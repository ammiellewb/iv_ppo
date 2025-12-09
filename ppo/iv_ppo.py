"""
Inverse-Variance PPO (IV-PPO) Implementation

This module implements IV-RL methodology for PPO.
Paper: "Sample Efficient Deep Reinforcement Learning via Uncertainty Estimation"
https://openreview.net/forum?id=vrW3tvDfOJQ

Extends EnsemblePPO with:
A. Variance Head for Aleatoric Uncertainty - ValueNetworkWithVariance
B. Batch Inverse-Variance (BIV) Weighting
C. Loss Attenuation (LA) for learning variance
D. Combined IV-RL Loss

The combined loss accounts for both model-based (epistemic) and 
data-driven (aleatoric) uncertainty when updating the critic.
"""

from .ensemblePPO import EnsemblePPO, compute_eff_bs, get_optimal_xi
from .networks import (PolicyNetwork, ValueNetwork, DiscretePolicyNetwork, 
                       ValueNetworkWithVariance)
from .ppo import RolloutBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import wandb


class IV_PPO(EnsemblePPO):
    """Inverse-Variance PPO with BIV weighting and loss attenuation.
    
    Implements the full IV-RL methodology:
    
    A. Aleatoric Uncertainty: ValueNetworkWithVariance predicts σ²_aleatoric(s)
       using loss attenuation (negative log-likelihood):
       L_LA = 0.5 * log(σ²) + 0.5 * ((y - V(s)) / σ)²
    
    B. Epistemic Uncertainty: Ensemble variance σ²_epistemic from target networks:
       σ²_epistemic(s') = Var_k[V_target_k(s')]
    
    C. BIV Weighting: Per-sample weights based on total variance:
       σ²_total = σ²_aleatoric(s) + σ²_epistemic(s')
       w_i = 1 / (σ²_total_i + ξ)
       w̃_i = w_i / Σ_j w_j  (normalized)
    
    D. Combined Loss:
       L_IV-RL = λ * L_BIV + (1-λ) * L_LA
    """
    
    def __init__(self, env, opt, device="cpu"):
        """Initialize IV_PPO.
        
        Params
        ======
            env: gym environment
            opt: configuration options with IV-specific parameters:
                - xi: regularization for BIV weights (default: 0.1)
                - dynamic_xi: whether to adapt xi to maintain eff batch size
                - minimal_eff_bs: target effective batch size for dynamic xi
                - lambda_biv: weight for BIV loss vs LA loss (default: 0.5)
        """
        # Initialize base attributes manually (don't call super().__init__())
        self.env = env
        self.opt = opt
        self.device = device
        
        self.ent_coef = 0.1
        self.n_epochs = 10
        
        # Determine state/action sizes
        try:
            self.state_size = env.observation_space.shape[0]
            if isinstance(env.action_space, gym.spaces.Box):
                self.continuous = True
                self.action_size = env.action_space.shape[0]
            elif isinstance(env.action_space, gym.spaces.Discrete):
                self.continuous = False
                self.action_size = env.action_space.n
            else:
                raise ValueError("Environment must have Box or Discrete action space")
        except AttributeError as e:
            raise ValueError(f"Environment must have Box or Discrete action space: {e}")
            
        self.seed = random.seed(opt.net_seed)
        self.test_scores = []
        
        # Number of ensemble members
        self.num_nets = getattr(opt, 'num_nets', 5)
        
        # Policy network
        if self.continuous:
            self.policy = PolicyNetwork(self.state_size, self.action_size, 
                                       opt.net_seed).to(self.device)
        else:
            self.policy = DiscretePolicyNetwork(self.state_size, self.action_size, 
                                               opt.net_seed).to(self.device)
        
        # Ensemble of value networks WITH variance heads for aleatoric uncertainty
        self.value_nets = nn.ModuleList([
            ValueNetworkWithVariance(self.state_size, opt.net_seed + i).to(self.device)
            for i in range(self.num_nets)
        ])
        
        # Target value networks (standard, for epistemic uncertainty)
        self.target_value_nets = nn.ModuleList([
            ValueNetwork(self.state_size, opt.net_seed + i).to(self.device)
            for i in range(self.num_nets)
        ])
        
        # Initialize targets from value_nets (just the value part)
        for i in range(self.num_nets):
            # Copy shared layers
            self.target_value_nets[i].fc1.load_state_dict(self.value_nets[i].fc1.state_dict())
            self.target_value_nets[i].fc2.load_state_dict(self.value_nets[i].fc2.state_dict())
            self.target_value_nets[i].fc3.load_state_dict(self.value_nets[i].value_head.state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=opt.policy_lr)
        self.value_optimizers = [
            optim.Adam(self.value_nets[i].parameters(), lr=opt.qf_lr)
            for i in range(self.num_nets)
        ]
        
        # Compatibility with parent
        self.value = self.value_nets[0]
        self.value_optimizer = self.value_optimizers[0]
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=opt.buffer_size,
            state_size=self.state_size,
            action_size=self.action_size,
            device=self.device,
            continuous=self.continuous
        )
        
        # PPO hyperparameters
        self.gamma = opt.gamma
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.target_kl = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.ppo_epochs = 10
        self.mini_batch_size = opt.batch_size
        
        # IV-specific parameters
        self.xi = getattr(opt, 'xi', 0.1)  # Regularization for BIV
        self.dynamic_xi = getattr(opt, 'dynamic_xi', False)
        self.minimal_eff_bs = getattr(opt, 'minimal_eff_bs', 32)
        self.lambda_biv = getattr(opt, 'lambda_biv', 0.5)  # BIV vs LA weighting
        self.tau = getattr(opt, 'tau', 0.005)
        
        # Tracking
        self.t_step = 0
        self.episode = 0
        
        # Initialize environment
        reset_output = self.env.reset()
        if isinstance(reset_output, tuple):
            self.current_state, _ = reset_output
        else:
            self.current_state = reset_output
        self.episode_reward = 0
        self.episode_length = 0
        
        # Wandb
        if opt.tag:
            # Use WANDB_PROJECT env var if set, otherwise construct from opt
            import os
            project_name = os.getenv('WANDB_PROJECT', f"{opt.exp}_{opt.env}_{opt.model}")
            wandb.init(project=project_name, 
                      name=f"{opt.env}_{opt.model}_seed{opt.env_seed}",
                      config=vars(opt),
                      tags=[opt.tag, opt.env, opt.model])
    
    def iv_weights(self, variance):
        """Compute inverse variance weights with regularization.
        
        w_i = 1 / (σ²_i + ξ)
        w̃_i = w_i / Σ_j w_j
        """
        weights = 1.0 / (variance + self.xi)
        weights = weights / weights.sum(dim=0, keepdim=True)
        return weights
    
    def get_mse_weights(self, variance):
        """Override parent to use IV weights."""
        return self.iv_weights(variance)
    
    def get_aleatoric_variance(self, states, net_idx=None):
        """Get aleatoric (learned) variance from value network(s).
        
        Returns the σ²_aleatoric predicted by the variance head.
        """
        if net_idx is not None:
            _, variance = self.value_nets[net_idx](states, return_variance=True)
            return variance
        else:
            # Mean aleatoric variance across ensemble
            variances = torch.stack([
                self.value_nets[i](states, return_variance=True)[1]
                for i in range(self.num_nets)
            ], dim=0)
            return variances.mean(dim=0)
    
    def get_epistemic_variance(self, states, use_target=True):
        """Get epistemic variance from ensemble disagreement.
        
        σ²_epistemic = (1/K) * Σ(V_k - μ_V)²
        """
        if use_target:
            with torch.no_grad():
                values = torch.stack([
                    net(states) for net in self.target_value_nets
                ], dim=0)
        else:
            values = torch.stack([
                self.value_nets[i](states, return_variance=False)
                for i in range(self.num_nets)
            ], dim=0)
        
        variance = values.var(dim=0)
        return variance
    
    def compute_loss_attenuation(self, values, targets, log_variances):
        """Compute loss attenuation (negative log-likelihood) loss.
        
        L_LA = 0.5 * log(σ²) + 0.5 * ((y - V(s)) / σ)²
             = 0.5 * log_var + 0.5 * exp(-log_var) * (y - V)²
        
        This learns to predict aleatoric uncertainty while being robust
        to high-variance (noisy) samples.
        """
        td_error_sq = (targets - values) ** 2
        loss = 0.5 * log_variances + 0.5 * torch.exp(-log_variances) * td_error_sq
        return loss.mean()
    
    def compute_biv_loss(self, values, targets, weights):
        """Compute BIV-weighted MSE loss.
        
        L_BIV = Σ w̃_i * (y_i - V(s_i))²
        """
        td_error_sq = (targets - values) ** 2
        weighted_loss = weights * td_error_sq
        return weighted_loss.sum()
    
    def compute_iv_rl_loss(self, values, targets, log_variances, total_variance, net_idx=0):
        """Compute combined IV-RL loss.
        
        L_IV-RL = λ * L_BIV + (1-λ) * L_LA
        
        Params
        ======
            values: predicted values V(s)
            targets: TD targets y = r + γV(s')
            log_variances: log σ²_aleatoric from variance head
            total_variance: σ²_total = σ²_aleatoric + σ²_epistemic
            net_idx: which network (for logging)
            
        Returns
        =======
            total_loss: combined IV-RL loss
            biv_loss: BIV component (for logging)
            la_loss: Loss attenuation component (for logging)
        """
        # Compute dynamic xi if enabled
        if self.dynamic_xi:
            current_xi = get_optimal_xi(
                total_variance.detach().cpu().numpy().flatten(),
                self.minimal_eff_bs,
                self.xi
            )
        else:
            current_xi = self.xi
        
        # Compute BIV weights
        weights = 1.0 / (total_variance + current_xi)
        weights = weights / weights.sum()
        
        # BIV loss
        biv_loss = self.compute_biv_loss(values, targets, weights.detach())
        
        # Loss attenuation
        la_loss = self.compute_loss_attenuation(values, targets, log_variances)
        
        # Combined loss
        total_loss = self.lambda_biv * biv_loss + (1 - self.lambda_biv) * la_loss
        
        return total_loss, biv_loss, la_loss, weights, current_xi
    
    def select_action(self, state, deterministic=False):
        """Select action using mean ensemble value (ignoring variance for action)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.continuous:
                mean, log_std = self.policy(state_tensor)
                if deterministic:
                    action = mean
                    log_prob = torch.zeros(1)
                else:
                    std = torch.exp(log_std)
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                action = action.cpu().numpy()[0]
            else:
                logits = self.policy(state_tensor)
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                    log_prob = torch.zeros(1)
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                action = action.cpu().item()
            
            # Mean value from ensemble (just value, not variance)
            values = torch.stack([
                net(state_tensor, return_variance=False) 
                for net in self.value_nets
            ], dim=0)
            value = values.mean(dim=0).cpu().item()
            log_prob = log_prob.cpu().item()
            
        return action, log_prob, value
    
    def evaluate_actions(self, states, actions, net_idx=None):
        """Evaluate actions, optionally with variance."""
        if self.continuous:
            mean, log_std = self.policy(states)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
        else:
            logits = self.policy(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions.squeeze())
            entropy = dist.entropy().mean()
        
        if net_idx is not None:
            values = self.value_nets[net_idx](states, return_variance=False).squeeze()
        else:
            values_stack = torch.stack([
                net(states, return_variance=False) 
                for net in self.value_nets
            ], dim=0)
            values = values_stack.mean(dim=0).squeeze()
        
        return log_probs, values, entropy
    
    def soft_update_targets(self):
        """Soft update target networks."""
        for i in range(self.num_nets):
            # Update fc1
            for target_param, param in zip(self.target_value_nets[i].fc1.parameters(),
                                           self.value_nets[i].fc1.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )
            # Update fc2
            for target_param, param in zip(self.target_value_nets[i].fc2.parameters(),
                                           self.value_nets[i].fc2.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )
            # Update fc3 (value head -> fc3)
            for target_param, param in zip(self.target_value_nets[i].fc3.parameters(),
                                           self.value_nets[i].value_head.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )
    
    def update(self):
        """Update with IV-RL loss combining BIV weighting and loss attenuation."""
        data = self.buffer.get()
        
        states = data['states']
        actions = data['actions']
        old_log_probs = data['log_probs']
        advantages = data['advantages']
        returns = data['returns']
        
        policy_losses = []
        value_losses = []
        biv_losses = []
        la_losses = []
        eff_batch_sizes = []
        xi_values = []
        
        for epoch in range(self.ppo_epochs):
            batch_size = states.size(0)
            indices = np.arange(batch_size)
            np.random.shuffle(indices)
            
            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                if end > batch_size:
                    continue
                    
                mb_indices = indices[start:end]
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]  # These are the TD targets
                
                # Get epistemic variance from target ensemble on current states
                # (for weighting the learning signal)
                epistemic_var = self.get_epistemic_variance(mb_states, use_target=True)
                
                # Update each value network with IV-RL loss
                for i in range(self.num_nets):
                    # Get value and aleatoric variance from network i
                    values, aleatoric_var = self.value_nets[i](mb_states, return_variance=True)
                    values = values.squeeze()
                    aleatoric_var = aleatoric_var.squeeze()
                    log_var = self.value_nets[i].get_log_variance(mb_states).squeeze()
                    
                    # Total variance for BIV weighting
                    # Note: using epistemic from targets, aleatoric from online network
                    total_var = aleatoric_var + epistemic_var.squeeze()
                    
                    # Compute IV-RL loss
                    value_loss, biv_loss, la_loss, weights, current_xi = self.compute_iv_rl_loss(
                        values, mb_returns, log_var, total_var, net_idx=i
                    )
                    
                    # Update value network
                    self.value_optimizers[i].zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.value_nets[i].parameters(), 0.5)
                    self.value_optimizers[i].step()
                    
                    value_losses.append(value_loss.item())
                    biv_losses.append(biv_loss.item())
                    la_losses.append(la_loss.item())
                    eff_batch_sizes.append(compute_eff_bs(weights.detach().cpu().numpy().flatten()))
                    xi_values.append(current_xi)
                
                # Policy update
                log_probs, _, entropy = self.evaluate_actions(mb_states, mb_actions)
                
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy
                
                total_policy_loss = policy_loss + self.entropy_coef * entropy_loss
                
                self.policy_optimizer.zero_grad()
                total_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()
                
                policy_losses.append(policy_loss.item())
            
            # Soft update target networks
            self.soft_update_targets()
            
            # Early stopping
            with torch.no_grad():
                log_probs_new, _, _ = self.evaluate_actions(states, actions)
                kl = (old_log_probs - log_probs_new).mean().item()
                if kl > 1.5 * self.target_kl:
                    break
        
        mean_policy_loss = np.mean(policy_losses)
        mean_value_loss = np.mean(value_losses)
        mean_biv_loss = np.mean(biv_losses)
        mean_la_loss = np.mean(la_losses)
        mean_eff_bs = np.mean(eff_batch_sizes)
        mean_xi = np.mean(xi_values)
        
        if self.opt.tag:
            wandb.log({
                "policy_loss": mean_policy_loss,
                "value_loss": mean_value_loss,
                "biv_loss": mean_biv_loss,
                "la_loss": mean_la_loss,
                "effective_batch_size": mean_eff_bs,
                "xi": mean_xi,
                "entropy": entropy.item(),
                "approx_kl": kl,
            })
        
        return mean_policy_loss, mean_value_loss
