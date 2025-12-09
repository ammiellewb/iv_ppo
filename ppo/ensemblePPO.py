"""
Ensemble PPO Implementation

PPO with ensemble of value networks for epistemic uncertainty estimation.
Paper: "Sample Efficient Deep Reinforcement Learning via Uncertainty Estimation"
https://openreview.net/forum?id=vrW3tvDfOJQ

Following the same structure as ensembleDQN.py and ensembleSAC.py.
"""

from .ppo import PPOAgent, RolloutBuffer
from .networks import PolicyNetwork, ValueNetwork, DiscretePolicyNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import os
import wandb
from collections import deque
from scipy.optimize import minimize


def compute_eff_bs(weights):
    """Compute effective batch size from normalized weights.
    
    The effective batch size measures how many samples are effectively 
    contributing to the gradient. A uniform distribution gives eff_bs = n,
    while a concentrated distribution gives eff_bs close to 1.
    
    eff_bs = 1 / sum(w_i^2)
    """
    return 1.0 / np.sum(np.square(weights))


def get_optimal_xi(variances, minimal_size, xi_start=0):
    """Find optimal xi to achieve a target effective batch size.
    
    Uses Nelder-Mead optimization to find the regularization parameter
    xi that achieves the desired effective batch size when computing
    inverse variance weights.
    
    Params
    ======
        variances (array): per-sample variances
        minimal_size (int): target effective batch size
        xi_start (float): starting value for optimization
        
    Returns
    =======
        xi (float): optimal regularization parameter
    """
    minimal_size = min(variances.shape[0] - 1, minimal_size)
    
    # First check if IV weights already achieve target without regularization
    iv_weights = 1.0 / (variances + 1e-8)
    iv_weights = iv_weights / iv_weights.sum()
    if compute_eff_bs(iv_weights) >= minimal_size:
        return 0.0
    
    # Optimize to find xi that achieves target effective batch size
    def objective(x):
        xi = np.abs(x[0])
        w = 1.0 / (variances + xi)
        w = w / w.sum()
        return np.abs(compute_eff_bs(w) - minimal_size)
    
    result = minimize(objective, [0], method='Nelder-Mead', 
                     options={'fatol': 1.0, 'maxiter': 100})
    xi = np.abs(result.x[0])
    return xi if xi is not None else 0.0


class EnsemblePPO(PPOAgent):
    """PPO with ensemble of value networks for epistemic uncertainty estimation.
    
    Maintains K value networks whose disagreement (variance) captures
    epistemic uncertainty in value estimates. This is used for:
    1. Better exploration via uncertainty-aware action selection
    2. Computing epistemic uncertainty σ²_ensemble for BIV weighting
    
    The variance over the ensemble provides an estimate of model uncertainty:
    σ²_ensemble(s) = (1/K) * Σ(V_k(s) - μ_V(s))²
    """
    
    def __init__(self, env, opt, device="cpu"):
        """Initialize EnsemblePPO with K value networks.
        
        Params
        ======
            env: gym environment
            opt: configuration options (must include num_nets)
            device: cpu or cuda
        """
        # Don't call super().__init__() fully - we need custom network setup
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
        
        # Policy network (single, shared across ensemble)
        if self.continuous:
            self.policy = PolicyNetwork(self.state_size, self.action_size, 
                                       opt.net_seed).to(self.device)
        else:
            self.policy = DiscretePolicyNetwork(self.state_size, self.action_size, 
                                               opt.net_seed).to(self.device)
        
        # Ensemble of value networks
        self.value_nets = nn.ModuleList([
            ValueNetwork(self.state_size, opt.net_seed + i).to(self.device)
            for i in range(self.num_nets)
        ])
        
        # Target value networks (for stable target computation)
        self.target_value_nets = nn.ModuleList([
            ValueNetwork(self.state_size, opt.net_seed + i).to(self.device)
            for i in range(self.num_nets)
        ])
        
        # Initialize targets to match online networks
        for i in range(self.num_nets):
            self.target_value_nets[i].load_state_dict(self.value_nets[i].state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=opt.policy_lr)
        self.value_optimizers = [
            optim.Adam(self.value_nets[i].parameters(), lr=opt.qf_lr)
            for i in range(self.num_nets)
        ]
        
        # For compatibility with parent class
        self.value = self.value_nets[0]  # Default to first network
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
        
        # Ensemble-specific parameters
        self.tau = getattr(opt, 'tau', 0.005)  # Soft update coefficient
        self.xi = getattr(opt, 'xi', 0.1)  # For subclasses
        
        # Tracking
        self.t_step = 0
        self.episode = 0
        
        # Initialize environment state
        reset_output = self.env.reset()
        if isinstance(reset_output, tuple):
            self.current_state, _ = reset_output
        else:
            self.current_state = reset_output
        self.episode_reward = 0
        self.episode_length = 0
        
        # Wandb logging
        if opt.tag:
            # Use WANDB_PROJECT env var if set, otherwise construct from opt
            import os
            project_name = os.getenv('WANDB_PROJECT', f"{opt.exp}_{opt.env}_{opt.model}")
            wandb.init(project=project_name, 
                      name=f"{opt.env}_{opt.model}_seed{opt.env_seed}",
                      config=vars(opt),
                      tags=[opt.tag, opt.env, opt.model])
    
    def get_ensemble_value(self, state):
        """Get mean and variance of value estimates from the ensemble.
        
        Params
        ======
            state: input state tensor
            
        Returns
        =======
            mean_value: mean of ensemble predictions
            variance: variance (epistemic uncertainty) across ensemble
        """
        with torch.no_grad():
            values = torch.stack([net(state) for net in self.value_nets], dim=0)
            mean_value = values.mean(dim=0)
            variance = values.var(dim=0)
        return mean_value, variance
    
    def get_target_ensemble_value(self, state):
        """Get mean and variance from target ensemble (for TD targets).
        
        Used for computing epistemic uncertainty in the bootstrapped targets:
        σ²_target(s') = (1/K) * Σ(V_target_k(s') - μ_target(s'))²
        """
        with torch.no_grad():
            values = torch.stack([net(state) for net in self.target_value_nets], dim=0)
            mean_value = values.mean(dim=0)
            variance = values.var(dim=0)
        return mean_value, variance
    
    def soft_update_targets(self):
        """Soft update target networks: θ_target = τ*θ + (1-τ)*θ_target"""
        for i in range(self.num_nets):
            for target_param, param in zip(self.target_value_nets[i].parameters(),
                                           self.value_nets[i].parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )
    
    def select_action(self, state, deterministic=False):
        """Select action using mean ensemble value."""
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
            
            # Use mean ensemble value
            mean_value, _ = self.get_ensemble_value(state_tensor)
            value = mean_value.cpu().item()
            log_prob = log_prob.cpu().item()
            
        return action, log_prob, value
    
    def evaluate_actions(self, states, actions, net_idx=None):
        """Evaluate actions, optionally using specific network."""
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
            values = self.value_nets[net_idx](states).squeeze()
        else:
            # Use mean ensemble value
            values_stack = torch.stack([net(states) for net in self.value_nets], dim=0)
            values = values_stack.mean(dim=0).squeeze()
        
        return log_probs, values, entropy
    
    def get_mse_weights(self, variance):
        """Get MSE weights - uniform for base ensemble (overridden in IV_PPO)."""
        weights = torch.ones(variance.size()).to(self.device) / variance.size(0)
        return weights
    
    def update(self):
        """Update policy and ensemble value networks."""
        data = self.buffer.get()
        
        states = data['states']
        actions = data['actions']
        old_log_probs = data['log_probs']
        advantages = data['advantages']
        returns = data['returns']
        
        policy_losses = []
        value_losses = []
        eff_batch_size_list = []
        
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
                mb_returns = returns[mb_indices]
                
                # Get target variance for weighting
                _, target_var = self.get_target_ensemble_value(mb_states)
                weights = self.get_mse_weights(target_var)
                
                # Update each value network in ensemble
                for i in range(self.num_nets):
                    log_probs, values, entropy = self.evaluate_actions(
                        mb_states, mb_actions, net_idx=i
                    )
                    
                    # Weighted value loss for this network
                    td_error_sq = (values - mb_returns) ** 2
                    value_loss = (weights.squeeze() * td_error_sq).sum()
                    
                    # Update value network
                    self.value_optimizers[i].zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.value_nets[i].parameters(), 0.5)
                    self.value_optimizers[i].step()
                    
                    value_losses.append(value_loss.item())
                
                eff_batch_size_list.append(
                    compute_eff_bs(weights.detach().cpu().numpy().flatten()))
                
                # Policy update (using mean ensemble value)
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
            
            # Soft update target networks after each epoch
            self.soft_update_targets()
            
            # Early stopping based on KL
            with torch.no_grad():
                log_probs_new, _, _ = self.evaluate_actions(states, actions)
                kl = (old_log_probs - log_probs_new).mean().item()
                if kl > 1.5 * self.target_kl:
                    break
        
        mean_policy_loss = np.mean(policy_losses)
        mean_value_loss = np.mean(value_losses)
        mean_eff_bs = np.mean(eff_batch_size_list)
        
        if self.opt.tag:
            wandb.log({
                "policy_loss": mean_policy_loss,
                "value_loss": mean_value_loss,
                "entropy": entropy.item(),
                "approx_kl": kl,
                "effective_batch_size": mean_eff_bs,
            })
        
        return mean_policy_loss, mean_value_loss
    
    def train_log(self, var, weights, eff_batch_size, xi):
        """Log training statistics."""
        if self.opt.tag:
            wandb.log({
                "IV Weights(VAR)": np.var(weights),
                "IV Weights(Mean)": np.mean(weights),
                "IV Weights(Min)": np.min(weights),
                "IV Weights(Max)": np.max(weights),
                "IV Weights(Median)": np.median(weights),
                "Variance(V) (VAR)": np.var(var),
                "Variance(V) (Mean)": np.mean(var),
                "Variance(V) (Min)": np.min(var),
                "Variance(V) (Max)": np.max(var),
                "Variance(V) (Median)": np.median(var),
                "Effective Batch Size": eff_batch_size,
                "Xi": xi,
            }, commit=False)
