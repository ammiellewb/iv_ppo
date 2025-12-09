import torch
import torch.optim as optim
import random
import numpy as np
import gymnasium as gym
# from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
import os
import wandb

from utils import *

class RolloutBuffer:
    """Fixed-size buffer to store experience tuples for PPO."""
    
    def __init__(self, buffer_size, state_size, action_size, device, continuous=False):
        """Initialize a RolloutBuffer object.
        
        Params
        ======
            buffer_size (int): maximum size of buffer
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            device (str): cpu or gpu
            continuous (bool): whether action space is continuous
        """
        self.states = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_size if continuous else 1), dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = buffer_size
        self.device = device
        self.continuous = continuous
        
    def add(self, state, action, log_prob, reward, value, done):
        """Add a new experience to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1
        
    def finish_path(self, last_value=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This computes advantage estimates using
        Generalized Advantage Estimation (GAE-Lambda).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        # GAE-Lambda advantage calculation
        deltas = rewards[:-1] + 0.99 * values[1:] - values[:-1]
        self.advantages[path_slice] = self._discount_cumsum(deltas, 0.99 * 0.95)
        
        # Compute returns (targets for value function)
        self.returns[path_slice] = self._discount_cumsum(rewards, 0.99)[:-1]
        
        self.path_start_idx = self.ptr
        
    def _discount_cumsum(self, x, discount):
        """
        Compute cumulative sums of vectors with discount factor.
        input: [x0, x1, x2]
        output: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
        """
        return np.array([np.sum(discount ** np.arange(len(x) - i) * x[i:]) 
                        for i in range(len(x))])
        
    def get(self):
        """
        Get all data from the buffer, and reset pointers.
        """
        assert self.ptr > 0, "Buffer is empty"
        
        # Normalize advantages
        adv_mean = np.mean(self.advantages[:self.ptr])
        adv_std = np.std(self.advantages[:self.ptr])
        self.advantages[:self.ptr] = (self.advantages[:self.ptr] - adv_mean) / (adv_std + 1e-8)
        
        data = dict(
            states=torch.FloatTensor(self.states[:self.ptr]).to(self.device),
            actions=torch.FloatTensor(self.actions[:self.ptr]).to(self.device),
            log_probs=torch.FloatTensor(self.log_probs[:self.ptr]).to(self.device),
            advantages=torch.FloatTensor(self.advantages[:self.ptr]).to(self.device),
            returns=torch.FloatTensor(self.returns[:self.ptr]).to(self.device),
        )
        
        # Reset pointers
        self.ptr = 0
        self.path_start_idx = 0
        
        return data
        
    def __len__(self):
        return self.ptr


class PPOAgent:
    """PPO Agent that interacts with and learns from the environment."""
    
    def __init__(self, env, opt, device="cpu"):
        """Initialize a PPO Agent.
        
        Params
        ======
            env (gym object): Initialized gym environment
            opt (dict): command line options for the model
            device (str): cpu or gpu
        """
        self.env = env
        self.opt = opt
        self.device = device

        self.ent_coef = 0.1  # Increase from default (was likely too small or 0)
        self.n_epochs = 10   # Multiple update passes per rollout
        
        # Determine if continuous or discrete action space
        try:
            self.state_size = env.observation_space.shape[0]
            # Check if action space is continuous (Box) or discrete (Discrete)
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
        
        # Networks
        if self.continuous:
            from .networks import PolicyNetwork, ValueNetwork
            self.policy = PolicyNetwork(self.state_size, self.action_size, 
                                       opt.net_seed).to(self.device)
            self.value = ValueNetwork(self.state_size, opt.net_seed).to(self.device)
        else:
            from .networks import DiscretePolicyNetwork, ValueNetwork
            self.policy = DiscretePolicyNetwork(self.state_size, self.action_size, 
                                               opt.net_seed).to(self.device)
            self.value = ValueNetwork(self.state_size, opt.net_seed).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=opt.policy_lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=opt.qf_lr)
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=opt.buffer_size,
            state_size=self.state_size,
            action_size=self.action_size,
            device=self.device,
            continuous=self.continuous
        )
        
        # PPO hyperparameters
        self.gamma = opt.gamma  # 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.target_kl = 0.2  # Higher to prevent premature early stopping
        self.entropy_coef = 0.01  # C_2 in notebook
        self.value_coef = 0.5  # C_1 in notebook
        self.ppo_epochs = 10  # K in notebook
        self.mini_batch_size = opt.batch_size  # M=64 in notebook
        
        # Tracking
        self.t_step = 0
        self.episode = 0
        
        # Initialize environment state (persistent across rollouts)
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
                      
    def select_action(self, state, deterministic=False):
        """Select an action from the policy.
        
        Params
        ======
            state (array_like): current state
            deterministic (bool): if True, use mean action (for eval)
        
        Returns
        =======
            action (array): selected action
            log_prob (float): log probability of the action
            value (float): value estimate of the state
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.continuous:
                mean, log_std = self.policy(state)
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
                logits = self.policy(state)
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                    log_prob = torch.zeros(1)
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                action = action.cpu().item()
                
            value = self.value(state)
            value = value.cpu().item()
            log_prob = log_prob.cpu().item()
            
        return action, log_prob, value
        
    def evaluate_actions(self, states, actions):
        """Evaluate actions according to current policy.
        
        Params
        ======
            states (Tensor): batch of states
            actions (Tensor): batch of actions
            
        Returns
        =======
            log_probs (Tensor): log probabilities of actions
            values (Tensor): value estimates
            entropy (Tensor): entropy of action distribution
        """
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
            
        values = self.value(states).squeeze()
        
        return log_probs, values, entropy
        
    def collect_rollout(self, n_steps):
        """Collect n_steps of experience using current policy.
        
        Params
        ======
            n_steps (int): number of steps to collect
        """
        # Use persistent state (don't reset env at start of rollout!)
        state = self.current_state
        
        for step in range(n_steps):
            action, log_prob, value = self.select_action(state)
            
            # Take action in environment (handle both old and new gym API)
            step_output = self.env.step(action)
            if len(step_output) == 5:
                next_state, reward, done, truncated, info = step_output
                done = done or truncated
            else:
                next_state, reward, done, info = step_output
                
            self.episode_reward += reward
            self.episode_length += 1
            
            # Store experience
            self.buffer.add(state, action, log_prob, reward, value, done)
            
            state = next_state
            
            if done:
                self.buffer.finish_path(last_value=0)
                
                # Log episode stats
                self.test_scores.append(self.episode_reward)
                
                # Print when goal is reached (reward > -200)
                if self.episode_reward > -200:
                    print(f"🎉 SUCCESS! Episode {self.episode}: Reward = {self.episode_reward:.0f}, Length = {self.episode_length}")
                
                if self.opt.tag:
                    wandb.log({
                        "episode_reward": self.episode_reward,
                        "episode_length": self.episode_length,
                        "episode": self.episode
                    })
                    
                self.episode += 1
                
                # Reset environment for next episode
                reset_output = self.env.reset()
                if isinstance(reset_output, tuple):
                    state, _ = reset_output
                else:
                    state = reset_output
                    
                self.episode_reward = 0
                self.episode_length = 0
                
        # Update persistent state
        self.current_state = state
        
        # If trajectory didn't end, bootstrap value
        if not done:
            _, _, last_value = self.select_action(state)
            self.buffer.finish_path(last_value=last_value)
            
    def update(self):
        """Update policy and value networks using collected rollouts."""
        data = self.buffer.get()
        
        states = data['states']
        actions = data['actions']
        old_log_probs = data['log_probs']
        advantages = data['advantages']
        returns = data['returns']
        
        # PPO update for multiple epochs
        for epoch in range(self.ppo_epochs):
            # Generate random indices for mini-batches
            batch_size = states.size(0)
            indices = np.arange(batch_size)
            np.random.shuffle(indices)
            
            # Process mini-batches
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
                
                # Evaluate actions with current policy
                log_probs, values, entropy = self.evaluate_actions(mb_states, mb_actions)
                
                # Compute ratio (pi_theta / pi_theta_old)
                ratio = torch.exp(log_probs - mb_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
                
                # PPO policy loss (maximize)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, mb_returns)
                
                # Entropy bonus (encourage exploration)
                entropy_loss = -entropy
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update policy
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
            # Check KL divergence for early stopping
            with torch.no_grad():
                log_probs_new, _, _ = self.evaluate_actions(states, actions)
                kl = (old_log_probs - log_probs_new).mean().item()
                if kl > 1.5 * self.target_kl:
                    print(f"Early stopping at epoch {epoch} due to reaching max KL.")
                    break
                    
        # Log training stats
        if self.opt.tag:
            wandb.log({
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.item(),
                "approx_kl": kl,
            })
            
        return policy_loss.item(), value_loss.item()
        
    def train(self, n_episodes=1000, steps_per_epoch=2048):
        """Train the agent.
        
        Params
        ======
            n_episodes (int): maximum number of training episodes
            steps_per_epoch (int): number of steps to collect per update
        """
        scores_window = deque(maxlen=100)
        
        print(f"Training PPO on {self.opt.env}...")
        print(f"State size: {self.state_size}, Action size: {self.action_size}")
        print(f"Continuous: {self.continuous}")
        
        epoch = 0
        while self.episode < n_episodes:
            # Collect rollouts
            self.collect_rollout(steps_per_epoch)
            
            # Update policy (update() already has internal loop over ppo_epochs)
            policy_loss, value_loss = self.update()
            
            # Track progress
            if len(self.test_scores) > 0:
                scores_window.append(self.test_scores[-1])
                
            epoch += 1
            
            # Print status
            if epoch % 10 == 0:
                avg_score = np.mean(scores_window) if len(scores_window) > 0 else 0
                print(f"Epoch {epoch}, Episode {self.episode}, "
                      f"Avg Score: {avg_score:.2f}, "
                      f"Policy Loss: {policy_loss:.4f}, "
                      f"Value Loss: {value_loss:.4f}")
                      
            # Save checkpoint
            if self.opt.save_freq > 0 and epoch % self.opt.save_freq == 0:
                self.save(f"{self.opt.log_dir}/ppo_{self.opt.env}_epoch_{epoch}.pth")
                
        print("Training completed!")
        
        # Save training scores to .npy file (same format as DQN/SAC)
        scores_filename = os.path.join(self.opt.log_dir,
                                       f"logs_{self.opt.exp}_{self.opt.env}_{self.opt.model}_seed_{self.opt.env_seed}_net_seed_{self.opt.net_seed}.npy")
        np.save(scores_filename, np.array(self.test_scores))
        print(f"Saved episode scores to: {scores_filename}")
        
        return self.test_scores
        
    def save(self, filepath):
        """Save model parameters."""
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
        
    def load(self, filepath):
        """Load model parameters."""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        print(f"Model loaded from {filepath}")
    
