import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNetwork(nn.Module):
    """Actor (Policy) Network for continuous actions."""
    
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(PolicyNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc_mean = nn.Linear(fc2_units, action_size)
        self.fc_logstd = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action distribution (mean, log_std)."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Stabilize training
        return mean, log_std


class ValueNetwork(nn.Module):
    """Critic (Value) Network."""
    
    def __init__(self, state_size, seed, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Dimension of each state
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer (256 like notebook)
            fc2_units (int): Number of nodes in second hidden layer (256 like notebook)
        """
        super(ValueNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
        # Orthogonal initialization for better RL performance
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, state):
        """Build a network that maps state -> value."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DiscretePolicyNetwork(nn.Module):
    """Actor (Policy) Network for discrete actions."""
    
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer (256 like notebook)
            fc2_units (int): Number of nodes in second hidden layer (256 like notebook)
        """
        super(DiscretePolicyNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        # Orthogonal initialization for better RL performance
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)  # Small init for policy output
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, state):
        """Build a network that maps state -> action logits."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ValueNetworkWithVariance(nn.Module):
    """Critic (Value) Network with variance head for aleatoric uncertainty.
    
    Augments the standard value network with a second output head σ²(s) 
    to predict aleatoric (data) uncertainty directly from observations.
    """
    
    def __init__(self, state_size, seed, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Dimension of each state
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(ValueNetworkWithVariance, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Shared feature extractor
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        
        # Value head (mean prediction)
        self.value_head = nn.Linear(fc2_units, 1)
        
        # Variance head (aleatoric uncertainty) - outputs log(σ²) for numerical stability
        self.log_var_head = nn.Linear(fc2_units, 1)
        
        # Orthogonal initialization
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.orthogonal_(self.log_var_head.weight, gain=0.1)  # Small init for variance
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.value_head.bias, 0)
        nn.init.constant_(self.log_var_head.bias, 0)
        
    def forward(self, state, return_variance=False):
        """Build a network that maps state -> value (and optionally variance).
        
        Params
        ======
            state: input state tensor
            return_variance: if True, returns (value, variance), else just value
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        value = self.value_head(x)
        
        if return_variance:
            log_var = self.log_var_head(x)
            # Clamp log_var for numerical stability: exp(-10) ≈ 4.5e-5, exp(2) ≈ 7.4
            log_var = torch.clamp(log_var, min=-10, max=2)
            variance = torch.exp(log_var)
            return value, variance
        
        return value
    
    def get_log_variance(self, state):
        """Get log variance for loss attenuation computation."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        log_var = self.log_var_head(x)
        log_var = torch.clamp(log_var, min=-10, max=2)
        return log_var
