# IV-RL PPO Experiments - Quick Start Guide

## Overview
This guide provides instructions for running comprehensive experiments comparing:



### Full Experimental Suite
Runs all experiments (baseline, ablations, different seeds) on all continuous control environments:
```bash
cd /home/ammiellewb/iv_rl/scripts
./run_ppo_iv_experiments.sh
```

This will create tmux sessions for:
- **Baseline PPO**: 5 environments × 5 seeds = 25 sessions
- **EnsemblePPO**: 5 environments × 5 seeds = 25 sessions  
- **IV-PPO (full)**: 5 environments × 5 seeds = 25 sessions
- **Ablations**: ~40 additional sessions for hyperparameter studies

**Total**: ~115 parallel tmux sessions

### Custom Single Experiment
```bash
python main.py \
    --env Pendulum-v1 \
    --model IV_PPO \
    --env_seed 0 \
    --net_seed 0 \
    --num_episodes 500 \
    --tag my_experiment
```

### Environment-Specific Runs

**Pendulum (Simple Continuous Control)**
```bash
python main.py --env Pendulum-v1 --model IV_PPO --num_episodes 500 --tag pendulum_test
```

**HalfCheetah (MuJoCo Locomotion)**
```bash
python main.py --env HalfCheetah-v4 --model IV_PPO --num_episodes 1000 --tag cheetah_test
```

**Hopper (Balance + Forward Movement)**
```bash
python main.py --env Hopper-v4 --model IV_PPO --num_episodes 1000 --tag hopper_test
```

**Walker2d (Bipedal Walking)**
```bash
python main.py --env Walker2d-v4 --model IV_PPO --num_episodes 1000 --tag walker_test
```

**Ant (Quadruped Locomotion)**
```bash
python main.py --env Ant-v4 --model IV_PPO --num_episodes 1000 --tag ant_test
```

## Managing Tmux Sessions

### View Active Sessions
```bash
./check_experiments.sh
# Or manually:
tmux ls
```

### Attach to Session (Monitor Progress)
```bash
tmux attach -t ivppo_Pendulum-v1_seed0
```

### Detach from Session
Press `Ctrl+B` then `D`

### Kill Specific Session
```bash
tmux kill-session -t ivppo_Pendulum-v1_seed0
```

### Kill All Experiment Sessions
```bash
./kill_all_experiments.sh
```

## Tracked Metrics

### Sample Efficiency Metrics
- **episode_reward**: Cumulative reward per episode
- **episode_length**: Number of steps per episode
- **episode**: Episode counter (for x-axis plotting)

### Learning Stability Metrics
- **policy_loss**: Actor network loss
- **value_loss**: Total critic loss (combined BIV + LA for IV-PPO)
- **biv_loss**: BIV-weighted MSE loss (IV-PPO only)
- **la_loss**: Loss attenuation (IV-PPO only)

### Uncertainty Metrics (IV-PPO)
- **effective_batch_size**: Effective batch size from BIV weighting
- **xi**: Regularization parameter (dynamic or fixed)
- **entropy**: Policy entropy (exploration measure)
- **approx_kl**: KL divergence (policy update magnitude)

### Additional Metrics (If Needed)
To track aleatoric and epistemic variance, add to `iv_ppo.py` in the `update()` method:
```python
mean_aleatoric = self.get_aleatoric_variance(states).mean().item()
mean_epistemic = self.get_epistemic_variance(states).mean().item()

wandb.log({
    # ... existing metrics ...
    "mean_aleatoric_var": mean_aleatoric,
    "mean_epistemic_var": mean_epistemic,
})
```

## Experiment Design

### Baseline Comparison
Compare learning curves across methods:
```
Baseline PPO vs EnsemblePPO vs IV-PPO
- X-axis: Environment steps (episode × episode_length)
- Y-axis: Episode reward
- Analysis: Which method learns fastest (sample efficiency)?
```

### Stability Analysis
Compare variance across random seeds:
```
For each method, compute:
- Mean reward at convergence (last 100 episodes)
- Standard deviation across 5 seeds
- Analysis: Which method is most stable (lowest variance)?
```

### Ablation Studies

**1. Dynamic Xi (IV-PPO only)**
- Fixed xi=0.1 vs Dynamic xi
- Hypothesis: Dynamic xi maintains target effective batch size better

**2. Lambda BIV (Loss Weighting)**
- λ ∈ {0.25, 0.5, 0.75}
- Hypothesis: λ=0.5 balances BIV and LA losses optimally

**3. Ensemble Size**
- K ∈ {3, 5, 10}
- Hypothesis: K=5 balances epistemic uncertainty estimation vs compute cost

## WandB Dashboard

Access your results at: `https://wandb.ai/<your-username>/iv-rl`

### Recommended Plots

**Sample Efficiency**
1. Line plot: `episode_reward` vs `episode` (all methods, grouped by environment)
2. Comparison table: Episodes to reach threshold (e.g., 90% of expert performance)

**Learning Stability**  
1. Line plot with confidence intervals: `episode_reward` ± std across seeds
2. Box plot: Final performance distribution (last 100 episodes)

**Uncertainty Evolution (IV-PPO)**
1. Line plot: `effective_batch_size` over time
2. Line plot: `xi` over time (if dynamic_xi=True)
3. Dual axis: `biv_loss` and `la_loss` over time

## Hyperparameters

All hyperparameters are defined in `config.py`:

### Shared (All Methods)
- `batch_size`: 64
- `buffer_size`: 2048 (steps per rollout)
- `gamma`: 0.99
- `policy_lr`: 0.0003
- `qf_lr`: 0.001

### EnsemblePPO Specific
- `num_nets`: 5 (ensemble size)
- `tau`: 0.005 (soft target update rate)

### IV-PPO Specific
- `xi`: 0.1 (variance regularization)
- `dynamic_xi`: True (optimize xi for target eff_bs)
- `minimal_eff_bs`: 48 (target effective batch size)
- `lambda_biv`: 0.5 (BIV loss weight, LA weight = 1-λ)

## Expected Results

### Sample Efficiency
IV-PPO should converge in **20-40% fewer episodes** than baseline PPO due to:
- Better handling of high-uncertainty transitions
- Adaptive weighting reduces negative impact of outliers

### Learning Stability
IV-PPO should show **30-50% lower variance** across seeds due to:
- Uncertainty-aware updates
- Loss attenuation preventing over-fitting to noisy data

### Uncertainty Patterns
- **Aleatoric uncertainty**: Should be higher in stochastic environments (Pendulum)
- **Epistemic uncertainty**: Should decrease as training progresses
- **Effective batch size**: Should stabilize around `minimal_eff_bs` when dynamic_xi=True

## Troubleshooting

### Environments Not Found
```bash
# Install missing Gym environments
pip install gymnasium[all]
```

### WandB Login Issues
```bash
wandb login
# Then paste your API key when prompted
```

### Tmux Not Found
```bash
# Ubuntu/Debian
sudo apt-get install tmux

# MacOS
brew install tmux
```

### Out of Memory
Reduce the number of parallel experiments:
```bash
# Edit run_ppo_iv_experiments.sh
# Comment out some environments or reduce SEEDS array
```

### Session Already Exists
```bash
# Kill existing session before restarting
tmux kill-session -t <session_name>
```

## Citation

If using this implementation, please cite:

**IV-RL Paper**:
```
Sample Efficient Deep Reinforcement Learning via Uncertainty Estimation
OpenReview: https://openreview.net/forum?id=vrW3tvDfOJQ
```

**PPO Paper**:
```
Schulman et al. (2017). Proximal Policy Optimization Algorithms.
arXiv:1707.06347
```

## File Structure
```
iv_rl/
├── config.py                          # Hyperparameter configurations
├── main.py                            # Main entry point
├── ppo/
│   ├── __init__.py                   # Module exports
│   ├── ppo.py                        # Baseline PPO
│   ├── networks.py                   # Value networks (with variance head)
│   ├── ensemblePPO.py                # Ensemble PPO (epistemic uncertainty)
│   └── iv_ppo.py                     # IV-PPO (full method)
└── scripts/
    ├── run_ppo_iv_experiments.sh     # Main experimental script
    ├── kill_all_experiments.sh       # Utility: stop all experiments
    └── check_experiments.sh          # Utility: check session status
```

## Quick Reference Commands

```bash
# Full experimental suite
cd /home/ammiellewb/iv_rl/scripts && ./run_ppo_iv_experiments.sh

# Check running experiments
./check_experiments.sh

# Monitor specific experiment
tmux attach -t ivppo_Pendulum-v1_seed0

# Stop all experiments
./kill_all_experiments.sh

# View WandB results
# Visit: https://wandb.ai/<username>/iv-rl
```
