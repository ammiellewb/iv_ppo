# IV-RL Experimental Setup Summary

## What Was Done

This setup implements comprehensive experiments for evaluating **Inverse-Variance Reinforcement Learning (IV-RL)** adapted to Proximal Policy Optimization (PPO) on continuous control tasks.

### Implementation Complete ✓

1. **ValueNetworkWithVariance** (`ppo/networks.py`)
   - Dual-head architecture for value estimation + variance prediction
   - Learns aleatoric (data) uncertainty via log-variance head
   - Used by IV-PPO for loss attenuation

2. **EnsemblePPO** (`ppo/ensemblePPO.py`)
   - K parallel value networks with soft target updates
   - Estimates epistemic (model) uncertainty via ensemble disagreement
   - Base class for IV-PPO

3. **IV_PPO** (`ppo/iv_ppo.py`)
   - Full IV-RL methodology combining both uncertainties
   - BIV weighting: w_i = 1/(σ²_total,i + ξ)
   - Loss attenuation: L_LA = 0.5*log(σ²) + 0.5*((y-V)/σ)²
   - Combined loss: L = λ*L_BIV + (1-λ)*L_LA
   - Dynamic xi optimization for target effective batch size

4. **Environment Configurations** (`config.py`)
   - Pendulum-v1: Simple continuous control
   - HalfCheetah-v4, Hopper-v4, Walker2d-v4, Ant-v4: MuJoCo locomotion
   - Tuned hyperparameters per environment

5. **Experimental Infrastructure**
   - `run_ppo_iv_experiments.sh`: Full experimental suite with tmux
   - `quick_test.sh`: Quick validation on Pendulum
   - WandB integration with project name "iv-rl"
   - Automated tracking of sample efficiency and learning stability

---

## Experiment Design

### Primary Comparisons

**1. Baseline PPO** (Standard method)
- Single value network
- No uncertainty estimation
- Serves as reference for sample efficiency

**2. EnsemblePPO** (Ablation: Epistemic only)
- Ensemble of K=5 value networks
- Epistemic uncertainty from disagreement
- Tests if ensemble alone improves performance

**3. IV-PPO** (Full method)
- Ensemble + variance head
- Both aleatoric and epistemic uncertainty
- BIV weighting + loss attenuation
- Expected to show best sample efficiency and stability

### Environments

| Environment | Type | Difficulty | Action Space | Observation Space |
|------------|------|------------|--------------|-------------------|
| Pendulum-v1 | Swing-up | Easy | Continuous(1) | Continuous(3) |
| HalfCheetah-v4 | Locomotion | Medium | Continuous(6) | Continuous(17) |
| Hopper-v4 | Balance+Walk | Medium | Continuous(3) | Continuous(11) |
| Walker2d-v4 | Bipedal Walk | Hard | Continuous(6) | Continuous(17) |
| Ant-v4 | Quadruped | Hard | Continuous(8) | Continuous(27) |

### Ablations

**1. Dynamic Xi**
- Hypothesis: Optimizing xi to maintain target effective batch size improves stability
- Comparison: `dynamic_xi=True` vs `dynamic_xi=False` (fixed xi=0.1)
- Environments: Pendulum, HalfCheetah, Hopper

**2. Lambda BIV (Loss Weighting)**
- Hypothesis: Balanced weighting (λ=0.5) optimal for combining BIV and LA
- Values tested: λ ∈ {0.25, 0.5, 0.75}
- Environments: Pendulum, HalfCheetah, Hopper

**3. Ensemble Size**
- Hypothesis: K=5 balances uncertainty estimation vs computational cost
- Values tested: K ∈ {3, 5, 10}
- Environments: Pendulum, HalfCheetah, Hopper

---

## Tracked Metrics

### Sample Efficiency (Primary)
- **episode_reward**: Cumulative reward per episode
- **episode_length**: Steps per episode
- **episode**: Episode counter
- **Analysis**: Plot reward vs total environment steps; compare convergence speed

### Learning Stability (Primary)
- Compute variance of `episode_reward` across random seeds
- Expected: IV-PPO shows lower variance, smoother learning curves
- **Analysis**: Error bands (mean ± std) across seeds

### Algorithmic Insights
- **policy_loss**: Actor network loss
- **value_loss**: Critic loss (BIV + LA for IV-PPO)
- **biv_loss**: BIV-weighted MSE component
- **la_loss**: Loss attenuation component
- **effective_batch_size**: Effective samples after BIV weighting
- **xi**: Regularization parameter value
- **entropy**: Policy entropy (exploration)
- **approx_kl**: Policy update magnitude

---

## Usage

### Quick Test (Single Environment)
```bash
export WANDB_API_KEY='your_key_here'
cd /home/ammiellewb/iv_rl/scripts
./quick_test.sh
```
Runs PPO, EnsemblePPO, IV-PPO on Pendulum-v1 (500 episodes each, ~10-15 min total).

### Full Experimental Suite
```bash
export WANDB_API_KEY='your_key_here'
cd /home/ammiellewb/iv_rl/scripts
./run_ppo_iv_experiments.sh
```
Launches ~115 tmux sessions:
- 5 environments × 3 methods × 5 seeds = 75 base experiments
- ~40 ablation experiments
- Total runtime: 24-48 hours on multi-GPU system

### Custom Single Run
```bash
export WANDB_PROJECT="iv-rl"
python main.py \
    --env Hopper-v4 \
    --model IV_PPO \
    --env_seed 0 \
    --net_seed 0 \
    --num_episodes 1000 \
    --tag my_experiment
```

### Manage Tmux Sessions
```bash
# List all active experiments
./check_experiments.sh

# Attach to specific experiment (monitor progress)
tmux attach -t ivppo_Pendulum-v1_seed0

# Detach: Ctrl+B then D

# Kill all experiments
./kill_all_experiments.sh
```

---

## Expected Results

### Hypothesis 1: Sample Efficiency
**IV-PPO converges 20-40% faster than baseline PPO**

Mechanism:
- BIV weighting down-weights high-uncertainty (noisy) transitions
- Effective batch contains more informative samples
- Faster, more stable policy updates

Evidence:
- Plot: Episode reward vs environment steps (sooner plateau)
- Metric: Episodes to reach 90% of expert performance

### Hypothesis 2: Learning Stability
**IV-PPO shows 30-50% lower variance across seeds**

Mechanism:
- Loss attenuation prevents over-fitting to outliers
- Uncertainty-aware updates smooth the learning process
- Ensemble reduces sensitivity to initialization

Evidence:
- Plot: Error bands (mean ± std) narrower for IV-PPO
- Metric: Std. deviation of final performance across 5 seeds

### Hypothesis 3: Uncertainty Evolution
**Epistemic uncertainty decreases, aleatoric remains stable**

Mechanism:
- Epistemic (model) uncertainty: high initially, decreases as model learns
- Aleatoric (data) uncertainty: inherent to environment, remains constant

Evidence:
- Plot: Epistemic variance vs episodes (decreasing trend)
- Plot: Aleatoric variance vs episodes (stable)

---

## Files Created/Modified

### New Files
```
ppo/
├── ensemblePPO.py          # Ensemble-based PPO (546 lines)
├── iv_ppo.py               # IV-RL PPO implementation (492 lines)

scripts/
├── run_ppo_iv_experiments.sh  # Main experimental script
├── quick_test.sh              # Quick validation script
├── kill_all_experiments.sh    # Auto-generated utility
└── check_experiments.sh       # Auto-generated utility

EXPERIMENTS_GUIDE.md           # User guide
EXPERIMENT_SUMMARY.md          # This file
```

### Modified Files
```
ppo/
├── networks.py             # Added ValueNetworkWithVariance class
├── __init__.py            # Updated exports
└── ppo.py                 # Updated WandB init

config.py                  # Added configs for Pendulum, MuJoCo envs
main.py                    # Already had get_ppo_dict() registration
```

---

## WandB Integration

### Project Setup
- **Project name**: `iv-rl` (set via `WANDB_PROJECT` env var)
- **Run naming**: `{env}_{model}_seed{seed}` (e.g., `Pendulum-v1_IV_PPO_seed0`)
- **Tags**: `[tag, env, model]` for filtering

### Dashboard Access
`https://wandb.ai/<your-username>/iv-rl`

### Recommended Visualizations

**1. Sample Efficiency Comparison**
```
X-axis: episode (or episode × episode_length for total steps)
Y-axis: episode_reward
Group by: model (PPO, EnsemblePPO, IV_PPO)
Aggregate: mean ± std across seeds
```

**2. Learning Curves by Environment**
```
Separate panels for each environment
Compare all three methods
Show confidence intervals
```

**3. Effective Batch Size Evolution (IV-PPO only)**
```
X-axis: episode
Y-axis: effective_batch_size
Expected: stabilizes around minimal_eff_bs=48
```

**4. Loss Components (IV-PPO only)**
```
X-axis: episode
Y-axis: biv_loss and la_loss
Show both on same plot (dual axis if needed)
```

**5. Ablation Studies**
```
Create separate workspace sections for:
- Dynamic xi: fixed vs dynamic
- Lambda BIV: 0.25 vs 0.5 vs 0.75
- Ensemble size: K=3 vs K=5 vs K=10
```

---

## Troubleshooting

### Import Errors
```bash
# Make sure all dependencies installed
pip install wandb gymnasium torch numpy
pip install gymnasium[mujoco]  # For HalfCheetah, Hopper, Walker2d, Ant
```

### WandB Not Logging
```bash
# Verify environment variable set
echo $WANDB_PROJECT  # Should output: iv-rl

# Verify tag is provided (required for logging)
python main.py --env Pendulum-v1 --model PPO --tag test  # ✓ Will log
python main.py --env Pendulum-v1 --model PPO              # ✗ No logging
```

### Tmux Session Conflicts
```bash
# If session already exists
tmux kill-session -t ivppo_Pendulum-v1_seed0

# Or kill all at once
./kill_all_experiments.sh
```

### Out of Memory
```bash
# Reduce parallelism in run_ppo_iv_experiments.sh
# Edit SEEDS array: SEEDS=(0 1 2)  # Use 3 instead of 5 seeds
```

---

## Next Steps

1. **Validate Setup**
   ```bash
   ./quick_test.sh
   # Wait ~15 min, verify 3 runs appear in WandB
   ```

2. **Run Full Suite**
   ```bash
   ./run_ppo_iv_experiments.sh
   # Monitor with: ./check_experiments.sh
   ```

3. **Analyze Results**
   - WandB dashboard: Compare learning curves
   - Download data: `wandb export iv-rl`
   - Statistical tests: Compare mean performance across seeds

4. **Iterate**
   - If results unclear: Try more seeds or longer training
   - If one environment dominates: Adjust episode counts
   - If ablations inconclusive: Test wider hyperparameter ranges

---

## Paper Reference

**Title**: Sample Efficient Deep Reinforcement Learning via Uncertainty Estimation  
**Link**: https://openreview.net/forum?id=vrW3tvDfOJQ  
**Key Contributions**:
- Bayesian framework combining aleatoric and epistemic uncertainty
- BIV weighting for sample-efficient updates
- Loss attenuation for robustness to noisy labels
- Dynamic xi optimization for adaptive regularization

This implementation faithfully adapts the IV-RL methodology from DQN/SAC to the on-policy PPO algorithm, maintaining the core principles while adapting to the actor-critic architecture and rollout-based training of PPO.

---

## Contact & Support

For issues or questions:
1. Check `EXPERIMENTS_GUIDE.md` for detailed usage instructions
2. Verify configurations in `config.py`
3. Review code implementation in `ppo/iv_ppo.py`
4. Check WandB logs for runtime errors or unexpected metrics

Good luck with your experiments! 🚀
