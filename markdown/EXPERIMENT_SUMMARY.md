# IV-RL Experimental Setup Summary

## What Was Done

1. **Baseline PPO**: Standard Proximal Policy Optimization
2. **EnsemblePPO**: PPO with ensemble of value networks (epistemic uncertainty only)
3. **IV-PPO**: Full Inverse-Variance Reinforcement Learning with both uncertainties

---

## Implementation 

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
