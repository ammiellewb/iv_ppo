# IV-PPO Experimental Results

## Experimental Setup

We evaluate IV-PPO against two baselines across four continuous control environments:

1. **PPO**: Standard Proximal Policy Optimization with a single value network
2. **EnsemblePPO**: PPO with K=5 ensemble value networks (epistemic uncertainty only, no variance head)
3. **IV-PPO**: Full inverse-variance method with ensemble + variance heads (both epistemic and aleatoric uncertainty)

All experiments use 5 random seeds per method. Final performance is computed as the mean return over the last 100 episodes. Error bars and shaded regions represent 95% confidence intervals computed using the t-distribution.

### Environments

| Environment | State Dim | Action Dim | Characteristics |
|-------------|-----------|------------|-----------------|
| Pendulum-v1 | 3 | 1 | Deterministic dynamics, homoscedastic rewards |
| LunarLanderContinuous-v2 | 8 | 2 | Stochastic wind, heteroscedastic transitions |
| HalfCheetah-v4 | 17 | 6 | High-dimensional, contact-rich dynamics |
| Walker2d-v4 | 17 | 6 | Balance task with discrete contact events |

---

## Learning Curves

### LunarLanderContinuous-v2

![LunarLanderContinuous-v2 Learning Curves](figures/LunarLanderContinuous-v2_learning_curves.png)

### Pendulum-v1

![Pendulum-v1 Learning Curves](figures/Pendulum-v1_learning_curves.png)

### HalfCheetah-v4

![HalfCheetah-v4 Learning Curves](figures/HalfCheetah-v4_learning_curves.png)

### Walker2d-v4

![Walker2d-v4 Learning Curves](figures/Walker2d-v4_learning_curves.png)

### Combined View

![Combined Learning Curves](figures/combined_learning_curves.png)

---

## Final Performance Summary

![Final Performance Comparison](figures/final_performance_comparison.png)

### Numerical Results (Mean ± Std over 5 seeds, last 100 episodes)

| Environment | PPO | IV-PPO | Improvement | EnsemblePPO |
|-------------|-----|--------|-------------|-------------|
| Pendulum-v1 | -1396.5 ± 195.0 | -1398.1 ± 160.0 | -0.1% | -1272.4 ± 175.7 (+8.9%) |
| LunarLanderContinuous-v2 | 106.1 ± 24.8 | 129.3 ± 29.9 | **+21.9%** | 121.8 ± 21.2 (+14.8%) |
| HalfCheetah-v4 | -37744.6 ± 434.3 | -39197.1 ± 580.1 | -3.8% | -37720.5 ± 277.4 (+0.1%) |
| Walker2d-v4 | 151.9 ± 0.6 | 154.7 ± 1.4 | +1.8% | 153.0 ± 1.7 (+0.7%) |

---

## Analysis by Environment

### LunarLanderContinuous-v2: Heteroscedastic Environment

LunarLanderContinuous-v2 includes stochastic wind mechanics. The numerical results show:

- **IV-PPO: 129.3 ± 29.9** (+21.9% vs PPO)
- **EnsemblePPO: 121.8 ± 21.2** (+14.8% vs PPO)
- **PPO: 106.1 ± 24.8**

IV-PPO achieves the highest mean return on this environment.

### Pendulum-v1: Homoscedastic Baseline

Pendulum-v1 has deterministic dynamics. Results show:

- **IV-PPO: -1398.1 ± 160.0** (-0.1% vs PPO)
- **EnsemblePPO: -1272.4 ± 175.7** (+8.9% vs PPO)
- **PPO: -1396.5 ± 195.0**

IV-PPO and PPO achieve comparable performance. EnsemblePPO achieves the highest mean return.

### HalfCheetah-v4: Contact-Rich Dynamics

HalfCheetah-v4 is a high-dimensional locomotion task. Results show:

- **IV-PPO: -39197.1 ± 580.1** (-3.8% vs PPO)
- **EnsemblePPO: -37720.5 ± 277.4** (+0.1% vs PPO)
- **PPO: -37744.6 ± 434.3**

All methods exhibit negative cumulative returns. IV-PPO underperforms both baselines on this environment.

### Walker2d-v4: Balance Task

Walker2d-v4 is a bipedal locomotion task. Results show:

- **IV-PPO: 154.7 ± 1.4** (+1.8% vs PPO)
- **EnsemblePPO: 153.0 ± 1.7** (+0.7% vs PPO)
- **PPO: 151.9 ± 0.6**

All methods achieve similar final performance with low variance across seeds.

---

## Method Comparison

The three methods differ in their uncertainty estimation components:

| Method | Epistemic (Ensemble) | Aleatoric (Variance Head) | BIV Weighting | Loss Attenuation |
|--------|---------------------|---------------------------|---------------|------------------|
| PPO | ✗ | ✗ | ✗ | ✗ |
| EnsemblePPO | ✓ | ✗ | ✗ | ✗ |
| IV-PPO | ✓ | ✓ | ✓ | ✓ |

### Observed Results

**LunarLanderContinuous-v2**:
- IV-PPO: +21.9% vs PPO
- EnsemblePPO: +14.8% vs PPO

**Pendulum-v1**:
- IV-PPO: -0.1% vs PPO
- EnsemblePPO: +8.9% vs PPO

**HalfCheetah-v4**:
- IV-PPO: -3.8% vs PPO
- EnsemblePPO: +0.1% vs PPO

**Walker2d-v4**:
- IV-PPO: +1.8% vs PPO
- EnsemblePPO: +0.7% vs PPO

IV-PPO shows the largest improvement on LunarLanderContinuous-v2. Performance varies across environments, with IV-PPO underperforming on HalfCheetah-v4.

---

## Discussion

### Cross-Seed Variance

Standard deviations across 5 seeds (from the data):
- LunarLander: IV-PPO std = 29.9, PPO std = 24.8
- Pendulum: IV-PPO std = 160.0, PPO std = 195.0
- HalfCheetah: IV-PPO std = 580.1, PPO std = 434.3
- Walker2d: IV-PPO std = 1.4, PPO std = 0.6

---

## Limitations

1. **Limited environment coverage**: Results are based on 4 environments; broader evaluation across more heteroscedastic benchmarks would strengthen conclusions.

2. **No direct uncertainty visualization**: While the method estimates epistemic and aleatoric variance, these quantities were not logged during training, preventing direct analysis of how uncertainty evolves or correlates with learning progress.

3. **Hyperparameter sensitivity**: IV-PPO introduces additional hyperparameters (ξ, λ_BIV, τ) that may require environment-specific tuning for optimal results.

4. **Computational overhead**: IV-PPO requires ~1.2-1.4× the training time of PPO due to ensemble forward passes and variance computation.

---

## Summary

IV-PPO achieves +21.9% improvement over PPO on LunarLanderContinuous-v2, -0.1% on Pendulum-v1, -3.8% on HalfCheetah-v4, and +1.8% on Walker2d-v4. Uncertainty estimates (epistemic and aleatoric variance) were not logged during training, preventing direct analysis of how the weighting mechanism affects learning dynamics.
