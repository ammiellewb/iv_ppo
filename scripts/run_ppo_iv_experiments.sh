#!/bin/bash

# IV-RL PPO Experimental Script
# Runs baseline, ensemble, and IV-PPO experiments with ablations
# Uses tmux for long-running experiments and WandB for tracking

# ============================================================================
# SETUP
# ============================================================================

# WandB Configuration
# Project name as specified by user
export WANDB_PROJECT="iv-rl"

# Check if WandB API key is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY not set. Please export WANDB_API_KEY before running."
    echo "Example: export WANDB_API_KEY='your_api_key_here'"
    exit 1
fi

# Login to WandB (suppress output if already logged in)
wandb login $WANDB_API_KEY 2>/dev/null || echo "WandB login attempted..."

# Project root directory
PROJECT_ROOT="/home/ammiellewb/iv_rl"
cd $PROJECT_ROOT

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

# Continuous control environments
ENVS=(
    "Pendulum-v0"
    "HalfCheetah-v3"
    "Hopper-v3"
    "Walker2d-v3"
    "Ant-v3"
)

# Seeds for reproducibility
SEEDS=(0 1 2 3 4)

# Episode counts (adjusted for continuous control)
EPISODES_PENDULUM=500
EPISODES_MUJOCO=1000

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Function to create and run experiment in tmux session
run_experiment() {
    local session_name=$1
    local env=$2
    local model=$3
    local seed=$4
    local net_seed=$5
    local tag=$6
    local extra_args=$7
    
    # Determine episode count based on environment
    local num_episodes
    if [ "$env" = "Pendulum-v1" ]; then
        num_episodes=$EPISODES_PENDULUM
    else
        num_episodes=$EPISODES_MUJOCO
    fi
    
    # Create tmux session
    tmux new-session -d -s "$session_name"
    
    # Run experiment in tmux session
    tmux send-keys -t "$session_name" "cd $PROJECT_ROOT" C-m
    tmux send-keys -t "$session_name" "export WANDB_API_KEY=$WANDB_API_KEY" C-m
    tmux send-keys -t "$session_name" "export WANDB_PROJECT=iv-rl" C-m
    tmux send-keys -t "$session_name" "python main.py \
        --env $env \
        --model $model \
        --env_seed $seed \
        --net_seed $net_seed \
        --num_episodes $num_episodes \
        --tag $tag \
        $extra_args" C-m
    
    echo "Started experiment: $session_name (env=$env, model=$model, seed=$seed)"
}

# Function to check if tmux session exists
session_exists() {
    tmux has-session -t "$1" 2>/dev/null
}

# ============================================================================
# BASELINE EXPERIMENTS
# ============================================================================

echo "==============================================="
echo "BASELINE EXPERIMENTS: Standard PPO"
echo "==============================================="

for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        session_name="ppo_${env}_seed${seed}"
        tag="baseline_ppo"
        
        run_experiment "$session_name" "$env" "PPO" "$seed" "$seed" "$tag" ""
        
        # Small delay to avoid overwhelming the system
        sleep 2
    done
done

echo "Baseline experiments started."
echo ""

# ============================================================================
# ENSEMBLE EXPERIMENTS (Ablation: Epistemic Uncertainty Only)
# ============================================================================

echo "==============================================="
echo "ABLATION: EnsemblePPO (Epistemic Uncertainty)"
echo "==============================================="

for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        session_name="ensemble_${env}_seed${seed}"
        tag="ablation_ensemble"
        
        run_experiment "$session_name" "$env" "EnsemblePPO" "$seed" "$seed" "$tag" ""
        
        sleep 2
    done
done

echo "Ensemble ablation experiments started."
echo ""

# ============================================================================
# IV-PPO EXPERIMENTS: Full Method
# ============================================================================

echo "==============================================="
echo "IV-PPO: Full Method (Default Hyperparameters)"
echo "==============================================="

for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        session_name="ivppo_${env}_seed${seed}"
        tag="iv_ppo_full"
        
        run_experiment "$session_name" "$env" "IV_PPO" "$seed" "$seed" "$tag" ""
        
        sleep 2
    done
done

echo "IV-PPO full method experiments started."
echo ""

# ============================================================================
# ABLATION: Dynamic Xi On/Off
# ============================================================================

echo "==============================================="
echo "ABLATION: Dynamic Xi (On vs Off)"
echo "==============================================="

# Test on subset of environments (to save compute)
ABLATION_ENVS=("Pendulum-v0" "HalfCheetah-v3" "Hopper-v3")

for env in "${ABLATION_ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # Dynamic xi OFF (fixed xi=0.1)
        session_name="ivppo_fixedxi_${env}_seed${seed}"
        tag="ablation_fixed_xi"
        
        run_experiment "$session_name" "$env" "IV_PPO" "$seed" "$seed" "$tag" "--dynamic_xi False --xi 0.1"
        
        sleep 2
    done
done

echo "Dynamic xi ablation experiments started."
echo ""

# ============================================================================
# ABLATION: Lambda BIV (Loss Weighting)
# ============================================================================

echo "==============================================="
echo "ABLATION: Lambda BIV (Loss Weighting)"
echo "==============================================="

# Test different lambda values: 0.25, 0.5 (default), 0.75
LAMBDA_VALUES=(0.25 0.75)

for env in "${ABLATION_ENVS[@]}"; do
    for lambda_val in "${LAMBDA_VALUES[@]}"; do
        for seed in 0 1 2; do  # Use fewer seeds for lambda ablation
            session_name="ivppo_lambda${lambda_val}_${env}_seed${seed}"
            tag="ablation_lambda_${lambda_val}"
            
            run_experiment "$session_name" "$env" "IV_PPO" "$seed" "$seed" "$tag" "--lambda_biv $lambda_val"
            
            sleep 2
        done
    done
done

echo "Lambda BIV ablation experiments started."
echo ""

# ============================================================================
# ABLATION: Ensemble Size
# ============================================================================

echo "==============================================="
echo "ABLATION: Ensemble Size (3, 5, 10)"
echo "==============================================="

# Test different ensemble sizes
ENSEMBLE_SIZES=(3 10)  # 5 is default

for env in "${ABLATION_ENVS[@]}"; do
    for num_nets in "${ENSEMBLE_SIZES[@]}"; do
        for seed in 0 1 2; do
            session_name="ivppo_nets${num_nets}_${env}_seed${seed}"
            tag="ablation_ensemble_size_${num_nets}"
            
            run_experiment "$session_name" "$env" "IV_PPO" "$seed" "$seed" "$tag" "--num_nets $num_nets"
            
            sleep 2
        done
    done
done

echo "Ensemble size ablation experiments started."
echo ""

# ============================================================================
# MONITORING AND UTILITY COMMANDS
# ============================================================================

echo "==============================================="
echo "ALL EXPERIMENTS STARTED"
echo "==============================================="
echo ""
echo "To monitor experiments:"
echo "  - List all sessions: tmux ls"
echo "  - Attach to session: tmux attach -t <session_name>"
echo "  - Detach from session: Ctrl+B then D"
echo "  - Kill session: tmux kill-session -t <session_name>"
echo ""
echo "To monitor WandB:"
echo "  - Visit: https://wandb.ai/<your-username>/iv-rl"
echo ""
echo "Tracked Metrics:"
echo "  - episode_reward: Cumulative reward per episode"
echo "  - episode_length: Steps per episode"
echo "  - policy_loss: Actor loss"
echo "  - value_loss: Total critic loss"
echo "  - biv_loss: BIV-weighted MSE loss (IV-PPO only)"
echo "  - la_loss: Loss attenuation (IV-PPO only)"
echo "  - effective_batch_size: Effective batch size from BIV weights"
echo "  - mean_aleatoric_var: Average aleatoric uncertainty"
echo "  - mean_epistemic_var: Average epistemic uncertainty"
echo ""
echo "Sample Efficiency Analysis:"
echo "  - Compare episode_reward vs environment steps across methods"
echo "  - Look for faster convergence in IV-PPO"
echo ""
echo "Learning Stability Analysis:"
echo "  - Compare variance of episode_reward across seeds"
echo "  - Look for lower variance and smoother learning curves in IV-PPO"
echo ""

# ============================================================================
# UTILITY SCRIPT: Kill all experiment sessions
# ============================================================================

cat > kill_all_experiments.sh << 'EOF'
#!/bin/bash
# Kill all tmux sessions related to experiments
echo "Killing all experiment tmux sessions..."
tmux ls | grep -E "(ppo_|ensemble_|ivppo_)" | cut -d: -f1 | xargs -I {} tmux kill-session -t {}
echo "Done."
EOF

chmod +x kill_all_experiments.sh

echo "Created utility script: kill_all_experiments.sh"
echo "Run './kill_all_experiments.sh' to stop all experiments."
echo ""

# ============================================================================
# SUMMARY SCRIPT: Check experiment status
# ============================================================================

cat > check_experiments.sh << 'EOF'
#!/bin/bash
# Check status of all experiment sessions
echo "Active experiment sessions:"
echo "----------------------------"
tmux ls 2>/dev/null | grep -E "(ppo_|ensemble_|ivppo_)" || echo "No active experiment sessions found."
echo ""
echo "Total active sessions: $(tmux ls 2>/dev/null | grep -E '(ppo_|ensemble_|ivppo_)' | wc -l)"
EOF

chmod +x check_experiments.sh

echo "Created utility script: check_experiments.sh"
echo "Run './check_experiments.sh' to view active experiments."
echo ""
echo "==============================================="
