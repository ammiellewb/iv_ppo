#!/bin/bash

# Quick test script for IV-RL PPO on Pendulum environment
# This runs a single experiment of each method for validation

# WandB Configuration
export WANDB_PROJECT="iv-rl"

if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY not set. Please export WANDB_API_KEY before running."
    echo "Example: export WANDB_API_KEY='your_api_key_here'"
    exit 1
fi

# Login to WandB
wandb login $WANDB_API_KEY 2>/dev/null || echo "WandB login attempted..."

# Project root
PROJECT_ROOT="/home/ammiellewb/iv_rl"
cd $PROJECT_ROOT

echo "==============================================="
echo "Quick Test: Pendulum-v0 (500 episodes)"
echo "Running: PPO, EnsemblePPO, IV_PPO"
echo "==============================================="
echo ""

# Test Baseline PPO
echo "1/3 Testing Baseline PPO..."
python main.py \
    --env Pendulum-v0 \
    --model PPO \
    --env_seed 0 \
    --net_seed 0 \
    --num_episodes 500 \
    --tag test_baseline &

PPO_PID=$!

# Test EnsemblePPO
echo "2/3 Testing EnsemblePPO..."
python main.py \
    --env Pendulum-v0 \
    --model EnsemblePPO \
    --env_seed 0 \
    --net_seed 0 \
    --num_episodes 500 \
    --tag test_ensemble &

ENSEMBLE_PID=$!

# Test IV-PPO
echo "3/3 Testing IV-PPO..."
python main.py \
    --env Pendulum-v0 \
    --model IV_PPO \
    --env_seed 0 \
    --net_seed 0 \
    --num_episodes 500 \
    --tag test_ivppo &

IV_PID=$!

echo ""
echo "All tests launched in background!"
echo "PIDs: PPO=$PPO_PID, Ensemble=$ENSEMBLE_PID, IV-PPO=$IV_PID"
echo ""
echo "Monitor progress:"
echo "  - View processes: ps -p $PPO_PID,$ENSEMBLE_PID,$IV_PID"
echo "  - View WandB: https://wandb.ai/<username>/iv-rl"
echo ""
echo "Wait for completion (this will block until all finish):"

# Wait for all background jobs
wait $PPO_PID
echo "✓ Baseline PPO complete"

wait $ENSEMBLE_PID
echo "✓ EnsemblePPO complete"

wait $IV_PID
echo "✓ IV-PPO complete"

echo ""
echo "==============================================="
echo "All tests complete! Check WandB for results."
echo "==============================================="
