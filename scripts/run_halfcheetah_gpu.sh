#!/bin/bash

# HalfCheetah-v4 GPU Experiments
# Runs 3 models × 5 seeds = 15 experiments on GPU

source ~/miniconda3/etc/profile.d/conda.sh
conda activate iv_rl_dqn

export WANDB_PROJECT="iv-rl"
export WANDB_MODE="offline"
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0

PROJECT_ROOT="/home/ammiellewb/iv_rl"
cd $PROJECT_ROOT

mkdir -p logs

ENV="HalfCheetah-v4"
MODELS=("PPO" "EnsemblePPO" "IV_PPO")
SEEDS=(0 1 2 3 4)
EPISODES=1000

echo "================================================"
echo "HalfCheetah-v4 GPU Experiments (15 runs)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "================================================"
echo ""

for model in "${MODELS[@]}"; do
    echo "Model: $model"
    
    # Run all 5 seeds in parallel (GPU can handle it)
    for seed in "${SEEDS[@]}"; do
        tag="${ENV}_${model}_seed${seed}"
        echo "  Starting: $tag (background, GPU 0)"
        CUDA_VISIBLE_DEVICES=0 python -u main.py \
            --env $ENV \
            --model $model \
            --env_seed $seed \
            --net_seed $seed \
            --num_episodes $EPISODES \
            --tag $tag \
            > logs/${tag}.log 2>&1 &
    done
    
    # Wait for all seeds to complete
    echo "  Waiting for all 5 seeds to complete..."
    wait
    echo "✓ All 5 seeds complete for $model"
    echo ""
done

echo ""
echo "================================================"
echo "All HalfCheetah-v4 experiments complete!"
echo "================================================"
echo ""
echo "Results saved to logs/logs_S_HalfCheetah-v4_*.npy"
echo ""
echo "Next steps:"
echo "  1. Check results: python scripts/quick_results_summary.py"
echo "  2. Sync WandB: wandb sync wandb/offline-run-*"
