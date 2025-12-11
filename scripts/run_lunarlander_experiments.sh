#!/bin/bash

# LunarLanderContinuous-v2 Experimental Suite
# Runs 3 models × 5 seeds = 15 experiments in parallel batches

source ~/miniconda3/etc/profile.d/conda.sh
conda activate iv_rl_dqn

export WANDB_PROJECT="iv-rl"
export WANDB_MODE="offline"

PROJECT_ROOT="/home/ammiellewb/iv_rl"
cd $PROJECT_ROOT

mkdir -p logs

ENV="LunarLanderContinuous-v2"
MODELS=("PPO" "EnsemblePPO" "IV_PPO")
SEEDS=(0 1 2 3 4)

echo "================================================"
echo "LunarLanderContinuous-v3 Experiments (15 runs)"
echo "================================================"
echo ""

for MODEL in "${MODELS[@]}"; do
    echo "Model: $MODEL"
    
    # Run seeds 0-2 in parallel
    echo "  Starting seeds 0, 1, 2 (background)..."
    for SEED in 0 1 2; do
        TAG="${ENV}_${MODEL}_seed${SEED}"
        LOG_FILE="logs/${TAG}.log"
        
        python -u main.py \
            --env "$ENV" \
            --model "$MODEL" \
            --env_seed $SEED \
            --net_seed $SEED \
            --num_episodes 1000 \
            --tag "$TAG" \
            > "$LOG_FILE" 2>&1 &
        
        echo "    Started: $TAG (PID: $!)"
    done
    
    # Wait for first batch to complete
    echo "  Waiting for seeds 0-2 to complete..."
    wait
    echo "  ✓ Seeds 0,1,2 complete"
    
    # Run seeds 3-4 in parallel
    echo "  Starting seeds 3, 4 (background)..."
    for SEED in 3 4; do
        TAG="${ENV}_${MODEL}_seed${SEED}"
        LOG_FILE="logs/${TAG}.log"
        
        python -u main.py \
            --env "$ENV" \
            --model "$MODEL" \
            --env_seed $SEED \
            --net_seed $SEED \
            --num_episodes 1000 \
            --tag "$TAG" \
            > "$LOG_FILE" 2>&1 &
        
        echo "    Started: $TAG (PID: $!)"
    done
    
    # Wait for second batch to complete
    echo "  Waiting for seeds 3-4 to complete..."
    wait
    echo "  ✓ Seeds 3,4 complete"
    echo "✓ All seeds complete for $MODEL"
    echo ""
done

echo "================================================"
echo "All LunarLanderContinuous-v3 experiments complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Sync WandB data: wandb sync wandb/offline-run-*"
echo "  2. Analyze results: python scripts/analyze_results.py"
