#!/bin/bash

# Parallel experimental suite - runs 3 seeds simultaneously
# Faster execution while keeping CPU load reasonable

source ~/miniconda3/etc/profile.d/conda.sh
conda activate iv_rl_dqn

export WANDB_PROJECT="iv-rl"
export WANDB_MODE="offline"

PROJECT_ROOT="/home/ammiellewb/iv_rl"
cd $PROJECT_ROOT

SEEDS=(0 1 2 3 4)
ENVS=("Pendulum-v1")
MODELS=("PPO" "EnsemblePPO" "IV_PPO")
EPISODES=1000

echo "==============================================="
echo "Parallel Experimental Suite (3 concurrent jobs)"
echo "==============================================="
echo ""

mkdir -p logs

# Run experiments in batches of 3 (to avoid overloading CPU)
for env in "${ENVS[@]}"; do
    echo "Environment: $env"
    
    for model in "${MODELS[@]}"; do
        echo "  Model: $model"
        
        # Run seeds 0,1,2 in parallel
        for seed in 0 1 2; do
            tag="${env}_${model}_seed${seed}"
            echo "    Starting: $tag (background)"
            python -u main.py \
                --env $env \
                --model $model \
                --env_seed $seed \
                --net_seed $seed \
                --num_episodes $EPISODES \
                --tag $tag \
                > logs/${tag}.log 2>&1 &
        done
        
        # Wait for first batch
        wait
        echo "    ✓ Seeds 0,1,2 complete"
        
        # Run seeds 3,4 in parallel
        for seed in 3 4; do
            tag="${env}_${model}_seed${seed}"
            echo "    Starting: $tag (background)"
            python -u main.py \
                --env $env \
                --model $model \
                --env_seed $seed \
                --net_seed $seed \
                --num_episodes $EPISODES \
                --tag $tag \
                > logs/${tag}.log 2>&1 &
        done
        
        # Wait for second batch
        wait
        echo "    ✓ Seeds 3,4 complete"
        echo "  ✓ All seeds complete for $model"
        echo ""
    done
done

echo ""
echo "==============================================="
echo "All experiments complete!"
echo "==============================================="
