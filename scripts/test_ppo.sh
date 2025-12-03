#!/bin/bash

# Test PPO on continuous control environment
echo "Testing PPO on gym_hopper (continuous actions)..."
python main.py --env gym_hopper --model PPO --net_seed 0 --env_seed 0 --num_episodes 10 --buffer_size 2048 --batch_size 64

# Test PPO on discrete control environment  
echo "Testing PPO on MountainCar-v0 (discrete actions)..."
python main.py --env MountainCar-v0 --model PPO --net_seed 0 --env_seed 0 --num_episodes 10 --buffer_size 2048 --batch_size 64
