#!/bin/bash

# Use the new configuration system
# The script now uses a YAML config file - all parameters are in the config

args=(
    --config ./configs/tulip_kitti_config.yaml
    )

# real batch size in training = batch_size * nproc_per_node
torchrun --nproc_per_node=1 tulip/main_lidar_upsampling.py "${args[@]}"