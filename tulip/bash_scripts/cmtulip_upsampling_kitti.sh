#!/bin/bash

# Use the new configuration system
# The script now uses a YAML config file - all parameters are in the config

# Usage: script.sh <NPROC> [eval]
if [[ -z "$1" ]]; then
    echo "Usage: $0 <NPROC> [eval]"
    exit 1
fi
NPROC="$1"
if ! [[ "$NPROC" =~ ^[0-9]+$ ]] || [[ "$NPROC" -lt 1 ]]; then
    echo "Error: <NPROC> must be a positive integer"
    exit 1
fi

args=(
    --config ./configs/cmtulip_kitti_config.yaml
    )

# Append --eval if second argument is 'eval'
if [[ "${2:-}" == "eval" ]]; then
    args+=( --eval )
fi

# real batch size in training = batch_size * nproc_per_node
torchrun --nproc_per_node="$NPROC" tulip/main_lidar_upsampling.py "${args[@]}"