#!/bin/bash

args=(
    --num_data_train 1000
    --num_data_val 1000
    --output_path_name_train train
    --output_path_name_val val
    --nuscenes_dataroot ./data/nuscenes/
    --nuscenes_version v1.0-trainval
    --create_val
    )

python ./nuscenes_utils/sample_nuscenes_dataset.py "${args[@]}"
