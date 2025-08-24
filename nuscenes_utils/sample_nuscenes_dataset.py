import numpy as np
import os
import argparse
import cv2
from glob import glob
import pathlib
import random
import shutil
import sys

# Add parent directory to path to import from kitti_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kitti_utils.sample_kitti_dataset import create_range_map, load_from_bin


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_data_train', type=int, default=21000)
    parser.add_argument('--num_data_val', type=int, default=2500)
    parser.add_argument("--input_path", type=str, default="/cluster/work/riner/users/biyang/dataset/nuscenes/")
    parser.add_argument("--output_path_name_train", type=str, default="nuscenes_train")
    parser.add_argument("--output_path_name_val", type=str, default="nuscenes_val")
    parser.add_argument("--create_val", action='store_true', default=False)
    parser.add_argument("--data_type", type=str, default="lidar", choices=["lidar", "radar"])
   
    return parser.parse_args()


def load_nuscenes_lidar_data(bin_path):
    """Load nuScenes LiDAR data from .bin file"""
    lidar_data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)  # nuScenes has 5 channels: x, y, z, intensity, ring_index
    # Return only x, y, z, intensity (ignore ring_index for compatibility with KITTI format)
    return lidar_data[:, :4]


def find_nuscenes_files(input_path, data_type="lidar"):
    """Find all nuScenes data files in the input directory"""
    if data_type == "lidar":
        # Look for .bin files in typical nuScenes directory structure
        search_pattern = os.path.join(input_path, "**", "*.bin")
        files = glob(search_pattern, recursive=True)
    else:  # radar
        # Look for .pcd files for radar data
        search_pattern = os.path.join(input_path, "**", "*.pcd")
        files = glob(search_pattern, recursive=True)
    
    return files


def main(args):
    num_data_train = args.num_data_train
    num_data_val = args.num_data_val
    dir_name = os.path.dirname(args.input_path)
    output_dir_name_train = os.path.join(dir_name, args.output_path_name_train)
    pathlib.Path(output_dir_name_train).mkdir(parents=True, exist_ok=True)
    
    if args.create_val:
        output_dir_name_val = os.path.join(dir_name, args.output_path_name_val)
        pathlib.Path(output_dir_name_val).mkdir(parents=True, exist_ok=True)

    # Find all available data files
    all_files = find_nuscenes_files(args.input_path, args.data_type)
    
    if len(all_files) == 0:
        print(f"No {args.data_type} files found in {args.input_path}")
        return
    
    print(f"Found {len(all_files)} {args.data_type} files")
    
    # Shuffle files for random sampling
    random.shuffle(all_files)
    
    # Split into train and validation
    if args.create_val:
        split_idx = int(len(all_files) * 0.8)  # 80% train, 20% val
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        
        # Sample from available files
        if num_data_train <= len(train_files):
            train_data = random.sample(train_files, num_data_train)
        else:
            # If we need more data than available, sample with replacement
            train_data = random.choices(train_files, k=num_data_train)
            
        if num_data_val <= len(val_files):
            val_data = random.sample(val_files, num_data_val)
        else:
            val_data = random.choices(val_files, k=num_data_val)
    else:
        # Only training data
        if num_data_train <= len(all_files):
            train_data = random.sample(all_files, num_data_train)
        else:
            train_data = random.choices(all_files, k=num_data_train)
        val_data = []

    print(f"Selected {len(train_data)} training files")
    if args.create_val:
        print(f"Selected {len(val_data)} validation files")

    # nuScenes LiDAR parameters (similar to KITTI but may need adjustment)
    image_rows = 32
    image_cols = 1024 # todo: check if this is correct
    ang_start_y = 10.67
    ang_res_y = 41.33 / (image_rows - 1)
    ang_res_x = 360 / image_cols
    max_range = 100
    min_range = 0
    
    # Process training data
    for i, data_path in enumerate(train_data):
        try:
            if args.data_type == "lidar":
                lidar_data = load_nuscenes_lidar_data(data_path)
            else:
                # For radar data, you might need a different loading function
                lidar_data = load_from_bin(data_path)  # Fallback to KITTI loader
            
            range_intensity_map = create_range_map(
                lidar_data, 
                image_rows_full=image_rows, 
                image_cols=image_cols, 
                ang_start_y=ang_start_y, 
                ang_res_y=ang_res_y, 
                ang_res_x=ang_res_x, 
                max_range=max_range, 
                min_range=min_range
            )

            output_filename = f'{i:08d}.npy'
            np.save(os.path.join(output_dir_name_train, output_filename), range_intensity_map.astype(np.float32))
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(train_data)} training files")
                
        except Exception as e:
            print(f"Error processing {data_path}: {e}")
            continue

    # Process validation data if requested
    if args.create_val:
        for j, data_path in enumerate(val_data):
            try:
                if args.data_type == "lidar":
                    lidar_data = load_nuscenes_lidar_data(data_path)
                else:
                    lidar_data = load_from_bin(data_path)
                
                range_intensity_map = create_range_map(
                    lidar_data, 
                    image_rows_full=image_rows, 
                    image_cols=image_cols, 
                    ang_start_y=ang_start_y, 
                    ang_res_y=ang_res_y, 
                    ang_res_x=ang_res_x, 
                    max_range=max_range, 
                    min_range=min_range
                )

                output_filename = f'{j:08d}.npy'
                np.save(os.path.join(output_dir_name_val, output_filename), range_intensity_map.astype(np.float32))
                
                if (j + 1) % 100 == 0:
                    print(f"Processed {j + 1}/{len(val_data)} validation files")
                    
            except Exception as e:
                print(f"Error processing {data_path}: {e}")
                continue

    print("Dataset creation completed!")


if __name__ == "__main__":
    args = read_args()
    main(args)

