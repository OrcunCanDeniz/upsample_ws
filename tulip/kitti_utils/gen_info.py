import argparse
import pickle
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from glob import glob
from scipy.spatial.transform import Rotation
import random
# Import functions from sample_kitti_dataset.py
from sample_kitti_dataset import create_range_map, load_from_bin


def read_calib_file(filepath):
    """Read calibration file and return dictionary of matrices."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                # Skip non-numeric values (like deprecated fields)
                continue
    
    # Reshape matrices
    for key, val in data.items():
        if key.startswith('R') or key.startswith('P'):
            # R matrices are 3x3, P matrices are 3x4
            if key.startswith('P'):
                data[key] = val.reshape(3, 4)
            else:
                data[key] = val.reshape(3, 3)
        elif key.startswith('T'):
            data[key] = val.reshape(3, 1)
        elif key.startswith('S') or key.startswith('D'):
            data[key] = val.reshape(-1, 1)
    
    return data


def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion [w, x, y, z] format."""
    rot = Rotation.from_matrix(R)
    quat = rot.as_quat()  # Returns [x, y, z, w]
    # Convert to [w, x, y, z] format
    return [quat[3], quat[0], quat[1], quat[2]]


def compute_cam2lidar(calib_velo_to_cam, calib_cam_to_cam):
    """
    Compute transformation from rectified left color camera (image_02) to lidar (velodyne).
    
    According to KITTI dataset developer guide:
    Forward transformation: y = P_rect_02 @ R_rect_00 @ T_velo_to_cam @ x
    Where:
    - x: point in velodyne coordinates [x, y, z, 1]'
    - T_velo_to_cam: velodyne -> cam0 (non-rectified): R_velo_to_cam0, T_velo_to_cam0
    - R_rect_00: cam0 (non-rectified) -> cam0 (rectified)
    - P_rect_02: cam0 (rectified) -> image plane
    
    Inverse transformation (rectified cam0 -> velo):
    1. cam0 (rectified) -> cam0 (non-rectified): R_rect_00.T
    2. cam0 (non-rectified) -> velo: 
       - Rotation: R_velo_to_cam0.T
       - Translation: -R_velo_to_cam0.T @ T_velo_to_cam0
    
    Args:
        calib_velo_to_cam: dict with velodyne-to-camera calibration
        calib_cam_to_cam: dict with camera-to-camera calibration (required)
    
    Returns:
        R: 3x3 rotation matrix (rectified camera to lidar)
        T: 3x1 translation vector (rectified camera to lidar)
        P_rect: 3x4 projection matrix (for camera intrinsic)
    """
    # Get velo to cam0 transformation (non-rectified)
    R_velo_to_cam0 = calib_velo_to_cam['R'].reshape(3, 3)
    T_velo_to_cam0 = calib_velo_to_cam['T'].reshape(3, 1)
    
    # Get rectification rotation: cam0 (non-rectified) -> cam0 (rectified)
    R_rect_00 = calib_cam_to_cam['R_rect_00'].reshape(3, 3)
    
    # Get projection matrix for rectified camera 02 (left color)
    P_rect = calib_cam_to_cam['P_rect_02'].reshape(3, 4)
    
    # Inverse transformation: rectified cam0 -> velo
    # Step 1: rectified cam0 -> non-rectified cam0 (inverse of R_rect_00)
    R_rect_to_nonrect = R_rect_00.T
    
    # Step 2: non-rectified cam0 -> velo (inverse of R_velo_to_cam0, T_velo_to_cam0)
    # If x_cam = R_velo_to_cam0 @ x_velo + T_velo_to_cam0
    # Then: x_velo = R_velo_to_cam0.T @ (x_cam - T_velo_to_cam0)
    #      = R_velo_to_cam0.T @ x_cam - R_velo_to_cam0.T @ T_velo_to_cam0
    R_cam0_to_velo = R_velo_to_cam0.T
    T_cam0_to_velo = -R_velo_to_cam0.T @ T_velo_to_cam0
    
    # Combine transformations: rectified cam0 -> velo
    # Apply rectification inverse first, then velo_to_cam inverse
    R_cam_to_velo = R_cam0_to_velo @ R_rect_to_nonrect
    
    # For translation: T_total = R_cam0_to_velo @ (R_rect_to_nonrect @ 0 + 0) + T_cam0_to_velo
    # Since rectification is rotation only (no translation), we have:
    T_cam_to_velo = T_cam0_to_velo
    
    return R_cam_to_velo, T_cam_to_velo, P_rect


def read_timestamps(timestamp_file):
    """Read timestamps from file. Handles both numeric and datetime string formats."""
    timestamps = []
    with open(timestamp_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Try to parse as datetime string (e.g., '2011-09-26 13:02:25.951199337')
                # Handle variable precision in microseconds
                if '.' in line:
                    # Has microseconds
                    dt = datetime.strptime(line, '%Y-%m-%d %H:%M:%S.%f')
                else:
                    # No microseconds
                    dt = datetime.strptime(line, '%Y-%m-%d %H:%M:%S')
                # Convert to Unix timestamp (seconds since epoch)
                timestamp = dt.timestamp()
                timestamps.append(timestamp)
            except ValueError:
                try:
                    # If datetime parsing fails, try as float
                    timestamps.append(float(line))
                except ValueError:
                    # Skip invalid lines
                    continue
    return timestamps


def generate_info(data_root, split_name, sampled_frames):
    """
    Generate info files for KITTI dataset.
    
    Args:
        data_root: Root directory of KITTI data
        split_name: 'train' or 'val'
        sampled_frames: List of (seq_path, lidar_file) tuples to process
    """
    # Create output directories for range view images
    split_rv_dir = os.path.join(data_root, split_name)
    split_head_target_dir = os.path.join(data_root, f'{split_name}_head_target')
    os.makedirs(split_rv_dir, exist_ok=True)
    os.makedirs(split_head_target_dir, exist_ok=True)
    print(f"Created directory: {split_rv_dir}")
    print(f"Processing {len(sampled_frames)} frames for {split_name} split")
    
    # Range image parameters (from sample_kitti_dataset.py)
    image_rows = 64
    image_cols = 1024
    ang_start_y = 24.8
    ang_res_y = 26.8 / (image_rows - 1)
    ang_res_x = 360 / image_cols
    max_range = 120
    min_range = 0
    
    infos = []
    # Only use left color camera (image_02)
    cam_name = 'image_02'
    
    # Cache calibration data per sequence to avoid re-reading
    calib_cache = {}
    timestamp_cache = {}
    
    for seq_path, lidar_file in tqdm(sampled_frames, desc=f"Processing {split_name} frames"):
        full_seq_path = os.path.join(data_root, seq_path)
        
        # Get or cache calibration files
        if seq_path not in calib_cache:
            # Find calibration files (they might be in parent directory)
            date_folder = os.path.dirname(full_seq_path)
            calib_velo_to_cam_path = os.path.join(date_folder, 'calib_velo_to_cam.txt')
            calib_cam_to_cam_path = os.path.join(date_folder, 'calib_cam_to_cam.txt')
            
            # If not found, try in sequence folder
            if not os.path.exists(calib_velo_to_cam_path):
                calib_velo_to_cam_path = os.path.join(full_seq_path, 'calib_velo_to_cam.txt')
            if not os.path.exists(calib_cam_to_cam_path):
                calib_cam_to_cam_path = os.path.join(full_seq_path, 'calib_cam_to_cam.txt')
            
            if not os.path.exists(calib_velo_to_cam_path):
                raise FileNotFoundError(
                    f"Velodyne-to-camera calibration file not found for {seq_path}. "
                    f"Looked in: {os.path.join(date_folder, 'calib_velo_to_cam.txt')} and "
                    f"{os.path.join(full_seq_path, 'calib_velo_to_cam.txt')}"
                )
            
            if not os.path.exists(calib_cam_to_cam_path):
                raise FileNotFoundError(
                    f"Camera-to-camera calibration file not found for {seq_path}. "
                    f"Looked in: {os.path.join(date_folder, 'calib_cam_to_cam.txt')} and "
                    f"{os.path.join(full_seq_path, 'calib_cam_to_cam.txt')}"
                )
            
            # Read calibration files
            try:
                calib_velo_to_cam = read_calib_file(calib_velo_to_cam_path)
                calib_cam_to_cam = read_calib_file(calib_cam_to_cam_path)
                calib_cache[seq_path] = (calib_velo_to_cam, calib_cam_to_cam)
            except Exception as e:
                raise RuntimeError(f"Failed to read calibration files for {seq_path}: {e}")
        else:
            calib_velo_to_cam, calib_cam_to_cam = calib_cache[seq_path]
        
        # Get image dimensions from calibration file (S_rect_02: [width, height])
        if 'S_rect_02' not in calib_cam_to_cam:
            raise KeyError(f"S_rect_02 not found in calibration file for {seq_path}")
        S_rect_02 = calib_cam_to_cam['S_rect_02'].flatten()
        img_width = int(S_rect_02[0])
        img_height = int(S_rect_02[1])
        
        # Get or cache timestamps
        if seq_path not in timestamp_cache:
            timestamp_file = os.path.join(full_seq_path, 'velodyne_points', 'timestamps.txt')
            if os.path.exists(timestamp_file):
                timestamps = read_timestamps(timestamp_file)
            else:
                # Get all lidar files to determine length
                lidar_dir = os.path.join(full_seq_path, 'velodyne_points', 'data')
                lidar_files = sorted(glob(os.path.join(lidar_dir, '*.bin')))
                timestamps = [None] * len(lidar_files)
            timestamp_cache[seq_path] = timestamps
        else:
            timestamps = timestamp_cache[seq_path]
        
        # Find frame index for this lidar file
        lidar_dir = os.path.join(full_seq_path, 'velodyne_points', 'data')
        lidar_files = sorted(glob(os.path.join(lidar_dir, '*.bin')))
        frame_idx = lidar_files.index(lidar_file)
        
        # Process this frame
        frame_name = os.path.splitext(os.path.basename(lidar_file))[0]
        
        info = dict()
        
        # Basic info
        info['sample_token'] = f"{frame_name}"
        info['timestamp'] = timestamps[frame_idx] if frame_idx < len(timestamps) else None
        info['scene_token'] = seq_path
        
        # Load lidar pointcloud
        lidar_pts = load_from_bin(lidar_file)
        
        # Convert to range image
        range_intensity_map = create_range_map(
            lidar_pts, 
            image_rows_full=image_rows, 
            image_cols=image_cols, 
            ang_start_y=ang_start_y, 
            ang_res_y=ang_res_y, 
            ang_res_x=ang_res_x, 
            max_range=max_range, 
            min_range=min_range
        )
        
        # Add mask channel to match nuScenes format (range, intensity, mask)
        # Mask is 1 where range > 0 (valid points)
        mask = (range_intensity_map[..., 0] > 0).astype(np.float32)
        range_intensity_mask = np.concatenate(
            [range_intensity_map, mask[..., np.newaxis]], axis=-1
        )
        
        # Save range image
        sanitized_seq_path = seq_path.replace('/', '_').replace('\\', '_')
        rv_out_name = f"{sanitized_seq_path}_{frame_name}_RV.npy"
        rv_out_path = os.path.join(split_rv_dir, rv_out_name)
        np.save(rv_out_path, range_intensity_mask.astype(np.float32))
        
        # Create depth target (optional, commented out in nuScenes version)
        head_target_out_path = os.path.join(split_head_target_dir, rv_out_name)
        # range_head_target = create_depth_target(
        #     range_intensity_mask[..., 0], 
        #     range_intensity_mask[..., -1].astype(np.bool)
        # )
        # np.save(head_target_out_path, range_head_target.astype(np.float32))
        
        # Lidar info
        lidar_info = dict()
        lidar_info['sample_token'] = info['sample_token']
        lidar_info['timestamp'] = info['timestamp']
        lidar_info['filename'] = os.path.relpath(lidar_file, data_root).replace('\\', '/')
        # Get relative path in format: {split_name}/RV.npy
        # Since split_rv_dir is absolute, we need to construct the relative path manually
        rv_rel_path = f"{split_name}/{rv_out_name}"
        head_target_rel_path = os.path.relpath(head_target_out_path, data_root).replace('\\', '/')
        lidar_info['rv_path'] = rv_rel_path
        lidar_info['range_head_target_path'] = head_target_rel_path
        
        # For KITTI, lidar is at origin (no transformation needed)
        lidar_info['calibrated_sensor'] = {
            'translation': [0.0, 0.0, 0.0],
            'rotation': [1.0, 0.0, 0.0, 0.0]  # w, x, y, z quaternion
        }
        lidar_info['ego_pose'] = {
            'translation': [0.0, 0.0, 0.0],
            'rotation': [1.0, 0.0, 0.0, 0.0]
        }
        
        # Camera info (only image_02 - left color camera)
        cam_infos = dict()
        cam_dir = os.path.join(full_seq_path, cam_name, 'data')
        
        if os.path.exists(cam_dir):
            cam_files = sorted(glob(os.path.join(cam_dir, '*.png')))
            
            if frame_idx < len(cam_files):
                cam_file = cam_files[frame_idx]
                cam_frame_name = os.path.splitext(os.path.basename(cam_file))[0]
                
                # Verify frame numbers match
                if cam_frame_name != frame_name:
                    # Try to find matching frame
                    matching_files = [f for f in cam_files if os.path.splitext(os.path.basename(f))[0] == frame_name]
                    if matching_files:
                        cam_file = matching_files[0]
                    else:
                        cam_file = None
                
                if cam_file is not None:
                    # Compute camera to lidar transformation
                    R_cam_to_velo, T_cam_to_velo, P_rect = compute_cam2lidar(
                        calib_velo_to_cam, calib_cam_to_cam
                    )
                    
                    sweep_cam_info = dict()
                    sweep_cam_info['sample_token'] = info['sample_token']
                    sweep_cam_info['timestamp'] = info['timestamp']
                    sweep_cam_info['is_key_frame'] = True  # All frames are key frames in KITTI
                    sweep_cam_info['height'] = img_height
                    sweep_cam_info['width'] = img_width
                    sweep_cam_info['filename'] = os.path.relpath(cam_file, data_root).replace('\\', '/')
                    
                    # Convert rotation matrix to quaternion [w, x, y, z]
                    quat_cam_to_velo = rotation_matrix_to_quaternion(R_cam_to_velo)
                    
                    # Calibrated sensor info (matching nuScenes format)
                    # Extract 3x3 intrinsic matrix from P_rect (first 3 columns)
                    # P_rect is 3x4: [K | t] where K is 3x3 intrinsic matrix
                    if P_rect is not None:
                        # Extract intrinsic matrix K from P_rect (first 3 columns)
                        K = P_rect[:, :3]  # 3x3 intrinsic matrix
                        camera_intrinsic = K.tolist()
                    else:
                        camera_intrinsic = None
                    
                    calibrated_sensor = {
                        'translation': T_cam_to_velo.flatten().tolist(),
                        'rotation': quat_cam_to_velo
                    }
                    if camera_intrinsic is not None:
                        calibrated_sensor['camera_intrinsic'] = camera_intrinsic
                    sweep_cam_info['calibrated_sensor'] = calibrated_sensor
                    
                    # Ego pose (same as lidar for KITTI, no ego motion between sensors)
                    sweep_cam_info['ego_pose'] = {
                        'translation': [0.0, 0.0, 0.0],
                        'rotation': [1.0, 0.0, 0.0, 0.0]
                    }
                    
                    # Sensor to lidar transformation (used by RVWithImageDataset.get_image())
                    sweep_cam_info['sensor2lidar_translation'] = T_cam_to_velo.flatten().tolist()
                    sweep_cam_info['sensor2lidar_rotation'] = R_cam_to_velo.tolist()
                    
                    cam_infos[cam_name] = sweep_cam_info
        
        info['cam_infos'] = cam_infos
        info['lidar_info'] = lidar_info
        info['cam_sweeps'] = list()
        info['lidar_sweeps'] = list()
        
        infos.append(info)
    
    return infos


def main():
    parser = argparse.ArgumentParser(description='Generate info files for KITTI dataset')
    parser.add_argument('--data_root', type=str, default='./data/KITTI/', 
                        help='Root directory of KITTI data (default: ./data/KITTI/)')
    parser.add_argument('--num_data_train', type=int, default=21000,
                        help='Number of training samples to generate (default: 21000)')
    parser.add_argument('--num_data_val', type=int, default=2500,
                        help='Number of validation samples to generate (default: 2500)')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    data_root = args.data_root
    num_data_train = args.num_data_train
    num_data_val = args.num_data_val
    
    print(f"Scanning data_root: {data_root}")
    print(f"Target samples - Train: {num_data_train}, Val: {num_data_val}")
    
    # Scan all pointcloud files in data_root
    all_lidar_files = []  # List of (seq_path, lidar_file) tuples
    
    # Find all sequence folders (date/drive_sync format)
    for date_folder in glob(os.path.join(data_root, '2011_*')):
        if os.path.isdir(date_folder):
            for drive_folder in glob(os.path.join(date_folder, '*_sync')):
                if os.path.isdir(drive_folder):
                    # Get relative sequence path
                    seq_path = os.path.relpath(drive_folder, data_root)
                    lidar_dir = os.path.join(drive_folder, 'velodyne_points', 'data')
                    
                    if os.path.exists(lidar_dir):
                        lidar_files = sorted(glob(os.path.join(lidar_dir, '*.bin')))
                        for lidar_file in lidar_files:
                            all_lidar_files.append((seq_path, lidar_file))
    
    print(f"Found {len(all_lidar_files)} total lidar files")
    
    if len(all_lidar_files) < num_data_train + num_data_val:
        print(f"Warning: Only {len(all_lidar_files)} files available, but requested {num_data_train + num_data_val} samples")
        print(f"Adjusting: Train={min(num_data_train, len(all_lidar_files))}, Val={min(num_data_val, len(all_lidar_files) - min(num_data_train, len(all_lidar_files)))}")
        num_data_train = min(num_data_train, len(all_lidar_files))
        num_data_val = min(num_data_val, len(all_lidar_files) - num_data_train)
    
    # Shuffle all files for random split
    random.shuffle(all_lidar_files)
    
    # Split into train and val
    train_frames = all_lidar_files[:num_data_train]
    val_frames = all_lidar_files[num_data_train:num_data_train + num_data_val]
    
    print(f"Split: {len(train_frames)} train files, {len(val_frames)} val files")
    
    # Generate train info
    print("Generating train info...")
    train_infos = generate_info(data_root, 'train', train_frames)
    with open(os.path.join(data_root, 'kitti_upsample_infos_train.pkl'), 'wb') as f:
        pickle.dump(train_infos, f)
    print(f"Saved {len(train_infos)} train samples")
    
    # Generate val info
    print("Generating val info...")
    val_infos = generate_info(data_root, 'val', val_frames)
    with open(os.path.join(data_root, 'kitti_upsample_infos_val.pkl'), 'wb') as f:
        pickle.dump(val_infos, f)
    print(f"Saved {len(val_infos)} val samples")


if __name__ == '__main__':
    main()
