import numpy as np
import os
import argparse
# import cv2
from glob import glob
import pathlib
import random
import shutil
import sys
import pdb
# Add parent directory to path to import from kitti_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kitti_utils.sample_kitti_dataset import create_range_map, load_from_bin

try:
    from nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import view_points
    NUSCENES_AVAILABLE = True
except ImportError:
    print("Warning: nuScenes SDK not available. Install with: pip install nuscenes-devkit")
    NUSCENES_AVAILABLE = False


class NuScenesPointCloudToRangeImage:
    """
    Convert a nuScenes LiDAR point cloud (HDL 32E) to a range image.

    Expected point format: [x, y, z, intensity, ring]
    Rows come either from ring indices or from nearest elevation in a shared table
    Columns come from azimuth atan2(y, x) to the nearest bin center
    Collisions keep the nearest range
    Returns range, intensity, and a mask
    """

    def __init__(self,
                 min_depth=2.0,
                 max_depth=50.0,
                 flip_vertical=True,
                 log_scale=False,
                 inverse_scale=False,
                 intensity_clip=None,
                 add_channel_dim=True):
        self.width = 1024
        self.height = 32
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.flip_vertical = bool(flip_vertical)
        self.log_scale = bool(log_scale)
        self.inverse_scale = bool(inverse_scale) and (not log_scale)
        self.intensity_clip = intensity_clip
        self.add_channel_dim = bool(add_channel_dim)
        self.h_per_row = np.array([-0.00216031, -0.00098729, -0.00020528,  0.00174976,  0.0044868 , -0.00294233,
                                    -0.00059629, -0.00020528,  0.00174976, -0.00294233, -0.0013783 ,  0.00018573,
                                    0.00253177, -0.00098729,  0.00018573,  0.00096774, -0.00411535, -0.0013783,
                                    0.00018573,  0.00018573, -0.00294233, -0.0013783 , -0.00098729, -0.00020528,
                                    0.00018573,  0.00018573,  0.00018573, -0.00020528,  0.00018573,  0.00018573,
                                    0.00018573,  0.00018573,], dtype=np.float32)

        # Default HDL 32E per ring elevations in degrees, interleaved order
        self._base_elev_deg = np.array([
                -30.67, -9.33,  -29.33,  -8.00,
                -28.00, -6.66,  -26.66,  -5.33,
                -25.33, -4.00,  -24.00,  -2.67,
                -22.67, -1.33,  -21.33,   0.00,
                -20.00,  1.33,  -18.67,   2.67,
                -17.33,  4.00,  -16.00,   5.33,
                -14.67,  6.67,  -13.33,   8.00,
                -12.00,  9.33,  -10.67,  10.67
            ], dtype=np.float32)

    def __call__(self, pc: np.ndarray):
        if pc.size == 0:
            H0 = len(self._base_elev_deg) if self.height is None else self.height
            return self._empty_outputs(inferred_height=H0)

        assert pc.shape[1] >= 5, "pc must have at least 5 columns: [x,y,z,intensity,ring]"

        x = pc[:, 0].astype(np.float32, copy=False)
        y = pc[:, 1].astype(np.float32, copy=False)
        z = pc[:, 2].astype(np.float32, copy=False)
        intensity = pc[:, 3].astype(np.float32, copy=False)
        ring = pc[:, 4].astype(np.int32, copy=False)

        # Height, prefer explicit height, else infer from ring, else from table
        if self.height is not None:
            H = int(self.height)
        else:
            H = int(ring.max()) + 1 if ring.size else len(self._base_elev_deg)
        W = self.width

        # Range and masks
        r = np.sqrt(x*x + y*y + z*z, dtype=np.float32)
        depth_mask = (r >= self.min_depth) & (r <= self.max_depth) & np.isfinite(r)
        ring_mask = (ring >= 0) & (ring < H)
        valid = depth_mask & ring_mask
        if not np.any(valid):
            return self._empty_outputs(inferred_height=H)

        xv = x[valid]; yv = y[valid]; zv = z[valid]
        rv = r[valid]; iv = intensity[valid]
        ringv = ring[valid]
        h_corr = self.h_per_row[ringv]
        zv = zv - h_corr

        # Row indices from shared elevation lookup or ring
        # row elev table, index equals ring id
        # point elevation in degrees
        vdeg = np.rad2deg(np.arctan2(zv, np.sqrt(xv * xv + yv * yv)))
        # nearest ring by elevation
        # shape: [Nv, H] differences, argmin over H
        diffs = np.abs(vdeg[:, None] - self._base_elev_deg[None, :])
        nearest_ring = np.argmin(diffs, axis=1).astype(np.int32)
        rows = nearest_ring
        if self.flip_vertical:
            rows = (H - 1 - rows).astype(np.int32, copy=False)

        # Columns from azimuth to nearest bin center
        az = np.arctan2(yv, xv)
        col_center = ((az + np.pi) / (2.0 * np.pi) * W) - 0.5
        cols = np.round(col_center).astype(np.int32) % W

        # Optional intensity clipping
        if self.intensity_clip is not None:
            lo, hi = self.intensity_clip
            iv = np.clip(iv, lo, hi)

        # Nearest per pixel
        order = np.argsort(rv, kind="stable")
        rows_sorted = rows[order]
        cols_sorted = cols[order]
        rv_sorted = rv[order]
        iv_sorted = iv[order]

        lin_sorted = rows_sorted * W + cols_sorted
        _, first_pos = np.unique(lin_sorted, return_index=True)

        rr = rv_sorted[first_pos]
        ii = iv_sorted[first_pos]
        rr_rows = rows_sorted[first_pos]
        rr_cols = cols_sorted[first_pos]

        # Outputs
        range_img = np.zeros((H, W), dtype=np.float32)
        intensity_img = np.zeros((H, W), dtype=np.float32)
        mask_img = np.zeros((H, W), dtype=np.uint8)

        range_img[rr_rows, rr_cols] = rr
        intensity_img[rr_rows, rr_cols] = ii
        mask_img[rr_rows, rr_cols] = 1

        if self.log_scale:
            scaled = np.log2(rr + 1.0, dtype=np.float32) / 6.0
            range_img[rr_rows, rr_cols] = scaled
        elif self.inverse_scale:
            inv = (1.0 / np.maximum(rr, 1e-6)).astype(np.float32)
            range_img[rr_rows, rr_cols] = inv

        out = np.stack([range_img, intensity_img, mask_img.astype(np.float32)], axis=-1)
        return out.astype(np.float32, copy=False)

    def _empty_outputs(self, inferred_height):
        H = inferred_height
        W = self.width
        range_img = np.zeros((H, W), dtype=np.float32)
        intensity_img = np.zeros((H, W), dtype=np.float32)
        mask_img = np.zeros((H, W), dtype=np.uint8)

        if self.add_channel_dim:
            return np.stack([range_img, intensity_img, mask_img.astype(np.float32)], axis=-1)
        return {"range": range_img, "intensity": intensity_img, "mask": mask_img}

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_data_train', type=int, default=21000)
    parser.add_argument('--num_data_val', type=int, default=2500)
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-trainval")
    parser.add_argument("--nuscenes_dataroot", type=str, required=True, help="Path to nuScenes dataset root directory")
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


def get_nuscenes_lidar_samples(nusc, split='train'):
    """Get all LiDAR sample paths for a given split using nuScenes SDK"""
    if not NUSCENES_AVAILABLE:
        raise ImportError("nuScenes SDK is required for this functionality")
    
    # Get all samples for the specified split
    if split == 'train':
        samples = [samp for samp in nusc.sample if nusc.get('scene', samp['scene_token'])['name'].startswith('scene-0')]
    elif split == 'val':
        samples = [samp for samp in nusc.sample if nusc.get('scene', samp['scene_token'])['name'].startswith('scene-1')]
    else:
        # For test or other splits, get all samples
        samples = nusc.sample
    
    lidar_samples = []
    
    for sample in samples:
        # Get the first LiDAR sample (usually LIDAR_TOP)
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        if lidar_data['filename'].endswith('.bin'):
            # Construct full path
            full_path = os.path.join(nusc.dataroot, lidar_data['filename'])
            if os.path.exists(full_path):
                lidar_samples.append({
                    'sample_token': sample['token'],
                    'file_path': full_path,
                    'timestamp': sample['timestamp']
                })
    
    return lidar_samples


def main(args):
    if not NUSCENES_AVAILABLE:
        print("Error: nuScenes SDK is required. Install with: pip install nuscenes-devkit")
        return
    
    # Set random seeds for reproducibility
    print(f"Setting random seed: 42")
    random.seed(42)
    np.random.seed(42)
    
    # Initialize nuScenes
    try:
        nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_dataroot, verbose=True)
        print(f"Loaded nuScenes dataset: {args.nuscenes_version}")
        print(f"Dataset root: {nusc.dataroot}")
    except Exception as e:
        print(f"Error loading nuScenes dataset: {e}")
        return
    
    num_data_train = args.num_data_train
    num_data_val = args.num_data_val
    
    # Create output directories
    output_dir_name_train = os.path.join(args.nuscenes_dataroot, args.output_path_name_train)
    pathlib.Path(output_dir_name_train).mkdir(parents=True, exist_ok=True)
    
    if args.create_val:
        output_dir_name_val = os.path.join(args.nuscenes_dataroot, args.output_path_name_val)
        pathlib.Path(output_dir_name_val).mkdir(parents=True, exist_ok=True)

    # Get LiDAR samples using nuScenes SDK
    print("Getting training samples...")
    train_samples = get_nuscenes_lidar_samples(nusc, 'train')
    print(f"Found {len(train_samples)} training samples")
    
    if args.create_val:
        print("Getting validation samples...")
        val_samples = get_nuscenes_lidar_samples(nusc, 'val')
        print(f"Found {len(val_samples)} validation samples")
    
    # Sort samples by timestamp for consistent ordering before sampling
    print("Sorting samples by timestamp for consistent sampling...")
    train_samples = sorted(train_samples, key=lambda x: x['timestamp'])
    if args.create_val:
        val_samples = sorted(val_samples, key=lambda x: x['timestamp'])
    
    # Sample from available data
    if num_data_train <= len(train_samples):
        selected_train = random.sample(train_samples, num_data_train)
    else:
        # If we need more data than available, sample with replacement
        selected_train = random.choices(train_samples, k=num_data_train)
        print(f"Warning: Requested {num_data_train} training samples but only {len(train_samples)} available. Using replacement sampling.")
    
    if args.create_val:
        if num_data_val <= len(val_samples):
            selected_val = random.sample(val_samples, num_data_val)
        else:
            selected_val = random.choices(val_samples, k=num_data_val)
            print(f"Warning: Requested {num_data_val} validation samples but only {len(val_samples)} available. Using replacement sampling.")
    
    print(f"Selected {len(selected_train)} training samples")
    if args.create_val:
        print(f"Selected {len(selected_val)} validation samples")

    # Initialize the range image converter
    converter = NuScenesPointCloudToRangeImage(
        width=1024,
        height=32,
        min_depth=2.0,
        max_depth=100.0
    )
    
    # Process training data
    print("Processing training data...")
    for i, sample_info in enumerate(selected_train):
        try:
            data_path = sample_info['file_path']
            # Load full nuScenes data (including ring_index)
            lidar_data = np.fromfile(data_path, dtype=np.float32).reshape(-1, 5)
            
            # Convert to range image using the new converter
            range_intensity_map = converter(lidar_data)

            output_filename = f'{i:08d}.npy'
            np.save(os.path.join(output_dir_name_train, output_filename), range_intensity_map.astype(np.float32))
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(selected_train)} training samples")
                
        except Exception as e:
            print(f"Error processing {data_path}: {e}")
            continue

    # Process validation data if requested
    if args.create_val:
        print("Processing validation data...")
        for j, sample_info in enumerate(selected_val):
            try:
                data_path = sample_info['file_path']
                # Load full nuScenes data (including ring_index)
                lidar_data = np.fromfile(data_path, dtype=np.float32).reshape(-1, 5)
                
                # Convert to range image using the new converter
                range_intensity_map = converter(lidar_data)

                output_filename = f'{j:08d}.npy'
                np.save(os.path.join(output_dir_name_val, output_filename), range_intensity_map.astype(np.float32))
                
                if (j + 1) % 100 == 0:
                    print(f"Processed {j + 1}/{len(selected_val)} validation samples")
                    
            except Exception as e:
                print(f"Error processing {data_path}: {e}")
                continue

    print("Dataset creation completed!")
    print(f"Training data saved to: {output_dir_name_train}")
    if args.create_val:
        print(f"Validation data saved to: {output_dir_name_val}")


if __name__ == "__main__":
    args = read_args()
    main(args)

