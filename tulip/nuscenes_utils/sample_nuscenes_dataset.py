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


class PointCloudToRangeImage:
    """Convert nuScenes LiDAR point clouds to range images"""
    
    def __init__(self, 
                 width=1024, 
                 height=32,
                 pc_range=[-50.2, -50.2, -3., 50.2, 50.2, 1.], 
                 log=False, 
                 normalize_volume_densities=True,
                 inverse=False,
                 min_depth=2.0,
                 max_depth=100.0) -> None:
        
        self.height = height
        self.width = width
        self.pc_range = pc_range
        self.log = log
        self.normalize_volume_densities = normalize_volume_densities
        self.inverse = inverse
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # nuScenes specific parameters
        self.zenith = np.array([ 1.86705767e-01,  1.63245357e-01,  1.39784946e-01,  1.16324536e-01,
                                 9.28641251e-02,  7.01857283e-02,  4.67253177e-02,  2.32649071e-02,
                                -1.95503421e-04, -2.28739003e-02, -4.63343109e-02, -6.97947214e-02,
                                -9.32551320e-01, -1.15933529e-01, -1.39393939e-01, -1.62854350e-01,
                                -1.85532747e-01, -2.08993157e-01, -2.32453568e-01, -2.55913978e-01,
                                -2.78592375e-01, -3.02052786e-01, -3.25513196e-01, -3.48973607e-01,
                                -3.72434018e-01, -3.95894428e-01, -4.19354839e-01, -4.42033236e-01,
                                -4.65493646e-01, -4.88954057e-01, -5.12414467e-01, -5.35874878e-01], dtype=np.float32)
        
        self.incl = -self.zenith
        
        # Statistics for normalization
        self.mean = 50.0
        self.std = 50.0
    
    def get_row_inds(self, pc):
        """Get row indices for nuScenes LiDAR data using ring index"""
        # nuScenes already has the row indices in the 5th column (ring_index)
        # Convert to 0-based indexing and flip vertically
        row_inds = self.height - 1 - pc[:, 4].astype(np.int32)
        return row_inds
    
    def __call__(self, pc):
        """Convert point cloud to range image"""
        # Filter by depth
        depth = np.linalg.norm(pc[:, :3], axis=1)
        mask = (depth > self.min_depth) & (depth < self.max_depth)
        pc = pc[mask, :]
        
        if len(pc) == 0:
            # Return empty range image if no valid points
            return np.full((self.height, self.width, 2), -1, dtype=np.float32)
        
        # Get row and column indices
        row_inds = self.get_row_inds(pc)
        
        # Calculate azimuth angles
        azi = np.arctan2(pc[:, 1], pc[:, 0])
        col_inds = self.width - 1.0 + 0.5 - (azi + np.pi) / (2.0 * np.pi) * self.width
        col_inds = np.round(col_inds).astype(np.int32)
        
        # Handle boundary conditions
        col_inds[col_inds == self.width] = self.width - 1
        col_inds[col_inds < 0] = 0
        
        # Initialize empty range image
        empty_range_image = np.full((self.height, self.width, 2), -1, dtype=np.float32)
        
        # Adjust height using zenith angles
        pc[:, 2] -= self.zenith[row_inds]
        
        # Calculate point ranges
        point_range = np.linalg.norm(pc[:, :3], axis=1, ord=2)
        # point_range[point_range > self.max_depth] = self.max_depth
        
        # Sort by range (farthest first for proper occlusion handling)
        order = np.argsort(-point_range)
        point_range = point_range[order]
        pc = pc[order]
        row_inds = row_inds[order]
        col_inds = col_inds[order]
        
        # Apply transformations if specified
        if self.log:
            point_range = np.log2(point_range + 1) / 6
        elif self.inverse:
            point_range = 1 / point_range
        
        # Create range-intensity pairs
        range_intensity = np.concatenate([point_range[:, None], pc[:, 3:4]], axis=1)
        
        # Fill the range image
        empty_range_image[row_inds, col_inds, :] = range_intensity
        
        return empty_range_image



import numpy as np

class NuScenesPointCloudToRangeImage:
    """
    Convert a nuScenes LiDAR point cloud (HDL‑32E) to a range image.

    Expected point format: [x, y, z, intensity, ring] for each point.
    - Rows are set by the ring index (0-based if you pass raw nuScenes ring).
    - Columns are set by azimuth atan2(y, x) mapped to [0, width).
    - Collisions (multiple points to same pixel): keeps the *nearest* range.
    - Returns range, intensity, and a validity mask. Optional scaling of range.

    Parameters
    ----------
    width : int
        Horizontal resolution (columns) of the range image (e.g., 1024).
    height : int or None
        Vertical resolution (rows). If None, inferred from max ring + 1.
    min_depth, max_depth : float
        Range filter in meters (strictly enforce physical limits).
    flip_vertical : bool
        If True, flips ring index v -> (H-1 - v) to match “top row = ring 0” conventions.
    log_scale : bool
        If True, replaces range r with log2(r + 1) / 6 (bounded to ~[0,1] for r<=63m).
    inverse_scale : bool
        If True, replaces range r with 1 / r. (Ignored if log_scale=True.)
    add_channel_dim : bool
        If True, returns HxWxC stacked array with channels [range, intensity, mask].
        If False, returns a dict with arrays.
    intensity_clip : tuple[float, float]
        Optional (min, max) to clip intensity before writing.
    """

    def __init__(self,
                 width=1024,
                 height=None,
                 min_depth=2.0,
                 max_depth=100.0,
                 flip_vertical=True,
                 log_scale=False,
                 inverse_scale=False,
                 intensity_clip=None):
        self.width = int(width)
        self.height = None if height is None else int(height)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.flip_vertical = bool(flip_vertical)
        self.log_scale = bool(log_scale)
        self.inverse_scale = bool(inverse_scale) and (not log_scale)
        self.intensity_clip = intensity_clip

    def __call__(self, pc: np.ndarray):
        """
        Parameters
        ----------
        pc : (N, 5) float/np.ndarray
            Columns: x, y, z, intensity, ring.

        Returns
        -------
        If add_channel_dim:
            range_image : (H, W, 3) float32, channels = [range, intensity, mask]
        else:
            {
              "range":    (H, W) float32,
              "intensity":(H, W) float32,
              "mask":     (H, W) uint8,
            }
        """
        if pc.size == 0:
            return self._empty_outputs(inferred_height=32 if self.height is None else self.height)

        assert pc.shape[1] >= 5, "pc must have at least 5 columns: [x,y,z,intensity,ring]"

        x = pc[:, 0].astype(np.float32, copy=False)
        y = pc[:, 1].astype(np.float32, copy=False)
        z = pc[:, 2].astype(np.float32, copy=False)
        intensity = pc[:, 3].astype(np.float32, copy=False)
        ring = pc[:, 4].astype(np.int32, copy=False)

        # Infer height from ring if not provided
        H = self.height if self.height is not None else (int(ring.max()) + 1)
        W = self.width

        # Compute Euclidean range and filter by min/max range
        r = np.sqrt(x*x + y*y + z*z, dtype=np.float32)
        depth_mask = (r >= self.min_depth) & (r <= self.max_depth) & np.isfinite(r)

        # Filter invalid rings (robustness)
        ring_mask = (ring >= 0) & (ring < H)

        valid = depth_mask & ring_mask
        if not np.any(valid):
            return self._empty_outputs(inferred_height=H)

        xv = x[valid]; yv = y[valid]; zv = z[valid]
        rv = r[valid]; iv = intensity[valid]
        ringv = ring[valid]

        # Row indices (optionally flipped so ring 0 at top or bottom)
        rows = ringv.copy()
        if self.flip_vertical:
            rows = (H - 1 - rows).astype(np.int32, copy=False)

        # Column indices from azimuth
        az = np.arctan2(yv, xv)  # [-pi, pi]
        cols = ((az + np.pi) / (2.0 * np.pi) * W).astype(np.int32)  # [0, W)
        # Robust wrap (rare rounding at boundary)
        cols %= W

        # Optional intensity clipping
        if self.intensity_clip is not None:
            lo, hi = self.intensity_clip
            iv = np.clip(iv, lo, hi)

        # Choose nearest point per (row, col)
        # Sort by range ascending so the first occurrence is the closest
        order = np.argsort(rv, kind="stable")
        rows_sorted = rows[order]
        cols_sorted = cols[order]
        rv_sorted = rv[order]
        iv_sorted = iv[order]

        lin_sorted = rows_sorted * W + cols_sorted
        # First occurrence per unique lin index gives the nearest point
        _, first_pos = np.unique(lin_sorted, return_index=True)

        rr = rv_sorted[first_pos]
        ii = iv_sorted[first_pos]
        rr_rows = rows_sorted[first_pos]
        rr_cols = cols_sorted[first_pos]

        # Allocate outputs
        range_img = np.zeros((H, W), dtype=np.float32)
        intensity_img = np.zeros((H, W), dtype=np.float32)
        mask_img = np.zeros((H, W), dtype=np.uint8)

        # Write nearest values
        range_img[rr_rows, rr_cols] = rr
        intensity_img[rr_rows, rr_cols] = ii
        mask_img[rr_rows, rr_cols] = 1

        # Optional scaling of the range channel (apply ONLY to valid pixels)
        if self.log_scale:
            scaled = np.log2(rr + 1.0, dtype=np.float32) / 6.0  # ~[0,1] for <=63m
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

