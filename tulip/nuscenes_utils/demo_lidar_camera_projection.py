#!/usr/bin/env python3
"""
Demo script that reads a random data point from RVWithImageDataset and projects 
lidar points onto each camera view, showing all outputs tiled in the same plot.

This script demonstrates:
1. Loading data from RVWithImageDataset
2. Converting range view to 3D lidar points
3. Projecting lidar points to each camera using lidar2img matrices
4. Visualizing projections overlaid on camera images
"""

import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2
import pdb
# Add the parent directory to the path to import from tulip
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from tulip.util.datasets import (
    RVWithImageDataset,
    transforms,
    ScaleTensor,
    FilterInvalidPixels,
    DownsampleTensor,
    DownsampleTensorWidth,
    LogTransform,
)

# nuScenes devkit imports (ensure nuscenes-devkit is installed)
from nuscenes.nuscenes import NuScenes

NUSC_AVAILABLE = True



def build_dataset(data_root: str, info_file: str, final_dim=(256, 704)) -> RVWithImageDataset:
    """
    Build the RVWithImageDataset with the same transforms as used in training.
    """
    input_size = (8, 1024)
    output_size = (32, 1024)
    
    # Low resolution transforms
    t_low_res = [
        transforms.ToTensor(),
        ScaleTensor(1/55),
        FilterInvalidPixels(min_range=0, max_range=55/55),
        DownsampleTensor(
            h_high_res=output_size[0],
            downsample_factor=output_size[0] // input_size[0],
        ),
        LogTransform(),
    ]
    
    # High resolution transforms  
    t_high_res = [
        transforms.ToTensor(),
        ScaleTensor(1/55),
        FilterInvalidPixels(min_range=0, max_range=55/55),
        LogTransform(),
    ]
    
    transform_low_res = transforms.Compose(t_low_res)
    transform_high_res = transforms.Compose(t_high_res)
    
    dataset = RVWithImageDataset(
        root=data_root,
        high_res_transform=transform_high_res,
        low_res_transform=transform_low_res,
        info_file=info_file,
        final_dim=final_dim,
    )
    return dataset


def load_nuscenes_point_cloud(bin_path):
    """Load point cloud from nuScenes .bin file."""
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)


def project_points_to_camera(points, lidar2img_matrices, img_shapes):
    """
    Project 3D points to camera image coordinates using einsum for efficient broadcasting.
    Handles batched samples and stacked/batched lidar2img matrices for multiple cameras.
    
    Args:
        points: (B, N, 3) or (N, 3) array of 3D points in lidar frame
        lidar2img_matrices: (B, num_cam, 4, 4) or (num_cam, 4, 4) stacked transformation matrices
        img_shapes: (B, num_cam, 2) or (num_cam, 2) array of (H, W) image shapes
        
    Returns:
        projected_points: (B, num_cam, N, 2) or (num_cam, N, 2) array of image coordinates
        valid_masks: (B, num_cam, N) or (num_cam, N) boolean masks for points within image bounds
    """
    imgH, imgW = img_shapes[0,0,0], img_shapes[0,0,1]
    # Handle both batched and non-batched inputs
    if points.ndim == 2:
        # Non-batched: add batch dimension
        points = points[None, ...]  # (1, N, 3)
        lidar2img_matrices = lidar2img_matrices[None, ...]  # (1, num_cam, 4, 4)
        img_shapes = img_shapes[None, ...]  # (1, num_cam, 2)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, N, _ = points.shape
    _, num_cam, _, _ = lidar2img_matrices.shape
    
    if N == 0:
        return (np.array([]).reshape(B, num_cam, 0, 2) if not squeeze_output 
                else np.array([]).reshape(num_cam, 0, 2)), \
               (np.array([]).reshape(B, num_cam, 0) if not squeeze_output 
                else np.array([]).reshape(num_cam, 0))
    
    # Convert to homogeneous coordinates
    points_homo = np.concatenate([points, np.ones((B, N, 1))], axis=-1)  # (B, N, 4)
    
    # Project to image coordinates using einsum for efficient broadcasting across batch and cameras
    # einsum('bcij,bnj->bcni', lidar2img_matrices, points_homo)
    # b: batch, c: camera, i: output dim, j: input dim, n: points
    reference_points_cam = np.einsum('bcij,bnj->bcni', lidar2img_matrices, points_homo)  # (B, num_cam, N, 4)
    
    # Convert from homogeneous coordinates
    eps = 1e-5

    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / np.maximum(
        reference_points_cam[..., 2:3], np.ones_like(reference_points_cam[..., 2:3]) * eps)

    # Normalize by image shapes - handle batched img_shapes
    # img_shapes has shape (B, num_cam, 2) where last dim is (H, W)
    # img_shapes_expanded = img_shapes[:, :, None, :]  # (B, num_cam, 1, 2)
    # reference_points_cam[..., 0] /= img_shapes_expanded[..., 1]  # Divide by width
    # reference_points_cam[..., 1] /= img_shapes_expanded[..., 0]  # Divide by height
    mask = mask & (reference_points_cam[..., 1:2] > 0.0) \
            & (reference_points_cam[..., 1:2] < imgH) \
            & (reference_points_cam[..., 0:1] < imgW) \
            & (reference_points_cam[..., 0:1] > 0.0)
                
    return reference_points_cam, mask


def denormalize_image(img_tensor, img_mean, img_std, to_rgb=True):
    """
    Denormalize image tensor back to displayable format.
    
    Args:
        img_tensor: (3, H, W) normalized image tensor
        img_mean: Mean values used for normalization
        img_std: Std values used for normalization
        to_rgb: Whether to convert BGR to RGB
        
    Returns:
        img: (H, W, 3) numpy array in [0, 255] range
    """
    # Convert to numpy and transpose to (H, W, 3)
    img = img_tensor.numpy().transpose(1, 2, 0)
    
    # Denormalize
    img = img * img_std + img_mean
    
    # Clip to valid range
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Convert BGR to RGB if needed
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def render_with_nuscenes_devkit(nusc: 'NuScenes', sample_token: str, cam_name: str):
    """
    Render point cloud projection using nuScenes render_pointcloud_in_image.
    Returns the rendered image with point cloud overlay.
    """
    try:
        # Use the render function which handles all the projection internally
        rendered_img = nusc.render_pointcloud_in_image(sample_token, pointsensor_channel='LIDAR_TOP', camera_channel=cam_name)
        return rendered_img
    except Exception as e:
        print(f"Error rendering with devkit: {e}")
        return None


def visualize_projections(sample_data, dataset, sample_idx):
    """
    Visualize lidar point projections on all camera views.
    
    Args:
        sample_data: Data from dataset.__getitem__()
        dataset: RVWithImageDataset instance
        sample_idx: Index of the sample
    """
    # Unpack sample data returns 10 items:
    # sweep_imgs, sweep_intrins, sweep_lidar2img_rts, sweep_lidar2cam_rts,
    # low_res_rv, high_res_rv, lidar2ego_mat, mask_sample, img_shape, sample_token
    (sweep_imgs, sweep_intrins, sweep_lidar2img_rts, sweep_lidar2cam_rts,
     low_res_rv, high_res_rv, lidar2ego_mat, mask_sample, img_shape, sample_token) = sample_data
    
    # Get camera names
    cam_names = dataset.cam_names
    
    # Get original point cloud path from sample info
    sample_info = dataset.infos[sample_idx]
    lidar_info = sample_info['lidar_info']
    original_bin_path = os.path.join(dataset.data_root, lidar_info['filename'])
    
    print(f"Loading original point cloud from: {original_bin_path}")
    
    if not os.path.exists(original_bin_path):
        print(f"❌ Original point cloud file not found: {original_bin_path}")
        return
    
    # Load original point cloud
    points_lidar = load_nuscenes_point_cloud(original_bin_path)
    print(f"Loaded {len(points_lidar)} points from .bin file")
    
    # Extract xyz coordinates
    points_3d = points_lidar[:, :3]  # [x, y, z, intensity, ring] -> [x, y, z]
    
    if len(points_3d) == 0:
        print("No valid points found in point cloud!")
        return
    

    use_devkit = False
    if use_devkit:
        nusc = NuScenes(version='v1.0-trainval', dataroot=dataset.data_root, verbose=False)
    imgs = []
    
    # Collect all lidar2img matrices and image shapes for batched processing
    lidar2img_matrices = []
    img_shapes = []
    
    # Process each camera to collect data
    for cam_idx, cam_name in enumerate(cam_names):
        # Get image
        img_tensor = sweep_imgs[0, cam_idx]  # First sweep, camera index
        
        # Denormalize image
        img = denormalize_image(img_tensor, dataset.img_mean, dataset.img_std, dataset.to_rgb)
        imgs.append(img)
        
        # Collect lidar2img matrix and image shape
        lidar2img_matrix = sweep_lidar2img_rts[cam_idx].numpy()
        lidar2img_matrices.append(lidar2img_matrix)
        img_shapes.append(img.shape[:2])
        
        print(f'{sample_token}')
        # nuScenes devkit projection for comparison (overlay cyan x)
        dev_xy = np.zeros((0, 2), dtype=np.float32)
        if use_devkit:
            print("Using nuScenes devkit")
            sample = nusc.get('sample', sample_token)
            
            cam_sd_token = sample['data'][cam_name]
            lidar_sd_token = sample['data']['LIDAR_TOP']
            # Get devkit points in original image coords            
            devkit_cam_out = f"devkit_render_{cam_name.lower()}_sample_{sample_idx}.png"
            devkit_render=nusc.render_pointcloud_in_image(sample['token'], pointsensor_channel='LIDAR_TOP', camera_channel=cam_name, out_path=devkit_cam_out)

    # Batch process all cameras at once using einsum
    lidar2img_matrices = np.array(lidar2img_matrices)  # (num_cam, 4, 4)
    img_shapes = np.array(img_shapes)  # (num_cam, 2)
    
    # Project points to all cameras using batched einsum
    # Add batch dimension to points for consistency
    points_3d_batched = points_3d[None, ...]  # (1, N, 3)
    lidar2img_matrices_batched = lidar2img_matrices[None, ...]  # (1, num_cam, 4, 4)
    img_shapes_batched = img_shapes[None, ...]  # (1, num_cam, 2)
    
    all_pts_xy, all_pts_mask = project_points_to_camera(
        points_3d_batched, lidar2img_matrices_batched, img_shapes_batched
    )
    # Remove batch dimension and convert results back to list format for compatibility
    all_pts_xy = all_pts_xy[0]  # (num_cam, N, 2)
    all_pts_mask = all_pts_mask[0]  # (num_cam, N)
    
    pts = [all_pts_xy[cam_idx] for cam_idx in range(len(cam_names))]
    pts_mask = [all_pts_mask[cam_idx] for cam_idx in range(len(cam_names))]

       
    
    # vis loop
    # Create figure with subplots for each camera
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for cam_idx, cam_name in enumerate(cam_names):
        ax = axes[cam_idx]
        img = imgs[cam_idx]
        ax.imshow(img)
        ax.set_title(f'{cam_name}', fontsize=12, fontweight='bold')
        ours_xy = pts[cam_idx]
        ours_mask = pts_mask[cam_idx]
    
        # Plot our projection
        if len(ours_xy) > 0 and ours_mask.sum() > 0:
            # Squeeze the extra dimension from the mask if it exists
            if ours_mask.ndim > 1:
                ours_mask = ours_mask.squeeze()
            ours_valid = ours_xy[ours_mask]  # Apply mask to first dimension (points)

            # Plot points as small circles
            ax.scatter(ours_valid[:, 0], ours_valid[:, 1], c='red', s=2, alpha=0.7, marker='o', label='ours')
        else:
            print("No points manual projected")
        # Overlay devkit points if available
        if len(ours_xy) > 0 and ours_mask.sum() > 0:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.5)
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
    # Add overall title
    fig.suptitle(f'Lidar Point Projections - Sample {sample_token[:8]}', 
                 fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save figure
    output_path = f'lidar_camera_projections_sample_{sample_idx}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    
    # Show plot
    plt.show()
    
    # (Optional) Separate devkit rendering figure removed to avoid confusion


def main():
    """Main demo function."""
    print("=== Lidar-Camera Projection Demo ===\n")
    
    # Setup paths
    data_root = "./data/nuscenes"
    info_file = "nuscenes_upsample_infos_val.pkl"
    
    # Check if data exists
    if not os.path.exists(data_root):
        print(f"❌ Data root not found: {data_root}")
        print("Please ensure nuScenes dataset is properly set up.")
        return
    
    info_path = os.path.join(data_root, info_file)
    if not os.path.exists(info_path):
        print(f"❌ Info file not found: {info_path}")
        print("Please run gen_info.py first to generate the info files.")
        return
    
    print("✅ Data paths found")
    
    # Create dataset
    print("📦 Creating RVWithImageDataset...")
    dataset = build_dataset(data_root, info_file)
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # Select random sample
    sample_idx = random.randint(0, len(dataset) - 1)
    print(f"🎲 Selected random sample: {sample_idx}")
    
    # Get sample data
    print("📥 Loading sample data...")
    sample_data = dataset[sample_idx]
    
    # Print sample info
    sample_token = sample_data[9]  # sample_token is at index 9 in NuscSCADataset structure
    print(f"📋 Sample token: {sample_token}")
    
    # Visualize projections
    print("🎨 Creating visualization...")
    visualize_projections(sample_data, dataset, sample_idx)
    
    print("\n✅ Demo completed successfully!")
    print("The script shows how lidar points are projected onto each camera view.")


if __name__ == "__main__":
    main()
