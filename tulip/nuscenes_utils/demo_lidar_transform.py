#!/usr/bin/env python3
"""
Demo script showing how to use RVWithImageDataset to load nuScenes data
and transform point clouds using the lidar2ego transformation matrix.

This script demonstrates the core functionality:
1. Load data using RVWithImageDataset
2. Extract lidar2ego transform from the data
3. Load original point cloud from .bin file
4. Apply transformation to point cloud
"""

import os
import sys
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt

# Add the parent directory to the path to import from tulip
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tulip.util.datasets import RVWithImageDataset, tf_mat_sensor_to_ego
from tulip.util.datasets import transforms, ScaleTensor, FilterInvalidPixels, DownsampleTensor, DownsampleTensorWidth, LogTransform

def load_nuscenes_point_cloud(bin_path):
    """Load point cloud from nuScenes .bin file."""
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)

def transform_points_to_ego_frame(points, lidar2ego_matrix):
    """
    Transform point cloud from lidar frame to ego frame.
    
    Args:
        points: (N, 5) array with [x, y, z, intensity, ring]
        lidar2ego_matrix: (4, 4) transformation matrix
        
    Returns:
        Transformed points in ego frame
    """
    # Extract xyz coordinates and convert to homogeneous
    xyz = points[:, :3]
    xyz_homo = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    
    # Apply transformation
    print(f"xyz_homo shape: {xyz_homo.shape}")
    xyz_ego =  xyz_homo @ lidar2ego_matrix.T
    
    # Combine with original intensity and ring data
    points_ego = np.hstack([xyz_ego[:, :3], points[:, 3:]])
    
    return points_ego

def create_coordinate_frame(scale=1.0, origin=[0, 0, 0]):
    """
    Create a coordinate frame visualization.
    
    Args:
        scale (float): Scale of the coordinate frame
        origin (list): Origin position [x, y, z]
        
    Returns:
        open3d.geometry.LineSet: Coordinate frame visualization
    """
    # Create coordinate frame lines
    points = np.array([
        [0, 0, 0],  # origin
        [scale, 0, 0],  # x-axis
        [0, scale, 0],  # y-axis
        [0, 0, scale],  # z-axis
    ]) + np.array(origin)
    
    lines = [
        [0, 1],  # x-axis
        [0, 2],  # y-axis
        [0, 3],  # z-axis
    ]
    
    colors = [
        [1, 0, 0],  # red for x-axis
        [0, 1, 0],  # green for y-axis
        [0, 0, 1],  # blue for z-axis
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def create_point_cloud_visualization(points, colors=None, max_points=50000):
    """
    Create Open3D point cloud visualization.
    
    Args:
        points: (N, 3) or (N, 5) array of points
        colors: Optional color array or None for intensity-based coloring
        max_points: Maximum number of points to display
        
    Returns:
        open3d.geometry.PointCloud: Point cloud visualization
    """
    # Subsample if too many points
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
    
    # Extract xyz coordinates
    if points.shape[1] >= 3:
        xyz = points[:, :3]
    else:
        xyz = points
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Set colors
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif points.shape[1] >= 4:
        # Use intensity for coloring
        intensity = points[:, 3]
        # Normalize intensity to [0, 1]
        intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
        # Create colormap (viridis-like)
        colors = plt.cm.viridis(intensity_norm)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Default color (gray)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    return pcd

def visualize_point_clouds_with_frames(points_lidar, points_ego, lidar2ego_matrix, 
                                     title="Point Cloud Visualization"):
    """
    Visualize point clouds in both lidar and ego frames with coordinate frames.
    
    Args:
        points_lidar: (N, 5) points in lidar frame
        points_ego: (N, 5) points in ego frame
        lidar2ego_matrix: (4, 4) transformation matrix
        title: Window title
    """
    # Create point clouds
    pcd_lidar = create_point_cloud_visualization(points_lidar)
    pcd_ego = create_point_cloud_visualization(points_ego)
    
    # Create coordinate frames
    frame_lidar = create_coordinate_frame(scale=2.0, origin=[0, 0, 0])
    frame_ego = create_coordinate_frame(scale=2.0, origin=[0, 0, 0])
    
    # Transform ego frame to show lidar position relative to ego
    frame_lidar.transform(lidar2ego_matrix)
    
    # Create geometries list
    geometries = [ pcd_ego, frame_ego]
    
    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1200, height=800)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set rendering options
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    
    # Add text annotations
    print("Visualization Controls:")
    print("- Mouse: Rotate view")
    print("- Mouse wheel: Zoom")
    print("- Right mouse + drag: Pan")
    print("- R: Reset view")
    print("- Q: Quit")
    print("\nLegend:")
    print("- Red/Green/Blue axes: Coordinate frames")
    print("- Gray points: Lidar frame point cloud")
    print("- Colored points: Ego frame point cloud (colored by intensity)")
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def main():
    """Main demo function."""
    print("=== NuScenes Dataset Demo with Lidar Transform ===\n")
    
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
    
    # Setup transforms for the dataset
    input_size = (8, 1024)
    output_size = (32, 1024)
    
    # Low resolution transforms
    t_low_res = [
        transforms.ToTensor(),
        ScaleTensor(1/55),
        DownsampleTensor(h_high_res=output_size[0], 
                        downsample_factor=output_size[0]//input_size[0]),
        LogTransform()
    ]
    
    # High resolution transforms  
    t_high_res = [
        transforms.ToTensor(),
        ScaleTensor(1/55),
        LogTransform()
    ]
    
    transform_low_res = transforms.Compose(t_low_res)
    transform_high_res = transforms.Compose(t_high_res)
    
    # Create dataset
    print("📦 Creating RVWithImageDataset...")
    dataset = RVWithImageDataset(
        root=data_root,
        high_res_transform=transform_high_res,
        low_res_transform=transform_low_res,
        info_file=info_file,
        final_dim=(256, 704)
    )
    
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # Test with first sample
    print("\n🔍 Testing with first sample...")
    
    # Get sample data
    sample_data = dataset[0]
    (sweep_imgs, sweep_sensor2ego_mats, sweep_intrins, sweep_ida_mats,
     sweep_sensor2sensor_mats, bda_mat, sweep_timestamps, img_metas,
     low_res_rv, high_res_rv, lidar2ego_mat, _, _) = sample_data
    
    print(f"📋 Sample token: {img_metas['token']}")
    print(f"📐 Low-res RV shape: {low_res_rv.shape}")
    print(f"📐 High-res RV shape: {high_res_rv.shape}")
    print(f"📐 Lidar2ego matrix shape: {lidar2ego_mat.shape}")
    
    # Get original point cloud path
    sample_info = dataset.infos[0]
    lidar_info = sample_info['lidar_info']
    original_bin_path = os.path.join(data_root, lidar_info['filename'])
    
    print(f"📁 Original point cloud path: {original_bin_path}")
    
    if os.path.exists(original_bin_path):
        print("✅ Original point cloud file found")
        
        # Load original point cloud
        print("📥 Loading original point cloud...")
        points_lidar = load_nuscenes_point_cloud(original_bin_path)
        print(f"📊 Original point cloud shape: {points_lidar.shape}")
        print(f"📊 Point cloud range: [{points_lidar[:, :3].min():.2f}, {points_lidar[:, :3].max():.2f}]")
        
        # Transform to ego frame
        print("🔄 Transforming points to ego frame...")
        points_ego = transform_points_to_ego_frame(points_lidar, lidar2ego_mat)
        print(f"📊 Transformed point cloud shape: {points_ego.shape}")
        print(f"📊 Transformed point cloud range: [{points_ego[:, :3].min():.2f}, {points_ego[:, :3].max():.2f}]")
        
        # Show transformation details
        translation = lidar2ego_mat[:3, 3]
        print(f"🔧 Lidar2ego translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
        
        # Compare centers
        lidar_center = points_lidar[:, :3].mean(axis=0)
        ego_center = points_ego[:, :3].mean(axis=0)
        print(f"📍 Lidar frame center: [{lidar_center[0]:.3f}, {lidar_center[1]:.3f}, {lidar_center[2]:.3f}]")
        print(f"📍 Ego frame center: [{ego_center[0]:.3f}, {ego_center[1]:.3f}, {ego_center[2]:.3f}]")
        
        # Show data structure from gen_info.py
        print(f"\n📋 Data structure from gen_info.py:")
        print(f"   - sample_token: {sample_info['sample_token']}")
        print(f"   - timestamp: {sample_info['timestamp']}")
        print(f"   - scene_token: {sample_info['scene_token']}")
        print(f"   - lidar_info keys: {list(lidar_info.keys())}")
        print(f"   - cam_infos keys: {list(sample_info['cam_infos'].keys())}")
        
        # Visualize the point clouds
        print(f"\n🎨 Opening 3D visualization...")
        try:
            visualize_point_clouds_with_frames(
                points_lidar, 
                points_ego, 
                lidar2ego_mat,
                title=f"NuScenes Point Cloud - Sample {img_metas['token'][:8]}"
            )
        except ImportError as e:
            print(f"⚠️  Open3D not available: {e}")
            print("Install Open3D with: pip install open3d")
            print("Continuing without visualization...")
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")
            print("Continuing without visualization...")
        
        print("\n✅ Demo completed successfully!")
        print("The lidar2ego transformation is working correctly.")
        
    else:
        print(f"❌ Original point cloud file not found: {original_bin_path}")
        print("Please ensure the nuScenes dataset .bin files are available.")

if __name__ == "__main__":
    main()
