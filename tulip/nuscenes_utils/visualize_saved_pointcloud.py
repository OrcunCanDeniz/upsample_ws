#!/usr/bin/env python3
"""
Script to load and visualize the saved pointcloud tensor using Open3D
"""

import torch
import numpy as np
import open3d as o3d
import os

def load_and_visualize_pointcloud():
    """Load the saved pointcloud tensor and visualize it with Open3D"""
    
    # Load the saved tensor
    file_path = "./debug_p_lidar/p_lidar_h_batched.pt"
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return
    
    print(f"Loading pointcloud from {file_path}...")
    p_lidar_h = torch.load(file_path)
    
    print(f"Tensor shape: {p_lidar_h.shape}")
    print(f"Tensor dtype: {p_lidar_h.dtype}")
    print(f"Tensor device: {p_lidar_h.device}")
    
    # Convert to numpy array
    p_lidar_np = p_lidar_h.numpy()
    
    # Handle different tensor shapes
    if len(p_lidar_np.shape) == 3:
        # Shape: [batch_size, num_points, features] where features can be 3 or 4
        print(f"Batch tensor detected with shape: {p_lidar_np.shape}")
        
        # Take the first sample from the batch
        points_full = p_lidar_np[0]  # Shape: [num_points, features]
        
        # Extract only x, y, z coordinates (first 3 features)
        points = points_full[:, :3]  # Shape: [num_points, 3]
        
        # Store the 4th feature (likely intensity) for coloring
        if points_full.shape[1] >= 4:
            intensity = points_full[:, 3]
            print(f"Detected 4D pointcloud with intensity values")
        else:
            intensity = None
            print(f"Detected 3D pointcloud")
        
        print(f"Using first sample from batch. Points shape: {points.shape}")
        
    elif len(p_lidar_np.shape) == 2:
        # Shape: [num_points, features] where features can be 3 or 4
        points_full = p_lidar_np
        
        # Extract only x, y, z coordinates (first 3 features)
        points = points_full[:, :3]  # Shape: [num_points, 3]
        
        # Store the 4th feature (likely intensity) for coloring
        if points_full.shape[1] >= 4:
            intensity = points_full[:, 3]
            print(f"Detected 4D pointcloud with intensity values")
        else:
            intensity = None
            print(f"Detected 3D pointcloud")
        
        print(f"Single pointcloud detected. Points shape: {points.shape}")
    
    else:
        print(f"Unexpected tensor shape: {p_lidar_np.shape}")
        return
    
    # Print some statistics
    print(f"\nPointcloud statistics:")
    print(f"Number of points: {points.shape[0]}")
    print(f"X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Add color based on intensity (if available) or height
    colors = np.zeros_like(points)
    
    if intensity is not None:
        # Color based on intensity values
        print(f"Intensity range: [{intensity.min():.3f}, {intensity.max():.3f}]")
        int_min, int_max = intensity.min(), intensity.max()
        if int_max > int_min:
            normalized_intensity = (intensity - int_min) / (int_max - int_min)
            colors[:, 0] = normalized_intensity  # Red channel based on intensity
            colors[:, 1] = 1 - normalized_intensity  # Green channel (inverse)
            colors[:, 2] = 0.3  # Blue channel constant
        else:
            colors[:, :] = [0.5, 0.5, 0.5]  # Gray if no intensity variation
    else:
        # Color based on height (Z coordinate)
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        if z_max > z_min:
            normalized_z = (points[:, 2] - z_min) / (z_max - z_min)
            colors[:, 0] = normalized_z  # Red channel based on height
            colors[:, 1] = 1 - normalized_z  # Green channel (inverse)
            colors[:, 2] = 0.5  # Blue channel constant
        else:
            colors[:, :] = [0.5, 0.5, 0.5]  # Gray if no height variation
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    print("\nOpening Open3D visualization window...")
    print("Controls:")
    print("- Mouse: Rotate view")
    print("- Scroll: Zoom in/out")
    print("- Right-click + drag: Pan")
    print("- Close window to exit")
    
    o3d.visualization.draw_geometries([pcd], 
                                    window_name="Saved Pointcloud Visualization",
                                    width=1200, 
                                    height=800)
    
    print("Visualization completed!")

if __name__ == "__main__":
    load_and_visualize_pointcloud()
