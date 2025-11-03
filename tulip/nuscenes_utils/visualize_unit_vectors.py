#!/usr/bin/env python3
"""
Minimal script to generate unit vectors similar to query_generator.py
and visualize the resulting point cloud with Open3D.
"""

import math
import torch
import numpy as np
import open3d as o3d

# NuScenes elevation angles per ring (same as in query_generator.py)
ELEV_DEG_PER_RING_NUCSENES = torch.tensor([-30.67, -29.33, -28., -26.66, -25.33, -24., -22.67, -21.33,
                               -20., -18.67, -17.33, -16., -14.67, -13.33, -12., -10.67,
                                -9.33, -8., -6.66, -5.33, -4., -2.67, -1.33, 0.,
                                1.33, 2.67, 4., 5.33, 6.67, 8., 9.33, 10.67], dtype=torch.float32)

def generate_unit_vectors_exact(nqw=1, nqh=1, in_rv_size=(32, 1024), og_rv_size=(32, 1024)):
    """
    Generate unit vectors exactly like SimpleQueryGenerator.
    
    Args:
        nqw: Number of queries per width
        nqh: Number of queries per height
        in_rv_size: Input range view size (height, width)
        og_rv_size: Original range view size (height, width)
    
    Returns:
        torch.Tensor: Unit vectors of shape (H, W, num_q_per_latent_cell, 3)
    """
    print(f"Generating unit vectors exactly like SimpleQueryGenerator...")
    print(f"nqw={nqw}, nqh={nqh}, in_rv_size={in_rv_size}, og_rv_size={og_rv_size}")
    
    # Parameters from SimpleQueryGenerator
    num_q_per_latent_cell = nqw * nqh
    og_Hrv, og_Wrv = int(og_rv_size[0]), int(og_rv_size[1])
    in_Hrv, in_Wrv = int(in_rv_size[0]), int(in_rv_size[1])
    ds_factor_h = og_Hrv // in_Hrv
    ds_factor_w = og_Wrv // in_Wrv
    
    # bin azimuths into Wrv bins (exactly like query_generator.py)
    assert nqw * in_Wrv <= og_Wrv, "Total number of horizontal samples must be less than or equal to original width"
    az_line = torch.linspace(-math.pi, math.pi, og_Wrv + 1)[:-1]
    az_line += (az_line[1] - az_line[0]) / 2
    az_binned_ = az_line.reshape(in_Wrv, nqw, -1)
    
    # Process azimuth (exactly like query_generator.py)
    assert ds_factor_w % nqw == 0, "in_Wrv must be divisible by n_q_w"
    az_per_q = az_binned_.mean(-1)
    az_per_q = az_per_q[None, ...]
    az_per_q = az_per_q.repeat(in_Hrv, 1, 1)  # (in_Hrv, in_Wrv, n_q_w)
    
    # bin elevations into Hrv bins (exactly like query_generator.py)
    assert nqh * in_Hrv <= og_Hrv, "Total number of vertical samples must be less than or equal to original height"
    elev = torch.deg2rad(ELEV_DEG_PER_RING_NUCSENES.flip(0))
    elev_binned_ = elev.reshape(in_Hrv, nqh, -1)
    
    # Process elevation (exactly like query_generator.py)
    assert ds_factor_h % nqh == 0, "in_Hrv must be divisible by n_q_h"
    elev_per_q = elev_binned_.mean(-1)  # (in_Hrv, n_q_h)
    elev_per_q = elev_per_q[:, None, :]  # (in_Hrv, 1, n_q_h)
    elev_per_q = elev_per_q.repeat(1, in_Wrv, 1)  # (in_Hrv, in_Wrv, n_q_h)
    
    # Compute unit vectors exactly like query_generator.py
    az = az_per_q
    elev = elev_per_q
    
    # compute unit vector per range view pixel 
    cos_az = torch.cos(az).unsqueeze(-1)
    sin_az = torch.sin(az).unsqueeze(-1)
    cos_el = torch.cos(elev).unsqueeze(-2)
    sin_el = torch.sin(elev)
    
    u_vec_x_grid = cos_az * cos_el
    u_vec_y_grid = sin_az * cos_el
    u_vec_z_grid = sin_el.unsqueeze(2).expand_as(u_vec_x_grid)

    u_vec_x = u_vec_x_grid.reshape(in_Hrv, in_Wrv, num_q_per_latent_cell)
    u_vec_y = u_vec_y_grid.reshape(in_Hrv, in_Wrv, num_q_per_latent_cell)
    u_vec_z = u_vec_z_grid.reshape(in_Hrv, in_Wrv, num_q_per_latent_cell)
    u_vec = torch.stack([u_vec_x, u_vec_y, u_vec_z], dim=-1)
    
    print(f"Generated unit vectors with shape: {u_vec.shape}")
    print(f"az_per_q shape: {az_per_q.shape}")
    print(f"elev_per_q shape: {elev_per_q.shape}")
    
    return u_vec

def create_point_cloud(unit_vectors, range_value=10.0):
    """
    Create point cloud by multiplying unit vectors with range.
    
    Args:
        unit_vectors: Unit vectors of shape (H, W, num_q_per_latent_cell, 3)
        range_value: Range value to multiply with unit vectors
    
    Returns:
        numpy.ndarray: Point cloud coordinates of shape (N, 3)
    """
    # Convert to numpy and apply range
    points = unit_vectors.numpy() * range_value
    
    # Reshape to (N, 3) for point cloud
    points = points.reshape(-1, 3)
    
    return points

def visualize_point_cloud(points, title="Unit Vector Point Cloud"):
    """
    Visualize point cloud using Open3D with coordinate frame.
    
    Args:
        points: Point cloud coordinates of shape (N, 3)
        title: Window title for visualization
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Color points based on height (Z coordinate)
    colors = np.zeros_like(points)
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    if z_max > z_min:
        colors[:, 2] = (points[:, 2] - z_min) / (z_max - z_min)  # Blue channel based on height
        colors[:, 1] = 1.0 - colors[:, 2]  # Green channel (inverse of blue)
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    
    # Visualize
    print(f"Visualizing {len(points)} points...")
    print(f"Point cloud bounds: X=[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
          f"Y=[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
          f"Z=[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    print("Coordinate frame: Red=X, Green=Y, Blue=Z")
    
    o3d.visualization.draw_geometries([pcd, coordinate_frame], window_name=title, width=800, height=600)

def main():
    """Main function to generate unit vectors and visualize point cloud."""
    print("Generating unit vectors exactly like SimpleQueryGenerator...")
    
    # Test case 1: Simple case with nqw=1, nqh=1 (like UpsampledQueryGenerator)
    print("\n=== Test Case 1: nqw=1, nqh=1, in_rv_size=(32,1024) ===")
    unit_vectors_1 = generate_unit_vectors_exact(nqw=1, nqh=1, in_rv_size=(32, 1024), og_rv_size=(32, 1024))
    
    # Create point cloud with arbitrary range
    range_value = 15.0  # meters
    points_1 = create_point_cloud(unit_vectors_1, range_value)
    print(f"Created point cloud with {len(points_1)} points at range {range_value}m")
    
    # Visualize the point cloud
    visualize_point_cloud(points_1, f"Unit Vector Point Cloud (nqw=1, nqh=1, Range: {range_value}m)")
    
    # Test case 2: Multiple queries per cell
    print("\n=== Test Case 2: nqw=4, nqh=4, in_rv_size=(2,64) ===")
    unit_vectors_2 = generate_unit_vectors_exact(nqw=4, nqh=4, in_rv_size=(2, 64), og_rv_size=(32, 1024))
    
    points_2 = create_point_cloud(unit_vectors_2, range_value)
    print(f"Created point cloud with {len(points_2)} points at range {range_value}m")
    
    # Visualize the second point cloud
    visualize_point_cloud(points_2, f"Unit Vector Point Cloud (nqw=4, nqh=4, Range: {range_value}m)")
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()