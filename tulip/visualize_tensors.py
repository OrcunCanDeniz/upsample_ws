#!/usr/bin/env python3
"""
Script to load and visualize in_imgs and depths tensors saved to disk.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path

def load_tensors(imgs_path, depths_path):
    """
    Load the tensors from disk.
    
    Args:
        imgs_path: Path to the images tensor file (.pt)
        depths_path: Path to the depths tensor file (.pt)
    
    Returns:
        tuple: (in_imgs, depths) tensors
    """
    print(f"Loading images from: {imgs_path}")
    in_imgs = torch.load(imgs_path, map_location='cpu')
    print(f"Images tensor shape: {in_imgs.shape}")
    
    print(f"Loading depths from: {depths_path}")
    depths = torch.load(depths_path, map_location='cpu')
    print(f"Depths tensor shape: {depths.shape}")
    
    return in_imgs, depths

def visualize_images(in_imgs, max_images=6, figsize=(15, 10)):
    """
    Visualize the input images.
    
    Args:
        in_imgs: Tensor of shape [B, 1, N, C, H, W] where N is number of cameras
        max_images: Maximum number of images to display
        figsize: Figure size for matplotlib
    """
    # Convert to numpy and handle different tensor shapes
    if in_imgs.dim() == 6:  # [B, S, N, C, H, W]
        print(f"in_imgs shape: {in_imgs.shape}")
        B, S, N, C, H, W = in_imgs.shape
        if S != 1:
            in_imgs = in_imgs[:,:1,:,:,:,:]
        print(f"in_imgs shape: {in_imgs.shape}")
        # Reshape to [B*N, C, H, W] for easier visualization
        imgs_vis = in_imgs.reshape(B * N, C, H, W)
    elif in_imgs.dim() == 5:  # [B, N, C, H, W]
        B, N, C, H, W = in_imgs.shape
        imgs_vis = in_imgs.reshape(B * N, C, H, W)
    elif in_imgs.dim() == 4:  # [B, C, H, W]
        imgs_vis = in_imgs
    else:
        raise ValueError(f"Unexpected tensor shape: {in_imgs.shape}")
    
    # Limit number of images to display
    num_images = min(imgs_vis.shape[0], max_images)
    
    # Create subplot grid
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(num_images):
        img = imgs_vis[i].detach().cpu().numpy()
        
        # Handle different channel configurations
        if img.shape[0] == 3:  # RGB
            img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        elif img.shape[0] == 1:  # Grayscale
            img = img[0]  # Remove channel dimension
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        else:
            # For other channel counts, take the first channel
            img = img[0]
            img = (img - img.min()) / (img.max() - img.min())
        
        axes[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[i].set_title(f'Image {i}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Input Images Visualization', fontsize=16, y=0.98)
    plt.show()

def visualize_depths(depths, max_images=6, figsize=(15, 10), bin_size=0.5):
    """
    Visualize the depth maps from probability distributions.
    
    Args:
        depths: Tensor of shape [B*N, D, H, W] or [B, N, D, H, W] - depth probability distribution
        max_images: Maximum number of depth maps to display
        figsize: Figure size for matplotlib
        bin_size: Size of each depth bin in meters (default: 0.5m)
    """
    # Handle different tensor shapes
    if depths.dim() == 5:  # [B, N, D, H, W]
        B, N, D, H, W = depths.shape
        depths_vis = depths.reshape(B * N, D, H, W)
    elif depths.dim() == 4:  # [B*N, D, H, W]
        depths_vis = depths
    else:
        raise ValueError(f"Unexpected tensor shape: {depths.shape}")
    
    # Limit number of images to display
    num_images = min(depths_vis.shape[0], max_images)
    
    # Create subplot grid
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(num_images):
        depth_probs = depths_vis[i].detach().cpu().numpy()  # Shape: [D, H, W]
        
        # Find the highest probability bin for each pixel
        max_prob_bins = np.argmax(depth_probs, axis=0)  # Shape: [H, W]
        
        # Convert bin indices to actual depth values
        # Each bin represents a depth range, so we use the bin index * bin_size
        depth_vis = max_prob_bins.astype(np.float32) * bin_size
        
        # Normalize depth for visualization (optional, for better contrast)
        depth_vis_normalized = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min())
        
        im = axes[i].imshow(depth_vis_normalized, cmap='viridis')
        axes[i].set_title(f'Depth {i} (max: {depth_vis.max():.1f}m)')
        axes[i].axis('off')
        
        # Add colorbar with actual depth values
        cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.set_label('Depth (m)')
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Depth Maps Visualization (Highest Probability Bins)', fontsize=16, y=0.98)
    plt.show()

def visualize_depth_bins_unused(depths, depth_idx=0, max_images=6, figsize=(15, 10), bin_size=0.5):
    """
    Visualize specific depth bins from the depth probability distribution.
    
    Args:
        depths: Tensor of shape [B*N, D, H, W] or [B, N, D, H, W] - depth probability distribution
        depth_idx: Index of the depth bin to visualize
        max_images: Maximum number of images to display
        figsize: Figure size for matplotlib
        bin_size: Size of each depth bin in meters (default: 0.5m)
    """
    # Handle different tensor shapes
    if depths.dim() == 5:  # [B, N, D, H, W]
        B, N, D, H, W = depths.shape
        depths_vis = depths.reshape(B * N, D, H, W)
    elif depths.dim() == 4:  # [B*N, D, H, W]
        depths_vis = depths
    else:
        raise ValueError(f"Unexpected tensor shape: {depths.shape}")
    
    if depth_idx >= depths_vis.shape[1]:
        print(f"Warning: depth_idx {depth_idx} is out of range. Available depth bins: {depths_vis.shape[1]}")
        depth_idx = 0
    
    # Limit number of images to display
    num_images = min(depths_vis.shape[0], max_images)
    
    # Create subplot grid
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(num_images):
        depth_prob = depths_vis[i, depth_idx].detach().cpu().numpy()  # Shape: [H, W]
        
        # Normalize probability for visualization
        depth_prob_normalized = (depth_prob - depth_prob.min()) / (depth_prob.max() - depth_prob.min())
        
        im = axes[i].imshow(depth_prob_normalized, cmap='viridis')
        axes[i].set_title(f'Depth Bin {depth_idx} (depth: {depth_idx * bin_size:.1f}m) - Image {i}')
        axes[i].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.set_label('Probability')
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Depth Bin {depth_idx} Probability Visualization (Depth: {depth_idx * bin_size:.1f}m)', fontsize=16, y=0.98)
    plt.show()

def visualize_depths_raw_unused(depths, max_images=6, figsize=(15, 10), bin_size=0.5):
    """
    Visualize the raw depth values (not normalized) from probability distributions.
    
    Args:
        depths: Tensor of shape [B*N, D, H, W] or [B, N, D, H, W] - depth probability distribution
        max_images: Maximum number of depth maps to display
        figsize: Figure size for matplotlib
        bin_size: Size of each depth bin in meters (default: 0.5m)
    """
    # Handle different tensor shapes
    if depths.dim() == 5:  # [B, N, D, H, W]
        B, N, D, H, W = depths.shape
        depths_vis = depths.reshape(B * N, D, H, W)
    elif depths.dim() == 4:  # [B*N, D, H, W]
        depths_vis = depths
    else:
        raise ValueError(f"Unexpected tensor shape: {depths.shape}")
    
    # Limit number of images to display
    num_images = min(depths_vis.shape[0], max_images)
    
    # Create subplot grid
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(num_images):
        depth_probs = depths_vis[i].detach().cpu().numpy()  # Shape: [D, H, W]
        
        # Find the highest probability bin for each pixel
        max_prob_bins = np.argmax(depth_probs, axis=0)  # Shape: [H, W]
        
        # Convert bin indices to actual depth values
        depth_vis = max_prob_bins.astype(np.float32) * bin_size
        
        im = axes[i].imshow(depth_vis, cmap='viridis', vmin=0, vmax=depth_vis.max())
        axes[i].set_title(f'Depth {i} (max: {depth_vis.max():.1f}m)')
        axes[i].axis('off')
        
        # Add colorbar with actual depth values
        cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.set_label('Depth (m)')
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Raw Depth Maps (Highest Probability Bins)', fontsize=16, y=0.98)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize in_imgs and depths tensors')
    parser.add_argument('--set', type=str, default='tulip',
                        help='Set to visualize')
    parser.add_argument('--imgs_path', type=str, default='',
                        help='Path to the images tensor file')
    parser.add_argument('--depths_path', type=str, default='',
                        help='Path to the depths tensor file')
    parser.add_argument('--max_images', type=int, default=6,
                        help='Maximum number of images to display')
    parser.add_argument('--bin_size', type=float, default=0.5,
                        help='Size of each depth bin in meters (default: 0.5m)')
    parser.add_argument('--show_images', action='store_true', default=True,
                        help='Show input images')
    parser.add_argument('--show_depths', action='store_true', default=True,
                        help='Show depth maps')
    
    args = parser.parse_args()
    
    if args.set == 'tulip':
        args.imgs_path = './imgs.pt'
        args.depths_path = './depths.pt'
    elif args.set == 'bevdepth':
        args.imgs_path = '../BEVDepth/debug_outputs/bd_x_tensor.pt'
        args.depths_path = '../BEVDepth/debug_outputs/bd_depths_tensor.pt'
    else:
        print(f"Error: Invalid set: {args.set}")
        return
    
    # Check if files exist
    if not os.path.exists(args.imgs_path):
        print(f"Error: Images file not found: {args.imgs_path}")
        return
    
    if not os.path.exists(args.depths_path):
        print(f"Error: Depths file not found: {args.depths_path}")
        return
    
    # Load tensors
    in_imgs, depths = load_tensors(args.imgs_path, args.depths_path)
    
    # Visualize images
    if args.show_images:
        print("\nVisualizing input images...")
        visualize_images(in_imgs, max_images=args.max_images)
    
    # Visualize depths
    if args.show_depths:
        print("\nVisualizing depth maps...")
        visualize_depths(depths, max_images=args.max_images, bin_size=args.bin_size)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
