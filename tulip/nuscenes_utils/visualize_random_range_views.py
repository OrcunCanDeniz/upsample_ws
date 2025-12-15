#!/usr/bin/env python3
"""
Visualize random nuScenes range-view samples using RVWithImageDataset.

This script constructs the dataset in the same way as demo_lidar_transform.py
and displays randomly selected low-/high-resolution range-view tensors.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add the parent directory to the path to import from tulip
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tulip.util.datasets import (
    RVWithImageDataset,
    transforms,
    ScaleTensor,
    FilterInvalidPixels,
    DownsampleTensor,
    DownsampleTensorWidth,
    LogTransform,
)


def build_dataset(data_root: str, info_file: str, final_dim=(256, 704)) -> RVWithImageDataset:
    """
    Build the RVWithImageDataset with the same transforms as the demo script.
    """
    input_size = (8, 1024)
    output_size = (32, 1024)
    max_range = 1

    t_low_res = [
        transforms.ToTensor(),
        ScaleTensor(1 / 80),
        FilterInvalidPixels(min_range=0, max_range=max_range),
        DownsampleTensor(
            h_high_res=output_size[0],
            downsample_factor=output_size[0] // input_size[0],
        ),
        LogTransform(),
    ]

    t_high_res = [
        transforms.ToTensor(),
        ScaleTensor(1 / 80),
        FilterInvalidPixels(min_range=0, max_range=max_range),
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


def tensor_channel_to_numpy_img(t: torch.Tensor, channel: int = 0) -> np.ndarray:
    """
    Convert a CxHxW tensor to a 2D numpy image by selecting a channel.
    """
    if isinstance(t, torch.Tensor):
        # Expect shape [C, H, W]
        if t.dim() == 3:
            c, h, w = t.shape
            ch = max(0, min(channel, c - 1))
            img = t[ch].detach().cpu().numpy()
        elif t.dim() == 2:
            img = t.detach().cpu().numpy()
        else:
            raise ValueError(f"Unexpected tensor shape: {tuple(t.shape)}")
    else:
        arr = np.asarray(t)
        if arr.ndim == 3:
            ch = max(0, min(channel, arr.shape[0] - 1))
            img = arr[ch]
        elif arr.ndim == 2:
            img = arr
        else:
            raise ValueError(f"Unexpected array shape: {arr.shape}")
    return img


def visualize_samples(dataset: RVWithImageDataset, num_samples: int = 4, seed: int = 42):
    if len(dataset) == 0:
        print("No samples available in dataset.")
        return

    rng = random.Random(seed)
    indices = [rng.randrange(0, len(dataset)) for _ in range(num_samples)]

    ncols = 2  # low-res and high-res
    nrows = num_samples
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 3 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    for row_idx, idx in enumerate(indices):
        sample = dataset[idx]
        (
            _sweep_imgs,
            _sweep_sensor2ego_mats,
            _sweep_intrins,
            _sweep_ida_mats,
            _sweep_sensor2sensor_mats,
            _bda_mat,
            _sweep_timestamps,
            img_metas,
            low_res_rv,
            high_res_rv,
            _lidar2ego_mat,
        ) = sample

        # Visualize the range channel (0) from low/high res tensors
        low_img = tensor_channel_to_numpy_img(low_res_rv, channel=0)
        high_img = tensor_channel_to_numpy_img(high_res_rv, channel=0)

        ax_low = axes[row_idx, 0]
        ax_high = axes[row_idx, 1]

        im0 = ax_low.imshow(low_img, cmap="viridis")
        ax_low.set_title(f"Low-res range | idx={idx}")
        ax_low.axis("off")
        fig.colorbar(im0, ax=ax_low, fraction=0.046, pad=0.04)

        im1 = ax_high.imshow(high_img, cmap="viridis")
        ax_high.set_title(
            f"High-res range | token={str(img_metas.get('token', ''))[:8]}"
        )
        ax_high.axis("off")
        fig.colorbar(im1, ax=ax_high, fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize random nuScenes range-view samples"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/nuscenes",
        help="Root directory to nuScenes data",
    )
    parser.add_argument(
        "--info-file",
        type=str,
        default="nuscenes_upsample_infos_val.pkl",
        help="Info pickle file name placed under data root",
    )
    parser.add_argument(
        "--num-samples", type=int, default=4, help="Number of random samples to show"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling indices"
    )
    args = parser.parse_args()

    info_path = os.path.join(args.data_root, args.info_file)
    if not os.path.exists(args.data_root):
        print(f"❌ Data root not found: {args.data_root}")
        return
    if not os.path.exists(info_path):
        print(f"❌ Info file not found: {info_path}")
        print("Please run gen_info.py first to generate the info files.")
        return

    print("📦 Creating RVWithImageDataset...")
    dataset = build_dataset(args.data_root, args.info_file)
    print(f"✅ Dataset created with {len(dataset)} samples")

    print(f"🎨 Visualizing {args.num_samples} random samples...")
    visualize_samples(dataset, num_samples=args.num_samples, seed=args.seed)


if __name__ == "__main__":
    main()


