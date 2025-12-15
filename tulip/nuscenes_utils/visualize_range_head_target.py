import os
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch


def ensure_project_cwd():
    """Change working directory to the `tulip` project root so dataset paths resolve.

    The dataset builder uses a relative root of "./data/nuscenes". This function ensures
    we run from the `tulip` directory where that path exists.
    """
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_file_dir, os.pardir))
    if os.path.basename(project_root) != "tulip":
        # Fallback: walk up until we find the tulip directory that has a data folder
        cur = this_file_dir
        while True:
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            if os.path.basename(parent) == "tulip" and os.path.isdir(os.path.join(parent, "data", "nuscenes")):
                project_root = parent
                break
            cur = parent
    os.chdir(project_root)


def get_dataset(is_train: bool):
    # Import after setting CWD so any relative paths resolve as expected
    from tulip.util.datasets import build_nuscenes_w_image_upsampling_dataset
    return build_nuscenes_w_image_upsampling_dataset(is_train=is_train, log_transform=False)


def visualize_range_head_target(dset, num_samples: int = 5, cmap: str = "viridis"):
    """Iterate over dataset items and visualize the original range view + 2-channel target.

    Top: original high-res range view (post-transform); Middle: mean; Bottom: std.
    Press any key (or close the window) to advance to the next sample.
    """
    plt.ion()
    # Use a GridSpec with a narrow colorbar column to keep plots aligned
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(10, 9))
    gs = GridSpec(nrows=3, ncols=2, width_ratios=[20, 1], figure=fig)
    axes_img = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    axes_cbar = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    for idx in range(min(num_samples, len(dset))):
        sample = dset[idx]
        # original high-res range view is index 9; range_head_target is last
        orig_rv = sample[9]
        rht = sample[-1]
        if isinstance(rht, torch.Tensor):
            rht_np = rht.detach().cpu().numpy()
        else:
            rht_np = np.asarray(rht)

        if isinstance(orig_rv, torch.Tensor):
            orig_np = orig_rv.detach().cpu().numpy()
        else:
            orig_np = np.asarray(orig_rv)

        # orig_np expected shape (1, H, W) after ToTensor; squeeze channel
        if orig_np.ndim == 3 and orig_np.shape[0] == 1:
            orig_np = orig_np[0]

        assert rht_np.ndim == 3 and rht_np.shape[0] == 2, (
            f"Expected (2, H, W) for range_head_target, got {rht_np.shape}")

        mean_map = rht_np[0]
        std_map = rht_np[1]

        for ax in axes_img + axes_cbar:
            ax.clear()
        # Original range view
        im_orig = axes_img[0].imshow(orig_np, cmap=cmap, aspect='auto')
        axes_img[0].set_title("original range view (high-res)")
        fig.colorbar(im_orig, cax=axes_cbar[0])

        # Mean
        im0 = axes_img[1].imshow(mean_map, cmap=cmap, aspect='auto')
        axes_img[1].set_title("range_head_target: mean")
        fig.colorbar(im0, cax=axes_cbar[1])

        # Std
        im1 = axes_img[2].imshow(std_map, cmap=cmap, aspect='auto')
        axes_img[2].set_title("range_head_target: std")
        fig.colorbar(im1, cax=axes_cbar[2])

        fig.suptitle(f"Sample {idx} — shape: {rht_np.shape}")
        plt.tight_layout()
        plt.show()

        # Wait for a key press or window close to continue
        # Return value: True if a key is pressed, False if timeout
        plt.waitforbuttonpress(timeout=-1)

    plt.ioff()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize range_head_target from nuScenes w/ images dataset")
    parser.add_argument("--train", action="store_true", help="Use training split (default: val split)")
    parser.add_argument("--num", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap name")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_cwd()
    dset = get_dataset(is_train=args.train)
    print(f"Loaded dataset with {len(dset)} samples")
    visualize_range_head_target(dset, num_samples=args.num, cmap=args.cmap)


if __name__ == "__main__":
    main()


