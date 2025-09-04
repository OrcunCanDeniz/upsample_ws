import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False


# Ensure we can import from the inner `tulip` package directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)  # /.../tulip
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from tulip.util.evaluation import img_to_pcd_nuscenes  # noqa: E402
from nuscenes_utils.sample_nuscenes_dataset import NuScenesPointCloudToRangeImage  # noqa: E402

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize nuScenes range image and reconstructed point cloud")
    parser.add_argument("npy_path", type=str, help="Path to .npy file saved by sample_nuscenes_dataset.py")
    parser.add_argument("--maximum_range", type=float, default=80.0, help="Max range used when range is normalized [0,1]")
    parser.add_argument("--sample", type=int, default=150000, help="Max number of points to plot (subsample if larger)")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap for range image")
    parser.add_argument("--no_range_plot", action="store_true", help="Skip matplotlib range image plot")
    parser.add_argument("--save_range", type=str, default=None, help="Save range image to this path (no legend/axes)")
    parser.add_argument("--save_dpi", type=int, default=150, help="DPI when saving range image")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.npy_path):
        raise FileNotFoundError(f"File not found: {args.npy_path}")

    # We only use the provided path to extract the token and locate GT .bin
    npy_dir = os.path.dirname(os.path.abspath(args.npy_path))
    npy_base = os.path.basename(args.npy_path)
    if not npy_base.endswith("_RV.npy"):
        print(f"Warning: filename does not end with '_RV.npy': {npy_base}")
    token = npy_base.replace("_RV.npy", "").replace(".npy", "")
    # ../samples/LIDAR_TOP/<TOKEN>.bin relative to the npy_dir
    original_bin_path = os.path.abspath(os.path.join(npy_dir, os.pardir, "samples", "LIDAR_TOP", f"{token}.pcd.bin"))

    # Load original point cloud from .bin (required)
    if not os.path.isfile(original_bin_path):
        raise FileNotFoundError(f"Original point cloud not found at: {original_bin_path}")
    try:
        raw = np.fromfile(original_bin_path, dtype=np.float32).reshape(-1, 5)
        orig_points = raw[:, :3].astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load original point cloud: {e}")

    # Render ground-truth point cloud to a range view using the same converter
    flip_vertical = True
    converter = NuScenesPointCloudToRangeImage(flip_vertical=flip_vertical)
    rv = converter(raw)
    range_img = rv[..., 0].astype(np.float32)

    # Optional: Visualize range image (show values as-is; could be meters or normalized)
    if not args.no_range_plot:
        plt.figure(figsize=(7, 5))
        im = plt.imshow(range_img, cmap=args.cmap)
        plt.title("Range image")
        plt.xlabel("u (cols)")
        plt.ylabel("v (rows)")
        plt.colorbar(im, fraction=0.046, pad=0.04)

    # Reconstruct point cloud from the rendered range view
    points = img_to_pcd_nuscenes(rv[...,0], maximum_range=args.maximum_range, flip_vertical=flip_vertical, eval=False)

    if points.shape[0] == 0:
        print("No valid points reconstructed from the image.")
        plt.show()
        return

    # Subsample for visualization if too many points
    if points.shape[0] > args.sample:
        idx = np.random.choice(points.shape[0], args.sample, replace=False)
        vis_pts = points[idx]
    else:
        vis_pts = points

    # orig_points already loaded above

    if not _HAS_O3D:
        print("Open3D is not installed. Install with: pip install open3d")
        if not args.no_range_plot:
            plt.show()
        return

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vis_pts[:, :3])

    # Color reconstructed (range-view) cloud solid red
    rv_color = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(rv_color, (vis_pts.shape[0], 1)))

    geometries = [pcd]

    # Original point cloud geometry (colored differently)
    if orig_points is not None and orig_points.shape[0] > 0:
        if orig_points.shape[0] > args.sample:
            idx2 = np.random.choice(orig_points.shape[0], args.sample, replace=False)
            orig_vis = orig_points[idx2]
        else:
            orig_vis = orig_points
        pcd_orig = o3d.geometry.PointCloud()
        pcd_orig.points = o3d.utility.Vector3dVector(orig_vis[:, :3])
        # Color original (ground truth) cloud solid green
        gt_color = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        pcd_orig.colors = o3d.utility.Vector3dVector(np.tile(gt_color, (orig_vis.shape[0], 1)))
        geometries.append(pcd_orig)
        print("Open3D: showing reconstructed (red) and original (green) point clouds together.")
    else:
        print("Open3D: showing reconstructed point cloud (red). Original not available.")

    # Visualize with Open3D (blocks until window is closed)
    o3d.visualization.draw_geometries(geometries, window_name="nuScenes Reconstructed vs Original Point Clouds",
                                      width=1280, height=720, left=100, top=100,
                                      point_show_normal=False)

    # Show the range plot after closing o3d window, if requested
    if not args.no_range_plot:
        plt.show()


if __name__ == "__main__":
    main()


