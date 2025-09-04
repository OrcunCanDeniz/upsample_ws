import numpy as np
import os
import sys
import pdb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ELEV_DEG_PER_RING_NUCSENES = np.array([-30.67, -29.33, -28., -26.66, -25.33, -24., -22.67, -21.33,
                               -20., -18.67, -17.33, -16., -14.67, -13.33, -12., -10.67,
                                -9.33, -8., -6.66, -5.33, -4., -2.67, -1.33, 0.,
                                1.33, 2.67, 4., 5.33, 6.67, 8., 9.33, 10.67], dtype=np.float32)
ELEV_DEG_PER_RING_NUCSENES_RAD = np.deg2rad(ELEV_DEG_PER_RING_NUCSENES)

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
                 min_depth=0.0,
                 max_depth=100.0,
                 flip_vertical=True,
                 intensity_clip=None,
                 add_channel_dim=True):
        width = 1024
        height = 32
        self.width = int(width)
        self.height = int(height) if height is not None else None
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.flip_vertical = bool(flip_vertical)
        self.intensity_clip = intensity_clip
        self.add_channel_dim = bool(add_channel_dim)

    def __call__(self, pc: np.ndarray):
        if pc.size == 0:
            return self._empty_outputs(inferred_height=self.height)

        assert pc.shape[1] >= 5, "pc must have at least 5 columns: [x,y,z,intensity,ring]"

        x = pc[:, 0].astype(np.float32, copy=False)
        y = pc[:, 1].astype(np.float32, copy=False)
        z = pc[:, 2].astype(np.float32, copy=False)
        intensity = pc[:, 3].astype(np.float32, copy=False)
        ring = pc[:, 4].astype(np.int32, copy=False)

        H = int(self.height)
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

        # row indices: directly use ring information
        rows = np.clip(ringv, 0, H - 1)
        if self.flip_vertical:
            rows = (H - 1 - rows).astype(np.int32, copy=False)

        # cols from azimuth to nearest bin center
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

        range_img = np.zeros((H, W), dtype=np.float32)
        intensity_img = np.zeros((H, W), dtype=np.float32)
        mask_img = np.zeros((H, W), dtype=np.uint8)

        range_img[rr_rows, rr_cols] = rr
        intensity_img[rr_rows, rr_cols] = ii
        mask_img[rr_rows, rr_cols] = 1

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

