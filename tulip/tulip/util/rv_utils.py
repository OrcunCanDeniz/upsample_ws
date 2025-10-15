# Range view processing utils for nuscenes

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb

ELEV_DEG_PER_RING_NUCSENES = np.array([-30.67, -29.33, -28., -26.66, -25.33, -24., -22.67, -21.33,
                               -20., -18.67, -17.33, -16., -14.67, -13.33, -12., -10.67,
                                -9.33, -8., -6.66, -5.33, -4., -2.67, -1.33, 0.,
                                1.33, 2.67, 4., 5.33, 6.67, 8., 9.33, 10.67], dtype=np.float32)
ELEV_DEG_PER_RING_NUCSENES_RAD = np.deg2rad(ELEV_DEG_PER_RING_NUCSENES)


def compute_polar_ang_per_pixel(rv_size):
    """
    Given sensor vfov, quantize fov into RV pixels, return the bin center for each pixel.
    rv_size : spatial dim of RV representation
    """
    
    rv_h, rv_w = int(rv_size[0]), int(rv_size[1])
    az_line = torch.linspace(-math.pi, math.pi, rv_w + 1)[:-1]   # [Wrv]
    az = az_line.view(1,rv_w).expand(rv_h,rv_w)
    az_step = az_line[1] - az_line[0]
    az = az + 0.5 * az_step   # center of each bin
    
    elev_segments = np.split(ELEV_DEG_PER_RING_NUCSENES_RAD[::-1], rv_h)
    elev_bin_bounds = [(elev_bin.max(), elev_bin.min()) for elev_bin in elev_segments]
    elev_bin_bounds = np.array(elev_bin_bounds)
    elev = np.array(elev_segments)
    elev = elev.mean(axis=1).repeat(rv_w).reshape(rv_h,rv_w)
    
    return az, elev, az_step, elev_bin_bounds
    
def compute_uvec_per_pixel(az_table, elev_table):
    """
    Compute unit vectors given the ray direction of RV pixel.
    """
    cos_az = np.cos(az)
    sin_az = np.sin(az)
    cos_el = np.cos(elev)
    sin_el = np.sin(elev)
    u_vec_x = cos_az * cos_el
    u_vec_y = sin_az * cos_el
    u_vec_z = sin_el[:, :]
    u_vec = np.stack([u_vec_x, u_vec_y, u_vec_z])
    
    return u_vec


if __name__ == "__main__":
    
    az, elev, az_step, elev_step = compute_polar_ang_per_pixel((2,64))
    uvec = compute_uvec_per_pixel(az, elev)
    
    print(f"az table shape: {az.shape}")
    print(f"eleb table shape: {elev.shape}")
    print(f"uvec table shape: {uvec.shape}")