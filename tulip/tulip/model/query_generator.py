import math
from typing import Optional, Tuple
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pdb

ELEV_DEG_PER_RING_NUCSENES = torch.tensor([-30.67, -29.33, -28., -26.66, -25.33, -24., -22.67, -21.33,
                               -20., -18.67, -17.33, -16., -14.67, -13.33, -12., -10.67,
                                -9.33, -8., -6.66, -5.33, -4., -2.67, -1.33, 0.,
                                1.33, 2.67, 4., 5.33, 6.67, 8., 9.33, 10.67], dtype=torch.float32)


class SimpleQueryGenerator(nn.Module):
    """
    Given feature map from range view, generate nqw*nqh numver of 3D query points in ego frame.
    """
    def __init__(self,
                 rmax=51.2,
                 C_rv=384,
                 nqw=5,
                 nqh=4,
                 in_rv_size=(2,64),
                 og_rv_size=(32,1024)):
        super().__init__()
        
  
        self.rmax = rmax
        self.n_q_w = nqw
        self.n_q_h = nqh
        self.num_q_per_latent_cell = self.n_q_w * self.n_q_h
        
        # 1D range proposal head for processing expanded features
        self.range_head =  nn.Conv2d(C_rv, 1, 1, bias=True)
        
        # Build azimuth and zenith maps by in_rv_size (width spans [-pi, pi))
        self.og_Hrv, self.og_Wrv = int(og_rv_size[0]), int(og_rv_size[1])
        self.in_Hrv, self.in_Wrv = int(in_rv_size[0]), int(in_rv_size[1])
        self.ds_factor_h = self.og_Hrv // self.in_Hrv
        self.ds_factor_w = self.og_Wrv // self.in_Wrv
        
        self.set_geometry()
        
    def set_geometry(self):
        # bin azimuths into Wrv bins
        assert self.n_q_w * self.in_Wrv <= self.og_Wrv, "Total number  of horizontal samples must be less than or equal to original width"
        az_line = torch.linspace(-math.pi, math.pi, self.og_Wrv + 1)[:-1]
        az_line += (az_line[1] - az_line[0])/2
        self.az_binned_ = az_line.reshape(self.in_Wrv, self.n_q_w, -1)
        self.process_az()

        # bin elevations into Hrv bins
        assert self.n_q_h * self.in_Hrv <= self.og_Hrv, "Total number of vertical samples must be less than or equal to original height"
        elev = torch.deg2rad(ELEV_DEG_PER_RING_NUCSENES.flip(0))
        self.elev_binned_ = elev.reshape(self.in_Hrv, self.n_q_h, -1)
        self.process_elev()
        
        az = self.az_per_q # (in_Hrv, in_Wrv, n_q_w)
        elev = self.elev_per_q # (in_Hrv, in_Wrv, n_q_h)
        # compute unit vector per range view pixel 
        # pdb.set_trace()
        cos_az = torch.cos(az).unsqueeze(-1) # (in_Hrv, in_Wrv, n_q_w, 1)
        sin_az = torch.sin(az).unsqueeze(-1) # (in_Hrv, in_Wrv, n_q_w, 1)
        cos_el = torch.cos(elev).unsqueeze(-2)
        sin_el = torch.sin(elev) # (in_Hrv, in_Wrv, n_q_h)
        
        u_vec_x_grid = cos_az * cos_el# (in_Hrv, in_Wrv, n_q_w, n_q_h)
        u_vec_y_grid = sin_az * cos_el# (in_Hrv, in_Wrv, n_q_w, n_q_h)
        u_vec_z_grid = sin_el.unsqueeze(2).expand_as(u_vec_x_grid)

        u_vec_x = u_vec_x_grid.permute(3,2,0,1)# 
        u_vec_y = u_vec_y_grid.permute(3,2,0,1)# (n_q_h, n_q_w, in_Hrv, in_Wrv)
        u_vec_z = u_vec_z_grid.permute(3,2,0,1)# 
        u_vec = torch.stack([u_vec_x, u_vec_y, u_vec_z], dim=0)
        
        u_vec_x = u_vec_x.unsqueeze(0).contiguous()
        u_vec_y = u_vec_y.unsqueeze(0).contiguous()
        u_vec_z = u_vec_z.unsqueeze(0).contiguous()
        self.register_buffer("u_vec_x", u_vec_x, persistent=False)
        self.register_buffer("u_vec_y", u_vec_y, persistent=False)
        self.register_buffer("u_vec_z", u_vec_z, persistent=False)

        
    def process_az(self):
        # this will process binned azimuth angles to produce useful features
        # self.az_binned_ is (in_Wrv, n_q_w, ds_factor_w//n_q_w)
        # get linearly spaced azimuths for each latent pixel
        assert self.ds_factor_w % self.n_q_w == 0, "in_Wrv must be divisible by n_q_w"
        
        az_per_q = self.az_binned_.mean(-1)
        az_per_q = az_per_q[None, ...]
        self.az_per_q = az_per_q.repeat(self.in_Hrv, 1, 1) # (in_Hrv, in_Wrv, n_q_w)
        az_per_pixel = az_per_q.mean(-1) # (in_Hrv, in_Wrv)
        
    def process_elev(self):
        assert self.ds_factor_h % self.n_q_h == 0, "in_Hrv must be divisible by n_q_h"
        
        elev_per_q = self.elev_binned_.mean(-1) # (in_Hrv, n_q_h)
        elev_per_q = elev_per_q[:, None, :] # (in_Hrv, 1, n_q_h)
        self.elev_per_q = elev_per_q.repeat(1, self.in_Wrv, 1) # (in_Hrv, in_Wrv, n_q_h)
        elev_per_pixel = elev_per_q.mean(-1) # (in_Hrv, in_Wrv)

    def point_hypothesis(self, x_rv):
        """
        Generate points in lidar frame by predicting intermediate normalized depth and unit vectors
        x_rv: [B, Crv, Hrv, Wrv]
        return: [B, Hrv* Wrv, 3]
        """
        B, Crv, Hrv, Wrv = x_rv.shape
        assert Hrv == self.in_Hrv and Wrv == self.in_Wrv, f"Input range view size {Hrv}x{Wrv} does not match expected size {self.in_Hrv}x{self.in_Wrv}"
        # Apply range head to each range view pixel
        interm_depths_norm = self.range_head(x_rv).sigmoid() # normalized intermediate depth
        interm_depths = (interm_depths_norm * self.rmax).detach()
        
        interm_depths = interm_depths.view(B, self.n_q_h, self.n_q_w, self.in_Hrv, self.in_Wrv)

        x_l = (interm_depths * self.u_vec_x).flatten(1) # [B, n_total_q, Hrv, Wrv] 
        y_l = (interm_depths * self.u_vec_y).flatten(1) # [B, n_total_q, Hrv, Wrv] 
        z_l = (interm_depths * self.u_vec_z).flatten(1) # [B, n_total_q, Hrv, Wrv] 
        
        p_lidar_h = torch.stack([x_l, y_l, z_l], dim=-1)
        
        return p_lidar_h, interm_depths_norm
        
        
    def forward(self, x_rv, tf_matrix=None):
        """
        Generate query points in desired frame. One per each pixel in output space.
        x_rv: [B, Crv, H_lrv, W_lrv]
        """

        B, Crv, H_lrv, W_lrv = x_rv.shape # range view latent size
        assert H_lrv == self.in_Hrv and W_lrv == self.in_Wrv, f"Input range view latent size {H_lrv}x{W_lrv} does not match expected size {self.in_Hrv}x{self.in_Wrv}"
        if self.num_q_per_latent_cell >1:
            raise NotImplementedError("Expanding a latent cell is not supported yet")

        points, interm_depths = self.point_hypothesis(x_rv)
        
        if tf_matrix is not None:
            ones = torch.ones_like(points[..., :1])
            points = torch.cat([points, ones], dim=-1) # [B, Hrv* Wrv, 4]

            # LiDAR -> Target frame
            points = points @ tf_matrix.mT
            points = points.view(B, self.og_Hrv, self.og_Wrv, 4)
            
        return points, interm_depths


class UpsampledQueryGenerator(nn.Module):
    """
    Upsample given RV feature map to a higher resolution, and generate query points at the new resolution.
    """
    def __init__(self,
                 rmax=51.2,
                 C_rv=384,
                 nqw=5,
                 nqh=4,
                 in_rv_size=(2,64),
                 og_rv_size=(32,1024)):
        super().__init__(rmax=rmax, C_rv=C_rv, nqw=nqw, nqh=nqh, in_rv_size=in_rv_size, og_rv_size=og_rv_size)
        
        
        

if __name__ == "__main__":
    # Minimal test snippet
    
    qg = SimpleQueryGenerator(rmax=51.2, C_rv=384, nqw=4, nqh=2, in_rv_size=(2,64), og_rv_size=(32,1024))
    
    assert qg.az_per_q.shape == (2, 64, 4)
    assert qg.elev_per_q.shape == (2, 64, 2)
    
    del qg
    
    qg = SimpleQueryGenerator(rmax=51.2, C_rv=384, nqw=1, nqh=1, in_rv_size=(32,1024), og_rv_size=(32,1024))
    
    assert qg.az_per_q.shape == (32, 1024, 1)
    assert qg.elev_per_q.shape == (32, 1024, 1)
    
    del qg
    
    qg = SimpleQueryGenerator(rmax=51.2, C_rv=384, nqw=2, nqh=2, in_rv_size=(16,512), og_rv_size=(32,1024))
    
    assert qg.az_per_q.shape == (16, 512, 2)
    assert qg.elev_per_q.shape == (16, 512, 2)
    
    print("All tests passed!")