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
        self.range_head_1d = nn.Sequential(
            nn.Linear(C_rv + 3, 128, bias=True),
            nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 64, bias=True),
            nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 1, bias=True) # depth distribution logits for single query
        )
        
        # Build azimuth and zenith maps by in_rv_size (width spans [-pi, pi))
        self.og_Hrv, self.og_Wrv = int(og_rv_size[0]), int(og_rv_size[1])
        self.in_Hrv, self.in_Wrv = int(in_rv_size[0]), int(in_rv_size[1])
        self.ds_factor_h = self.og_Hrv // self.in_Hrv
        self.ds_factor_w = self.og_Wrv // self.in_Wrv
        
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
        
        az = self.az_per_q
        elev = self.elev_per_q
        # compute unit vector per range view pixel 
        cos_az = torch.cos(az).unsqueeze(-1)
        sin_az = torch.sin(az).unsqueeze(-1)
        cos_el = torch.cos(elev).unsqueeze(-2)
        sin_el = torch.sin(elev)
        
        u_vec_x_grid = cos_az * cos_el
        u_vec_y_grid = sin_az * cos_el
        u_vec_z_grid = sin_el.unsqueeze(2).expand_as(u_vec_x_grid)


        u_vec_x = u_vec_x_grid.reshape(self.in_Hrv, self.in_Wrv, self.num_q_per_latent_cell)
        u_vec_y = u_vec_y_grid.reshape(self.in_Hrv, self.in_Wrv, self.num_q_per_latent_cell)
        u_vec_z = u_vec_z_grid.reshape(self.in_Hrv, self.in_Wrv, self.num_q_per_latent_cell)
        u_vec = torch.stack([u_vec_x, u_vec_y, u_vec_z], dim=-1)
        
        
        u_vec_mean = u_vec.mean(-2)
        u_vec_norm = u_vec_mean / torch.norm(u_vec_mean, dim=-1, keepdim=True)
        u_vec_norm = u_vec_norm.permute(2,0,1)

        self.register_buffer("u_vec_mean", u_vec_norm, persistent=False)
        self.register_buffer("u_vec_x", u_vec_x, persistent=False)
        self.register_buffer("u_vec_y", u_vec_y, persistent=False)
        self.register_buffer("u_vec_z", u_vec_z, persistent=False)
        self.register_buffer("u_vec", u_vec, persistent=False)
    
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

        
    def forward(self, x_rv, lidar2ego_mat):
        B, Crv, Hrv, Wrv = x_rv.shape
        assert Hrv == self.in_Hrv and Wrv == self.in_Wrv, f"Input range view size {Hrv}x{Wrv} does not match expected size {self.in_Hrv}x{self.in_Wrv}"
        
        # Expand features for each query unit vector
        # x_rv: [B, Crv, Hrv, Wrv]
        # u_vec: [Hrv, Wrv, num_q_per_latent_cell, 3]
        
        # Expand x_rv to [B, Crv, Hrv, Wrv, num_q_per_latent_cell]
        x_rv_expanded = x_rv.unsqueeze(-1).expand(-1, -1, -1, -1, self.num_q_per_latent_cell)
        
        # Get unit vectors for each pixel: [Hrv, Wrv, num_q_per_latent_cell, 3]
        u_vec_per_pixel = self.u_vec  # [Hrv, Wrv, num_q_per_latent_cell, 3]
        
        # Expand unit vectors for batch: [B, Hrv, Wrv, num_q_per_latent_cell, 3]
        batch_u_vec = u_vec_per_pixel.unsqueeze(0).expand(B, -1, -1, -1, -1)
        
        # Create expanded input by concatenating features with unit vectors
        # We need to concatenate along the channel dimension
        # x_rv_expanded: [B, Crv, Hrv, Wrv, num_q_per_latent_cell]
        # batch_u_vec: [B, Hrv, Wrv, num_q_per_latent_cell, 3]
        
        # Permute x_rv_expanded to [B, Hrv, Wrv, num_q_per_latent_cell, Crv]
        x_rv_for_concat = x_rv_expanded.permute(0, 2, 3, 4, 1)
        
        # Concatenate features with unit vectors: [B, Hrv, Wrv, num_q_per_latent_cell, Crv+3]
        range_in_expanded = torch.cat([x_rv_for_concat, batch_u_vec], dim=-1)
        
        # Reshape for processing: [B*Hrv*Wrv*num_q_per_latent_cell, Crv+3]
        range_in_flat = range_in_expanded.view(-1, Crv + 3)
        
        # Apply range head to each expanded feature
        depth_logits_flat = self.range_head_1d(range_in_flat)
        
        # Reshape back: [B, Hrv, Wrv, num_q_per_latent_cell]
        depth_logits = depth_logits_flat.view(B, Hrv, Wrv, self.num_q_per_latent_cell)
        depth_activation = depth_logits.sigmoid()

        # Project expected depth `mu_d` into 3D space
        mu_d = depth_activation * self.rmax  # [B, Hrv, Wrv, num_q_per_latent_cell]
        
        # Expand unit vectors for batch operations
        u_vec_x_batch = self.u_vec_x.unsqueeze(0).expand(B, -1, -1, -1)  # [B, Hrv, Wrv, num_q_per_latent_cell]
        u_vec_y_batch = self.u_vec_y.unsqueeze(0).expand(B, -1, -1, -1)  # [B, Hrv, Wrv, num_q_per_latent_cell]
        u_vec_z_batch = self.u_vec_z.unsqueeze(0).expand(B, -1, -1, -1)  # [B, Hrv, Wrv, num_q_per_latent_cell]
        
        x_l = mu_d * u_vec_x_batch
        y_l = mu_d * u_vec_y_batch
        z_l = mu_d * u_vec_z_batch
        
        ones = torch.ones_like(x_l)
        # Stack coordinates: [B, Hrv, Wrv, num_q_per_latent_cell, 4]
        p_lidar_h = torch.stack([x_l, y_l, z_l, ones], dim=-1)
        p_lidar_h = p_lidar_h.view(B, -1, 4)

        # LiDAR -> Ego
        p_ego_h = p_lidar_h @ lidar2ego_mat.T
        p_ego_h = p_ego_h.view(B, Hrv, Wrv, self.num_q_per_latent_cell, 4)
        
        return p_ego_h, depth_activation


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
    
    qg = SimpleQueryGenerator(rmax=51.2, C_rv=384, nqw=4, nqh=4, in_rv_size=(2,64), og_rv_size=(32,1024))
    
    assert qg.az_per_q.shape == (2, 64, 4)
    assert qg.elev_per_q.shape == (2, 64, 4)
    assert qg.u_vec_mean.shape == (3,2, 64)
    
    del qg
    
    qg = SimpleQueryGenerator(rmax=51.2, C_rv=384, nqw=1, nqh=1, in_rv_size=(32,1024), og_rv_size=(32,1024))
    
    assert qg.az_per_q.shape == (32, 1024, 1)
    assert qg.elev_per_q.shape == (32, 1024, 1)
    assert qg.u_vec_mean.shape == (3,32, 1024)
    
    del qg
    
    qg = SimpleQueryGenerator(rmax=51.2, C_rv=384, nqw=2, nqh=4, in_rv_size=(16,512), og_rv_size=(32,1024))
    
    assert qg.az_per_q.shape == (16, 512, 2)
    assert qg.elev_per_q.shape == (16, 512, 2)
    assert qg.u_vec_mean.shape == (3,16, 512)
    
    print("All tests passed!")