from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention as MSDA
import pdb

ELEV_DEG_PER_RING_NUCSENES = np.array([-30.67, -29.33, -28., -26.66, -25.33, -24., -22.67, -21.33,
                               -20., -18.67, -17.33, -16., -14.67, -13.33, -12., -10.67,
                                -9.33, -8., -6.66, -5.33, -4., -2.67, -1.33, 0.,
                                1.33, 2.67, 4., 5.33, 6.67, 8., 9.33, 10.67], dtype=np.float32)

class RV2BEVFrustumAttn(nn.Module):
    def __init__(self, C_rv, C_bev, d=128,
                 rmax=51.2, # K is removed, we use the full distribution
                 bin_size=0.8, bev_extent=(-51.2,51.2,-51.2,51.2),
                 n_heads=8,
                 msda_points=6,
                 rv_size=(2,64),
                 vfov=((-30.67,10.67)),
                 x_bound=(51.2,51.2),
                 y_bound=(51.2,51.2),
                 z_bound=(51.2,51.2),
                 og_rv_size=(32,1024)
                 ):
        super().__init__()
        self.d = d
        self.rmax = float(rmax)
        self.bin_size = float(bin_size)
        self.xmin, self.xmax, self.ymin, self.ymax = bev_extent
        self.n_heads = n_heads
        self.msda_points = msda_points

        # Projections
        self.proj_q = nn.Conv2d(C_rv, d, 1, bias=True)
        self.proj_v = nn.Conv2d(C_bev, d, 1, bias=True)

        # Light range proposal head
        self.n_q_w = 5
        self.n_q_h = 4
        self.num_q_per_latent_cell = self.n_q_w * self.n_q_h
        
        self.proj_o =  nn.Sequential(
            nn.Conv2d(self.d*self.num_q_per_latent_cell, C_rv*2, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(C_rv*2, C_rv, 1, bias=True)
        )
        
        self.add_norm = nn.LayerNorm(C_rv)
        self.ffn_norm = nn.LayerNorm(C_rv)
        self.ffn = nn.Sequential(
            nn.Linear(C_rv, C_rv),
            nn.GELU(),
            nn.Linear(C_rv, C_rv)
        )
        
        self.n_bins = int(self.rmax // self.bin_size)
        self.range_head = nn.Sequential(
            nn.Conv2d(C_rv + 3, 128, 1, bias=True),
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1, padding_mode="circular", bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, self.num_q_per_latent_cell, 1, bias=True) # depth distribution logits
        )
        # Build azimuth and zenith maps by rv_size (width spans [-pi, pi))
        self.og_Hrv, self.og_Wrv = int(og_rv_size[0]), int(og_rv_size[1])
        self.Hrv, self.Wrv = int(rv_size[0]), int(rv_size[1])
        self.ds_factor_h = self.og_Hrv // self.Hrv
        self.ds_factor_w = self.og_Wrv // self.Wrv
        
        az_line = torch.linspace(-math.pi, math.pi, self.Wrv + 1)[:-1]   # [Wrv]
        az_step = az_line[1] - az_line[0]
        az_offset_template = torch.linspace(-0.5, 0.5, self.n_q_w + 1)[:-1] + 0.5 / self.n_q_w
        az_line += 0.5*az_step # bin centers
        az = az_line[..., None] + az_offset_template*az_step
        az = az.repeat(2, 1, 1)

        elev = np.array(np.split(ELEV_DEG_PER_RING_NUCSENES[::-1], self.Hrv))
        inds = np.arange(self.n_q_h) * (self.og_Hrv/rv_size[0]/self.n_q_h)
        elev = elev[:, inds.astype(int)]
        elev = elev[:,None,:].repeat(64,1)
        elev = torch.from_numpy(np.deg2rad(elev)).float()

        # compute unit vector per range view pixel 
        cos_az = torch.cos(az).unsqueeze(-1)
        sin_az = torch.sin(az).unsqueeze(-1)
        cos_el = torch.cos(elev).unsqueeze(-2)
        sin_el = torch.sin(elev)
        u_vec_x_grid = cos_az * cos_el
        u_vec_y_grid = sin_az * cos_el
        u_vec_z_grid = sin_el.unsqueeze(2).expand_as(u_vec_x_grid)

        num_points = self.n_q_w * self.n_q_h
        u_vec_x = u_vec_x_grid.reshape(self.Hrv, self.Wrv, num_points)
        u_vec_y = u_vec_y_grid.reshape(self.Hrv, self.Wrv, num_points)
        u_vec_z = u_vec_z_grid.reshape(self.Hrv, self.Wrv, num_points)
        u_vec = torch.stack([u_vec_x, u_vec_y, u_vec_z], dim=-1)
        
        u_vec_mean = u_vec.mean(-2)
        u_vec_norm = u_vec_mean / torch.norm(u_vec_mean, dim=-1, keepdim=True)
        u_vec_norm = u_vec_norm.permute(2,0,1)

        self.register_buffer("u_vec", u_vec_norm, persistent=False)
        self.register_buffer("u_vec_x", u_vec_x, persistent=False)
        self.register_buffer("u_vec_y", u_vec_y, persistent=False)
        self.register_buffer("u_vec_z", u_vec_z, persistent=False)
        self.kl_weight    = 1e-4   # tiny KL prior

        #  deformAttn (1 level, 2D)
        self.msda = MSDA(embed_dims=d,
                         num_heads=n_heads,
                         num_levels=1,
                         num_points=6,
                         batch_first=True)

        pe = self.build_sinusoidal_bev_pe(H=128, W=128, C=d, device='cpu')
        self.register_buffer("bev_pe", pe, persistent=False)

        bin_centers = torch.arange(0.5, self.n_bins + 0.5, 1.0, dtype=torch.float32) * self.bin_size
        self.register_buffer("bin_centers", bin_centers.view(1, -1, 1, 1), persistent=False)

    def build_sinusoidal_bev_pe(self, H, W, C, device):
        assert C % 2 == 0, "Embedding dim must be even for sine/cosine PE."
        pe = torch.zeros(1, C, H, W, device=device)
        y_pos = torch.arange(0, H, dtype=torch.float, device=device).unsqueeze(1)
        x_pos = torch.arange(0, W, dtype=torch.float, device=device).unsqueeze(0)

        div_term = torch.exp(torch.arange(0, C // 2, 2, device=device).float() * (-math.log(10000.0) / (C // 2)))
        pe[0, 0:C//4*2:2, :, :] = torch.sin(x_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
        pe[0, 1:C//4*2:2, :, :] = torch.cos(x_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
        pe[0, C//2::2, :, :] = torch.sin(y_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
        pe[0, C//2+1::2, :, :] = torch.cos(y_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
        return pe

    def forward(self, x_rv, bev, lidar2ego_mat, temperature=1.0):
        B, Hrv, Wrv, Crv = x_rv.shape
        B2, Cb, Hbev, Wbev = bev.shape
        if B < B2:
            bev = bev[:B, ...]
            B2, Cb, Hbev, Wbev = bev.shape
        assert B == B2
        x_rv = x_rv.permute(0, 3, 1, 2).contiguous()
        batch_u_vec = self.u_vec.expand(B, -1, -1, -1)

        # Projections
        Q0   = self.proj_q(x_rv).permute(0,2,3,1)
        Vmap = self.proj_v(bev)
        Vmap = Vmap + self.bev_pe

        # Range head -> distribution
        range_in = torch.cat([x_rv, batch_u_vec], 1)
        depth_logits = self.range_head(range_in)
        depth_activation = depth_logits.sigmoid()

        mu_d = (depth_activation * self.rmax).permute(0,2,3,1)

        # Project expected depth `mu_d` into 3D space
        x_l = mu_d * self.u_vec_x
        y_l = mu_d * self.u_vec_y
        z_l = mu_d * self.u_vec_z

        ones = torch.ones_like(x_l)
        # Note: Shape is now [B, 1, Hrv, Wrv], we squeeze the singular dim 1 for stacking
        p_lidar_h = torch.stack([x_l.squeeze(1), y_l.squeeze(1), z_l.squeeze(1), ones.squeeze(1)], dim=-1)
        p_lidar_h = p_lidar_h.view(B, -1, 4)

        # LiDAR -> Ego
        p_ego_h = p_lidar_h @ lidar2ego_mat
        p_ego_h = p_ego_h.view(B, Hrv, Wrv, self.num_q_per_latent_cell, 4)

        # Splat onto BEV plane and normalize to get reference points
        x_ego = p_ego_h[..., 0]
        y_ego = p_ego_h[..., 1]
        rx = (x_ego - self.xmin) / (self.xmax - self.xmin)
        ry = (y_ego - self.ymin) / (self.ymax - self.ymin)
        # Note: Shape is now [B, Hrv*Wrv, 1, 2]
        # Stack coordinates into the last dimension
        ref = torch.stack([rx.clamp(0, 1), ry.clamp(0, 1)], dim=-1)
        # Reshape to [B, num_queries, num_points, 2]
        ref = ref.view(B, Hrv * Wrv *self.num_q_per_latent_cell, 2)
        
        # Flatten query and value for MSDA
        query = Q0.unsqueeze(3).expand(B, Hrv, Wrv, self.num_q_per_latent_cell, self.d)          # [B, Hrv, Wrv, P, d]
        query = query.reshape(B, -1, self.d).contiguous()                 # [B, Len_q_pts, d]

        # 2) Flatten ref points to align with query, keep num_levels=1 and num_points=1 for MSDA input
        #    MMCV will broadcast its internal num_points over this singleton
        reference_points = ref.reshape(B, -1, 2)                     # [B, Len_q_pts, 2]
        reference_points = reference_points.unsqueeze(2)       # [B, Len_q_pts, 1, 1, 2]


        value = Vmap.flatten(2).transpose(1,2).contiguous() # [B, Hbev*Wbev, d]

        original_dtype = query.dtype
        value = value.to(torch.float32)
        query = query.to(torch.float32)
        spatial_shapes = torch.as_tensor([[Hbev, Wbev]], device=bev.device, dtype=torch.long)
        level_start_index = torch.as_tensor([0], device=bev.device, dtype=torch.long)

        with torch.cuda.amp.autocast(enabled=False):
            y_msda = self.msda(query=query,
                                reference_points=reference_points,
                                value=value,
                                spatial_shapes=spatial_shapes,
                                level_start_index=level_start_index,
                                key_padding_mask=None)
        # Reshape output back to image format
        y_msda = y_msda.transpose(1, 2).contiguous().view(B, -1, Hrv, Wrv)
        y_msda = self.proj_o(y_msda) # to reshape to C_rv
        y = y_msda + x_rv
        y = self.add_norm(y.permute(0,2,3,1))
        y_ffn = self.ffn(y)
        y = self.ffn_norm(y_ffn + y)
        
        y = y.contiguous()

        # Return the KL loss to be added to the main training objective
        return y, depth_logits, torch.tensor(0.0)
