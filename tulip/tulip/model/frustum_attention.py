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
    def __init__(self, C_rv, C_bev, C_out=128, d=128,
                 rmax=55.0, K=24, c=2.0,
                 grid_m=0.5, bev_extent=(-55,55,-55,55),
                 gumbel_topk=False, topk=8, lambda_ent=0.0,
                 n_heads=8,    # <-- add: MSDA heads
                 msda_points=6, # <-- num_points per head in MSDA (keep small)
                 rv_size=(2,64),
                 vfov=((-30.67,10.67))
                 ):
        super().__init__()
        self.d = d
        self.K = K
        self.c = float(c)
        self.rmax = float(rmax)
        self.grid_m = float(grid_m)
        self.xmin, self.xmax, self.ymin, self.ymax = bev_extent
        self.gumbel_topk = gumbel_topk
        self.topk = topk
        self.lambda_ent = lambda_ent
        self.n_heads = n_heads
        self.msda_points = msda_points

        # Projections
        self.proj_q = nn.Conv2d(C_rv, d, 1, bias=True)
        self.proj_v = nn.Conv2d(C_bev, d, 1, bias=True)
        self.proj_o = nn.Conv2d(d, C_out, 1, bias=True)

        # σ-aware query lift: concat [logσ, 1/σ] to Q, bring back to d
        self.q_sigma = nn.Sequential(
            nn.Conv2d(d + 2, d, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(d, d, 1, bias=True)
        )

        # Light range proposal head 
        self.range_head = nn.Sequential(
            nn.Conv2d(C_rv + 3, 128, 1, bias=True),
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1, padding_mode="circular", bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 2, 1, bias=True) # sigma, mu for depth distribution
        )

        # Build azimuth and zenith maps by rv_size (width spans [-pi, pi))
        self.Hrv, self.Wrv = int(rv_size[0]), int(rv_size[1])
        az_line = torch.linspace(-math.pi, math.pi, self.Wrv + 1)[:-1]   # [Wrv]
        az = az_line.view(1, 1, 1, self.Wrv).expand(1, 1, self.Hrv, self.Wrv)
        elev = np.array(np.split(ELEV_DEG_PER_RING_NUCSENES[::-1], self.Hrv))
        elev = elev.mean(axis=1).repeat(self.Wrv).reshape(self.Hrv, self.Wrv)
        elev = torch.from_numpy(np.deg2rad(elev)).float()
        
        self.register_buffer("elevation", elev, persistent=False)
        self.register_buffer("azimuth", az, persistent=False)

        # compute unit vector per range view pixel 
        cos_az = torch.cos(az)
        sin_az = torch.sin(az)
        cos_el = torch.cos(elev)
        sin_el = torch.sin(elev)
        u_vec_x = cos_az * cos_el
        u_vec_y = sin_az * cos_el
        u_vec_z = sin_el[None, None, :, :]
        u_vec = torch.cat([u_vec_x, u_vec_y, u_vec_z], dim=1)
        self.register_buffer("u_vec", u_vec, persistent=False)
        self.register_buffer("u_vec_x", u_vec_x, persistent=False)
        self.register_buffer("u_vec_y", u_vec_y, persistent=False)
        self.register_buffer("u_vec_z", u_vec_z, persistent=False)
        
        self.jitter_scale = 0.05   # γ: 5% of sigma
        self.edge_gain    = 4.0    # smooth “clamp” strength for ref coords
        self.kl_weight    = 1e-4   # tiny KL prior
        

        # ---- NEW: MSDeformAttn (1 level, 2D) ----
        self.msda = MSDA(embed_dims=d,
                         num_heads=n_heads,
                         num_levels=1,
                         num_points=msda_points,
                         batch_first=True)

    def forward(self, x_rv, bev, lidar2ego_mat, temperature=1.0):
        """
            x_rv (_type_): [B, Hrv, Wrv, Crv]
            bev (_type_): [B, Cb, Hbev, Wbev]
            lidar2ego_mat (_type_): [1, 4, 4]
        """
        B, Hrv, Wrv, Crv = x_rv.shape
        B2, Cb, Hbev, Wbev = bev.shape
        if B < B2: # handle this inside bev feature extractor bevdepth/base_lss_fpn.py, it happens because of cacheing the batched geometries
            bev = bev[:B, ...]
            B2, Cb, Hbev, Wbev = bev.shape
        assert B == B2
        x_rv = x_rv.permute(0, 3, 1, 2).contiguous()
        # Azimuth & Coord
        batch_az = self.azimuth.expand(B, -1, -1, -1)
        batch_u_vec = self.u_vec.expand(B, -1, -1, -1)

        # Projections
        Q0   = self.proj_q(x_rv)         # [B,d,Hrv,Wrv]
        Vmap = self.proj_v(bev)          # [B,d,Hbev,Wbev]

        # Range head -> μ, σ
        range_in = torch.cat([x_rv, batch_u_vec], 1)
        mu_raw, sigma_raw = torch.chunk(self.range_head(range_in), 2, dim=1)
        mu = torch.sigmoid(mu_raw) * self.rmax
        sigma = F.relu(sigma_raw).clamp_min(1e-3) * self.rmax                        # [B,1,Hrv,Wrv]
        # mu    = mu.squeeze(1).unsqueeze(1).clamp(0.0, self.rmax)  # [B,1,Hrv,Wrv]

        # ---- Build reference points from μ, az ----
        # μ along ray to (x,y,z) in meters (real world position)
        # Build reference points from μ, az, elevation
        az_ = batch_az.squeeze(1)  # [B,Hrv,Wrv], LiDAR-frame azimuth (radians)
        el_ = self.elevation.unsqueeze(0).expand(B, -1, -1)  # [B,Hrv,Wrv], LiDAR-frame elevation

        mu_used = mu.squeeze(1)  # [B,Hrv,Wrv]
        # sg_ = sigma.squeeze(1)

        # mu_used = mu_
        # beam endpoint in LiDAR frame
        x_l = mu_used * self.u_vec_x #torch.cos(el_) * torch.cos(az_)
        y_l = mu_used * self.u_vec_y #torch.cos(el_) * torch.sin(az_)
        z_l = mu_used * self.u_vec_z #torch.sin(el_)

        ones = torch.ones_like(x_l)
        p_lidar_h = torch.stack([x_l, y_l, z_l, ones], dim=-1)   # [B,Hrv,Wrv,4]
        p_lidar_h = p_lidar_h.view(B, Hrv*Wrv, 4)                # [B,L,4]

        # LiDAR -> Ego
        # lidar2ego_mat is [B,4,4] (or [1,4,4] broadcastable)
        # print dtypes
        p_ego_h = p_lidar_h @ lidar2ego_mat                      # [B,L,4]
        p_ego_h = p_ego_h.view(B, Hrv, Wrv, 4) # xyzw per latent beam

        # splat onto bev plane
        x_ego = p_ego_h[..., 0]
        y_ego = p_ego_h[..., 1]

        # Normalize to [0,1]
        rx = (x_ego - self.xmin) / (self.xmax - self.xmin)   # [B,Hrv,Wrv]
        ry = (y_ego - self.ymin) / (self.ymax - self.ymin)   # [B,Hrv,Wrv]
        rx = torch.sigmoid(self.edge_gain * (rx - 0.5))
        ry = torch.sigmoid(self.edge_gain * (ry - 0.5))
        ref = torch.stack([rx, ry], dim=-1)          # [B,Hrv,Wrv,2]
        ref = ref.view(B, Hrv*Wrv, 1, 2)                                   # [B,L,1,2], 1 level

        # ---- σ-aware query augmentation ----
        sig_feat = torch.cat([sigma, (1.0 / (sigma + 1e-6))], dim=1)   # [B,2,Hrv,Wrv]
        Q = self.q_sigma(torch.cat([Q0, sig_feat], dim=1))                 # [B,d,Hrv,Wrv]

        # flatten for MSDA
        query = Q.permute(0,2,3,1).reshape(B, Hrv*Wrv, self.d)         
        value = Vmap.flatten(2).transpose(1,2).contiguous()   
        original_dtype = query.dtype
        value = value.to(torch.float32) # [B,Hbev*Wbev,d]
        query = query.to(torch.float32)
        spatial_shapes = torch.as_tensor([[Hbev, Wbev]], device=bev.device, dtype=torch.long)
        level_start_index = torch.as_tensor([0], device=bev.device, dtype=torch.long)

        # without disabling autocasts, amp casts tensors to half which msda does not support
        with torch.cuda.amp.autocast(enabled=False):
            y_msda = self.msda(query=query,
                                reference_points=ref,
                                value=value,
                                spatial_shapes=spatial_shapes,
                                level_start_index=level_start_index,
                                key_padding_mask=None)
        y_msda = y_msda.to(original_dtype)
        y = y_msda.view(B, Hrv, Wrv, self.d).permute(0,3,1,2).contiguous()  # [B,d,Hrv,Wrv]
        y = self.proj_o(y)                                                  # [B,C_out,Hrv,Wrv]

        # KL to N(0,1) on normalized μ/rmax and σ
        # mu_norm = mu_ / self.rmax
        # L_kl = 0.5 * (mu_norm**2 + sg_**2 - (sg_**2 + 1e-8).log() - 1.0)
        aux = None
        # aux["L_kl_mu_sigma"] = self.kl_weight * L_kl.mean()
        # aux["mu_mean"] = mu_.mean().detach()
        # aux["sigma_mean"] = sg_.mean().detach()
        y = y.view(B, Hrv, Wrv, -1)

        return y, aux, (mu/self.rmax, sigma)

__all__ = ["AngleBinner3D"]


