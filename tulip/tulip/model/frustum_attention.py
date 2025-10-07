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
                 rmax=55.0, K=5, c=2.0,
                 grid_m=0.5, bev_extent=(-55,55,-55,55),
                 gumbel_topk=False, topk=8, lambda_ent=0.0,
                 n_heads=8,    # <-- add: MSDA heads
                 msda_points=4, # <-- num_points per head in MSDA (keep small)
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
            nn.Conv2d(d + 6, d, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(d, d, 1, bias=True)
        )

        # Light range proposal head 
        self.range_head = nn.Sequential(
            nn.Conv2d(C_rv + 3, 128, 1, bias=True),
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1, padding_mode="circular", bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 4, 1, bias=True) # mu_depth, sigma_depth, sigma_elev, sigma_azimuth
        )

        # Build azimuth and zenith maps by rv_size (width spans [-pi, pi))
        self.Hrv, self.Wrv = int(rv_size[0]), int(rv_size[1])
        az_line = torch.linspace(-math.pi, math.pi, self.Wrv + 1)[:-1]   # [Wrv]
        self.az_step = az_line[1] - az_line[0]
        az_line = az_line + self.az_step / 2
        az = az_line.view(1, 1, 1, self.Wrv).expand(1, 1, self.Hrv, self.Wrv)
        # elev angles are already bin centers so no need to add half step
        elev = np.array(np.split(ELEV_DEG_PER_RING_NUCSENES[::-1], self.Hrv))
        el_steps = elev.max(axis=1) - elev.min(axis=1)
        el_steps = torch.from_numpy(np.deg2rad(el_steps)).float()
        self.register_buffer("elev_steps", el_steps.repeat(self.Wrv).reshape(self.Hrv, self.Wrv), persistent=False)
        elev = elev.mean(axis=1).repeat(self.Wrv).reshape(self.Hrv, self.Wrv)
        elev = torch.from_numpy(np.deg2rad(elev)).float()
        
        self.register_buffer("elevation", elev, persistent=False)
        self.register_buffer("azimuth", az, persistent=False)

        # compute unit vector per range view pixel 
        self.register_buffer("offsets", torch.tensor(np.linspace(-0.6, 0.6, self.K), dtype=torch.float32), persistent=False)
        
        self.jitter_scale = 0.05   # γ: 5% of sigma
        self.edge_gain    = 4.0    # smooth “clamp” strength for ref coords
        self.kl_weight    = 1e-4   # tiny KL prior
        

        self.msda = MSDA(embed_dims=d,
                         num_heads=n_heads,
                         num_levels=self.K,
                         num_points=msda_points,
                         batch_first=True)

    def forward(self, x_rv, bev, lidar2ego_mat):
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
        mu, sigma_depth, sigma_elev, sigma_azimuth = torch.chunk(self.range_head(range_in), 4, dim=1)
        mu = torch.sigmoid(mu) * self.rmax
        sigma_depth = F.softplus(sigma_depth) + 1e-3  # [B,1,Hrv,Wrv]
        sigma_elev = torch.sigmoid(sigma_elev) * self.elev_steps + 1e-3  # [B,1,Hrv,Wrv]
        sigma_azimuth = torch.sigmoid(sigma_azimuth) * self.az_step + 1e-3  # [B,1,Hrv,Wrv]
        # mu    = mu.squeeze(1).unsqueeze(1).clamp(0.0, self.rmax)  # [B,1,Hrv,Wrv]

        # ---- Build reference points from μ, az ----
        # μ along ray to (x,y,z) in meters (real world position)
        # Build reference points from μ, az, elevation
        az_ = batch_az.squeeze(1)  # [B,Hrv,Wrv], LiDAR-frame azimuth (radians)
        el_ = self.elevation.unsqueeze(0).expand(B, -1, -1)  # [B,Hrv,Wrv], LiDAR-frame elevation

        mu_ = mu.squeeze(1)  # [B,Hrv,Wrv]
        sg_depth = sigma_depth.squeeze(1)
        sg_elev = sigma_elev.squeeze(1)
        sg_azimuth = sigma_azimuth.squeeze(1)
        
        depth_samples = mu_[...,None] + (sg_depth[...,None] * self.offsets) # [B, Hrv, Wrv, K]
        elev_samples = el_[...,None] + (sg_elev[...,None] * self.offsets) # [B, Hrv, Wrv, K]
        azimuth_samples = az_[...,None] + (sg_azimuth[...,None] * self.offsets) # [B, Hrv, Wrv, K]

        # beam endpoint in LiDAR frame
        x_l = depth_samples * torch.cos(elev_samples) * torch.cos(azimuth_samples) #[B, Hrv, Wrv, K]
        y_l = depth_samples * torch.cos(elev_samples) * torch.sin(azimuth_samples)
        z_l = depth_samples * torch.sin(elev_samples)

        ones = torch.ones_like(x_l)
        p_lidar_h = torch.stack([x_l.reshape(B, Hrv*Wrv*self.K), 
                                 y_l.reshape(B, Hrv*Wrv*self.K), 
                                 z_l.reshape(B, Hrv*Wrv*self.K), 
                                 ones.reshape(B, Hrv*Wrv*self.K)], dim=-1)   # [B,Hrv*Wrv*self.K,4]

        # LiDAR -> Ego
        # lidar2ego_mat is [B,4,4] (or [1,4,4] broadcastable)
        # print dtypes
        p_ego_h = p_lidar_h @ lidar2ego_mat                      # [B,L,4]
        p_ego_h = p_ego_h.view(B, Hrv, Wrv, self.K, 4) # xyzw per latent beam

        # splat onto bev plane
        x_ego = p_ego_h[..., 0]
        y_ego = p_ego_h[..., 1]

        # Normalize to [0,1]
        rx = (x_ego - self.xmin) / (self.xmax - self.xmin)   # [B,Hrv,Wrv,K]
        ry = (y_ego - self.ymin) / (self.ymax - self.ymin)   # [B,Hrv,Wrv,K]
        rx = torch.sigmoid(self.edge_gain * (rx - 0.5))
        ry = torch.sigmoid(self.edge_gain * (ry - 0.5))
        ref = torch.stack([rx, ry], dim=-1) # [B,Hrv,Wrv,k,2]
        ref = ref.view(B, Hrv*Wrv, self.K, 2)     # [B,L,K,2], K level

        # ---- σ-aware query augmentation ----
        sig_feat = torch.cat([sigma_depth, sigma_elev, sigma_azimuth, 
                              (1.0 / (sigma_depth + 1e-6)),
                              (1.0 / (sigma_elev + 1e-6)),
                              (1.0 / (sigma_azimuth + 1e-6))], dim=1)   # [B,6,Hrv,Wrv]
        Q = self.q_sigma(torch.cat([Q0, sig_feat], dim=1))                 # [B,d,Hrv,Wrv]

        # flatten for MSDA
        query = Q.permute(0,2,3,1).reshape(B, Hrv*Wrv, self.d)         
        value = Vmap.flatten(2).transpose(1,2).contiguous()   
        value = value.repeat(1, self.K, 1)
        original_dtype = query.dtype
        value = value.to(torch.float32) # [B,Hbev*Wbev,d]
        query = query.to(torch.float32)
        spatial_shapes = torch.as_tensor([[Hbev, Wbev]] * self.K, device=bev.device, dtype=torch.long)
        level_start_index = torch.as_tensor(
                                            [i * Hbev * Wbev for i in range(self.K)], device=bev.device, dtype=torch.long
                                        )

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

        mu_norm = mu / self.rmax
        sigma_r = sigma_depth.clamp_min(1e-6)
        sigma_az = sigma_azimuth / self.az_step      # make dimensionless
        sigma_el = sigma_elev / self.elev_steps      # make dimensionless

        L_kl_r  = 0.5 * (mu_norm.pow(2) + sigma_r.pow(2) - sigma_r.pow(2).log() - 1.0)
        L_kl_az = 0.5 * (sigma_az.pow(2) - sigma_az.pow(2).log() - 1.0)
        L_kl_el = 0.5 * (sigma_el.pow(2) - sigma_el.pow(2).log() - 1.0)

        L_kl = L_kl_r + L_kl_az + L_kl_el
        
        aux = {}
        aux["L_kl_mu_sigma"] = self.kl_weight * L_kl.mean()
        y = y.view(B, Hrv, Wrv, -1)

        return y, aux


