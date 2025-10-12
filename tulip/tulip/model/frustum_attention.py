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
                 rmax=55.0, K=5,
                 bin_size=0.5, bev_extent=(-55,55,-55,55),
                 n_heads=8,    # <-- add: MSDA heads
                 msda_points=6, # <-- num_points per head in MSDA (keep small)
                 rv_size=(2,64),
                 vfov=((-30.67,10.67))
                 ):
        super().__init__()
        self.d = d
        self.K = K
        self.rmax = float(rmax)
        self.bin_size = float(bin_size)
        self.xmin, self.xmax, self.ymin, self.ymax = bev_extent
        self.n_heads = n_heads
        self.msda_points = msda_points

        # Projections
        self.proj_q = nn.Conv2d(C_rv, d, 1, bias=True)
        self.proj_v = nn.Conv2d(C_bev, d, 1, bias=True)
        self.proj_o = nn.Conv2d(d, C_out, 1, bias=True)

        # depth-aware query lift: concat [logσ, 1/σ] to Q, bring back to d
        self.q_depth = nn.Sequential(
            nn.Conv2d(d + 1, d, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(d, d, 1, bias=True)
        )

        # Light range proposal head 
        self.n_bins = self.rmax // self.bin_size
        self.range_head = nn.Sequential(
            nn.Conv2d(C_rv + 3, 128, 1, bias=True),
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1, padding_mode="circular", bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, int(self.n_bins), 1, bias=True) # depth distribution logits
        )
        # Build azimuth and zenith maps by rv_size (width spans [-pi, pi))
        self.Hrv, self.Wrv = int(rv_size[0]), int(rv_size[1])
        az_line = torch.linspace(-math.pi, math.pi, self.Wrv + 1)[:-1]   # [Wrv]
        az = az_line.view(1, 1, 1, self.Wrv).expand(1, 1, self.Hrv, self.Wrv)
        az_step = az_line[1] - az_line[0]
        az = az + 0.5 * az_step   # center of each bin
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
        
        self.edge_gain    = 4.0    # smooth “clamp” strength for ref coords
        self.kl_weight    = 1e-4   # tiny KL prior
        

        #  deformAttn (1 level, 2D)
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
        batch_u_vec = self.u_vec.expand(B, -1, -1, -1)

        # Projections
        Q0   = self.proj_q(x_rv)         # [B,d,Hrv,Wrv]
        Vmap = self.proj_v(bev)          # [B,d,Hbev,Wbev]

        # Range head -> μ, σ
        range_in = torch.cat([x_rv, batch_u_vec], 1)
        depth_logits = self.range_head(range_in)
        depth_dist = F.softmax(depth_logits, dim=1)
        topk_prob, topk_idx = torch.topk(depth_dist, k=self.K, dim=1)   # [B,K,Hrv,Wrv]
        # Gather bin centers
        # bin_centers: [1,n_bins,1,1] -> expand along H,W
        topk_depths = self.bin_size * (topk_idx.to(depth_logits.dtype) + 0.5)
        topk_depths = topk_depths.clamp_max(self.rmax - 0.5 * self.bin_size)

        # beam endpoint in LiDAR frame
        x_l = topk_depths * self.u_vec_x #torch.cos(el_) * torch.cos(az_)
        y_l = topk_depths * self.u_vec_y #torch.cos(el_) * torch.sin(az_)
        z_l = topk_depths * self.u_vec_z #torch.sin(el_)

        ones = torch.ones_like(x_l)
        p_lidar_h = torch.stack([x_l, y_l, z_l, ones], dim=-1)   # [B,Hrv,Wrv,4]
        p_lidar_h = p_lidar_h.view(B, self.K*Hrv*Wrv, 4)                # [B,L,4]

        # LiDAR -> Ego
        # lidar2ego_mat is [B,4,4] (or [1,4,4] broadcastable)
        p_ego_h = p_lidar_h @ lidar2ego_mat                      # [B,L,4]
        p_ego_h = p_ego_h.view(B, self.K, Hrv, Wrv, 4) # xyzw per latent beam

        # splat onto bev plane
        x_ego = p_ego_h[..., 0]
        y_ego = p_ego_h[..., 1]

        # Normalize to [0,1] linearly
        rx = (x_ego - self.xmin) / (self.xmax - self.xmin)   # [B,Hrv,Wrv]
        ry = (y_ego - self.ymin) / (self.ymax - self.ymin)   # [B,Hrv,Wrv]
        valid_ref = (rx >= 0) & (rx <= 1) & (ry >= 0) & (ry <= 1)
        ref = torch.stack([rx.clamp(0,1), ry.clamp(0,1)], dim=-1).view(B, self.K*Hrv*Wrv, 1, 2)

        # ---- σ-aware query augmentation ----
        Q_tiled = Q0.unsqueeze(1).repeat(1, self.K, 1, 1, 1)        # [B,K,d,H,W]
        depth_norm = (topk_depths / self.rmax).unsqueeze(2)         # [B,K,1,H,W]
        Q_depth = torch.cat([Q_tiled, depth_norm], dim=2)           # [B,K,d+1,H,W]
        Q_depth = Q_depth.view(B*self.K, self.d+1, Hrv, Wrv)
        Q_depth = self.q_depth(Q_depth)               

        # Flatten to (B, L*K, d)
        query = Q_depth.permute(0,2,3,1).reshape(B, self.K*Hrv*Wrv, self.d)  # [B,L*K,d]
        value = Vmap.flatten(2).transpose(1,2).contiguous()    
        # flatten for MSDA
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
        y_msda = y_msda.view(B, self.K, Hrv, Wrv, self.d) 
        w = topk_prob / (topk_prob.sum(dim=1, keepdim=True) + 1e-8)  # [B,K,H,W]
        y = (y_msda * w.unsqueeze(-1)).sum(dim=1)               # [B,H,W,d]
        y = y.permute(0,3,1,2).contiguous()     
        y = self.proj_o(y)                                                  # [B,C_out,Hrv,Wrv]
        y = y.permute(0,2,3,1).contiguous()

        return y, depth_logits, None

