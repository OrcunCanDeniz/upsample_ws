from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as func


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention as MSDA
import pdb

class RV2BEVFrustumAttn(nn.Module):
    def __init__(self, C_rv, C_bev, C_out=128, d=128,
                 rmax=55.0, K=24, c=2.0,
                 grid_m=0.5, bev_extent=(-50,50,-50,50),
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

        # Projections (unchanged)
        self.proj_q = nn.Conv2d(C_rv, d, 1, bias=True)
        self.proj_v = nn.Conv2d(C_bev, d, 1, bias=True)
        self.proj_o = nn.Conv2d(d, C_out, 1, bias=True)

        # σ-aware query lift: concat [logσ, 1/σ] to Q, bring back to d
        self.q_sigma = nn.Sequential(
            nn.Conv2d(d + 2, d, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(d, d, 1, bias=True)
        )

        # Light range proposal head (unchanged)
        self.range_head = nn.Sequential(
            nn.Conv2d(C_rv + 3, 128, 1, bias=True),
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1, padding_mode="circular", bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 2, 1, bias=True)
        )

        # Fixed offsets (kept for aux/visualization)
        ti = torch.linspace(-self.c, self.c, steps=self.K)
        self.register_buffer("t_offsets", ti, persistent=False)

        # Build azimuth map by rv_size (width spans [-pi, pi))
        self.Hrv, self.Wrv = int(rv_size[0]), int(rv_size[1])
        az_line = torch.linspace(-math.pi, math.pi, self.Wrv + 1)[:-1]   # [Wrv]
        az = az_line.view(1, 1, 1, self.Wrv).expand(1, 1, self.Hrv, self.Wrv)
        self.register_buffer("azimuth", az, persistent=False)

        # ---- NEW: MSDeformAttn (1 level, 2D) ----
        self.msda = MSDA(embed_dims=d,
                         num_heads=n_heads,
                         num_levels=1,
                         num_points=msda_points,
                         batch_first=True)

    def forward(self, x_rv, bev, temperature=1.0):
        """_summary_

        Args:
            x_rv (_type_): [B, Hrv, Wrv, Crv]
            bev (_type_): [B, Cb, Hbev, Wbev]
        """
        
        B, Hrv, Wrv, Crv = x_rv.shape
        B2, Cb, Hbev, Wbev = bev.shape
        assert B == B2
        x_rv = x_rv.permute(0, 3, 1, 2).contiguous()
        # Azimuth & Coord
        az = self.azimuth.expand(B, -1, -1, -1)
        coord = torch.cat([torch.sin(az), torch.cos(az), torch.zeros_like(az)], dim=1)

        # Projections
        Q0   = self.proj_q(x_rv)         # [B,d,Hrv,Wrv]
        Vmap = self.proj_v(bev)          # [B,d,Hbev,Wbev]

        # Range head -> μ, σ
        mu, log_sigma = torch.chunk(self.range_head(torch.cat([x_rv, coord], 1)), 2, dim=1)
        log_sigma = log_sigma.clamp(min=-5.0, max=3.0)
        sigma = log_sigma.exp()                          # [B,1,Hrv,Wrv]
        mu    = mu.squeeze(1).unsqueeze(1).clamp(0.0, self.rmax)  # [B,1,Hrv,Wrv]

        # ---- Build reference points from μ, az ----
        # μ along ray to (x,y) in meters (real world position)
        x_mu = mu.squeeze(1) * torch.cos(az.squeeze(1))  # [B,Hrv,Wrv]
        y_mu = mu.squeeze(1) * torch.sin(az.squeeze(1))  # [B,Hrv,Wrv]
        # normalize to [0,1] 
        rx = (x_mu - self.xmin) / (self.xmax - self.xmin)
        ry = (y_mu - self.ymin) / (self.ymax - self.ymin)
        ref = torch.stack([rx.clamp(0,1), ry.clamp(0,1)], dim=-1)          # [B,Hrv,Wrv,2]
        ref = ref.view(B, Hrv*Wrv, 1, 2)                                   # [B,L,1,2], 1 level

        # ---- σ-aware query augmentation ----
        sig_feat = torch.cat([log_sigma, (1.0 / (sigma + 1e-6))], dim=1)   # [B,2,Hrv,Wrv]
        Q = self.q_sigma(torch.cat([Q0, sig_feat], dim=1))                 # [B,d,Hrv,Wrv]

        # flatten for MSDA
        query = Q.permute(0,2,3,1).reshape(B, Hrv*Wrv, self.d)         
        value = Vmap.flatten(2).transpose(1,2).contiguous()   
        original_dtype = query.dtype
        value = value.to(torch.float32) # [B,Hbev*Wbev,d]
        query = query.to(torch.float32)
        spatial_shapes = torch.as_tensor([[Hbev, Wbev]], device=bev.device, dtype=torch.long)
        level_start_index = torch.as_tensor([0], device=bev.device, dtype=torch.long)
        # ---- MSDeformAttn call ----
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

        return y.view(B, Hrv, Wrv, -1)

__all__ = ["AngleBinner3D"]


