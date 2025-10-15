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
                 rmax=51.2, # K is removed, we use the full distribution
                 bin_size=0.8, bev_extent=(-51.2,51.2,-51.2,51.2),
                 n_heads=8,
                 msda_points=6,
                 rv_size=(2,64),
                 vfov=((-30.67,10.67))
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
        self.proj_o = nn.Conv2d(d, C_out, 1, bias=True)

        self.q_depth = nn.Sequential(
            nn.Conv2d(d + 2, d, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(d, d, 1, bias=True)
        )

        # Light range proposal head
        self.n_bins = int(self.rmax // self.bin_size)
        self.range_head = nn.Sequential(
            nn.Conv2d(C_rv + 3, 128, 1, bias=True),
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1, padding_mode="circular", bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, self.n_bins, 1, bias=True) # depth distribution logits
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

        self.kl_weight    = 1e-4   # tiny KL prior

        #  deformAttn (1 level, 2D)
        self.msda = MSDA(embed_dims=d,
                         num_heads=n_heads,
                         num_levels=1,
                         num_points=msda_points,
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
        Q0   = self.proj_q(x_rv)
        Vmap = self.proj_v(bev)
        Vmap = Vmap + self.bev_pe

        # Range head -> distribution
        range_in = torch.cat([x_rv, batch_u_vec], 1)
        depth_logits = self.range_head(range_in)
        depth_dist = F.softmax(depth_logits / temperature, dim=1)

        # Instead of topk, compute expected depth (μ) and standard deviation (σ)
        mu_d = torch.sum(depth_dist * self.bin_centers, dim=1, keepdim=True)

        # Var[d] = E[d^2] - (E[d])^2
        var_d = torch.sum(depth_dist * (self.bin_centers ** 2), dim=1, keepdim=True) - (mu_d ** 2)
        sigma_d = torch.sqrt(var_d.clamp(min=1e-6)) # Add epsilon for stability

        # Project expected depth `mu_d` into 3D space
        expected_depths = mu_d.clamp_max(self.rmax - 0.5 * self.bin_size)
        x_l = expected_depths * self.u_vec_x
        y_l = expected_depths * self.u_vec_y
        z_l = expected_depths * self.u_vec_z
        # ==============================================================================

        ones = torch.ones_like(x_l)
        # Note: Shape is now [B, 1, Hrv, Wrv], we squeeze the singular dim 1 for stacking
        p_lidar_h = torch.stack([x_l.squeeze(1), y_l.squeeze(1), z_l.squeeze(1), ones.squeeze(1)], dim=-1)
        p_lidar_h = p_lidar_h.view(B, Hrv*Wrv, 4)

        # LiDAR -> Ego
        p_ego_h = p_lidar_h @ lidar2ego_mat
        p_ego_h = p_ego_h.view(B, Hrv, Wrv, 4)

        # Splat onto BEV plane and normalize to get reference points
        x_ego = p_ego_h[..., 0]
        y_ego = p_ego_h[..., 1]
        rx = (x_ego - self.xmin) / (self.xmax - self.xmin)
        ry = (y_ego - self.ymin) / (self.ymax - self.ymin)
        # Note: Shape is now [B, Hrv*Wrv, 1, 2]
        ref = torch.stack([ry.clamp(0,1), rx.clamp(0,1)], dim=-1).view(B, Hrv*Wrv, 1, 2)

        # Augment query with depth mean and std dev, normalized for stability
        mu_norm = (mu_d / self.rmax).clamp(0, 1)
        sigma_norm = (sigma_d / self.rmax).clamp(0, 1)

        Q_aug = torch.cat([Q0, mu_norm, sigma_norm], dim=1) # [B, d+2, Hrv, Wrv]
        Q_depth = self.q_depth(Q_aug)

        # Flatten query and value for MSDA
        query = Q_depth.view(B, self.d, Hrv * Wrv).transpose(1, 2).contiguous() # [B, Hrv*Wrv, d]
        value = Vmap.flatten(2).transpose(1,2).contiguous() # [B, Hbev*Wbev, d]

        original_dtype = query.dtype
        value = value.to(torch.float32)
        query = query.to(torch.float32)
        spatial_shapes = torch.as_tensor([[Hbev, Wbev]], device=bev.device, dtype=torch.long)
        level_start_index = torch.as_tensor([0], device=bev.device, dtype=torch.long)

        with torch.cuda.amp.autocast(enabled=False):
            y_msda = self.msda(query=query,
                                reference_points=ref,
                                value=value,
                                spatial_shapes=spatial_shapes,
                                level_start_index=level_start_index,
                                key_padding_mask=None)

        # Reshape output back to image format
        y = y_msda.transpose(1, 2).contiguous().view(B, self.d, Hrv, Wrv)
        y = self.proj_o(y)
        y = y.permute(0,2,3,1).contiguous()

        # Encourages smoother, less peaky distributions
        log_p = F.log_softmax(depth_logits, dim=1)
        # Target is a uniform distribution
        log_q_uniform = torch.full_like(log_p, -math.log(self.n_bins))
        # KL(P || Q) = Σ P(x) * (log P(x) - log Q(x))
        # Note: F.kl_div expects (input, target) where input is log-prob
        kl_loss = F.kl_div(log_p, log_q_uniform, reduction='batchmean', log_target=True) * self.kl_weight

        # Return the KL loss to be added to the main training objective
        return y, depth_logits, kl_loss
