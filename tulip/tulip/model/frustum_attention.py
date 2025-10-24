from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .query_generator import SimpleQueryGenerator, ELEV_DEG_PER_RING_NUCSENES
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention as MSDA
import pdb


class BEV_MSDA(nn.Module):
    def __init__(self, embed_dims=d, num_heads=n_heads,
                        num_levels=1, num_points=6,
                        batch_first=True):
        super().__init__()
        
        self.msda = MSDA(embed_dims=d,
                         num_heads=n_heads,
                         num_levels=num_levels,
                         num_points=num_points,
                         batch_first=True)
         
        self.add_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d)
        )
        self.ffn_norm = nn.LayerNorm(d)
        
    def forward(self, query, value, reference_points, spatial_shapes, level_start_index):
        
        with torch.cuda.amp.autocast(enabled=False):
            y_msda = self.msda(query=query,
                                reference_points=reference_points,
                                value=value,
                                spatial_shapes=spatial_shapes,
                                level_start_index=level_start_index,
                                key_padding_mask=None)
            
        y = self.add_norm(y_msda + query)
        y_ffn = self.ffn(y)
        y = self.ffn_norm(y_ffn + y)
        
        return y


class RV2BEVFrustumAttn(nn.Module):
    def __init__(self, C_rv, C_bev, d=128,
                 n_heads=8, msda_points=6, num_msda=1,
                 rmax=51.2, bin_size=0.8, bev_extent=(-51.2,51.2,-51.2,51.2),
                 rv_size=(2,64),
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
        self.proj_q = nn.Conv2d(d, d, 1, bias=True)
        self.proj_v = nn.Conv2d(C_bev, d, 1, bias=True)
        self.proj_o = nn.Conv2d(d, C_rv, 1, bias=True)
        
        self.query_generator = SimpleQueryGenerator(C_rv=C_rv, rmax=rmax,
                                                    in_rv_size=rv_size, 
                                                    og_rv_size=og_rv_size)
        
        self.msda_layers = nn.ModuleList()
        for l in range(num_msda):
            self.msda_layers.append(
                    BEV_MSDA(embed_dims=self.d, num_heads=self.n_heads,
                                num_levels=1, num_points=msda_points,
                                batch_first=True)
                )

        pe = self.build_sinusoidal_bev_pe(H=128, W=128, C=d, device='cpu')
        self.register_buffer("bev_pe", pe, persistent=False)

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

        # Projections
        Vmap = self.proj_v(bev)
        Vmap = Vmap + self.bev_pe

        p_ego_h, depth_logits, rv_feats = self.query_generator(x_rv, lidar2ego_mat, ret_feats=True)
        Q0 = self.proj_q(rv_feats).permute(0,2,3,1) # [BHWC]
        # Splat onto BEV plane and normalize to get reference points
        x_ego = p_ego_h[..., 0]
        y_ego = p_ego_h[..., 1]
        rx = (x_ego - self.xmin) / (self.xmax - self.xmin)
        ry = (y_ego - self.ymin) / (self.ymax - self.ymin)
        # Stack coordinates into the last dimension
        ref = torch.stack([rx.clamp(0, 1), ry.clamp(0, 1)], dim=-1)
        ref = ref.view(B, -1, 2) # [B, len_q, 2]
        
        # Flatten query and value for MSDA
        query = query.flatten(B, -1, self.d).contiguous() # [B, Len_q_pts, d]

        reference_points = ref.reshape(B, -1, 2)                     # [B, Len_q_pts, 2]
        reference_points = reference_points.unsqueeze(2)       # [B, Len_q_pts, 1, 1, 2]

        value = Vmap.flatten(2).transpose(1,2).contiguous() # [B, Hbev*Wbev, d]

        original_dtype = query.dtype
        value = value.to(torch.float32)
        query = query.to(torch.float32)
        spatial_shapes = torch.as_tensor([[Hbev, Wbev]], device=bev.device, dtype=torch.long)
        level_start_index = torch.as_tensor([0], device=bev.device, dtype=torch.long)

        for msda_layer in msda_layers:
            query = msda_layer(query, 
                               value, 
                               reference_points, 
                               spatial_shapes, 
                               level_start_index)

        y = self.proj_o(query)
        
        y = y.contiguous()

        # Return the KL loss to be added to the main training objective
        return y, depth_logits, torch.tensor(0.0)


if __name__ == "__main__":
    # Minimal test snippet
    print("Testing RV2BEVFrustumAttn...")
    
    # Create test inputs
    B, Hrv, Wrv, Crv = 2, 2, 64, 384
    Cb, Hbev, Wbev = 80, 128, 128
    
    # Create dummy range view features
    x_rv = torch.randn(B, Hrv, Wrv, Crv)
    
    # Create dummy BEV features
    bev = torch.randn(B, Cb, Hbev, Wbev)
    
    # Create dummy lidar2ego transformation matrix
    lidar2ego_mat = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    
    # Initialize the model
    model = RV2BEVFrustumAttn(
        C_rv=Crv,
        C_bev=Cb,
        d=128,
        rmax=51.2,
        bin_size=0.8,
        bev_extent=(-51.2, 51.2, -51.2, 51.2),
        n_heads=8,
        msda_points=6,
        rv_size=(2, 64),
        og_rv_size=(32, 1024)
    )
    
    # Test forward pass
    try:
        with torch.no_grad():
            output, depth_logits, kl_loss = model(x_rv, bev, lidar2ego_mat)
        
        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {x_rv.shape}")
        print(f"  BEV shape: {bev.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Depth logits shape: {depth_logits.shape}")
        print(f"  KL loss: {kl_loss}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("Test completed.")