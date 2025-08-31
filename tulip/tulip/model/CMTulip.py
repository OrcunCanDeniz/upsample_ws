import torch
import torch.nn as nn
import torch.nn.functional as func

from einops import rearrange
from typing import Optional, Tuple

from functools import partial
from tulip.util.filter import *

from tulip.util.evaluation import inverse_huber_loss
from tulip.model.swin_transformer_v2 import SwinTransformerBlockV2, PatchMergingV2
from .tulip import TULIP
from bevdepth.layers.backbones.tulip_lss_fpn import BaseLSSFPN

import collections.abc


class TokenLearnerBEV(nn.Module):
    """
    X: (B, C, H, W)  ->  K tokens (B, K, C)
    'og' style: LayerNorm on inputs (per-position over channels).
    """
    def __init__(self, C: int, K: int = 4, hidden: int = 0,
                 norm_masks: str = "softmax", temperature: float = 1.0,
                 ln_on_tokens: bool = False):
        super().__init__()
        self.K = K
        self.norm_masks = norm_masks
        self.temperature = temperature

        # LN over channel dim for each spatial position (B, H, W, C)
        self.in_ln = nn.LayerNorm(C)

        if hidden and hidden > 0:
            self.mask_net = nn.Sequential(
                nn.Conv2d(C, hidden, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(hidden, K, kernel_size=1, bias=True),
            )
        else:
            self.mask_net = nn.Conv2d(C, K, kernel_size=1, bias=True)

        # Optional LN on output tokens (per token over channels)
        self.out_ln = nn.LayerNorm(C) if ln_on_tokens else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: (B, K, C)
        """
        B, C, H, W = x.shape

        # ---- LayerNorm on inputs (per-position over channels) ----
        # (B,C,H,W) -> (B,H,W,C) -> LN(C) -> (B,C,H,W)
        x_ln = x.permute(0, 2, 3, 1)                      # (B,H,W,C)
        x_ln = self.in_ln(x_ln)
        x_ln = x_ln.permute(0, 3, 1, 2).contiguous()      # (B,C,H,W)

        # ---- Predict K spatial masks ----
        logits = self.mask_net(x_ln)                      # (B,K,H,W)

        A = F.softmax(logits.flatten(2) / max(self.temperature, 1e-6), dim=-1)
        A = A.view(B, self.K, H, W)                  # (B,K,H,W)

        # ---- Weighted pooling over HxW ----
        x_flat = x_ln.view(B, C, H * W)                   # (B,C,HW)
        A_flat = A.view(B, self.K, H * W)                 # (B,K,HW)
        Z = torch.einsum('bkh,bch->bkc', A_flat, x_flat)  # (B,K,C)

        # ---- Optional LN on tokens ----
        # Z = self.out_ln(Z)                                # (B,K,C)
        return Z


class CMTULIP(TULIP):
    def __init__(self, backbone_config, **kwargs):
        super(CMTULIP, self).__init__(**kwargs)

        self.multiview_backbone = BaseLSSFPN(**backbone_config)
        self.token_learner = TokenLearnerBEV(C=embed_dim, K=16, hidden=128, temperature=1.0)

        self.apply(self.init_weights)


    def forward(self, x, in_imgs, mats_dict, timestamps, target, eval = False, mc_drop = False):
        
        # get depth features
        bev_feat = self.multiview_backbone(in_imgs, mats_dict, timestamps)
        bev_tokens = self.token_learner(bev_feat)

        x = self.patch_embed(x) 
        x = self.pos_drop(x) 
        x_save = []
        for i, layer in enumerate(self.layers):
            x_save.append(x)
            x = layer(x)
            
        x = self.first_patch_expanding(x)


        for i, layer in enumerate(self.layers_up):
            x = torch.cat([x, x_save[len(x_save) - i - 2]], -1)
            x = self.skip_connection_layers[i](x)
            x = layer(x)

        
        x = self.norm_up(x)
        

        if self.pixel_shuffle:
            x = rearrange(x, 'B H W C -> B C H W')
            x = self.ps_head(x.contiguous())
        else:
            x = self.final_patch_expanding(x)
            x = rearrange(x, 'B H W C -> B C H W')


        x = self.decoder_pred(x.contiguous())
            
        if mc_drop:
            return x
        else:
            total_loss, pixel_loss = self.forward_loss(x, target)
            return x, total_loss, pixel_loss

def tulip_base(**kwargs):
    model = CMTULIP(
        depths=(2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model

def tulip_large(**kwargs):
    model = CMTULIP(
        depths=(2, 2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24, 48),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model





