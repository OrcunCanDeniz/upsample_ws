import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from functools import partial
from util.filter import *

from .tulip import TULIP
from bevdepth.layers.backbones.base_lss_fpn import BaseLSSFPN
import os

import pdb


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


class GlobalBEVAttention(nn.Module):
    """
    Cross-attn: RV BHWC -> global g_tokens (no Conv2d, channel-only Linear).
    x_mid: (B,H,W,C_mid)  e.g., (1,2,64,384)
    g_tokens: (B,K,C_bev) from TokenLearner
    """
    def __init__(self, c_mid=384, c_bev=32, d=128, heads=4, ln_eps=1e-6):
        super().__init__()
        assert d % heads == 0
        self.h = heads
        self.dk = d // heads

        # per-position LN over channels (BHWC)
        self.ln_rv = nn.LayerNorm(c_mid, eps=ln_eps)

        # channel-only projections
        self.q_proj = nn.Linear(c_mid, d, bias=False)  # RV -> Q
        self.k_proj = nn.Linear(c_bev, d, bias=False)  # globals -> K
        self.v_proj = nn.Linear(c_bev, d, bias=False)  # globals -> V
        self.out_proj = nn.Linear(d, c_mid, bias=False)

        # gated residual
        self.gate = nn.Parameter(torch.zeros(1))       # tanh(gate) starts ~0

    def _split_heads(self, t):  # (B,L,d) -> (B,h,L,dk)
        B, L, d = t.shape
        return t.view(B, L, self.h, self.dk).permute(0, 2, 1, 3).contiguous()

    def forward(self, x_mid_bhwc, g_tokens):
        B, H, W, C = x_mid_bhwc.shape
        N = H * W
        K = g_tokens.size(1)

        # Pre-norm over C at each (h,w)
        x = self.ln_rv(x_mid_bhwc)        # (B,H,W,C)
        x = x.view(B, N, C)               # (B,N,C)

        Q = self.q_proj(x)                # (B,N,d)
        Kt = self.k_proj(g_tokens)        # (B,K,d)
        Vt = self.v_proj(g_tokens)        # (B,K,d)

        Qh = self._split_heads(Q)         # (B,h,N,dk)
        Kh = self._split_heads(Kt)        # (B,h,K,dk)
        Vh = self._split_heads(Vt)        # (B,h,K,dk)

        attn = torch.matmul(Qh, Kh.transpose(-2, -1)) * (self.dk ** -0.5)  # (B,h,N,K)
        attn = attn.softmax(dim=-1)
        ctx  = torch.matmul(attn, Vh)     # (B,h,N,dk)

        # merge heads
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(B, N, self.h * self.dk)  # (B,N,d)
        y   = self.out_proj(ctx).view(B, H, W, C)                                # (B,H,W,C)

        return x_mid_bhwc + torch.tanh(self.gate) * y


class CMTULIP(TULIP):
    def __init__(self, backbone_config, lss_weights_path, **kwargs):
        super(CMTULIP, self).__init__(**kwargs)

        # TODO: Integrate the multiview backbone and token learner
        self.token_learner = TokenLearnerBEV(C=80, K=16, hidden=128, temperature=1.0)
        self.global_attention = GlobalBEVAttention(c_bev=80, c_mid=384, d=128, heads=4, ln_eps=1e-6)

        self.init = False
        self.apply(self.init_weights)
        self.multiview_backbone = BaseLSSFPN(**backbone_config)
        self.load_lss_weights(lss_weights_path)
    
    def load_lss_weights(self, lss_weights_path, strict=True):
        """
        Load LSS weights into the multiview_backbone from a saved checkpoint.
        
        Args:
            lss_weights_path: Path to the LSS weights (.pth file)
            strict: Whether to strictly enforce key matching
        
        Returns:
            bool: True if loading was successful
        """
        
        print(f"Loading LSS weights from: {lss_weights_path}")
        
        if not os.path.exists(lss_weights_path):
            print(f"Error: LSS weights file not found: {lss_weights_path}")
            return False
        
        # Load the LSS weights
        lss_data = torch.load(lss_weights_path, map_location='cpu')
        
        if 'state_dict' in lss_data:
            lss_state_dict = lss_data['state_dict']
        else:
            lss_state_dict = lss_data
        
        print(f"Loaded {len(lss_state_dict)} LSS weight keys")
        
        # Get the multiview_backbone state dict
        backbone_state_dict = self.multiview_backbone.state_dict()
        print(f"Multiview backbone expects {len(backbone_state_dict)} keys")
        import pdb; pdb.set_trace()
        
        # Check for matching keys
        matching_keys = []
        missing_keys = []
        unexpected_keys = []
        
        for key in backbone_state_dict.keys():
            if key in lss_state_dict:
                matching_keys.append(key)
            else:
                missing_keys.append(key)
        
        for key in lss_state_dict.keys():
            if key not in backbone_state_dict:
                unexpected_keys.append(key)
        
        print(f"Matching keys: {len(matching_keys)}")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        
        if missing_keys and strict:
            print("Missing keys (strict mode):")
            for key in missing_keys[:10]:  # Show first 10
                print(f"  - {key}")
            if len(missing_keys) > 10:
                print(f"  ... and {len(missing_keys) - 10} more")
            return False
        
        if unexpected_keys:
            print("Unexpected keys:")
            for key in unexpected_keys[:10]:  # Show first 10
                print(f"  + {key}")
            if len(unexpected_keys) > 10:
                print(f"  ... and {len(unexpected_keys) - 10} more")
        
        # Load the matching weights
        try:
            self.multiview_backbone.load_state_dict(lss_state_dict, strict=False)
            print(f"✅ Successfully loaded {len(matching_keys)} LSS weights into multiview_backbone")
            return True
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            return False


    def forward(self, x, in_imgs, mats_dict, timestamps, target, mc_drop = False):
        if self.init:
            self.multiview_backbone.eval()
            self.init = True
        
        # get depth features
        with torch.no_grad():
            bev_feat = self.multiview_backbone(in_imgs, mats_dict, timestamps)
            
        bev_tokens = self.token_learner(bev_feat)
        x = self.patch_embed(x) 
        x = self.pos_drop(x) 
        x_save = []
        for i, layer in enumerate(self.layers):
            x_save.append(x)
            x = layer(x)
            
        x = self.global_attention(x, bev_tokens)
            
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





