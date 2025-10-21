import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
import math
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
import pdb

class ImageBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_backbone = build_backbone(cfg.img_backbone_conf)
        self.img_neck = build_neck(cfg.img_neck_conf)
        self.img_backbone.init_weights()
        self.img_neck.init_weights()

        # neck out_channels can be int or list per level
        outc = cfg.img_neck_conf.get('out_channels', 256)
        self.num_feature_levels = cfg.img_neck_conf.get('num_outs', 4)


        self.embed_dims = int(outc)
        self.proj_per_level = None  # already unified

        assert self.embed_dims % 2 == 0, "embed_dims must be even for 2D sine-cos PE"

        self.num_cams = cfg.get('num_cams', 6)

        # embeddings at unified dim
        self.level_embeds = nn.Parameter(torch.empty(self.num_feature_levels, self.embed_dims))
        self.cams_embeds  = nn.Parameter(torch.empty(self.num_cams, self.embed_dims))
        nn.init.normal_(self.level_embeds, std=0.02)
        nn.init.normal_(self.cams_embeds,  std=0.02)

        self._pe_cache = {}  # optional: cache PE by (H,W,dtype,device)
        
    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        normal_(self.level_embeds, mean=0, std=0.02)
        normal_(self.cams_embeds, mean=0, std=0.02)
        
    @staticmethod
    def _build_2d_sincos_pos_embed(C: int, H: int, W: int, device, dtype,
                                temperature: float = 10000.,
                                normalize: bool = True,
                                scale: float = 2*math.pi) -> torch.Tensor:
        """
        Return a [C, H, W] 2D sine-cosine positional encoding that matches any C (even or odd).
        We allocate Cx channels to X and Cy=C-Cx to Y, and fill each exactly.

        - If normalize=True, positions are in [0, scale] for better conditioning across sizes.
        - Works for arbitrary C (no need for C % 2 == 0).
        """

        # grid in [0,scale] (or raw indices)
        if normalize:
            yv = torch.linspace(0, 1, steps=H, device=device, dtype=dtype) * scale
            xv = torch.linspace(0, 1, steps=W, device=device, dtype=dtype) * scale
        else:
            yv = torch.arange(H, device=device, dtype=dtype)
            xv = torch.arange(W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(yv, xv, indexing='ij')          # [H,W]

        # split channels between x and y (handle odd C)
        Cx = C // 2
        Cy = C - Cx

        def _sincos_1d(pos: torch.Tensor, Cout: int) -> torch.Tensor:
            """
            Build a [H,W,Cout] (or [W,Cout]) 1D sincos PE for the given scalar field `pos`.
            Exactly Cout channels are produced:
            - use pairs of (sin, cos) for 2*k channels
            - if Cout is odd, append one extra sin channel at the highest frequency
            """
            if Cout == 0:
                # no channels requested
                return torch.empty((*pos.shape, 0), device=pos.device, dtype=pos.dtype)

            # number of sin/cos pairs
            pairs = Cout // 2
            # frequencies
            dim_t = temperature ** (2 * torch.arange(pairs, device=pos.device, dtype=pos.dtype) / max(1, pairs))
            # angles
            ang = pos[..., None] / dim_t                     # [..., pairs]
            emb = [ang.sin(), ang.cos()]                    # two lists of [..., pairs]
            emb = torch.cat(emb, dim=-1)                    # [..., 2*pairs]

            if Cout % 2 == 1:
                # append one extra sin at the highest frequency
                extra_dim_t = temperature ** (2 * torch.tensor(pairs, device=pos.device, dtype=pos.dtype) / max(1, pairs+1))
                extra = (pos / extra_dim_t).sin()[..., None]  # [...,1]
                emb = torch.cat([emb, extra], dim=-1)       # [..., 2*pairs+1] == Cout

            return emb

        # build per-axis embeddings with exact channel counts
        ex = _sincos_1d(xx, Cx)   # [H,W,Cx]
        ey = _sincos_1d(yy, Cy)   # [H,W,Cy]

        pos = torch.cat([ex, ey], dim=-1)   # [H,W,C]
        return pos.permute(2, 0, 1).contiguous()  # [C,H,W]
    
    def forward(self, imgs):
        B, S, num_cams, C, H, W = imgs.shape
        assert S == 1 and num_cams == self.num_cams
        x = imgs.flatten(0, 2)  # [B*Cams, C, H, W]

        feats = self.img_backbone(x)
        mlvl_feats = self.img_neck(feats)  # list of L tensors

        feat_flatten, spatial_shapes = [], []
        for lvl, feat in enumerate(mlvl_feats):
            BC, Cemb, h, w = feat.shape
            bs = BC // num_cams
            feat = feat.contiguous()

            # (1) project per level to unified embed_dims if needed
            if self.proj_per_level is not None:
                feat = self.proj_per_level[lvl](feat)
                Cemb = self.embed_dims
            else:
                assert Cemb == self.embed_dims, f"Level {lvl}: got {Cemb}, expected {self.embed_dims}"

            # (2) add 2D sine-cos PE (cache by size/type/device)
            key = (h, w, feat.dtype, feat.device)
            pe = self._pe_cache.get(key)
            if pe is None:
                pe = self._build_2d_sincos_pos_embed(Cemb, h, w, feat.device, feat.dtype)
                self._pe_cache[key] = pe
            feat = feat + pe.unsqueeze(0)  # [BC,C,h,w] + [1,C,h,w]

            # (3) reshape and add camera + level embeddings
            feat = feat.view(bs, num_cams, Cemb, h, w).flatten(3).permute(1, 0, 3, 2)  # (cam,B,h*w,C)
            feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl+1, :].to(feat.dtype)

            spatial_shapes.append((h, w))
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, dim=2)                 # (cam,B,Σ(h*w),C)
        feat_flatten = feat_flatten.permute(0, 2, 1, 3).contiguous()  # (cam,Σ(h*w),B,C)

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=imgs.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(1),
                                    spatial_shapes.prod(1).cumsum(0)[:-1]))
        return {
            'mlvl_feats': feat_flatten,
            'spatial_shapes': spatial_shapes,
            'level_start_index': level_start_index
        }
