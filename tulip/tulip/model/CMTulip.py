import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from functools import partial
from util.filter import *

from .tulip import TULIP
from bevdepth.layers.backbones.base_lss_fpn import BaseLSSFPN
from .ray_masked_bev_attn import RayMaskedBEVAttention
import os

import pdb

def _set_bn_eval(module: nn.Module):
    # Works for BN2d/1d/3d and SyncBatchNorm (all subclass _BatchNorm)
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()

def freeze_multiview_backbone(model):
    """
    Freeze all parameters of model.multiview_backbone and set its BN layers to eval().
    Call this AFTER model.to(device) / BEFORE building the optimizer for Phase A.
    Re-apply _set_bn_eval after any global model.train() call.
    """
    m = getattr(model, "module", model)  # handle DDP
    assert hasattr(m, "multiview_backbone"), "model has no attribute 'multiview_backbone'"

    # 1) Freeze params (incl. BN affine)
    for p in m.multiview_backbone.parameters():
        p.requires_grad = False

    # 2) Stop BN running stats updates
    m.multiview_backbone.apply(_set_bn_eval)

def unfreeze_multiview_backbone(model, train_bn: bool = True):
    """
    Unfreeze backbone parameters. If train_bn=True, set BN back to train mode.
    Otherwise, keep BN stats frozen but allow affine to train (set requires_grad True above).
    """
    m = getattr(model, "module", model)
    assert hasattr(m, "multiview_backbone"), "model has no attribute 'multiview_backbone'"

    # 1) Unfreeze params
    for p in m.multiview_backbone.parameters():
        p.requires_grad = True

    # 2) BN behavior
    if train_bn:
        m.multiview_backbone.train()      # BN stats update again
    else:
        m.multiview_backbone.apply(_set_bn_eval)  # keep stats frozen

class CMTULIP(TULIP):
    def __init__(self, backbone_config, lss_weights_path, im2col_step=128, **kwargs):
        super(CMTULIP, self).__init__(**kwargs)

        self.init = False
        self.apply(self.init_weights)
        self.multiview_backbone = BaseLSSFPN(**backbone_config)
        self.load_lss_weights(lss_weights_path)
        self.multiview_backbone.depth_net.depth_conv[4].im2col_step = im2col_step
        self.max_range = 55
        self.frust_attn = RayMaskedBEVAttention()
        self.range_head_weight = 0.05
    
    def load_lss_weights(self, lss_weights_path, strict=False):
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


    def forward(self, x, in_imgs, mats_dict, timestamps, target, lidar2ego_mat, mc_drop = False):
    def forward(self, x, in_imgs, mats_dict, timestamps, target, lidar2ego_mat, mc_drop = False):
        bev_feat = self.multiview_backbone(in_imgs, mats_dict, timestamps)
            
        x = self.patch_embed(x) 
        x = self.pos_drop(x) 
        x_save = []
        for i, layer in enumerate(self.layers):
            x_save.append(x)
            x = layer(x)

        x = self.first_patch_expanding(x)
        x, attn = self.frust_attn(x, bev_feat, lidar2ego_mat)
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
            total_loss, pixel_loss, range_head_loss = self.forward_loss(x, target)
            return x, total_loss, pixel_loss

    def forward_loss(self, pred, target):
        # range_head_target is normalized by max_range
        
        loss_rh = F.mse_loss(rh_preds, target)
        
        loss = (pred - target).abs()
        loss = loss.mean()
        
        total_loss = loss + loss_rh * self.range_head_weight
        
        if self.log_transform:
            pixel_loss = (torch.expm1(pred) - torch.expm1(target)).abs().mean()
        else:
            pixel_loss = loss.clone()
        
        return loss, pixel_loss, 0
    
    
    def gaussian_bins_targets(self, gt_depth, bin_size=0.8, rmax=51.2, sigma_bins=1.0):
        # returns soft targets: [B, n_bins, H, W], sum=1 per pixel
        B, H, W = gt_depth.shape
        gt_depth = (gt_depth * rmax)
        gt_depth =  torch.clamp(gt_depth, 0, rmax)
        n_bins = int(round(rmax / bin_size))
        centers = (torch.arange(n_bins, device=gt_depth.device, dtype=gt_depth.dtype) + 0.5) * bin_size  # [n_bins]
        # [B, n_bins, H, W]
        diff = centers.view(1, n_bins, 1, 1) - gt_depth.view(B, 1, H, W)
        sigma_m = sigma_bins * bin_size
        logits = -0.5 * (diff / (sigma_m + 1e-8))**2                  # unnormalized log-prob
        probs  = torch.softmax(logits, dim=1)
        return probs

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





