import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import TORCH_VERSION, digit_version
from torch.nn.init import normal_
from .spatial_cross_attn import SpatialCrossAttention
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import build_norm_layer
import math
import numpy as np
from .query_generator import SimpleQueryGenerator
from .ppe import PointCoordEncoder
import pdb



class CrossModalSCA(nn.Module):
    def __init__(self, embed_dims=128, msda_points=8, num_cams=6, dropout=0.1, num_levels=4, 
                 deformable_attention = dict( type='MSDeformableAttention3D',
                                                embed_dims=256,
                                                num_levels=4,
                                                im2col_step=256)
                    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.msda_points = msda_points
        self.num_cams = num_cams
        self.dropout = dropout
        
        norm1_name, self.norm1 = build_norm_layer(dict(type='LN'), num_features=embed_dims)
        self.spatial_cross_attention = SpatialCrossAttention(num_cams=num_cams, embed_dims=embed_dims, num_levels=4, deformable_attention=deformable_attention)
        self.dropout1 = nn.Dropout(dropout)
        norm2_name, self.norm2 = build_norm_layer(dict(type='LN'), num_features=embed_dims)
        self.ffn = FFN(embed_dims, embed_dims)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, query, img_feats, reference_points_cam, mask):
        query = self.norm1(query)

        # spatial_shapes: [num_levels, 2]; level_start_index: [num_levels]
        # It returns slots: [B, N, C_rv] 
        fused = self.spatial_cross_attention(
                    query=query,           # Pass NORMALIZED query
                    key=img_feats['mlvl_feats'],              # Use img_feats as key
                    value=img_feats['mlvl_feats'],            # Use img_feats as value
                    reference_points_cam=reference_points_cam,  # [num_cam, B, N, K, 2]
                    bev_mask=mask,                          # [num_cam, B, N, K]
                    spatial_shapes=img_feats['spatial_shapes'],
                    level_start_index=img_feats['level_start_index'],
                    flag='encoder'
                )
        
        normed_fused = self.norm2(fused)
        ffn_output = self.ffn(normed_fused)
        fused = normed_fused + self.dropout2(ffn_output)

        return fused

class FFN(nn.Module):
    """
    A simple Feed-Forward Network module.
    """
    def __init__(self, embed_dim, feedforward_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, embed_dim)

    def forward(self, x):
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return x

class RV2MVImgAttn(nn.Module):
    def __init__(self, C_rv, rmax, embed_dims=128, 
                 msda_points=8, num_cams=6, num_levels=4, num_layers=1, im2col_step=256,
                 dropout=0.1, in_rv_size=(32,1024), only_low_res=False,
                 dataset_name='nuscenes'):
        super().__init__()
        self.C_rv = C_rv
        self.rmax = rmax
        self.msda_points = msda_points
        self.num_cams = num_cams
        self.embed_dims = embed_dims
        self.in_Hrv, self.in_Wrv = in_rv_size
        
        self.ref_pts_generator = SimpleQueryGenerator(C_rv=C_rv, rmax=rmax, 
                                                      in_rv_size=in_rv_size,
                                                      only_low_res=only_low_res,
                                                      dataset_name=dataset_name)

        self.proj_q = nn.Sequential(
            nn.Conv2d(C_rv, embed_dims, 1, bias=True), 
            nn.GELU(),
            nn.Conv2d(embed_dims, embed_dims, 1, bias=True)
        )
        
        self.proj_out = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), 
            nn.GELU(),
            nn.Linear(embed_dims, C_rv)
        )
        
        self.cross_modal_sca_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.cross_modal_sca_layers.append(
                CrossModalSCA(embed_dims=embed_dims, msda_points=msda_points, 
                                num_cams=num_cams, num_levels=num_levels, dropout=dropout,
                                deformable_attention=dict(
                                                            type='MSDeformableAttention3D',
                                                            embed_dims=embed_dims,
                                                            num_levels=4,
                                                            im2col_step=im2col_step))
                                                        )
        self.n_q_h = self.ref_pts_generator.n_q_h
        self.n_q_w = self.ref_pts_generator.n_q_w
        self.num_q_per_latent_cell = self.ref_pts_generator.num_q_per_latent_cell
        
        if self.num_q_per_latent_cell > 1:
        # conditional instantiation bc ddp will complain about unused parameters otherwise
            self.query_aggregate = nn.Conv2d(C_rv, C_rv, kernel_size=(self.n_q_h, self.n_q_w), stride=(self.n_q_h, self.n_q_w))

    # This function must use fp32!!!
    @force_fp32(apply_to=('points_lidar', 'lidar2img', 'img_shapes'))
    def point_sampling(self, points_lidar, lidar2img, img_shapes):
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        ones = torch.ones_like(points_lidar[..., :1])
        points_lidar = torch.cat([points_lidar, ones], dim=-1) # [B, Hrv* Wrv, 4]
        reference_points_cam = torch.einsum('bcij,bnj->bcni',
                            lidar2img.to(torch.float32),    # [B, num_cam, 4, 4]
                            points_lidar.to(torch.float32))    # [B, N, 4]
        eps = 1e-5

        mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        # assumes all images have the same shape
        reference_points_cam[..., 0] /= img_shapes[0][1]
        reference_points_cam[..., 1] /= img_shapes[0][0]

        mask = mask & (reference_points_cam[..., 1:2] > 0.0) \
                & (reference_points_cam[..., 1:2] < 1.0) \
                & (reference_points_cam[..., 0:1] < 1.0) \
                & (reference_points_cam[..., 0:1] > 0.0)
                
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            mask = torch.nan_to_num(mask)
        else:
            mask = mask.new_tensor(np.nan_to_num(mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.unsqueeze(-2)

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_cam, mask
    
    def forward(self, in_rv_feat, img_feats, lidar2img_rts, img_shapes, return_nhwc=False, lr_depths=None, target_depths=None, gt_mixture_weight = 0.0):
        """
        Fuses range-view (RV) tokens with multi-view image features using SpatialCrossAttention.

        Expected inputs (two supported styles):

        rv_feat: Tensor [B, C_rv Hrv, Wrv]
        depths: Tensor [B, Hrv, Wrv, 1]
        img_feats: dict containing:
            'mlvl_feats': Tensor [num_cams, spatial_flattened, B, C]
            'spatial_shapes': Tensor [num_levels, 2]
            'level_start_index': Tensor [num_levels]
        
        Returns:
            fused_rv: Tensor [B, Hrv, Wrv, C_rv]  # same layout as rv_feat if input was NHWC; otherwise returns NHWC.
        """
        B, in_C, in_Hrv, in_Wrv = in_rv_feat.shape # range view latent size
        points_lidar, interm_depths, rv_feat = self.ref_pts_generator(in_rv_feat, ret_feats=True, lr_depths=lr_depths,
                                                                      target_depths=target_depths, gt_mixture_weight=gt_mixture_weight) 
        rv_feat = self.proj_q(rv_feat)

        reference_points_cam, mask = self.point_sampling(points_lidar, lidar2img_rts, img_shapes)
        #   reference_points_cam: [B, num_cam, N, 1, 2]  (N=ogHrv*ogWrv)
        #   mask:                [B, num_cam, N, 1]
        # SpatialCrossAttention expects [num_cam, B, N, 1, 2] and mask [num_cam, B, N, 1]
        reference_points_cam = reference_points_cam.permute(1, 0, 2, 3, 4).contiguous()
        
        mask = mask.permute(1, 0, 2, 3).contiguous()   # keep last singleton (K=1)

        # attend per-latent-sample; if created multiple latent samples per RV cell flatten them here.
        query = rv_feat.flatten(2).transpose(1, 2)  # [B, N, C_rv], N must equal N used in point_sampling
       
        for layer in self.cross_modal_sca_layers:
            query = layer(query, img_feats, reference_points_cam, mask)
        
        # query is in shape [B, N, C_rv]
        fused = self.proj_out(query)

        if self.num_q_per_latent_cell > 1:
            fused = fused.transpose(1, 2).reshape(B, self.C_rv, (self.in_Hrv * self.n_q_h), (self.in_Wrv * self.n_q_w))
            fused = self.query_aggregate(fused)  # [B, inC_rv, inHrv, inWrv]
            fused = fused.permute(0,2,3,1).contiguous()  # [B, inHrv, inWrv, inC_rv]
        else:
            fused = fused.view(B, (self.in_Hrv * self.n_q_h), (self.in_Wrv * self.n_q_w), self.C_rv) # [B, inHrv, inWrv, inC_rv]
            
        if not return_nhwc:
            fused = fused.permute(0,3,1,2).contiguous()  # [B, inC_rv, inHrv, inWrv]
                
        return fused, interm_depths


if __name__ == "__main__":
    rv2mvimg_attn = RV2MVImgAttn(C_rv=96, rmax=51.2, msda_points=4)
    range_img = torch.randn(1, 96, 32, 1024)
    img_feats = {
        'mlvl_feats': torch.randn(1, 6, 720*1080, 96),
        'spatial_shapes': torch.tensor([[720, 1080]]),
        'level_start_index': torch.tensor([0])
    }
    lidar2img_rts = torch.randn(1, 6, 4, 4)
    img_shapes = torch.tensor([[720, 1080]])
    
    fused = rv2mvimg_attn.forward(range_img, img_feats, lidar2img_rts, img_shapes)
    print(fused.shape)