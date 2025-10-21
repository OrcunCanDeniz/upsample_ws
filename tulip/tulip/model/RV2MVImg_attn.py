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
from .query_generator import ELEV_DEG_PER_RING_NUCSENES
import pdb


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
    def __init__(self, C_rv, rmax, embed_dims=128, msda_points=8, num_cams=6,
                 num_q_w=1, num_q_h=1, dropout=0.1, in_rv_size=(32,1024), og_rv_size=(32,1024)):
        super().__init__()
        self.C_rv = C_rv
        self.rmax = rmax
        self.msda_points = msda_points
        self.num_cams = num_cams
        self.embed_dims = embed_dims

        self.proj_q = nn.Conv2d(C_rv, embed_dims, 1, bias=True)
        self.range_head = nn.Conv2d(C_rv, 1, 1) # predicts intermediate normalized depth
        norm1_name, self.norm1 = build_norm_layer(dict(type='LN'), num_features=embed_dims)
        self.spatial_cross_attention = SpatialCrossAttention(num_cams=num_cams, embed_dims=embed_dims, num_levels=4)
        self.dropout1 = nn.Dropout(dropout)
        norm2_name, self.norm2 = build_norm_layer(dict(type='LN'), num_features=embed_dims)
        self.ffn = FFN(embed_dims, embed_dims)
        self.dropout2 = nn.Dropout(dropout)
        
        self.n_q_w = num_q_w
        self.n_q_h = num_q_h
        self.num_q_per_latent_cell = self.n_q_w * self.n_q_h
        self.in_Hrv, self.in_Wrv = int(in_rv_size[0]), int(in_rv_size[1])
        self.og_Hrv, self.og_Wrv = int(og_rv_size[0]), int(og_rv_size[1])
        self.ds_factor_h = self.og_Hrv // self.in_Hrv
        self.ds_factor_w = self.og_Wrv // self.in_Wrv
        
        self.set_geometry()
        
        
    def set_geometry(self):
        # bin azimuths into Wrv bins
        assert self.n_q_w * self.in_Wrv <= self.og_Wrv, "Total number  of horizontal samples must be less than or equal to original width"
        az_line = torch.linspace(-math.pi, math.pi, self.og_Wrv + 1)[:-1]
        az_line += (az_line[1] - az_line[0])/2
        self.az_binned_ = az_line.reshape(self.in_Wrv, self.n_q_w, -1)
        self.process_az()

        # bin elevations into Hrv bins
        assert self.n_q_h * self.in_Hrv <= self.og_Hrv, "Total number of vertical samples must be less than or equal to original height"
        elev = torch.deg2rad(ELEV_DEG_PER_RING_NUCSENES.flip(0))
        self.elev_binned_ = elev.reshape(self.in_Hrv, self.n_q_h, -1)
        self.process_elev()
        
        az = self.az_per_q
        elev = self.elev_per_q
        # compute unit vector per range view pixel 
        cos_az = torch.cos(az).unsqueeze(-1)
        sin_az = torch.sin(az).unsqueeze(-1)
        cos_el = torch.cos(elev).unsqueeze(-2)
        sin_el = torch.sin(elev)
        
        u_vec_x_grid = cos_az * cos_el
        u_vec_y_grid = sin_az * cos_el
        u_vec_z_grid = sin_el.unsqueeze(2).expand_as(u_vec_x_grid)


        u_vec_x = u_vec_x_grid.reshape(self.num_q_per_latent_cell, self.in_Hrv, self.in_Wrv)
        u_vec_y = u_vec_y_grid.reshape(self.num_q_per_latent_cell, self.in_Hrv, self.in_Wrv)
        u_vec_z = u_vec_z_grid.reshape(self.num_q_per_latent_cell, self.in_Hrv, self.in_Wrv)
        u_vec = torch.stack([u_vec_x, u_vec_y, u_vec_z], dim=0)
        
        u_vec_mean = u_vec.mean(1)
        u_vec_norm = u_vec_mean / torch.norm(u_vec_mean, dim=0, keepdim=True)

        self.register_buffer("u_vec_mean", u_vec_norm, persistent=False)
        u_vec_x = u_vec_x.permute(1, 2, 0).contiguous()
        u_vec_y = u_vec_y.permute(1, 2, 0).contiguous()
        u_vec_z = u_vec_z.permute(1, 2, 0).contiguous()
        self.register_buffer("u_vec_x", u_vec_x, persistent=False)
        self.register_buffer("u_vec_y", u_vec_y, persistent=False)
        self.register_buffer("u_vec_z", u_vec_z, persistent=False)
        self.register_buffer("u_vec", u_vec, persistent=False)
        
    def process_az(self):
        # this will process binned azimuth angles to produce useful features
        # self.az_binned_ is (in_Wrv, n_q_w, ds_factor_w//n_q_w)
        # get linearly spaced azimuths for each latent pixel
        assert self.ds_factor_w % self.n_q_w == 0, "in_Wrv must be divisible by n_q_w"
        
        az_per_q = self.az_binned_.mean(-1)
        az_per_q = az_per_q[None, ...]
        self.az_per_q = az_per_q.repeat(self.in_Hrv, 1, 1) # (in_Hrv, in_Wrv, n_q_w)
        az_per_pixel = az_per_q.mean(-1) # (in_Hrv, in_Wrv)
        
    def process_elev(self):
        assert self.ds_factor_h % self.n_q_h == 0, "in_Hrv must be divisible by n_q_h"
        
        elev_per_q = self.elev_binned_.mean(-1) # (in_Hrv, n_q_h)
        elev_per_q = elev_per_q[:, None, :] # (in_Hrv, 1, n_q_h)
        self.elev_per_q = elev_per_q.repeat(1, self.in_Wrv, 1) # (in_Hrv, in_Wrv, n_q_h)
        elev_per_pixel = elev_per_q.mean(-1) # (in_Hrv, in_Wrv)


    # This function must use fp32!!!
    @force_fp32(apply_to=('range_img', 'lidar2img', 'img_shapes'))
    def point_sampling(self, range_img, lidar2img, img_shapes):
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        B, Hrv, Wrv, Crv = range_img.shape
        assert Crv == 1, "Range image must have 1 channel"
        num_cam = lidar2img.size(1) # B, num_cam, 4, 4
        
        # Expand unit vectors for batch operations
        u_vec_x_batch = self.u_vec_x.unsqueeze(0)
        u_vec_y_batch = self.u_vec_y.unsqueeze(0)
        u_vec_z_batch = self.u_vec_z.unsqueeze(0)
        x_l = range_img[..., 0:1] * u_vec_x_batch
        y_l = range_img[..., 0:1] * u_vec_y_batch
        z_l = range_img[..., 0:1]* u_vec_z_batch
        
        ones = torch.ones_like(x_l)
        # Stack coordinates: [B, Hrv, Wrv, num_q_per_latent_cell, 4]
        p_lidar_h = torch.stack([x_l, y_l, z_l, ones], dim=-1)
        p_lidar_h = p_lidar_h.view(B, -1, 4)

        reference_points_cam = torch.einsum('bcij,bnj->bcni',
                            lidar2img.to(torch.float32),    # [B, num_cam, 4, 4]
                            p_lidar_h.to(torch.float32))    # [B, N, 4]
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
    
    def forward(self, in_rv_feat, img_feats, lidar2img_rts, img_shapes):
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

        rv_feat = self.proj_q(in_rv_feat)
        B, C_rv, Hrv, Wrv = rv_feat.shape
        
        interm_depths_norm = self.range_head(in_rv_feat).sigmoid() # normalized intermediate depth
        interm_depths = (interm_depths_norm * self.rmax).detach()

        # ---- Build reference points & masks per camera (K=1 anchor per RV pixel) ----
        # point_sampling expects range_img = [B,Hrv,Wrv,Crv] (Crv can be 1)
        reference_points_cam, mask = self.point_sampling(interm_depths.permute(0, 2, 3, 1).contiguous(), lidar2img_rts, img_shapes)
        # current shapes from point_sampling:
        #   reference_points_cam: [B, num_cam, N, 1, 2]  (N=Hrv*Wrv*self.num_q_per_latent_cell)
        #   mask:                [B, num_cam, N, 1]
        # SpatialCrossAttention expects [num_cam, B, N, K, 2] and mask [num_cam, B, N, K]
        reference_points_cam = reference_points_cam.permute(1, 0, 2, 3, 4).contiguous()
        
        mask = mask.permute(1, 0, 2, 3).contiguous()   # keep last singleton (K=1)

        # ---- Prepare queries from RV tokens --------------------------------------
        # We attend per-latent-sample; if you created multiple latent samples per RV cell,
        # flatten them here. point_sampling already used 'num_q_per_latent_cell' in N.
        query = rv_feat.flatten(2).transpose(1, 2)  # [B, N, C_rv], N must equal N used in point_sampling
        if self.num_q_per_latent_cell > 1:
            query = query.repeat_interleave(self.num_q_per_latent_cell, dim=1)  # [B, H*W*Q, C]
        query = self.norm1(query)

        # ---- Call SpatialCrossAttention (DeformableAttn under the hood) ----------
        # spatial_shapes: [num_levels, 2]; level_start_index: [num_levels]
        # It returns slots: [B, N, C_rv] (residual is added inside)
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
        fused = fused + self.dropout2(ffn_output)

        fused = fused.view(B, C_rv, Hrv, Wrv)

        return fused, interm_depths_norm


if __name__ == "__main__":
    rv2mvimg_attn = RV2MVImgAttn(C_rv=96, rmax=51.2, msda_points=4)
    range_img = torch.randn(1, 32, 1024, 1)
    depths = torch.randn(1, 32, 1024, 1)
    img_metas = [
        {
            'lidar2img': torch.randn(1, 4, 4),
            'img_shape': [(720, 1080)]
        },
        {
            'lidar2img': torch.randn(1, 4, 4),
            'img_shape': [(720, 1080)]
        }
    ]
    
    fused = rv2mvimg_attn.forward(range_img, depths, img_metas)
    print(fused.shape)