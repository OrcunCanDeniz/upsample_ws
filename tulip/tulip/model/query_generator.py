import math
from typing import Optional, Tuple
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F
from .coord_conv import CoordConv2d
import pdb

ELEV_DEG_PER_RING_NUCSENES = torch.tensor([-30.67, -29.33, -28., -26.66, -25.33, -24., -22.67, -21.33,
                               -20., -18.67, -17.33, -16., -14.67, -13.33, -12., -10.67,
                                -9.33, -8., -6.66, -5.33, -4., -2.67, -1.33, 0.,
                                1.33, 2.67, 4., 5.33, 6.67, 8., 9.33, 10.67], dtype=torch.float32)


class SimpleQueryGenerator(nn.Module):
    """
    Given feature map from range view, generate nqw*nqh numver of 3D query points in ego frame.
    
    Args:
        rmax: maximum range in meters used to normalize depths
        C_rv: number of channels in range view feature map
        in_rv_size: size of range view feature map to be processed
        og_rv_size: original size of range view image
        only_low_res: If True, only low resolution gt depths will be used to generate points. 
                        If False, range prediction head is used to generate intermediate depths, with resolution of og_rv_size.
    """
    def __init__(self,
                 rmax=51.2,
                 C_rv=384,
                 in_rv_size=(2,64),
                 og_rv_size=(32,1024),
                 only_low_res=False):
        super().__init__()
        
        self.C_rv = C_rv
        self.rmax = rmax     
        self.only_low_res = only_low_res
        
        # 1D range proposal head for processing expanded features
        if not only_low_res:
            self.range_head = nn.Sequential(
                                            nn.GroupNorm(num_groups=8, num_channels=C_rv),
                                            CoordConv2d(C_rv, 64, kernel_size=3, padding=1, bias=True),
                                            nn.SiLU(inplace=True),
                                            nn.Conv2d(64, 2, kernel_size=1, bias=True)
                                        )
            self.offset_mult = nn.Parameter(torch.tensor([-1.0, 0.0, 1.0]), requires_grad=True)
        else:
            self.save_in_rv_size = in_rv_size
            in_rv_size = (32, 1024)
        
        # Build azimuth and zenith maps by in_rv_size (width spans [-pi, pi))
        self.og_Hrv, self.og_Wrv = int(og_rv_size[0]), int(og_rv_size[1])
        self.in_Hrv, self.in_Wrv = int(in_rv_size[0]), int(in_rv_size[1])
        self.ds_factor_h = self.og_Hrv // self.in_Hrv
        self.ds_factor_w = self.og_Wrv // self.in_Wrv
        
        self.n_q_w = int(self.ds_factor_w)
        self.n_q_h = int(self.ds_factor_h)
        assert self.n_q_w == self.ds_factor_w, "ogW / inW must be int" 
        assert self.n_q_h == self.ds_factor_h, "ogH / inH must be int" 
        
        self.num_q_per_latent_cell = self.n_q_w * self.n_q_h
        
        if self.num_q_per_latent_cell > 1:
            assert self.n_q_w == self.n_q_h, "Only symmetrical expanding supported"
            num_ups = int(math.log(self.n_q_w, 2))
            up_layers = [CoordConv2d(C_rv, C_rv, kernel_size=1, padding=0, bias=False)]
            # TODO Dropouts?
            for _ in range(num_ups):
                up_layers.extend([
                    nn.Conv2d(in_channels=C_rv, out_channels=C_rv*4, kernel_size=(1, 1)),
                    nn.BatchNorm2d(C_rv*4),
                    nn.GELU(),
                    nn.Conv2d(in_channels=C_rv*4, out_channels=C_rv*4, kernel_size=(1, 1)),
                    nn.PixelShuffle(upscale_factor=2)
                ])
            self.spatial_expand = nn.Sequential(*up_layers)
        
        self.set_geometry(True)
        self.low_res_index = range(0, 32, 4)
        
        if only_low_res:
            self.u_vec_x = self.u_vec_x[:, :, :, self.low_res_index, :]
            self.u_vec_y = self.u_vec_y[:, :, :, self.low_res_index, :]
            self.u_vec_z = self.u_vec_z[:, :, :, self.low_res_index, :]
            self.in_Hrv = self.save_in_rv_size[0]
            self.in_Wrv = self.save_in_rv_size[1]
            self.ds_factor_h = 8 // self.in_Hrv
            self.ds_factor_w = 1024 // self.in_Wrv
            
            self.n_q_w = int(self.ds_factor_w)
            self.n_q_h = int(self.ds_factor_h)
            
            assert self.n_q_w == self.ds_factor_w, "ogW / inW must be int" 
            assert self.n_q_h == self.ds_factor_h, "ogH / inH must be int" 
            
            self.num_q_per_latent_cell = self.n_q_w * self.n_q_h
            num_addit_ch = self.C_rv * self.num_q_per_latent_cell
            self.populate_channels = nn.Sequential(
                nn.Conv2d(in_channels=C_rv, out_channels=num_addit_ch, kernel_size=(1, 1)),
                nn.BatchNorm2d(num_addit_ch),
                nn.GELU(),
                nn.Conv2d(in_channels=num_addit_ch, out_channels=num_addit_ch, kernel_size=(1, 1)),
            )
        else:
            self.ds_factor_h = 32 // self.in_Hrv
            self.ds_factor_w = 1024 // self.in_Wrv
            
            self.n_q_w = int(self.ds_factor_w)
            self.n_q_h = int(self.ds_factor_h)
            
            assert self.n_q_w == self.ds_factor_w, "ogW / inW must be int" 
            assert self.n_q_h == self.ds_factor_h, "ogH / inH must be int" 
            
            self.num_q_per_latent_cell = self.n_q_w * self.n_q_h
                 
    def set_geometry(self, no_binning=False):
        # bin azimuths into Wrv bins
        assert self.n_q_w * self.in_Wrv <= self.og_Wrv, "Total number  of horizontal samples must be less than or equal to original width"
        az_line = torch.linspace(-math.pi, math.pi, self.og_Wrv + 1)[:-1]
        az_line += (az_line[1] - az_line[0])/2
        if no_binning:
            self.az_per_q = az_line[None, :, None].repeat(self.og_Hrv, 1, 1) # (in_Hrv, in_Wrv, n_q_w)
        else:
            self.az_binned_ = az_line.reshape(self.in_Wrv, self.n_q_w, -1)
            self.process_az()

        # bin elevations into Hrv bins
        assert self.n_q_h * self.in_Hrv <= self.og_Hrv, "Total number of vertical samples must be less than or equal to original height"
        elev = torch.deg2rad(ELEV_DEG_PER_RING_NUCSENES.flip(0))
        if no_binning:
            self.elev_per_q = elev[:, None, None].repeat(1, self.og_Wrv, 1) # (in_Hrv, in_Wrv, n_q_h)
        else:
            self.elev_binned_ = elev.reshape(self.in_Hrv, self.n_q_h, -1)
            self.process_elev()
        az = self.az_per_q # (in_Hrv, in_Wrv, n_q_w)
        elev = self.elev_per_q # (in_Hrv, in_Wrv, n_q_h)
        # compute unit vector per range view pixel 
        # pdb.set_trace()
        cos_az = torch.cos(az).unsqueeze(-1) # (in_Hrv, in_Wrv, n_q_w, 1)
        sin_az = torch.sin(az).unsqueeze(-1) # (in_Hrv, in_Wrv, n_q_w, 1)
        cos_el = torch.cos(elev).unsqueeze(-2)
        sin_el = torch.sin(elev) # (in_Hrv, in_Wrv, n_q_h)
        
        u_vec_x_grid = cos_az * cos_el# (in_Hrv, in_Wrv, n_q_w, n_q_h)
        u_vec_y_grid = sin_az * cos_el# (in_Hrv, in_Wrv, n_q_w, n_q_h)
        u_vec_z_grid = sin_el.unsqueeze(2).expand_as(u_vec_x_grid)

        u_vec_x = u_vec_x_grid.permute(3,2,0,1)# 
        u_vec_y = u_vec_y_grid.permute(3,2,0,1)# (n_q_h, n_q_w, in_Hrv, in_Wrv)
        u_vec_z = u_vec_z_grid.permute(3,2,0,1)# 
        u_vec = torch.stack([u_vec_x, u_vec_y, u_vec_z], dim=0)
        
        if not no_binning:
            u_vec_x = u_vec_x.unsqueeze(0).contiguous()
            u_vec_y = u_vec_y.unsqueeze(0).contiguous()
            u_vec_z = u_vec_z.unsqueeze(0).contiguous()
        else:
            nqh, nqw, _, _ = u_vec_x.shape
            assert nqh == 1 and nqw == 1, "When no binning, n_q_h and n_q_w must be 1"
            u_vec_x = u_vec_x.squeeze(0).contiguous()
            u_vec_y = u_vec_y.squeeze(0).contiguous()
            u_vec_z = u_vec_z.squeeze(0).contiguous()
            # only squeeze one dim to keep (1, in_Hrv, in_Wrv) shape, broadcastable to batch dim later
        self.register_buffer("u_vec_x", u_vec_x, persistent=False)
        self.register_buffer("u_vec_y", u_vec_y, persistent=False)
        self.register_buffer("u_vec_z", u_vec_z, persistent=False)

        
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

    def _point_hypothesis(self, x_rv, lr_depths=None, target_depths=None, gt_mixture_weight = 0.0):
        """
        Generate points in lidar frame by predicting intermediate normalized depth and unit vectors
        This will always get rv with original size.
        Use forward() since it also handles spatial expanding if needed.
        
        x_rv: [B, Crv, Hrv, Wrv]
        lr_depths: 0-1 ranged low resolution depth image fed as input to the model
        return: [B, Hrv* Wrv, 3]
        """
        B, Crv, Hrv, Wrv = x_rv.shape
        if not self.only_low_res:
            assert Hrv == self.og_Hrv and Wrv == self.og_Wrv, f"Input range view size {Hrv}x{Wrv} does not match expected size {self.og_Hrv}x{self.og_Wrv}"
        # Apply range head to each range view pixel

        if self.only_low_res:
            # if only low res gt preds will be used to generate points. no range prediction head
            interm_depths = lr_depths.unsqueeze(2)
            interm_depths_norm = None
        else:
            # range prediction is performed on spatially expanded features but still lr depths are scattered into predicted intermediate depths
            rh_pred = self.range_head(x_rv)  # normalized intermediate depth
            interm_depths_norm = torch.sigmoid(rh_pred[:, 0:1, :, :])  # [B, 1, Hrv, Wrv]
            interm_depths_sigma = F.relu(rh_pred[:, 1:2, :, :])  # [B, 1, Hrv, Wrv]
            
            if target_depths is not None:
                interm_depths = (interm_depths_norm) * (1 - gt_mixture_weight) + target_depths.to(interm_depths_norm.dtype) * gt_mixture_weight
            else:
                interm_depths = interm_depths_norm.clone()
            interm_depths[:, :, self.low_res_index, :] = lr_depths.to(interm_depths.dtype) # [B, 1, Hrv, Wrv]
            sampled_depths = interm_depths + interm_depths_sigma * self.offset_mult.view(1, 3, 1, 1)  # [B, 3, Hrv, Wrv]
            sampled_depths = torch.clamp(sampled_depths, 0.0, 1.0)
            sampled_depths *= self.rmax
            # interm_depths = interm_depths.unsqueeze(1)

        # [B, 3, Hrv, Wrv] * [1, 1, Hrv, Wrv] -> [B, 3, Hrv, Wrv]
        x_l = (sampled_depths * self.u_vec_x).flatten(1) # [B, n_total_q * Hrv * Wrv] 
        y_l = (sampled_depths * self.u_vec_y).flatten(1) # [B, n_total_q * Hrv * Wrv] 
        z_l = (sampled_depths * self.u_vec_z).flatten(1) # [B, n_total_q * Hrv * Wrv] 
        
        p_lidar_h = torch.stack([x_l, y_l, z_l], dim=-1) # [B, n_total_q * Hrv * Wrv, 3]
        
        return p_lidar_h.detach(), (interm_depths_norm, sampled_depths)
        
        
    def forward(self, x_rv, tf_matrix=None, ret_feats=False, lr_depths=None, target_depths=None, gt_mixture_weight = 0.0):
        """
        Generate query points in desired frame. One per each pixel in output space.
        x_rv: [B, Crv, H_lrv, W_lrv]
        """ 

        B, Crv, H_lrv, W_lrv = x_rv.shape # range view latent size
        assert H_lrv == self.in_Hrv and W_lrv == self.in_Wrv, f"Input range view latent size {H_lrv}x{W_lrv} does not match expected size {self.in_Hrv}x{self.in_Wrv}"
        assert Crv == self.C_rv, f"in C {Crv} != expected C {self.C_rv}"
        
        x_ret = x_rv.clone() if ret_feats else None
        if self.num_q_per_latent_cell >1:
            if self.only_low_res:
                # doing this in 2 steps bc cant use PixelShuffle for asymmetric upsampling, 
                # still could create a custom nn.module class for rearrange to be wrapped by nn.sequential
                x_rv = self.populate_channels(x_rv)
                x_rv = rearrange(x_rv, 'B (P1 P2 C) H W -> B C (H P1) (W P2)', P1=self.n_q_h, P2=self.n_q_w)
            else:
                x_rv = self.spatial_expand(x_rv)
            x_ret = x_rv.clone() if ret_feats else None
            
        assert x_rv.shape[1] == self.C_rv, f"After expansion, in C {x_rv.shape[1]} != expected C {self.C_rv}"
        points, interm_depths = self._point_hypothesis(x_rv, lr_depths=lr_depths, target_depths=target_depths, gt_mixture_weight=gt_mixture_weight) # [B, Hrv* Wrv, 3]
        
        if tf_matrix is not None:
            ones = torch.ones_like(points[..., :1])
            points = torch.cat([points, ones], dim=-1) # [B, Hrv* Wrv, 4]

            # LiDAR -> Target frame
            points = points @ tf_matrix.mT
            points = points.view(B, self.og_Hrv, self.og_Wrv, 4)
            
        return points, interm_depths, x_ret

if __name__ == "__main__":
    # Minimal test snippet
    
    qg = SimpleQueryGenerator(rmax=51.2, C_rv=384, in_rv_size=(2,64), og_rv_size=(32,1024))
    
    assert qg.az_per_q.shape == (2, 64, 16)
    assert qg.elev_per_q.shape == (2, 64, 16)
    
    del qg
    
    qg = SimpleQueryGenerator(rmax=51.2, C_rv=384, in_rv_size=(32,1024), og_rv_size=(32,1024))
    
    assert qg.az_per_q.shape == (32, 1024, 1)
    assert qg.elev_per_q.shape == (32, 1024, 1)
    
    del qg
    
    qg = SimpleQueryGenerator(rmax=51.2, C_rv=384, in_rv_size=(16,512), og_rv_size=(32,1024))
    
    assert qg.az_per_q.shape == (16, 512, 2)
    assert qg.elev_per_q.shape == (16, 512, 2)
    
    in_t = torch.randn(2, 384, 16, 512)

    pts, interm_depths, q_features = qg(in_t, ret_feats=True)

    assert pts.shape == (2, 32*1024, 3)
    assert interm_depths.shape == (2, 1, 32, 1024)
    assert q_features.shape == (2, 384, 32, 1024)
    print("All tests passed!")