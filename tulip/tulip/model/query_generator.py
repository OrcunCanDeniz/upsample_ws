import math
from typing import Optional, Tuple
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F

# Handle both relative import (when used as module) and absolute import (when run as script)
try:
    from .coord_conv import CoordConv2d
except ImportError:
    # When running as script, import directly from current directory
    import sys
    from pathlib import Path
    # Add current directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    from coord_conv import CoordConv2d

import pdb

ELEV_DEG_PER_RING_NUCSENES = torch.tensor([-30.67, -29.33, -28., -26.66, -25.33, -24., -22.67, -21.33,
                               -20., -18.67, -17.33, -16., -14.67, -13.33, -12., -10.67,
                                -9.33, -8., -6.66, -5.33, -4., -2.67, -1.33, 0.,
                                1.33, 2.67, 4., 5.33, 6.67, 8., 9.33, 10.67], dtype=torch.float32)


class SimpleQueryGenerator(nn.Module):
    """
    Given feature map from range view, generate nqw*nqh number of 3D query points in ego frame.
    
    Args:
        rmax: maximum range in meters used to normalize depths
        C_rv: number of channels in range view feature map
        in_rv_size: size of range view feature map to be processed
        og_rv_size: original size of range view image
        only_low_res: If True, only low resolution gt depths will be used to generate query vectors, for each vector gt gepth will be used to generate 3D query points . 
                    If False, query token generation done in the target(upsampled) resolution, then a range prediction head is used to generate depths that will be used to generate 3D query points
    """
    def __init__(self,
                 rmax,
                 C_rv=384,
                 in_rv_size=(2,64),
                 only_low_res=False,
                 dataset_name='nuscenes'):
        super().__init__()
        
        self.C_rv = C_rv
        self.rmax = rmax     
        self.only_low_res = only_low_res
        if dataset_name == 'nuscenes':
            self.lr_Hrv = 8
            self.lr_Wrv = 1024
            og_rv_size = (32, 1024)
        elif dataset_name == 'kitti':
            self.lr_Hrv = 16
            self.lr_Wrv = 1024
            og_rv_size = (64, 1024)
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        self.dataset_name = dataset_name
        
        # Build azimuth and zenith maps by in_rv_size (width spans [-pi, pi))
        self.og_Hrv, self.og_Wrv = int(og_rv_size[0]), int(og_rv_size[1])
        self.in_Hrv, self.in_Wrv = int(in_rv_size[0]), int(in_rv_size[1])
        self.in_spatial_expand_f_h = (self.lr_Hrv // self.in_Hrv) if only_low_res else (self.og_Hrv // self.in_Hrv)
        self.in_spatial_expand_f_w = (self.lr_Wrv // self.in_Wrv) if only_low_res else (self.og_Wrv // self.in_Wrv)
        
        self.n_q_w = int(self.in_spatial_expand_f_w)
        self.n_q_h = int(self.in_spatial_expand_f_h)
        assert self.n_q_w == self.in_spatial_expand_f_w, "ogW / inW must be int" 
        assert self.n_q_h == self.in_spatial_expand_f_h, "ogH / inH must be int" 
        
        self.num_q_per_latent_cell = self.n_q_w * self.n_q_h

        if not only_low_res:
            # 1D range proposal head for processing expanded features
            self.range_head = nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=C_rv),
                CoordConv2d(C_rv, 64, kernel_size=3, padding=1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1, bias=False)
            )

            # Spatial expansion (PixelShuffle) path requires symmetrical upsampling
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
        # Always build geometry without binning at the original range-view resolution.
        # Any lower-resolution variants should be obtained by subsampling this table.
        self.set_geometry()
        self.low_res_index = range(0, og_rv_size[0], 4) # which indices were used to create low res rv
        
        if only_low_res:
            self.u_vec_x = self.u_vec_x[:, :, :, self.low_res_index, :]
            self.u_vec_y = self.u_vec_y[:, :, :, self.low_res_index, :]
            self.u_vec_z = self.u_vec_z[:, :, :, self.low_res_index, :]
            
            # Only create populate_channels when spatial expansion is needed
            if self.num_q_per_latent_cell > 1:
                num_addit_ch = self.C_rv * self.num_q_per_latent_cell
                self.populate_channels = nn.Sequential(
                    nn.Conv2d(in_channels=C_rv, out_channels=num_addit_ch, kernel_size=(1, 1)),
                    nn.BatchNorm2d(num_addit_ch),
                    nn.GELU(),
                    nn.Conv2d(in_channels=num_addit_ch, out_channels=num_addit_ch, kernel_size=(1, 1)),
                )
       
    def set_geometry(self):
        """
        Build per-pixel unit direction vectors on the *original* range-view grid
        without any angular binning. This produces a canonical table that can
        be safely subsampled for lower-resolution variants.
        """
        # Sanity check: total number of latent samples should not exceed original grid
        assert self.n_q_w * self.in_Wrv <= self.og_Wrv, "Total number  of horizontal samples must be less than or equal to original width"
        az_line = torch.linspace(-math.pi, math.pi, self.og_Wrv + 1)[:-1]
        az_line += (az_line[1] - az_line[0])/2
        # (og_Hrv, og_Wrv, 1)
        self.az_per_q = az_line[None, :, None].repeat(self.og_Hrv, 1, 1)

        assert self.n_q_h * self.in_Hrv <= self.og_Hrv, "Total number of vertical samples must be less than or equal to original height"
        if self.dataset_name == 'nuscenes':
            elev = torch.deg2rad(ELEV_DEG_PER_RING_NUCSENES.flip(0))
        elif self.dataset_name == 'kitti':
            ang_start_y = 24.8
            ang_res_y = 26.8 / (self.og_Hrv -1)
            elev_arr = torch.arange(self.og_Hrv)*ang_res_y - ang_start_y
            elev = torch.deg2rad(elev_arr) # descending order
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset_name}")

        # (og_Hrv, og_Wrv, 1)
        self.elev_per_q = elev[:, None, None].repeat(1, self.og_Wrv, 1)
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
        
        u_vec_x = u_vec_x.unsqueeze(0).contiguous()
        u_vec_y = u_vec_y.unsqueeze(0).contiguous()
        u_vec_z = u_vec_z.unsqueeze(0).contiguous()
        self.register_buffer("u_vec_x", u_vec_x, persistent=False)
        self.register_buffer("u_vec_y", u_vec_y, persistent=False)
        self.register_buffer("u_vec_z", u_vec_z, persistent=False)

    def _point_hypothesis(self, x_rv, lr_depths=None, target_depths=None, gt_mixture_weight = 0.0):
        """
        Generate points in lidar frame by predicting intermediate normalized depth and unit vectors
        This will always get rv with original size.
        Use forward() since it also handles spatial expanding if needed.
        
        x_rv: [B, Crv, Hrv, Wrv]
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
            interm_depths_norm = self.range_head(x_rv).sigmoid()  # normalized intermediate depth
            if target_depths is not None:
                interm_depths = (interm_depths_norm) * (1 - gt_mixture_weight) + target_depths.to(interm_depths_norm.dtype) * gt_mixture_weight
            else:
                interm_depths = interm_depths_norm.clone().float()
            interm_depths[:, :, self.low_res_index, :] = lr_depths.to(interm_depths.dtype)
            interm_depths *= float(self.rmax)
            interm_depths = interm_depths.unsqueeze(1)

        x_l = (interm_depths * self.u_vec_x).float().flatten(1) # [B, n_total_q * Hrv * Wrv] 
        y_l = (interm_depths * self.u_vec_y).float().flatten(1) # [B, n_total_q * Hrv * Wrv] 
        z_l = (interm_depths * self.u_vec_z).float().flatten(1) # [B, n_total_q * Hrv * Wrv] 
        
        p_lidar_h = torch.stack([x_l, y_l, z_l], dim=-1)
        
        return p_lidar_h.detach(), interm_depths_norm
        
        
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
    print("Testing only_low_res=False (default)...")
    
    # Test 1: only_low_res=False with spatial expansion
    qg = SimpleQueryGenerator(rmax=51.2, C_rv=384, in_rv_size=(2,64), og_rv_size=(32,1024), only_low_res=False)
    assert qg.az_per_q.shape == (32, 1024, 1)
    assert qg.elev_per_q.shape == (32, 1024, 1)
    assert hasattr(qg, 'range_head'), "range_head should exist for only_low_res=False"
    assert qg.u_vec_x.shape == (1, 1, 1, 32, 1024), "u_vec_x should have full resolution"
    del qg
    
    # Test 2: only_low_res=False without spatial expansion (num_q_per_latent_cell == 1)
    qg = SimpleQueryGenerator(rmax=51.2, C_rv=384, in_rv_size=(32,1024), og_rv_size=(32,1024), only_low_res=False)
    assert qg.az_per_q.shape == (32, 1024, 1)
    assert qg.elev_per_q.shape == (32, 1024, 1)
    assert qg.num_q_per_latent_cell == 1, "No expansion expected"
    assert not hasattr(qg, 'spatial_expand'), "spatial_expand should not exist when num_q_per_latent_cell == 1"
    del qg
    
    # Test 3: only_low_res=False with 2x2 spatial expansion
    qg = SimpleQueryGenerator(rmax=51.2, C_rv=384, in_rv_size=(16,512), og_rv_size=(32,1024), only_low_res=False)
    assert qg.az_per_q.shape == (32, 1024, 1)
    assert qg.elev_per_q.shape == (32, 1024, 1)
    assert qg.num_q_per_latent_cell == 4, "Should have 2x2 = 4 queries per cell"
    assert hasattr(qg, 'spatial_expand'), "spatial_expand should exist"
    
    in_t = torch.randn(2, 384, 16, 512)
    lr_depths = torch.rand(2, 1, 8, 1024)  # low-res depths: 8 rows (every 4th row of 32)
    pts, interm_depths, q_features = qg(in_t, ret_feats=True, lr_depths=lr_depths)
    
    assert pts.shape == (2, 32*1024, 3), f"Expected (2, 32768, 3), got {pts.shape}"
    assert interm_depths is not None, "interm_depths should not be None for only_low_res=False"
    assert interm_depths.shape == (2, 1, 32, 1024), f"Expected (2, 1, 32, 1024), got {interm_depths.shape}"
    assert q_features.shape == (2, 384, 32, 1024), f"Expected (2, 384, 32, 1024), got {q_features.shape}"
    del qg
    
    print("  ✓ only_low_res=False tests passed")
    
    print("\nTesting only_low_res=True...")
    
    # Test 4: only_low_res=True with spatial expansion (asymmetric allowed)
    qg = SimpleQueryGenerator(rmax=51.2, C_rv=384, in_rv_size=(2,64), og_rv_size=(32,1024), only_low_res=True)
    assert qg.az_per_q.shape == (32, 1024, 1), "Geometry should be built at full resolution"
    assert qg.elev_per_q.shape == (32, 1024, 1)
    assert not hasattr(qg, 'range_head'), "range_head should not exist for only_low_res=True"
    assert not hasattr(qg, 'spatial_expand'), "spatial_expand should not exist for only_low_res=True"
    assert hasattr(qg, 'populate_channels'), "populate_channels should exist for only_low_res=True"
    # After subsampling, u_vec should have low-res height
    lr_Hrv = len(qg.low_res_index)  # Should be 8 for nusc (0, 4, 8, ..., 28)
    assert lr_Hrv == 8, f"Expected lr_Hrv to be 8, got {lr_Hrv}"
    assert qg.u_vec_x.shape == (1, 1, 1, lr_Hrv, 1024), f"Expected (1, 1, 1, {lr_Hrv}, 1024), got {qg.u_vec_x.shape}"
    assert qg.n_q_h == 4 and qg.n_q_w == 16, "Asymmetric expansion should be allowed"
    assert qg.num_q_per_latent_cell == 64, "Should have 4*16 = 64 queries per cell"
    
    in_t = torch.randn(2, 384, 2, 64)
    lr_depths = torch.rand(2, 1, lr_Hrv, 1024)  # low-res depths matching subsampled geometry
    pts, interm_depths, q_features = qg(in_t, ret_feats=True, lr_depths=lr_depths)
    
    assert pts.shape == (2, lr_Hrv*1024, 3), f"Expected (2, {lr_Hrv*1024}, 3), got {pts.shape}"
    assert interm_depths is None, "interm_depths should be None for only_low_res=True"
    assert q_features.shape == (2, 384, lr_Hrv, 1024), f"Expected (2, 384, {lr_Hrv}, 1024), got {q_features.shape}"
    del qg
    
    # Test 5: only_low_res=True without spatial expansion
    qg = SimpleQueryGenerator(rmax=51.2, C_rv=384, in_rv_size=(8,1024), og_rv_size=(32,1024), only_low_res=True)
    assert qg.num_q_per_latent_cell == 1, "No expansion expected (8*1 = 8, matching lr_Hrv)"
    assert not hasattr(qg, 'populate_channels'), "populate_channels should not exist when num_q_per_latent_cell == 1"
    lr_Hrv = len(qg.low_res_index)
    assert qg.u_vec_x.shape == (1, 1, 1, lr_Hrv, 1024)
    
    in_t = torch.randn(2, 384, 8, 1024)
    lr_depths = torch.rand(2, 1, lr_Hrv, 1024)
    pts, interm_depths, q_features = qg(in_t, ret_feats=True, lr_depths=lr_depths)
    
    assert pts.shape == (2, lr_Hrv*1024, 3)
    assert interm_depths is None
    assert q_features.shape == (2, 384, lr_Hrv, 1024)
    del qg
    
    print("  ✓ only_low_res=True tests passed")
    
    print("\nAll tests passed!")