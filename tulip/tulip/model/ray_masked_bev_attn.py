import sys
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

# Add the tulip directory to the path so we can import from util
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util.rv_utils import compute_polar_ang_per_pixel

class RayMaskedBEVAttention(nn.Module):
    """
    Run attention from range view tokens to BEV features. Maske bev cells that are not within the fov of the range view tokens.
    """
    def __init__(self, bev_extent=(-51.2,51.2,-51.2,51.2), bev_cell_reso=0.8, bev_ch=80, 
                 spatial_rv_shape=(2,64), rv_ch=384):
        super().__init__()
        attention_dim = 128
        self.bev_extent = bev_extent
        self.grid_m = bev_cell_reso
        self.bev_ch = bev_ch
        self.rv_ch = rv_ch
        self.out_ch = rv_ch
        self.rv_size = spatial_rv_shape
        self.Hr, self.Wr = self.rv_size
        self.Nr = self.Hr * self.Wr

        bev_size_x = (bev_extent[1] - bev_extent[0]) / bev_cell_reso
        bev_size_y = (bev_extent[3] - bev_extent[2]) / bev_cell_reso
        assert bev_size_x == int(bev_size_x) and bev_size_y == int(bev_size_y), "BEV extent must be divisible by cell resolution"
        self.bev_size = (int(bev_size_y), int(bev_size_x)) # (H, W)
        self.Hb, self.Wb = self.bev_size
        self.Nb = self.Hb * self.Wb
        
        # --- Define projection and attention layers ---
        self.q_proj = nn.Linear(rv_ch, attention_dim)
        self.k_proj = nn.Linear(bev_ch, attention_dim)
        self.v_proj = nn.Linear(bev_ch, attention_dim)
        self.out_proj = nn.Linear(attention_dim, rv_ch)
        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)
        
        self.cache_built = False
        
        
    @torch.no_grad()
    def _build_cache(self, lidar2ego_tf_mat):
        device = lidar2ego_tf_mat.device
        Hb, Wb = self.bev_size
        Hr, Wr = self.rv_size

        az, elev, az_step, elev_bin_bounds = compute_polar_ang_per_pixel(self.rv_size)
        elev_centers = torch.from_numpy(elev).mean(dim=1).float().to(device)

        xmin, xmax, ymin, ymax = self.bev_extent
        xs = torch.linspace(xmin + self.grid_m/2, xmax - self.grid_m/2, int(Wb)).to(device)
        ys = torch.linspace(ymin + self.grid_m/2, ymax - self.grid_m/2, int(Hb)).to(device)

        # --- BEV cell centers in ego frame (z=0 plane) ---
        X, Y = torch.meshgrid(xs, ys, indexing='xy')         # X:[Wb,Hb], Y:[Wb,Hb]
        X = X.T.contiguous()                                  # -> [Hb,Wb]
        Y = Y.T.contiguous()
        Z = torch.zeros_like(X)                               # ground plane z=0 (ego)
        Nb = int(Hb * Wb)
        P_ego = torch.stack([X, Y, Z, torch.ones_like(X)], dim=-1).view(Nb, 4)  # [Nb,4]

        # --- Ego -> LiDAR frame ---
        # lidar2ego maps LiDAR -> Ego; we need its inverse to map Ego -> LiDAR.
        if lidar2ego_tf_mat.ndim == 3:   # [B,4,4] → assume same for all in batch; take first
            lidar2ego_tf_mat = lidar2ego_tf_mat[0]
        ego2lidar = torch.inverse(lidar2ego_tf_mat)          # [4,4]

        # Column-vector convention: p_lidar = ego2lidar @ p_ego  (we have row vectors → right-multiply by T)
        P_lidar = (P_ego @ ego2lidar.T)                      # [Nb,4]
        Xl, Yl, Zl = P_lidar[:,0], P_lidar[:,1], P_lidar[:,2]

        theta_bev_lidar   = torch.atan2(Yl, Xl)                     # [-pi, pi]
        rho        = torch.sqrt(Xl*Xl + Yl*Yl).clamp_min(1e-6)
        elev_bev_lidar = torch.atan2(Zl, rho)                    # elevation
        r_bev_ego = torch.sqrt(X*X + Y*Y).view(-1)               # [Nb], ego-plane radius

        diffs = torch.abs(elev_bev_lidar[:,None] - elev_centers[None,:])  # [Nb,Hr]
        ring_ids = torch.argmin(diffs, dim=1)

        # If you also need BEV→RV column assignment (angles-only):
        theta_cols = (((theta_bev_lidar + math.pi) / (2*math.pi)) * Wr).long().clamp(0, Wr-1)  # [Nb]
        bev_by_col  = [torch.nonzero(theta_cols == c, as_tuple=False).flatten() for c in range(Wr)]
        bev_by_ring = [torch.nonzero(ring_ids   == r, as_tuple=False).flatten() for r in range(Hr)]

        # For faster intersection later, also build boolean masks (optional optimization)
        col_mask  = torch.zeros(Wr, Nb, dtype=torch.bool)
        ring_mask = torch.zeros(Hr, Nb, dtype=torch.bool)
        for c in range(Wr):
            if bev_by_col[c].numel() > 0:
                col_mask[c, bev_by_col[c]] = True
        for r in range(Hr):
            if bev_by_ring[r].numel() > 0:
                ring_mask[r, bev_by_ring[r]] = True

        # Build candidate list per RV token (ring,col)
        cand_lists = []
        for r in range(Hr):
            # precompute ring-band mask
            r_ids = r % Hr
            ring_band_mask = ring_mask[r_ids]                      # [Nb] bool
            for c in range(Wr):
                c_ids = c % Wr
                az_band_mask = col_mask[c_ids]                      # [Nb] bool
                # Intersect both ring and azimuth masks
                keep = (ring_band_mask & az_band_mask).nonzero(as_tuple=False).flatten()  # [M?]
                if keep.numel() == 0:
                    cand_lists.append(keep)
                    continue
                # sort by ego radius (near→far), then subsample
                r_vals = r_bev_ego[keep]
                order  = torch.argsort(r_vals)
                keep   = keep[order]
                cand_lists.append(keep)

        # --- Pad to rectangular [Nr, M] ---
        Nr = Hr * Wr
        M  = max((ci.numel() for ci in cand_lists), default=1)
        cand_mat = torch.full((Nr, M), -1, dtype=torch.long)
        for i, ci in enumerate(cand_lists):
            if ci.numel() > 0:
                m = min(M, ci.numel())
                cand_mat[i, :m] = ci[:m]
        pad_mask = (cand_mat < 0)[None, None, :, :]  # [1,1,Nr,M]
        
        return cand_mat, pad_mask
    
    def _flatten_rv(self, x):
        # x: [B, Hr, Wr, C_rv] -> [B, Nr, C_rv]
        B, Hr, Wr, C = x.shape
        assert Hr == self.Hr and Wr == self.Wr, "RV spatial size mismatch."
        return x.reshape(B, self.Nr, C)

    def _flatten_bev(self, bev_feat):
        # bev_feat: [B, C_bev, Hb, Wb] -> [B, Nb, C_bev]
        B, Cb, Hb, Wb = bev_feat.shape
        assert Hb == self.Hb and Wb == self.Wb, "BEV spatial size mismatch."
        Nb = Hb * Wb
        return bev_feat.permute(0, 2, 3, 1).reshape(B, Nb, Cb)

    @staticmethod
    def _batched_gather_bev(feats_bnC, idx_NrM, fill_value=0.0):
        """
        feats_bnC: [B, Nb, C]  (BEV flattened)
        idx_NrM : [Nr, M] (indices into Nb; -1 indicates padding)
        returns:  [B, Nr, M, C]
        """
        B, Nb, C = feats_bnC.shape
        Nr, M = idx_NrM.shape

        idx_safe = idx_NrM.clamp_min(0)                         # [-1] -> [0] for gather
        idx_flat = idx_safe.reshape(1, Nr * M, 1).expand(B, -1, C)   # [B, Nr*M, C]
        # gather along dim=1 (Nb axis)
        gathered = feats_bnC.gather(1, idx_flat)                     # [B, Nr*M, C]
        gathered = gathered.reshape(B, Nr, M, C)

        if (idx_NrM < 0).any():
            # zero out padded positions explicitly (avoid leaking BEV[0])
            mask = (idx_NrM < 0).view(1, Nr, M, 1).expand(B, -1, -1, C)
            if fill_value == 0.0:
                gathered = gathered.masked_fill(mask, 0.0)
            else:
                gathered = torch.where(mask, torch.as_tensor(fill_value, dtype=gathered.dtype, device=gathered.device), gathered)
        return gathered  # [B, Nr, M, C]

    def forward(self, rv_x, bev_feat, lidar2ego_tf_mat):
        """
        rv_x:    [B, rv_ch, Hr, Wr]     (range-view latent tokens)
        bev_feat:[B, bev_ch, Hb, Wb]    (BEV feature map)
        returns:
          rv_out:        [B, out_ch, Hr, Wr]  (RV tokens enriched by local BEV context)
          attn_weights:  [B, Nr, M]           (softmax weights per RV token over its BEV candidates)
        """
        if not self.cache_built:
            cand_mat, pad_mask = self._build_cache(lidar2ego_tf_mat)
            self.M = cand_mat.shape[1]
            # store as regular attributes to avoid DDP buffer sync requirements
            self._cand_mat = cand_mat.contiguous()
            self._pad_mask = pad_mask.contiguous()
            self.cache_built = True
        B = rv_x.shape[0]
        device = rv_x.device
        # Ensure cached tensors on correct device/dtype
        cand_mat = self._cand_mat.to(device=device)
        pad_mask = self._pad_mask.to(device=device)  # [1,1,Nr,M] bool

        # 1) flatten and project queries
        rv_flat = self._flatten_rv(rv_x)                           # [B, Nr, C_rv]
        Q = self.q_proj(rv_flat)                                   # [B, Nr, d]
        d = Q.shape[-1]
        Q = F.normalize(Q, dim=-1)                                 # stabilizes dot products

        # 2) gather candidate BEV features per RV token and project K,V
        bev_flat = self._flatten_bev(bev_feat)                     # [B, Nb, C_bev]
        bev_sel = self._batched_gather_bev(bev_flat, cand_mat)     # [B, Nr, M, C_bev]
        K = self.k_proj(bev_sel)                                   # [B, Nr, M, d]
        V = self.v_proj(bev_sel)                                   # [B, Nr, M, d]
        K = F.normalize(K, dim=-1)

        # 3) scaled dot-product attention with padding mask
        # logits_{b,nr,m} = <Q_{b,nr,:}, K_{b,nr,m,:}> / sqrt(d)
        logits = torch.einsum('bnd,bnmd->bnm', Q, K) / math.sqrt(d)  # [B, Nr, M]
        if pad_mask is not None and pad_mask.numel() > 0:
            # pad_mask: [1,1,Nr,M] -> [B,Nr,M]
            m = pad_mask.view(1, 1, self.Nr, self.M).expand(B, 1, -1, -1).squeeze(1)
            logits = logits.masked_fill(m, float('-inf'))

        attn = torch.softmax(logits, dim=-1)                       # [B, Nr, M]
        attn = self.attn_drop(attn)

        # 4) weighted sum of values: out_{b,nr,:} = sum_m attn * V
        out = torch.einsum('bnm,bnmd->bnd', attn, V)               # [B, Nr, d]
        out = self.out_proj(out)                                   # [B, Nr, out_ch]
        out = self.proj_drop(out)

        # 5) reshape back to RV map
        out = out.view(B, self.Hr, self.Wr, self.out_ch).contiguous()  # [B,out_ch,Hr,Wr]
        return out, attn



if __name__ == "__main__":
    # --- Example Configuration ---
    BATCH_SIZE = 2
    BEV_EXTENT = (-51.2, 51.2, -51.2, 51.2)
    BEV_CELL_RESO = 0.8
    BEV_CH = 80
    SPATIAL_RV_SHAPE = (2, 64)
    RV_CH = 384

    # --- Create Model and Dummy Inputs ---
    # Identity matrix means LiDAR and Ego frames are the same
    lidar2ego_tf_mat = torch.eye(4).float().cuda()
    
    print("Initializing RayMaskedBEVAttention module...")
    model = RayMaskedBEVAttention(
        bev_extent=BEV_EXTENT,
        bev_cell_reso=BEV_CELL_RESO,
        bev_ch=BEV_CH,
        spatial_rv_shape=SPATIAL_RV_SHAPE,
        rv_ch=RV_CH,
    ).cuda()
    print("Initialization complete.\n")

    dummy_rv_x = torch.randn(BATCH_SIZE, RV_CH, SPATIAL_RV_SHAPE[0], SPATIAL_RV_SHAPE[1]).cuda()
    # BEV shape: H=128, W=128 (calculated from extent and resolution)
    dummy_bev_feat = torch.randn(BATCH_SIZE, BEV_CH, 128, 128).cuda()

    print(f"Input RV shape:    {dummy_rv_x.shape}")
    print(f"Input BEV shape:   {dummy_bev_feat.shape}")

    # --- Run Forward Pass ---
    enriched_rv, attn_weights = model(dummy_rv_x, dummy_bev_feat, lidar2ego_tf_mat)

    print(f"\nOutput RV shape:   {enriched_rv.shape}")
    print(f"Attention weights shape: {attn_weights.shape} (Batch, Num_RV_Tokens, Max_BEV_Candidates)")
    
    assert enriched_rv.shape == dummy_rv_x.shape
    print("\nShape verification successful!")
