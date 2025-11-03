import sys
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def invert_tf_mat(tf_mat):
    R = tf_mat[:3, :3]
    t = tf_mat[:3, 3]
    inv_R = R.T
    inv_t = -inv_R @ t
    inv_tf_mat = torch.eye(4, device=tf_mat.device, dtype=tf_mat.dtype)
    inv_tf_mat[:3, :3] = inv_R
    inv_tf_mat[:3, 3] = inv_t
    return inv_tf_mat

# Add the tulip directory to the path so we can import from util
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util.rv_utils import compute_polar_ang_per_pixel, ELEV_DEG_PER_RING_NUCSENES_RAD

class MaskedBEVTransformer(nn.Module):
    def __init__(self, bev_extent=(-51.2,51.2,-51.2,51.2), bev_cell_reso=0.8, bev_ch=80, 
                 spatial_rv_shape=(2,64), rv_ch=384, attention_dim=128, num_layers=1,
                 num_heads=4, drop=0.1, mlp_ratio=4.0):
        super().__init__()
        self.rv_ch = rv_ch
        self.bev_extent = bev_extent
        self.grid_m = bev_cell_reso
        self.rv_size = spatial_rv_shape
        self.Hr, self.Wr = self.rv_size
        self.Nr = self.Hr * self.Wr

        bev_size_x = (bev_extent[1] - bev_extent[0]) / bev_cell_reso
        bev_size_y = (bev_extent[3] - bev_extent[2]) / bev_cell_reso
        assert bev_size_x == int(bev_size_x) and bev_size_y == int(bev_size_y), "BEV extent must be divisible by cell resolution"
        self.bev_size = (int(bev_size_y), int(bev_size_x)) # (H, W)
        self.Hb, self.Wb = self.bev_size
        self.Nb = self.Hb * self.Wb
        
        # Positional encoding for BEV features (fixed 2D sinusoidal)
        assert bev_ch % 2 == 0, "bev_ch must be even for 2D sinusoidal positional encoding"
        self.register_buffer('pos_bev', self._build_2d_sincos_pos(self.Hb, self.Wb, bev_ch), persistent=False)

        self.q_norm = nn.LayerNorm(rv_ch)
        self.q_proj = nn.Linear(rv_ch, attention_dim)
        self.k_proj = nn.Linear(bev_ch, attention_dim)
        self.v_proj = nn.Linear(bev_ch, attention_dim)
        
        self.attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(MaskedBEVAttention(spatial_rv_shape=spatial_rv_shape, rv_ch=rv_ch, 
                                                       attention_dim=attention_dim, num_heads=num_heads, 
                                                       drop=drop, mlp_ratio=mlp_ratio))
        
        self.out_proj = nn.Linear(attention_dim, rv_ch)
        
        self.cache_built = False
        
        
    def _build_2d_sincos_pos(self, H, W, C):
        def get_1d_pos(d, length):
            pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
            i = torch.arange(d, dtype=torch.float32).unsqueeze(0)
            div = torch.exp(-math.log(10000.0) * (2 * (i//2)) / d)
            angle = pos * div
            sin = torch.sin(angle)
            cos = torch.cos(angle)
            out = torch.zeros(length, d)
            out[:, 0::2] = sin[:, 0::2]
            out[:, 1::2] = cos[:, 1::2]
            return out  # [length, d]

        d_half = C // 2
        px = get_1d_pos(d_half, W)  # [W, d_half]
        py = get_1d_pos(d_half, H)  # [H, d_half]
        # Combine by broadcasting to grid then concat on channel
        px_grid = px.unsqueeze(0).expand(H, -1, -1)          # [H, W, d_half]
        py_grid = py.unsqueeze(1).expand(-1, W, -1)          # [H, W, d_half]
        pos = torch.cat([py_grid, px_grid], dim=-1)          # [H, W, C]
        # to conv-friendly [1,C,H,W]
        pos = pos.permute(2, 0, 1).unsqueeze(0)              # [1, C, H, W]
        return pos

    @torch.no_grad()
    def _build_cache(self, lidar2ego_tf_mat):
        device = lidar2ego_tf_mat.device
        Hb, Wb = self.bev_size
        Hr, Wr = self.rv_size

        az, elev, az_step, elev_bin_bounds = compute_polar_ang_per_pixel(self.rv_size)
        elev_centers = torch.from_numpy(elev).mean(dim=1).float().to(device)
        t_elevs = torch.from_numpy(ELEV_DEG_PER_RING_NUCSENES_RAD).flip(0).float().to(device)
        elev_min = t_elevs.min()
        elev_split = torch.split(t_elevs, 32//self.Hr)
        elev_max = torch.stack([elev.max() for elev in elev_split])

        xmin, xmax, ymin, ymax = self.bev_extent
        xs = torch.linspace(xmin + self.grid_m/2, xmax - self.grid_m/2, int(Wb)).to(device)
        ys = torch.linspace(ymin + self.grid_m/2, ymax - self.grid_m/2, int(Hb)).to(device)

        # --- BEV cell centers in ego frame (z=0 plane) ---
        X, Y = torch.meshgrid(xs, ys, indexing='xy')         # X:[Wb,Hb], Y:[Wb,Hb]
        X = X.T.contiguous()                                  # -> [Hb,Wb]
        Y = Y.T.contiguous()
        Z = torch.zeros_like(X)                               # ground plane z=0 (ego)
        Nb = int(Hb * Wb)
        P_ego = torch.stack([X, Y, Z, torch.ones_like(X)], dim=-1) # [Hb, Wb, 4]
        # --- Ego -> LiDAR frame ---
        # lidar2ego maps LiDAR -> Ego; we need its inverse to map Ego -> LiDAR.
        if lidar2ego_tf_mat.ndim == 3:   # [B,4,4] → assume same for all in batch; take first
            lidar2ego_tf_mat = lidar2ego_tf_mat[0]
        ego2lidar = invert_tf_mat(lidar2ego_tf_mat)          # [4,4]

        # Column-vector convention: p_lidar = ego2lidar @ p_ego  (we have row vectors → right-multiply by T)
        P_lidar = (P_ego @ ego2lidar.mT)                      # [Nb,4]

        theta_bev_lidar   = torch.atan2(P_lidar[..., 0], P_lidar[..., 1])                     # [-pi, pi]
        rho        = torch.norm(P_lidar[..., 0:2], dim=-1).clamp_min(1e-6)
        elev_bev_lidar = torch.atan2(P_lidar[..., 2], rho)                    # elevation
        r_bev_ego = torch.norm(P_ego[..., 0:2], dim=-1)              # [Nb], ego-plane radius
        # which row each bev cell can interact with [BevH, BevW, 1] last dim is for target row idx, a cell can interact with multiple rows
        elev_mask_per_cell = elev_bev_lidar.unsqueeze(-1) < elev_max
        elev_inds_per_cell = elev_mask_per_cell.nonzero()

        # which column each bev cell can interact with
        theta_cols = (((theta_bev_lidar + math.pi) / (2*math.pi)) * Wr).long().clamp(0, Wr-1)  # [Hb,Wb]

        # --- Build RV->BEV candidate matrix ---
        # Flatten BEV indices in row-major order consistent with _flatten_bev
        bev_lin_idx = torch.arange(Hb * Wb, device=device, dtype=torch.long).view(Hb, Wb)

        # Collect candidate BEV indices for each RV token (row r, col c)
        rv_candidates = []  # length = Nr, each is 1D LongTensor of variable length
        for r in range(Hr):
            mask_r = elev_mask_per_cell[:, :, r]  # [Hb,Wb]
            for c in range(Wr):
                mask_c = (theta_cols == c)  # [Hb,Wb]
                mask_rc = mask_r & mask_c   # [Hb,Wb]
                idxs = bev_lin_idx[mask_rc].view(-1)
                # Guarantee at least one candidate to avoid all-masked softmax NaNs
                if idxs.numel() == 0:
                    idxs = bev_lin_idx.new_tensor([0])
                rv_candidates.append(idxs)

        Nr = Hr * Wr
        assert len(rv_candidates) == Nr
        # Determine max candidate list length (M) and pad
        M = max(int(t.numel()) for t in rv_candidates)
        cand_mat = torch.full((Nr, M), -1, device=device, dtype=torch.long)
        pad_mask = torch.ones((1, 1, Nr, M), device=device, dtype=torch.bool)  # True = padded
        for n, idxs in enumerate(rv_candidates):
            mlen = int(idxs.numel())
            cand_mat[n, :mlen] = idxs
            pad_mask[0, 0, n, :mlen] = False

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

        # Prepare static tensors: add BEV positional encoding then flatten
        bev_feat = bev_feat + self.pos_bev.to(bev_feat.dtype)      # [B,C,Hb,Wb]
        bev_flat = self._flatten_bev(bev_feat)                     # [B, Nb, C_bev]
        bev_kv_in = self._batched_gather_bev(bev_flat, cand_mat)     # [B, Nr, M, C_bev]
        
        K_all = self.k_proj(bev_kv_in)  # [B,Nr,M,C_bev]
        V_all = self.v_proj(bev_kv_in)  # [B,Nr,M,C_bev]

        rv_flat = self._flatten_rv(rv_x) # [B, Nr, C_rv]
        rv_q_in = self.q_norm(rv_flat)
        Q_rv = self.q_proj(rv_q_in) # [B, Nr, H*D]
        
        for attn_layer in self.attn_layers:
            Q_rv = attn_layer(Q_rv, K_all, V_all, pad_mask)
        
        fused_rv = self.out_proj(Q_rv)
        fused_rv = fused_rv.view(B, self.Hr, self.Wr, self.rv_ch).contiguous()
            
        return fused_rv


class MaskedBEVAttention(nn.Module):
    """
    Run attention from range view tokens to BEV features. Maske bev cells that are not within the fov of the range view tokens.
    """
    def __init__(self, spatial_rv_shape=(2,64), rv_ch=384, 
                 attention_dim=128, num_heads=4, 
                 drop=0.1, mlp_ratio=4.0):
        super().__init__()
        self.Hr, self.Wr = spatial_rv_shape
        self.Nr = self.Hr * self.Wr
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads

        # --- Define projection and attention layers (multi-head) ---
        self.attn_drop = nn.Dropout(drop)
        self.add_norm1 = nn.LayerNorm(attention_dim)
        hidden_dim = int(mlp_ratio * attention_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, attention_dim),
            nn.Dropout(drop),
        )
        

    def forward(self, Q_rv, K_bev, V_bev, pad_mask):
        B = Q_rv.shape[0]
        H = self.num_heads
        D = self.head_dim
        Nr = self.Nr
        # reshape inputs into [B,Nr,H,D] and [B,Nr,M,H,D]
        Q = Q_rv.view(B, Nr, H, D)
        M = K_bev.shape[2]
        K = K_bev.view(B, Nr, M, H, D)
        V = V_bev.view(B, Nr, M, H, D)

        # attention logits and mask
        logits = torch.einsum('bnhd,bnmhd->bnhm', Q, K) / math.sqrt(D)
        if pad_mask is not None and pad_mask.numel() > 0:
            m = pad_mask.view(1, Nr, 1, M).expand(B, Nr, 1, M)
            logits = logits.masked_fill(m, float('-inf'))
        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_drop(attn)

        # aggregate
        out = torch.einsum('bnhm,bnmhd->bnhd', attn, V)            # [B,Nr,H,D]
        out_flat = out.reshape(B, Nr, H * D)
        Q_flat = Q.reshape(B, Nr, H * D)
        out = self.add_norm1(Q_flat + out_flat)
        fused_rv = out + self.mlp(out)
       
        return fused_rv


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
