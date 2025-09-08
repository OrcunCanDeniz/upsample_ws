import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        # Shared MLP implemented with 1x1 convs to preserve (B,C,1,1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        avg_pool = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)  # (B, C, 1, 1)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        attn = self.sig(attn)  # (B, C, 1, 1)
        return x * attn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        avg_map = torch.mean(x, dim=1, keepdim=True)          # (B,1,H,W)
        max_map, _ = torch.max(x, dim=1, keepdim=True)        # (B,1,H,W)
        m = torch.cat([avg_map, max_map], dim=1)              # (B,2,H,W)
        m = self.conv(m)                                      # (B,1,H,W)
        m = self.sig(m)
        return x * m

class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7, residual: bool = True):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)
        self.residual = residual

    def forward(self, x):
        out = self.ca(x)
        out = self.sa(out)
        return x + out if self.residual else out