import math
import torch
import torch.nn as nn


class PointCoordEncoder(nn.Module):
    """
    3D point positional encoder as in the figure.
    Inputs:  points [B, N, 3]  (x, y, z)
    Output:  features [B, N, C]

    Sine() maps 1D -> 2L dims via [sin(2^k pi t), cos(2^k pi t)], k=0..L-1.
    We pick L = C // 4 so that concat(x,y,z) gives 3C/2 dims, then MLP -> C.
    """
    def __init__(
        self,
        out_dim: int,                 # C
        num_frequencies: int = None,  # L; if None, inferred from out_dim
        hidden_dim: int = None,       # first MLP width; default = out_dim
        freq_base: float = 2.0,       # 2^k growth
        use_pi: bool = True,           # multiply by pi as in NeRF
        dim: int = 3
    ):
        super().__init__()
        if num_frequencies is None:
            if out_dim % 4 != 0:
                raise ValueError("out_dim must be divisible by 4 if num_frequencies is not given.")
            num_frequencies = out_dim // 4  # so 6L = 3C/2
        self.C = out_dim
        self.L = num_frequencies
        self.use_pi = use_pi
        self.freq_base = freq_base
        self.dim = dim
        
        if self.dim == 3:
            in_dim = 6 * self.L  # concat of x,y,z encodings (each 2L)
        elif self.dim == 2:
            in_dim = 4 * self.L  # concat of x,y encodings (each 2L)
        elif self.dim == 1:
            in_dim = 2 * self.L  # x encoding (2L)
        else:
            raise ValueError(f"Unsupported dim: {self.dim}")
        
        h = out_dim if hidden_dim is None else hidden_dim

        self.mlp = nn.Sequential(
            nn.Conv2d(in_dim, h, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, out_dim, 1, bias=True),
        )

        # Precompute frequency vector (registered as buffer for device moves)
        freqs = (freq_base ** torch.arange(self.L)).float()
        if use_pi:
            freqs = math.pi * freqs
        self.register_buffer("freqs", freqs[None, :, None, None], persistent=False)

    def sine_encode_1d(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B, N, 1]  ->  [B, N, 2L] via sin/cos at exponentially spaced freqs.
        """
        angles = t * self.freqs  # broadcast
        ret = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [B, 2L, H, W]
        return ret

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        enc = self.sine_encode_1d(x)  # [B,N,2L]
        return self.mlp(enc) # [B, C, N]