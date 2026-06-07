"""
Continuous-time transformer for irregular-cadence light curves.

Same overall shape as `DisentangledLightCurveTransformer`:
  Fourier time-series embedding → transformer trunk → masked-mean pool
  → z_sig (InfoNCE) and z_qual (MSE on log median σ) heads.

Difference: every attention layer's logits are biased by a learned
function of Δt = t_i − t_j between the query and key timestamps. Time
information is no longer trapped at the input — it modulates attention
all the way down. The pairwise time-feature tensor is computed *once*
outside the trunk and reused at every layer (only the per-head
projection is per-layer), so memory cost is O(B·L²·F) once rather than
O(L_layers · B·L²·F).

Drop-in compatible with the existing training script via the same forward
signature (time, flux, flux_err, mask) → {z_sig, z_qual, pooled}.
"""

from __future__ import annotations

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fourier_embedding import FourierTimeSeriesEmbedding


class ContinuousTimeAttention(nn.Module):
    """Multi-head self-attention with an additive Δt bias on the logits.

    Each layer owns a small linear projection that maps the SHARED
    pairwise time features (sin/cos at K learnable timescales) to one
    scalar per head — that's the per-pair, per-head additive bias.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Per-layer projection from time features to per-head bias scalars.
        # Initialized small so attention starts close to vanilla transformer
        # behavior — the time bias is a perturbation that grows during training.
        self.time_bias_proj: Optional[nn.Linear] = None  # set after construction

    def set_time_bias_proj(self, n_time_features: int) -> None:
        """Called by the parent module so the projection's input size matches
        the SHARED time-feature tensor."""
        self.time_bias_proj = nn.Linear(n_time_features, self.nhead)
        # Keep initial bias near zero
        nn.init.zeros_(self.time_bias_proj.weight)
        nn.init.zeros_(self.time_bias_proj.bias)

    def forward(
        self,
        x: torch.Tensor,                       # (B, L, d_model)
        time_features: torch.Tensor,           # (B, L, L, n_time_features)
        key_padding_mask: Optional[torch.Tensor],  # (B, L) — True = valid
    ) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Standard scaled dot-product logits
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, L, L)

        # Δt-bias: project shared (B, L, L, F) features to (B, L, L, H), permute
        if self.time_bias_proj is not None:
            time_bias = self.time_bias_proj(time_features)            # (B, L, L, H)
            scores = scores + time_bias.permute(0, 3, 1, 2)           # (B, H, L, L)

        # Mask out padded keys
        if key_padding_mask is not None:
            # True = valid → expand to attention mask (B, 1, 1, L)
            ignore = (~key_padding_mask).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(ignore, float("-inf"))

        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v                                               # (B, H, L, hd)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class ContinuousTimeBlock(nn.Module):
    """Pre-norm transformer block with continuous-time attention."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ContinuousTimeAttention(d_model, nhead, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, time_features, key_padding_mask):
        x = x + self.drop(self.attn(self.ln1(x), time_features, key_padding_mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class ContinuousTimeLightCurveTransformer(nn.Module):
    """Drop-in replacement for ``DisentangledLightCurveTransformer`` with
    continuous-time attention bias inside every transformer layer."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        n_fourier_features: int = 64,
        d_sig: int = 128,
        d_qual: int = 32,
        # Continuous-time-specific
        n_time_bias_pairs: int = 16,        # K (gives 2K = 32 features)
        time_bias_min_period: float = 0.001,    # days
        time_bias_max_period: float = 1000.0,   # days
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = FourierTimeSeriesEmbedding(
            d_model=d_model,
            n_fourier_features=n_fourier_features,
            dropout=dropout,
        )
        self.layers = nn.ModuleList([
            ContinuousTimeBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # SHARED log-spaced timescales for Δt → sin/cos features
        self.n_time_bias_pairs = n_time_bias_pairs
        log_freqs = torch.linspace(
            math.log10(1.0 / time_bias_max_period),
            math.log10(1.0 / time_bias_min_period),
            n_time_bias_pairs,
        )
        self.time_bias_freqs = nn.Parameter(10.0 ** log_freqs)  # learnable

        # Per-layer projection (each block's attention owns its own)
        n_time_features = 2 * n_time_bias_pairs
        for blk in self.layers:
            blk.attn.set_time_bias_proj(n_time_features)

        # Heads — same as DisentangledLightCurveTransformer
        self.z_sig_head = nn.Sequential(
            nn.Linear(d_model, d_sig),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.z_qual_head = nn.Sequential(
            nn.Linear(d_model, d_qual),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_qual, 1),
        )

    def _masked_mean_pool(self, x, mask):
        if mask is None:
            return x.mean(dim=1)
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)

    def _build_time_features(self, time, mask):
        """Build the shared (B, L, L, 2K) sin/cos pairwise-Δt tensor.

        Padding pairs are still computed but their attention contribution
        is masked downstream by the key-padding mask, so we don't bother
        zero-ing them here.
        """
        # Δt: (B, L, L) — anti-symmetric on the diagonal
        dt = time.unsqueeze(2) - time.unsqueeze(1)
        # Angles: 2π · Δt · f for each shared frequency
        angles = 2.0 * math.pi * dt.unsqueeze(-1) * self.time_bias_freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(
        self,
        time: torch.Tensor,        # (B, L)
        flux: torch.Tensor,        # (B, L)
        flux_err: torch.Tensor,    # (B, L)
        mask: Optional[torch.Tensor] = None,  # (B, L) True=valid
    ) -> Dict[str, torch.Tensor]:
        x = self.embedding(time, flux, flux_err)              # (B, L, D)
        time_features = self._build_time_features(time, mask)  # (B, L, L, 2K)

        for blk in self.layers:
            x = blk(x, time_features, mask)
        x = self.norm(x)

        pooled = self._masked_mean_pool(x, mask)
        return {
            "z_sig":  self.z_sig_head(pooled),
            "z_qual": self.z_qual_head(pooled),
            "pooled": pooled,
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L = 4, 64
    t = torch.sort(torch.rand(B, L) * 1000 + 2458000).values
    f = torch.randn(B, L) * 0.01
    fe = torch.abs(torch.randn(B, L)) * 0.001
    m = torch.ones(B, L, dtype=torch.bool); m[:, 50:] = False

    model = ContinuousTimeLightCurveTransformer(
        d_model=128, nhead=4, num_layers=3, dim_feedforward=256,
        n_fourier_features=32, n_time_bias_pairs=8,
    )
    out = model(t, f, fe, m)
    n = sum(p.numel() for p in model.parameters())
    print(f"Params: {n:,}")
    print(f"z_sig: {out['z_sig'].shape}, z_qual: {out['z_qual'].shape}, pooled: {out['pooled'].shape}")
    print(f"Time-feature tensor would be: ({B}, {L}, {L}, {2*8}) = "
          f"{B*L*L*2*8 * 4 / 1e6:.1f} MB at fp32")
