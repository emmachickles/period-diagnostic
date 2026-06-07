"""
CNN and RNN encoder baselines for the ZTF SSL benchmark.

Both encoders match ContinuousTimeLightCurveTransformer's forward signature:
    forward(time, flux, flux_err, mask) -> {z_sig, z_qual, pooled}

so they slot into the existing DDP training loop and contrastive loss without
any changes upstream. Time enters as a 3rd channel (log Δt to previous obs)
rather than via Fourier features + B(Δt) bias — that's the whole point of the
baselines: handicap the cadence treatment to measure how much our continuous-
time inductive bias actually buys us.

Channel layout (B, L, 3): [flux, log10(flux_err+ε), log10(Δt+ε)]
  * Δt is computed as time[:, t] - time[:, t-1], with t=0 set to 0
  * log scale on err and Δt because both are positive and span ~3 orders of
    magnitude on ZTF
  * mask=False positions are zeroed out before the encoder; pooling is over
    mask=True only
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_channels(time, flux, flux_err, mask):
    """Stack (B, L) tensors into (B, 3, L) = [flux, log10 err, log10 Δt].

    Mask is applied after stacking — invalid positions are zeroed.
    """
    eps = 1e-8

    log_err = torch.log10(flux_err.clamp(min=eps))

    # Δt: pad with 0 at position 0 (no previous obs)
    dt = torch.zeros_like(time)
    dt[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp(min=0.0)
    log_dt = torch.log10(dt.clamp(min=eps))

    x = torch.stack([flux, log_err, log_dt], dim=1)  # (B, 3, L)

    if mask is not None:
        m = mask.to(x.dtype).unsqueeze(1)  # (B, 1, L)
        x = x * m

    return x


def _masked_mean_pool(x, mask):
    """x: (B, D, L); mask: (B, L) bool. Returns (B, D)."""
    if mask is None:
        return x.mean(dim=-1)
    m = mask.to(x.dtype).unsqueeze(1)  # (B, 1, L)
    n = m.sum(dim=-1).clamp(min=1.0)   # (B, 1)
    return (x * m).sum(dim=-1) / n


class CNN1DEncoder(nn.Module):
    """1D CNN over [flux, log_err, log_dt] sequence.

    Architecture: stack of (Conv1d → GroupNorm → GELU) blocks with stride-1
    convs and increasing channels. No pooling — keeps full sequence length so
    the masked-mean pool sees one feature vector per real observation. Final
    z_sig / z_qual heads are 1×1 projections after pool.
    """

    def __init__(
        self,
        d_model: int = 192,
        num_layers: int = 4,
        kernel_size: int = 7,
        d_sig: int = 128,
        d_qual: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_ch = 3
        for i in range(num_layers):
            out_ch = d_model
            layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2)
            )
            layers.append(nn.GroupNorm(8, out_ch))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)
        # Heads match ContinuousTimeLightCurveTransformer exactly so the
        # downstream loss (DDPDisentangledLoss expects z_qual shape (B,1) for
        # MSE against log_median_sigma) works without changes.
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

    def forward(
        self,
        time: torch.Tensor,
        flux: torch.Tensor,
        flux_err: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = _build_channels(time, flux, flux_err, mask)  # (B, 3, L)
        h = self.encoder(x)                              # (B, D, L)
        pooled = _masked_mean_pool(h, mask)              # (B, D)
        return {
            "z_sig": self.z_sig_head(pooled),
            "z_qual": self.z_qual_head(pooled),
            "pooled": pooled,
        }


class RNNEncoder(nn.Module):
    """Bidirectional GRU over [flux, log_err, log_dt] sequence.

    GRU > LSTM here: similar accuracy on this kind of variable-cadence
    sequence in informal trials, ~25% fewer params.
    """

    def __init__(
        self,
        d_model: int = 192,
        num_layers: int = 2,
        d_sig: int = 128,
        d_qual: int = 32,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=3,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = d_model * (2 if bidirectional else 1)
        self.z_sig_head = nn.Sequential(
            nn.Linear(out_dim, d_sig),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.z_qual_head = nn.Sequential(
            nn.Linear(out_dim, d_qual),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_qual, 1),
        )

    def forward(
        self,
        time: torch.Tensor,
        flux: torch.Tensor,
        flux_err: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = _build_channels(time, flux, flux_err, mask)  # (B, 3, L)
        x = x.transpose(1, 2)                            # (B, L, 3)
        h, _ = self.rnn(x)                               # (B, L, D*dirs)
        h = h.transpose(1, 2)                            # (B, D*dirs, L)
        pooled = _masked_mean_pool(h, mask)              # (B, D*dirs)
        return {
            "z_sig": self.z_sig_head(pooled),
            "z_qual": self.z_qual_head(pooled),
            "pooled": pooled,
        }


def build_encoder(name: str, **kwargs):
    """Factory: 'cnn' or 'rnn'."""
    name = name.lower()
    if name == "cnn":
        return CNN1DEncoder(**kwargs)
    if name == "rnn":
        return RNNEncoder(**kwargs)
    raise ValueError(f"Unknown encoder: {name!r}")
