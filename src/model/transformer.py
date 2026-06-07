"""
Disentangled Light Curve Transformer.

Shared transformer backbone producing two disentangled embedding spaces:
  z_sig  — astrophysical signal structure (128-d, contrastive learning)
  z_qual — photometric quality / uncertainty (32-d, regression target)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .fourier_embedding import FourierTimeSeriesEmbedding


class TransformerEncoder(nn.Module):
    """Standard Transformer encoder with multi-head self-attention."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # pre-norm (more stable)
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        src : (batch, seq_len, d_model)
        src_key_padding_mask : (batch, seq_len)
            True = valid data, False = padding (converted internally).
        """
        # Convert: dataset uses True=valid, transformer uses True=ignore
        if src_key_padding_mask is not None:
            src_key_padding_mask = ~src_key_padding_mask

        return self.encoder(
            src, mask=mask, src_key_padding_mask=src_key_padding_mask
        )


class DisentangledLightCurveTransformer(nn.Module):
    """
    Self-supervised transformer for irregular light curves with
    disentangled embedding heads.

    Architecture:
        1. FourierTimeSeriesEmbedding (flux + Fourier time encoding)
        2. TransformerEncoder (shared backbone)
        3. Masked mean pooling
        4. z_sig head (128-d, for contrastive learning on astrophysical signal)
        5. z_qual head (32-d → scalar, for predicting photometric uncertainty)

    Parameters
    ----------
    d_model : int
        Transformer hidden dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of transformer layers.
    dim_feedforward : int
        Feedforward network dimension.
    dropout : float
        Dropout rate.
    n_fourier_features : int
        Number of Fourier frequency pairs for time encoding.
    d_sig : int
        Dimension of z_sig embedding.
    d_qual : int
        Dimension of z_qual embedding (before scalar projection).
    """

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
    ):
        super().__init__()
        self.d_model = d_model
        self.d_sig = d_sig
        self.d_qual = d_qual

        # Embedding: (flux, flux_err, timestamps) -> d_model
        self.embedding = FourierTimeSeriesEmbedding(
            d_model=d_model,
            n_fourier_features=n_fourier_features,
            dropout=dropout,
        )

        # Shared transformer backbone
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # z_sig head: astrophysical signal embedding
        self.z_sig_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_sig),
        )

        # z_qual head: photometric quality embedding -> scalar
        self.z_qual_head = nn.Sequential(
            nn.Linear(d_model, d_qual),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_qual, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _masked_mean_pool(
        self, encoded: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mean pooling over valid (unmasked) sequence positions."""
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            masked_encoded = encoded * mask_expanded
            pooled = masked_encoded.sum(dim=1) / mask_expanded.sum(dim=1).clamp(
                min=1e-6
            )
        else:
            pooled = encoded.mean(dim=1)
        return pooled

    def forward(
        self,
        time: torch.Tensor,
        flux: torch.Tensor,
        flux_err: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        time : (batch, seq_len)
            Timestamps in BJD.
        flux : (batch, seq_len)
            Normalized flux.
        flux_err : (batch, seq_len)
            Flux uncertainties.
        mask : (batch, seq_len), optional
            Boolean mask (True = valid).

        Returns
        -------
        dict with keys:
            z_sig : (batch, d_sig)
            z_qual : (batch, 1)  — predicted log(median_sigma)
            pooled : (batch, d_model)  — pooled backbone output
        """
        # Embed
        embeddings = self.embedding(time, flux, flux_err)

        # Encode
        encoded = self.encoder(embeddings, src_key_padding_mask=mask)

        # Pool
        pooled = self._masked_mean_pool(encoded, mask)

        # Disentangled heads
        z_sig = self.z_sig_head(pooled)  # (batch, d_sig)
        z_qual = self.z_qual_head(pooled)  # (batch, 1)

        return {
            "z_sig": z_sig,
            "z_qual": z_qual,
            "pooled": pooled,
        }

    @torch.no_grad()
    def get_embeddings(
        self,
        time: torch.Tensor,
        flux: torch.Tensor,
        flux_err: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract embeddings for downstream evaluation."""
        return self.forward(time, flux, flux_err, mask)


if __name__ == "__main__":
    print("Testing DisentangledLightCurveTransformer...")

    model = DisentangledLightCurveTransformer(
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        d_sig=128,
        d_qual=32,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Dummy input
    batch_size, seq_len = 8, 256
    time = torch.sort(torch.rand(batch_size, seq_len) * 2000 + 2458000).values
    flux = torch.randn(batch_size, seq_len) * 0.01
    flux_err = torch.abs(torch.randn(batch_size, seq_len)) * 0.001
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, 200:] = False  # simulate padding

    output = model(time, flux, flux_err, mask)

    print(f"\nOutput shapes:")
    for key, val in output.items():
        print(f"  {key}: {val.shape}")

    print("\nAll tests passed.")
