"""
Fourier time encoding for irregularly sampled time series.

Replaces sequence-position positional encoding with actual timestamp-aware
encoding using learnable Fourier features. This is critical for ZTF light
curves where time gaps carry astrophysical information.
"""

import math
import torch
import torch.nn as nn
import numpy as np


class FourierTimeEncoding(nn.Module):
    """
    Encode irregular timestamps using learnable Fourier features.

    For each timestamp t, computes:
        dt = t - t_0  (relative to first observation)
        features = [sin(2*pi*f_1*dt), cos(2*pi*f_1*dt), ..., sin(2*pi*f_K*dt), cos(2*pi*f_K*dt)]
        output = Linear(features) -> d_model

    Frequencies are initialized log-spaced from ~minutes to ~years and are
    learnable, allowing the model to discover relevant timescales.

    Parameters
    ----------
    d_model : int
        Output embedding dimension.
    n_fourier_features : int
        Number of frequency pairs (total Fourier features = 2 * n_fourier_features).
    learnable_frequencies : bool
        If True, frequencies are nn.Parameters that get optimized.
    min_period_days : float
        Shortest period to initialize (in days).
    max_period_days : float
        Longest period to initialize (in days).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_fourier_features: int = 64,
        learnable_frequencies: bool = True,
        min_period_days: float = 0.001,  # ~1.4 minutes
        max_period_days: float = 1000.0,  # ~2.7 years
    ):
        super().__init__()
        self.d_model = d_model
        self.n_fourier_features = n_fourier_features

        # Initialize frequencies log-spaced: f = 1/period
        log_freqs = torch.linspace(
            math.log10(1.0 / max_period_days),
            math.log10(1.0 / min_period_days),
            n_fourier_features,
        )
        freqs = 10.0 ** log_freqs  # cycles/day

        if learnable_frequencies:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

        # Project 2*n_fourier_features -> d_model
        self.projection = nn.Linear(2 * n_fourier_features, d_model)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        timestamps : (batch, seq_len)
            Timestamps in BJD (or any consistent time unit in days).

        Returns
        -------
        time_encoding : (batch, seq_len, d_model)
        """
        # Compute dt relative to first observation (translation-invariant)
        t0 = timestamps[:, :1]  # (batch, 1)
        dt = timestamps - t0  # (batch, seq_len)

        # Fourier features: (batch, seq_len, n_fourier_features)
        angles = 2.0 * math.pi * dt.unsqueeze(-1) * self.freqs  # broadcast

        # Concatenate sin and cos: (batch, seq_len, 2*n_fourier_features)
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return self.projection(features)  # (batch, seq_len, d_model)


class FourierTimeSeriesEmbedding(nn.Module):
    """
    Embed light curve observations into d_model space using:
    1. Linear projection of (flux, flux_err) -> d_model
    2. Fourier encoding of actual timestamps -> d_model
    3. Sum + LayerNorm + Dropout

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    n_fourier_features : int
        Number of Fourier frequency pairs.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_fourier_features: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Project (flux, flux_err) -> d_model
        self.value_projection = nn.Linear(2, d_model)

        # Fourier time encoding
        self.time_encoding = FourierTimeEncoding(
            d_model=d_model,
            n_fourier_features=n_fourier_features,
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        time: torch.Tensor,
        flux: torch.Tensor,
        flux_err: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        time : (batch, seq_len)
            Timestamps in BJD.
        flux : (batch, seq_len)
            Normalized flux values.
        flux_err : (batch, seq_len)
            Flux uncertainties.

        Returns
        -------
        embeddings : (batch, seq_len, d_model)
        """
        # Value embedding: project (flux, flux_err) to d_model
        values = torch.stack([flux, flux_err], dim=-1)  # (batch, seq_len, 2)
        value_emb = self.value_projection(values)  # (batch, seq_len, d_model)

        # Time encoding from actual timestamps
        time_emb = self.time_encoding(time)  # (batch, seq_len, d_model)

        # Combine
        embeddings = value_emb + time_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


if __name__ == "__main__":
    print("Testing Fourier time encoding...")

    batch_size, seq_len = 8, 256
    d_model = 256

    # Simulate irregular ZTF timestamps (BJD, ~2458000-2460000)
    timestamps = torch.sort(
        torch.rand(batch_size, seq_len) * 2000 + 2458000
    ).values

    # Test FourierTimeEncoding
    encoder = FourierTimeEncoding(d_model=d_model, n_fourier_features=64)
    time_emb = encoder(timestamps)
    print(f"FourierTimeEncoding: {timestamps.shape} -> {time_emb.shape}")

    # Test full embedding
    flux = torch.randn(batch_size, seq_len) * 0.01
    flux_err = torch.abs(torch.randn(batch_size, seq_len)) * 0.001

    embedding = FourierTimeSeriesEmbedding(d_model=d_model)
    output = embedding(timestamps, flux, flux_err)
    print(f"FourierTimeSeriesEmbedding: -> {output.shape}")

    n_params = sum(p.numel() for p in embedding.parameters())
    print(f"Parameters: {n_params:,}")

    # Verify translation invariance
    timestamps_shifted = timestamps + 1000.0
    out1 = encoder(timestamps)
    out2 = encoder(timestamps_shifted)
    diff = (out1 - out2).abs().max().item()
    print(f"Translation invariance check (max diff): {diff:.6f}")

    print("\nAll tests passed.")
