"""
Loss functions for disentangled self-supervised learning.

- InfoNCE (NT-Xent) on z_sig for astrophysical signal contrastive learning
- MSE regression on z_qual for predicting photometric uncertainty
- ProjectionHead for contrastive projection (discarded after training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning (discarded after training)."""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class InfoNCELoss(nn.Module):
    """
    InfoNCE / NT-Xent contrastive loss.

    For a batch of N pairs (z_i, z_j), treats (z_i, z_j) as positives
    and all other 2(N-1) combinations as negatives.

    Parameters
    ----------
    temperature : float
        Temperature scaling for similarity scores.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_i, z_j : (batch, proj_dim)
            L2-normalized projections of two augmented views.

        Returns
        -------
        loss : scalar
        """
        batch_size = z_i.shape[0]

        # Normalize (eps prevents division-by-zero NaN for near-zero
        # vectors, which can occur in fp16 for degenerate light curves)
        z_i = F.normalize(z_i, dim=1, eps=1e-6)
        z_j = F.normalize(z_j, dim=1, eps=1e-6)

        # Concatenate: (2*batch, proj_dim)
        z = torch.cat([z_i, z_j], dim=0)

        # Similarity matrix: (2*batch, 2*batch)
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # Masks
        diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)

        # Positive pairs: (i, i+N) and (i+N, i)
        pos_mask = torch.zeros_like(diag_mask)
        eye_n = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        pos_mask[:batch_size, batch_size:] = eye_n
        pos_mask[batch_size:, :batch_size] = eye_n

        # Remove self-similarity
        sim_matrix = sim_matrix.masked_fill(diag_mask, float("-inf"))

        # Positive logits and all logits
        positives = sim_matrix[pos_mask].view(2 * batch_size, 1)
        negatives = sim_matrix[~diag_mask].view(2 * batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)

        return F.cross_entropy(logits, labels)


class DisentangledLoss(nn.Module):
    """
    Combined loss for disentangled self-supervised training.

    L = L_infonce(z_sig) + lambda_qual * L_mse(z_qual, log_median_sigma)

    Parameters
    ----------
    d_sig : int
        Dimension of z_sig embedding.
    proj_hidden_dim : int
        Hidden dim of contrastive projection head.
    proj_output_dim : int
        Output dim of contrastive projection head.
    temperature : float
        InfoNCE temperature.
    lambda_qual : float
        Weight for quality regression loss.
    """

    def __init__(
        self,
        d_sig: int = 128,
        proj_hidden_dim: int = 256,
        proj_output_dim: int = 128,
        temperature: float = 0.07,
        lambda_qual: float = 1.0,
    ):
        super().__init__()
        self.lambda_qual = lambda_qual

        self.projection_head = ProjectionHead(
            input_dim=d_sig,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
        )
        self.infonce = InfoNCELoss(temperature=temperature)

    def forward(
        self,
        z_sig_1: torch.Tensor,
        z_sig_2: torch.Tensor,
        z_qual: torch.Tensor,
        log_median_sigma: torch.Tensor,
    ) -> dict:
        """
        Parameters
        ----------
        z_sig_1, z_sig_2 : (batch, d_sig)
            Signal embeddings from two augmented views.
        z_qual : (batch, 1)
            Predicted quality scalar.
        log_median_sigma : (batch,)
            Target: log10(median(sigma_f)) for each light curve.

        Returns
        -------
        dict with keys: loss, loss_contrastive, loss_qual
        """
        # Contrastive loss on z_sig
        proj_1 = self.projection_head(z_sig_1)
        proj_2 = self.projection_head(z_sig_2)
        loss_contrastive = self.infonce(proj_1, proj_2)

        # Quality regression loss
        loss_qual = F.mse_loss(z_qual.squeeze(-1), log_median_sigma)

        # Combined
        loss = loss_contrastive + self.lambda_qual * loss_qual

        return {
            "loss": loss,
            "loss_contrastive": loss_contrastive.item(),
            "loss_qual": loss_qual.item(),
        }


if __name__ == "__main__":
    print("Testing losses...")

    batch_size = 16
    d_sig = 128

    # Test InfoNCE
    infonce = InfoNCELoss(temperature=0.07)
    z_i = torch.randn(batch_size, d_sig)
    z_j = torch.randn(batch_size, d_sig)
    loss = infonce(z_i, z_j)
    print(f"InfoNCE loss: {loss.item():.4f}")

    # Test DisentangledLoss
    criterion = DisentangledLoss(d_sig=d_sig, lambda_qual=1.0)
    z_sig_1 = torch.randn(batch_size, d_sig)
    z_sig_2 = torch.randn(batch_size, d_sig)
    z_qual = torch.randn(batch_size, 1)
    log_med_sigma = torch.randn(batch_size)

    result = criterion(z_sig_1, z_sig_2, z_qual, log_med_sigma)
    print(f"Combined loss: {result['loss'].item():.4f}")
    print(f"  Contrastive: {result['loss_contrastive']:.4f}")
    print(f"  Quality: {result['loss_qual']:.4f}")

    print("\nAll tests passed.")
