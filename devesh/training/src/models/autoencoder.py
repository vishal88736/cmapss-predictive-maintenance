"""
DM-TCAE: Denoising Masked Temporal Convolutional Autoencoder.

Architecture:
  Input:   [B, W, 14]  (corrupted)
  Encoder: Residual 1D-CNN  14→32→16  → Latent [B, W, 16]
  Decoder: 1D-CNN           16→32→14  → Reconstruction [B, W, 14]

The model reconstructs clean input from corrupted input.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ResidualBlock1D(nn.Module):
    """
    Residual 1D-Convolutional Block.

    Input/output channels must be the same for the skip connection.
    If they differ, a 1×1 projection is applied.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return out


class DMTCAE(nn.Module):
    """
    Denoising Masked Temporal Convolutional Autoencoder.

    Input shape :  [B, W, 14]   (time-last after transpose)
    Latent shape:  [B, W, 16]
    Output shape:  [B, W, 14]
    """

    def __init__(self, in_channels: int = 14, latent_channels: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Encoder: 14 → 32 → 16
        self.encoder = nn.Sequential(
            ResidualBlock1D(in_channels, 32, kernel_size=3),
            ResidualBlock1D(32, latent_channels, kernel_size=3),
        )

        # Decoder: 16 → 32 → 14
        self.decoder = nn.Sequential(
            ResidualBlock1D(latent_channels, 32, kernel_size=3),
            nn.Conv1d(32, in_channels, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : [B, W, in_channels]

        Returns
        -------
        reconstruction : [B, W, in_channels]
        latent         : [B, W, latent_channels]
        """
        # Conv1d expects [B, C, L]
        x_t = x.permute(0, 2, 1)           # [B, 14, W]
        z_t = self.encoder(x_t)             # [B, 16, W]
        out_t = self.decoder(z_t)            # [B, 14, W]

        reconstruction = out_t.permute(0, 2, 1)  # [B, W, 14]
        latent = z_t.permute(0, 2, 1)            # [B, W, 16]

        assert reconstruction.shape == x.shape, \
            f"Reconstruction shape {reconstruction.shape} ≠ input {x.shape}"
        assert latent.shape == (x.shape[0], x.shape[1], self.latent_channels), \
            f"Latent shape {latent.shape} unexpected"

        return reconstruction, latent


class MaskedReconstructionLoss(nn.Module):
    """
    Combined reconstruction loss:
      L = (1 - full_weight) * masked_MSE  +  full_weight * full_MSE

    masked_MSE: MSE only on positions where mask == 1 (corrupted elements)
    full_MSE  : MSE on all positions (general denoising signal)
    """

    def __init__(self, full_weight: float = 0.1):
        super().__init__()
        self.full_weight = full_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sq_err = (pred - target) ** 2  # [B, W, F]

        if mask is not None and mask.sum() > 0:
            masked_loss = (sq_err * mask).sum() / (mask.sum() + 1e-8)
        else:
            masked_loss = sq_err.mean()

        full_loss = sq_err.mean()
        loss = (1.0 - self.full_weight) * masked_loss + self.full_weight * full_loss

        if not torch.isfinite(loss):
            loss = full_loss  # fallback if masked region is empty

        return loss
