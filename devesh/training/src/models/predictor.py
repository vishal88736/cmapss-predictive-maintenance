"""
SupervisedPredictor: CNN-Transformer RUL Predictor.

Architecture:
  1. Pretrained DM-TCAE autoencoder produces:
       X_hat [B, W, 14]  (clean reconstruction)
       Z     [B, W, 16]  (latent features)
  2. Concatenate along feature dim: [B, W, 30]
  3. 1D-CNN feature extractor (kernel=3, 64 channels)
  4. Two Transformer encoder layers (d_model=64, heads=4, ffn=128)
  5. Temporal mean pooling → [B, 64]
  6. Regression head → [B, 1]  (RUL ∈ [0, max_rul])
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from .autoencoder import DMTCAE


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.drop(x)


class SupervisedPredictor(nn.Module):
    """
    Full CNN-Transformer RUL predictor.

    If a pretrained_autoencoder is provided, its weights are inherited and
    it is further fine-tuned jointly. Otherwise a fresh DMTCAE is created.
    """

    def __init__(
        self,
        in_channels: int = 14,
        latent_channels: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        max_rul: Optional[float] = 125.0,
        pretrained_autoencoder: Optional[DMTCAE] = None,
    ):
        super().__init__()
        self.max_rul = max_rul
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.fused_channels = in_channels + latent_channels  # 14 + 16 = 30

        # ── Autoencoder ─────────────────────────────────────────────────────
        if pretrained_autoencoder is not None:
            self.autoencoder = pretrained_autoencoder
        else:
            self.autoencoder = DMTCAE(in_channels=in_channels, latent_channels=latent_channels)

        # ── 1D-CNN feature projector: 30 → 64 ───────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv1d(self.fused_channels, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Positional encoding ─────────────────────────────────────────────
        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)

        # ── Transformer encoder ─────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ── Regression head ─────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : [B, W, 14]  normalised sensor windows

        Returns
        -------
        y_pred  : [B, 1]   predicted RUL (clamped to [0, max_rul])
        h       : [B, 64]  temporal mean-pooled feature vector (TD3 state)
        x_hat   : [B, W, 14]  reconstructed clean window
        z       : [B, W, 16]  latent features from encoder
        """
        assert x.shape[-1] == self.in_channels, \
            f"Expected {self.in_channels} input features, got {x.shape[-1]}"

        # ── Step 1: Autoencoder ─────────────────────────────────────────────
        x_hat, z = self.autoencoder(x)   # [B,W,14], [B,W,16]

        # ── Step 2: Fusion ──────────────────────────────────────────────────
        fused = torch.cat([x_hat, z], dim=-1)   # [B, W, 30]

        # ── Step 3: CNN ──────────────────────────────────────────────────────
        fused_t = fused.permute(0, 2, 1)        # [B, 30, W]
        feat_t = self.cnn(fused_t)              # [B, 64, W]
        feat = feat_t.permute(0, 2, 1)          # [B, W, 64]

        # ── Step 4: Transformer ──────────────────────────────────────────────
        feat = self.pos_enc(feat)
        encoded = self.transformer(feat)        # [B, W, 64]

        # ── Step 5: Last-token pooling ───────────────────────────────────────
        h = encoded[:, -1, :]                   # [B, 64]

        # ── Step 6: Regression head ──────────────────────────────────────────
        y_raw = self.head(h)                    # [B, 1]

        if self.max_rul is not None:
            y_pred = torch.clamp(y_raw, min=0.0, max=self.max_rul)
        else:
            y_pred = torch.clamp(y_raw, min=0.0)

        return y_pred, h, x_hat, z


class CombinedPredictorLoss(nn.Module):
    """
    L_total = L_main + lambda_rec * L_rec

    L_main : Huber loss (or MSE) on RUL prediction
    L_rec  : MSE reconstruction loss (denoising auxiliary signal)
    """

    def __init__(self, loss_type: str = "huber", lambda_rec: float = 0.2, delta: float = 10.0, linex_weight: float = 0.1):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.linex_weight = linex_weight
        if loss_type == "huber":
            self.main_criterion = nn.HuberLoss(delta=delta)
        else:
            self.main_criterion = nn.MSELoss()
        self.rec_criterion = nn.MSELoss()

    def forward(
        self,
        y_pred: torch.Tensor,   # [B, 1]
        y_true: torch.Tensor,   # [B, 1] or [B]
        x_hat: torch.Tensor,    # [B, W, 14]
        x_clean: torch.Tensor,  # [B, W, 14]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if y_true.ndim == 1:
            y_true = y_true.unsqueeze(-1)
        main_loss = self.main_criterion(y_pred, y_true)
        
        # Asymmetric LinEx penalty for overestimation
        d = y_pred - y_true
        linex_loss = (torch.exp(0.1 * d) - 0.1 * d - 1).mean()
        main_loss = main_loss + self.linex_weight * linex_loss
        
        rec_loss = self.rec_criterion(x_hat, x_clean)
        total = main_loss + self.lambda_rec * rec_loss
        return total, main_loss, rec_loss
