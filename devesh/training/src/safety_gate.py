"""
Validation-Based Safety Gate.

Calibrated ONLY on validation data. The test set is never used during calibration.

The gate:
  1. Grid-searches blending coefficient alpha ∈ [0, 1] on validation RMSE
     y_gated = alpha * y_corrected + (1 - alpha) * y_base
  2. Rejects any correction that is NaN, Inf, or out of [0, max_rul]
  3. Can reject harmful corrections (corrected worse than baseline)
"""

import os
import json
import numpy as np
from typing import Dict, Optional, Tuple
from .metrics import cmapss_score


class SafetyGate:
    """
    Validation-calibrated safety gate for TD3 corrections.

    Attributes
    ----------
    alpha      : blending weight (0 = use baseline, 1 = use full correction)
    max_rul    : hard upper bound on any prediction
    gate_threshold : correction magnitude threshold beyond which harder rejection applies
    """

    def __init__(
        self,
        max_allowed_correction: float = 6.0,
        gate_threshold: float = 4.0,
        max_rul: float = 125.0,
    ):
        self.max_allowed_correction = max_allowed_correction
        self.gate_threshold = gate_threshold
        self.max_rul = max_rul
        self.alpha: float = 0.5  # default; calibrated in calibrate()

    def calibrate(
        self,
        val_y_base: np.ndarray,      # baseline predictions on val set
        val_a_lqr: np.ndarray,       # LQR-filtered corrections on val set
        val_y_true: np.ndarray,      # true RUL on val set
        n_alphas: int = 21,
        metric: str = "score",
    ) -> Dict:
        """
        Grid-search alpha to minimise validation score or RMSE.
        Sets self.alpha in-place.

        Returns dict with calibration statistics:
          - validation_base_score, validation_calibrated_score
          - validation_base_rmse, validation_calibrated_rmse
          - best_alpha
        """
        best_alpha = 0.0
        best_metric_val = float("inf")

        val_y_base = np.asarray(val_y_base, dtype=np.float64).flatten()
        val_a_lqr = np.asarray(val_a_lqr, dtype=np.float64).flatten()
        val_y_true = np.asarray(val_y_true, dtype=np.float64).flatten()

        val_y_corr = val_y_base + val_a_lqr

        for alpha in np.linspace(0.0, 1.0, n_alphas):
            y_gated = np.clip(
                alpha * val_y_corr + (1.0 - alpha) * val_y_base,
                0.0,
                self.max_rul,
            )
            if metric == "rmse":
                val = float(np.sqrt(np.mean((y_gated - val_y_true) ** 2)))
            else:
                val = cmapss_score(y_gated, val_y_true)

            if np.isfinite(val) and val < best_metric_val:
                best_metric_val = val
                best_alpha = float(alpha)

        self.alpha = best_alpha

        best_y_gated = np.clip(
            best_alpha * val_y_corr + (1.0 - best_alpha) * val_y_base,
            0.0,
            self.max_rul,
        )

        base_score = cmapss_score(val_y_base, val_y_true)
        calibrated_score = cmapss_score(best_y_gated, val_y_true)
        base_rmse = float(np.sqrt(np.mean((val_y_base - val_y_true) ** 2)))
        calibrated_rmse = float(np.sqrt(np.mean((best_y_gated - val_y_true) ** 2)))

        return {
            "validation_base_score": base_score,
            "validation_calibrated_score": calibrated_score,
            "validation_base_rmse": base_rmse,
            "validation_calibrated_rmse": calibrated_rmse,
            "best_alpha": best_alpha,
        }

    def apply_gate(
        self,
        y_base: np.ndarray,
        a_corrected: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply gate to produce safe predictions.

        Returns
        -------
        y_final   : [N]  final gated predictions
        accepted  : [N]  boolean mask (True = correction accepted)
        """
        y_base = np.asarray(y_base, dtype=np.float64)
        a_corrected = np.asarray(a_corrected, dtype=np.float64)

        y_corr = y_base + a_corrected

        # Reject: NaN, Inf, out-of-bounds
        valid = (
            np.isfinite(a_corrected)
            & np.isfinite(y_corr)
            & (y_corr >= 0.0)
            & (y_corr <= self.max_rul)
            & (np.abs(a_corrected) <= self.max_allowed_correction)
        )

        # Blend where valid
        y_final = np.where(
            valid,
            np.clip(
                self.alpha * y_corr + (1.0 - self.alpha) * y_base,
                0.0,
                self.max_rul,
            ),
            np.clip(y_base, 0.0, self.max_rul),
        )

        return y_final.astype(np.float32), valid

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "alpha": self.alpha,
                    "max_allowed_correction": self.max_allowed_correction,
                    "gate_threshold": self.gate_threshold,
                    "max_rul": self.max_rul,
                },
                f,
                indent=2,
            )

    def load(self, path: str) -> None:
        with open(path) as f:
            cfg = json.load(f)
        self.alpha = cfg["alpha"]
        self.max_allowed_correction = cfg["max_allowed_correction"]
        self.gate_threshold = cfg["gate_threshold"]
        self.max_rul = cfg["max_rul"]
