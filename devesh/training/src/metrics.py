"""
Metrics for RUL prediction evaluation.

RMSE    : Root Mean Squared Error (primary benchmark metric)
MAE     : Mean Absolute Error
CMAPSS_Score : NASA asymmetric scoring function (penalises late predictions more)
"""

import numpy as np
from typing import Dict


def cmapss_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    NASA C-MAPSS asymmetric scoring function.

    d  = y_pred - y_true
    s  = sum(exp(-d/13) - 1 for d < 0) + sum(exp(d/10) - 1 for d >= 0)
    """
    d = y_pred.flatten() - y_true.flatten()
    s = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(s))


def calculate_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    label: str = "",
) -> Dict[str, float]:
    """
    Compute RMSE, MAE, and NASA C-MAPSS score.

    Parameters
    ----------
    y_pred : predicted RUL values [N]
    y_true : true RUL values [N]
    label  : optional string label for display

    Returns
    -------
    dict with 'RMSE', 'MAE', 'CMAPSS_Score', 'N'
    """
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    assert len(y_pred) == len(y_true), "Prediction/truth length mismatch"

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    score = cmapss_score(y_pred, y_true)
    n = len(y_pred)

    if label:
        print(f"  [{label}] N={n:5d} | RMSE={rmse:.3f} | MAE={mae:.3f} | CMAPSS_Score={score:.1f}")

    return {"RMSE": rmse, "MAE": mae, "CMAPSS_Score": score, "N": n}
