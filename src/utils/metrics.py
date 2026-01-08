# src/utils/metrics.py

import torch
import torch.nn.functional as F


# ============================================================
# Root Mean Squared Error (RMSE)
# ============================================================
def compute_rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Standard RMSE for RUL evaluation.
    """
    return torch.sqrt(F.mse_loss(y_pred, y_true)).item()


# ============================================================
# Mean Absolute Error (MAE)
# ============================================================
def compute_mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return F.l1_loss(y_pred, y_true).item()


# ============================================================
# NASA PHM Scoring Function
# ============================================================
def compute_phm_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Official NASA scoring rule (exponential penalty).
    Used in CMAPSS and PHM 2008 competition.
    """

    errors = (y_pred - y_true).cpu()

    score = 0.0
    for e in errors:
        e = e.item()
        if e < 0:  # late prediction → more penalty
            score += torch.exp(torch.tensor(-e / 13.0)).item() - 1
        else:      # early prediction → less penalty
            score += torch.exp(torch.tensor(e / 10.0)).item() - 1

    return score


# ============================================================
# R2 Score
# ============================================================
def compute_r2(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Coefficient of determination.
    """
    ss_res = torch.sum((y_true - y_pred) ** 2).item()
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


# ============================================================
# Absolute Error Vector
# ============================================================
def compute_errors(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Returns |y_pred – y_true| for error plots."""
    return torch.abs(y_pred - y_true).cpu()
