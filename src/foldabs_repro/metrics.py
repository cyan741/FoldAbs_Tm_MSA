from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    result = spearmanr(y_true, y_pred)
    rho = float(result.statistic if hasattr(result, "statistic") else result[0])
    if math.isnan(rho):
        rho = 0.0
    return {
        "spearman_rho": rho,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }

