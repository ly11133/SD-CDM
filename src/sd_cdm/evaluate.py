"""Evaluation metrics for cognitive diagnosis."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score


def binary_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    y_label = (y_pred >= 0.5).astype(int)
    metrics = {
        "acc": float(accuracy_score(y_true, y_label)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_pred))
    else:
        metrics["auc"] = float("nan")
    return metrics
