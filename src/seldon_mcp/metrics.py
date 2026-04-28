from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute classification metrics from true and predicted labels."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics from true and predicted values."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "mse": round(mse, 4),
        "rmse": round(math.sqrt(mse), 4),
        "r2": round(r2_score(y_true, y_pred), 4),
        "median_absolute_error": round(median_absolute_error(y_true, y_pred), 4),
    }
