from __future__ import annotations

import numpy as np

from seldon_mcp.metrics import compute_classification_metrics, compute_regression_metrics


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = np.array(["cat", "dog", "cat", "dog"])
        y_pred = np.array(["cat", "dog", "cat", "dog"])
        m = compute_classification_metrics(y_true, y_pred)

        assert m["accuracy"] == 1.0
        assert m["f1_score"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0

    def test_partial_predictions(self):
        y_true = np.array(["cat", "dog", "cat", "dog", "cat"])
        y_pred = np.array(["cat", "dog", "dog", "dog", "cat"])
        m = compute_classification_metrics(y_true, y_pred)

        assert m["accuracy"] == 0.8
        assert 0 < m["f1_score"] < 1.0

    def test_confusion_matrix_shape(self):
        y_true = np.array(["a", "b", "c", "a", "b", "c"])
        y_pred = np.array(["a", "b", "a", "a", "c", "c"])
        m = compute_classification_metrics(y_true, y_pred)

        assert len(m["confusion_matrix"]) == 3
        assert all(len(row) == 3 for row in m["confusion_matrix"])

    def test_classification_report_is_string(self):
        y_true = np.array(["x", "y", "x"])
        y_pred = np.array(["x", "x", "x"])
        m = compute_classification_metrics(y_true, y_pred)

        assert isinstance(m["classification_report"], str)
        assert "precision" in m["classification_report"]

    def test_all_wrong(self):
        y_true = np.array(["a", "a", "a"])
        y_pred = np.array(["b", "b", "b"])
        m = compute_classification_metrics(y_true, y_pred)

        assert m["accuracy"] == 0.0


class TestRegressionMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        m = compute_regression_metrics(y_true, y_pred)

        assert m["mae"] == 0.0
        assert m["mse"] == 0.0
        assert m["rmse"] == 0.0
        assert m["r2"] == 1.0
        assert m["median_absolute_error"] == 0.0

    def test_known_errors(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
        m = compute_regression_metrics(y_true, y_pred)

        assert m["mae"] > 0
        assert m["rmse"] >= m["mae"]
        assert m["r2"] > 0.9
        assert m["mse"] > 0
        assert m["median_absolute_error"] > 0

    def test_rmse_is_sqrt_mse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        m = compute_regression_metrics(y_true, y_pred)

        import math
        assert abs(m["rmse"] - math.sqrt(m["mse"])) < 0.001

    def test_negative_r2_possible(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        m = compute_regression_metrics(y_true, y_pred)

        assert m["r2"] < 0
