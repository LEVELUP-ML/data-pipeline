"""
Tests for model bias detection.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "energy"))


def test_assign_age_bucket():
    """Test age bucket assignment."""
    from model_bias_detection import assign_age_bucket

    assert assign_age_bucket(15) == "<20"
    assert assign_age_bucket(25) == "20-29"
    assert assign_age_bucket(35) == "30-39"
    assert assign_age_bucket(45) == "40+"
    assert assign_age_bucket(None) == "unknown"
    assert assign_age_bucket("abc") == "unknown"


def test_compute_slice_metrics():
    """Test per-slice metric computation."""
    from model_bias_detection import compute_slice_metrics

    y_true = np.array([50, 60, 70, 80])
    y_pred = np.array([52, 58, 72, 78])
    global_mae = 2.0

    metrics = compute_slice_metrics(y_true, y_pred, global_mae)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "count" in metrics
    assert metrics["count"] == 4
    assert metrics["mae"] == 2.0
    assert not metrics["flagged"]  # No deviation from global


def test_compute_slice_metrics_flagged():
    """Test that large deviation triggers flag."""
    from model_bias_detection import compute_slice_metrics

    y_true = np.array([50, 60, 70, 80])
    y_pred = np.array([40, 50, 80, 90])  # Large errors
    global_mae = 2.0  # Much lower than slice MAE

    metrics = compute_slice_metrics(y_true, y_pred, global_mae)

    assert metrics["flagged"], "Should flag large deviation from global MAE"
    assert metrics["mae_deviation_pct"] > 20


def test_compute_slice_metrics_zero_global():
    """Test handling of zero global MAE."""
    from model_bias_detection import compute_slice_metrics

    y_true = np.array([50, 60])
    y_pred = np.array([50, 60])

    metrics = compute_slice_metrics(y_true, y_pred, 0.0)
    assert metrics["mae"] == 0.0
    assert not metrics["flagged"]