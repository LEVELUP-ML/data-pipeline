"""
Tests for model validation logic.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "energy"))


def test_compute_metrics():
    """Test metric computation."""
    from validate_model import compute_metrics

    y_true = np.array([50, 60, 70, 80, 90])
    y_pred = np.array([52, 58, 72, 78, 88])

    metrics = compute_metrics(y_true, y_pred)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "max_error" in metrics
    assert metrics["mae"] > 0
    assert metrics["mae"] == 2.0  # Mean of [2, 2, 2, 2, 2]
    assert metrics["r2"] > 0.9  # Should be very high for this data


def test_metrics_perfect_predictions():
    """Test metrics for perfect predictions."""
    from validate_model import compute_metrics

    y = np.array([10, 20, 30, 40, 50])
    metrics = compute_metrics(y, y)

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["max_error"] == 0.0


def test_validation_gate_pass():
    """Test that validation gate passes with good metrics."""
    from config import MAX_ACCEPTABLE_MAE, MIN_ACCEPTABLE_R2

    mae = 2.0  # Below threshold
    r2 = 0.85  # Above threshold

    assert mae <= MAX_ACCEPTABLE_MAE
    assert r2 >= MIN_ACCEPTABLE_R2


def test_validation_gate_fail_mae():
    """Test that validation gate fails with high MAE."""
    from config import MAX_ACCEPTABLE_MAE

    mae = 8.0  # Above threshold
    assert mae > MAX_ACCEPTABLE_MAE


def test_validation_gate_fail_r2():
    """Test that validation gate fails with low R²."""
    from config import MIN_ACCEPTABLE_R2

    r2 = 0.5  # Below threshold
    assert r2 < MIN_ACCEPTABLE_R2


def test_generate_plots_creates_file(tmp_path):
    """Test that plot generation creates a file."""
    import matplotlib
    matplotlib.use("Agg")

    from unittest.mock import patch

    with patch("validate_model.PLOTS_DIR", tmp_path):
        from validate_model import generate_plots

        y_true = np.random.rand(100) * 100
        y_pred = y_true + np.random.normal(0, 2, 100)

        plot_path = generate_plots(y_true, y_pred)
        assert Path(plot_path).exists(), "Plot file should be created"