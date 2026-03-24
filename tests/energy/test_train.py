"""
Tests for the energy score ML training pipeline.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "energy"))


@pytest.fixture
def sample_dataframe():
    """Create a sample daily_joined DataFrame for testing."""
    np.random.seed(42)
    n = 200

    return pd.DataFrame({
        "user_id":                [f"user_{i:03d}" for i in np.random.randint(1, 20, n)],
        "date":                   pd.date_range("2025-01-01", periods=n, freq="D"),
        "sleep_hours":            np.random.normal(7, 1.2, n).clip(2, 14),
        "sleep_satisfaction":     np.random.uniform(0, 1, n),
        "rolling_sleep_hours_7d": np.random.normal(7, 0.8, n).clip(3, 12),
        "bedtime_variability_7d": np.random.uniform(10, 120, n),
        "avg_accuracy":           np.random.uniform(0.3, 1.0, n),
        "int_score":              np.random.uniform(20, 100, n),
        "rolling_int_7d":         np.random.uniform(25, 95, n),
        "bmr":                    np.random.randint(1200, 2200, n).astype(float),
        "attempts_count":         np.random.randint(0, 5, n).astype(float),
        "age":                    np.random.randint(14, 60, n).astype(float),
        "sex":                    np.random.choice(["Male", "Female"], n),
        "height":                 np.random.randint(150, 195, n).astype(float),
        "weight":                 np.random.randint(45, 120, n).astype(float),
    })


@pytest.fixture
def sample_parquet(tmp_path, sample_dataframe):
    """Save sample data as parquet and return the path."""
    path = tmp_path / "data" / "processed" / "daily_joined.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_dataframe.to_parquet(path, index=False)
    return path


def test_load_and_prepare_data(sample_parquet):
    """Test data loading and feature preparation."""
    from train_model import load_and_prepare_data

    with patch("train_model.DAILY_JOINED", sample_parquet):
        X, y, df, feature_names = load_and_prepare_data()

    assert len(X) > 0, "Should load data"
    assert len(X) == len(y), "X and y should have same length"
    assert len(feature_names) > 0, "Should have feature names"
    assert all(0 <= v <= 100 for v in y), "Target should be in [0, 100]"


def test_train_models_returns_valid_models():
    """Test that training produces valid models."""
    from train_model import train_models

    np.random.seed(42)
    X = np.random.rand(100, 8)
    y = np.random.rand(100) * 100

    models = train_models(X, y)
    assert "random_forest" in models, "Should train Random Forest"

    for name, (model, cv_scores) in models.items():
        assert hasattr(model, "predict"), f"{name} should have predict method"
        assert len(cv_scores) == 5, f"{name} should have 5 CV scores"
        assert all(s >= 0 for s in cv_scores), f"{name} CV scores should be positive"


def test_select_best_model():
    """Test model selection logic."""
    from train_model import select_best_model
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    X = np.random.rand(50, 5)
    y = np.random.rand(50) * 100
    rf.fit(X, y)

    models = {
        "model_a": (rf, np.array([5.0, 4.0, 6.0, 5.5, 4.5])),
        "model_b": (rf, np.array([3.0, 2.5, 3.5, 3.0, 2.8])),
    }

    name, model, mae = select_best_model(models)
    assert name == "model_b", "Should select model with lower MAE"
    assert mae < 4.0, "Should report correct MAE"


def test_predictions_in_range():
    """Test that model predictions are in a valid range."""
    from sklearn.ensemble import RandomForestRegressor

    np.random.seed(42)
    X = np.random.rand(100, 8)
    y = np.random.rand(100) * 100

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    assert all(p >= -10 for p in preds), "Predictions should not be extremely negative"
    assert all(p <= 110 for p in preds), "Predictions should not be extremely high"


def test_energy_score_range(sample_parquet):
    """Energy score target should always be clipped to [0, 100]."""
    from train_model import load_and_prepare_data

    with patch("train_model.DAILY_JOINED", sample_parquet):
        _, y, _, _ = load_and_prepare_data()

    assert y.min() >= 0.0, "Energy scores must be >= 0"
    assert y.max() <= 100.0, "Energy scores must be <= 100"