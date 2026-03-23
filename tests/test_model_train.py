"""
tests/test_model_train.py

Unit tests for model_train.py — data splits, feature prep,
metric functions, sensitivity, and the training gate.
All tests run without Firestore, GCS, or MLflow.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


#  Stub optional heavy deps so tests run without GPU/mlflow 

def _stub_optional():
    for name in ("mlflow", "mlflow.sklearn"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.set_tracking_uri     = lambda *a, **k: None
            mod.set_experiment       = lambda *a, **k: None
            mod.start_run            = lambda **k: MagicMock(__enter__=lambda s, *a: s,
                                                             __exit__=lambda *a: None)
            mod.log_param = mod.log_params = mod.log_metric = mod.log_dict = \
                mod.log_artifact = lambda *a, **k: None
            mod.end_run              = lambda: None
            sys.modules[name]        = mod

    for name in ("fairlearn", "fairlearn.metrics"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

_stub_optional()


#  Load model_train module 

import importlib.util

def _load_model_train():
    path = Path(__file__).parents[1] / "scripts" / "model_train.py"
    if not path.exists():
        pytest.skip(f"model_train.py not found: {path}")
    spec = importlib.util.spec_from_file_location("model_train", str(path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mt():
    return _load_model_train()


#  Fixtures 

def _make_features(n_users=5, n_days=40, seed=42) -> pd.DataFrame:
    """
    Generate a synthetic features DataFrame matching the schema produced
    by flexibility_features_dag.py, with no Firestore dependency.
    """
    rng    = np.random.default_rng(seed)
    rows   = []
    start  = pd.Timestamp("2026-01-01")

    for u in range(n_users):
        base_score = rng.uniform(30, 70)
        for d in range(n_days):
            score = float(np.clip(base_score + rng.normal(0, 1) * d * 0.1, 0, 100))
            row   = {
                "user_id":          f"u{u:03d}",
                "ref_date":         start + pd.Timedelta(days=d),
                "ref_score":        score,
                "mean_score_5":     score + rng.normal(0, 0.5),
                "score_trend_5":    rng.uniform(-0.5, 0.5),
                "workout_count_7d": int(rng.integers(1, 5)),
                "workout_count_14d":int(rng.integers(2, 9)),
                "days_since_last":  int(rng.integers(1, 4)),
                "current_streak":   int(rng.integers(1, 8)),
                "mean_effort_5":    rng.uniform(2, 4),
                "age":              float(rng.integers(18, 55)),
                "sex_encoded":      float(rng.integers(0, 2)),
                "bmr":              float(rng.integers(1400, 2200)),
                "age_bucket_enc":   float(rng.integers(0, 4)),
                "sex_raw":          "Male" if rng.random() > 0.5 else "Female",
                "age_raw":          float(rng.integers(18, 55)),
            }
            # Lag features
            for lag in range(1, 6):
                row[f"score_lag_{lag}"]    = float(np.clip(score + rng.normal(0, 1), 0, 100))
                row[f"effort_lag_{lag}"]   = float(rng.integers(1, 6))
                row[f"duration_lag_{lag}"] = float(rng.integers(20, 90))
                row[f"reach_lag_{lag}"]    = float(np.clip(10 + score * 0.4 + rng.normal(0, 1), 2, 55))
                row[f"days_ago_lag_{lag}"] = float(rng.integers(1, 14))
            # Targets — last 5 days per user have no d14 target (realistic)
            row["target_d1"]  = score + rng.normal(0.5, 0.5) if d < n_days - 1  else np.nan
            row["target_d3"]  = score + rng.normal(1.0, 0.8) if d < n_days - 3  else np.nan
            row["target_d7"]  = score + rng.normal(1.5, 1.2) if d < n_days - 7  else np.nan
            row["target_d14"] = score + rng.normal(2.0, 1.8) if d < n_days - 14 else np.nan
            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def features_df():
    return _make_features()


#  time_split 

class TestTimeSplit:
    def test_no_overlap(self, mt, features_df):
        train, test = mt.time_split(features_df)
        assert len(set(train["ref_date"]) & set(test["ref_date"])) == 0

    def test_train_before_test(self, mt, features_df):
        train, test = mt.time_split(features_df)
        assert train["ref_date"].max() < test["ref_date"].min()

    def test_approximate_split_ratio(self, mt, features_df):
        train, test = mt.time_split(features_df)
        total = len(train) + len(test)
        test_ratio = len(test) / total
        assert 0.15 <= test_ratio <= 0.30

    def test_no_data_lost(self, mt, features_df):
        train, test = mt.time_split(features_df)
        assert len(train) + len(test) == len(features_df)


#  get_feature_cols 

class TestGetFeatureCols:
    def test_excludes_identifiers(self, mt, features_df):
        cols = mt.get_feature_cols(features_df)
        for excluded in ("user_id", "ref_date", "ref_score", "sex_raw", "age_raw"):
            assert excluded not in cols

    def test_excludes_targets(self, mt, features_df):
        cols = mt.get_feature_cols(features_df)
        for h in [1, 3, 7, 14]:
            assert f"target_d{h}" not in cols

    def test_includes_lag_features(self, mt, features_df):
        cols = mt.get_feature_cols(features_df)
        assert "score_lag_1" in cols
        assert "effort_lag_3" in cols
        assert "duration_lag_5" in cols

    def test_includes_profile_features(self, mt, features_df):
        cols = mt.get_feature_cols(features_df)
        for f in ("age", "sex_encoded", "bmr", "age_bucket_enc"):
            assert f in cols


#  prepare 

class TestPrepare:
    def test_drops_null_targets(self, mt, features_df):
        feature_cols = mt.get_feature_cols(features_df)
        X, Y = mt.prepare(features_df, feature_cols)
        assert Y.isna().sum().sum() == 0

    def test_x_y_same_length(self, mt, features_df):
        feature_cols = mt.get_feature_cols(features_df)
        X, Y = mt.prepare(features_df, feature_cols)
        assert len(X) == len(Y)

    def test_no_nan_in_x(self, mt, features_df):
        feature_cols = mt.get_feature_cols(features_df)
        X, Y = mt.prepare(features_df, feature_cols)
        assert X.isna().sum().sum() == 0

    def test_correct_target_columns(self, mt, features_df):
        feature_cols = mt.get_feature_cols(features_df)
        _, Y = mt.prepare(features_df, feature_cols)
        assert list(Y.columns) == ["target_d1", "target_d3", "target_d7", "target_d14"]


#  rmse 

class TestRmse:
    def test_perfect_predictions(self, mt):
        y = np.array([1.0, 2.0, 3.0])
        assert mt.rmse(y, y) == pytest.approx(0.0, abs=1e-8)

    def test_known_value(self, mt):
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([2.0, 2.0, 2.0, 2.0])
        assert mt.rmse(y_true, y_pred) == pytest.approx(2.0, abs=1e-6)

    def test_always_non_negative(self, mt):
        rng    = np.random.default_rng(0)
        y_true = rng.normal(50, 10, 100)
        y_pred = rng.normal(50, 12, 100)
        assert mt.rmse(y_true, y_pred) >= 0


#  hyperparam_sensitivity 

class TestHyperparamSensitivity:
    def _make_cv_results(self, n=30, seed=0):
        rng = np.random.default_rng(seed)
        # n_estimators positively correlated with score
        n_est  = rng.integers(100, 400, n).astype(float)
        score  = -5.0 + n_est * 0.01 + rng.normal(0, 0.5, n)
        return {
            "param_estimator__n_estimators": n_est,
            "param_estimator__max_depth":    rng.integers(3, 7, n).astype(float),
            "mean_test_score":               score,
        }

    def test_returns_correlations_dict(self, mt):
        cv = self._make_cv_results()
        result = mt.hyperparam_sensitivity(cv)
        assert "correlations" in result
        assert isinstance(result["correlations"], dict)

    def test_positive_correlation_detected(self, mt):
        cv     = self._make_cv_results()
        result = mt.hyperparam_sensitivity(cv)
        # n_estimators is positively correlated in our fake data
        assert result["correlations"].get("n_estimators", -1) > 0

    def test_handles_empty_input(self, mt):
        result = mt.hyperparam_sensitivity({})
        assert "available" in result or "correlations" in result


#  Full training smoke test (fast, in-memory) 

class TestTrainSmoke:
    """
    Runs the full train() function on synthetic data without hitting
    Firestore, GCS, or MLflow. Uses tmp_path for file I/O.
    """

    def test_train_produces_artifacts(self, mt, features_df, tmp_path, monkeypatch):
        # Point file paths to tmp_path
        monkeypatch.setattr(mt, "FEATURES_PATH", tmp_path / "flexibility_features.parquet")
        monkeypatch.setattr(mt, "MODELS_DIR",    tmp_path / "models")
        monkeypatch.setattr(mt, "MODEL_PATH",    tmp_path / "models" / "model.pkl")
        monkeypatch.setattr(mt, "METRICS_PATH",  tmp_path / "models" / "metrics.json")
        monkeypatch.setattr(mt, "BIAS_PATH",     tmp_path / "models" / "bias_report.json")
        monkeypatch.setattr(mt, "SHAP_PATH",     tmp_path / "models" / "shap_summary.png")
        monkeypatch.setenv("MODEL_RMSE_THRESHOLD", "999")  # always pass gate

        features_df.to_parquet(str(tmp_path / "flexibility_features.parquet"))

        result = mt.train(run_id="test_run")

        assert (tmp_path / "models" / "model.pkl").exists()
        assert (tmp_path / "models" / "metrics.json").exists()
        assert (tmp_path / "models" / "bias_report.json").exists()

        metrics = json.loads((tmp_path / "models" / "metrics.json").read_text())
        assert "test_metrics" in metrics
        assert "model_comparison" in metrics
        assert "winner_model" in metrics
        assert metrics["winner_model"] in ("Ridge", "Random Forest", "XGBoost")
        for h in [1, 3, 7, 14]:
            assert f"d{h}" in metrics["test_metrics"]

    @pytest.mark.slow
    def test_gate_fails_on_tight_threshold(self, mt, features_df, tmp_path, monkeypatch):
        monkeypatch.setattr(mt, "FEATURES_PATH", tmp_path / "flexibility_features.parquet")
        monkeypatch.setattr(mt, "MODELS_DIR",    tmp_path / "models")
        monkeypatch.setattr(mt, "MODEL_PATH",    tmp_path / "models" / "model.pkl")
        monkeypatch.setattr(mt, "METRICS_PATH",  tmp_path / "models" / "metrics.json")
        monkeypatch.setattr(mt, "BIAS_PATH",     tmp_path / "models" / "bias_report.json")
        monkeypatch.setattr(mt, "SHAP_PATH",     tmp_path / "models" / "shap_summary.png")
        monkeypatch.setenv("MODEL_RMSE_THRESHOLD", "0.0001")  # impossibly tight

        features_df.to_parquet(str(tmp_path / "flexibility_features.parquet"))

        with pytest.raises(ValueError, match="gate FAILED"):
            mt.train(run_id="test_tight")