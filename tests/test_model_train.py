"""
tests/test_model_train.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _stub_optional():
    for name in ("mlflow", "mlflow.sklearn"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.set_tracking_uri     = lambda *a, **k: None
            mod.set_experiment       = lambda *a, **k: None
            mod.start_run            = lambda **k: MagicMock(
                __enter__=lambda s, *a: s, __exit__=lambda *a: None
            )
            mod.log_param = mod.log_params = mod.log_metric = mod.log_dict = \
                mod.log_artifact = lambda *a, **k: None
            mod.end_run = lambda: None
            sys.modules[name] = mod
    for name in ("fairlearn", "fairlearn.metrics"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


_stub_optional()

import importlib.util


def _load_model_train():
    for parent in [Path(__file__).parent, Path(__file__).parents[1]]:
        path = parent / "scripts" / "model_train.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location("model_train", str(path))
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    pytest.skip("model_train.py not found in scripts/")


@pytest.fixture(scope="module")
def mt():
    return _load_model_train()


def _make_features(n_users=6, n_days=90, seed=42) -> pd.DataFrame:
    """
    All targets explicitly populated — no NaN anywhere.
    n_users=6, n_days=90 → 540 rows.
    After 80/20 time split: ~432 train, ~108 test.
    Both splits have ample rows after prepare() (which only drops NaN targets).
    """
    rng   = np.random.default_rng(seed)
    rows  = []
    start = pd.Timestamp("2026-01-01")

    for u in range(n_users):
        base = rng.uniform(30, 70)
        for d in range(n_days):
            score = float(np.clip(base + d * 0.05 + rng.normal(0, 0.5), 0, 100))
            row   = {
                "user_id":          f"u{u:03d}",
                "ref_date":         start + pd.Timedelta(days=d),
                "ref_score":        score,
                "mean_score_5":     score + rng.normal(0, 0.3),
                "score_trend_5":    rng.uniform(-0.3, 0.3),
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
            for lag in range(1, 6):
                row[f"score_lag_{lag}"]    = float(np.clip(score + rng.normal(0, 1), 0, 100))
                row[f"effort_lag_{lag}"]   = float(rng.integers(1, 6))
                row[f"duration_lag_{lag}"] = float(rng.integers(20, 90))
                row[f"reach_lag_{lag}"]    = float(np.clip(10 + score * 0.4, 2, 55))
                row[f"days_ago_lag_{lag}"] = float(rng.integers(1, 14))

            # All targets fully populated — no NaN
            row["target_d1"]  = float(np.clip(score + rng.normal(0.3, 0.3), 0, 100))
            row["target_d3"]  = float(np.clip(score + rng.normal(0.6, 0.5), 0, 100))
            row["target_d7"]  = float(np.clip(score + rng.normal(1.0, 0.8), 0, 100))
            row["target_d14"] = float(np.clip(score + rng.normal(1.5, 1.0), 0, 100))
            rows.append(row)

    df = pd.DataFrame(rows)
    assert df[["target_d1","target_d3","target_d7","target_d14"]].isna().sum().sum() == 0
    return df


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
        ratio = len(test) / (len(train) + len(test))
        assert 0.15 <= ratio <= 0.30

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


#  prepare ─

class TestPrepare:
    def test_drops_null_targets(self, mt, features_df):
        X, Y = mt.prepare(features_df, mt.get_feature_cols(features_df))
        assert Y.isna().sum().sum() == 0

    def test_x_y_same_length(self, mt, features_df):
        X, Y = mt.prepare(features_df, mt.get_feature_cols(features_df))
        assert len(X) == len(Y)

    def test_no_nan_in_x(self, mt, features_df):
        X, Y = mt.prepare(features_df, mt.get_feature_cols(features_df))
        assert X.isna().sum().sum() == 0

    def test_correct_target_columns(self, mt, features_df):
        _, Y = mt.prepare(features_df, mt.get_feature_cols(features_df))
        assert list(Y.columns) == ["target_d1", "target_d3", "target_d7", "target_d14"]


#  rmse 

class TestRmse:
    def test_perfect_predictions(self, mt):
        y = np.array([1.0, 2.0, 3.0])
        assert mt.rmse(y, y) == pytest.approx(0.0, abs=1e-8)

    def test_known_value(self, mt):
        assert mt.rmse(
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([2.0, 2.0, 2.0, 2.0])
        ) == pytest.approx(2.0, abs=1e-6)

    def test_always_non_negative(self, mt):
        rng = np.random.default_rng(0)
        assert mt.rmse(rng.normal(50, 10, 100), rng.normal(50, 12, 100)) >= 0


#  hyperparam_sensitivity 

class TestHyperparamSensitivity:
    def _cv(self, n=30, seed=0):
        rng   = np.random.default_rng(seed)
        n_est = rng.integers(100, 400, n).astype(float)
        return {
            "param_estimator__n_estimators": n_est,
            "param_estimator__max_depth":    rng.integers(3, 7, n).astype(float),
            "mean_test_score":               -5.0 + n_est * 0.01 + rng.normal(0, 0.5, n),
        }

    def test_returns_correlations_dict(self, mt):
        assert "correlations" in mt.hyperparam_sensitivity(self._cv())

    def test_positive_correlation_detected(self, mt):
        assert mt.hyperparam_sensitivity(self._cv())["correlations"].get("n_estimators", -1) > 0

    def test_handles_empty_input(self, mt):
        result = mt.hyperparam_sensitivity({})
        assert "available" in result or "correlations" in result


#  smoke tests ─

class TestTrainSmoke:
    """
    Patches load_data() directly so train() uses our in-memory DataFrame
    regardless of what FEATURES_PATH points to.
    """

    def _run_train(self, mt, tmp_path, df, rmse_threshold="999"):
        models_dir = tmp_path / "models" / "flexibility"
        models_dir.mkdir(parents=True)

        import os
        os.environ["MODEL_RMSE_THRESHOLD"] = rmse_threshold

        with patch.object(mt, "load_data", return_value=df), \
             patch.object(mt, "MODELS_DIR",   models_dir), \
             patch.object(mt, "MODEL_PATH",   models_dir / "model.pkl"), \
             patch.object(mt, "METRICS_PATH", models_dir / "metrics.json"), \
             patch.object(mt, "BIAS_PATH",    models_dir / "bias_report.json"), \
             patch.object(mt, "SHAP_PATH",    models_dir / "shap_summary.png"):
            return mt.train(run_id="test_run"), models_dir

    def test_train_produces_artifacts(self, mt, tmp_path):
        df = _make_features()
        result, models_dir = self._run_train(mt, tmp_path, df)

        assert (models_dir / "model.pkl").exists()
        assert (models_dir / "metrics.json").exists()
        assert (models_dir / "bias_report.json").exists()

        metrics = json.loads((models_dir / "metrics.json").read_text())
        assert "test_metrics" in metrics
        assert "model_comparison" in metrics
        assert metrics["winner_model"] in ("Ridge", "Random Forest", "XGBoost")
        for h in [1, 3, 7, 14]:
            assert f"d{h}" in metrics["test_metrics"]

    @pytest.mark.slow
    def test_gate_fails_on_tight_threshold(self, mt, tmp_path):
        df = _make_features()
        with pytest.raises(ValueError, match="gate FAILED"):
            self._run_train(mt, tmp_path / "tight", df, rmse_threshold="0.0001")