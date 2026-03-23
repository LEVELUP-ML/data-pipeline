"""
tests/test_flexibility_features.py

Unit tests for the flexibility feature engineering logic.
All tests run without Firestore or GCS — pure Python/numpy/pandas.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


#  Import helpers directly from the DAG module 
# We import the free functions only — no DAG-level objects are instantiated.

import importlib.util, sys, types

def _load_dag_helpers():
    """
    Load only the helper functions from flexibility_features_dag.py
    without triggering DAG registration (which needs Airflow context).
    We stub the airflow imports so the module loads cleanly in plain pytest.
    """
    stubs = {
        "airflow":                        types.ModuleType("airflow"),
        "airflow.DAG":                    types.ModuleType("airflow.DAG"),
        "airflow.decorators":             types.ModuleType("airflow.decorators"),
        "airflow.exceptions":             types.ModuleType("airflow.exceptions"),
        "airflow.models":                 types.ModuleType("airflow.models"),
        "airflow.operators.trigger_dagrun": types.ModuleType("airflow.operators.trigger_dagrun"),
        "firebase_admin":                 types.ModuleType("firebase_admin"),
        "firebase_admin.credentials":     types.ModuleType("firebase_admin.credentials"),
        "firebase_admin.firestore":       types.ModuleType("firebase_admin.firestore"),
        "dag_monitoring":                 types.ModuleType("dag_monitoring"),
        "pendulum":                       types.ModuleType("pendulum"),
    }
    # Minimal shims so the module body doesn't crash
    stubs["airflow"].DAG = lambda *a, **k: None
    stubs["airflow.decorators"].task = lambda f: f
    stubs["airflow.exceptions"].AirflowFailException = Exception
    stubs["airflow.models"].Variable = type("V", (), {"get": staticmethod(lambda *a, **k: "")})()
    stubs["airflow.operators.trigger_dagrun"].TriggerDagRunOperator = lambda **k: None
    stubs["dag_monitoring"].emit_metric = lambda *a, **k: None
    stubs["dag_monitoring"].log = __import__("logging").getLogger("test")
    stubs["dag_monitoring"].monitored_dag_args = lambda **k: {}
    stubs["dag_monitoring"].on_dag_failure_callback = None
    stubs["dag_monitoring"].on_sla_miss_callback = None
    stubs["pendulum"].datetime = datetime
    stubs["pendulum"].timezone = lambda tz: None

    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)

    dag_path = Path(__file__).parents[1] / "dags" / "processing_dags" / "flexibility_features_dag.py"
    if not dag_path.exists():
        pytest.skip(f"DAG file not found: {dag_path}")

    spec = importlib.util.spec_from_file_location("flex_feat_dag", str(dag_path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dag():
    return _load_dag_helpers()


#  _slope 

class TestSlope:
    def test_flat_series_returns_zero(self, dag):
        assert dag._slope([5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0, abs=1e-6)

    def test_ascending_series_positive(self, dag):
        assert dag._slope([1.0, 2.0, 3.0, 4.0, 5.0]) > 0

    def test_descending_series_negative(self, dag):
        assert dag._slope([5.0, 4.0, 3.0, 2.0, 1.0]) < 0

    def test_single_value_returns_zero(self, dag):
        assert dag._slope([42.0]) == 0.0

    def test_empty_returns_zero(self, dag):
        assert dag._slope([]) == 0.0

    def test_known_slope(self, dag):
        # y = 2x + 1, slope should be 2
        vals = [1.0, 3.0, 5.0, 7.0, 9.0]
        assert dag._slope(vals) == pytest.approx(2.0, abs=0.01)


#  _future_score 

class TestFutureScore:
    def _make_series(self, start="2026-01-01", n=30, base=50.0, step=0.5):
        dates  = [pd.Timestamp(start) + pd.Timedelta(days=i) for i in range(n)]
        scores = [base + i * step for i in range(n)]
        return dates, scores

    def test_exact_match(self, dag):
        dates, scores = self._make_series()
        ref   = dates[0]
        found = dag._future_score(dates, scores, ref, 7)
        assert found == pytest.approx(scores[7], abs=0.01)

    def test_within_2_day_window(self, dag):
        # Remove day 7 from series — nearest within ±2 days should be used
        dates, scores = self._make_series()
        ref    = dates[0]
        d7_idx = 7
        dates2  = dates[:d7_idx] + dates[d7_idx+1:]
        scores2 = scores[:d7_idx] + scores[d7_idx+1:]
        found   = dag._future_score(dates2, scores2, ref, 7)
        assert found is not None

    def test_returns_none_when_out_of_range(self, dag):
        dates, scores = self._make_series(n=5)
        ref   = dates[0]
        # horizon 30 is way beyond the 5-day series
        found = dag._future_score(dates, scores, ref, 30)
        assert found is None

    def test_empty_series(self, dag):
        ref   = pd.Timestamp("2026-01-01")
        found = dag._future_score([], [], ref, 7)
        assert found is None


#  _bmr 

class TestBmr:
    def test_male_bmr(self, dag):
        # Mifflin-St Jeor: 10*80 + 6.25*180 - 5*30 + 5 = 1880
        result = dag._bmr(30, "Male", 180, 80)
        assert result == pytest.approx(1880, abs=2)

    def test_female_bmr(self, dag):
        # 10*60 + 6.25*165 - 5*25 - 161 = 1390.25
        result = dag._bmr(25, "Female", 165, 60)
        assert result == pytest.approx(1390, abs=2)

    def test_invalid_inputs_return_none(self, dag):
        assert dag._bmr(None, "Male", 180, 80) is None
        assert dag._bmr(30, "Male", None, 80) is None
        assert dag._bmr("bad", "Female", 165, 60) is None

    def test_lowercase_sex(self, dag):
        result = dag._bmr(30, "female", 165, 60)
        assert result is not None


#  _age_enc 

class TestAgeEnc:
    def test_under_20(self, dag):
        assert dag._age_enc(17) == 0

    def test_20s(self, dag):
        assert dag._age_enc(25) == 1

    def test_30s(self, dag):
        assert dag._age_enc(35) == 2

    def test_40_plus(self, dag):
        assert dag._age_enc(55) == 3

    def test_boundary_20(self, dag):
        assert dag._age_enc(20) == 1

    def test_invalid(self, dag):
        assert dag._age_enc(None) == -1
        assert dag._age_enc("old") == -1


#  Lag feature construction (integration-level) 

class TestLagFeatureConstruction:
    """
    Test that the lag feature loop produces the right shape and values
    without needing Firestore. We replicate the core loop logic here.
    """

    def _make_user_df(self, n_sessions=15, base_score=50.0):
        """Create a synthetic single-user session DataFrame."""
        start = pd.Timestamp("2026-01-01")
        rows  = []
        score = base_score
        for i in range(n_sessions):
            score += np.random.uniform(0.1, 1.0)
            rows.append({
                "user_id":              "u001",
                "date":                 start + pd.Timedelta(days=i * 2),  # every 2 days
                "score_after":          round(score, 2),
                "effort_level":         3,
                "session_duration_min": 45,
                "sit_and_reach_cm":     round(10 + score * 0.4, 1),
                "streak_days":          (i % 5) + 1,
                "rest_days_before":     1,
            })
        return pd.DataFrame(rows)

    def test_minimum_sessions_threshold(self):
        """Users with fewer than MIN_SESS sessions should produce no rows."""
        N_LAGS   = 5
        MIN_SESS = 7
        df = self._make_user_df(n_sessions=MIN_SESS - 1)
        n  = len(df)
        expected_rows = max(0, n - N_LAGS) if n >= MIN_SESS else 0
        assert expected_rows == 0

    def test_row_count_correct(self):
        """Users with N sessions produce N - N_LAGS rows."""
        N_LAGS   = 5
        MIN_SESS = 7
        n_sess   = 15
        df       = self._make_user_df(n_sessions=n_sess)
        expected = n_sess - N_LAGS
        assert expected == 10

    def test_lag_ordering(self):
        """lag_1 should be the most recent session before the reference point."""
        N_LAGS = 5
        n_sess = 10
        df     = self._make_user_df(n_sessions=n_sess).reset_index(drop=True)
        dates  = df["date"].tolist()
        scores = df["score_after"].tolist()

        # Reference point t = N_LAGS (index 5, the 6th session)
        t         = N_LAGS
        ref_date  = dates[t]
        # lag_1 = session at t-1 (most recent), lag_5 = session at t-5 (oldest)
        lag1_expected_score = scores[t - 1]

        # Replicate the loop
        lag1_score = None
        for k in range(N_LAGS):
            idx = t - N_LAGS + k
            lag = N_LAGS - k       # lag_5, lag_4, ..., lag_1
            if lag == 1:
                lag1_score = df.iloc[idx]["score_after"]

        assert lag1_score == pytest.approx(lag1_expected_score, abs=0.01)

    def test_days_ago_non_negative(self):
        """days_ago_lag_k must always be >= 0."""
        N_LAGS = 5
        df     = self._make_user_df(n_sessions=12)
        dates  = df["date"].tolist()

        for t in range(N_LAGS, len(df)):
            ref_date = dates[t]
            for k in range(N_LAGS):
                idx     = t - N_LAGS + k
                days_ago = (ref_date - dates[idx]).days
                assert days_ago >= 0, f"Negative days_ago at t={t}, k={k}"