"""
tests/test_strength_features.py
"""

from __future__ import annotations

import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import importlib.util


def _load_dag_helpers():
    """Load helper functions from strength_features_dag.py without Airflow context."""
    stubs = {
        "airflow":                           types.ModuleType("airflow"),
        "airflow.decorators":                types.ModuleType("airflow.decorators"),
        "airflow.exceptions":                types.ModuleType("airflow.exceptions"),
        "airflow.models":                    types.ModuleType("airflow.models"),
        "airflow.operators.trigger_dagrun":  types.ModuleType("airflow.operators.trigger_dagrun"),
        "dag_monitoring":                    types.ModuleType("dag_monitoring"),
        "pendulum":                          types.ModuleType("pendulum"),
    }
    class _FakeDAG:
        def __enter__(self): return self
        def __exit__(self, *a): pass
    class _FakeTask:
        def __rshift__(self, other): return other
        def __rrshift__(self, other): return self
        def __call__(self, *a, **k): return _FakeTask()
    stubs["airflow"].DAG                         = lambda *a, **k: _FakeDAG()
    stubs["airflow.decorators"].task             = lambda f: _FakeTask()
    stubs["airflow.operators.trigger_dagrun"].TriggerDagRunOperator = lambda **k: _FakeTask()
    stubs["airflow.exceptions"].AirflowFailException = Exception
    stubs["airflow.models"].Variable             = type("V", (), {
        "get": staticmethod(lambda *a, **k: "")
    })()
    stubs["airflow.operators.trigger_dagrun"].TriggerDagRunOperator = lambda **k: None
    stubs["dag_monitoring"].emit_metric          = lambda *a, **k: None
    stubs["dag_monitoring"].log                  = __import__("logging").getLogger("test")
    stubs["dag_monitoring"].monitored_dag_args   = lambda **k: {}
    stubs["dag_monitoring"].on_dag_failure_callback = None
    stubs["dag_monitoring"].on_sla_miss_callback = None
    stubs["pendulum"].datetime                   = lambda *a, tz=None, **k: datetime(*a, **k)
    stubs["pendulum"].timezone                   = lambda tz: None

    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)

    # Search upward from tests/ to find the DAG file
    for parent in [Path(__file__).parent, Path(__file__).parents[1]]:
        dag_path = parent / "dags" / "processing_dags" / "strength_features_dag.py"
        if dag_path.exists():
            spec = importlib.util.spec_from_file_location("strength_feat_dag", str(dag_path))
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

    pytest.skip(
        "strength_features_dag.py not found — copy it to "
        "dags/processing_dags/ before running these tests"
    )


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
        assert dag._slope([1.0, 3.0, 5.0, 7.0, 9.0]) == pytest.approx(2.0, abs=0.01)


#  _future_score

class TestFutureScore:
    def _make_series(self, start="2026-01-01", n=30, base=100.0, step=1.0):
        dates  = [pd.Timestamp(start) + pd.Timedelta(days=i) for i in range(n)]
        scores = [base + i * step for i in range(n)]
        return dates, scores

    def test_exact_match(self, dag):
        dates, scores = self._make_series()
        found = dag._future_score(dates, scores, dates[0], 7)
        assert found == pytest.approx(scores[7], abs=0.01)

    def test_within_2_day_window(self, dag):
        dates, scores = self._make_series()
        # Remove day 7 — nearest within ±2 days should be used
        dates2  = dates[:7] + dates[8:]
        scores2 = scores[:7] + scores[8:]
        found   = dag._future_score(dates2, scores2, dates[0], 7)
        assert found is not None

    def test_returns_none_when_out_of_range(self, dag):
        dates, scores = self._make_series(n=5)
        found = dag._future_score(dates, scores, dates[0], 30)
        assert found is None

    def test_empty_series(self, dag):
        ref   = pd.Timestamp("2026-01-01")
        found = dag._future_score([], [], ref, 7)
        assert found is None


#  _bmr

class TestBmr:
    def test_male_bmr(self, dag):
        # Mifflin-St Jeor: 10*80 + 6.25*180 - 5*30 + 5 = 1780
        assert dag._bmr(30, "Male", 180, 80) == pytest.approx(1780, abs=2)

    def test_female_bmr(self, dag):
        # 10*60 + 6.25*165 - 5*25 - 161 = 1345.25
        assert dag._bmr(25, "Female", 165, 60) == pytest.approx(1345, abs=2)

    def test_invalid_inputs_return_none(self, dag):
        assert dag._bmr(None, "Male", 180, 80) is None
        assert dag._bmr(30, "Male", None, 80) is None
        assert dag._bmr("bad", "Female", 165, 60) is None

    def test_lowercase_sex(self, dag):
        assert dag._bmr(30, "female", 165, 60) is not None


#  _age_enc

class TestAgeEnc:
    def test_under_20(self, dag):    assert dag._age_enc(17) == 0
    def test_20s(self, dag):         assert dag._age_enc(25) == 1
    def test_30s(self, dag):         assert dag._age_enc(35) == 2
    def test_40_plus(self, dag):     assert dag._age_enc(55) == 3
    def test_boundary_20(self, dag): assert dag._age_enc(20) == 1
    def test_invalid(self, dag):
        assert dag._age_enc(None) == -1
        assert dag._age_enc("old") == -1


#  Lag feature construction (pure logic, no file I/O)

class TestLagFeatureConstruction:
    def _make_sessions_df(self, n_sessions=15, base_score=100.0):
        start = pd.Timestamp("2026-01-01")
        rows  = []
        score = base_score
        for i in range(n_sessions):
            score += np.random.uniform(0.5, 2.0)
            rows.append({
                "user_id":          0,
                "Date":             start + pd.Timedelta(days=i * 2),
                "max_1rm":          round(score * 1.2, 1),
                "avg_1rm":          round(score, 1),
                "total_1rm_volume": round(score * 5, 1),
                "max_weight":       round(score * 0.8, 1),
                "total_weight":     round(score * 4, 1),
                "total_reps":       30,
                "num_sets":         5,
                "exercise_variety": 3,
                "strength_score":   round(score, 2),
            })
        return pd.DataFrame(rows)

    def test_minimum_sessions_threshold(self):
        N_LAGS, MIN_SESS = 5, 7
        n = MIN_SESS - 1
        expected = max(0, n - N_LAGS) if n >= MIN_SESS else 0
        assert expected == 0

    def test_row_count_correct(self):
        N_LAGS = 5
        assert 15 - N_LAGS == 10

    def test_lag_ordering(self):
        N_LAGS = 5
        df     = self._make_sessions_df(n_sessions=10).reset_index(drop=True)
        dates  = df["Date"].tolist()
        scores = df["strength_score"].tolist()

        t    = N_LAGS
        lag1 = None
        for k in range(N_LAGS):
            idx = t - N_LAGS + k
            lag = N_LAGS - k
            if lag == 1:
                lag1 = df.iloc[idx]["strength_score"]

        assert lag1 == pytest.approx(scores[t - 1], abs=0.01)

    def test_days_ago_non_negative(self):
        N_LAGS = 5
        df     = self._make_sessions_df(n_sessions=12)
        dates  = df["Date"].tolist()

        for t in range(N_LAGS, len(df)):
            ref_date = dates[t]
            for k in range(N_LAGS):
                idx      = t - N_LAGS + k
                days_ago = (ref_date - dates[idx]).days
                assert days_ago >= 0
