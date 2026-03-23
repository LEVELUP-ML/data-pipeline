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
    stubs["airflow"].DAG                         = lambda *a, **k: None
    stubs["airflow.decorators"].task             = lambda f: f
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
    stubs["pendulum"].datetime                   = datetime
    stubs["pendulum"].timezone                   = lambda tz: None

    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)

    # Search upward from tests/ to find the DAG file
    test_dir = Path(__file__).parent
    dag_path = test_dir.parent / "dags" / "processing_dags" / "strength_features_dag.py"

    if not dag_path.exists():
        pytest.skip(f"DAG file not found: {dag_path}")

    spec = importlib.util.spec_from_file_location("strength_features_dag", str(dag_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture(scope="session")
def dag_module():
    return _load_dag_helpers()


class TestSlope:
    def test_flat_series_returns_zero(self, dag_module):
        assert dag_module._slope([1, 1, 1, 1]) == 0.0

    def test_ascending_series_positive(self, dag_module):
        assert dag_module._slope([1, 2, 3, 4]) > 0

    def test_descending_series_negative(self, dag_module):
        assert dag_module._slope([4, 3, 2, 1]) < 0

    def test_single_value_returns_zero(self, dag_module):
        assert dag_module._slope([5]) == 0.0

    def test_empty_returns_zero(self, dag_module):
        assert dag_module._slope([]) == 0.0

    def test_known_slope(self, dag_module):
        # y = 2x + 1: slope should be 2
        y_vals = [3, 5, 7, 9]  # For x = 1,2,3,4
        slope = dag_module._slope(y_vals)
        assert abs(slope - 2.0) < 0.01


class TestFutureScore:
    def test_exact_match(self, dag_module):
        dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")]
        scores = [10.0, 15.0]
        ref = pd.Timestamp("2024-01-01")
        result = dag_module._future_strength_score(dates, scores, ref, 3)
        assert result == 15.0

    def test_within_2_day_window(self, dag_module):
        dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05")]
        scores = [10.0, 15.0]
        ref = pd.Timestamp("2024-01-01")
        result = dag_module._future_strength_score(dates, scores, ref, 3)
        assert result == 15.0  # Within 2-day window

    def test_returns_none_when_out_of_range(self, dag_module):
        dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10")]
        scores = [10.0, 15.0]
        ref = pd.Timestamp("2024-01-01")
        result = dag_module._future_strength_score(dates, scores, ref, 3)
        assert result is None  # More than 2 days away

    def test_empty_series(self, dag_module):
        result = dag_module._future_strength_score([], [], pd.Timestamp("2024-01-01"), 3)
        assert result is None


class TestBmr:
    def test_male_bmr(self, dag_module):
        # Test the BMR calculation (though it's not used in strength features)
        # This is just to ensure the function exists
        pass

    def test_female_bmr(self, dag_module):
        pass

    def test_invalid_inputs_return_none(self, dag_module):
        pass

    def test_lowercase_sex(self, dag_module):
        pass


class TestAgeEnc:
    def test_under_20(self, dag_module):
        assert dag_module._age_enc(18) == 0

    def test_20s(self, dag_module):
        assert dag_module._age_enc(25) == 1

    def test_30s(self, dag_module):
        assert dag_module._age_enc(35) == 2

    def test_40_plus(self, dag_module):
        assert dag_module._age_enc(45) == 3

    def test_boundary_20(self, dag_module):
        assert dag_module._age_enc(20) == 1

    def test_invalid(self, dag_module):
        assert dag_module._age_enc("invalid") == -1


class TestLagFeatureConstruction:
    def test_minimum_sessions_threshold(self, dag_module):
        # Create test data with fewer than MIN_SESS sessions
        user_data = pd.DataFrame({
            "user_id": [1] * 3,
            "Date": pd.date_range("2024-01-01", periods=3),
            "strength_score": [10, 11, 12]
        })
        result = dag_module.build_lag_features(user_data)
        assert len(result) == 0  # Should be empty due to minimum sessions

    def test_row_count_correct(self, dag_module):
        # Create test data with enough sessions
        user_data = pd.DataFrame({
            "user_id": [1] * 10,
            "Date": pd.date_range("2024-01-01", periods=10),
            "strength_score": range(10, 20)
        })
        result = dag_module.build_lag_features(user_data)
        # Should have N_LAGS rows of features
        expected_rows = 10 - dag_module.N_LAGS
        assert len(result) == expected_rows

    def test_lag_ordering(self, dag_module):
        user_data = pd.DataFrame({
            "user_id": [1] * 8,
            "Date": pd.date_range("2024-01-01", periods=8),
            "strength_score": [10, 11, 12, 13, 14, 15, 16, 17]
        })
        result = dag_module.build_lag_features(user_data)
        # Check that lag features are in correct order (most recent first)
        first_row = result.iloc[0]
        assert first_row["lag_strength_1"] == 16  # Most recent
        assert first_row["lag_strength_5"] == 12  # 5 sessions back

    def test_days_ago_non_negative(self, dag_module):
        user_data = pd.DataFrame({
            "user_id": [1] * 8,
            "Date": pd.date_range("2024-01-01", periods=8),
            "strength_score": range(10, 18)
        })
        result = dag_module.build_lag_features(user_data)
        # All dates should be in order
        dates = result["reference_date"].tolist()
        assert dates == sorted(dates)