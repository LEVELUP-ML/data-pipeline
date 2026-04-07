"""Tests for stamina engine, windowing, and anomaly detection.

Logic under test lives in dags/lib/wisdm.py — no inline copies here.
"""

import numpy as np
import pandas as pd
import pytest

from lib.wisdm import (
    analyze_bias,
    compute_stamina,
    create_row_windows,
    detect_anomalies,
    tag_row_issues,
)


#  Fixtures 


@pytest.fixture
def accel_df():
    """Clean 100-row WISDM-like DataFrame with no issues."""
    np.random.seed(0)
    n = 100
    return pd.DataFrame(
        {
            "user": pd.array([1600] * n, dtype="Int64"),
            "activity": np.random.choice(list("ABCDE"), n),
            "timestamp": range(n),
            "x": np.random.normal(0.1, 0.5, n),
            "y": np.random.normal(-0.2, 0.5, n),
            "z": np.random.normal(9.8, 0.3, n),
        }
    )


@pytest.fixture
def windowed_df(accel_df):
    return create_row_windows(accel_df, window_size=10)


#  tag_row_issues 


class TestTagRowIssues:

    def _make_row(self, **kwargs):
        base = {
            "user": pd.array([1600], dtype="Int64")[0],
            "activity": "A",
            "timestamp": 1000,
            "x": 0.1,
            "y": -0.2,
            "z": 9.8,
        }
        base.update(kwargs)
        return pd.Series(base)

    def test_valid_row_no_issues(self):
        assert tag_row_issues(self._make_row()) == []

    def test_missing_user_flagged(self):
        issues = tag_row_issues(self._make_row(user=None))
        assert "missing_user" in issues

    def test_invalid_activity_flagged(self):
        issues = tag_row_issues(self._make_row(activity="Z"))
        assert any("invalid_activity" in i for i in issues)

    def test_missing_x_flagged(self):
        issues = tag_row_issues(self._make_row(x=np.nan))
        assert "missing_x" in issues

    def test_extreme_magnitude_flagged(self):
        # sqrt(300^2 + 0 + 0) = 300 > 200 ceiling
        issues = tag_row_issues(self._make_row(x=300.0, y=0.0, z=0.0))
        assert any("extreme_magnitude" in i for i in issues)

    def test_normal_magnitude_not_flagged(self):
        issues = tag_row_issues(self._make_row(x=0.5, y=-0.3, z=9.8))
        assert not any("extreme_magnitude" in i for i in issues)


#  create_row_windows 


class TestCreateRowWindows:

    def test_correct_number_of_windows(self, accel_df):
        windowed = create_row_windows(accel_df, window_size=10)
        assert len(windowed) == 10  # 100 rows / 10

    def test_window_columns_present(self, accel_df):
        windowed = create_row_windows(accel_df, window_size=10)
        for col in ("user", "activity", "x", "y", "z"):
            assert col in windowed.columns

    def test_xyz_are_means_not_raw(self, accel_df):
        # First window: rows 0-9
        expected_x = float(accel_df["x"].iloc[:10].mean())
        windowed = create_row_windows(accel_df, window_size=10)
        assert windowed["x"].iloc[0] == pytest.approx(expected_x, rel=1e-5)

    def test_window_size_equals_total_gives_one_row(self, accel_df):
        windowed = create_row_windows(accel_df, window_size=len(accel_df))
        assert len(windowed) == 1

    def test_single_row_df(self):
        df = pd.DataFrame({"user": [1], "activity": ["A"], "x": [1.0], "y": [2.0], "z": [3.0]})
        windowed = create_row_windows(df, window_size=10)
        assert len(windowed) == 1


#  compute_stamina 


class TestComputeStamina:

    def test_stamina_column_added(self, windowed_df):
        result = compute_stamina(windowed_df)
        assert "stamina" in result.columns

    def test_stamina_starts_near_max(self, windowed_df):
        result = compute_stamina(windowed_df, max_stamina=100.0)
        assert result["stamina"].iloc[0] <= 100.0

    def test_stamina_is_monotonically_non_increasing(self, windowed_df):
        result = compute_stamina(windowed_df, max_stamina=100.0, fatigue_rate=0.01)
        assert (result["stamina"].diff().dropna() <= 0).all()

    def test_stamina_never_negative(self, windowed_df):
        result = compute_stamina(windowed_df, max_stamina=100.0, fatigue_rate=10.0)
        assert (result["stamina"] >= 0).all()

    def test_input_df_not_mutated(self, windowed_df):
        original_cols = set(windowed_df.columns)
        compute_stamina(windowed_df)
        assert set(windowed_df.columns) == original_cols

    def test_custom_fatigue_rate(self, windowed_df):
        slow = compute_stamina(windowed_df, fatigue_rate=0.001)
        fast = compute_stamina(windowed_df, fatigue_rate=0.1)
        assert slow["stamina"].iloc[-1] > fast["stamina"].iloc[-1]


#  detect_anomalies 


class TestDetectAnomalies:

    def test_returns_expected_keys(self, accel_df):
        result = detect_anomalies(accel_df)
        assert {"anomaly_count", "threshold", "mean_magnitude", "std_magnitude"} == set(result)

    def test_no_anomalies_in_clean_data(self, accel_df):
        # Data has std ~1; 5-sigma threshold >> any value
        result = detect_anomalies(accel_df)
        assert result["anomaly_count"] == 0

    def test_spike_detected(self, accel_df):
        df = accel_df.copy()
        df.loc[0, "x"] = 10_000.0  # far beyond 5-sigma
        result = detect_anomalies(df)
        assert result["anomaly_count"] >= 1

    def test_threshold_above_mean(self, accel_df):
        result = detect_anomalies(accel_df)
        assert result["threshold"] > result["mean_magnitude"]


#  analyze_bias 


class TestAnalyzeBias:

    def test_returns_dict_keyed_by_activity(self, windowed_df):
        result = compute_stamina(windowed_df)
        bias = analyze_bias(result)
        assert isinstance(bias, dict)
        assert all(isinstance(v, float) for v in bias.values())

    def test_all_activities_present(self, accel_df):
        windowed = create_row_windows(accel_df, window_size=10)
        result = compute_stamina(windowed)
        bias = analyze_bias(result)
        unique_activities = set(windowed["activity"])
        assert set(bias.keys()) == unique_activities