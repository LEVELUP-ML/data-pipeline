"""Tests for stamina engine, windowing, and anomaly detection."""

import numpy as np
import pandas as pd
import pytest


#  Inline logic

VALID_ACTIVITIES = set("ABCDEFGHIJKLMOPQRS")
ACCEL_MAG_CEILING = 200.0


def create_row_windows(df: pd.DataFrame, window_size: int = 200) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df["window_id"] = df.index // window_size
    windowed = (
        df.groupby("window_id")
        .agg(
            {
                "user": "first",
                "activity": "first",
                "x": "mean",
                "y": "mean",
                "z": "mean",
            }
        )
        .reset_index(drop=True)
    )
    return windowed


def compute_stamina(
    df: pd.DataFrame, max_stamina: float = 100.0, fatigue_rate: float = 0.01
) -> pd.DataFrame:
    stamina = max_stamina
    values = []
    for _, row in df.iterrows():
        intensity = np.sqrt(row["x"] ** 2 + row["y"] ** 2 + row["z"] ** 2)
        stamina = max(0.0, stamina - intensity * fatigue_rate)
        values.append(round(stamina, 4))
    df = df.copy()
    df["stamina"] = values
    return df


def detect_anomalies(df: pd.DataFrame) -> dict:
    mag = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    threshold = mag.mean() + 5 * mag.std()
    count = (mag > threshold).sum()
    return {
        "anomaly_count": int(count),
        "threshold": round(float(threshold), 4),
        "mean_magnitude": round(float(mag.mean()), 4),
        "std_magnitude": round(float(mag.std()), 4),
    }


def tag_row_issues(row: pd.Series) -> list:
    issues = []
    if pd.isna(row.get("user")):
        issues.append("missing_user")
    if pd.isna(row.get("activity")) or row["activity"] not in VALID_ACTIVITIES:
        issues.append(f"invalid_activity({row.get('activity')})")
    for ax in ("x", "y", "z"):
        if pd.isna(row.get(ax)):
            issues.append(f"missing_{ax}")
    if pd.isna(row.get("timestamp")):
        issues.append("missing_timestamp")
    if all(pd.notna(row.get(ax)) for ax in ("x", "y", "z")):
        mag = np.sqrt(row["x"] ** 2 + row["y"] ** 2 + row["z"] ** 2)
        if mag > ACCEL_MAG_CEILING:
            issues.append(f"extreme_magnitude({mag:.1f})")
    return issues


#  Fixtures


@pytest.fixture
def accel_df():
    """Simulated accel data: 400 rows, 2 windows of 200."""
    np.random.seed(42)
    n = 400
    return pd.DataFrame(
        {
            "user": [1600] * n,
            "activity": ["A"] * 200 + ["B"] * 200,
            "timestamp": range(n),
            "x": np.random.normal(0.1, 0.5, n),
            "y": np.random.normal(-0.2, 0.5, n),
            "z": np.random.normal(9.8, 0.3, n),
        }
    )


@pytest.fixture
def windowed_df(accel_df):
    return create_row_windows(accel_df, window_size=200)


#  Windowing Tests


class TestWindowing:

    def test_correct_window_count(self, accel_df):
        w = create_row_windows(accel_df, window_size=200)
        assert len(w) == 2

    def test_window_preserves_first_activity(self, accel_df):
        w = create_row_windows(accel_df, window_size=200)
        assert w["activity"].iloc[0] == "A"
        assert w["activity"].iloc[1] == "B"

    def test_window_aggregates_mean(self, accel_df):
        w = create_row_windows(accel_df, window_size=200)
        expected_z = accel_df.iloc[:200]["z"].mean()
        assert w["z"].iloc[0] == pytest.approx(expected_z)

    def test_single_row_window(self):
        df = pd.DataFrame(
            {
                "user": [1600],
                "activity": ["A"],
                "x": [0.1],
                "y": [-0.2],
                "z": [9.8],
            }
        )
        w = create_row_windows(df, window_size=1)
        assert len(w) == 1

    def test_window_size_larger_than_data(self):
        df = pd.DataFrame(
            {
                "user": [1600] * 5,
                "activity": ["A"] * 5,
                "x": [0.1] * 5,
                "y": [-0.2] * 5,
                "z": [9.8] * 5,
            }
        )
        w = create_row_windows(df, window_size=200)
        assert len(w) == 1  # all rows in one window

    def test_empty_input(self):
        df = pd.DataFrame(columns=["user", "activity", "x", "y", "z"])
        w = create_row_windows(df, window_size=200)
        assert len(w) == 0


#  Stamina Tests


class TestStamina:

    def test_stamina_starts_at_max(self, windowed_df):
        result = compute_stamina(windowed_df, max_stamina=100.0, fatigue_rate=0.01)
        # First value should be slightly less than 100 (after first drain)
        assert result["stamina"].iloc[0] < 100.0
        assert result["stamina"].iloc[0] > 90.0

    def test_stamina_monotonically_decreases(self, windowed_df):
        result = compute_stamina(windowed_df, max_stamina=100.0, fatigue_rate=0.01)
        for i in range(1, len(result)):
            assert result["stamina"].iloc[i] <= result["stamina"].iloc[i - 1]

    def test_stamina_never_negative(self, windowed_df):
        result = compute_stamina(windowed_df, max_stamina=100.0, fatigue_rate=1.0)
        assert (result["stamina"] >= 0).all()

    def test_stamina_floors_at_zero(self):
        df = pd.DataFrame(
            {
                "user": [1600] * 50,
                "activity": ["A"] * 50,
                "x": [50.0] * 50,
                "y": [50.0] * 50,
                "z": [50.0] * 50,
            }
        )
        result = compute_stamina(df, max_stamina=100.0, fatigue_rate=1.0)
        assert result["stamina"].iloc[-1] == 0.0

    def test_zero_movement_no_fatigue(self):
        df = pd.DataFrame(
            {
                "user": [1600] * 5,
                "activity": ["A"] * 5,
                "x": [0.0] * 5,
                "y": [0.0] * 5,
                "z": [0.0] * 5,
            }
        )
        result = compute_stamina(df, max_stamina=100.0, fatigue_rate=0.01)
        assert (result["stamina"] == 100.0).all()

    def test_higher_fatigue_rate_drains_faster(self, windowed_df):
        slow = compute_stamina(windowed_df, fatigue_rate=0.001)
        fast = compute_stamina(windowed_df, fatigue_rate=0.1)
        assert fast["stamina"].iloc[-1] < slow["stamina"].iloc[-1]

    def test_stamina_column_added(self, windowed_df):
        result = compute_stamina(windowed_df)
        assert "stamina" in result.columns

    def test_original_df_not_mutated(self, windowed_df):
        original_cols = set(windowed_df.columns)
        compute_stamina(windowed_df)
        assert "stamina" not in windowed_df.columns
        assert set(windowed_df.columns) == original_cols


#  Anomaly Detection Tests


class TestAnomalyDetection:

    def test_no_anomalies_in_normal_data(self, accel_df):
        result = detect_anomalies(accel_df)
        assert result["anomaly_count"] == 0

    def test_detects_spike(self, accel_df):
        spiked = accel_df.copy()
        spiked.loc[0, "x"] = 500.0
        spiked.loc[1, "y"] = -500.0
        result = detect_anomalies(spiked)
        assert result["anomaly_count"] >= 1

    def test_returns_expected_keys(self, accel_df):
        result = detect_anomalies(accel_df)
        assert set(result.keys()) == {
            "anomaly_count",
            "threshold",
            "mean_magnitude",
            "std_magnitude",
        }

    def test_threshold_is_positive(self, accel_df):
        result = detect_anomalies(accel_df)
        assert result["threshold"] > 0

    def test_all_extreme_data(self):
        """Uniform extreme values = 0 std = no anomalies."""
        df = pd.DataFrame(
            {
                "x": [100.0] * 10,
                "y": [100.0] * 10,
                "z": [100.0] * 10,
            }
        )
        result = detect_anomalies(df)
        assert result["anomaly_count"] == 0


#  WISDM Row Validation Tests


class TestWisdmRowValidation:

    def test_valid_row_no_issues(self):
        row = pd.Series(
            {
                "user": 1600,
                "activity": "A",
                "timestamp": 123,
                "x": 0.1,
                "y": -0.2,
                "z": 9.8,
            }
        )
        assert tag_row_issues(row) == []

    def test_missing_user_flagged(self):
        row = pd.Series(
            {
                "user": pd.NA,
                "activity": "A",
                "timestamp": 123,
                "x": 0.1,
                "y": -0.2,
                "z": 9.8,
            }
        )
        assert "missing_user" in tag_row_issues(row)

    def test_invalid_activity_flagged(self):
        row = pd.Series(
            {
                "user": 1600,
                "activity": "N",
                "timestamp": 123,
                "x": 0.1,
                "y": -0.2,
                "z": 9.8,
            }
        )
        issues = tag_row_issues(row)
        assert any("invalid_activity" in i for i in issues)

    def test_missing_axis_flagged(self):
        row = pd.Series(
            {
                "user": 1600,
                "activity": "A",
                "timestamp": 123,
                "x": np.nan,
                "y": -0.2,
                "z": 9.8,
            }
        )
        assert "missing_x" in tag_row_issues(row)

    def test_extreme_magnitude_flagged(self):
        row = pd.Series(
            {
                "user": 1600,
                "activity": "A",
                "timestamp": 123,
                "x": 150.0,
                "y": 150.0,
                "z": 150.0,
            }
        )
        issues = tag_row_issues(row)
        assert any("extreme_magnitude" in i for i in issues)

    def test_missing_timestamp_flagged(self):
        row = pd.Series(
            {
                "user": 1600,
                "activity": "A",
                "timestamp": np.nan,
                "x": 0.1,
                "y": -0.2,
                "z": 9.8,
            }
        )
        assert "missing_timestamp" in tag_row_issues(row)

    def test_activity_N_is_invalid(self):
        """Dataset docs say no 'N' activity."""
        row = pd.Series(
            {
                "user": 1600,
                "activity": "N",
                "timestamp": 123,
                "x": 0.1,
                "y": -0.2,
                "z": 9.8,
            }
        )
        assert any("invalid_activity" in i for i in tag_row_issues(row))

    def test_all_valid_activities(self):
        for act in "ABCDEFGHIJKLMOPQRS":
            row = pd.Series(
                {
                    "user": 1600,
                    "activity": act,
                    "timestamp": 123,
                    "x": 0.1,
                    "y": -0.2,
                    "z": 9.8,
                }
            )
            assert tag_row_issues(row) == [], f"Activity {act} should be valid"
