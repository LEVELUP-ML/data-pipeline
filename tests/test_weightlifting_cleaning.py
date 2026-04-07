"""Tests for weightlifting CSV cleaning logic.

Logic under test lives in dags/lib/weightlifting.py — no inline copies here.
"""

import numpy as np
import pandas as pd
import pytest

from lib.weightlifting import (
    REQUIRED_COLS,
    clean_dataframe,
    tag_row_issues,
    validate_schema,
)


#  Fixtures 


@pytest.fixture
def valid_df():
    return pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "Workout Name": ["Push Day", "Push Day", "Pull Day"],
            "Exercise Name": ["Bench Press", "OHP", "Deadlift"],
            "Set Order": pd.array([1, 2, 1], dtype="Int64"),
            "Weight": [80.0, 40.0, 120.0],
            "Reps": pd.array([8, 10, 5], dtype="Int64"),
            "Seconds": pd.array([None, None, None], dtype="Int64"),
            "Distance": [None, None, None],
            "_parsed_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
        }
    )


@pytest.fixture
def dirty_df():
    return pd.DataFrame(
        {
            "Date": ["2024-01-01", "not-a-date", "2024-01-03", "2024-01-04", "2024-01-05"],
            "Workout Name": ["Push", "", "Pull", "Legs", "Push"],
            "Exercise Name": ["Bench", "OHP", "", "Squat", "Bench"],
            "Set Order": pd.array([1, 2, 1, 0, 1], dtype="Int64"),
            "Weight": [80.0, -5.0, 100.0, 800.0, 50.0],
            "Reps": pd.array([8, 10, -3, 5, 300], dtype="Int64"),
            "Seconds": pd.array([None, None, None, -10, None], dtype="Int64"),
            "Distance": [None, None, None, None, 200_000.0],
            "_parsed_date": pd.to_datetime(
                ["2024-01-01", pd.NaT, "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
        }
    )


#  tag_row_issues 


class TestTagRowIssues:

    def test_valid_rows_have_no_issues(self, valid_df):
        for _, row in valid_df.iterrows():
            assert tag_row_issues(row) == []

    def test_invalid_date_flagged(self, dirty_df):
        issues = tag_row_issues(dirty_df.iloc[1])
        assert "invalid_or_missing_date" in issues

    def test_missing_workout_name_flagged(self, dirty_df):
        issues = tag_row_issues(dirty_df.iloc[1])
        assert "missing_workout_name" in issues

    def test_missing_exercise_name_flagged(self, dirty_df):
        issues = tag_row_issues(dirty_df.iloc[2])
        assert "missing_exercise_name" in issues

    def test_set_order_zero_flagged(self, dirty_df):
        issues = tag_row_issues(dirty_df.iloc[3])
        assert any("set_order_out_of_range" in i for i in issues)

    def test_negative_weight_flagged(self, dirty_df):
        issues = tag_row_issues(dirty_df.iloc[1])
        assert any("negative_weight" in i for i in issues)

    def test_weight_exceeds_max_flagged(self, dirty_df):
        issues = tag_row_issues(dirty_df.iloc[3])
        assert any("weight_exceeds_max" in i for i in issues)

    def test_negative_reps_flagged(self, dirty_df):
        issues = tag_row_issues(dirty_df.iloc[2])
        assert any("negative_reps" in i for i in issues)

    def test_reps_exceeds_max_flagged(self, dirty_df):
        issues = tag_row_issues(dirty_df.iloc[4])
        assert any("reps_exceeds_max" in i for i in issues)

    def test_negative_seconds_flagged(self, dirty_df):
        issues = tag_row_issues(dirty_df.iloc[3])
        assert any("negative_seconds" in i for i in issues)

    def test_distance_exceeds_max_flagged(self, dirty_df):
        issues = tag_row_issues(dirty_df.iloc[4])
        assert any("distance_exceeds_max" in i for i in issues)

    def test_null_optional_fields_no_issue(self):
        row = pd.Series(
            {
                "Date": "2024-01-01",
                "Workout Name": "Push",
                "Exercise Name": "Bench",
                "Set Order": 1,
                "Weight": np.nan,
                "Reps": np.nan,
                "Seconds": np.nan,
                "Distance": np.nan,
                "_parsed_date": pd.Timestamp("2024-01-01"),
            }
        )
        assert tag_row_issues(row) == []


#  validate_schema 


class TestValidateSchema:

    def test_valid_df_no_missing(self, valid_df):
        assert validate_schema(valid_df) == set()

    def test_detects_missing_columns(self):
        df = pd.DataFrame({"Date": [], "Weight": []})
        missing = validate_schema(df)
        assert "Exercise Name" in missing
        assert "Set Order" in missing


#  clean_dataframe 


class TestCleanDataframe:

    def _make_raw_df(self, rows):
        """Build a minimal raw CSV-like DataFrame."""
        return pd.DataFrame(
            rows,
            columns=["Date", "Workout Name", "Exercise Name", "Set Order", "Weight", "Reps"],
        )

    def test_clean_rows_survive(self):
        df = self._make_raw_df(
            [
                ["2024-01-01", "Push", "Bench", 1, "80.0", "8"],
                ["2024-01-01", "Push", "OHP", 2, "40.0", "10"],
            ]
        )
        result = clean_dataframe(df)
        assert len(result["clean"]) == 2
        assert len(result["rejected"]) == 0

    def test_dirty_rows_rejected(self):
        df = self._make_raw_df(
            [
                ["not-a-date", "Push", "Bench", 1, "80.0", "8"],
                ["2024-01-01", "Push", "Bench", 1, "-5.0", "8"],
            ]
        )
        result = clean_dataframe(df)
        assert len(result["rejected"]) == 2

    def test_duplicates_dropped(self):
        df = self._make_raw_df(
            [
                ["2024-01-01", "Push", "Bench", 1, "80.0", "8"],
                ["2024-01-01", "Push", "Bench", 1, "80.0", "8"],  # exact duplicate
                ["2024-01-01", "Push", "Bench", 2, "82.5", "6"],
            ]
        )
        result = clean_dataframe(df)
        assert result["duplicates_dropped"] == 1
        assert len(result["clean"]) == 2

    def test_clean_df_sorted(self):
        df = self._make_raw_df(
            [
                ["2024-01-02", "Push", "OHP", 1, "40.0", "10"],
                ["2024-01-01", "Push", "Bench", 1, "80.0", "8"],
            ]
        )
        result = clean_dataframe(df)
        dates = list(result["clean"]["Date"])
        assert dates == sorted(dates)

    def test_reject_pct_calculated_correctly(self):
        df = self._make_raw_df(
            [
                ["2024-01-01", "Push", "Bench", 1, "80.0", "8"],   # clean
                ["bad-date",  "Push", "Bench", 1, "80.0", "8"],    # rejected
                ["2024-01-02", "Push", "OHP",   2, "40.0", "10"],  # clean
                ["2024-01-03", "Push", "OHP",   2, "-1.0", "10"],  # rejected
            ]
        )
        result = clean_dataframe(df)
        assert result["reject_pct"] == pytest.approx(50.0)