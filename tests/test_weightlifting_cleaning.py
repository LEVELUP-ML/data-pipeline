"""Tests for weightlifting CSV cleaning logic."""

import pandas as pd
import numpy as np
import pytest

#  Inline cleaning logic

MAX_WEIGHT_KG = 700.0
MAX_REPS = 200
MAX_SET_ORDER = 100
MAX_SECONDS = 36_000
MAX_DISTANCE_M = 100_000
REQUIRED_COLS = {"Date", "Workout Name", "Exercise Name", "Set Order", "Weight", "Reps"}


def tag_row_issues(row: pd.Series) -> list:
    issues = []
    if pd.isna(row.get("_parsed_date")):
        issues.append("invalid_or_missing_date")
    if pd.isna(row.get("Exercise Name")) or str(row["Exercise Name"]).strip() == "":
        issues.append("missing_exercise_name")
    if pd.isna(row.get("Workout Name")) or str(row["Workout Name"]).strip() == "":
        issues.append("missing_workout_name")
    so = row.get("Set Order")
    if pd.isna(so):
        issues.append("missing_set_order")
    elif so < 1 or so > MAX_SET_ORDER:
        issues.append(f"set_order_out_of_range({so})")
    w = row.get("Weight")
    if pd.notna(w):
        if w < 0:
            issues.append(f"negative_weight({w})")
        elif w > MAX_WEIGHT_KG:
            issues.append(f"weight_exceeds_max({w})")
    r = row.get("Reps")
    if pd.notna(r):
        if r < 0:
            issues.append(f"negative_reps({r})")
        elif r > MAX_REPS:
            issues.append(f"reps_exceeds_max({r})")
    s = row.get("Seconds")
    if pd.notna(s):
        if s < 0:
            issues.append(f"negative_seconds({s})")
        elif s > MAX_SECONDS:
            issues.append(f"seconds_exceeds_max({s})")
    d = row.get("Distance")
    if pd.notna(d):
        if d < 0:
            issues.append(f"negative_distance({d})")
        elif d > MAX_DISTANCE_M:
            issues.append(f"distance_exceeds_max({d})")
    return issues


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
            "Date": [
                "2024-01-01",
                "not-a-date",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
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


#  Tests


class TestTagRowIssues:

    def test_valid_rows_have_no_issues(self, valid_df):
        for _, row in valid_df.iterrows():
            assert tag_row_issues(row) == []

    def test_invalid_date_flagged(self, dirty_df):
        row = dirty_df.iloc[1]
        issues = tag_row_issues(row)
        assert "invalid_or_missing_date" in issues

    def test_missing_workout_name_flagged(self, dirty_df):
        row = dirty_df.iloc[1]
        issues = tag_row_issues(row)
        assert "missing_workout_name" in issues

    def test_missing_exercise_name_flagged(self, dirty_df):
        row = dirty_df.iloc[2]
        issues = tag_row_issues(row)
        assert "missing_exercise_name" in issues

    def test_set_order_zero_flagged(self, dirty_df):
        row = dirty_df.iloc[3]
        issues = tag_row_issues(row)
        assert any("set_order_out_of_range" in i for i in issues)

    def test_negative_weight_flagged(self, dirty_df):
        row = dirty_df.iloc[1]
        issues = tag_row_issues(row)
        assert any("negative_weight" in i for i in issues)

    def test_weight_exceeds_max_flagged(self, dirty_df):
        row = dirty_df.iloc[3]
        issues = tag_row_issues(row)
        assert any("weight_exceeds_max" in i for i in issues)

    def test_negative_reps_flagged(self, dirty_df):
        row = dirty_df.iloc[2]
        issues = tag_row_issues(row)
        assert any("negative_reps" in i for i in issues)

    def test_reps_exceeds_max_flagged(self, dirty_df):
        row = dirty_df.iloc[4]
        issues = tag_row_issues(row)
        assert any("reps_exceeds_max" in i for i in issues)

    def test_negative_seconds_flagged(self, dirty_df):
        row = dirty_df.iloc[3]
        issues = tag_row_issues(row)
        assert any("negative_seconds" in i for i in issues)

    def test_distance_exceeds_max_flagged(self, dirty_df):
        row = dirty_df.iloc[4]
        issues = tag_row_issues(row)
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
        issues = tag_row_issues(row)
        assert issues == []


class TestSchemaCheck:

    def test_required_columns_present(self, valid_df):
        missing = REQUIRED_COLS - set(valid_df.columns)
        assert len(missing) == 0

    def test_missing_column_detected(self):
        df = pd.DataFrame({"Date": [], "Weight": []})
        missing = REQUIRED_COLS - set(df.columns)
        assert "Exercise Name" in missing
        assert "Set Order" in missing


class TestDeduplication:

    def test_duplicates_removed(self):
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01"] * 3,
                "Workout Name": ["Push"] * 3,
                "Exercise Name": ["Bench"] * 3,
                "Set Order": [1, 1, 2],
                "Weight": [80.0, 80.0, 85.0],
                "Reps": [8, 8, 6],
            }
        )
        deduped = df.drop_duplicates(
            subset=[
                "Date",
                "Workout Name",
                "Exercise Name",
                "Set Order",
                "Weight",
                "Reps",
            ]
        )
        assert len(deduped) == 2

    def test_no_false_dedup(self):
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-01"],
                "Workout Name": ["Push", "Push"],
                "Exercise Name": ["Bench", "Bench"],
                "Set Order": [1, 2],
                "Weight": [80.0, 82.5],
                "Reps": [8, 6],
            }
        )
        deduped = df.drop_duplicates(
            subset=[
                "Date",
                "Workout Name",
                "Exercise Name",
                "Set Order",
                "Weight",
                "Reps",
            ]
        )
        assert len(deduped) == 2
