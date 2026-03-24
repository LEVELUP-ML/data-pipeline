"""Tests for synthetic data pipeline scripts."""

import json
import numpy as np
import pandas as pd
import pytest


#  Inline functions (mirrors scripts/) 


# preprocess_sleep.py
def parse_time_to_minutes(time_str):
    try:
        parts = str(time_str).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError, TypeError):
        return np.nan


# build_features.py
def compute_bmr(age, sex, height_cm, weight_kg):
    try:
        base = 10 * float(weight_kg) + 6.25 * float(height_cm) - 5 * float(age)
        if str(sex).strip().lower() in ("female", "f"):
            return round(base - 161)
        return round(base + 5)
    except (ValueError, TypeError):
        return None


# preprocess_quiz.py
def compute_int_score(accuracy, streak_days=1):
    acc_comp = min(1.0, accuracy) * 75
    streak_comp = min(1.0, streak_days / 7) * 25
    return round(min(100, max(0, acc_comp + streak_comp)), 2)


# anomaly_detection.py
def detect_anomalies(df):
    anomalies = []
    for field in ["sleep_hours", "attempts_count", "avg_accuracy"]:
        if field in df.columns:
            miss_pct = df[field].isna().mean() * 100
            if miss_pct > 20:
                anomalies.append(
                    {
                        "type": "high_missingness",
                        "field": field,
                        "missing_pct": round(miss_pct, 2),
                        "threshold": 20,
                        "severity": "warning",
                    }
                )
    if "sleep_hours" in df.columns:
        bad = df[
            (df["sleep_hours"].notna())
            & ((df["sleep_hours"] < 2) | (df["sleep_hours"] > 14))
        ]
        if len(bad) > 0:
            anomalies.append(
                {
                    "type": "out_of_range",
                    "field": "sleep_hours",
                    "count": len(bad),
                    "range": [2, 14],
                    "severity": "error",
                }
            )
    if "avg_accuracy" in df.columns:
        bad = df[
            (df["avg_accuracy"].notna())
            & ((df["avg_accuracy"] < 0) | (df["avg_accuracy"] > 1))
        ]
        if len(bad) > 0:
            anomalies.append(
                {
                    "type": "out_of_range",
                    "field": "avg_accuracy",
                    "count": len(bad),
                    "range": [0, 1],
                    "severity": "error",
                }
            )
    if "avg_time_per_question" in df.columns:
        bad = df[
            (df["avg_time_per_question"].notna())
            & ((df["avg_time_per_question"] <= 0) | (df["avg_time_per_question"] > 300))
        ]
        if len(bad) > 0:
            anomalies.append(
                {
                    "type": "out_of_range",
                    "field": "avg_time_per_question",
                    "count": len(bad),
                    "range": [0.01, 300],
                    "severity": "warning",
                }
            )
    for field in ["sleep_hours", "attempts_count", "int_score", "bmr"]:
        if field in df.columns:
            neg = df[(df[field].notna()) & (df[field] < 0)]
            if len(neg) > 0:
                anomalies.append(
                    {
                        "type": "negative_value",
                        "field": field,
                        "count": len(neg),
                        "severity": "error",
                    }
                )
    return anomalies


# bias_report.py
AGE_BINS = [(0, 19, "<20"), (20, 29, "20-29"), (30, 39, "30-39"), (40, 200, "40+")]
IMBALANCE_THRESHOLD = 10


def assign_age_bucket(age):
    try:
        age = int(age)
        for lo, hi, label in AGE_BINS:
            if lo <= age <= hi:
                return label
    except (ValueError, TypeError):
        pass
    return "unknown"


# validate_schema.py
PROFILE_SCHEMA = {
    "user_id": str,
    "age": (int, float),
    "sex": str,
    "height": (int, float),
    "weight": (int, float),
}
SLEEP_SCHEMA = {
    "user_id": str,
    "date": str,
    "sleepHours": (int, float, type(None)),
    "quality": (int, float, type(None)),
}
QUIZ_SCHEMA = {
    "user_id": str,
    "timestamp": str,
    "num_questions": (int, float),
    "num_correct": (int, float),
    "total_time_seconds": (int, float),
}


def validate_records(records, schema, name):
    errors = []
    for i, rec in enumerate(records):
        for field, expected_type in schema.items():
            if field not in rec:
                errors.append(f"{name}[{i}]: missing field '{field}'")
            elif not isinstance(rec[field], expected_type):
                errors.append(
                    f"{name}[{i}]: field '{field}' expected {expected_type}, got {type(rec[field]).__name__}"
                )
    return errors


#  Fixtures 


@pytest.fixture
def joined_df():
    """Simulated daily_joined dataframe."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "user_id": [f"user_{i:03d}" for i in np.random.randint(1, 20, n)],
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "sleep_hours": np.random.normal(7.0, 1.2, n).clip(2, 14),
            "avg_accuracy": np.random.uniform(0.3, 1.0, n),
            "avg_time_per_question": np.random.uniform(5, 60, n),
            "int_score": np.random.uniform(20, 90, n),
            "attempts_count": np.random.randint(1, 5, n),
            "age": np.random.randint(14, 60, n),
            "sex": np.random.choice(["Male", "Female"], n),
            "bmr": np.random.randint(1200, 2200, n),
        }
    )


#  Sleep Preprocessing Tests 


class TestParseTimeToMinutes:

    def test_midnight(self):
        assert parse_time_to_minutes("00:00") == 0

    def test_noon(self):
        assert parse_time_to_minutes("12:00") == 720

    def test_late_night(self):
        assert parse_time_to_minutes("23:30") == 23 * 60 + 30

    def test_invalid_string(self):
        assert np.isnan(parse_time_to_minutes("not-a-time"))

    def test_none(self):
        assert np.isnan(parse_time_to_minutes(None))

    def test_partial(self):
        assert np.isnan(parse_time_to_minutes("12"))


#  BMR Tests 


class TestComputeBMR:

    def test_male(self):
        # 10*70 + 6.25*175 - 5*25 + 5 = 700 + 1093.75 - 125 + 5 = 1673.75 → 1674
        assert compute_bmr(25, "Male", 175, 70) == 1674

    def test_female(self):
        # 10*60 + 6.25*165 - 5*30 - 161 = 600 + 1031.25 - 150 - 161 = 1320.25 → 1320
        assert compute_bmr(30, "Female", 165, 60) == 1320

    def test_female_lowercase(self):
        assert compute_bmr(30, "female", 165, 60) == 1320

    def test_female_f(self):
        assert compute_bmr(30, "f", 165, 60) == 1320

    def test_invalid_returns_none(self):
        assert compute_bmr("abc", "Male", 175, 70) is None

    def test_none_weight(self):
        assert compute_bmr(25, "Male", 175, None) is None


#  INT Score Tests 


class TestComputeIntScore:

    def test_perfect_accuracy_no_streak(self):
        score = compute_int_score(1.0, streak_days=1)
        assert score == pytest.approx(75 + 25 * (1 / 7), abs=0.1)

    def test_zero_accuracy(self):
        assert compute_int_score(0.0) == pytest.approx(25 * (1 / 7), abs=0.1)

    def test_full_streak(self):
        score = compute_int_score(1.0, streak_days=7)
        assert score == 100.0

    def test_capped_at_100(self):
        score = compute_int_score(1.5, streak_days=10)
        assert score <= 100.0

    def test_floored_at_0(self):
        score = compute_int_score(0.0, streak_days=0)
        assert score >= 0.0

    def test_half_accuracy_half_streak(self):
        score = compute_int_score(0.5, streak_days=3)
        expected = 0.5 * 75 + min(1.0, 3 / 7) * 25
        assert score == pytest.approx(expected, abs=0.1)


#  Anomaly Detection Tests 


class TestDetectAnomalies:

    def test_clean_data_no_anomalies(self, joined_df):
        anomalies = detect_anomalies(joined_df)
        assert len(anomalies) == 0

    def test_out_of_range_sleep(self):
        df = pd.DataFrame({"sleep_hours": [1.0, 7.0, 15.0, 8.0]})
        anomalies = detect_anomalies(df)
        sleep_anomalies = [
            a
            for a in anomalies
            if a["field"] == "sleep_hours" and a["type"] == "out_of_range"
        ]
        assert len(sleep_anomalies) == 1
        assert sleep_anomalies[0]["count"] == 2

    def test_negative_values(self):
        df = pd.DataFrame({"int_score": [-5, 50, 80], "sleep_hours": [7, 7, 7]})
        anomalies = detect_anomalies(df)
        neg = [a for a in anomalies if a["type"] == "negative_value"]
        assert len(neg) == 1
        assert neg[0]["field"] == "int_score"

    def test_accuracy_out_of_range(self):
        df = pd.DataFrame({"avg_accuracy": [0.5, 1.5, -0.1]})
        anomalies = detect_anomalies(df)
        acc = [a for a in anomalies if a["field"] == "avg_accuracy"]
        assert len(acc) == 1
        assert acc[0]["count"] == 2

    def test_high_missingness(self):
        df = pd.DataFrame({"sleep_hours": [np.nan] * 25 + [7.0] * 75})
        anomalies = detect_anomalies(df)
        miss = [a for a in anomalies if a["type"] == "high_missingness"]
        assert len(miss) == 1

    def test_time_per_question_out_of_range(self):
        df = pd.DataFrame({"avg_time_per_question": [10, 0, 400, 30]})
        anomalies = detect_anomalies(df)
        time_a = [a for a in anomalies if a["field"] == "avg_time_per_question"]
        assert len(time_a) == 1
        assert time_a[0]["count"] == 2

    def test_empty_df_no_crash(self):
        df = pd.DataFrame()
        anomalies = detect_anomalies(df)
        assert anomalies == []


#  Bias / Age Bucket Tests 


class TestAssignAgeBucket:

    @pytest.mark.parametrize(
        "age,expected",
        [
            (15, "<20"),
            (19, "<20"),
            (20, "20-29"),
            (25, "20-29"),
            (30, "30-39"),
            (39, "30-39"),
            (40, "40+"),
            (60, "40+"),
        ],
    )
    def test_correct_buckets(self, age, expected):
        assert assign_age_bucket(age) == expected

    def test_none(self):
        assert assign_age_bucket(None) == "unknown"

    def test_string_number(self):
        assert assign_age_bucket("25") == "20-29"

    def test_invalid_string(self):
        assert assign_age_bucket("abc") == "unknown"

    def test_zero(self):
        assert assign_age_bucket(0) == "<20"


#  Schema Validation Tests 


class TestValidateRecords:

    def test_valid_profile(self):
        rec = [{"user_id": "u1", "age": 25, "sex": "Male", "height": 175, "weight": 70}]
        assert validate_records(rec, PROFILE_SCHEMA, "profile") == []

    def test_missing_field(self):
        rec = [{"user_id": "u1", "age": 25}]
        errors = validate_records(rec, PROFILE_SCHEMA, "profile")
        assert any("missing field 'sex'" in e for e in errors)

    def test_wrong_type(self):
        rec = [{"user_id": 123, "age": 25, "sex": "Male", "height": 175, "weight": 70}]
        errors = validate_records(rec, PROFILE_SCHEMA, "profile")
        assert any("user_id" in e for e in errors)

    def test_valid_sleep(self):
        rec = [{"user_id": "u1", "date": "2024-01-01", "sleepHours": 7.5, "quality": 4}]
        assert validate_records(rec, SLEEP_SCHEMA, "sleep") == []

    def test_sleep_null_quality_ok(self):
        rec = [
            {"user_id": "u1", "date": "2024-01-01", "sleepHours": 7.5, "quality": None}
        ]
        assert validate_records(rec, SLEEP_SCHEMA, "sleep") == []

    def test_valid_quiz(self):
        rec = [
            {
                "user_id": "u1",
                "timestamp": "2024-01-01T10:00:00",
                "num_questions": 10,
                "num_correct": 8,
                "total_time_seconds": 120,
            }
        ]
        assert validate_records(rec, QUIZ_SCHEMA, "quiz") == []

    def test_multiple_errors(self):
        rec = [{}]
        errors = validate_records(rec, PROFILE_SCHEMA, "profile")
        assert len(errors) == len(PROFILE_SCHEMA)
