"""
DAG: process_synthetic_data

Validates, preprocesses, joins, and generates stats for
the synthetic Firestore-seeded data (profiles, sleep, quiz).

Expects raw JSON files already in data/raw/:
  - profiles_raw.json
  - sleep_logs_raw.json
  - quiz_attempts_raw.json

Triggers synthetic_anomaly_and_bias on success.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from dag_monitoring import (
    monitored_dag_args,
    on_dag_failure_callback,
    on_sla_miss_callback,
    emit_metric,
    log,
)

AIRFLOW_HOME = "/opt/airflow"
RAW_DIR = Path(f"{AIRFLOW_HOME}/data/raw")
PROCESSED_DIR = Path(f"{AIRFLOW_HOME}/data/processed")
REPORTS_DIR = Path(f"{AIRFLOW_HOME}/data/reports")

PROFILES_RAW = RAW_DIR / "profiles_raw.json"
SLEEP_RAW = RAW_DIR / "sleep_logs_raw.json"
QUIZ_RAW = RAW_DIR / "quiz_attempts_raw.json"
SLEEP_DAILY = PROCESSED_DIR / "sleep_daily.parquet"
INT_DAILY = PROCESSED_DIR / "int_daily.parquet"
DAILY_JOINED = PROCESSED_DIR / "daily_joined.parquet"
SCHEMA_REPORT = REPORTS_DIR / "schema.json"
STATS_REPORT = REPORTS_DIR / "stats.json"

#  Schema definitions

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


#  Helpers


def ensure_dirs():
    for d in [RAW_DIR, PROCESSED_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def validate_records(records, schema, name):
    errors = []
    for i, rec in enumerate(records):
        for field, expected_type in schema.items():
            if field not in rec:
                errors.append(f"{name}[{i}]: missing field '{field}'")
            elif not isinstance(rec[field], expected_type):
                errors.append(
                    f"{name}[{i}]: field '{field}' expected "
                    f"{expected_type}, got {type(rec[field]).__name__}"
                )
    return errors


def parse_time_to_minutes(time_str):
    try:
        parts = str(time_str).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError, TypeError):
        return np.nan


def compute_int_score(accuracy, streak_days=1):
    acc_comp = min(1.0, accuracy) * 75
    streak_comp = min(1.0, streak_days / 7) * 25
    return round(min(100, max(0, acc_comp + streak_comp)), 2)


def compute_bmr(age, sex, height_cm, weight_kg):
    try:
        base = 10 * float(weight_kg) + 6.25 * float(height_cm) - 5 * float(age)
        if str(sex).strip().lower() in ("female", "f"):
            return round(base - 161)
        return round(base + 5)
    except (ValueError, TypeError):
        return None


#  DAG

with DAG(
    dag_id="process_synthetic_data",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=monitored_dag_args(retries=1, sla_minutes=30),
    on_failure_callback=on_dag_failure_callback,
    sla_miss_callback=on_sla_miss_callback,
    tags=["synthetic", "processing", "firebase"],
) as dag:

    @task
    def validate_schemas() -> Dict[str, Any]:
        all_errors = {}

        for path, schema, name in [
            (PROFILES_RAW, PROFILE_SCHEMA, "profiles"),
            (SLEEP_RAW, SLEEP_SCHEMA, "sleep_logs"),
            (QUIZ_RAW, QUIZ_SCHEMA, "quiz_attempts"),
        ]:
            if not path.exists():
                raise AirflowFailException(f"Required file missing: {path}")
            records = json.loads(path.read_text())
            errs = validate_records(records, schema, name)
            log.info("%s: %d records, %d errors", name, len(records), len(errs))
            if errs:
                all_errors[name] = errs
                for e in errs[:5]:
                    log.warning("Schema: %s", e)

        total_errors = sum(len(v) for v in all_errors.values())
        result = {
            "total_errors": total_errors,
            "errors_by_source": {k: len(v) for k, v in all_errors.items()},
            "passed": total_errors == 0,
        }

        emit_metric("process_synthetic_data", "validate_schemas", result)
        log.info("Schema validation: %s", result)
        return result

    @task
    def preprocess_sleep() -> Dict[str, Any]:
        ensure_dirs()

        raw = json.loads(SLEEP_RAW.read_text())
        log.info("Loaded %d raw sleep records", len(raw))
        df = pd.DataFrame(raw).copy()

        # Clamp / clean
        df["sleep_hours"] = pd.to_numeric(df.get("sleepHours"), errors="coerce")
        df.loc[(df["sleep_hours"] < 2) | (df["sleep_hours"] > 14), "sleep_hours"] = (
            np.nan
        )

        # Parse times
        df["sleep_start_minutes"] = df.get("bedTime", pd.Series(dtype=str)).apply(
            parse_time_to_minutes
        )
        df["sleep_end_minutes"] = df.get("wakeTime", pd.Series(dtype=str)).apply(
            parse_time_to_minutes
        )

        # Midpoint
        df["sleep_midpoint"] = (df["sleep_start_minutes"] + df["sleep_end_minutes"]) / 2
        mask = df["sleep_start_minutes"] > df["sleep_end_minutes"]
        df.loc[mask, "sleep_midpoint"] = (
            (
                df.loc[mask, "sleep_start_minutes"]
                + df.loc[mask, "sleep_end_minutes"]
                + 1440
            )
            / 2
        ) % 1440

        # Satisfaction
        quality = pd.to_numeric(df.get("quality"), errors="coerce")
        df["sleep_satisfaction"] = (quality - 1) / 4

        # Sort + deduplicate
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["user_id", "date"])
        df = df.drop_duplicates(subset=["user_id", "date"], keep="first")

        # Rolling features
        parts = []
        for uid, group in df.groupby("user_id"):
            group = group.set_index("date").sort_index()
            group["rolling_sleep_hours_3d"] = (
                group["sleep_hours"].rolling("3D", min_periods=1).mean().round(2)
            )
            group["rolling_sleep_hours_7d"] = (
                group["sleep_hours"].rolling("7D", min_periods=1).mean().round(2)
            )
            group["bedtime_variability_7d"] = (
                group["sleep_start_minutes"].rolling("7D", min_periods=2).std().round(2)
            )
            parts.append(group.reset_index())
        df = pd.concat(parts, ignore_index=True)

        cols = [
            "user_id",
            "date",
            "sleep_hours",
            "sleep_start_minutes",
            "sleep_end_minutes",
            "sleep_midpoint",
            "sleep_satisfaction",
            "rolling_sleep_hours_3d",
            "rolling_sleep_hours_7d",
            "bedtime_variability_7d",
        ]
        result = df[cols].reset_index(drop=True)
        result.to_parquet(str(SLEEP_DAILY), index=False)

        summary = {
            "raw_records": len(raw),
            "processed_rows": len(result),
            "unique_users": int(result["user_id"].nunique()),
            "avg_sleep_hours": round(float(result["sleep_hours"].mean()), 2),
            "missing_sleep_pct": round(
                float(result["sleep_hours"].isna().mean() * 100), 2
            ),
        }

        emit_metric("process_synthetic_data", "preprocess_sleep", summary)
        log.info("Sleep preprocessing: %s", json.dumps(summary, indent=2))
        return summary

    @task
    def preprocess_quiz() -> Dict[str, Any]:
        ensure_dirs()

        raw = json.loads(QUIZ_RAW.read_text())
        log.info("Loaded %d raw quiz records", len(raw))
        df = pd.DataFrame(raw).copy()

        # Date
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["date"] = df["timestamp"].dt.date
        df["date"] = pd.to_datetime(df["date"])

        # Accuracy
        df["num_questions"] = pd.to_numeric(df.get("num_questions"), errors="coerce")
        df["num_correct"] = pd.to_numeric(df.get("num_correct"), errors="coerce")
        df["accuracy"] = np.where(
            df["num_questions"] > 0,
            (df["num_correct"] / df["num_questions"]).clip(0, 1),
            np.nan,
        )

        # Clean time
        df["avg_time_per_question"] = pd.to_numeric(
            df.get("avg_time_per_question_seconds"), errors="coerce"
        )
        df.loc[
            (df["avg_time_per_question"] <= 0) | (df["avg_time_per_question"] > 300),
            "avg_time_per_question",
        ] = np.nan

        # Remove impossible
        df = df[df["num_correct"] >= 0].copy()

        # Daily aggregation
        daily = (
            df.groupby(["user_id", "date"])
            .agg(
                attempts_count=("accuracy", "count"),
                avg_accuracy=("accuracy", "mean"),
                avg_time_per_question=("avg_time_per_question", "mean"),
            )
            .reset_index()
        )
        daily["avg_accuracy"] = daily["avg_accuracy"].round(4)
        daily["avg_time_per_question"] = daily["avg_time_per_question"].round(2)

        # Streak + INT score + rolling
        daily = daily.sort_values(["user_id", "date"])
        parts = []
        for uid, group in daily.groupby("user_id"):
            group = group.set_index("date").sort_index()
            streak = 0
            streaks = []
            prev_date = None
            for d in group.index:
                if prev_date is not None and (d - prev_date).days == 1:
                    streak += 1
                else:
                    streak = 1
                streaks.append(streak)
                prev_date = d
            group["streak"] = streaks
            group["int_score"] = [
                compute_int_score(acc, st)
                for acc, st in zip(group["avg_accuracy"], group["streak"])
            ]
            group["rolling_int_3d"] = (
                group["int_score"].rolling("3D", min_periods=1).mean().round(2)
            )
            group["rolling_int_7d"] = (
                group["int_score"].rolling("7D", min_periods=1).mean().round(2)
            )
            parts.append(group.reset_index())
        daily = pd.concat(parts, ignore_index=True)

        cols = [
            "user_id",
            "date",
            "attempts_count",
            "avg_accuracy",
            "avg_time_per_question",
            "int_score",
            "rolling_int_3d",
            "rolling_int_7d",
        ]
        result = daily[cols].reset_index(drop=True)
        result.to_parquet(str(INT_DAILY), index=False)

        summary = {
            "raw_records": len(raw),
            "processed_rows": len(result),
            "unique_users": int(result["user_id"].nunique()),
            "avg_accuracy": round(float(result["avg_accuracy"].mean()), 4),
            "avg_int_score": round(float(result["int_score"].mean()), 2),
        }

        emit_metric("process_synthetic_data", "preprocess_quiz", summary)
        log.info("Quiz preprocessing: %s", json.dumps(summary, indent=2))
        return summary

    @task
    def build_features() -> Dict[str, Any]:
        ensure_dirs()

        sleep = pd.read_parquet(str(SLEEP_DAILY))
        quiz = pd.read_parquet(str(INT_DAILY))
        profiles = pd.DataFrame(json.loads(PROFILES_RAW.read_text()))

        log.info(
            "Sleep: %d, Quiz: %d, Profiles: %d", len(sleep), len(quiz), len(profiles)
        )

        profiles["bmr"] = profiles.apply(
            lambda r: compute_bmr(
                r.get("age"), r.get("sex"), r.get("height"), r.get("weight")
            ),
            axis=1,
        )

        sleep["date"] = pd.to_datetime(sleep["date"])
        quiz["date"] = pd.to_datetime(quiz["date"])

        joined = pd.merge(
            sleep,
            quiz,
            on=["user_id", "date"],
            how="outer",
            suffixes=("_sleep", "_quiz"),
        )
        joined = pd.merge(joined, profiles, on="user_id", how="left")
        joined = joined.sort_values(["user_id", "date"]).reset_index(drop=True)

        joined.to_parquet(str(DAILY_JOINED), index=False)

        summary = {
            "joined_rows": len(joined),
            "columns": len(joined.columns),
            "unique_users": int(joined["user_id"].nunique()),
        }

        emit_metric("process_synthetic_data", "build_features", summary)
        log.info("Feature join: %s", json.dumps(summary, indent=2))
        return summary

    @task
    def generate_stats_task() -> Dict[str, Any]:
        ensure_dirs()

        df = pd.read_parquet(str(DAILY_JOINED))
        log.info("Loaded %d rows for schema/stats generation", len(df))

        # Schema
        schema = {}
        for col in df.columns:
            schema[col] = {
                "dtype": str(df[col].dtype),
                "nullable": bool(df[col].isna().any()),
                "n_unique": int(df[col].nunique()),
            }
        SCHEMA_REPORT.write_text(json.dumps(schema, indent=2))
        log.info("Schema report -> %s", SCHEMA_REPORT)

        # Stats
        stats = {"row_count": len(df), "column_count": len(df.columns), "columns": {}}
        for col in df.columns:
            col_stats = {
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
                "missing_pct": round(df[col].isna().mean() * 100, 2),
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                desc = df[col].describe()
                col_stats.update(
                    {
                        "mean": round(float(desc.get("mean", 0)), 4),
                        "std": round(float(desc.get("std", 0)), 4),
                        "min": round(float(desc.get("min", 0)), 4),
                        "max": round(float(desc.get("max", 0)), 4),
                    }
                )
            stats["columns"][col] = col_stats
        STATS_REPORT.write_text(json.dumps(stats, indent=2))
        log.info("Stats report -> %s", STATS_REPORT)

        summary = {
            "row_count": stats["row_count"],
            "column_count": stats["column_count"],
        }

        emit_metric("process_synthetic_data", "generate_stats", summary)
        return summary

    trigger_monitoring = TriggerDagRunOperator(
        task_id="trigger_anomaly_and_bias",
        trigger_dag_id="synthetic_anomaly_and_bias",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    # Task flow
    schema = validate_schemas()

    sleep = preprocess_sleep()
    quiz = preprocess_quiz()
    schema >> [sleep, quiz]

    features = build_features()
    [sleep, quiz] >> features

    stats = generate_stats_task()
    features >> stats

    stats >> trigger_monitoring
