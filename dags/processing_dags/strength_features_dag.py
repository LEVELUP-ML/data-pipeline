"""
DAG: strength_features

Reads cleaned weightlifting data, builds supervised learning features
for strength score forecasting, and writes data/processed/strength_features.parquet.

Each training row = a user at reference session t, with:
  - last N_LAGS=5 sessions as strength features (volume, max weight, etc.)
  - target strength scores at t+1, t+3, t+7, t+14 days ahead

Triggers strength_model DAG on success.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pendulum
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from dag_monitoring import (
    emit_metric, log, monitored_dag_args,
    on_dag_failure_callback, on_sla_miss_callback,
)

AIRFLOW_HOME  = "/opt/airflow"
PROCESSED_DIR = Path(f"{AIRFLOW_HOME}/data/processed")
FEATURES_PATH = PROCESSED_DIR / "strength_features.parquet"

N_LAGS   = 5
MIN_SESS = 7
HORIZONS = [1, 3, 7, 14]
AGE_BINS = [(0, 19, 0), (20, 29, 1), (30, 39, 2), (40, 200, 3)]


def _age_enc(age):
    try:
        a = int(age)
        for lo, hi, code in AGE_BINS:
            if lo <= a <= hi:
                return code
    except (TypeError, ValueError):
        pass
    return -1


def _slope(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    x = np.arange(len(vals), dtype=float) - np.arange(len(vals), dtype=float).mean()
    y = np.array(vals, dtype=float)
    d = (x ** 2).sum()
    return float(np.dot(x, y) / d) if d else 0.0


def _future_strength_score(dates, scores, ref, h):
    target = ref + pd.Timedelta(days=h)
    best, best_gap = None, float("inf")
    for d, s in zip(dates, scores):
        g = abs((d - target).days)
        if g <= 2 and g < best_gap:
            best, best_gap = s, g
    return best


@task
def load_cleaned_data() -> pd.DataFrame:
    """Load cleaned weightlifting data and compute per-session strength metrics."""
    clean_path = PROCESSED_DIR / "weightlifting_cleaned"
    if not clean_path.exists():
        raise AirflowFailException(f"Cleaned weightlifting data not found: {clean_path}")

    # Load all cleaned parquet files
    dfs = []
    for f in clean_path.glob("*.parquet"):
        df = pd.read_parquet(f)
        dfs.append(df)

    if not dfs:
        raise AirflowFailException("No cleaned weightlifting data files found")

    df = pd.concat(dfs, ignore_index=True)

    # Ensure we have required columns
    required = ["Date", "Workout Name", "Exercise Name", "Set Order", "Weight", "Reps"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise AirflowFailException(f"Missing required columns: {missing}")

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Workout Name", "Set Order"]).reset_index(drop=True)

    # Add synthetic user_id if not present (for demo data)
    if "user_id" not in df.columns:
        # Group by workout sessions to create synthetic users
        df["session_id"] = df.groupby(["Date", "Workout Name"]).ngroup()
        df["user_id"] = df["session_id"] % 100  # Distribute across 100 synthetic users

    log.info(f"Loaded {len(df)} weightlifting records from {len(df['user_id'].unique())} users")
    return df


@task
def compute_session_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strength metrics per session per user."""
    # Calculate 1RM estimates using Epley formula: weight * (1 + reps/30)
    df["estimated_1rm"] = df["Weight"] * (1 + df["Reps"] / 30.0)

    # Group by user and date to get session-level metrics
    session_metrics = df.groupby(["user_id", "Date"]).agg({
        "estimated_1rm": ["max", "mean", "sum"],  # Max 1RM, avg 1RM, total volume
        "Weight": ["max", "sum"],  # Max weight, total weight moved
        "Reps": ["sum", "count"],  # Total reps, number of sets
        "Exercise Name": "nunique",  # Exercise variety
    }).reset_index()

    # Flatten column names
    session_metrics.columns = [
        "user_id", "Date",
        "max_1rm", "avg_1rm", "total_1rm_volume",
        "max_weight", "total_weight",
        "total_reps", "num_sets",
        "exercise_variety"
    ]

    # Calculate strength score as weighted combination
    session_metrics["strength_score"] = (
        session_metrics["max_1rm"] * 0.4 +
        session_metrics["total_1rm_volume"] * 0.3 +
        session_metrics["total_weight"] * 0.2 +
        session_metrics["num_sets"] * 0.1
    )

    log.info(f"Computed strength metrics for {len(session_metrics)} sessions")
    return session_metrics


@task
def build_lag_features(session_df: pd.DataFrame) -> pd.DataFrame:
    """Build lag features for each user at each reference session."""
    features = []

    for user_id in session_df["user_id"].unique():
        user_data = session_df[session_df["user_id"] == user_id].sort_values("Date")

        if len(user_data) < MIN_SESS:
            continue  # Skip users with too few sessions

        for i in range(N_LAGS, len(user_data)):
            ref_date = user_data.iloc[i]["Date"]
            past_sessions = user_data.iloc[i-N_LAGS:i]

            # Lag features
            lag_max_1rm = past_sessions["max_1rm"].tolist()
            lag_avg_1rm = past_sessions["avg_1rm"].tolist()
            lag_total_volume = past_sessions["total_1rm_volume"].tolist()
            lag_strength_scores = past_sessions["strength_score"].tolist()

            # Trends
            max_1rm_trend = _slope(lag_max_1rm)
            strength_trend = _slope(lag_strength_scores)

            # Recent performance
            recent_max_1rm = lag_max_1rm[-1]
            recent_avg_1rm = lag_avg_1rm[-1]
            recent_volume = lag_total_volume[-1]

            # Target values (future strength scores)
            future_dates = user_data.iloc[i:]["Date"].tolist()
            future_scores = user_data.iloc[i:]["strength_score"].tolist()

            targets = {}
            for h in HORIZONS:
                targets[f"target_d{h}"] = _future_strength_score(
                    future_dates, future_scores, ref_date, h
                )

            # Only include if we have targets
            if any(t is not None for t in targets.values()):
                features.append({
                    "user_id": user_id,
                    "reference_date": ref_date,
                    # Lag features
                    **{f"lag_max_1rm_{j+1}": v for j, v in enumerate(reversed(lag_max_1rm))},
                    **{f"lag_avg_1rm_{j+1}": v for j, v in enumerate(reversed(lag_avg_1rm))},
                    **{f"lag_volume_{j+1}": v for j, v in enumerate(reversed(lag_total_volume))},
                    **{f"lag_strength_{j+1}": v for j, v in enumerate(reversed(lag_strength_scores))},
                    # Derived features
                    "max_1rm_trend": max_1rm_trend,
                    "strength_trend": strength_trend,
                    "recent_max_1rm": recent_max_1rm,
                    "recent_avg_1rm": recent_avg_1rm,
                    "recent_volume": recent_volume,
                    # Targets
                    **targets
                })

    features_df = pd.DataFrame(features)
    log.info(f"Built {len(features_df)} feature rows from {len(session_df['user_id'].unique())} users")
    return features_df


@task
def join_user_profiles(features_df: pd.DataFrame) -> pd.DataFrame:
    """Add user profile features (age, sex, etc.) - using synthetic profiles for demo."""
    # For demo purposes, create synthetic user profiles
    # In production, this would join with actual user data from Firestore

    np.random.seed(42)  # For reproducible synthetic data
    user_ids = features_df["user_id"].unique()

    profiles = {}
    for uid in user_ids:
        profiles[uid] = {
            "age": np.random.randint(18, 65),
            "sex": np.random.choice(["Male", "Female"]),
            "height_cm": np.random.normal(170 if np.random.random() > 0.5 else 160, 10),
            "weight_kg": np.random.normal(75, 15),
        }

    # Add profile features to each row
    profile_features = []
    for _, row in features_df.iterrows():
        uid = row["user_id"]
        profile = profiles[uid]

        profile_features.append({
            **row.to_dict(),
            "age": profile["age"],
            "sex": profile["sex"],
            "height_cm": profile["height_cm"],
            "weight_kg": profile["weight_kg"],
            "age_encoded": _age_enc(profile["age"]),
            "sex_encoded": 1 if profile["sex"] == "Male" else 0,
        })

    result_df = pd.DataFrame(profile_features)
    log.info(f"Added profile features for {len(result_df)} rows")
    return result_df


@task
def quality_gate(features_df: pd.DataFrame) -> Dict[str, Any]:
    """Quality checks on the feature dataset."""
    rows = len(features_df)
    users = features_df["user_id"].nunique()

    # Check target coverage
    target_cols = [f"target_d{h}" for h in HORIZONS]
    coverage = {}
    for col in target_cols:
        coverage[col] = features_df[col].notna().mean() * 100

    d1_cov = coverage["target_d1"]
    d14_cov = coverage["target_d14"]

    if rows < 100:
        raise AirflowFailException(f"Only {rows} rows — need ≥100. Generate more workout data.")
    if d1_cov < 50.0:
        raise AirflowFailException(f"target_d1 coverage {d1_cov}% < 50% — check session dates.")

    log.info("Quality gate PASSED: %d rows, %d users, d1_cov=%.1f%%, d14_cov=%.1f%%",
             rows, users, d1_cov, d14_cov)
    emit_metric("strength_features", "quality_gate",
                {"rows": rows, "users": users, "passed": True, **coverage})
    return {"rows": rows, "users": users, "passed": True, **coverage}


@task
def save_features(features_df: pd.DataFrame) -> Dict[str, Any]:
    """Save features to parquet and return metadata."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Drop rows with no targets
    target_cols = [f"target_d{h}" for h in HORIZONS]
    valid_df = features_df.dropna(subset=target_cols, how="all")

    valid_df.to_parquet(FEATURES_PATH, index=False)

    result = {
        "path": str(FEATURES_PATH),
        "rows": len(valid_df),
        "users": valid_df["user_id"].nunique(),
        "features": len([c for c in valid_df.columns if not c.startswith("target_")]),
        "targets": len([c for c in valid_df.columns if c.startswith("target_")]),
    }

    log.info(f"Saved {result['rows']} feature rows to {FEATURES_PATH}")
    emit_metric("strength_features", "save_features", result)
    return result


with DAG(
    dag_id="strength_features",
    start_date=pendulum.datetime(2026, 2, 1, tz="America/New_York"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=monitored_dag_args(retries=1, sla_minutes=60),
    on_failure_callback=on_dag_failure_callback,
    sla_miss_callback=on_sla_miss_callback,
    tags=["strength", "features", "ingest"],
) as dag:

    trigger_model = TriggerDagRunOperator(
        task_id="trigger_strength_model",
        trigger_dag_id="strength_model",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    raw     = load_cleaned_data()
    sessions = compute_session_strength(raw)
    lags    = build_lag_features(sessions)
    joined  = join_user_profiles(lags)
    gate    = quality_gate(joined)
    saved   = save_features(joined)
    saved >> trigger_model