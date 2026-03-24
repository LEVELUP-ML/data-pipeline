"""
DAG: strength_features

Reads cleaned weightlifting data, builds supervised learning features
for strength score forecasting, and writes data/processed/strength_features.parquet.

Each training row = a user at reference session t, with:
  - last N_LAGS=5 sessions as lag features (1RM, volume, etc.)
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
RAW_DIR       = Path(f"{AIRFLOW_HOME}/data/raw")
PROCESSED_DIR = Path(f"{AIRFLOW_HOME}/data/processed")
SESSIONS_PATH = PROCESSED_DIR / "strength_sessions.parquet"
FEATURES_PATH = PROCESSED_DIR / "strength_features.parquet"

N_LAGS   = 5
MIN_SESS = 7
HORIZONS = [1, 3, 7, 14]
AGE_BINS = [(0, 19, 0), (20, 29, 1), (30, 39, 2), (40, 200, 3)]


def _bmr(age, sex, h, w):
    try:
        b = 10 * float(w) + 6.25 * float(h) - 5 * float(age)
        return round(b - 161 if str(sex).lower() in ("female", "f") else b + 5)
    except (TypeError, ValueError):
        return None


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


def _future_score(dates, scores, ref, h):
    target = ref + pd.Timedelta(days=h)
    best, best_gap = None, float("inf")
    for d, s in zip(dates, scores):
        g = abs((d - target).days)
        if g <= 2 and g < best_gap:
            best, best_gap = s, g
    return best


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

    @task
    def load_strength_sessions() -> Dict[str, Any]:
        """Load cleaned weightlifting data, compute per-session strength metrics, save sessions parquet."""
        clean_path = PROCESSED_DIR / "weightlifting_cleaned"
        if not clean_path.exists():
            raise AirflowFailException(f"Cleaned weightlifting data not found: {clean_path}")

        dfs = []
        for f in clean_path.glob("*.parquet"):
            dfs.append(pd.read_parquet(f))

        if not dfs:
            raise AirflowFailException("No cleaned weightlifting data files found")

        df = pd.concat(dfs, ignore_index=True)

        required = ["Date", "Workout Name", "Exercise Name", "Set Order", "Weight", "Reps"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise AirflowFailException(f"Missing required columns: {missing}")

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Date", "Workout Name", "Set Order"]).reset_index(drop=True)

        if "user_id" not in df.columns:
            df["session_id"] = df.groupby(["Date", "Workout Name"]).ngroup()
            df["user_id"] = df["session_id"] % 100

        # Epley 1RM estimate: weight * (1 + reps/30)
        df["estimated_1rm"] = df["Weight"] * (1 + df["Reps"] / 30.0)

        sessions = df.groupby(["user_id", "Date"]).agg(
            max_1rm=("estimated_1rm", "max"),
            avg_1rm=("estimated_1rm", "mean"),
            total_1rm_volume=("estimated_1rm", "sum"),
            max_weight=("Weight", "max"),
            total_weight=("Weight", "sum"),
            total_reps=("Reps", "sum"),
            num_sets=("Reps", "count"),
            exercise_variety=("Exercise Name", "nunique"),
        ).reset_index()

        sessions["strength_score"] = (
            sessions["max_1rm"] * 0.4
            + sessions["total_1rm_volume"] * 0.3
            + sessions["total_weight"] * 0.2
            + sessions["num_sets"] * 0.1
        )

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        sessions.to_parquet(SESSIONS_PATH, index=False)

        result = {"users": int(sessions["user_id"].nunique()), "sessions": len(sessions)}
        log.info("Loaded %d sessions from %d users", result["sessions"], result["users"])
        emit_metric("strength_features", "load_strength_sessions", result)
        return result

    @task
    def build_lag_features(load_result: Dict) -> Dict[str, Any]:
        if not SESSIONS_PATH.exists():
            raise AirflowFailException(f"Missing: {SESSIONS_PATH}")

        sessions = pd.read_parquet(SESSIONS_PATH)
        sessions["Date"] = pd.to_datetime(sessions["Date"])
        sessions = sessions.sort_values(["user_id", "Date"]).reset_index(drop=True)

        rows: List[Dict] = []

        for uid, grp in sessions.groupby("user_id"):
            grp    = grp.sort_values("Date").reset_index(drop=True)
            n      = len(grp)
            if n < MIN_SESS:
                continue

            dates  = grp["Date"].tolist()
            scores = grp["strength_score"].tolist()

            for t in range(N_LAGS, n):
                ref_date = dates[t]
                row: Dict[str, Any] = {
                    "user_id":   uid,
                    "ref_date":  ref_date,
                    "ref_score": scores[t],
                }

                # lag features: lag_1 = most recent session before t
                for k in range(N_LAGS):
                    idx = t - N_LAGS + k    # oldest → newest
                    lag = N_LAGS - k        # lag_5 → lag_1
                    s   = grp.iloc[idx]
                    row[f"max_1rm_lag_{lag}"]   = s["max_1rm"]
                    row[f"avg_1rm_lag_{lag}"]   = s["avg_1rm"]
                    row[f"volume_lag_{lag}"]    = s["total_1rm_volume"]
                    row[f"strength_lag_{lag}"]  = s["strength_score"]
                    row[f"days_ago_lag_{lag}"]  = (ref_date - s["Date"]).days

                last5_scores = scores[t - N_LAGS: t]

                row["workout_count_7d"]  = sum(1 for d in dates[:t] if (ref_date - d).days <= 7)
                row["workout_count_14d"] = sum(1 for d in dates[:t] if (ref_date - d).days <= 14)
                row["mean_score_5"]      = float(np.nanmean(last5_scores))
                row["score_trend_5"]     = _slope(last5_scores)
                row["days_since_last"]   = (ref_date - dates[t - 1]).days

                for h in HORIZONS:
                    row[f"target_d{h}"] = _future_score(dates, scores, ref_date, h)

                rows.append(row)

        if not rows:
            raise AirflowFailException(
                f"No feature rows built. Need ≥{MIN_SESS} sessions per user."
            )

        out = pd.DataFrame(rows)
        (PROCESSED_DIR / "strength_features_noprofile.parquet").parent.mkdir(
            parents=True, exist_ok=True
        )
        out.to_parquet(str(PROCESSED_DIR / "strength_features_noprofile.parquet"), index=False)

        result = {
            "rows":         len(out),
            "users":        int(out["user_id"].nunique()),
            "d1_null_pct":  round(out["target_d1"].isna().mean() * 100, 2),
            "d14_null_pct": round(out["target_d14"].isna().mean() * 100, 2),
        }
        log.info("Lag features: %s", result)
        emit_metric("strength_features", "build_lag_features", result)
        return result

    @task
    def join_profiles(lag_result: Dict) -> Dict[str, Any]:
        feat_path = PROCESSED_DIR / "strength_features_noprofile.parquet"
        if not feat_path.exists():
            raise AirflowFailException(f"Missing: {feat_path}")

        feat = pd.read_parquet(str(feat_path))

        # Synthetic profiles (strength data has no Firestore user profiles)
        user_ids = sorted(feat["user_id"].unique())
        profiles = []
        for uid in user_ids:
            rng = np.random.default_rng(int(uid))
            sex = rng.choice(["Male", "Female"])
            age = int(rng.integers(18, 65))
            h   = float(rng.normal(170 if sex == "Male" else 160, 10))
            w   = float(rng.normal(75, 15))
            profiles.append({"user_id": uid, "age": age, "sex": sex,
                              "height_cm": h, "weight_kg": w})

        profs = pd.DataFrame(profiles)
        profs["bmr"]            = profs.apply(
            lambda r: _bmr(r["age"], r["sex"], r["height_cm"], r["weight_kg"]), axis=1
        )
        profs["sex_encoded"]    = profs["sex"].map({"Male": 1, "male": 1, "M": 1, "Female": 0, "female": 0, "F": 0})
        profs["age_bucket_enc"] = profs["age"].apply(_age_enc)

        # Keep raw sex/age columns separately for bias slicing (mirrors flexibility)
        profs_model = profs[["user_id", "age", "sex_encoded", "bmr", "age_bucket_enc"]].copy()
        profs_bias  = profs[["user_id", "sex", "age"]].rename(
            columns={"sex": "sex_raw", "age": "age_raw"}
        )

        joined = pd.merge(feat, profs_model, on="user_id", how="left")
        joined = pd.merge(joined, profs_bias,  on="user_id", how="left")
        joined = joined.sort_values(["user_id", "ref_date"]).reset_index(drop=True)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        joined.to_parquet(str(FEATURES_PATH), index=False)

        result = {
            "final_rows":       len(joined),
            "columns":          len(joined.columns),
            "users":            int(joined["user_id"].nunique()),
            "missing_bmr_pct":  round(joined["bmr"].isna().mean() * 100, 2),
            "d1_coverage_pct":  round((1 - joined["target_d1"].isna().mean()) * 100, 2),
            "d14_coverage_pct": round((1 - joined["target_d14"].isna().mean()) * 100, 2),
        }
        log.info("Final features: %s", result)
        emit_metric("strength_features", "join_profiles", result)
        return result

    @task
    def quality_gate(load_r: Dict, lag_r: Dict, join_r: Dict) -> Dict[str, Any]:
        rows    = join_r["final_rows"]
        users   = join_r["users"]
        d1_cov  = join_r["d1_coverage_pct"]
        d14_cov = join_r["d14_coverage_pct"]

        if rows < 100:
            raise AirflowFailException(f"Only {rows} rows — need ≥100. Generate more workout data.")
        if d1_cov < 50.0:
            raise AirflowFailException(f"target_d1 coverage {d1_cov}% < 50% — check session dates.")

        log.info("Quality gate PASSED: %d rows, %d users, d1_cov=%.1f%%, d14_cov=%.1f%%",
                 rows, users, d1_cov, d14_cov)
        emit_metric("strength_features", "quality_gate",
                    {"rows": rows, "users": users, "passed": True})
        return {"rows": rows, "users": users, "passed": True}

    trigger_model = TriggerDagRunOperator(
        task_id="trigger_strength_model",
        trigger_dag_id="strength_model",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    load_r   = load_strength_sessions()
    lags     = build_lag_features(load_r)
    joined   = join_profiles(lags)
    gate     = quality_gate(load_r, lags, joined)
    gate >> trigger_model
