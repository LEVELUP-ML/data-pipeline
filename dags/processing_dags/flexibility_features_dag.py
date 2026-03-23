"""
DAG: flexibility_features

Fetches flexibility_workouts from Firestore, builds supervised learning
features for multi-step score forecasting, joins with user profiles,
and writes data/processed/flexibility_features.parquet.

Each training row = a user at reference session t, with:
  - last N_LAGS=5 sessions as lag features
  - target scores at t+1, t+3, t+7, t+14 days ahead

Triggers flexibility_model DAG on success.
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

import firebase_admin
from firebase_admin import credentials, firestore
from dag_monitoring import (
    emit_metric, log, monitored_dag_args,
    on_dag_failure_callback, on_sla_miss_callback,
)

AIRFLOW_HOME  = "/opt/airflow"
RAW_DIR       = Path(f"{AIRFLOW_HOME}/data/raw")
PROCESSED_DIR = Path(f"{AIRFLOW_HOME}/data/processed")
FEATURES_PATH = PROCESSED_DIR / "flexibility_features.parquet"

N_LAGS   = 5
MIN_SESS = 7
HORIZONS = [1, 3, 7, 14]
AGE_BINS = [(0, 19, 0), (20, 29, 1), (30, 39, 2), (40, 200, 3)]


def _get_db():
    if not firebase_admin._apps:
        sa = Variable.get("FIREBASE_SERVICE_ACCOUNT_PATH")
        if not os.path.exists(sa):
            raise AirflowFailException(f"Service account not found: {sa}")
        firebase_admin.initialize_app(credentials.Certificate(sa))
    return firestore.client()


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
    dag_id="flexibility_features",
    start_date=pendulum.datetime(2026, 2, 1, tz="America/New_York"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=monitored_dag_args(retries=2, sla_minutes=30),
    on_failure_callback=on_dag_failure_callback,
    sla_miss_callback=on_sla_miss_callback,
    tags=["flexibility", "features", "modeling"],
) as dag:

    @task
    def download_workout_sessions() -> Dict[str, Any]:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        db = _get_db()

        users = list(db.collection("users").stream())
        log.info("Found %d users", len(users))

        all_workouts, profiles = [], []

        for u in users:
            uid    = u.id
            udata  = u.to_dict() or {}
            prof   = udata.get("profile", {})
            profiles.append({
                "user_id": uid, "age": prof.get("age"), "sex": prof.get("sex"),
                "height_cm": prof.get("height_cm"), "weight_kg": prof.get("weight_kg"),
            })
            for w in (db.collection("users").document(uid)
                       .collection("flexibility_workouts").stream()):
                d = w.to_dict() or {}
                for f in ("timestamp", "seededAt"):
                    v = d.get(f)
                    if hasattr(v, "isoformat"):
                        d[f] = v.isoformat()
                d["user_id"] = uid
                all_workouts.append(d)

        if not all_workouts:
            raise AirflowFailException(
                "No flexibility_workouts found. Run data_seeding/main.py first."
            )

        (RAW_DIR / "flexibility_workouts_raw.jsonl").write_text(
            "\n".join(json.dumps(r, default=str) for r in all_workouts)
        )
        (RAW_DIR / "profiles_raw.json").write_text(json.dumps(profiles, indent=2))

        result = {"users": len(users), "workouts": len(all_workouts)}
        log.info("Downloaded: %s", result)
        emit_metric("flexibility_features", "download_sessions", result)
        return result

    @task
    def build_lag_features(download_result: Dict) -> Dict[str, Any]:
        raw_path = RAW_DIR / "flexibility_workouts_raw.jsonl"
        if not raw_path.exists():
            raise AirflowFailException(f"Missing: {raw_path}")

        recs = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
        df   = pd.DataFrame(recs)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "user_id"]).sort_values(["user_id", "date"]).reset_index(drop=True)

        for c in ("score_after", "effort_level", "session_duration_min",
                  "sit_and_reach_cm", "streak_days", "rest_days_before"):
            df[c] = pd.to_numeric(df.get(c, pd.Series(dtype=float)), errors="coerce")

        rows: List[Dict] = []

        for uid, grp in df.groupby("user_id"):
            grp    = grp.sort_values("date").reset_index(drop=True)
            n      = len(grp)
            if n < MIN_SESS:
                continue

            dates  = grp["date"].tolist()
            scores = grp["score_after"].tolist()

            for t in range(N_LAGS, n):
                ref_date = dates[t]
                row: Dict[str, Any] = {
                    "user_id":  uid,
                    "ref_date": ref_date,
                    "ref_score": scores[t],
                }

                # lag features: lag_1 = most recent session before t
                for k in range(N_LAGS):
                    idx = t - N_LAGS + k    # oldest → newest
                    lag = N_LAGS - k        # lag_5 → lag_1
                    s   = grp.iloc[idx]
                    row[f"score_lag_{lag}"]    = s["score_after"]
                    row[f"effort_lag_{lag}"]   = s["effort_level"]
                    row[f"duration_lag_{lag}"] = s["session_duration_min"]
                    row[f"reach_lag_{lag}"]    = s["sit_and_reach_cm"]
                    row[f"days_ago_lag_{lag}"] = (ref_date - s["date"]).days

                last5_scores  = scores[t - N_LAGS: t]
                last5_efforts = grp["effort_level"].iloc[t - N_LAGS: t].tolist()

                row["workout_count_7d"]  = sum(1 for d in dates[:t] if (ref_date - d).days <= 7)
                row["workout_count_14d"] = sum(1 for d in dates[:t] if (ref_date - d).days <= 14)
                row["mean_score_5"]      = float(np.nanmean(last5_scores))
                row["score_trend_5"]     = _slope(last5_scores)
                row["mean_effort_5"]     = float(np.nanmean(last5_efforts))
                row["days_since_last"]   = (ref_date - dates[t - 1]).days
                row["current_streak"]    = int(grp.iloc[t]["streak_days"])

                for h in HORIZONS:
                    row[f"target_d{h}"] = _future_score(dates, scores, ref_date, h)

                rows.append(row)

        if not rows:
            raise AirflowFailException(
                f"No feature rows built. Need ≥{MIN_SESS} sessions per user."
            )

        out = pd.DataFrame(rows)
        (PROCESSED_DIR / "flexibility_features_noprofile.parquet").parent.mkdir(
            parents=True, exist_ok=True
        )
        out.to_parquet(str(PROCESSED_DIR / "flexibility_features_noprofile.parquet"), index=False)

        result = {
            "rows":   len(out),
            "users":  int(out["user_id"].nunique()),
            "d1_null_pct":  round(out["target_d1"].isna().mean() * 100, 2),
            "d14_null_pct": round(out["target_d14"].isna().mean() * 100, 2),
        }
        log.info("Lag features: %s", result)
        emit_metric("flexibility_features", "build_lag_features", result)
        return result

    @task
    def join_profiles(lag_result: Dict) -> Dict[str, Any]:
        feat_path = PROCESSED_DIR / "flexibility_features_noprofile.parquet"
        prof_path = RAW_DIR / "profiles_raw.json"

        for p in (feat_path, prof_path):
            if not p.exists():
                raise AirflowFailException(f"Missing: {p}")

        feat = pd.read_parquet(str(feat_path))
        profs = pd.DataFrame(json.loads(prof_path.read_text()))

        profs["bmr"]           = profs.apply(lambda r: _bmr(r.get("age"), r.get("sex"), r.get("height_cm"), r.get("weight_kg")), axis=1)
        profs["sex_encoded"]   = profs["sex"].map({"Male": 1, "male": 1, "M": 1, "Female": 0, "female": 0, "F": 0})
        profs["age_bucket_enc"] = profs["age"].apply(_age_enc)

        # Keep raw sex/age columns separately for bias slicing
        profs_model = profs[["user_id", "age", "sex_encoded", "bmr", "age_bucket_enc"]].copy()
        profs_bias  = profs[["user_id", "sex", "age"]].rename(columns={"sex": "sex_raw", "age": "age_raw"})

        joined = pd.merge(feat, profs_model, on="user_id", how="left")
        joined = pd.merge(joined, profs_bias,  on="user_id", how="left")
        joined = joined.sort_values(["user_id", "ref_date"]).reset_index(drop=True)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        joined.to_parquet(str(FEATURES_PATH), index=False)

        result = {
            "final_rows":         len(joined),
            "columns":            len(joined.columns),
            "users":              int(joined["user_id"].nunique()),
            "missing_bmr_pct":    round(joined["bmr"].isna().mean() * 100, 2),
            "d1_coverage_pct":    round((1 - joined["target_d1"].isna().mean()) * 100, 2),
            "d14_coverage_pct":   round((1 - joined["target_d14"].isna().mean()) * 100, 2),
        }
        log.info("Final features: %s", result)
        emit_metric("flexibility_features", "join_profiles", result)
        return result

    @task
    def quality_gate(download_r: Dict, lag_r: Dict, join_r: Dict) -> Dict[str, Any]:
        rows    = join_r["final_rows"]
        users   = join_r["users"]
        d1_cov  = join_r["d1_coverage_pct"]
        d14_cov = join_r["d14_coverage_pct"]

        if rows < 100:
            raise AirflowFailException(f"Only {rows} rows — need ≥100. Seed more users/days.")
        if d1_cov < 50.0:
            raise AirflowFailException(f"target_d1 coverage {d1_cov}% < 50% — check session dates.")

        log.info("Quality gate PASSED: %d rows, %d users, d1_cov=%.1f%%, d14_cov=%.1f%%",
                 rows, users, d1_cov, d14_cov)
        emit_metric("flexibility_features", "quality_gate",
                    {"rows": rows, "users": users, "passed": True})
        return {"rows": rows, "users": users, "passed": True}

    trigger_model = TriggerDagRunOperator(
        task_id="trigger_flexibility_model",
        trigger_dag_id="flexibility_model",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    dl     = download_workout_sessions()
    lags   = build_lag_features(dl)
    joined = join_profiles(lags)
    gate   = quality_gate(dl, lags, joined)
    gate >> trigger_model