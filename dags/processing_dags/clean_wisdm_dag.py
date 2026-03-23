"""
DAG: clean_wisdm_accel_data

Reads raw WISDM accelerometer txt files, validates, cleans,
windows, computes stamina, runs bias analysis, and writes
processed output to data/processed/wisdm/.

Triggers DVC backup on success.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List

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
RAW_DIR = f"{AIRFLOW_HOME}/data/raw/wisdm"
CLEAN_DIR = f"{AIRFLOW_HOME}/data/processed/wisdm"
REJECT_DIR = f"{AIRFLOW_HOME}/data/processed/wisdm_rejected"

REQUIRED_COLS = ["user", "activity", "x", "y", "z"]
VALID_ACTIVITIES = set("ABCDEFGHIJKLMOPQRS")  # A-S, no N per dataset docs
ACCEL_MAG_FLOOR = 0.0
ACCEL_MAG_CEILING = 200.0  # reasonable accelerometer magnitude cap


#  Loading


def load_wisdm_file(path: str) -> pd.DataFrame:
    """Load a single WISDM accel txt file."""
    df = pd.read_csv(
        path,
        header=None,
        names=["user", "activity", "timestamp", "x", "y", "z"],
        sep=",",
        on_bad_lines="skip",
    )
    df["z"] = df["z"].astype(str).str.replace(";", "", regex=False)
    for col in ("x", "y", "z", "timestamp"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["user"] = pd.to_numeric(df["user"], errors="coerce").astype("Int64")
    df["activity"] = df["activity"].astype(str).str.strip()
    df["_source_file"] = os.path.basename(path)
    return df


#  Validation


def tag_row_issues(row: pd.Series) -> List[str]:
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

    # magnitude check
    if all(pd.notna(row.get(ax)) for ax in ("x", "y", "z")):
        mag = np.sqrt(row["x"] ** 2 + row["y"] ** 2 + row["z"] ** 2)
        if mag > ACCEL_MAG_CEILING:
            issues.append(f"extreme_magnitude({mag:.1f})")

    return issues


#  Anomaly detection


def detect_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    mag = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    threshold = mag.mean() + 5 * mag.std()
    anomaly_mask = mag > threshold
    count = anomaly_mask.sum()
    return {
        "anomaly_count": int(count),
        "threshold": round(float(threshold), 4),
        "mean_magnitude": round(float(mag.mean()), 4),
        "std_magnitude": round(float(mag.std()), 4),
    }


#  Windowing


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


#  Stamina engine


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


#  Bias analysis


def analyze_bias(df: pd.DataFrame) -> Dict[str, float]:
    return df.groupby("activity")["stamina"].mean().round(4).to_dict()


#  DAG

with DAG(
    dag_id="clean_wisdm_accel_data",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=monitored_dag_args(retries=2, sla_minutes=30),
    on_failure_callback=on_dag_failure_callback,
    sla_miss_callback=on_sla_miss_callback,
    tags=["cleaning", "wisdm", "accelerometer", "data-quality"],
) as dag:

    @task
    def discover_files() -> List[str]:
        if not os.path.isdir(RAW_DIR):
            raise AirflowFailException(
                f"Raw dir {RAW_DIR} not found. Run download_wisdm_accel first."
            )
        paths = sorted(
            [
                os.path.join(RAW_DIR, f)
                for f in os.listdir(RAW_DIR)
                if f.endswith(".txt") and "accel" in f
            ]
        )
        if not paths:
            raise AirflowFailException(f"No accel txt files in {RAW_DIR}")
        log.info("Discovered %d accel files", len(paths))
        return paths

    @task
    def load_and_validate(file_paths: List[str]) -> Dict[str, Any]:
        frames = []
        for p in file_paths:
            log.info("Loading %s", os.path.basename(p))
            frames.append(load_wisdm_file(p))

        raw = pd.concat(frames, ignore_index=True)
        total_raw = len(raw)
        log.info("Total raw rows: %d", total_raw)

        # Schema check
        missing = set(REQUIRED_COLS) - set(raw.columns)
        if missing:
            log.error("Missing columns: %s", missing)
            raise AirflowFailException(f"Missing columns: {missing}")

        # Deduplicate
        before = len(raw)
        raw = raw.drop_duplicates(
            subset=["user", "activity", "timestamp", "x", "y", "z"]
        )
        dupes = before - len(raw)
        if dupes:
            log.warning("Dropped %d duplicate rows", dupes)

        # Row-level validation
        raw["_issues"] = raw.apply(tag_row_issues, axis=1)
        raw["_is_clean"] = raw["_issues"].apply(lambda x: len(x) == 0)

        clean = raw[raw["_is_clean"]].copy()
        rejected = raw[~raw["_is_clean"]].copy()

        clean = clean.drop(columns=["_issues", "_is_clean"])
        clean = clean.sort_values(["user", "timestamp"]).reset_index(drop=True)

        # Write outputs
        os.makedirs(CLEAN_DIR, exist_ok=True)
        os.makedirs(REJECT_DIR, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

        clean_path = os.path.join(CLEAN_DIR, f"wisdm_accel_clean_{ts}.parquet")
        clean.to_parquet(clean_path, index=False)
        log.info("Wrote %d clean rows -> %s", len(clean), clean_path)

        reject_path = os.path.join(REJECT_DIR, f"wisdm_accel_rejected_{ts}.csv")
        if len(rejected) > 0:
            rejected["_issues"] = rejected["_issues"].apply(lambda x: "; ".join(x))
            rejected = rejected.drop(columns=["_is_clean"])
            rejected.to_csv(reject_path, index=False)
            log.warning("Wrote %d rejected rows -> %s", len(rejected), reject_path)

        summary = {
            "total_raw": total_raw,
            "duplicates_dropped": dupes,
            "clean_rows": len(clean),
            "rejected_rows": len(rejected),
            "reject_pct": (
                round(len(rejected) / total_raw * 100, 2) if total_raw else 0.0
            ),
            "unique_users": int(clean["user"].nunique()),
            "unique_activities": int(clean["activity"].nunique()),
            "clean_path": clean_path,
            "reject_path": reject_path if len(rejected) > 0 else None,
        }

        emit_metric("clean_wisdm_accel_data", "load_and_validate", summary)
        log.info("Validation summary: %s", json.dumps(summary, indent=2))
        return summary

    @task
    def quality_gate(summary: Dict[str, Any]) -> str:
        max_reject_pct = 10.0
        if summary["reject_pct"] > max_reject_pct:
            log.error(
                "Reject rate %.2f%% exceeds %.1f%%",
                summary["reject_pct"],
                max_reject_pct,
            )
            raise AirflowFailException(
                f"Rejection rate {summary['reject_pct']}% exceeds {max_reject_pct}%"
            )
        if summary["clean_rows"] == 0:
            raise AirflowFailException("Zero clean rows")
        log.info(
            "Quality gate PASSED: %d clean, %.2f%% rejected",
            summary["clean_rows"],
            summary["reject_pct"],
        )
        return summary["clean_path"]

    @task
    def run_anomaly_detection(clean_path: str) -> str:
        df = pd.read_parquet(clean_path)
        result = detect_anomalies(df)
        log.info("Anomaly detection: %s", result)

        emit_metric("clean_wisdm_accel_data", "anomaly_detection", result)

        if result["anomaly_count"] > 0:
            log.warning(
                "%d anomalies detected (threshold=%.2f)",
                result["anomaly_count"],
                result["threshold"],
            )
        else:
            log.info("No anomalies detected")
        return clean_path

    @task
    def window_and_compute_stamina(clean_path: str) -> Dict[str, Any]:
        df = pd.read_parquet(clean_path)
        log.info("Windowing %d rows (window_size=200)", len(df))

        windowed = create_row_windows(df, window_size=200)
        log.info("Created %d windows", len(windowed))

        if windowed.empty:
            raise AirflowFailException("Windowed dataset is empty")

        log.info("Computing stamina across %d windows", len(windowed))
        result = compute_stamina(windowed, max_stamina=100.0, fatigue_rate=0.01)

        # Bias analysis
        bias = analyze_bias(result)
        log.info("Stamina by activity: %s", json.dumps(bias, indent=2))

        emit_metric(
            "clean_wisdm_accel_data",
            "stamina_computation",
            {
                "windows": len(result),
                "mean_stamina": round(float(result["stamina"].mean()), 4),
                "min_stamina": round(float(result["stamina"].min()), 4),
                "max_stamina": round(float(result["stamina"].max()), 4),
                "activities": len(bias),
            },
        )

        # Save
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        out_path = os.path.join(CLEAN_DIR, f"wisdm_stamina_{ts}.parquet")
        result.to_parquet(out_path, index=False)
        log.info("Wrote stamina output (%d rows) -> %s", len(result), out_path)

        bias_path = os.path.join(CLEAN_DIR, f"wisdm_bias_{ts}.json")
        with open(bias_path, "w") as f:
            json.dump(bias, f, indent=2)
        log.info("Wrote bias report -> %s", bias_path)

        return {
            "stamina_path": out_path,
            "bias_path": bias_path,
            "windows": len(result),
            "bias_summary": bias,
        }

    trigger_dvc = TriggerDagRunOperator(
        task_id="trigger_dvc_backup",
        trigger_dag_id="dvc_backup_to_gcp",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    files = discover_files()
    summary = load_and_validate(files)
    clean_path = quality_gate(summary)
    checked_path = run_anomaly_detection(clean_path)
    stamina_result = window_and_compute_stamina(checked_path)
    stamina_result >> trigger_dvc
