"""
DAG: clean_wisdm_accel_data

Reads raw WISDM accelerometer txt files, validates, cleans,
windows, computes stamina, runs bias analysis, and writes
processed output to data/processed/wisdm/.

Core logic lives in dags/lib/wisdm.py — import from there for tests.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from dag_monitoring import (
    emit_metric,
    log,
    monitored_dag_args,
    on_dag_failure_callback,
    on_sla_miss_callback,
)
from lib.wisdm import (
    analyze_bias,
    compute_stamina,
    create_row_windows,
    detect_anomalies,
    load_wisdm_file,
    tag_row_issues,
    validate_schema,
)

AIRFLOW_HOME = "/opt/airflow"
RAW_DIR      = f"{AIRFLOW_HOME}/data/raw/wisdm"
CLEAN_DIR    = f"{AIRFLOW_HOME}/data/processed/wisdm"
REJECT_DIR   = f"{AIRFLOW_HOME}/data/processed/wisdm_rejected"

MAX_REJECT_PCT = 10.0


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
            os.path.join(RAW_DIR, f)
            for f in os.listdir(RAW_DIR)
            if f.endswith(".txt") and "accel" in f
        )
        if not paths:
            raise AirflowFailException(f"No accel txt files in {RAW_DIR}")
        log.info("Discovered %d accel files", len(paths))
        return paths

    @task
    def load_and_validate(file_paths: List[str]) -> Dict[str, Any]:
        frames = [load_wisdm_file(p) for p in file_paths]
        raw = pd.concat(frames, ignore_index=True)
        total_raw = len(raw)

        missing = validate_schema(raw)
        if missing:
            raise AirflowFailException(f"Missing columns: {missing}")

        before = len(raw)
        raw = raw.drop_duplicates(subset=["user", "activity", "timestamp", "x", "y", "z"])
        dupes = before - len(raw)
        if dupes:
            log.warning("Dropped %d duplicate rows", dupes)

        raw["_issues"] = raw.apply(tag_row_issues, axis=1)
        raw["_is_clean"] = raw["_issues"].apply(lambda x: len(x) == 0)

        clean    = raw[raw["_is_clean"]].drop(columns=["_issues", "_is_clean"])
        rejected = raw[~raw["_is_clean"]].copy()

        clean = clean.sort_values(["user", "timestamp"]).reset_index(drop=True)

        os.makedirs(CLEAN_DIR, exist_ok=True)
        os.makedirs(REJECT_DIR, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

        clean_path = os.path.join(CLEAN_DIR, f"wisdm_accel_clean_{ts}.parquet")
        clean.to_parquet(clean_path, index=False)

        reject_path = os.path.join(REJECT_DIR, f"wisdm_accel_rejected_{ts}.csv")
        if len(rejected):
            rejected["_issues"] = rejected["_issues"].apply("; ".join)
            rejected.drop(columns=["_is_clean"]).to_csv(reject_path, index=False)

        summary = {
            "total_raw": total_raw,
            "duplicates_dropped": dupes,
            "clean_rows": len(clean),
            "rejected_rows": len(rejected),
            "reject_pct": round(len(rejected) / total_raw * 100, 2) if total_raw else 0.0,
            "unique_users": int(clean["user"].nunique()),
            "unique_activities": int(clean["activity"].nunique()),
            "clean_path": clean_path,
            "reject_path": reject_path if len(rejected) else None,
        }
        emit_metric("clean_wisdm_accel_data", "load_and_validate", summary)
        log.info("Validation summary: %s", json.dumps(summary, indent=2))
        return summary

    @task
    def quality_gate(summary: Dict[str, Any]) -> str:
        if summary["reject_pct"] > MAX_REJECT_PCT:
            raise AirflowFailException(
                f"Rejection rate {summary['reject_pct']}% exceeds {MAX_REJECT_PCT}%"
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
        emit_metric("clean_wisdm_accel_data", "anomaly_detection", result)
        log.info("Anomaly detection: %s", result)
        if result["anomaly_count"]:
            log.warning(
                "%d anomalies detected (threshold=%.2f)",
                result["anomaly_count"],
                result["threshold"],
            )
        return clean_path

    @task
    def window_and_compute_stamina(clean_path: str) -> Dict[str, Any]:
        df = pd.read_parquet(clean_path)
        log.info("Windowing %d rows (window_size=200)", len(df))

        windowed = create_row_windows(df, window_size=200)
        if windowed.empty:
            raise AirflowFailException("Windowed dataset is empty")

        result = compute_stamina(windowed)
        bias   = analyze_bias(result)

        emit_metric(
            "clean_wisdm_accel_data",
            "stamina_computation",
            {
                "windows": len(result),
                "mean_stamina": round(float(result["stamina"].mean()), 4),
                "min_stamina":  round(float(result["stamina"].min()),  4),
                "max_stamina":  round(float(result["stamina"].max()),  4),
                "activities":   len(bias),
            },
        )

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        out_path  = os.path.join(CLEAN_DIR, f"wisdm_stamina_{ts}.parquet")
        bias_path = os.path.join(CLEAN_DIR, f"wisdm_bias_{ts}.json")

        result.to_parquet(out_path, index=False)
        with open(bias_path, "w") as f:
            json.dump(bias, f, indent=2)

        log.info("Wrote stamina -> %s, bias -> %s", out_path, bias_path)
        return {"stamina_path": out_path, "bias_path": bias_path, "windows": len(result)}

    trigger_dvc = TriggerDagRunOperator(
        task_id="trigger_dvc_backup",
        trigger_dag_id="dvc_backup_to_gcp",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    files         = discover_files()
    summary       = load_and_validate(files)
    clean_path    = quality_gate(summary)
    checked_path  = run_anomaly_detection(clean_path)
    stamina_result = window_and_compute_stamina(checked_path)
    stamina_result >> trigger_dvc