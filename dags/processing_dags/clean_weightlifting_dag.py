"""
DAG: clean_weightlifting_data

Reads raw CSVs from the Kaggle ingest dirs, applies cleaning /
validation rules, writes cleaned Parquet + a rejection log, then
triggers the DVC backup DAG.

Core logic lives in dags/lib/weightlifting.py — import from there for tests.
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
from lib.weightlifting import clean_dataframe, validate_schema

AIRFLOW_HOME = "/opt/airflow"
RAW_DIRS = [
    f"{AIRFLOW_HOME}/data/raw/weightlifting",
    f"{AIRFLOW_HOME}/data/raw/strength",
]
CLEAN_DIR  = f"{AIRFLOW_HOME}/data/processed/weightlifting_cleaned"
REJECT_DIR = f"{AIRFLOW_HOME}/data/processed/weightlifting_rejected"

MAX_REJECT_PCT = 15.0


def _find_csvs() -> List[str]:
    paths = []
    for d in RAW_DIRS:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(".csv"):
                paths.append(os.path.join(d, f))
    return sorted(set(paths))


with DAG(
    dag_id="clean_weightlifting_data",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=monitored_dag_args(retries=2, sla_minutes=30),
    on_failure_callback=on_dag_failure_callback,
    sla_miss_callback=on_sla_miss_callback,
    tags=["cleaning", "weightlifting", "data-quality"],
) as dag:

    @task
    def discover_files() -> List[str]:
        paths = _find_csvs()
        if not paths:
            raise AirflowFailException(
                f"No CSVs found in {RAW_DIRS}. Run a kaggle_download DAG first."
            )
        log.info("Discovered %d CSV(s): %s", len(paths), paths)
        return paths

    @task
    def clean_and_validate(csv_paths: List[str]) -> Dict[str, Any]:
        frames = []
        for p in csv_paths:
            df = pd.read_csv(p, dtype=str)
            df["_source_file"] = os.path.basename(p)
            frames.append(df)

        raw = pd.concat(frames, ignore_index=True)
        total_raw = len(raw)

        missing_cols = validate_schema(raw)
        if missing_cols:
            raise AirflowFailException(f"CSV(s) missing required columns: {missing_cols}")

        result = clean_dataframe(raw)
        clean    = result["clean"]
        rejected = result["rejected"]

        os.makedirs(CLEAN_DIR, exist_ok=True)
        os.makedirs(REJECT_DIR, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

        clean_path = os.path.join(CLEAN_DIR, f"workouts_clean_{ts}.parquet")
        clean.to_parquet(clean_path, index=False)

        reject_path = os.path.join(REJECT_DIR, f"workouts_rejected_{ts}.csv")
        if len(rejected):
            out = rejected.copy()
            out["_issues"] = out["_issues"].apply("; ".join)
            out.drop(columns=["_parsed_date", "_is_clean"], errors="ignore").to_csv(
                reject_path, index=False
            )

        summary = {
            "total_raw_rows":   total_raw,
            "duplicates_dropped": result["duplicates_dropped"],
            "clean_rows":       len(clean),
            "rejected_rows":    len(rejected),
            "reject_pct":       result["reject_pct"],
            "clean_path":       clean_path,
            "reject_path":      reject_path if len(rejected) else None,
            "date_range": {
                "min": str(clean["Date"].min()) if len(clean) else None,
                "max": str(clean["Date"].max()) if len(clean) else None,
            },
            "unique_exercises": int(clean["Exercise Name"].nunique()) if len(clean) else 0,
            "unique_workouts":  int(clean["Workout Name"].nunique()) if len(clean) else 0,
        }

        emit_metric("clean_weightlifting_data", "clean_and_validate", summary)
        log.info("Summary: %s", json.dumps(summary, indent=2))
        return summary

    @task
    def quality_gate(summary: Dict[str, Any]) -> Dict[str, Any]:
        if summary["reject_pct"] > MAX_REJECT_PCT:
            raise AirflowFailException(
                f"Rejection rate {summary['reject_pct']}% exceeds {MAX_REJECT_PCT}%"
            )
        if summary["clean_rows"] == 0:
            raise AirflowFailException("Zero clean rows — nothing to back up.")
        log.info(
            "Quality gate PASSED: %d clean rows, %.2f%% rejected",
            summary["clean_rows"],
            summary["reject_pct"],
        )
        return summary

    trigger_dvc = TriggerDagRunOperator(
        task_id="trigger_dvc_backup",
        trigger_dag_id="dvc_backup_to_gcp",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    files   = discover_files()
    summary = clean_and_validate(files)
    gate    = quality_gate(summary)
    gate >> trigger_dvc