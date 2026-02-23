"""
DAG: clean_weightlifting_data

Reads raw CSVs from the Kaggle ingest dirs, applies cleaning /
validation rules, writes cleaned Parquet + a rejection log, then
triggers the DVC backup DAG to push everything to GCS.

Trigger manually or wire it downstream of the kaggle_download_* DAGs.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# Paths

AIRFLOW_HOME = "/opt/airflow"
RAW_DIRS = [
    f"{AIRFLOW_HOME}/data/raw/weightlifting",
    f"{AIRFLOW_HOME}/data/raw/strength",
]
CLEAN_DIR = f"{AIRFLOW_HOME}/data/processed/weightlifting_cleaned"
REJECT_DIR = f"{AIRFLOW_HOME}/data/processed/weightlifting_rejected"

# Cleaning constants

REQUIRED_COLS = {
    "Date",
    "Workout Name",
    "Exercise Name",
    "Set Order",
    "Weight",
    "Reps",
}

# Plausible physical limits
MAX_WEIGHT_KG = 700.0  # beyond equipped all-time records
MAX_REPS = 200
MAX_SET_ORDER = 100
MAX_SECONDS = 36_000  # 10 hours
MAX_DISTANCE_M = 100_000  # 100 km


# Helpers


def _find_csvs() -> List[str]:
    """Glob all CSVs from the known raw directories."""
    paths = []
    for d in RAW_DIRS:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(".csv"):
                paths.append(os.path.join(d, f))
    return sorted(set(paths))


def _tag_row_issues(row: pd.Series) -> List[str]:
    """Return a list of issue strings for a single row (empty = clean)."""
    issues: List[str] = []

    #  Date 
    if pd.isna(row.get("_parsed_date")):
        issues.append("invalid_or_missing_date")

    #  Exercise Name 
    if pd.isna(row.get("Exercise Name")) or str(row["Exercise Name"]).strip() == "":
        issues.append("missing_exercise_name")

    #  Workout Name 
    if pd.isna(row.get("Workout Name")) or str(row["Workout Name"]).strip() == "":
        issues.append("missing_workout_name")

    #  Set Order 
    so = row.get("Set Order")
    if pd.isna(so):
        issues.append("missing_set_order")
    elif so < 1 or so > MAX_SET_ORDER:
        issues.append(f"set_order_out_of_range({so})")

    #  Weight 
    w = row.get("Weight")
    if pd.notna(w):
        if w < 0:
            issues.append(f"negative_weight({w})")
        elif w > MAX_WEIGHT_KG:
            issues.append(f"weight_exceeds_max({w})")

    #  Reps 
    r = row.get("Reps")
    if pd.notna(r):
        if r < 0:
            issues.append(f"negative_reps({r})")
        elif r > MAX_REPS:
            issues.append(f"reps_exceeds_max({r})")

    #  Seconds 
    s = row.get("Seconds")
    if pd.notna(s):
        if s < 0:
            issues.append(f"negative_seconds({s})")
        elif s > MAX_SECONDS:
            issues.append(f"seconds_exceeds_max({s})")

    #  Distance 
    d = row.get("Distance")
    if pd.notna(d):
        if d < 0:
            issues.append(f"negative_distance({d})")
        elif d > MAX_DISTANCE_M:
            issues.append(f"distance_exceeds_max({d})")

    return issues


# DAG

with DAG(
    dag_id="clean_weightlifting_data",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["cleaning", "weightlifting", "data-quality"],
) as dag:

    @task
    def discover_files() -> List[str]:
        paths = _find_csvs()
        if not paths:
            raise AirflowFailException(
                f"No CSVs found in {RAW_DIRS}. Run a kaggle_download DAG first."
            )
        print(f"Discovered {len(paths)} CSV(s): {paths}")
        return paths

    @task
    def clean_and_validate(csv_paths: List[str]) -> Dict[str, Any]:
        frames: List[pd.DataFrame] = []

        for p in csv_paths:
            print(f"Reading {p}")
            df = pd.read_csv(p, dtype=str)  # read everything as str first
            df["_source_file"] = os.path.basename(p)
            frames.append(df)

        raw = pd.concat(frames, ignore_index=True)
        total_raw = len(raw)
        print(f"Total raw rows: {total_raw}")

        #  Schema check
        missing_cols = REQUIRED_COLS - set(raw.columns)
        if missing_cols:
            raise AirflowFailException(
                f"CSV(s) missing required columns: {missing_cols}"
            )

        #  Strip whitespace from all string columns
        str_cols = raw.select_dtypes(include="object").columns
        for c in str_cols:
            raw[c] = raw[c].str.strip()

        #  Type coercion 
        raw["_parsed_date"] = pd.to_datetime(
            raw["Date"], format="mixed", dayfirst=False, errors="coerce"
        )
        for col in ("Weight", "Distance"):
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        for col in ("Set Order", "Reps", "Seconds"):
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
            # convert to nullable int so NaN is preserved
            raw[col] = raw[col].astype("Int64")

        #  Drop full duplicates 
        before_dedup = len(raw)
        raw = raw.drop_duplicates(
            subset=[
                "Date",
                "Workout Name",
                "Exercise Name",
                "Set Order",
                "Weight",
                "Reps",
            ],
        )
        dupes_dropped = before_dedup - len(raw)

        #  Row-level validation 
        raw["_issues"] = raw.apply(_tag_row_issues, axis=1)
        raw["_is_clean"] = raw["_issues"].apply(lambda x: len(x) == 0)

        clean = raw[raw["_is_clean"]].copy()
        rejected = raw[~raw["_is_clean"]].copy()

        #  Normalise clean output 
        clean["Date"] = clean["_parsed_date"]
        clean = clean.drop(columns=["_parsed_date", "_issues", "_is_clean"])

        # Fill missing optional numeric fields with explicit NaN
        for col in ("Weight", "Reps", "Distance", "Seconds"):
            if col in clean.columns:
                clean[col] = clean[col]

        # Sort for deterministic output
        clean = clean.sort_values(
            ["Date", "Workout Name", "Exercise Name", "Set Order"]
        ).reset_index(drop=True)

        #  Write outputs 
        os.makedirs(CLEAN_DIR, exist_ok=True)
        os.makedirs(REJECT_DIR, exist_ok=True)

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

        clean_path = os.path.join(CLEAN_DIR, f"workouts_clean_{ts}.parquet")
        clean.to_parquet(clean_path, index=False)
        print(f"Wrote {len(clean)} clean rows → {clean_path}")

        reject_path = os.path.join(REJECT_DIR, f"workouts_rejected_{ts}.csv")
        if len(rejected) > 0:
            rejected["_issues"] = rejected["_issues"].apply(lambda lst: "; ".join(lst))
            rejected = rejected.drop(columns=["_parsed_date", "_is_clean"])
            rejected.to_csv(reject_path, index=False)
            print(f"Wrote {len(rejected)} rejected rows → {reject_path}")

        #  Summary 
        summary = {
            "total_raw_rows": total_raw,
            "duplicates_dropped": dupes_dropped,
            "clean_rows": len(clean),
            "rejected_rows": len(rejected),
            "reject_pct": (
                round(len(rejected) / total_raw * 100, 2) if total_raw else 0.0
            ),
            "clean_path": clean_path,
            "reject_path": reject_path if len(rejected) > 0 else None,
            "date_range": {
                "min": str(clean["Date"].min()) if len(clean) else None,
                "max": str(clean["Date"].max()) if len(clean) else None,
            },
            "unique_exercises": (
                int(clean["Exercise Name"].nunique()) if len(clean) else 0
            ),
            "unique_workouts": (
                int(clean["Workout Name"].nunique()) if len(clean) else 0
            ),
        }
        print(json.dumps(summary, indent=2))
        return summary

    @task
    def quality_gate(summary: Dict[str, Any]) -> Dict[str, Any]:
        """Block the pipeline if rejection rate is too high."""
        max_reject_pct = 15.0
        if summary["reject_pct"] > max_reject_pct:
            raise AirflowFailException(
                f"Rejection rate {summary['reject_pct']}% exceeds "
                f"threshold {max_reject_pct}%. Review rejected rows at "
                f"{summary['reject_path']}"
            )
        if summary["clean_rows"] == 0:
            raise AirflowFailException("Zero clean rows — nothing to back up.")
        print(
            f"Quality gate passed: {summary['clean_rows']} clean rows, "
            f"{summary['reject_pct']}% rejected."
        )
        return summary

    trigger_dvc = TriggerDagRunOperator(
        task_id="trigger_dvc_backup",
        trigger_dag_id="dvc_backup_to_gcp",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    files = discover_files()
    summary = clean_and_validate(files)
    gate = quality_gate(summary)
    gate >> trigger_dvc
