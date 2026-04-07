"""
DAG: download_synthetic_from_firestore

Pulls seeded synthetic data (profiles, sleep_logs, quiz_attempts)
from Firestore and writes raw JSON files for the processing pipeline.

Output:
  - data/raw/profiles_raw.json
  - data/raw/sleep_logs_raw.json
  - data/raw/quiz_attempts_raw.json

Triggers process_synthetic_data on success.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

import firebase_admin
from firebase_admin import credentials, firestore
from dag_monitoring import (
    monitored_dag_args,
    on_dag_failure_callback,
    on_sla_miss_callback,
    emit_metric,
    log,
)

AIRFLOW_HOME = "/opt/airflow"
RAW_DIR = Path(f"{AIRFLOW_HOME}/data/raw")


def get_firestore_client():
    if not firebase_admin._apps:
        sa_path = Variable.get("FIREBASE_SERVICE_ACCOUNT_PATH")
        if not os.path.exists(sa_path):
            raise AirflowFailException(f"Service account JSON not found: {sa_path}")
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()


def doc_to_dict(doc) -> dict:
    """Convert Firestore doc to JSON-safe dict."""
    d = doc.to_dict() or {}
    for k, v in d.items():
        if hasattr(v, "isoformat"):
            d[k] = v.isoformat()
    return d


with DAG(
    dag_id="download_synthetic_from_firestore",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=monitored_dag_args(retries=2, sla_minutes=15),
    on_failure_callback=on_dag_failure_callback,
    sla_miss_callback=on_sla_miss_callback,
    tags=["download", "firestore", "synthetic", "ingest"],
) as dag:

    @task
    def download_profiles() -> Dict[str, Any]:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        db = get_firestore_client()

        log.info("Fetching users from Firestore")
        users = db.collection("users").stream()

        profiles = []
        user_ids = []
        for doc in users:
            d = doc_to_dict(doc)
            profile = d.get("profile", {})
            profiles.append(
                {
                    "user_id": doc.id,
                    "age": profile.get("age"),
                    "sex": profile.get("sex"),
                    "height": profile.get("height_cm"),
                    "weight": profile.get("weight_kg"),
                }
            )
            user_ids.append(doc.id)

        out = RAW_DIR / "profiles_raw.json"
        out.write_text(json.dumps(profiles, indent=2))

        log.info("Wrote %d profiles -> %s", len(profiles), out)

        emit_metric(
            "download_synthetic_from_firestore",
            "download_profiles",
            {
                "profile_count": len(profiles),
            },
        )

        return {"profile_count": len(profiles), "user_ids": user_ids}

    @task
    def download_sleep_logs(profiles_result: Dict[str, Any]) -> Dict[str, Any]:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        db = get_firestore_client()
        user_ids = profiles_result["user_ids"]

        log.info("Fetching sleep_logs for %d users", len(user_ids))

        all_logs = []
        for uid in user_ids:
            docs = (
                db.collection("users").document(uid).collection("sleep_logs").stream()
            )
            for doc in docs:
                d = doc_to_dict(doc)
                d["user_id"] = uid
                all_logs.append(d)

        out = RAW_DIR / "sleep_logs_raw.json"
        out.write_text(json.dumps(all_logs, indent=2, default=str))

        log.info("Wrote %d sleep logs -> %s", len(all_logs), out)

        emit_metric(
            "download_synthetic_from_firestore",
            "download_sleep_logs",
            {
                "sleep_log_count": len(all_logs),
            },
        )

        return {"sleep_log_count": len(all_logs)}

    @task
    def download_quiz_attempts(profiles_result: Dict[str, Any]) -> Dict[str, Any]:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        db = get_firestore_client()
        user_ids = profiles_result["user_ids"]

        log.info("Fetching quiz_attempts for %d users", len(user_ids))

        all_attempts = []
        for uid in user_ids:
            docs = (
                db.collection("users")
                .document(uid)
                .collection("quiz_attempts")
                .stream()
            )
            for doc in docs:
                d = doc_to_dict(doc)
                d["user_id"] = uid
                # Normalize field name for downstream compatibility
                if "timestamp" in d and hasattr(d["timestamp"], "isoformat"):
                    d["timestamp"] = d["timestamp"].isoformat()
                if (
                    "avg_time_per_question_seconds" not in d
                    and "avg_time_per_question" in d
                ):
                    d["avg_time_per_question_seconds"] = d["avg_time_per_question"]
                all_attempts.append(d)

        out = RAW_DIR / "quiz_attempts_raw.json"
        out.write_text(json.dumps(all_attempts, indent=2, default=str))

        log.info("Wrote %d quiz attempts -> %s", len(all_attempts), out)

        emit_metric(
            "download_synthetic_from_firestore",
            "download_quiz_attempts",
            {
                "quiz_attempt_count": len(all_attempts),
            },
        )

        return {"quiz_attempt_count": len(all_attempts)}

    @task
    def log_summary(
        profiles_result: Dict[str, Any],
        sleep_result: Dict[str, Any],
        quiz_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        summary = {
            "profiles": profiles_result["profile_count"],
            "sleep_logs": sleep_result["sleep_log_count"],
            "quiz_attempts": quiz_result["quiz_attempt_count"],
        }
        log.info("Download complete: %s", json.dumps(summary))

        if summary["profiles"] == 0:
            raise AirflowFailException(
                "No profiles found in Firestore. Run the seeding script first."
            )

        return summary

    trigger_processing = TriggerDagRunOperator(
        task_id="trigger_processing",
        trigger_dag_id="process_synthetic_data",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    # Task flow
    profiles = download_profiles()

    sleep = download_sleep_logs(profiles)
    quiz = download_quiz_attempts(profiles)

    summary = log_summary(profiles, sleep, quiz)
    summary >> trigger_processing