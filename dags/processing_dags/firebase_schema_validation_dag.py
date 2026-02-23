from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import pendulum
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import get_current_context

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import FieldFilter
from datetime import datetime


VALID_METRICS = {"strength", "stamina", "speed", "flexibility", "intelligence"}
VALID_SOURCES = {"wearable", "app_tasks", "workout_log", "manual"}
VALID_EVENT_TYPES = {"model_update"}
VALID_TOPICS = {"Biology", "Chemistry", "Physics", "Math", "History", "CS", "English"}
VALID_SEXES = {"Male", "Female"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}$")

EXPECTED_COMPONENTS = {
    "strength": {"bench_1rm", "squat_1rm", "deadlift_1rm"},
    "stamina": {"vo2max", "resting_hr"},
    "speed": {"sprint_100m_sec"},
    "flexibility": {"sit_and_reach_cm", "shoulder_mobility"},
    "intelligence": {"memory_task", "reaction_time_ms"},
}

MAX_VALIDATION_DOCS = 600_000
SAMPLE_LIMIT = 50_000


# Helpers
def get_firestore_client() -> firestore.Client:
    if not firebase_admin._apps:
        sa_path = Variable.get("FIREBASE_SERVICE_ACCOUNT_PATH")
        if not os.path.exists(sa_path):
            raise AirflowFailException(f"Service-account JSON not found: {sa_path}")
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()


@dataclass
class ValidationReport:
    collection: str
    docs_checked: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, doc_id: str, msg: str):
        if len(self.errors) < 200:  # cap stored messages
            self.errors.append(f"[{doc_id}] {msg}")

    def add_warning(self, doc_id: str, msg: str):
        if len(self.warnings) < 200:
            self.warnings.append(f"[{doc_id}] {msg}")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "collection": self.collection,
            "docs_checked": self.docs_checked,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "passed": self.passed,
            "sample_errors": self.errors[:25],
            "sample_warnings": self.warnings[:25],
        }


def _check_type(
    d: dict, key: str, expected, doc_id: str, rpt: ValidationReport, nullable=False
):
    """Validate that d[key] exists and has the right type."""
    val = d.get(key)
    if val is None:
        if nullable:
            return val
        rpt.add_error(doc_id, f"missing or null field '{key}'")
        return None
    if not isinstance(val, expected):
        rpt.add_error(
            doc_id, f"'{key}' expected {expected.__name__}, got {type(val).__name__}"
        )
        return None
    return val


def _check_range(val, lo, hi, key: str, doc_id: str, rpt: ValidationReport):
    if val is None:
        return
    if not (lo <= val <= hi):
        rpt.add_error(doc_id, f"'{key}' value {val} outside [{lo}, {hi}]")


# Per-collection validators


def validate_metric_event(d: dict, doc_id: str, rpt: ValidationReport, target_day: str):
    """Validate a single metric_events document."""
    rpt.docs_checked += 1

    # day
    day = _check_type(d, "day", str, doc_id, rpt)
    if day:
        if not DATE_RE.match(day):
            rpt.add_error(doc_id, f"'day' bad format: {day}")
        if day != target_day:
            rpt.add_warning(doc_id, f"'day' {day} != target {target_day}")

    # metric
    metric = _check_type(d, "metric", str, doc_id, rpt)
    if metric and metric not in VALID_METRICS:
        rpt.add_error(doc_id, f"unknown metric '{metric}'")

    # type
    etype = _check_type(d, "type", str, doc_id, rpt)
    if etype and etype not in VALID_EVENT_TYPES:
        rpt.add_warning(doc_id, f"unexpected event type '{etype}'")

    # source
    src = _check_type(d, "source", str, doc_id, rpt)
    if src and src not in VALID_SOURCES:
        rpt.add_warning(doc_id, f"unexpected source '{src}'")

    # score
    score = d.get("score")
    if score is not None:
        if not isinstance(score, (int, float)):
            rpt.add_error(doc_id, f"'score' not numeric: {type(score).__name__}")
        else:
            _check_range(score, 0.0, 100.0, "score", doc_id, rpt)
    else:
        rpt.add_error(doc_id, "missing 'score'")

    # delta
    delta = d.get("delta")
    if delta is not None and not isinstance(delta, (int, float)):
        rpt.add_error(doc_id, f"'delta' not numeric: {type(delta).__name__}")

    # confidence
    conf = d.get("confidence")
    if conf is not None:
        if not isinstance(conf, (int, float)):
            rpt.add_error(doc_id, f"'confidence' not numeric")
        else:
            _check_range(conf, 0.0, 1.0, "confidence", doc_id, rpt)

    # payload / components
    payload = d.get("payload")
    if payload is not None and metric and metric in EXPECTED_COMPONENTS:
        if not isinstance(payload, dict):
            rpt.add_error(doc_id, f"'payload' not a dict")
        else:
            missing = EXPECTED_COMPONENTS[metric] - set(payload.keys())
            if missing:
                rpt.add_error(doc_id, f"payload missing keys for {metric}: {missing}")


def validate_sleep_log(d: dict, doc_id: str, rpt: ValidationReport):
    rpt.docs_checked += 1

    _check_type(d, "user_id", str, doc_id, rpt)

    date = _check_type(d, "date", str, doc_id, rpt)
    if date and not DATE_RE.match(date):
        rpt.add_error(doc_id, f"'date' bad format: {date}")

    bed = _check_type(d, "bedTime", str, doc_id, rpt)
    if bed and not TIME_RE.match(bed):
        rpt.add_error(doc_id, f"'bedTime' bad format: {bed}")

    wake = _check_type(d, "wakeTime", str, doc_id, rpt)
    if wake and not TIME_RE.match(wake):
        rpt.add_error(doc_id, f"'wakeTime' bad format: {wake}")

    hrs = d.get("sleepHours")
    if hrs is not None:
        if not isinstance(hrs, (int, float)):
            rpt.add_error(doc_id, "'sleepHours' not numeric")
        else:
            _check_range(hrs, 0.0, 24.0, "sleepHours", doc_id, rpt)
    else:
        rpt.add_error(doc_id, "missing 'sleepHours'")

    quality = d.get("quality")
    if quality is not None:
        if not isinstance(quality, int):
            rpt.add_error(doc_id, f"'quality' not int: {type(quality).__name__}")
        else:
            _check_range(quality, 1, 5, "quality", doc_id, rpt)
    # quality can be null (seeded anomaly), so just warn
    if quality is None:
        rpt.add_warning(doc_id, "'quality' is null")


def validate_quiz_attempt(d: dict, doc_id: str, rpt: ValidationReport):
    rpt.docs_checked += 1

    _check_type(d, "user_id", str, doc_id, rpt)
    _check_type(d, "quiz_id", str, doc_id, rpt)

    topic = _check_type(d, "topic", str, doc_id, rpt)
    if topic and topic not in VALID_TOPICS:
        rpt.add_error(doc_id, f"unknown topic '{topic}'")

    n_q = d.get("num_questions")
    if n_q is not None:
        if not isinstance(n_q, int):
            rpt.add_error(doc_id, "'num_questions' not int")
        else:
            _check_range(n_q, 1, 200, "num_questions", doc_id, rpt)

    n_c = d.get("num_correct")
    if n_c is not None:
        if not isinstance(n_c, int):
            rpt.add_error(doc_id, "'num_correct' not int")
        elif n_c < 0:
            rpt.add_error(doc_id, f"'num_correct' is negative: {n_c}")
        elif n_q is not None and isinstance(n_q, int) and n_c > n_q:
            rpt.add_error(doc_id, f"'num_correct' ({n_c}) > 'num_questions' ({n_q})")

    time_s = d.get("total_time_seconds")
    if time_s is not None:
        if not isinstance(time_s, (int, float)):
            rpt.add_error(doc_id, "'total_time_seconds' not numeric")
        else:
            _check_range(time_s, 0, 7200, "total_time_seconds", doc_id, rpt)

    avg = d.get("avg_time_per_question_seconds")
    if avg is not None and isinstance(avg, (int, float)):
        if avg > 300:
            rpt.add_warning(
                doc_id, f"'avg_time_per_question_seconds' suspiciously high: {avg}"
            )

    diff = d.get("difficulty")
    if diff is not None:
        if not isinstance(diff, int):
            rpt.add_error(doc_id, "'difficulty' not int")
        else:
            _check_range(diff, 1, 5, "difficulty", doc_id, rpt)

    pct = d.get("percent")
    if pct is not None:
        if not isinstance(pct, int):
            rpt.add_error(doc_id, "'percent' not int")
        else:
            _check_range(pct, 0, 100, "percent", doc_id, rpt)


# DAG

with DAG(
    dag_id="firestore_schema_validation",
    start_date=pendulum.datetime(2026, 2, 1, tz="America/New_York"),
    schedule="0 2 * * *",  # runs 1 h before the export DAG (03:00)
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1},
    tags=["firestore", "validation", "quality"],
) as dag:

    @task
    def validate_metric_events() -> Dict[str, Any]:
        ctx = get_current_context()
        ny = pendulum.timezone("America/New_York")
        target_day = (
            ctx["logical_date"].in_timezone(ny).subtract(days=1).format("YYYY-MM-DD")
        )

        db = get_firestore_client()
        rpt = ValidationReport(collection="metric_events")

        q = db.collection_group("metric_events").where(
            filter=FieldFilter("day", "==", target_day)
        )
        for doc in q.stream():
            d = doc.to_dict() or {}
            validate_metric_event(d, doc.id, rpt, target_day)
            if rpt.docs_checked >= MAX_VALIDATION_DOCS:
                rpt.add_warning("GLOBAL", "hit MAX_VALIDATION_DOCS cap")
                break

        return rpt.as_dict()

    @task
    def validate_sleep_logs() -> Dict[str, Any]:
        ctx = get_current_context()
        ny = pendulum.timezone("America/New_York")
        target_day = (
            ctx["logical_date"].in_timezone(ny).subtract(days=1).format("YYYY-MM-DD")
        )

        db = get_firestore_client()
        rpt = ValidationReport(collection="sleep_logs")

        # No collection-group index assumed; iterate known users
        users = db.collection("users").select([]).limit(SAMPLE_LIMIT).stream()
        for u in users:
            doc_ref = (
                db.collection("users")
                .document(u.id)
                .collection("sleep_logs")
                .document(target_day)
            )
            snap = doc_ref.get()
            if snap.exists:
                validate_sleep_log(snap.to_dict() or {}, snap.id, rpt)

        return rpt.as_dict()

    @task
    def validate_quiz_attempts() -> Dict[str, Any]:
        ctx = get_current_context()
        ny = pendulum.timezone("America/New_York")
        target_day = (
            ctx["logical_date"].in_timezone(ny).subtract(days=1).format("YYYY-MM-DD")
        )

        db = get_firestore_client()
        rpt = ValidationReport(collection="quiz_attempts")

        users = db.collection("users").select([]).limit(SAMPLE_LIMIT).stream()
        for u in users:
            col = db.collection("users").document(u.id).collection("quiz_attempts")
            # Filter to docs whose timestamp falls on target_day
            day_start = pendulum.parse(target_day, tz="UTC")
            day_end = day_start.add(days=1)
            q = col.where(filter=FieldFilter("timestamp", ">=", day_start)).where(
                filter=FieldFilter("timestamp", "<", day_end)
            )
            for snap in q.stream():
                validate_quiz_attempt(snap.to_dict() or {}, snap.id, rpt)

        return rpt.as_dict()

    @task
    def gate_export(
        metric_result: Dict[str, Any],
        sleep_result: Dict[str, Any],
        quiz_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Aggregate reports. Fail the task (blocking downstream trigger)
        if any collection has errors above the allowed threshold.
        """
        error_threshold_pct = float(
            Variable.get("VALIDATION_ERROR_THRESHOLD_PCT", default_var="5.0")
        )

        summary = {}
        any_blocked = False

        for rpt in (metric_result, sleep_result, quiz_result):
            col = rpt["collection"]
            checked = rpt["docs_checked"]
            errs = rpt["error_count"]
            pct = (errs / checked * 100) if checked else 0.0
            ok = pct <= error_threshold_pct

            summary[col] = {
                "docs_checked": checked,
                "errors": errs,
                "warnings": rpt["warning_count"],
                "error_pct": round(pct, 2),
                "passed": ok,
            }

            if not ok:
                any_blocked = True

        summary["all_passed"] = not any_blocked

        if any_blocked:
            raise AirflowFailException(
                f"Schema validation blocked export. Summary: {summary}"
            )

        return summary

    trigger_export = TriggerDagRunOperator(
        task_id="trigger_export_dag",
        trigger_dag_id="firestore_metric_events_to_gcs",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    me = validate_metric_events()
    sl = validate_sleep_logs()
    qa = validate_quiz_attempts()
    g = gate_export(me, sl, qa)
    g >> trigger_export
