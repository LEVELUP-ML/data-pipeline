"""
Shared monitoring utilities for all DAGs.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from airflow.models import Variable

#  Logger 
log = logging.getLogger("airflow.task")


#  Slack / Webhook alerting
def _send_webhook(payload: dict):
    """
    Post a JSON payload to a webhook URL (Slack, Teams, Discord, etc.).
    Set Airflow Variable ALERT_WEBHOOK_URL to enable.
    """
    try:
        url = Variable.get("ALERT_WEBHOOK_URL", default_var="")
    except Exception:
        url = ""
    if not url:
        log.debug("No ALERT_WEBHOOK_URL set — skipping webhook.")
        return

    import urllib.request

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            log.info("Webhook sent (%s)", resp.status)
    except Exception as e:
        log.warning("Webhook failed: %s", e)


#  Callbacks


def on_failure_callback(context: Dict[str, Any]):
    """Called when any task fails."""
    ti = context["task_instance"]
    exc = context.get("exception", "")

    log.error(
        "TASK FAILED | dag=%s | task=%s | run=%s | try=%d | error=%s",
        ti.dag_id,
        ti.task_id,
        ti.run_id,
        ti.try_number,
        exc,
    )

    _send_webhook(
        {
            "text": (
                f":red_circle: *Task Failed*\n"
                f"• DAG: `{ti.dag_id}`\n"
                f"• Task: `{ti.task_id}`\n"
                f"• Run: `{ti.run_id}`\n"
                f"• Attempt: {ti.try_number}\n"
                f"• Error: ```{str(exc)[:500]}```"
            )
        }
    )


def on_success_callback(context: Dict[str, Any]):
    """Called when a task succeeds."""
    ti = context["task_instance"]

    log.info(
        "TASK OK | dag=%s | task=%s | duration=%.1fs",
        ti.dag_id,
        ti.task_id,
        ti.duration or 0,
    )


def on_retry_callback(context: Dict[str, Any]):
    """Called when a task retries."""
    ti = context["task_instance"]
    exc = context.get("exception", "")

    log.warning(
        "TASK RETRY | dag=%s | task=%s | try=%d | error=%s",
        ti.dag_id,
        ti.task_id,
        ti.try_number,
        exc,
    )


def on_dag_failure_callback(context: Dict[str, Any]):
    """Called when the entire DAG run fails."""
    dag_id = (
        context.get("dag", context.get("dag_run", "")).dag_id
        if "dag" in context
        else "unknown"
    )
    run_id = context.get("run_id", "unknown")

    log.error("DAG FAILED | dag=%s | run=%s", dag_id, run_id)

    _send_webhook({"text": f":fire: *DAG Failed*: `{dag_id}` (run: `{run_id}`)"})


def on_sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis):
    """Called when any task misses its SLA."""
    tasks = ", ".join(str(t) for t in task_list)
    log.warning("SLA MISS | dag=%s | tasks=%s", dag.dag_id, tasks)

    _send_webhook(
        {
            "text": (
                f":warning: *SLA Miss*\n"
                f"• DAG: `{dag.dag_id}`\n"
                f"• Tasks: `{tasks}`"
            )
        }
    )


#  Metrics helper

METRICS_DIR = "/opt/airflow/logs/dag_metrics"


def emit_metric(
    dag_id: str,
    task_id: str,
    metrics: Dict[str, Any],
    run_id: Optional[str] = None,
):
    """
    Write a structured JSON metrics line to a file for external scrapers
    (Prometheus node_exporter textfile, Datadog, Grafana Loki, etc.).

    Call this from inside any @task:
        emit_metric("clean_weightlifting_data", "clean_and_validate", {
            "raw_rows": 9932,
            "clean_rows": 9800,
            "rejected_rows": 132,
            "reject_pct": 1.33,
        })
    """
    os.makedirs(METRICS_DIR, exist_ok=True)

    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "dag_id": dag_id,
        "task_id": task_id,
        "run_id": run_id,
        **metrics,
    }

    path = os.path.join(METRICS_DIR, f"{dag_id}.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    log.info("METRIC | %s.%s | %s", dag_id, task_id, json.dumps(metrics, default=str))


#  Pre-built default_args 


def monitored_dag_args(
    retries: int = 2,
    retry_delay_min: int = 5,
    email: Optional[list] = None,
    sla_minutes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Returns a default_args dict with all callbacks wired up.

    Usage:
        with DAG(
            dag_id="my_dag",
            default_args=monitored_dag_args(retries=3, sla_minutes=30),
            on_failure_callback=on_dag_failure_callback,
            sla_miss_callback=on_sla_miss_callback,
            ...
        ) as dag:
    """
    args: Dict[str, Any] = {
        "retries": retries,
        "retry_delay": timedelta(minutes=retry_delay_min),
        "on_failure_callback": on_failure_callback,
        "on_success_callback": on_success_callback,
        "on_retry_callback": on_retry_callback,
    }
    if email:
        args["email"] = email
        args["email_on_failure"] = True
        args["email_on_retry"] = True
    if sla_minutes:
        args["sla"] = timedelta(minutes=sla_minutes)
    return args
