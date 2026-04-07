"""
DAG: synthetic_anomaly_and_bias

Runs anomaly detection and bias analysis on the processed
synthetic data (daily_joined.parquet), sends a Slack report,
then triggers DVC backup.

Triggered by process_synthetic_data DAG.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from dag_monitoring import (
    monitored_dag_args,
    on_dag_failure_callback,
    on_sla_miss_callback,
    emit_metric,
    log,
)

AIRFLOW_HOME = "/opt/airflow"
PROCESSED_DIR = Path(f"{AIRFLOW_HOME}/data/processed")
REPORTS_DIR = Path(f"{AIRFLOW_HOME}/data/reports")
DAILY_JOINED = PROCESSED_DIR / "daily_joined.parquet"
ANOMALY_REPORT = REPORTS_DIR / "anomaly_report.json"
BIAS_REPORT = REPORTS_DIR / "bias_report.json"

AGE_BINS = [(0, 19, "<20"), (20, 29, "20-29"), (30, 39, "30-39"), (40, 200, "40+")]
IMBALANCE_THRESHOLD = 10


# Helpers


def ensure_dirs():
    for d in [PROCESSED_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def detect_anomalies(df):
    anomalies = []
    for field in ["sleep_hours", "attempts_count", "avg_accuracy"]:
        if field in df.columns:
            miss_pct = df[field].isna().mean() * 100
            if miss_pct > 20:
                anomalies.append(
                    {
                        "type": "high_missingness",
                        "field": field,
                        "missing_pct": round(miss_pct, 2),
                        "threshold": 20,
                        "severity": "warning",
                    }
                )
    if "sleep_hours" in df.columns:
        bad = df[
            (df["sleep_hours"].notna())
            & ((df["sleep_hours"] < 2) | (df["sleep_hours"] > 14))
        ]
        if len(bad) > 0:
            anomalies.append(
                {
                    "type": "out_of_range",
                    "field": "sleep_hours",
                    "count": len(bad),
                    "range": [2, 14],
                    "severity": "error",
                }
            )
    if "avg_accuracy" in df.columns:
        bad = df[
            (df["avg_accuracy"].notna())
            & ((df["avg_accuracy"] < 0) | (df["avg_accuracy"] > 1))
        ]
        if len(bad) > 0:
            anomalies.append(
                {
                    "type": "out_of_range",
                    "field": "avg_accuracy",
                    "count": len(bad),
                    "range": [0, 1],
                    "severity": "error",
                }
            )
    if "avg_time_per_question" in df.columns:
        bad = df[
            (df["avg_time_per_question"].notna())
            & ((df["avg_time_per_question"] <= 0) | (df["avg_time_per_question"] > 300))
        ]
        if len(bad) > 0:
            anomalies.append(
                {
                    "type": "out_of_range",
                    "field": "avg_time_per_question",
                    "count": len(bad),
                    "range": [0.01, 300],
                    "severity": "warning",
                }
            )
    for field in ["sleep_hours", "attempts_count", "int_score", "bmr"]:
        if field in df.columns:
            neg = df[(df[field].notna()) & (df[field] < 0)]
            if len(neg) > 0:
                anomalies.append(
                    {
                        "type": "negative_value",
                        "field": field,
                        "count": len(neg),
                        "severity": "error",
                    }
                )
    return anomalies


def assign_age_bucket(age):
    try:
        age = int(age)
        for lo, hi, label in AGE_BINS:
            if lo <= age <= hi:
                return label
    except (ValueError, TypeError):
        pass
    return "unknown"


def compute_slice_metrics(df, slice_col, slice_name):
    slices = []
    for val, group in df.groupby(slice_col):
        n = len(group)
        slices.append(
            {
                "slice_by": slice_name,
                "value": str(val),
                "count": n,
                "mean_int_score": (
                    round(group["int_score"].mean(), 2)
                    if "int_score" in group
                    else None
                ),
                "mean_avg_accuracy": (
                    round(group["avg_accuracy"].mean(), 4)
                    if "avg_accuracy" in group
                    else None
                ),
                "mean_sleep_hours": (
                    round(group["sleep_hours"].mean(), 2)
                    if "sleep_hours" in group
                    else None
                ),
                "flagged_imbalance": n < IMBALANCE_THRESHOLD,
            }
        )
    return slices


# DAG

with DAG(
    dag_id="synthetic_anomaly_and_bias",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=monitored_dag_args(retries=1, sla_minutes=20),
    on_failure_callback=on_dag_failure_callback,
    sla_miss_callback=on_sla_miss_callback,
    tags=["synthetic", "monitoring", "bias", "anomaly"],
) as dag:

    @task
    def run_anomaly_detection() -> Dict[str, Any]:
        ensure_dirs()

        df = pd.read_parquet(str(DAILY_JOINED))
        log.info("Loaded %d rows for anomaly detection", len(df))

        anomalies = detect_anomalies(df)

        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_rows": len(df),
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
            "status": "PASS" if len(anomalies) == 0 else "FAIL",
        }

        ANOMALY_REPORT.write_text(json.dumps(report, indent=2))
        log.info("Anomaly report -> %s (%d issues)", ANOMALY_REPORT, len(anomalies))

        if anomalies:
            for a in anomalies:
                log.warning(
                    "Anomaly: %s on '%s' (%s)", a["type"], a["field"], a["severity"]
                )

            # Slack alert for anomalies
            webhook = os.environ.get("SLACK_WEBHOOK_URL") or Variable.get(
                "ALERT_WEBHOOK_URL", default_var=""
            )
            if webhook:
                try:
                    msg = (
                        f"⚠️ Synthetic Pipeline: {len(anomalies)} anomalies detected!\n"
                    )
                    for a in anomalies[:5]:
                        msg += f"  • {a['type']}: {a['field']} ({a['severity']})\n"
                    payload = json.dumps({"text": msg}).encode("utf-8")
                    req = urllib.request.Request(
                        webhook,
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    urllib.request.urlopen(req, timeout=10)
                    log.info("Anomaly Slack alert sent")
                except Exception as e:
                    log.warning("Failed to send anomaly alert: %s", e)

        emit_metric(
            "synthetic_anomaly_and_bias",
            "anomaly_detection",
            {
                "anomaly_count": len(anomalies),
                "status": report["status"],
            },
        )

        return report

    @task
    def run_bias_analysis() -> Dict[str, Any]:
        ensure_dirs()

        df = pd.read_parquet(str(DAILY_JOINED))
        log.info("Loaded %d rows for bias analysis", len(df))

        df["age_bucket"] = df.get("age", pd.Series(dtype=float)).apply(
            assign_age_bucket
        )

        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_rows": len(df),
            "slices": {},
        }

        if "sex" in df.columns:
            report["slices"]["sex"] = compute_slice_metrics(df, "sex", "sex")

        report["slices"]["age_bucket"] = compute_slice_metrics(
            df, "age_bucket", "age_bucket"
        )

        imbalanced = []
        for category, slices in report["slices"].items():
            for s in slices:
                if s.get("flagged_imbalance"):
                    imbalanced.append(f"{category}={s['value']} (n={s['count']})")

        report["imbalanced_slices"] = imbalanced
        report["mitigation_notes"] = (
            "If slices are imbalanced, consider: "
            "1) collecting more data from underrepresented groups, "
            "2) applying sample weighting during model training, "
            "3) using stratified train/test splits, "
            "4) monitoring per-slice model performance separately."
        )

        BIAS_REPORT.write_text(json.dumps(report, indent=2))
        log.info(
            "Bias report -> %s (%d slices)",
            BIAS_REPORT,
            sum(len(v) for v in report["slices"].values()),
        )

        if imbalanced:
            log.warning("Data imbalance detected: %s", ", ".join(imbalanced))

        emit_metric(
            "synthetic_anomaly_and_bias",
            "bias_analysis",
            {
                "total_slices": sum(len(v) for v in report["slices"].values()),
                "imbalanced_count": len(imbalanced),
            },
        )

        return report

    @task
    def send_slack_summary(
        anomaly_result: Dict[str, Any],
        bias_result: Dict[str, Any],
    ):
        url = Variable.get("ALERT_WEBHOOK_URL", default_var="")
        if not url:
            log.warning("No ALERT_WEBHOOK_URL — skipping Slack")
            return

        anomaly_status = anomaly_result.get("status", "UNKNOWN")
        anomaly_count = anomaly_result.get("anomaly_count", 0)
        imbalanced = bias_result.get("imbalanced_slices", [])
        has_issues = anomaly_status != "PASS" or len(imbalanced) > 0
        status_emoji = ":red_circle:" if has_issues else ":large_green_circle:"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Synthetic Data — Anomaly & Bias Report",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Status:* {status_emoji} {'ISSUES FOUND' if has_issues else 'ALL CLEAR'}\n"
                        f"*Generated:* {anomaly_result.get('timestamp', 'N/A')}\n"
                        f"*Rows analyzed:* {anomaly_result.get('total_rows', 0):,}"
                    ),
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Anomaly Detection*\n"
                        f"Status: {'✅ PASS' if anomaly_status == 'PASS' else '🚨 FAIL'} | "
                        f"Issues: {anomaly_count}"
                    ),
                },
            },
        ]

        if anomaly_count > 0:
            lines = []
            for a in anomaly_result.get("anomalies", [])[:8]:
                sev = "🔴" if a["severity"] == "error" else "🟡"
                lines.append(f"{sev} `{a['type']}` on `{a['field']}`")
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "\n".join(lines)},
                }
            )

        blocks.append({"type": "divider"})

        # Bias section
        bias_slices = bias_result.get("slices", {})
        bias_lines = ["*Bias Analysis*"]
        for category, slices in bias_slices.items():
            table = f"`{category}`: "
            table += " | ".join(f"{s['value']}: n={s['count']}" for s in slices)
            bias_lines.append(table)

        if imbalanced:
            bias_lines.append(
                f"\n:warning: *Imbalanced slices:* {', '.join(imbalanced)}"
            )

        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "\n".join(bias_lines)},
            }
        )

        # Per-slice metric tables
        for category, slices in bias_slices.items():
            has_metrics = any(s.get("mean_int_score") is not None for s in slices)
            if has_metrics:
                table = f"```{'Slice':<12} | {'INT':>6} | {'Acc':>6} | {'Sleep':>6} | {'N':>5}\n"
                table += "-" * 50 + "\n"
                for s in slices:
                    table += (
                        f"{str(s['value']):<12} | "
                        f"{s.get('mean_int_score', 'N/A'):>6} | "
                        f"{s.get('mean_avg_accuracy', 'N/A'):>6} | "
                        f"{s.get('mean_sleep_hours', 'N/A'):>6} | "
                        f"{s['count']:>5}\n"
                    )
                table += "```"
                blocks.append(
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*{category} breakdown:*\n{table}",
                        },
                    }
                )

        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "_Automated report from `synthetic_anomaly_and_bias` DAG_",
                    }
                ],
            }
        )

        payload = json.dumps({"blocks": blocks}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                log.info("Slack summary sent (status=%s)", resp.status)
        except Exception as e:
            log.error("Slack send failed: %s", e)

    trigger_dvc = TriggerDagRunOperator(
        task_id="trigger_dvc_backup",
        trigger_dag_id="dvc_backup_to_gcp",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    # Task flow
    anomalies = run_anomaly_detection()
    bias = run_bias_analysis()
    anomalies >> bias

    slack = send_slack_summary(anomalies, bias)
    bias >> slack
    slack >> trigger_dvc
