"""
DAG: daily_bias_monitoring

Runs once a day at 06:00 ET. Scans all processed data
(weightlifting + WISDM stamina), computes bias metrics,
and sends a report to Slack.

Configure:
  Airflow Variable ALERT_WEBHOOK_URL = "https://hooks.slack.com/services/..."
"""

from __future__ import annotations

import json
import os
import glob
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import urllib.request
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from dag_monitoring import (
    monitored_dag_args,
    on_dag_failure_callback,
    on_sla_miss_callback,
    emit_metric,
    log,
)

AIRFLOW_HOME = "/opt/airflow"
WISDM_DIR = f"{AIRFLOW_HOME}/data/processed/wisdm"
WEIGHTLIFTING_DIR = f"{AIRFLOW_HOME}/data/processed/weightlifting_cleaned"
REPORT_DIR = f"{AIRFLOW_HOME}/data/processed/bias_reports"


#  Helpers


def latest_parquet(directory: str, pattern: str) -> str | None:
    """Return the most recently modified parquet matching pattern."""
    files = sorted(
        glob.glob(os.path.join(directory, pattern)),
        key=os.path.getmtime,
        reverse=True,
    )
    return files[0] if files else None


def pct_diff(a: float, b: float) -> float:
    base = (a + b) / 2
    if base == 0:
        return 0.0
    return round(abs(a - b) / base * 100, 2)


#  DAG

with DAG(
    dag_id="daily_bias_monitoring",
    start_date=datetime(2024, 1, 1),
    schedule="0 6 * * *",
    catchup=False,
    max_active_runs=1,
    default_args=monitored_dag_args(retries=1, sla_minutes=20),
    on_failure_callback=on_dag_failure_callback,
    sla_miss_callback=on_sla_miss_callback,
    tags=["bias", "monitoring", "data-quality", "daily"],
) as dag:

    @task
    def analyze_wisdm_bias() -> Dict[str, Any]:
        """Bias analysis on WISDM stamina data."""
        report = {"dataset": "wisdm_stamina", "available": False}

        path = latest_parquet(WISDM_DIR, "wisdm_stamina_*.parquet")
        if not path:
            log.warning("No WISDM stamina parquet found in %s", WISDM_DIR)
            return report

        log.info("Loading WISDM stamina: %s", path)
        df = pd.read_parquet(path)
        report["available"] = True
        report["file"] = os.path.basename(path)
        report["total_rows"] = len(df)

        # Stamina by activity
        if "stamina" in df.columns and "activity" in df.columns:
            by_activity = (
                df.groupby("activity")["stamina"]
                .agg(["mean", "std", "min", "max", "count"])
                .round(4)
            )
            report["stamina_by_activity"] = by_activity.to_dict(orient="index")

            means = by_activity["mean"]
            report["highest_stamina_activity"] = means.idxmax()
            report["lowest_stamina_activity"] = means.idxmin()
            report["max_activity_gap_pct"] = pct_diff(means.max(), means.min())

        # Stamina by user
        if "stamina" in df.columns and "user" in df.columns:
            by_user = (
                df.groupby("user")["stamina"].agg(["mean", "std", "count"]).round(4)
            )
            report["user_count"] = len(by_user)
            user_means = by_user["mean"]
            report["user_stamina_spread"] = {
                "min_user_mean": round(float(user_means.min()), 4),
                "max_user_mean": round(float(user_means.max()), 4),
                "std_across_users": round(float(user_means.std()), 4),
                "gap_pct": pct_diff(user_means.max(), user_means.min()),
            }

            # Flag users with mean stamina >2 std from global mean
            global_mean = user_means.mean()
            global_std = user_means.std()
            outliers = by_user[
                (user_means < global_mean - 2 * global_std)
                | (user_means > global_mean + 2 * global_std)
            ]
            report["outlier_user_count"] = len(outliers)
            if len(outliers) > 0:
                report["outlier_users"] = outliers.index.tolist()[:20]

        log.info("WISDM bias report: %s", json.dumps(report, indent=2, default=str))
        return report

    @task
    def analyze_weightlifting_bias() -> Dict[str, Any]:
        """Bias analysis on cleaned weightlifting data."""
        report = {"dataset": "weightlifting", "available": False}

        path = latest_parquet(WEIGHTLIFTING_DIR, "workouts_clean_*.parquet")
        if not path:
            log.warning("No weightlifting parquet found in %s", WEIGHTLIFTING_DIR)
            return report

        log.info("Loading weightlifting: %s", path)
        df = pd.read_parquet(path)
        report["available"] = True
        report["file"] = os.path.basename(path)
        report["total_rows"] = len(df)

        # Volume by exercise (sets × reps × weight)
        if all(c in df.columns for c in ("Exercise Name", "Weight", "Reps")):
            df["volume"] = df["Weight"].fillna(0) * df["Reps"].fillna(0)

            by_exercise = (
                df.groupby("Exercise Name")
                .agg(
                    total_volume=("volume", "sum"),
                    avg_weight=("Weight", "mean"),
                    avg_reps=("Reps", "mean"),
                    set_count=("volume", "count"),
                )
                .round(2)
            )

            top_10 = by_exercise.nlargest(10, "total_volume")
            report["top_10_exercises"] = top_10.to_dict(orient="index")
            report["unique_exercises"] = len(by_exercise)

            # Distribution skew: top 5 exercises as % of total volume
            total_vol = by_exercise["total_volume"].sum()
            top5_vol = by_exercise.nlargest(5, "total_volume")["total_volume"].sum()
            report["top5_volume_pct"] = (
                round(top5_vol / total_vol * 100, 2) if total_vol > 0 else 0.0
            )

        # Weight progression check
        if all(c in df.columns for c in ("Date", "Exercise Name", "Weight")):
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            recent = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=30)]
            older = df[df["Date"] < df["Date"].max() - pd.Timedelta(days=30)]

            if len(recent) > 0 and len(older) > 0:
                recent_avg = recent["Weight"].mean()
                older_avg = older["Weight"].mean()
                report["weight_trend"] = {
                    "recent_30d_avg": round(float(recent_avg), 2),
                    "older_avg": round(float(older_avg), 2),
                    "change_pct": pct_diff(recent_avg, older_avg),
                    "direction": "up" if recent_avg > older_avg else "down",
                }

        # Workout frequency by day of week
        if "Date" in df.columns:
            df["_dow"] = pd.to_datetime(df["Date"], errors="coerce").dt.day_name()
            dow_counts = df.groupby("_dow").size().to_dict()
            report["sets_by_day_of_week"] = dow_counts

        log.info(
            "Weightlifting bias report: %s", json.dumps(report, indent=2, default=str)
        )
        return report

    @task
    def build_report(
        wisdm_report: Dict[str, Any],
        wl_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate bias reports, save JSON, build email HTML."""

        combined = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "wisdm": wisdm_report,
            "weightlifting": wl_report,
            "alerts": [],
        }

        # Alert rules
        if wisdm_report.get("available"):
            gap = wisdm_report.get("max_activity_gap_pct", 0)
            if gap > 50:
                combined["alerts"].append(
                    f"WISDM: Large stamina gap between activities ({gap}%)"
                )
            outliers = wisdm_report.get("outlier_user_count", 0)
            if outliers > 0:
                combined["alerts"].append(
                    f"WISDM: {outliers} user(s) with outlier stamina"
                )

        if wl_report.get("available"):
            top5 = wl_report.get("top5_volume_pct", 0)
            if top5 > 80:
                combined["alerts"].append(
                    f"Weightlifting: Top 5 exercises dominate {top5}% of volume"
                )

        # Save JSON report
        os.makedirs(REPORT_DIR, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        report_path = os.path.join(REPORT_DIR, f"bias_report_{ts}.json")
        with open(report_path, "w") as f:
            json.dump(combined, f, indent=2, default=str)
        log.info("Saved report -> %s", report_path)

        emit_metric(
            "daily_bias_monitoring",
            "build_report",
            {
                "wisdm_available": wisdm_report.get("available", False),
                "wl_available": wl_report.get("available", False),
                "alert_count": len(combined["alerts"]),
            },
        )

        return {
            "report_path": report_path,
            "alert_count": len(combined["alerts"]),
        }

    def _build_slack_blocks(report: Dict[str, Any]) -> List[Dict]:
        """Build Slack Block Kit message."""
        alerts = report.get("alerts", [])
        status = ":red_circle: ALERTS" if alerts else ":large_green_circle: ALL CLEAR"

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Daily Bias Monitoring Report"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Status:* {status}\n*Generated:* {report['generated_at']}",
                },
            },
        ]

        if alerts:
            alert_text = "\n".join(f"• {a}" for a in alerts)
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":warning: *Alerts:*\n{alert_text}",
                    },
                }
            )

        blocks.append({"type": "divider"})

        # WISDM section
        w = report.get("wisdm", {})
        if w.get("available"):
            lines = [f"*WISDM Accelerometer / Stamina*"]
            lines.append(
                f"File: `{w.get('file', 'N/A')}` | Rows: {w.get('total_rows', 0):,}"
            )

            if "stamina_by_activity" in w:
                lines.append(
                    f"Highest stamina: *{w.get('highest_stamina_activity')}* | Lowest: *{w.get('lowest_stamina_activity')}* | Gap: *{w.get('max_activity_gap_pct', 0)}%*"
                )

                # Compact table of top activities
                acts = sorted(
                    w["stamina_by_activity"].items(),
                    key=lambda x: x[1]["mean"],
                    reverse=True,
                )
                table = "```Activity | Mean   | Std    | Count\n"
                table += "-" * 38 + "\n"
                for act, vals in acts[:10]:
                    table += f"  {act:>6}  | {vals['mean']:>6.2f} | {vals['std']:>6.2f} | {vals['count']}\n"
                table += "```"
                lines.append(table)

            if "user_stamina_spread" in w:
                s = w["user_stamina_spread"]
                lines.append(
                    f"Users: {w.get('user_count', 0)} | Mean range: {s['min_user_mean']} – {s['max_user_mean']} (gap {s['gap_pct']}%)"
                )
                if w.get("outlier_user_count", 0) > 0:
                    lines.append(f":warning: Outlier users: `{w['outlier_users']}`")

            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "\n".join(lines)},
                }
            )
        else:
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "*WISDM:* No data available"},
                }
            )

        blocks.append({"type": "divider"})

        # Weightlifting section
        wl = report.get("weightlifting", {})
        if wl.get("available"):
            lines = ["*Weightlifting*"]
            lines.append(
                f"File: `{wl.get('file', 'N/A')}` | Rows: {wl.get('total_rows', 0):,} | Exercises: {wl.get('unique_exercises', 0)}"
            )

            if "top_10_exercises" in wl:
                table = "```Exercise                    | Volume     | Sets\n"
                table += "-" * 50 + "\n"
                for ex, vals in list(wl["top_10_exercises"].items())[:8]:
                    name = ex[:27]
                    table += f"{name:<28}| {vals['total_volume']:>10,.0f} | {vals['set_count']}\n"
                table += "```"
                lines.append(table)
                lines.append(
                    f"Top 5 volume concentration: *{wl.get('top5_volume_pct', 0)}%*"
                )

            if "weight_trend" in wl:
                t = wl["weight_trend"]
                arrow = ":arrow_up:" if t["direction"] == "up" else ":arrow_down:"
                lines.append(
                    f"Weight trend (30d): {t['older_avg']} → {t['recent_30d_avg']} {arrow} {t['change_pct']}%"
                )

            if "sets_by_day_of_week" in wl:
                dow = " | ".join(
                    f"{d[:3]}: {wl['sets_by_day_of_week'].get(d, 0)}"
                    for d in [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday",
                    ]
                )
                lines.append(f"Sets by day: {dow}")

            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "\n".join(lines)},
                }
            )
        else:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Weightlifting:* No data available",
                    },
                }
            )

        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "_Automated report from Airflow `daily_bias_monitoring` DAG_",
                    }
                ],
            }
        )

        return blocks

    @task
    def send_slack_report(report_result: Dict[str, Any]):
        """Send bias report to Slack via webhook."""
        url = Variable.get("ALERT_WEBHOOK_URL", default_var="")
        if not url:
            log.warning("No ALERT_WEBHOOK_URL set — skipping Slack notification")
            return

        # Load the full report JSON to build the message
        report_path = report_result["report_path"]
        with open(report_path) as f:
            report = json.load(f)

        blocks = _build_slack_blocks(report)

        # Slack webhook payload
        payload = json.dumps({"blocks": blocks}).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                log.info("Slack report sent (status=%s)", resp.status)
        except Exception as e:
            log.error("Failed to send Slack report: %s", e)
            raise AirflowFailException(f"Slack send failed: {e}")

    w_report = analyze_wisdm_bias()
    wl_report = analyze_weightlifting_bias()
    final = build_report(w_report, wl_report)
    final >> send_slack_report(final)
