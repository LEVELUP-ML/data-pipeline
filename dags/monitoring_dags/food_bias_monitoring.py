"""
DAG: food_bias_monitoring

Analyzes Food-101 processed data for:
  - Class imbalance across 101 food categories
  - Prediction confidence bias per class
  - Under/over-represented classes
  - Train/val distribution skew

Sends Slack report. Triggered by clean_food_data or manually.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
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
FOOD_DIR = Path(f"{AIRFLOW_HOME}/data/processed/food")
REPORTS_DIR = Path(f"{AIRFLOW_HOME}/data/processed/food/reports")


with DAG(
    dag_id="food_bias_monitoring",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=monitored_dag_args(retries=1, sla_minutes=15),
    on_failure_callback=on_dag_failure_callback,
    sla_miss_callback=on_sla_miss_callback,
    tags=["monitoring", "food", "bias", "data-quality"],
) as dag:

    @task
    def analyze_class_balance() -> Dict[str, Any]:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        dist_path = FOOD_DIR / "food_class_distribution.csv"
        if not dist_path.exists():
            raise AirflowFailException(f"Class distribution not found: {dist_path}")

        dist = pd.read_csv(dist_path)
        log.info("Loaded class distribution: %d classes", len(dist))

        mean_count = dist["count"].mean()
        std_count = dist["count"].std()
        median_count = dist["count"].median()

        # Flag under-represented: < mean - 2*std
        threshold_low = max(1, mean_count - 2 * std_count)
        threshold_high = mean_count + 2 * std_count

        under = dist[dist["count"] < threshold_low]
        over = dist[dist["count"] > threshold_high]

        # Imbalance ratio: max / min
        imbalance_ratio = (
            round(dist["count"].max() / dist["count"].min(), 2)
            if dist["count"].min() > 0
            else float("inf")
        )

        # Top 10 and bottom 10
        top10 = dist.nlargest(10, "count")[["label", "count"]].to_dict(orient="records")
        bottom10 = dist.nsmallest(10, "count")[["label", "count"]].to_dict(
            orient="records"
        )

        report = {
            "num_classes": len(dist),
            "mean_per_class": round(float(mean_count), 1),
            "std_per_class": round(float(std_count), 1),
            "median_per_class": round(float(median_count), 1),
            "imbalance_ratio": imbalance_ratio,
            "under_represented_count": len(under),
            "under_represented": under["label"].tolist() if len(under) > 0 else [],
            "over_represented_count": len(over),
            "over_represented": over["label"].tolist() if len(over) > 0 else [],
            "top_10": top10,
            "bottom_10": bottom10,
        }

        log.info(
            "Class balance: %d classes, ratio=%.2f, under=%d, over=%d",
            len(dist),
            imbalance_ratio,
            len(under),
            len(over),
        )

        emit_metric(
            "food_bias_monitoring",
            "class_balance",
            {
                "num_classes": len(dist),
                "imbalance_ratio": imbalance_ratio,
                "under_represented": len(under),
                "over_represented": len(over),
            },
        )

        return report

    @task
    def analyze_split_skew() -> Dict[str, Any]:
        train_path = FOOD_DIR / "food_train.csv"
        val_path = FOOD_DIR / "food_val.csv"

        if not train_path.exists() or not val_path.exists():
            log.warning("Train/val splits not found — skipping split skew analysis")
            return {"available": False}

        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)

        train_dist = train["label"].value_counts(normalize=True)
        val_dist = val["label"].value_counts(normalize=True)

        # Align on same labels
        all_labels = sorted(set(train_dist.index) | set(val_dist.index))
        train_pcts = [train_dist.get(l, 0) for l in all_labels]
        val_pcts = [val_dist.get(l, 0) for l in all_labels]

        # Max absolute difference in class proportions
        diffs = [abs(t - v) for t, v in zip(train_pcts, val_pcts)]
        max_diff_idx = np.argmax(diffs)
        max_diff_label = all_labels[max_diff_idx]
        max_diff_value = round(diffs[max_diff_idx] * 100, 3)

        # Mean absolute difference
        mean_diff = round(np.mean(diffs) * 100, 3)

        report = {
            "available": True,
            "train_rows": len(train),
            "val_rows": len(val),
            "train_classes": int(train["label"].nunique()),
            "val_classes": int(val["label"].nunique()),
            "max_proportion_diff_pct": max_diff_value,
            "max_diff_class": max_diff_label,
            "mean_proportion_diff_pct": mean_diff,
        }

        log.info(
            "Split skew: max diff=%.3f%% (%s), mean diff=%.3f%%",
            max_diff_value,
            max_diff_label,
            mean_diff,
        )

        emit_metric("food_bias_monitoring", "split_skew", report)
        return report

    @task
    def analyze_prediction_bias() -> Dict[str, Any]:
        pred_path = FOOD_DIR / "food_predictions.jsonl"
        if not pred_path.exists():
            log.warning("Predictions file not found — skipping prediction bias")
            return {"available": False}

        records = [
            json.loads(line)
            for line in pred_path.read_text().splitlines()
            if line.strip()
        ]
        df = pd.DataFrame(records)
        log.info("Loaded %d predictions", len(df))

        # Confidence by true label
        conf_by_class = (
            df.groupby("true_label")["confidence"]
            .agg(["mean", "std", "count"])
            .round(4)
        )

        low_conf = conf_by_class[conf_by_class["mean"] < 0.7]
        high_conf = conf_by_class[conf_by_class["mean"] > 0.9]

        # Accuracy (mock: predicted matches true)
        df["correct"] = df["predicted_food"] == df["true_label"].str.replace("_", " ")
        accuracy = round(df["correct"].mean(), 4)

        # Per-class accuracy
        acc_by_class = df.groupby("true_label")["correct"].mean().round(4)
        worst_classes = acc_by_class.nsmallest(5).to_dict()
        best_classes = acc_by_class.nlargest(5).to_dict()

        report = {
            "available": True,
            "total_predictions": len(df),
            "overall_accuracy": accuracy,
            "avg_confidence": round(float(df["confidence"].mean()), 4),
            "low_confidence_classes": low_conf.index.tolist(),
            "high_confidence_classes": high_conf.index.tolist(),
            "worst_5_accuracy": worst_classes,
            "best_5_accuracy": best_classes,
        }

        log.info(
            "Prediction bias: accuracy=%.4f, avg_conf=%.4f, low_conf_classes=%d",
            accuracy,
            df["confidence"].mean(),
            len(low_conf),
        )

        emit_metric(
            "food_bias_monitoring",
            "prediction_bias",
            {
                "accuracy": accuracy,
                "avg_confidence": report["avg_confidence"],
                "low_conf_classes": len(low_conf),
            },
        )

        return report

    @task
    def build_and_send_report(
        class_result: Dict[str, Any],
        split_result: Dict[str, Any],
        pred_result: Dict[str, Any],
    ):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        combined = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "class_balance": class_result,
            "split_skew": split_result,
            "prediction_bias": pred_result,
            "alerts": [],
        }

        # Alert rules
        if class_result.get("imbalance_ratio", 1) > 3:
            combined["alerts"].append(
                f"Class imbalance ratio {class_result['imbalance_ratio']} exceeds 3x"
            )
        if class_result.get("under_represented_count", 0) > 0:
            combined["alerts"].append(
                f"{class_result['under_represented_count']} under-represented classes detected"
            )
        if (
            split_result.get("available")
            and split_result.get("max_proportion_diff_pct", 0) > 1.0
        ):
            combined["alerts"].append(
                f"Train/val split skew: {split_result['max_proportion_diff_pct']}% max diff on {split_result['max_diff_class']}"
            )
        if (
            pred_result.get("available")
            and pred_result.get("overall_accuracy", 1) < 0.8
        ):
            combined["alerts"].append(
                f"Low overall accuracy: {pred_result['overall_accuracy']}"
            )
        if (
            pred_result.get("available")
            and len(pred_result.get("low_confidence_classes", [])) > 5
        ):
            combined["alerts"].append(
                f"{len(pred_result['low_confidence_classes'])} classes with low confidence (<0.7)"
            )

        # Save report
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        report_path = REPORTS_DIR / f"food_bias_report_{ts}.json"
        report_path.write_text(json.dumps(combined, indent=2, default=str))
        log.info("Saved report -> %s", report_path)

        emit_metric(
            "food_bias_monitoring",
            "build_report",
            {
                "alert_count": len(combined["alerts"]),
            },
        )

        # Slack
        url = Variable.get("ALERT_WEBHOOK_URL", default_var="")
        if not url:
            log.warning("No ALERT_WEBHOOK_URL — skipping Slack")
            return

        alerts = combined["alerts"]
        status = ":red_circle: ALERTS" if alerts else ":large_green_circle: ALL CLEAR"

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Food-101 Bias Report"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Status:* {status}\n*Generated:* {combined['generated_at']}"
                    ),
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

        # Class balance
        c = class_result
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Class Balance*\n"
                        f"Classes: {c['num_classes']} | Mean: {c['mean_per_class']} | "
                        f"Std: {c['std_per_class']} | Imbalance ratio: {c['imbalance_ratio']}\n"
                        f"Under-represented: {c['under_represented_count']} | Over-represented: {c['over_represented_count']}"
                    ),
                },
            }
        )

        # Bottom 5 classes
        if c.get("bottom_10"):
            bottom5 = c["bottom_10"][:5]
            table = "```Class                | Count\n" + "-" * 30 + "\n"
            for item in bottom5:
                table += f"{item['label']:<21}| {item['count']}\n"
            table += "```"
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Smallest classes:*\n{table}"},
                }
            )

        # Split skew
        if split_result.get("available"):
            s = split_result
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*Train/Val Split*\n"
                            f"Train: {s['train_rows']:,} | Val: {s['val_rows']:,}\n"
                            f"Max proportion diff: {s['max_proportion_diff_pct']}% ({s['max_diff_class']})\n"
                            f"Mean proportion diff: {s['mean_proportion_diff_pct']}%"
                        ),
                    },
                }
            )

        # Prediction bias
        if pred_result.get("available"):
            p = pred_result
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*Prediction Bias*\n"
                            f"Accuracy: {p['overall_accuracy']} | Avg confidence: {p['avg_confidence']}\n"
                            f"Low confidence classes: {len(p.get('low_confidence_classes', []))}"
                        ),
                    },
                }
            )

            if p.get("worst_5_accuracy"):
                worst = "\n".join(
                    f"• `{k}`: {v}" for k, v in p["worst_5_accuracy"].items()
                )
                blocks.append(
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Worst accuracy:*\n{worst}",
                        },
                    }
                )

        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "_Automated report from `food_bias_monitoring` DAG_",
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
                log.info("Slack report sent (status=%s)", resp.status)
        except Exception as e:
            log.error("Slack send failed: %s", e)

    # Task flow
    class_bal = analyze_class_balance()
    split_skew = analyze_split_skew()
    pred_bias = analyze_prediction_bias()

    report = build_and_send_report(class_bal, split_skew, pred_bias)
