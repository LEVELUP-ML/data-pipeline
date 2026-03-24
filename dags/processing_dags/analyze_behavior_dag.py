"""
DAG: clean_food_data

Preprocesses Food-101 dataset:
  1. Preprocess food images — build manifest, train/val split, class distribution
  2. Run Gemini inference on val set (mock mode if no API key)
  3. Evaluate predictions — compute accuracy metrics
  4. Quality gate
  5. Trigger bias monitoring + DVC backup

Input:  data/raw/food-101/
Output: data/processed/food/
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from dag_monitoring import (
    monitored_dag_args,
    on_dag_failure_callback,
    on_sla_miss_callback,
    emit_metric,
    log,
)

AIRFLOW_HOME  = "/opt/airflow"
RAW_ROOT      = f"{AIRFLOW_HOME}/data/raw/food-101"
OUT_DIR       = f"{AIRFLOW_HOME}/data/processed/food"
SCRIPTS_DIR   = f"{AIRFLOW_HOME}/scripts/food"


with DAG(
    dag_id="clean_food_data",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=monitored_dag_args(retries=1, sla_minutes=30),
    on_failure_callback=on_dag_failure_callback,
    sla_miss_callback=on_sla_miss_callback,
    tags=["processing", "food", "data-quality"],
) as dag:

    @task
    def preprocess() -> Dict[str, Any]:
        """
        Build manifest, train/val split, and class distribution
        by calling preprocess_food_images.py.
        """
        import subprocess

        Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [
                "python", f"{SCRIPTS_DIR}/preprocess_food_images.py",
                "--raw_root", RAW_ROOT,
                "--out_dir", OUT_DIR,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise AirflowFailException(
                f"preprocess_food_images.py failed:\n{result.stderr}"
            )

        log.info(result.stdout)

        # Read back summary stats from generated files
        import pandas as pd
        manifest_path = Path(OUT_DIR) / "food_manifest.csv"
        dist_path     = Path(OUT_DIR) / "food_class_distribution.csv"
        train_path    = Path(OUT_DIR) / "food_train.csv"
        val_path      = Path(OUT_DIR) / "food_val.csv"

        manifest = pd.read_csv(manifest_path)
        dist     = pd.read_csv(dist_path)
        train_df = pd.read_csv(train_path)
        val_df   = pd.read_csv(val_path)

        summary = {
            "total_items":    len(manifest),
            "unique_classes": int(manifest["label"].nunique()),
            "train_rows":     len(train_df),
            "val_rows":       len(val_df),
            "num_classes":    len(dist),
            "mean_per_class": round(float(dist["count"].mean()), 1),
            "std_per_class":  round(float(dist["count"].std()), 1),
            "min_class":      dist.iloc[-1]["label"],
            "min_count":      int(dist.iloc[-1]["count"]),
            "max_class":      dist.iloc[0]["label"],
            "max_count":      int(dist.iloc[0]["count"]),
        }

        emit_metric("clean_food_data", "preprocess", summary)
        log.info("Preprocess summary: %s", summary)
        return summary

    @task
    def run_inference() -> Dict[str, Any]:
        """
        Run Gemini food inference on val set.
        Uses real Gemini API if GEMINI_API_KEY is set,
        otherwise falls back to mock mode.
        """
        import subprocess

        gemini_key = Variable.get("GEMINI_API_KEY", default_var="")
        use_mock   = not bool(gemini_key)

        if use_mock:
            log.info("GEMINI_API_KEY not set — running in mock mode")
        else:
            os.environ["GEMINI_API_KEY"] = gemini_key
            log.info("GEMINI_API_KEY found — running real Gemini inference")

        cmd = [
            "python", f"{SCRIPTS_DIR}/infer_food_gemini.py",
            "--input_csv",  f"{OUT_DIR}/food_val.csv",
            "--out_jsonl",  f"{OUT_DIR}/food_predictions.jsonl",
            "--max_images", "25",
        ]
        if use_mock:
            cmd.append("--mock")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise AirflowFailException(
                f"infer_food_gemini.py failed:\n{result.stderr}"
            )

        log.info(result.stdout)

        # Count predictions written
        pred_path = Path(OUT_DIR) / "food_predictions.jsonl"
        n_preds   = sum(1 for _ in pred_path.open()) if pred_path.exists() else 0

        summary = {
            "predictions": n_preds,
            "mode": "mock" if use_mock else "gemini",
        }

        emit_metric("clean_food_data", "inference", summary)
        return summary

    @task
    def evaluate(inference_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate predictions — compute accuracy metrics
        by calling evaluate_food_model.py.
        """
        import subprocess

        result = subprocess.run(
            [
                "python", f"{SCRIPTS_DIR}/evaluate_food_model.py",
                "--predictions_jsonl", f"{OUT_DIR}/food_predictions.jsonl",
                "--out_metrics",       f"{OUT_DIR}/food_eval_metrics.json",
                "--out_results_csv",   f"{OUT_DIR}/food_eval_results.csv",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise AirflowFailException(
                f"evaluate_food_model.py failed:\n{result.stderr}"
            )

        log.info(result.stdout)

        metrics = json.loads((Path(OUT_DIR) / "food_eval_metrics.json").read_text())

        emit_metric("clean_food_data", "evaluate", metrics)
        log.info("Evaluation metrics: %s", metrics)
        return metrics

    @task
    def quality_gate(
        preprocess_result: Dict[str, Any],
        eval_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        total   = preprocess_result.get("total_items", 0)
        classes = preprocess_result.get("num_classes", 0)

        if classes < 10:
            raise AirflowFailException(
                f"Only {classes} classes found — expected ~101"
            )

        if (
            preprocess_result.get("std_per_class", 0)
            > preprocess_result.get("mean_per_class", 1) * 0.5
        ):
            log.warning(
                "Class distribution has high variance (std=%.1f, mean=%.1f)",
                preprocess_result["std_per_class"],
                preprocess_result["mean_per_class"],
            )

        accuracy = eval_result.get("accuracy", 0)
        if accuracy < 0.5:
            log.warning(
                "Model accuracy %.4f is below 0.5 — check inference quality",
                accuracy,
            )

        log.info(
            "Quality gate PASSED: %d items, %d classes, accuracy=%.4f",
            total,
            classes,
            accuracy,
        )

        return {
            "total_items": total,
            "classes":     classes,
            "accuracy":    accuracy,
            "passed":      True,
        }

    trigger_bias = TriggerDagRunOperator(
        task_id="trigger_food_bias",
        trigger_dag_id="food_bias_monitoring",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    trigger_dvc = TriggerDagRunOperator(
        task_id="trigger_dvc_backup",
        trigger_dag_id="dvc_backup_to_gcp",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    # Task flow
    pre       = preprocess()
    infer     = run_inference()
    pre >> infer

    eval_res  = evaluate(infer)
    gate      = quality_gate(pre, eval_res)
    gate >> [trigger_bias, trigger_dvc]