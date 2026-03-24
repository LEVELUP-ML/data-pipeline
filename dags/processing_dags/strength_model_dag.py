"""
DAG: strength_model

Orchestrates the full modeling pipeline per the Model Development Guidelines:
  1. run_training         — calls scripts/model_train.py, XGBoost + hyperparameter search
  2. validate_model       — enforces per-horizon RMSE gate, checks bias report
  3. rollback_check       — compares against previous registered model; blocks if regressed
  4. push_to_registry     — uploads artifacts to gs://{bucket}/model_registry/strength/
  5. trigger_dvc_backup

Triggered by: strength_features DAG
Can also run manually.

Airflow Variables:
  MODEL_RMSE_THRESHOLD         — gate on d7 RMSE (default 15.0 for strength)
  BIAS_MAX_RMSE_RATIO          — hard block if any group RMSE ratio > this (default 2.0)
  GCS_BACKUP_BUCKET            — GCS bucket name
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pendulum
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook

from dag_monitoring import (
    emit_metric, log, monitored_dag_args,
    on_dag_failure_callback, on_sla_miss_callback,
)

AIRFLOW_HOME = "/opt/airflow"
MODELS_DIR   = Path(f"{AIRFLOW_HOME}/data/models/strength")
SCRIPTS_DIR  = Path(f"{AIRFLOW_HOME}/scripts")
MODEL_PATH   = MODELS_DIR / "model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
BIAS_PATH    = MODELS_DIR / "bias_report.json"
SHAP_PATH    = MODELS_DIR / "shap_summary.png"
REGISTRY_GCS = "model_registry/strength"


with DAG(
    dag_id="strength_model",
    start_date=pendulum.datetime(2026, 2, 1, tz="America/New_York"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=monitored_dag_args(retries=1, sla_minutes=90),
    on_failure_callback=on_dag_failure_callback,
    sla_miss_callback=on_sla_miss_callback,
    tags=["strength", "model", "training", "mlops"],
) as dag:

    @task
    def run_training() -> Dict[str, Any]:
        """
        Dynamically imports scripts/model_train.py and calls train() for strength.
        Keeping logic in a standalone script means it is also:
          - runnable from CLI for local iteration
          - tracked as a DVC stage (dvc repro train_strength_model)
          - testable without Airflow
        """
        script_path = SCRIPTS_DIR / "model_train.py"
        if not script_path.exists():
            raise AirflowFailException(
                f"Training script not found: {script_path}\n"
                "Expected at scripts/model_train.py in the Airflow container."
            )

        # Surface Airflow Variable into env so model_train.py reads it
        os.environ["MODEL_RMSE_THRESHOLD"] = Variable.get(
            "MODEL_RMSE_THRESHOLD", default_var="15.0"
        )

        spec   = importlib.util.spec_from_file_location("model_train", str(script_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        run_id = f"airflow_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
        try:
            metrics = module.train(run_id=run_id, model_type="strength")
        except ValueError as e:
            raise AirflowFailException(str(e))  # RMSE gate failure

        emit_metric("strength_model", "run_training", {
            "gate_rmse": metrics["gate_rmse"],
            "d1_rmse":   metrics["test_metrics"]["d1"]["rmse"],
            "d7_rmse":   metrics["test_metrics"]["d7"]["rmse"],
            "d14_rmse":  metrics["test_metrics"]["d14"]["rmse"],
            "train_rows": metrics["train_rows"],
        })
        log.info("Training complete — gate RMSE (d7)=%.4f", metrics["gate_rmse"])
        return metrics

    @task
    def validate_model(training_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reads saved metrics.json and bias_report.json.
        Enforces:
          - d7 RMSE below threshold (double-check after training gate)
          - No demographic group with RMSE ratio > BIAS_MAX_RMSE_RATIO (hard block)
          - model.pkl exists
        Warns (but does not block) on softer bias flags.
        """
        for p in (MODEL_PATH, METRICS_PATH):
            if not p.exists():
                raise AirflowFailException(f"Missing artifact: {p}")

        metrics   = json.loads(METRICS_PATH.read_text())
        bias      = json.loads(BIAS_PATH.read_text()) if BIAS_PATH.exists() else {"available": False}

        threshold  = float(Variable.get("MODEL_RMSE_THRESHOLD", default_var="15.0"))
        max_ratio  = float(Variable.get("BIAS_MAX_RMSE_RATIO",  default_var="2.0"))
        gate_rmse  = metrics["gate_rmse"]

        log.info("Validation: gate RMSE=%.4f (threshold=%.1f)", gate_rmse, threshold)

        if gate_rmse > threshold:
            raise AirflowFailException(
                f"Validation FAILED: d7 RMSE {gate_rmse:.4f} > {threshold}."
            )

        # Bias hard block (extreme disparity)
        if bias.get("available"):
            for slice_col, slice_data in bias.get("slices", {}).items():
                overall = slice_data.get("overall_rmse", 0)
                for group, g_rmse in slice_data.get("by_group", {}).items():
                    ratio = g_rmse / overall if overall else 1.0
                    if ratio > max_ratio:
                        raise AirflowFailException(
                            f"Bias BLOCKED: {slice_col}={group} RMSE={g_rmse:.2f} "
                            f"({ratio:.1f}× overall). Model not deployed."
                        )

        flagged = bias.get("flagged", [])
        if flagged:
            log.warning("Soft bias flags (not blocking): %s", flagged)

        result = {
            "gate_rmse":          gate_rmse,
            "d1_rmse":            metrics["test_metrics"]["d1"]["rmse"],
            "d7_rmse":            metrics["test_metrics"]["d7"]["rmse"],
            "d14_rmse":           metrics["test_metrics"]["d14"]["rmse"],
            "bias_flagged_count": len(flagged),
            "passed":             True,
        }
        emit_metric("strength_model", "validate_model", result)
        log.info("Validation PASSED: %s", result)
        return result

    @task
    def rollback_check(validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compares the new model's d7 RMSE against the previously deployed model
        stored in gs://{bucket}/model_registry/strength/latest.json.

        Blocks deployment if the new model is more than 10% worse.
        On first deploy (no previous model) — always passes.
        """
        bucket = Variable.get("GCS_BACKUP_BUCKET")
        gcs    = GCSHook(gcp_conn_id="google_cloud_default")

        new_rmse = validation_result["d7_rmse"]
        latest_key = f"{REGISTRY_GCS}/latest.json"

        try:
            raw  = gcs.download(bucket_name=bucket, object_name=latest_key)
            prev = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            prev_rmse = float(prev.get("gate_rmse", float("inf")))
            log.info("Previous model d7 RMSE: %.4f | New: %.4f", prev_rmse, new_rmse)
        except Exception:
            log.info("No previous model found in registry — first deploy, skipping rollback check.")
            return {"prev_rmse": None, "new_rmse": new_rmse, "action": "first_deploy"}

        # Block if new model is more than 10% worse
        regression_threshold = prev_rmse * 1.10
        if new_rmse > regression_threshold:
            raise AirflowFailException(
                f"Rollback triggered: new model d7 RMSE {new_rmse:.4f} is more than 10% "
                f"worse than deployed model {prev_rmse:.4f} "
                f"(threshold={regression_threshold:.4f}). "
                "Retrain with more data or tune hyperparameters."
            )

        action = "improved" if new_rmse < prev_rmse else "similar"
        log.info("Rollback check PASSED — action=%s", action)
        emit_metric("strength_model", "rollback_check", {
            "prev_rmse": prev_rmse, "new_rmse": new_rmse, "action": action
        })
        return {"prev_rmse": prev_rmse, "new_rmse": new_rmse, "action": action}

    @task
    def push_to_registry(
        validation_result: Dict[str, Any],
        rollback_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Uploads model artifacts to:
          gs://{bucket}/model_registry/strength/{timestamp}/
            model.pkl
            metrics.json
            bias_report.json
            shap_summary.png

        Updates latest.json pointer so rollback_check can compare next run.
        """
        bucket    = Variable.get("GCS_BACKUP_BUCKET")
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        prefix    = f"{REGISTRY_GCS}/{timestamp}"
        gcs       = GCSHook(gcp_conn_id="google_cloud_default")

        uploaded = []
        for local_path in [MODEL_PATH, METRICS_PATH, BIAS_PATH, SHAP_PATH]:
            if not local_path.exists():
                if local_path == SHAP_PATH:
                    log.warning("SHAP plot not found — skipping.")
                    continue
                raise AirflowFailException(f"Required artifact missing: {local_path}")

            obj_name = f"{prefix}/{local_path.name}"
            gcs.upload(
                bucket_name=bucket,
                object_name=obj_name,
                data=local_path.read_bytes(),
                mime_type="application/octet-stream",
            )
            uploaded.append(obj_name)
            log.info("Uploaded → gs://%s/%s", bucket, obj_name)

        # Update latest.json pointer (used by rollback_check on next run)
        metrics  = json.loads(METRICS_PATH.read_text())
        latest = {
            "timestamp":  timestamp,
            "prefix":     f"gs://{bucket}/{prefix}",
            "gate_rmse":  validation_result["d7_rmse"],
            "d1_rmse":    validation_result["d1_rmse"],
            "d14_rmse":   validation_result["d14_rmse"],
            "bias_flags": metrics.get("bias_flagged", []),
        }
        gcs.upload(
            bucket_name=bucket,
            object_name=f"{REGISTRY_GCS}/latest.json",
            data=json.dumps(latest, indent=2).encode("utf-8"),
            mime_type="application/json",
        )
        log.info("Updated latest.json pointer")

        result = {
            "registry_path":  f"gs://{bucket}/{prefix}",
            "files_uploaded": len(uploaded),
            "timestamp":      timestamp,
        }
        emit_metric("strength_model", "push_to_registry", result)
        return result

    trigger_dvc = TriggerDagRunOperator(
        task_id="trigger_dvc_backup",
        trigger_dag_id="dvc_backup_to_gcp",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    # ── Task flow (matches PDF section 7 order) ───────────────────────────────
    training   = run_training()
    validation = validate_model(training)
    rollback   = rollback_check(validation)
    registry   = push_to_registry(validation, rollback)
    registry >> trigger_dvc
