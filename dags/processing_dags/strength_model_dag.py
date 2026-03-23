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


@task
def run_training() -> Dict[str, Any]:
    """
    Dynamically imports scripts/model_train.py and calls train() for strength.
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
    })
    return metrics


@task
def validate_model(training_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate model performance and bias.
    """
    # Check RMSE gate
    gate_rmse = training_metrics["gate_rmse"]
    threshold = float(Variable.get("MODEL_RMSE_THRESHOLD", default_var="15.0"))

    if gate_rmse > threshold:
        raise AirflowFailException(
            f"Model validation FAILED: d7 RMSE {gate_rmse:.4f} > {threshold}. "
            "Model performance below acceptable threshold."
        )

    # Check bias report exists
    if not BIAS_PATH.exists():
        raise AirflowFailException(f"Bias report not found: {BIAS_PATH}")

    with open(BIAS_PATH) as f:
        bias_report = json.load(f)

    # Check for problematic bias
    max_ratio = float(Variable.get("BIAS_MAX_RMSE_RATIO", default_var="2.0"))
    problematic_groups = []

    for group, metrics in bias_report.items():
        if isinstance(metrics, dict) and "rmse" in metrics:
            ratio = metrics["rmse"] / gate_rmse if gate_rmse > 0 else float("inf")
            if ratio > max_ratio:
                problematic_groups.append(f"{group}: {ratio:.2f}")

    if problematic_groups:
        raise AirflowFailException(
            f"Model bias check FAILED: Groups with RMSE ratio > {max_ratio}: "
            f"{', '.join(problematic_groups)}"
        )

    log.info("Model validation PASSED: RMSE=%.4f <= %.4f, bias check OK",
             gate_rmse, threshold)
    emit_metric("strength_model", "validate_model", {
        "gate_rmse": gate_rmse,
        "threshold": threshold,
        "bias_max_ratio": max_ratio,
        "passed": True
    })
    return {"passed": True, "metrics": training_metrics}


@task
def rollback_check(validation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare against previous model version.
    """
    current_rmse = validation["metrics"]["gate_rmse"]

    # Try to load previous metrics
    prev_metrics = None
    try:
        bucket = Variable.get("GCS_BACKUP_BUCKET", default_var="")
        if bucket:
            hook = GCSHook()
            prev_path = f"{REGISTRY_GCS}/latest.json"
            if hook.exists(bucket, prev_path):
                content = hook.download_as_string(bucket, prev_path)
                prev_data = json.loads(content)
                prev_metrics = prev_data.get("metrics", {}).get("gate_rmse")
    except Exception as e:
        log.warning("Could not load previous model metrics: %s", e)

    if prev_metrics is not None:
        degradation = (current_rmse - prev_metrics) / prev_metrics
        max_degradation = 0.10  # 10% degradation allowed

        if degradation > max_degradation:
            raise AirflowFailException(
                f"Rollback check FAILED: Model regressed by {degradation:.1%} "
                f"(current: {current_rmse:.4f}, previous: {prev_metrics:.4f})"
            )

        log.info("Rollback check PASSED: Degradation %.1f%% <= %.1f%%",
                 degradation * 100, max_degradation * 100)
    else:
        log.info("No previous model found - skipping rollback check")

    emit_metric("strength_model", "rollback_check", {
        "current_rmse": current_rmse,
        "previous_rmse": prev_metrics,
        "passed": True
    })
    return {"passed": True, "current_rmse": current_rmse, "previous_rmse": prev_metrics}


@task
def push_to_registry(validation: Dict[str, Any], rollback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload model artifacts to GCS registry.
    """
    bucket = Variable.get("GCS_BACKUP_BUCKET", default_var="")
    if not bucket:
        raise AirflowFailException("GCS_BACKUP_BUCKET variable not set")

    hook = GCSHook()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Upload artifacts
    uploaded = []
    artifacts = [
        (MODEL_PATH, f"{REGISTRY_GCS}/{timestamp}/model.pkl"),
        (METRICS_PATH, f"{REGISTRY_GCS}/{timestamp}/metrics.json"),
        (BIAS_PATH, f"{REGISTRY_GCS}/{timestamp}/bias_report.json"),
        (SHAP_PATH, f"{REGISTRY_GCS}/{timestamp}/shap_summary.png"),
    ]

    for local_path, gcs_path in artifacts:
        if local_path.exists():
            hook.upload(bucket, gcs_path, local_path)
            uploaded.append(gcs_path)
            log.info("Uploaded %s to gs://%s/%s", local_path.name, bucket, gcs_path)

    # Update latest.json pointer
    latest = {
        "timestamp": timestamp,
        "artifacts": uploaded,
        "metrics": validation["metrics"],
    }

    hook.upload(
        bucket,
        f"{REGISTRY_GCS}/latest.json",
        json.dumps(latest, indent=2).encode("utf-8"),
        mime_type="application/json",
    )
    log.info("Updated latest.json pointer")

    result = {
        "registry_path": f"gs://{bucket}/{REGISTRY_GCS}/{timestamp}",
        "files_uploaded": len(uploaded),
        "timestamp": timestamp,
    }
    emit_metric("strength_model", "push_to_registry", result)
    return result


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