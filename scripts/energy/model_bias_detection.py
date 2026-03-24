#!/usr/bin/env python3
"""
Model bias detection via data slicing.

Evaluates model performance across demographic slices (sex, age bucket)
to identify potential fairness issues.

Usage:
    python model_bias_detection.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BEST_MODEL_PATH, BIAS_DETECTION_REPORT, DAILY_JOINED,
    FEATURE_COLUMNS, TARGET_COLUMN, AGE_BINS,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, ensure_dirs,
)
from train_model import load_and_prepare_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEVIATION_THRESHOLD = 0.20  # Flag if slice MAE deviates > 20% from global


def assign_age_bucket(age):
    """Map age to bucket label."""
    try:
        age = int(age)
        for lo, hi, label in AGE_BINS:
            if lo <= age <= hi:
                return label
    except (ValueError, TypeError):
        pass
    return "unknown"


def compute_slice_metrics(y_true, y_pred, global_mae):
    """Compute metrics for a single slice."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    deviation = abs(mae - global_mae) / global_mae if global_mae > 0 else 0

    return {
        "count": len(y_true),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mean_prediction": round(float(np.mean(y_pred)), 4),
        "mean_actual": round(float(np.mean(y_true)), 4),
        "prediction_spread": round(float(np.std(y_pred)), 4),
        "mae_deviation_pct": round(deviation * 100, 2),
        "flagged": deviation > DEVIATION_THRESHOLD,
    }


def main():
    """Run bias detection analysis."""
    ensure_dirs()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Load model
    model = joblib.load(BEST_MODEL_PATH)
    log.info("Loaded model from %s", BEST_MODEL_PATH)

    # Load full dataset
    X, y, df, feature_names = load_and_prepare_data()
    y_pred = model.predict(X)

    # Global metrics
    global_mae = mean_absolute_error(y, y_pred)
    log.info("Global MAE: %.4f", global_mae)

    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "global_mae": round(global_mae, 4),
        "total_samples": len(y),
        "deviation_threshold_pct": DEVIATION_THRESHOLD * 100,
        "slices": {},
        "flagged_slices": [],
    }

    # Slice by sex
    if "sex" in df.columns:
        report["slices"]["sex"] = {}
        for sex_val in df["sex"].unique():
            mask = df["sex"] == sex_val
            if mask.sum() > 0:
                metrics = compute_slice_metrics(y[mask], y_pred[mask], global_mae)
                report["slices"]["sex"][str(sex_val)] = metrics
                log.info("  sex=%s: MAE=%.4f (n=%d)%s",
                         sex_val, metrics["mae"], metrics["count"],
                         " ⚠️ FLAGGED" if metrics["flagged"] else "")
                if metrics["flagged"]:
                    report["flagged_slices"].append(f"sex={sex_val}")

    # Slice by age bucket
    if "age" in df.columns:
        df["age_bucket"] = df["age"].apply(assign_age_bucket)
        report["slices"]["age_bucket"] = {}
        for bucket in df["age_bucket"].unique():
            mask = df["age_bucket"] == bucket
            if mask.sum() > 0:
                metrics = compute_slice_metrics(y[mask], y_pred[mask], global_mae)
                report["slices"]["age_bucket"][str(bucket)] = metrics
                log.info("  age_bucket=%s: MAE=%.4f (n=%d)%s",
                         bucket, metrics["mae"], metrics["count"],
                         " ⚠️ FLAGGED" if metrics["flagged"] else "")
                if metrics["flagged"]:
                    report["flagged_slices"].append(f"age_bucket={bucket}")

    # Mitigation documentation
    report["mitigation_steps"] = [
        "1. Stratified train/test splitting ensures proportional representation.",
        "2. Sample weighting can be applied to underrepresented groups.",
        "3. Per-slice performance monitoring catches degradation early.",
        "4. Data augmentation or synthetic oversampling for small slices.",
        "5. Fairness constraints can be added during model optimization.",
        "6. Regular bias audits post-deployment track drift over time.",
    ]

    # Save report
    BIAS_DETECTION_REPORT.write_text(json.dumps(report, indent=2))
    log.info("Bias detection report → %s", BIAS_DETECTION_REPORT)

    # Log to MLflow
    with mlflow.start_run(run_name="bias_detection"):
        mlflow.log_metric("global_mae", global_mae)
        mlflow.log_metric("n_flagged_slices", len(report["flagged_slices"]))
        mlflow.log_artifact(str(BIAS_DETECTION_REPORT))

    if report["flagged_slices"]:
        log.warning("⚠️ Bias detected in %d slices: %s",
                     len(report["flagged_slices"]), report["flagged_slices"])
    else:
        log.info("✅ No significant bias detected across slices")

    return report


if __name__ == "__main__":
    main()