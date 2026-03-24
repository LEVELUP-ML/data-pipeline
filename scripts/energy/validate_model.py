#!/usr/bin/env python3
"""
Model validation with quality gates.

Loads the trained model and test data, computes metrics,
generates diagnostic plots, and enforces validation thresholds.

Exit code 0 = PASS, 1 = FAIL (used by CI/CD as a gate).

Usage:
    python validate_model.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BEST_MODEL_PATH, VALIDATION_REPORT, PLOTS_DIR,
    MAX_ACCEPTABLE_MAE, MIN_ACCEPTABLE_R2,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, ensure_dirs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_test_data():
    """Load saved test data from training."""
    test_path = BEST_MODEL_PATH.parent / "test_data.json"
    data = json.loads(test_path.read_text())
    X_test = np.array(data["X_test"])
    y_test = np.array(data["y_test"])
    feature_names = data["feature_names"]
    return X_test, y_test, feature_names


def compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "r2": round(r2_score(y_true, y_pred), 4),
        "max_error": round(float(np.max(np.abs(y_true - y_pred))), 4),
        "mean_prediction": round(float(np.mean(y_pred)), 4),
        "mean_actual": round(float(np.mean(y_true)), 4),
        "prediction_std": round(float(np.std(y_pred)), 4),
    }


def generate_plots(y_true, y_pred):
    """Generate validation diagnostic plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, s=10, color="#4A90D9")
    ax1.plot([0, 100], [0, 100], "r--", linewidth=1, label="Perfect prediction")
    ax1.set_xlabel("Actual Energy Score")
    ax1.set_ylabel("Predicted Energy Score")
    ax1.set_title("Actual vs Predicted")
    ax1.legend()
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)

    residuals = y_true - y_pred
    ax2 = axes[1]
    ax2.hist(residuals, bins=40, color="#5CB85C", edgecolor="white", alpha=0.8)
    ax2.axvline(0, color="red", linestyle="--", linewidth=1)
    ax2.set_xlabel("Residual (Actual - Predicted)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Residual Distribution (mean={residuals.mean():.2f}, std={residuals.std():.2f})")

    plt.tight_layout()
    plot_path = PLOTS_DIR / "validation_diagnostics.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved validation plots → %s", plot_path)

    return str(plot_path)


def main():
    """Run model validation with gates."""
    ensure_dirs()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    log.info("Loading model from %s", BEST_MODEL_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    X_test, y_test, feature_names = load_test_data()

    log.info("Validating on %d test samples", len(X_test))

    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    log.info("Validation Metrics:")
    for k, v in metrics.items():
        log.info("  %s: %.4f", k, v)

    plot_path = generate_plots(y_test, y_pred)

    gates = {
        "mae_gate": {
            "threshold": MAX_ACCEPTABLE_MAE,
            "actual": metrics["mae"],
            "passed": metrics["mae"] <= MAX_ACCEPTABLE_MAE,
        },
        "r2_gate": {
            "threshold": MIN_ACCEPTABLE_R2,
            "actual": metrics["r2"],
            "passed": metrics["r2"] >= MIN_ACCEPTABLE_R2,
        },
    }

    all_passed = all(g["passed"] for g in gates.values())

    report = {
        "status": "PASS" if all_passed else "FAIL",
        "metrics": metrics,
        "gates": gates,
        "test_samples": len(X_test),
        "feature_names": feature_names,
        "plot_path": plot_path,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    VALIDATION_REPORT.write_text(json.dumps(report, indent=2))
    log.info("Validation report → %s", VALIDATION_REPORT)

    with mlflow.start_run(run_name="validation"):
        for k, v in metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        mlflow.log_param("validation_status", report["status"])
        mlflow.log_artifact(plot_path)
        mlflow.log_artifact(str(VALIDATION_REPORT))

    if not all_passed:
        failed = [k for k, v in gates.items() if not v["passed"]]
        log.error("❌ VALIDATION FAILED: %s", failed)
        for g in failed:
            gate = gates[g]
            log.error("  %s: %.4f (threshold: %.4f)", g, gate["actual"], gate["threshold"])
        sys.exit(1)

    log.info("✅ All validation gates passed")
    return report


if __name__ == "__main__":
    main()