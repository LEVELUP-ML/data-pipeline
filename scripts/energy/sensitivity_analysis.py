#!/usr/bin/env python3
"""
Sensitivity analysis using SHAP (SHapley Additive exPlanations).

Computes feature importance via TreeSHAP for the trained model,
generates summary plots, and saves importance rankings.

Usage:
    python sensitivity_analysis.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BEST_MODEL_PATH, SHAP_REPORT, PLOTS_DIR,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, ensure_dirs,
)
from train_model import load_and_prepare_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _get_importance(model, X_sample, feature_names):
    """
    Get feature importance array and method name.

    Uses SHAP TreeExplainer for RandomForest. For XGBoost, falls back to
    feature_importances_ due to a known SHAP/XGBoost base_score
    incompatibility where base_score is stored as '[5.47E1]' format.
    """
    try:
        import xgboost
        is_xgb = isinstance(model, xgboost.XGBRegressor)
    except ImportError:
        is_xgb = False

    if is_xgb:
        log.warning(
            "Skipping SHAP TreeExplainer for XGBoost due to known base_score "
            "incompatibility — using feature_importances_ instead."
        )
        return model.feature_importances_, "xgboost_feature_importances"

    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    log.info("SHAP values computed successfully")
    return np.abs(shap_values).mean(axis=0), "shap"


def main():
    """Run SHAP sensitivity analysis."""
    ensure_dirs()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Load model and data
    model = joblib.load(BEST_MODEL_PATH)
    X, y, df, feature_names = load_and_prepare_data()
    log.info("Computing feature importance for %d samples, %d features...", len(X), len(feature_names))

    try:
        import shap  # noqa: F401 — ensure shap is installed before proceeding

        # Sample data if too large (SHAP can be slow)
        if len(X) > 1000:
            idx = np.random.RandomState(42).choice(len(X), 1000, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        importance_array, method = _get_importance(model, X_sample, feature_names)

        importance_dict = {
            feature_names[i]: round(float(importance_array[i]), 4)
            for i in range(len(feature_names))
        }
        importance_sorted = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        log.info("Feature importance ranking (method=%s):", method)
        for rank, (feat, imp) in enumerate(importance_sorted.items(), 1):
            log.info("  %d. %s: %.4f", rank, feat, imp)

        # Generate summary bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        features_sorted = list(importance_sorted.keys())
        values_sorted = list(importance_sorted.values())
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features_sorted)))

        ax.barh(range(len(features_sorted)), values_sorted[::-1], color=colors[::-1])
        ax.set_yticks(range(len(features_sorted)))
        ax.set_yticklabels(features_sorted[::-1])
        ax.set_xlabel("Mean |SHAP value|" if method == "shap" else "Feature Importance")
        ax.set_title("Feature Importance (SHAP)" if method == "shap" else "Feature Importance (XGBoost)")
        plt.tight_layout()

        bar_path = PLOTS_DIR / "shap_importance_bar.png"
        fig.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved importance bar plot → %s", bar_path)

        # Generate SHAP beeswarm plot (only available when using SHAP)
        beeswarm_path = None
        if method == "shap":
            try:
                import shap as shap_mod
                explainer = shap_mod.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                plt.figure(figsize=(12, 8))
                shap_mod.summary_plot(
                    shap_values, X_sample,
                    feature_names=feature_names,
                    show=False,
                )
                beeswarm_path = PLOTS_DIR / "shap_beeswarm.png"
                plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
                plt.close()
                log.info("Saved SHAP beeswarm plot → %s", beeswarm_path)
            except Exception as e:
                log.warning("Could not generate beeswarm plot: %s", e)
                beeswarm_path = None

        # Save report
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "method": method,
            "n_samples_analyzed": len(X_sample),
            "n_features": len(feature_names),
            "feature_importance": importance_sorted,
            "plots": {
                "bar_plot": str(bar_path),
                "beeswarm_plot": str(beeswarm_path) if beeswarm_path else None,
            },
        }

        SHAP_REPORT.write_text(json.dumps(report, indent=2))
        log.info("SHAP report → %s", SHAP_REPORT)

        # Log to MLflow
        with mlflow.start_run(run_name="sensitivity_analysis"):
            for feat, imp in importance_sorted.items():
                mlflow.log_metric(f"shap_{feat}", imp)
            mlflow.log_artifact(str(bar_path))
            mlflow.log_artifact(str(SHAP_REPORT))
            if beeswarm_path:
                mlflow.log_artifact(str(beeswarm_path))

        log.info("✅ Sensitivity analysis complete")
        return report

    except ImportError:
        log.warning("SHAP not installed. Falling back to sklearn feature_importances_.")

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            importance_dict = {
                feature_names[i]: round(float(importance[i]), 4)
                for i in range(len(feature_names))
            }
            importance_sorted = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

            report = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "method": "sklearn_feature_importances",
                "n_features": len(feature_names),
                "feature_importance": importance_sorted,
            }
            SHAP_REPORT.write_text(json.dumps(report, indent=2))
            log.info("Feature importance report (sklearn) → %s", SHAP_REPORT)
            return report
        else:
            log.error("Model has no feature_importances_ attribute")
            return None


if __name__ == "__main__":
    main()