#!/usr/bin/env python3
"""
Hyperparameter tuning via RandomizedSearchCV.

Tunes Random Forest and XGBoost, logs all trials to MLflow,
and outputs the best parameters as JSON for the training script.

Usage:
    python hyperparameter_tuning.py
"""

import json
import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DAILY_JOINED, MODELS_DIR, MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME, ensure_dirs,
)
from train_model import load_and_prepare_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Parameter grids ──

RF_PARAM_GRID = {
    "n_estimators": randint(50, 300),
    "max_depth": randint(4, 15),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", None],
}

XGB_PARAM_GRID = {
    "n_estimators": randint(50, 300),
    "max_depth": randint(3, 12),
    "learning_rate": uniform(0.01, 0.3),
    "min_child_weight": randint(1, 10),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "gamma": uniform(0, 5),
}


def tune_model(model_class, param_grid, X, y, model_name, n_iter=30):
    """Run RandomizedSearchCV and log results to MLflow."""
    log.info("Tuning %s (%d iterations)...", model_name, n_iter)

    search = RandomizedSearchCV(
        model_class(random_state=42, n_jobs=-1),
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)

    best_score = -search.best_score_
    log.info("  Best %s CV MAE: %.3f", model_name, best_score)
    log.info("  Best params: %s", search.best_params_)

    # Log to MLflow
    with mlflow.start_run(run_name=f"tune_{model_name}", nested=True):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_search_iterations", n_iter)
        mlflow.log_metric("best_cv_mae", best_score)
        for k, v in search.best_params_.items():
            try:
                mlflow.log_param(f"best_{k}", v)
            except Exception:
                pass

    return search.best_params_, best_score


def main():
    """Run hyperparameter tuning for all model types."""
    ensure_dirs()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X, y, _, feature_names = load_and_prepare_data()

    results = {}

    with mlflow.start_run(run_name="hyperparameter_tuning"):
        # Random Forest
        rf_params, rf_score = tune_model(
            RandomForestRegressor, RF_PARAM_GRID, X, y,
            "random_forest", n_iter=30,
        )
        # Add fixed params
        rf_params["random_state"] = 42
        rf_params["n_jobs"] = -1
        results["random_forest"] = rf_params

        # XGBoost
        try:
            from xgboost import XGBRegressor
            xgb_params, xgb_score = tune_model(
                XGBRegressor, XGB_PARAM_GRID, X, y,
                "xgboost", n_iter=30,
            )
            xgb_params["random_state"] = 42
            xgb_params["n_jobs"] = -1
            results["xgboost"] = xgb_params
        except ImportError:
            log.warning("XGBoost not installed, skipping tuning")

        mlflow.log_param("models_tuned", list(results.keys()))

    # Save best params
    output_path = MODELS_DIR / "best_params.json"

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = {
        k: {pk: convert(pv) for pk, pv in v.items()}
        for k, v in results.items()
    }
    output_path.write_text(json.dumps(serializable, indent=2))
    log.info("Best hyperparameters saved → %s", output_path)

    return results


if __name__ == "__main__":
    main()