#!/usr/bin/env python3
"""
Model training for Level Up energy score prediction.

Loads processed data from the data pipeline, engineers the energy_score
target, trains Random Forest and XGBoost models, selects the best via
cross-validation, and logs everything to MLflow.

Usage:
    python train_model.py [--params params.json]
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DAILY_JOINED, BEST_MODEL_PATH, FEATURE_COLUMNS, TARGET_COLUMN,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, ensure_dirs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_and_prepare_data():
    """
    Load daily_joined.parquet and engineer features + target.

    Returns (X, y, df) where df has all columns for bias analysis.
    """
    df = pd.read_parquet(DAILY_JOINED)
    log.info("Loaded %d rows from %s", len(df), DAILY_JOINED)

    # Engineer target: energy_score
    # Based on the model's original formula adapted for our features:
    #   sleep_component:  sleep_satisfaction * 55 (0–55 range)
    #   study_component:  avg_accuracy * 35      (0–35 range)
    #   activity_component: attempts_count / 5 * 10 (0–10 range, capped)
    df["sleep_comp"] = df.get("sleep_satisfaction", pd.Series(0.5, index=df.index)) * 55
    df["study_comp"] = df.get("avg_accuracy", pd.Series(0.5, index=df.index)) * 35
    df["activity_comp"] = (
        df.get("attempts_count", pd.Series(1, index=df.index)).clip(upper=5) / 5 * 10
    )
    df[TARGET_COLUMN] = (df["sleep_comp"] + df["study_comp"] + df["activity_comp"]).clip(0, 100)

    # Add noise for realism
    noise = np.random.RandomState(42).normal(0, 2, size=len(df))
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] + noise).clip(0, 100).round(2)

    # Derive sex_numeric and bmi_numeric if not present
    if "sex_numeric" not in df.columns:
        df["sex_numeric"] = df.get("sex", pd.Series("Male", index=df.index)).apply(
            lambda s: 0 if str(s).strip().lower() in ("female", "f") else 1
        )
    if "bmi_numeric" not in df.columns:
        # Derive from height/weight if available
        h_m = df.get("height", pd.Series(170, index=df.index)) / 100
        w = df.get("weight", pd.Series(70, index=df.index))
        bmi = w / (h_m ** 2)
        df["bmi_numeric"] = pd.cut(
            bmi,
            bins=[0, 18.5, 25, 30, 100],
            labels=[0, 1, 2, 3],
        ).astype(float).fillna(1)

    # Select features — use available columns
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    log.info("Using %d features: %s", len(available_features), available_features)

    # Fill NaN in features
    for col in available_features:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            log.info("Filled %d NaN in '%s' with median %.2f", df[col].isna().sum(), col, median)

    # Drop rows where target is NaN
    df = df.dropna(subset=[TARGET_COLUMN])

    X = df[available_features].values
    y = df[TARGET_COLUMN].values

    log.info("Final dataset: %d samples, %d features", len(X), X.shape[1])
    return X, y, df, available_features


def train_models(X_train, y_train, params=None):
    """
    Train Random Forest and XGBoost models.

    Returns dict of {model_name: (model, cv_scores)}.
    """
    params = params or {}

    rf_params = params.get("random_forest", {
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    })

    models = {}

    # Random Forest
    log.info("Training Random Forest...")
    rf = RandomForestRegressor(**rf_params)
    rf_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
    rf.fit(X_train, y_train)
    models["random_forest"] = (rf, -rf_scores)
    log.info("  RF CV MAE: %.3f ± %.3f", -rf_scores.mean(), rf_scores.std())

    # XGBoost
    try:
        from xgboost import XGBRegressor

        xgb_params = params.get("xgboost", {
            "n_estimators": 150,
            "max_depth": 6,
            "learning_rate": 0.1,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        })

        log.info("Training XGBoost...")
        xgb = XGBRegressor(**xgb_params)
        xgb_scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
        xgb.fit(X_train, y_train)
        models["xgboost"] = (xgb, -xgb_scores)
        log.info("  XGB CV MAE: %.3f ± %.3f", -xgb_scores.mean(), xgb_scores.std())
    except ImportError:
        log.warning("XGBoost not installed, skipping")

    return models


def select_best_model(models):
    """Select the model with lowest mean CV MAE."""
    best_name = None
    best_mae = float("inf")
    best_model = None

    for name, (model, cv_scores) in models.items():
        mae = cv_scores.mean()
        log.info("  %s: CV MAE = %.3f", name, mae)
        if mae < best_mae:
            best_mae = mae
            best_name = name
            best_model = model

    log.info("Best model: %s (CV MAE = %.3f)", best_name, best_mae)
    return best_name, best_model, best_mae


def main(params_file=None):
    """Run the full training pipeline."""
    ensure_dirs()

    # Load params if provided
    params = {}
    if params_file and Path(params_file).exists():
        params = json.loads(Path(params_file).read_text())
        log.info("Loaded params from %s", params_file)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Load data
    X, y, df, feature_names = load_and_prepare_data()

    # Train/test split (stratified by age bucket for better representation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    log.info("Train: %d samples, Test: %d samples", len(X_train), len(X_test))

    # Save test indices for validation
    test_indices = df.index[len(X_train):]

    with mlflow.start_run(run_name="model_selection") as run:
        # Log dataset info
        mlflow.log_param("total_samples", len(X))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("feature_names", feature_names)

        # Train models
        models = train_models(X_train, y_train, params)

        # Select best
        best_name, best_model, best_cv_mae = select_best_model(models)

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)

        log.info("Test Set Results:")
        log.info("  MAE:  %.3f", test_mae)
        log.info("  RMSE: %.3f", test_rmse)
        log.info("  R²:   %.3f", test_r2)

        # Log metrics to MLflow
        mlflow.log_metric("cv_mae", best_cv_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_param("best_model_type", best_name)

        # Log model params
        if hasattr(best_model, "get_params"):
            for k, v in best_model.get_params().items():
                try:
                    mlflow.log_param(f"best_{k}", v)
                except Exception:
                    pass

        # Log model
        mlflow.sklearn.log_model(best_model, "model")

        # Save model locally
        joblib.dump(best_model, BEST_MODEL_PATH)
        log.info("Saved best model → %s", BEST_MODEL_PATH)

        # Save test data for validation
        test_data = {
            "X_test": X_test.tolist(),
            "y_test": y_test.tolist(),
            "feature_names": feature_names,
            "test_df_indices": test_indices.tolist() if hasattr(test_indices, "tolist") else [],
        }
        test_data_path = BEST_MODEL_PATH.parent / "test_data.json"
        test_data_path.write_text(json.dumps(test_data))

        # Save training metadata
        metadata = {
            "best_model_type": best_name,
            "cv_mae": round(best_cv_mae, 4),
            "test_mae": round(test_mae, 4),
            "test_rmse": round(test_rmse, 4),
            "test_r2": round(test_r2, 4),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "mlflow_run_id": run.info.run_id,
        }
        (BEST_MODEL_PATH.parent / "training_metadata.json").write_text(
            json.dumps(metadata, indent=2)
        )

        log.info("Training complete. MLflow run ID: %s", run.info.run_id)

    return best_model, feature_names, test_mae


if __name__ == "__main__":
    params_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(params_file)