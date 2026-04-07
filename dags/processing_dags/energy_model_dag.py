"""
dags/processing_dags/energy_model_dag.py

End-to-end energy score model pipeline orchestrated by Airflow.

Chain:
    build_features
        → run_training        (RandomForest / XGBoost, MLflow tracking)
        → validate_model      (MAE + R² gates)
        → bias_detection      (sex + age slices)
        → shap_analysis       (feature importance)
        → export_model        (model_weights.json for the app)
        → push_to_registry    (GCS model_registry/energy/)

Airflow Variables:
    ENERGY_MAE_THRESHOLD  — float, default 5.0
    ENERGY_R2_THRESHOLD   — float, default 0.7
    GCS_BACKUP_BUCKET     — GCS bucket name, default raw_data_lvlup

Input:
    /opt/airflow/data/raw/sleep_health/Sleep_health_and_lifestyle_dataset.csv

Outputs:
    /opt/airflow/data/processed/daily_joined.parquet
    /opt/airflow/data/models/energy/best_model.joblib
    /opt/airflow/data/models/energy/model_weights.json
    /opt/airflow/data/models/energy/reports/
    /opt/airflow/data/models/energy/plots/
"""

import importlib.util
import sys
from pathlib import Path

import pendulum
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models import Variable

from dag_monitoring import (
    monitored_dag_args,
    on_dag_failure_callback,
    on_sla_miss_callback,
)

#  Paths 
RAW_CSV    = Path("/opt/airflow/data/raw/sleep_health/Sleep_health_and_lifestyle_dataset.csv")
FEAT_PATH  = Path("/opt/airflow/data/processed/daily_joined.parquet")
SCRIPTS    = Path("/opt/airflow/scripts/energy")

MAE_THRESHOLD = float(Variable.get("ENERGY_MAE_THRESHOLD", default_var="5.0"))
R2_THRESHOLD  = float(Variable.get("ENERGY_R2_THRESHOLD",  default_var="0.7"))
GCS_BUCKET    = Variable.get("GCS_BACKUP_BUCKET",          default_var="raw_data_lvlup")


def _load(script_name: str):
    """Dynamically load a script from scripts/energy/ as a module."""
    path = SCRIPTS / script_name
    if not path.exists():
        raise AirflowFailException(f"Script not found: {path}")
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), str(path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#  DAG 
with DAG(
    dag_id="energy_model_dag",
    start_date=pendulum.datetime(2026, 2, 1, tz="America/New_York"),
    schedule=None,
    catchup=False,
    default_args=monitored_dag_args(retries=1, sla_minutes=90),
    on_failure_callback=on_dag_failure_callback,
    sla_miss_callback=on_sla_miss_callback,
    tags=["energy", "ml", "training"],
) as dag:

    @task()
    def build_features():
        """
        Engineer daily_joined.parquet from the raw Kaggle CSV.

        The Sleep Health dataset has one row per person. We expand it to
        look like the synthetic daily_joined schema that the energy model
        expects (sleep_hours, sleep_satisfaction, avg_accuracy, int_score,
        rolling averages, BMR, etc.) so that train_model.py works unchanged.
        """
        import numpy as np
        import pandas as pd

        if not RAW_CSV.exists():
            raise AirflowFailException(
                f"Raw CSV not found: {RAW_CSV}\n"
                "Run kaggle_download_sleep_health first."
            )

        FEAT_PATH.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(RAW_CSV)
        print(f"Loaded {len(df)} rows from {RAW_CSV}")

        rng = np.random.default_rng(42)

        #  Map Kaggle columns → pipeline feature names 
        out = pd.DataFrame()

        out["user_id"]   = df["Person ID"].astype(str)
        out["age"]       = df["Age"].astype(float)
        out["sex"]       = df["Gender"]

        # Sleep Duration (hrs) → sleep_hours
        out["sleep_hours"] = pd.to_numeric(df["Sleep Duration"], errors="coerce").clip(2, 14)

        # Quality of Sleep (1–10) → sleep_satisfaction (0–1)
        out["sleep_satisfaction"] = (
            pd.to_numeric(df["Quality of Sleep"], errors="coerce").clip(1, 10) - 1
        ) / 9.0

        # Rolling sleep avg — add small noise for variation
        out["rolling_sleep_hours_7d"] = (
            out["sleep_hours"] + rng.normal(0, 0.3, len(df))
        ).clip(2, 14)

        # Bedtime variability — proxy from stress level if available
        if "Stress Level" in df.columns:
            out["bedtime_variability_7d"] = (
                pd.to_numeric(df["Stress Level"], errors="coerce").fillna(5) * 12
            ).clip(5, 120)
        else:
            out["bedtime_variability_7d"] = rng.uniform(10, 90, len(df))

        # Physical Activity Level → attempts_count proxy (0–5 range)
        out["attempts_count"] = (
            pd.to_numeric(df["Physical Activity Level"], errors="coerce")
            .fillna(50).clip(0, 100) / 20.0
        ).round(1)

        # avg_accuracy — proxy from Quality of Sleep normalised
        out["avg_accuracy"] = out["sleep_satisfaction"].clip(0.3, 1.0)

        # int_score — proxy: accuracy * 75 + (attempts_count/5)*25
        out["int_score"] = (
            out["avg_accuracy"] * 75 + (out["attempts_count"] / 5.0) * 25
        ).clip(0, 100)

        out["rolling_int_7d"] = (
            out["int_score"] + rng.normal(0, 3, len(df))
        ).clip(0, 100)

        # BMI Category → bmi_numeric (0=underweight,1=normal,2=overweight,3=obese)
        bmi_map = {"Underweight": 0, "Normal": 1, "Normal Weight": 1,
                   "Overweight": 2, "Obese": 3}
        out["bmi_numeric"] = (
            df["BMI Category"].map(bmi_map).fillna(1).astype(float)
        )

        # BMR — Mifflin-St Jeor approximation (height/weight not in dataset,
        # use age + sex proxy with reasonable defaults)
        def _bmr(row):
            age = row["age"]
            if str(row["sex"]).strip().lower() in ("male", "m"):
                return round(10 * 70 + 6.25 * 175 - 5 * age + 5)
            return round(10 * 60 + 6.25 * 163 - 5 * age - 161)

        out["bmr"] = out.apply(_bmr, axis=1).astype(float)

        # sex_numeric
        out["sex_numeric"] = out["sex"].apply(
            lambda s: 0 if str(s).strip().lower() in ("female", "f") else 1
        ).astype(float)

        # Date — assign synthetic dates for time-series compatibility
        base_date = pd.Timestamp("2025-01-01")
        out["date"] = [base_date + pd.Timedelta(days=i) for i in range(len(out))]

        # Drop rows with any NaN in key columns
        key_cols = ["sleep_hours", "sleep_satisfaction", "avg_accuracy",
                    "int_score", "bmr", "age"]
        before = len(out)
        out    = out.dropna(subset=key_cols)
        print(f"Dropped {before - len(out)} rows with NaN in key columns")

        out.to_parquet(str(FEAT_PATH), index=False)
        print(f"Saved {len(out)} rows → {FEAT_PATH}")
        print(f"Columns: {list(out.columns)}")

        return str(FEAT_PATH)

    @task()
    def run_training(feat_path: str):
        """Train RandomForest + XGBoost, select best by CV MAE, log to MLflow."""
        import os
        os.environ["ENERGY_FEATURES_PATH"] = feat_path

        sys.path.insert(0, str(SCRIPTS))
        mod     = _load("train_model.py")
        _, _, test_mae = mod.main()

        print(f"Training complete — test MAE: {test_mae:.4f}")
        return test_mae

    @task()
    def validate_model(test_mae: float):
        """Validate model against MAE and R² gates."""
        import json

        sys.path.insert(0, str(SCRIPTS))
        mod = _load("validate_model.py")

        try:
            report = mod.main()
        except SystemExit as e:
            if e.code != 0:
                raise AirflowFailException(
                    f"Energy model validation FAILED (MAE threshold={MAE_THRESHOLD}, "
                    f"R² threshold={R2_THRESHOLD})"
                )
            report = {}

        status = report.get("status", "UNKNOWN") if report else "UNKNOWN"
        print(f"Validation status: {status}")

        if status != "PASS":
            raise AirflowFailException(f"Validation gate FAILED: {status}")

        return status

    @task()
    def bias_detection(validation_status: str):
        """Slice model performance by sex and age bucket."""
        sys.path.insert(0, str(SCRIPTS))
        mod    = _load("model_bias_detection.py")
        report = mod.main()

        flagged = report.get("flagged_slices", []) if report else []

        if flagged:
            print(f"WARNING: Bias detected in {len(flagged)} slice(s): {flagged}")
        else:
            print("No significant bias detected.")

        return len(flagged)

    @task()
    def shap_analysis(n_flagged: int):
        """Compute SHAP feature importance and generate plots."""
        sys.path.insert(0, str(SCRIPTS))
        mod    = _load("sensitivity_analysis.py")
        report = mod.main()

        if report and "feature_importance" in report:
            top3 = list(report["feature_importance"].keys())[:3]
            print(f"Top 3 features: {top3}")

        return True

    @task()
    def export_model(shap_done: bool):
        """Export trained model to model_weights.json for the app."""
        sys.path.insert(0, str(SCRIPTS))
        mod = _load("export_model.py")
        mod.main()
        print("Model exported to model_weights.json")
        return True

    @task()
    def push_to_registry(exported: bool):
        """Push model package to GCS model_registry/energy/."""
        import os
        os.environ["GCS_BACKUP_BUCKET"] = GCS_BUCKET

        sys.path.insert(0, str(SCRIPTS))
        mod      = _load("registry_push.py")
        location = mod.main()

        print(f"Pushed to registry: {location}")

    #  Wire up the chain 
    feat       = build_features()
    train_mae  = run_training(feat)
    val_status = validate_model(train_mae)
    n_flagged  = bias_detection(val_status)
    shap_done  = shap_analysis(n_flagged)
    exported   = export_model(shap_done)
    push_to_registry(exported)