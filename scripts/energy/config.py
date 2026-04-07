"""
Centralized configuration for the Level Up Energy Score ML Pipeline.

All energy model scripts import paths and constants from here.
Paths are relative to the pipeline repo root so this works both
locally (inside Docker) and in GitHub Actions CI.
"""

import os
from pathlib import Path

#  Repo root (two levels up from scripts/energy/) 
SCRIPTS_ENERGY_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR        = SCRIPTS_ENERGY_DIR.parent
REPO_ROOT          = SCRIPTS_DIR.parent

#  Input data — produced by the synthetic + Firestore DAGs 
# In Airflow: /opt/airflow/data/processed/daily_joined.parquet
# In CI:      data/processed/daily_joined.parquet (generated synthetically)
DAILY_JOINED = Path(
    os.environ.get("ENERGY_FEATURES_PATH",
                   str(REPO_ROOT / "data" / "processed" / "daily_joined.parquet"))
)

#  Energy model outputs 
MODELS_DIR  = REPO_ROOT / "data" / "models" / "energy"
REPORTS_DIR = MODELS_DIR / "reports"
PLOTS_DIR   = MODELS_DIR / "plots"

#  Model files 
BEST_MODEL_PATH      = MODELS_DIR / "best_model.joblib"
MODEL_WEIGHTS_JSON   = MODELS_DIR / "model_weights.json"
VALIDATION_REPORT    = REPORTS_DIR / "validation_report.json"
BIAS_DETECTION_REPORT = REPORTS_DIR / "bias_detection_report.json"
SHAP_REPORT          = REPORTS_DIR / "shap_feature_importance.json"

#  MLflow 
MLFLOW_TRACKING_URI    = os.environ.get("MLFLOW_TRACKING_URI",
                                        "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME = "levelup-energy-prediction"

#  Validation thresholds 
MAX_ACCEPTABLE_MAE = float(os.environ.get("ENERGY_MAE_THRESHOLD", "5.0"))
MIN_ACCEPTABLE_R2  = float(os.environ.get("ENERGY_R2_THRESHOLD", "0.7"))

#  Feature configuration 
FEATURE_COLUMNS = [
    "age", "sex_numeric", "bmi_numeric", "sleep_hours",
    "sleep_satisfaction", "rolling_sleep_hours_7d",
    "bedtime_variability_7d", "avg_accuracy",
    "int_score", "rolling_int_7d", "bmr",
    "attempts_count",
]
TARGET_COLUMN = "energy_score"

#  Age buckets for bias slicing (matches flexibility model) 
AGE_BINS = [(0, 19, "<20"), (20, 29, "20-29"), (30, 39, "30-39"), (40, 200, "40+")]

#  GCS registry (same bucket as flexibility model, different prefix) 
GCS_BUCKET         = os.environ.get("GCS_BACKUP_BUCKET", "raw_data_lvlup")
GCS_REGISTRY_PREFIX = "model_registry/energy"


def ensure_dirs():
    """Create all output directories."""
    for d in [MODELS_DIR, REPORTS_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)