# Level Up — MLOps Full Pipeline

End-to-end MLOps pipeline for a lifestyle gamification app. Covers data ingestion, cleaning, validation, feature engineering, model training, bias detection, experiment tracking, CI/CD automation, and alerting — orchestrated with Apache Airflow, versioned with DVC, and tracked with MLflow.

## Requirements

- Python 3.11+
- Docker + Docker Compose
- Kaggle API credentials (for dataset downloads)
- GCP service account (for Firestore + GCS)
- Slack webhook URL (for alerts)

## Setup

### 1. Clone and configure environment

Create a `.env` file (do not commit):

```bash
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
AIRFLOW_UID=1000
```

### 2. Add secrets

```bash
mkdir -p secrets
cp /path/to/your/gcp-service-account.json secrets/gcp-sa.json
cp /path/to/your/firebase-admin.json secrets/firebase-admin.json
```

### 3. Start all services

```bash
docker compose up -d
docker compose ps
```

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow UI | http://localhost:8081 | admin / admin |
| MLflow UI | http://localhost:5001 | — |

### 4. Initialize DVC

```bash
dvc init
dvc remote add -d gcs_remote gs://your-bucket/dvc-store
dvc remote modify gcs_remote credentialpath secrets/gcp-sa.json

dvc add data/raw
dvc add data/processed

git add dvc.yaml dvc.lock data/raw.dvc data/processed.dvc .dvc/ .gitignore
git commit -m "feat: initialize DVC tracking"
dvc push
```

### 5. Seed Firestore and run the flexibility pipeline

```bash
# Seed 20 users, 90 days of workout data
docker exec airflow-scheduler python /opt/airflow/data_seeding/main.py \
  --service-account /opt/airflow/secrets/firebase-admin.json \
  --num-users 20 --days 90 --write-rollups --seed 42

# Run feature engineering (downloads from Firestore, builds lag features)
docker exec airflow-scheduler airflow dags trigger flexibility_features

# Model DAG triggers automatically on success, or trigger manually
docker exec airflow-scheduler airflow dags trigger flexibility_model
```

### 6. Run Tests

```bash
pip install pytest pytest-cov
pytest
pytest --cov=tests --cov-report=term-missing
```

---

## Reproducing on Another Machine

### Prerequisites

- Docker and Docker Compose installed
- Git installed
- GCP service account JSON with Firestore + GCS access
- Kaggle account (for dataset downloads)
- Python 3.11+ (only needed for running tests locally outside Docker)

### Step-by-step

```bash
# 1. Clone the repo
git clone <repo-url>
cd <repo>

# 2. Create .env
cat > .env << EOF
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
AIRFLOW_UID=1000
EOF

# 3. Add service account keys
mkdir -p secrets
cp /path/to/gcp-sa.json secrets/gcp-sa.json
cp /path/to/firebase-admin.json secrets/firebase-admin.json

# 4. Restore data from DVC
pip install "dvc[gs]"
dvc pull

# 5. Start all services
docker compose up -d

# 6. Verify
docker compose ps
docker exec -it airflow-scheduler airflow dags list

# 7. Run any pipeline
docker exec -it airflow-scheduler airflow dags trigger download_food_data
docker exec -it airflow-scheduler airflow dags trigger download_wisdm_accel
docker exec -it airflow-scheduler airflow dags trigger download_synthetic_from_firestore

# 8. Run tests
pip install -r requirements.txt
pytest
```

### If you don't have GCP credentials

```bash
docker compose up -d

# Kaggle datasets (requires Kaggle credentials only)
docker exec -it airflow-scheduler airflow dags trigger kaggle_download_strength
docker exec -it airflow-scheduler airflow dags trigger clean_weightlifting_data

# Public datasets (no credentials needed)
docker exec -it airflow-scheduler airflow dags trigger download_wisdm_accel
docker exec -it airflow-scheduler airflow dags trigger clean_wisdm_accel_data
docker exec -it airflow-scheduler airflow dags trigger download_food_data
```

---

## Pipeline Overview

![Pipeline Architecture](pipeline.png)

### DAG Dependency Flow

```
Download DAGs (manual trigger)
├── download_wisdm_accel                → data/raw/wisdm/
├── kaggle_download_strength            → data/raw/strength/
├── kaggle_download_flexibility         → data/raw/flexibility/
├── download_food_data                  → data/raw/food-101/
│       ↓ triggers
│   clean_food_data (processing_dags/)
│     manifest → splits → distribution + inference → quality gate
│       ↓ triggers
│   food_bias_monitoring (monitoring_dags/)
│     class balance → split skew → prediction bias → slack
│       ↓ triggers
│   dvc_backup_to_gcp
└── download_synthetic_from_firestore   → data/raw/*.json (profiles, sleep, quiz)
        │                                    ↓ triggers
        │                               process_synthetic_data (processing_dags/)
        │                                 validate → sleep + quiz (parallel) → join → stats
        │                                    ↓ triggers
        │                               synthetic_anomaly_and_bias (monitoring_dags/)
        │                                 anomaly detection → bias analysis → slack
        ▼                                    ↓ triggers
Processing DAGs (manual or triggered)   dvc_backup_to_gcp
├── clean_weightlifting_data       → data/processed/weightlifting_cleaned/
│   └── triggers: dvc_backup_to_gcp
├── clean_wisdm_accel_data         → data/processed/wisdm/
│   └── triggers: dvc_backup_to_gcp
├── firestore_schema_validation    (daily 02:00 ET)
│   └── triggers: firestore_metric_events_to_gcs → GCS JSONL partitioned by day/metric
├── flexibility_features           → data/processed/flexibility_features.parquet
│   └── triggers: flexibility_model
└── flexibility_model              → data/models/flexibility/
    run_training → validate_model → rollback_check → push_to_registry
        ↓ triggers
    dvc_backup_to_gcp

Backup DAGs
├── dvc_backup_to_gcp              → DVC add/push raw + processed + models, git commit
└── firestore_metric_events_to_gcs → GCS JSONL partitioned by day/metric

Monitoring DAGs (daily 06:00 ET)
├── daily_bias_monitoring          → Slack report (WISDM + weightlifting)
└── food_bias_monitoring           → Slack report (Food-101 class/prediction bias)
```

---

## Flexibility Model Pipeline

The flexibility model predicts a user's flexibility score at +1, +3, +7, and +14 days ahead, given their last 5 workout sessions. This is the primary ML deliverable of the project.

### End-to-end flow

```
data_seeding/main.py
  Seeds Firestore with per-session workout data
  (exercise_type, duration, effort, sit_and_reach_cm, score_before/after, streak)
        ↓
flexibility_features DAG
  download_workout_sessions   pulls flexibility_workouts subcollection from Firestore
  build_lag_features          engineers 5-session lag features + aggregate features per user
  join_profiles               joins age, sex, BMR from user profiles
  quality_gate                enforces minimum row count and target coverage
        ↓ triggers
flexibility_model DAG
  run_training                trains Ridge, Random Forest, XGBoost; selects winner by d7 RMSE
  validate_model              enforces RMSE gate + Fairlearn bias check (sex, age group)
  rollback_check              blocks deploy if new model > 10% worse than previous
  push_to_registry            uploads model + metrics + plots to GCS model registry
        ↓ triggers
dvc_backup_to_gcp
```

### Feature design

Each training row represents a user at reference session `t`. Features are derived from the 5 sessions immediately before `t`.

**Lag features (per session, 5 lags):**
- `score_lag_1` .. `score_lag_5` — flexibility score at that session
- `effort_lag_1` .. `effort_lag_5` — effort level (1–5)
- `duration_lag_1` .. `duration_lag_5` — session duration in minutes
- `reach_lag_1` .. `reach_lag_5` — sit-and-reach measurement (cm)
- `days_ago_lag_1` .. `days_ago_lag_5` — days before the reference session

**Aggregate features:**
- `workout_count_7d`, `workout_count_14d` — session frequency
- `mean_score_5`, `score_trend_5` — recent score level and trajectory
- `mean_effort_5` — recent effort consistency
- `days_since_last`, `current_streak` — rest and consistency indicators

**Profile features:**
- `age`, `sex_encoded`, `bmr`, `age_bucket_enc`

**Targets:**
- `target_d1`, `target_d3`, `target_d7`, `target_d14` — score N days after `t`

### Model training

Three architectures are trained and compared on every run:

| Model | Notes |
|-------|-------|
| Ridge regression | Linear baseline, fast |
| Random Forest | 200 estimators, max_depth 8 |
| XGBoost | RandomizedSearchCV, 20 iterations × 3-fold CV |

The winner is selected by lowest d7 RMSE on the time-based test set (last 20% of dates). All three results are logged to MLflow for comparison.

**Gate:** deployment is blocked if d7 RMSE exceeds `MODEL_RMSE_THRESHOLD` (default 10.0, configurable via Airflow Variable).

### Experiment tracking (MLflow)

MLflow runs at http://localhost:5001. Each training run logs:
- Hyperparameters (best params from RandomizedSearchCV)
- Per-horizon metrics (RMSE, MAE, R²) for all three models
- SHAP feature importance (top 10, d7 horizon)
- Hyperparameter sensitivity (CV correlation analysis)
- Bias report (Fairlearn per-slice RMSE)
- Model artifact registered as `flexibility_score_forecaster`

### Model outputs

After every successful training run, the following are written to `data/models/flexibility/` and uploaded to GCS:

```
data/models/flexibility/
├── model.pkl                        winning model (MultiOutputRegressor)
├── metrics.json                     full metrics, model comparison, SHAP, sensitivity
├── bias_report.json                 Fairlearn per-slice RMSE (sex, age group)
├── shap_summary.png                 SHAP beeswarm plot (d7 horizon)
└── plots/
    ├── 01_horizon_rmse_comparison.png   RMSE per horizon, all models
    ├── 02_model_selection.png           RMSE + R² at d7, winner highlighted
    ├── 03_shap_top10.png                top-10 feature importances
    ├── 04_bias_sex.png                  per-sex RMSE vs overall
    ├── 05_bias_age.png                  per-age-group RMSE vs overall
    ├── 06_hyperparam_sensitivity.png    hyperparameter sensitivity
    └── 07_score_distribution.png       target score distributions per horizon
```

### Bias detection

Fairlearn slices the test set by `sex` and `age_bucket` (<20, 20–29, 30–39, 40+) and evaluates d7 RMSE per group. Groups with RMSE >50% above overall are flagged; groups >100% above block deployment.

Mitigation strategies documented in `bias_report.json`:
- Collect more data from underrepresented groups
- Apply inverse-frequency sample weighting in XGBoost
- Use stratified CV splits by demographic group

### Rollback mechanism

Before every registry push, `rollback_check` fetches `latest.json` from GCS and compares the new model's d7 RMSE against the previously deployed model. Deployment is blocked if the new model is more than 10% worse.

---

## CI/CD

GitHub Actions workflow at `.github/workflows/flexibility_model_ci.yml` triggers on any push to `main` or `model/*` branches that touches model or feature code.

### Jobs

| Job | Trigger | What it does |
|-----|---------|--------------|
| `test` | every push | Lint (ruff) + pytest |
| `train` | after test passes | Feature engineering, training, validation, bias check, artifact upload |
| `deploy` | main branch only | Rollback check, GCS registry push, DVC pointer commit |

### GitHub secrets required

| Secret | How to generate |
|--------|----------------|
| `GCP_SA_KEY` | `base64 -i secrets/gcp-sa.json \| tr -d '\n'` |
| `FIREBASE_SA_KEY` | `base64 -i secrets/firebase-admin.json \| tr -d '\n'` |
| `KAGGLE_USERNAME` | plain text |
| `KAGGLE_KEY` | plain text |
| `SLACK_WEBHOOK_URL` | plain text |

### GitHub variables required

| Variable | Value |
|----------|-------|
| `GCS_BACKUP_BUCKET` | `raw_data_lvlup` |
| `MODEL_RMSE_THRESHOLD` | `10.0` |

---

## Food-101 Pipeline

```
download_food_data
  download + extract Food-101 tarball (101 classes, ~101k images)
        ↓ triggers
clean_food_data
  build_manifest → create_splits (80/20 stratified) → class_distribution + mock_inference → quality_gate
        ↓ triggers both
food_bias_monitoring                    dvc_backup_to_gcp
  class_balance (imbalance ratio,
    under/over-represented ±2σ)
  split_skew (train vs val proportions)
  prediction_bias (confidence + accuracy per class)
  → Slack report
```

---

## Synthetic Data Pipeline (Firestore)

```
download_synthetic_from_firestore
  download_profiles → download_sleep_logs + download_quiz_attempts (parallel)
        ↓ triggers
process_synthetic_data
  validate_schemas → preprocess_sleep + preprocess_quiz (parallel) → build_features → generate_stats
        ↓ triggers
synthetic_anomaly_and_bias
  anomaly_detection → bias_analysis → slack_summary
        ↓ triggers
dvc_backup_to_gcp
```

**Download** (`download_synthetic_from_firestore`):
- Pulls user profiles (age, sex, height, weight) from `users/{uid}.profile`
- Pulls sleep logs from `users/{uid}/sleep_logs/*`
- Pulls quiz attempts from `users/{uid}/quiz_attempts/*`
- Writes `profiles_raw.json`, `sleep_logs_raw.json`, `quiz_attempts_raw.json` to `data/raw/`

**Processing** (`process_synthetic_data`):
- Validates raw JSON schemas (required fields, correct types)
- Preprocesses sleep: clamps hours to [2,14], parses bed/wake times, computes midpoint, satisfaction, rolling 3d/7d averages, bedtime variability
- Preprocesses quiz: computes accuracy, filters impossible values, daily aggregation, streak tracking, INT score (accuracy × 75 + streak bonus × 25), rolling 3d/7d averages
- Joins sleep + quiz + profiles with outer merge, adds BMR (Mifflin–St Jeor equation)
- Generates schema report and descriptive statistics

**Anomaly Detection & Bias** (`synthetic_anomaly_and_bias`):
- Missingness >20% in key fields
- Out-of-range: sleep outside [2,14]h, accuracy outside [0,1], time per question >300s
- Negative values in sleep_hours, attempts_count, int_score, bmr
- Bias slicing by sex and age bucket (<20, 20–29, 30–39, 40+)
- Sends Slack report with anomaly + bias summary

---

## Data Sources

| Source | Type | DAG | Raw Location |
|--------|------|-----|--------------|
| Kaggle weightlifting | CSV (9,932 rows, 10 cols) | `kaggle_download_strength` | `data/raw/strength/` |
| UCI WISDM | Accel txt (15.6M rows, 6 cols) | `download_wisdm_accel` | `data/raw/wisdm/` |
| ETH Food-101 | Images (101 classes, 101k images) | `download_food_data` | `data/raw/food-101/` |
| Firestore (synthetic) | JSON (profiles, sleep, quiz) | `download_synthetic_from_firestore` | `data/raw/*.json` |
| Firestore (flexibility) | NoSQL (flexibility_workouts) | `flexibility_features` | Firestore subcollection |
| Firestore (live) | NoSQL (metric_events) | `firestore_metric_events_to_gcs` | GCS bucket |

---

## Data Cleaning

**Weightlifting** (`clean_weightlifting_dag.py`):
- Schema validation (required columns: Date, Workout Name, Exercise Name, Set Order, Weight, Reps)
- Type coercion, deduplication, row-level validation (negative values, out-of-range weight >700kg, reps >200)
- Output: clean Parquet + rejection CSV; quality gate blocks if rejection rate >15%

**WISDM** (`clean_wisdm_dag.py`):
- Semicolon stripping, type coercion, deduplication
- Row validation: valid activity codes (A–S), missing axes, extreme magnitude (>200)
- 200-row windowing, sequential stamina computation, anomaly detection (5σ spikes)
- Output: clean Parquet, stamina Parquet, bias JSON

**Food-101** (`clean_food_dag.py`):
- Manifest from meta/train.txt + meta/test.txt, stratified 80/20 split
- Mock Gemini inference on val set; quality gate blocks if >20% images missing

**Firestore Live** (`firebase_schema_validation_dag.py`):
- Validates metric_events: metric ∈ {strength, stamina, speed, flexibility, intelligence}, score ∈ [0,100], confidence ∈ [0,1]
- Validates sleep_logs: sleepHours ∈ [0,24], quality ∈ [1,5], time formats
- Validates quiz_attempts: valid topics, num_correct ≤ num_questions, difficulty ∈ [1,5]
- Configurable error threshold gate before GCS export

---

## Bias Detection and Monitoring

**Flexibility model** (via `flexibility_model` DAG):
- Fairlearn slicing by sex and age bucket on d7 RMSE
- Groups >50% worse flagged; groups >100% worse block deployment
- Bias report + per-slice bar charts saved to `data/models/flexibility/plots/`

**WISDM / Stamina** (via `daily_bias_monitoring`):
- Stamina distribution across 18 activity types
- Per-user outlier detection (>2σ from global mean)
- Activity gap analysis

**Weightlifting** (via `daily_bias_monitoring`):
- Exercise volume concentration (top 5 as % of total)
- Weight progression trend and workout frequency by day of week

**Food-101** (via `food_bias_monitoring`):
- Class imbalance ratio, flags under/over-represented classes (±2σ)
- Train/val split skew, prediction confidence bias per class
- Alerts if imbalance ratio >3x, split skew >1%, or accuracy <0.8

**Synthetic / Firestore** (via `synthetic_anomaly_and_bias`):
- Slicing by sex and age bucket
- Per-slice: mean INT score, mean accuracy, mean sleep hours
- Flags slices with <10 samples

---

## Logging and Monitoring

All DAGs use `dag_monitoring.py` providing:
- Structured logging via `logging.getLogger("airflow.task")`
- Task callbacks: `on_failure_callback`, `on_success_callback`, `on_retry_callback`
- DAG callbacks: `on_dag_failure_callback`, SLA monitoring
- Slack webhook alerts on failures and SLA misses
- `emit_metric()` — writes structured JSONL to `/opt/airflow/logs/dag_metrics/`

| DAG | SLA Budget |
|-----|-----------|
| Kaggle downloads | 15 min |
| Food-101 download | 30 min |
| Food-101 cleaning | 30 min |
| Food bias monitoring | 15 min |
| Firestore synthetic download | 15 min |
| DVC backup | 20 min |
| Weightlifting cleaning | 30 min |
| WISDM cleaning | 30 min |
| Synthetic data processing | 30 min |
| Synthetic anomaly & bias | 20 min |
| Firestore validation | 45 min |
| Firestore export | 60 min |
| Daily bias monitoring | 20 min |
| Flexibility features | 30 min |
| Flexibility model | 90 min |

---

## Data Versioning

DVC tracks `data/raw/`, `data/processed/`, and `data/models/` with a GCS remote backend. The `dvc_backup_to_gcp` DAG automates:
1. `dvc add data/raw` and `dvc add data/processed` (in parallel)
2. `dvc push` to GCS
3. `git commit` of `.dvc` pointer files

The `dvc.yaml` file defines the full pipeline for local `dvc repro` execution outside of Airflow.

---

## Testing

~160 tests across 8 modules using pytest:

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_wisdm_loader.py` | 11 | File loading, semicolon parsing, malformed/empty files, multi-file concat |
| `test_weightlifting_cleaning.py` | 14 | Row validation (all issue types), schema check, deduplication |
| `test_stamina_and_anomaly.py` | 22 | Windowing, stamina monotonic decrease, anomaly detection, WISDM row validation |
| `test_schema_validation.py` | 20 | Firestore metric_events/sleep_logs/quiz_attempts, parametrized across all metrics |
| `test_synthetic_pipeline.py` | 33 | Time parsing, BMR, INT score, anomaly detection, age bucketing |
| `test_food_pipeline.py` | 30 | Manifest, class balance, split skew, mock inference, prediction bias, quality gate |
| `test_flexibility_features.py` | 16 | `_slope`, `_future_score`, `_bmr`, `_age_enc`, lag feature construction, days_ago ordering |
| `test_model_train.py` | 14 | Time split, feature cols, prepare, RMSE, sensitivity, full training smoke test, gate |

```bash
pytest                                        # all tests
pytest tests/test_flexibility_features.py     # single module
pytest tests/test_model_train.py -m "not slow" # skip slow smoke tests
pytest -k "test_stamina"                      # by name pattern
pytest --cov --cov-report=term-missing        # with coverage
```

---

## Project Structure

```
.
├── .github/workflows/
│   └── flexibility_model_ci.yml    CI/CD for flexibility model
├── dags/
│   ├── dag_monitoring.py           shared monitoring, callbacks, emit_metric
│   ├── backup_dags/                dvc_backup_to_gcp, firestore_metric_events_to_gcs
│   ├── download_dags/              kaggle, wisdm, food, synthetic, flexibility
│   ├── monitoring_dags/            daily_bias, food_bias, synthetic_anomaly_and_bias
│   └── processing_dags/            clean_*, firebase_schema_validation,
│                                   flexibility_features, flexibility_model
├── data_seeding/
│   └── main.py                     seeds Firestore with synthetic workout data
├── scripts/
│   ├── model_train.py              trains Ridge/RF/XGBoost, SHAP, bias, MLflow
│   └── generate_plots.py           generates 7 submission plots from metrics.json
├── tests/                          pytest test suite (~160 tests)
├── data/
│   ├── raw/                        DVC-tracked raw data
│   ├── processed/                  DVC-tracked processed features
│   └── models/flexibility/         trained model, metrics, bias report, plots
├── secrets/                        GCP + Firebase credentials (never committed)
├── docker-compose.yml              Airflow + Postgres + MLflow
├── Dockerfile                      Airflow image with all dependencies
├── requirements.txt                Python dependencies
└── dvc.yaml                        `DVC pipeline stages
```

---

## Useful Commands

```bash
# Airflow
docker exec -it airflow-scheduler bash
airflow dags list
airflow dags trigger <dag_id>
airflow dags list-import-errors
airflow tasks logs <dag_id> <task_id> <run_id>

# Live log monitoring
docker exec airflow-scheduler bash -c \
  "find /opt/airflow/logs/dag_id=flexibility_model -name '*.log' | sort | tail -1 | xargs tail -f"

# Flexibility pipeline end-to-end
docker exec airflow-scheduler python /opt/airflow/data_seeding/main.py \
  --service-account /opt/airflow/secrets/firebase-admin.json \
  --num-users 20 --days 90 --seed 42
docker exec airflow-scheduler airflow dags trigger flexibility_features

# Check model outputs
docker exec airflow-scheduler cat /opt/airflow/data/models/flexibility/metrics.json | \
  python -c "import json,sys; m=json.load(sys.stdin); print('Winner:', m['winner_model'], '| d7 RMSE:', m['gate_rmse'])"

# Regenerate plots from existing metrics
docker exec airflow-scheduler python /opt/airflow/scripts/generate_plots.py

# DVC
dvc status
dvc repro
dvc push
dvc pull
dvc diff

# MLflow
open http://localhost:5001

# Logs
docker compose logs -f airflow-scheduler
docker compose logs -f airflow-mlflow

# Reset (dev only)
docker compose down -v
docker compose up -d
```