# Level Up - MLOps Data Pipeline

End-to-end MLOps data pipeline for fitness and activity tracking data. Covers ingestion, cleaning, validation, stamina computation, bias monitoring, versioning, and alerting — orchestrated with Apache Airflow and versioned with DVC.

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
```

### 2. Start Airflow

```bash
docker compose up -d
docker compose ps
```

Airflow UI: [http://localhost:8080](http://localhost:8080) (default: `admin` / `admin`)

### 3. Initialize DVC

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

### 4. Run Tests

```bash
pip install pytest pytest-cov
pytest
pytest --cov=tests --cov-report=term-missing
```

## Pipeline Overview

### DAG Dependency Flow

```
Download DAGs (manual trigger)
├── download_wisdm_accel           → data/raw/wisdm/
├── kaggle_download_strength       → data/raw/strength/
└── kaggle_download_flexibility    → data/raw/flexibility/
        │
        ▼
Processing DAGs (manual or triggered)
├── clean_weightlifting_data       → data/processed/weightlifting_cleaned/
│   └── triggers: dvc_backup_to_gcp
├── clean_wisdm_accel_data         → data/processed/wisdm/
│   └── triggers: dvc_backup_to_gcp
└── firestore_schema_validation    (daily 02:00 ET)
    └── triggers: firestore_metric_events_to_gcs
        │
        ▼
Backup DAGs
├── dvc_backup_to_gcp              → DVC add/push raw + processed, git commit
└── firestore_metric_events_to_gcs → GCS JSONL partitioned by day/metric
        │
        ▼
Monitoring DAGs (daily 06:00 ET)
└── daily_bias_monitoring          → Slack report
```

### Data Sources

| Source | Type | DAG | Raw Location |
|--------|------|-----|--------------|
| Kaggle weightlifting | CSV (9,932 rows, 10 cols) | `kaggle_download_strength` | `data/raw/strength/` |
| UCI WISDM | Accel txt (15.6M rows, 6 cols) | `download_wisdm_accel` | `data/raw/wisdm/` |
| Firestore | NoSQL (metric_events, sleep_logs, quiz_attempts) | `firestore_metric_events_to_gcs` | GCS bucket |

### Data Cleaning

**Weightlifting** (`clean_weightlifting_dag.py`):
- Schema validation (required columns: Date, Workout Name, Exercise Name, Set Order, Weight, Reps)
- Whitespace stripping on all string columns
- Type coercion (dates, floats, nullable ints)
- Deduplication on natural key
- Row-level validation: negative values, out-of-range weight (>700kg), reps (>200), set order, seconds, distance
- Output: clean Parquet + rejection CSV with tagged reasons
- Quality gate: blocks pipeline if rejection rate >15%

**WISDM** (`clean_wisdm_dag.py`):
- Semicolon stripping from z-axis values
- Type coercion for all numeric columns
- Deduplication
- Row validation: valid activity codes (A-S, no N), missing axes, extreme magnitude (>200)
- 200-row windowing with mean aggregation
- Sequential stamina computation (fatigue drain based on acceleration magnitude)
- Anomaly detection (5σ magnitude spikes)
- Per-activity bias analysis
- Output: clean Parquet, stamina Parquet, bias JSON

**Firestore** (`firebase_schema_validation_dag.py`):
- Validates metric_events: metric ∈ {strength, stamina, speed, flexibility, intelligence}, score ∈ [0,100], confidence ∈ [0,1], payload component keys
- Validates sleep_logs: sleepHours ∈ [0,24], quality ∈ [1,5], time formats
- Validates quiz_attempts: valid topics, num_correct ≥ 0, num_correct ≤ num_questions, difficulty ∈ [1,5]
- Configurable error threshold gate before allowing GCS export

### Bias Detection and Monitoring

The `daily_bias_monitoring` DAG runs daily and analyzes:

**WISDM / Stamina:**
- Stamina distribution across all 18 activity types (mean, std, min, max)
- Per-user stamina spread and outlier detection (>2σ from global mean)
- Activity gap analysis (% difference between highest and lowest mean stamina)

**Weightlifting:**
- Exercise volume concentration (top 5 exercises as % of total volume)
- Weight progression trend (recent 30 days vs older data)
- Workout frequency distribution by day of week

**Alert thresholds:**
- Activity stamina gap >50%
- Outlier users detected
- Top 5 exercises >80% of total volume

Reports are sent to Slack via webhook with formatted Block Kit messages.

### Logging and Monitoring

All DAGs use a shared `dag_monitoring.py` module providing:
- Structured logging via `logging.getLogger("airflow.task")` (replaces all `print()` calls)
- Task-level callbacks: `on_failure_callback`, `on_success_callback`, `on_retry_callback`
- DAG-level callbacks: `on_dag_failure_callback`
- SLA monitoring with `on_sla_miss_callback`
- Slack webhook alerts on failures and SLA misses
- `emit_metric()` — writes structured JSONL to `/opt/airflow/logs/dag_metrics/` for external scraping

| DAG | SLA Budget |
|-----|-----------|
| Kaggle downloads | 15 min |
| DVC backup | 20 min |
| Weightlifting cleaning | 30 min |
| WISDM cleaning | 30 min |
| Firestore validation | 45 min |
| Firestore export | 60 min |
| Bias monitoring | 20 min |

### Data Versioning

DVC tracks `data/raw/` and `data/processed/` with a GCS remote backend. The `dvc_backup_to_gcp` DAG automates:
1. `dvc add data/raw` and `dvc add data/processed` (in parallel)
2. `dvc push` to GCS
3. `git commit` of `.dvc` files

Reproducibility: clone the repo, run `dvc pull`, and all data is restored from GCS.

The `dvc.yaml` file also defines the full pipeline for local `dvc repro` execution outside of Airflow.

### Testing

67 tests across 4 modules using pytest:

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_wisdm_loader.py` | 11 | File loading, semicolon parsing, malformed/empty files, multi-file concat |
| `test_weightlifting_cleaning.py` | 14 | Row validation (all issue types), schema check, deduplication |
| `test_stamina_and_anomaly.py` | 22 | Windowing (size, edge cases), stamina (monotonic decrease, floor at 0, fatigue rates), anomaly detection, WISDM row validation |
| `test_schema_validation.py` | 20 | Firestore metric_events/sleep_logs/quiz_attempts validation, parametrized across all metrics and topics |

Run:
```bash
pytest                                    # all tests
pytest tests/test_wisdm_loader.py         # single module
pytest -k "test_stamina"                  # by name pattern
pytest --cov --cov-report=term-missing    # with coverage
```

## Useful Commands

```bash
# Airflow
docker exec -it airflow-scheduler bash
airflow dags list
airflow dags trigger <dag_id>
airflow tasks test <dag_id> <task_id> <date>
airflow dags list-import-errors

# DVC
dvc status
dvc repro
dvc push
dvc pull
dvc diff

# Logs
docker compose logs -f airflow-scheduler
docker compose logs -f postgres

# Reset (dev only)
docker compose down -v
docker compose up -d
```