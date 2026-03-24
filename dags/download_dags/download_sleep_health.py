"""
dags/download_dags/kaggle_download_sleep_health.py

Downloads the Sleep Health and Lifestyle dataset from Kaggle.
Dataset: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

Outputs:
    /opt/airflow/data/raw/sleep_health/Sleep_health_and_lifestyle_dataset.csv

Triggers:
    energy_model_dag (on success)
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago

from dag_monitoring import (
    monitored_dag_args,
    on_dag_failure_callback,
    on_sla_miss_callback,
)

DATASET = "uom190346a/sleep-health-and-lifestyle-dataset"
OUT_DIR = "/opt/airflow/data/raw/sleep_health"

with DAG(
    dag_id="kaggle_download_sleep_health",
    start_date=days_ago(1),
    schedule=None,
    catchup=False,
    default_args=monitored_dag_args(retries=2, sla_minutes=15),
    on_failure_callback=on_dag_failure_callback,
    sla_miss_callback=on_sla_miss_callback,
    tags=["kaggle", "ingest", "energy"],
) as dag:

    download = BashOperator(
        task_id="download",
        bash_command=f"""
        set -euo pipefail
        mkdir -p {OUT_DIR}

        echo "Downloading {DATASET}..."
        kaggle datasets download \\
            -d {DATASET} \\
            -p {OUT_DIR} \\
            --unzip

        echo "Downloaded files:"
        ls -lah {OUT_DIR}

        if [ ! -f "{OUT_DIR}/Sleep_health_and_lifestyle_dataset.csv" ]; then
            echo "ERROR: Expected CSV not found in {OUT_DIR}"
            ls {OUT_DIR}
            exit 1
        fi

        echo "Row count:"
        wc -l "{OUT_DIR}/Sleep_health_and_lifestyle_dataset.csv"
        """,
    )

    validate = BashOperator(
        task_id="validate",
        bash_command=f"""
        set -euo pipefail
        python3 - <<'PYEOF'
import pandas as pd, sys

path = "{OUT_DIR}/Sleep_health_and_lifestyle_dataset.csv"
df   = pd.read_csv(path)

print(f"Shape: {{df.shape}}")
print(f"Columns: {{list(df.columns)}}")
print(f"\\nNull counts:\\n{{df.isnull().sum()}}")

required = [
    "Person ID", "Age", "Gender", "Sleep Duration",
    "Quality of Sleep", "Physical Activity Level",
    "BMI Category", "Heart Rate", "Daily Steps", "Sleep Disorder",
]
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"ERROR: Missing columns: {{missing}}")
    sys.exit(1)

print(f"All {{len(required)}} required columns present.")
print(f"Rows: {{len(df)}}  Age range: {{df['Age'].min()}}–{{df['Age'].max()}}")
print(f"Gender values: {{df['Gender'].unique().tolist()}}")
PYEOF
        """,
    )

    trigger_energy = TriggerDagRunOperator(
        task_id="trigger_energy_model",
        trigger_dag_id="energy_model_dag",
        wait_for_completion=False,
    )

    download >> validate >> trigger_energy