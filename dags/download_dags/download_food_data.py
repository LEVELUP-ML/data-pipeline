"""
DAG: download_food_data

Downloads the Food-101 dataset tarball and extracts it.
Triggers clean_food_data on success.

Output: data/raw/food-101/
"""

from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from dag_monitoring import (
    monitored_dag_args,
    on_dag_failure_callback,
    on_sla_miss_callback,
)

AIRFLOW_HOME = "/opt/airflow"
RAW_DIR = f"{AIRFLOW_HOME}/data/raw"
URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"

with DAG(
    dag_id="download_food_data",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    default_args=monitored_dag_args(retries=2, sla_minutes=30),
    on_failure_callback=on_dag_failure_callback,
    sla_miss_callback=on_sla_miss_callback,
    tags=["download", "food", "ingest"],
) as dag:

    download_and_extract = BashOperator(
        task_id="download_and_extract",
        bash_command=f"""
        set -euo pipefail
        mkdir -p {RAW_DIR}

        TAR="{RAW_DIR}/food-101.tar.gz"
        MARKER="{RAW_DIR}/food-101/meta/train.txt"

        if [ ! -f "$TAR" ]; then
            echo "Downloading Food-101..."
            curl -L -o "$TAR" "{URL}"
        else
            echo "Tarball already exists"
        fi

        if [ ! -f "$MARKER" ]; then
            echo "Extracting..."
            cd {RAW_DIR}
            python -c "
import tarfile
with tarfile.open('food-101.tar.gz', 'r:gz') as t:
    t.extractall('.')
print('Extracted')
"
        else
            echo "Already extracted"
        fi

        echo "=== Contents ==="
        ls -lah {RAW_DIR}/food-101/
        echo "Classes: $(ls {RAW_DIR}/food-101/images/ | wc -l)"
        """,
    )

    trigger_processing = TriggerDagRunOperator(
        task_id="trigger_clean_food",
        trigger_dag_id="clean_food_data",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    download_and_extract >> trigger_processing
