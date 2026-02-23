from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

DATASET = "joep89/weightlifting"
OUT_DIR = "/opt/airflow/data/raw/weightlifting"

with DAG(
    dag_id="kaggle_download_weightlifting",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["kaggle", "ingest"],
) as dag:

    download = BashOperator(
        task_id="download",
        bash_command=f"""
        set -euo pipefail
        mkdir -p {OUT_DIR}
        kaggle datasets download -d {DATASET} -p {OUT_DIR} --unzip
        ls -lah {OUT_DIR}
        """,
    )
