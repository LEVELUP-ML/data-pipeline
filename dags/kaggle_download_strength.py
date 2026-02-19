from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

DATASET = Variable.get("strength_dataset", default_var="joep89/weightlifting")
OUT_DIR = "/opt/airflow/data/raw/strength"

with DAG(
    dag_id="kaggle_download_strength",
    start_date=days_ago(1),
    schedule=None,
    catchup=False,
    tags=["kaggle", "ingest"],
) as dag:

    download = BashOperator(
        task_id="download",
        bash_command=f"""
        set -euo pipefail
        mkdir -p {OUT_DIR}
        if [ -z "{DATASET}" ]; then
          echo "No strength dataset configured (Airflow Variable 'strength_dataset'). Exiting."
          exit 1
        fi
        kaggle datasets download -d {DATASET} -p {OUT_DIR} --unzip
        ls -lah {OUT_DIR}
        """,
    )
