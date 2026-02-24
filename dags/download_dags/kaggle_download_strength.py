from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from dag_monitoring import (
    monitored_dag_args,
    on_dag_failure_callback,
    on_sla_miss_callback,
)

DATASET = Variable.get("strength_dataset", default_var="joep89/weightlifting")
OUT_DIR = "/opt/airflow/data/raw/strength"

with DAG(
    dag_id="kaggle_download_strength",
    start_date=days_ago(1),
    schedule=None,
    catchup=False,
    default_args=monitored_dag_args(retries=2, sla_minutes=15),
    on_failure_callback=on_dag_failure_callback,
    sla_miss_callback=on_sla_miss_callback,
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
