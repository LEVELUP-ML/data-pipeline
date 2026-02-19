from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

AIRFLOW_HOME = "/opt/airflow"
DATA_DIR = f"{AIRFLOW_HOME}/data/raw"
SECRETS_DIR = f"{AIRFLOW_HOME}/secrets"

with DAG(
    dag_id="dvc_backup_to_gcp",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["dvc", "gcp", "backup"],
) as dag:

    dvc_add = BashOperator(
        task_id="dvc_add",
        bash_command=f"""
        set -euo pipefail
        cd {AIRFLOW_HOME}
        dvc add data/raw
        echo "DVC add complete. Staged:"
        cat data/raw.dvc
        """,
    )

    dvc_push = BashOperator(
        task_id="dvc_push",
        bash_command=f"""
        set -euo pipefail
        cd {AIRFLOW_HOME}
        dvc push
        echo "DVC push to GCS complete."
        """,
    )

    git_commit = BashOperator(
        task_id="git_commit",
        bash_command=f"""
        set -euo pipefail
        cd {AIRFLOW_HOME}
        git config --global --add safe.directory /opt/airflow
        git config user.email "airflow@pipeline"
        git config user.name "Airflow"
        git add data/raw.dvc
        git diff --cached --quiet && echo "Nothing to commit" || git commit -m "chore: update data/raw.dvc [airflow]"
        echo "Git commit complete."
        """,
    )

    dvc_add >> dvc_push >> git_commit
