from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

AIRFLOW_HOME = "/opt/airflow"

with DAG(
    dag_id="dvc_backup_to_gcp",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["dvc", "gcp", "backup"],
) as dag:

    dvc_add_raw = BashOperator(
        task_id="dvc_add_raw",
        bash_command=f"""
        set -euo pipefail
        cd {AIRFLOW_HOME}
        dvc add data/raw
        echo "DVC add data/raw complete. Staged:"
        cat data/raw.dvc
        """,
    )

    dvc_add_processed = BashOperator(
        task_id="dvc_add_processed",
        bash_command=f"""
        set -euo pipefail
        cd {AIRFLOW_HOME}
        dvc add data/processed
        echo "DVC add data/processed complete. Staged:"
        cat data/processed.dvc
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
        git add data/raw.dvc data/processed.dvc
        git diff --cached --quiet && echo "Nothing to commit" || git commit -m "chore: update data/raw.dvc data/processed.dvc [airflow]"
        echo "Git commit complete."
        """,
    )

    [dvc_add_raw, dvc_add_processed] >> dvc_push >> git_commit
 