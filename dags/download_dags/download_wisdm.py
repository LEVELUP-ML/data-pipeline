from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
from dag_monitoring import (
    monitored_dag_args,
    on_dag_failure_callback,
    on_sla_miss_callback,
)

OUT_DIR = "/opt/airflow/data/raw/wisdm"
TMP_DIR = "/tmp/wisdm_download"
URL = "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"

with DAG(
    dag_id="download_wisdm_accel",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    default_args=monitored_dag_args(retries=2, sla_minutes=20),
    on_failure_callback=on_dag_failure_callback,
    sla_miss_callback=on_sla_miss_callback,
    tags=["uci", "ingest", "wisdm"],
) as dag:

    download = BashOperator(
        task_id="download_and_extract",
        bash_command=f"""
        set -euo pipefail

        echo "=== Downloading WISDM dataset ==="
        mkdir -p {TMP_DIR} {OUT_DIR}
        curl -L -o {TMP_DIR}/wisdm.zip "{URL}"

        echo "=== Extracting (double-zip) ==="
        python -c "
import zipfile, os, shutil, glob

tmp = '{TMP_DIR}'
out = '{OUT_DIR}'

# Extract outer zip
with zipfile.ZipFile(os.path.join(tmp, 'wisdm.zip')) as z:
    z.extractall(tmp)

# Find and extract inner zip
for inner in glob.glob(os.path.join(tmp, '**/*.zip'), recursive=True):
    print(f'Extracting inner zip: {{inner}}')
    with zipfile.ZipFile(inner) as z2:
        for name in z2.namelist():
            if '/raw/' in name and 'accel' in name and name.endswith('.txt'):
                z2.extract(name, tmp)

# Flatten accel txt files into output dir
for f in glob.glob(os.path.join(tmp, '**/*accel*.txt'), recursive=True):
    shutil.copy2(f, out)
    print(f'Copied: {{os.path.basename(f)}}')
"

        echo "=== Cleanup ==="
        rm -rf {TMP_DIR}

        echo "=== Final contents ==="
        ls -lah {OUT_DIR}
        echo "File count: $(ls {OUT_DIR} | wc -l)"
        """,
    )
