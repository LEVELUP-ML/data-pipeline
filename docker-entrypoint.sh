#!/bin/bash
gcloud auth activate-service-account \
    --key-file=/opt/airflow/secrets/gcp-sa.json 2>/dev/null || true
exec "$@"