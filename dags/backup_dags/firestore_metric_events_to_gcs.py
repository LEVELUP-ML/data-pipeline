from __future__ import annotations

import io
import json
import os
from collections import defaultdict
from typing import Dict, List, Any
from datetime import datetime

import pendulum
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import get_current_context
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from airflow.providers.google.cloud.hooks.gcs import GCSHook

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import FieldFilter


DEFAULT_BACKUP_PREFIX = "firestore_backups/metric_events"


def get_firestore_client() -> firestore.Client:
    """
    Initialize firebase-admin exactly once per worker process.
    """
    if not firebase_admin._apps:
        sa_path = Variable.get("FIREBASE_SERVICE_ACCOUNT_PATH")
        if not os.path.exists(sa_path):
            raise AirflowFailException(
                f"Service account JSON not found at {sa_path}. "
                "Mount it into the Airflow container and set FIREBASE_SERVICE_ACCOUNT_PATH."
            )
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()


def jsonl_from_rows(rows: List[Dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    for r in rows:
        buf.write(
            json.dumps(
                r, separators=(",", ":"), ensure_ascii=False, default=_json_default
            )
        )
        buf.write("\n")
    return buf.getvalue().encode("utf-8")


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, GeoPoint):
        return {"lat": obj.latitude, "lng": obj.longitude}
    if isinstance(obj, DocumentReference):
        return obj.path
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


with DAG(
    dag_id="firestore_metric_events_to_gcs",
    start_date=pendulum.datetime(2026, 2, 1, tz="America/New_York"),
    schedule="0 3 * * *",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2},
    tags=["firestore", "backup", "gcs", "metrics"],
) as dag:

    @task
    def export_day_to_gcs() -> Dict[str, Any]:
        """
        Export yesterday's metric_events (by the stored 'day' field) into GCS,
        partitioned by day and metric.
        """
        bucket = Variable.get("GCS_BACKUP_BUCKET")
        prefix = Variable.get("BACKUP_PREFIX", default_var=DEFAULT_BACKUP_PREFIX)
        max_docs = int(Variable.get("MAX_DOCS_PER_RUN", default_var="500000"))
        chunk_size = int(Variable.get("CHUNK_SIZE", default_var="5000"))

        ctx = get_current_context()
        logical_date = ctx["logical_date"]  # tz-aware pendulum dt
        ny = pendulum.timezone("America/New_York")
        target_day = logical_date.in_timezone(ny).subtract(days=1).format("YYYY-MM-DD")

        db = get_firestore_client()

        # Collection group query across all users/*/metric_events/*
        q = db.collection_group("metric_events").where(
            filter=FieldFilter("day", "==", target_day)
        )
        docs_iter = q.stream()

        per_metric_buffers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        per_metric_part_idx: Dict[str, int] = defaultdict(int)

        gcs = GCSHook(gcp_conn_id="google_cloud_default")

        total_docs = 0
        uploaded_objects = 0

        def flush_metric(metric: str):
            nonlocal uploaded_objects
            rows = per_metric_buffers[metric]
            if not rows:
                return
            per_metric_part_idx[metric] += 1
            part = per_metric_part_idx[metric]
            object_name = (
                f"{prefix}/day={target_day}/metric={metric}/part-{part:05d}.jsonl"
            )
            gcs.upload(
                bucket_name=bucket,
                object_name=object_name,
                data=jsonl_from_rows(rows),
                mime_type="application/json",
            )
            uploaded_objects += 1
            per_metric_buffers[metric].clear()

        for doc in docs_iter:
            total_docs += 1
            if total_docs > max_docs:
                raise AirflowFailException(
                    f"Aborting: exceeded MAX_DOCS_PER_RUN={max_docs}. "
                    "Either raise the cap or fix your export window/filters."
                )

            d = doc.to_dict() or {}
            metric = d.get("metric")
            if not metric:
                raise AirflowFailException(
                    f"metric_events doc {doc.id} missing 'metric'"
                )

            # extract uid from path users/{uid}/metric_events/{eventId}
            path_parts = doc.reference.path.split("/")
            uid = None
            try:
                uid = path_parts[path_parts.index("users") + 1]
            except Exception:
                pass

            per_metric_buffers[metric].append({"uid": uid, "eventId": doc.id, **d})

            if len(per_metric_buffers[metric]) >= chunk_size:
                flush_metric(metric)

        for metric in list(per_metric_buffers.keys()):
            flush_metric(metric)

        return {
            "day": target_day,
            "docs_exported": total_docs,
            "objects_written": uploaded_objects,
            "bucket": bucket,
            "prefix": prefix,
        }

    export_day_to_gcs()
