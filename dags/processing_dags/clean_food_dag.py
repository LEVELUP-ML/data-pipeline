"""
DAG: clean_food_data

Preprocesses Food-101 dataset:
  1. Build manifest from meta/train.txt + meta/test.txt
  2. Create train/val split (80/20 stratified)
  3. Generate class distribution
  4. Run mock Gemini inference on val set
  5. Trigger bias monitoring + DVC backup

Input:  data/raw/food-101/
Output: data/processed/food/
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from dag_monitoring import (
    monitored_dag_args,
    on_dag_failure_callback,
    on_sla_miss_callback,
    emit_metric,
    log,
)

AIRFLOW_HOME = "/opt/airflow"
RAW_ROOT = Path(f"{AIRFLOW_HOME}/data/raw/food-101")
OUT_DIR = Path(f"{AIRFLOW_HOME}/data/processed/food")


with DAG(
    dag_id="clean_food_data",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=monitored_dag_args(retries=1, sla_minutes=30),
    on_failure_callback=on_dag_failure_callback,
    sla_miss_callback=on_sla_miss_callback,
    tags=["processing", "food", "data-quality"],
) as dag:

    @task
    def build_manifest() -> Dict[str, Any]:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        meta = RAW_ROOT / "meta"
        images = RAW_ROOT / "images"

        if not meta.exists():
            raise AirflowFailException(
                f"Meta dir not found: {meta}. Run download_food_data first."
            )

        train_items = [
            x.strip()
            for x in (meta / "train.txt").read_text().splitlines()
            if x.strip()
        ]
        test_items = [
            x.strip() for x in (meta / "test.txt").read_text().splitlines() if x.strip()
        ]

        log.info("Train items: %d, Test items: %d", len(train_items), len(test_items))

        rows = []
        missing = 0
        for split, items in [("train", train_items), ("test", test_items)]:
            for item in items:
                label = item.split("/")[0]
                img_path = images / f"{item}.jpg"
                exists = img_path.exists()
                if not exists:
                    missing += 1
                rows.append(
                    {
                        "split": split,
                        "label": label,
                        "image_path": str(img_path),
                        "exists": exists,
                    }
                )

        df = pd.DataFrame(rows)

        if missing > 0:
            log.warning("%d images referenced but not found on disk", missing)

        manifest_path = OUT_DIR / "food_manifest.csv"
        df.to_csv(manifest_path, index=False)
        log.info("Wrote manifest: %s (%d rows)", manifest_path, len(df))

        summary = {
            "total_items": len(df),
            "train_items": len(train_items),
            "test_items": len(test_items),
            "unique_classes": int(df["label"].nunique()),
            "missing_images": missing,
            "manifest_path": str(manifest_path),
        }

        emit_metric("clean_food_data", "build_manifest", summary)
        return summary

    @task
    def create_splits() -> Dict[str, Any]:
        manifest = pd.read_csv(OUT_DIR / "food_manifest.csv")
        train_df = manifest[manifest["split"] == "train"].copy()

        log.info("Creating train/val split from %d train rows", len(train_df))

        train_split, val_split = train_test_split(
            train_df, test_size=0.2, random_state=42, stratify=train_df["label"]
        )

        train_path = OUT_DIR / "food_train.csv"
        val_path = OUT_DIR / "food_val.csv"
        train_split.to_csv(train_path, index=False)
        val_split.to_csv(val_path, index=False)

        log.info("Train: %d rows -> %s", len(train_split), train_path)
        log.info("Val: %d rows -> %s", len(val_split), val_path)

        summary = {
            "train_rows": len(train_split),
            "val_rows": len(val_split),
            "train_classes": int(train_split["label"].nunique()),
            "val_classes": int(val_split["label"].nunique()),
        }

        emit_metric("clean_food_data", "create_splits", summary)
        return summary

    @task
    def generate_class_distribution() -> Dict[str, Any]:
        train_df = pd.read_csv(OUT_DIR / "food_train.csv")

        dist = train_df["label"].value_counts().reset_index()
        dist.columns = ["label", "count"]

        dist_path = OUT_DIR / "food_class_distribution.csv"
        dist.to_csv(dist_path, index=False)

        # Stats for logging
        mean_count = dist["count"].mean()
        std_count = dist["count"].std()
        min_class = dist.iloc[-1]
        max_class = dist.iloc[0]

        log.info(
            "Class distribution: %d classes, mean=%.1f, std=%.1f",
            len(dist),
            mean_count,
            std_count,
        )
        log.info(
            "Largest: %s (%d), Smallest: %s (%d)",
            max_class["label"],
            max_class["count"],
            min_class["label"],
            min_class["count"],
        )

        summary = {
            "num_classes": len(dist),
            "mean_per_class": round(float(mean_count), 1),
            "std_per_class": round(float(std_count), 1),
            "min_class": min_class["label"],
            "min_count": int(min_class["count"]),
            "max_class": max_class["label"],
            "max_count": int(max_class["count"]),
        }

        emit_metric("clean_food_data", "class_distribution", summary)
        return summary

    @task
    def run_mock_inference() -> Dict[str, Any]:
        val_df = pd.read_csv(OUT_DIR / "food_val.csv").head(25)
        out_path = OUT_DIR / "food_predictions.jsonl"

        log.info("Running mock inference on %d val images", len(val_df))

        random.seed(42)
        records = []
        for _, row in val_df.iterrows():
            pred = {
                "image_path": row["image_path"],
                "true_label": row["label"],
                "predicted_food": row["label"].replace("_", " "),
                "confidence": round(random.uniform(0.6, 0.95), 2),
                "notes": "mock_mode",
            }
            records.append(pred)

        with out_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        log.info("Wrote predictions: %s (%d rows)", out_path, len(records))

        avg_conf = sum(r["confidence"] for r in records) / len(records)

        summary = {
            "predictions": len(records),
            "avg_confidence": round(avg_conf, 4),
            "output_path": str(out_path),
        }

        emit_metric("clean_food_data", "mock_inference", summary)
        return summary

    @task
    def quality_gate(
        manifest_result: Dict[str, Any],
        split_result: Dict[str, Any],
        dist_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Check: images actually exist
        missing = manifest_result.get("missing_images", 0)
        total = manifest_result.get("total_items", 0)
        missing_pct = (missing / total * 100) if total > 0 else 0

        if missing_pct > 20:
            raise AirflowFailException(
                f"Too many missing images: {missing}/{total} ({missing_pct:.1f}%)"
            )

        # Check: reasonable class count
        if dist_result.get("num_classes", 0) < 10:
            raise AirflowFailException(
                f"Only {dist_result['num_classes']} classes found — expected ~101"
            )

        # Check: class balance
        if (
            dist_result.get("std_per_class", 0)
            > dist_result.get("mean_per_class", 1) * 0.5
        ):
            log.warning(
                "Class distribution has high variance (std=%.1f, mean=%.1f)",
                dist_result["std_per_class"],
                dist_result["mean_per_class"],
            )

        log.info(
            "Quality gate PASSED: %d items, %d classes, %d missing images",
            total,
            dist_result["num_classes"],
            missing,
        )

        return {
            "total_items": total,
            "classes": dist_result["num_classes"],
            "missing_images": missing,
            "passed": True,
        }

    trigger_bias = TriggerDagRunOperator(
        task_id="trigger_food_bias",
        trigger_dag_id="food_bias_monitoring",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    trigger_dvc = TriggerDagRunOperator(
        task_id="trigger_dvc_backup",
        trigger_dag_id="dvc_backup_to_gcp",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    # Task flow
    manifest = build_manifest()
    splits = create_splits()
    manifest >> splits

    dist = generate_class_distribution()
    splits >> dist

    inference = run_mock_inference()
    splits >> inference

    gate = quality_gate(manifest, splits, dist)
    [dist, inference] >> gate

    gate >> [trigger_bias, trigger_dvc]
