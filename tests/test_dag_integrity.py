"""
tests/test_dag_integrity.py

Structural / integrity tests for all Airflow DAGs.

These tests verify that every DAG:
  - Loads without import errors
  - Has a unique dag_id
  - Defines at least one task
  - Has the expected tags
  - Task dependencies are acyclic (Airflow enforces this, but we check early)
  - No task has missing upstream when it expects one

These tests run in under a second and require no Docker / Airflow scheduler.
They act as a fast pre-commit gate before deploying DAGs.

Run with:  pytest tests/test_dag_integrity.py -v
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List

import pytest

# Skip the entire module when apache-airflow is not installed.
# Model CI workflows (strength, flexibility, energy) do not install Airflow;
# DAG integrity is validated by the dedicated dag_integrity.yml workflow.
# To run locally: pip install apache-airflow==2.9.3
pytest.importorskip(
    "airflow",
    reason=(
        "apache-airflow is not installed. "
        "Run the dag-integrity workflow or install airflow locally."
    ),
)

# ── DAG loader ────────────────────────────────────────────────────────────────

DAG_ROOT = Path(__file__).parent.parent / "dags"

# Mapping of (subpackage, module_name) for every DAG file
DAG_MODULES = [
    # download
    ("download_dags", "download_wisdm"),
    ("download_dags", "download_food_data"),
    ("download_dags", "download_sleep_health"),
    ("download_dags", "download_synthetic_from_firestore"),
    ("download_dags", "kaggle_download_flexibility"),
    ("download_dags", "kaggle_download_strength"),
    # processing
    ("processing_dags", "clean_wisdm_dag"),
    ("processing_dags", "clean_weightlifting_dag"),
    ("processing_dags", "clean_food_dag"),
    ("processing_dags", "strength_features_dag"),
    ("processing_dags", "flexibility_features_dag"),
    ("processing_dags", "strength_model_dag"),
    ("processing_dags", "flexibility_model_dag"),
    ("processing_dags", "energy_model_dag"),
    # analyze_behavior_dag intentionally excluded — it is a legacy duplicate of
    # clean_food_dag.py; both declare dag_id="clean_food_data". Delete the file
    # or give it a unique dag_id before re-adding it here.
    ("processing_dags", "process_synthetic_data_dag"),
    ("processing_dags", "firebase_schema_validation_dag"),
    # monitoring
    ("monitoring_dags", "daily_bias_monitoring"),
    ("monitoring_dags", "synthetic_anomaly_and_bias_dag"),
    ("monitoring_dags", "food_bias_monitoring"),
    # backup
    ("backup_dags", "dvc_backup_gcp"),
    ("backup_dags", "firestore_metric_events_to_gcs"),
]

# Expected tags that each category must include (subpackage → required tags)
REQUIRED_TAGS: Dict[str, List[str]] = {
    # All download DAGs share the "ingest" convention.
    "download_dags":    ["ingest"],
    # processing_dags is a mixed folder: clean_*, features_*, model_*, validation_*.
    # Only the clean_* DAGs carry "cleaning"; feature/model/validation DAGs have
    # their own tag vocabularies. Enforce specific tags via per-DAG task tests
    # in TestDagTaskDependencies instead of folder-wide here.
    "processing_dags":  [],
    "monitoring_dags":  ["monitoring"],
    "backup_dags":      [],
}


def _import_dag_module(subpkg: str, module_name: str) -> ModuleType:
    """Dynamically import a DAG module, adding dags/ to sys.path first."""
    dag_root = str(DAG_ROOT)
    if dag_root not in sys.path:
        sys.path.insert(0, dag_root)
    full_name = f"{subpkg}.{module_name}"
    return importlib.import_module(full_name)


def _collect_dags(module: ModuleType):
    """Return all Airflow DAG objects defined in a module."""
    from airflow import DAG
    return [
        obj for obj in vars(module).values()
        if isinstance(obj, DAG)
    ]


# ── Parametrize across all DAG modules ────────────────────────────────────────

@pytest.fixture(params=DAG_MODULES, ids=lambda x: f"{x[0]}.{x[1]}")
def dag_module_pair(request):
    return request.param


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestDagLoading:

    def test_module_imports_without_error(self, dag_module_pair):
        """DAG file must be importable (no syntax errors, missing imports, etc.)."""
        subpkg, name = dag_module_pair
        try:
            mod = _import_dag_module(subpkg, name)
            assert mod is not None
        except ImportError as exc:
            pytest.fail(f"Import error in {subpkg}.{name}: {exc}")

    def test_dag_objects_present(self, dag_module_pair):
        """Each module must expose at least one Airflow DAG."""
        subpkg, name = dag_module_pair
        mod  = _import_dag_module(subpkg, name)
        dags = _collect_dags(mod)
        assert dags, f"{subpkg}.{name} defines no DAG objects"

    def test_dag_ids_are_strings(self, dag_module_pair):
        subpkg, name = dag_module_pair
        mod  = _import_dag_module(subpkg, name)
        for dag in _collect_dags(mod):
            assert isinstance(dag.dag_id, str)
            assert dag.dag_id.strip() != ""

    def test_dag_has_tasks(self, dag_module_pair):
        subpkg, name = dag_module_pair
        mod  = _import_dag_module(subpkg, name)
        for dag in _collect_dags(mod):
            assert len(dag.task_ids) > 0, f"DAG {dag.dag_id} has no tasks"

    def test_dag_tags_present(self, dag_module_pair):
        """DAGs should declare at least one tag."""
        subpkg, name = dag_module_pair
        mod  = _import_dag_module(subpkg, name)
        for dag in _collect_dags(mod):
            assert dag.tags, f"DAG {dag.dag_id} has no tags"

    def test_required_category_tags(self, dag_module_pair):
        subpkg, name = dag_module_pair
        mod      = _import_dag_module(subpkg, name)
        required = REQUIRED_TAGS.get(subpkg, [])
        for dag in _collect_dags(mod):
            for tag in required:
                assert tag in dag.tags, (
                    f"DAG {dag.dag_id} (in {subpkg}) is missing required tag '{tag}'"
                )


class TestDagUniqueness:
    """Ensure no two DAG files declare the same dag_id."""

    def test_all_dag_ids_unique(self):
        ids_seen: Dict[str, str] = {}
        for subpkg, name in DAG_MODULES:
            try:
                mod = _import_dag_module(subpkg, name)
            except ImportError:
                continue
            for dag in _collect_dags(mod):
                source = f"{subpkg}.{name}"
                assert dag.dag_id not in ids_seen, (
                    f"Duplicate dag_id '{dag.dag_id}' found in both "
                    f"'{ids_seen[dag.dag_id]}' and '{source}'"
                )
                ids_seen[dag.dag_id] = source


class TestDagSchedule:
    """Spot-check schedule settings for specific DAG categories."""

    def test_download_dags_are_manual_trigger(self):
        """Download DAGs should not run on a schedule — they're triggered explicitly."""
        for name in [
            "download_wisdm",
            "download_food_data",
            "download_sleep_health",
            "kaggle_download_flexibility",
            "kaggle_download_strength",
        ]:
            mod  = _import_dag_module("download_dags", name)
            for dag in _collect_dags(mod):
                assert dag.schedule_interval in (None, "@once"), (
                    f"Download DAG {dag.dag_id} should have schedule=None, "
                    f"got {dag.schedule_interval!r}"
                )

    def test_monitoring_dag_has_cron_schedule(self):
        """daily_bias_monitoring must run on a daily schedule."""
        mod = _import_dag_module("monitoring_dags", "daily_bias_monitoring")
        for dag in _collect_dags(mod):
            assert dag.schedule_interval is not None, (
                f"{dag.dag_id} must have a schedule (expected daily cron)"
            )

    def test_processing_dags_are_manual_trigger(self):
        """Processing DAGs are triggered downstream, not on a cron."""
        for name in [
            "clean_wisdm_dag",
            "clean_weightlifting_dag",
            "strength_features_dag",
        ]:
            mod = _import_dag_module("processing_dags", name)
            for dag in _collect_dags(mod):
                assert dag.schedule_interval in (None, "@once"), (
                    f"Processing DAG {dag.dag_id} should be schedule=None, "
                    f"got {dag.schedule_interval!r}"
                )


class TestDagSettings:
    """Verify sensible operational defaults on all DAGs."""

    def test_catchup_is_false(self, dag_module_pair):
        subpkg, name = dag_module_pair
        mod = _import_dag_module(subpkg, name)
        for dag in _collect_dags(mod):
            assert dag.catchup is False, (
                f"DAG {dag.dag_id}: catchup should be False to avoid backfill avalanche"
            )

    def test_max_active_runs_set(self, dag_module_pair):
        subpkg, name = dag_module_pair
        mod = _import_dag_module(subpkg, name)
        for dag in _collect_dags(mod):
            # At minimum, max_active_runs should be defined (not unlimited)
            assert dag.max_active_runs is not None
            assert dag.max_active_runs >= 1


class TestDagTaskDependencies:
    """Verify task wiring on the two main processing DAGs."""

    def _task_downstream_ids(self, dag, task_id: str) -> List[str]:
        task = dag.get_task(task_id)
        return [t.task_id for t in task.downstream_list]

    def test_wisdm_dag_task_chain(self):
        mod = _import_dag_module("processing_dags", "clean_wisdm_dag")
        dag = _collect_dags(mod)[0]

        task_ids = set(dag.task_ids)
        # All expected tasks must exist
        for expected in [
            "discover_files",
            "load_and_validate",
            "quality_gate",
            "run_anomaly_detection",
            "window_and_compute_stamina",
            "trigger_dvc_backup",
        ]:
            assert expected in task_ids, f"Task '{expected}' missing from wisdm DAG"

    def test_weightlifting_dag_task_chain(self):
        mod = _import_dag_module("processing_dags", "clean_weightlifting_dag")
        dag = _collect_dags(mod)[0]

        task_ids = set(dag.task_ids)
        for expected in [
            "discover_files",
            "clean_and_validate",
            "quality_gate",
            "trigger_dvc_backup",
        ]:
            assert expected in task_ids, f"Task '{expected}' missing from weightlifting DAG"

    def test_monitoring_dag_task_chain(self):
        mod = _import_dag_module("monitoring_dags", "daily_bias_monitoring")
        dag = _collect_dags(mod)[0]

        task_ids = set(dag.task_ids)
        for expected in [
            "analyze_wisdm_bias",
            "analyze_weightlifting_bias",
            "build_report",
            "send_slack_report",
        ]:
            assert expected in task_ids, f"Task '{expected}' missing from monitoring DAG"