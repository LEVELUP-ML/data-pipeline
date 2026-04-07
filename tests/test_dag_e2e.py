"""
tests/test_dag_e2e.py

End-to-end tests for DAG task functions.

These tests exercise the full task-function chain (discover → load → validate
→ quality_gate → anomaly → stamina) against temporary on-disk fixtures.

No live Airflow scheduler is needed — we call the underlying Python functions
directly, the same way Airflow's TaskInstance does when it executes a @task.

Run with:  pytest tests/test_dag_e2e.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

#  Import shared lib 
from lib.wisdm import (
    analyze_bias,
    compute_stamina,
    create_row_windows,
    detect_anomalies,
    load_wisdm_file,
    tag_row_issues,
    validate_schema,
)
from lib.weightlifting import clean_dataframe, validate_schema as wl_validate_schema


#  WISDM end-to-end helpers (mirrors DAG task bodies) 

def _load_and_validate_wisdm(
    file_paths: List[str], clean_dir: str, reject_dir: str
) -> Dict[str, Any]:
    """Mirrors the load_and_validate @task from clean_wisdm_dag."""
    frames = [load_wisdm_file(p) for p in file_paths]
    raw = pd.concat(frames, ignore_index=True)
    total_raw = len(raw)

    missing = validate_schema(raw)
    assert not missing, f"Missing columns: {missing}"

    before = len(raw)
    raw = raw.drop_duplicates(subset=["user", "activity", "timestamp", "x", "y", "z"])
    dupes = before - len(raw)

    raw["_issues"] = raw.apply(tag_row_issues, axis=1)
    raw["_is_clean"] = raw["_issues"].apply(lambda x: len(x) == 0)

    clean    = raw[raw["_is_clean"]].drop(columns=["_issues", "_is_clean"])
    rejected = raw[~raw["_is_clean"]].copy()
    clean    = clean.sort_values(["user", "timestamp"]).reset_index(drop=True)

    os.makedirs(clean_dir,  exist_ok=True)
    os.makedirs(reject_dir, exist_ok=True)

    clean_path  = os.path.join(clean_dir,  "wisdm_accel_clean_test.parquet")
    reject_path = os.path.join(reject_dir, "wisdm_accel_rejected_test.csv")
    clean.to_parquet(clean_path, index=False)
    if len(rejected):
        rejected["_issues"] = rejected["_issues"].apply("; ".join)
        rejected.drop(columns=["_is_clean"]).to_csv(reject_path, index=False)

    return {
        "total_raw": total_raw,
        "duplicates_dropped": dupes,
        "clean_rows": len(clean),
        "rejected_rows": len(rejected),
        "reject_pct": round(len(rejected) / total_raw * 100, 2) if total_raw else 0.0,
        "unique_users": int(clean["user"].nunique()),
        "unique_activities": int(clean["activity"].nunique()),
        "clean_path": clean_path,
    }


def _quality_gate_wisdm(summary: Dict[str, Any], max_pct: float = 10.0) -> str:
    assert summary["reject_pct"] <= max_pct, (
        f"Rejection rate {summary['reject_pct']}% exceeds {max_pct}%"
    )
    assert summary["clean_rows"] > 0, "Zero clean rows"
    return summary["clean_path"]


def _anomaly_detection_wisdm(clean_path: str) -> str:
    df = pd.read_parquet(clean_path)
    result = detect_anomalies(df)
    assert isinstance(result["anomaly_count"], int)
    return clean_path


def _stamina_pipeline(clean_path: str, clean_dir: str) -> Dict[str, Any]:
    df = pd.read_parquet(clean_path)
    windowed = create_row_windows(df, window_size=5)
    assert not windowed.empty, "Windowed dataset is empty"
    result = compute_stamina(windowed)
    bias   = analyze_bias(result)

    out_path  = os.path.join(clean_dir, "wisdm_stamina_test.parquet")
    bias_path = os.path.join(clean_dir, "wisdm_bias_test.json")
    result.to_parquet(out_path, index=False)
    with open(bias_path, "w") as f:
        json.dump(bias, f)

    return {"stamina_path": out_path, "bias_path": bias_path, "windows": len(result)}


#  WISDM E2E fixtures 

def _write_wisdm_txt(path: Path, n_rows: int = 50, user: int = 1600) -> None:
    np.random.seed(42)
    lines = []
    for i in range(n_rows):
        x, y, z = np.random.normal(0.1, 0.5), np.random.normal(-0.2, 0.5), np.random.normal(9.8, 0.3)
        activity = np.random.choice(list("ABCDE"))
        lines.append(f"{user},{activity},{1_000_000 + i},{x:.4f},{y:.4f},{z:.4f};\n")
    path.write_text("".join(lines))


@pytest.fixture
def wisdm_workspace(tmp_path):
    raw_dir    = tmp_path / "raw" / "wisdm"
    clean_dir  = tmp_path / "processed" / "wisdm"
    reject_dir = tmp_path / "processed" / "wisdm_rejected"
    raw_dir.mkdir(parents=True)

    _write_wisdm_txt(raw_dir / "data_1600_accel_phone.txt", n_rows=50, user=1600)
    _write_wisdm_txt(raw_dir / "data_1601_accel_phone.txt", n_rows=50, user=1601)

    return {
        "raw_dir":   str(raw_dir),
        "clean_dir": str(clean_dir),
        "reject_dir": str(reject_dir),
    }


@pytest.fixture
def wisdm_workspace_with_dirty(tmp_path):
    raw_dir    = tmp_path / "raw" / "wisdm"
    clean_dir  = tmp_path / "processed" / "wisdm"
    reject_dir = tmp_path / "processed" / "wisdm_rejected"
    raw_dir.mkdir(parents=True)

    # Good file
    _write_wisdm_txt(raw_dir / "data_1600_accel_phone.txt", n_rows=50, user=1600)

    # Dirty file: invalid activities, extreme magnitudes
    dirty = raw_dir / "data_bad_accel_phone.txt"
    dirty.write_text(
        "9999,Z,100,0.1,-0.2,9.8;\n"   # invalid activity Z
        "9999,A,101,999,999,999;\n"     # extreme magnitude
        ",A,102,0.1,-0.2,9.8;\n"        # missing user
    )
    return {
        "raw_dir":   str(raw_dir),
        "clean_dir": str(clean_dir),
        "reject_dir": str(reject_dir),
    }


#  WISDM E2E tests 


class TestWisdmDagE2E:

    def _file_paths(self, ws: dict) -> List[str]:
        raw_dir = ws["raw_dir"]
        return sorted(
            os.path.join(raw_dir, f)
            for f in os.listdir(raw_dir)
            if f.endswith(".txt") and "accel" in f
        )

    def test_full_happy_path(self, wisdm_workspace):
        ws = wisdm_workspace
        paths = self._file_paths(ws)
        assert len(paths) == 2

        # Step 1: discover + validate
        summary = _load_and_validate_wisdm(paths, ws["clean_dir"], ws["reject_dir"])
        assert summary["clean_rows"] == 100
        assert summary["rejected_rows"] == 0
        assert summary["reject_pct"] == 0.0
        assert summary["unique_users"] == 2

        # Step 2: quality gate
        clean_path = _quality_gate_wisdm(summary)
        assert clean_path == summary["clean_path"]
        assert os.path.exists(clean_path)

        # Step 3: anomaly detection
        after_anomaly = _anomaly_detection_wisdm(clean_path)
        assert after_anomaly == clean_path

        # Step 4: windowing + stamina
        result = _stamina_pipeline(clean_path, ws["clean_dir"])
        assert result["windows"] > 0
        assert os.path.exists(result["stamina_path"])
        assert os.path.exists(result["bias_path"])

        # Validate stamina parquet
        stamina_df = pd.read_parquet(result["stamina_path"])
        assert "stamina" in stamina_df.columns
        assert (stamina_df["stamina"] >= 0).all()
        assert (stamina_df["stamina"] <= 100.0).all()

    def test_dirty_data_partially_rejected(self, wisdm_workspace_with_dirty):
        ws = wisdm_workspace_with_dirty
        paths = self._file_paths(ws)
        summary = _load_and_validate_wisdm(paths, ws["clean_dir"], ws["reject_dir"])
        # 50 good + 3 dirty (2 invalid activity/magnitude, 1 missing user)
        assert summary["rejected_rows"] >= 2
        assert summary["clean_rows"] == 50
        assert summary["reject_pct"] > 0

    def test_quality_gate_passes_below_threshold(self, wisdm_workspace):
        ws = wisdm_workspace
        paths = self._file_paths(ws)
        summary = _load_and_validate_wisdm(paths, ws["clean_dir"], ws["reject_dir"])
        # Should not raise
        clean_path = _quality_gate_wisdm(summary, max_pct=10.0)
        assert clean_path

    def test_quality_gate_fails_above_threshold(self, wisdm_workspace_with_dirty):
        ws = wisdm_workspace_with_dirty
        paths = self._file_paths(ws)
        summary = _load_and_validate_wisdm(paths, ws["clean_dir"], ws["reject_dir"])
        # Force threshold to 0 to trigger failure
        with pytest.raises(AssertionError, match="Rejection rate"):
            _quality_gate_wisdm(summary, max_pct=0.0)

    def test_stamina_output_non_increasing(self, wisdm_workspace):
        ws = wisdm_workspace
        paths = self._file_paths(ws)
        summary = _load_and_validate_wisdm(paths, ws["clean_dir"], ws["reject_dir"])
        clean_path = _quality_gate_wisdm(summary)
        _anomaly_detection_wisdm(clean_path)
        result = _stamina_pipeline(clean_path, ws["clean_dir"])

        stamina_df = pd.read_parquet(result["stamina_path"])
        diffs = stamina_df["stamina"].diff().dropna()
        assert (diffs <= 0).all(), "Stamina increased somewhere — fatigue logic broken"

    def test_bias_report_has_correct_activities(self, wisdm_workspace):
        ws = wisdm_workspace
        paths = self._file_paths(ws)
        summary = _load_and_validate_wisdm(paths, ws["clean_dir"], ws["reject_dir"])
        clean_path = _quality_gate_wisdm(summary)
        result = _stamina_pipeline(clean_path, ws["clean_dir"])

        with open(result["bias_path"]) as f:
            bias = json.load(f)
        assert len(bias) > 0
        assert all(isinstance(v, float) for v in bias.values())


#  Weightlifting E2E helpers 

def _wl_clean_and_validate(csv_paths: List[str], clean_dir: str, reject_dir: str) -> Dict[str, Any]:
    """Mirrors the clean_and_validate @task from clean_weightlifting_dag."""
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p, dtype=str)
        df["_source_file"] = os.path.basename(p)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    total_raw = len(raw)

    missing = wl_validate_schema(raw)
    assert not missing, f"Missing required columns: {missing}"

    result = clean_dataframe(raw)
    clean, rejected = result["clean"], result["rejected"]

    os.makedirs(clean_dir,  exist_ok=True)
    os.makedirs(reject_dir, exist_ok=True)

    clean_path = os.path.join(clean_dir,  "workouts_clean_test.parquet")
    clean.to_parquet(clean_path, index=False)

    return {
        "total_raw_rows":    total_raw,
        "duplicates_dropped": result["duplicates_dropped"],
        "clean_rows":        len(clean),
        "rejected_rows":     len(rejected),
        "reject_pct":        result["reject_pct"],
        "clean_path":        clean_path,
        "unique_exercises":  int(clean["Exercise Name"].nunique()) if len(clean) else 0,
    }


def _wl_quality_gate(summary: Dict[str, Any], max_pct: float = 15.0) -> Dict[str, Any]:
    assert summary["reject_pct"] <= max_pct
    assert summary["clean_rows"] > 0, "Zero clean rows"
    return summary


#  Weightlifting E2E fixtures 

@pytest.fixture
def wl_workspace(tmp_path):
    raw_dir   = tmp_path / "raw" / "weightlifting"
    clean_dir = tmp_path / "processed" / "weightlifting"
    reject_dir = tmp_path / "processed" / "weightlifting_rejected"
    raw_dir.mkdir(parents=True)

    csv = raw_dir / "workouts.csv"
    rows = []
    for i in range(20):
        rows.append(
            f"2024-01-{(i % 28) + 1:02d},Push Day,Bench Press,{(i % 4) + 1},{60 + i * 2.5},{8 - i % 3}"
        )
    csv.write_text(
        "Date,Workout Name,Exercise Name,Set Order,Weight,Reps\n" + "\n".join(rows) + "\n"
    )
    return {
        "raw_dir":    str(raw_dir),
        "clean_dir":  str(clean_dir),
        "reject_dir": str(reject_dir),
    }


@pytest.fixture
def wl_workspace_dirty(tmp_path):
    raw_dir   = tmp_path / "raw" / "weightlifting"
    clean_dir = tmp_path / "processed" / "weightlifting"
    reject_dir = tmp_path / "processed" / "weightlifting_rejected"
    raw_dir.mkdir(parents=True)

    csv = raw_dir / "workouts_dirty.csv"
    csv.write_text(
        "Date,Workout Name,Exercise Name,Set Order,Weight,Reps\n"
        "2024-01-01,Push,Bench,1,80,8\n"           # clean
        "not-a-date,,Bench,1,80,8\n"               # bad date + missing workout
        "2024-01-02,Push,,1,-5,8\n"                # missing exercise + negative weight
        "2024-01-03,Push,OHP,1,80,300\n"           # reps exceeds max
        "2024-01-04,Pull,Deadlift,1,120,5\n"       # clean
    )
    return {
        "raw_dir":    str(raw_dir),
        "clean_dir":  str(clean_dir),
        "reject_dir": str(reject_dir),
    }


#  Weightlifting E2E tests 


class TestWeightliftingDagE2E:

    def _csv_paths(self, ws: dict) -> List[str]:
        return sorted(
            os.path.join(ws["raw_dir"], f)
            for f in os.listdir(ws["raw_dir"])
            if f.endswith(".csv")
        )

    def test_full_happy_path(self, wl_workspace):
        ws = wl_workspace
        paths = self._csv_paths(ws)
        assert len(paths) == 1

        summary = _wl_clean_and_validate(paths, ws["clean_dir"], ws["reject_dir"])
        assert summary["clean_rows"] == 20
        assert summary["rejected_rows"] == 0
        assert summary["reject_pct"] == 0.0
        assert summary["unique_exercises"] == 1

        gate = _wl_quality_gate(summary)
        assert gate["clean_rows"] == 20

        # Parquet exists and is loadable
        df = pd.read_parquet(summary["clean_path"])
        assert len(df) == 20
        assert "Date" in df.columns

    def test_dirty_data_partially_rejected(self, wl_workspace_dirty):
        ws = wl_workspace_dirty
        paths = self._csv_paths(ws)
        summary = _wl_clean_and_validate(paths, ws["clean_dir"], ws["reject_dir"])
        # 2 clean, 3 dirty
        assert summary["clean_rows"] == 2
        assert summary["rejected_rows"] == 3
        assert summary["reject_pct"] == pytest.approx(60.0)

    def test_quality_gate_fails_on_high_rejection(self, wl_workspace_dirty):
        ws = wl_workspace_dirty
        paths = self._csv_paths(ws)
        summary = _wl_clean_and_validate(paths, ws["clean_dir"], ws["reject_dir"])
        with pytest.raises(AssertionError):
            _wl_quality_gate(summary, max_pct=10.0)  # 60% > 10%

    def test_clean_df_sorted_by_date(self, wl_workspace):
        ws = wl_workspace
        paths = self._csv_paths(ws)
        summary = _wl_clean_and_validate(paths, ws["clean_dir"], ws["reject_dir"])
        df = pd.read_parquet(summary["clean_path"])
        assert list(df["Date"]) == sorted(df["Date"])

    def test_duplicate_rows_dropped(self, tmp_path):
        raw_dir   = tmp_path / "raw"
        clean_dir = tmp_path / "clean"
        raw_dir.mkdir()
        csv = raw_dir / "dupes.csv"
        csv.write_text(
            "Date,Workout Name,Exercise Name,Set Order,Weight,Reps\n"
            "2024-01-01,Push,Bench,1,80,8\n"
            "2024-01-01,Push,Bench,1,80,8\n"  # exact duplicate
            "2024-01-01,Push,Bench,2,85,6\n"
        )
        summary = _wl_clean_and_validate(
            [str(csv)], str(clean_dir), str(tmp_path / "reject")
        )
        assert summary["duplicates_dropped"] == 1
        assert summary["clean_rows"] == 2