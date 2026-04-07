"""
dags/lib/wisdm.py

Pure functions for WISDM accelerometer data processing.
No Airflow imports — importable in tests without a running scheduler.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

VALID_ACTIVITIES: set[str] = set("ABCDEFGHIJKLMOPQRS")  # A-S, no N per dataset docs
ACCEL_MAG_CEILING: float = 200.0
ACCEL_MAG_FLOOR: float = 0.0
REQUIRED_COLS: List[str] = ["user", "activity", "x", "y", "z"]


#  Loading 

def load_wisdm_file(path: str) -> pd.DataFrame:
    """Load a single WISDM accel txt file into a DataFrame."""
    df = pd.read_csv(
        path,
        header=None,
        names=["user", "activity", "timestamp", "x", "y", "z"],
        sep=",",
        on_bad_lines="skip",
    )
    df["z"] = df["z"].astype(str).str.replace(";", "", regex=False)
    for col in ("x", "y", "z", "timestamp"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["user"] = pd.to_numeric(df["user"], errors="coerce").astype("Int64")
    df["activity"] = df["activity"].astype(str).str.strip()
    df["_source_file"] = os.path.basename(path)
    return df


#  Validation 

def tag_row_issues(row: pd.Series) -> List[str]:
    """Return a list of issue strings for a single WISDM row."""
    issues: List[str] = []

    if pd.isna(row.get("user")):
        issues.append("missing_user")
    if pd.isna(row.get("activity")) or row["activity"] not in VALID_ACTIVITIES:
        issues.append(f"invalid_activity({row.get('activity')})")
    for ax in ("x", "y", "z"):
        if pd.isna(row.get(ax)):
            issues.append(f"missing_{ax}")
    if pd.isna(row.get("timestamp")):
        issues.append("missing_timestamp")

    if all(pd.notna(row.get(ax)) for ax in ("x", "y", "z")):
        mag = float(np.sqrt(row["x"] ** 2 + row["y"] ** 2 + row["z"] ** 2))
        if mag > ACCEL_MAG_CEILING:
            issues.append(f"extreme_magnitude({mag:.1f})")

    return issues


def validate_schema(df: pd.DataFrame) -> set[str]:
    """Return the set of missing required columns (empty = ok)."""
    return set(REQUIRED_COLS) - set(df.columns)


#  Anomaly detection 

def detect_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Flag rows whose acceleration magnitude exceeds mean + 5σ.

    Returns a summary dict (never raises — callers decide what to do).
    """
    mag = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    mean_mag = float(mag.mean())
    std_mag = float(mag.std())
    threshold = mean_mag + 5 * std_mag
    count = int((mag > threshold).sum())
    return {
        "anomaly_count": count,
        "threshold": round(threshold, 4),
        "mean_magnitude": round(mean_mag, 4),
        "std_magnitude": round(std_mag, 4),
    }


#  Windowing 

def create_row_windows(df: pd.DataFrame, window_size: int = 200) -> pd.DataFrame:
    """
    Aggregate rows into fixed-size non-overlapping windows.

    Each window retains the first user/activity and the mean xyz values.
    """
    df = df.reset_index(drop=True)
    df["window_id"] = df.index // window_size
    windowed = (
        df.groupby("window_id")
        .agg(
            user=("user", "first"),
            activity=("activity", "first"),
            x=("x", "mean"),
            y=("y", "mean"),
            z=("z", "mean"),
        )
        .reset_index(drop=True)
    )
    return windowed


#  Stamina engine 

def compute_stamina(
    df: pd.DataFrame,
    max_stamina: float = 100.0,
    fatigue_rate: float = 0.01,
) -> pd.DataFrame:
    """
    Simulate cumulative stamina drain row by row.

    Stamina decreases by `intensity × fatigue_rate` each window,
    clamped to [0, max_stamina].
    """
    stamina = max_stamina
    values: List[float] = []
    for _, row in df.iterrows():
        intensity = float(np.sqrt(row["x"] ** 2 + row["y"] ** 2 + row["z"] ** 2))
        stamina = max(0.0, stamina - intensity * fatigue_rate)
        values.append(round(stamina, 4))
    out = df.copy()
    out["stamina"] = values
    return out


#  Bias analysis 

def analyze_bias(df: pd.DataFrame) -> Dict[str, float]:
    """Return mean stamina per activity label."""
    return df.groupby("activity")["stamina"].mean().round(4).to_dict()