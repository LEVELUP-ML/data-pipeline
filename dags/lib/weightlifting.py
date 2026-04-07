"""
dags/lib/weightlifting.py

Pure functions for weightlifting CSV cleaning and validation.
No Airflow imports — importable in tests without a running scheduler.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

import pandas as pd

REQUIRED_COLS: Set[str] = {
    "Date",
    "Workout Name",
    "Exercise Name",
    "Set Order",
    "Weight",
    "Reps",
}

MAX_WEIGHT_KG: float = 700.0
MAX_REPS: int = 200
MAX_SET_ORDER: int = 100
MAX_SECONDS: int = 36_000
MAX_DISTANCE_M: int = 100_000

DEDUP_COLS: List[str] = [
    "Date",
    "Workout Name",
    "Exercise Name",
    "Set Order",
    "Weight",
    "Reps",
]


#  Validation 

def tag_row_issues(row: pd.Series) -> List[str]:
    """Return a list of issue strings for a single weightlifting row."""
    issues: List[str] = []

    if pd.isna(row.get("_parsed_date")):
        issues.append("invalid_or_missing_date")
    if pd.isna(row.get("Exercise Name")) or str(row["Exercise Name"]).strip() == "":
        issues.append("missing_exercise_name")
    if pd.isna(row.get("Workout Name")) or str(row["Workout Name"]).strip() == "":
        issues.append("missing_workout_name")

    so = row.get("Set Order")
    if pd.isna(so):
        issues.append("missing_set_order")
    elif int(so) < 1 or int(so) > MAX_SET_ORDER:
        issues.append(f"set_order_out_of_range({so})")

    w = row.get("Weight")
    if pd.notna(w):
        if w < 0:
            issues.append(f"negative_weight({w})")
        elif w > MAX_WEIGHT_KG:
            issues.append(f"weight_exceeds_max({w})")

    r = row.get("Reps")
    if pd.notna(r):
        if int(r) < 0:
            issues.append(f"negative_reps({r})")
        elif int(r) > MAX_REPS:
            issues.append(f"reps_exceeds_max({r})")

    s = row.get("Seconds")
    if pd.notna(s):
        if int(s) < 0:
            issues.append(f"negative_seconds({s})")
        elif int(s) > MAX_SECONDS:
            issues.append(f"seconds_exceeds_max({s})")

    d = row.get("Distance")
    if pd.notna(d):
        if d < 0:
            issues.append(f"negative_distance({d})")
        elif d > MAX_DISTANCE_M:
            issues.append(f"distance_exceeds_max({d})")

    return issues


def validate_schema(df: pd.DataFrame) -> Set[str]:
    """Return set of missing required columns (empty = ok)."""
    return REQUIRED_COLS - set(df.columns)


#  Cleaning 

def cast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce all columns to the expected dtypes.

    Mutates a copy — does not modify the caller's DataFrame.
    """
    df = df.copy()

    # Strip whitespace from every string column
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()

    df["_parsed_date"] = pd.to_datetime(
        df["Date"], format="mixed", dayfirst=False, errors="coerce"
    )
    for col in ("Weight", "Distance"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("Set Order", "Reps", "Seconds"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def clean_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Full cleaning pipeline: cast → dedup → tag issues → split clean/rejected.

    Returns:
        {
            "clean": pd.DataFrame,
            "rejected": pd.DataFrame,
            "duplicates_dropped": int,
            "reject_pct": float,
        }
    """
    df = cast_columns(df)

    before = len(df)
    df = df.drop_duplicates(subset=DEDUP_COLS)
    dupes = before - len(df)

    df["_issues"] = df.apply(tag_row_issues, axis=1)
    df["_is_clean"] = df["_issues"].apply(lambda x: len(x) == 0)

    clean = df[df["_is_clean"]].copy()
    rejected = df[~df["_is_clean"]].copy()

    clean["Date"] = clean["_parsed_date"]
    clean = clean.drop(columns=["_parsed_date", "_issues", "_is_clean"])
    clean = clean.sort_values(
        ["Date", "Workout Name", "Exercise Name", "Set Order"]
    ).reset_index(drop=True)

    total = before
    return {
        "clean": clean,
        "rejected": rejected,
        "duplicates_dropped": dupes,
        "reject_pct": round(len(rejected) / total * 100, 2) if total else 0.0,
    }