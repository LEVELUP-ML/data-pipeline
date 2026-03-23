"""Shared fixtures for all test modules."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_wisdm_df():
    """100-row WISDM-like accelerometer dataframe."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "user": [1600] * n,
            "activity": np.random.choice(list("ABCDEFGHIJKLMOPQRS"), n),
            "timestamp": range(1000, 1000 + n),
            "x": np.random.normal(0.1, 0.5, n),
            "y": np.random.normal(-0.2, 0.5, n),
            "z": np.random.normal(9.8, 0.3, n),
        }
    )


@pytest.fixture
def sample_weightlifting_df():
    """Small weightlifting dataframe."""
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01"] * 4 + ["2024-01-02"] * 4),
            "Workout Name": ["Push"] * 4 + ["Pull"] * 4,
            "Exercise Name": [
                "Bench",
                "Bench",
                "OHP",
                "OHP",
                "Deadlift",
                "Deadlift",
                "Row",
                "Row",
            ],
            "Set Order": pd.array([1, 2, 1, 2, 1, 2, 1, 2], dtype="Int64"),
            "Weight": [80.0, 82.5, 40.0, 42.5, 120.0, 125.0, 60.0, 62.5],
            "Reps": pd.array([8, 6, 10, 8, 5, 4, 10, 8], dtype="Int64"),
            "Distance": [None] * 8,
            "Seconds": pd.array([None] * 8, dtype="Int64"),
            "_source_file": ["test.csv"] * 8,
        }
    )
