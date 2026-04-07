"""Tests for WISDM accelerometer data loading.

Logic under test lives in dags/lib/wisdm.py — no inline copies here.
"""

import pandas as pd
import pytest

from lib.wisdm import load_wisdm_file


#  Fixtures 

VALID_LINES = (
    "1600,A,123456789,0.12,-0.34,9.81;\n"
    "1600,A,123456790,0.15,-0.31,9.78;\n"
    "1600,B,123456791,1.20,-2.10,8.50;\n"
)


@pytest.fixture
def valid_file(tmp_path):
    f = tmp_path / "data_1600_accel_phone.txt"
    f.write_text(VALID_LINES)
    return str(f)


@pytest.fixture
def empty_file(tmp_path):
    f = tmp_path / "data_empty.txt"
    f.write_text("")
    return str(f)


@pytest.fixture
def malformed_file(tmp_path):
    content = (
        "1600,A,123456789,0.12,-0.34,9.81;\n"
        "garbage_line_no_commas\n"
        "1600,B,123456791,bad,bad,bad;\n"
        "1600,C,123456792,1.0,-1.0,9.5;\n"
    )
    f = tmp_path / "data_malformed.txt"
    f.write_text(content)
    return str(f)


@pytest.fixture
def missing_values_file(tmp_path):
    content = (
        "1600,A,123456789,0.12,,9.81;\n"
        ",A,123456790,0.15,-0.31,;\n"
        "1600,,123456791,1.20,-2.10,8.50;\n"
    )
    f = tmp_path / "data_missing.txt"
    f.write_text(content)
    return str(f)


#  Tests: single file 


class TestLoadWisdmFile:

    def test_loads_valid_file(self, valid_file):
        df = load_wisdm_file(valid_file)
        assert len(df) == 3
        assert list(df.columns[:6]) == ["user", "activity", "timestamp", "x", "y", "z"]

    def test_strips_semicolons_from_z(self, valid_file):
        df = load_wisdm_file(valid_file)
        assert df["z"].dtype in ("float64", "float32")
        assert df["z"].iloc[0] == pytest.approx(9.81)

    def test_numeric_columns_are_float(self, valid_file):
        df = load_wisdm_file(valid_file)
        for col in ("x", "y", "z"):
            assert pd.api.types.is_float_dtype(df[col])

    def test_user_ids_parsed(self, valid_file):
        df = load_wisdm_file(valid_file)
        assert (df["user"] == 1600).all()

    def test_activity_codes_stripped(self, valid_file):
        df = load_wisdm_file(valid_file)
        assert set(df["activity"]) == {"A", "B"}

    def test_source_file_tracked(self, valid_file):
        df = load_wisdm_file(valid_file)
        assert df["_source_file"].iloc[0] == "data_1600_accel_phone.txt"

    def test_empty_file_returns_empty_df(self, empty_file):
        df = load_wisdm_file(empty_file)
        assert len(df) == 0

    def test_malformed_lines_coerced_to_nan(self, malformed_file):
        df = load_wisdm_file(malformed_file)
        nan_rows = df[df[["x", "y", "z"]].isna().any(axis=1)]
        assert len(nan_rows) >= 1

    def test_missing_values_become_nan(self, missing_values_file):
        df = load_wisdm_file(missing_values_file)
        assert df[["x", "y", "z"]].isna().any().any()


#  Tests: multiple files 


class TestLoadMultipleFiles:

    def test_concat_multiple_files(self, tmp_path):
        for uid in (1600, 1601):
            f = tmp_path / f"data_{uid}_accel_phone.txt"
            f.write_text(
                f"{uid},A,100,0.1,-0.2,9.8;\n{uid},B,101,0.3,-0.4,9.7;\n"
            )

        frames = [load_wisdm_file(str(f)) for f in sorted(tmp_path.glob("*.txt"))]
        combined = pd.concat(frames, ignore_index=True)
        assert len(combined) == 4
        assert set(combined["user"]) == {1600, 1601}

    def test_no_txt_files_produces_empty_list(self, tmp_path):
        frames = [load_wisdm_file(str(f)) for f in tmp_path.glob("*.txt")]
        assert frames == []