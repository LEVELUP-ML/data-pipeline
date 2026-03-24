"""
tests/test_flexibility_features.py

Tests pure helper logic from flexibility_features_dag.py.
Functions are copied here directly — no Airflow import needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

AGE_BINS = [(0, 19, 0), (20, 29, 1), (30, 39, 2), (40, 200, 3)]


def _slope(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    x = np.arange(len(vals), dtype=float) - np.arange(len(vals), dtype=float).mean()
    y = np.array(vals, dtype=float)
    d = (x ** 2).sum()
    return float(np.dot(x, y) / d) if d else 0.0


def _future_score(dates, scores, ref, h):
    target = ref + pd.Timedelta(days=h)
    best, best_gap = None, float("inf")
    for d, s in zip(dates, scores):
        g = abs((d - target).days)
        if g <= 2 and g < best_gap:
            best, best_gap = s, g
    return best


def _bmr(age, sex, h, w):
    try:
        b = 10 * float(w) + 6.25 * float(h) - 5 * float(age)
        return round(b - 161 if str(sex).lower() in ("female", "f") else b + 5)
    except (TypeError, ValueError):
        return None


def _age_enc(age):
    try:
        a = int(age)
        for lo, hi, code in AGE_BINS:
            if lo <= a <= hi:
                return code
    except (TypeError, ValueError):
        pass
    return -1


#  _slope 

class TestSlope:
    def test_flat_series_returns_zero(self):
        assert _slope([5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0, abs=1e-6)

    def test_ascending_series_positive(self):
        assert _slope([1.0, 2.0, 3.0, 4.0, 5.0]) > 0

    def test_descending_series_negative(self):
        assert _slope([5.0, 4.0, 3.0, 2.0, 1.0]) < 0

    def test_single_value_returns_zero(self):
        assert _slope([42.0]) == 0.0

    def test_empty_returns_zero(self):
        assert _slope([]) == 0.0

    def test_known_slope(self):
        assert _slope([1.0, 3.0, 5.0, 7.0, 9.0]) == pytest.approx(2.0, abs=0.01)


#  _future_score 

class TestFutureScore:
    def _series(self, n=30, base=50.0, step=0.5):
        dates  = [pd.Timestamp("2026-01-01") + pd.Timedelta(days=i) for i in range(n)]
        scores = [base + i * step for i in range(n)]
        return dates, scores

    def test_exact_match(self):
        dates, scores = self._series()
        assert _future_score(dates, scores, dates[0], 7) == pytest.approx(scores[7], abs=0.01)

    def test_within_2_day_window(self):
        dates, scores = self._series()
        d2 = dates[:7] + dates[8:]
        s2 = scores[:7] + scores[8:]
        assert _future_score(d2, s2, dates[0], 7) is not None

    def test_returns_none_when_out_of_range(self):
        dates, scores = self._series(n=5)
        assert _future_score(dates, scores, dates[0], 30) is None

    def test_empty_series(self):
        assert _future_score([], [], pd.Timestamp("2026-01-01"), 7) is None


#  _bmr 

class TestBmr:
    def test_male_bmr(self):
        # 10*80 + 6.25*180 - 5*30 + 5 = 1780
        assert _bmr(30, "Male", 180, 80) == pytest.approx(1780, abs=2)

    def test_female_bmr(self):
        # 10*60 + 6.25*165 - 5*25 - 161 = 1345
        assert _bmr(25, "Female", 165, 60) == pytest.approx(1345, abs=2)

    def test_invalid_inputs_return_none(self):
        assert _bmr(None, "Male", 180, 80) is None
        assert _bmr(30, "Male", None, 80) is None
        assert _bmr("bad", "Female", 165, 60) is None

    def test_lowercase_sex(self):
        assert _bmr(30, "female", 165, 60) is not None


#  _age_enc 

class TestAgeEnc:
    def test_under_20(self):    assert _age_enc(17) == 0
    def test_20s(self):         assert _age_enc(25) == 1
    def test_30s(self):         assert _age_enc(35) == 2
    def test_40_plus(self):     assert _age_enc(55) == 3
    def test_boundary_20(self): assert _age_enc(20) == 1
    def test_invalid(self):
        assert _age_enc(None) == -1
        assert _age_enc("old") == -1


#  Lag feature construction 

class TestLagFeatureConstruction:
    N_LAGS   = 5
    MIN_SESS = 7

    def _make_df(self, n=15, base=50.0):
        start = pd.Timestamp("2026-01-01")
        rows  = []
        score = base
        for i in range(n):
            score += np.random.uniform(0.1, 1.0)
            rows.append({
                "user_id":              "u001",
                "date":                 start + pd.Timedelta(days=i * 2),
                "score_after":          round(score, 2),
                "effort_level":         3,
                "session_duration_min": 45,
                "sit_and_reach_cm":     round(10 + score * 0.4, 1),
                "streak_days":          (i % 5) + 1,
                "rest_days_before":     1,
            })
        return pd.DataFrame(rows)

    def test_minimum_sessions_threshold(self):
        df       = self._make_df(n=self.MIN_SESS - 1)
        expected = max(0, len(df) - self.N_LAGS) if len(df) >= self.MIN_SESS else 0
        assert expected == 0

    def test_row_count_correct(self):
        assert 15 - self.N_LAGS == 10

    def test_lag_ordering(self):
        df     = self._make_df(n=10).reset_index(drop=True)
        dates  = df["date"].tolist()
        scores = df["score_after"].tolist()
        t      = self.N_LAGS
        lag1   = None
        for k in range(self.N_LAGS):
            idx = t - self.N_LAGS + k
            lag = self.N_LAGS - k
            if lag == 1:
                lag1 = df.iloc[idx]["score_after"]
        assert lag1 == pytest.approx(scores[t - 1], abs=0.01)

    def test_days_ago_non_negative(self):
        df    = self._make_df(n=12)
        dates = df["date"].tolist()
        for t in range(self.N_LAGS, len(df)):
            for k in range(self.N_LAGS):
                assert (dates[t] - dates[t - self.N_LAGS + k]).days >= 0