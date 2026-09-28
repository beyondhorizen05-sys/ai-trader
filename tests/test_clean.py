"""Unit tests for src.data.clean."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.clean import normalize, resample_ohlcv, validate


def _df(**overrides) -> pd.DataFrame:
    """Build a valid 3-row OHLCV DataFrame; overrides patch specific cells."""
    idx = pd.date_range("2024-01-02", periods=3, freq="D", name="date")
    base = pd.DataFrame(
        {
            "open":      [100.0, 101.0, 102.0],
            "high":      [105.0, 106.0, 107.0],
            "low":       [ 99.0, 100.0, 101.0],
            "close":     [104.0, 105.0, 106.0],
            "adj close": [104.0, 105.0, 106.0],
            "volume":    [1000, 1100, 1200],
        },
        index=idx,
    )
    for k, v in overrides.items():
        base.loc[base.index[0], k] = v
    return base


# --- validate: happy path ---

def test_validate_passes_on_clean_frame():
    validate(_df(), "TEST")  # should not raise


# --- validate: each failure mode ---

def test_validate_rejects_missing_column():
    df = _df().drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing columns"):
        validate(df, "TEST")


def test_validate_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        validate(_df().iloc[0:0], "TEST")


def test_validate_rejects_non_datetime_index():
    df = _df().reset_index(drop=True)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate(df, "TEST")


def test_validate_rejects_duplicate_dates():
    df = _df()
    df = pd.concat([df, df.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        validate(df, "TEST")


def test_validate_rejects_unsorted_index():
    df = _df().iloc[::-1]
    with pytest.raises(ValueError, match="not sorted"):
        validate(df, "TEST")


def test_validate_rejects_nans():
    df = _df()
    df.loc[df.index[0], "close"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate(df, "TEST")


def test_validate_rejects_high_below_low():
    df = _df()
    df.loc[df.index[0], "high"] = 90.0
    with pytest.raises(ValueError, match="high < low"):
        validate(df, "TEST")


def test_validate_rejects_high_below_close():
    df = _df()
    df.loc[df.index[0], "high"] = 103.0  # close is 104
    with pytest.raises(ValueError, match="high < max"):
        validate(df, "TEST")


def test_validate_rejects_low_above_open():
    df = _df()
    df.loc[df.index[0], "low"] = 101.0  # open is 100
    with pytest.raises(ValueError, match="low > min"):
        validate(df, "TEST")


def test_validate_rejects_negative_volume():
    df = _df()
    df.loc[df.index[0], "volume"] = -1
    with pytest.raises(ValueError, match="negative volume"):
        validate(df, "TEST")


# --- normalize ---

def test_normalize_sorts_and_dedupes():
    df = _df()
    shuffled = pd.concat([df.iloc[[2]], df.iloc[[0]], df.iloc[[1]], df.iloc[[0]]])
    out = normalize(shuffled)
    assert out.index.is_monotonic_increasing
    assert not out.index.has_duplicates
    assert len(out) == 3


def test_normalize_enforces_dtypes():
    df = _df()
    df["volume"] = df["volume"].astype("int32")
    out = normalize(df)
    assert out["volume"].dtype == "int64"
    assert out["close"].dtype == "float64"


def test_normalize_then_validate_accepts_dirty_input():
    """normalize() should repair issues that validate() alone would reject."""
    df = _df()
    dirty = pd.concat([df.iloc[[1]], df.iloc[[0]], df.iloc[[2]]])  # unsorted
    clean = normalize(dirty)
    validate(clean, "TEST")  # should not raise


# --- resample ---

def test_resample_weekly_aggregates_correctly():
    idx = pd.date_range("2024-01-01", periods=10, freq="D", name="date")
    df = pd.DataFrame(
        {
            "open":      np.arange(100.0, 110.0),
            "high":      np.arange(105.0, 115.0),
            "low":       np.arange( 95.0, 105.0),
            "close":     np.arange(101.0, 111.0),
            "adj close": np.arange(101.0, 111.0),
            "volume":    np.arange(1000, 1010),
        },
        index=idx,
    )
    weekly = resample_ohlcv(df, "1W")
    assert weekly.index.name == "date"
    first_week = weekly.iloc[0]
    assert first_week["open"] == 100.0
    assert first_week["close"] == 107.0
    assert first_week["high"] == 111.0
    assert first_week["low"] == 95.0
    assert first_week["volume"] == 1000 + 1001 + 1002 + 1003 + 1004 + 1005 + 1006