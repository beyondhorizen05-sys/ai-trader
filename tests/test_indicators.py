"""Unit tests for src.features.indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.indicators import (
    add_all,
    atr,
    bollinger,
    ema,
    macd,
    returns,
    rolling_vol,
    rsi,
    sma,
    true_range,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prices() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=30, freq="D", name="date")
    vals = 100 + np.arange(30) * 0.5 + np.sin(np.arange(30)) * 2
    return pd.Series(vals, index=idx, name="close")


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=60, freq="D", name="date")
    close = pd.Series(100 + np.arange(60) * 0.3 + np.sin(np.arange(60)), index=idx)
    high = close + 1.5
    low = close - 1.5
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(1_000_000, index=idx, dtype="int64")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj close": close,
            "volume": volume,
        }
    )


# ---------------------------------------------------------------------------
# returns
# ---------------------------------------------------------------------------

def test_returns_simple_matches_manual(prices):
    r = returns(prices, kind="simple")
    assert r.iloc[0] != r.iloc[0]  # NaN
    expected = prices.iloc[1] / prices.iloc[0] - 1
    assert r.iloc[1] == pytest.approx(expected)


def test_returns_log_matches_manual(prices):
    r = returns(prices, kind="log")
    expected = np.log(prices.iloc[1] / prices.iloc[0])
    assert r.iloc[1] == pytest.approx(expected)


def test_returns_rejects_bad_kind(prices):
    with pytest.raises(ValueError, match="simple.*log"):
        returns(prices, kind="banana")


# ---------------------------------------------------------------------------
# moving averages
# ---------------------------------------------------------------------------

def test_sma_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, window=3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[3] == pytest.approx(3.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_ema_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ema(s, window=3)
    # min_periods=3 -> first non-NaN at index 2, but the recursion
    # seeds from x_0 (not from the SMA), so values differ from SMA-seeded EMA.
    assert out.iloc[:2].isna().all()
    # alpha = 2/(3+1) = 0.5 ; adjust=False seeds with x_0 = 1.0
    assert out.iloc[2] == pytest.approx(2.25)     # 0.5*3 + 0.5*1.5
    assert out.iloc[3] == pytest.approx(3.125)    # 0.5*4 + 0.5*2.25
    assert out.iloc[4] == pytest.approx(4.0625)   # 0.5*5 + 0.5*3.125


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1.0, 31.0))
    out = rsi(s, window=14)
    assert out.iloc[-1] == pytest.approx(100.0)


def test_rsi_all_losses_is_0():
    s = pd.Series(np.arange(30.0, 0.0, -1.0))
    out = rsi(s, window=14)
    assert out.iloc[-1] == pytest.approx(0.0)


def test_rsi_range_and_warmup(prices):
    out = rsi(prices, window=14)
    assert out.iloc[:14].isna().all()
    assert pd.notna(out.iloc[14])                 # first valid at index 14
    valid = out.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


# ---------------------------------------------------------------------------
# True Range / ATR
# ---------------------------------------------------------------------------

def test_true_range_uses_prev_close():
    high = pd.Series([10.0, 12.0, 11.0])
    low = pd.Series([8.0, 9.0, 10.0])
    close = pd.Series([9.0, 11.0, 10.5])
    tr = true_range(high, low, close)
    assert tr.iloc[0] == pytest.approx(2.0)       # 10 - 8 ; no prev close
    assert tr.iloc[1] == pytest.approx(3.0)       # max(3, |12-9|, |9-9|) = 3
    assert tr.iloc[2] == pytest.approx(1.0)       # max(1, |11-11|, |10-11|) = 1


def test_atr_positive_and_warmup(ohlcv):
    out = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], window=14)
    # min_periods=14 -> first valid value is at index 13 (the 14th bar).
    assert out.iloc[:13].isna().all()
    assert pd.notna(out.iloc[13])
    assert (out.dropna() > 0).all()


# ---------------------------------------------------------------------------
# Bollinger
# ---------------------------------------------------------------------------

def test_bollinger_bands_ordering(prices):
    bb = bollinger(prices, window=20, num_std=2.0)
    valid = bb.dropna()
    assert (valid["bb_upper"] >= valid["bb_mid"]).all()
    assert (valid["bb_mid"] >= valid["bb_lower"]).all()


def test_bollinger_mid_equals_sma(prices):
    bb = bollinger(prices, window=20)
    expected = sma(prices, 20)
    pd.testing.assert_series_equal(
        bb["bb_mid"].dropna(), expected.dropna(), check_names=False
    )


def test_bollinger_columns_present(prices):
    bb = bollinger(prices)
    assert list(bb.columns) == ["bb_mid", "bb_upper", "bb_lower", "bb_pctb", "bb_width"]


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def test_macd_columns_and_hist_consistency(prices):
    m = macd(prices)
    assert list(m.columns) == ["macd", "macd_signal", "macd_hist"]
    diff = (m["macd"] - m["macd_signal"]) - m["macd_hist"]
    assert diff.abs().max() < 1e-12


# ---------------------------------------------------------------------------
# add_all
# ---------------------------------------------------------------------------

def test_add_all_preserves_index_and_adds_columns(ohlcv):
    out = add_all(ohlcv)
    assert out.index.equals(ohlcv.index)
    for col in ["ret_log", "ret_simple", "vol_20", "sma_20", "ema_20",
                "rsi_14", "atr_14", "bb_mid", "macd", "macd_signal"]:
        assert col in out.columns, f"missing {col}"


def test_add_all_does_not_mutate_input(ohlcv):
    before = ohlcv.copy()
    _ = add_all(ohlcv)
    pd.testing.assert_frame_equal(ohlcv, before)


def test_add_all_custom_windows(ohlcv):
    out = add_all(ohlcv, windows=(5, 15))
    assert "sma_5" in out.columns
    assert "ema_15" in out.columns
    assert "sma_20" not in out.columns


def test_rolling_vol_shape_and_warmup(prices):
    v = rolling_vol(prices, window=10)
    assert v.iloc[:10].isna().all()
    assert (v.dropna() > 0).all()