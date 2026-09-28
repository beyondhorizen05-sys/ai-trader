"""Tests for src.risk.sizing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.sizing import (
    apply_max_drawdown_brake,
    atr_stop,
    fixed_fraction,
    vol_target,
)


def _pos(vals, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="D", name="date")
    return pd.Series(vals, index=idx, dtype="int8", name="position")


def _rets(vals):
    idx = pd.date_range("2024-01-01", periods=len(vals), freq="D", name="date")
    return pd.Series(vals, index=idx, dtype="float64")


# ---------------------------------------------------------------------------
# fixed_fraction
# ---------------------------------------------------------------------------

def test_fixed_fraction_halves_positions():
    p = _pos([1, -1, 1, 0])
    out = fixed_fraction(p, 0.5)
    assert list(out) == [0.5, -0.5, 0.5, 0.0]


def test_fixed_fraction_rejects_bad_fraction():
    with pytest.raises(ValueError):
        fixed_fraction(_pos([1]), 0.0)
    with pytest.raises(ValueError):
        fixed_fraction(_pos([1]), 1.5)


# ---------------------------------------------------------------------------
# vol_target
# ---------------------------------------------------------------------------

def test_vol_target_shrinks_position_when_vol_high():
    """High realized vol -> smaller leverage -> smaller position."""
    rng = np.random.default_rng(0)
    p = _pos([1] * 60)
    low_vol = _rets(rng.normal(0.0, 0.001, 60))     # ~0.1% daily
    high_vol = _rets(rng.normal(0.0, 0.05, 60))     # ~5% daily
    lo = vol_target(p, low_vol, target_vol=0.15, window=20)
    hi = vol_target(p, high_vol, target_vol=0.15, window=20)

    assert hi.iloc[-1] > 0
    assert lo.iloc[-1] > 0
    assert hi.iloc[-1] < lo.iloc[-1]


def test_vol_target_respects_max_leverage():
    rng = np.random.default_rng(1)
    p = _pos([1] * 60)
    tiny = _rets(rng.normal(0.0, 1e-6, 60))
    out = vol_target(p, tiny, target_vol=0.15, window=20, max_leverage=0.8)
    assert out.iloc[-1] <= 0.8 + 1e-9


def test_vol_target_validates_inputs():
    with pytest.raises(ValueError):
        vol_target(_pos([1]), _rets([0.01]), target_vol=0)
    with pytest.raises(ValueError):
        vol_target(_pos([1]), _rets([0.01]), max_leverage=0)


def test_vol_target_warmup_is_flat():
    """During warm-up (before `window` returns accumulated), position is 0."""
    rng = np.random.default_rng(2)
    p = _pos([1] * 40)
    r = _rets(rng.normal(0.0, 0.01, 40))
    out = vol_target(p, r, window=20)
    # shift(1) + min_periods=20 -> first 20 bars are 0
    assert (out.iloc[:20] == 0).all()
    assert (out.iloc[20:] > 0).any()


# ---------------------------------------------------------------------------
# atr_stop
# ---------------------------------------------------------------------------

def test_atr_stop_zeroes_on_adverse_move_and_stays_flat():
    """After the adverse move trips the stop, position remains 0 until the
    raw signal would naturally flip. In this test the raw signal is always 1,
    so once stopped, it stays stopped."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", name="date")
    p = pd.Series([1, 1, 1, 1, 1], index=idx, dtype="int8", name="position")
    close = pd.Series([100, 100, 90, 90, 90], index=idx)
    atr = pd.Series([1.0] * 5, index=idx)
    out = atr_stop(p, close, atr, multiple=2.0)

    # Before the crash: still long
    assert out.iloc[0] == 1
    assert out.iloc[1] == 1
    # Bar 2 is the crash -> stop trips
    assert out.iloc[2] == 0
    # Bars 3-4: no new signal change, so stop stays latched
    assert out.iloc[3] == 0
    assert out.iloc[4] == 0


def test_atr_stop_keeps_position_within_threshold():
    idx = pd.date_range("2024-01-01", periods=5, freq="D", name="date")
    p = pd.Series([1, 1, 1, 1, 1], index=idx, dtype="int8", name="position")
    close = pd.Series([100, 100.5, 100.3, 100.1, 100.2], index=idx)
    atr = pd.Series([1.0] * 5, index=idx)
    out = atr_stop(p, close, atr, multiple=2.0)
    assert out.iloc[-1] == 1


def test_atr_stop_resets_when_signal_flips():
    """If the raw strategy signal changes, the stop should reset so a fresh
    position can be opened."""
    idx = pd.date_range("2024-01-01", periods=8, freq="D", name="date")
    # Raw signal: long, long, long, long, flat, flat, long, long
    p = pd.Series([1, 1, 1, 1, 0, 0, 1, 1], index=idx, dtype="int8", name="position")
    # Crash on bar 2
    close = pd.Series([100, 100, 90, 90, 90, 90, 90, 90], index=idx)
    atr = pd.Series([1.0] * 8, index=idx)
    out = atr_stop(p, close, atr, multiple=2.0)

    assert out.iloc[2] == 0                     # stop tripped
    assert out.iloc[3] == 0                     # still latched
    assert out.iloc[4] == 0                     # signal went flat
    # On bar 6 raw signal is 1 again -> should be allowed back in
    # (close is flat 90->90 so no new adverse move)
    assert out.iloc[6] == 1


def test_atr_stop_rejects_bad_multiple():
    p = _pos([1, 1, 1])
    c = pd.Series([100.0, 100.0, 100.0])
    a = pd.Series([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        atr_stop(p, c, a, multiple=0)


# ---------------------------------------------------------------------------
# apply_max_drawdown_brake
# ---------------------------------------------------------------------------

def test_max_drawdown_brake_triggers_and_cools_down():
    n = 50
    r = _rets([-0.05] * n)
    p = _pos([1] * n)
    out = apply_max_drawdown_brake(p, r, max_dd=0.10, cooldown=10)
    assert (out == 0).sum() > 0
    assert out.iloc[0] == 1


def test_max_drawdown_brake_never_engages_on_flat_returns():
    n = 30
    r = _rets([0.0] * n)
    p = _pos([1] * n)
    out = apply_max_drawdown_brake(p, r, max_dd=0.10, cooldown=5)
    assert (out == 1).all()


def test_max_drawdown_brake_rejects_bad_max_dd():
    p = _pos([1, 1, 1])
    r = _rets([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        apply_max_drawdown_brake(p, r, max_dd=0.0)
    with pytest.raises(ValueError):
        apply_max_drawdown_brake(p, r, max_dd=1.0)