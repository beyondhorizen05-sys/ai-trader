"""Unit tests for src.strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.base import Strategy
from src.strategies.sma_cross import SmaCross


def _features(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq="D", name="date")
    s = pd.Series(closes, index=idx, dtype="float64")
    return pd.DataFrame({"close": s})


# ---------------------------------------------------------------------------
# base class contract
# ---------------------------------------------------------------------------

def test_base_is_abstract():
    with pytest.raises(TypeError):
        Strategy()  # type: ignore[abstract]


def test_validate_output_rejects_wrong_index():
    class Bad(Strategy):
        name = "bad"

        def generate_signals(self, df):
            return pd.Series([0, 0, 0], name="position")  # no index

    df = _features([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="index"):
        Bad()._validate_output(df, Bad().generate_signals(df))


def test_validate_output_rejects_bad_values():
    class Bad(Strategy):
        name = "bad"

        def generate_signals(self, df):
            return pd.Series([0, 2, 0], index=df.index, name="position")

    df = _features([1.0, 2.0, 3.0])
    pos = Bad().generate_signals(df)
    with pytest.raises(ValueError, match="invalid position values"):
        Bad()._validate_output(df, pos)


# ---------------------------------------------------------------------------
# SmaCross construction
# ---------------------------------------------------------------------------

def test_sma_cross_rejects_fast_ge_slow():
    with pytest.raises(ValueError, match="fast.*slow"):
        SmaCross(fast=50, slow=20)
    with pytest.raises(ValueError, match="fast.*slow"):
        SmaCross(fast=20, slow=20)


def test_sma_cross_requires_close_column():
    df = pd.DataFrame({"open": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="close"):
        SmaCross(fast=2, slow=3).generate_signals(df)


# ---------------------------------------------------------------------------
# SmaCross behaviour
# ---------------------------------------------------------------------------

def test_sma_cross_warmup_is_flat():
    df = _features([10.0] * 30)
    pos = SmaCross(fast=5, slow=10).generate_signals(df)
    # first non-NaN of slow is at index slow-1 = 9
    assert (pos.iloc[:9] == 0).all()
    # after warm-up, fast == slow -> neither > nor <, so stays flat
    assert (pos.iloc[9:] == 0).all()


def test_sma_cross_long_when_fast_above_slow():
    # strong uptrend -> fast > slow after warm-up
    closes = list(np.linspace(100, 200, 60))
    df = _features(closes)
    pos = SmaCross(fast=5, slow=20, allow_short=False).generate_signals(df)

    assert pos.iloc[:19].eq(0).all()          # warm-up flat
    tail = pos.iloc[25:]
    assert tail.eq(1).all()                    # fully long once trend is established


def test_sma_cross_flat_when_fast_below_slow_and_short_disabled():
    # strong downtrend, long-only mode -> flat everywhere
    closes = list(np.linspace(200, 100, 60))
    df = _features(closes)
    pos = SmaCross(fast=5, slow=20, allow_short=False).generate_signals(df)
    assert pos.eq(0).all()


def test_sma_cross_short_when_fast_below_slow_and_short_enabled():
    closes = list(np.linspace(200, 100, 60))
    df = _features(closes)
    pos = SmaCross(fast=5, slow=20, allow_short=True).generate_signals(df)
    assert pos.iloc[:19].eq(0).all()
    assert pos.iloc[25:].eq(-1).all()


def test_sma_cross_positions_are_int8_and_named():
    df = _features(list(np.linspace(100, 200, 40)))
    pos = SmaCross(fast=5, slow=10).generate_signals(df)
    assert pos.name == "position"
    assert pos.dtype == "int8"


def test_sma_cross_preserves_index():
    df = _features(list(np.linspace(100, 200, 40)))
    pos = SmaCross(fast=5, slow=10).generate_signals(df)
    assert pos.index.equals(df.index)


def test_sma_cross_does_not_mutate_input():
    df = _features(list(np.linspace(100, 200, 40)))
    before = df.copy()
    SmaCross(fast=5, slow=10).generate_signals(df)
    pd.testing.assert_frame_equal(df, before)


def test_sma_cross_detects_crossover():
    """Construct prices that go up then down; the position should flip."""
    up = list(np.linspace(100, 200, 40))
    down = list(np.linspace(200, 100, 40))
    df = _features(up + down)
    pos = SmaCross(fast=5, slow=20, allow_short=True).generate_signals(df)

    # During the rise, eventually long.
    assert (pos.iloc[25:35] == 1).all()
    # During the fall, eventually short.
    assert (pos.iloc[-5:] == -1).all()