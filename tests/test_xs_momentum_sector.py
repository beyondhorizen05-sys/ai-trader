"""Tests for the sector-neutral path in XSMomentum."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.xs_momentum import XSMomentum


def _prices(symbols: list[str], n: int = 200, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B", name="date")
    out = {}
    for i, s in enumerate(symbols):
        drift = 0.001 * (i - len(symbols) // 2)
        rets = rng.normal(drift, 0.005, size=n)
        close = 100 * np.exp(np.cumsum(rets))
        out[s] = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "adj close": close,
             "volume": np.full(n, 1_000_000, dtype="int64")},
            index=idx,
        )
    return out


def test_skip_recent_validation():
    with pytest.raises(ValueError):
        XSMomentum(skip_recent=-1)
    with pytest.raises(ValueError):
        XSMomentum(lookback=20, skip_recent=20)


def test_skip_recent_changes_signal():
    prices = _prices(["A", "B", "C", "D", "E", "F"])
    w0 = XSMomentum(lookback=40, top_n=2, bottom_n=0, min_history=6,
                    skip_recent=0).generate_weights(prices)
    w1 = XSMomentum(lookback=40, top_n=2, bottom_n=0, min_history=6,
                    skip_recent=10).generate_weights(prices)
    assert not w0.equals(w1)


def test_sector_neutral_respects_sector_boundaries():
    """Force 2 sectors of 3 symbols each. Long 2 total should be 1 per sector."""
    symbols = ["A1", "A2", "A3", "B1", "B2", "B3"]
    sectors = {"A1": "A", "A2": "A", "A3": "A", "B1": "B", "B2": "B", "B3": "B"}
    prices = _prices(symbols, n=200, seed=1)
    strat = XSMomentum(lookback=40, top_n=2, bottom_n=0, min_history=6,
                       sectors=sectors)
    w = strat.generate_weights(prices)

    tail = w.iloc[-30:]
    # On bars where positions are non-zero, at most one of {A1, A2, A3} should
    # be long and at most one of {B1, B2, B3}.
    a_cols = ["A1", "A2", "A3"]
    b_cols = ["B1", "B2", "B3"]
    a_count = (tail[a_cols] > 0).sum(axis=1)
    b_count = (tail[b_cols] > 0).sum(axis=1)
    assert (a_count <= 1).all()
    assert (b_count <= 1).all()