"""Tests for src.strategies.xs_momentum."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.xs_momentum import XSMomentum


def _make_prices(symbols: dict[str, float], n: int = 200, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Build price frames where each symbol drifts at a fixed per-bar rate."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B", name="date")
    out = {}
    for i, (sym, drift) in enumerate(symbols.items()):
        rets = rng.normal(drift, 0.005, size=n)
        close = 100 * np.exp(np.cumsum(rets))
        out[sym] = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "adj close": close,
                "volume": np.full(n, 1_000_000, dtype="int64"),
            },
            index=idx,
        )
    return out


def test_xs_momentum_rejects_bad_params():
    with pytest.raises(ValueError):
        XSMomentum(lookback=0)
    with pytest.raises(ValueError):
        XSMomentum(top_n=0)
    with pytest.raises(ValueError):
        XSMomentum(bottom_n=-1)
    with pytest.raises(ValueError):
        XSMomentum(gross_exposure=0)


def test_xs_momentum_long_short_gross_is_one():
    """Long/short configs normalize gross exposure to 1.0 (0.5 long + 0.5 short)."""
    prices = _make_prices({"A": 0.001, "B": 0.002, "C": -0.001, "D": -0.002, "E": 0.0})
    strat = XSMomentum(lookback=20, top_n=2, bottom_n=2, min_history=5)
    w = strat.generate_weights(prices)

    post = w.iloc[30:]
    gross = post.abs().sum(axis=1)
    nonzero = gross[gross > 0]
    assert len(nonzero) > 0
    assert np.allclose(nonzero.values, 1.0, atol=1e-9)


def test_xs_momentum_long_only_gross_is_one():
    prices = _make_prices({"A": 0.001, "B": 0.002, "C": -0.001, "D": -0.002, "E": 0.0})
    strat = XSMomentum(lookback=20, top_n=2, bottom_n=0, min_history=5)
    w = strat.generate_weights(prices)
    post = w.iloc[30:]
    gross = post.abs().sum(axis=1)
    nonzero = gross[gross > 0]
    assert len(nonzero) > 0
    assert np.allclose(nonzero.values, 1.0, atol=1e-9)


def test_xs_momentum_picks_top_drifting_symbols():
    prices = _make_prices(
        {"WIN": 0.003, "MID": 0.001, "LOSE": -0.003, "FLAT": 0.0},
        n=200, seed=1,
    )
    strat = XSMomentum(lookback=20, top_n=1, bottom_n=1, min_history=4)
    w = strat.generate_weights(prices)

    tail = w.iloc[-20:]
    assert (tail["WIN"] > 0).sum() >= 15
    assert (tail["LOSE"] < 0).sum() >= 15


def test_xs_momentum_long_only_when_bottom_n_zero():
    prices = _make_prices({"A": 0.002, "B": 0.001, "C": -0.001})
    strat = XSMomentum(lookback=20, top_n=1, bottom_n=0, min_history=3)
    w = strat.generate_weights(prices)
    assert (w >= -1e-12).all().all()


def test_xs_momentum_skips_warmup():
    prices = _make_prices({f"S{i}": 0.001 * i for i in range(6)}, n=100)
    strat = XSMomentum(lookback=30, top_n=2, bottom_n=2, min_history=5)
    w = strat.generate_weights(prices)
    assert (w.iloc[:31] == 0).all().all()