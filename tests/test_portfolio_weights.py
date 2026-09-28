"""Tests for run_portfolio_weights (cross-sectional path)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestConfig
from src.backtest.portfolio import run_portfolio_weights


def _prices(symbols: list[str], n: int = 100, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B", name="date")
    out = {}
    for s in symbols:
        rets = rng.normal(0.001, 0.01, size=n)
        close = 100 * np.exp(np.cumsum(rets))
        out[s] = pd.DataFrame(
            {
                "open": close, "high": close * 1.01, "low": close * 0.99,
                "close": close, "adj close": close,
                "volume": np.full(n, 1_000_000, dtype="int64"),
            },
            index=idx,
        )
    return out


def test_weights_zero_gives_flat_equity():
    prices = _prices(["A", "B", "C"])
    weights = pd.DataFrame(0.0, index=list(prices.values())[0].index, columns=["A", "B", "C"])
    result = run_portfolio_weights(prices, weights, BacktestConfig(commission_bps=0, slippage_bps=0))
    assert (result.equity == result.equity.iloc[0]).all()
    assert result.metrics["total_return"] == pytest.approx(0.0)


def test_weights_reject_empty():
    prices = _prices(["A"])
    with pytest.raises(ValueError):
        run_portfolio_weights(prices, pd.DataFrame())


def test_weights_align_to_common_index():
    prices = _prices(["A", "B"], n=50)
    weights = pd.DataFrame(0.5, index=list(prices.values())[0].index, columns=["A", "B"])
    result = run_portfolio_weights(prices, weights)
    assert not result.equity.empty
    assert result.weights.shape[1] == 2