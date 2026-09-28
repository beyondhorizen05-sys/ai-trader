"""Tests for src.backtest.portfolio."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestConfig
from src.backtest.portfolio import run_portfolio
from src.strategies.sma_cross import SmaCross


def _prices(n: int, start: float = 100.0, drift: float = 0.001,
            seed: int = 1, symbol_offset: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.01, size=n)
    close = (start + symbol_offset) * np.exp(np.cumsum(rets))
    idx = pd.date_range("2023-01-01", periods=n, freq="B", name="date")
    return pd.DataFrame(
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


def test_portfolio_runs_single_symbol():
    data = {"AAPL": _prices(300)}
    result = run_portfolio(
        data,
        strategy_factory=lambda sym: SmaCross(fast=10, slow=30),
        config=BacktestConfig(),
    )
    assert not result.equity.empty
    assert result.equity.iloc[0] == pytest.approx(
        100_000.0 * (1 + result.returns.iloc[0])
    )
    assert set(result.per_symbol) == {"AAPL"}


def test_portfolio_runs_multi_symbol():
    data = {
        "AAPL": _prices(300, seed=1),
        "MSFT": _prices(300, seed=2),
        "SPY":  _prices(300, seed=3),
    }
    result = run_portfolio(data, strategy_factory=lambda sym: SmaCross(fast=10, slow=30))
    assert set(result.per_symbol) == {"AAPL", "MSFT", "SPY"}
    assert result.weights.shape[1] == 3
    row_sums = result.weights.sum(axis=1)
    assert ((row_sums == 0) | np.isclose(row_sums, 1.0)).all()


def test_portfolio_weights_zero_when_flat():
    data = {"AAPL": _prices(300)}
    result = run_portfolio(data, strategy_factory=lambda sym: SmaCross(fast=10, slow=30))
    flat_bars = result.weights.sum(axis=1) == 0
    assert flat_bars.any()
    assert (result.returns[flat_bars].abs() < 1e-12).all()


def test_portfolio_rejects_empty_input():
    with pytest.raises(ValueError):
        run_portfolio({}, strategy_factory=lambda s: SmaCross())


def test_portfolio_metrics_keys():
    data = {"AAPL": _prices(300), "MSFT": _prices(300, seed=2)}
    result = run_portfolio(data, strategy_factory=lambda s: SmaCross(fast=10, slow=30))
    for k in ["total_return", "cagr", "sharpe", "max_drawdown", "exposure"]:
        assert k in result.metrics