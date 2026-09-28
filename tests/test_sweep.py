"""Unit tests for src.backtest.sweep."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestConfig
from src.backtest.sweep import SweepResult, _expand_grid, sweep
from src.strategies.base import Strategy
from src.strategies.sma_cross import SmaCross


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _prices(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="D", name="date")
    # trending-up series with a bit of noise so different SMA windows matter
    rng = np.random.default_rng(42)
    rets = rng.normal(0.001, 0.01, size=n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * 1.01
    low = close * 0.99
    open_ = close * 0.999
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj close": close,      # no split/dividend
            "volume": np.full(n, 1_000_000, dtype="int64"),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# _expand_grid
# ---------------------------------------------------------------------------

def test_expand_grid_single_key():
    out = _expand_grid({"a": [1, 2]})
    assert out == [{"a": 1}, {"a": 2}]


def test_expand_grid_cartesian_product():
    out = _expand_grid({"a": [1, 2], "b": [10, 20]})
    assert len(out) == 4
    assert {"a": 1, "b": 10} in out
    assert {"a": 2, "b": 20} in out


def test_expand_grid_empty():
    assert _expand_grid({}) == []


# ---------------------------------------------------------------------------
# sweep — happy path
# ---------------------------------------------------------------------------

def test_sweep_runs_all_valid_combos():
    prices = _prices(200)
    grid = {"fast": [5, 10], "slow": [30, 50]}

    result = sweep(prices, SmaCross, grid)

    assert isinstance(result, SweepResult)
    # 2x2 = 4 combos, all valid because fast < slow
    assert len(result.metrics) == 4
    assert set(result.param_names) == {"fast", "slow"}
    assert "sharpe" in result.metrics.columns
    assert "total_return" in result.metrics.columns
    assert len(result.equity) == 4


def test_sweep_skips_invalid_combos_but_keeps_others():
    """fast >= slow should raise inside SmaCross and be skipped, not kill the sweep."""
    prices = _prices(200)
    grid = {"fast": [5, 50], "slow": [30, 50]}
    # combos: (5,30) ok, (5,50) ok, (50,30) invalid, (50,50) invalid
    result = sweep(prices, SmaCross, grid)
    assert len(result.metrics) == 2
    valid = result.metrics[["fast", "slow"]]
    assert (valid["fast"] < valid["slow"]).all()


def test_sweep_metrics_are_finite_numbers():
    prices = _prices(300)
    grid = {"fast": [5, 10], "slow": [30, 60]}
    result = sweep(prices, SmaCross, grid)

    for col in ["total_return", "sharpe", "max_drawdown", "exposure"]:
        vals = result.metrics[col]
        assert vals.notna().all(), f"{col} has NaN"
        assert np.isfinite(vals).all(), f"{col} has inf"


def test_sweep_equity_curves_align_with_prices():
    prices = _prices(200)
    grid = {"fast": [5], "slow": [30]}
    result = sweep(prices, SmaCross, grid)

    (run_id, eq), = result.equity.items()
    assert eq.index.equals(prices.index)
    assert eq.iloc[0] == pytest.approx(100_000.0)


def test_sweep_feature_builder_is_applied_once():
    """If feature_builder adds a column, the strategy can read it — but here
    we just verify the callable runs and doesn't break the pipeline."""
    prices = _prices(200)
    calls = {"n": 0}

    def fb(df):
        calls["n"] += 1
        return df

    result = sweep(
        prices, SmaCross,
        {"fast": [5, 10], "slow": [30, 60]},
        feature_builder=fb,
    )
    assert calls["n"] == 1
    assert len(result.metrics) == 4


# ---------------------------------------------------------------------------
# SweepResult helpers
# ---------------------------------------------------------------------------

def test_best_sorts_by_metric():
    prices = _prices(300)
    grid = {"fast": [5, 10, 20], "slow": [30, 60, 100]}
    result = sweep(prices, SmaCross, grid)

    top = result.best("sharpe", top_n=3)
    assert len(top) <= 3
    # descending Sharpe
    assert (top["sharpe"].diff().dropna() <= 0).all()


def test_best_metric_missing_raises():
    prices = _prices(200)
    result = sweep(prices, SmaCross, {"fast": [5], "slow": [30]})
    with pytest.raises(KeyError):
        result.best("nonexistent_metric")


def test_pivot_shape_matches_grid():
    prices = _prices(300)
    grid = {"fast": [5, 10, 20], "slow": [30, 60]}
    result = sweep(prices, SmaCross, grid)

    mat = result.pivot("sharpe", x="fast", y="slow")
    assert mat.shape == (2, 3)          # slow rows, fast columns
    assert list(mat.index) == [30, 60]
    assert list(mat.columns) == [5, 10, 20]


def test_pivot_rejects_unknown_column():
    prices = _prices(200)
    result = sweep(prices, SmaCross, {"fast": [5], "slow": [30]})
    with pytest.raises(KeyError):
        result.pivot("sharpe", x="nope", y="slow")


# ---------------------------------------------------------------------------
# config plumbing
# ---------------------------------------------------------------------------

def test_sweep_uses_provided_config():
    """Higher costs should lower total return for identical signals."""
    prices = _prices(500)
    grid = {"fast": [10], "slow": [30]}

    cheap = sweep(prices, SmaCross, grid, BacktestConfig(commission_bps=0, slippage_bps=0))
    pricey = sweep(prices, SmaCross, grid, BacktestConfig(commission_bps=20, slippage_bps=20))

    cheap_ret = cheap.metrics["total_return"].iloc[0]
    pricey_ret = pricey.metrics["total_return"].iloc[0]
    assert pricey_ret < cheap_ret