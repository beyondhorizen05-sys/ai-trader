"""Tests for src.backtest.walk_forward_xs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.walk_forward_xs import walk_forward_xs
from src.strategies.xs_momentum import XSMomentum


def _prices_by_symbol(
    n_symbols: int = 12,
    years: int = 8,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = years * 252
    idx = pd.date_range("2015-01-01", periods=n, freq="B", name="date")
    out = {}
    for i in range(n_symbols):
        drift = 0.0003 * (i - n_symbols / 2) / n_symbols
        rets = rng.normal(drift, 0.012, size=n)
        close = 100 * np.exp(np.cumsum(rets))
        out[f"S{i:02d}"] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "adj close": close,
                "volume": np.full(n, 1_000_000, dtype="int64"),
            },
            index=idx,
        )
    return out


def test_walk_forward_xs_runs_end_to_end():
    prices = _prices_by_symbol(n_symbols=12, years=8)
    res = walk_forward_xs(
        prices,
        strategy_factory=XSMomentum,
        param_grid={"lookback": [20, 60], "top_n": [3, 5]},
        train_years=3, test_years=1, step_years=1,
        min_symbols=5,
    )
    assert not res.oos_equity.empty
    assert not res.folds.empty
    assert len(res.folds) == len(res.best_params_per_fold)
    # Each fold chose lookback in {20, 60} and top_n in {3, 5}
    for p in res.best_params_per_fold:
        assert p["lookback"] in {20, 60}
        assert p["top_n"] in {3, 5}


def test_walk_forward_xs_oos_returns_sorted_and_unique():
    prices = _prices_by_symbol(n_symbols=12, years=8)
    res = walk_forward_xs(
        prices,
        strategy_factory=XSMomentum,
        param_grid={"lookback": [60], "top_n": [5]},
        train_years=3, test_years=1, step_years=1,
        min_symbols=5,
    )
    assert res.oos_returns.index.is_monotonic_increasing
    assert not res.oos_returns.index.has_duplicates


def test_walk_forward_xs_no_look_ahead():
    """Poisoning future data must not change earlier folds' chosen params."""
    prices = _prices_by_symbol(n_symbols=12, years=8)

    res_a = walk_forward_xs(
        prices,
        strategy_factory=XSMomentum,
        param_grid={"lookback": [20, 60], "top_n": [3, 5]},
        train_years=3, test_years=1, step_years=1,
        min_symbols=5,
    )

    # Poison all data after the second fold's test_end
    cutoff = pd.Timestamp(res_a.folds.iloc[1]["test_end"])
    poisoned = {}
    for sym, df in prices.items():
        df2 = df.copy()
        df2.loc[df2.index > cutoff, ["open", "high", "low", "close", "adj close"]] *= 10
        poisoned[sym] = df2

    res_b = walk_forward_xs(
        poisoned,
        strategy_factory=XSMomentum,
        param_grid={"lookback": [20, 60], "top_n": [3, 5]},
        train_years=3, test_years=1, step_years=1,
        min_symbols=5,
    )

    assert res_a.best_params_per_fold[0] == res_b.best_params_per_fold[0]
    assert res_a.best_params_per_fold[1] == res_b.best_params_per_fold[1]


def test_walk_forward_xs_rejects_bad_metric():
    prices = _prices_by_symbol(n_symbols=12, years=8)
    with pytest.raises(ValueError, match="select_metric"):
        walk_forward_xs(
            prices,
            strategy_factory=XSMomentum,
            param_grid={"lookback": [60], "top_n": [5]},
            train_years=3, test_years=1, step_years=1,
            select_metric="not_a_metric",
        )


def test_walk_forward_xs_rejects_empty_universe():
    with pytest.raises(ValueError):
        walk_forward_xs(
            {},
            strategy_factory=XSMomentum,
            param_grid={"lookback": [60], "top_n": [5]},
        )


def test_walk_forward_xs_rejects_empty_grid():
    prices = _prices_by_symbol(n_symbols=12, years=8)
    with pytest.raises(ValueError):
        walk_forward_xs(
            prices,
            strategy_factory=XSMomentum,
            param_grid={},
            train_years=3, test_years=1, step_years=1,
        )


def test_walk_forward_xs_raises_when_history_too_short():
    prices = _prices_by_symbol(n_symbols=12, years=2)
    with pytest.raises(ValueError):
        walk_forward_xs(
            prices,
            strategy_factory=XSMomentum,
            param_grid={"lookback": [20], "top_n": [3]},
            train_years=5, test_years=1, step_years=1,
            min_symbols=5,
        )


def test_walk_forward_xs_skips_folds_with_too_few_symbols():
    prices = _prices_by_symbol(n_symbols=3, years=8)   # only 3 symbols
    with pytest.raises(RuntimeError):
        walk_forward_xs(
            prices,
            strategy_factory=XSMomentum,
            param_grid={"lookback": [20], "top_n": [2]},
            train_years=3, test_years=1, step_years=1,
            min_symbols=5,      # requires 5 -> all folds skipped -> no OOS
        )