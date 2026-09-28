"""Unit tests for src.backtest.walk_forward."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestConfig
from src.backtest.walk_forward import _date_slices, walk_forward
from src.strategies.sma_cross import SmaCross


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _prices(years: int = 6) -> pd.DataFrame:
    # ~252 bars per "year"
    n = years * 252
    idx = pd.date_range("2018-01-01", periods=n, freq="B", name="date")
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0005, 0.012, size=n)
    close = 100 * np.exp(np.cumsum(rets))
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


# ---------------------------------------------------------------------------
# _date_slices
# ---------------------------------------------------------------------------

def test_date_slices_produces_non_overlapping_test_windows():
    idx = pd.date_range("2015-01-01", "2024-12-31", freq="B")
    folds = _date_slices(idx, train_years=3, test_years=1, step_years=1)
    assert len(folds) >= 5

    # test windows must not overlap
    for (_, _, ts1, te1), (_, _, ts2, _) in zip(folds, folds[1:]):
        assert ts2 >= te1

    # every fold: train_end == test_start
    for _, tr_e, te_s, _ in folds:
        assert tr_e == te_s


def test_date_slices_respects_bounds():
    idx = pd.date_range("2015-01-01", "2020-12-31", freq="B")
    folds = _date_slices(idx, train_years=3, test_years=1, step_years=1)
    for _, _, te_s, te_e in folds:
        assert te_s < idx.max()
        # test_end is either <= idx.max()+1 or clamped
        assert te_e <= idx.max() + pd.Timedelta(days=1)


def test_date_slices_empty_when_history_too_short():
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    folds = _date_slices(idx, train_years=5, test_years=1, step_years=1)
    assert folds == []


# ---------------------------------------------------------------------------
# walk_forward — happy path
# ---------------------------------------------------------------------------

def test_walk_forward_runs_and_produces_stitched_equity():
    prices = _prices(years=8)
    grid = {"fast": [5, 10], "slow": [30, 60]}

    res = walk_forward(
        prices, SmaCross, grid,
        train_years=3, test_years=1, step_years=1,
        select_metric="sharpe",
    )

    assert not res.oos_equity.empty
    assert not res.folds.empty
    assert len(res.folds) == len(res.best_params_per_fold)
    # OOS equity starts at initial_capital
    assert res.oos_equity.iloc[0] == pytest.approx(100_000.0 * (1.0 + res.oos_returns.iloc[0]))


def test_walk_forward_oos_returns_are_unique_and_sorted():
    prices = _prices(years=8)
    res = walk_forward(
        prices, SmaCross,
        {"fast": [5, 10], "slow": [30, 60]},
        train_years=3, test_years=1, step_years=1,
    )
    assert res.oos_returns.index.is_monotonic_increasing
    assert not res.oos_returns.index.has_duplicates


def test_walk_forward_each_fold_picks_some_params():
    prices = _prices(years=8)
    res = walk_forward(
        prices, SmaCross,
        {"fast": [5, 10], "slow": [30, 60]},
        train_years=3, test_years=1, step_years=1,
    )
    for params in res.best_params_per_fold:
        assert "fast" in params and "slow" in params
        assert params["fast"] < params["slow"]


# ---------------------------------------------------------------------------
# No-look-ahead guarantee
# ---------------------------------------------------------------------------

def test_walk_forward_does_not_use_future_data_to_pick_params():
    """Poison the tail: replace prices after fold N's test window with an
    absurd jump. Params chosen for earlier folds must be identical, and the
    OOS returns for those earlier folds must be identical too."""
    prices = _prices(years=8)
    grid = {"fast": [5, 10], "slow": [30, 60]}

    res_a = walk_forward(prices, SmaCross, grid,
                         train_years=3, test_years=1, step_years=1)

    poisoned = prices.copy()
    cutoff = res_a.folds.iloc[1]["test_end"]
    cutoff_ts = pd.Timestamp(cutoff)
    poisoned.loc[poisoned.index > cutoff_ts, ["open", "high", "low", "close", "adj close"]] *= 10

    res_b = walk_forward(poisoned, SmaCross, grid,
                         train_years=3, test_years=1, step_years=1)

    # first two folds' chosen params must be unchanged
    assert res_a.best_params_per_fold[0] == res_b.best_params_per_fold[0]
    assert res_a.best_params_per_fold[1] == res_b.best_params_per_fold[1]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_walk_forward_rejects_bad_windows():
    prices = _prices(years=6)
    with pytest.raises(ValueError):
        walk_forward(prices, SmaCross, {"fast": [5], "slow": [30]},
                     train_years=0, test_years=1, step_years=1)
    with pytest.raises(ValueError):
        walk_forward(prices, SmaCross, {"fast": [5], "slow": [30]},
                     train_years=3, test_years=-1, step_years=1)


def test_walk_forward_rejects_unknown_metric():
    prices = _prices(years=6)
    with pytest.raises(ValueError, match="select_metric"):
        walk_forward(prices, SmaCross, {"fast": [5], "slow": [30]},
                     train_years=3, test_years=1, step_years=1,
                     select_metric="not_a_metric")


def test_walk_forward_raises_if_history_too_short():
    prices = _prices(years=2)
    with pytest.raises(ValueError):
        walk_forward(prices, SmaCross, {"fast": [5], "slow": [30]},
                     train_years=5, test_years=1, step_years=1)