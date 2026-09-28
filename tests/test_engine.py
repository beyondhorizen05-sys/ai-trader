"""Unit tests for src.backtest.engine and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import metrics as M
from src.backtest.engine import (
    BacktestConfig,
    extract_trades,
    prepare_prices,
    run_backtest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _prices(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    """Minimal price frame with open == close so fills are exact and simple."""
    idx = pd.date_range(start, periods=len(closes), freq="D", name="date")
    c = np.array(closes, dtype="float64")
    return pd.DataFrame(
        {
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "adj close": c,     # no split/dividend -> ratio == 1
            "volume": np.full(len(c), 1_000_000, dtype="int64"),
        },
        index=idx,
    )


def _signals(idx: pd.Index, values: list[int]) -> pd.Series:
    return pd.Series(values, index=idx, name="position", dtype="int8")


# ---------------------------------------------------------------------------
# prepare_prices
# ---------------------------------------------------------------------------

def test_prepare_prices_scales_ohlc_and_sets_close_to_adj():
    idx = pd.date_range("2024-01-01", periods=3, freq="D", name="date")
    df = pd.DataFrame(
        {
            "open":      [100.0, 100.0, 100.0],
            "high":      [101.0, 101.0, 101.0],
            "low":       [ 99.0,  99.0,  99.0],
            "close":     [100.0, 100.0, 100.0],
            "adj close": [ 50.0,  25.0,  12.5],   # 0.5x, 0.25x, 0.125x
            "volume":    [1000, 1000, 1000],
        },
        index=idx,
    )
    out = prepare_prices(df)
    assert out["close"].iloc[0] == 50.0
    assert out["open"].iloc[0] == 50.0
    assert out["high"].iloc[0] == 50.5
    assert out["low"].iloc[0] == 49.5
    assert out["open"].iloc[1] == 25.0


def test_prepare_prices_rejects_missing_columns():
    df = pd.DataFrame({"open": [1.0], "close": [1.0]})
    with pytest.raises(ValueError, match="missing columns"):
        prepare_prices(df)


# ---------------------------------------------------------------------------
# BacktestConfig
# ---------------------------------------------------------------------------

def test_config_rejects_bad_params():
    with pytest.raises(ValueError):
        BacktestConfig(initial_capital=0)
    with pytest.raises(ValueError):
        BacktestConfig(commission_bps=-1)
    with pytest.raises(ValueError):
        BacktestConfig(position_size=0)
    with pytest.raises(ValueError):
        BacktestConfig(position_size=1.5)


# ---------------------------------------------------------------------------
# extract_trades
# ---------------------------------------------------------------------------

def test_extract_trades_single_long_roundtrip():
    idx = pd.date_range("2024-01-01", periods=5, freq="D", name="date")
    pos = pd.Series([0, 1, 1, 0, 0], index=idx, dtype="int8")
    price = pd.Series([100.0, 100.0, 110.0, 120.0, 120.0], index=idx)
    trades = extract_trades(pos, price)

    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["side"] == "long"
    assert row["entry_price"] == 100.0
    assert row["exit_price"] == 120.0
    assert row["return"] == pytest.approx(0.20)


def test_extract_trades_flip_creates_two_trades():
    idx = pd.date_range("2024-01-01", periods=5, freq="D", name="date")
    pos = pd.Series([0, 1, -1, -1, 0], index=idx, dtype="int8")
    price = pd.Series([100.0, 100.0, 110.0, 90.0, 80.0], index=idx)
    trades = extract_trades(pos, price)

    assert len(trades) == 2
    assert trades.iloc[0]["side"] == "long"
    assert trades.iloc[0]["return"] == pytest.approx(0.10)     # 100 -> 110
    assert trades.iloc[1]["side"] == "short"
    # short 110 -> 80 : return = -1 * (80/110 - 1) = +0.2727
    assert trades.iloc[1]["return"] == pytest.approx(30 / 110)


def test_extract_trades_empty_when_flat():
    idx = pd.date_range("2024-01-01", periods=4, freq="D", name="date")
    pos = pd.Series([0, 0, 0, 0], index=idx, dtype="int8")
    price = pd.Series([100.0, 101.0, 102.0, 103.0], index=idx)
    trades = extract_trades(pos, price)
    assert trades.empty


# ---------------------------------------------------------------------------
# run_backtest: flat strategy
# ---------------------------------------------------------------------------

def test_flat_strategy_produces_flat_equity_and_no_costs():
    prices = _prices([100, 101, 102, 103, 104])
    sig = _signals(prices.index, [0, 0, 0, 0, 0])
    res = run_backtest(prices, sig, BacktestConfig(commission_bps=5, slippage_bps=5))

    assert (res.position == 0).all()
    assert res.equity.iloc[-1] == pytest.approx(100_000.0)
    assert res.metrics["n_trades"] == 0
    assert res.metrics["total_return"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run_backtest: always long
# ---------------------------------------------------------------------------

def test_always_long_tracks_price_return_minus_costs():
    """Signal is +1 from bar 1 onwards; execution begins at bar 2.
    So we hold the return of bars 2..end. With no costs, equity should equal
    the product of those close-to-close returns."""
    closes = [100, 100, 110, 121, 133.1]     # 10% up each bar from index 2
    prices = _prices(closes)
    sig = _signals(prices.index, [0, 1, 1, 1, 1])
    res = run_backtest(prices, sig, BacktestConfig(commission_bps=0, slippage_bps=0))

    # position: shift(1) -> [0,0,1,1,1]
    expected_pos = [0, 0, 1, 1, 1]
    assert list(res.position) == expected_pos

    # returns: [0, 0, 0.10, 0.10, 0.10]
    expected_final = 100_000.0 * 1.10 ** 3
    assert res.equity.iloc[-1] == pytest.approx(expected_final)


def test_costs_are_charged_on_entry_and_exit():
    """With 10 bps total cost, a full 0->1->0 round trip pays 10 + 10 bps
    on the two turnover bars."""
    closes = [100, 100, 100, 100, 100]      # no price move
    prices = _prices(closes)
    sig = _signals(prices.index, [0, 1, 1, 0, 0])
    res = run_backtest(prices, sig, BacktestConfig(commission_bps=5, slippage_bps=5))

    # executed position: [0, 0, 1, 1, 0]
    # turnover:          [0, 0, 1, 0, 1]  -> costs at bars 2 and 4
    # equity after bar 2: 100000 * (1 - 0.001) = 99900
    # equity after bar 4: 99900 * (1 - 0.001) = 99800.1
    assert res.equity.iloc[-1] == pytest.approx(100_000 * 0.999 * 0.999)


# ---------------------------------------------------------------------------
# run_backtest: look-ahead guard
# ---------------------------------------------------------------------------

def test_signal_at_last_bar_does_not_change_prior_equity():
    closes = [100, 100, 110, 121, 133.1]
    prices = _prices(closes)
    sig_a = _signals(prices.index, [0, 0, 0, 0, 0])
    sig_b = _signals(prices.index, [0, 0, 0, 0, 1])   # only last bar differs

    res_a = run_backtest(prices, sig_a, BacktestConfig(commission_bps=0, slippage_bps=0))
    res_b = run_backtest(prices, sig_b, BacktestConfig(commission_bps=0, slippage_bps=0))

    # Equity series should be identical because the last-bar signal is
    # executed at a bar that doesn't exist (shift(1) puts it off the end).
    pd.testing.assert_series_equal(res_a.equity, res_b.equity)


# ---------------------------------------------------------------------------
# run_backtest: metric sanity
# ---------------------------------------------------------------------------

def test_metrics_have_expected_keys_and_ranges():
    closes = list(np.linspace(100, 150, 60))
    prices = _prices(closes)
    sig = _signals(prices.index, [1] * 60)
    res = run_backtest(prices, sig)

    for k in [
        "total_return", "cagr", "volatility", "sharpe", "sortino",
        "max_drawdown", "calmar", "hit_rate", "profit_factor",
        "exposure", "n_trades",
    ]:
        assert k in res.metrics

    assert 0 <= res.metrics["max_drawdown"] <= 1
    assert 0 <= res.metrics["exposure"] <= 1
    assert res.metrics["sharpe"] > 0               # monotone up -> positive Sharpe


# ---------------------------------------------------------------------------
# metrics module
# ---------------------------------------------------------------------------

def test_metrics_max_drawdown_known_path():
    equity = pd.Series([100, 120, 90, 110, 80, 130])
    # peak 120, trough 80 -> 1 - 80/120 = 0.3333
    assert M.max_drawdown(equity) == pytest.approx(1 - 80 / 120)


def test_metrics_total_return():
    eq = pd.Series([100, 110, 121])
    assert M.total_return(eq) == pytest.approx(0.21)


def test_metrics_sharpe_zero_vol_is_zero():
    r = pd.Series([0.01, 0.01, 0.01, 0.01])
    assert M.sharpe(r) == 0.0


def test_metrics_hit_rate_and_profit_factor():
    r = pd.Series([0.10, -0.05, 0.20, -0.02])
    assert M.hit_rate(r) == pytest.approx(0.5)
    assert M.profit_factor(r) == pytest.approx(0.30 / 0.07)


def test_metrics_exposure():
    p = pd.Series([0, 1, 1, 0, -1])
    assert M.exposure(p) == pytest.approx(3 / 5)