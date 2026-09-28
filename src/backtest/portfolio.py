"""Multi-symbol portfolio backtest.

Two entry points:
  - run_portfolio: run a single-symbol Strategy on each name, equal-weight
  - run_portfolio_weights: apply pre-computed (dates x symbols) target weights
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.backtest import metrics as M
from src.backtest.engine import BacktestConfig, prepare_prices
from src.strategies.base import Strategy


@dataclass
class PortfolioResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    per_symbol: dict[str, pd.Series] = field(default_factory=dict)
    metrics: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    config: BacktestConfig | None = field(default=None, repr=False)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"symbols          {len(self.per_symbol)}",
            f"total return     {m['total_return']:.2%}",
            f"CAGR             {m['cagr']:.2%}",
            f"volatility       {m['volatility']:.2%}",
            f"Sharpe           {m['sharpe']:.2f}",
            f"max drawdown     {m['max_drawdown']:.2%}",
            f"calmar           {m['calmar']:.2f}",
        ]
        return "\n".join(lines)


def _finalize(
    returns_df: pd.DataFrame,
    weights: pd.DataFrame,
    cfg: BacktestConfig,
    per_symbol_returns: dict[str, pd.Series],
) -> PortfolioResult:
    portfolio_ret = (weights * returns_df).sum(axis=1).rename("net_return")
    equity = cfg.initial_capital * (1.0 + portfolio_ret).cumprod()
    metrics = pd.Series(
        {
            "total_return": M.total_return(equity),
            "cagr": M.cagr(equity),
            "volatility": M.volatility(portfolio_ret),
            "sharpe": M.sharpe(portfolio_ret, rf=cfg.rf),
            "sortino": M.sortino(portfolio_ret, rf=cfg.rf),
            "max_drawdown": M.max_drawdown(equity),
            "calmar": M.calmar(equity),
            "exposure": float((weights.abs().sum(axis=1) > 0).mean()),
        }
    )
    return PortfolioResult(
        equity=equity,
        returns=portfolio_ret,
        weights=weights,
        per_symbol=per_symbol_returns,
        metrics=metrics,
        config=cfg,
    )


def _aligned_common_index(prices_by_symbol: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    common_index: pd.DatetimeIndex | None = None
    for df in prices_by_symbol.values():
        common_index = df.index if common_index is None else common_index.union(df.index)
    assert common_index is not None
    return common_index.sort_values()


def _per_symbol_returns(
    prices_by_symbol: dict[str, pd.DataFrame],
    common_index: pd.DatetimeIndex,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    cols = {}
    for sym, df in prices_by_symbol.items():
        aligned = df.reindex(common_index).ffill()
        adj = prepare_prices(aligned) if "adj close" in aligned.columns else aligned.copy()
        cols[sym] = adj["close"].pct_change().fillna(0.0).rename(sym)
    return pd.DataFrame(cols)


def run_portfolio(
    prices_by_symbol: dict[str, pd.DataFrame],
    strategy_factory: Callable[[str], Strategy],
    config: BacktestConfig | None = None,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> PortfolioResult:
    cfg = config or BacktestConfig()
    if not prices_by_symbol:
        raise ValueError("prices_by_symbol is empty")

    common_index = _aligned_common_index(prices_by_symbol)

    per_symbol_returns: dict[str, pd.Series] = {}
    per_symbol_signals: dict[str, pd.Series] = {}

    for sym, df in prices_by_symbol.items():
        aligned = df.reindex(common_index).ffill()
        feats = feature_builder(aligned) if feature_builder else aligned
        strat = strategy_factory(sym)
        sig = strat.generate_signals(feats)

        adj = prepare_prices(aligned) if "adj close" in aligned.columns else aligned.copy()
        close_ret = adj["close"].pct_change().fillna(0.0)

        target = sig.reindex(common_index).fillna(0).astype(int)
        executed = target.shift(1).fillna(0).astype(int) * cfg.position_size

        gross = executed * close_ret
        turnover = executed.diff().abs().fillna(executed.abs())
        cost = turnover * ((cfg.commission_bps + cfg.slippage_bps) / 10_000.0)
        net = gross - cost

        per_symbol_returns[sym] = net.rename(sym)
        per_symbol_signals[sym] = executed.rename(sym)

    returns_df = pd.DataFrame(per_symbol_returns)
    signals_df = pd.DataFrame(per_symbol_signals)

    active = (signals_df != 0)
    n_active = active.sum(axis=1).replace(0, np.nan)
    weights = active.div(n_active, axis=0).fillna(0.0)

    return _finalize(returns_df, weights, cfg, per_symbol_returns)


def run_portfolio_weights(
    prices_by_symbol: dict[str, pd.DataFrame],
    target_weights: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> PortfolioResult:
    """Apply pre-computed target weights (dates x symbols).

    `target_weights` is the *desired* weight at the close of each bar.
    Execution lags by one bar internally.
    """
    cfg = config or BacktestConfig()
    if not prices_by_symbol:
        raise ValueError("prices_by_symbol is empty")
    if target_weights.empty:
        raise ValueError("target_weights is empty")

    common_index = _aligned_common_index(prices_by_symbol)

    returns_df = _per_symbol_returns(prices_by_symbol, common_index, cfg)
    per_symbol_returns = {c: returns_df[c].rename(c) for c in returns_df.columns}

    tw = target_weights.reindex(common_index).reindex(columns=returns_df.columns).fillna(0.0)
    executed = tw.shift(1).fillna(0.0) * cfg.position_size

    gross = (executed * returns_df).sum(axis=1)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    cost = turnover * ((cfg.commission_bps + cfg.slippage_bps) / 10_000.0)
    net = (gross - cost).rename("net_return")

    equity = cfg.initial_capital * (1.0 + net).cumprod()
    metrics = pd.Series(
        {
            "total_return": M.total_return(equity),
            "cagr": M.cagr(equity),
            "volatility": M.volatility(net),
            "sharpe": M.sharpe(net, rf=cfg.rf),
            "sortino": M.sortino(net, rf=cfg.rf),
            "max_drawdown": M.max_drawdown(equity),
            "calmar": M.calmar(equity),
            "exposure": float((executed.abs().sum(axis=1) > 0).mean()),
        }
    )

    return PortfolioResult(
        equity=equity,
        returns=net,
        weights=executed,
        per_symbol=per_symbol_returns,
        metrics=metrics,
        config=cfg,
    )