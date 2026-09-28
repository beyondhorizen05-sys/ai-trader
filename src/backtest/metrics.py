"""Performance metrics for a return or equity series. Pure functions."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate from an equity curve."""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = len(equity) / TRADING_DAYS
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)


def volatility(returns: pd.Series, annualize: bool = True) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    vol = float(r.std(ddof=1))
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


def sharpe(returns: pd.Series, rf: float = 0.0, annualize: bool = True) -> float:
    """Annualized Sharpe ratio. rf is an annual rate (e.g. 0.04 for 4%)."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - rf / TRADING_DAYS
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    s = float(excess.mean() / sd)
    return s * np.sqrt(TRADING_DAYS) if annualize else s


def sortino(returns: pd.Series, rf: float = 0.0, annualize: bool = True) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - rf / TRADING_DAYS
    downside = excess[excess < 0]
    dd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if dd == 0 or np.isnan(dd):
        return 0.0
    s = float(excess.mean() / dd)
    return s * np.sqrt(TRADING_DAYS) if annualize else s


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough drop as a positive fraction (e.g., 0.25 = 25%)."""
    if len(equity) < 2:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(-dd.min())


def calmar(equity: pd.Series) -> float:
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return cagr(equity) / mdd


def hit_rate(trade_returns: pd.Series) -> float:
    r = trade_returns.dropna()
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def profit_factor(trade_returns: pd.Series) -> float:
    r = trade_returns.dropna()
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def exposure(position: pd.Series) -> float:
    """Fraction of bars with a non-zero position."""
    p = position.dropna()
    if len(p) == 0:
        return 0.0
    return float((p != 0).mean())