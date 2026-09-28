"""Position sizing and risk overlays. Pure functions on Series/DataFrames."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def fixed_fraction(position: pd.Series, fraction: float) -> pd.Series:
    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in (0, 1]")
    return (position * fraction).rename("position")


def vol_target(
    position: pd.Series,
    returns: pd.Series,
    target_vol: float = 0.15,
    window: int = 20,
    max_leverage: float = 1.0,
) -> pd.Series:
    if target_vol <= 0:
        raise ValueError("target_vol must be > 0")
    if max_leverage <= 0:
        raise ValueError("max_leverage must be > 0")

    realized = returns.rolling(window=window, min_periods=window).std() * np.sqrt(TRADING_DAYS)
    leverage = (target_vol / realized.replace(0.0, np.nan)).clip(upper=max_leverage)
    leverage = leverage.shift(1)

    return (position * leverage).fillna(0.0).rename("position")


def atr_stop(
    position: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    multiple: float = 2.0,
) -> pd.Series:
    if multiple <= 0:
        raise ValueError("multiple must be > 0")
    adverse = (close - close.shift(1)) * -np.sign(position)
    threshold = multiple * atr
    hit = (adverse.abs() > threshold).fillna(False)
    stopped = position.where(~hit, 0)
    return stopped.rename("position")


def apply_max_drawdown_brake(
    position: pd.Series,
    returns: pd.Series,
    max_dd: float = 0.20,
    cooldown: int = 20,
) -> pd.Series:
    if not (0 < max_dd < 1):
        raise ValueError("max_dd must be in (0, 1)")
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0

    brake = pd.Series(False, index=position.index)
    trigger = (dd < -max_dd).to_numpy()
    out_vals = position.to_numpy(copy=True)
    armed_until = -1
    for i in range(len(position)):
        if i <= armed_until:
            out_vals[i] = 0
        elif trigger[i]:
            armed_until = i + cooldown
            out_vals[i] = 0
    return pd.Series(out_vals, index=position.index, name="position")