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
    """Scale positions so realized vol matches `target_vol` (annualized).

    Uses trailing realized vol of the underlying return series. Never
    looks ahead: leverage at t uses returns up to t-1.
    """
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
    """Zero out the position once price moves `multiple * ATR` against entry.

    The stop is **sticky**: once tripped, the position stays flat until the
    strategy's raw signal flips sign or goes to zero (i.e. until the position
    would naturally be re-entered or is already flat).
    """
    if multiple <= 0:
        raise ValueError("multiple must be > 0")

    position = position.reindex(close.index).fillna(0)
    price_delta = close.diff()
    adverse = -price_delta * np.sign(position)
    threshold = multiple * atr

    raw_hit = (adverse.abs() > threshold).fillna(False).to_numpy()
    raw_pos = position.to_numpy(copy=True)

    out = raw_pos.copy()
    stopped = False
    prev_signal = 0

    for i in range(len(raw_pos)):
        sig = raw_pos[i]

        # If the raw signal changed (or went flat), reset the stop.
        if sig != prev_signal:
            stopped = False

        if stopped:
            out[i] = 0
        elif raw_hit[i] and sig != 0:
            # Trip the stop on this bar.
            stopped = True
            out[i] = 0

        prev_signal = sig

    return pd.Series(out, index=position.index, name="position")


def apply_max_drawdown_brake(
    position: pd.Series,
    returns: pd.Series,
    max_dd: float = 0.20,
    cooldown: int = 20,
) -> pd.Series:
    """Force flat after equity drops more than `max_dd` from peak, for `cooldown` bars."""
    if not (0 < max_dd < 1):
        raise ValueError("max_dd must be in (0, 1)")

    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0

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