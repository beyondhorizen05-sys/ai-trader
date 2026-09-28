"""Technical indicators. Pure functions: OHLCV in, Series/DataFrame out.

Every function:
  - preserves the input index
  - never mutates the input
  - returns NaN for the warm-up window (no forward-filling)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def returns(close: pd.Series, kind: str = "log") -> pd.Series:
    """Simple or log returns of a price series.

    kind="simple": close_t / close_{t-1} - 1
    kind="log":    ln(close_t / close_{t-1})
    """
    if kind == "simple":
        out = close.pct_change()
    elif kind == "log":
        out = np.log(close / close.shift(1))
    else:
        raise ValueError(f"kind must be 'simple' or 'log', got {kind!r}")
    out.name = f"ret_{kind}"
    return out


# ---------------------------------------------------------------------------
# Rolling volatility
# ---------------------------------------------------------------------------

def rolling_vol(close: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Rolling std of log returns, optionally annualized."""
    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(window=window, min_periods=window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    vol.name = f"vol_{window}"
    return vol


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def sma(series: pd.Series, window: int) -> pd.Series:
    out = series.rolling(window=window, min_periods=window).mean()
    out.name = f"sma_{window}"
    return out


def ema(series: pd.Series, window: int) -> pd.Series:
    out = series.ewm(span=window, adjust=False, min_periods=window).mean()
    out.name = f"ema_{window}"
    return out


# ---------------------------------------------------------------------------
# RSI (Wilder)
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI. 0..100. Uses EWM with alpha = 1/window."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # If avg_loss == 0 -> all gains -> RSI 100. If avg_gain == 0 -> RSI 0.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(avg_gain != 0, 0.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    out.name = f"rsi_{window}"
    return out


# ---------------------------------------------------------------------------
# Average True Range (Wilder)
# ---------------------------------------------------------------------------

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr.name = "tr"
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    out = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    out.name = f"atr_{window}"
    return out


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Returns DataFrame with columns: bb_mid, bb_upper, bb_lower, bb_pctb, bb_width."""
    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pctb = (close - lower) / (upper - lower).replace(0.0, np.nan)
    width = (upper - lower) / mid.replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_pctb": pctb,
            "bb_width": width,
        }
    )


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Returns DataFrame with columns: macd, macd_signal, macd_hist."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {
            "macd": line,
            "macd_signal": sig,
            "macd_hist": line - sig,
        }
    )


# ---------------------------------------------------------------------------
# Convenience: attach a default feature set
# ---------------------------------------------------------------------------

DEFAULT_WINDOWS = (10, 20, 50, 200)


def add_all(df: pd.DataFrame, windows: tuple[int, ...] = DEFAULT_WINDOWS) -> pd.DataFrame:
    """Attach a default battery of indicators. Input must have lowercased OHLCV columns."""
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]

    out["ret_log"] = returns(close, "log")
    out["ret_simple"] = returns(close, "simple")
    out["vol_20"] = rolling_vol(close, window=20)

    for w in windows:
        out[f"sma_{w}"] = sma(close, w)
        out[f"ema_{w}"] = ema(close, w)

    out["rsi_14"] = rsi(close, 14)
    out["atr_14"] = atr(high, low, close, 14)

    bb = bollinger(close, window=20, num_std=2.0)
    for col in bb.columns:
        out[col] = bb[col]

    m = macd(close)
    for col in m.columns:
        out[col] = m[col]

    return out