"""Simple moving-average crossover strategy."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.features.indicators import sma
from src.strategies.base import Strategy


@dataclass
class SmaCross(Strategy):
    """Long when fast SMA > slow SMA; flat (or short) otherwise.

    Parameters
    ----------
    fast, slow : int
        Window lengths for the two SMAs. Must satisfy ``fast < slow``.
    allow_short : bool
        If True, emit -1 when fast < slow (long/short). If False, emit 0
        (long/flat).
    """

    fast: int = 20
    slow: int = 50
    allow_short: bool = False
    name: str = "sma_cross"

    def __post_init__(self) -> None:
        if self.fast >= self.slow:
            raise ValueError(f"SmaCross: fast ({self.fast}) must be < slow ({self.slow})")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if "close" not in df.columns:
            raise ValueError("SmaCross: input must have a 'close' column")

        close = df["close"]
        fast = sma(close, self.fast)
        slow = sma(close, self.slow)

        # NaN during warm-up -> flat
        valid = fast.notna() & slow.notna()

        long_mask = valid & (fast > slow)
        short_mask = valid & (fast < slow)

        pos = self._empty_position(df)
        pos[long_mask] = 1
        if self.allow_short:
            pos[short_mask] = -1

        # No look-ahead: SMA at t uses closes up to t inclusive; that's
        # decision data at the close of bar t. Backtester must execute at
        # t+1 open (or t+1 close) — that's the engine's responsibility.
        return self._validate_output(df, pos)