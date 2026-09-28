"""Abstract Strategy interface. Every strategy maps a feature frame to a position series."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Base class for signal generators.

    Contract:
      - ``generate_signals(df)`` receives a features DataFrame (output of
        :func:`src.features.indicators.add_all` or a similar pipeline).
      - It returns a ``pd.Series`` named ``position`` with the **same index**
        as ``df``.
      - Values are in ``{-1, 0, +1}``: short, flat, long.
      - Signals are decided using information available **at or before**
        each timestamp (no look-ahead). Implementations are responsible for
        ensuring this — typically by using already-lagged indicators or by
        shifting at the very end.
    """

    name: str = "strategy"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:  # pragma: no cover
        raise NotImplementedError

    # --- helpers -----------------------------------------------------------

    def _empty_position(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=df.index, dtype="int8", name="position")

    def _validate_output(self, df: pd.DataFrame, position: pd.Series) -> pd.Series:
        if not position.index.equals(df.index):
            raise ValueError(
                f"{self.name}: position index does not match input index"
            )
        if position.name != "position":
            position = position.rename("position")
        bad = set(position.dropna().unique()) - {-1, 0, 1}
        if bad:
            raise ValueError(f"{self.name}: invalid position values {sorted(bad)}")
        return position.astype("int8")