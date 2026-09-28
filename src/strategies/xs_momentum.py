"""Cross-sectional momentum: long top-N by trailing return, optionally short bottom-N.

Gross exposure is normalized to 1.0 for all configs (long-only and long/short
alike) so that comparisons across configurations are apples-to-apples.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.cross_sectional import to_panel, top_n_mask


@dataclass
class XSMomentum:
    """Cross-sectional momentum with equal weights within each leg.

    Parameters
    ----------
    lookback : int
        Trailing return window in bars.
    top_n : int
        Number of symbols to go long each bar.
    bottom_n : int
        Number of symbols to short each bar. 0 -> long-only.
    min_history : int
        Skip bars with fewer than this many symbols having valid data.
    skip_recent : int
        Bars to exclude from the momentum signal. 0 = classic momentum,
        21 = "12-1" style (skip the most recent month).
    sectors : dict[str, str], optional
        Map symbol -> sector. When provided, ranking is performed *within*
        sector, then long/short selections are pooled across sectors.
    gross_exposure : float
        Total gross exposure (sum of |weights| across all positions).
        Default 1.0. A long/short config with gross=1.0 puts 0.5 on the
        long side and 0.5 on the short side.
    """

    lookback: int = 60
    top_n: int = 5
    bottom_n: int = 5
    min_history: int = 20
    skip_recent: int = 0
    sectors: dict[str, str] = field(default_factory=dict)
    gross_exposure: float = 1.0

    def __post_init__(self) -> None:
        if self.lookback <= 0:
            raise ValueError("lookback must be > 0")
        if self.top_n <= 0:
            raise ValueError("top_n must be > 0")
        if self.bottom_n < 0:
            raise ValueError("bottom_n must be >= 0")
        if self.min_history <= 0:
            raise ValueError("min_history must be > 0")
        if self.skip_recent < 0:
            raise ValueError("skip_recent must be >= 0")
        if self.skip_recent >= self.lookback:
            raise ValueError("skip_recent must be < lookback")
        if self.gross_exposure <= 0:
            raise ValueError("gross_exposure must be > 0")

    def _momentum_panel(self, closes: dict[str, pd.Series]) -> pd.DataFrame:
        panel = to_panel(closes).sort_index()
        if self.skip_recent > 0:
            p0 = panel.shift(self.lookback)
            p1 = panel.shift(self.skip_recent)
            return (p1 / p0) - 1.0
        return panel.pct_change(self.lookback)

    def _long_short_masks(
        self, momentum: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        long_mask = pd.DataFrame(False, index=momentum.index, columns=momentum.columns)
        short_mask = pd.DataFrame(False, index=momentum.index, columns=momentum.columns)

        if not self.sectors:
            long_mask = top_n_mask(momentum, self.top_n, ascending=False)
            if self.bottom_n > 0:
                short_mask = top_n_mask(momentum, self.bottom_n, ascending=True)
            return long_mask.fillna(False), short_mask.fillna(False)

        by_sector: dict[str, list[str]] = {}
        for sym, sec in self.sectors.items():
            if sym in momentum.columns:
                by_sector.setdefault(sec, []).append(sym)

        n_sectors = max(len(by_sector), 1)
        longs_per_sector = max(self.top_n // n_sectors, 1)
        shorts_per_sector = max(self.bottom_n // n_sectors, 1) if self.bottom_n > 0 else 0

        for sec, cols in by_sector.items():
            sub = momentum[cols]
            lmask = top_n_mask(sub, min(longs_per_sector, len(cols)), ascending=False)
            long_mask.loc[:, cols] = lmask.fillna(False).to_numpy()
            if shorts_per_sector > 0:
                smask = top_n_mask(sub, min(shorts_per_sector, len(cols)), ascending=True)
                short_mask.loc[:, cols] = smask.fillna(False).to_numpy()

        return long_mask, short_mask

    def generate_weights(self, prices_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
        closes = {
            sym: df["adj close"] if "adj close" in df.columns else df["close"]
            for sym, df in prices_by_symbol.items()
        }
        momentum = self._momentum_panel(closes)

        long_mask, short_mask = self._long_short_masks(momentum)

        n_valid = momentum.notna().sum(axis=1)
        enough = n_valid >= self.min_history
        long_mask = long_mask.where(enough, False).fillna(False).astype(bool)
        short_mask = short_mask.where(enough, False).fillna(False).astype(bool)

        # --- Normalize so gross exposure is exactly `gross_exposure`.
        n_long = long_mask.sum(axis=1).replace(0, np.nan)
        n_short = short_mask.sum(axis=1).replace(0, np.nan)

        # Split gross equally between long and short legs when both are active.
        if self.bottom_n > 0:
            half = self.gross_exposure / 2.0
            w_long = long_mask.astype(float).div(n_long, axis=0) * half
            w_short = short_mask.astype(float).div(n_short, axis=0) * half
            weights = (w_long - w_short).fillna(0.0)
        else:
            w_long = long_mask.astype(float).div(n_long, axis=0) * self.gross_exposure
            weights = w_long.fillna(0.0)

        weights = weights.shift(1).fillna(0.0)
        return weights