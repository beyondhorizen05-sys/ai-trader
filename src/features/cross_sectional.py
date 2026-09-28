"""Cross-sectional features: rank, z-score, and demean across symbols each bar."""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_panel(series_by_symbol: dict[str, pd.Series]) -> pd.DataFrame:
    """Stack per-symbol series into a (dates x symbols) DataFrame."""
    if not series_by_symbol:
        raise ValueError("series_by_symbol is empty")
    return pd.DataFrame(series_by_symbol)


def cross_sectional_rank(panel: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    """Rank each row across symbols. Ranks are 1..N, ties broken arbitrarily."""
    return panel.rank(axis=1, ascending=ascending, method="average")


def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row across symbols. NaN rows stay NaN."""
    mean = panel.mean(axis=1)
    std = panel.std(axis=1, ddof=1)
    return panel.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)


def demean(panel: pd.DataFrame) -> pd.DataFrame:
    """Subtract the row mean from each cell."""
    return panel.sub(panel.mean(axis=1), axis=0)


def top_n_mask(panel: pd.DataFrame, n: int, ascending: bool = False) -> pd.DataFrame:
    """Boolean mask: True where the value is in the top-n of its row.

    ascending=False -> top-n largest; ascending=True -> top-n smallest.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    ranks = panel.rank(axis=1, ascending=ascending, method="first")
    return ranks <= n