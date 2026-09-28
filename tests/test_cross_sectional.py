"""Tests for src.features.cross_sectional."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.cross_sectional import (
    cross_sectional_rank,
    cross_sectional_zscore,
    demean,
    to_panel,
    top_n_mask,
)


def _panel() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", name="date")
    return pd.DataFrame(
        {
            "A": [1.0, 4.0, 7.0],
            "B": [2.0, 5.0, 8.0],
            "C": [3.0, 6.0, 9.0],
        },
        index=idx,
    )


def test_to_panel_builds_dataframe():
    s1 = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2))
    s2 = pd.Series([3.0, 4.0], index=pd.date_range("2024-01-01", periods=2))
    panel = to_panel({"a": s1, "b": s2})
    assert list(panel.columns) == ["a", "b"]
    assert panel.shape == (2, 2)


def test_to_panel_rejects_empty():
    with pytest.raises(ValueError):
        to_panel({})


def test_rank_ascending():
    r = cross_sectional_rank(_panel(), ascending=True)
    assert r.iloc[0].tolist() == [1.0, 2.0, 3.0]
    assert r.iloc[2].tolist() == [1.0, 2.0, 3.0]


def test_rank_descending():
    r = cross_sectional_rank(_panel(), ascending=False)
    assert r.iloc[0].tolist() == [3.0, 2.0, 1.0]


def test_zscore_row_has_mean_zero_std_one():
    z = cross_sectional_zscore(_panel())
    for i in range(len(z)):
        row = z.iloc[i]
        assert abs(row.mean()) < 1e-12
        assert abs(row.std(ddof=1) - 1.0) < 1e-12


def test_demean_row_sums_to_zero():
    d = demean(_panel())
    for i in range(len(d)):
        assert abs(d.iloc[i].sum()) < 1e-12


def test_top_n_mask_top_largest():
    m = top_n_mask(_panel(), n=1, ascending=False)
    # row 0 max is C=3 -> only C True
    assert m.iloc[0].tolist() == [False, False, True]
    # row 2 max is C=9 -> only C True
    assert m.iloc[2].tolist() == [False, False, True]


def test_top_n_mask_top_smallest():
    m = top_n_mask(_panel(), n=1, ascending=True)
    # row 0 min is A=1 -> only A True
    assert m.iloc[0].tolist() == [True, False, False]


def test_top_n_mask_rejects_bad_n():
    with pytest.raises(ValueError):
        top_n_mask(_panel(), n=0)