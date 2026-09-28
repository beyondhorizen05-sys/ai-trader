"""Tests for src.data.sectors."""
from __future__ import annotations

import pytest

from src.data.sectors import (
    SECTORS_LARGE_CAP_30,
    SECTORS_LARGE_CAP_100,
    get_sectors,
)
from src.data.universe import get_universe


def test_sector_map_covers_most_of_universe_30():
    universe = set(get_universe("large_cap_30"))
    sectors = set(SECTORS_LARGE_CAP_30.keys())
    missing = universe - sectors
    assert len(missing) <= 1, f"missing sector labels for {missing}"


def test_sector_map_covers_most_of_universe_100():
    universe = set(get_universe("large_cap_100"))
    sectors = set(SECTORS_LARGE_CAP_100.keys())
    missing = universe - sectors
    assert len(missing) <= 5, f"missing sector labels for {missing}"


def test_get_sectors_returns_copy():
    s1 = get_sectors("large_cap_100")
    s1["FAKE"] = "test"
    s2 = get_sectors("large_cap_100")
    assert "FAKE" not in s2


def test_get_sectors_unknown_universe():
    with pytest.raises(KeyError):
        get_sectors("does_not_exist")