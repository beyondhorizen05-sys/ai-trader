"""Fetch a named universe of symbols. Idempotent — skips already-downloaded files."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.data.clean import load_settings
from src.data.fetch import fetch_symbol, save_raw
from src.data.universe import SURVIVORSHIP_WARNING, get_universe

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(universe_name: str = "large_cap_30", limit: int | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(SURVIVORSHIP_WARNING)

    symbols = get_universe(universe_name)
    if limit is not None:
        symbols = symbols[:limit]
    print(f"\nFetching {len(symbols)} symbols from universe {universe_name!r}\n")

    cfg = load_settings()["data"]
    raw_dir = PROJECT_ROOT / "data" / "raw"

    for i, sym in enumerate(symbols, 1):
        out_path = raw_dir / f"{sym.lower()}.parquet"
        if out_path.exists():
            print(f"[{i:>3}/{len(symbols)}] {sym}  (cached)")
            continue
        try:
            df = fetch_symbol(sym, cfg["start"], cfg["end"], cfg["timeframe"])
            save_raw(df, sym)
            print(f"[{i:>3}/{len(symbols)}] {sym}  {len(df)} rows")
        except Exception as e:  # noqa: BLE001
            print(f"[{i:>3}/{len(symbols)}] {sym}  FAILED: {e}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "large_cap_30"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(name, limit)