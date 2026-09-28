"""Run the full pipeline end-to-end. Idempotent.

Steps:
  1. fetch    (data/raw)
  2. clean    (data/processed)
  3. backtest (single-symbol demo)
  4. walk_forward
  5. portfolio
  6. fetch universe + xs momentum (if enough symbols)
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("fetch",         ["-m", "src.data.fetch"]),
    ("clean",         ["-m", "src.data.clean"]),
    ("backtest",      ["-m", "scripts.run_backtest"]),
    ("walk_forward",  ["-m", "scripts.run_walk_forward"]),
    ("portfolio",     ["-m", "scripts.run_portfolio"]),
]


def run_step(name: str, args: list[str]) -> None:
    print(f"\n{'=' * 60}\nSTEP: {name}\n{'=' * 60}")
    result = subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"step {name!r} failed with exit code {result.returncode}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name, args in STEPS:
        run_step(name, args)
    print("\n✓ all steps completed")


if __name__ == "__main__":
    main()