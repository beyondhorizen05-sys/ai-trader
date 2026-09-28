"""End-to-end demo: load processed AAPL, generate signals, run backtest, plot.

Usage
-----
    python -m scripts.run_backtest
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest
from src.data.clean import load_raw, normalize
from src.features.indicators import add_all
from src.strategies.sma_cross import SmaCross

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "notebooks"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # 1. Load and clean the raw AAPL bars.
    raw = load_raw("AAPL")
    prices = normalize(raw)
    print(f"Loaded {len(prices)} bars: {prices.index[0].date()} -> {prices.index[-1].date()}")

    # 2. Features (not strictly needed for SmaCross but mirrors a real pipeline).
    features = add_all(prices)

    # 3. Signal.
    strat = SmaCross(fast=20, slow=50, allow_short=False)
    signals = strat.generate_signals(features)
    print(f"Strategy: {strat.name} fast={strat.fast} slow={strat.slow} short={strat.allow_short}")
    print(f"Bars long: {(signals == 1).sum()} / {len(signals)}")

    # 4. Backtest.
    cfg = BacktestConfig(
        initial_capital=100_000.0,
        commission_bps=1.0,
        slippage_bps=2.0,
        position_size=1.0,
    )
    result = run_backtest(prices, signals, cfg)
    print("\n=== Results ===")
    print(result.summary())

    if not result.trades.empty:
        print("\n=== First 5 trades ===")
        print(result.trades.head().to_string(index=False))

    # 5. Plot equity + drawdown.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(result.equity.index, result.equity, label="Strategy", linewidth=1.4)
    bh = prices["adj close"] / prices["adj close"].iloc[0] * cfg.initial_capital
    ax1.plot(bh.index, bh, label="Buy & Hold", linewidth=1.0, alpha=0.7)
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(f"AAPL — {strat.name} vs Buy & Hold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    dd = result.equity / result.equity.cummax() - 1.0
    ax2.fill_between(dd.index, dd * 100, 0, color="red", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "equity_curve.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path}")

    # 6. Persist results for later inspection.
    result.equity.to_frame("equity").to_parquet(OUT_DIR / "equity.parquet")
    result.trades.to_parquet(OUT_DIR / "trades.parquet")
    print(f"Saved equity and trades to {OUT_DIR}")


if __name__ == "__main__":
    main()