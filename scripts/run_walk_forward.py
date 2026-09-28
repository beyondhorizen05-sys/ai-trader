"""Walk-forward validation demo on processed AAPL data.

Produces:
  notebooks/wf_equity.png       OOS equity vs. buy & hold vs. in-sample best
  notebooks/wf_folds.csv        per-fold: dates, chosen params, train/test metrics
  notebooks/wf_oos_equity.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.walk_forward import walk_forward
from src.data.clean import load_raw, normalize
from src.features.indicators import add_all
from src.strategies.sma_cross import SmaCross

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "notebooks"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    prices = normalize(load_raw("AAPL"))
    print(f"Loaded {len(prices)} bars: {prices.index[0].date()} -> {prices.index[-1].date()}")

    grid = {"fast": [5, 10, 15, 20, 30], "slow": [30, 50, 75, 100, 150]}
    cfg = BacktestConfig(commission_bps=1.0, slippage_bps=2.0)

    result = walk_forward(
        prices=prices,
        strategy_cls=SmaCross,
        param_grid=grid,
        train_years=3,
        test_years=1,
        step_years=1,
        select_metric="sharpe",
        config=cfg,
        feature_builder=add_all,
    )

    print("\n=== Walk-forward summary ===")
    print(result.summary())

    print("\n=== Per-fold detail ===")
    print(result.folds.to_string(index=False))

    result.folds.to_csv(OUT_DIR / "wf_folds.csv", index=False)
    result.oos_equity.to_frame("equity").to_parquet(OUT_DIR / "wf_oos_equity.parquet")
    print(f"\nsaved {OUT_DIR / 'wf_folds.csv'}")
    print(f"saved {OUT_DIR / 'wf_oos_equity.parquet'}")

    # Reference curves for the plot
    bh = prices["adj close"] / prices["adj close"].iloc[0] * cfg.initial_capital

    in_sample = run_backtest(
        prices,
        SmaCross(fast=20, slow=50, allow_short=False).generate_signals(add_all(prices)),
        cfg,
    )

    # Plot on OOS date range
    start = result.oos_equity.index[0]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(result.oos_equity.index, result.oos_equity, label="Walk-forward OOS", linewidth=1.6)
    ax.plot(bh.loc[start:].index, bh.loc[start:], label="Buy & Hold", linewidth=1.0, alpha=0.8)
    ax.plot(in_sample.equity.loc[start:].index, in_sample.equity.loc[start:],
            label="In-sample SMA(20,50)", linewidth=1.0, alpha=0.7, linestyle="--")
    ax.set_title("AAPL — Walk-forward out-of-sample vs. references")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / "wf_equity.png"
    plt.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()