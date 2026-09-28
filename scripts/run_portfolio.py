"""Multi-symbol portfolio backtest on the config symbols.

Produces:
  notebooks/portfolio_equity.png
  notebooks/portfolio_metrics.txt
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from src.backtest.engine import BacktestConfig
from src.backtest.portfolio import run_portfolio
from src.data.clean import load_raw, load_settings, normalize
from src.features.indicators import add_all
from src.strategies.sma_cross import SmaCross

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "notebooks"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = load_settings()["data"]
    symbols = cfg["symbols"]

    prices_by_symbol = {}
    for sym in symbols:
        try:
            prices_by_symbol[sym] = normalize(load_raw(sym))
        except FileNotFoundError:
            print(f"skipping {sym}: no raw data (run python -m src.data.fetch)")

    if not prices_by_symbol:
        raise SystemExit("no data available — fetch first")

    result = run_portfolio(
        prices_by_symbol=prices_by_symbol,
        strategy_factory=lambda sym: SmaCross(fast=20, slow=50, allow_short=False),
        config=BacktestConfig(commission_bps=1.0, slippage_bps=2.0),
        feature_builder=add_all,
    )

    print("\n=== Portfolio summary ===")
    print(result.summary())
    print("\n=== Per-symbol mean return (last 60 bars) ===")
    print({k: round(float(v.tail(60).mean()), 6) for k, v in result.per_symbol.items()})

    result.equity.to_frame("equity").to_parquet(OUT_DIR / "portfolio_equity.parquet")
    (OUT_DIR / "portfolio_metrics.txt").write_text(result.summary(), encoding="utf-8")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax1.plot(result.equity.index, result.equity, label="Portfolio", linewidth=1.4)
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(f"Portfolio — {len(prices_by_symbol)} symbols, SmaCross(20,50)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    dd = result.equity / result.equity.cummax() - 1.0
    ax2.fill_between(dd.index, dd * 100, 0, color="red", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "portfolio_equity.png", dpi=120)
    print(f"saved {OUT_DIR / 'portfolio_equity.png'}")


if __name__ == "__main__":
    main()