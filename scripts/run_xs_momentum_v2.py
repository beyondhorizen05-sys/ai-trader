"""XS momentum v2 — redesign with monthly rebalance, 12-1 momentum.

Compares v1 (daily) against v2 variants on the wider universe.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.engine import BacktestConfig
from src.backtest.portfolio import run_portfolio_weights
from src.data.clean import load_raw, normalize
from src.data.sectors import get_sectors
from src.data.universe import SURVIVORSHIP_WARNING, get_universe
from src.strategies.xs_momentum import XSMomentum

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "notebooks"
OUT_DIR.mkdir(exist_ok=True)


def _load_prices(symbols: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for sym in symbols:
        try:
            out[sym] = normalize(load_raw(sym))
        except FileNotFoundError:
            pass
    return out


def _equal_weight_bh(prices: dict[str, pd.DataFrame], capital: float) -> pd.Series:
    common = None
    for df in prices.values():
        common = df.index if common is None else common.union(df.index)
    common = common.sort_values()
    closes = pd.DataFrame({s: df["adj close"].reindex(common).ffill() for s, df in prices.items()})
    ew = closes.pct_change().mean(axis=1).fillna(0.0)
    return capital * (1 + ew).cumprod()


def main(universe_name: str = "large_cap_100") -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(SURVIVORSHIP_WARNING)

    symbols = get_universe(universe_name)
    prices = _load_prices(symbols)
    print(f"\nLoaded {len(prices)} symbols\n")
    if len(prices) < 20:
        raise SystemExit(f"only {len(prices)} symbols; run scripts/fetch_universe {universe_name}")

    sectors = get_sectors(universe_name)
    cfg = BacktestConfig(commission_bps=1.0, slippage_bps=2.0)

    configs = [
        ("v1_daily_60d", dict(lookback=60, top_n=5, bottom_n=0,
                              rebalance_freq=None, skip_recent=0, sectors={})),
        ("v2_weekly_60d", dict(lookback=60, top_n=5, bottom_n=0,
                               rebalance_freq="W", skip_recent=0, sectors={})),
        ("v2_monthly_60d", dict(lookback=60, top_n=5, bottom_n=0,
                                rebalance_freq="ME", skip_recent=0, sectors={})),
        ("v2_monthly_12_1", dict(lookback=252, top_n=5, bottom_n=0,
                                 rebalance_freq="ME", skip_recent=21, sectors={})),
        ("v2_monthly_sect", dict(lookback=252, top_n=5, bottom_n=0,
                                 rebalance_freq="ME", skip_recent=21, sectors=sectors)),
    ]

    results = {}
    rows = []
    for name, kwargs in configs:
        strat = XSMomentum(min_history=30, **kwargs)
        weights = strat.generate_weights(prices)
        r = run_portfolio_weights(prices, weights, cfg)
        results[name] = r
        rows.append({"config": name, **r.metrics.to_dict()})
        print(f"\n=== {name} ===")
        print(r.summary())

    comparison = pd.DataFrame(rows).set_index("config")
    print("\n=== Comparison ===")
    print(comparison.to_string())
    comparison.to_csv(OUT_DIR / "xsm_v2_comparison.csv")

    ew_bh = _equal_weight_bh(prices, cfg.initial_capital)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    for name, r in results.items():
        ax1.plot(r.equity.index, r.equity, label=name, linewidth=1.3)
    ax1.plot(ew_bh.index, ew_bh, label="Equal-weight B&H",
             linewidth=1.0, alpha=0.6, linestyle="--", color="black")
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(f"XS Momentum v2 variants — {universe_name} ({len(prices)} symbols)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    best_name = max(results, key=lambda k: results[k].metrics["sharpe"])
    best_eq = results[best_name].equity
    dd = best_eq / best_eq.cummax() - 1.0
    ax2.fill_between(dd.index, dd * 100, 0, color="red", alpha=0.4)
    ax2.set_ylabel(f"DD% ({best_name})")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "xsm_v2_equity.png", dpi=120)
    print(f"\nsaved {OUT_DIR / 'xsm_v2_equity.png'}")
    print(f"saved {OUT_DIR / 'xsm_v2_comparison.csv'}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "large_cap_100"
    main(name)