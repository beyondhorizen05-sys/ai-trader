"""Walk-forward the redesigned long-only XS momentum (monthly, 12-1)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.engine import BacktestConfig
from src.backtest.portfolio import run_portfolio_weights
from src.backtest.walk_forward_xs import walk_forward_xs
from src.data.clean import load_raw, normalize
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
        raise SystemExit(f"only {len(prices)} symbols available")

    cfg = BacktestConfig(commission_bps=1.0, slippage_bps=2.0)

    def factory(**params) -> XSMomentum:
        return XSMomentum(bottom_n=0, min_history=30, rebalance_freq="ME",
                          skip_recent=21, **params)

    result = walk_forward_xs(
        prices_by_symbol=prices,
        strategy_factory=factory,
        param_grid={
            "lookback": [126, 189, 252],     # 6, 9, 12 months
            "top_n":    [5, 10, 20],
        },
        train_years=3,
        test_years=1,
        step_years=1,
        select_metric="sharpe",
        config=cfg,
        min_symbols=20,
    )

    print("\n=== Walk-forward XS momentum v2 summary ===")
    print(result.summary())
    print("\n=== Per-fold detail ===")
    print(result.folds.to_string(index=False))

    result.folds.to_csv(OUT_DIR / "wf_xsm_v2_folds.csv", index=False)
    result.oos_equity.to_frame("equity").to_parquet(OUT_DIR / "wf_xsm_v2_oos_equity.parquet")

    ew_bh = _equal_weight_bh(prices, cfg.initial_capital)

    fixed = XSMomentum(lookback=252, top_n=10, bottom_n=0, min_history=30,
                       rebalance_freq="ME", skip_recent=21)
    is_weights = fixed.generate_weights(prices)
    is_result = run_portfolio_weights(prices, is_weights, cfg)

    start = result.oos_equity.index[0]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(result.oos_equity.index, result.oos_equity,
            label="WF OOS (v2)", linewidth=1.8)
    ax.plot(ew_bh.loc[start:].index, ew_bh.loc[start:],
            label="Equal-weight B&H", linewidth=1.0, alpha=0.8)
    ax.plot(is_result.equity.loc[start:].index, is_result.equity.loc[start:],
            label="In-sample (LB=252, top=10)", linewidth=1.0,
            alpha=0.6, linestyle="--")
    ax.set_title(f"XS Momentum v2 — walk-forward OOS ({len(prices)} symbols)")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "wf_xsm_v2_equity.png", dpi=120)
    print(f"\nsaved {OUT_DIR / 'wf_xsm_v2_equity.png'}")
    print(f"saved {OUT_DIR / 'wf_xsm_v2_folds.csv'}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "large_cap_100"
    main(name)