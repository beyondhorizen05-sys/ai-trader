"""Grid-search SmaCross parameters on processed AAPL data.

Produces:
  notebooks/sweep_results.csv       every (fast, slow) row with metrics
  notebooks/sweep_sharpe.png        heatmap of Sharpe
  notebooks/sweep_return.png        heatmap of total return
  notebooks/sweep_drawdown.png      heatmap of max drawdown
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.backtest.engine import BacktestConfig
from src.backtest.sweep import sweep
from src.data.clean import load_raw, normalize
from src.features.indicators import add_all
from src.strategies.sma_cross import SmaCross

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "notebooks"
OUT_DIR.mkdir(exist_ok=True)


def _heatmap(matrix, title: str, xlabel: str, ylabel: str, cbar_label: str,
             cmap: str = "viridis", fmt: str = ".2f", out_path: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix.values, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, format(val, fmt), ha="center", va="center",
                    color="white" if _dark(val, matrix.values) else "black", fontsize=8)

    fig.colorbar(im, ax=ax, label=cbar_label)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
        print(f"saved {out_path}")
    plt.close(fig)


def _dark(val: float, arr) -> bool:
    """Rough contrast check — dark cells get white text."""
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    if vmax == vmin:
        return True
    return (val - vmin) / (vmax - vmin) > 0.6


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # 1. Load data
    prices = normalize(load_raw("AAPL"))
    print(f"Loaded {len(prices)} bars: {prices.index[0].date()} -> {prices.index[-1].date()}")

    # 2. Grid
    fasts = [5, 10, 15, 20, 30, 40]
    slows = [30, 50, 75, 100, 150, 200]
    # Keep fast < slow; sweep() catches invalid combos via SmaCross.__post_init__
    grid = {"fast": fasts, "slow": slows}

    # 3. Run
    cfg = BacktestConfig(commission_bps=1.0, slippage_bps=2.0)
    result = sweep(
        prices=prices,
        strategy_cls=SmaCross,
        param_grid=grid,
        config=cfg,
        feature_builder=add_all,
    )

    # 4. Save raw results
    csv_path = OUT_DIR / "sweep_results.csv"
    result.to_csv(str(csv_path))
    print(f"saved {csv_path}  ({len(result.metrics)} runs)")

    # 5. Ranked tables
    print("\n=== Top 10 by Sharpe ===")
    print(result.best("sharpe", top_n=10).to_string())
    print("\n=== Top 10 by Total Return ===")
    print(result.best("total_return", top_n=10).to_string())
    print("\n=== Top 10 by Calmar (return / max DD) ===")
    print(result.best("calmar", top_n=10).to_string())

    # 6. Heatmaps
    sharpe = result.pivot("sharpe", x="fast", y="slow")
    total_ret = result.pivot("total_return", x="fast", y="slow") * 100
    mdd = result.pivot("max_drawdown", x="fast", y="slow") * 100

    _heatmap(sharpe, "Sharpe (fast x slow)", "fast", "slow", "Sharpe",
             out_path=OUT_DIR / "sweep_sharpe.png")
    _heatmap(total_ret, "Total return % (fast x slow)", "fast", "slow", "Total %",
             cmap="RdYlGn", out_path=OUT_DIR / "sweep_return.png")
    _heatmap(mdd, "Max drawdown % (fast x slow)", "fast", "slow", "MDD %",
             cmap="RdYlGn_r", out_path=OUT_DIR / "sweep_drawdown.png")

    # 7. Baseline row for reference
    print("\n=== Baseline SMA(20,50) ===")
    baseline = result.metrics[(result.metrics["fast"] == 20) & (result.metrics["slow"] == 50)]
    print(baseline.to_string())


if __name__ == "__main__":
    main()