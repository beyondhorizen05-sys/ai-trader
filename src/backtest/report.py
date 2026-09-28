"""Report generation: text summaries + matplotlib figures."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.engine import BacktestResult


def plot_equity(
    result: BacktestResult,
    benchmark: Optional[pd.Series] = None,
    title: str = "Equity curve",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax1.plot(result.equity.index, result.equity, label="Strategy", linewidth=1.4)
    if benchmark is not None:
        ax1.plot(benchmark.index, benchmark, label="Benchmark", linewidth=1.0, alpha=0.8)
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(alpha=0.3)

    dd = result.equity / result.equity.cummax() - 1.0
    ax2.fill_between(dd.index, dd * 100, 0, color="red", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
    return fig


def plot_trades(
    result: BacktestResult,
    prices: pd.Series,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(prices.index, prices, color="black", linewidth=1.0, label="Price")
    trades = result.trades
    if not trades.empty:
        for _, t in trades.iterrows():
            color = "green" if t["side"] == "long" else "red"
            ax.axvspan(t["entry_date"], t["exit_date"], color=color, alpha=0.1)
    ax.set_ylabel("Price")
    ax.set_title("Trade windows")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
    return fig


def write_text_report(result: BacktestResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 60,
        "BACKTEST REPORT",
        "=" * 60,
        result.summary(),
        "",
        "=" * 60,
        "TRADES (first 20)",
        "=" * 60,
    ]
    if result.trades.empty:
        lines.append("(no trades)")
    else:
        lines.append(result.trades.head(20).to_string(index=False))
    out_path.write_text("\n".join(lines), encoding="utf-8")