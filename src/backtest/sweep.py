"""Parameter sweep utilities: grid-search a strategy over multiple parameter sets.

Reuses the existing backtest engine. Does not mutate global state.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest
from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    """Holds every run's metrics plus the parameters that produced them."""

    param_names: list[str]
    metrics: pd.DataFrame          # index: run id; columns: params + metric names
    equity: dict[int, pd.Series] = field(default_factory=dict, repr=False)

    def best(self, metric: str, maximize: bool = True, top_n: int = 10) -> pd.DataFrame:
        if metric not in self.metrics.columns:
            raise KeyError(f"metric {metric!r} not in results")
        return self.metrics.sort_values(metric, ascending=not maximize).head(top_n)

    def pivot(self, metric: str, x: str, y: str) -> pd.DataFrame:
        """Return an x-by-y matrix of the given metric (useful for heatmaps)."""
        if x not in self.metrics.columns or y not in self.metrics.columns:
            raise KeyError(f"{x!r} or {y!r} not in param columns")
        return self.metrics.pivot_table(index=y, columns=x, values=metric)

    def to_csv(self, path: str) -> None:
        self.metrics.to_csv(path, index_label="run_id")


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def _expand_grid(grid: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    """Expand a dict of lists into a list of single-value dicts.

    An empty grid produces an empty list (not ``[{}]``) so callers don't
    accidentally run a strategy with untested default parameters.
    """
    if not grid:
        return []
    keys = list(grid.keys())
    combos = list(itertools.product(*(list(grid[k]) for k in keys)))
    return [dict(zip(keys, c)) for c in combos]


def sweep(
    prices: pd.DataFrame,
    strategy_cls: type[Strategy],
    param_grid: dict[str, Iterable[Any]],
    config: BacktestConfig | None = None,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> SweepResult:
    """Run a backtest for every combination of parameters in `param_grid`.

    Parameters
    ----------
    prices : DataFrame
        Raw OHLCV frame (with `adj close`) for a single symbol.
    strategy_cls : type[Strategy]
        A Strategy subclass. Will be instantiated with `**params`.
    param_grid : dict[str, Iterable]
        e.g. {"fast": [5, 10, 20], "slow": [30, 50, 100]}
    config : BacktestConfig, optional
    feature_builder : callable, optional
        If provided, applied once to `prices` before signal generation.
        Use this to attach indicator columns (e.g. ``add_all``). If None,
        the strategy is responsible for computing its own indicators.
    """
    cfg = config or BacktestConfig()
    combos = _expand_grid(param_grid)
    param_names = list(param_grid.keys())

    if not combos:
        raise ValueError("param_grid produced no combinations")

    features = feature_builder(prices) if feature_builder is not None else prices

    rows: list[dict[str, Any]] = []
    equity: dict[int, pd.Series] = {}

    for i, params in enumerate(combos):
        try:
            strat = strategy_cls(**params)
            signals = strat.generate_signals(features)
            result = run_backtest(prices, signals, cfg)
        except Exception as e:  # noqa: BLE001 — we want to record and continue
            logger.warning("run %d failed for params %s: %s", i, params, e)
            continue

        row: dict[str, Any] = {**params}
        row.update(result.metrics.to_dict())
        row["run_id"] = i
        rows.append(row)
        equity[i] = result.equity

        logger.info(
            "run %3d/%d %s  total_ret=%6.2f%%  sharpe=%5.2f  mdd=%5.2f%%",
            i + 1,
            len(combos),
            params,
            row["total_return"] * 100,
            row["sharpe"],
            row["max_drawdown"] * 100,
        )

    metrics_df = pd.DataFrame(rows)
    # Stable column ordering: params first, then metrics
    cols = param_names + ["run_id"] + [c for c in metrics_df.columns if c not in param_names + ["run_id"]]
    metrics_df = metrics_df[cols].set_index("run_id")

    return SweepResult(param_names=param_names, metrics=metrics_df, equity=equity)