"""Walk-forward validation for cross-sectional (weight-based) strategies.

Each fold:
  - Grid-search parameters on the TRAIN window.
  - Skip params whose lookback exceeds the test window (avoid silent no-trades).
  - Pick the params that maximize `select_metric` on train.
  - Apply them to the TEST window (out-of-sample).
  - Append test-period portfolio returns to the stitched OOS series.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

import pandas as pd

from src.backtest.engine import BacktestConfig
from src.backtest.portfolio import run_portfolio_weights
from src.backtest.sweep import _expand_grid
from src.backtest.walk_forward import date_slices, max_dd, sharpe

logger = logging.getLogger(__name__)


class XSStrategy(Protocol):
    def generate_weights(self, prices_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame: ...


@dataclass
class WalkForwardXSResult:
    oos_equity: pd.Series
    oos_returns: pd.Series
    folds: pd.DataFrame
    best_params_per_fold: list[dict[str, Any]] = field(default_factory=list)
    config: BacktestConfig | None = field(default=None, repr=False)

    def summary(self) -> str:
        if self.folds.empty:
            return "no folds"
        lines = [
            f"folds            {len(self.folds)}",
            f"OOS total return {self.oos_equity.iloc[-1] / self.oos_equity.iloc[0] - 1:.2%}",
            f"OOS Sharpe       {sharpe(self.oos_returns):.2f}",
            f"OOS max DD       {max_dd(self.oos_equity):.2%}",
        ]
        return "\n".join(lines)


def _slice_prices(
    prices_by_symbol: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym, df in prices_by_symbol.items():
        sub = df.loc[(df.index >= start) & (df.index < end)]
        if not sub.empty:
            out[sym] = sub
    return out


def _lookback_fits(lookback: Any, test_bars: int, train_bars: int) -> bool:
    """A parameter set is valid only if the lookback fits in BOTH windows.

    We use the train window (3x longer) for warm-up during grid search, and
    the test window for the OOS run. If the lookback exceeds the test window
    length, the strategy will produce no signals during the test period —
    silently returning zeros. Reject such params up front.
    """
    try:
        lb = int(lookback)
    except (TypeError, ValueError):
        return True
    # Require at least 5 tradable bars after warm-up in the test window.
    return lb + 5 <= test_bars


def walk_forward_xs(
    prices_by_symbol: dict[str, pd.DataFrame],
    strategy_factory: Callable[..., XSStrategy],
    param_grid: dict[str, Iterable[Any]],
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
    select_metric: str = "sharpe",
    config: BacktestConfig | None = None,
    min_symbols: int = 5,
) -> WalkForwardXSResult:
    if select_metric not in {"sharpe", "total_return", "calmar", "sortino"}:
        raise ValueError(f"unsupported select_metric: {select_metric!r}")
    if not prices_by_symbol:
        raise ValueError("prices_by_symbol is empty")

    cfg = config or BacktestConfig()
    combos = _expand_grid(param_grid)
    if not combos:
        raise ValueError("param_grid produced no combinations")

    common_index: pd.DatetimeIndex | None = None
    for df in prices_by_symbol.values():
        common_index = df.index if common_index is None else common_index.union(df.index)
    assert common_index is not None
    common_index = common_index.sort_values()

    folds = date_slices(common_index, train_years, test_years, step_years)
    if not folds:
        raise ValueError("not enough history for the requested train/test windows")

    oos_pieces: list[pd.Series] = []
    fold_rows: list[dict[str, Any]] = []
    best_params_per_fold: list[dict[str, Any]] = []

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        train = _slice_prices(prices_by_symbol, tr_s, tr_e)
        test = _slice_prices(prices_by_symbol, te_s, te_e)
        if len(train) < min_symbols or len(test) < min_symbols:
            logger.warning(
                "fold %d skipping: train=%d test=%d (min=%d)",
                i, len(train), len(test), min_symbols,
            )
            continue

        # Number of bars in each window (use the longest available).
        train_bars = max((len(df) for df in train.values()), default=0)
        test_bars = max((len(df) for df in test.values()), default=0)

        # Filter params whose lookback can't fit the test window.
        valid_combos = [
            p for p in combos
            if _lookback_fits(p.get("lookback"), test_bars, train_bars)
        ]
        if not valid_combos:
            logger.warning(
                "fold %d skipping: no params fit (test_bars=%d, combos=%d)",
                i, test_bars, len(combos),
            )
            continue

        # --- Grid-search on train
        best_row: pd.Series | None = None
        best_params: dict[str, Any] | None = None
        for params in valid_combos:
            try:
                strat = strategy_factory(**params)
                weights = strat.generate_weights(train)
                res = run_portfolio_weights(train, weights, cfg)
            except Exception as e:  # noqa: BLE001
                logger.debug("fold %d train failed for %s: %s", i, params, e)
                continue
            m = res.metrics[select_metric]
            if best_row is None or m > best_row[select_metric]:
                best_row = res.metrics
                best_params = params

        if best_params is None:
            logger.warning("fold %d: no valid params, skipping", i)
            continue

        # --- Apply chosen params on test
        strat = strategy_factory(**best_params)
        test_weights = strat.generate_weights(test)
        test_result = run_portfolio_weights(test, test_weights, cfg)

        oos_pieces.append(test_result.returns)
        best_params_per_fold.append({**best_params, "fold": i})

        fold_rows.append(
            {
                "fold": i,
                "train_start": tr_s.date(),
                "train_end": (tr_e - pd.Timedelta(days=1)).date(),
                "test_start": te_s.date(),
                "test_end": (te_e - pd.Timedelta(days=1)).date(),
                "n_train_symbols": len(train),
                "n_test_symbols": len(test),
                "train_bars": train_bars,
                "test_bars": test_bars,
                **{f"best_{k}": v for k, v in best_params.items()},
                "train_sharpe": float(best_row["sharpe"]),
                "train_total_return": float(best_row["total_return"]),
                "test_sharpe": float(test_result.metrics["sharpe"]),
                "test_total_return": float(test_result.metrics["total_return"]),
                "test_max_drawdown": float(test_result.metrics["max_drawdown"]),
            }
        )

        logger.info(
            "fold %d  train %s..%s  test %s..%s  symbols=%d  params=%s  "
            "train %s=%.3f  test %s=%.3f",
            i,
            tr_s.date(), (tr_e - pd.Timedelta(days=1)).date(),
            te_s.date(), (te_e - pd.Timedelta(days=1)).date(),
            len(test),
            best_params,
            select_metric, float(best_row[select_metric]),
            select_metric, float(test_result.metrics[select_metric]),
        )

    if not oos_pieces:
        raise RuntimeError("no folds produced out-of-sample returns")

    oos_returns = pd.concat(oos_pieces).sort_index()
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    oos_equity = cfg.initial_capital * (1.0 + oos_returns).cumprod()

    return WalkForwardXSResult(
        oos_equity=oos_equity,
        oos_returns=oos_returns,
        folds=pd.DataFrame(fold_rows),
        best_params_per_fold=best_params_per_fold,
        config=cfg,
    )