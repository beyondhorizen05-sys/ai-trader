"""Walk-forward validation.

Rolling train/test windows: pick the best parameters on each train slice,
apply them to the next unseen test slice, stitch the out-of-sample equity
together. This is the honest test of whether a parameter search generalizes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.sweep import _expand_grid
from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
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
            f"OOS Sharpe       {_sharpe(self.oos_returns):.2f}",
            f"OOS max DD       {_max_dd(self.oos_equity):.2%}",
        ]
        return "\n".join(lines)


def _sharpe(returns: pd.Series, periods: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * periods ** 0.5)


def _max_dd(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    dd = equity / equity.cummax() - 1.0
    return float(-dd.min())


def _date_slices(
    index: pd.DatetimeIndex,
    train_years: int,
    test_years: int,
    step_years: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    start = index.min()
    end = index.max()
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=test_years)
        if test_start >= end:
            break
        test_end = min(test_end, end + pd.Timedelta(days=1))
        folds.append((train_start, train_end, test_start, test_end))
        train_start = train_start + pd.DateOffset(years=step_years)
    return folds


def walk_forward(
    prices: pd.DataFrame,
    strategy_cls: type[Strategy],
    param_grid: dict[str, Iterable[Any]],
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
    select_metric: str = "sharpe",
    config: BacktestConfig | None = None,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> WalkForwardResult:
    if train_years <= 0 or test_years <= 0 or step_years <= 0:
        raise ValueError("train_years, test_years, step_years must be positive")
    if select_metric not in {"sharpe", "total_return", "calmar", "sortino"}:
        raise ValueError(f"unsupported select_metric: {select_metric!r}")

    cfg = config or BacktestConfig()
    combos = _expand_grid(param_grid)
    if not combos:
        raise ValueError("param_grid produced no combinations")

    index = prices.index
    folds = _date_slices(index, train_years, test_years, step_years)
    if not folds:
        raise ValueError("not enough history for the requested train/test windows")

    oos_pieces: list[pd.Series] = []
    fold_rows: list[dict[str, Any]] = []
    best_params_per_fold: list[dict[str, Any]] = []

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        train = prices.loc[(index >= tr_s) & (index < tr_e)]
        test = prices.loc[(index >= te_s) & (index < te_e)]
        if train.empty or test.empty:
            logger.warning("fold %d empty slice, skipping", i)
            continue

        train_feat = feature_builder(train) if feature_builder else train
        test_feat = feature_builder(test) if feature_builder else test

        best_row: pd.Series | None = None
        best_params: dict[str, Any] | None = None
        for params in combos:
            try:
                strat = strategy_cls(**params)
                sig = strat.generate_signals(train_feat)
                res = run_backtest(train, sig, cfg)
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

        strat = strategy_cls(**best_params)
        sig = strat.generate_signals(test_feat)
        test_result = run_backtest(test, sig, cfg)

        oos_pieces.append(test_result.returns)
        best_params_per_fold.append({**best_params, "fold": i})

        fold_rows.append(
            {
                "fold": i,
                "train_start": tr_s.date(),
                "train_end": (tr_e - pd.Timedelta(days=1)).date(),
                "test_start": te_s.date(),
                "test_end": (te_e - pd.Timedelta(days=1)).date(),
                **{f"best_{k}": v for k, v in best_params.items()},
                "train_sharpe": float(best_row["sharpe"]),
                "train_total_return": float(best_row["total_return"]),
                "test_sharpe": float(test_result.metrics["sharpe"]),
                "test_total_return": float(test_result.metrics["total_return"]),
                "test_max_drawdown": float(test_result.metrics["max_drawdown"]),
            }
        )

        logger.info(
            "fold %d  train %s..%s  test %s..%s  params=%s  train %s=%.3f  test %s=%.3f",
            i,
            tr_s.date(), (tr_e - pd.Timedelta(days=1)).date(),
            te_s.date(), (te_e - pd.Timedelta(days=1)).date(),
            best_params,
            select_metric, float(best_row[select_metric]),
            select_metric, float(test_result.metrics[select_metric]),
        )

    if not oos_pieces:
        raise RuntimeError("no folds produced out-of-sample returns")

    oos_returns = pd.concat(oos_pieces).sort_index()
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    oos_equity = cfg.initial_capital * (1.0 + oos_returns).cumprod()

    return WalkForwardResult(
        oos_equity=oos_equity,
        oos_returns=oos_returns,
        folds=pd.DataFrame(fold_rows),
        best_params_per_fold=best_params_per_fold,
        config=cfg,
    )