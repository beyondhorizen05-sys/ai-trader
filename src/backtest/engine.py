"""Event-driven backtest engine.

Conventions
-----------
- Signals live at the close of bar ``t`` and are executed at the open of ``t+1``.
- Prices are total-return adjusted (OHLC scaled by adj_close / close) so that
  splits and dividends don't create fake gaps in the equity curve.
- Costs: ``commission_bps`` + ``slippage_bps`` are charged on the **turnover**
  at each bar, where turnover = |position_t - position_{t-1}|.
- No leverage, no margin, no borrow cost. Short positions earn ``-ret``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest import metrics as M

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_bps: float = 1.0     # one-way, per unit turnover
    slippage_bps: float = 2.0       # one-way, per unit turnover
    position_size: float = 1.0      # fraction of equity, in (0, 1]
    rf: float = 0.0                 # annual risk-free rate for Sharpe/Sortino

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs must be non-negative")
        if not (0 < self.position_size <= 1):
            raise ValueError("position_size must be in (0, 1]")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    price: pd.Series           # execution price series (adjusted open)
    trades: pd.DataFrame
    metrics: pd.Series
    config: BacktestConfig = field(repr=False)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"trades           {int(m['n_trades'])}",
            f"total return     {m['total_return']:.2%}",
            f"CAGR             {m['cagr']:.2%}",
            f"volatility       {m['volatility']:.2%}",
            f"Sharpe           {m['sharpe']:.2f}",
            f"Sortino          {m['sortino']:.2f}",
            f"max drawdown     {m['max_drawdown']:.2%}",
            f"calmar           {m['calmar']:.2f}",
            f"hit rate         {m['hit_rate']:.2%}",
            f"profit factor    {m['profit_factor']:.2f}",
            f"exposure         {m['exposure']:.2%}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Price preparation
# ---------------------------------------------------------------------------

def prepare_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Scale OHLC by adj_close / close so all price columns are total-return adjusted.

    Returns a DataFrame with columns: open, high, low, close, volume.
    The 'close' column becomes the adjusted close.
    """
    required = {"open", "high", "low", "close", "adj close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    ratio = df["adj close"] / df["close"]
    out = pd.DataFrame(index=df.index)
    for c in ["open", "high", "low"]:
        out[c] = df[c] * ratio
    out["close"] = df["adj close"]
    out["volume"] = df["volume"]
    out.index.name = df.index.name
    return out


# ---------------------------------------------------------------------------
# Trade extraction
# ---------------------------------------------------------------------------

def extract_trades(
    position: pd.Series,
    fill_price: pd.Series,
) -> pd.DataFrame:
    """Walk the executed position series and emit one row per round-trip trade.

    `position` is the *executed* position (already lagged).
    `fill_price` is the price at which the position change is filled.
    """
    rows = []
    pos = position.fillna(0).astype(int)
    idx = pos.index

    open_side = 0
    open_date = None
    open_price = None

    for i, ts in enumerate(idx):
        p = int(pos.iloc[i])
        price = float(fill_price.iloc[i])

        if p == open_side:
            continue

        # close existing trade if any
        if open_side != 0 and open_date is not None:
            ret = open_side * (price / open_price - 1.0)
            rows.append(
                {
                    "entry_date": open_date,
                    "exit_date": ts,
                    "side": "long" if open_side > 0 else "short",
                    "entry_price": open_price,
                    "exit_price": price,
                    "bars_held": int(i - idx.get_loc(open_date)),
                    "return": float(ret),
                }
            )

        # open new trade if p != 0
        if p != 0:
            open_side = p
            open_date = ts
            open_price = price
        else:
            open_side = 0
            open_date = None
            open_price = None

    # force-close any open trade on last bar
    if open_side != 0 and open_date is not None:
        ts = idx[-1]
        price = float(fill_price.iloc[-1])
        ret = open_side * (price / open_price - 1.0)
        rows.append(
            {
                "entry_date": open_date,
                "exit_date": ts,
                "side": "long" if open_side > 0 else "short",
                "entry_price": open_price,
                "exit_price": price,
                "bars_held": int(len(idx) - 1 - idx.get_loc(open_date)),
                "return": float(ret),
            }
        )

    cols = ["entry_date", "exit_date", "side", "entry_price", "exit_price",
            "bars_held", "return"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def run_backtest(
    prices: pd.DataFrame,
    signals: pd.Series,
    config: Optional[BacktestConfig] = None,
) -> BacktestResult:
    """Run a single-asset backtest.

    Parameters
    ----------
    prices : DataFrame
        Must contain at least `open`, `close` (raw). If `adj close` is present
        it will be used to total-return adjust all prices.
    signals : Series
        Position target in {-1, 0, +1} indexed the same as `prices`.
        Signals are treated as decisions taken at the **close** of each bar and
        are executed at the **open** of the next bar.
    config : BacktestConfig, optional
    """
    cfg = config or BacktestConfig()

    if not prices.index.equals(signals.index):
        raise ValueError("prices and signals must share the same index")
    if signals.dropna().pipe(lambda s: set(s.unique()) - {-1, 0, 1}):
        raise ValueError("signals must only contain -1, 0, 1")

    # ---- 1. total-return adjust prices
    adj = prepare_prices(prices) if "adj close" in prices.columns else prices.copy()
    exec_price = adj["open"]                    # fill price = next bar open

    # ---- 2. lag signals: signal at close of t -> executed position from t+1
    target = signals.reindex(prices.index).fillna(0).astype(int)
    executed = target.shift(1).fillna(0).astype(int)

    # ---- 3. position sizing
    pos_sized = executed.astype(float) * cfg.position_size

    # ---- 4. bar returns at the fill price
    #     Return from open of t to open of t+1 is the P&L earned for holding
    #     during bar t. We compute the (adjusted) close-to-close return and
    #     apply it to the position held during bar t.
    close_ret = adj["close"].pct_change().fillna(0.0)

    gross_ret = pos_sized * close_ret

    # ---- 5. costs applied on turnover at each bar
    turnover = pos_sized.diff().abs().fillna(pos_sized.abs())
    cost_bps = cfg.commission_bps + cfg.slippage_bps
    cost_ret = turnover * (cost_bps / 10_000.0)

    net_ret = gross_ret - cost_ret

    # ---- 6. equity curve
    equity = cfg.initial_capital * (1.0 + net_ret).cumprod()

    # ---- 7. trades
    trades = extract_trades(executed, exec_price)

    # ---- 8. metrics
    trade_rets = trades["return"] if len(trades) else pd.Series(dtype=float)
    metrics = pd.Series(
        {
            "total_return": M.total_return(equity),
            "cagr": M.cagr(equity),
            "volatility": M.volatility(net_ret),
            "sharpe": M.sharpe(net_ret, rf=cfg.rf),
            "sortino": M.sortino(net_ret, rf=cfg.rf),
            "max_drawdown": M.max_drawdown(equity),
            "calmar": M.calmar(equity),
            "hit_rate": M.hit_rate(trade_rets),
            "profit_factor": M.profit_factor(trade_rets),
            "exposure": M.exposure(executed),
            "n_trades": float(len(trades)),
        }
    )

    return BacktestResult(
        equity=equity,
        returns=net_ret.rename("net_return"),
        position=executed.rename("position"),
        price=exec_price.rename("exec_price"),
        trades=trades,
        metrics=metrics,
        config=cfg,
    )