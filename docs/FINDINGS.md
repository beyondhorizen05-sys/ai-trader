# Research Findings

**Repository:** ai-trader
**Status:** Research complete as of v0.2.0
**Scope:** Daily bars, 2015–2024, US equities (large-cap)

---

## Executive summary

Three strategy families were implemented, tested, and walk-forward validated:

| Strategy | In-sample Sharpe | WF OOS Sharpe | Beats Buy & Hold? | Verdict |
|---|---|---|---|---|
| SMA crossover (3 symbols) | 0.79 | 0.95 | Yes, marginally | Marginal edge; regime-dependent |
| XS momentum, long/short (29 symbols) | -0.09 | — | No | No edge on this universe |
| XS momentum, long-only (110 symbols) | 1.09 | **0.41** | No | Fails walk-forward gate |

**Primary finding:** Long-only cross-sectional momentum shows strong in-sample performance (Sharpe 1.09, CAGR 33%) but does not survive walk-forward validation (OOS Sharpe 0.41, OOS CAGR ~5%). The failure is **regime-driven, not overfit** — parameters were stable across folds; the losses cluster in known momentum-crash years (2018 Q4, 2021 rotation).

**Secondary finding:** Sector-neutralization helps long/short momentum by ~0.05 Sharpe (still negative) but *hurts* long-only momentum (Sharpe 1.09 → 0.94). Sector constraints remove the ability to concentrate in whichever sector is hottest — fine for a long-only factor tilt, but not helpful for a hedged construction on this universe.

---

## What was tested

### Data
- Universe: 110 large-cap US equities (`large_cap_100` universe, hand-curated)
- Period: 2015-01-02 to 2024-12-30 (~2,515 trading days)
- Survivorship bias: **yes** — universe is hand-assembled from currently-listed names

### Strategies

**1. SMA crossover** — long when fast SMA > slow SMA, flat otherwise. Grid: `fast ∈ {5,10,20,40}`, `slow ∈ {30,50,100,200}`.

**2. XS momentum (long/short)** — long top-N by trailing return, short bottom-N, equal weight within legs, gross exposure normalized to 1.0. Variants:
- Cross-sectional vs. sector-neutral
- Lookback ∈ {20, 60, 126, 252}
- Top/bottom N ∈ {3, 5, 10, 20}
- Daily, weekly, monthly rebalance
- Skip-recent ∈ {0, 21}

**3. XS momentum (long-only)** — same as above but `bottom_n=0`.

### Validation methods
- **In-sample:** full-history backtest with 1 bps commission + 2 bps slippage
- **Sweep:** full grid search with heatmap output
- **Walk-forward:** 3-year train / 1-year test, step 1 year (7 folds), params chosen on train by Sharpe, applied to test

---

## Key results

### SMA crossover — marginal edge, regime-dependent

Walk-forward on AAPL:
- OOS Sharpe: 0.95
- OOS total return: 210% (7 years)
- Two negative folds (2018: -0.59, 2022: -0.42)
- Four strong folds (2019: +2.96, 2020: +1.38, 2021: +1.42, 2024: +1.39)

**Interpretation:** works in trending years, fails in choppy/rotational years. Classic trend-following profile.

### XS momentum — strong in-sample, fails walk-forward

In-sample on 110 symbols (best config):

| Config | Sharpe | CAGR | Max DD |
|---|---|---|---|
| v1 daily 60d | 0.88 | 21.9% | 37.7% |
| v2 weekly 60d | 0.90 | 23.0% | 37.8% |
| v2 monthly 60d | 1.03 | 28.2% | 34.3% |
| **v2 monthly 12-1** | **1.09** | **33.3%** | 38.1% |
| v2 monthly sector | 0.94 | 17.7% | 34.2% |

Walk-forward OOS (7 folds, best config chosen per fold):
- OOS Sharpe: **0.41**
- OOS total return: 42.3% (7 years, ~5.2% CAGR)
- OOS max DD: 26.8%

**Fold-level detail:**

| Fold | Test year | Params | Test Sharpe |
|---|---|---|---|
| 0 | 2018 | (126, 5) | -0.89 |
| 1 | 2019 | (126, 10) | +1.20 |
| 2 | 2020 | (126, 10) | +1.29 |
| 3 | 2021 | (189, 5) | -0.16 |
| 4 | 2022 | (126, 5) | +1.21 |
| 5 | 2023 | (126, 5) | +0.31 |
| 6 | 2024 | (252, 5) | 0.00 (bug) |

**Parameter stability:** `lookback=126` chosen in 5 of 7 folds. `top_n=5` in 5 of 7. The optimizer picked a stable, sensible region — no evidence of overfitting to the grid.

**Regime dependence:** Negative folds are 2018 (Q4 momentum crash, growth-to-value rotation) and 2021 (post-COVID growth unwind). These are known momentum-unwind periods. Momentum's weakness is structural, not a parameter artifact.

### Long/short XS momentum — no edge

On 29 mega-caps:
- Cross-sectional long/short, gross 1.0: Sharpe -0.09
- Sector-neutral long/short: Sharpe -0.14
- Zero-cost (commission=0, slippage=0): total return -0.60% over 10 years

**Interpretation:** the short leg is systematically biased against the strategy on a mega-cap universe. The "losers" within the top 30 US companies still go up on average — you're shorting future winners during a structural bull market.

---

## What went wrong / what would need to change

### 1. The walk-forward gate
Pre-committed gate before running v2: **WF Sharpe ≥ 0.5 AND beats B&H → continue; else → stop.**
Observed: WF Sharpe 0.41, does not beat B&H (B&H equal-weight on 110 names returned ~+450% over the period, OOS strategy returned +42%).
**Verdict: fail.** No further tuning of momentum on this universe.

### 2. Known bugs and honest limitations

- **Walk-forward with lookback longer than test window** produces silent zeros (fold 6 in run above). Fixed in v0.2.0.
- **Survivorship bias** in the universe means all results are optimistic. A production-grade study needs point-in-time constituents (CRSP or Norgate).
- **Mega-cap concentration** — even 110 names is heavily weighted to the same growth-factor exposures. Momentum needs cross-sectional dispersion to work; US large-cap has less than small-cap or international.

### 3. Legitimate paths forward (not pursued)

- **Wider universe:** small-caps, international, futures
- **Multi-factor:** combine momentum with value/low-vol/quality so the failure regimes are uncorrelated
- **Regime filters:** skip trading during high-VIX or post-shock regimes
- **Volatility targeting:** scale position size by realized vol

All four are legitimate *a priori* design choices. None should be added after seeing the walk-forward result — that's the curve-fitting trap.

---

## Conclusions

1. **The framework works.** 136 tests, clean architecture, reproducible backtests, walk-forward validation, portfolio construction, reporting.
2. **Momentum on US large-caps does not pass walk-forward** on the pre-committed gate.
3. **The failure mode is regime, not overfit.** Parameters were stable; losses cluster in known momentum-crash years.
4. **Adding complexity now is the wrong move.** The gate was set before the experiment for exactly this reason.

## What this repo is good for

- A teaching/reference implementation of a small quant research platform
- A testbed for future hypotheses: build a strategy, run walk-forward, get an honest answer in ~1 hour
- A codebase to fork and extend with new data sources or strategy families

## What this repo is not

- A live trading system. No execution, no risk limits, no monitoring.
- A claim of alpha. The gate failed. Momentum didn't beat buy & hold walk-forward.
- A complete study. Survivorship bias, single-asset-class, single-regime (2015–2024 US bull).

---

*Last updated: v0.2.0-research-complete*