# Pre-registered candidates: study 2

This file was committed **before** any out-of-sample (OOS) number for these
candidates was computed. The selection rule, the shortlist rule and every
parameter below are fixed. Nothing is re-tuned after the OOS run.

Code: `src/portfolio_backtest.py`, `src/candidates.py`, `scripts/run_preregistered_study.py`.

## Periods

| Period | Dates | Use |
|---|---|---|
| Warm-up | 2005-01-03 to 2006-12-29 | Look-back windows only |
| In-sample (IS) | 2007-01-03 to 2019-12-31 | All selection |
| Out-of-sample (OOS) | 2020-01-02 to latest close | Evaluated once, shortlist only |

The IS phase (`--phase is`) truncates the data at 2019-12-31 before computing
any signal, so it cannot see OOS prices.

## Universe

These ETFs all traded before 2007, so there is no survivorship selection after
the sample starts. DBC (inception Feb 2006) gets zero weight until it has 12
months of history.

* Equity: SPY, QQQ, IWM, EFA, EEM
* Bonds: TLT, IEF
* Real assets: GLD, DBC
* SPDR sectors (1998): XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY

Prices are Yahoo Finance daily bars, adjusted for splits and dividends. The
risk-free rate is the 3-month T-bill (^IRX).

## Execution and costs (all candidates and the benchmark)

* Weights are decided on a close using data up to that close only. They fill
  at the **next open**.
* Monthly candidates decide on the last trading day of each month.
* The volatility-regime strategy (family E) decides daily.
* Costs per side, on traded notional:
  * 2 bp: SPY, QQQ, IWM, EFA, TLT, IEF, GLD
  * 3 bp: sector SPDRs
  * 5 bp: EEM, DBC
* Idle cash earns the T-bill rate.
* Borrowed cash (gross exposure > 100%) pays **T-bill + 50 bp**.
* Short positions pay a 1%/yr borrow fee.
* Benchmark: SPY buy-and-hold, bought at the first open of each period and
  charged the same cost.

## Base candidates (14)

"Vol target" means scaling weights so that the trailing 63-day volatility of
the portfolio is 15%/yr, capped at the stated maximum gross leverage.

| ID | Family | Description |
|---|---|---|
| A1 | Vol-managed SPY | SPY weight = min(15% / 21-day realised vol, 1.5) |
| A2 | Vol-managed SPY | SPY weight = min(15% / 63-day realised vol, 1.5) |
| B1 | Trend equity | SPY if its close > 10-month (210-day) SMA, else IEF; unlevered |
| B2 | Trend equity | B1, vol target 15%, max 1.5x |
| B3 | Dual momentum (Antonacci GEM) | Hold SPY or EFA, whichever has the higher 12-month return, if that return beats T-bills over the same 12 months; else IEF; unlevered |
| B4 | Dual momentum | B3, vol target 15%, max 1.5x |
| C1 | Multi-asset trend | 9 ETFs (SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC); long the ones with a positive 12-month excess return, inverse-63-day-vol weights; portfolio vol target 15%, max 2.0x |
| C2 | Risk parity | Same 9 ETFs, always long, inverse-63-day-vol weights; vol target 15%, max 2.0x |
| C3 | Risk parity + trend | C2, but an asset gets zero weight when its close is below its 10-month SMA; vol target 15%, max 2.0x |
| D1 | QQQ/SPY rotation | Hold whichever of QQQ and SPY has the higher 6-month return; IEF if that ETF is below its 10-month SMA; unlevered |
| D2 | Sector momentum | Equal-weight the top 3 of the 9 sectors by 12-1 month return; unlevered |
| D3 | Sector momentum + filter | D2 while SPY is above its 10-month SMA, else IEF; unlevered |
| E1 | Repo strategy, levered | VolatilityRegimeStrategy defaults (ML off); signal x min(15% / 63-day SPY vol, 1.5), leverage reset monthly |
| E2 | Repo strategy, levered | E1 with the walk-forward ML gate on (0.55 / 0.45) |

N = 14 trials for the multiple-testing adjustment.

## Selection rule (in-sample only)

1. Compute every candidate's IS Sharpe ratio: daily returns minus T-bills,
   annualised.
2. **Selected = the candidate with the highest IS Sharpe.**
3. **Vol-matched version.** `selected_volmatched` scales the selected
   candidate's weights by k = (SPY IS volatility) / (selected IS volatility),
   both measured in-sample. It is capped at 2.0x gross and financed at
   T-bill + 50 bp. k is fixed from IS data and is not changed OOS. This is the
   ex-ante test of whether a Sharpe edge turns into a higher raw return than
   SPY.
4. Robustness reported (not used to select):
   * IS Sharpe in four sub-periods: 2007-09, 2010-12, 2013-15, 2016-19.
   * A walk-forward check inside IS: each year from 2010 to 2019, pick the
     candidate with the best Sharpe from 2007 to the prior year, and record its
     return in the following year.
   * The Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) of the selected
     candidate, with N = 14 trials and the cross-candidate variance of IS
     Sharpe.

## OOS shortlist (fixed by rule, all reported in full)

* The selected candidate and `selected_volmatched`
* The best-IS-Sharpe candidate from each family A, B, C, D, E (no duplicates)
* The candidate with the highest IS CAGR
* SPY buy-and-hold

OOS also reports an **ex-post leverage-matched comparison** for each
shortlisted strategy. Its excess returns are scaled so its OOS volatility
equals SPY's OOS volatility, with borrowed cash financed at T-bill + 50 bp. It
is labelled hypothetical because the scaling uses OOS volatility.

## What counts as "beats SPY"

A candidate beats SPY OOS on raw return only if its OOS CAGR, net of costs and
financing, is higher than SPY buy-and-hold's CAGR on the same dates. The
ex-post leverage-matched numbers are reported for context only.
