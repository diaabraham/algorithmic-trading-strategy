# Study 3 protocol: walk-forward machine learning (pre-registered)

This file was committed before any study 3 result was computed. The pipeline
configuration is frozen in `results/frozen_config.md` after the development
run and before the holdout run.

## Contamination note

SPY over 2020-01-02 to 2026-10-23 was already looked at in studies 1 and 2.
It is not a clean test any more. In this study, 2020-2023 is part of
development and is reported as development. The clean holdouts are:

* SPY from 2024-01-01 to the latest close.
* Markets never used in development.

## Development (SPY, data cut at 2023-12-31)

* **Recursive learning.**
  * An expanding-window walk-forward refits every model on the first trading
    day of each month. It uses all rows whose 21-day label is fully observed by
    that close.
  * Hyper-parameters are re-chosen once a year by purged, embargoed 5-fold CV
    inside the training window (purge = horizon + 1 bars, embargo = 21 bars).
  * The first refit needs 5 years (1,260 rows) of usable history. Every
    prediction is out of sample in time.
* **Label.** Log return from the next open to the open 21 bars later, minus
  T-bills.
* **Features (24, all point in time):**
  * Own price: 5/21/63/126/252-day returns, 12-1 momentum, gaps to the
    50/200-day SMA, 21/63-day volatility, the 21/252 volatility ratio, 63-day
    downside volatility and the 252-day drawdown.
  * VIX: level, 252-day z-score, 5-day change, and implied minus realised
    volatility.
  * Rates: T-bill level and its 63-day change; FRED T10Y3M term spread.
  * Credit: FRED BAA10Y spread and its 63-day change.
  * FRED initial claims (4-week average, year-on-year).
  * Sector breadth: share of the 9 SPDR sectors above their 200-day SMA; NaN
    before 1999.
  * Publication lags: FRED daily series +1 day; weekly claims +6 days after
    the week-ending date. FRED serves current vintages (revisions to claims are
    small; the market series are not revised).
* **Models (3 + ensemble):**
  * ridge, alpha in {10, 1000}
  * LightGBM, 200 trees, (leaves, min leaf) in {(7, 200), (15, 50)}
  * MLP (32, 16), L2 in {1e-3, 1e-1}, early stopping
  * ens: the mean of the three predictions
* **Sizing rules, decided weekly (Friday close), filled at the next open:**
  * `voltarget` (no ML; reference): w = long-run vol / 63-day vol.
  * `mv`: w = annualised predicted excess return / (3 x variance).
  * `tilt`: w = voltarget x (1 + 0.5 z), where z is the prediction divided by
    the standard deviation of the training-set predictions, clipped to +/-1.
  * `gate`: w = voltarget if the prediction is > 0, else 0.
  * All rules: weights in [0, 1.5]. The position is changed only when the
    target moves by more than 0.10.
* **Costs.** 2 bp per side (SPY, EFA, S&P index), 5 bp (EWJ, EWG, EWU, EWC).
  Idle cash earns T-bills; borrowed cash pays T-bill + 50 bp.
* **Trials.** 4 models x 3 ML sizings + the voltarget reference = **13
  configs**. Each one is logged in `results/trials_log.csv`. Studies 1 and 2
  tried 18 + 14 = 32 more on SPY, so the deflated Sharpe uses N = 45.
* **Development evaluation period:** 2000-01-03 to 2023-12-31. Benchmark: SPY
  buy-and-hold with the same fills and costs.
* **Selection:** the config with the highest development CAGR net of costs.

## Holdouts (run once with the frozen pipeline, no changes)

| Holdout | Instrument | Fills | Evaluation |
|---|---|---|---|
| H0 | SPY | next open | 2024-01-01 to latest |
| X1 | EFA | next open | first full month with predictions to latest |
| X2 | EWJ | next open | same |
| X3 | EWG | next open | same |
| X4 | EWU | next open | same |
| X5 | EWC | next open | same |
| X6 | ^SP500TR (total-return index, data cut at 1999-12-31) | next close (index opens unreliable) | 1996-01-02 to 1999-12-31 |

In X1-X6 each market is walk-forward trained on its own history. The macro
features are the same US series.

## "Reliably beats" criterion (all must hold)

1. Development CAGR > SPY buy-and-hold CAGR (2000-2023), net of costs.
2. H0 CAGR > SPY buy-and-hold CAGR.
3. CAGR > the market's own buy-and-hold CAGR in at least 4 of the 6
   cross-market holdouts X1-X6.
4. Deflated Sharpe ratio of the development active return (strategy minus
   buy-and-hold, daily) >= 0.95 with N = 45 trials.
5. Rolling 3-year (756-day) win rate vs buy-and-hold >= 60% in development.

If any one fails, the verdict is "does not reliably beat the S&P 500".
