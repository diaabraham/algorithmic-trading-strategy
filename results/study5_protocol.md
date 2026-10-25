# Study 5 protocol: ML combiner over anomaly and industry portfolios (pre-registered)

This file was committed before any study 5 return, feature or model output
was computed. Only metadata was inspected: file layouts, column names, date
ranges, signal counts, publication years and categories. No returns were
inspected.

## Why this design

Studies 3-4 timed a single index with small models, or picked a single
factor. The documented ML edge in equities (Gu, Kelly & Xiu 2020; Chen &
Zimmermann's Open Source Asset Pricing) comes from the cross-section: ranking
many portfolios built from many published signals. This study does that, at
the portfolio level.

## Data (free, point in time, survivorship-free)

* **Chen & Zimmermann, Open Source Asset Pricing.** `deciles_vw`:
  value-weighted decile portfolios and the decile 10 - decile 1 long-short
  portfolio for 179 published predictors, monthly, 1926-01 to 2024-12. They
  are built from CRSP/Compustat, including delisted stocks.
  * The signal documentation gives each predictor's publication year, sign
    and data category.
* **Ken French Data Library.** 49 value-weighted industry portfolios, the US
  market and the 1-month T-bill.
* **FRED monthly macro:** GS10, TB3MS, BAA, AAA, CPIAUCSL. Each is lagged one
  extra month for publication (two months for CPI). These are current
  vintages; ALFRED real-time vintages are not used. Only the yield and spread
  series are market data and are never revised.
* **SPY.** Total return, monthly (Yahoo, adjusted), as the holdout benchmark.

**Why not stock level.** No free survivorship-free stock-level price and
fundamentals source is available (Yahoo drops delisted firms; CRSP and
Compustat are paid). So the study works at the portfolio level, which is how
Chen & Zimmermann distribute these data.

**Point-in-time signal universe.** At the end of month t, a predictor is
usable only if its publication year is earlier than t's calendar year. Once
usable, its full history may be used for training, because a researcher at t
could rebuild it from the paper. Before 1990 only 13 predictors qualify; by
2014 there are 174.

**Not used:**
* LLMs. Every available LLM was trained on text written after most
  prediction dates, so any LLM-derived feature (news reading, signal
  generation) would leak future knowledge. They are excluded.
* Equal-weighted portfolios. They are dominated by microcaps and are not
  investable at these costs.

## Assets

* Long-only universe: the long leg of each usable predictor (decile 10 if its
  sign is +1, decile 1 if it is -1, value-weighted) plus the 49 industries.
* Overlay universe: each usable predictor's long-short portfolio, signed so
  that positive means the published direction.

## Features (end of month t, all known at t)

* **Own:** last-month return; 2-12 month return; 12-month volatility; 36- and
  60-month mean return; 36-month market beta; 12-month max drawdown.
* **Cross-sectional ranks** of the own 2-12 month return and volatility.
* **Signal:** years since publication; data category (one-hot: Accounting,
  Price, Trading, Analyst, Options, Other/13F/Event); an industry flag.
* **Factor momentum:** the predictor's long-short last-month and 2-12 month
  returns (0 for industries).
* **Macro:** market last-month and 12-month returns; market 12-month
  volatility; T-bill; term spread (GS10 - TB3MS); default spread (BAA - AAA);
  CPI inflation (year on year).

**Target:** next-month return minus that month's cross-sectional mean
(long-only panel), or the next-month long-short return (overlay panel).

## Recursive learning

* An expanding-window walk-forward, refit every month.
* At decision month t the model is trained on every row whose target month
  is <= t, for predictors usable at t.
* Predictions are made for month t + 1. The first prediction is for 1975-01.
* **LightGBM** (CPU, fixed): 300 trees, learning rate 0.05, 31 leaves, min
  leaf 500, bagging and feature fraction 0.7, seed 7.
* **MLP** (GPU, torch-directml on an AMD RX 9070): 3 hidden layers
  (512-256-128), ReLU, dropout 0.1, AdamW (lr 1e-3, weight decay 1e-4),
  batch 2048, 6 epochs, seed 7. Inputs are standardised on the training rows.
  CPU and GPU timings are reported on the same machine.
* Hyper-parameters are fixed in advance, with no tuning. That avoids spending
  extra trials.

## Candidates (5 new; trials carried forward 50, so N = 55)

| ID | Strategy |
|---|---|
| L0 | No ML: top-10 long-only assets by 2-12 month return (factor/industry momentum baseline) |
| L1 | Top-10 long-only assets by LightGBM prediction, equal weight, monthly |
| L2 | Top-10 by MLP prediction |
| L3 | Top-10 by the ensemble (average of the LightGBM and MLP cross-sectional ranks) |
| O1 | Market + overlay: 100% market (SPY in the holdout) plus k x the equal-weight top-10 long-short portfolios by ensemble rank. k = min(0.05 / trailing 36-month vol of that basket, 1.0), i.e. a 5% tracking-error budget. |

## Costs (pre-set)

* **Cost per side:** 20 bp for predictor portfolios (value-weighted deciles
  include small caps), 10 bp for industries.
* **Within-portfolio turnover** (one-way, per year), from the documented
  rebalancing period:
  * Monthly price/trading signals: 300%
  * Other monthly signals: 150%
  * Quarterly/semiannual: 100%
  * Annual or longer: 50%
  * Industries: 10%
  * Drag = 2 x turnover x cost per side / 12, per month held.
* **Switching cost:** sum |change in weight| x cost per side, each month.
* **O1:** both legs are charged. There is a 0.5%/yr short-borrow fee and
  0.5%/yr financing (T-bill + 50 bp minus the T-bill rebate on the extra long
  notional), both times k.
* **2x stress:** all of the above doubled.

## Periods

* Training history only: before 1975.
* **Development (selection):** 1975-01 to 2014-12. Benchmark: the French US
  market (SPY did not exist for most of it).
* **Holdout (never inspected):** 2015-01 to 2024-12. Benchmark: SPY total
  return.
* The development phase truncates every dataset at 2014-12 before computing
  anything.

## Selection rule

**Selected = the candidate with the highest development CAGR net of costs.**
The holdout is run once for all five candidates. Only the selected one
decides the verdict; all are reported.

## Pass criteria for the selected candidate (all must hold)

1. Holdout CAGR > SPY CAGR (2015-01 to 2024-12), net of costs.
2. Holdout CAGR with 2x costs > SPY CAGR.
3. It beats SPY in at least 2 of 3 holdout subperiods: 2015-01 to 2018-04,
   2018-05 to 2021-08, 2021-09 to 2024-12.
4. a) Deflated Sharpe ratio of the development active return (vs the market),
      N = 55, is >= 0.95.
   b) Probabilistic Sharpe ratio (vs 0) of the holdout active return vs SPY
      is >= 0.95.

## Capacity and small-fund feasibility

Reported from the share of value-weighted decile and industry market value
involved, the number of stocks per long leg (Nlong), and turnover.
