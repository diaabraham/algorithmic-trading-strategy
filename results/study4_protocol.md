# Study 4 protocol: long-only factor tilts in large caps (pre-registered)

Committed before any study 4 return was computed or looked at. Only the file
layouts (column names and date ranges) of the data were inspected.

## Why this approach

Studies 1-3 tried to time the market and failed. Study 4 tests a different
source of return: holding a subset of large-cap stocks with a documented
factor premium (momentum, value, profitability, low volatility), fully
invested and unlevered.

## Data, and why not individual stocks

* **No free survivorship-free stock data.** Free point-in-time S&P 500
  membership lists exist (rebuilt from Wikipedia change logs), but free price
  histories for delisted members do not. Yahoo Finance drops most delisted
  tickers, so a free stock-level backtest would carry survivorship bias. It is
  not used.
* **What is used instead.** The Kenneth R. French Data Library: portfolio
  returns built from CRSP (US) and Bloomberg (developed ex-US), including
  delisted stocks. Its sorts use only information available at formation:
  * Momentum uses months t-12 to t-2.
  * Book equity and operating profitability come from the prior fiscal year,
    and the portfolios are formed in June.
  * Variance uses the 60 days before formation.
* **Limits:**
  * Returns are gross. Trading costs are applied below as a pre-set annual
    drag from turnover assumptions, because French does not publish
    turnover.
  * Rebalancing is assumed at month-end closes.
  * "BIG" means above the NYSE median market cap: large and upper mid caps,
    value-weighted.
  * SPY returns (Yahoo Finance, adjusted for splits and dividends, monthly)
    are the benchmark from 2000 on.

## Candidates (5; trial count carried forward 45, so N = 50)

| ID | Portfolio (value-weighted) | Source file | Assumed one-way turnover / yr | Cost drag / yr |
|---|---|---|---:|---:|
| F1 | BIG HiPRIOR (large-cap winners) | 6_Portfolios_ME_Prior_12_2 | 200% | 0.40% + 0.15% fee = 0.55% |
| F2 | BIG HiBM (large-cap value) | 6_Portfolios_2x3 | 30% | 0.06% + 0.15% = 0.21% |
| F3 | BIG HiOP (large-cap robust profitability) | 6_Portfolios_ME_OP_2x3 | 30% | 0.06% + 0.15% = 0.21% |
| F4 | Lo 20 variance quintile | Portfolios_Formed_on_VAR | 100% | 0.20% + 0.15% = 0.35% |
| F5 | Equal mix of F1, F2, F3, rebalanced monthly | - | 100% | 0.20% + 0.15% = 0.35% |

* **Cost drag** = 2 sides x turnover x 10 bp, plus 0.15%/yr for
  fees/implementation, deducted monthly (drag / 12).
* **Stress test:** 2x the drag.

## Periods

* **Development (selection):** US, 1963-07 to 1999-12. Benchmark: the French
  US market (Mkt-RF + RF).
* **H1, primary holdout (fresh):** US, 2000-01 to the latest month in the
  data. Benchmark: **SPY** total return.
* **H2, pre-sample holdout:** US, 1927-01 to 1963-06, vs the French US
  market. Only for candidates with data (F1, F2, and F5 restricted to F1 + F2
  is not used; F5 is n/a).
* **H3, cross-market holdout:** Developed ex-US, 1990-07 to the latest month,
  vs the French developed ex-US market. Only for F1, F2, F3 and F5 (F4 has no
  ex-US file).

Caveat: academic work published before this study documents these factors in
general terms, including post-2000 weakness for value. That knowledge cannot
be removed; it is why the selection rule and costs are fixed here in advance.

## Selection rule (development only)

**Selected = the candidate with the highest development CAGR net of the cost
drag.** The holdouts are run once for all five candidates. All results are
reported, but only the selected candidate decides the verdict.

## Pass criteria for the selected candidate (all applicable ones must hold)

1. H1 CAGR net of costs > SPY CAGR over the same months.
2. H1 CAGR with 2x costs > SPY CAGR.
3. H1 beats SPY in at least 2 of 3 subperiods: 2000-01 to 2008-12, 2009-01 to
   2017-12, 2018-01 to the end.
4. a) Deflated Sharpe ratio of the development active return (vs the market),
      N = 50 trials, is >= 0.95.
   b) Probabilistic Sharpe ratio (vs 0) of the H1 active return vs SPY is
      >= 0.95.
5. Robustness: beats its market's CAGR in H2 and H3, where data exist.

Verdict: PASS only if every applicable criterion holds.

## Small-fund feasibility

* Capacity is discussed from the portfolio definition: large caps,
  value-weighted.
* The assumed turnover is shown next to listed ETF implementations (MTUM,
  VLUE, QUAL, USMV/SPLV). Those ETFs only began in 2011-2013 and are not used
  in the test.
