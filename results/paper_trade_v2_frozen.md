# paper-trade-v2: frozen study 5 rule L0

Frozen on 2026-10-25 at tag `paper-trade-v2`. No parameter differs from
`results/study5_frozen_config.md`.

## Rule

At each month end t:

1. For every asset in the frozen universe, compute its 2-12 month momentum:
   the compounded return over months t-11 to t-1.
2. An asset is eligible if it:
   * has a return at month t;
   * has 11 months of momentum history;
   * if it is an anomaly, was published before t's calendar year.
3. Hold the 10 eligible assets with the highest momentum, equal-weighted, for
   month t+1.

## Universe

`results/paper_trade_v2_universe.csv` lists the 228 assets:

* 179 Chen-Zimmermann anomalies. Each is the value-weighted decile leg in the
  published long direction.
* The 49 French value-weighted industries.

## Costs (study 5)

* Within-portfolio drag: 2 x turnover x cost per side / 12, with the per-asset
  turnover in the universe file.
* Switching: the sum of |weight change| x cost per side.
* Cost per side: 20 bp for anomaly legs, 10 bp for industries.
* Each tracking window starts from cash, so the full entry cost is charged.
* SPY is charged 2 bp at entry.

## Dates

* **Freeze date:** 2026-10-25.
* **First decision:** 2024-12-31. This is the latest month end with data for the
  whole universe: Chen-Zimmermann release 2025-10 has data through 2024-12;
  French industries run through 2026-08.
* **Pre-live months, 2025-01 to 2026-10:** tracked but not judged. The holding
  months come before the freeze.
* **Official live window:** starts with holding month 2026-11, decided at the
  2026-10-31 month end.

## How live returns are measured

The anomaly legs cannot be priced in real time. Chen-Zimmermann build them from
CRSP and Compustat and publish them about once a year:

* data through 2022-12: released 2023-08;
* data through 2023-12: released 2024-08;
* data through 2024-12: released 2025-10.

`scripts/paper_trade_update.py` handles this as follows:

* A decision is logged once the first release covering its decision month exists.
* The log (`results/paper_trade_live_v2_picks.csv`) is append-only. A logged
  pick is never recomputed from a later, revised release.
* Monthly returns are filled in once a release covers the holding month.
* The returns come from the published portfolio returns, not from an ETF proxy.

The code and parameters are fixed at this tag, and every pick is a mechanical
function of third-party data. The freeze therefore shows the picks cannot have
been chosen with hindsight.

## Judging

The rule is judged at 12 months (holding months 2026-11 to 2027-10) and at
24 months (2026-11 to 2028-10). At each checkpoint it passes only if its CAGR
net of costs beats SPY total return over the same months.

Expected timing, if the release pattern holds:

* 12-month result: needs data through 2027-10, so it should be measurable around
  late 2028.
* 24-month result: around late 2029.
