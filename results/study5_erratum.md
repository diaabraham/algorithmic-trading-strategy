# Study 5 erratum: publication filter missing from L0

This was found on 2026-10-25, after the holdout, while writing the live tracker.

## The problem

The protocol says an anomaly is usable only from the year after its paper was
published. The ML candidates applied this rule in `walkforward.py`, but the
momentum baseline L0 did not. Its score in `evaluate.scores` ranked every asset
with data, including anomalies that had not been published yet:

* development: 1,913 of its 4,800 positions;
* holdout: 8 of 1,200 positions.

## Effect

These figures were recomputed with the filter applied, for disclosure only.
`results/study5_dev.json` and `results/study5_holdout.json` are left as they were
originally recorded.

| | As recorded | With publication filter |
|---|---:|---:|
| Dev CAGR (1975-2014) | 16.75% | 16.47% |
| Dev CAGR, 2x costs | 15.11% | 15.42% |
| Dev Sharpe / max DD | 0.60 / -57.7% | 0.61 / -57.8% |
| Dev deflated Sharpe (N=55) | 0.75 | 0.75 |
| Holdout CAGR (2015-2024) | 18.38% | 18.35% |
| Holdout CAGR, 2x costs | 16.78% | 16.76% |
| Holdout Sharpe / max DD | 0.82 / -21.6% | 0.82 / -21.6% |
| Subperiods won vs SPY | 2/3 | 2/3 |
| Holdout active PSR | 0.932 | 0.932 |

## What it changes

* The selection does not change. L0 still has the highest development CAGR;
  the next best is L3 at 15.98%.
* The verdict does not change. It is still FAIL, on criteria 4a and 4b.
* The live tracker (`src/study5/live.py`) applies the filter, as the protocol
  requires.
* For live decisions the filter never binds. The latest publication year in the
  universe is 2016.
