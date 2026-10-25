# Study 5 frozen configuration

This file was frozen before the holdout run. The code and development results
are at commit `804ef6484cd8028355cb1125e2fd0c7301ed9336`. The protocol is `results/study5_protocol.md`
(commit `469f7eb`).

The holdout is run once:

1. `build --phase holdout` (data to 2024-12)
2. LightGBM and MLP walk-forward for decision months 2014-11 to 2024-11
3. `evaluate --phase holdout`

Nothing is changed afterwards.

## Selected: L0_momentum_top10 (no ML)

Each month, equal-weight the 10 usable assets with the highest 2-12 month
return. The assets are the long legs of published anomalies and the 49
industries. The pre-registered rule (highest development CAGR net of costs)
picked it over the three ML combiners and the overlay.

## Development, 1975-01 to 2014-12, net of costs, vs the CRSP value-weighted market

| Candidate | CAGR | 2x costs | Sharpe | Vol | Max DD | Dev DSR (N=55) |
|---|---:|---:|---:|---:|---:|---:|
| L0 momentum top-10 (no ML) | 16.75% | 15.11% | 0.60 | 21.9% | -57.7% | 0.75 |
| L3 ensemble top-10 | 15.98% | 13.72% | 0.59 | 20.7% | -65.3% | 0.73 |
| L1 LightGBM top-10 | 15.06% | 12.71% | 0.55 | 20.8% | -65.1% | 0.56 |
| L2 MLP top-10 | 14.12% | 11.91% | 0.50 | 21.2% | -68.5% | 0.36 |
| O1 market + LS overlay (from 1986-02) | 10.86% | 7.21% | 0.49 | 17.1% | -49.2% | 0.15 |
| Market | 12.54% | - | 0.53 | 15.5% | -50.3% | - |

* None of the ML combiners beat the simple momentum ranking in development.
* **Criterion 4a already fails** (DSR 0.75 < 0.95), so the study cannot
  pass. The holdout is still run once, as registered.
* O1 starts in 1986-02 because it needs at least 10 published long-short
  portfolios.

## Note recorded before results were computed

Chen & Zimmermann's "LS" column is decile 10 minus decile 1 for every signal.
It was flipped by each signal's documented sign, so that positive means the
published direction, as the protocol specifies. This was found with a
mechanical identity check before any returns were evaluated.
