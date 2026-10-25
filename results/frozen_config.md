# Study 3 frozen configuration

This file was frozen before the holdout run. Code and development results are
at commit `935ce2a9a20b649e83a8d7d2b4ccee1011221a9b`. The holdout is run once with
`python scripts/run_ml_study.py --phase holdout` and nothing is changed
afterwards.

## Selected config: `none_voltarget`

The pre-registered rule (highest development CAGR among the 13 configs in
`results/trials_log.csv`) selected the **reference rule that uses no ML
prediction**:

* weight = long-run realised vol (expanding, all history to date) / 63-day
  realised vol
* clipped to [0, 1.5]
* decided at the Friday close, filled at the next open
* changed only when the target moves by more than 0.10
* active from the first month in which the walk-forward model has a
  prediction, i.e. after 5 years of usable history

Every ML config (ridge, LightGBM, MLP, ensemble x mv/tilt/gate) had a lower
development CAGR than this reference.

## Development (SPY, 2000-01-03 to 2023-12-31, net of costs)

| | CAGR | Sharpe | Max DD | Vol | Avg leverage |
|---|---:|---:|---:|---:|---:|
| none_voltarget | 8.44% | 0.45 | -44.5% | 18.2% | 1.14 |
| SPY buy-and-hold | 6.92% | 0.35 | -55.2% | 19.6% | 1.00 |

* Rolling 3-year win rate vs SPY: 73%.
* Deflated Sharpe ratio of the active return: 0.29 with N = 45 trials (0.40
  with N = 13).
* So **criterion 4 (DSR >= 0.95) already fails**. The study cannot pass,
  whatever the holdout shows. The holdout is still run once, as registered.

## Fixed parameters

* MAX_LEVERAGE 1.5, BAND 0.10, HORIZON 21, MIN_TRAIN_DAYS 1260
* Costs: 2 bp per side (SPY, EFA, ^SP500TR), 5 bp (EWJ, EWG, EWU, EWC)
* Financing: T-bill + 50 bp. Cash earns T-bills.
* Holdouts: H0 SPY 2024-01-01 onward; X1-X6 as in `results/study3_protocol.md`
