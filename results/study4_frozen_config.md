# Study 4 frozen configuration

This file was frozen before the holdout run. The code and development results
are at commit `371788522831d5ce18c55898f458ab5173b3a38a`. The protocol is `results/study4_protocol.md`
(commit `f761f04`). The holdout is run once with
`python scripts/run_factor_study.py --phase holdout`, and nothing is changed
afterwards.

## Selected: F1, large-cap momentum

Kenneth French "BIG HiPRIOR": value-weighted stocks above the NYSE median
size with the top-30% 12-2 month return, rebalanced monthly. It carries a
0.55%/yr cost drag (1.10% in the 2x stress test).

## Development, US 1963-07 to 1999-12, net of costs

| | CAGR | Sharpe | Vol | Max DD |
|---|---:|---:|---:|---:|
| F1 large-cap momentum | 16.13% | 0.60 | 17.1% | -40.4% |
| F2 large-cap value | 15.13% | 0.60 | 15.2% | -31.9% |
| F5 mix (F1+F2+F3) | 14.90% | 0.59 | 15.2% | -39.0% |
| F3 large-cap profitability | 13.16% | 0.48 | 15.6% | -47.9% |
| F4 low-variance quintile | 11.84% | 0.48 | 12.3% | -42.7% |
| US market | 12.63% | 0.46 | 15.2% | -46.5% |

**Criterion 4a already fails.** F1's active return vs the market has a
Sharpe of 0.56 and a probabilistic Sharpe of 0.9996. But its deflated Sharpe
with N = 50 is only 0.07. The null maximum Sharpe is 0.80, driven by the wide
spread of active Sharpes across the five candidates. The study therefore
cannot pass. The holdout is still run once, as registered.

## Implementation pass-through

* H1 runs through the last month in which the French market and T-bill
  series are published. The momentum file can be one month ahead, so that
  month is dropped.
* This line was added before the holdout run.
