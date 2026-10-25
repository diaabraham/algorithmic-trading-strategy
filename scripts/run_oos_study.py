"""Run the SPY in-sample / out-of-sample study and write results/.

    python scripts/run_oos_study.py [--end-date YYYY-MM-DD] [--cost-bps 2]
"""

import argparse
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data_loader import DataLoader  # noqa: E402
from src.ml_signal import attach_recursive_ml_up_proba  # noqa: E402
from src.oos_study import IS_START, OOS_START, run_study, t_bill_daily  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--cost-bps", type=float, default=2.0)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    loader = DataLoader("SPY", "2007-01-01", args.end_date, sources=("yfinance",))
    data = loader.get_data()
    # Walk-forward ML probabilities: each 21-bar block is scored by a model
    # trained only on earlier bars, starting at the in-sample start.
    data = attach_recursive_ml_up_proba(data, eval_start=IS_START)
    rf = t_bill_daily(data.index)

    result = run_study(data, rf, cost_bps=args.cost_bps)
    sims = result.pop("_sims")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "oos_study.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    oos = data.index >= OOS_START
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in result["out_of_sample"]:
        ret = sims[name]["ret"].loc[oos]
        ax.plot((1 + ret).cumprod(), label=name)
    spy = data["Close"].loc[oos]
    ax.plot(spy / spy.iloc[0], label="SPY buy-and-hold", color="black", linewidth=1)
    ax.set_title("Out-of-sample growth of $1 (net of costs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "oos_equity_curve.png"), dpi=110)

    def row(label, p):
        print(f"{label:<28} {p['start']}..{p['end']}  CAGR {p['cagr']:7.2%}  total {p['total_return']:8.2%}  "
              f"Sharpe {p['sharpe']:5.2f}  vol {p['ann_vol']:6.2%}  MaxDD {p['max_drawdown']:7.2%}")

    print("In-sample (selection period):")
    for name, p in sorted(result["in_sample"].items(), key=lambda kv: -kv[1]["sharpe"]):
        row(name, p)
    row("SPY buy-and-hold", result["in_sample_benchmark_spy"])
    print(f"\nSelected by in-sample Sharpe: {result['selected']}")
    print("\nOut-of-sample (evaluated once):")
    for name, p in result["out_of_sample"].items():
        row(name, p)
    row("SPY buy-and-hold", result["out_of_sample_benchmark_spy"])


if __name__ == "__main__":
    main()
