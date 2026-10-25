"""Study 3: walk-forward ML (see results/study3_protocol.md and results/frozen_config.md).

    python scripts/run_ml_study.py --phase dev       # SPY, data cut at 2023-12-31; logs every config
    python scripts/run_ml_study.py --phase holdout   # frozen config only; run once
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
import pandas as pd  # noqa: E402

from src.ml_research.pipeline import BASE_MODELS, buy_and_hold, load_market, predictions, rolling_win_rate, simulate  # noqa: E402
from src.ml_research.sizing import decided_weights  # noqa: E402
from src.portfolio_backtest import deflated_sharpe, stats  # noqa: E402

OUT = "results"
DEV_END, DEV_START = "2023-12-31", "2000-01-03"
N_TRIALS_TOTAL = 45
MODELS = BASE_MODELS + ["ens"]
RULES = ["mv", "tilt", "gate"]
HOLDOUTS = {
    "H0_SPY_2024plus": {"ticker": "SPY", "fill": "open", "cost": 2.0, "start": "2024-01-01", "end": None},
    "X1_EFA": {"ticker": "EFA", "fill": "open", "cost": 2.0, "start": None, "end": None},
    "X2_EWJ": {"ticker": "EWJ", "fill": "open", "cost": 5.0, "start": None, "end": None},
    "X3_EWG": {"ticker": "EWG", "fill": "open", "cost": 5.0, "start": None, "end": None},
    "X4_EWU": {"ticker": "EWU", "fill": "open", "cost": 5.0, "start": None, "end": None},
    "X5_EWC": {"ticker": "EWC", "fill": "open", "cost": 5.0, "start": None, "end": None},
    "X6_SP500TR_pre2000": {"ticker": "^SP500TR", "fill": "close", "cost": 2.0, "start": "1996-01-02", "end": "1999-12-31"},
}


def summarize(sim, bench, rf):
    s = stats(sim, rf)
    b = stats(bench, rf)
    s["bh_cagr"], s["bh_sharpe"], s["bh_max_drawdown"], s["bh_ann_vol"] = b["cagr"], b["sharpe"], b["max_drawdown"], b["ann_vol"]
    s["beats_bh_cagr"] = bool(s["cagr"] > b["cagr"])
    s["rolling3y_win_rate"] = rolling_win_rate(sim["ret"], bench["ret"])
    return s


def phase_dev():
    t0 = time.time()
    mkt = load_market("SPY", DEV_END)
    preds = predictions(mkt, BASE_MODELS, "open")
    close = mkt["asset"]["Close"]
    bench = buy_and_hold(mkt, "open", 2.0, DEV_START, DEV_END)
    configs = [("none", "voltarget")] + [(m, r) for m in MODELS for r in RULES]
    rows, sims = [], {}
    for model, rule in configs:
        p = preds["ridge" if model == "none" else model]
        w = decided_weights(rule, p, close)
        sim = simulate(mkt, w, "open", 2.0, DEV_START, DEV_END)
        s = summarize(sim, bench, mkt["rf"])
        s["sub_2000_2009_cagr"] = float((1 + sim["ret"].loc[:"2009"]).prod() ** (252 / len(sim["ret"].loc[:"2009"])) - 1)
        s["sub_2010_2019_cagr"] = float((1 + sim["ret"].loc["2010":"2019"]).prod() ** (252 / len(sim["ret"].loc["2010":"2019"])) - 1)
        s["sub_2020_2023_cagr_contaminated"] = float((1 + sim["ret"].loc["2020":]).prod() ** (252 / len(sim["ret"].loc["2020":])) - 1)
        name = f"{model}_{rule}"
        sims[name] = sim
        rows.append({"config": name, "model": model, "rule": rule, **{k: v for k, v in s.items() if not isinstance(v, (dict, list))}})
    trial_active_sharpes = []
    for r in rows:
        act = sims[r["config"]]["ret"] - bench["ret"]
        r["active_sharpe"] = float(act.mean() / act.std() * np.sqrt(252)) if act.std() > 0 else 0.0
        trial_active_sharpes.append(r["active_sharpe"])
    selected = max(rows, key=lambda r: r["cagr"])["config"]
    act = sims[selected]["ret"] - bench["ret"]
    zero = pd.Series(0.0, index=act.index)
    dsr = deflated_sharpe(act, zero, trial_active_sharpes)
    dsr45 = dict(dsr)
    # Re-evaluate the null with N = 45 total trials (same cross-trial variance).
    from scipy.stats import norm
    sr = act.mean() / act.std()
    var = np.var(np.asarray(trial_active_sharpes) / np.sqrt(252), ddof=1)
    g = 0.5772156649
    sr0 = np.sqrt(var) * ((1 - g) * norm.ppf(1 - 1 / N_TRIALS_TOTAL) + g * norm.ppf(1 - 1 / (N_TRIALS_TOTAL * np.e)))
    den = np.sqrt(1 - dsr["skew"] * sr + (dsr["kurtosis"] - 1) / 4 * sr ** 2)
    dsr45.update({"trials": N_TRIALS_TOTAL, "expected_max_sharpe_under_null_annual": float(sr0 * np.sqrt(252)),
                  "deflated_sharpe_ratio": float(norm.cdf((sr - sr0) * np.sqrt(len(act) - 1) / den))})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "trials_log.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    sel_row = next(r for r in rows if r["config"] == selected)
    result = {"protocol": "results/study3_protocol.md", "dev_period": [DEV_START, DEV_END], "configs_tried": len(rows),
              "selected": selected, "selected_dev": sel_row, "spy_bh_dev": stats(bench, mkt["rf"]),
              "deflated_sharpe_active_N13": dsr, "deflated_sharpe_active_N45": dsr45,
              "criterion_dev": {"1_dev_cagr_beats_spy": sel_row["beats_bh_cagr"],
                                "4_dsr_ge_0.95": dsr45["deflated_sharpe_ratio"] >= 0.95,
                                "5_rolling3y_win_ge_0.60": sel_row["rolling3y_win_rate"] >= 0.60},
              "hyperparameters_chosen": {m: preds[m].attrs.get("params") for m in BASE_MODELS},
              "runtime_seconds": time.time() - t0}
    with open(os.path.join(OUT, "study3_dev.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    b = result["spy_bh_dev"]
    print(f"SPY buy-and-hold dev: CAGR {b['cagr']:.2%} Sharpe {b['sharpe']:.2f} MaxDD {b['max_drawdown']:.2%} vol {b['ann_vol']:.2%}")
    for r in sorted(rows, key=lambda r: -r["cagr"]):
        print(f"{r['config']:<16} CAGR {r['cagr']:7.2%} Sharpe {r['sharpe']:5.2f} MaxDD {r['max_drawdown']:7.2%} vol {r['ann_vol']:6.2%} "
              f"lev {r['avg_gross_leverage']:.2f} cost {r['costs_per_year']:.3%} roll3y {r['rolling3y_win_rate']:.0%} "
              f"| 00-09 {r['sub_2000_2009_cagr']:6.2%} 10-19 {r['sub_2010_2019_cagr']:6.2%} 20-23* {r['sub_2020_2023_cagr_contaminated']:6.2%}")
    print("selected:", selected, "| DSR N=13:", round(dsr["deflated_sharpe_ratio"], 4), "N=45:", round(dsr45["deflated_sharpe_ratio"], 4))
    print("criterion (dev part):", result["criterion_dev"], f"runtime {result['runtime_seconds']:.0f}s")


def phase_holdout(end_date):
    with open(os.path.join(OUT, "study3_dev.json")) as fh:
        dev = json.load(fh)
    model, rule = dev["selected"].split("_", 1)
    models = BASE_MODELS if model in ("ens", "none") else [model]
    results = {}
    for name, h in HOLDOUTS.items():
        end = h["end"] or end_date
        mkt = load_market(h["ticker"], end)
        preds = predictions(mkt, models, h["fill"])
        p = preds["ridge" if model == "none" else model]
        w = decided_weights(rule, p, mkt["asset"]["Close"])
        start = h["start"]
        if start is None:
            first = p["pred"].first_valid_index()
            start = str((first + pd.offsets.MonthBegin(1)).date())
        sim = simulate(mkt, w, h["fill"], h["cost"], start, h["end"])
        bench = buy_and_hold(mkt, h["fill"], h["cost"], start, h["end"])
        results[name] = summarize(sim, bench, mkt["rf"])
        r = results[name]
        print(f"{name:<20} {r['start']}..{r['end']} CAGR {r['cagr']:7.2%} vs B&H {r['bh_cagr']:7.2%} | Sharpe {r['sharpe']:.2f} vs {r['bh_sharpe']:.2f} "
              f"| MaxDD {r['max_drawdown']:.1%} vs {r['bh_max_drawdown']:.1%} | vol {r['ann_vol']:.1%} lev {r['avg_gross_leverage']:.2f}/{r['max_gross_leverage']:.2f} "
              f"cost {r['costs_per_year']:.3%} roll3y {r['rolling3y_win_rate']:.0%} beats={r['beats_bh_cagr']}")
    cross = [results[k]["beats_bh_cagr"] for k in results if k.startswith("X")]
    crit = dict(dev["criterion_dev"])
    crit["2_h0_beats_spy"] = results["H0_SPY_2024plus"]["beats_bh_cagr"]
    crit["3_cross_market_majority"] = sum(cross) >= 4
    crit["cross_market_wins"] = f"{sum(cross)}/{len(cross)}"
    crit["PASS"] = all(v for k, v in crit.items() if k[0].isdigit())
    out = {"frozen_config": "results/frozen_config.md", "selected": dev["selected"], "holdouts": results, "criterion": crit}
    with open(os.path.join(OUT, "study3_holdout.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("criterion:", crit)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["dev", "holdout"], required=True)
    ap.add_argument("--end-date", default=date.today().isoformat())
    a = ap.parse_args()
    phase_dev() if a.phase == "dev" else phase_holdout(a.end_date)
