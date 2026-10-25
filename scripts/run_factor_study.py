"""Study 4 (see results/study4_protocol.md).

    python scripts/run_factor_study.py --phase dev       # US 1963-07..1999-12 only
    python scripts/run_factor_study.py --phase holdout   # run once after results/study4_frozen_config.md
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from src.factor_study import MIX, N_TRIALS, load, monthly_stats, net_returns, sharpe_stats, spy_monthly  # noqa: E402

OUT = "results"
DEV = ("1963-07-31", "1999-12-31")
H1_START = "2000-01-31"
H2 = ("1927-01-31", "1963-06-30")
H3_START = "1990-07-31"
SUBPERIODS = [("2000-01-31", "2008-12-31"), ("2009-01-31", "2017-12-31"), ("2018-01-31", None)]


def cagr(r):
    r = r.dropna()
    return float((1 + r).prod() ** (12 / len(r)) - 1)


def phase_dev():
    d = load(end=DEV[1])
    net = net_returns(d["us"]).loc[DEV[0]:DEV[1]]
    mkt = d["us_market"].loc[DEV[0]:DEV[1]]
    rf = d["us_rf"]
    table = {k: monthly_stats(net[k], rf) for k in net.columns}
    actives = {k: net[k] - mkt for k in net.columns}
    trial_sr = [sharpe_stats(actives[k])["active_sharpe_annual"] for k in net.columns]
    for k in net.columns:
        table[k]["active_vs_market"] = sharpe_stats(actives[k], trial_sr, N_TRIALS)
    selected = max(table, key=lambda k: table[k]["cagr"])
    res = {"protocol": "results/study4_protocol.md", "dev_period": DEV, "candidates": table,
           "market": monthly_stats(mkt, rf), "selected": selected,
           "criterion_4a_dsr_ge_0.95": table[selected]["active_vs_market"]["deflated_sharpe_ratio"] >= 0.95}
    with open(os.path.join(OUT, "study4_dev.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    show(table, res["market"], "US market")
    print("selected:", selected, "| DSR (N=50):", round(table[selected]["active_vs_market"]["deflated_sharpe_ratio"], 4))


def phase_holdout():
    with open(os.path.join(OUT, "study4_dev.json")) as fh:
        sel = json.load(fh)["selected"]
    d = load()
    rf = d["us_rf"]
    # Last month for which the market and T-bill series are published (the momentum file can run a month ahead).
    end = str(min(d["us_rf"].index[-1], d["us_market"].index[-1]).date())
    net, net2 = net_returns(d["us"]).loc[:end], net_returns(d["us"], 2.0).loc[:end]
    spy = spy_monthly(end).loc[H1_START:]
    common = net.loc[H1_START:].index.intersection(spy.index)
    spy = spy.loc[common]
    h1, h1_2x = net.loc[common], net2.loc[common]
    out = {"selected": sel, "H1_period": [str(common[0].date()), str(common[-1].date())]}
    out["H1"] = {k: {**monthly_stats(h1[k], rf), "cagr_2x_costs": cagr(h1_2x[k]),
                     "subperiod_cagr": [cagr(h1[k].loc[a:b]) for a, b in SUBPERIODS],
                     "active_vs_spy": sharpe_stats(h1[k] - spy)} for k in h1.columns}
    out["H1_spy"] = {**monthly_stats(spy, rf), "subperiod_cagr": [cagr(spy.loc[a:b]) for a, b in SUBPERIODS]}
    h2 = net.loc[H2[0]:H2[1]]
    mkt2 = d["us_market"].loc[H2[0]:H2[1]]
    out["H2"] = {k: monthly_stats(h2[k], rf) for k in h2.columns if h2[k].notna().sum() >= len(mkt2) * 0.95}
    out["H2_market"] = monthly_stats(mkt2, rf)
    ex = net_returns(d["ex"]).loc[H3_START:]
    mkt3 = d["ex_market"].loc[ex.index]
    out["H3"] = {k: monthly_stats(ex[k], d["ex_rf"]) for k in ex.columns}
    out["H3_market"] = monthly_stats(mkt3, d["ex_rf"])

    s, b = out["H1"][sel], out["H1_spy"]
    sub_wins = sum(x > y for x, y in zip(s["subperiod_cagr"], b["subperiod_cagr"]))
    with open(os.path.join(OUT, "study4_dev.json")) as fh:
        dev = json.load(fh)
    crit = {
        "1_h1_cagr_beats_spy": s["cagr"] > b["cagr"],
        "2_h1_2x_costs_beats_spy": s["cagr_2x_costs"] > b["cagr"],
        "3_subperiods_beaten": f"{sub_wins}/3",
        "3_pass": sub_wins >= 2,
        "4a_dev_dsr_ge_0.95": dev["criterion_4a_dsr_ge_0.95"],
        "4b_h1_psr_ge_0.95": s["active_vs_spy"]["psr_vs_zero"] >= 0.95,
        "5_h2_beats_market": (out["H2"][sel]["cagr"] > out["H2_market"]["cagr"]) if sel in out["H2"] else "n/a",
        "5_h3_beats_market": (out["H3"][sel]["cagr"] > out["H3_market"]["cagr"]) if sel in out["H3"] else "n/a",
    }
    crit["PASS"] = all(v for k, v in crit.items() if k[0].isdigit() and k != "3_subperiods_beaten" and v != "n/a")
    out["criterion"] = crit
    with open(os.path.join(OUT, "study4_holdout.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"H1 {out['H1_period']} vs SPY")
    show(out["H1"], b, "SPY")
    for k, v in out["H1"].items():
        print(f"  {k}: 2x costs CAGR {v['cagr_2x_costs']:.2%}; subperiods {[f'{x:.1%}' for x in v['subperiod_cagr']]} "
              f"(SPY {[f'{x:.1%}' for x in b['subperiod_cagr']]}); active PSR {v['active_vs_spy']['psr_vs_zero']:.3f}")
    print("H2 1927-1963"); show(out["H2"], out["H2_market"], "US market")
    print("H3 developed ex-US"); show(out["H3"], out["H3_market"], "ex-US market")
    print("criterion:", crit)


def show(table, bench, label):
    for k, v in list(table.items()) + [(label, bench)]:
        print(f"  {k:<24} {v['start']}..{v['end']} CAGR {v['cagr']:6.2%} Sharpe {v['sharpe']:5.2f} vol {v['ann_vol']:6.2%} MaxDD {v['max_drawdown']:7.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["dev", "holdout"], required=True)
    a = ap.parse_args()
    phase_dev() if a.phase == "dev" else phase_holdout()
