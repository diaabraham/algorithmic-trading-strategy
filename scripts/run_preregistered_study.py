"""Pre-registered study 2 (see results/candidates.md).

    python scripts/run_preregistered_study.py --phase is    # selection, data cut at 2019-12-31
    python scripts/run_preregistered_study.py --phase oos   # evaluates the saved shortlist once

The OOS phase reads the shortlist and the vol-matching factor from
results/study2_is_selection.json; it does not re-select anything.
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.candidates import COST_BPS, UNIVERSE, build_candidates, scale_weights, vr_signal_from_repo  # noqa: E402
from src.portfolio_backtest import deflated_sharpe, load_prices, simulate_weights, stats, vol_matched_returns  # noqa: E402

DATA_START = "2005-01-01"
IS_START, IS_END = "2007-01-03", "2019-12-31"
OOS_START = "2020-01-02"
SUBPERIODS = [("2007-01-03", "2009-12-31"), ("2010-01-01", "2012-12-31"), ("2013-01-01", "2015-12-31"), ("2016-01-01", "2019-12-31")]
OUT = "results"
SEL_FILE = os.path.join(OUT, "study2_is_selection.json")


def period_sim(prices, weights, start, end):
    """Flat at the close before ``start``; first fills at the first open in the period."""
    close, open_, rf = prices["Close"], prices["Open"], prices["rf_daily"]
    idx = close.index
    i0 = idx.searchsorted(pd.Timestamp(start)) - 1
    i1 = idx.searchsorted(pd.Timestamp(end), side="right")
    sl = idx[max(i0, 0):i1]
    w = weights.reindex(sl).fillna(0.0)
    sim = simulate_weights(open_.loc[sl], close.loc[sl], w, COST_BPS, rf.loc[sl])
    return sim.iloc[1:]


def spy_weights(index):
    return pd.DataFrame({"SPY": 1.0}, index=index)


def sharpe(ret, rf):
    ex = ret - rf.reindex(ret.index).fillna(0.0)
    return float(ex.mean() / ex.std() * np.sqrt(252))


def candidates_for(prices, end):
    vr = {"off": vr_signal_from_repo(end, ml=False), "on": vr_signal_from_repo(end, ml=True)}
    return build_candidates(prices["Close"], prices["rf_daily"], vr)


def truncate(prices, end):
    e = pd.Timestamp(end)
    return {k: v.loc[:e] for k, v in prices.items()}


def phase_is(end_date):
    prices = truncate(load_prices(UNIVERSE, DATA_START, end_date), IS_END)
    rf = prices["rf_daily"]
    cands = candidates_for(prices, IS_END)
    sims = {name: period_sim(prices, w, IS_START, IS_END) for name, w in cands.items()}
    spy = period_sim(prices, spy_weights(prices["Close"].index), IS_START, IS_END)
    table = {}
    for name, sim in sims.items():
        s = stats(sim, rf)
        s["subperiod_sharpe"] = {f"{a[:4]}-{b[:4]}": sharpe(sim["ret"].loc[a:b], rf) for a, b in SUBPERIODS}
        table[name] = s
    spy_stats = stats(spy, rf)
    spy_stats["subperiod_sharpe"] = {f"{a[:4]}-{b[:4]}": sharpe(spy["ret"].loc[a:b], rf) for a, b in SUBPERIODS}

    selected = max(table, key=lambda n: table[n]["sharpe"])
    k = spy_stats["ann_vol"] / table[selected]["ann_vol"]
    vm_sim = period_sim(prices, scale_weights(cands[selected], k, 2.0), IS_START, IS_END)
    vm_stats = stats(vm_sim, rf)

    # Walk-forward check of the selection rule inside IS.
    wf = []
    for year in range(2010, 2020):
        hist_end = f"{year - 1}-12-31"
        pick = max(sims, key=lambda n: sharpe(sims[n]["ret"].loc[IS_START:hist_end], rf))
        r = sims[pick]["ret"].loc[f"{year}-01-01":f"{year}-12-31"]
        rs = spy["ret"].loc[f"{year}-01-01":f"{year}-12-31"]
        wf.append({"year": year, "pick": pick, "pick_return": float((1 + r).prod() - 1), "spy_return": float((1 + rs).prod() - 1)})
    wf_ret = pd.concat([sims[w["pick"]]["ret"].loc[f"{w['year']}-01-01":f"{w['year']}-12-31"] for w in wf])
    wf_spy = spy["ret"].loc["2010-01-01":IS_END]
    wf_summary = {
        "cagr": float((1 + wf_ret).prod() ** (252 / len(wf_ret)) - 1), "sharpe": sharpe(wf_ret, rf),
        "spy_cagr": float((1 + wf_spy).prod() ** (252 / len(wf_spy)) - 1), "spy_sharpe": sharpe(wf_spy, rf),
        "years_pick_beat_spy": int(sum(w["pick_return"] > w["spy_return"] for w in wf)),
    }

    dsr = deflated_sharpe(sims[selected]["ret"], rf, [table[n]["sharpe"] for n in table])

    families = {}
    for name in sorted(table, key=lambda n: -table[n]["sharpe"]):
        families.setdefault(name[0], name)
    best_cagr = max(table, key=lambda n: table[n]["cagr"])
    shortlist = list(dict.fromkeys([selected, *families.values(), best_cagr]))

    result = {
        "protocol": "results/candidates.md",
        "data_cut": IS_END,
        "in_sample": table,
        "spy_buy_and_hold": spy_stats,
        "selected": selected,
        "selection_reason": f"highest IS Sharpe ({table[selected]['sharpe']:.3f})",
        "volmatch_k": k,
        "selected_volmatched_in_sample": vm_stats,
        "walk_forward_inside_is": {"years": wf, "summary": wf_summary},
        "deflated_sharpe_selected": dsr,
        "oos_shortlist": shortlist,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(SEL_FILE, "w") as fh:
        json.dump(result, fh, indent=2)

    print("In-sample 2007-01-03 .. 2019-12-31 (sorted by Sharpe)")
    for name in sorted(table, key=lambda n: -table[n]["sharpe"]):
        row(name, table[name])
    row("SPY buy-and-hold", spy_stats)
    print(f"\nSelected: {selected}; vol-match k = {k:.3f}")
    row("selected_volmatched", vm_stats)
    print("Walk-forward inside IS:", wf_summary)
    print("Deflated Sharpe:", {a: round(b, 4) if isinstance(b, float) else b for a, b in dsr.items()})
    print("OOS shortlist:", shortlist)


def phase_oos(end_date):
    with open(SEL_FILE) as fh:
        sel = json.load(fh)
    prices = load_prices(UNIVERSE, DATA_START, end_date)
    rf = prices["rf_daily"]
    end = str(prices["Close"].index[-1].date())
    cands = candidates_for(prices, end_date)

    # Look-ahead check: IS numbers from full data must equal the IS phase numbers.
    drift = 0.0
    for name in sel["in_sample"]:
        s = stats(period_sim(prices, cands[name], IS_START, IS_END), rf)
        drift = max(drift, abs(s["cagr"] - sel["in_sample"][name]["cagr"]))

    weights = {name: cands[name] for name in sel["oos_shortlist"]}
    weights["selected_volmatched"] = scale_weights(cands[sel["selected"]], sel["volmatch_k"], 2.0)
    spy = period_sim(prices, spy_weights(prices["Close"].index), OOS_START, end)
    spy_stats = stats(spy, rf)
    table, sims = {}, {}
    for name, w in weights.items():
        sim = period_sim(prices, w, OOS_START, end)
        sims[name] = sim
        s = stats(sim, rf)
        s["beats_spy_cagr"] = bool(s["cagr"] > spy_stats["cagr"])
        k_post = spy_stats["ann_vol"] / s["ann_vol"]
        vm = stats(vol_matched_returns(sim, rf, k_post), rf)
        s["ex_post_vol_matched"] = {"k": k_post, **{key: vm[key] for key in ("cagr", "sharpe", "ann_vol", "max_drawdown", "avg_gross_leverage")},
                                    "beats_spy_cagr": bool(vm["cagr"] > spy_stats["cagr"])}
        table[name] = s
    result = {"protocol": "results/candidates.md", "selected": sel["selected"], "oos_start": OOS_START, "oos_end": end,
              "is_reproduction_max_cagr_diff": drift, "out_of_sample": table, "spy_buy_and_hold": spy_stats,
              "average_t_bill": float(rf.loc[OOS_START:end].mean() * 252)}
    with open(os.path.join(OUT, "study2_oos.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, sim in sims.items():
        ax.plot(sim["nav"] / sim["nav"].iloc[0], label=name, linewidth=1.2)
    ax.plot(spy["nav"] / spy["nav"].iloc[0], label="SPY buy-and-hold", color="black", linewidth=1.8)
    ax.set_title(f"Pre-registered study, out-of-sample {OOS_START} to {end} (net of costs and financing)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "study2_oos_equity.png"), dpi=110)

    print(f"OOS {OOS_START} .. {end}; IS reproduction max CAGR diff = {drift:.2e}")
    for name, s in table.items():
        row(name, s)
        v = s["ex_post_vol_matched"]
        print(f"    ex-post vol-matched k={v['k']:.2f}: CAGR {v['cagr']:.2%} Sharpe {v['sharpe']:.2f} MaxDD {v['max_drawdown']:.2%}")
    row("SPY buy-and-hold", spy_stats)


def row(label, s):
    print(f"{label:<30} CAGR {s['cagr']:7.2%}  Sharpe {s['sharpe']:5.2f}  vol {s['ann_vol']:6.2%}  MaxDD {s['max_drawdown']:7.2%}  "
          f"lev {s['avg_gross_leverage']:.2f}/{s['max_gross_leverage']:.2f}  cost/yr {s['costs_per_year']:.3%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["is", "oos"], required=True)
    ap.add_argument("--end-date", default=date.today().isoformat())
    a = ap.parse_args()
    phase_is(a.end_date) if a.phase == "is" else phase_oos(a.end_date)
