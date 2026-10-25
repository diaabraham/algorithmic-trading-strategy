"""Portfolio construction, costs and statistics for study 5 (results/study5_protocol.md)."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src.factor_study import monthly_stats, sharpe_stats, spy_monthly
from src.study5.data import CACHE, load

TOP_K = 10
COST_SIGNAL, COST_INDUSTRY = 0.0020, 0.0010
TE_BUDGET, MAX_K = 0.05, 1.0
SHORT_FEE, FIN_SPREAD = 0.005, 0.005
N_TRIALS = 55
CANDIDATES = ["L0_momentum_top10", "L1_lgbm_top10", "L2_mlp_top10", "L3_ensemble_top10", "O1_market_plus_ls_overlay"]
SUBPERIODS = [("2015-01-31", "2018-04-30"), ("2018-05-31", "2021-08-31"), ("2021-09-30", "2024-12-31")]
OUT = "results"


def turnover_of(meta: pd.DataFrame, asset: str) -> float:
    if asset.startswith("IND_"):
        return 0.10
    per, cat = meta.at[asset, "Portfolio Period"], meta.at[asset, "Cat.Data"]
    if per == 1:
        return 3.0 if cat in ("Price", "Trading") else 1.5
    if per in (3, 6):
        return 1.0
    return 0.5


def cost_of(asset: str) -> float:
    return COST_INDUSTRY if asset.startswith("IND_") else COST_SIGNAL


def _xs_rank(panel, col):
    return panel.groupby("month")[col].rank(pct=True)


def scores(panel: pd.DataFrame, preds: dict) -> pd.DataFrame:
    s = pd.DataFrame(index=panel.index)
    s["L0_momentum_top10"] = panel["r2_12"]
    if "lgbm" in preds:
        panel = panel.assign(_l=preds["lgbm"])
        s["L1_lgbm_top10"] = panel["_l"]
    if "mlp" in preds:
        panel = panel.assign(_m=preds["mlp"])
        s["L2_mlp_top10"] = panel["_m"]
    if "lgbm" in preds and "mlp" in preds:
        ok = panel["_l"].notna() & panel["_m"].notna()
        rl = panel["_l"].where(ok).groupby(panel["month"]).rank(pct=True)
        rm = panel["_m"].where(ok).groupby(panel["month"]).rank(pct=True)
        s["L3_ensemble_top10"] = (rl + rm) / 2
    return s


def long_only(panel, score, meta, months, cost_mult=1.0):
    """Monthly net returns (indexed by holding month end) of the top-K equal-weight portfolio."""
    out, prev, held = {}, {}, []
    for t in months:
        rows = panel[(panel["month"] == t) & score.notna() & panel["next_ret"].notna()]
        if len(rows) < TOP_K:
            continue
        top = rows.loc[score.loc[rows.index].nlargest(TOP_K).index]
        w = {a: 1.0 / TOP_K for a in top["asset"]}
        gross = float(top["next_ret"].mean())
        within = sum(w[a] * 2 * turnover_of(meta, a) * cost_of(a) / 12 for a in w)
        switch = sum(abs(w.get(a, 0) - prev.get(a, 0)) * cost_of(a) for a in set(w) | set(prev))
        date = top["date"].iloc[0] + pd.offsets.MonthEnd(1)
        out[date] = gross - cost_mult * (within + switch)
        held.append({"date": date, "assets": list(w), "switch_turnover": sum(abs(w.get(a, 0) - prev.get(a, 0)) for a in set(w) | set(prev)) / 2,
                     "n_industries": sum(a.startswith("IND_") for a in w)})
        prev = w
    return pd.Series(out).sort_index(), held


def overlay(panel_ls, score, meta, months, ls_wide, base, cost_mult=1.0):
    out, prev, ks = {}, {}, {}
    for t in months:
        rows = panel_ls[(panel_ls["month"] == t) & score.notna() & panel_ls["next_ret"].notna()]
        if len(rows) < TOP_K:
            continue
        top = rows.loc[score.loc[rows.index].nlargest(TOP_K).index]
        assets = list(top["asset"])
        date0 = top["date"].iloc[0]
        hist = ls_wide.loc[:date0, assets].tail(36).mean(axis=1)
        vol = float(hist.std() * np.sqrt(12)) if hist.notna().sum() >= 12 else np.nan
        k = min(TE_BUDGET / vol, MAX_K) if vol and np.isfinite(vol) and vol > 0 else 0.0
        w = {a: k / TOP_K for a in assets}
        basket = float(top["next_ret"].mean())
        within = sum(w[a] * 2 * 2 * turnover_of(meta, a) * cost_of(a) / 12 for a in w)  # two legs
        switch = 2 * sum(abs(w.get(a, 0) - prev.get(a, 0)) * cost_of(a) for a in set(w) | set(prev))
        carry = k * (SHORT_FEE + FIN_SPREAD) / 12
        date = date0 + pd.offsets.MonthEnd(1)
        if date not in base.index:
            continue
        out[date] = float(base.loc[date]) + k * basket - cost_mult * (within + switch + carry)
        ks[date] = k
        prev = w
    return pd.Series(out).sort_index(), pd.Series(ks).sort_index()


def build_all(phase, d, months, base, cost_mult=1.0):
    pl = pd.read_pickle(os.path.join(CACHE, f"panel_{phase}_long.pkl"))
    ps = pd.read_pickle(os.path.join(CACHE, f"panel_{phase}_ls.pkl"))
    pred = lambda m, k: np.load(os.path.join(CACHE, f"pred_{phase}_{m}_{k}.npy"))
    sl = scores(pl, {"lgbm": pred("lgbm", "long"), "mlp": pred("mlp", "long")})
    ss = scores(ps, {"lgbm": pred("lgbm", "ls"), "mlp": pred("mlp", "ls")})
    rets, info = {}, {}
    for c in CANDIDATES[:4]:
        r, held = long_only(pl, sl[c], d["meta"], months, cost_mult)
        rets[c] = r
        info[c] = {"avg_switch_turnover_per_month": float(np.mean([h["switch_turnover"] for h in held])),
                   "avg_industries_held": float(np.mean([h["n_industries"] for h in held])),
                   "avg_nlong_of_signal_legs": float(np.nanmean([d["nlong"].at[h["date"] - pd.offsets.MonthEnd(1), a]
                                                                 for h in held for a in h["assets"] if not a.startswith("IND_")] or [np.nan]))}
    r, ks = overlay(ps, ss["L3_ensemble_top10"], d["meta"], months, d["ls"], base, cost_mult)
    rets["O1_market_plus_ls_overlay"] = r
    info["O1_market_plus_ls_overlay"] = {"avg_k": float(ks.mean()), "max_k": float(ks.max())}
    return rets, info


def _months(first, last):
    return range(first, last + 1)


def evaluate_dev(first, last):
    d = load("2014-12-31")
    base = d["market"]
    months = _months(first, last)
    rets, info = build_all("dev", d, months, base)
    rets2, _ = build_all("dev", d, months, base, 2.0)
    rf = d["rf"]
    table = {}
    actives = {c: (rets[c] - base.reindex(rets[c].index)).dropna() for c in CANDIDATES}
    trial_sr = [sharpe_stats(a)["active_sharpe_annual"] for a in actives.values()]
    for c in CANDIDATES:
        table[c] = {**monthly_stats(rets[c], rf), "cagr_2x_costs": float(monthly_stats(rets2[c], rf)["cagr"]),
                    "active_vs_market": sharpe_stats(actives[c], trial_sr, N_TRIALS), **info[c]}
    mkt = monthly_stats(base.reindex(rets[CANDIDATES[0]].index), rf)
    sel = max(table, key=lambda c: table[c]["cagr"])
    res = {"protocol": "results/study5_protocol.md", "period": [table[sel]["start"], table[sel]["end"]], "candidates": table,
           "market": mkt, "selected": sel,
           "criterion_4a_dsr_ge_0.95": table[sel]["active_vs_market"]["deflated_sharpe_ratio"] >= 0.95}
    with open(os.path.join(OUT, "study5_dev.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    return res


def evaluate_holdout(first, last):
    with open(os.path.join(OUT, "study5_dev.json")) as fh:
        dev = json.load(fh)
    sel = dev["selected"]
    d = load("2024-12-31")
    spy = spy_monthly("2024-12-31").loc["2015-01-31":]
    months = _months(first, last)
    rets, info = build_all("holdout", d, months, spy)
    rets2, _ = build_all("holdout", d, months, spy, 2.0)
    rf = d["rf"]
    table = {}
    for c in CANDIDATES:
        r = rets[c].reindex(spy.index).dropna()
        sub = [(monthly_stats(r.loc[a:b], rf)["cagr"], monthly_stats(spy.loc[a:b], rf)["cagr"]) for a, b in SUBPERIODS]
        table[c] = {**monthly_stats(r, rf), "cagr_2x_costs": monthly_stats(rets2[c].reindex(spy.index).dropna(), rf)["cagr"],
                    "subperiods_cagr_vs_spy": sub, "active_vs_spy": sharpe_stats(r - spy.reindex(r.index)), **info[c]}
    spy_stats = monthly_stats(spy, rf)
    s = table[sel]
    wins = sum(a > b for a, b in s["subperiods_cagr_vs_spy"])
    crit = {"1_holdout_cagr_beats_spy": s["cagr"] > spy_stats["cagr"],
            "2_2x_costs_beats_spy": s["cagr_2x_costs"] > spy_stats["cagr"],
            "3_subperiods_won": f"{wins}/3", "3_pass": wins >= 2,
            "4a_dev_dsr_ge_0.95": dev["criterion_4a_dsr_ge_0.95"],
            "4b_holdout_psr_ge_0.95": s["active_vs_spy"]["psr_vs_zero"] >= 0.95}
    crit["PASS"] = all(v for k, v in crit.items() if k[0].isdigit() and k != "3_subperiods_won")
    res = {"frozen_config": "results/study5_frozen_config.md", "selected": sel, "candidates": table, "spy": spy_stats, "criterion": crit}
    with open(os.path.join(OUT, "study5_holdout.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    return res


def evaluate(phase):
    first, last = PHASE_MONTHS[phase]
    res = evaluate_dev(first, last) if phase == "dev" else evaluate_holdout(first, last)
    bench = res.get("market") or res.get("spy")
    for c, v in list(res["candidates"].items()) + [("benchmark", bench)]:
        extra = f" 2x {v['cagr_2x_costs']:.2%}" if "cagr_2x_costs" in v else ""
        print(f"{c:<28} {v['start']}..{v['end']} CAGR {v['cagr']:6.2%}{extra} Sharpe {v['sharpe']:5.2f} vol {v['ann_vol']:6.2%} MaxDD {v['max_drawdown']:7.2%}")
    print("selected:", res["selected"])
    if "criterion" in res:
        print("criterion:", res["criterion"])
    return res


PHASE_MONTHS = {"dev": ((1974 - 1900) * 12 + 11, (2014 - 1900) * 12 + 10),
                "holdout": ((2014 - 1900) * 12 + 11, (2024 - 1900) * 12 + 10)}
