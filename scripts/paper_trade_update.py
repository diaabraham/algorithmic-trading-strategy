"""Live paper tracking of the two frozen rules against SPY.

paper-trade-v1: study 3 rule none_voltarget on SPY (results/frozen_config.md).
    SPY weight = long-run realised vol / 63-day realised vol, clipped to [0, 1.5],
    decided at each Friday close, filled at the next open, changed only if the
    target moves by more than 0.10. Costs 2 bp per side, cash at T-bills,
    borrowing at T-bill + 50 bp. Tracked daily from the 2026-10-26 open.

paper-trade-v2: study 5 rule L0 (results/study5_frozen_config.md,
    results/paper_trade_v2_frozen.md). Top-10 by 2-12 month momentum across the
    frozen 228 Chen-Zimmermann anomaly legs and French 49 industries, monthly,
    with study 5 costs. Tracked monthly. The official live window starts with the
    holding month 2026-11 (decision at the 2026-10-31 month end). See
    src/study5/live.py for why its returns are only measurable once the
    published portfolio data covers them.

Run monthly (any day works; reruns are idempotent):

    pip install -U openassetpricing && python scripts/paper_trade_update.py

Outputs:
    results/paper_trade_summary.json        both rules vs SPY, judging status
    results/paper_trade_live_v1.csv         v1 daily NAV vs SPY
    results/paper_trade_live_v2_picks.csv   v2 decisions (append-only log)
    results/paper_trade_live_v2.csv         v2 monthly net returns vs SPY
"""

import json
import os
import sys
import warnings
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.ml_research.sizing import decided_weights  # noqa: E402
from src.portfolio_backtest import simulate_weights, stats  # noqa: E402

OUT = "results"
SPY_COST_BPS = 2.0
CHECKPOINTS_MONTHS = (12, 24)
PASS_RULE = "CAGR net of costs above SPY total return (2 bp entry) over the same live window"

# v1 (tag paper-trade-v1)
V1_FREEZE_DECISION_DATE = "2026-10-23"
V1_LIVE_START = "2026-10-26"
V1_COST_BPS = 2.0

# v2 (tag paper-trade-v2)
V2_FREEZE_DATE = "2026-10-25"
V2_FIRST_DECISION = "2024-12-31"        # latest month end with data for the whole universe at the freeze
V2_FIRST_LIVE_HOLDING = "2026-11-30"    # first holding month decided after the freeze
V2_UNIVERSE = os.path.join(OUT, "paper_trade_v2_universe.csv")
V2_PICKS = os.path.join(OUT, "paper_trade_live_v2_picks.csv")
V2_RETURNS = os.path.join(OUT, "paper_trade_live_v2.csv")
STUDY5_SEED = os.path.join("data", "cache", "study5", "cz_deciles_vw.pkl")


# ---------------------------------------------------------------- v1 (daily)

def _yahoo(t):
    import yfinance as yf

    d = yf.download(t, period="max", auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.index = pd.to_datetime(d.index).tz_localize(None)
    return d


def v1_weights(spy: pd.DataFrame) -> pd.Series:
    """The frozen rule over full history (the 0.10 band makes the state path-dependent)."""
    close = spy["Close"]
    # none_voltarget ignores the prediction value; it only needs one to exist.
    # In the frozen pipeline SPY has had predictions since 1998, so it is always on in the live window.
    first_pred = pd.Timestamp("1998-05-01")
    pred = pd.DataFrame({"pred": np.where(close.index >= first_pred, 0.0, np.nan), "pred_sd": 1.0}, index=close.index)
    return decided_weights("voltarget", pred, close)


def _cagr(nav_end: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    return float(nav_end ** (365.25 / max((end - start).days, 1)) - 1)


def run_v1(spy: pd.DataFrame, rf: pd.Series) -> dict:
    w = v1_weights(spy)
    idx = spy.index
    live = idx[idx >= pd.Timestamp(V1_LIVE_START)]
    s = {"tag": "paper-trade-v1", "rule": "none_voltarget (SPY volatility targeting)", "frequency": "daily",
         "freeze_decision_date": V1_FREEZE_DECISION_DATE, "live_start": V1_LIVE_START,
         "weight_at_freeze": float(w.loc[:V1_FREEZE_DECISION_DATE].iloc[-1]),
         "measurable": "daily, from the first close after the 2026-10-26 open fill"}
    checks = {f"{m}_months": {"window_end": str((pd.Timestamp(V1_LIVE_START) + pd.DateOffset(months=m)).date()),
                              "status": "pending"} for m in CHECKPOINTS_MONTHS}
    if len(live) == 0:
        s["status"] = "no live bars yet"
        s["checkpoints"] = checks
        return s
    i0 = idx.searchsorted(pd.Timestamp(V1_LIVE_START)) - 1  # flat at the freeze close
    sl = idx[i0:]
    px_o = spy[["Open"]].loc[sl].rename(columns={"Open": "SPY"})
    px_c = spy[["Close"]].loc[sl].rename(columns={"Close": "SPY"})
    strat = simulate_weights(px_o, px_c, w.loc[sl].to_frame("SPY"), {"SPY": V1_COST_BPS}, rf.loc[sl]).iloc[1:]
    bench = simulate_weights(px_o, px_c, pd.DataFrame({"SPY": 1.0}, index=sl), {"SPY": SPY_COST_BPS}, rf.loc[sl]).iloc[1:]
    table = pd.DataFrame({"weight": w.loc[strat.index], "strategy_nav": strat["nav"], "spy_nav": bench["nav"]})
    table.to_csv(os.path.join(OUT, "paper_trade_live_v1.csv"), float_format="%.6f")
    t0 = sl[0]
    for m in CHECKPOINTS_MONTHS:
        end = pd.Timestamp(checks[f"{m}_months"]["window_end"])
        if strat.index[-1] >= end:
            last = strat.index[strat.index <= end][-1]
            c_s, c_b = _cagr(strat["nav"].loc[last], t0, last), _cagr(bench["nav"].loc[last], t0, last)
            checks[f"{m}_months"].update({"status": "PASS" if c_s > c_b else "FAIL", "last_bar": str(last.date()),
                                          "strategy_cagr": c_s, "spy_cagr": c_b})
    s.update({"status": "live", "last_bar": str(strat.index[-1].date()), "trading_days": len(strat),
              "strategy_return": float(strat["nav"].iloc[-1] - 1), "spy_return": float(bench["nav"].iloc[-1] - 1),
              "checkpoints": checks})
    if len(strat) >= 63:
        s["strategy"] = stats(strat, rf)
        s["spy"] = stats(bench, rf)
    return s


# ---------------------------------------------------------------- v2 (monthly)

def spy_month_returns(spy: pd.DataFrame) -> pd.Series:
    """SPY total-return months that are complete (dividend-adjusted closes)."""
    m = spy["Close"].resample("ME").last().pct_change().dropna()
    return m[m.index < pd.Timestamp(date.today()) + pd.offsets.MonthEnd(0)]


def update_picks(R, pub, names, info, picks_path=V2_PICKS) -> pd.DataFrame:
    """Append decisions that became computable; never rewrite a logged one."""
    from src.study5.live import momentum_2_12, picks_at

    cols = ["decision_month_end", "holding_month_end", "phase", "picks", "names", "cz_release", "cz_data_end",
            "industries_data_end", "logged_on"]
    log = pd.read_csv(picks_path, dtype=str, keep_default_na=False) if os.path.exists(picks_path) else pd.DataFrame(columns=cols)
    done = set(log["decision_month_end"])
    data_end = min(pd.Timestamp(info["cz_data_end"]), pd.Timestamp(info["industries_data_end"]))
    mom = momentum_2_12(R)
    rows = []
    for t in R.index[(R.index >= pd.Timestamp(V2_FIRST_DECISION)) & (R.index <= data_end)]:
        key = str(t.date())
        if key in done:
            continue
        p = picks_at(R, pub, t, mom)
        if p is None:
            continue
        h = t + pd.offsets.MonthEnd(1)
        rows.append({"decision_month_end": key, "holding_month_end": str(h.date()),
                     "phase": "live" if h >= pd.Timestamp(V2_FIRST_LIVE_HOLDING) else "pre_live_not_judged",
                     "picks": ";".join(p), "names": "; ".join(names.get(a, a) for a in p),
                     "cz_release": info["cz_release"], "cz_data_end": info["cz_data_end"],
                     "industries_data_end": info["industries_data_end"], "logged_on": info["fetched_on"]})
    if rows:
        log = pd.concat([log, pd.DataFrame(rows)], ignore_index=True)
    log = log.sort_values("decision_month_end").reset_index(drop=True)
    log[cols].to_csv(picks_path, index=False)
    return log


def v2_returns(log: pd.DataFrame, R: pd.DataFrame, turnover: dict, cost: dict, spy_m: pd.Series) -> pd.DataFrame:
    """Monthly net returns of the logged picks; each phase starts from cash (full entry cost)."""
    from src.study5.live import month_cost

    out, prev, prev_phase = [], [], None
    for _, row in log.iterrows():
        h = pd.Timestamp(row["holding_month_end"])
        if row["phase"] != prev_phase:
            prev, prev_phase, first = [], row["phase"], True
        else:
            first = False
        picks = row["picks"].split(";")
        if h not in R.index or R.loc[h, picks].notna().sum() == 0 or h not in spy_m.index:
            break  # returns for this month are not published yet
        gross = float(R.loc[h, picks].mean())  # equal weight over legs with a published return
        c = month_cost(picks, prev, turnover, cost)
        out.append({"holding_month_end": str(h.date()), "phase": row["phase"], "gross": gross, "cost": c,
                    "net": gross - c, "net_2x_costs": gross - 2 * c,
                    "missing_legs": int(R.loc[h, picks].isna().sum()),
                    "spy": float(spy_m.loc[h]) - (SPY_COST_BPS / 1e4 if first else 0.0)})
        prev = picks
    cols = ["holding_month_end", "phase", "gross", "cost", "net", "net_2x_costs", "missing_legs", "spy"]
    return pd.DataFrame(out, columns=cols)


def _window(df: pd.DataFrame, n: int) -> dict:
    sub = df.head(n)
    yrs = len(sub) / 12
    c = lambda col: float((1 + sub[col]).prod() ** (1 / yrs) - 1)  # noqa: E731
    return {"months": len(sub), "start": sub["holding_month_end"].iloc[0][:7], "end": sub["holding_month_end"].iloc[-1][:7],
            "strategy_cagr": c("net"), "strategy_cagr_2x_costs": c("net_2x_costs"), "spy_cagr": c("spy")}


def run_v2(spy: pd.DataFrame, seed: str | None = STUDY5_SEED) -> dict:
    from src.study5.live import load_universe

    uni = pd.read_csv(V2_UNIVERSE).set_index("asset")  # frozen: names, publication years, turnover, costs
    R, info = load_universe(list(uni.index), seed=seed)
    pub = uni["pub_year"].fillna(-1).astype(int)
    log = update_picks(R, pub, uni["name"].to_dict(), info)
    rets = v2_returns(log, R, uni["turnover_per_year"].to_dict(), uni["cost_per_side"].to_dict(), spy_month_returns(spy))
    rets.to_csv(V2_RETURNS, index=False, float_format="%.6f")
    first_live = pd.Timestamp(V2_FIRST_LIVE_HOLDING)
    s = {"tag": "paper-trade-v2", "rule": "L0_momentum_top10 (study 5)", "frequency": "monthly",
         "freeze_date": V2_FREEZE_DATE, "first_decision": V2_FIRST_DECISION,
         "first_live_holding_month": V2_FIRST_LIVE_HOLDING[:7], "data": info,
         "measurable": ("when a Chen-Zimmermann release covers the month; the anomaly legs are not priced in real "
                        "time, so decisions and returns are filled in from published portfolio returns"),
         "latest_logged_decision": log.iloc[-1][["decision_month_end", "holding_month_end", "phase", "names"]].to_dict()
         if len(log) else None}
    pre = rets[rets["phase"] == "pre_live_not_judged"] if len(rets) else rets
    live = rets[rets["phase"] == "live"] if len(rets) else rets
    if len(pre):
        s["pre_live_not_judged"] = _window(pre, len(pre))
    checks = {}
    for m in CHECKPOINTS_MONTHS:
        end = first_live + pd.offsets.MonthEnd(m - 1)
        c = {"window": f"{first_live:%Y-%m}..{end:%Y-%m}", "status": "pending",
             "needs_data_through": f"{end:%Y-%m}"}
        if len(live) >= m:
            w = _window(live, m)
            c.update(w)
            c["status"] = "PASS" if w["strategy_cagr"] > w["spy_cagr"] else "FAIL"
        checks[f"{m}_months"] = c
    s["checkpoints"] = checks
    s["status"] = f"live, {len(live)} months measured" if len(live) else "no live returns yet"
    if len(live):
        s["live_to_date"] = _window(live, len(live))
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    spy, irx = _yahoo("SPY"), _yahoo("^IRX")
    rf = irx["Close"].reindex(spy.index).ffill().bfill() / 100 / 252
    summary = {"updated_on": str(date.today()), "pass_rule": PASS_RULE, "judged_at_months": list(CHECKPOINTS_MONTHS),
               "v1": run_v1(spy, rf), "v2": run_v2(spy)}
    with open(os.path.join(OUT, "paper_trade_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    v1, v2 = summary["v1"], summary["v2"]
    print("v1:", v1["status"], "| weight at freeze", round(v1["weight_at_freeze"], 3))
    print("v2:", v2["status"], "| data", v2["data"])
    if v2["latest_logged_decision"]:
        d = v2["latest_logged_decision"]
        print(f"v2 latest picks (decided {d['decision_month_end']}, held {d['holding_month_end'][:7]}): {d['names']}")


if __name__ == "__main__":
    main()
