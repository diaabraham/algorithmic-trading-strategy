"""Live paper test of the frozen study 3 rule (tag paper-trade-v1).

Rule (none_voltarget, results/frozen_config.md): SPY weight = long-run
realised vol / 63-day realised vol, clipped to [0, 1.5], decided at each
Friday close, filled at the next open, changed only if the target moves by
more than 0.10. Costs 2 bp per side, cash at T-bills, borrowing at
T-bill + 50 bp.

The live test starts with the first fill after the freeze (decision at the
2026-10-23 close, fill at the 2026-10-26 open). Run this script any time; it
re-downloads prices and rewrites results/paper_trade_live.csv and
results/paper_trade_summary.json from the freeze date onward.

    python scripts/paper_trade_update.py
"""

import json
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.ml_research.sizing import decided_weights  # noqa: E402
from src.portfolio_backtest import simulate_weights, stats  # noqa: E402

FREEZE_DECISION_DATE = "2026-10-23"
LIVE_START = "2026-10-26"
COST_BPS = 2.0
OUT = "results"


def download():
    import yfinance as yf

    def one(t):
        d = yf.download(t, period="max", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.index = pd.to_datetime(d.index).tz_localize(None)
        return d

    spy, irx = one("SPY"), one("^IRX")
    rf = irx["Close"].reindex(spy.index).ffill().bfill() / 100 / 252
    return spy, rf


def live_weights(spy: pd.DataFrame) -> pd.Series:
    """The frozen rule over full history (the 0.10 band makes the state path-dependent)."""
    close = spy["Close"]
    # none_voltarget ignores the prediction value; it only needs one to exist.
    # In the frozen pipeline SPY has had predictions since 1998, so it is always on in the live window.
    first_pred = pd.Timestamp("1998-05-01")
    pred = pd.DataFrame({"pred": np.where(close.index >= first_pred, 0.0, np.nan), "pred_sd": 1.0}, index=close.index)
    return decided_weights("voltarget", pred, close)


def main():
    spy, rf = download()
    w = live_weights(spy)
    idx = spy.index
    live = idx[idx >= pd.Timestamp(LIVE_START)]
    os.makedirs(OUT, exist_ok=True)
    summary = {"tag": "paper-trade-v1", "rule": "none_voltarget", "freeze_decision_date": FREEZE_DECISION_DATE,
               "live_start": LIVE_START, "weight_at_freeze": float(w.loc[:FREEZE_DECISION_DATE].iloc[-1]),
               "judged_at": ["12 months", "24 months"],
               "pass_rule": "CAGR net of costs above SPY buy-and-hold over the same live window"}
    if len(live) == 0:
        summary["status"] = "no live bars yet"
        print(f"No bars after the freeze yet; target weight decided on {FREEZE_DECISION_DATE}: {summary['weight_at_freeze']:.3f}")
    else:
        i0 = idx.searchsorted(pd.Timestamp(LIVE_START)) - 1  # flat at the freeze close
        sl = idx[i0:]
        px_o, px_c = spy[["Open"]].loc[sl].rename(columns={"Open": "SPY"}), spy[["Close"]].loc[sl].rename(columns={"Close": "SPY"})
        strat = simulate_weights(px_o, px_c, w.loc[sl].to_frame("SPY"), {"SPY": COST_BPS}, rf.loc[sl]).iloc[1:]
        bench = simulate_weights(px_o, px_c, pd.DataFrame({"SPY": 1.0}, index=sl), {"SPY": COST_BPS}, rf.loc[sl]).iloc[1:]
        table = pd.DataFrame({"weight": w.loc[strat.index], "strategy_nav": strat["nav"], "spy_nav": bench["nav"]})
        table.to_csv(os.path.join(OUT, "paper_trade_live.csv"), float_format="%.6f")
        summary["status"] = "live"
        summary["last_bar"] = str(strat.index[-1].date())
        summary["trading_days"] = len(strat)
        summary["strategy_return"] = float(strat["nav"].iloc[-1] - 1)
        summary["spy_return"] = float(bench["nav"].iloc[-1] - 1)
        if len(strat) >= 63:
            summary["strategy"] = stats(strat, rf)
            summary["spy"] = stats(bench, rf)
        print(f"Live {LIVE_START}..{summary['last_bar']} ({len(strat)} days): strategy {summary['strategy_return']:+.2%} vs SPY {summary['spy_return']:+.2%}")
    with open(os.path.join(OUT, "paper_trade_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
