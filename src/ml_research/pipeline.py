"""End-to-end pipeline for one market: features -> walk-forward models -> weights -> simulation."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.ml_research.data import SECTORS, fred, yahoo
from src.ml_research.features import build_features, forward_label
from src.ml_research.sizing import HORIZON, decided_weights
from src.ml_research.walkforward import walk_forward
from src.portfolio_backtest import simulate_weights

MIN_TRAIN_DAYS = 1260
BASE_MODELS = ["ridge", "lgbm", "mlp"]


def load_market(ticker: str, end: str) -> dict:
    tickers = sorted(set([ticker, "^VIX", "^IRX", *SECTORS]))
    y = yahoo(tickers, end)
    asset = y[ticker].dropna(subset=["Close"])
    asset = asset[asset["Close"] > 0]
    rf = y["^IRX"]["Close"].reindex(asset.index).ffill().bfill() / 100 / 252
    return {"asset": asset, "yahoo": y, "fred": fred(end), "rf": rf}


def predictions(mkt: dict, models: List[str], fill: str) -> Dict[str, pd.DataFrame]:
    asset = mkt["asset"]
    feats = build_features(asset, mkt["yahoo"], mkt["fred"])
    fill_px = asset["Open"] if fill == "open" else asset["Close"]
    label = forward_label(fill_px, mkt["rf"], HORIZON)
    preds = {m: walk_forward(feats, label, m, HORIZON, MIN_TRAIN_DAYS) for m in models}
    if "ens" in models or set(BASE_MODELS) <= set(preds):
        p = pd.concat([preds[m]["pred"] for m in BASE_MODELS], axis=1)
        sd = pd.concat([preds[m]["pred_sd"] for m in BASE_MODELS], axis=1)
        preds["ens"] = pd.DataFrame({"pred": p.mean(axis=1, skipna=False), "pred_sd": sd.mean(axis=1, skipna=False)})
    return preds


def simulate(mkt: dict, weights: pd.Series, fill: str, cost_bps: float, start: str, end: str = None) -> pd.DataFrame:
    asset, rf = mkt["asset"], mkt["rf"]
    idx = asset.index
    i0 = max(idx.searchsorted(pd.Timestamp(start)) - 1, 0)
    i1 = idx.searchsorted(pd.Timestamp(end), side="right") if end else len(idx)
    sl = idx[i0:i1]
    open_ = asset[["Open"]] if fill == "open" else asset[["Close"]].rename(columns={"Close": "Open"})
    w = weights.reindex(sl).fillna(0.0).to_frame("A")
    sim = simulate_weights(open_.loc[sl].rename(columns={"Open": "A"}), asset[["Close"]].loc[sl].rename(columns={"Close": "A"}),
                           w, {"A": cost_bps}, rf.loc[sl])
    return sim.iloc[1:]


def buy_and_hold(mkt, fill, cost_bps, start, end=None):
    w = pd.Series(1.0, index=mkt["asset"].index)
    return simulate(mkt, w, fill, cost_bps, start, end)


def rolling_win_rate(strat: pd.Series, bench: pd.Series, window: int = 756) -> float:
    a = np.log1p(strat).rolling(window).sum()
    b = np.log1p(bench).rolling(window).sum()
    ok = a.notna() & b.notna()
    return float((a[ok] > b[ok]).mean()) if ok.any() else float("nan")
