"""Multi-asset backtesting utilities: data, a next-open simulator, statistics.

Target weights (fractions of NAV) are decided on a bar's close and filled at
the next bar's open. Holdings are share counts between fills. Costs are per
side in basis points of traded notional, per asset. Cash earns the T-bill
rate; borrowed cash pays T-bill + ``borrow_spread``; shorts pay ``short_fee``.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")


def load_prices(tickers: Iterable[str], start: str, end: str, cache: bool = True) -> Dict[str, pd.DataFrame]:
    """Adjusted daily Open/Close per ticker from Yahoo Finance (naive date index)."""
    import yfinance as yf

    tickers = sorted(set(tickers))
    key = os.path.join(CACHE_DIR, f"prices_{'_'.join(tickers)}_{start}_{end}.pkl")
    if cache and os.path.exists(key):
        with open(key, "rb") as fh:
            return pickle.load(fh)
    raw = yf.download(tickers, start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
                      auto_adjust=True, progress=False, group_by="column")
    out = {}
    for field in ("Open", "Close"):
        frame = raw[field].copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        out[field] = frame[tickers]
    irx = yf.download("^IRX", start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
                      auto_adjust=False, progress=False)["Close"]
    if isinstance(irx, pd.DataFrame):
        irx = irx.iloc[:, 0]
    irx.index = pd.to_datetime(irx.index).tz_localize(None)
    out["rf_daily"] = irx.reindex(out["Close"].index).ffill().bfill() / 100.0 / 252.0
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(key, "wb") as fh:
            pickle.dump(out, fh)
    return out


def simulate_weights(
    open_: pd.DataFrame,
    close: pd.DataFrame,
    target: pd.DataFrame,
    cost_bps: Dict[str, float],
    rf_daily: pd.Series,
    borrow_spread: float = 0.005,
    short_fee: float = 0.01,
    band: float = 1e-4,
) -> pd.DataFrame:
    """Simulate decided target weights. Returns nav, ret, gross, turnover, cost."""
    idx = close.index
    cols = list(target.columns)
    tgt = target.reindex(idx).ffill().fillna(0.0)[cols].to_numpy(float)
    o = open_[cols].reindex(idx).to_numpy(float)
    c = close[cols].reindex(idx).ffill().to_numpy(float)
    rf = rf_daily.reindex(idx).ffill().fillna(0.0).to_numpy(float)
    bps = np.array([cost_bps.get(t, 5.0) for t in cols]) * 1e-4

    shares = np.zeros(len(cols))
    cash = 1.0
    last = np.zeros(len(cols))
    n = len(idx)
    nav = np.empty(n)
    gross = np.zeros(n)
    turn = np.zeros(n)
    cost = np.zeros(n)
    for t in range(n):
        if t > 0:
            cash *= 1.0 + (rf[t] if cash >= 0 else rf[t] + borrow_spread / 252.0)
            prev_c = np.nan_to_num(c[t - 1])
            cash -= np.sum(np.abs(np.minimum(shares, 0)) * prev_c) * short_fee / 252.0
            want = tgt[t - 1]
            if np.max(np.abs(want - last)) > band:
                px = np.where(np.isfinite(o[t]) & (o[t] > 0), o[t], np.nan_to_num(c[t - 1]))
                nav_open = cash + np.sum(shares * px)
                tradable = px > 0
                new_shares = np.where(tradable, want * nav_open / np.where(tradable, px, 1.0), 0.0)
                traded = np.abs(new_shares - shares) * px
                fee = float(np.sum(traded * bps))
                cash -= float(np.sum((new_shares - shares) * px)) + fee
                turn[t] = float(np.sum(traded)) / nav_open if nav_open else 0.0
                cost[t] = fee / nav_open if nav_open else 0.0
                shares = new_shares
                last = want.copy()
        value = shares * np.nan_to_num(c[t])
        nav[t] = cash + value.sum()
        gross[t] = np.abs(value).sum() / nav[t] if nav[t] else 0.0
    out = pd.DataFrame({"nav": nav, "gross": gross, "turnover": turn, "cost": cost}, index=idx)
    out["ret"] = out["nav"].pct_change().fillna(out["nav"].iloc[0] - 1.0)
    return out


def stats(sim: pd.DataFrame, rf_daily: pd.Series) -> dict:
    ret = sim["ret"]
    rf = rf_daily.reindex(ret.index).fillna(0.0)
    years = len(ret) / 252.0
    growth = float((1 + ret).prod())
    ex = ret - rf
    eq = (1 + ret).cumprod()
    return {
        "start": str(ret.index[0].date()),
        "end": str(ret.index[-1].date()),
        "cagr": growth ** (1 / years) - 1,
        "total_return": growth - 1,
        "ann_vol": float(ret.std() * np.sqrt(252)),
        "sharpe": float(ex.mean() / ex.std() * np.sqrt(252)),
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "avg_gross_leverage": float(sim["gross"].mean()),
        "max_gross_leverage": float(sim["gross"].max()),
        "turnover_per_year": float(sim["turnover"].sum() / years),
        "costs_per_year": float(sim["cost"].sum() / years),
    }


def vol_matched_returns(sim: pd.DataFrame, rf_daily: pd.Series, k: float, borrow_spread: float = 0.005) -> pd.DataFrame:
    """Scale excess returns by k; extra borrowing is financed at rf + spread."""
    rf = rf_daily.reindex(sim.index).fillna(0.0)
    borrowed = (k * sim["gross"].shift(1).fillna(0.0) - 1.0).clip(lower=0.0)
    ret = rf + k * (sim["ret"] - rf) - borrowed * borrow_spread / 252.0
    out = pd.DataFrame({"ret": ret, "gross": k * sim["gross"], "turnover": k * sim["turnover"], "cost": k * sim["cost"]})
    out["nav"] = (1 + out["ret"]).cumprod()
    return out


def deflated_sharpe(returns: pd.Series, rf_daily: pd.Series, trial_sharpes_annual: Iterable[float]) -> dict:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014), per-period units."""
    ex = (returns - rf_daily.reindex(returns.index).fillna(0.0)).dropna()
    t = len(ex)
    sr = ex.mean() / ex.std()
    trials = np.asarray(list(trial_sharpes_annual), float) / np.sqrt(252)
    n = len(trials)
    var = float(np.var(trials, ddof=1))
    gamma = 0.5772156649
    sr0 = np.sqrt(var) * ((1 - gamma) * norm.ppf(1 - 1 / n) + gamma * norm.ppf(1 - 1 / (n * np.e)))
    g3 = float(skew(ex))
    g4 = float(kurtosis(ex, fisher=False))
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    dsr = float(norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom))
    psr0 = float(norm.cdf(sr * np.sqrt(t - 1) / denom))
    return {
        "trials": n,
        "sharpe_annual": float(sr * np.sqrt(252)),
        "expected_max_sharpe_under_null_annual": float(sr0 * np.sqrt(252)),
        "deflated_sharpe_ratio": dsr,
        "probabilistic_sharpe_vs_zero": psr0,
        "skew": g3,
        "kurtosis": g4,
        "observations": t,
    }
