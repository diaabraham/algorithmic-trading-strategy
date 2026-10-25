"""Single-asset execution simulator: decisions at the close, fills at the next open.

``simulate_exposure`` takes the exposure (fraction of NAV, e.g. 1 = fully long,
0 = flat, -1 = fully short, 1.5 = 50% leverage) *decided* at the close of each
bar and trades to it at the following bar's open. Positions are held as share
counts between decisions, so prices drift the exposure naturally; trades occur
only when the decided exposure changes.

Costs and carry:
* ``cost_bps`` per side on traded notional (commission + half-spread + impact)
* idle cash earns ``rf_daily`` (e.g. T-bill / 252); borrowed cash pays
  ``rf_daily + borrow_spread / 252``; short positions pay ``short_fee / 252``
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def simulate_exposure(
    open_: pd.Series,
    close: pd.Series,
    decided: pd.Series,
    cost_bps: float = 2.0,
    rf_daily: Optional[pd.Series] = None,
    borrow_spread: float = 0.01,
    short_fee: float = 0.01,
) -> pd.DataFrame:
    """Return a frame with columns nav, ret, exposure, turnover (one row per bar)."""
    idx = close.index
    decided = decided.reindex(idx).ffill().fillna(0.0).astype(float)
    rf = (rf_daily.reindex(idx).ffill().fillna(0.0) if rf_daily is not None else pd.Series(0.0, index=idx)).values
    o, c, d = open_.values.astype(float), close.values.astype(float), decided.values

    shares, cash = 0.0, 1.0
    last_target = 0.0
    nav = np.empty(len(idx))
    expo = np.zeros(len(idx))
    turn = np.zeros(len(idx))
    for t in range(len(idx)):
        if t > 0:
            rate = rf[t] if cash >= 0 else rf[t] + borrow_spread / 252.0
            cash *= 1.0 + rate
            if shares < 0:
                cash -= abs(shares) * c[t - 1] * short_fee / 252.0
            target = d[t - 1]
            if target != last_target and np.isfinite(o[t]) and o[t] > 0:
                nav_open = cash + shares * o[t]
                new_shares = target * nav_open / o[t]
                traded = abs(new_shares - shares) * o[t]
                cash -= (new_shares - shares) * o[t] + traded * cost_bps * 1e-4
                turn[t] = traded / nav_open if nav_open else 0.0
                shares = new_shares
                last_target = target
        nav[t] = cash + shares * c[t]
        expo[t] = shares * c[t] / nav[t] if nav[t] else 0.0
    out = pd.DataFrame({"nav": nav, "exposure": expo, "turnover": turn}, index=idx)
    out["ret"] = out["nav"].pct_change().fillna(out["nav"].iloc[0] - 1.0)
    return out


def performance(ret: pd.Series, rf_daily: Optional[pd.Series] = None) -> dict:
    """CAGR, annualised vol, Sharpe (excess of rf), max drawdown, total return."""
    ret = ret.dropna()
    years = len(ret) / 252.0
    growth = float((1 + ret).prod())
    ex = ret - (rf_daily.reindex(ret.index).fillna(0.0) if rf_daily is not None else 0.0)
    eq = (1 + ret).cumprod()
    return {
        "start": str(pd.Timestamp(ret.index[0]).date()),
        "end": str(pd.Timestamp(ret.index[-1]).date()),
        "total_return": growth - 1,
        "cagr": growth ** (1 / years) - 1 if years > 0 and growth > 0 else float("nan"),
        "ann_vol": float(ret.std() * np.sqrt(252)),
        "sharpe": float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else float("nan"),
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
    }
