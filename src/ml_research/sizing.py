"""Turn predictions into target weights (fraction of NAV), decided weekly."""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_LEVERAGE = 1.5
GAMMA = 3.0      # risk aversion for mean-variance sizing
TILT = 0.5       # weight multiplier per unit of standardised prediction
BAND = 0.10      # trade only when the target moves by more than this
HORIZON = 21


def week_end_mask(index: pd.DatetimeIndex) -> np.ndarray:
    wk = index.to_period("W-FRI")
    return np.r_[wk[1:] != wk[:-1], True]


def raw_weights(rule: str, pred: pd.DataFrame, close: pd.Series) -> pd.Series:
    r = np.log(close).diff()
    vol = (r.rolling(63).std() * np.sqrt(252)).clip(lower=0.02)
    long_run = r.expanding(252).std() * np.sqrt(252)  # all history to date
    vt = (long_run / vol).clip(upper=MAX_LEVERAGE)
    yhat = pred["pred"]
    z = (yhat / pred["pred_sd"]).clip(-1, 1)
    if rule == "voltarget":
        w = vt.where(yhat.notna())
    elif rule == "mv":
        w = (yhat * 252 / HORIZON) / (GAMMA * vol ** 2)
    elif rule == "tilt":
        w = vt * (1 + TILT * z)
    elif rule == "gate":
        w = vt.where(yhat > 0, 0.0).where(yhat.notna())
    else:
        raise ValueError(rule)
    return w.clip(0.0, MAX_LEVERAGE)


def decided_weights(rule: str, pred: pd.DataFrame, close: pd.Series) -> pd.Series:
    """Weekly decisions with a no-trade band; NaN (no model yet) means flat."""
    w = raw_weights(rule, pred, close).to_numpy(float)
    week_end = week_end_mask(close.index)
    out = np.zeros(len(w))
    cur = 0.0
    for t in range(len(w)):
        if week_end[t] and np.isfinite(w[t]) and abs(w[t] - cur) > BAND:
            cur = float(w[t])
        elif week_end[t] and not np.isfinite(w[t]):
            cur = 0.0
        out[t] = cur
    return pd.Series(out, index=close.index)
