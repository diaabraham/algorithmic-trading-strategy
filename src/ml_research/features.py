"""Feature matrix for one traded asset. Every value at date t uses data up to t's close."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.ml_research.data import SECTORS, asof


def build_features(asset: pd.DataFrame, yahoo: Dict[str, pd.DataFrame], fred: Dict[str, pd.Series]) -> pd.DataFrame:
    idx = asset.index
    c = asset["Close"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=idx)
    for h in (5, 21, 63, 126, 252):
        f[f"ret_{h}"] = np.log(c / c.shift(h))
    f["mom_12_1"] = np.log(c.shift(21) / c.shift(252))
    for n in (50, 200):
        f[f"sma_gap_{n}"] = c / c.rolling(n).mean() - 1
    f["vol_21"] = r.rolling(21).std() * np.sqrt(252)
    f["vol_63"] = r.rolling(63).std() * np.sqrt(252)
    f["vol_ratio"] = f["vol_21"] / (r.rolling(252).std() * np.sqrt(252))
    f["downside_vol_63"] = r.clip(upper=0).rolling(63).std() * np.sqrt(252)
    f["drawdown_252"] = c / c.rolling(252).max() - 1

    vix = yahoo["^VIX"]["Close"].reindex(idx).ffill()
    f["vix"] = vix
    f["vix_z"] = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
    f["vix_chg_5"] = np.log(vix / vix.shift(5))
    f["vrp"] = vix / 100 - f["vol_21"]  # implied minus realised
    irx = yahoo["^IRX"]["Close"].reindex(idx).ffill() / 100
    f["tbill"] = irx
    f["tbill_chg_63"] = irx - irx.shift(63)
    f["term_10y3m"] = asof(fred["T10Y3M"], idx)
    baa = asof(fred["BAA10Y"], idx)
    f["credit_baa10y"] = baa
    f["credit_chg_63"] = baa - baa.shift(63)
    claims = asof(fred["ICSA"].rolling(4).mean(), idx)
    f["claims_yoy"] = np.log(claims / claims.shift(252))
    sectors = pd.concat([yahoo[s]["Close"].reindex(idx).ffill() for s in SECTORS], axis=1)
    above = (sectors > sectors.rolling(200).mean()).where(sectors.rolling(200).count() == 200)
    f["breadth_200"] = above.mean(axis=1, skipna=False)
    return f.replace([np.inf, -np.inf], np.nan)


def forward_label(fill_price: pd.Series, rf_daily: pd.Series, horizon: int) -> pd.Series:
    """Log return from the fill after date t (t+1) to the fill h bars later, minus T-bills."""
    p = fill_price
    lab = np.log(p.shift(-(horizon + 1)) / p.shift(-1))
    rf = np.log1p(rf_daily.reindex(p.index).fillna(0.0)).rolling(horizon).sum().shift(-(horizon + 1))
    return lab - rf
