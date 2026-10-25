"""Pre-registered candidates for study 2 (see results/candidates.md).

Every function returns a DataFrame of target weights (fraction of NAV)
decided at each close using only data up to that close.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

MULTI = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC"]
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
UNIVERSE = sorted(set(MULTI + SECTORS))
COST_BPS = {**{t: 2.0 for t in ["SPY", "QQQ", "IWM", "EFA", "TLT", "IEF", "GLD"]},
            **{t: 3.0 for t in SECTORS}, "EEM": 5.0, "DBC": 5.0}
TARGET_VOL = 0.15
SMA_DAYS = 210  # 10 months


def month_end_mask(index: pd.DatetimeIndex) -> np.ndarray:
    p = index.to_period("M")
    return np.r_[p[1:] != p[:-1], True]


def _monthly(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.where(pd.Series(month_end_mask(frame.index), index=frame.index), axis=0).ffill().fillna(0.0)


def _weights(index, columns) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=index, columns=columns)


def _vol(close: pd.DataFrame, days: int) -> pd.DataFrame:
    return close.pct_change().rolling(days).std() * np.sqrt(252)


def _tbill_12m(rf_daily: pd.Series) -> pd.Series:
    return np.exp(np.log1p(rf_daily).rolling(252).sum()) - 1


def vol_target(raw: pd.DataFrame, close: pd.DataFrame, target: float, max_gross: float, days: int = 63) -> pd.DataFrame:
    """Scale each row of raw weights to a trailing portfolio vol of ``target``."""
    rets = close[raw.columns].pct_change().fillna(0.0).to_numpy()
    w = raw.to_numpy(float)
    out = np.zeros_like(w)
    for t in range(days, len(w)):
        g = np.abs(w[t]).sum()
        if g == 0:
            continue
        pv = rets[t - days + 1 : t + 1] @ w[t]
        vol = pv.std() * np.sqrt(252)
        scale = target / vol if vol > 0 else 0.0
        scale = min(scale, max_gross / g)
        out[t] = w[t] * scale
    return pd.DataFrame(out, index=raw.index, columns=raw.columns)


# ---- Family A: vol-managed SPY -------------------------------------------------
def vol_managed_spy(close: pd.DataFrame, days: int, cap: float = 1.5) -> pd.DataFrame:
    w = _weights(close.index, ["SPY"])
    w["SPY"] = (TARGET_VOL / _vol(close[["SPY"]], days)["SPY"]).clip(upper=cap).fillna(0.0)
    return _monthly(w)


# ---- Family B: trend equity / dual momentum ------------------------------------
def spy_sma_filter(close: pd.DataFrame) -> pd.DataFrame:
    w = _weights(close.index, ["SPY", "IEF"])
    above = close["SPY"] > close["SPY"].rolling(SMA_DAYS).mean()
    w["SPY"] = above.astype(float)
    w["IEF"] = (~above).astype(float)
    w.iloc[: SMA_DAYS] = 0.0
    return _monthly(w)


def dual_momentum(close: pd.DataFrame, rf_daily: pd.Series) -> pd.DataFrame:
    w = _weights(close.index, ["SPY", "EFA", "IEF"])
    r12 = close[["SPY", "EFA"]] / close[["SPY", "EFA"]].shift(252) - 1
    tb = _tbill_12m(rf_daily.reindex(close.index).ffill())
    spy_wins = r12["SPY"] >= r12["EFA"]
    best = r12.max(axis=1)
    risk_on = best > tb
    w["SPY"] = (risk_on & spy_wins).astype(float)
    w["EFA"] = (risk_on & ~spy_wins).astype(float)
    w["IEF"] = (~risk_on).astype(float)
    w[r12.isna().any(axis=1) | tb.isna()] = 0.0
    return _monthly(w)


# ---- Family C: multi-asset --------------------------------------------------------
def _inv_vol(close: pd.DataFrame) -> pd.DataFrame:
    iv = 1.0 / _vol(close[MULTI], 63)
    iv = iv.where(close[MULTI].notna())
    return iv


def multi_asset(close: pd.DataFrame, rf_daily: pd.Series, mode: str) -> pd.DataFrame:
    sub = close[MULTI]
    iv = _inv_vol(close)
    if mode == "tsmom":
        r12 = sub / sub.shift(252) - 1
        tb = _tbill_12m(rf_daily.reindex(close.index).ffill())
        iv = iv.where(r12.sub(tb, axis=0) > 0)
    elif mode == "rp_trend":
        iv = iv.where(sub > sub.rolling(SMA_DAYS).mean())
    elif mode != "rp":
        raise ValueError(mode)
    iv = iv.where(sub.shift(252).notna())  # require 12 months of history
    raw = iv.div(iv.sum(axis=1), axis=0).fillna(0.0)
    raw = raw.where(pd.Series(month_end_mask(raw.index), index=raw.index), axis=0).ffill().fillna(0.0)
    scaled = vol_target(raw, close, TARGET_VOL, 2.0)
    return _monthly(scaled)


# ---- Family D: rotation ------------------------------------------------------------
def qqq_spy_rotation(close: pd.DataFrame) -> pd.DataFrame:
    w = _weights(close.index, ["QQQ", "SPY", "IEF"])
    r6 = close[["QQQ", "SPY"]] / close[["QQQ", "SPY"]].shift(126) - 1
    pick_qqq = r6["QQQ"] > r6["SPY"]
    px = close["QQQ"].where(pick_qqq, close["SPY"])
    sma = close["QQQ"].rolling(SMA_DAYS).mean().where(pick_qqq, close["SPY"].rolling(SMA_DAYS).mean())
    on = px > sma
    w["QQQ"] = (on & pick_qqq).astype(float)
    w["SPY"] = (on & ~pick_qqq).astype(float)
    w["IEF"] = (~on).astype(float)
    w.iloc[: SMA_DAYS] = 0.0
    return _monthly(w)


def sector_momentum(close: pd.DataFrame, spy_filter: bool) -> pd.DataFrame:
    w = _weights(close.index, SECTORS + ["IEF"])
    mom = close[SECTORS].shift(21) / close[SECTORS].shift(252) - 1
    rank = mom.rank(axis=1, ascending=False)
    top = (rank <= 3).astype(float) / 3.0
    top[mom.isna().any(axis=1)] = 0.0
    w[SECTORS] = top
    if spy_filter:
        on = close["SPY"] > close["SPY"].rolling(SMA_DAYS).mean()
        w.loc[~on, SECTORS] = 0.0
        w.loc[~on & mom.notna().all(axis=1), "IEF"] = 1.0
    return _monthly(w)


# ---- Family E: the repository's volatility-regime strategy, levered ---------------
def vr_levered(close: pd.DataFrame, vr_signal: pd.Series) -> pd.DataFrame:
    w = _weights(close.index, ["SPY"])
    lev = (TARGET_VOL / _vol(close[["SPY"]], 63)["SPY"]).clip(upper=1.5)
    lev = lev.where(pd.Series(month_end_mask(close.index), index=close.index)).ffill().fillna(0.0)
    w["SPY"] = vr_signal.reindex(close.index).fillna(0.0) * lev
    return w


def vr_signal_from_repo(end: str, ml: bool, eval_start: str = "2007-01-03") -> pd.Series:
    """Decided exposure (+1/0/-1) of VolatilityRegimeStrategy with default parameters."""
    from dataclasses import replace

    from src.data_loader import DataLoader
    from src.ml_signal import attach_recursive_ml_up_proba
    from src.strategy import StrategyParams, VolatilityRegimeStrategy

    data = DataLoader("SPY", "2005-01-01", end, sources=("yfinance",)).get_data()
    params = StrategyParams(cost_bps=0.0)
    if ml:
        data = attach_recursive_ml_up_proba(data, eval_start=eval_start)
        params = replace(params, ml_up_min_long=0.55, ml_up_max_short=0.45)
    strat = VolatilityRegimeStrategy(data, params=params)
    strat.run_backtest()
    sig = strat.decided_exposure()
    sig.index = pd.DatetimeIndex(sig.index).tz_localize(None).normalize()
    return sig


def build_candidates(close: pd.DataFrame, rf_daily: pd.Series, vr_signals: Dict[str, pd.Series]) -> Dict[str, pd.DataFrame]:
    return {
        "A1_volmanaged_spy_21d": vol_managed_spy(close, 21),
        "A2_volmanaged_spy_63d": vol_managed_spy(close, 63),
        "B1_spy_sma10m": spy_sma_filter(close),
        "B2_spy_sma10m_voltarget": vol_target(spy_sma_filter(close), close, TARGET_VOL, 1.5).pipe(_monthly),
        "B3_dual_momentum": dual_momentum(close, rf_daily),
        "B4_dual_momentum_voltarget": vol_target(dual_momentum(close, rf_daily), close, TARGET_VOL, 1.5).pipe(_monthly),
        "C1_multi_asset_tsmom": multi_asset(close, rf_daily, "tsmom"),
        "C2_risk_parity": multi_asset(close, rf_daily, "rp"),
        "C3_risk_parity_trend": multi_asset(close, rf_daily, "rp_trend"),
        "D1_qqq_spy_rotation": qqq_spy_rotation(close),
        "D2_sector_momentum": sector_momentum(close, spy_filter=False),
        "D3_sector_momentum_filtered": sector_momentum(close, spy_filter=True),
        "E1_vr_levered": vr_levered(close, vr_signals["off"]),
        "E2_vr_ml_levered": vr_levered(close, vr_signals["on"]),
    }


def scale_weights(w: pd.DataFrame, k: float, max_gross: float = 2.0) -> pd.DataFrame:
    gross = w.abs().sum(axis=1)
    factor = np.minimum(k, max_gross / gross.where(gross > 0, np.inf))
    return w.mul(factor, axis=0)
