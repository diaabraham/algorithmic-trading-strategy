"""In-sample / out-of-sample study on SPY.

Protocol (fixed before looking at out-of-sample data):
* Candidates are a small, pre-declared grid (see ``candidate_exposures``).
* The winner is the candidate with the highest in-sample Sharpe ratio
  (daily returns in excess of 3-month T-bills, 2010-01-01 .. 2019-12-31).
* The winner and the repository's default configuration are then evaluated
  once on 2020-01-01 .. latest, next to SPY buy-and-hold on the same data.

Every candidate is simulated with ``src.execution.simulate_exposure``:
decisions at the close, fills at the next open, ``cost_bps`` per side, idle
cash earning T-bills, leverage financed at T-bills + 1%, shorts paying 1%/yr.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

import numpy as np
import pandas as pd

from src.execution import performance, simulate_exposure
from src.strategy import StrategyParams, VolatilityRegimeStrategy

IS_START, IS_END = "2010-01-01", "2019-12-31"
OOS_START = "2020-01-01"
# StrategyParams defaults (ML gate off). Before --train-ml was wired up, main.py
# always applied the 0.55 / 0.45 ML gate, so that variant is reported too.
REPO_DEFAULT = "vr_vol0.7_hold5_mloff"
PREVIOUS_CLI_DEFAULT = "vr_vol0.7_hold5_mlon"


def t_bill_daily(index: pd.DatetimeIndex) -> pd.Series:
    """3-month T-bill (^IRX, % annualised) as a daily simple rate on ``index``."""
    import yfinance as yf

    irx = yf.download("^IRX", start=str(index[0].date() - pd.Timedelta(days=10)), progress=False, auto_adjust=False)
    col = irx["Close"]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    col.index = pd.to_datetime(col.index).tz_localize(None)
    naive = pd.DatetimeIndex(index).tz_localize(None) if index.tz is not None else pd.DatetimeIndex(index)
    rate = col.reindex(naive, method="ffill").fillna(0.0) / 100.0 / 252.0
    rate.index = index
    return rate


def _month_end_mask(index: pd.DatetimeIndex) -> np.ndarray:
    periods = index.to_period("M") if index.tz is None else index.tz_localize(None).to_period("M")
    return np.r_[periods[1:] != periods[:-1], True]


def sma_exposure(close: pd.Series, window: int) -> pd.Series:
    """Faber-style trend filter: long when the month-end close is above its SMA."""
    sma = close.rolling(window).mean()
    month_end = _month_end_mask(close.index)
    signal = pd.Series(np.where(close > sma, 1.0, 0.0), index=close.index)
    signal[sma.isna()] = 0.0
    return signal.where(month_end).ffill().fillna(0.0)


def vol_managed_exposure(close: pd.Series, target: float, max_leverage: float = 1.5, lookback: int = 21) -> pd.Series:
    """Moreira-Muir style: exposure = target / trailing realised vol, set at month ends."""
    vol = close.pct_change().rolling(lookback).std() * np.sqrt(252)
    raw = (target / vol).clip(upper=max_leverage)
    month_end = _month_end_mask(close.index)
    return raw.where(month_end).ffill().fillna(0.0)


def vr_exposure(data: pd.DataFrame, params: StrategyParams) -> pd.Series:
    strat = VolatilityRegimeStrategy(data, params=params)
    strat.run_backtest()
    return strat.decided_exposure()


def candidate_exposures(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """All pre-declared candidates (decided exposure at each close)."""
    close = data["Close"].astype(float)
    out: Dict[str, pd.Series] = {}
    base = StrategyParams(cost_bps=0.0)
    for vol in (0.6, 0.7, 0.8):
        for hold in (5, 10):
            for ml in (False, True):
                params = replace(
                    base,
                    vol_threshold=vol,
                    max_hold_days=hold,
                    ml_up_min_long=0.55 if ml else 0.0,
                    ml_up_max_short=0.45 if ml else 1.0,
                )
                out[f"vr_vol{vol}_hold{hold}_ml{'on' if ml else 'off'}"] = vr_exposure(data, params)
    for window in (150, 200):
        out[f"sma{window}_monthly"] = sma_exposure(close, window)
    for target in (0.12, 0.16):
        out[f"volmanaged_target{int(target * 100)}"] = vol_managed_exposure(close, target)
    return out


def run_study(data: pd.DataFrame, rf_daily: pd.Series, cost_bps: float = 2.0) -> dict:
    idx = data.index
    is_mask = (idx >= pd.Timestamp(IS_START, tz=idx.tz)) & (idx <= pd.Timestamp(IS_END, tz=idx.tz) + pd.Timedelta(days=1))
    oos_mask = idx >= pd.Timestamp(OOS_START, tz=idx.tz)
    sims = {}
    for name, exposure in candidate_exposures(data).items():
        sims[name] = simulate_exposure(data["Open"], data["Close"], exposure, cost_bps=cost_bps, rf_daily=rf_daily)

    def stats(name: str, mask) -> dict:
        sim = sims[name].loc[mask]
        perf = performance(sim["ret"], rf_daily.loc[mask])
        perf["avg_exposure"] = float(sim["exposure"].mean())
        perf["turnover_per_year"] = float(sim["turnover"].sum() / (len(sim) / 252.0))
        perf["rebalances"] = int((sim["turnover"] > 0).sum())
        return perf

    in_sample = {name: stats(name, is_mask) for name in sims}
    selected = max(in_sample, key=lambda n: in_sample[n]["sharpe"])

    def bench(mask) -> dict:
        sub = data.loc[mask]
        sim = simulate_exposure(sub["Open"], sub["Close"], pd.Series(1.0, index=sub.index), cost_bps=cost_bps, rf_daily=rf_daily.loc[mask])
        return performance(sim["ret"], rf_daily.loc[mask])

    reported = list(dict.fromkeys([selected, REPO_DEFAULT, PREVIOUS_CLI_DEFAULT]))
    oos = {name: stats(name, oos_mask) for name in reported}
    return {
        "protocol": {
            "in_sample": [IS_START, IS_END],
            "out_of_sample_start": OOS_START,
            "selection": "highest in-sample Sharpe (excess of T-bills)",
            "execution": f"next-open fills, {cost_bps} bp per side, cash at T-bills, leverage at T-bills + 1%",
        },
        "in_sample": in_sample,
        "in_sample_benchmark_spy": bench(is_mask),
        "selected": selected,
        "out_of_sample": oos,
        "out_of_sample_benchmark_spy": bench(oos_mask),
        "average_t_bill_oos": float(rf_daily.loc[oos_mask].mean() * 252),
        "_sims": sims,
    }
