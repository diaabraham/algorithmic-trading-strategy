"""Point-in-time data: Yahoo Finance prices and FRED macro series with publication lags."""

from __future__ import annotations

import io
import os
import pickle
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import requests

CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
# FRED series and the calendar-day lag after the observation date before the value is public.
# ICSA: week ending Saturday, released the following Thursday (8:30 ET) -> usable at Friday's close.
FRED_SERIES = {"T10Y3M": 1, "BAA10Y": 1, "ICSA": 6}


def _cached(name: str, fn):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"study3_{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    obj = fn()
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    return obj


def yahoo(tickers: Iterable[str], end: str) -> Dict[str, pd.DataFrame]:
    import yfinance as yf

    tickers = list(tickers)

    def fetch():
        out = {}
        for t in tickers:
            d = yf.download(t, period="max", auto_adjust=True, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d.index = pd.to_datetime(d.index).tz_localize(None)
            out[t] = d[["Open", "High", "Low", "Close", "Volume"]].astype(float).loc[:end]
        return out

    data = _cached("yahoo_" + "_".join(sorted(t.replace("^", "") for t in tickers)) + "_" + end, fetch)
    return {t: d.loc[:end] for t, d in data.items()}


def fred(end: str) -> Dict[str, pd.Series]:
    def fetch():
        out = {}
        for sid in FRED_SERIES:
            txt = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=60).text
            df = pd.read_csv(io.StringIO(txt))
            s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            s.index = pd.to_datetime(df.iloc[:, 0])
            out[sid] = s.dropna().loc[:end]
        return out

    raw = _cached("fred_" + end, fetch)
    out = {}
    for sid, s in raw.items():
        avail = s.copy()
        avail.index = avail.index + pd.Timedelta(days=FRED_SERIES[sid])  # date the value becomes public
        out[sid] = avail.loc[: pd.Timestamp(end)]
    return out


def asof(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Last value public on or before each date in ``index``."""
    s = series.sort_index()
    pos = s.index.searchsorted(index, side="right") - 1
    vals = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=index)
