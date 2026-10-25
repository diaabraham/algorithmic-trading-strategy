"""Point-in-time inputs for study 5. All returns are monthly decimals indexed by month-end."""

from __future__ import annotations

import io
import os
import pickle

import numpy as np
import pandas as pd
import requests

from src.factor_study import _download as french_download, parse_monthly

CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache", "study5")
FRED_LAGS = {"GS10": 1, "TB3MS": 1, "BAA": 1, "AAA": 1, "CPIAUCSL": 2}  # extra months before public
CAT_GROUPS = ["Accounting", "Price", "Trading", "Analyst", "Options"]  # rest -> "Other"


def _cache(name, fn):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name + ".pkl")
    if os.path.exists(p):
        with open(p, "rb") as fh:
            return pickle.load(fh)
    obj = fn()
    with open(p, "wb") as fh:
        pickle.dump(obj, fh)
    return obj


def _month_end(s):
    return pd.to_datetime(s) + pd.offsets.MonthEnd(0)


def cz_portfolios():
    def fetch():
        import openassetpricing as oap

        o = oap.OpenAP()
        return o.dl_port("deciles_vw", "pandas"), o.dl_signal_doc("pandas")

    return _cache("cz_deciles_vw", fetch)


def long_legs(port: pd.DataFrame, doc: pd.DataFrame):
    """Long-direction legs, signed long-shorts, Nlong and metadata from CZ deciles and SignalDoc."""
    port = port.copy()
    port["date"] = _month_end(port["date"])
    doc = doc.drop_duplicates("Acronym").set_index("Acronym")
    names = sorted(set(port["signalname"]) & set(doc.index))
    meta = doc.loc[names, ["Year", "Sign", "Cat.Data", "Portfolio Period"]].copy()
    wide = port.pivot_table(index="date", columns=["signalname", "port"], values="ret") / 100.0
    long_leg, ls, nlong = {}, {}, {}
    nl = port.pivot_table(index="date", columns=["signalname", "port"], values="Nlong")
    for s in names:
        leg = "10" if meta.at[s, "Sign"] > 0 else "01"
        if (s, leg) in wide.columns and (s, "LS") in wide.columns:
            long_leg[s] = wide[(s, leg)]
            # CZ "LS" is decile 10 minus decile 1; flip it so positive is the published direction.
            ls[s] = float(np.sign(meta.at[s, "Sign"])) * wide[(s, "LS")]
            nlong[s] = nl[(s, leg)] if (s, leg) in nl.columns else np.nan
    long_leg, ls = pd.DataFrame(long_leg), pd.DataFrame(ls)
    return long_leg, ls, pd.DataFrame(nlong), meta.loc[long_leg.columns]


def load(end: str) -> dict:
    """Wide matrices of long legs, long-shorts, industries, market, T-bill and macro, cut at ``end``."""
    e = pd.Timestamp(end) + pd.offsets.MonthEnd(0)
    port, doc = cz_portfolios()
    port = port[_month_end(port["date"]) <= e]
    long_leg, ls, nlong, meta = long_legs(port, doc)
    ind = parse_monthly(french_download("49_Industry_Portfolios"), "Average Value Weighted Returns -- Monthly")
    ind.columns = ["IND_" + c.strip() for c in ind.columns]
    ff = parse_monthly(french_download("F-F_Research_Data_Factors"))
    out = {
        "long_leg": long_leg, "ls": ls, "industries": ind.loc[:e], "nlong": nlong,
        "market": (ff["Mkt-RF"] + ff["RF"]).loc[:e], "rf": ff["RF"].loc[:e],
        "meta": meta, "macro": macro(e),
    }
    return out


def macro(end: pd.Timestamp) -> pd.DataFrame:
    def fetch():
        out = {}
        for sid in FRED_LAGS:
            txt = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=60).text
            df = pd.read_csv(io.StringIO(txt))
            s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            s.index = _month_end(df.iloc[:, 0])
            out[sid] = s
        return out

    raw = _cache("fred_monthly", fetch)
    m = {}
    for sid, s in raw.items():
        m[sid] = s.copy()
        m[sid].index = m[sid].index + pd.offsets.MonthEnd(FRED_LAGS[sid])
    df = pd.DataFrame(m)
    out = pd.DataFrame(index=df.index)
    out["term_spread"] = df["GS10"] - df["TB3MS"]
    out["default_spread"] = df["BAA"] - df["AAA"]
    out["inflation_yoy"] = np.log(df["CPIAUCSL"] / df["CPIAUCSL"].shift(12))
    return out.loc[:end]
