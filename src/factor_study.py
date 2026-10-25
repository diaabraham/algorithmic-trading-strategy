"""Study 4: long-only large-cap factor tilts from the Kenneth R. French Data Library.

See results/study4_protocol.md. Returns are monthly, in decimals.
"""

from __future__ import annotations

import io
import os
import zipfile
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
from scipy.stats import kurtosis, norm, skew

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "french")

# id -> (US file, column, ex-US file or None, annual one-way turnover)
CANDIDATES = {
    "F1_big_momentum": ("6_Portfolios_ME_Prior_12_2", "BIG HiPRIOR", "Developed_ex_US_6_Portfolios_ME_Prior_12_2", 2.00),
    "F2_big_value": ("6_Portfolios_2x3", "BIG HiBM", "Developed_ex_US_6_Portfolios_ME_BE-ME", 0.30),
    "F3_big_profitability": ("6_Portfolios_ME_OP_2x3", "BIG HiOP", "Developed_ex_US_6_Portfolios_ME_OP", 0.30),
    "F4_low_variance": ("Portfolios_Formed_on_VAR", "Lo 20", None, 1.00),
}
MIX = "F5_mix_mom_value_prof"
MIX_PARTS = ["F1_big_momentum", "F2_big_value", "F3_big_profitability"]
MIX_TURNOVER = 1.00
COST_PER_SIDE = 0.0010
FEE = 0.0015
N_TRIALS = 50


def cost_drag(turnover: float, multiple: float = 1.0) -> float:
    """Annual drag: 2 sides x turnover x cost per side, plus fees."""
    return multiple * (2 * turnover * COST_PER_SIDE + FEE)


def _download(name: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.csv")
    if not os.path.exists(path):
        raw = requests.get(f"{BASE}/{name}_CSV.zip", timeout=60).content
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        with open(path, "w") as fh:
            fh.write(text)
    with open(path) as fh:
        return fh.read()


def parse_monthly(text: str, section: Optional[str] = None) -> pd.DataFrame:
    """First monthly table (or the one after ``section``) as decimals, indexed by month end."""
    lines = text.replace("\r", "").split("\n")
    start = 0
    if section is not None:
        start = next(i for i, l in enumerate(lines) if section in l)
    hdr = next(i for i in range(start, len(lines)) if lines[i].startswith(",") or lines[i].strip().startswith(",") )
    cols = [c.strip() for c in lines[hdr].split(",")[1:]]
    rows = []
    for l in lines[hdr + 1 :]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) < 2 or not (parts[0].isdigit() and len(parts[0]) == 6):
            break
        rows.append([parts[0]] + parts[1:])
    df = pd.DataFrame([r[1:] for r in rows], columns=cols[: len(rows[0]) - 1], dtype=float)
    df.index = pd.to_datetime([r[0] for r in rows], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.where(df > -99)
    return df / 100.0


def load(end: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Gross candidate returns (US and ex-US), markets and T-bills, truncated at ``end``."""
    us_ff = parse_monthly(_download("F-F_Research_Data_Factors"))
    ex_ff = parse_monthly(_download("Developed_ex_US_3_Factors"))
    us = pd.DataFrame({k: parse_monthly(_download(f), "Value Weighted Returns -- Monthly")[c] for k, (f, c, _, _) in CANDIDATES.items()})
    ex = pd.DataFrame({k: parse_monthly(_download(x), "Value Weighted Returns -- Monthly")[c]
                       for k, (_, c, x, _) in CANDIDATES.items() if x})
    out = {
        "us": us, "ex": ex,
        "us_market": us_ff["Mkt-RF"] + us_ff["RF"], "us_rf": us_ff["RF"],
        "ex_market": ex_ff["Mkt-RF"] + ex_ff["RF"], "ex_rf": ex_ff["RF"],
    }
    if end is not None:
        e = pd.Timestamp(end)
        out = {k: v.loc[:e] for k, v in out.items()}
    return out


def net_returns(gross: pd.DataFrame, cost_multiple: float = 1.0) -> pd.DataFrame:
    """Candidate returns net of the pre-registered monthly cost drag, plus the F5 mix."""
    net = pd.DataFrame(index=gross.index)
    for k in gross.columns:
        net[k] = gross[k] - cost_drag(CANDIDATES[k][3], cost_multiple) / 12
    if all(p in gross.columns for p in MIX_PARTS):
        net[MIX] = gross[MIX_PARTS].mean(axis=1, skipna=False) - cost_drag(MIX_TURNOVER, cost_multiple) / 12
    return net


def monthly_stats(r: pd.Series, rf: pd.Series) -> dict:
    r = r.dropna()
    rf = rf.reindex(r.index).fillna(0.0)
    years = len(r) / 12
    growth = float((1 + r).prod())
    ex = r - rf
    eq = (1 + r).cumprod()
    return {"start": str(r.index[0].date())[:7], "end": str(r.index[-1].date())[:7], "months": len(r),
            "cagr": growth ** (1 / years) - 1, "sharpe": float(ex.mean() / ex.std() * np.sqrt(12)),
            "ann_vol": float(r.std() * np.sqrt(12)), "max_drawdown": float((eq / eq.cummax() - 1).min())}


def sharpe_stats(active: pd.Series, trial_sharpes_annual=None, n_trials: int = N_TRIALS) -> dict:
    """Probabilistic and deflated Sharpe ratio of monthly active returns."""
    a = active.dropna()
    t = len(a)
    sr = a.mean() / a.std()
    g3, g4 = float(skew(a)), float(kurtosis(a, fisher=False))
    den = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    out = {"active_sharpe_annual": float(sr * np.sqrt(12)), "months": t,
           "psr_vs_zero": float(norm.cdf(sr * np.sqrt(t - 1) / den))}
    if trial_sharpes_annual is not None:
        var = float(np.var(np.asarray(trial_sharpes_annual) / np.sqrt(12), ddof=1))
        gam = 0.5772156649
        sr0 = np.sqrt(var) * ((1 - gam) * norm.ppf(1 - 1 / n_trials) + gam * norm.ppf(1 - 1 / (n_trials * np.e)))
        out.update({"trials": n_trials, "null_max_sharpe_annual": float(sr0 * np.sqrt(12)),
                    "deflated_sharpe_ratio": float(norm.cdf((sr - sr0) * np.sqrt(t - 1) / den))})
    return out


def spy_monthly(end: str) -> pd.Series:
    import yfinance as yf

    d = yf.download("SPY", period="max", auto_adjust=True, progress=False)["Close"]
    if isinstance(d, pd.DataFrame):
        d = d.iloc[:, 0]
    d.index = pd.to_datetime(d.index).tz_localize(None)
    m = d.resample("ME").last()
    return m.pct_change().dropna().loc[: pd.Timestamp(end)]
