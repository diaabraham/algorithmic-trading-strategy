"""Live paper tracking of the frozen study 5 rule L0 (tag paper-trade-v2).

Rule (results/study5_frozen_config.md, unchanged): at each month end t, hold for
month t+1 an equal-weight portfolio of the 10 assets with the highest 2-12 month
return (months t-11..t-1, compounded). Assets are the long-direction value-weighted
decile legs of the Chen-Zimmermann anomalies and the French 49 value-weighted
industries. An anomaly is eligible only if it was published before t's calendar year
and has a return at t. Costs are those of study 5: within-portfolio drag of
2 x turnover x cost / 12 plus switching cost, at 20 bp per side for anomaly legs and
10 bp for industries.

The anomaly legs are not priced in real time. Chen-Zimmermann publish them about
once a year, a year or more after the fact. Each decision is therefore computed when
the first data release that covers the decision month appears. It is logged
append-only and never recomputed. Returns are filled in when a release covering the
holding month appears. The code and the parameters are fixed at the tag, so the
picks are a mechanical function of published data and cannot be chosen with
hindsight.
"""

from __future__ import annotations

import io
import os
import pickle
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import requests

from src.factor_study import BASE as FRENCH_BASE, parse_monthly
from src.study5.data import long_legs
from src.study5.evaluate import cost_of, turnover_of

TOP_K = 10
CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache", "live")


def momentum_2_12(R: pd.DataFrame) -> pd.DataFrame:
    return np.expm1(np.log1p(R).shift(1).rolling(11, min_periods=11).sum())


def picks_at(R: pd.DataFrame, pub_year: pd.Series, t: pd.Timestamp, mom: pd.DataFrame | None = None):
    """The frozen L0 picks decided at month end ``t`` (None if fewer than 10 eligible assets)."""
    if t not in R.index:
        return None
    mom = momentum_2_12(R) if mom is None else mom
    m = mom.loc[t]
    ok = R.loc[t].notna() & m.notna() & (pub_year.reindex(R.columns) < t.year)
    cand = m[ok].sort_index()  # ties broken alphabetically, as in the study 5 panel
    if len(cand) < TOP_K:
        return None
    return list(cand.nlargest(TOP_K).index)


def frozen_costs(meta: pd.DataFrame, assets: list) -> tuple[dict, dict]:
    """Per-asset annual turnover and cost per side, as in study 5."""
    return {a: turnover_of(meta, a) for a in assets}, {a: cost_of(a) for a in assets}


def month_cost(new: list, prev: list, turnover: dict, cost: dict, cost_mult: float = 1.0) -> float:
    """Within-portfolio drag of the new holdings plus the cost of switching from ``prev``."""
    w = {a: 1.0 / TOP_K for a in new}
    p = {a: 1.0 / TOP_K for a in prev}
    within = sum(w[a] * 2 * turnover[a] * cost[a] / 12 for a in w)
    switch = sum(abs(w.get(a, 0) - p.get(a, 0)) * cost[a] for a in set(w) | set(p))
    return cost_mult * (within + switch)


def _cache_path(name):
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, name)


def cz_release() -> str:
    import openassetpricing.urls as u

    return max(x for x in dir(u) if x.startswith("release")).replace("_url", "")


def cz_data(seed: str | None = None):
    """CZ deciles_vw and SignalDoc of the newest release the installed package knows (cached per release)."""
    rel = cz_release()
    path = _cache_path(f"cz_deciles_vw_{rel}.pkl")
    if not os.path.exists(path):
        if seed and os.path.exists(seed):
            with open(seed, "rb") as fh:
                obj = pickle.load(fh)
        else:
            import openassetpricing as oap

            o = oap.OpenAP()
            obj = (o.dl_port("deciles_vw", "pandas"), o.dl_signal_doc("pandas"))
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)
    with open(path, "rb") as fh:
        port, doc = pickle.load(fh)
    return rel, port, doc


def french_fresh(name: str) -> str:
    """Re-download a Ken French file on every run (falls back to the last copy offline)."""
    path = _cache_path(f"{name}.csv")
    try:
        raw = requests.get(f"{FRENCH_BASE}/{name}_CSV.zip", timeout=60).content
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        with open(path, "w") as fh:
            fh.write(text)
    except Exception:  # noqa: BLE001
        if not os.path.exists(path):
            raise
    with open(path) as fh:
        return fh.read()


def load_universe(universe: list, seed: str | None = None):
    """Monthly returns of the frozen 228-asset universe and data-vintage info."""
    rel, port, doc = cz_data(seed)
    long_leg = long_legs(port, doc)[0]
    ind = parse_monthly(french_fresh("49_Industry_Portfolios"), "Average Value Weighted Returns -- Monthly")
    ind.columns = ["IND_" + c.strip() for c in ind.columns]
    R = pd.concat([long_leg, ind], axis=1)
    R = R.loc[R.index >= "1926-07-31"].reindex(columns=sorted(universe))
    info = {"cz_release": rel, "cz_data_end": str(long_leg.dropna(how="all").index.max().date()),
            "industries_data_end": str(ind.dropna(how="all").index.max().date()), "fetched_on": str(date.today())}
    return R, info
