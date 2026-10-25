"""Long-format feature panels. A row (m, a) holds features known at the end of month m
and the target for month m + 1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.study5.data import CAT_GROUPS

FEATURES = [
    "r1", "r2_12", "vol12", "mean36", "mean60", "beta36", "dd12", "rank_r2_12", "rank_vol12",
    "years_since_pub", "is_industry", *[f"cat_{c}" for c in CAT_GROUPS], "cat_Other",
    "ls_r1", "ls_r2_12", "mkt_r1", "mkt_r12", "mkt_vol12", "rf", "term_spread", "default_spread", "inflation_yoy",
]
START = "1930-01-31"


def _own(R: pd.DataFrame, mkt: pd.Series) -> dict:
    lr = np.log1p(R)
    f = {"r1": R, "r2_12": np.expm1(lr.shift(1).rolling(11, min_periods=11).sum()),
         "vol12": R.rolling(12, min_periods=12).std(), "mean36": R.rolling(36, min_periods=24).mean(),
         "mean60": R.rolling(60, min_periods=36).mean()}
    m = mkt.reindex(R.index)
    cov = R.rolling(36, min_periods=24).cov(m)
    f["beta36"] = cov.div(m.rolling(36, min_periods=24).var(), axis=0)
    c = lr.cumsum().where(R.notna())
    f["dd12"] = np.expm1(c - c.rolling(12, min_periods=1).max())
    return f


def _usable(pub_year: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """True where the asset is usable at that month (published before that calendar year)."""
    years = index.year.to_numpy()[:, None]
    return pd.DataFrame(pub_year.to_numpy()[None, :] < years, index=index, columns=pub_year.index)


def _rank_among(values: pd.DataFrame, usable: pd.DataFrame) -> pd.DataFrame:
    """Percentile of each asset's value within the assets usable that month."""
    out = np.full(values.shape, np.nan)
    v = values.to_numpy()
    u = usable.to_numpy()
    for i in range(len(v)):
        ref = np.sort(v[i][u[i] & np.isfinite(v[i])])
        if len(ref) < 5:
            continue
        ok = np.isfinite(v[i])
        out[i, ok] = np.searchsorted(ref, v[i, ok], side="right") / len(ref)
    return pd.DataFrame(out, index=values.index, columns=values.columns)


def build(d: dict, kind: str) -> pd.DataFrame:
    """kind = 'long' (long legs + industries, demeaned target) or 'ls' (long-shorts, raw target)."""
    meta = d["meta"]
    if kind == "long":
        R = pd.concat([d["long_leg"], d["industries"]], axis=1)
        pub = pd.concat([meta["Year"], pd.Series(-1, index=d["industries"].columns)])
        cats = pd.concat([meta["Cat.Data"], pd.Series("IND", index=d["industries"].columns)])
        ls = d["ls"].reindex(columns=R.columns)
    else:
        R = d["ls"]
        pub, cats = meta["Year"], meta["Cat.Data"]
        ls = None
    R = R.loc[R.index >= "1926-07-31"]
    idx = R.index
    usable = _usable(pub, idx)
    feats = _own(R, d["market"])
    feats["rank_r2_12"] = _rank_among(feats["r2_12"], usable)
    feats["rank_vol12"] = _rank_among(feats["vol12"], usable)
    if ls is not None:
        ls = ls.reindex(idx)
        feats["ls_r1"] = ls.fillna(0.0).where(R.notna())
        feats["ls_r2_12"] = np.expm1(np.log1p(ls).shift(1).rolling(11, min_periods=11).sum()).fillna(0.0).where(R.notna())
    else:
        feats["ls_r1"] = R * 0.0
        feats["ls_r2_12"] = R * 0.0
    nxt = R.shift(-1)
    if kind == "long":
        mask_next = usable.to_numpy() & np.isfinite(nxt.to_numpy())
        xs_mean = pd.Series(np.nanmean(np.where(mask_next, nxt.to_numpy(), np.nan), axis=1), index=idx)
        target = nxt.sub(xs_mean, axis=0)
    else:
        target = nxt
    rows = []
    mkt = d["market"].reindex(idx)
    macro = pd.DataFrame({
        "mkt_r1": mkt, "mkt_r12": np.expm1(np.log1p(mkt).rolling(12).sum()),
        "mkt_vol12": mkt.rolling(12).std(), "rf": d["rf"].reindex(idx),
    }).join(d["macro"].reindex(idx))
    for a in R.columns:
        df = pd.DataFrame({k: v[a] for k, v in feats.items()})
        df["target"] = target[a]
        df["next_ret"] = nxt[a]
        df["asset"] = a
        df["pub_year"] = int(pub[a])
        df["is_industry"] = float(cats[a] == "IND")
        df["years_since_pub"] = (idx.year - pub[a]).astype(float) if cats[a] != "IND" else 0.0
        for c in CAT_GROUPS:
            df[f"cat_{c}"] = float(cats[a] == c)
        df["cat_Other"] = float(cats[a] not in CAT_GROUPS and cats[a] != "IND")
        df = df.join(macro)
        df = df[R[a].notna()]
        rows.append(df)
    panel = pd.concat(rows)
    panel.index.name = "date"
    panel = panel.reset_index()
    panel = panel[panel["date"] >= START]
    panel["month"] = (panel["date"].dt.year - 1900) * 12 + panel["date"].dt.month - 1
    return panel.sort_values(["month", "asset"]).reset_index(drop=True)


def to_arrays(panel: pd.DataFrame) -> dict:
    return {
        "X": panel[FEATURES].to_numpy(np.float32),
        "y": panel["target"].to_numpy(np.float32),
        "month": panel["month"].to_numpy(np.int32),
        "pub_year": panel["pub_year"].to_numpy(np.int32),
    }
