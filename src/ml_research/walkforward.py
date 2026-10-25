"""Expanding-window walk-forward predictions ("recursive learning").

At each refit date (first trading day of each month) the model is retrained on
every row whose label was fully observed by that date's close. Hyper-parameters
are re-chosen by purged k-fold CV inside the training window once a year. The
model then predicts every day until the next refit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml_research.models import Model, choose_params


def walk_forward(features: pd.DataFrame, label: pd.Series, kind: str, horizon: int, min_train_days: int,
                 start: pd.Timestamp = None) -> pd.DataFrame:
    idx = features.index
    X = features.to_numpy(float)
    y = label.reindex(idx).to_numpy(float)
    usable = np.flatnonzero(np.isfinite(y) & (np.isfinite(X).sum(1) >= X.shape[1] * 0.6))
    if len(usable) == 0:
        raise ValueError("no usable rows")
    first = usable[0]
    months = idx.to_period("M")
    refit = np.r_[True, months[1:] != months[:-1]]
    pred = np.full(len(idx), np.nan)
    psd = np.full(len(idx), np.nan)
    chosen = []
    model, params, params_year = None, None, None
    for t in np.flatnonzero(refit):
        if t - first < min_train_days or (start is not None and idx[t] < start - pd.Timedelta(days=40)):
            continue
        # Rows s whose label ends at the fill on bar s + horizon + 1 <= t are known at t's close.
        train = usable[usable + horizon + 1 <= t]
        if len(train) < min_train_days * 0.8:
            continue
        if params is None or idx[t].year != params_year:
            params = choose_params(kind, X[train], y[train], horizon)
            params_year = idx[t].year
            chosen.append((str(idx[t].date()), params))
        model = Model(kind, params).fit(X[train], y[train])
        nxt = np.flatnonzero(refit[t + 1 :])
        end = t + 1 + nxt[0] if len(nxt) else len(idx)
        pred[t:end] = model.predict(X[t:end])
        psd[t:end] = model.train_pred_sd
    out = pd.DataFrame({"pred": pred, "pred_sd": psd}, index=idx)
    out.attrs["params"] = chosen
    return out
