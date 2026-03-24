"""
Walk-forward ML overlay: train only on bars strictly before eval_start, then score later bars.

Uses a small RandomForest on lagged returns and volatility to estimate P(up next day).
Does not retrain inside the backtest loop (fast, conservative vs leakage).
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def attach_ml_up_proba(
    df: pd.DataFrame,
    eval_start: Union[str, pd.Timestamp],
    min_train_rows: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Add column ML_Up_Proba. Training rows: index < eval_start; applied rows: index >= eval_start.
    Rows before eval get 0.5 (neutral) so strategy ML gate should use ml_up_min_long only with eval window.
    """
    out = df.copy()
    if "Close" not in out.columns:
        out["ML_Up_Proba"] = 0.5
        return out

    eval_ts = pd.to_datetime(eval_start, utc=True)
    close = out["Close"].astype(float)
    ret1 = close.pct_change()
    ret2 = close.pct_change(2)
    ret5 = close.pct_change(5)

    vol = out["Volatility"] if "Volatility" in out.columns else ret1.rolling(20).std() * np.sqrt(252)

    feats = pd.DataFrame(
        {
            "r1": ret1,
            "r2": ret2,
            "r5": ret5,
            "vol": vol,
        },
        index=out.index,
    )
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y_raw = (close.shift(-1) > close).astype(float)
    train_ix = out.index < eval_ts
    X_train = feats.loc[train_ix]
    y_train = y_raw.loc[X_train.index]
    valid = y_train.notna()
    X_train = X_train.loc[valid]
    y_train = y_train.loc[valid].astype(int)

    proba = np.full(len(out), 0.5, dtype=float)
    if len(X_train) < min_train_rows or y_train.nunique() < 2:
        logger.warning("ML train skipped: rows=%s unique_y=%s", len(X_train), y_train.nunique())
        out["ML_Up_Proba"] = proba
        return out

    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=80,
        max_depth=5,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train.values, y_train.values)

    X_all = feats.fillna(0.0).values
    classes = list(clf.classes_)
    if 1 not in classes:
        out["ML_Up_Proba"] = proba
        return out
    pos = classes.index(1)
    p = clf.predict_proba(X_all)[:, pos]
    apply_mask = (out.index >= eval_ts).to_numpy()
    proba[apply_mask] = p[apply_mask]
    out["ML_Up_Proba"] = proba
    return out
