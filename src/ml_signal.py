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


def training_history_start(eval_start: str, calendar_days: int = 504) -> str:
    """Calendar start that leaves history strictly before eval_start for training."""
    start = pd.Timestamp(eval_start)
    return (start - pd.Timedelta(days=calendar_days)).date().isoformat()


def _ml_feature_frame(out: pd.DataFrame) -> pd.DataFrame:
    close = out["Close"].astype(float)
    ret1 = close.pct_change()
    vol = out["Volatility"] if "Volatility" in out.columns else ret1.rolling(20).std() * np.sqrt(252)
    if "Vol_Percentile" in out.columns:
        vol_pct = out["Vol_Percentile"]
    else:
        vol_pct = vol.rolling(252, min_periods=60).rank(pct=True)
    distance = (close / out["EMA_20"] - 1.0) if "EMA_20" in out.columns else ret1 * 0.0
    feats = pd.DataFrame(
        {
            "r1": ret1,
            "r2": close.pct_change(2),
            "r5": close.pct_change(5),
            "vol": vol,
            "vol_pct": vol_pct,
            "ema_distance": distance,
        },
        index=out.index,
    )
    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fit_up_proba(feats: pd.DataFrame, y_raw: pd.Series, min_train_rows: int, random_state: int):
    """Train on rows whose next-bar label is already known. Returns P(up) or None."""
    valid = y_raw.notna()
    X_train = feats.loc[valid]
    y_train = y_raw.loc[valid].astype(int)
    if len(X_train) < min_train_rows or y_train.nunique() < 2:
        return None
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=80,
        max_depth=5,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=1,
    )
    clf.fit(X_train.values, y_train.values)
    classes = list(clf.classes_)
    if 1 not in classes:
        return None
    return clf, classes.index(1)


def attach_recursive_ml_up_proba(
    df: pd.DataFrame,
    eval_start: Union[str, pd.Timestamp],
    refit_every: int = 21,
    min_train_rows: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Walk-forward RandomForest. Refit every `refit_every` bars on labels whose next close
    is strictly before that block, then score only the block. Adapts to the current regime
    without using future prices.
    """
    out = df.copy()
    proba = np.full(len(out), 0.5, dtype=float)
    if "Close" not in out.columns or len(out) == 0:
        out["ML_Up_Proba"] = proba
        return out

    eval_ts = pd.to_datetime(eval_start, utc=True)
    feats = _ml_feature_frame(out)
    close = out["Close"].astype(float)
    y_raw = (close.shift(-1) > close).astype(float)
    idx = out.index
    start_pos = int(idx.searchsorted(eval_ts))
    step = max(int(refit_every), 1)
    pos = start_pos
    while pos < len(idx):
        block_end = min(pos + step, len(idx))
        # The label on bar t uses the close at t+1. Drop the bar immediately
        # before the block so that close is not in the training label.
        train_stop = max(pos - 1, 0)
        fitted = _fit_up_proba(feats.iloc[:train_stop], y_raw.iloc[:train_stop], min_train_rows, random_state)
        if fitted is not None:
            clf, up_col = fitted
            scored = clf.predict_proba(feats.iloc[pos:block_end].values)[:, up_col]
            proba[pos:block_end] = scored
        else:
            logger.warning("Recursive ML refit skipped at %s", idx[pos])
        pos = block_end
    out["ML_Up_Proba"] = proba
    return out


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
    feats = _ml_feature_frame(out)
    y_raw = (close.shift(-1) > close).astype(float)
    train_ix = out.index < eval_ts
    # Drop the last pre-eval bar: its label is the first eval close.
    if train_ix.any():
        last_train = out.index[train_ix][-1]
        train_ix = train_ix & (out.index < last_train)

    proba = np.full(len(out), 0.5, dtype=float)
    fitted = _fit_up_proba(feats.loc[train_ix], y_raw.loc[train_ix], min_train_rows, random_state)
    if fitted is None:
        logger.warning("ML train skipped")
        out["ML_Up_Proba"] = proba
        return out
    clf, up_col = fitted
    apply_mask = (out.index >= eval_ts).to_numpy()
    if apply_mask.any():
        proba[apply_mask] = clf.predict_proba(feats.loc[apply_mask].values)[:, up_col]
    out["ML_Up_Proba"] = proba
    return out
