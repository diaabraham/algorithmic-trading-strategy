"""Look-ahead and protocol tests for the study 3 ML pipeline (offline, synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from src.ml_research.data import SECTORS, asof
from src.ml_research.features import build_features, forward_label
from src.ml_research.models import purged_kfold
from src.ml_research.sizing import decided_weights, week_end_mask
from src.ml_research.walkforward import walk_forward


def _market(n=2200, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)

    def bars(drift=0.0003, vol=0.01):
        c = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
        o = c * np.exp(rng.normal(0, 0.002, n))
        return pd.DataFrame({"Open": o, "High": np.maximum(o, c), "Low": np.minimum(o, c), "Close": c, "Volume": 1.0}, index=idx)

    yahoo = {"^VIX": bars(0, 0.05), "^IRX": bars(0, 0.01), **{s: bars() for s in SECTORS}}
    fred = {k: pd.Series(rng.normal(size=n).cumsum(), index=idx + pd.Timedelta(days=1)) for k in ("T10Y3M", "BAA10Y")}
    weekly = pd.date_range("1999-01-02", idx[-1], freq="W-SAT")
    fred["ICSA"] = pd.Series(300000 + rng.normal(0, 1000, len(weekly)), index=weekly + pd.Timedelta(days=6))
    return bars(), yahoo, fred


def _cut(obj, t):
    if isinstance(obj, dict):
        return {k: _cut(v, t) for k, v in obj.items()}
    return obj.loc[:t]


def test_features_use_only_past_data():
    asset, yahoo, fred = _market()
    full = build_features(asset, yahoo, fred)
    t = asset.index[1500]
    part = build_features(_cut(asset, t), _cut(yahoo, t), _cut(fred, t))
    pd.testing.assert_frame_equal(full.loc[:t], part, check_exact=False, atol=1e-10)


def test_asof_respects_publication_dates():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-10", "2020-01-17"]))  # dates already shifted to release
    idx = pd.to_datetime(["2020-01-09", "2020-01-10", "2020-01-16", "2020-01-17"])
    assert asof(s, idx).tolist()[1:] == [1.0, 1.0, 2.0] and np.isnan(asof(s, idx).iloc[0])


def test_label_starts_at_next_fill():
    idx = pd.bdate_range("2020-01-01", periods=30)
    px = pd.Series(np.arange(100.0, 130.0), index=idx)
    lab = forward_label(px, pd.Series(0.0, index=idx), 5)
    assert lab.iloc[0] == pytest.approx(np.log(px.iloc[6] / px.iloc[1]))
    assert lab.iloc[-6:].isna().all()


def test_purged_kfold_removes_overlapping_labels():
    for train, test in purged_kfold(1000, horizon=21, folds=5):
        a, b = test.min(), test.max()
        assert not ((train >= a - 22) & (train <= b + 21)).any()


@pytest.mark.parametrize("kind", ["ridge", "lgbm"])
def test_walk_forward_predictions_do_not_change_when_future_is_removed(kind):
    asset, yahoo, fred = _market()
    feats = build_features(asset, yahoo, fred)
    lab = forward_label(asset["Open"], pd.Series(0.0, index=asset.index), 21)
    full = walk_forward(feats, lab, kind, 21, 400)
    t = asset.index[1800]
    f_cut = feats.loc[:t]
    # Labels that need prices after t are unknown at t.
    lab_cut = forward_label(asset["Open"].loc[:t], pd.Series(0.0, index=f_cut.index), 21)
    part = walk_forward(f_cut, lab_cut, kind, 21, 400)
    np.testing.assert_allclose(full["pred"].loc[:t].to_numpy(), part["pred"].to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)


def test_weights_only_change_on_week_ends_and_respect_cap():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=idx)
    pred = pd.DataFrame({"pred": rng.normal(0, 0.02, 400), "pred_sd": 0.02}, index=idx)
    for rule in ("mv", "tilt", "gate", "voltarget"):
        w = decided_weights(rule, pred, close)
        changes = w.diff().fillna(0) != 0
        assert (~changes | week_end_mask(idx)).all()
        assert w.max() <= 1.5 and w.min() >= 0
