"""Offline tests for study 5: point-in-time universe, walk-forward masks and costs."""

import numpy as np
import pandas as pd
import pytest

from src.study5 import walkforward
from src.study5.evaluate import cost_of, long_only, turnover_of
from src.study5.panel import _rank_among, _usable


def test_usable_requires_publication_before_the_year():
    idx = pd.to_datetime(["1999-12-31", "2000-01-31", "2001-01-31"])
    u = _usable(pd.Series({"A": 2000, "IND_x": -1}), idx)
    assert u["A"].tolist() == [False, False, True]
    assert u["IND_x"].all()


def test_rank_only_against_usable_assets():
    idx = pd.to_datetime(["2001-01-31"])
    vals = pd.DataFrame({f"a{i}": [float(i)] for i in range(6)}, index=idx)
    usable = pd.DataFrame({f"a{i}": [i < 5] for i in range(6)}, index=idx)
    r = _rank_among(vals, usable)
    assert r["a4"].iloc[0] == 1.0 and r["a5"].iloc[0] == 1.0  # a5 (unusable) is ranked against usable only
    assert r["a0"].iloc[0] == pytest.approx(0.2)


def test_walk_forward_never_trains_on_future_targets(monkeypatch):
    seen = []

    def fake(Xtr, ytr, Xte, n_jobs=8):
        seen.append(Xtr[:, 0].max())
        return np.zeros(len(Xte))

    monkeypatch.setattr(walkforward, "fit_predict_lgbm", fake)
    n = 3000
    month = np.repeat(np.arange(1000, 1100), 30)
    X = month.reshape(-1, 1).astype(np.float32)
    arr = {"X": X, "y": np.ones(n, np.float32), "month": month.astype(np.int32), "pub_year": np.full(n, -1, np.int32)}
    walkforward.run(arr, 1050, 1060, "lgbm", log_every=0)
    # Refit at decision month t sees features from months <= t - 1 only.
    assert seen == [float(t - 1) for t in range(1050, 1061)]


def test_unpublished_assets_are_not_predicted():
    month = np.repeat(np.arange(1300, 1340), 50)
    pub = np.tile(np.r_[np.full(25, -1), np.full(25, 2005)], 40).astype(np.int32)
    arr = {"X": np.random.default_rng(0).normal(size=(len(month), 3)).astype(np.float32),
           "y": np.random.default_rng(1).normal(size=len(month)).astype(np.float32), "month": month.astype(np.int32), "pub_year": pub}
    pred, _ = walkforward.run(arr, 1330, 1339, "lgbm", log_every=0)
    yr = 1900 + month // 12
    assert np.isnan(pred[(pub >= yr)]).all()


def test_costs_and_turnover_mapping():
    meta = pd.DataFrame({"Portfolio Period": [1, 1, 12], "Cat.Data": ["Price", "Accounting", "Accounting"]}, index=["p", "a", "y"])
    assert turnover_of(meta, "p") == 3.0 and turnover_of(meta, "a") == 1.5 and turnover_of(meta, "y") == 0.5
    assert turnover_of(meta, "IND_Food") == 0.1 and cost_of("IND_Food") == 0.001 and cost_of("p") == 0.002


def test_long_only_picks_top_scores_and_charges_costs():
    dates = pd.to_datetime(["2000-01-31"] * 12)
    panel = pd.DataFrame({"month": 1200, "date": dates, "asset": [f"IND_{i}" for i in range(12)], "next_ret": np.arange(12) / 100.0})
    score = pd.Series(np.arange(12.0))
    r, held = long_only(panel, score, pd.DataFrame(), [1200])
    gross = np.arange(2, 12).mean() / 100
    expected = gross - (2 * 0.10 * 0.001 / 12) - 1.0 * 0.001
    assert r.iloc[0] == pytest.approx(expected)
    assert r.index[0] == pd.Timestamp("2000-02-29")
