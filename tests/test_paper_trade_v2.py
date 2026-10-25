import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

from src.study5 import evaluate as E
from src.study5.live import TOP_K, momentum_2_12, month_cost, picks_at

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _script():
    spec = importlib.util.spec_from_file_location("ptu", os.path.join(ROOT, "scripts", "paper_trade_update.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _returns(start="2023-01-31", periods=30, n=14, seed=0):
    idx = pd.date_range(start, periods=periods, freq="ME")
    rng = np.random.default_rng(seed)
    cols = [f"S{i:02d}" for i in range(n - 2)] + ["IND_A", "IND_B"]
    return pd.DataFrame(rng.normal(0.01, 0.05, (periods, n)), index=idx, columns=cols)


def test_picks_need_eleven_months_and_skip_last_month():
    R = _returns()
    pub = pd.Series(1990, index=R.columns)
    assert picks_at(R, pub, R.index[10]) is None  # only 10 prior months
    t = R.index[20]
    mom = np.expm1(np.log1p(R.loc[R.index[9]:R.index[19]]).sum())
    assert sorted(picks_at(R, pub, t)) == sorted(mom.nlargest(TOP_K).index)


def test_unpublished_and_missing_assets_are_excluded():
    R = _returns()
    t = R.index[20]
    R.loc[R.index[9]:R.index[19], "S00"] = 0.5  # would be the top pick
    pub = pd.Series(1990, index=R.columns)
    assert "S00" in picks_at(R, pub, t)
    pub["S00"] = t.year  # published this year: not usable yet
    assert "S00" not in picks_at(R, pub, t)
    pub["S00"] = 1990
    R.loc[t, "S00"] = np.nan  # no return at the decision month
    assert "S00" not in picks_at(R, pub, t)


def test_month_cost_matches_study5():
    meta = pd.DataFrame({"Portfolio Period": [1, 12], "Cat.Data": ["Price", "Accounting"]}, index=["S00", "S01"])
    new = ["S00", "S01"] + [f"IND_{i}" for i in range(8)]
    prev = ["S00"] + [f"IND_{i}" for i in range(9)]
    turnover = {a: E.turnover_of(meta, a) for a in set(new) | set(prev)}
    cost = {a: E.cost_of(a) for a in turnover}
    w = 1 / TOP_K
    within = w * 2 * (3.0 * 0.002 + 0.5 * 0.002 + 8 * 0.1 * 0.001) / 12
    switch = w * 0.002 + w * 0.001
    assert month_cost(new, prev, turnover, cost) == pytest.approx(within + switch)
    assert month_cost(new, [], turnover, cost) == pytest.approx(within + 0.002 * 2 * w + 0.001 * 8 * w)


def test_picks_log_is_append_only(tmp_path):
    m = _script()
    R = _returns(start="2023-06-30", periods=24)
    pub = pd.Series(1990, index=R.columns)
    info = {"cz_release": "r1", "cz_data_end": "2024-12-31", "industries_data_end": "2025-03-31", "fetched_on": "2026-10-25"}
    path = str(tmp_path / "picks.csv")
    log = m.update_picks(R, pub, {}, info, picks_path=path)
    assert list(log["decision_month_end"]) == ["2024-12-31"]  # limited by the CZ data end
    first = log.iloc[0]["picks"]
    R2 = R.copy()
    R2.loc["2024-02-29":"2024-12-31"] = -R2.loc["2024-02-29":"2024-12-31"]  # a revised release
    info2 = dict(info, cz_release="r2", cz_data_end="2025-03-31")
    log2 = m.update_picks(R2, pub, {}, info2, picks_path=path)
    assert log2.iloc[0]["picks"] == first and log2.iloc[0]["cz_release"] == "r1"
    assert list(log2["decision_month_end"]) == ["2024-12-31", "2025-01-31", "2025-02-28", "2025-03-31"]
    assert set(log2["phase"]) == {"pre_live_not_judged"}


def test_returns_stop_where_data_ends_and_entry_cost_is_charged():
    m = _script()
    R = _returns(start="2023-06-30", periods=21)  # through 2025-02
    picks = ";".join(R.columns[:TOP_K])
    log = pd.DataFrame({"holding_month_end": ["2025-01-31", "2025-02-28", "2025-03-31"],
                        "phase": ["pre_live_not_judged"] * 3, "picks": [picks] * 3})
    spy = pd.Series(0.01, index=pd.date_range("2025-01-31", periods=3, freq="ME"))
    turnover = {a: 0.1 for a in R.columns}
    cost = {a: 0.001 for a in R.columns}
    out = m.v2_returns(log, R, turnover, cost, spy)
    assert len(out) == 2
    assert out["cost"].iloc[0] > out["cost"].iloc[1]
    assert out["spy"].iloc[0] == pytest.approx(0.01 - 0.0002) and out["spy"].iloc[1] == pytest.approx(0.01)
    assert out["net"].iloc[1] == pytest.approx(R.loc["2025-02-28", R.columns[:TOP_K]].mean() - out["cost"].iloc[1])


def test_window_cagr_and_checkpoint_rule():
    m = _script()
    df = pd.DataFrame({"holding_month_end": [str(d.date()) for d in pd.date_range("2026-11-30", periods=12, freq="ME")],
                       "net": 0.01, "net_2x_costs": 0.009, "spy": 0.005})
    w = m._window(df, 12)
    assert w["strategy_cagr"] == pytest.approx(1.01 ** 12 - 1)
    assert w["strategy_cagr"] > w["spy_cagr"]


def test_momentum_matches_study5_panel_definition():
    R = _returns()
    expected = np.expm1(np.log1p(R).shift(1).rolling(11, min_periods=11).sum())
    pd.testing.assert_frame_equal(momentum_2_12(R), expected)
