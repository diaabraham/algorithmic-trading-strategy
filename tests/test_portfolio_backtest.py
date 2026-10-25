"""Offline tests for the multi-asset simulator and the pre-registered candidates."""

import numpy as np
import pandas as pd
import pytest

from src import candidates as C
from src.portfolio_backtest import deflated_sharpe, simulate_weights, stats, vol_matched_returns


def _synthetic(n=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n)
    tickers = C.UNIVERSE
    rets = rng.normal(0.0003, 0.012, size=(n, len(tickers)))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=tickers)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, size=close.shape))
    rf = pd.Series(0.02 / 252, index=idx)
    return open_, close, rf


def test_simulator_fills_next_open_and_finances_leverage():
    idx = pd.bdate_range("2020-01-01", periods=4)
    o = pd.DataFrame({"SPY": [100.0, 100.0, 100.0, 100.0]}, index=idx)
    c = pd.DataFrame({"SPY": [100.0, 110.0, 110.0, 110.0]}, index=idx)
    w = pd.DataFrame({"SPY": [1.5, 1.5, 1.5, 1.5]}, index=idx)
    rf = pd.Series(0.0, index=idx)
    sim = simulate_weights(o, c, w, {"SPY": 0.0}, rf, borrow_spread=0.0)
    # Bought 1.5x at the bar-1 open (100); bar-1 close is 110 -> +15%.
    assert sim["nav"].iloc[1] == pytest.approx(1.15)
    sim2 = simulate_weights(o, c, w, {"SPY": 0.0}, rf, borrow_spread=0.252)
    assert sim2["nav"].iloc[-1] < sim["nav"].iloc[-1]  # financing charged on the 0.5x borrowed


def test_costs_reduce_nav_by_traded_notional():
    idx = pd.bdate_range("2020-01-01", periods=3)
    px = pd.DataFrame({"SPY": [100.0] * 3}, index=idx)
    w = pd.DataFrame({"SPY": [1.0, 0.0, 0.0]}, index=idx)
    sim = simulate_weights(px, px, w, {"SPY": 10.0}, pd.Series(0.0, index=idx))
    assert sim["nav"].iloc[-1] == pytest.approx(1 - 0.001 - 0.001, rel=1e-6)


@pytest.mark.parametrize("name", [
    "A1_volmanaged_spy_21d", "B1_spy_sma10m", "B3_dual_momentum", "C1_multi_asset_tsmom",
    "C2_risk_parity", "C3_risk_parity_trend", "D1_qqq_spy_rotation", "D3_sector_momentum_filtered",
])
def test_candidates_have_no_look_ahead(name):
    _, close, rf = _synthetic()
    vr = {"off": pd.Series(1.0, index=close.index), "on": pd.Series(1.0, index=close.index)}
    full = C.build_candidates(close, rf, vr)[name]
    # Cut at a real month end (as the IS phase does at 2019-12-31): the last
    # bar of a truncated series is otherwise treated as a decision day.
    cut = close.index[C.month_end_mask(close.index)][28]
    part = C.build_candidates(close.loc[:cut], rf.loc[:cut], vr)[name]
    pd.testing.assert_frame_equal(full.loc[:cut], part, check_exact=False, atol=1e-12)


def test_gross_leverage_caps_hold_at_decision():
    _, close, rf = _synthetic()
    vr = {"off": pd.Series(1.0, index=close.index), "on": pd.Series(1.0, index=close.index)}
    cands = C.build_candidates(close, rf, vr)
    assert cands["A1_volmanaged_spy_21d"].abs().sum(axis=1).max() <= 1.5 + 1e-12
    assert cands["C2_risk_parity"].abs().sum(axis=1).max() <= 2.0 + 1e-12
    scaled = C.scale_weights(cands["B1_spy_sma10m"], 3.0, max_gross=2.0)
    assert scaled.abs().sum(axis=1).max() <= 2.0 + 1e-12


def test_vol_matched_and_deflated_sharpe():
    o, c, rf = _synthetic()
    w = pd.DataFrame({"SPY": 1.0}, index=c.index)
    sim = simulate_weights(o[["SPY"]], c[["SPY"]], w, {"SPY": 2.0}, rf)
    vm = vol_matched_returns(sim.iloc[1:], rf, 2.0)
    assert vm["ret"].std() == pytest.approx(2.0 * sim["ret"].iloc[1:].std(), rel=1e-2)
    s = stats(sim.iloc[1:], rf)
    assert -1 < s["max_drawdown"] <= 0
    d = deflated_sharpe(sim["ret"], rf, [0.1, 0.3, 0.5, 0.2, 0.4])
    assert 0.0 <= d["deflated_sharpe_ratio"] <= d["probabilistic_sharpe_vs_zero"] <= 1.0
