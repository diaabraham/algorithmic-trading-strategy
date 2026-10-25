"""Next-open execution, costs and the IS/OOS study helpers (offline)."""

import numpy as np
import pandas as pd
import pytest

from src.execution import performance, simulate_exposure
from src.oos_study import sma_exposure, vol_managed_exposure
from src.strategy import StrategyParams, VolatilityRegimeStrategy
from src.trade import Trade


def _bars(opens, closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    return pd.Series(opens, index=idx, dtype=float), pd.Series(closes, index=idx, dtype=float)


def test_fill_happens_at_next_open_not_signal_close():
    o, c = _bars([100, 110, 121, 121], [100, 120, 121, 121])
    decided = pd.Series([1.0, 1.0, 1.0, 1.0], index=c.index)
    sim = simulate_exposure(o, c, decided, cost_bps=0.0)
    # Bought at the open of bar 1 (110), not the close of bar 0 (100).
    assert sim["nav"].iloc[0] == pytest.approx(1.0)
    assert sim["nav"].iloc[1] == pytest.approx(120 / 110)
    assert sim["nav"].iloc[-1] == pytest.approx(121 / 110)


def test_costs_are_charged_per_side_on_traded_notional():
    o, c = _bars([100] * 4, [100] * 4)
    decided = pd.Series([1.0, 0.0, 0.0, 0.0], index=c.index)
    sim = simulate_exposure(o, c, decided, cost_bps=10.0)
    # Buy at bar 1 open, sell at bar 2 open: two sides of ~10 bp each.
    assert sim["nav"].iloc[-1] == pytest.approx((1 - 0.001) * (1 - 0.001), rel=1e-4)
    assert (sim["turnover"] > 0).sum() == 2


def test_cash_earns_rf_and_leverage_pays_spread():
    o, c = _bars([100] * 253, [100] * 253)
    rf = pd.Series(0.04 / 252, index=c.index)
    flat = simulate_exposure(o, c, pd.Series(0.0, index=c.index), cost_bps=0.0, rf_daily=rf)
    assert flat["nav"].iloc[-1] == pytest.approx((1 + 0.04 / 252) ** 252)
    levered = simulate_exposure(o, c, pd.Series(1.5, index=c.index), cost_bps=0.0, rf_daily=rf, borrow_spread=0.01)
    assert levered["nav"].iloc[-1] < 1.0  # paying 5% on 0.5x borrowed with flat prices


def test_performance_metrics_on_known_series():
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    ret = pd.Series(0.0, index=idx)
    ret.iloc[10] = 0.10
    ret.iloc[20] = -0.20
    perf = performance(ret)
    assert perf["total_return"] == pytest.approx(1.1 * 0.8 - 1)
    assert perf["max_drawdown"] == pytest.approx(-0.20)
    assert perf["cagr"] == pytest.approx(perf["total_return"])


def test_trend_filter_only_changes_at_month_end():
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    close = pd.Series(np.linspace(100, 200, 400), index=idx)
    exp = sma_exposure(close, 150)
    changes = exp.index[exp.diff().fillna(0) != 0]
    for ts in changes:
        nxt = exp.index[exp.index.get_loc(ts) + 1] if exp.index.get_loc(ts) + 1 < len(exp) else None
        assert nxt is None or nxt.month != ts.month
    assert exp.iloc[-1] == 1.0


def test_vol_managed_exposure_is_capped():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(np.full(300, 0.0001))), index=idx)
    close.iloc[::2] *= 1.0000001  # near-zero vol -> leverage cap binds
    exp = vol_managed_exposure(close, target=0.16, max_leverage=1.5)
    assert exp.max() <= 1.5 + 1e-12


def test_strategy_trade_prices_use_next_open_and_net_costs():
    idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
    data = pd.DataFrame({"Open": [100, 101, 102, 103, 104], "Close": [100.5, 101.5, 102.5, 103.5, 104.5]}, index=idx)
    strat = VolatilityRegimeStrategy.__new__(VolatilityRegimeStrategy)
    strat.data = data
    strat.params = StrategyParams(cost_bps=5.0)
    assert strat._fill_price(0) == 101
    assert strat._fill_price(4) == 104.5  # no next bar: last close
    trade = Trade(entry_date=idx[0], exit_date=idx[2], entry_price=101.0, exit_price=103.0, position=1, pnl=0.0, holding_period=2)
    assert strat._net_trade_return(trade) == pytest.approx(2 / 101 - 0.001)
    strat.trades = [trade]
    decided = strat.decided_exposure()
    assert decided.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
