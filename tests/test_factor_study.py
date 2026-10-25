"""Offline tests for the study 4 parser, cost drag and Sharpe statistics."""

import numpy as np
import pandas as pd
import pytest

from src.factor_study import CANDIDATES, cost_drag, monthly_stats, net_returns, parse_monthly, sharpe_stats

SAMPLE = """This file was created using the CRSP database.

  Average Value Weighted Returns -- Monthly
,SMALL LoPRIOR,BIG HiPRIOR
192701,   1.00,   2.00
192702, -99.99,  -1.00

  Average Equal Weighted Returns -- Monthly
,SMALL LoPRIOR,BIG HiPRIOR
192701,   9.00,   9.00
"""


def test_parse_monthly_reads_the_requested_section():
    df = parse_monthly(SAMPLE, "Value Weighted Returns -- Monthly")
    assert list(df.columns) == ["SMALL LoPRIOR", "BIG HiPRIOR"]
    assert df.index[0] == pd.Timestamp("1927-01-31")
    assert df["BIG HiPRIOR"].tolist() == [0.02, -0.01]
    assert np.isnan(df["SMALL LoPRIOR"].iloc[1])


def test_cost_drag_matches_protocol():
    assert cost_drag(2.0) == pytest.approx(0.0055)
    assert cost_drag(0.3) == pytest.approx(0.0021)
    assert cost_drag(1.0, 2.0) == pytest.approx(0.0070)


def test_net_returns_and_mix():
    idx = pd.date_range("2000-01-31", periods=3, freq="ME")
    gross = pd.DataFrame({k: 0.01 for k in CANDIDATES}, index=idx)
    net = net_returns(gross)
    assert net["F1_big_momentum"].iloc[0] == pytest.approx(0.01 - 0.0055 / 12)
    assert net["F5_mix_mom_value_prof"].iloc[0] == pytest.approx(0.01 - 0.0035 / 12)


def test_stats_on_known_series():
    idx = pd.date_range("2000-01-31", periods=24, freq="ME")
    r = pd.Series(0.01, index=idx)
    r.iloc[5] = -0.10
    s = monthly_stats(r, pd.Series(0.0, index=idx))
    assert s["max_drawdown"] == pytest.approx(-0.10)
    assert s["cagr"] == pytest.approx((1.01 ** 23 * 0.9) ** 0.5 - 1)
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0.005, 0.02, 400))
    out = sharpe_stats(a, [0.1, 0.5, 0.9, 0.3, 0.2], 50)
    assert 0 <= out["deflated_sharpe_ratio"] <= out["psr_vs_zero"] <= 1
