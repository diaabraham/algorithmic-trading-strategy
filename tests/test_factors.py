import numpy as np
import pandas as pd

from src.black_scholes import black_scholes_call, black_scholes_up_proba, expected_move
from src.brinson import brinson_fachler, brinson_single_asset
from src.ml_signal import attach_recursive_ml_up_proba


def test_black_scholes_up_proba_orders_moneyness():
    deep_itm = float(black_scholes_up_proba(120, 100, 0.2, 21))
    deep_otm = float(black_scholes_up_proba(80, 100, 0.2, 21))
    assert 0.0 < deep_otm < deep_itm < 1.0
    call_low_vol = float(black_scholes_call(100, 100, 0.1, 21))
    call_high_vol = float(black_scholes_call(100, 100, 0.4, 21))
    assert 0.0 < call_low_vol < call_high_vol
    move = float(expected_move(100, 0.2, 252))
    assert move == 20.0


def test_brinson_single_asset_splits_flat_and_short():
    idx = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    position = pd.Series([1.0, 0.0, -1.0], index=idx)
    asset = pd.Series([0.01, 0.02, 0.03], index=idx)
    factors = brinson_single_asset(position, asset)
    assert factors["allocation"] == -0.02
    assert factors["selection"] == -0.06
    assert abs(factors["interaction"]) < 1e-12
    assert abs(factors["allocation"] + factors["selection"] + factors["interaction"] - factors["arithmetic_excess"]) < 1e-12


def test_brinson_fachler_book():
    factors = brinson_fachler(
        portfolio_weights=[0.5, 0.5],
        benchmark_weights=[0.5, 0.5],
        portfolio_returns=[0.10, -0.04],
        benchmark_returns=[0.02, 0.06],
    )
    assert factors["allocation"] == 0.0
    assert abs(factors["selection"] - (0.5 * 0.08 + 0.5 * -0.10)) < 1e-12
    assert factors["interaction"] == 0.0


def test_recursive_ml_block_ignores_later_prices():
    idx = pd.date_range("2025-01-01", periods=160, freq="D", tz="UTC")
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 1, len(idx)))
    df = pd.DataFrame({"Close": close, "Volatility": np.full(len(idx), 0.25)}, index=idx)
    base = attach_recursive_ml_up_proba(df, eval_start="2025-03-01", refit_every=40, min_train_rows=30)
    shocked = df.copy()
    shocked.iloc[-5:, shocked.columns.get_loc("Close")] = shocked["Close"].iloc[-5:] * 3
    alt = attach_recursive_ml_up_proba(shocked, eval_start="2025-03-01", refit_every=40, min_train_rows=30)
    first_block = (base.index >= pd.Timestamp("2025-03-01", tz="UTC")) & (
        base.index < pd.Timestamp("2025-04-10", tz="UTC")
    )
    assert base.loc[first_block, "ML_Up_Proba"].equals(alt.loc[first_block, "ML_Up_Proba"])
    assert base["ML_Up_Proba"].between(0.0, 1.0).all()
