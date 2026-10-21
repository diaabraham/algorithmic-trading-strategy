"""Live daily bars through the same provider chain as the backtest."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytestmark = pytest.mark.integration

_EVAL = "2025-10-21"
_END = "2026-10-21"


def test_spy_live_bars_stay_invested_in_the_uptrend():
    from src.data_loader import DataLoader
    from src.ml_signal import attach_recursive_ml_up_proba, training_history_start
    from src.strategy import StrategyParams, VolatilityRegimeStrategy

    loader = DataLoader("SPY", training_history_start(_EVAL), _END)
    data = loader.get_data()
    assert loader.data_source in {"massive", "databento", "polygon", "yfinance", "stooq"}
    assert loader.symbol == "SPY"
    window = data.loc[data.index >= pd.Timestamp(_EVAL, tz="UTC")]
    assert len(window) >= 200
    assert float(window["Close"].min()) > 0

    data = attach_recursive_ml_up_proba(data, eval_start=_EVAL)
    strategy = VolatilityRegimeStrategy(
        data,
        params=StrategyParams(max_hold_days=5, ml_up_min_long=0.55, ml_up_max_short=0.45),
        eval_start_date=_EVAL,
    )
    metrics = strategy.run_backtest()
    position, _asset = strategy._daily_exposure()
    assert metrics["total_trades"] >= 1
    assert float(position.abs().mean()) > 0.5
    assert metrics["brinson_benchmark_return"] > 0


def test_wti_live_bars_resolve_to_uso():
    from src.data_loader import DataLoader

    loader = DataLoader("WTI", "2026-07-01", _END)
    data = loader.get_data()
    assert loader.symbol == "USO"
    assert loader.data_source in {"massive", "databento", "polygon", "yfinance", "stooq"}
    october = data.loc[data.index >= pd.Timestamp("2026-10-01", tz="UTC")]
    assert len(october) >= 10
    assert float(october["Close"].iloc[-1]) > 0
