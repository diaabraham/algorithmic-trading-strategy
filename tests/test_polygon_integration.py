"""
Real Polygon.io daily data — skipped automatically when POLYGON_API_KEY is unset.

Run locally:
  export POLYGON_API_KEY=...   # or use repo-root .env
  pytest tests/test_polygon_integration.py -v -s

This is the test suite’s **market** validation; unit tests with mocks are not substitutes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("POLYGON_API_KEY", "").strip(),
        reason="POLYGON_API_KEY missing — set in .env to run real-market integration tests",
    ),
]


def test_polygon_spy_backtest_non_empty():
    from src.data_loader import DataLoader
    from src.strategy import StrategyParams, VolatilityRegimeStrategy

    loader = DataLoader("SPY", "2024-01-01", "2024-09-30")
    data = loader.get_data()
    assert not data.empty
    assert "Close" in data.columns

    strat = VolatilityRegimeStrategy(data, 100_000.0, params=StrategyParams(max_hold_days=5))
    m = strat.run_backtest()
    assert "total_trades" in m and "win_rate" in m
    assert m["total_trades"] >= 0


def test_polygon_wti_resolves_to_uso():
    from src.data_loader import DataLoader

    loader = DataLoader("WTI", "2024-01-01", "2024-06-30")
    data = loader.get_data()
    assert not data.empty
    assert loader.symbol == "USO"
