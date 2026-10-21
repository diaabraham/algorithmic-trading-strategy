"""
End-to-end portfolio Monte Carlo without network: DataLoader.get_data is stubbed.
"""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.portfolio_sim import MonteCarloResult, run_portfolio_monte_carlo


def _synthetic_market_bars(n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    close = 100.0 + np.cumsum(rng.normal(0, 1.2, n))
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close * (1.0 + rng.uniform(0, 0.02, n)),
            "Low": close * (1.0 - rng.uniform(0, 0.02, n)),
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std() * np.sqrt(252)
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Vol_Percentile"] = df["Volatility"].expanding().rank(pct=True)
    df["Upper_Band"] = df["EMA_20"] + 2 * df["Volatility"]
    df["Lower_Band"] = df["EMA_20"] - 2 * df["Volatility"]
    return df.dropna()


def _patched_get_data(self):
    self.symbol = getattr(self, "requested_symbol", "MOCK").strip().upper()
    return _synthetic_market_bars()


def test_run_portfolio_monte_carlo_smoke(monkeypatch):
    monkeypatch.setattr(DataLoader, "get_data", _patched_get_data)
    out = run_portfolio_monte_carlo(
        symbols=["AAA", "BBB"],
        window_start="2025-03-01",
        window_end="2025-06-01",
        initial_cad=25_000.0,
        cad_usd=0.74,
        scenarios=520,
        seed=7,
        tune=False,
        use_ml=False,
    )
    assert isinstance(out, MonteCarloResult)
    assert out.scenarios == 520
    assert len(out.symbols_used) == 2
    assert out.initial_usd == pytest.approx(25_000.0 * 0.74)
    assert out.pnl_pct_p05 <= out.pnl_pct_p50 <= out.pnl_pct_p95


def test_failed_symbol_does_not_drop_its_capital(monkeypatch):
    def fake(symbol, *args, **kwargs):
        if symbol == "BAD":
            raise RuntimeError("no data")
        return [0.0]

    monkeypatch.setattr("src.portfolio_sim._trade_returns_for_symbol", fake)
    out = run_portfolio_monte_carlo(
        symbols=["AAA", "BAD"],
        window_start="2025-03-01",
        window_end="2025-06-01",
        initial_cad=100.0,
        cad_usd=1.0,
        scenarios=20,
        seed=1,
    )
    assert out.symbols_used == ["AAA"]
    assert out.pnl_pct_mean == pytest.approx(0.0)
    assert out.final_usd_p50 == pytest.approx(100.0)
