import sys
import types

import numpy as np
import pandas as pd

from src.data_loader import DataLoader


def test_wti_alias_candidates_include_proxy():
    loader = DataLoader("WTI", "2024-01-01", "2024-02-01")
    candidates = loader._candidate_tickers()
    assert "USO" in candidates
    assert candidates[0] in {"CL1!", "CL=F", "USO"}


def test_regular_symbol_candidates_identity():
    loader = DataLoader("NVDA", "2024-01-01", "2024-02-01")
    assert loader._candidate_tickers() == ["NVDA"]


def _raw_bars(n: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100, 130, n), index=idx)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def _clear_data_keys(monkeypatch):
    for name in ("MASSIVE_API_KEY", "DATABENTO_API_KEY", "POLYGON_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_provider_order_skips_paid_sources_without_keys(monkeypatch):
    _clear_data_keys(monkeypatch)
    loader = DataLoader("SPY", "2024-01-01", "2024-02-01")
    assert loader._provider_order() == ["yfinance", "stooq"]


def test_provider_order_prefers_massive_then_databento_then_polygon(monkeypatch):
    _clear_data_keys(monkeypatch)
    monkeypatch.setenv("POLYGON_API_KEY", "polygon-key")
    monkeypatch.setenv("DATABENTO_API_KEY", "databento-key")
    monkeypatch.setenv("MASSIVE_API_KEY", "massive-key")
    loader = DataLoader("SPY", "2024-01-01", "2024-02-01")
    assert loader._provider_order() == ["massive", "databento", "polygon", "yfinance", "stooq"]


def test_polygon_failure_falls_back_to_yfinance(monkeypatch):
    _clear_data_keys(monkeypatch)
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    calls = []

    def fail_polygon(self):
        calls.append("polygon")
        raise RuntimeError("polygon down")

    def yahoo(self):
        calls.append("yfinance")
        return "SPY", _raw_bars()

    def fail_stooq(self):
        calls.append("stooq")
        raise AssertionError("Stooq should not run after Yahoo succeeds")

    monkeypatch.setattr(DataLoader, "_fetch_polygon", fail_polygon)
    monkeypatch.setattr(DataLoader, "_fetch_yfinance", yahoo)
    monkeypatch.setattr(DataLoader, "_fetch_stooq", fail_stooq)
    loader = DataLoader("SPY", "2024-01-01", "2024-02-01")
    frame = loader.fetch_data()
    assert calls == ["polygon", "yfinance"]
    assert loader.data_source == "yfinance"
    assert "Close" in frame.columns and len(frame) == 30


def test_yfinance_failure_falls_back_to_stooq(monkeypatch):
    _clear_data_keys(monkeypatch)

    def fail_yahoo(self):
        raise RuntimeError("yahoo down")

    def stooq(self):
        return "USO", _raw_bars(25)

    monkeypatch.setattr(DataLoader, "_fetch_yfinance", fail_yahoo)
    monkeypatch.setattr(DataLoader, "_fetch_stooq", stooq)
    loader = DataLoader("WTI", "2024-01-01", "2024-02-01")
    loader.fetch_data()
    assert loader.data_source == "stooq"
    assert loader.symbol == "USO"


def test_fetch_massive_reads_daily_aggs(monkeypatch):
    _clear_data_keys(monkeypatch)
    monkeypatch.setenv("MASSIVE_API_KEY", "massive-key")

    class Bar:
        def __init__(self, ts, close):
            self.timestamp = ts
            self.open = close
            self.high = close + 1
            self.low = close - 1
            self.close = close
            self.volume = 1_000

    class RESTClient:
        def __init__(self, api_key):
            assert api_key == "massive-key"

        def list_aggs(self, **kwargs):
            assert kwargs["ticker"] == "SPY"
            assert kwargs["timespan"] == "day"
            start = pd.Timestamp("2024-01-02", tz="UTC")
            return [Bar(int((start + pd.Timedelta(days=i)).timestamp() * 1000), 100 + i) for i in range(5)]

    fake = types.ModuleType("massive")
    fake.RESTClient = RESTClient
    monkeypatch.setitem(sys.modules, "massive", fake)
    loader = DataLoader("SPY", "2024-01-01", "2024-02-01", sources=("massive",))
    frame = loader.fetch_data()
    assert loader.data_source == "massive"
    assert len(frame) == 5
    assert frame["Close"].iloc[-1] == 104


def test_fetch_databento_normalizes_ohlcv(monkeypatch):
    _clear_data_keys(monkeypatch)
    monkeypatch.setenv("DATABENTO_API_KEY", "db-key")

    class Store:
        def to_df(self):
            return _raw_bars()

    class Timeseries:
        @staticmethod
        def get_range(**kwargs):
            assert kwargs["dataset"] == "EQUS.SUMMARY"
            assert kwargs["schema"] == "ohlcv-1d"
            assert kwargs["symbols"] == ["SPY"]
            assert kwargs["start"] == "2024-01-01"
            assert kwargs["end"] == "2024-02-02"
            return Store()

    class Historical:
        def __init__(self, key):
            assert key == "db-key"
            self.timeseries = Timeseries()

    fake = types.ModuleType("databento")
    fake.Historical = Historical
    monkeypatch.setitem(sys.modules, "databento", fake)
    loader = DataLoader("SPY", "2024-01-01", "2024-02-01", sources=("databento",))
    frame = loader.fetch_data()
    assert loader.data_source == "databento"
    assert loader.symbol == "SPY"
    assert len(frame) == 30


def test_volatility_bands_use_price_scale():
    idx = pd.date_range("2024-01-01", periods=40, freq="B", tz="UTC")
    close = pd.Series(np.linspace(100, 140, len(idx)), index=idx)
    loader = DataLoader("SPY", "2024-01-01", "2024-03-01")
    loader.data = pd.DataFrame(
        {
            "Close": close,
            "Volatility": 0.25,
            "EMA_20": close.ewm(span=20, adjust=False).mean(),
        },
        index=idx,
    )
    out = loader.preprocess_data()
    width = float((out["Upper_Band"] - out["Lower_Band"]).median())
    # Annualized return vol (~0.25) added to price would make the band about 1 point wide.
    assert width > 2.0
