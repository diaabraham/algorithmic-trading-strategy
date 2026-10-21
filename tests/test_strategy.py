import numpy as np
import pandas as pd

from src.model_runner import optimize_for_period
from src.strategy import StrategyParams, VolatilityRegimeStrategy


def _price_frame(index, close, ema):
    close = np.asarray(close, dtype=float)
    ema = np.asarray(ema, dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "EMA_20": ema,
            "Volatility": np.full(len(index), 0.4),
            "Vol_Percentile": np.full(len(index), 0.95),
        },
        index=index,
    )


class _ForcedEntry(VolatilityRegimeStrategy):
    def __init__(self, data, entry_i, momentum, params):
        super().__init__(data, params=params)
        self._entry_i = entry_i
        self._momentum = momentum

    def generate_signals(self):
        signals = super().generate_signals()
        for col in ("long_entry", "short_entry", "long_momentum", "short_momentum"):
            signals[col] = False
        signals.loc[signals.index[self._entry_i], "long_entry"] = True
        if self._momentum:
            signals.loc[signals.index[self._entry_i], "long_momentum"] = True
        return signals


def test_generate_signals_does_not_backfill_future_values():
    idx = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    close = np.full(len(idx), 100.0)
    df = _price_frame(idx, close, close)
    df["Volatility"] = np.nan
    df.loc[df.index[-1], "Volatility"] = 9.9
    strategy = VolatilityRegimeStrategy(df)
    strategy.generate_signals()
    assert pd.isna(strategy.data["Volatility"].iloc[0])


def test_momentum_entry_is_not_closed_by_mean_reversion_exit():
    idx = pd.date_range("2024-01-01", periods=80, freq="D", tz="UTC")
    close = np.linspace(50, 80, len(idx))
    close[40:] = close[40] + 15
    entry_i = 45
    strategy = _ForcedEntry(
        _price_frame(idx, close, close - 5),
        entry_i=entry_i,
        momentum=True,
        params=StrategyParams(max_hold_days=5),
    )
    strategy.run_backtest()
    trade = strategy.trades[0]
    assert trade.holding_period == 5
    assert trade.exit_date == idx[entry_i + 5]


def test_mean_reversion_exit_still_closes_extension():
    idx = pd.date_range("2024-01-01", periods=80, freq="D", tz="UTC")
    close = np.linspace(50, 80, len(idx))
    close[40:] = close[40] + 15
    entry_i = 45
    strategy = _ForcedEntry(
        _price_frame(idx, close, close - 5),
        entry_i=entry_i,
        momentum=False,
        params=StrategyParams(max_hold_days=5),
    )
    strategy.run_backtest()
    trade = strategy.trades[0]
    assert trade.holding_period == 1
    assert trade.exit_date == idx[entry_i + 1]


def test_time_stop_counts_bars_not_calendar_gaps():
    entry_i = 30
    dates = list(pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC"))
    gap = pd.Timedelta(days=20)
    dates = [d + gap if i > entry_i else d for i, d in enumerate(dates)]
    idx = pd.DatetimeIndex(dates)
    close = np.full(len(idx), 100.0)
    strategy = _ForcedEntry(
        _price_frame(idx, close, close),
        entry_i=entry_i,
        momentum=False,
        params=StrategyParams(max_hold_days=5),
    )
    strategy.run_backtest()
    trade = strategy.trades[0]
    assert trade.holding_period == 5
    assert trade.exit_date == idx[entry_i + 5]


def test_cagr_is_finite_when_the_sample_is_shorter_than_a_day():
    idx = pd.date_range("2024-06-03 14:00", periods=30, freq="min", tz="UTC")
    close = np.linspace(100, 110, len(idx))
    strategy = _ForcedEntry(
        _price_frame(idx, close, close),
        entry_i=22,
        momentum=False,
        params=StrategyParams(max_hold_days=3),
    )
    metrics = strategy.run_backtest()
    assert np.isfinite(metrics["cagr"])
    assert np.isfinite(metrics["sharpe_ratio"])
    assert np.isfinite(metrics["max_drawdown"])


def test_optimize_for_period_scores_the_eval_window(monkeypatch):
    captured = []

    def fake_run_on_data(**kwargs):
        captured.append(kwargs.get("eval_start_date"))
        return {
            "win_rate": 0.0,
            "avg_return": 0.0,
            "sharpe_ratio": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
        }

    monkeypatch.setattr("src.model_runner._run_on_data", fake_run_on_data)
    optimize_for_period(
        "SPY",
        "2025-03-01",
        "2025-06-01",
        base_data=pd.DataFrame({"Close": [1.0]}),
        fixed_hold_days=2,
    )
    assert captured
    assert set(captured) == {"2025-03-01"}
