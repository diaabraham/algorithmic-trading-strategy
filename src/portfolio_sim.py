"""
Equal-weight portfolio Monte Carlo: bootstrap per-symbol trade returns, then aggregate USD notional.

Assumes symbols are traded independently with equal capital slices (approximation; no correlation model).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import List, Sequence

import numpy as np

from src.data_loader import DataLoader
from src.ml_signal import attach_ml_up_proba
from src.model_runner import optimize_for_period
from src.strategy import StrategyParams, VolatilityRegimeStrategy


MARCH_2026_HIGH_BETA_UNIVERSE: List[str] = [
    "WTI",
    "TSLA",
    "NVDA",
    "MSTR",
    "AMD",
    "COIN",
    "SPY",
    "QQQ",
    "GLD",
]


@dataclass
class MonteCarloResult:
    initial_usd: float
    scenarios: int
    symbols_used: List[str]
    pnl_pct_mean: float
    pnl_pct_p05: float
    pnl_pct_p50: float
    pnl_pct_p95: float
    final_usd_p05: float
    final_usd_p50: float
    final_usd_p95: float


def _warmup_start(window_start: str) -> str:
    d = date.fromisoformat(window_start)
    return (d - timedelta(days=75)).isoformat()


def _trade_returns_for_symbol(
    symbol: str,
    window_start: str,
    window_end: str,
    tune: bool,
    use_ml: bool,
    ml_long_min: float,
    ml_short_max: float,
    alpha_vantage_api_key: str | None,
    use_twitter: bool,
) -> List[float]:
    warm = _warmup_start(window_start)
    loader = DataLoader(symbol, warm, window_end)
    data = loader.get_data().copy()
    if alpha_vantage_api_key or use_twitter:
        from src.sentiment import combined_sentiment

        score = combined_sentiment(
            symbol=symbol,
            start_date=window_start,
            end_date=window_end,
            alpha_vantage_api_key=alpha_vantage_api_key,
            twitter_query=f"${symbol}" if use_twitter else None,
        )
        data["News_Sentiment"] = score

    if use_ml:
        data = attach_ml_up_proba(data, eval_start=window_start)

    if tune:
        params = optimize_for_period(
            symbol=symbol,
            start_date=window_start,
            end_date=window_end,
            base_data=data,
            alpha_vantage_api_key=alpha_vantage_api_key,
            use_twitter=use_twitter,
        )
    else:
        st = 0.4 if (alpha_vantage_api_key or use_twitter) else -1.0
        params = StrategyParams(max_hold_days=5, sentiment_threshold=st)

    ml_min = ml_long_min if use_ml else 0.0
    ml_max = ml_short_max if use_ml else 1.0
    params = StrategyParams(**{**asdict(params), "ml_up_min_long": ml_min, "ml_up_max_short": ml_max})

    strat = VolatilityRegimeStrategy(
        data,
        initial_capital=100_000.0,
        params=params,
        eval_start_date=window_start,
    )
    strat.run_backtest()
    return [float(t.pnl) for t in strat.trades]


def _bootstrap_slice(
    returns: Sequence[float],
    rng: np.random.Generator,
    capital: float,
) -> float:
    if not returns:
        return capital
    r = np.array(returns, dtype=float)
    idx = rng.integers(0, len(r), size=len(r), endpoint=False)
    mult = float(np.prod(1.0 + r[idx]))
    return capital * mult


def run_portfolio_monte_carlo(
    symbols: Sequence[str],
    window_start: str,
    window_end: str,
    initial_cad: float,
    cad_usd: float,
    scenarios: int = 500,
    seed: int = 42,
    tune: bool = False,
    use_ml: bool = False,
    ml_long_min: float = 0.55,
    ml_short_max: float = 0.45,
    alpha_vantage_api_key: str | None = None,
    use_twitter: bool = False,
) -> MonteCarloResult:
    if scenarios < 1:
        raise ValueError("scenarios must be >= 1")
    usd0 = float(initial_cad) * float(cad_usd)
    n = len(symbols)
    if n == 0:
        raise ValueError("symbols must be non-empty")

    per = usd0 / n
    by_sym: dict[str, List[float]] = {}
    used: List[str] = []
    for sym in symbols:
        s = sym.strip().upper()
        try:
            rets = _trade_returns_for_symbol(
                s,
                window_start,
                window_end,
                tune=tune,
                use_ml=use_ml,
                ml_long_min=ml_long_min if use_ml else 0.0,
                ml_short_max=ml_short_max if use_ml else 1.0,
                alpha_vantage_api_key=alpha_vantage_api_key,
                use_twitter=use_twitter,
            )
        except Exception:
            continue
        by_sym[s] = rets
        used.append(s)

    if not used:
        raise RuntimeError("No symbols produced trades; check data/API and date window.")

    rng = np.random.default_rng(seed)
    pnl_samples = np.empty(scenarios, dtype=float)
    for i in range(scenarios):
        final = 0.0
        for s in used:
            final += _bootstrap_slice(by_sym[s], rng, per)
        pnl_samples[i] = (final - usd0) / usd0

    return MonteCarloResult(
        initial_usd=usd0,
        scenarios=scenarios,
        symbols_used=used,
        pnl_pct_mean=float(np.mean(pnl_samples)),
        pnl_pct_p05=float(np.quantile(pnl_samples, 0.05)),
        pnl_pct_p50=float(np.quantile(pnl_samples, 0.50)),
        pnl_pct_p95=float(np.quantile(pnl_samples, 0.95)),
        final_usd_p05=float(usd0 * (1.0 + np.quantile(pnl_samples, 0.05))),
        final_usd_p50=float(usd0 * (1.0 + np.quantile(pnl_samples, 0.50))),
        final_usd_p95=float(usd0 * (1.0 + np.quantile(pnl_samples, 0.95))),
    )
