from dataclasses import asdict
from datetime import date, timedelta, datetime
from typing import Dict, List

import pandas as pd

from src.data_loader import DataLoader
from src.sentiment import combined_sentiment
from src.strategy import StrategyParams, VolatilityRegimeStrategy


def _run_on_data(
    symbol: str,
    data: pd.DataFrame,
    params: StrategyParams,
    eval_start_date: str | None = None,
) -> Dict:
    strategy = VolatilityRegimeStrategy(data.copy(), params=params, eval_start_date=eval_start_date)
    metrics = strategy.run_backtest()
    return metrics


def _load_data_with_sentiment(
    symbol: str,
    start_date: str,
    end_date: str,
    alpha_vantage_api_key: str | None,
    use_twitter: bool,
) -> pd.DataFrame:
    warmup_start = (datetime.fromisoformat(start_date) - timedelta(days=60)).date().isoformat()
    loader = DataLoader(symbol, warmup_start, end_date)
    data = loader.get_data().copy()
    if alpha_vantage_api_key or use_twitter:
        score = combined_sentiment(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            alpha_vantage_api_key=alpha_vantage_api_key,
            twitter_query=f"${symbol}" if use_twitter else None,
        )
        data["News_Sentiment"] = score
    return data


def _run_once(
    symbol: str,
    start_date: str,
    end_date: str,
    data: pd.DataFrame,
    params: StrategyParams,
) -> Dict:
    metrics = _run_on_data(symbol=symbol, data=data, params=params, eval_start_date=start_date)
    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "params": asdict(params),
        **metrics,
    }


def optimize_for_period(
    symbol: str,
    start_date: str,
    end_date: str,
    base_data: pd.DataFrame | None = None,
    alpha_vantage_api_key: str | None = None,
    use_twitter: bool = False,
    fixed_hold_days: int | None = None,
) -> StrategyParams:
    if base_data is None:
        base_data = _load_data_with_sentiment(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            alpha_vantage_api_key=alpha_vantage_api_key,
            use_twitter=use_twitter,
        )

    grid: List[StrategyParams] = []
    hold_values = (fixed_hold_days,) if fixed_hold_days is not None else (2, 5, 8)
    for hold in hold_values:
        for mr_z in (0.9, 1.1, 1.3, 1.5, 1.7):
            for vol in (0.65, 0.70, 0.80, 0.90, 0.95):
                for atr_stop, atr_take in ((1.0, 1.8), (1.25, 2.0), (1.5, 2.5)):
                    grid.append(
                        StrategyParams(
                            vol_threshold=vol,
                            mr_z_threshold=mr_z,
                            max_hold_days=hold,
                            atr_stop_mult=atr_stop,
                            atr_take_mult=atr_take,
                            sentiment_threshold=0.4 if (alpha_vantage_api_key or use_twitter) else -1.0,
                        )
                    )

    best = None
    best_score = -1e9
    for params in grid:
        metrics = _run_on_data(symbol=symbol, data=base_data, params=params)
        # Reward win rate and avg return; penalize drawdown.
        score = (
            metrics["win_rate"] * 100
            + metrics["avg_return"] * 100
            + metrics["sharpe_ratio"] * 2
            + metrics["cagr"] * 20
            + metrics["max_drawdown"] * 20
        )
        if fixed_hold_days is None and metrics["total_trades"] >= 10:
            score += 10
        if score > best_score:
            best = params
            best_score = score
    return best or StrategyParams()


def run_rigorous_suite(
    symbols: List[str],
    as_of: date,
    alpha_vantage_api_key: str | None = None,
    use_twitter: bool = False,
) -> pd.DataFrame:
    week_start = as_of - timedelta(days=6)
    six_month_start = as_of - timedelta(days=182)

    rows = []
    for symbol in symbols:
        try:
            six_month_data = _load_data_with_sentiment(
                symbol=symbol,
                start_date=six_month_start.isoformat(),
                end_date=as_of.isoformat(),
                alpha_vantage_api_key=alpha_vantage_api_key,
                use_twitter=use_twitter,
            )
            week_data = _load_data_with_sentiment(
                symbol=symbol,
                start_date=week_start.isoformat(),
                end_date=as_of.isoformat(),
                alpha_vantage_api_key=alpha_vantage_api_key,
                use_twitter=use_twitter,
            )
        except Exception:
            continue

        tuned = optimize_for_period(
            symbol=symbol,
            start_date=six_month_start.isoformat(),
            end_date=as_of.isoformat(),
            base_data=six_month_data,
            alpha_vantage_api_key=alpha_vantage_api_key,
            use_twitter=use_twitter,
        )
        rows.append(
            _run_once(
                symbol=symbol,
                start_date=six_month_start.isoformat(),
                end_date=as_of.isoformat(),
                data=six_month_data,
                params=tuned,
            )
            | {"window": "6m", "style": "tuned"}
        )
        for hold, style in ((2, "short"), (8, "long")):
            tuned_week = optimize_for_period(
                symbol=symbol,
                start_date=week_start.isoformat(),
                end_date=as_of.isoformat(),
                base_data=week_data,
                alpha_vantage_api_key=alpha_vantage_api_key,
                use_twitter=use_twitter,
                fixed_hold_days=hold,
            )
            p = StrategyParams(**{**asdict(tuned_week), "max_hold_days": hold})
            rows.append(
                _run_once(
                    symbol=symbol,
                    start_date=week_start.isoformat(),
                    end_date=as_of.isoformat(),
                    data=week_data,
                    params=p,
                )
                | {"window": "1w", "style": style}
            )

    return pd.DataFrame(rows)
