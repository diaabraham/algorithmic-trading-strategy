#!/usr/bin/env python3
"""
Backtest on live daily bars.

Uses the same provider order as the app: Massive, Databento, and Polygon when
those keys are set, then Yahoo Finance, then Stooq. A missing Polygon key does
not block the run.

Usage:
  python scripts/verify_real_market.py
  python scripts/verify_real_market.py --symbol WTI --start-date 2025-10-21 --end-date 2026-10-21
  python scripts/verify_real_market.py --portfolio-mc --mc-symbols SPY,NVDA --capital-cad 25000
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _bootstrap() -> None:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")


def main() -> None:
    _bootstrap()

    end_default = date.today().isoformat()
    start_default = (date.today() - timedelta(days=365)).isoformat()
    parser = argparse.ArgumentParser(description="Backtest on live daily bars")
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--start-date", type=str, default=start_default)
    parser.add_argument("--end-date", type=str, default=end_default)
    parser.add_argument("--holding-days", type=int, default=5)
    parser.add_argument("--portfolio-mc", action="store_true")
    parser.add_argument("--mc-symbols", type=str, default="SPY,NVDA")
    parser.add_argument("--mc-window-start", type=str, default=start_default)
    parser.add_argument("--as-of", type=str, default=end_default)
    parser.add_argument("--capital-cad", type=float, default=25_000.0)
    parser.add_argument("--mc-scenarios", type=int, default=500)
    args = parser.parse_args()

    from src.config import get_settings
    from src.data_loader import DataLoader
    from src.ml_signal import attach_recursive_ml_up_proba, training_history_start
    from src.portfolio_sim import run_portfolio_monte_carlo
    from src.strategy import StrategyParams, VolatilityRegimeStrategy

    load_start = training_history_start(args.start_date)
    print("=== Live daily bars ===")
    print(f"  {args.symbol}  {args.start_date} .. {args.end_date}")
    loader = DataLoader(args.symbol, load_start, args.end_date)
    data = loader.get_data()
    if data.empty:
        print("ERROR: no bars returned.")
        sys.exit(2)
    data = attach_recursive_ml_up_proba(data, eval_start=args.start_date)
    print(f"  Source: {loader.data_source}  ticker: {loader.symbol}  rows={len(data)}")
    params = StrategyParams(
        max_hold_days=args.holding_days,
        ml_up_min_long=0.55,
        ml_up_max_short=0.45,
    )
    strat = VolatilityRegimeStrategy(
        data,
        100_000.0,
        params=params,
        eval_start_date=args.start_date,
    )
    m = strat.run_backtest()
    print(f"  Total trades:     {m['total_trades']}")
    print(f"  Win rate:         {m['win_rate']:.2%}")
    print(f"  Avg return/trade: {m['avg_return']:.2%}")
    print(f"  Sharpe:           {m['sharpe_ratio']:.2f}")
    print(f"  Max drawdown:     {m['max_drawdown']:.2%}")
    print(f"  CAGR:             {m['cagr']:.2%}")
    print(f"  Strategy return:  {m['brinson_portfolio_return']:+.2%}")
    print(f"  Buy-and-hold:     {m['brinson_benchmark_return']:+.2%}")
    print(f"  Allocation:       {m['brinson_allocation']:+.2%}")
    print(f"  Selection:        {m['brinson_selection']:+.2%}")

    if args.portfolio_mc:
        settings = get_settings()
        syms = [s.strip().upper() for s in args.mc_symbols.split(",") if s.strip()]
        print()
        print("=== Live daily bars: portfolio Monte Carlo ===")
        print(f"  Symbols: {syms}  window {args.mc_window_start} .. {args.as_of}")
        r = run_portfolio_monte_carlo(
            symbols=syms,
            window_start=args.mc_window_start,
            window_end=args.as_of,
            initial_cad=args.capital_cad,
            cad_usd=settings.cad_usd_fx,
            scenarios=max(500, args.mc_scenarios),
            tune=False,
            use_ml=False,
        )
        print(f"  Symbols used:     {r.symbols_used}")
        print(f"  Scenarios:        {r.scenarios}")
        print(f"  Initial USD:      ${r.initial_usd:,.2f}")
        print(f"  PnL % mean/p05/p50/p95:  {r.pnl_pct_mean:+.2%} / {r.pnl_pct_p05:+.2%} / {r.pnl_pct_p50:+.2%} / {r.pnl_pct_p95:+.2%}")
        print(f"  Final USD p05/p50/p95:   ${r.final_usd_p05:,.2f} / ${r.final_usd_p50:,.2f} / ${r.final_usd_p95:,.2f}")


if __name__ == "__main__":
    main()
