#!/usr/bin/env python3
"""
Backtest and optional portfolio Monte Carlo on **real** Polygon daily bars.

Requires POLYGON_API_KEY in the environment or a .env file in the repository root.
This is what produces actual market metrics (not pytest smoke tests, not synthetic OHLCV).

Usage:
  python scripts/verify_real_market.py
  python scripts/verify_real_market.py --symbol WTI --start-date 2024-01-01 --end-date 2025-06-01
  python scripts/verify_real_market.py --portfolio-mc --mc-symbols SPY,NVDA --capital-cad 25000
"""
from __future__ import annotations

import argparse
import os
import sys
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

    parser = argparse.ArgumentParser(description="Real Polygon backtest / portfolio MC")
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--end-date", type=str, default="2025-06-30")
    parser.add_argument("--holding-days", type=int, default=5)
    parser.add_argument("--portfolio-mc", action="store_true")
    parser.add_argument("--mc-symbols", type=str, default="SPY,NVDA")
    parser.add_argument("--mc-window-start", type=str, default="2025-01-01")
    parser.add_argument("--as-of", type=str, default="2025-06-30")
    parser.add_argument("--capital-cad", type=float, default=25_000.0)
    parser.add_argument("--mc-scenarios", type=int, default=500)
    args = parser.parse_args()

    from src.config import get_settings
    from src.data_loader import DataLoader
    from src.portfolio_sim import run_portfolio_monte_carlo
    from src.strategy import StrategyParams, VolatilityRegimeStrategy

    get_settings()

    print("=== Real market: single-symbol backtest (Polygon) ===")
    print(f"  {args.symbol}  {args.start_date} .. {args.end_date}")
    loader = DataLoader(args.symbol, args.start_date, args.end_date)
    data = loader.get_data()
    if data.empty:
        print("ERROR: no bars returned.")
        sys.exit(2)
    resolved = getattr(loader, "symbol", args.symbol)
    print(f"  Resolved ticker: {resolved}  rows={len(data)}")
    params = StrategyParams(max_hold_days=args.holding_days)
    strat = VolatilityRegimeStrategy(data, 100_000.0, params=params)
    m = strat.run_backtest()
    print(f"  Total trades:     {m['total_trades']}")
    print(f"  Win rate:         {m['win_rate']:.2%}")
    print(f"  Avg return/trade: {m['avg_return']:.2%}")
    print(f"  Sharpe:           {m['sharpe_ratio']:.2f}")
    print(f"  Max drawdown:     {m['max_drawdown']:.2%}")
    print(f"  CAGR:             {m['cagr']:.2%}")

    if args.portfolio_mc:
        settings = get_settings()
        syms = [s.strip().upper() for s in args.mc_symbols.split(",") if s.strip()]
        print()
        print("=== Real market: portfolio Monte Carlo (Polygon) ===")
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
    try:
        main()
    except RuntimeError as e:
        if "POLYGON_API_KEY" in str(e):
            print(
                "ERROR: Set POLYGON_API_KEY in .env at the repo root (see .env.example), then re-run.",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
