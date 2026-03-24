import argparse
from datetime import datetime, date
import logging
from src.data_loader import DataLoader
from src.strategy import VolatilityRegimeStrategy, StrategyParams
from src.visualization import StrategyVisualizer
from src.config import get_settings
from src.live_trading import LiveTradingEngine
from src.model_runner import run_rigorous_suite
from src.portfolio_sim import MARCH_2026_HIGH_BETA_UNIVERSE, run_portfolio_monte_carlo

def parse_args():
    parser = argparse.ArgumentParser(description='Run Volatility Regime Mean Reversion Strategy Backtest')
    parser.add_argument('--symbol', type=str, default='SPY', help='Stock/ETF symbol (default: SPY)')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float, default=100000.0, help='Initial capital (default: 100000)')
    parser.add_argument('--plot', action='store_true', help='Show performance plots')
    parser.add_argument('--live', action='store_true', help='Enable live order routing to IBKR (must also set LIVE_TRADING_ENABLED=true)')
    parser.add_argument('--order-qty', type=int, default=1, help='Order quantity for live mode (default: 1)')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--holding-days', type=int, default=5, help='Max holding days for trades (default: 5)')
    parser.add_argument('--sentiment', action='store_true', help='Use Alpha Vantage news + optional Twitter sentiment filter')
    parser.add_argument('--alpha-vantage-key', type=str, default='', help='Alpha Vantage API key for news sentiment (or set ALPHAVANTAGE_API_KEY env)')
    parser.add_argument('--use-twitter', action='store_true', help='Use Twitter scrape sentiment in addition to news')
    parser.add_argument('--rigorous-test', action='store_true', help='Run rigorous suite (6m + 1w short/long) and exit')
    parser.add_argument('--symbols', type=str, default='WTI,TSLA,NVDA', help='Comma-separated symbols for rigorous testing')
    parser.add_argument('--as-of', type=str, default='2026-03-24', help='Anchor date (YYYY-MM-DD) for rigorous suite and MC window end')
    parser.add_argument('--portfolio-mc', action='store_true', help='Run equal-weight portfolio Monte Carlo bootstrap on a date window')
    parser.add_argument('--mc-scenarios', type=int, default=500, help='Monte Carlo scenarios (default: 500)')
    parser.add_argument('--capital-cad', type=float, default=25000.0, help='Portfolio notional in CAD for MC (default: 25000)')
    parser.add_argument('--mc-window-start', type=str, default='2026-03-01', help='Evaluation window start for portfolio MC')
    parser.add_argument('--mc-symbols', type=str, default='', help='Comma-separated symbols for MC (default: March 2026 high-beta preset)')
    parser.add_argument('--mc-tune', action='store_true', help='Grid-search params per symbol before MC (slower, more data)')
    parser.add_argument('--mc-ml', action='store_true', help='Enable ML probability gate during MC backtest')
    parser.add_argument('--train-ml', action='store_true', help='Attach walk-forward ML column and gate entries in single backtest')
    parser.add_argument('--ml-long-min', type=float, default=0.55, help='Min ML P(up) for long entries when ML enabled')
    parser.add_argument('--ml-short-max', type=float, default=0.45, help='Max ML P(up) for short entries when ML enabled')
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    
    settings = get_settings()
    alpha_key = args.alpha_vantage_key.strip() or __import__("os").getenv("ALPHAVANTAGE_API_KEY", "").strip()

    if args.portfolio_mc:
        as_of = date.fromisoformat(args.as_of)
        if args.mc_symbols.strip():
            mc_syms = [s.strip().upper() for s in args.mc_symbols.split(",") if s.strip()]
        else:
            mc_syms = list(MARCH_2026_HIGH_BETA_UNIVERSE)
        report = run_portfolio_monte_carlo(
            symbols=mc_syms,
            window_start=args.mc_window_start,
            window_end=as_of.isoformat(),
            initial_cad=args.capital_cad,
            cad_usd=settings.cad_usd_fx,
            scenarios=max(500, args.mc_scenarios),
            tune=args.mc_tune,
            use_ml=args.mc_ml,
            ml_long_min=args.ml_long_min,
            ml_short_max=args.ml_short_max,
            alpha_vantage_api_key=alpha_key if args.sentiment else None,
            use_twitter=args.use_twitter and args.sentiment,
        )
        print("\nPortfolio Monte Carlo (equal-weight slices, bootstrap trade returns):")
        print(f"  Symbols used: {report.symbols_used}")
        print(f"  Scenarios: {report.scenarios}  Initial USD (from CAD): ${report.initial_usd:,.2f}")
        print(f"  PnL %%  mean={report.pnl_pct_mean:.2%}  p05={report.pnl_pct_p05:.2%}  p50={report.pnl_pct_p50:.2%}  p95={report.pnl_pct_p95:.2%}")
        print(
            f"  Final USD  p05=${report.final_usd_p05:,.2f}  p50=${report.final_usd_p50:,.2f}  p95=${report.final_usd_p95:,.2f}"
        )
        return

    if args.rigorous_test:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        report = run_rigorous_suite(
            symbols=symbols,
            as_of=date.fromisoformat(args.as_of),
            alpha_vantage_api_key=alpha_key if args.sentiment else None,
            use_twitter=args.use_twitter and args.sentiment,
        )
        if report.empty:
            print("No rigorous test results produced.")
            return
        print("\nRigorous Test Results:")
        print(report[["symbol", "window", "style", "total_trades", "win_rate", "avg_return", "sharpe_ratio", "cagr"]].to_string(index=False))
        total_trades = report["total_trades"].sum()
        if total_trades > 0:
            weighted_win = (report["win_rate"] * report["total_trades"]).sum() / total_trades
            weighted_avg = (report["avg_return"] * report["total_trades"]).sum() / total_trades
        else:
            weighted_win = 0.0
            weighted_avg = 0.0
        print(f"\nOverall weighted win rate: {weighted_win:.2%}")
        print(f"Overall weighted avg return: {weighted_avg:.2%}")
        return

    print(f"Running backtest for {args.symbol} from {args.start_date} to {args.end_date}")
    print(f"Initial capital: ${args.initial_capital:,.2f}")

    data_loader = DataLoader(args.symbol, args.start_date, args.end_date)
    data = data_loader.get_data()
    if args.sentiment:
        from src.sentiment import combined_sentiment
        sentiment_score = combined_sentiment(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            alpha_vantage_api_key=alpha_key if alpha_key else None,
            twitter_query=f"${args.symbol}" if args.use_twitter else None,
        )
        data["News_Sentiment"] = sentiment_score

    if args.train_ml:
        from src.ml_signal import attach_ml_up_proba

        data = attach_ml_up_proba(data, eval_start=args.start_date)
    
    if data.empty:
        print("Error: No data available for the specified period")
        return
    
    # Run strategy
    params = StrategyParams(
        max_hold_days=args.holding_days,
        sentiment_threshold=0.4 if args.sentiment else -1.0,
        ml_up_min_long=args.ml_long_min if args.train_ml else 0.0,
        ml_up_max_short=args.ml_short_max if args.train_ml else 1.0,
    )
    strategy = VolatilityRegimeStrategy(data, args.initial_capital, params=params)
    metrics = strategy.run_backtest()
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Average Return: {metrics['avg_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"CAGR: {metrics['cagr']:.2%}")
    
    # Plot results if requested
    if args.plot:
        visualizer = StrategyVisualizer(data, strategy.trades)
        visualizer.plot_all()

    if args.live:
        if not settings.live_trading_enabled:
            raise RuntimeError(
                "Live mode requested but LIVE_TRADING_ENABLED is false. "
                "Set LIVE_TRADING_ENABLED=true to allow order routing."
            )
        if metrics["sharpe_ratio"] <= 0 or metrics["cagr"] <= 0 or metrics["win_rate"] < 0.45:
            raise RuntimeError(
                "Live routing blocked by quality gate: strategy metrics are below minimum thresholds "
                "(requires Sharpe>0, CAGR>0, WinRate>=45%)."
            )
        if strategy.trades:
            latest_trade = strategy.trades[-1]
            latest_signal = "long" if latest_trade.position == 1 else "short"
            engine = LiveTradingEngine(settings)
            engine.connect()
            try:
                last_price = float(data["Close"].iloc[-1]) if not data.empty else None
                routed_symbol = getattr(data_loader, "symbol", None) or args.symbol
                engine.submit_signal(routed_symbol, latest_signal, args.order_qty, last_price=last_price)
                mode = "paper" if settings.paper_trading_enabled else "live"
                print(f"{mode.capitalize()} order submitted to IBKR.")
            finally:
                engine.disconnect()
        
if __name__ == "__main__":
    main() 