import argparse
from datetime import datetime
import pandas as pd
from src.data_loader import DataLoader
from src.strategy import VolatilityRegimeStrategy
from src.visualization import StrategyVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Run Volatility Regime Mean Reversion Strategy Backtest')
    parser.add_argument('--symbol', type=str, default='SPY', help='Stock/ETF symbol (default: SPY)')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float, default=100000.0, help='Initial capital (default: 100000)')
    parser.add_argument('--plot', action='store_true', help='Show performance plots')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Running backtest for {args.symbol} from {args.start_date} to {args.end_date}")
    print(f"Initial capital: ${args.initial_capital:,.2f}")
    
    # Load and preprocess data
    data_loader = DataLoader(args.symbol, args.start_date, args.end_date)
    data = data_loader.get_data()
    
    if data.empty:
        print("Error: No data available for the specified period")
        return
    
    # Run strategy
    strategy = VolatilityRegimeStrategy(data, args.initial_capital)
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
        
if __name__ == "__main__":
    main() 