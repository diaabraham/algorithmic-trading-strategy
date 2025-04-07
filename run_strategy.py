from src.data_loader import DataLoader
from src.strategy import VolatilityRegimeStrategy
from src.visualization import StrategyVisualizer

# Load and preprocess data
data_loader = DataLoader('SPY', '2020-01-01', '2023-12-31')
data = data_loader.get_data()

# Initialize and run the strategy
strategy = VolatilityRegimeStrategy(data)
metrics = strategy.run_backtest()

# Print performance metrics
print("\nStrategy Performance Metrics:")
print(f"Total Trades: {metrics['total_trades']}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
print(f"Average Return per Trade: {metrics['avg_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
print(f"CAGR: {metrics['cagr']:.2%}")

# Create visualizer and plot results
visualizer = StrategyVisualizer(data, strategy.trades)
visualizer.plot_all()