# QuantVol-revert: Volatility Regime-Based Mean Reversion Strategy

A Python-based backtesting engine that implements a volatility regime-based mean reversion trading strategy. The strategy dynamically adjusts risk exposure based on rolling volatility bands and capitalizes on mean-reversion opportunities in high-volatility environments.

## Features

- Volatility regime detection using rolling standard deviation
- Mean-reversion trade signals in high-volatility environments
- Comprehensive backtesting with realistic transaction costs
- Performance metrics including Sharpe ratio, max drawdown, and CAGR
- Interactive visualization of strategy performance
- Support for multiple assets (SPY, QQQ, IWM)

## Project Structure

```
quantvol-revert/
├── data/                  # Data storage directory
├── notebooks/             # Jupyter notebooks for analysis
│   └── strategy_analysis.ipynb  # Interactive strategy analysis
├── src/                   # Source code
│   ├── __init__.py       # Package initialization
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── strategy.py       # Strategy implementation
│   └── visualization.py  # Plotting utilities
├── main.py               # CLI interface
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quantvol-revert.git
cd quantvol-revert
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run a backtest with default parameters:
```bash
python main.py --symbol SPY --start-date 2020-01-01 --end-date 2023-12-31 --plot
```

Available options:
- `--symbol`: Stock/ETF symbol (default: SPY)
- `--start-date`: Start date in YYYY-MM-DD format
- `--end-date`: End date in YYYY-MM-DD format
- `--initial-capital`: Initial capital (default: 100000)
- `--plot`: Show performance plots

### Jupyter Notebook

For interactive analysis:
```bash
jupyter notebook notebooks/strategy_analysis.ipynb
```

## Strategy Overview

The strategy operates in the following way:

1. **Volatility Regime Detection**:
   - Calculates 20-day rolling volatility
   - Identifies low and high volatility regimes using percentile thresholds
   - Low Vol Regime = σ < σₚ₅
   - High Vol Regime = σ > σₚ₉₀

2. **Signal Generation**:
   - Monitors price deviation from 20-day EMA
   - Enters mean-reversion trades when:
     - In high volatility regime
     - Price is ≥ 2 standard deviations from EMA

3. **Trade Management**:
   - Entry: Price > 2σ from EMA in high-vol regime
   - Exit: Price reverts within 0.5σ of EMA or after 5-day timeout
   - Stop-loss: 1.5x entry volatility

## Performance Metrics

The strategy tracks:
- Sharpe Ratio
- Maximum Drawdown
- CAGR (Compounded Annual Growth Rate)
- Win Rate
- Average Gain/Loss
- Exposure Ratio

## Dependencies

- Python 3.8+
- pandas>=1.3.0
- numpy>=1.21.0
- matplotlib>=3.4.0
- yfinance>=0.1.70
- scipy>=1.7.0
- jupyter>=1.0.0
- streamlit>=1.0.0
- ta>=0.10.0
- scikit-learn>=0.24.0
- tqdm>=4.62.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 