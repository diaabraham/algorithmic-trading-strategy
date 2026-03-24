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
git clone https://github.com/diaabraham/algorithmic-trading-strategy.git
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
- `--live`: Route latest strategy signal to IBKR (guarded)
- `--order-qty`: Quantity used in live mode (default: 1)
- `--log-level`: Runtime logging level
- `--holding-days`: Max holding days per trade (short/long style control)
- `--sentiment`: Enable Alpha Vantage news sentiment filter
- `--use-twitter`: Blend in Twitter sentiment (snscrape)
- `--rigorous-test`: Run 6-month + 1-week short/long validation suite
- `--symbols`: Comma-separated symbols for rigorous testing
- `--as-of`: Anchor date for rigorous suite and portfolio MC end (default: `2026-03-24`)
- `--portfolio-mc`: Equal-weight portfolio Monte Carlo (bootstrap trade returns, default ≥500 paths)
- `--mc-scenarios`: Scenario count (floored to at least 500 when using `--portfolio-mc`)
- `--capital-cad`: Starting notional in CAD (default: `25000`)
- `--mc-window-start` / `--mc-symbols`: Evaluation start and symbol list (empty uses March 2026 high-beta preset)
- `--mc-tune`: Per-symbol grid search before MC (slower, more API usage)
- `--mc-ml`: Enable RandomForest probability gate during MC backtests
- `--train-ml`: Single-symbol backtest with walk-forward `ML_Up_Proba` overlay
- `--ml-long-min` / `--ml-short-max`: Thresholds when ML is enabled

### Environment Setup

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Set:
- `POLYGON_API_KEY` (required)
- `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID` (for IBKR)
- `IBKR_TIMEOUT` (seconds, default 15), `IBKR_READONLY` (read-only API session)
- `IBKR_CONTRACT_JSON` (optional): full `ib_insync` `Contract` fields as JSON for futures/options (overrides plain stock routing)
- `LIVE_TRADING_ENABLED=true` only when you intentionally want live orders
- `PAPER_TRADING_ENABLED=true` to keep routing in paper mode
- `MAX_ORDER_QTY`, `MAX_NOTIONAL_PER_ORDER` risk caps
- `CAD_USD_FX` (spot approximation for converting CAD notionals to USD in portfolio MC, default `0.74`)
- `ALPHAVANTAGE_API_KEY` (optional, for news sentiment)

### IBKR connection behavior

- On `connect()`, the client uses `timeout`/`readonly` from settings, verifies `isConnected()`, and logs connect/disconnect.
- Before `placeOrder`, contracts are **qualified** via `qualifyContracts`; unfilled symbols raise a clear error.
- Default routing is `Stock(symbol, "SMART", "USD")`. Futures/options require `IBKR_CONTRACT_JSON` (example in `.env.example`).
- Live orders use the **resolved** Polygon ticker when applicable (e.g. `WTI` → `USO`).

### Data coverage: stocks vs options/futures

- Daily OHLCV comes from **Polygon aggregates** for equities/ETFs. `WTI`/`CL`/`OIL` map to **`USO`** as a liquid proxy (avoids futures entitlements in typical starter accounts).
- Listed options and futures chains are **not** modeled in the backtester here; use ETF proxies (e.g. `USO`, `GLD`, index ETFs) or bring your own bars. IBKR live routing for non-stock products uses `IBKR_CONTRACT_JSON`.

### March 2026 research preset

- Portfolio MC defaults to a high-beta basket: `WTI, TSLA, NVDA, MSTR, AMD, COIN, SPY, QQQ, GLD` (see `src/portfolio_sim.py`).
- Window defaults: `--mc-window-start 2026-03-01` through `--as-of 2026-03-24` (adjust as needed).

### Machine learning overlay

- `attach_ml_up_proba` (`src/ml_signal.py`) trains a **single** RandomForest on bars **strictly before** `eval_start`, then scores `ML_Up_Proba` from `eval_start` onward (reduces lookahead vs in-sample fitting).
- Strategy gates long/short when `ml_up_min_long` / `ml_up_max_short` are set (`--train-ml` sets this in `main.py`). This is a **research filter**, not a guarantee of live edge; small samples (e.g. one week) will be noisy.

### Portfolio Monte Carlo (25k CAD, ≥500 scenarios)

- Capital is converted with `CAD_USD_FX`, split **equally** across symbols, each slice bootstraps **with replacement** over that symbol’s trade returns from the evaluation window (independent symbols, no correlation model—documented limitation).
- Example:

```bash
python main.py --portfolio-mc --capital-cad 25000 --mc-scenarios 500 --log-level WARNING
```

- Add `--mc-tune` or `--mc-ml` for heavier runs; requires Polygon (and sentiment keys if you also pass `--sentiment`).

### Real market results (not unit-test placeholders)

Fast unit tests use mocks or synthetic OHLCV; they **do not** prove performance on live data.

With `POLYGON_API_KEY` set (repo-root `.env`):

```bash
python scripts/verify_real_market.py
python scripts/verify_real_market.py --portfolio-mc --mc-symbols SPY,NVDA --capital-cad 25000
pytest tests/test_polygon_integration.py -v   # integration tests; skipped if key missing
```

### Symbol Notes

- `WTI` is supported as an alias and maps to `USO` (liquid WTI proxy ETF).
- Equity tickers like `NVDA`, `TSLA`, `SPY` are queried directly.

## Easier Usage

Single backtest with optional sentiment:

```bash
python main.py --symbol WTI --start-date 2025-09-24 --end-date 2026-03-24 --holding-days 2 --sentiment --use-twitter
```

Rigorous validation suite (6-month + past-week short/long styles):

```bash
python main.py --rigorous-test --symbols WTI,TSLA,NVDA --as-of 2026-03-24 --log-level WARNING
```

The suite prints weighted overall win rate and weighted average return.

Walk-forward ML on a single symbol backtest:

```bash
python main.py --symbol SPY --start-date 2024-01-01 --end-date 2026-03-24 --train-ml --ml-long-min 0.55 --ml-short-max 0.45
```

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
- polygon-api-client>=1.16.0
- ib-insync>=0.9.86
- python-dotenv>=1.0.0
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