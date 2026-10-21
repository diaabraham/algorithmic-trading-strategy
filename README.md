# QuantVol-revert: Volatility Regime Strategy

A Python backtester for a volatility-regime book. In a rising market it stays long while price holds the 50-day average and the Black-Scholes cone still favors a finish above the 20-day mean. Mean-reversion fades and breakout trades are reserved for high-volatility dislocations. A walk-forward RandomForest and Brinson attribution run with every single-symbol backtest.

## Features

- Trend sleeve for a calm uptrend, so the book is not stuck in cash when price never leaves the volatility cone
- Mean-reversion and momentum entries only in a high 252-day volatility rank
- Walk-forward RandomForest (`ML_Up_Proba`), refit every 21 bars on labels that are already known
- Black-Scholes expected-move cone from realized vol, used to gate fades
- Brinson split versus buy-and-hold: allocation, selection, and interaction
- Sharpe, max drawdown, and CAGR from daily close-to-close returns
- Daily bars from Massive, Databento, Polygon, Yahoo Finance, then Stooq

## Project Structure

```
algorithmic-trading-strategy/
├── notebooks/             # strategy1.ipynb, strategy1_clean.ipynb, strategy2.ipynb, strategy3.ipynb
├── scripts/               # verify_real_market.py (live daily bars)
├── src/
│   ├── data_loader.py     # Massive, Databento, Polygon, Yahoo, Stooq
│   ├── strategy.py        # Trend sleeve, high-vol fades, exits
│   ├── ml_signal.py       # Walk-forward RandomForest
│   ├── black_scholes.py   # Realized-vol cone and N(d2)
│   ├── brinson.py         # Brinson-Fachler versus buy-and-hold
│   ├── model_runner.py    # Rigorous suite and parameter grid
│   ├── portfolio_sim.py   # Equal-weight Monte Carlo
│   ├── broker_ibkr.py     # Paper-guarded IBKR routing
│   ├── live_trading.py    # Symbol aliases and open-position side
│   └── visualization.py
├── tests/
├── main.py
└── requirements.txt
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
- `--ml-long-min` / `--ml-short-max`: Fade thresholds (default `0.55` / `0.45`). The trend sleeve uses a softer `0.50` floor. `--train-ml` does not change this path; a single-symbol run always attaches the walk-forward model.

### Environment Setup

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Set:
- `MASSIVE_API_KEY` (optional). Massive daily aggregates, tried first when set.
- `DATABENTO_API_KEY` (optional). Databento `ohlcv-1d` bars. `DATABENTO_DATASET` defaults to `EQUS.SUMMARY`.
- `POLYGON_API_KEY` (optional). When a paid key is missing or that source fails, daily bars continue to the next source, then Yahoo Finance, then Stooq.
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
- Live orders use the resolved ticker when applicable (e.g. `WTI` → `USO`).
- Paper mode refuses IBKR live ports `7496` and `4001`. Use `7497` or `4002`. A read-only session does not place orders. An order is sent only when the last bar is still long or short.

### Data coverage: stocks vs options/futures

- Daily OHLCV tries **Massive**, **Databento**, and **Polygon** when those keys are set, then **Yahoo Finance** (`yfinance`), then **Stooq**. Stooq often returns a bot check or 404 to an unattended client; the loader records that failure and continues. `WTI`/`CL`/`OIL` map to **`USO`**.
- Listed options and futures chains are **not** modeled in the backtester here; use ETF proxies (e.g. `USO`, `GLD`, index ETFs) or bring your own bars. IBKR live routing for non-stock products uses `IBKR_CONTRACT_JSON`.

### March 2026 research preset

- Portfolio MC defaults to a high-beta basket: `WTI, TSLA, NVDA, MSTR, AMD, COIN, SPY, QQQ, GLD` (see `src/portfolio_sim.py`).
- Window defaults: `--mc-window-start 2026-03-01` through `--as-of 2026-03-24` (adjust as needed).

### Scenario adjustment

- **Trend sleeve.** Long when the close is above the 50-day average and risk-neutral `N(d2)` says a finish above the 20-day EMA is the more likely outcome (`BS_Up_Proba >= 0.5`). The hold stays on through a tag of the 50-day average. It exits only when price is more than 3% under that average and `N(d2)` has fallen below `0.45`. Shorts are not opened while price is still above the 50-day average. This sleeve does not require a high-vol rank or a break of the cone, and it does not use the 5-day holding limit.
- **High-vol fades.** Mean reversion and momentum still require the 252-day volatility rank to be above `0.70`. Mean reversion also has to clear `0.75` times the hold-horizon expected move, `S * sigma * sqrt(T)`. Those trades still exit on a revert, an ATR stop or target, or `max_hold_days` trading bars.
- **Walk-forward model.** `attach_recursive_ml_up_proba` refits a RandomForest every 21 bars. Training labels stop before the block being scored. Fade longs need `ML_Up_Proba >= --ml-long-min` (default `0.55`); fade shorts need `<= --ml-short-max` (default `0.45`). The trend sleeve only needs `>= 0.50`, so a neutral model does not flatten a rising market.
- **Brinson.** Versus buy-and-hold, allocation is time spent flat, selection is time spent short, and interaction is the residual. The single-symbol report prints all three plus the compounded strategy, buy-and-hold, and excess returns. The cone is realized vol, not listed-option implied vol.

### Portfolio Monte Carlo (25k CAD, ≥500 scenarios)

- Capital is converted with `CAD_USD_FX`, split **equally** across symbols, each slice bootstraps **with replacement** over that symbol’s trade returns from the evaluation window (independent symbols, no correlation model—documented limitation).
- Example:

```bash
python main.py --portfolio-mc --capital-cad 25000 --mc-scenarios 500 --log-level WARNING
```

- Add `--mc-tune` or `--mc-ml` for heavier runs. Price history uses Massive, Databento, or Polygon when those keys are set, otherwise Yahoo Finance and then Stooq. `--sentiment` also needs an Alpha Vantage key.

### Live daily bars

`scripts/verify_real_market.py` and `tests/test_market_integration.py` pull real sessions through the provider chain (Massive, Databento, Polygon when a key is set, then Yahoo Finance, then Stooq). The SPY check covers 2025-10-21 through 2026-10-21 and requires the book to be invested in that uptrend.

```bash
python scripts/verify_real_market.py
python scripts/verify_real_market.py --symbol WTI --start-date 2025-10-21 --end-date 2026-10-21
pytest tests/test_market_integration.py -v
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

Single-symbol backtest. History before `--start-date` is loaded only to train the walk-forward model; trades are scored from that date:

```bash
python main.py --symbol SPY --start-date 2025-10-21 --end-date 2026-10-21
```

### Jupyter Notebook

```bash
jupyter notebook notebooks/strategy1.ipynb
```

## Strategy Overview

1. **Volatility.** 20-day realized vol, ranked over the last 252 sessions (minimum 60). High vol means that rank is above `0.70`.
2. **Trend.** Long above the 50-day average when the cone agrees. Exit on a 3% break of that average together with a failed cone. No short in that uptrend.
3. **Fades.** In high vol, fade a move outside the expected-move cone back toward the 20-day EMA, or take a breakout. Those positions use an ATR stop and target and a bar-count time stop (`--holding-days`, default 5).
4. **Model.** Walk-forward `ML_Up_Proba` gates fades at `0.55` / `0.45` and the trend sleeve at `0.50`.

## Performance Metrics

The strategy tracks:
- Sharpe ratio and max drawdown from daily close-to-close returns, including flat days
- CAGR over the evaluation window
- Win rate and average trade return
- Brinson allocation, selection, and interaction versus buy-and-hold

## Dependencies

- Python 3.8+
- pandas, numpy, scipy, scikit-learn, matplotlib
- yfinance, massive, databento, polygon-api-client, requests
- ib-insync, python-dotenv
- jupyter, streamlit, ta, tqdm, pytest, vaderSentiment, snscrape

Versions are pinned as minimums in `requirements.txt`.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 