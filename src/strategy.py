import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from src.black_scholes import black_scholes_up_proba, expected_move
from src.brinson import brinson_single_asset
from src.trade import Trade

@dataclass
class StrategyParams:
    vol_threshold: float = 0.70
    mr_z_threshold: float = 1.1
    rsi_long: float = 35.0
    rsi_short: float = 65.0
    trend_slope_flat: float = 0.03
    trend_slope_strong: float = 0.01
    momentum_atr_mult: float = 1.0
    momentum_rsi_long: float = 55.0
    momentum_rsi_short: float = 45.0
    mean_reversion_exit_z: float = 0.2
    stop_z_fallback: float = 2.1
    atr_stop_mult: float = 1.25
    atr_take_mult: float = 2.0
    max_hold_days: int = 5
    sentiment_threshold: float = -1.0  # disabled when negative
    # ML gate: requires column ML_Up_Proba (from src.ml_signal). Disabled when ml_up_min_long <= 0.
    ml_up_min_long: float = 0.0
    ml_up_max_short: float = 1.0
    # Trend sleeve uses a softer probability floor so a rising market is held, not faded.
    ml_trend_min: float = 0.50
    # Exit a trend only after price is this far under the 50-day average. A tag of the average is a pause.
    trend_exit_band: float = 0.03
    vol_lookback: int = 252
    # Black-Scholes cone: fade only dislocations outside the hold-horizon expected move,
    # and only when risk-neutral P(finish beyond the 20-day EMA) still supports the trade.
    use_bs_gate: bool = True
    bs_move_mult: float = 0.75
    bs_long_min: float = 0.45
    bs_short_max: float = 0.55

class VolatilityRegimeStrategy:
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        params: StrategyParams | None = None,
        eval_start_date: str | None = None,
    ):
        """
        Initialize the strategy with data and initial capital.
        
        Args:
            data (pd.DataFrame): Preprocessed market data
            initial_capital (float): Initial capital for backtesting
        """
        # Work on an internal copy to avoid chained-assignment/pandas view bugs.
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.params = params or StrategyParams()
        self.current_capital = initial_capital
        self.position = 0
        self.trades: List[Trade] = []
        self.current_trade = None
        self.entry_atr = None
        self.entry_signal_strength = None
        self.entry_style = None
        self.entry_index = None
        self.eval_start_date = pd.to_datetime(eval_start_date, utc=True) if eval_start_date else None
        
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on volatility regime and price deviation.
        
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        signals = pd.DataFrame(index=self.data.index)
        
        # Calculate indicators
        self.data['Price_Deviation'] = (self.data['Close'] - self.data['EMA_20']) / self.data['EMA_20']
        self.data['Deviation_Std'] = self.data['Price_Deviation'].rolling(window=20).std()
        self.data['EMA_50'] = self.data['Close'].ewm(span=50, adjust=False).mean()
        self.data['Trend_Slope'] = self.data['EMA_20'].pct_change(5)
        self.data['ATR'] = (self.data['High'] - self.data['Low']).rolling(window=14).mean()
        
        # Calculate RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Forward-fill only. Back-fill would copy later bars into earlier ones.
        self.data = self.data.ffill()

        # Current-scenario vol: rank against the last year, not the whole history.
        lookback = max(int(self.params.vol_lookback), 2)
        current_vol_pct = self.data["Volatility"].rolling(lookback, min_periods=min(60, lookback)).rank(pct=True)
        if "Vol_Percentile" in self.data.columns:
            current_vol_pct = current_vol_pct.fillna(self.data["Vol_Percentile"])
        self.data["Vol_Percentile_Current"] = current_vol_pct

        sigma = self.data["Volatility"].replace(0, np.nan).fillna(0.2).clip(lower=1e-4)
        self.data["BS_Expected_Move"] = expected_move(self.data["Close"], sigma, self.params.max_hold_days)
        self.data["BS_Up_Proba"] = black_scholes_up_proba(
            self.data["Close"],
            self.data["EMA_20"].clip(lower=1e-8),
            sigma,
            self.params.max_hold_days,
        )
        
        # Generate signals with stricter regime + trend context filters.
        high_vol_regime = self.data["Vol_Percentile_Current"] > self.params.vol_threshold
        long_mean_reversion = self.data['Price_Deviation'] < -self.params.mr_z_threshold * self.data['Deviation_Std']
        short_mean_reversion = self.data['Price_Deviation'] > self.params.mr_z_threshold * self.data['Deviation_Std']
        uptrend = self.data['Close'] > self.data['EMA_50']
        downtrend = self.data['Close'] < self.data['EMA_50']
        trend_flattening = self.data['Trend_Slope'].abs() < self.params.trend_slope_flat
        strong_uptrend = (self.data['EMA_20'] > self.data['EMA_50']) & (self.data['Trend_Slope'] > self.params.trend_slope_strong)
        strong_downtrend = (self.data['EMA_20'] < self.data['EMA_50']) & (self.data['Trend_Slope'] < -self.params.trend_slope_strong)

        # In strong trends, mean-reversion often fails; allow breakout/momentum entries.
        momentum_long = (
            high_vol_regime &
            strong_uptrend &
            (self.data['Close'] > self.data['EMA_20'] + self.params.momentum_atr_mult * self.data['ATR']) &
            (self.data['RSI'] > self.params.momentum_rsi_long)
        )
        momentum_short = (
            high_vol_regime &
            strong_downtrend &
            (self.data['Close'] < self.data['EMA_20'] - self.params.momentum_atr_mult * self.data['ATR']) &
            (self.data['RSI'] < self.params.momentum_rsi_short)
        )

        mean_reversion_long = (
            high_vol_regime &
            long_mean_reversion &
            (self.data['RSI'] < self.params.rsi_long) &
            (uptrend | trend_flattening) &
            (~strong_downtrend)
        )
        mean_reversion_short = (
            high_vol_regime &
            short_mean_reversion &
            (self.data['RSI'] > self.params.rsi_short) &
            (downtrend | trend_flattening) &
            (~strong_uptrend)
        )

        if self.params.use_bs_gate:
            dislocated = (self.data["Close"] - self.data["EMA_20"]).abs() >= (
                self.params.bs_move_mult * self.data["BS_Expected_Move"]
            )
            bs_up = self.data["BS_Up_Proba"]
            mean_reversion_long = mean_reversion_long & dislocated & (bs_up >= self.params.bs_long_min)
            mean_reversion_short = mean_reversion_short & dislocated & (bs_up <= self.params.bs_short_max)
            momentum_long = momentum_long & (bs_up >= 0.5)
            momentum_short = momentum_short & (bs_up <= 0.5)

        # Rising market: stay long once price is above the 50-day average and the
        # cone says a finish above the 20-day mean is the more likely outcome.
        # Waiting for the 20-day average to cross buys the rally after it has run.
        trend_long = uptrend & (self.data["BS_Up_Proba"] >= 0.5)
        # Do not short a market that is still above the 50-day average.
        mean_reversion_short = mean_reversion_short & ~uptrend
        momentum_short = momentum_short & ~uptrend

        if self.params.sentiment_threshold >= 0 and "News_Sentiment" in self.data.columns:
            sentiment = self.data["News_Sentiment"].fillna(0)
            mean_reversion_long = mean_reversion_long & (sentiment >= -self.params.sentiment_threshold)
            momentum_long = momentum_long & (sentiment >= -self.params.sentiment_threshold)
            trend_long = trend_long & (sentiment >= -self.params.sentiment_threshold)
            mean_reversion_short = mean_reversion_short & (sentiment <= self.params.sentiment_threshold)
            momentum_short = momentum_short & (sentiment <= self.params.sentiment_threshold)

        signals['long_entry'] = mean_reversion_long | momentum_long | trend_long
        signals['short_entry'] = mean_reversion_short | momentum_short
        # Style flags stay independent of the entry filters below so an entry
        # that survives the filters still knows which exit rules apply.
        signals['long_momentum'] = momentum_long
        signals['short_momentum'] = momentum_short
        signals['long_trend'] = trend_long

        if self.params.ml_up_min_long > 0 and "ML_Up_Proba" in self.data.columns:
            m = self.data["ML_Up_Proba"].fillna(0.5).clip(0.0, 1.0)
            fade_long = (mean_reversion_long | momentum_long) & (m >= self.params.ml_up_min_long)
            trend_ok = trend_long & (m >= self.params.ml_trend_min)
            signals["long_entry"] = fade_long | trend_ok
            signals["long_trend"] = trend_ok
            signals["short_entry"] = signals["short_entry"] & (m <= self.params.ml_up_max_short)

        for col in ("long_entry", "short_entry", "long_momentum", "short_momentum", "long_trend"):
            signals[col] = signals[col].fillna(False)

        return signals
        
    def run_backtest(self) -> Dict:
        """
        Run the backtest and return performance metrics.
        
        Returns:
            Dict: Dictionary containing performance metrics
        """
        signals = self.generate_signals()
        self.trades = []
        self.position = 0
        self.current_capital = self.initial_capital
        self.current_trade = None
        self.entry_atr = None
        self.entry_signal_strength = None
        self.entry_style = None
        self.entry_index = None
        
        for i in range(len(self.data)):
            if i < 20:  # Skip the first 20 days
                continue
                
            current_date = self.data.index[i]
            current_price = self.data['Close'].iloc[i]
            price_deviation = self.data['Price_Deviation'].iloc[i]
            deviation_std = self.data['Deviation_Std'].iloc[i]
            rsi = self.data['RSI'].iloc[i]
            atr = self.data['ATR'].iloc[i]

            if self.eval_start_date is not None and current_date < self.eval_start_date:
                continue
            
            # Check for exit signals first
            if self.position != 0 and self.current_trade is not None:
                exit_signal = False
                held_bars = i - self.entry_index
                self.current_trade.holding_period = held_bars
                # Mean-reversion fades and RSI exits contradict momentum and trend entries.
                mean_reversion_style = self.entry_style == "mean_reversion"

                if self.entry_style == "trend" and self.position == 1:
                    ema_50 = self.data["EMA_50"].iloc[i]
                    broken = current_price < ema_50 * (1.0 - self.params.trend_exit_band)
                    cone_failed = self.data["BS_Up_Proba"].iloc[i] < self.params.bs_long_min
                    if broken and cone_failed:
                        exit_signal = True
                elif mean_reversion_style:
                    if (self.position == 1 and price_deviation > -self.params.mean_reversion_exit_z * deviation_std) or \
                       (self.position == -1 and price_deviation < self.params.mean_reversion_exit_z * deviation_std):
                        exit_signal = True
                    elif (self.position == 1 and rsi > 55) or (self.position == -1 and rsi < 45):
                        exit_signal = True

                # ATR stop for fades. A trend hold exits on the 50-day break, not a one-bar ATR dip.
                if not exit_signal and self.entry_style != "trend" and self.entry_atr is not None and self.entry_atr > 0:
                    if self.position == 1:
                        stop_loss = current_price <= (self.current_trade.entry_price - self.params.atr_stop_mult * self.entry_atr)
                        take_profit = self.entry_style != "trend" and current_price >= (
                            self.current_trade.entry_price + self.params.atr_take_mult * self.entry_atr
                        )
                    else:
                        stop_loss = current_price >= (self.current_trade.entry_price + self.params.atr_stop_mult * self.entry_atr)
                        take_profit = current_price <= (self.current_trade.entry_price - self.params.atr_take_mult * self.entry_atr)
                    if stop_loss or take_profit:
                        exit_signal = True

                if not exit_signal and self.entry_style != "trend":
                    if (self.position == 1 and price_deviation < -self.params.stop_z_fallback * deviation_std) or \
                       (self.position == -1 and price_deviation > self.params.stop_z_fallback * deviation_std):
                        exit_signal = True

                # max_hold_days counts trading bars. Trend holds stay on until the trend breaks.
                if not exit_signal and self.entry_style != "trend" and held_bars >= self.params.max_hold_days:
                    exit_signal = True
                
                if exit_signal:
                    self.current_trade.exit_date = current_date
                    self.current_trade.exit_price = current_price
                    
                    if self.position == 1:
                        self.current_trade.pnl = (current_price - self.current_trade.entry_price) / self.current_trade.entry_price
                    else:
                        self.current_trade.pnl = (self.current_trade.entry_price - current_price) / self.current_trade.entry_price
                    
                    self.trades.append(self.current_trade)
                    self.position = 0
                    self.current_trade = None
                    self.entry_atr = None
                    self.entry_signal_strength = None
                    self.entry_style = None
                    self.entry_index = None
            
            # Check for entry signals
            elif self.position == 0:
                if signals['long_entry'].iloc[i]:
                    self.position = 1
                    if "long_trend" in signals.columns and bool(signals["long_trend"].iloc[i]):
                        self.entry_style = "trend"
                    elif bool(signals["long_momentum"].iloc[i]):
                        self.entry_style = "momentum"
                    else:
                        self.entry_style = "mean_reversion"
                    self.entry_index = i
                    self.current_trade = Trade(
                        entry_date=current_date,
                        exit_date=None,
                        entry_price=current_price,
                        exit_price=None,
                        position=1,
                        pnl=0.0,
                        holding_period=0
                    )
                    self.entry_atr = atr if not np.isnan(atr) else None
                    self.entry_signal_strength = abs(price_deviation / (deviation_std + 1e-9))
                    
                elif signals['short_entry'].iloc[i]:
                    self.position = -1
                    self.entry_style = "momentum" if bool(signals['short_momentum'].iloc[i]) else "mean_reversion"
                    self.entry_index = i
                    self.current_trade = Trade(
                        entry_date=current_date,
                        exit_date=None,
                        entry_price=current_price,
                        exit_price=None,
                        position=-1,
                        pnl=0.0,
                        holding_period=0
                    )
                    self.entry_atr = atr if not np.isnan(atr) else None
                    self.entry_signal_strength = abs(price_deviation / (deviation_std + 1e-9))
        
        # Close any remaining open trade. Leave self.position set so live
        # routing can tell a still-open position from a flat book.
        if self.current_trade is not None:
            self.current_trade.exit_date = self.data.index[-1]
            self.current_trade.exit_price = self.data['Close'].iloc[-1]
            if self.entry_index is not None:
                self.current_trade.holding_period = (len(self.data) - 1) - self.entry_index
            
            if self.position == 1:
                self.current_trade.pnl = (self.current_trade.exit_price - self.current_trade.entry_price) / self.current_trade.entry_price
            else:
                self.current_trade.pnl = (self.current_trade.entry_price - self.current_trade.exit_price) / self.current_trade.entry_price
            
            self.trades.append(self.current_trade)
        
        return self.calculate_metrics()

    def _daily_exposure(self) -> tuple[pd.Series, pd.Series]:
        """Position and close-to-close asset return over the evaluation window."""
        idx = self.data.index
        position = pd.Series(0.0, index=idx)
        for trade in self.trades:
            if trade.entry_date is None or trade.exit_date is None:
                continue
            held = (idx > trade.entry_date) & (idx <= trade.exit_date)
            position.loc[held] = float(trade.position)
        asset = self.data["Close"].astype(float).pct_change().fillna(0.0)
        if self.eval_start_date is not None:
            mask = idx >= self.eval_start_date
            position = position.loc[mask]
            asset = asset.loc[mask]
        position = position.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        asset = asset.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return position, asset

    def _strategy_daily_returns(self) -> pd.Series:
        """Close-to-close strategy returns, including flat days as zero."""
        position, asset = self._daily_exposure()
        return position * asset

    def _brinson_factors(self) -> Dict:
        position, asset = self._daily_exposure()
        factors = brinson_single_asset(position, asset)
        return {f"brinson_{key}": value for key, value in factors.items()}
        
    def calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics from the backtest results.
        
        Returns:
            Dict: Dictionary containing performance metrics
        """
        brinson = self._brinson_factors()
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'cagr': 0,
                **brinson,
            }
            
        # Calculate basic metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        win_rate = winning_trades / total_trades
        
        # Calculate returns
        returns = [trade.pnl for trade in self.trades]
        avg_return = np.mean(returns)
        
        daily = self._strategy_daily_returns()
        if len(daily) == 0:
            sharpe_ratio = 0.0
            max_drawdown = 0.0
        else:
            # Per-trade returns are not daily observations; scale the daily book.
            excess = daily.to_numpy(dtype=float) - (0.02 / 252)
            returns_std = float(np.std(excess))
            sharpe_ratio = 0.0 if returns_std == 0 else float(np.sqrt(252) * np.mean(excess) / returns_std)
            equity = np.cumprod(1.0 + daily.to_numpy(dtype=float))
            peak = np.maximum.accumulate(equity)
            max_drawdown = float(np.min(equity / peak - 1.0))
        
        # CAGR uses the evaluation window, not indicator warmup before it.
        total_return = float(np.prod(1 + np.array(returns)) - 1)
        window_start = self.data.index[0]
        if self.eval_start_date is not None and self.eval_start_date > window_start:
            window_start = self.eval_start_date
        span_days = (self.data.index[-1] - window_start).days
        if span_days <= 0 or total_return <= -1.0:
            cagr = total_return if total_return > -1.0 else -1.0
        else:
            cagr = float((1.0 + total_return) ** (1.0 / (span_days / 365.0)) - 1.0)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cagr': cagr,
            **brinson,
        }