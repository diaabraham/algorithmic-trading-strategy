import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
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
        
        # Fill NaN values
        self.data = self.data.ffill().bfill()
        
        # Generate signals with stricter regime + trend context filters.
        high_vol_regime = self.data['Vol_Percentile'] > self.params.vol_threshold
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

        signals['long_entry'] = mean_reversion_long | momentum_long
        
        signals['short_entry'] = mean_reversion_short | momentum_short

        if self.params.sentiment_threshold >= 0 and "News_Sentiment" in self.data.columns:
            sentiment = self.data["News_Sentiment"].fillna(0)
            signals['long_entry'] = signals['long_entry'] & (sentiment >= -self.params.sentiment_threshold)
            signals['short_entry'] = signals['short_entry'] & (sentiment <= self.params.sentiment_threshold)

        if self.params.ml_up_min_long > 0 and "ML_Up_Proba" in self.data.columns:
            m = self.data["ML_Up_Proba"].fillna(0.5).clip(0.0, 1.0)
            signals["long_entry"] = signals["long_entry"] & (m >= self.params.ml_up_min_long)
            signals["short_entry"] = signals["short_entry"] & (m <= self.params.ml_up_max_short)

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
                stop_loss = False
                take_profit = False
                
                # Mean reversion exit
                if (self.position == 1 and price_deviation > -self.params.mean_reversion_exit_z * deviation_std) or \
                   (self.position == -1 and price_deviation < self.params.mean_reversion_exit_z * deviation_std):
                    exit_signal = True
                
                # RSI extreme exit
                elif (self.position == 1 and rsi > 55) or (self.position == -1 and rsi < 45):
                    exit_signal = True
                
                # ATR-based stop/target helps prevent large losers in extended volatility.
                elif self.entry_atr is not None and self.entry_atr > 0:
                    if self.position == 1:
                        stop_loss = current_price <= (self.current_trade.entry_price - self.params.atr_stop_mult * self.entry_atr)
                        take_profit = current_price >= (self.current_trade.entry_price + self.params.atr_take_mult * self.entry_atr)
                    else:
                        stop_loss = current_price >= (self.current_trade.entry_price + self.params.atr_stop_mult * self.entry_atr)
                        take_profit = current_price <= (self.current_trade.entry_price - self.params.atr_take_mult * self.entry_atr)

                    if stop_loss or take_profit:
                        exit_signal = True

                # Fallback deviation stop loss
                elif (self.position == 1 and price_deviation < -self.params.stop_z_fallback * deviation_std) or \
                     (self.position == -1 and price_deviation > self.params.stop_z_fallback * deviation_std):
                    exit_signal = True
                
                # Time-based exit (5 trading days equivalent)
                elif self.current_trade.holding_period >= self.params.max_hold_days:
                    exit_signal = True
                
                if exit_signal:
                    self.current_trade.exit_date = current_date
                    self.current_trade.exit_price = current_price
                    self.current_trade.holding_period = (current_date - self.current_trade.entry_date).days
                    
                    if self.position == 1:
                        self.current_trade.pnl = (current_price - self.current_trade.entry_price) / self.current_trade.entry_price
                    else:
                        self.current_trade.pnl = (self.current_trade.entry_price - current_price) / self.current_trade.entry_price
                    
                    self.trades.append(self.current_trade)
                    self.position = 0
                    self.current_trade = None
                    self.entry_atr = None
                    self.entry_signal_strength = None
            
            # Check for entry signals
            elif self.position == 0:
                if signals['long_entry'].iloc[i]:
                    self.position = 1
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
            
            if self.current_trade is not None:
                self.current_trade.holding_period = (current_date - self.current_trade.entry_date).days
        
        # Close any remaining open trade
        if self.current_trade is not None:
            self.current_trade.exit_date = self.data.index[-1]
            self.current_trade.exit_price = self.data['Close'].iloc[-1]
            self.current_trade.holding_period = (self.current_trade.exit_date - self.current_trade.entry_date).days
            
            if self.position == 1:
                self.current_trade.pnl = (self.current_trade.exit_price - self.current_trade.entry_price) / self.current_trade.entry_price
            else:
                self.current_trade.pnl = (self.current_trade.entry_price - self.current_trade.exit_price) / self.current_trade.entry_price
            
            self.trades.append(self.current_trade)
        
        return self.calculate_metrics()
        
    def calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics from the backtest results.
        
        Returns:
            Dict: Dictionary containing performance metrics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'cagr': 0
            }
            
        # Calculate basic metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        win_rate = winning_trades / total_trades
        
        # Calculate returns
        returns = [trade.pnl for trade in self.trades]
        avg_return = np.mean(returns)
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # 2% annual risk-free rate
        excess_returns = np.array(returns) - risk_free_rate/252
        returns_std = np.std(excess_returns)
        sharpe_ratio = 0 if returns_std == 0 else (np.sqrt(252) * np.mean(excess_returns) / returns_std)
        
        # Calculate max drawdown
        cumulative_returns = np.cumprod(1 + np.array(returns))
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)
        
        # Calculate CAGR
        total_return = np.prod(1 + np.array(returns)) - 1
        years = (self.data.index[-1] - self.data.index[0]).days / 365
        cagr = (1 + total_return) ** (1/years) - 1
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cagr': cagr
        }