import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Trade:
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position: int  # 1 for long, -1 for short
    pnl: float
    holding_period: int

class VolatilityRegimeStrategy:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000.0):
        """
        Initialize the strategy with data and initial capital.
        
        Args:
            data (pd.DataFrame): Preprocessed market data
            initial_capital (float): Initial capital for backtesting
        """
        self.data = data
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0
        self.trades: List[Trade] = []
        self.current_trade = None
        
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on volatility regime and price deviation.
        
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        signals = pd.DataFrame(index=self.data.index)
        
        # Calculate price deviation from EMA
        self.data['Price_Deviation'] = (self.data['Close'] - self.data['EMA_20']) / self.data['EMA_20']
        
        # Calculate standard deviation of price deviation
        self.data['Deviation_Std'] = self.data['Price_Deviation'].rolling(window=20).std()
        
        # Entry signals
        signals['long_entry'] = (
            (self.data['Vol_Percentile'] > 0.7) &  # More sensitive to high volatility
            (self.data['Price_Deviation'] < -1.5 * self.data['Deviation_Std']) &  # More sensitive to oversold
            (self.position == 0)
        )
        
        signals['short_entry'] = (
            (self.data['Vol_Percentile'] > 0.7) &  # More sensitive to high volatility
            (self.data['Price_Deviation'] > 1.5 * self.data['Deviation_Std']) &  # More sensitive to overbought
            (self.position == 0)
        )
        
        # Exit signals
        signals['exit'] = (
            (self.position != 0) & (
                (self.data['Price_Deviation'].abs() < 0.3 * self.data['Deviation_Std']) |  # More lenient mean reversion
                (self.data['Close'] > self.data['Upper_Band'] * 1.2) |  # More lenient stop loss
                (self.data['Close'] < self.data['Lower_Band'] * 1.2)  # More lenient stop loss
            )
        )
        
        # Print signal statistics
        print("\nSignal Statistics:")
        print(f"Number of long entry signals: {signals['long_entry'].sum()}")
        print(f"Number of short entry signals: {signals['short_entry'].sum()}")
        print(f"Number of exit signals: {signals['exit'].sum()}")
        
        # Print some example conditions
        print("\nExample Conditions:")
        print(f"Volatility percentile > 0.7: {(self.data['Vol_Percentile'] > 0.7).sum()}")
        print(f"Price deviation < -1.5*std: {(self.data['Price_Deviation'] < -1.5 * self.data['Deviation_Std']).sum()}")
        print(f"Price deviation > 1.5*std: {(self.data['Price_Deviation'] > 1.5 * self.data['Deviation_Std']).sum()}")
        
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
        
        for i in range(len(self.data)):
            current_date = self.data.index[i]
            current_price = self.data['Close'].iloc[i]
            
            # Check for entry signals
            if signals['long_entry'].iloc[i] and self.position == 0:
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
                
            elif signals['short_entry'].iloc[i] and self.position == 0:
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
                
            # Check for exit signals
            elif signals['exit'].iloc[i] and self.position != 0:
                self.current_trade.exit_date = current_date
                self.current_trade.exit_price = current_price
                self.current_trade.holding_period = (current_date - self.current_trade.entry_date).days
                
                # Calculate PnL
                if self.position == 1:
                    self.current_trade.pnl = (current_price - self.current_trade.entry_price) / self.current_trade.entry_price
                else:
                    self.current_trade.pnl = (self.current_trade.entry_price - current_price) / self.current_trade.entry_price
                
                self.trades.append(self.current_trade)
                self.position = 0
                self.current_trade = None
                
            # Update holding period for current trade
            if self.current_trade is not None:
                self.current_trade.holding_period = (current_date - self.current_trade.entry_date).days
                
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
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        
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