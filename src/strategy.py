import sys
sys.path.append('.')  # Add current directory to Python path

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from src.data_loader import DataLoader
from src.visualization import StrategyVisualizer
from src.trade import Trade

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
        
        # Calculate indicators
        self.data['Price_Deviation'] = (self.data['Close'] - self.data['EMA_20']) / self.data['EMA_20']
        self.data['Deviation_Std'] = self.data['Price_Deviation'].rolling(window=20).std()
        
        # Calculate RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Fill NaN values
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        
        # Generate signals with balanced conditions
        signals['long_entry'] = (
            (self.data['Vol_Percentile'] > 0.7) &  # High volatility
            (self.data['Price_Deviation'] < -1.2 * self.data['Deviation_Std']) &  # Oversold
            (self.data['RSI'] < 35)  # Oversold
        )
        
        signals['short_entry'] = (
            (self.data['Vol_Percentile'] > 0.7) &  # High volatility
            (self.data['Price_Deviation'] > 1.2 * self.data['Deviation_Std']) &  # Overbought
            (self.data['RSI'] > 65)  # Overbought
        )
        
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
        
        for i in range(len(self.data)):
            if i < 20:  # Skip the first 20 days
                continue
                
            current_date = self.data.index[i]
            current_price = self.data['Close'].iloc[i]
            
            # Check for exit signals first
            if self.position != 0 and self.current_trade is not None:
                price_deviation = self.data['Price_Deviation'].iloc[i]
                deviation_std = self.data['Deviation_Std'].iloc[i]
                rsi = self.data['RSI'].iloc[i]
                
                exit_signal = False
                
                # Mean reversion exit
                if (self.position == 1 and price_deviation > -0.5 * deviation_std) or \
                   (self.position == -1 and price_deviation < 0.5 * deviation_std):
                    exit_signal = True
                
                # RSI extreme exit
                elif (self.position == 1 and rsi > 50) or (self.position == -1 and rsi < 50):
                    exit_signal = True
                
                # Stop loss
                elif (self.position == 1 and price_deviation < -1.8 * deviation_std) or \
                     (self.position == -1 and price_deviation > 1.8 * deviation_std):
                    exit_signal = True
                
                # Time-based exit (7 days)
                elif self.current_trade.holding_period >= 7:
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