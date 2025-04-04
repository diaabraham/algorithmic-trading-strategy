import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List
from .strategy import Trade
import os

class StrategyVisualizer:
    def __init__(self, data: pd.DataFrame, trades: List[Trade]):
        """
        Initialize the visualizer with data and trades.
        
        Args:
            data (pd.DataFrame): Market data
            trades (List[Trade]): List of trades from backtest
        """
        self.data = data
        self.trades = trades
        
    def plot_equity_curve(self, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot the equity curve of the strategy.
        
        Args:
            ax (plt.Axes, optional): Matplotlib axes to plot on
            
        Returns:
            plt.Axes: The axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            
        # Calculate cumulative returns
        equity = pd.Series(1.0, index=self.data.index)
        for trade in self.trades:
            equity.loc[trade.exit_date] *= (1 + trade.pnl)
            
        equity = equity.cumprod()
        
        # Plot equity curve
        ax.plot(equity.index, equity.values, label='Strategy Equity')
        ax.set_title('Strategy Equity Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        ax.grid(True)
        ax.legend()
        
        return ax
        
    def plot_drawdown(self, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot the drawdown curve of the strategy.
        
        Args:
            ax (plt.Axes, optional): Matplotlib axes to plot on
            
        Returns:
            plt.Axes: The axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            
        # Calculate drawdown
        equity = pd.Series(1.0, index=self.data.index)
        for trade in self.trades:
            equity.loc[trade.exit_date] *= (1 + trade.pnl)
            
        equity = equity.cumprod()
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        
        # Plot drawdown
        ax.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        ax.plot(drawdown.index, drawdown.values, color='red', label='Drawdown')
        ax.set_title('Strategy Drawdown')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.grid(True)
        ax.legend()
        
        return ax
        
    def plot_trades(self, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot the trades on top of the price chart.
        
        Args:
            ax (plt.Axes, optional): Matplotlib axes to plot on
            
        Returns:
            plt.Axes: The axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            
        # Plot price
        ax.plot(self.data.index, self.data['Close'], label='Price', alpha=0.5)
        
        # Plot trades
        for trade in self.trades:
            if trade.position == 1:  # Long trade
                ax.scatter(trade.entry_date, trade.entry_price, color='green', marker='^', s=100)
                ax.scatter(trade.exit_date, trade.exit_price, color='red', marker='v', s=100)
            else:  # Short trade
                ax.scatter(trade.entry_date, trade.entry_price, color='red', marker='v', s=100)
                ax.scatter(trade.exit_date, trade.exit_price, color='green', marker='^', s=100)
                
        ax.set_title('Trades on Price Chart')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True)
        ax.legend()
        
        return ax
        
    def plot_volatility_regimes(self, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot the volatility regimes.
        
        Args:
            ax (plt.Axes, optional): Matplotlib axes to plot on
            
        Returns:
            plt.Axes: The axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            
        # Plot volatility
        ax.plot(self.data.index, self.data['Volatility'], label='Volatility', color='blue')
        
        # Plot regime thresholds
        ax.axhline(y=self.data['Volatility'].quantile(0.9), color='red', linestyle='--', label='High Vol Regime')
        ax.axhline(y=self.data['Volatility'].quantile(0.5), color='green', linestyle='--', label='Low Vol Regime')
        
        ax.set_title('Volatility Regimes')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility')
        ax.grid(True)
        ax.legend()
        
        return ax
        
    def plot_all(self) -> None:
        """
        Create a comprehensive dashboard of all plots and save to files.
        """
        # Create plots directory if it doesn't exist
        if not os.path.exists('plots'):
            os.makedirs('plots')
            
        # Plot and save equity curve
        fig, ax = plt.subplots(figsize=(15, 8))
        self.plot_equity_curve(ax)
        plt.savefig('plots/equity_curve.png')
        plt.close()
        
        # Plot and save drawdown
        fig, ax = plt.subplots(figsize=(15, 8))
        self.plot_drawdown(ax)
        plt.savefig('plots/drawdown.png')
        plt.close()
        
        # Plot and save trades
        fig, ax = plt.subplots(figsize=(15, 8))
        self.plot_trades(ax)
        plt.savefig('plots/trades.png')
        plt.close()
        
        # Plot and save volatility regimes
        fig, ax = plt.subplots(figsize=(15, 8))
        self.plot_volatility_regimes(ax)
        plt.savefig('plots/volatility_regimes.png')
        plt.close()
        
        print("\nPlots have been saved to the 'plots' directory:")
        print("1. plots/equity_curve.png")
        print("2. plots/drawdown.png")
        print("3. plots/trades.png")
        print("4. plots/volatility_regimes.png") 