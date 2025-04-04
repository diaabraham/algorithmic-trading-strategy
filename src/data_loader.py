import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, symbol: str, start_date: str, end_date: str):
        """
        Initialize the data loader with symbol and date range.
        
        Args:
            symbol (str): Stock/ETF symbol (e.g., 'SPY')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Returns:
            pd.DataFrame: DataFrame containing OHLCV data
        """
        try:
            print("Fetching data from Yahoo Finance...")
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval='1d'
            )
            print(f"Data shape: {self.data.shape}")
            print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
            
            # Calculate daily returns
            self.data['Returns'] = self.data['Close'].pct_change()
            
            # Calculate 20-day rolling volatility
            self.data['Volatility'] = self.data['Returns'].rolling(window=20).std()
            
            # Calculate 20-day EMA
            self.data['EMA_20'] = self.data['Close'].ewm(span=20, adjust=False).mean()
            
            # Calculate volatility percentiles using a rolling window
            self.data['Vol_Percentile'] = self.data['Volatility'].rolling(window=252).rank(pct=True)
            
            print("\nData statistics:")
            print(f"Volatility mean: {self.data['Volatility'].mean():.4f}")
            print(f"Volatility max: {self.data['Volatility'].max():.4f}")
            print(f"Volatility min: {self.data['Volatility'].min():.4f}")
            print(f"Volatility percentile 90%: {self.data['Vol_Percentile'].quantile(0.9):.4f}")
            
            return self.data
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
            
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the data by adding technical indicators and regime signals.
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        if self.data is None or self.data.empty:
            print("No data available. Please fetch data first.")
            return pd.DataFrame()
            
        # Calculate standard deviation bands
        self.data['Upper_Band'] = self.data['EMA_20'] + 2 * self.data['Volatility']
        self.data['Lower_Band'] = self.data['EMA_20'] - 2 * self.data['Volatility']
        
        # Drop NaN values
        self.data = self.data.dropna()
        print(f"\nData after preprocessing shape: {self.data.shape}")
        
        return self.data
        
    def get_data(self) -> pd.DataFrame:
        """
        Get the complete processed dataset.
        
        Returns:
            pd.DataFrame: Complete processed dataset
        """
        if self.data is None or self.data.empty:
            self.fetch_data()
            self.preprocess_data()
        return self.data 