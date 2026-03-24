import pandas as pd
import numpy as np
import time
import logging
from polygon import RESTClient
from src.config import get_settings

logger = logging.getLogger(__name__)


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
        self.requested_symbol = symbol

    def _candidate_tickers(self) -> list[str]:
        symbol = self.requested_symbol.strip().upper()
        aliases = {
            # WTI workflows default to liquid proxy to avoid futures entitlement/rate issues.
            "WTI": ["USO"],
            "CL": ["USO"],
            "OIL": ["USO"],
        }
        return aliases.get(symbol, [symbol])
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Returns:
            pd.DataFrame: DataFrame containing OHLCV data
        """
        try:
            settings = get_settings()
            client = RESTClient(api_key=settings.polygon_api_key)
            last_error = None
            selected_symbol = self.symbol
            fetched_data = None

            for ticker in self._candidate_tickers():
                for attempt in range(1, 4):
                    try:
                        logger.info("Fetching %s attempt %s", ticker, attempt)
                        aggs = list(
                            client.get_aggs(
                                ticker=ticker,
                                multiplier=1,
                                timespan="day",
                                from_=self.start_date,
                                to=self.end_date,
                                adjusted=True,
                                sort="asc",
                                limit=50000,
                            )
                        )
                        if not aggs:
                            raise ValueError(
                                f"No bars returned by Polygon for {ticker} "
                                f"between {self.start_date} and {self.end_date}."
                            )

                        rows = []
                        for bar in aggs:
                            rows.append(
                                {
                                    "Date": pd.to_datetime(bar.timestamp, unit="ms", utc=True),
                                    "Open": bar.open,
                                    "High": bar.high,
                                    "Low": bar.low,
                                    "Close": bar.close,
                                    "Volume": bar.volume,
                                }
                            )
                        frame = pd.DataFrame(rows).set_index("Date").sort_index()
                        fetched_data = frame
                        selected_symbol = ticker
                        break
                    except Exception as fetch_error:
                        last_error = fetch_error
                        # Exponential backoff for transient network/API failures.
                        time.sleep(0.5 * (2 ** (attempt - 1)))
                        continue
                if fetched_data is not None and not fetched_data.empty:
                    break

            if fetched_data is None or fetched_data.empty:
                raise ValueError(
                    f"Unable to load data for requested symbol {self.requested_symbol}. "
                    f"Tried candidates: {self._candidate_tickers()}. Last error: {last_error}"
                )
            
            # Calculate daily returns
            fetched_data['Returns'] = fetched_data['Close'].pct_change()
            
            # Calculate 20-day rolling volatility
            fetched_data['Volatility'] = fetched_data['Returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized
            
            # Calculate 20-day EMA
            fetched_data['EMA_20'] = fetched_data['Close'].ewm(span=20, adjust=False).mean()
            
            # Calculate volatility percentile (using expanding window instead of rolling)
            fetched_data['Vol_Percentile'] = fetched_data['Volatility'].expanding().rank(pct=True)

            # Only commit object state after full successful fetch + feature generation.
            self.symbol = selected_symbol
            self.data = fetched_data
            return self.data
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data for {self.symbol}: {e}") from e
            
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the data by adding technical indicators and regime signals.
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        if self.data is None or self.data.empty:
            return pd.DataFrame()
            
        # Calculate standard deviation bands
        self.data['Upper_Band'] = self.data['EMA_20'] + 2 * self.data['Volatility']
        self.data['Lower_Band'] = self.data['EMA_20'] - 2 * self.data['Volatility']
        
        # Drop NaN values
        self.data = self.data.dropna()
        
        return self.data
        
    def get_data(self) -> pd.DataFrame:
        """
        Get the complete processed dataset.
        
        Returns:
            pd.DataFrame: Complete processed dataset
        """
        if self.data is None or self.data.empty:
            fetched = self.fetch_data()
            if fetched is None or fetched.empty:
                raise RuntimeError("Data fetch returned no usable rows.")
            processed = self.preprocess_data()
            if processed is None or processed.empty:
                raise RuntimeError("Data preprocessing returned no usable rows.")
        if self.data is None or self.data.empty:
            raise RuntimeError(
                "Processed dataset is empty. Data fetch/preprocessing did not produce usable rows."
            )
        return self.data