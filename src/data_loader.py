import io
import logging
import time

import numpy as np
import pandas as pd
import requests

from src.config import get_settings

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, symbol: str, start_date: str, end_date: str, sources: tuple[str, ...] | None = None):
        """
        Initialize the data loader with symbol and date range.
        
        Args:
            symbol (str): Stock/ETF symbol (e.g., 'SPY')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            sources: Provider order. None uses Massive, Databento, and Polygon when their keys are set, then Yahoo Finance, then Stooq.
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.requested_symbol = symbol
        self.sources = sources
        self.data_source: str | None = None

    def _candidate_tickers(self) -> list[str]:
        symbol = self.requested_symbol.strip().upper()
        aliases = {
            # WTI workflows default to liquid proxy to avoid futures entitlement/rate issues.
            "WTI": ["USO"],
            "CL": ["USO"],
            "OIL": ["USO"],
        }
        return aliases.get(symbol, [symbol])

    def _provider_order(self) -> list[str]:
        if self.sources is not None:
            return [name.strip().lower() for name in self.sources if name.strip()]
        settings = get_settings()
        order = ["yfinance", "stooq"]
        # Insert front-to-back so a configured key is preferred in this order:
        # Massive, Databento, Polygon, then the keyless fallbacks above.
        if settings.polygon_api_key:
            order.insert(0, "polygon")
        if settings.databento_api_key:
            order.insert(0, "databento")
        if settings.massive_api_key:
            order.insert(0, "massive")
        return order

    def _window_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        start = pd.Timestamp(self.start_date, tz="UTC")
        end = pd.Timestamp(self.end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return start, end

    def _normalize_ohlcv(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(0)
        lookup = {str(col).lower(): col for col in out.columns}
        missing = [name for name in ("open", "high", "low", "close", "volume") if name not in lookup]
        if missing:
            raise ValueError(f"Bars are missing columns: {missing}")
        out = pd.DataFrame(
            {
                "Open": out[lookup["open"]],
                "High": out[lookup["high"]],
                "Low": out[lookup["low"]],
                "Close": out[lookup["close"]],
                "Volume": out[lookup["volume"]],
            },
            index=out.index,
        )
        out.index = pd.to_datetime(out.index, utc=True)
        out.index.name = "Date"
        out = out.apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"]).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        start, end = self._window_bounds()
        out = out.loc[(out.index >= start) & (out.index <= end)]
        if out.empty:
            raise ValueError(
                f"No bars left for {self.requested_symbol} between {self.start_date} and {self.end_date}."
            )
        return out

    def _attach_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        featured = frame.copy()
        featured["Returns"] = featured["Close"].pct_change()
        featured["Volatility"] = featured["Returns"].rolling(window=20).std() * np.sqrt(252)
        featured["EMA_20"] = featured["Close"].ewm(span=20, adjust=False).mean()
        rolling_pct = featured["Volatility"].rolling(252, min_periods=60).rank(pct=True)
        expanding_pct = featured["Volatility"].expanding().rank(pct=True)
        featured["Vol_Percentile"] = rolling_pct.fillna(expanding_pct)
        return featured

    def _frame_from_aggregate_bars(self, bars, source_name: str, ticker: str) -> pd.DataFrame:
        if not bars:
            raise ValueError(
                f"No bars returned by {source_name} for {ticker} "
                f"between {self.start_date} and {self.end_date}."
            )
        rows = [
            {
                "Date": pd.to_datetime(bar.timestamp, unit="ms", utc=True),
                "Open": bar.open,
                "High": bar.high,
                "Low": bar.low,
                "Close": bar.close,
                "Volume": bar.volume,
            }
            for bar in bars
        ]
        return self._normalize_ohlcv(pd.DataFrame(rows).set_index("Date"))

    def _fetch_massive(self) -> tuple[str, pd.DataFrame]:
        try:
            from massive import RESTClient
        except ImportError as exc:
            raise RuntimeError("massive is not installed.") from exc

        api_key = get_settings().massive_api_key
        if not api_key:
            raise RuntimeError("Missing MASSIVE_API_KEY. Set it in your environment or .env file.")
        client = RESTClient(api_key=api_key)
        last_error: Exception | None = None
        for ticker in self._candidate_tickers():
            for attempt in range(1, 4):
                try:
                    logger.info("Fetching %s from Massive attempt %s", ticker, attempt)
                    aggs = list(
                        client.list_aggs(
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
                    return ticker, self._frame_from_aggregate_bars(aggs, "Massive", ticker)
                except Exception as fetch_error:
                    last_error = fetch_error
                    time.sleep(0.5 * (2 ** (attempt - 1)))
        raise RuntimeError(f"Massive failed for {self._candidate_tickers()}: {last_error}")

    def _fetch_databento(self) -> tuple[str, pd.DataFrame]:
        try:
            import databento as db
        except ImportError as exc:
            raise RuntimeError("databento is not installed.") from exc

        settings = get_settings()
        if not settings.databento_api_key:
            raise RuntimeError("Missing DATABENTO_API_KEY. Set it in your environment or .env file.")
        client = db.Historical(settings.databento_api_key)
        # get_range treats end as exclusive.
        end = (pd.Timestamp(self.end_date) + pd.Timedelta(days=1)).date().isoformat()
        last_error: Exception | None = None
        for ticker in self._candidate_tickers():
            try:
                logger.info("Fetching %s from Databento dataset %s", ticker, settings.databento_dataset)
                store = client.timeseries.get_range(
                    dataset=settings.databento_dataset,
                    symbols=[ticker],
                    schema="ohlcv-1d",
                    start=self.start_date,
                    end=end,
                )
                frame = store.to_df()
                if frame is None or frame.empty:
                    raise ValueError(f"Databento returned no bars for {ticker}.")
                return ticker, self._normalize_ohlcv(frame)
            except Exception as fetch_error:
                last_error = fetch_error
        raise RuntimeError(f"Databento failed for {self._candidate_tickers()}: {last_error}")

    def _fetch_polygon(self) -> tuple[str, pd.DataFrame]:
        from polygon import RESTClient

        api_key = get_settings().polygon_api_key
        if not api_key:
            raise RuntimeError("Missing POLYGON_API_KEY. Set it in your environment or .env file.")
        client = RESTClient(api_key=api_key)
        last_error: Exception | None = None
        for ticker in self._candidate_tickers():
            for attempt in range(1, 4):
                try:
                    logger.info("Fetching %s from Polygon attempt %s", ticker, attempt)
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
                    return ticker, self._frame_from_aggregate_bars(aggs, "Polygon", ticker)
                except Exception as fetch_error:
                    last_error = fetch_error
                    time.sleep(0.5 * (2 ** (attempt - 1)))
        raise RuntimeError(f"Polygon failed for {self._candidate_tickers()}: {last_error}")

    def _fetch_yfinance(self) -> tuple[str, pd.DataFrame]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance is not installed.") from exc

        end = (pd.Timestamp(self.end_date) + pd.Timedelta(days=1)).date().isoformat()
        last_error: Exception | None = None
        for ticker in self._candidate_tickers():
            try:
                logger.info("Fetching %s from Yahoo Finance", ticker)
                raw = yf.Ticker(ticker).history(
                    start=self.start_date,
                    end=end,
                    interval="1d",
                    auto_adjust=True,
                )
                if raw is None or raw.empty:
                    raise ValueError(f"Yahoo Finance returned no bars for {ticker}.")
                return ticker, self._normalize_ohlcv(raw)
            except Exception as fetch_error:
                last_error = fetch_error
        raise RuntimeError(f"Yahoo Finance failed for {self._candidate_tickers()}: {last_error}")

    def _fetch_stooq(self) -> tuple[str, pd.DataFrame]:
        last_error: Exception | None = None
        start_token = self.start_date.replace("-", "")
        end_token = self.end_date.replace("-", "")
        for ticker in self._candidate_tickers():
            stooq_symbol = ticker.lower() if "." in ticker else f"{ticker.lower()}.us"
            url = (
                "https://stooq.com/q/d/l/"
                f"?s={stooq_symbol}&d1={start_token}&d2={end_token}&i=d"
            )
            try:
                logger.info("Fetching %s from Stooq", stooq_symbol)
                response = requests.get(url, timeout=20)
                response.raise_for_status()
                body = response.text.strip()
                if not body.lower().startswith("date,"):
                    raise ValueError(f"Stooq returned no daily bars for {stooq_symbol}.")
                raw = pd.read_csv(io.StringIO(body), parse_dates=["Date"]).set_index("Date")
                return ticker, self._normalize_ohlcv(raw)
            except Exception as fetch_error:
                last_error = fetch_error
        raise RuntimeError(f"Stooq failed for {self._candidate_tickers()}: {last_error}")

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars. Configured Massive, Databento, and Polygon keys are tried first, then Yahoo Finance, then Stooq.
        """
        providers = {
            "massive": self._fetch_massive,
            "databento": self._fetch_databento,
            "polygon": self._fetch_polygon,
            "yfinance": self._fetch_yfinance,
            "stooq": self._fetch_stooq,
        }
        order = self._provider_order()
        unknown = [name for name in order if name not in providers]
        if unknown:
            raise ValueError(f"Unknown data sources: {unknown}")

        errors: list[str] = []
        for name in order:
            try:
                selected_symbol, bars = providers[name]()
                featured = self._attach_features(bars)
                self.symbol = selected_symbol
                self.data_source = name
                self.data = featured
                logger.info("Loaded %s from %s (%s rows)", selected_symbol, name, len(featured))
                return self.data
            except Exception as fetch_error:
                errors.append(f"{name}: {fetch_error}")
                logger.warning("Data source %s failed for %s: %s", name, self.requested_symbol, fetch_error)

        raise RuntimeError(
            f"Failed to fetch data for {self.requested_symbol}. Tried {order}. " + " | ".join(errors)
        )
            
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the data by adding technical indicators and regime signals.
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        if self.data is None or self.data.empty:
            return pd.DataFrame()
            
        # Bands are in price units. Volatility is an annualized return, so it
        # cannot be added to the EMA directly.
        price_std = self.data['Close'].rolling(window=20).std()
        self.data['Upper_Band'] = self.data['EMA_20'] + 2 * price_std
        self.data['Lower_Band'] = self.data['EMA_20'] - 2 * price_std
        
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