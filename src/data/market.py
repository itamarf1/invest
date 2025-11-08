import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging
from .multi_source_fetcher import MultiSourceDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataFetcher:
    def __init__(self, cache_expiry_minutes: int = 5):
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_expiry_minutes = cache_expiry_minutes
        self.multi_source_fetcher = MultiSourceDataFetcher(cache_expiry_minutes=30)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[cache_key]
        now = datetime.now()
        expiry = cache_time + timedelta(minutes=self.cache_expiry_minutes)
        
        return now < expiry
    
    def get_stock_data(
        self, 
        symbol: str, 
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        try:
            # Create cache key
            cache_key = f"{symbol}_{period}_{interval}"
            
            # Check cache first
            if cache_key in self.cache and self._is_cache_valid(cache_key):
                logger.info(f"Using cached data for {symbol} ({interval})")
                return self.cache[cache_key]
            
            # For historical data (daily), use multi-source validation for better reliability
            if interval == "1d" and self._should_use_multi_source_validation(symbol, period):
                logger.info(f"Using multi-source validation for {symbol} historical data")
                data = self.multi_source_fetcher.get_validated_stock_data(symbol, period, interval)
            else:
                # For intraday or recent data, use standard yfinance
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
            
            # Cache the data
            self.cache[cache_key] = data
            self.cache_timestamps[cache_key] = datetime.now()
            
            logger.info(f"Retrieved {len(data)} rows for {symbol} ({interval}, cached for {self.cache_expiry_minutes}min)")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _should_use_multi_source_validation(self, symbol: str, period: str) -> bool:
        """Determine if multi-source validation should be used"""
        # Always use for warrants (high risk of stale data)
        if symbol.endswith('W'):
            return True
        
        # Use for longer historical periods where data quality matters
        if period in ["6mo", "1y", "2y"]:
            return True
        
        # For penny stocks or low-volume securities
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            if price and price < 5.0:  # Penny stock threshold
                return True
        except:
            pass
        
        return False
    
    def get_multiple_stocks(
        self, 
        symbols: List[str], 
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_stock_data(symbol, period)
        return results
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('currentPrice') or info.get('regularMarketPrice')
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {str(e)}")
            return {}
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data
        
        data_copy = data.copy()
        data_copy['Daily_Return'] = data_copy['Close'].pct_change()
        data_copy['Cumulative_Return'] = (1 + data_copy['Daily_Return']).cumprod() - 1
        return data_copy
    
    def get_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data
        
        data_copy = data.copy()
        
        data_copy['SMA_20'] = data_copy['Close'].rolling(window=20).mean()
        data_copy['SMA_50'] = data_copy['Close'].rolling(window=50).mean()
        data_copy['SMA_200'] = data_copy['Close'].rolling(window=200).mean()
        
        data_copy['EMA_12'] = data_copy['Close'].ewm(span=12).mean()
        data_copy['EMA_26'] = data_copy['Close'].ewm(span=26).mean()
        data_copy['MACD'] = data_copy['EMA_12'] - data_copy['EMA_26']
        
        delta = data_copy['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data_copy['RSI'] = 100 - (100 / (1 + rs))
        
        data_copy['Volatility'] = data_copy['Close'].rolling(window=20).std()
        
        return data_copy
    
    def get_data_quality_report(self, symbol: str) -> Dict[str, Any]:
        """Get data quality assessment for a symbol"""
        return self.multi_source_fetcher.get_data_quality_report(symbol)


