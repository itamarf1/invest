import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
import logging
import time
import os
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class AlphaVantageConfig:
    api_key: str
    base_url: str = "https://www.alphavantage.co/query"
    rate_limit_delay: float = 12.0  # seconds between calls (free tier: 5 calls/min)
    timeout: int = 30
    max_retries: int = 3

class AlphaVantageClient:
    """
    Production-ready Alpha Vantage API client with rate limiting, caching, and error handling
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        
        if not self.api_key:
            logger.warning("No Alpha Vantage API key provided. Get free key at: https://www.alphavantage.co/support/#api-key")
            
        self.config = AlphaVantageConfig(api_key=self.api_key) if self.api_key else None
        self.last_request_time = 0
        self.request_count = 0
        self.daily_request_count = 0
        self.last_reset_date = datetime.now().date()
        
        # Cache for API responses (in-memory, could be extended to Redis/disk)
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes for intraday, longer for daily
        
        logger.info(f"Alpha Vantage client initialized. API key: {'✓' if self.api_key else '✗'}")
    
    def is_available(self) -> bool:
        """Check if Alpha Vantage is available (has API key)"""
        return self.api_key is not None
    
    def _enforce_rate_limits(self):
        """Enforce rate limiting to respect API limits"""
        if not self.is_available():
            raise ValueError("Alpha Vantage API key not available")
            
        # Reset daily counter if it's a new day
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_request_count = 0
            self.last_reset_date = today
        
        # Check daily limit (25 for free tier, 500 for some plans)
        if self.daily_request_count >= 25:
            raise ValueError("Alpha Vantage daily API limit reached (25 requests)")
        
        # Enforce rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
    
    def _make_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Make API request with error handling and retries"""
        self._enforce_rate_limits()
        
        # Add API key to params
        params['apikey'] = self.config.api_key
        
        # Check cache first
        cache_key = str(sorted(params.items()))
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                logger.debug(f"Using cached data for {params.get('symbol', 'unknown')}")
                return cached_data
        
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Alpha Vantage API request (attempt {attempt + 1}): {params.get('function')} for {params.get('symbol')}")
                
                response = requests.get(
                    self.config.base_url,
                    params=params,
                    timeout=self.config.timeout
                )
                
                self.last_request_time = time.time()
                self.request_count += 1
                self.daily_request_count += 1
                
                response.raise_for_status()
                data = response.json()
                
                # Check for API errors
                if 'Error Message' in data:
                    raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
                
                if 'Note' in data:
                    if 'call frequency' in data['Note'] or 'rate limit' in data['Note'].lower():
                        logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                        if attempt < self.config.max_retries - 1:
                            time.sleep(60)  # Wait 1 minute and retry
                            continue
                    raise ValueError(f"Alpha Vantage limit: {data['Note']}")
                
                # Cache successful response
                self._cache[cache_key] = (data, time.time())
                
                return data
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise
        
        raise ValueError("Alpha Vantage API request failed after all retries")
    
    def get_daily_data(self, symbol: str, outputsize: str = "compact") -> pd.DataFrame:
        """
        Get daily time series data
        outputsize: "compact" (latest 100 data points) or "full" (20+ years)
        """
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'outputsize': outputsize
            }
            
            data = self._make_request(params)
            
            # Extract time series
            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data:
                logger.error(f"No daily time series data found for {symbol}")
                return pd.DataFrame()
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            records = []
            for date_str, values in time_series.items():
                records.append({
                    'date': pd.to_datetime(date_str),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })
            
            if not records:
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            df = df.set_index('date').sort_index()
            
            logger.info(f"Retrieved {len(df)} daily records for {symbol} from Alpha Vantage")
            return df
            
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage daily data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_intraday_data(self, symbol: str, interval: str = "5min", 
                         outputsize: str = "compact") -> pd.DataFrame:
        """
        Get intraday time series data
        interval: "1min", "5min", "15min", "30min", "60min"
        outputsize: "compact" or "full"
        """
        try:
            if interval not in ["1min", "5min", "15min", "30min", "60min"]:
                raise ValueError(f"Invalid interval: {interval}")
            
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'outputsize': outputsize
            }
            
            data = self._make_request(params)
            
            # Extract time series
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                logger.error(f"No intraday time series data found for {symbol}")
                return pd.DataFrame()
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            records = []
            for datetime_str, values in time_series.items():
                records.append({
                    'datetime': pd.to_datetime(datetime_str),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })
            
            if not records:
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            df = df.set_index('datetime').sort_index()
            
            logger.info(f"Retrieved {len(df)} {interval} records for {symbol} from Alpha Vantage")
            return df
            
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage intraday data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol"""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol
            }
            
            data = self._make_request(params)
            
            quote_key = 'Global Quote'
            if quote_key not in data:
                logger.error(f"No quote data found for {symbol}")
                return {}
            
            quote = data[quote_key]
            
            return {
                'symbol': quote.get('01. symbol', symbol),
                'open': float(quote.get('02. open', 0)),
                'high': float(quote.get('03. high', 0)),
                'low': float(quote.get('04. low', 0)),
                'price': float(quote.get('05. price', 0)),
                'volume': int(quote.get('06. volume', 0)),
                'latest_trading_day': quote.get('07. latest trading day', ''),
                'previous_close': float(quote.get('08. previous close', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_percent': quote.get('10. change percent', '0%').rstrip('%')
            }
            
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage quote for {symbol}: {str(e)}")
            return {}
    
    def search_symbols(self, keywords: str) -> List[Dict[str, str]]:
        """Search for symbols by keywords"""
        try:
            params = {
                'function': 'SYMBOL_SEARCH',
                'keywords': keywords
            }
            
            data = self._make_request(params)
            
            matches_key = 'bestMatches'
            if matches_key not in data:
                return []
            
            matches = []
            for match in data[matches_key]:
                matches.append({
                    'symbol': match.get('1. symbol', ''),
                    'name': match.get('2. name', ''),
                    'type': match.get('3. type', ''),
                    'region': match.get('4. region', ''),
                    'market_open': match.get('5. marketOpen', ''),
                    'market_close': match.get('6. marketClose', ''),
                    'timezone': match.get('7. timezone', ''),
                    'currency': match.get('8. currency', ''),
                    'match_score': float(match.get('9. matchScore', 0))
                })
            
            return matches
            
        except Exception as e:
            logger.error(f"Error searching symbols for '{keywords}': {str(e)}")
            return []
    
    def get_api_usage_stats(self) -> Dict[str, Any]:
        """Get current API usage statistics"""
        return {
            'api_key_available': self.is_available(),
            'total_requests_today': self.daily_request_count,
            'daily_limit': 25,
            'remaining_requests': max(0, 25 - self.daily_request_count),
            'total_requests_session': self.request_count,
            'last_request_time': datetime.fromtimestamp(self.last_request_time).isoformat() if self.last_request_time > 0 else None,
            'cache_size': len(self._cache)
        }
    
    def clear_cache(self):
        """Clear the response cache"""
        self._cache.clear()
        logger.info("Alpha Vantage response cache cleared")


# Convenience function for easy imports
def get_alpha_vantage_client(api_key: str = None) -> AlphaVantageClient:
    """Get configured Alpha Vantage client"""
    return AlphaVantageClient(api_key)