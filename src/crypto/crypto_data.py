import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import time
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class CryptoPriceData:
    symbol: str
    price: float
    change_24h: float
    change_24h_pct: float
    volume_24h: float
    market_cap: float
    timestamp: datetime
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    supply_circulating: Optional[float] = None
    supply_total: Optional[float] = None

@dataclass
class CryptoTransaction:
    transaction_id: str
    symbol: str
    transaction_type: str  # 'buy' or 'sell'
    amount: float
    price: float
    total_value: float
    timestamp: datetime
    fee_amount: float
    fee_currency: str
    order_type: str

class CryptoDataFetcher:
    """Enhanced cryptocurrency data fetcher using Binance API (real data) with CoinGecko fallback"""
    
    def __init__(self):
        # Import here to avoid circular imports
        from src.crypto.binance_data_fetcher import BinanceDataFetcher
        self.binance_fetcher = BinanceDataFetcher()
        
        # Keep CoinGecko as fallback
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        self.coingecko_api_key = os.getenv('COINGECKO_API_KEY')
        self.session = requests.Session()
        
        # Symbol mapping for CoinGecko
        self.symbol_mapping = {
            'BTC': 'bitcoin',
            'BITCOIN': 'bitcoin', 
            'ETH': 'ethereum',
            'ETHEREUM': 'ethereum',
            'BNB': 'binancecoin',
            'ADA': 'cardano',
            'CARDANO': 'cardano',
            'SOL': 'solana',
            'SOLANA': 'solana',
            'XRP': 'ripple',
            'RIPPLE': 'ripple',
            'DOT': 'polkadot',
            'POLKADOT': 'polkadot',
            'DOGE': 'dogecoin',
            'DOGECOIN': 'dogecoin',
            'AVAX': 'avalanche-2',
            'AVALANCHE': 'avalanche-2',
            'SHIB': 'shiba-inu',
            'MATIC': 'matic-network',
            'POLYGON': 'matic-network',
            'LTC': 'litecoin',
            'LITECOIN': 'litecoin',
            'UNI': 'uniswap',
            'UNISWAP': 'uniswap'
        }
    
    def get_crypto_price(self, symbol: str) -> Optional[CryptoPriceData]:
        """Get current price data for a cryptocurrency using Binance (primary) or CoinGecko (fallback)"""
        try:
            # First try Binance (real-time, free, no API key needed)
            binance_data = self.binance_fetcher.get_crypto_price(symbol)
            if binance_data:
                # Convert to our format
                return CryptoPriceData(
                    symbol=symbol.upper(),
                    price=binance_data.price,
                    change_24h=binance_data.price * (binance_data.change_24h / 100),
                    change_24h_pct=binance_data.change_24h,
                    volume_24h=binance_data.volume_24h,
                    market_cap=binance_data.market_cap,
                    timestamp=binance_data.timestamp,
                    high_24h=binance_data.high_24h,
                    low_24h=binance_data.low_24h
                )
            
            # Fallback to CoinGecko
            logger.info(f"Binance data not available for {symbol}, trying CoinGecko")
            return self._get_coingecko_price(symbol)
            
        except Exception as e:
            logger.error(f"Error getting crypto price for {symbol}: {str(e)}")
            return None
    
    def get_multiple_crypto_prices(self, symbols: List[str]) -> Dict[str, CryptoPriceData]:
        """Get price data for multiple cryptocurrencies efficiently"""
        try:
            # Try Binance first (much faster for multiple symbols)
            binance_data = self.binance_fetcher.get_multiple_crypto_prices(symbols)
            results = {}
            
            # Convert Binance data to our format
            for symbol, data in binance_data.items():
                results[symbol] = CryptoPriceData(
                    symbol=symbol,
                    price=data.price,
                    change_24h=data.price * (data.change_24h / 100),
                    change_24h_pct=data.change_24h,
                    volume_24h=data.volume_24h,
                    market_cap=data.market_cap,
                    timestamp=data.timestamp,
                    high_24h=data.high_24h,
                    low_24h=data.low_24h
                )
            
            # For symbols not found in Binance, try CoinGecko
            missing_symbols = [s for s in symbols if s.upper() not in results]
            if missing_symbols:
                logger.info(f"Getting {len(missing_symbols)} symbols from CoinGecko fallback")
                for symbol in missing_symbols:
                    coingecko_data = self._get_coingecko_price(symbol)
                    if coingecko_data:
                        results[symbol.upper()] = coingecko_data
                    time.sleep(0.1)  # Rate limiting for CoinGecko
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting multiple crypto prices: {str(e)}")
            return {}
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical price data using Binance (primary) or CoinGecko (fallback)"""
        try:
            # Try Binance first (more granular data)
            binance_data = self.binance_fetcher.get_historical_data(symbol, days)
            if not binance_data.empty:
                return binance_data
            
            # Fallback to CoinGecko
            logger.info(f"Binance historical data not available for {symbol}, trying CoinGecko")
            return self._get_coingecko_historical(symbol, days)
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get cryptocurrency market overview using Binance (primary) or CoinGecko (fallback)"""
        try:
            # Try Binance first
            binance_overview = self.binance_fetcher.get_market_overview()
            if binance_overview:
                return binance_overview
            
            # Fallback to CoinGecko
            logger.info("Binance market overview not available, trying CoinGecko")
            return self._get_coingecko_market_overview()
            
        except Exception as e:
            logger.error(f"Error getting market overview: {str(e)}")
            return {}
    
    def get_top_cryptocurrencies(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get top cryptocurrencies by market cap using Binance (primary) or CoinGecko (fallback)"""
        try:
            # Try Binance first (sorted by volume)
            binance_top = self.binance_fetcher.get_top_cryptocurrencies(limit)
            if binance_top:
                return binance_top
            
            # Fallback to CoinGecko (sorted by market cap)
            logger.info("Binance top cryptos not available, trying CoinGecko")
            return self._get_coingecko_top_cryptos(limit)
            
        except Exception as e:
            logger.error(f"Error getting top cryptocurrencies: {str(e)}")
            return []
    
    def _get_coingecko_price(self, symbol: str) -> Optional[CryptoPriceData]:
        """Get price data from CoinGecko API"""
        try:
            # Convert symbol to CoinGecko ID
            coin_id = self.symbol_mapping.get(symbol.upper())
            if not coin_id:
                logger.warning(f"Unknown crypto symbol for CoinGecko: {symbol}")
                return None
            
            # Fetch from CoinGecko
            url = f"{self.coingecko_base_url}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true',
                'include_last_updated_at': 'true'
            }
            
            if self.coingecko_api_key:
                params['x_cg_demo_api_key'] = self.coingecko_api_key
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if coin_id not in data:
                return None
            
            coin_data = data[coin_id]
            
            return CryptoPriceData(
                symbol=symbol.upper(),
                price=coin_data.get('usd', 0),
                change_24h=coin_data.get('usd') * (coin_data.get('usd_24h_change', 0) / 100),
                change_24h_pct=coin_data.get('usd_24h_change', 0),
                volume_24h=coin_data.get('usd_24h_vol', 0),
                market_cap=coin_data.get('usd_market_cap', 0),
                timestamp=datetime.fromtimestamp(coin_data.get('last_updated_at', time.time()))
            )
            
        except Exception as e:
            logger.error(f"Error getting CoinGecko price for {symbol}: {str(e)}")
            return None
    
    def _get_coingecko_historical(self, symbol: str, days: int) -> pd.DataFrame:
        """Get historical data from CoinGecko API"""
        try:
            coin_id = self.symbol_mapping.get(symbol.upper())
            if not coin_id:
                return pd.DataFrame()
            
            url = f"{self.coingecko_base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days
            }
            
            if self.coingecko_api_key:
                params['x_cg_demo_api_key'] = self.coingecko_api_key
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if 'prices' not in data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            timestamps = [datetime.fromtimestamp(item[0] / 1000) for item in data['prices']]
            prices = [item[1] for item in data['prices']]
            
            df = pd.DataFrame({
                'Close': prices
            }, index=timestamps)
            
            # Add OHLV data (simplified - CoinGecko only provides close prices)
            df['Open'] = df['Close'].shift(1).fillna(df['Close'])
            df['High'] = df['Close']
            df['Low'] = df['Close']
            df['Volume'] = 1000000  # Placeholder volume
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            logger.error(f"Error getting CoinGecko historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _get_coingecko_market_overview(self) -> Dict[str, Any]:
        """Get market overview from CoinGecko API"""
        try:
            url = f"{self.coingecko_base_url}/global"
            params = {}
            
            if self.coingecko_api_key:
                params['x_cg_demo_api_key'] = self.coingecko_api_key
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            global_data = data.get('data', {})
            
            return {
                'total_market_cap': global_data.get('total_market_cap', {}).get('usd', 0),
                'total_volume_24h': global_data.get('total_volume', {}).get('usd', 0),
                'btc_dominance': global_data.get('market_cap_percentage', {}).get('btc', 0),
                'active_cryptocurrencies': global_data.get('active_cryptocurrencies', 0),
                'market_sentiment': 'neutral',
                'top_gainers': [],
                'top_losers': []
            }
            
        except Exception as e:
            logger.error(f"Error getting CoinGecko market overview: {str(e)}")
            return {}
    
    def _get_coingecko_top_cryptos(self, limit: int) -> List[Dict[str, Any]]:
        """Get top cryptocurrencies from CoinGecko API"""
        try:
            url = f"{self.coingecko_base_url}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': limit,
                'page': 1
            }
            
            if self.coingecko_api_key:
                params['x_cg_demo_api_key'] = self.coingecko_api_key
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            top_cryptos = []
            for i, coin in enumerate(data):
                top_cryptos.append({
                    'rank': coin.get('market_cap_rank', i + 1),
                    'symbol': coin.get('symbol', '').upper(),
                    'name': coin.get('name', ''),
                    'price': coin.get('current_price', 0),
                    'change_24h': coin.get('price_change_percentage_24h', 0),
                    'volume_24h': coin.get('total_volume', 0),
                    'market_cap': coin.get('market_cap', 0)
                })
            
            return top_cryptos
            
        except Exception as e:
            logger.error(f"Error getting CoinGecko top cryptocurrencies: {str(e)}")
            return []
