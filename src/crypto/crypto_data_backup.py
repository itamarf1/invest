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
class CryptoTrade:
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    fee: float
    fee_currency: str
    order_type: str

class CryptoDataFetcher:
    """Enhanced cryptocurrency data fetcher using Binance API with CoinGecko fallback"""
    
    def __init__(self):
        # Import here to avoid circular imports
        from src.crypto.binance_data_fetcher import BinanceDataFetcher
        self.binance_fetcher = BinanceDataFetcher()
        
        # Keep CoinGecko as fallback
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        self.coingecko_api_key = os.getenv('COINGECKO_API_KEY')
        
        # Symbol mapping for both APIs
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
    """Fetch cryptocurrency data using free APIs"""
    
    def __init__(self):
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        self.binance_base_url = "https://api.binance.com/api/v3"
        self.rate_limit_delay = 1.0  # seconds between requests
        
        # API keys (optional for higher rate limits)
        self.coingecko_api_key = os.getenv('COINGECKO_API_KEY')
        
        # Common cryptocurrency symbols mapping
        self.symbol_mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum', 
            'BNB': 'binancecoin',
            'ADA': 'cardano',
            'SOL': 'solana',
            'XRP': 'ripple',
            'DOT': 'polkadot',
            'DOGE': 'dogecoin',
            'AVAX': 'avalanche-2',
            'SHIB': 'shiba-inu',
            'MATIC': 'matic-network',
            'LINK': 'chainlink',
            'UNI': 'uniswap',
            'LTC': 'litecoin',
            'BCH': 'bitcoin-cash'
        }
        
    def get_crypto_price(self, symbol: str) -> Optional[CryptoPriceData]:
        """Get current price data for a cryptocurrency"""
        try:
            # Convert symbol to CoinGecko ID
            coin_id = self.symbol_mapping.get(symbol.upper())
            if not coin_id:
                logger.warning(f"Unknown crypto symbol: {symbol}")
                return self._get_mock_crypto_data(symbol)
            
            # Fetch from CoinGecko
            url = f"{self.coingecko_base_url}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            if self.coingecko_api_key:
                params['x_cg_demo_api_key'] = self.coingecko_api_key
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if coin_id not in data:
                return self._get_mock_crypto_data(symbol)
            
            coin_data = data[coin_id]
            
            return CryptoPriceData(
                symbol=symbol.upper(),
                price=coin_data['usd'],
                change_24h=coin_data.get('usd_24h_change', 0),
                change_24h_pct=coin_data.get('usd_24h_change', 0),
                volume_24h=coin_data.get('usd_24h_vol', 0),
                market_cap=coin_data.get('usd_market_cap', 0),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error fetching crypto price for {symbol}: {str(e)}")
            return self._get_mock_crypto_data(symbol)
    
    def get_multiple_crypto_prices(self, symbols: List[str]) -> Dict[str, CryptoPriceData]:
        """Get price data for multiple cryptocurrencies"""
        try:
            # Map symbols to CoinGecko IDs
            coin_ids = []
            symbol_to_id = {}
            
            for symbol in symbols:
                coin_id = self.symbol_mapping.get(symbol.upper())
                if coin_id:
                    coin_ids.append(coin_id)
                    symbol_to_id[coin_id] = symbol.upper()
            
            if not coin_ids:
                return {symbol: self._get_mock_crypto_data(symbol) for symbol in symbols}
            
            # Fetch from CoinGecko
            url = f"{self.coingecko_base_url}/simple/price"
            params = {
                'ids': ','.join(coin_ids),
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            if self.coingecko_api_key:
                params['x_cg_demo_api_key'] = self.coingecko_api_key
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = {}
            for coin_id, coin_data in data.items():
                symbol = symbol_to_id.get(coin_id)
                if symbol:
                    results[symbol] = CryptoPriceData(
                        symbol=symbol,
                        price=coin_data['usd'],
                        change_24h=coin_data.get('usd_24h_change', 0),
                        change_24h_pct=coin_data.get('usd_24h_change', 0),
                        volume_24h=coin_data.get('usd_24h_vol', 0),
                        market_cap=coin_data.get('usd_market_cap', 0),
                        timestamp=datetime.now()
                    )
            
            # Add mock data for missing symbols
            for symbol in symbols:
                if symbol.upper() not in results:
                    results[symbol.upper()] = self._get_mock_crypto_data(symbol)
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching multiple crypto prices: {str(e)}")
            return {symbol: self._get_mock_crypto_data(symbol) for symbol in symbols}
    
    def get_crypto_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical price data for a cryptocurrency"""
        try:
            coin_id = self.symbol_mapping.get(symbol.upper())
            if not coin_id:
                return self._get_mock_historical_data(symbol, days)
            
            url = f"{self.coingecko_base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days
            }
            
            if self.coingecko_api_key:
                params['x_cg_demo_api_key'] = self.coingecko_api_key
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            if not prices:
                return self._get_mock_historical_data(symbol, days)
            
            df_data = []
            for i, price_point in enumerate(prices):
                timestamp = datetime.fromtimestamp(price_point[0] / 1000)
                price = price_point[1]
                volume = volumes[i][1] if i < len(volumes) else 0
                
                df_data.append({
                    'timestamp': timestamp,
                    'price': price,
                    'volume': volume
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return self._get_mock_historical_data(symbol, days)
    
    def get_crypto_market_overview(self) -> Dict[str, Any]:
        """Get overall cryptocurrency market data"""
        try:
            url = f"{self.coingecko_base_url}/global"
            
            params = {}
            if self.coingecko_api_key:
                params['x_cg_demo_api_key'] = self.coingecko_api_key
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            market_data = data.get('data', {})
            
            return {
                'total_market_cap_usd': market_data.get('total_market_cap', {}).get('usd', 0),
                'total_volume_24h_usd': market_data.get('total_volume', {}).get('usd', 0),
                'market_cap_change_24h': market_data.get('market_cap_change_percentage_24h_usd', 0),
                'active_cryptocurrencies': market_data.get('active_cryptocurrencies', 0),
                'markets': market_data.get('markets', 0),
                'bitcoin_dominance': market_data.get('market_cap_percentage', {}).get('btc', 0),
                'ethereum_dominance': market_data.get('market_cap_percentage', {}).get('eth', 0),
                'updated_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching market overview: {str(e)}")
            return self._get_mock_market_overview()
    
    def get_top_cryptocurrencies(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get list of top cryptocurrencies by market cap"""
        try:
            url = f"{self.coingecko_base_url}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': limit,
                'page': 1,
                'sparkline': 'false'
            }
            
            if self.coingecko_api_key:
                params['x_cg_demo_api_key'] = self.coingecko_api_key
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            cryptocurrencies = []
            for coin in data:
                cryptocurrencies.append({
                    'id': coin['id'],
                    'symbol': coin['symbol'].upper(),
                    'name': coin['name'],
                    'current_price': coin['current_price'],
                    'market_cap': coin['market_cap'],
                    'market_cap_rank': coin['market_cap_rank'],
                    'total_volume': coin['total_volume'],
                    'price_change_24h': coin['price_change_24h'],
                    'price_change_percentage_24h': coin['price_change_percentage_24h'],
                    'circulating_supply': coin['circulating_supply'],
                    'total_supply': coin['total_supply'],
                    'max_supply': coin['max_supply']
                })
            
            return cryptocurrencies
            
        except Exception as e:
            logger.error(f"Error fetching top cryptocurrencies: {str(e)}")
            return self._get_mock_top_cryptos(limit)
    
    def _get_mock_crypto_data(self, symbol: str) -> CryptoPriceData:
        """Generate mock crypto data for testing"""
        base_prices = {
            'BTC': 65000, 'ETH': 3500, 'BNB': 600, 'ADA': 1.2, 'SOL': 150,
            'XRP': 0.6, 'DOT': 25, 'DOGE': 0.08, 'AVAX': 35, 'SHIB': 0.000025,
            'MATIC': 1.1, 'LINK': 15, 'UNI': 12, 'LTC': 180, 'BCH': 450
        }
        
        base_price = base_prices.get(symbol.upper(), 100)
        change_pct = np.random.uniform(-10, 10)
        
        return CryptoPriceData(
            symbol=symbol.upper(),
            price=base_price * (1 + change_pct / 100),
            change_24h=base_price * change_pct / 100,
            change_24h_pct=change_pct,
            volume_24h=np.random.uniform(1e9, 50e9),
            market_cap=np.random.uniform(1e9, 1000e9),
            timestamp=datetime.now()
        )
    
    def _get_mock_historical_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate mock historical data"""
        base_price = 100 if symbol.upper() not in ['BTC', 'ETH'] else (50000 if symbol.upper() == 'BTC' else 3000)
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        prices = []
        current_price = base_price
        
        for _ in range(days):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            current_price *= (1 + change)
            prices.append(current_price)
        
        volumes = np.random.uniform(1e9, 10e9, days)
        
        return pd.DataFrame({
            'price': prices,
            'volume': volumes
        }, index=dates)
    
    def _get_mock_market_overview(self) -> Dict[str, Any]:
        """Generate mock market overview"""
        return {
            'total_market_cap_usd': 2.5e12,
            'total_volume_24h_usd': 100e9,
            'market_cap_change_24h': np.random.uniform(-5, 5),
            'active_cryptocurrencies': 13000,
            'markets': 800,
            'bitcoin_dominance': 45.0,
            'ethereum_dominance': 18.0,
            'updated_at': datetime.now()
        }
    
    def _get_mock_top_cryptos(self, limit: int) -> List[Dict[str, Any]]:
        """Generate mock top cryptocurrencies list"""
        mock_cryptos = [
            {'symbol': 'BTC', 'name': 'Bitcoin', 'price': 65000, 'market_cap': 1.2e12},
            {'symbol': 'ETH', 'name': 'Ethereum', 'price': 3500, 'market_cap': 400e9},
            {'symbol': 'BNB', 'name': 'BNB', 'price': 600, 'market_cap': 90e9},
            {'symbol': 'ADA', 'name': 'Cardano', 'price': 1.2, 'market_cap': 40e9},
            {'symbol': 'SOL', 'name': 'Solana', 'price': 150, 'market_cap': 35e9},
        ]
        
        results = []
        for i, crypto in enumerate(mock_cryptos[:limit]):
            results.append({
                'id': crypto['name'].lower(),
                'symbol': crypto['symbol'],
                'name': crypto['name'],
                'current_price': crypto['price'],
                'market_cap': crypto['market_cap'],
                'market_cap_rank': i + 1,
                'total_volume': np.random.uniform(1e9, 10e9),
                'price_change_24h': np.random.uniform(-1000, 1000),
                'price_change_percentage_24h': np.random.uniform(-10, 10),
                'circulating_supply': np.random.uniform(1e6, 1e9),
                'total_supply': np.random.uniform(1e6, 1e9),
                'max_supply': np.random.uniform(1e6, 1e9)
            })
        
        return results
