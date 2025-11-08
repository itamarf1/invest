import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import time
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class CryptoPriceData:
    symbol: str
    price: float
    change_24h: float
    volume_24h: float
    market_cap: float
    high_24h: float
    low_24h: float
    timestamp: datetime

class BinanceDataFetcher:
    """Enhanced cryptocurrency data fetcher using Binance API (completely free)"""
    
    def __init__(self):
        self.binance_base_url = "https://api.binance.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Investment-Dashboard/1.0'
        })
        
        # Popular crypto symbols mapping
        self.symbol_mapping = {
            'BTC': 'BTCUSDT',
            'BITCOIN': 'BTCUSDT',
            'ETH': 'ETHUSDT', 
            'ETHEREUM': 'ETHUSDT',
            'BNB': 'BNBUSDT',
            'ADA': 'ADAUSDT',
            'CARDANO': 'ADAUSDT',
            'SOL': 'SOLUSDT',
            'SOLANA': 'SOLUSDT',
            'XRP': 'XRPUSDT',
            'RIPPLE': 'XRPUSDT',
            'DOT': 'DOTUSDT',
            'POLKADOT': 'DOTUSDT',
            'DOGE': 'DOGEUSDT',
            'DOGECOIN': 'DOGEUSDT',
            'AVAX': 'AVAXUSDT',
            'AVALANCHE': 'AVAXUSDT',
            'SHIB': 'SHIBUSDT',
            'MATIC': 'MATICUSDT',
            'POLYGON': 'MATICUSDT',
            'LTC': 'LTCUSDT',
            'LITECOIN': 'LTCUSDT',
            'UNI': 'UNIUSDT',
            'UNISWAP': 'UNIUSDT',
            'ATOM': 'ATOMUSDT',
            'COSMOS': 'ATOMUSDT',
            'LINK': 'LINKUSDT',
            'CHAINLINK': 'LINKUSDT',
            'BCH': 'BCHUSDT',
            'XLM': 'XLMUSDT',
            'STELLAR': 'XLMUSDT',
            'ALGO': 'ALGOUSDT',
            'ALGORAND': 'ALGOUSDT',
            'VET': 'VETUSDT',
            'VECHAIN': 'VETUSDT',
            'ICP': 'ICPUSDT',
            'INTERNET_COMPUTER': 'ICPUSDT',
            'FIL': 'FILUSDT',
            'FILECOIN': 'FILUSDT',
            'TRX': 'TRXUSDT',
            'TRON': 'TRXUSDT',
            'ETC': 'ETCUSDT',
            'ETHEREUM_CLASSIC': 'ETCUSDT'
        }
    
    def get_crypto_price(self, symbol: str) -> Optional[CryptoPriceData]:
        """Get current price data for a cryptocurrency from Binance"""
        try:
            # Convert symbol to Binance format
            binance_symbol = self.symbol_mapping.get(symbol.upper())
            if not binance_symbol:
                # Try direct symbol
                binance_symbol = f"{symbol.upper()}USDT"
            
            # Get 24hr ticker statistics
            url = f"{self.binance_base_url}/ticker/24hr"
            params = {'symbol': binance_symbol}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            return CryptoPriceData(
                symbol=symbol.upper(),
                price=float(data['lastPrice']),
                change_24h=float(data['priceChangePercent']),
                volume_24h=float(data['volume']),
                market_cap=0,  # Binance doesn't provide market cap directly
                high_24h=float(data['highPrice']),
                low_24h=float(data['lowPrice']),
                timestamp=datetime.now()
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching {symbol} from Binance: {str(e)}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Data parsing error for {symbol}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol}: {str(e)}")
            return None
    
    def get_multiple_crypto_prices(self, symbols: List[str]) -> Dict[str, CryptoPriceData]:
        """Get price data for multiple cryptocurrencies efficiently"""
        try:
            # Get all 24hr ticker statistics in one call
            url = f"{self.binance_base_url}/ticker/24hr"
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            all_tickers = response.json()
            
            # Create a mapping of Binance symbols to our symbols
            binance_to_symbol = {}
            for symbol in symbols:
                binance_symbol = self.symbol_mapping.get(symbol.upper())
                if not binance_symbol:
                    binance_symbol = f"{symbol.upper()}USDT"
                binance_to_symbol[binance_symbol] = symbol.upper()
            
            results = {}
            for ticker in all_tickers:
                binance_symbol = ticker['symbol']
                if binance_symbol in binance_to_symbol:
                    symbol = binance_to_symbol[binance_symbol]
                    try:
                        results[symbol] = CryptoPriceData(
                            symbol=symbol,
                            price=float(ticker['lastPrice']),
                            change_24h=float(ticker['priceChangePercent']),
                            volume_24h=float(ticker['volume']),
                            market_cap=0,
                            high_24h=float(ticker['highPrice']),
                            low_24h=float(ticker['lowPrice']),
                            timestamp=datetime.now()
                        )
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Error parsing data for {symbol}: {str(e)}")
                        continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching multiple crypto prices: {str(e)}")
            # Fallback to individual requests
            results = {}
            for symbol in symbols:
                price_data = self.get_crypto_price(symbol)
                if price_data:
                    results[symbol.upper()] = price_data
                time.sleep(0.1)  # Rate limiting
            return results
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical price data from Binance"""
        try:
            # Convert symbol to Binance format
            binance_symbol = self.symbol_mapping.get(symbol.upper())
            if not binance_symbol:
                binance_symbol = f"{symbol.upper()}USDT"
            
            # Determine interval based on days requested
            if days <= 1:
                interval = '1m'
                limit = min(days * 1440, 1000)  # 1440 minutes per day
            elif days <= 7:
                interval = '5m'
                limit = min(days * 288, 1000)  # 288 5-minute intervals per day
            elif days <= 30:
                interval = '1h'
                limit = min(days * 24, 1000)  # 24 hours per day
            else:
                interval = '1d'
                limit = min(days, 1000)
            
            # Get kline data
            url = f"{self.binance_base_url}/klines"
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                logger.warning(f"No historical data available for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamp and numeric columns
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get cryptocurrency market overview from Binance"""
        try:
            # Get 24hr ticker statistics for top cryptos
            url = f"{self.binance_base_url}/ticker/24hr"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            all_tickers = response.json()
            
            # Filter to USD pairs and sort by volume
            usd_pairs = [ticker for ticker in all_tickers 
                        if ticker['symbol'].endswith('USDT')]
            usd_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
            
            # Calculate market stats
            total_volume = sum(float(ticker['quoteVolume']) for ticker in usd_pairs[:100])
            positive_changes = sum(1 for ticker in usd_pairs[:100] 
                                 if float(ticker['priceChangePercent']) > 0)
            
            # Get top gainers and losers
            gainers = sorted(usd_pairs[:50], 
                           key=lambda x: float(x['priceChangePercent']), reverse=True)[:5]
            losers = sorted(usd_pairs[:50], 
                          key=lambda x: float(x['priceChangePercent']))[:5]
            
            return {
                'total_market_cap': 0,  # Not available from Binance
                'total_volume_24h': total_volume,
                'btc_dominance': 0,  # Not available from Binance
                'active_cryptocurrencies': len(usd_pairs),
                'market_sentiment': 'bullish' if positive_changes > 50 else 'bearish',
                'top_gainers': [
                    {
                        'symbol': ticker['symbol'].replace('USDT', ''),
                        'change_24h': float(ticker['priceChangePercent']),
                        'price': float(ticker['lastPrice'])
                    }
                    for ticker in gainers
                ],
                'top_losers': [
                    {
                        'symbol': ticker['symbol'].replace('USDT', ''),
                        'change_24h': float(ticker['priceChangePercent']),
                        'price': float(ticker['lastPrice'])
                    }
                    for ticker in losers
                ]
            }
            
        except Exception as e:
            logger.error(f"Error fetching market overview: {str(e)}")
            return {
                'total_market_cap': 0,
                'total_volume_24h': 0,
                'btc_dominance': 0,
                'active_cryptocurrencies': 0,
                'market_sentiment': 'neutral',
                'top_gainers': [],
                'top_losers': []
            }
    
    def get_top_cryptocurrencies(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get top cryptocurrencies by volume"""
        try:
            # Get 24hr ticker statistics
            url = f"{self.binance_base_url}/ticker/24hr"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            all_tickers = response.json()
            
            # Filter to USD pairs and sort by volume
            usd_pairs = [ticker for ticker in all_tickers 
                        if ticker['symbol'].endswith('USDT')]
            usd_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
            
            top_cryptos = []
            for i, ticker in enumerate(usd_pairs[:limit]):
                try:
                    symbol = ticker['symbol'].replace('USDT', '')
                    top_cryptos.append({
                        'rank': i + 1,
                        'symbol': symbol,
                        'name': symbol,  # Binance doesn't provide full names
                        'price': float(ticker['lastPrice']),
                        'change_24h': float(ticker['priceChangePercent']),
                        'volume_24h': float(ticker['quoteVolume']),
                        'market_cap': 0  # Not available from Binance
                    })
                except (KeyError, ValueError) as e:
                    logger.warning(f"Error parsing ticker data: {str(e)}")
                    continue
            
            return top_cryptos
            
        except Exception as e:
            logger.error(f"Error fetching top cryptocurrencies: {str(e)}")
            return []
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """Get Binance exchange information"""
        try:
            url = f"{self.binance_base_url}/exchangeInfo"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Count USDT pairs
            usdt_pairs = [symbol for symbol in data['symbols'] 
                         if symbol['symbol'].endswith('USDT') and symbol['status'] == 'TRADING']
            
            return {
                'exchange': 'Binance',
                'total_pairs': len(data['symbols']),
                'usdt_pairs': len(usdt_pairs),
                'server_time': datetime.fromtimestamp(data['serverTime'] / 1000),
                'timezone': data['timezone']
            }
            
        except Exception as e:
            logger.error(f"Error fetching exchange info: {str(e)}")
            return {}
