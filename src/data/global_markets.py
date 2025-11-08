import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class MarketRegion(Enum):
    NORTH_AMERICA = "north_america"
    EUROPE = "europe" 
    ASIA_PACIFIC = "asia_pacific"
    EMERGING_MARKETS = "emerging_markets"


class TradingSession(Enum):
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"


@dataclass
class MarketInfo:
    exchange: str
    country: str
    region: MarketRegion
    currency: str
    timezone: str
    open_time: str  # Local time (24h format)
    close_time: str  # Local time (24h format)
    lunch_break: Optional[Tuple[str, str]]  # Some Asian markets have lunch breaks
    market_cap_usd: float  # Approximate market cap in USD trillions
    major_indices: List[str]
    sample_tickers: List[str]


@dataclass
class GlobalMarketData:
    symbol: str
    exchange: str
    country: str
    company_name: str
    currency: str
    current_price: float
    local_price: float  # In local currency
    daily_change: float
    daily_change_pct: float
    volume: int
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    session_status: TradingSession
    local_time: str
    last_updated: datetime
    
    # Additional metrics
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    avg_volume: Optional[int] = None
    dividend_yield: Optional[float] = None


class GlobalMarketsDataFetcher:
    """Fetches data from major global stock exchanges"""
    
    def __init__(self):
        # Define major global exchanges by market cap
        self.markets = {
            # North America - Combined ~$50T+ market cap
            "nasdaq": MarketInfo(
                exchange="NASDAQ",
                country="United States", 
                region=MarketRegion.NORTH_AMERICA,
                currency="USD",
                timezone="America/New_York",
                open_time="09:30",
                close_time="16:00",
                lunch_break=None,
                market_cap_usd=25.0,  # ~$25T
                major_indices=["^IXIC", "^GSPC", "^DJI"],
                sample_tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
            ),
            
            # Asia Pacific
            "tokyo": MarketInfo(
                exchange="Tokyo Stock Exchange (TSE)",
                country="Japan",
                region=MarketRegion.ASIA_PACIFIC, 
                currency="JPY",
                timezone="Asia/Tokyo",
                open_time="09:00",
                close_time="15:00",
                lunch_break=("11:30", "12:30"),
                market_cap_usd=6.2,  # ~$6.2T
                major_indices=["^N225", "^TOPIX"],  # Nikkei 225, TOPIX
                sample_tickers=["7203.T", "9984.T", "6758.T", "8306.T", "9434.T"]  # Toyota, SoftBank, Sony, Mitsubishi, SoftBank
            ),
            
            "shanghai": MarketInfo(
                exchange="Shanghai Stock Exchange (SSE)",
                country="China",
                region=MarketRegion.ASIA_PACIFIC,
                currency="CNY", 
                timezone="Asia/Shanghai",
                open_time="09:30",
                close_time="15:00",
                lunch_break=("11:30", "13:00"),
                market_cap_usd=5.9,  # ~$5.9T
                major_indices=["000001.SS"],  # SSE Composite
                sample_tickers=["600519.SS", "000858.SS", "600036.SS", "601318.SS", "000001.SS"]  # Kweichow Moutai, WULIANGYE, China Merchants Bank, Ping An, Ping An Bank
            ),
            
            "hong_kong": MarketInfo(
                exchange="Hong Kong Stock Exchange (HKEX)",
                country="Hong Kong",
                region=MarketRegion.ASIA_PACIFIC,
                currency="HKD",
                timezone="Asia/Hong_Kong", 
                open_time="09:30",
                close_time="16:00",
                lunch_break=("12:00", "13:00"),
                market_cap_usd=5.4,  # ~$5.4T
                major_indices=["^HSI"],  # Hang Seng Index
                sample_tickers=["0700.HK", "9988.HK", "0005.HK", "0939.HK", "3690.HK"]  # Tencent, Alibaba, HSBC, China Construction Bank, Meituan
            ),
            
            # Europe
            "london": MarketInfo(
                exchange="London Stock Exchange (LSE)",
                country="United Kingdom",
                region=MarketRegion.EUROPE,
                currency="GBP",
                timezone="Europe/London",
                open_time="08:00", 
                close_time="16:30",
                lunch_break=None,
                market_cap_usd=3.9,  # ~$3.9T
                major_indices=["^FTSE"],  # FTSE 100
                sample_tickers=["SHEL.L", "AZN.L", "LSEG.L", "UNI.L", "ULVR.L"]  # Shell, AstraZeneca, London Stock Exchange, Unilever
            ),
            
            "euronext": MarketInfo(
                exchange="Euronext (Amsterdam, Paris, Brussels, Lisbon)",
                country="Europe (Multi-country)",
                region=MarketRegion.EUROPE,
                currency="EUR",
                timezone="Europe/Amsterdam", 
                open_time="09:00",
                close_time="17:30",
                lunch_break=None,
                market_cap_usd=6.9,  # ~$6.9T combined
                major_indices=["^FCHI", "^AEX"],  # CAC 40, AEX
                sample_tickers=["MC.PA", "OR.PA", "SAN.PA", "ASML.AS", "ADYEN.AS"]  # LVMH, L'Oréal, Sanofi, ASML, Adyen
            ),
            
            "frankfurt": MarketInfo(
                exchange="Frankfurt Stock Exchange (Xetra)",
                country="Germany", 
                region=MarketRegion.EUROPE,
                currency="EUR",
                timezone="Europe/Berlin",
                open_time="09:00",
                close_time="17:30", 
                lunch_break=None,
                market_cap_usd=2.3,  # ~$2.3T
                major_indices=["^GDAXI"],  # DAX
                sample_tickers=["SAP.DE", "ASME.DE", "SIE.DE", "DTE.DE", "ALV.DE"]  # SAP, ASML, Siemens, Deutsche Telekom, Allianz
            ),
            
            # Other Major Markets
            "toronto": MarketInfo(
                exchange="Toronto Stock Exchange (TSX)",
                country="Canada",
                region=MarketRegion.NORTH_AMERICA,
                currency="CAD",
                timezone="America/Toronto",
                open_time="09:30",
                close_time="16:00",
                lunch_break=None,
                market_cap_usd=3.0,  # ~$3.0T
                major_indices=["^GSPTSE"],  # S&P/TSX Composite
                sample_tickers=["SHOP.TO", "CNR.TO", "RY.TO", "BNS.TO", "ABX.TO"]  # Shopify, Canadian National Railway, Royal Bank, Bank of Nova Scotia, Barrick Gold
            ),
            
            "mumbai": MarketInfo(
                exchange="Bombay Stock Exchange (BSE) / National Stock Exchange (NSE)",
                country="India",
                region=MarketRegion.EMERGING_MARKETS,
                currency="INR", 
                timezone="Asia/Kolkata",
                open_time="09:15",
                close_time="15:30",
                lunch_break=None,
                market_cap_usd=4.3,  # ~$4.3T combined
                major_indices=["^BSESN", "^NSEI"],  # BSE Sensex, NSE Nifty 50
                sample_tickers=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "HINDUNILVR.NS"]  # Reliance, TCS, HDFC Bank, ICICI Bank, Hindustan Unilever
            ),
            
            "shenzhen": MarketInfo(
                exchange="Shenzhen Stock Exchange (SZSE)",
                country="China",
                region=MarketRegion.ASIA_PACIFIC,
                currency="CNY",
                timezone="Asia/Shanghai", 
                open_time="09:30",
                close_time="15:00",
                lunch_break=("11:30", "13:00"),
                market_cap_usd=3.5,  # ~$3.5T
                major_indices=["399001.SZ"],  # SZSE Component Index
                sample_tickers=["000002.SZ", "002415.SZ", "300750.SZ", "002594.SZ", "300059.SZ"]  # China Vanke, Hikvision, Contemporary Amperex, BYD, East Money
            )
        }
        
        # Market session times (rough approximations)
        self.global_trading_sessions = self._calculate_trading_sessions()
        
        logger.info(f"Global Markets Data Fetcher initialized with {len(self.markets)} major exchanges")
    
    def get_market_info(self, market_code: str) -> Optional[MarketInfo]:
        """Get information about a specific market"""
        return self.markets.get(market_code.lower())
    
    def get_all_markets(self) -> Dict[str, MarketInfo]:
        """Get information about all supported markets"""
        return self.markets
    
    def get_markets_by_region(self, region: MarketRegion) -> Dict[str, MarketInfo]:
        """Get markets filtered by region"""
        return {code: market for code, market in self.markets.items() 
                if market.region == region}
    
    def get_global_stock_data(self, symbol: str, market: str = None) -> Optional[GlobalMarketData]:
        """Get stock data with global market context"""
        try:
            # Auto-detect market from symbol suffix if not provided
            if market is None:
                market = self._detect_market_from_symbol(symbol)
            
            market_info = self.get_market_info(market) if market else None
            
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="5d")
            
            if hist.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return None
            
            # Current price and basic metrics
            current_price = info.get('currentPrice') or info.get('regularMarketPrice') or hist['Close'].iloc[-1]
            
            # Calculate daily change
            daily_change = 0.0
            daily_change_pct = 0.0
            if len(hist) >= 2:
                prev_close = hist['Close'].iloc[-2]
                daily_change = current_price - prev_close
                daily_change_pct = (daily_change / prev_close) * 100 if prev_close != 0 else 0
            
            # Determine trading session status
            session_status = self._get_trading_session_status(market_info) if market_info else TradingSession.REGULAR
            
            # Local time
            local_time = self._get_local_time(market_info.timezone if market_info else "UTC")
            
            # Exchange rate for local price (simplified - in production would use real FX rates)
            local_price = current_price
            if market_info and market_info.currency != "USD":
                # Simplified currency conversion (in production, use real-time FX rates)
                fx_rates = {"JPY": 150, "CNY": 7.3, "HKD": 7.8, "GBP": 0.8, "EUR": 0.9, "CAD": 1.35, "INR": 83}
                fx_rate = fx_rates.get(market_info.currency, 1)
                local_price = current_price * fx_rate
            
            return GlobalMarketData(
                symbol=symbol,
                exchange=market_info.exchange if market_info else "Unknown",
                country=market_info.country if market_info else "Unknown", 
                company_name=info.get('longName', info.get('shortName', symbol)),
                currency=market_info.currency if market_info else info.get('currency', 'USD'),
                current_price=current_price,
                local_price=local_price,
                daily_change=daily_change,
                daily_change_pct=daily_change_pct,
                volume=int(hist['Volume'].iloc[-1]) if not pd.isna(hist['Volume'].iloc[-1]) else 0,
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                session_status=session_status,
                local_time=local_time,
                last_updated=datetime.now(timezone.utc),
                week_52_high=info.get('fiftyTwoWeekHigh'),
                week_52_low=info.get('fiftyTwoWeekLow'),
                avg_volume=info.get('averageVolume'),
                dividend_yield=info.get('dividendYield')
            )
            
        except Exception as e:
            logger.error(f"Error fetching global stock data for {symbol}: {str(e)}")
            return None
    
    def get_market_indices(self, market: str) -> Dict[str, float]:
        """Get major market indices for a specific market"""
        try:
            market_info = self.get_market_info(market)
            if not market_info:
                return {}
            
            indices_data = {}
            for index_symbol in market_info.major_indices:
                try:
                    ticker = yf.Ticker(index_symbol)
                    hist = ticker.history(period="2d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_close = hist['Close'].iloc[-2] if len(hist) >= 2 else current_price
                        change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0
                        
                        indices_data[index_symbol] = {
                            'name': ticker.info.get('longName', index_symbol),
                            'price': current_price,
                            'change_pct': change_pct
                        }
                    time.sleep(0.1)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Error fetching index {index_symbol}: {str(e)}")
                    continue
            
            return indices_data
            
        except Exception as e:
            logger.error(f"Error fetching market indices for {market}: {str(e)}")
            return {}
    
    def get_global_market_overview(self) -> Dict[str, Any]:
        """Get overview of all global markets"""
        try:
            market_overview = {}
            
            for market_code, market_info in self.markets.items():
                try:
                    # Get main index for the market
                    main_index = market_info.major_indices[0] if market_info.major_indices else None
                    if main_index:
                        ticker = yf.Ticker(main_index)
                        hist = ticker.history(period="2d")
                        if not hist.empty:
                            current_price = hist['Close'].iloc[-1] 
                            prev_close = hist['Close'].iloc[-2] if len(hist) >= 2 else current_price
                            change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0
                            
                            # Trading session status
                            session_status = self._get_trading_session_status(market_info)
                            local_time = self._get_local_time(market_info.timezone)
                            
                            market_overview[market_code] = {
                                'exchange': market_info.exchange,
                                'country': market_info.country,
                                'region': market_info.region.value,
                                'currency': market_info.currency,
                                'market_cap_usd_trillions': market_info.market_cap_usd,
                                'main_index': {
                                    'symbol': main_index,
                                    'price': current_price,
                                    'change_pct': change_pct
                                },
                                'session_status': session_status.value,
                                'local_time': local_time,
                                'is_open': session_status == TradingSession.REGULAR
                            }
                    
                    time.sleep(0.1)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Error processing market {market_code}: {str(e)}")
                    continue
            
            return {
                'markets': market_overview,
                'total_market_cap_usd_trillions': sum(m.market_cap_usd for m in self.markets.values()),
                'summary': {
                    'total_markets': len(market_overview),
                    'open_markets': len([m for m in market_overview.values() if m['is_open']]),
                    'regions_covered': len(set(m.region.value for m in self.markets.values()))
                },
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting global market overview: {str(e)}")
            return {'error': str(e)}
    
    def get_sample_stocks_by_market(self, market: str, limit: int = 5) -> List[GlobalMarketData]:
        """Get sample stocks from a specific market"""
        try:
            market_info = self.get_market_info(market)
            if not market_info:
                return []
            
            sample_stocks = []
            for symbol in market_info.sample_tickers[:limit]:
                stock_data = self.get_global_stock_data(symbol, market)
                if stock_data:
                    sample_stocks.append(stock_data)
                time.sleep(0.1)  # Rate limiting
            
            return sample_stocks
            
        except Exception as e:
            logger.error(f"Error getting sample stocks for {market}: {str(e)}")
            return []
    
    def search_global_stocks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for stocks across all global markets"""
        try:
            # This is a simplified search - in production, would use a proper search API
            results = []
            
            # Search through sample tickers across all markets
            for market_code, market_info in self.markets.items():
                for symbol in market_info.sample_tickers:
                    if query.upper() in symbol.upper():
                        try:
                            ticker = yf.Ticker(symbol)
                            info = ticker.info
                            company_name = info.get('longName', info.get('shortName', symbol))
                            
                            if query.upper() in company_name.upper() or query.upper() in symbol.upper():
                                results.append({
                                    'symbol': symbol,
                                    'company_name': company_name,
                                    'exchange': market_info.exchange,
                                    'country': market_info.country,
                                    'currency': market_info.currency,
                                    'market_code': market_code
                                })
                            
                            if len(results) >= limit:
                                break
                                
                        except Exception as e:
                            logger.warning(f"Error searching symbol {symbol}: {str(e)}")
                            continue
                
                if len(results) >= limit:
                    break
                    
            return results
            
        except Exception as e:
            logger.error(f"Error searching global stocks: {str(e)}")
            return []
    
    def _detect_market_from_symbol(self, symbol: str) -> Optional[str]:
        """Auto-detect market from symbol suffix"""
        if symbol.endswith('.T'):
            return 'tokyo'
        elif symbol.endswith('.SS'):
            return 'shanghai'
        elif symbol.endswith('.SZ'):
            return 'shenzhen'
        elif symbol.endswith('.HK'):
            return 'hong_kong'
        elif symbol.endswith('.L'):
            return 'london'
        elif symbol.endswith('.PA'):
            return 'euronext'
        elif symbol.endswith('.AS'):
            return 'euronext'
        elif symbol.endswith('.DE'):
            return 'frankfurt'
        elif symbol.endswith('.TO'):
            return 'toronto'
        elif symbol.endswith('.NS') or symbol.endswith('.BO'):
            return 'mumbai'
        else:
            return 'nasdaq'  # Default to NASDAQ for US stocks
    
    def _get_trading_session_status(self, market_info: MarketInfo) -> TradingSession:
        """Determine current trading session status for a market"""
        try:
            # Simplified logic - in production would use proper timezone handling
            # For now, assume regular session for demonstration
            return TradingSession.REGULAR
        except:
            return TradingSession.CLOSED
    
    def _get_local_time(self, timezone_str: str) -> str:
        """Get local time for a timezone"""
        try:
            # Simplified - in production would use proper timezone conversion
            return datetime.now().strftime("%H:%M %Z")
        except:
            return "Unknown"
    
    def _calculate_trading_sessions(self) -> Dict[str, Any]:
        """Calculate global trading sessions schedule"""
        # Simplified implementation - would be more complex in production
        return {
            "next_major_open": "Tokyo (09:00 JST)",
            "currently_trading": ["nasdaq", "toronto"],  # Simplified
            "session_transitions": {}
        }