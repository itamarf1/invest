import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import time
from enum import Enum
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CommodityCategory(Enum):
    ENERGY = "energy"
    PRECIOUS_METALS = "precious_metals"
    INDUSTRIAL_METALS = "industrial_metals"
    AGRICULTURE = "agriculture"
    LIVESTOCK = "livestock"
    SOFT_COMMODITIES = "soft_commodities"


@dataclass
class CommodityData:
    symbol: str
    name: str
    price: float
    currency: str
    unit: str  # e.g., "per barrel", "per ounce", "per bushel"
    category: CommodityCategory
    change_24h: Optional[float]
    change_24h_pct: Optional[float]
    volume: Optional[int]
    timestamp: datetime
    
    # Additional metrics
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    volatility: Optional[float] = None
    storage_cost: Optional[float] = None
    seasonality_factor: Optional[str] = None


@dataclass
class CommodityFuturesData:
    symbol: str
    name: str
    front_month_price: float
    back_month_price: Optional[float]
    contango_backwardation: Optional[str]  # "contango", "backwardation", "normal"
    expiry_date: Optional[datetime]
    open_interest: Optional[int]
    volume: Optional[int]
    category: CommodityCategory
    timestamp: datetime


class CommoditiesDataFetcher:
    """Commodities data fetcher using ETFs and futures proxies"""
    
    def __init__(self):
        # Commodity ETFs as proxies for commodity prices
        self.commodity_etfs = {
            # Energy
            'USO': {'name': 'United States Oil Fund', 'category': CommodityCategory.ENERGY, 'underlying': 'Crude Oil WTI', 'unit': 'per barrel'},
            'UNG': {'name': 'United States Natural Gas Fund', 'category': CommodityCategory.ENERGY, 'underlying': 'Natural Gas', 'unit': 'per MMBtu'},
            'BNO': {'name': 'United States Brent Oil Fund', 'category': CommodityCategory.ENERGY, 'underlying': 'Brent Oil', 'unit': 'per barrel'},
            
            # Precious Metals
            'GLD': {'name': 'SPDR Gold Shares', 'category': CommodityCategory.PRECIOUS_METALS, 'underlying': 'Gold', 'unit': 'per ounce'},
            'SLV': {'name': 'iShares Silver Trust', 'category': CommodityCategory.PRECIOUS_METALS, 'underlying': 'Silver', 'unit': 'per ounce'},
            'PPLT': {'name': 'Aberdeen Standard Platinum Shares ETF', 'category': CommodityCategory.PRECIOUS_METALS, 'underlying': 'Platinum', 'unit': 'per ounce'},
            'PALL': {'name': 'Aberdeen Standard Palladium Shares ETF', 'category': CommodityCategory.PRECIOUS_METALS, 'underlying': 'Palladium', 'unit': 'per ounce'},
            
            # Industrial Metals
            'CPER': {'name': 'United States Copper Index Fund', 'category': CommodityCategory.INDUSTRIAL_METALS, 'underlying': 'Copper', 'unit': 'per pound'},
            'JJT': {'name': 'iPath Bloomberg Tin Subindex Total Return ETN', 'category': CommodityCategory.INDUSTRIAL_METALS, 'underlying': 'Tin', 'unit': 'per metric ton'},
            'JJN': {'name': 'iPath Bloomberg Nickel Subindex Total Return ETN', 'category': CommodityCategory.INDUSTRIAL_METALS, 'underlying': 'Nickel', 'unit': 'per pound'},
            
            # Agriculture
            'CORN': {'name': 'Teucrium Corn Fund', 'category': CommodityCategory.AGRICULTURE, 'underlying': 'Corn', 'unit': 'per bushel'},
            'SOYB': {'name': 'Teucrium Soybean Fund', 'category': CommodityCategory.AGRICULTURE, 'underlying': 'Soybeans', 'unit': 'per bushel'},
            'WEAT': {'name': 'Teucrium Wheat Fund', 'category': CommodityCategory.AGRICULTURE, 'underlying': 'Wheat', 'unit': 'per bushel'},
            'CANE': {'name': 'Teucrium Sugar Fund', 'category': CommodityCategory.SOFT_COMMODITIES, 'underlying': 'Sugar', 'unit': 'per pound'},
            
            # Soft Commodities  
            'JO': {'name': 'iPath Bloomberg Coffee Subindex Total Return ETN', 'category': CommodityCategory.SOFT_COMMODITIES, 'underlying': 'Coffee', 'unit': 'per pound'},
            'NIB': {'name': 'iPath Bloomberg Cocoa Subindex Total Return ETN', 'category': CommodityCategory.SOFT_COMMODITIES, 'underlying': 'Cocoa', 'unit': 'per metric ton'},
            'BAL': {'name': 'iPath Bloomberg Cotton Subindex Total Return ETN', 'category': CommodityCategory.SOFT_COMMODITIES, 'underlying': 'Cotton', 'unit': 'per pound'},
            
            # Livestock
            'COW': {'name': 'iPath Bloomberg Livestock Subindex Total Return ETN', 'category': CommodityCategory.LIVESTOCK, 'underlying': 'Live Cattle', 'unit': 'per pound'},
            
            # Broad Commodity Exposure
            'DJP': {'name': 'iPath Bloomberg Commodity Index Total Return ETN', 'category': CommodityCategory.ENERGY, 'underlying': 'Commodity Index', 'unit': 'index'},
            'GSG': {'name': 'iShares S&P GSCI Commodity-Indexed Trust', 'category': CommodityCategory.ENERGY, 'underlying': 'Commodity Index', 'unit': 'index'},
        }
        
        # Seasonal patterns for agricultural commodities
        self.seasonality_patterns = {
            CommodityCategory.AGRICULTURE: {
                "harvest_months": ["September", "October", "November"],
                "planting_months": ["March", "April", "May"],
                "pattern": "Prices typically lower during harvest, higher during planting"
            },
            CommodityCategory.ENERGY: {
                "high_demand": ["December", "January", "February", "July", "August"],
                "low_demand": ["March", "April", "May", "September", "October"],
                "pattern": "Higher demand in winter (heating) and summer (cooling)"
            },
            CommodityCategory.PRECIOUS_METALS: {
                "high_demand": ["October", "November", "December"],
                "pattern": "Jewelry demand increases during holiday season"
            }
        }
        
        logger.info("Commodities data fetcher initialized with ETF proxies")
    
    def get_commodity_data(self, symbol: str) -> Optional[CommodityData]:
        """Get commodity data for a specific ETF symbol"""
        try:
            if symbol.upper() not in self.commodity_etfs:
                logger.warning(f"Unknown commodity ETF symbol: {symbol}")
                return None
            
            commodity_info = self.commodity_etfs[symbol.upper()]
            
            # Get ETF data from yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            info = ticker.info
            
            if hist.empty:
                logger.error(f"No price data for commodity ETF {symbol}")
                return None
            
            current_price = hist['Close'].iloc[-1]
            volume = int(hist['Volume'].iloc[-1]) if not pd.isna(hist['Volume'].iloc[-1]) else None
            
            # Calculate 24h change
            if len(hist) >= 2:
                prev_price = hist['Close'].iloc[-2]
                change_24h = current_price - prev_price
                change_24h_pct = (change_24h / prev_price) * 100
            else:
                change_24h = None
                change_24h_pct = None
            
            # Calculate 52-week high/low
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()
            
            # Calculate volatility (annualized)
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 10 else None
            
            # Get seasonality information
            current_month = datetime.now().strftime("%B")
            seasonality = self._get_seasonality_info(commodity_info['category'], current_month)
            
            return CommodityData(
                symbol=symbol.upper(),
                name=commodity_info['name'],
                price=current_price,
                currency='USD',
                unit=commodity_info['unit'],
                category=commodity_info['category'],
                change_24h=change_24h,
                change_24h_pct=change_24h_pct,
                volume=volume,
                timestamp=datetime.now(),
                high_52w=high_52w,
                low_52w=low_52w,
                volatility=volatility,
                seasonality_factor=seasonality
            )
            
        except Exception as e:
            logger.error(f"Error fetching commodity data for {symbol}: {str(e)}")
            return None
    
    def get_multiple_commodities(self, symbols: List[str]) -> Dict[str, CommodityData]:
        """Get commodity data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                commodity_data = self.get_commodity_data(symbol)
                if commodity_data:
                    results[symbol.upper()] = commodity_data
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {str(e)}")
                continue
        
        return results
    
    def get_commodity_categories(self) -> Dict[str, List[Dict[str, str]]]:
        """Get commodities organized by category"""
        categories = {}
        
        for symbol, info in self.commodity_etfs.items():
            category = info['category'].value
            if category not in categories:
                categories[category] = []
            categories[category].append({
                'symbol': symbol,
                'name': info['name'],
                'underlying': info['underlying'],
                'unit': info['unit']
            })
        
        return categories
    
    def analyze_commodity_trends(self, symbols: List[str] = None, period: str = "1y") -> Dict[str, Any]:
        """Analyze commodity market trends"""
        try:
            if symbols is None:
                # Use representative commodities from each category
                symbols = ["GLD", "USO", "CORN", "CPER", "DJP"]
            
            trends_analysis = {}
            
            for symbol in symbols:
                try:
                    commodity = self.get_commodity_data(symbol)
                    if not commodity:
                        continue
                    
                    # Get historical data for trend analysis
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    
                    if hist.empty:
                        continue
                    
                    # Calculate trend metrics
                    start_price = hist['Close'].iloc[0]
                    end_price = hist['Close'].iloc[-1]
                    total_return = ((end_price - start_price) / start_price) * 100
                    
                    # Moving averages
                    ma_20 = hist['Close'].rolling(20).mean().iloc[-1]
                    ma_50 = hist['Close'].rolling(50).mean().iloc[-1]
                    
                    # Determine trend
                    if end_price > ma_20 > ma_50:
                        trend = "Strong Uptrend"
                    elif end_price > ma_20:
                        trend = "Uptrend" 
                    elif end_price < ma_20 < ma_50:
                        trend = "Strong Downtrend"
                    elif end_price < ma_20:
                        trend = "Downtrend"
                    else:
                        trend = "Sideways"
                    
                    # Volatility ranking
                    vol_level = "High" if commodity.volatility and commodity.volatility > 30 else \
                               "Medium" if commodity.volatility and commodity.volatility > 15 else "Low"
                    
                    trends_analysis[symbol] = {
                        'name': commodity.name,
                        'category': commodity.category.value,
                        'current_price': commodity.price,
                        'total_return_pct': total_return,
                        'trend': trend,
                        'volatility': commodity.volatility,
                        'volatility_level': vol_level,
                        'seasonality': commodity.seasonality_factor
                    }
                    
                except Exception as e:
                    logger.warning(f"Error analyzing {symbol}: {str(e)}")
                    continue
            
            # Market-wide analysis
            if trends_analysis:
                total_returns = [data['total_return_pct'] for data in trends_analysis.values() if data['total_return_pct']]
                avg_return = np.mean(total_returns) if total_returns else 0
                
                uptrend_count = len([data for data in trends_analysis.values() if 'Uptrend' in data['trend']])
                total_count = len(trends_analysis)
                market_sentiment = "Bullish" if uptrend_count / total_count > 0.6 else \
                                 "Bearish" if uptrend_count / total_count < 0.4 else "Mixed"
                
                market_analysis = {
                    'average_return': avg_return,
                    'market_sentiment': market_sentiment,
                    'uptrending_commodities': uptrend_count,
                    'total_analyzed': total_count,
                    'dominant_themes': self._identify_commodity_themes(trends_analysis)
                }
            else:
                market_analysis = {'error': 'No valid trend data'}
            
            return {
                'individual_analysis': trends_analysis,
                'market_analysis': market_analysis,
                'analysis_period': period,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing commodity trends: {str(e)}")
            return {'error': str(e)}
    
    def get_commodity_allocation_strategy(self, portfolio_size: float, risk_level: str = "moderate") -> Dict[str, Any]:
        """Get commodity allocation recommendations"""
        try:
            strategies = {
                'conservative': {
                    'allocation_pct': 5,  # 5% of portfolio in commodities
                    'focus': 'precious_metals',
                    'diversification': {
                        'GLD': 60,  # Gold as safe haven
                        'SLV': 25,  # Silver for diversification
                        'DJP': 15   # Broad commodity exposure
                    },
                    'rationale': 'Conservative hedge against inflation and currency debasement'
                },
                'moderate': {
                    'allocation_pct': 10,
                    'focus': 'diversified',
                    'diversification': {
                        'GLD': 35,   # Gold foundation
                        'USO': 20,   # Energy exposure
                        'CPER': 15,  # Industrial metals
                        'DJP': 15,   # Broad exposure
                        'CORN': 10,  # Agriculture
                        'SLV': 5     # Additional precious metals
                    },
                    'rationale': 'Balanced commodity exposure across major categories'
                },
                'aggressive': {
                    'allocation_pct': 15,
                    'focus': 'growth_and_volatility',
                    'diversification': {
                        'USO': 25,    # Energy - high volatility
                        'CPER': 20,   # Copper - economic growth proxy
                        'GLD': 15,    # Gold hedge
                        'CORN': 15,   # Agriculture
                        'SLV': 10,    # Silver - more volatile than gold
                        'UNG': 10,    # Natural gas
                        'PALL': 5     # Palladium - industrial demand
                    },
                    'rationale': 'Higher allocation with growth-oriented commodities'
                }
            }
            
            strategy = strategies.get(risk_level.lower(), strategies['moderate'])
            
            # Calculate dollar amounts
            total_commodity_allocation = portfolio_size * (strategy['allocation_pct'] / 100)
            
            etf_allocations = {}
            for etf, percentage in strategy['diversification'].items():
                dollar_amount = total_commodity_allocation * (percentage / 100)
                
                # Get current price to calculate shares
                commodity_data = self.get_commodity_data(etf)
                shares = int(dollar_amount / commodity_data.price) if commodity_data else 0
                
                etf_allocations[etf] = {
                    'percentage': percentage,
                    'dollar_amount': dollar_amount,
                    'shares': shares,
                    'current_price': commodity_data.price if commodity_data else None,
                    'name': self.commodity_etfs[etf]['name'],
                    'category': self.commodity_etfs[etf]['category'].value
                }
            
            return {
                'risk_level': risk_level,
                'total_portfolio_size': portfolio_size,
                'commodity_allocation_pct': strategy['allocation_pct'],
                'commodity_allocation_amount': total_commodity_allocation,
                'strategy_focus': strategy['focus'],
                'rationale': strategy['rationale'],
                'etf_allocations': etf_allocations,
                'rebalancing_frequency': 'Quarterly',
                'monitoring_recommendations': [
                    'Track inflation data and commodity price trends',
                    'Monitor geopolitical events affecting supply chains',
                    'Watch for changes in industrial demand indicators',
                    'Consider seasonal factors for agricultural commodities'
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating commodity allocation strategy: {str(e)}")
            return {'error': str(e)}
    
    def _get_seasonality_info(self, category: CommodityCategory, current_month: str) -> str:
        """Get seasonality information for commodity category"""
        try:
            if category not in self.seasonality_patterns:
                return "No specific seasonal pattern"
            
            pattern_info = self.seasonality_patterns[category]
            
            if 'high_demand' in pattern_info and current_month in pattern_info['high_demand']:
                return f"High demand season - {pattern_info['pattern']}"
            elif 'low_demand' in pattern_info and current_month in pattern_info['low_demand']:
                return f"Low demand season - {pattern_info['pattern']}"
            elif 'harvest_months' in pattern_info and current_month in pattern_info['harvest_months']:
                return f"Harvest season - {pattern_info['pattern']}"
            elif 'planting_months' in pattern_info and current_month in pattern_info['planting_months']:
                return f"Planting season - {pattern_info['pattern']}"
            else:
                return pattern_info.get('pattern', 'Neutral seasonal period')
                
        except Exception as e:
            logger.warning(f"Error getting seasonality info: {str(e)}")
            return "Seasonality data unavailable"
    
    def _identify_commodity_themes(self, trends_analysis: Dict[str, Any]) -> List[str]:
        """Identify dominant themes in commodity markets"""
        try:
            themes = []
            
            # Energy analysis
            energy_commodities = [k for k, v in trends_analysis.items() if v['category'] == 'energy']
            if energy_commodities:
                energy_trends = [trends_analysis[etf]['trend'] for etf in energy_commodities]
                if sum('Uptrend' in trend for trend in energy_trends) >= len(energy_trends) * 0.6:
                    themes.append("Energy sector strength")
                elif sum('Downtrend' in trend for trend in energy_trends) >= len(energy_trends) * 0.6:
                    themes.append("Energy sector weakness")
            
            # Precious metals analysis
            precious_metals = [k for k, v in trends_analysis.items() if v['category'] == 'precious_metals']
            if precious_metals:
                pm_trends = [trends_analysis[etf]['trend'] for etf in precious_metals]
                if sum('Uptrend' in trend for trend in pm_trends) >= len(pm_trends) * 0.6:
                    themes.append("Safe haven demand (precious metals up)")
                
            # High volatility theme
            high_vol_count = sum(1 for v in trends_analysis.values() if v.get('volatility_level') == 'High')
            if high_vol_count >= len(trends_analysis) * 0.5:
                themes.append("High market volatility across commodities")
            
            # Agriculture theme
            ag_commodities = [k for k, v in trends_analysis.items() if v['category'] == 'agriculture']
            if ag_commodities:
                ag_trends = [trends_analysis[etf]['trend'] for etf in ag_commodities]
                if sum('Uptrend' in trend for trend in ag_trends) >= len(ag_trends) * 0.6:
                    themes.append("Agricultural commodity strength")
            
            return themes if themes else ["Mixed commodity market conditions"]
            
        except Exception as e:
            logger.warning(f"Error identifying themes: {str(e)}")
            return ["Unable to identify market themes"]


