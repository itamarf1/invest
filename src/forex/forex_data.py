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


class CurrencyClass(Enum):
    MAJOR = "major"  # G7 currencies
    MINOR = "minor"  # Other developed market currencies
    EXOTIC = "exotic"  # Emerging market currencies
    COMMODITY = "commodity"  # Commodity-linked currencies


@dataclass
class ForexPair:
    base_currency: str
    quote_currency: str
    symbol: str
    name: str
    price: float
    bid: Optional[float]
    ask: Optional[float]
    spread: Optional[float]
    change_24h: Optional[float]
    change_24h_pct: Optional[float]
    currency_class: CurrencyClass
    timestamp: datetime
    
    # Additional metrics
    volatility: Optional[float] = None
    volume: Optional[int] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    central_bank_rate: Optional[float] = None
    inflation_rate: Optional[float] = None


@dataclass
class EconomicIndicator:
    country: str
    currency: str
    indicator_name: str
    current_value: float
    previous_value: Optional[float]
    forecast_value: Optional[float]
    impact_level: str  # "high", "medium", "low"
    release_date: datetime
    next_release: Optional[datetime]


class ForexDataFetcher:
    """Forex data fetcher using currency ETFs and economic data"""
    
    def __init__(self):
        # Currency ETFs as proxies for forex pairs
        self.currency_etfs = {
            # Major currencies (vs USD)
            'FXE': {'name': 'Invesco CurrencyShares Euro Trust', 'base': 'EUR', 'quote': 'USD', 'class': CurrencyClass.MAJOR},
            'FXY': {'name': 'Invesco CurrencyShares Japanese Yen Trust', 'base': 'JPY', 'quote': 'USD', 'class': CurrencyClass.MAJOR},
            'FXB': {'name': 'Invesco CurrencyShares British Pound Trust', 'base': 'GBP', 'quote': 'USD', 'class': CurrencyClass.MAJOR},
            'FXC': {'name': 'Invesco CurrencyShares Canadian Dollar Trust', 'base': 'CAD', 'quote': 'USD', 'class': CurrencyClass.MAJOR},
            'FXA': {'name': 'Invesco CurrencyShares Australian Dollar Trust', 'base': 'AUD', 'quote': 'USD', 'class': CurrencyClass.COMMODITY},
            'FXF': {'name': 'Invesco CurrencyShares Swiss Franc Trust', 'base': 'CHF', 'quote': 'USD', 'class': CurrencyClass.MAJOR},
            
            # Other developed markets
            'FXS': {'name': 'Invesco CurrencyShares Swedish Krona Trust', 'base': 'SEK', 'quote': 'USD', 'class': CurrencyClass.MINOR},
            'UUP': {'name': 'Invesco DB US Dollar Index Bullish Fund', 'base': 'USD', 'quote': 'DXY', 'class': CurrencyClass.MAJOR},
            
            # Emerging markets
            'CYB': {'name': 'WisdomTree Chinese Yuan Strategy Fund', 'base': 'CNY', 'quote': 'USD', 'class': CurrencyClass.EXOTIC},
            'BZF': {'name': 'Invesco CurrencyShares Brazilian Real Trust', 'base': 'BRL', 'quote': 'USD', 'class': CurrencyClass.EXOTIC},
            'FXM': {'name': 'Invesco CurrencyShares Mexican Peso Trust', 'base': 'MXN', 'quote': 'USD', 'class': CurrencyClass.EXOTIC},
            'ICN': {'name': 'Invesco CurrencyShares Indian Rupee Trust', 'base': 'INR', 'quote': 'USD', 'class': CurrencyClass.EXOTIC},
            
            # Multi-currency
            'CEW': {'name': 'WisdomTree Emerging Currency Strategy Fund', 'base': 'EM', 'quote': 'USD', 'class': CurrencyClass.EXOTIC},
            'DBV': {'name': 'Invesco DB G10 Currency Harvest Fund', 'base': 'G10', 'quote': 'USD', 'class': CurrencyClass.MAJOR},
        }
        
        # Major central bank interest rates (approximate current levels)
        self.central_bank_rates = {
            'USD': {'rate': 5.25, 'bank': 'Federal Reserve'},
            'EUR': {'rate': 4.50, 'bank': 'European Central Bank'},
            'GBP': {'rate': 5.25, 'bank': 'Bank of England'},
            'JPY': {'rate': 0.10, 'bank': 'Bank of Japan'},
            'CAD': {'rate': 5.00, 'bank': 'Bank of Canada'},
            'AUD': {'rate': 4.35, 'bank': 'Reserve Bank of Australia'},
            'CHF': {'rate': 1.75, 'bank': 'Swiss National Bank'},
            'CNY': {'rate': 3.45, 'bank': 'People\'s Bank of China'},
        }
        
        # Economic indicators impact on currencies
        self.economic_indicators = {
            'USD': ['Non-Farm Payrolls', 'CPI', 'Federal Funds Rate', 'GDP'],
            'EUR': ['ECB Interest Rate', 'Eurozone CPI', 'German GDP', 'Unemployment Rate'],
            'GBP': ['BoE Base Rate', 'UK CPI', 'GDP', 'Employment Data'],
            'JPY': ['BoJ Policy Rate', 'Tokyo CPI', 'GDP', 'Trade Balance'],
            'AUD': ['RBA Cash Rate', 'Employment Change', 'CPI', 'Commodity Prices'],
            'CAD': ['BoC Overnight Rate', 'Employment Change', 'CPI', 'Oil Prices']
        }
        
        logger.info("Forex data fetcher initialized with currency ETFs")
    
    def get_forex_pair(self, symbol: str) -> Optional[ForexPair]:
        """Get forex data for a specific currency ETF"""
        try:
            if symbol.upper() not in self.currency_etfs:
                logger.warning(f"Unknown forex ETF symbol: {symbol}")
                return None
            
            currency_info = self.currency_etfs[symbol.upper()]
            
            # Get ETF data from yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            info = ticker.info
            
            if hist.empty:
                logger.error(f"No price data for currency ETF {symbol}")
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
            
            # Calculate volatility (annualized)
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 10 else None
            
            # Calculate 52-week high/low
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()
            
            # Get central bank rate
            base_currency = currency_info['base']
            cb_rate = self.central_bank_rates.get(base_currency, {}).get('rate')
            
            # Estimate spread (simplified)
            spread = current_price * 0.0002 if current_price else None  # ~2 basis points
            
            return ForexPair(
                base_currency=currency_info['base'],
                quote_currency=currency_info['quote'],
                symbol=symbol.upper(),
                name=currency_info['name'],
                price=current_price,
                bid=current_price - (spread/2) if spread else None,
                ask=current_price + (spread/2) if spread else None,
                spread=spread,
                change_24h=change_24h,
                change_24h_pct=change_24h_pct,
                currency_class=currency_info['class'],
                timestamp=datetime.now(),
                volatility=volatility,
                volume=volume,
                high_52w=high_52w,
                low_52w=low_52w,
                central_bank_rate=cb_rate
            )
            
        except Exception as e:
            logger.error(f"Error fetching forex data for {symbol}: {str(e)}")
            return None
    
    def get_multiple_forex_pairs(self, symbols: List[str]) -> Dict[str, ForexPair]:
        """Get forex data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                forex_data = self.get_forex_pair(symbol)
                if forex_data:
                    results[symbol.upper()] = forex_data
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {str(e)}")
                continue
        
        return results
    
    def get_currency_strength_analysis(self, currencies: List[str] = None) -> Dict[str, Any]:
        """Analyze relative currency strength"""
        try:
            if currencies is None:
                currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF']
            
            # Map currencies to their primary ETFs
            currency_to_etf = {
                'EUR': 'FXE', 'JPY': 'FXY', 'GBP': 'FXB', 
                'CAD': 'FXC', 'AUD': 'FXA', 'CHF': 'FXF'
            }
            
            strength_analysis = {}
            
            for currency in currencies:
                if currency == 'USD':
                    # USD strength measured by DXY (UUP ETF)
                    usd_data = self.get_forex_pair('UUP')
                    if usd_data:
                        strength_analysis['USD'] = {
                            'strength_score': self._calculate_strength_score(usd_data),
                            'change_24h_pct': usd_data.change_24h_pct,
                            'volatility': usd_data.volatility,
                            'central_bank_rate': usd_data.central_bank_rate,
                            'trend': self._determine_trend(usd_data)
                        }
                elif currency in currency_to_etf:
                    etf_symbol = currency_to_etf[currency]
                    forex_data = self.get_forex_pair(etf_symbol)
                    if forex_data:
                        strength_analysis[currency] = {
                            'strength_score': self._calculate_strength_score(forex_data),
                            'change_24h_pct': forex_data.change_24h_pct,
                            'volatility': forex_data.volatility,
                            'central_bank_rate': forex_data.central_bank_rate,
                            'trend': self._determine_trend(forex_data)
                        }
            
            # Rank currencies by strength
            if strength_analysis:
                sorted_currencies = sorted(
                    strength_analysis.items(),
                    key=lambda x: x[1]['strength_score'],
                    reverse=True
                )
                
                # Calculate market sentiment
                positive_currencies = sum(1 for _, data in strength_analysis.items() 
                                        if data.get('change_24h_pct', 0) > 0)
                market_sentiment = "Risk-On" if positive_currencies >= len(strength_analysis) * 0.6 else \
                                 "Risk-Off" if positive_currencies <= len(strength_analysis) * 0.4 else "Mixed"
                
                return {
                    'currency_rankings': [
                        {'currency': currency, 'strength_score': data['strength_score'], 'trend': data['trend']}
                        for currency, data in sorted_currencies
                    ],
                    'detailed_analysis': strength_analysis,
                    'market_sentiment': market_sentiment,
                    'strongest_currency': sorted_currencies[0][0] if sorted_currencies else None,
                    'weakest_currency': sorted_currencies[-1][0] if sorted_currencies else None,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            else:
                return {'error': 'No currency data available for analysis'}
                
        except Exception as e:
            logger.error(f"Error analyzing currency strength: {str(e)}")
            return {'error': str(e)}
    
    def get_economic_calendar_impact(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get upcoming economic events and their currency impact"""
        try:
            # In production, this would fetch from economic calendar APIs
            # For now, provide representative high-impact events
            
            upcoming_events = {
                'USD': [
                    {
                        'event': 'Federal Reserve Meeting',
                        'impact': 'high',
                        'forecast': 'Hold rates at 5.25%',
                        'importance': 'Key for USD direction',
                        'date': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
                    },
                    {
                        'event': 'Non-Farm Payrolls',
                        'impact': 'high',
                        'forecast': '200K jobs added',
                        'importance': 'Employment strength indicator',
                        'date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    },
                    {
                        'event': 'CPI Release',
                        'impact': 'high',
                        'forecast': '3.2% YoY',
                        'importance': 'Inflation trend critical for Fed policy',
                        'date': (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')
                    }
                ],
                'EUR': [
                    {
                        'event': 'ECB Monetary Policy Meeting',
                        'impact': 'high',
                        'forecast': 'Hold rates at 4.50%',
                        'importance': 'Eurozone policy direction',
                        'date': (datetime.now() + timedelta(days=12)).strftime('%Y-%m-%d')
                    },
                    {
                        'event': 'Eurozone CPI',
                        'impact': 'medium',
                        'forecast': '2.8% YoY',
                        'importance': 'Inflation target progress',
                        'date': (datetime.now() + timedelta(days=5)).strftime('%Y-%m-%d')
                    }
                ],
                'GBP': [
                    {
                        'event': 'BoE Interest Rate Decision',
                        'impact': 'high',
                        'forecast': 'Hold at 5.25%',
                        'importance': 'UK monetary policy stance',
                        'date': (datetime.now() + timedelta(days=16)).strftime('%Y-%m-%d')
                    },
                    {
                        'event': 'UK Employment Data',
                        'impact': 'medium',
                        'forecast': '4.2% unemployment',
                        'importance': 'Labor market health',
                        'date': (datetime.now() + timedelta(days=8)).strftime('%Y-%m-%d')
                    }
                ],
                'JPY': [
                    {
                        'event': 'BoJ Policy Meeting',
                        'impact': 'high',
                        'forecast': 'Maintain ultra-loose policy',
                        'importance': 'Yield curve control continuation',
                        'date': (datetime.now() + timedelta(days=18)).strftime('%Y-%m-%d')
                    }
                ]
            }
            
            # Add trading recommendations based on events
            for currency, events in upcoming_events.items():
                for event in events:
                    if event['impact'] == 'high':
                        event['trading_note'] = f"High volatility expected for {currency} pairs"
                    else:
                        event['trading_note'] = f"Moderate impact expected on {currency}"
            
            return upcoming_events
            
        except Exception as e:
            logger.error(f"Error fetching economic calendar: {str(e)}")
            return {'error': str(e)}
    
    def get_forex_trading_recommendations(self, risk_level: str = "moderate", 
                                        trading_style: str = "swing") -> Dict[str, Any]:
        """Get forex trading recommendations based on analysis"""
        try:
            # Get current currency strength
            strength_analysis = self.get_currency_strength_analysis()
            
            if 'error' in strength_analysis:
                return strength_analysis
            
            recommendations = {
                'conservative': {
                    'pairs_to_watch': ['EUR/USD', 'GBP/USD', 'USD/CHF'],
                    'strategy': 'Major pairs with lower volatility',
                    'position_size': '1-2% risk per trade',
                    'timeframe': 'Daily/Weekly charts',
                    'focus': 'Central bank policy divergence'
                },
                'moderate': {
                    'pairs_to_watch': ['EUR/USD', 'GBP/USD', 'AUD/USD', 'USD/CAD'],
                    'strategy': 'Trend following with major pairs',
                    'position_size': '2-3% risk per trade', 
                    'timeframe': '4H/Daily charts',
                    'focus': 'Economic data releases and technical levels'
                },
                'aggressive': {
                    'pairs_to_watch': ['EUR/USD', 'GBP/USD', 'AUD/USD', 'USD/JPY', 'Exotic pairs'],
                    'strategy': 'Higher volatility pairs and carry trades',
                    'position_size': '3-5% risk per trade',
                    'timeframe': '1H/4H charts',
                    'focus': 'Momentum trading and news events'
                }
            }
            
            profile = recommendations.get(risk_level.lower(), recommendations['moderate'])
            
            # Generate specific recommendations based on current analysis
            strongest = strength_analysis.get('strongest_currency', 'USD')
            weakest = strength_analysis.get('weakest_currency', 'EUR')
            market_sentiment = strength_analysis.get('market_sentiment', 'Mixed')
            
            # Map currencies to ETF symbols for trading
            currency_etf_map = {
                'EUR': 'FXE', 'GBP': 'FXB', 'JPY': 'FXY',
                'AUD': 'FXA', 'CAD': 'FXC', 'CHF': 'FXF', 'USD': 'UUP'
            }
            
            current_opportunities = []
            
            if market_sentiment == "Risk-On":
                current_opportunities.extend([
                    f"Consider long {currency_etf_map.get(strongest, strongest)} (strongest currency)",
                    f"Consider short {currency_etf_map.get(weakest, weakest)} (weakest currency)",
                    "Risk-on sentiment favors commodity currencies (AUD, CAD)"
                ])
            elif market_sentiment == "Risk-Off":
                current_opportunities.extend([
                    "Safe haven currencies (USD, CHF, JPY) may outperform",
                    "Consider reducing exposure to emerging market currencies",
                    "Monitor central bank interventions"
                ])
            else:
                current_opportunities.append("Mixed sentiment - focus on individual currency fundamentals")
            
            return {
                'risk_level': risk_level,
                'trading_style': trading_style,
                'recommended_approach': profile,
                'current_market_analysis': {
                    'sentiment': market_sentiment,
                    'strongest_currency': strongest,
                    'weakest_currency': weakest
                },
                'current_opportunities': current_opportunities,
                'risk_management': [
                    'Always use stop losses',
                    'Monitor economic calendar for high-impact events',
                    'Be aware of central bank intervention levels',
                    'Consider correlation between currency pairs',
                    'Manage exposure during low liquidity periods'
                ],
                'recommended_etfs': [currency_etf_map.get(curr, curr) for curr in ['EUR', 'GBP', 'JPY', 'AUD', 'CAD']],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating forex recommendations: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_strength_score(self, forex_data: ForexPair) -> float:
        """Calculate currency strength score (0-100)"""
        try:
            score = 50  # Base score
            
            # Recent performance (24h change)
            if forex_data.change_24h_pct:
                score += forex_data.change_24h_pct * 2  # Weight recent performance
            
            # Volatility adjustment (lower volatility = more stable = higher score)
            if forex_data.volatility:
                vol_adjustment = max(-10, min(10, (20 - forex_data.volatility) / 2))
                score += vol_adjustment
            
            # Central bank rate advantage
            if forex_data.central_bank_rate:
                rate_advantage = min(10, forex_data.central_bank_rate)  # Cap at 10 points
                score += rate_advantage
            
            # 52-week position
            if forex_data.high_52w and forex_data.low_52w and forex_data.high_52w != forex_data.low_52w:
                range_position = ((forex_data.price - forex_data.low_52w) / 
                                (forex_data.high_52w - forex_data.low_52w)) * 20 - 10
                score += range_position
            
            return max(0, min(100, score))  # Clamp to 0-100
            
        except Exception as e:
            logger.warning(f"Error calculating strength score: {str(e)}")
            return 50  # Default neutral score
    
    def _determine_trend(self, forex_data: ForexPair) -> str:
        """Determine currency trend"""
        try:
            if not forex_data.change_24h_pct:
                return "Neutral"
            
            if forex_data.change_24h_pct > 0.5:
                return "Strong Bullish"
            elif forex_data.change_24h_pct > 0.1:
                return "Bullish"
            elif forex_data.change_24h_pct < -0.5:
                return "Strong Bearish"
            elif forex_data.change_24h_pct < -0.1:
                return "Bearish"
            else:
                return "Neutral"
                
        except Exception as e:
            logger.warning(f"Error determining trend: {str(e)}")
            return "Unknown"


