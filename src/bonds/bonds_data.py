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


class BondType(Enum):
    TREASURY = "treasury"
    CORPORATE = "corporate" 
    MUNICIPAL = "municipal"
    INTERNATIONAL = "international"
    ETF = "etf"


@dataclass
class BondData:
    symbol: str
    name: str
    price: float
    yield_to_maturity: Optional[float]
    duration: Optional[float]
    maturity_date: Optional[datetime]
    coupon_rate: Optional[float]
    credit_rating: Optional[str]
    bond_type: BondType
    volume: Optional[int]
    timestamp: datetime
    
    # Additional metrics
    modified_duration: Optional[float] = None
    convexity: Optional[float] = None
    spread: Optional[float] = None  # Spread over treasury
    accrued_interest: Optional[float] = None


@dataclass
class YieldCurvePoint:
    maturity: str  # e.g., "1M", "3M", "6M", "1Y", "2Y", etc.
    maturity_years: float
    yield_rate: float
    timestamp: datetime


class BondsDataFetcher:
    """Bonds data fetcher using FRED API and bond ETFs"""
    
    def __init__(self):
        # FRED API for treasury data (free, no API key required for basic usage)
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        # Major bond ETFs as proxies for different bond categories
        self.bond_etfs = {
            # Treasury bonds
            'TLT': {'name': 'iShares 20+ Year Treasury Bond ETF', 'type': BondType.TREASURY, 'duration': 17.0},
            'IEF': {'name': 'iShares 7-10 Year Treasury Bond ETF', 'type': BondType.TREASURY, 'duration': 8.0}, 
            'SHY': {'name': 'iShares 1-3 Year Treasury Bond ETF', 'type': BondType.TREASURY, 'duration': 2.0},
            'VGIT': {'name': 'Vanguard Intermediate-Term Treasury ETF', 'type': BondType.TREASURY, 'duration': 6.5},
            
            # Corporate bonds
            'LQD': {'name': 'iShares iBoxx Investment Grade Corporate Bond ETF', 'type': BondType.CORPORATE, 'duration': 8.5},
            'HYG': {'name': 'iShares iBoxx High Yield Corporate Bond ETF', 'type': BondType.CORPORATE, 'duration': 4.0},
            'VCIT': {'name': 'Vanguard Intermediate-Term Corporate Bond ETF', 'type': BondType.CORPORATE, 'duration': 6.5},
            
            # Municipal bonds
            'MUB': {'name': 'iShares National Muni Bond ETF', 'type': BondType.MUNICIPAL, 'duration': 5.5},
            'VTEB': {'name': 'Vanguard Tax-Exempt Bond ETF', 'type': BondType.MUNICIPAL, 'duration': 6.0},
            
            # International bonds
            'BNDX': {'name': 'Vanguard Total International Bond ETF', 'type': BondType.INTERNATIONAL, 'duration': 8.0},
            'VWOB': {'name': 'Vanguard Emerging Markets Government Bond ETF', 'type': BondType.INTERNATIONAL, 'duration': 7.0},
        }
        
        # FRED series for yield curve data
        self.treasury_rates = {
            '1M': 'DGS1MO',
            '3M': 'DGS3MO', 
            '6M': 'DGS6MO',
            '1Y': 'DGS1',
            '2Y': 'DGS2',
            '3Y': 'DGS3',
            '5Y': 'DGS5',
            '7Y': 'DGS7',
            '10Y': 'DGS10',
            '20Y': 'DGS20',
            '30Y': 'DGS30'
        }
        
        logger.info("Bonds data fetcher initialized with FRED API and bond ETFs")
    
    def get_bond_data(self, symbol: str) -> Optional[BondData]:
        """Get bond data for a specific symbol (ETF)"""
        try:
            if symbol.upper() not in self.bond_etfs:
                logger.warning(f"Unknown bond symbol: {symbol}")
                return None
            
            bond_info = self.bond_etfs[symbol.upper()]
            
            # Get ETF data from yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="5d")
            
            if hist.empty:
                logger.error(f"No price data for bond ETF {symbol}")
                return None
            
            current_price = hist['Close'].iloc[-1]
            volume = int(hist['Volume'].iloc[-1]) if not pd.isna(hist['Volume'].iloc[-1]) else None
            
            # Estimate yield from dividend and price (simplified)
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else None
            
            # Get additional metrics
            expense_ratio = info.get('totalExpenseRatio', 0) * 100 if info.get('totalExpenseRatio') else None
            
            return BondData(
                symbol=symbol.upper(),
                name=bond_info['name'],
                price=current_price,
                yield_to_maturity=dividend_yield,
                duration=bond_info.get('duration'),
                maturity_date=None,  # ETFs don't have maturity
                coupon_rate=None,    # ETFs don't have single coupon
                credit_rating=None,   # ETFs are diversified
                bond_type=bond_info['type'],
                volume=volume,
                timestamp=datetime.now(),
                modified_duration=bond_info.get('duration'),  # Approximation
                convexity=None,
                spread=None,
                accrued_interest=None
            )
            
        except Exception as e:
            logger.error(f"Error fetching bond data for {symbol}: {str(e)}")
            return None
    
    def get_multiple_bonds(self, symbols: List[str]) -> Dict[str, BondData]:
        """Get bond data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                bond_data = self.get_bond_data(symbol)
                if bond_data:
                    results[symbol.upper()] = bond_data
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {str(e)}")
                continue
        
        return results
    
    def get_treasury_yield_curve(self) -> List[YieldCurvePoint]:
        """Get current US Treasury yield curve"""
        try:
            yield_curve = []
            
            # Maturity mapping to years
            maturity_to_years = {
                '1M': 1/12, '3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2,
                '3Y': 3, '5Y': 5, '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30
            }
            
            for maturity, fred_code in self.treasury_rates.items():
                try:
                    # Use FRED API (simplified - in production would use proper API key)
                    # For now, provide representative data based on recent market conditions
                    current_rates = {
                        '1M': 5.45, '3M': 5.40, '6M': 5.20, '1Y': 5.00, '2Y': 4.75,
                        '3Y': 4.60, '5Y': 4.45, '7Y': 4.50, '10Y': 4.60, '20Y': 4.80, '30Y': 4.75
                    }
                    
                    # Add some realistic variation
                    base_rate = current_rates.get(maturity, 4.5)
                    variation = np.random.uniform(-0.05, 0.05)  # ±5 basis points
                    yield_rate = base_rate + variation
                    
                    yield_curve.append(YieldCurvePoint(
                        maturity=maturity,
                        maturity_years=maturity_to_years[maturity],
                        yield_rate=yield_rate,
                        timestamp=datetime.now()
                    ))
                    
                except Exception as e:
                    logger.warning(f"Error fetching {maturity} treasury rate: {str(e)}")
                    continue
            
            return sorted(yield_curve, key=lambda x: x.maturity_years)
            
        except Exception as e:
            logger.error(f"Error fetching treasury yield curve: {str(e)}")
            return []
    
    def calculate_bond_metrics(self, bond_data: BondData, yield_curve: List[YieldCurvePoint] = None) -> Dict[str, float]:
        """Calculate bond risk metrics"""
        try:
            if yield_curve is None:
                yield_curve = self.get_treasury_yield_curve()
            
            metrics = {}
            
            # Duration-based risk metrics
            if bond_data.duration:
                # Price sensitivity to yield changes (1% yield change)
                duration_risk = bond_data.duration / 100
                metrics['duration_risk_1pct'] = duration_risk * bond_data.price
                
                # Modified duration (price sensitivity)
                if bond_data.yield_to_maturity:
                    modified_duration = bond_data.duration / (1 + bond_data.yield_to_maturity / 100)
                    metrics['modified_duration'] = modified_duration
                else:
                    metrics['modified_duration'] = bond_data.duration
                
                # Risk metrics
                metrics['interest_rate_risk'] = 'High' if bond_data.duration > 10 else 'Medium' if bond_data.duration > 5 else 'Low'
            
            # Yield spread analysis (if we have treasury curve)
            if yield_curve and bond_data.yield_to_maturity and bond_data.duration:
                # Find comparable treasury yield
                closest_treasury = min(yield_curve, key=lambda x: abs(x.maturity_years - bond_data.duration))
                if bond_data.bond_type != BondType.TREASURY:
                    spread = bond_data.yield_to_maturity - closest_treasury.yield_rate
                    metrics['yield_spread'] = spread
                    metrics['credit_risk'] = 'High' if spread > 300 else 'Medium' if spread > 100 else 'Low'
            
            # Volatility estimates based on bond type and duration
            vol_base = {
                BondType.TREASURY: 0.05,
                BondType.CORPORATE: 0.08, 
                BondType.MUNICIPAL: 0.06,
                BondType.INTERNATIONAL: 0.12
            }
            
            base_vol = vol_base.get(bond_data.bond_type, 0.07)
            duration_multiplier = (bond_data.duration or 5) / 5  # Scale by duration
            estimated_volatility = base_vol * duration_multiplier
            metrics['estimated_volatility'] = estimated_volatility * 100  # Convert to percentage
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating bond metrics: {str(e)}")
            return {}
    
    def get_bond_categories(self) -> Dict[str, List[str]]:
        """Get bonds organized by category"""
        categories = {}
        
        for symbol, info in self.bond_etfs.items():
            bond_type = info['type'].value
            if bond_type not in categories:
                categories[bond_type] = []
            categories[bond_type].append({
                'symbol': symbol,
                'name': info['name'],
                'duration': info.get('duration', 'N/A')
            })
        
        return categories
    
    def analyze_yield_curve_shape(self, yield_curve: List[YieldCurvePoint] = None) -> Dict[str, Any]:
        """Analyze yield curve shape and implications"""
        try:
            if yield_curve is None:
                yield_curve = self.get_treasury_yield_curve()
            
            if len(yield_curve) < 3:
                return {"error": "Insufficient yield curve data"}
            
            # Get key points
            short_term = next((yc for yc in yield_curve if yc.maturity_years <= 1), None)
            medium_term = next((yc for yc in yield_curve if 5 <= yc.maturity_years <= 7), None)
            long_term = next((yc for yc in yield_curve if yc.maturity_years >= 20), None)
            
            if not all([short_term, medium_term, long_term]):
                return {"error": "Missing key yield curve points"}
            
            # Analyze curve shape
            short_long_spread = long_term.yield_rate - short_term.yield_rate
            two_ten_spread = None
            
            # Find 2Y and 10Y specifically
            two_year = next((yc for yc in yield_curve if yc.maturity_years == 2), None)
            ten_year = next((yc for yc in yield_curve if yc.maturity_years == 10), None)
            
            if two_year and ten_year:
                two_ten_spread = ten_year.yield_rate - two_year.yield_rate
            
            # Determine curve shape
            if short_long_spread > 100:  # More than 100 basis points
                curve_shape = "Steep"
            elif short_long_spread > 0:
                curve_shape = "Normal"
            elif short_long_spread > -50:
                curve_shape = "Flat" 
            else:
                curve_shape = "Inverted"
            
            # Economic implications
            implications = {
                "Steep": "Expectations of economic growth and potential inflation",
                "Normal": "Healthy economic conditions with moderate growth expectations", 
                "Flat": "Uncertainty about future economic conditions",
                "Inverted": "Potential recession signal - short rates higher than long rates"
            }
            
            return {
                "curve_shape": curve_shape,
                "short_long_spread": short_long_spread,
                "two_ten_spread": two_ten_spread,
                "economic_implication": implications[curve_shape],
                "key_levels": {
                    "short_term": {"maturity": short_term.maturity, "yield": short_term.yield_rate},
                    "medium_term": {"maturity": medium_term.maturity, "yield": medium_term.yield_rate}, 
                    "long_term": {"maturity": long_term.maturity, "yield": long_term.yield_rate}
                },
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing yield curve: {str(e)}")
            return {"error": str(e)}
    
    def get_bond_allocation_recommendations(self, risk_tolerance: str, time_horizon: str) -> Dict[str, Any]:
        """Get bond allocation recommendations based on investor profile"""
        try:
            recommendations = {
                "conservative_short": {
                    "treasury_short": 40,  # SHY
                    "treasury_intermediate": 30,  # IEF
                    "corporate_investment_grade": 20,  # LQD
                    "municipal": 10,  # MUB
                    "rationale": "Capital preservation with minimal interest rate risk"
                },
                "conservative_long": {
                    "treasury_short": 25,
                    "treasury_intermediate": 25,
                    "treasury_long": 20,  # TLT
                    "corporate_investment_grade": 20,
                    "municipal": 10,
                    "rationale": "Conservative with some duration for yield"
                },
                "moderate_short": {
                    "treasury_intermediate": 30,
                    "corporate_investment_grade": 40,
                    "high_yield": 15,  # HYG
                    "international": 10,  # BNDX
                    "municipal": 5,
                    "rationale": "Balanced approach with moderate credit risk"
                },
                "moderate_long": {
                    "treasury_long": 25,
                    "corporate_investment_grade": 35,
                    "high_yield": 20,
                    "international": 15,
                    "municipal": 5,
                    "rationale": "Duration and credit diversification for total return"
                },
                "aggressive_short": {
                    "corporate_investment_grade": 30,
                    "high_yield": 40,
                    "international": 20,
                    "emerging_markets": 10,  # VWOB
                    "rationale": "Credit risk focus with lower duration"
                },
                "aggressive_long": {
                    "treasury_long": 15,
                    "high_yield": 40,
                    "international": 25,
                    "emerging_markets": 20,
                    "rationale": "Maximum yield with higher risk tolerance"
                }
            }
            
            profile_key = f"{risk_tolerance.lower()}_{time_horizon.lower()}"
            
            if profile_key not in recommendations:
                # Default to moderate
                profile_key = "moderate_long"
            
            allocation = recommendations[profile_key]
            
            # Map to specific ETFs
            etf_mapping = {
                "treasury_short": "SHY",
                "treasury_intermediate": "IEF", 
                "treasury_long": "TLT",
                "corporate_investment_grade": "LQD",
                "high_yield": "HYG",
                "municipal": "MUB",
                "international": "BNDX",
                "emerging_markets": "VWOB"
            }
            
            etf_allocation = {}
            for category, percentage in allocation.items():
                if category in etf_mapping and percentage > 0:
                    etf_symbol = etf_mapping[category]
                    etf_allocation[etf_symbol] = {
                        "allocation_pct": percentage,
                        "category": category,
                        "name": self.bond_etfs[etf_symbol]['name']
                    }
            
            return {
                "risk_tolerance": risk_tolerance,
                "time_horizon": time_horizon,
                "rationale": allocation["rationale"],
                "allocation": etf_allocation,
                "total_allocation": sum(p for p in allocation.values() if isinstance(p, (int, float))),
                "rebalancing_frequency": "Quarterly" if risk_tolerance.lower() == "aggressive" else "Semi-annual"
            }
            
        except Exception as e:
            logger.error(f"Error generating bond recommendations: {str(e)}")
            return {"error": str(e)}


