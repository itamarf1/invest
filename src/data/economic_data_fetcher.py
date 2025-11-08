import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class EconomicIndicator:
    series_id: str
    title: str
    value: float
    date: datetime
    unit: str
    frequency: str
    change_from_previous: Optional[float] = None
    change_percentage: Optional[float] = None

@dataclass
class EconomicData:
    category: str
    indicators: Dict[str, EconomicIndicator]
    summary: str
    market_impact: str
    timestamp: datetime

class FREDDataFetcher:
    """Federal Reserve Economic Data (FRED) API fetcher - completely free"""
    
    def __init__(self):
        self.api_key = os.getenv('FRED_API_KEY')  # Get free key from fred.stlouisfed.org
        self.base_url = "https://api.stlouisfed.org/fred"
        self.session = requests.Session()
        
        if not self.api_key:
            logger.warning("FRED_API_KEY not found. Get free API key from fred.stlouisfed.org")
            # FRED API is completely free, but requires registration
            
        # Key economic indicators with their FRED series IDs
        self.key_indicators = {
            # Interest Rates
            'fed_funds_rate': 'FEDFUNDS',
            'treasury_10y': 'DGS10', 
            'treasury_2y': 'DGS2',
            'treasury_3m': 'DGS3MO',
            
            # Inflation
            'cpi': 'CPIAUCSL',
            'core_cpi': 'CPILFESL',
            'pce': 'PCEPI',
            'core_pce': 'PCEPILFE',
            
            # Employment
            'unemployment_rate': 'UNRATE',
            'nonfarm_payrolls': 'PAYEMS',
            'labor_force_participation': 'CIVPART',
            'initial_claims': 'ICSA',
            
            # GDP and Growth
            'gdp': 'GDP',
            'gdp_growth': 'A191RL1Q225SBEA',
            'real_gdp': 'GDPC1',
            'industrial_production': 'INDPRO',
            
            # Money Supply
            'm1_money_supply': 'M1SL',
            'm2_money_supply': 'M2SL',
            
            # Housing
            'housing_starts': 'HOUST',
            'existing_home_sales': 'EXHOSLUSM495S',
            'case_shiller_index': 'CSUSHPINSA',
            
            # Consumer
            'consumer_sentiment': 'UMCSENT',
            'retail_sales': 'RSAFS',
            'personal_income': 'PI',
            'personal_consumption': 'PCE',
            
            # Market Indicators
            'vix': 'VIXCLS',
            'dollar_index': 'DTWEXBGS',
            'gold_price': 'GOLDAMGBD228NLBM',
            'oil_price': 'DCOILWTICO'
        }
    
    def get_latest_indicators(self, categories: List[str] = None) -> Dict[str, EconomicData]:
        """Get latest economic indicators by category"""
        try:
            if categories is None:
                categories = ['rates', 'inflation', 'employment', 'growth']
            
            category_mapping = {
                'rates': ['fed_funds_rate', 'treasury_10y', 'treasury_2y', 'treasury_3m'],
                'inflation': ['cpi', 'core_cpi', 'pce', 'core_pce'],
                'employment': ['unemployment_rate', 'nonfarm_payrolls', 'labor_force_participation'],
                'growth': ['gdp_growth', 'industrial_production', 'retail_sales'],
                'housing': ['housing_starts', 'existing_home_sales', 'case_shiller_index'],
                'market': ['vix', 'dollar_index', 'gold_price', 'oil_price']
            }
            
            results = {}
            
            for category in categories:
                if category not in category_mapping:
                    continue
                    
                indicators = {}
                series_ids = category_mapping[category]
                
                for indicator_name in series_ids:
                    series_id = self.key_indicators.get(indicator_name)
                    if series_id:
                        try:
                            data = self.get_series_data(series_id, limit=2)
                            if data and not data.empty:
                                latest_value = data.iloc[-1]['value']
                                latest_date = data.index[-1]
                                
                                # Calculate change if we have previous data
                                change_from_previous = None
                                change_percentage = None
                                if len(data) > 1:
                                    previous_value = data.iloc[-2]['value']
                                    if not pd.isna(previous_value) and previous_value != 0:
                                        change_from_previous = latest_value - previous_value
                                        change_percentage = (change_from_previous / previous_value) * 100
                                
                                # Get series metadata
                                metadata = self.get_series_metadata(series_id)
                                
                                indicators[indicator_name] = EconomicIndicator(
                                    series_id=series_id,
                                    title=metadata.get('title', indicator_name),
                                    value=latest_value,
                                    date=latest_date,
                                    unit=metadata.get('units', ''),
                                    frequency=metadata.get('frequency', ''),
                                    change_from_previous=change_from_previous,
                                    change_percentage=change_percentage
                                )
                                
                        except Exception as e:
                            logger.warning(f"Error fetching {indicator_name}: {str(e)}")
                            continue
                
                if indicators:
                    summary = self._generate_category_summary(category, indicators)
                    market_impact = self._assess_market_impact(category, indicators)
                    
                    results[category] = EconomicData(
                        category=category,
                        indicators=indicators,
                        summary=summary,
                        market_impact=market_impact,
                        timestamp=datetime.now()
                    )
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting latest indicators: {str(e)}")
            return {}
    
    def get_series_data(self, series_id: str, limit: int = 100, 
                       start_date: Optional[str] = None) -> pd.DataFrame:
        """Get time series data for a specific FRED series"""
        try:
            if not self.api_key:
                # Return sample data for demo
                return self._get_sample_series_data(series_id, limit)
            
            url = f"{self.base_url}/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': limit,
                'sort_order': 'desc'  # Get most recent first
            }
            
            if start_date:
                params['start_date'] = start_date
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            observations = data.get('observations', [])
            if not observations:
                logger.warning(f"No data found for series {series_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_data = []
            for obs in observations:
                try:
                    date = pd.to_datetime(obs['date'])
                    value = obs['value']
                    
                    # Handle missing values represented as '.'
                    if value == '.':
                        value = None
                    else:
                        value = float(value)
                    
                    df_data.append({'date': date, 'value': value})
                except (ValueError, KeyError):
                    continue
            
            if not df_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(df_data)
            df = df.set_index('date').sort_index()
            df = df.dropna()  # Remove missing values
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching series {series_id}: {str(e)}")
            return self._get_sample_series_data(series_id, limit)
        except Exception as e:
            logger.error(f"Error fetching series {series_id}: {str(e)}")
            return pd.DataFrame()
    
    def get_series_metadata(self, series_id: str) -> Dict[str, Any]:
        """Get metadata for a FRED series"""
        try:
            if not self.api_key:
                return self._get_sample_metadata(series_id)
            
            url = f"{self.base_url}/series"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            series_info = data.get('seriess', [])
            if series_info:
                return series_info[0]
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting metadata for {series_id}: {str(e)}")
            return self._get_sample_metadata(series_id)
    
    def _get_sample_series_data(self, series_id: str, limit: int) -> pd.DataFrame:
        """Generate sample data for demonstration when API key not available"""
        try:
            # Generate realistic sample data based on series type
            end_date = datetime.now()
            dates = pd.date_range(end=end_date, periods=limit, freq='ME')
            
            # Base values for different series types
            base_values = {
                'FEDFUNDS': 5.25,      # Fed Funds Rate
                'DGS10': 4.50,         # 10Y Treasury
                'DGS2': 4.75,          # 2Y Treasury  
                'UNRATE': 3.8,         # Unemployment Rate
                'CPIAUCSL': 310.0,     # CPI
                'GDP': 27000.0,        # GDP (billions)
                'VIXCLS': 18.5,        # VIX
                'GOLDAMGBD228NLBM': 2000.0,  # Gold price
                'DCOILWTICO': 75.0     # Oil price
            }
            
            base_value = base_values.get(series_id, 100.0)
            
            # Generate values with some random variation
            np.random.seed(42)  # For consistent demo data
            values = []
            current_value = base_value
            
            for i in range(limit):
                # Add small random changes
                change_pct = np.random.normal(0, 0.02)  # 2% standard deviation
                current_value = current_value * (1 + change_pct)
                values.append(current_value)
            
            df = pd.DataFrame({
                'value': values
            }, index=dates)
            
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Error generating sample data: {str(e)}")
            return pd.DataFrame()
    
    def _get_sample_metadata(self, series_id: str) -> Dict[str, Any]:
        """Get sample metadata for demonstration"""
        metadata_map = {
            'FEDFUNDS': {
                'title': 'Federal Funds Rate',
                'units': 'Percent',
                'frequency': 'Monthly'
            },
            'DGS10': {
                'title': '10-Year Treasury Constant Maturity Rate',
                'units': 'Percent',
                'frequency': 'Daily'
            },
            'UNRATE': {
                'title': 'Unemployment Rate',
                'units': 'Percent',
                'frequency': 'Monthly'
            },
            'CPIAUCSL': {
                'title': 'Consumer Price Index for All Urban Consumers: All Items',
                'units': 'Index 1982-1984=100',
                'frequency': 'Monthly'
            }
        }
        
        return metadata_map.get(series_id, {
            'title': series_id,
            'units': 'Units',
            'frequency': 'Monthly'
        })
    
    def _generate_category_summary(self, category: str, indicators: Dict[str, EconomicIndicator]) -> str:
        """Generate a summary for a category of indicators"""
        try:
            if category == 'rates':
                fed_funds = indicators.get('fed_funds_rate')
                treasury_10y = indicators.get('treasury_10y')
                
                if fed_funds and treasury_10y:
                    yield_curve = treasury_10y.value - fed_funds.value
                    curve_status = "inverted" if yield_curve < 0 else "normal"
                    return f"Fed Funds at {fed_funds.value:.2f}%, 10Y Treasury at {treasury_10y.value:.2f}% (yield curve {curve_status})"
                
            elif category == 'inflation':
                cpi = indicators.get('cpi')
                if cpi and cpi.change_percentage:
                    trend = "rising" if cpi.change_percentage > 0 else "falling"
                    return f"CPI at {cpi.value:.1f}, {trend} {abs(cpi.change_percentage):.1f}% from previous period"
                
            elif category == 'employment':
                unemployment = indicators.get('unemployment_rate')
                if unemployment:
                    level = "low" if unemployment.value < 4.0 else "moderate" if unemployment.value < 6.0 else "high"
                    return f"Unemployment at {unemployment.value:.1f}% ({level} level)"
                
            elif category == 'growth':
                gdp_growth = indicators.get('gdp_growth')
                industrial = indicators.get('industrial_production')
                if gdp_growth:
                    return f"GDP growth at {gdp_growth.value:.1f}% (annualized)"
            
            return f"{category.title()} indicators updated"
            
        except Exception as e:
            logger.error(f"Error generating summary for {category}: {str(e)}")
            return f"{category.title()} data available"
    
    def _assess_market_impact(self, category: str, indicators: Dict[str, EconomicIndicator]) -> str:
        """Assess potential market impact of economic indicators"""
        try:
            if category == 'rates':
                fed_funds = indicators.get('fed_funds_rate')
                if fed_funds and fed_funds.change_from_previous:
                    if abs(fed_funds.change_from_previous) > 0.25:
                        return "High impact - significant rate change expected to affect all asset classes"
                    elif abs(fed_funds.change_from_previous) > 0:
                        return "Moderate impact - rate change may influence bond and equity markets"
                
            elif category == 'inflation':
                cpi = indicators.get('cpi')
                if cpi and cpi.change_percentage:
                    if abs(cpi.change_percentage) > 0.5:
                        return "High impact - significant inflation change may drive Fed policy decisions"
                    elif abs(cpi.change_percentage) > 0.2:
                        return "Moderate impact - inflation trends may influence market expectations"
                
            elif category == 'employment':
                unemployment = indicators.get('unemployment_rate')
                if unemployment and unemployment.change_from_previous:
                    if abs(unemployment.change_from_previous) > 0.3:
                        return "High impact - significant employment change may affect Fed policy"
                
            return "Low to moderate market impact expected"
            
        except Exception as e:
            logger.error(f"Error assessing market impact: {str(e)}")
            return "Market impact assessment unavailable"
    
    def get_recession_indicators(self) -> Dict[str, Any]:
        """Get key recession indicators"""
        try:
            recession_series = {
                'yield_curve_10y2y': ('DGS10', 'DGS2'),
                'unemployment_rate': 'UNRATE',
                'industrial_production': 'INDPRO',
                'real_gdp_growth': 'A191RL1Q225SBEA',
                'initial_claims': 'ICSA'
            }
            
            indicators = {}
            
            # Get yield curve spread (10Y - 2Y)
            if self.api_key:
                try:
                    treasury_10y = self.get_series_data('DGS10', limit=5)
                    treasury_2y = self.get_series_data('DGS2', limit=5)
                    
                    if not treasury_10y.empty and not treasury_2y.empty:
                        # Get most recent common date
                        common_dates = treasury_10y.index.intersection(treasury_2y.index)
                        if len(common_dates) > 0:
                            latest_date = common_dates[-1]
                            spread = treasury_10y.loc[latest_date, 'value'] - treasury_2y.loc[latest_date, 'value']
                            indicators['yield_curve_spread'] = {
                                'value': spread,
                                'signal': 'Inverted (recession warning)' if spread < 0 else 'Normal',
                                'date': latest_date
                            }
                except:
                    pass
            
            # Add other recession indicators
            for name, series_id in [('unemployment_rate', 'UNRATE'), ('gdp_growth', 'A191RL1Q225SBEA')]:
                try:
                    data = self.get_series_data(series_id, limit=5)
                    if not data.empty:
                        latest_value = data.iloc[-1]['value']
                        indicators[name] = {
                            'value': latest_value,
                            'date': data.index[-1]
                        }
                except:
                    continue
            
            # Generate recession probability (simplified)
            recession_probability = 0.15  # Base probability
            
            if 'yield_curve_spread' in indicators and indicators['yield_curve_spread']['value'] < 0:
                recession_probability += 0.25
            
            if 'unemployment_rate' in indicators and indicators['unemployment_rate']['value'] > 5.0:
                recession_probability += 0.20
            
            recession_probability = min(recession_probability, 0.85)
            
            return {
                'recession_probability': recession_probability,
                'indicators': indicators,
                'assessment': self._generate_recession_assessment(recession_probability),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting recession indicators: {str(e)}")
            return {
                'recession_probability': 0.15,
                'indicators': {},
                'assessment': 'Recession assessment unavailable',
                'timestamp': datetime.now()
            }
    
    def _generate_recession_assessment(self, probability: float) -> str:
        """Generate recession assessment based on probability"""
        if probability < 0.20:
            return "Low recession risk - economic indicators suggest continued expansion"
        elif probability < 0.40:
            return "Moderate recession risk - some warning signs present, monitor closely"
        elif probability < 0.60:
            return "Elevated recession risk - multiple indicators showing concern"
        else:
            return "High recession risk - significant economic warning signs present"
