import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
import logging
import requests
import time
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DataQuality:
    source: str
    symbol: str
    start_date: str
    end_date: str
    total_records: int
    missing_days: int
    zero_volume_days: int
    price_anomalies: int  # extreme price jumps
    avg_volume: float
    price_range: Tuple[float, float]  # (min, max)
    data_completeness: float  # percentage
    reliability_score: float  # 0-100

@dataclass
class ValidationResult:
    symbol: str
    test_period: str
    sources_compared: List[str]
    price_correlation: Dict[str, float]  # correlation between sources
    volume_correlation: Dict[str, float]
    discrepancies: Dict[str, List[Dict]]  # major differences
    recommended_source: str
    confidence: float
    quality_scores: Dict[str, DataQuality]

class DataSourceValidator:
    """
    Validates and compares data quality between different sources:
    - Yahoo Finance (yfinance)
    - Alpha Vantage API
    - Stooq data (when available)
    """
    
    def __init__(self, alpha_vantage_key: str = None):
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.rate_limit_delay = 12  # seconds between Alpha Vantage calls
        self.last_av_request = 0
        
        logger.info(f"Data source validator initialized. Alpha Vantage: {'✓' if self.alpha_vantage_key else '✗'}")
    
    def get_yahoo_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if data.empty:
                logger.warning(f"No Yahoo data for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            data = data.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume'
            })
            data['source'] = 'yahoo'
            
            logger.debug(f"Yahoo data for {symbol}: {len(data)} records")
            return data[['open', 'high', 'low', 'close', 'volume', 'source']]
            
        except Exception as e:
            logger.error(f"Error getting Yahoo data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_alpha_vantage_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get data from Alpha Vantage API"""
        if not self.alpha_vantage_key:
            logger.warning("No Alpha Vantage API key provided")
            return pd.DataFrame()
        
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_av_request < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - (current_time - self.last_av_request)
                logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'outputsize': 'full',  # Get full historical data
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params)
            self.last_av_request = time.time()
            
            if response.status_code != 200:
                logger.error(f"Alpha Vantage API error: {response.status_code}")
                return pd.DataFrame()
            
            data_json = response.json()
            
            if 'Error Message' in data_json:
                logger.error(f"Alpha Vantage error: {data_json['Error Message']}")
                return pd.DataFrame()
            
            if 'Note' in data_json:
                logger.warning(f"Alpha Vantage rate limit: {data_json['Note']}")
                return pd.DataFrame()
            
            # Extract time series data
            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data_json:
                logger.error(f"No time series data in Alpha Vantage response for {symbol}")
                return pd.DataFrame()
            
            time_series = data_json[time_series_key]
            
            # Convert to DataFrame
            records = []
            for date_str, values in time_series.items():
                date = pd.to_datetime(date_str)
                
                # Filter by date range
                if start_date and date < pd.to_datetime(start_date):
                    continue
                if end_date and date > pd.to_datetime(end_date):
                    continue
                
                records.append({
                    'date': date,
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume']),
                    'source': 'alpha_vantage'
                })
            
            if not records:
                logger.warning(f"No Alpha Vantage data for {symbol} in date range")
                return pd.DataFrame()
            
            data = pd.DataFrame(records)
            data = data.set_index('date').sort_index()
            
            logger.debug(f"Alpha Vantage data for {symbol}: {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def load_stooq_data(self, symbol: str, stooq_data_dir: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load data from Stooq CSV files (if available)"""
        try:
            # Stooq files are typically named like "aapl.us.txt"
            stooq_file = Path(stooq_data_dir) / f"{symbol.lower()}.us.txt"
            
            if not stooq_file.exists():
                logger.debug(f"No Stooq file for {symbol} at {stooq_file}")
                return pd.DataFrame()
            
            # Read CSV with expected Stooq format
            data = pd.read_csv(stooq_file, names=['date', 'open', 'high', 'low', 'close', 'volume'])
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date').sort_index()
            
            # Filter by date range
            if start_date:
                data = data[data.index >= pd.to_datetime(start_date)]
            if end_date:
                data = data[data.index <= pd.to_datetime(end_date)]
            
            data['source'] = 'stooq'
            
            logger.debug(f"Stooq data for {symbol}: {len(data)} records")
            return data[['open', 'high', 'low', 'close', 'volume', 'source']]
            
        except Exception as e:
            logger.error(f"Error loading Stooq data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def assess_data_quality(self, data: pd.DataFrame, symbol: str, source: str) -> DataQuality:
        """Assess the quality of a dataset"""
        if data.empty:
            return DataQuality(
                source=source, symbol=symbol, start_date='N/A', end_date='N/A',
                total_records=0, missing_days=0, zero_volume_days=0, price_anomalies=0,
                avg_volume=0, price_range=(0, 0), data_completeness=0, reliability_score=0
            )
        
        try:
            # Basic metrics
            total_records = len(data)
            start_date = data.index[0].strftime('%Y-%m-%d')
            end_date = data.index[-1].strftime('%Y-%m-%d')
            
            # Calculate expected trading days (rough estimate: 252 per year)
            date_range = (data.index[-1] - data.index[0]).days
            expected_records = int(date_range * 252 / 365)  # Approximate trading days
            missing_days = max(0, expected_records - total_records)
            
            # Volume analysis
            zero_volume_days = len(data[data['volume'] == 0])
            avg_volume = data['volume'].mean()
            
            # Price anomalies (>10% daily moves)
            price_changes = data['close'].pct_change().abs()
            price_anomalies = len(price_changes[price_changes > 0.10])
            
            # Price range
            price_range = (data['close'].min(), data['close'].max())
            
            # Data completeness
            data_completeness = (total_records / max(expected_records, 1)) * 100
            data_completeness = min(data_completeness, 100)  # Cap at 100%
            
            # Reliability score (composite)
            completeness_score = min(data_completeness, 100) * 0.4
            volume_score = min(100, (1 - zero_volume_days / max(total_records, 1)) * 100) * 0.3
            consistency_score = min(100, (1 - price_anomalies / max(total_records, 1) * 10) * 100) * 0.3
            reliability_score = completeness_score + volume_score + consistency_score
            
            return DataQuality(
                source=source,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                total_records=total_records,
                missing_days=missing_days,
                zero_volume_days=zero_volume_days,
                price_anomalies=price_anomalies,
                avg_volume=avg_volume,
                price_range=price_range,
                data_completeness=data_completeness,
                reliability_score=reliability_score
            )
            
        except Exception as e:
            logger.error(f"Error assessing data quality for {symbol} from {source}: {str(e)}")
            return DataQuality(
                source=source, symbol=symbol, start_date='ERROR', end_date='ERROR',
                total_records=0, missing_days=0, zero_volume_days=0, price_anomalies=0,
                avg_volume=0, price_range=(0, 0), data_completeness=0, reliability_score=0
            )
    
    def compare_sources(self, symbol: str, test_period_days: int = 365, 
                       stooq_data_dir: str = None) -> ValidationResult:
        """Compare data from multiple sources for validation"""
        try:
            # Calculate date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=test_period_days)).strftime('%Y-%m-%d')
            
            logger.info(f"Comparing data sources for {symbol} from {start_date} to {end_date}")
            
            # Get data from all available sources
            datasets = {}
            quality_scores = {}
            
            # Yahoo Finance
            yahoo_data = self.get_yahoo_data(symbol, start_date, end_date)
            if not yahoo_data.empty:
                datasets['yahoo'] = yahoo_data
                quality_scores['yahoo'] = self.assess_data_quality(yahoo_data, symbol, 'yahoo')
            
            # Alpha Vantage
            if self.alpha_vantage_key:
                av_data = self.get_alpha_vantage_data(symbol, start_date, end_date)
                if not av_data.empty:
                    datasets['alpha_vantage'] = av_data
                    quality_scores['alpha_vantage'] = self.assess_data_quality(av_data, symbol, 'alpha_vantage')
            
            # Stooq (if directory provided)
            if stooq_data_dir:
                stooq_data = self.load_stooq_data(symbol, stooq_data_dir, start_date, end_date)
                if not stooq_data.empty:
                    datasets['stooq'] = stooq_data
                    quality_scores['stooq'] = self.assess_data_quality(stooq_data, symbol, 'stooq')
            
            if len(datasets) < 2:
                logger.warning(f"Not enough data sources for comparison (only {len(datasets)})")
                return ValidationResult(
                    symbol=symbol,
                    test_period=f"{test_period_days} days",
                    sources_compared=list(datasets.keys()),
                    price_correlation={},
                    volume_correlation={},
                    discrepancies={},
                    recommended_source=list(datasets.keys())[0] if datasets else 'none',
                    confidence=0.0,
                    quality_scores=quality_scores
                )
            
            # Calculate correlations and discrepancies
            sources = list(datasets.keys())
            price_correlations = {}
            volume_correlations = {}
            discrepancies = {source: [] for source in sources}
            
            # Compare each pair of sources
            for i, source1 in enumerate(sources):
                for j, source2 in enumerate(sources):
                    if i >= j:
                        continue
                    
                    # Align data by date
                    common_dates = datasets[source1].index.intersection(datasets[source2].index)
                    if len(common_dates) < 10:
                        continue
                    
                    data1 = datasets[source1].loc[common_dates]
                    data2 = datasets[source2].loc[common_dates]
                    
                    # Price correlation
                    price_corr = data1['close'].corr(data2['close'])
                    price_correlations[f"{source1}_vs_{source2}"] = price_corr
                    
                    # Volume correlation
                    volume_corr = data1['volume'].corr(data2['volume'])
                    volume_correlations[f"{source1}_vs_{source2}"] = volume_corr
                    
                    # Find significant discrepancies (>5% price difference)
                    price_diff = (data1['close'] - data2['close']).abs() / data1['close']
                    large_diffs = price_diff[price_diff > 0.05]
                    
                    for date, diff in large_diffs.items():
                        discrepancies[source1].append({
                            'date': date.strftime('%Y-%m-%d'),
                            'vs_source': source2,
                            'price_diff_pct': diff * 100,
                            'price_1': data1.loc[date, 'close'],
                            'price_2': data2.loc[date, 'close']
                        })
            
            # Determine recommended source
            if quality_scores:
                best_source = max(quality_scores.keys(), 
                                key=lambda x: quality_scores[x].reliability_score)
                confidence = quality_scores[best_source].reliability_score / 100
            else:
                best_source = 'none'
                confidence = 0.0
            
            return ValidationResult(
                symbol=symbol,
                test_period=f"{test_period_days} days",
                sources_compared=sources,
                price_correlation=price_correlations,
                volume_correlation=volume_correlations,
                discrepancies=discrepancies,
                recommended_source=best_source,
                confidence=confidence,
                quality_scores=quality_scores
            )
            
        except Exception as e:
            logger.error(f"Error comparing sources for {symbol}: {str(e)}")
            return ValidationResult(
                symbol=symbol,
                test_period=f"{test_period_days} days",
                sources_compared=[],
                price_correlation={},
                volume_correlation={},
                discrepancies={},
                recommended_source='error',
                confidence=0.0,
                quality_scores={}
            )
    
    def batch_validate_symbols(self, symbols: List[str], test_period_days: int = 365,
                             stooq_data_dir: str = None) -> Dict[str, ValidationResult]:
        """Validate multiple symbols and generate a comprehensive report"""
        results = {}
        
        logger.info(f"Starting batch validation of {len(symbols)} symbols")
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Validating {symbol} ({i+1}/{len(symbols)})")
            
            try:
                result = self.compare_sources(symbol, test_period_days, stooq_data_dir)
                results[symbol] = result
                
                # Brief pause to respect rate limits
                if self.alpha_vantage_key and i < len(symbols) - 1:
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error validating {symbol}: {str(e)}")
                continue
        
        logger.info(f"Batch validation completed: {len(results)} symbols processed")
        return results
    
    def generate_validation_report(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Generate a comprehensive validation report"""
        if not results:
            return {"error": "No validation results available"}
        
        try:
            # Overall statistics
            total_symbols = len(results)
            symbols_with_data = len([r for r in results.values() if r.sources_compared])
            
            # Source availability
            source_counts = {}
            reliability_by_source = {}
            
            for result in results.values():
                for source in result.sources_compared:
                    source_counts[source] = source_counts.get(source, 0) + 1
                
                for source, quality in result.quality_scores.items():
                    if source not in reliability_by_source:
                        reliability_by_source[source] = []
                    reliability_by_source[source].append(quality.reliability_score)
            
            # Average reliability scores
            avg_reliability = {}
            for source, scores in reliability_by_source.items():
                avg_reliability[source] = sum(scores) / len(scores) if scores else 0
            
            # Correlation analysis
            all_price_correlations = []
            all_volume_correlations = []
            
            for result in results.values():
                all_price_correlations.extend(result.price_correlation.values())
                all_volume_correlations.extend(result.volume_correlation.values())
            
            avg_price_correlation = sum(all_price_correlations) / len(all_price_correlations) if all_price_correlations else 0
            avg_volume_correlation = sum(all_volume_correlations) / len(all_volume_correlations) if all_volume_correlations else 0
            
            # Recommended sources
            source_recommendations = {}
            for result in results.values():
                rec = result.recommended_source
                source_recommendations[rec] = source_recommendations.get(rec, 0) + 1
            
            return {
                "summary": {
                    "total_symbols_tested": total_symbols,
                    "symbols_with_data": symbols_with_data,
                    "data_coverage_rate": symbols_with_data / total_symbols * 100 if total_symbols > 0 else 0
                },
                "source_statistics": {
                    "availability": source_counts,
                    "average_reliability": avg_reliability,
                    "recommendations": source_recommendations
                },
                "correlation_analysis": {
                    "average_price_correlation": avg_price_correlation,
                    "average_volume_correlation": avg_volume_correlation,
                    "price_correlations_distribution": {
                        "min": min(all_price_correlations) if all_price_correlations else 0,
                        "max": max(all_price_correlations) if all_price_correlations else 0,
                        "count": len(all_price_correlations)
                    }
                },
                "detailed_results": {symbol: {
                    "sources": result.sources_compared,
                    "recommended": result.recommended_source,
                    "confidence": result.confidence,
                    "price_correlations": result.price_correlation,
                    "major_discrepancies": sum(len(discs) for discs in result.discrepancies.values()),
                    "quality_scores": {
                        source: {
                            "reliability": q.reliability_score,
                            "completeness": q.data_completeness,
                            "records": q.total_records
                        } for source, q in result.quality_scores.items()
                    }
                } for symbol, result in results.items()},
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating validation report: {str(e)}")
            return {"error": str(e)}