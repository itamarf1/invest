import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any, Union
import logging
import asyncio
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from .alpha_vantage_client import AlphaVantageClient
from .stooq_downloader import StooqDownloader
from .data_source_validator import DataSourceValidator, DataQuality, ValidationResult
import yfinance as yf

logger = logging.getLogger(__name__)

@dataclass
class DataSourcePriority:
    """Configuration for data source priority and usage"""
    primary_source: str = "yahoo"  # yahoo, stooq, alpha_vantage
    fallback_sources: List[str] = None  # Auto-determined if None
    quality_threshold: float = 75.0  # Minimum quality score to use a source
    max_age_hours: int = 24  # Max age for cached data
    enable_cross_validation: bool = True
    validation_sample_size: int = 30  # Days to sample for validation

@dataclass
class DataRequest:
    """Standardized data request format"""
    symbol: str
    start_date: str
    end_date: str
    interval: str = "1d"  # 1d, 1h, 5m, etc.
    include_validation: bool = False
    force_refresh: bool = False

@dataclass
class DataResponse:
    """Standardized data response with metadata"""
    symbol: str
    data: pd.DataFrame
    source: str  # primary source used
    quality_score: float
    validation_result: Optional[ValidationResult] = None
    metadata: Dict[str, Any] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class HybridDataPipeline:
    """
    Intelligent data pipeline that combines Yahoo Finance, Stooq, and Alpha Vantage
    with automatic quality validation and smart fallback logic
    """
    
    def __init__(self, 
                 alpha_vantage_key: str = None,
                 stooq_data_dir: str = "data/stooq",
                 cache_dir: str = "data/cache"):
        
        # Initialize data sources
        self.yahoo_available = True  # Always available
        
        self.alpha_vantage = AlphaVantageClient(alpha_vantage_key)
        self.alpha_vantage_available = self.alpha_vantage.is_available()
        
        self.stooq = StooqDownloader(stooq_data_dir)
        self.stooq_available = True  # Always available for download
        
        self.validator = DataSourceValidator(alpha_vantage_key)
        
        # Cache setup
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = DataSourcePriority()
        
        # Quality monitoring
        self.quality_history = {}  # symbol -> [quality_scores]
        self.source_reliability = {
            'yahoo': [],
            'stooq': [],
            'alpha_vantage': []
        }
        
        logger.info(f"Hybrid data pipeline initialized")
        logger.info(f"  Yahoo Finance: ✓")
        logger.info(f"  Alpha Vantage: {'✓' if self.alpha_vantage_available else '✗'}")
        logger.info(f"  Stooq: ✓")
    
    def configure(self, config: DataSourcePriority):
        """Update pipeline configuration"""
        self.config = config
        logger.info(f"Pipeline configured with primary source: {config.primary_source}")
    
    async def get_data(self, request: DataRequest) -> DataResponse:
        """Get data with intelligent source selection and validation"""
        try:
            logger.info(f"Processing data request: {request.symbol} ({request.start_date} to {request.end_date})")
            
            # Determine source priority for this symbol
            source_priority = self._determine_source_priority(request.symbol)
            
            # Try sources in priority order
            best_response = None
            validation_result = None
            
            for source in source_priority:
                try:
                    logger.debug(f"Trying source: {source} for {request.symbol}")
                    
                    # Get data from source
                    data, quality_score = await self._get_data_from_source(
                        source, request
                    )
                    
                    if data.empty:
                        logger.warning(f"No data from {source} for {request.symbol}")
                        continue
                    
                    # Check quality threshold
                    if quality_score < self.config.quality_threshold:
                        logger.warning(f"Quality score {quality_score:.1f} below threshold for {source}")
                        continue
                    
                    # Create response
                    response = DataResponse(
                        symbol=request.symbol,
                        data=data,
                        source=source,
                        quality_score=quality_score,
                        metadata={
                            'request': asdict(request),
                            'source_priority': source_priority,
                            'data_points': len(data)
                        }
                    )
                    
                    # If this is our first good response, save it
                    if best_response is None:
                        best_response = response
                    
                    # If quality is excellent, use it immediately
                    if quality_score >= 95.0:
                        logger.info(f"Excellent quality data from {source} ({quality_score:.1f}/100)")
                        best_response = response
                        break
                    
                    # If this is better than our current best, use it
                    if quality_score > best_response.quality_score:
                        best_response = response
                    
                except Exception as e:
                    logger.error(f"Error getting data from {source}: {str(e)}")
                    continue
            
            if best_response is None:
                logger.error(f"No usable data found for {request.symbol}")
                return DataResponse(
                    symbol=request.symbol,
                    data=pd.DataFrame(),
                    source="none",
                    quality_score=0.0,
                    metadata={'error': 'No usable data sources'}
                )
            
            # Perform cross-validation if enabled
            if request.include_validation and self.config.enable_cross_validation:
                validation_result = await self._validate_data(request, best_response)
                best_response.validation_result = validation_result
            
            # Update quality monitoring
            self._update_quality_monitoring(best_response.source, best_response.quality_score)
            
            logger.info(f"Data request completed: {request.symbol} from {best_response.source} "
                       f"({best_response.quality_score:.1f}/100, {len(best_response.data)} records)")
            
            return best_response
            
        except Exception as e:
            logger.error(f"Error in hybrid data pipeline for {request.symbol}: {str(e)}")
            return DataResponse(
                symbol=request.symbol,
                data=pd.DataFrame(),
                source="error",
                quality_score=0.0,
                metadata={'error': str(e)}
            )
    
    async def _get_data_from_source(self, source: str, request: DataRequest) -> Tuple[pd.DataFrame, float]:
        """Get data from a specific source"""
        
        if source == "yahoo":
            return await self._get_yahoo_data(request)
        elif source == "stooq":
            return await self._get_stooq_data(request)
        elif source == "alpha_vantage":
            return await self._get_alpha_vantage_data(request)
        else:
            raise ValueError(f"Unknown data source: {source}")
    
    async def _get_yahoo_data(self, request: DataRequest) -> Tuple[pd.DataFrame, float]:
        """Get data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(request.symbol)
            data = ticker.history(
                start=request.start_date,
                end=request.end_date,
                interval=request.interval
            )
            
            if data.empty:
                return pd.DataFrame(), 0.0
            
            # Standardize columns
            data = data.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })
            data = data[['open', 'high', 'low', 'close', 'volume']]
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(data, request.symbol)
            
            return data, quality_score
            
        except Exception as e:
            logger.error(f"Error getting Yahoo data: {str(e)}")
            return pd.DataFrame(), 0.0
    
    async def _get_stooq_data(self, request: DataRequest) -> Tuple[pd.DataFrame, float]:
        """Get data from Stooq"""
        try:
            # For now, use existing data if available, otherwise download
            data = self.stooq.get_symbol_data(
                request.symbol,
                start_date=request.start_date,
                end_date=request.end_date
            )
            
            if data.empty:
                # Try to download it
                logger.info(f"Downloading {request.symbol} from Stooq...")
                stooq_data = self.stooq.download_symbol_from_stooq(
                    request.symbol,
                    start_date=request.start_date.replace('-', '')
                )
                
                if not stooq_data.empty:
                    # Filter to requested date range
                    stooq_data['date'] = pd.to_datetime(stooq_data['date'])
                    mask = (
                        (stooq_data['date'] >= pd.to_datetime(request.start_date)) &
                        (stooq_data['date'] <= pd.to_datetime(request.end_date))
                    )
                    data = stooq_data[mask].copy()
                    data = data.set_index('date')
            
            if data.empty:
                return pd.DataFrame(), 0.0
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(data, request.symbol)
            
            return data, quality_score
            
        except Exception as e:
            logger.error(f"Error getting Stooq data: {str(e)}")
            return pd.DataFrame(), 0.0
    
    async def _get_alpha_vantage_data(self, request: DataRequest) -> Tuple[pd.DataFrame, float]:
        """Get data from Alpha Vantage"""
        try:
            if not self.alpha_vantage_available:
                return pd.DataFrame(), 0.0
            
            # Choose function based on interval
            if request.interval == "1d":
                data = self.alpha_vantage.get_daily_data(request.symbol, "full")
            else:
                # Map intervals
                interval_map = {
                    "1h": "60min", "5m": "5min", "15m": "15min", "30m": "30min"
                }
                av_interval = interval_map.get(request.interval, "5min")
                data = self.alpha_vantage.get_intraday_data(request.symbol, av_interval, "full")
            
            if data.empty:
                return pd.DataFrame(), 0.0
            
            # Filter to date range
            start = pd.to_datetime(request.start_date)
            end = pd.to_datetime(request.end_date)
            data = data[(data.index >= start) & (data.index <= end)]
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(data, request.symbol)
            
            return data, quality_score
            
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage data: {str(e)}")
            return pd.DataFrame(), 0.0
    
    def _determine_source_priority(self, symbol: str) -> List[str]:
        """Determine source priority for a given symbol"""
        
        # Start with configured primary source
        priority = [self.config.primary_source]
        
        # Add fallback sources
        all_sources = ["yahoo", "stooq", "alpha_vantage"]
        
        if self.config.fallback_sources:
            for source in self.config.fallback_sources:
                if source not in priority and source in all_sources:
                    priority.append(source)
        else:
            # Auto-determine fallbacks based on symbol characteristics and history
            remaining_sources = [s for s in all_sources if s not in priority]
            
            # Special logic for warrants (typically have issues with Yahoo)
            if symbol.endswith('W'):
                priority = ["stooq", "alpha_vantage", "yahoo"]
            
            # Check historical reliability for this symbol
            if symbol in self.quality_history:
                source_scores = {}
                for source in remaining_sources:
                    if source in self.source_reliability:
                        recent_scores = self.source_reliability[source][-10:]  # Last 10
                        if recent_scores:
                            source_scores[source] = np.mean(recent_scores)
                
                # Sort by reliability
                sorted_sources = sorted(source_scores.items(), key=lambda x: x[1], reverse=True)
                priority.extend([source for source, score in sorted_sources])
            
            # Add any remaining sources
            for source in remaining_sources:
                if source not in priority:
                    priority.append(source)
        
        # Filter out unavailable sources
        available_priority = []
        for source in priority:
            if source == "yahoo" and self.yahoo_available:
                available_priority.append(source)
            elif source == "stooq" and self.stooq_available:
                available_priority.append(source)
            elif source == "alpha_vantage" and self.alpha_vantage_available:
                available_priority.append(source)
        
        return available_priority
    
    def _calculate_quality_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Calculate quality score for data"""
        if data.empty:
            return 0.0
        
        try:
            score = 100.0
            
            # Completeness (no missing values)
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            score -= missing_ratio * 30
            
            # Volume activity
            if 'volume' in data.columns:
                zero_volume_ratio = len(data[data['volume'] == 0]) / len(data)
                score -= zero_volume_ratio * 25
            
            # Price consistency (not too many identical consecutive prices)
            if len(data) > 1:
                price_changes = data['close'].diff().fillna(0)
                no_change_ratio = len(price_changes[price_changes == 0]) / len(data)
                if no_change_ratio > 0.3:  # More than 30% no change is suspicious
                    score -= (no_change_ratio - 0.1) * 40
            
            # Extreme price movements (red flag for data errors)
            if len(data) > 1:
                price_changes = data['close'].pct_change().fillna(0)
                extreme_changes = len(price_changes[abs(price_changes) > 0.5])  # >50%
                if extreme_changes > 0:
                    score -= min(extreme_changes / len(data) * 100, 25)
            
            # Bonus for recency and size
            if len(data) > 100:
                score += 5  # Bonus for substantial data
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {str(e)}")
            return 50.0
    
    async def _validate_data(self, request: DataRequest, response: DataResponse) -> ValidationResult:
        """Cross-validate data against other sources"""
        try:
            # Create a validation request for a shorter period (to save API calls)
            val_end = pd.to_datetime(request.end_date)
            val_start = val_end - timedelta(days=self.config.validation_sample_size)
            
            val_request = DataRequest(
                symbol=request.symbol,
                start_date=val_start.strftime('%Y-%m-%d'),
                end_date=val_end.strftime('%Y-%m-%d'),
                interval=request.interval,
                include_validation=False  # Avoid recursion
            )
            
            # Get data from other sources for comparison
            comparison_data = {}
            available_sources = ["yahoo", "stooq", "alpha_vantage"]
            
            for source in available_sources:
                if source == response.source:
                    continue  # Skip the source we already used
                
                try:
                    data, quality = await self._get_data_from_source(source, val_request)
                    if not data.empty and quality > 50:
                        comparison_data[source] = data
                except:
                    continue
            
            if not comparison_data:
                logger.warning(f"No comparison data available for validation of {request.symbol}")
                return None
            
            # Perform comparison using existing validator logic
            # This is simplified - you could use the full ValidationResult logic here
            correlations = {}
            discrepancies = {}
            
            for source, data in comparison_data.items():
                # Calculate correlation with main data
                common_dates = response.data.index.intersection(data.index)
                if len(common_dates) > 5:
                    main_prices = response.data.loc[common_dates, 'close']
                    comp_prices = data.loc[common_dates, 'close']
                    correlation = main_prices.corr(comp_prices)
                    correlations[f"{response.source}_vs_{source}"] = correlation
            
            return ValidationResult(
                symbol=request.symbol,
                test_period=f"{self.config.validation_sample_size} days",
                sources_compared=[response.source] + list(comparison_data.keys()),
                price_correlation=correlations,
                volume_correlation={},  # Simplified
                discrepancies=discrepancies,
                recommended_source=response.source,
                confidence=response.quality_score / 100,
                quality_scores={}  # Simplified
            )
            
        except Exception as e:
            logger.error(f"Error in data validation: {str(e)}")
            return None
    
    def _update_quality_monitoring(self, source: str, quality_score: float):
        """Update quality monitoring statistics"""
        if source in self.source_reliability:
            self.source_reliability[source].append(quality_score)
            # Keep only last 100 scores
            if len(self.source_reliability[source]) > 100:
                self.source_reliability[source] = self.source_reliability[source][-100:]
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics"""
        stats = {
            'configuration': asdict(self.config),
            'source_availability': {
                'yahoo': self.yahoo_available,
                'stooq': self.stooq_available,
                'alpha_vantage': self.alpha_vantage_available
            },
            'source_reliability': {}
        }
        
        # Calculate reliability stats
        for source, scores in self.source_reliability.items():
            if scores:
                stats['source_reliability'][source] = {
                    'average_quality': np.mean(scores),
                    'recent_quality': np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores),
                    'samples': len(scores),
                    'min_quality': min(scores),
                    'max_quality': max(scores)
                }
        
        if self.alpha_vantage_available:
            stats['alpha_vantage_usage'] = self.alpha_vantage.get_api_usage_stats()
        
        return stats
    
    async def batch_get_data(self, requests: List[DataRequest]) -> List[DataResponse]:
        """Process multiple data requests efficiently"""
        logger.info(f"Processing batch request for {len(requests)} symbols")
        
        # Group requests by source priority to optimize API usage
        responses = []
        
        for request in requests:
            response = await self.get_data(request)
            responses.append(response)
            
            # Brief pause between requests to be respectful to APIs
            await asyncio.sleep(0.1)
        
        logger.info(f"Batch processing completed: {len(responses)} responses")
        return responses