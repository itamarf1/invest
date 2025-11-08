import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good" 
    POOR = "poor"
    STALE = "stale"


@dataclass
class DataSourceResult:
    source: str
    data: pd.DataFrame
    quality: DataQuality
    confidence: float
    issues: List[str]
    retrieved_at: datetime


class MultiSourceDataFetcher:
    """
    Advanced market data fetcher that cross-validates multiple data sources
    and caches the most reliable historical data for accurate analysis.
    """
    
    def __init__(self, cache_expiry_minutes: int = 30):
        self.cache_expiry_minutes = cache_expiry_minutes
        self.validated_cache = {}
        self.cache_timestamps = {}
        
    def get_validated_stock_data(self, symbol: str, period: str = "1y", 
                                interval: str = "1d") -> pd.DataFrame:
        """
        Get most reliable stock data by cross-validating multiple sources
        """
        try:
            cache_key = f"{symbol}_{period}_{interval}_validated"
            
            # Check validated cache first
            if self._is_validated_cache_valid(cache_key):
                logger.info(f"Using validated cached data for {symbol}")
                return self.validated_cache[cache_key]
            
            # Gather data from multiple sources
            sources = self._fetch_from_multiple_sources(symbol, period, interval)
            
            # Validate and select best source
            best_data = self._validate_and_select_best(sources, symbol)
            
            # Cache the validated result
            self.validated_cache[cache_key] = best_data
            self.cache_timestamps[cache_key] = datetime.now()
            
            return best_data
            
        except Exception as e:
            logger.error(f"Error in multi-source fetch for {symbol}: {str(e)}")
            # Fallback to basic yfinance
            return yf.Ticker(symbol).history(period=period, interval=interval)
    
    def _fetch_from_multiple_sources(self, symbol: str, period: str, 
                                   interval: str) -> List[DataSourceResult]:
        """Fetch data from multiple sources for cross-validation"""
        sources = []
        
        # Source 1: Standard yfinance daily
        try:
            ticker = yf.Ticker(symbol)
            data_daily = ticker.history(period=period, interval=interval)
            quality, confidence, issues = self._assess_data_quality(data_daily, symbol)
            
            sources.append(DataSourceResult(
                source="yfinance_daily",
                data=data_daily,
                quality=quality,
                confidence=confidence,
                issues=issues,
                retrieved_at=datetime.now()
            ))
            
        except Exception as e:
            logger.warning(f"yfinance daily failed for {symbol}: {str(e)}")
        
        # Source 2: Intraday aggregated to daily (for validation)
        if interval == "1d" and period in ["1y", "6mo", "3mo"]:
            try:
                # Get intraday data and aggregate to daily for comparison
                intraday_data = self._get_intraday_aggregated(symbol, period)
                if not intraday_data.empty:
                    quality, confidence, issues = self._assess_data_quality(intraday_data, symbol)
                    
                    sources.append(DataSourceResult(
                        source="yfinance_intraday_agg",
                        data=intraday_data,
                        quality=quality,
                        confidence=confidence,
                        issues=issues,
                        retrieved_at=datetime.now()
                    ))
                    
            except Exception as e:
                logger.warning(f"Intraday aggregation failed for {symbol}: {str(e)}")
        
        # Source 3: Multiple intervals cross-check
        if interval == "1h":
            try:
                # Also try 30min and aggregate up for validation
                data_30min = ticker.history(period=period, interval="30m")
                if not data_30min.empty:
                    # Aggregate 30min to 1hour
                    aggregated = self._aggregate_to_hourly(data_30min)
                    quality, confidence, issues = self._assess_data_quality(aggregated, symbol)
                    
                    sources.append(DataSourceResult(
                        source="yfinance_30min_agg",
                        data=aggregated,
                        quality=quality,
                        confidence=confidence,
                        issues=issues,
                        retrieved_at=datetime.now()
                    ))
                    
            except Exception as e:
                logger.warning(f"30min aggregation failed for {symbol}: {str(e)}")
        
        return sources
    
    def _get_intraday_aggregated(self, symbol: str, period: str) -> pd.DataFrame:
        """Get intraday data and aggregate to daily for validation"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get hourly data for shorter periods, daily for longer
            if period in ["1mo", "3mo"]:
                intraday = ticker.history(period=period, interval="1h")
                if intraday.empty:
                    return pd.DataFrame()
                
                # Aggregate hourly to daily
                daily_agg = intraday.groupby(intraday.index.date).agg({
                    'Open': 'first',
                    'High': 'max', 
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })
                
                # Convert index back to datetime
                daily_agg.index = pd.to_datetime(daily_agg.index)
                return daily_agg
                
            else:
                # For longer periods, try weekly aggregation
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Intraday aggregation error: {str(e)}")
            return pd.DataFrame()
    
    def _aggregate_to_hourly(self, data_30min: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 30-minute data to hourly"""
        try:
            hourly = data_30min.groupby(pd.Grouper(freq='1H')).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min', 
                'Close': 'last',
                'Volume': 'sum'
            })
            return hourly.dropna()
        except Exception as e:
            logger.error(f"Hourly aggregation error: {str(e)}")
            return pd.DataFrame()
    
    def _assess_data_quality(self, data: pd.DataFrame, symbol: str) -> Tuple[DataQuality, float, List[str]]:
        """Assess the quality of market data"""
        issues = []
        confidence = 1.0
        
        if data.empty:
            return DataQuality.STALE, 0.0, ["No data available"]
        
        # Check for stale/flat data (major red flag for warrants)
        unique_closes = data['Close'].nunique()
        total_points = len(data)
        
        if unique_closes == 1:
            issues.append("All closing prices identical - likely stale data")
            confidence *= 0.1
            
        elif unique_closes < max(3, total_points * 0.1):
            issues.append("Very few unique prices - possible stale data")
            confidence *= 0.3
        
        # Check for zero volume (especially bad for warrants)
        total_volume = data['Volume'].sum()
        zero_volume_days = (data['Volume'] == 0).sum()
        
        if total_volume == 0:
            issues.append("Zero trading volume - possibly delisted or illiquid")
            confidence *= 0.2
        elif zero_volume_days > total_points * 0.8:
            issues.append("Mostly zero volume days - low liquidity")
            confidence *= 0.5
        
        # Check for unrealistic OHLC patterns
        ohlc_identical = 0
        for _, row in data.iterrows():
            if row['Open'] == row['High'] == row['Low'] == row['Close']:
                ohlc_identical += 1
        
        if ohlc_identical > total_points * 0.8:
            issues.append("OHLC values frequently identical - suspicious")
            confidence *= 0.3
        
        # Check for data freshness (especially important for current analysis)
        latest_date = data.index[-1]
        days_old = (datetime.now() - latest_date.to_pydatetime().replace(tzinfo=None)).days
        
        if days_old > 7:
            issues.append(f"Data is {days_old} days old")
            confidence *= max(0.5, 1 - (days_old / 30))
        
        # Special warrant detection (symbols ending in W)
        if symbol.endswith('W'):
            # Warrants should have more volatile pricing
            price_volatility = data['Close'].std() / data['Close'].mean()
            if price_volatility < 0.01:
                issues.append("Warrant shows unrealistically low volatility")
                confidence *= 0.2
        
        # Determine overall quality
        if confidence >= 0.8:
            quality = DataQuality.EXCELLENT
        elif confidence >= 0.6:
            quality = DataQuality.GOOD
        elif confidence >= 0.3:
            quality = DataQuality.POOR
        else:
            quality = DataQuality.STALE
        
        return quality, confidence, issues
    
    def _validate_and_select_best(self, sources: List[DataSourceResult], 
                                 symbol: str) -> pd.DataFrame:
        """Cross-validate sources and select the most reliable data"""
        if not sources:
            raise ValueError("No data sources available")
        
        logger.info(f"Validating {len(sources)} data sources for {symbol}")
        
        # Log quality assessment for each source
        for source in sources:
            logger.info(f"{source.source}: {source.quality.value} (confidence: {source.confidence:.2f})")
            if source.issues:
                logger.info(f"  Issues: {', '.join(source.issues)}")
        
        # If we have multiple sources, cross-validate
        if len(sources) > 1:
            validated_source = self._cross_validate_sources(sources)
            if validated_source:
                logger.info(f"Selected {validated_source.source} after cross-validation")
                return validated_source.data
        
        # Otherwise select best single source
        best_source = max(sources, key=lambda x: x.confidence)
        logger.info(f"Selected {best_source.source} as best single source")
        
        # Apply corrections if needed
        return self._apply_data_corrections(best_source, sources)
    
    def _cross_validate_sources(self, sources: List[DataSourceResult]) -> Optional[DataSourceResult]:
        """Cross-validate multiple data sources to find inconsistencies"""
        try:
            # Compare recent closing prices between sources
            recent_closes = {}
            for source in sources:
                if not source.data.empty:
                    # Get last 5 closing prices
                    recent = source.data['Close'].tail(5).tolist()
                    recent_closes[source.source] = recent
            
            if len(recent_closes) >= 2:
                # Check for major discrepancies
                sources_list = list(recent_closes.keys())
                source1_closes = recent_closes[sources_list[0]]
                source2_closes = recent_closes[sources_list[1]]
                
                # Compare latest prices
                if source1_closes and source2_closes:
                    latest_diff = abs(source1_closes[-1] - source2_closes[-1])
                    price_threshold = max(source1_closes[-1], source2_closes[-1]) * 0.1  # 10% threshold
                    
                    if latest_diff > price_threshold:
                        logger.warning(f"Major price discrepancy detected: {latest_diff:.3f}")
                        # Return the source with better quality
                        return max(sources, key=lambda x: x.confidence)
            
            return None
            
        except Exception as e:
            logger.error(f"Cross-validation error: {str(e)}")
            return None
    
    def _apply_data_corrections(self, best_source: DataSourceResult, 
                              all_sources: List[DataSourceResult]) -> pd.DataFrame:
        """Apply intelligent corrections to improve data quality"""
        data = best_source.data.copy()
        
        if data.empty:
            return data
        
        # For poor quality data, try to get real-time price correction
        if best_source.quality in [DataQuality.POOR, DataQuality.STALE]:
            try:
                # Check if we can get a current price to correct the latest data point
                symbol = None
                for source in all_sources:
                    if hasattr(source, 'symbol'):
                        symbol = source.symbol
                        break
                
                if symbol:
                    ticker = yf.Ticker(symbol)
                    current_info = ticker.info
                    current_price = current_info.get('currentPrice') or current_info.get('regularMarketPrice')
                    
                    if current_price and abs(current_price - data['Close'].iloc[-1]) > data['Close'].iloc[-1] * 0.05:
                        logger.info(f"Correcting stale price: {data['Close'].iloc[-1]:.3f} -> {current_price:.3f}")
                        # Update the most recent close price
                        data.iloc[-1, data.columns.get_loc('Close')] = current_price
                        
                        # Also update High/Low if needed
                        if current_price > data.iloc[-1]['High']:
                            data.iloc[-1, data.columns.get_loc('High')] = current_price
                        if current_price < data.iloc[-1]['Low']:
                            data.iloc[-1, data.columns.get_loc('Low')] = current_price
                            
            except Exception as e:
                logger.warning(f"Could not apply real-time correction: {str(e)}")
        
        return data
    
    def _is_validated_cache_valid(self, cache_key: str) -> bool:
        """Check if validated cache is still valid"""
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[cache_key]
        now = datetime.now()
        expiry = cache_time + timedelta(minutes=self.cache_expiry_minutes)
        
        return now < expiry
    
    def get_data_quality_report(self, symbol: str) -> Dict[str, Any]:
        """Get detailed data quality report for a symbol"""
        try:
            sources = self._fetch_from_multiple_sources(symbol, "3mo", "1d")
            
            report = {
                "symbol": symbol,
                "sources_checked": len(sources),
                "sources": [],
                "recommendation": "",
                "overall_quality": "unknown"
            }
            
            for source in sources:
                report["sources"].append({
                    "name": source.source,
                    "quality": source.quality.value,
                    "confidence": source.confidence,
                    "issues": source.issues,
                    "data_points": len(source.data) if not source.data.empty else 0
                })
            
            if sources:
                best = max(sources, key=lambda x: x.confidence)
                report["recommendation"] = f"Use {best.source}"
                report["overall_quality"] = best.quality.value
            
            return report
            
        except Exception as e:
            return {"error": str(e)}