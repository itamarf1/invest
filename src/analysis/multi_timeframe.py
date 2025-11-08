import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from src.data.market import MarketDataFetcher
from src.trading.strategies.rsi_strategy import RSIStrategy
from src.trading.strategies.macd_strategy import MACDStrategy
from src.trading.strategies.bollinger_strategy import BollingerBandsStrategy

logger = logging.getLogger(__name__)

class Timeframe(Enum):
    """Available timeframes for analysis"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1wk"
    MONTH_1 = "1mo"

@dataclass
class TimeframeSignal:
    timeframe: str
    signal: str  # 'buy', 'sell', 'hold'
    confidence: float
    price: float
    indicators: Dict[str, float]
    trend: str  # 'uptrend', 'downtrend', 'sideways'
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    volume_trend: str = "neutral"  # 'increasing', 'decreasing', 'neutral'

@dataclass
class MultiTimeframeAnalysis:
    symbol: str
    timeframe_signals: Dict[str, TimeframeSignal]
    consensus_signal: str
    consensus_confidence: float
    trend_alignment: Dict[str, str]  # timeframe -> trend
    key_levels: Dict[str, List[float]]  # support/resistance by timeframe
    volume_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    timestamp: datetime

class TrendAnalyzer:
    """Analyze trends across different timeframes"""
    
    @staticmethod
    def identify_trend(data: pd.DataFrame, window: int = 20) -> Tuple[str, float]:
        """Identify trend direction and strength"""
        try:
            if len(data) < window:
                return "sideways", 0.0
            
            close_prices = data['Close']
            
            # Moving averages for trend identification
            ma_short = close_prices.rolling(window=window//2).mean()
            ma_long = close_prices.rolling(window=window).mean()
            
            current_price = close_prices.iloc[-1]
            ma_short_current = ma_short.iloc[-1]
            ma_long_current = ma_long.iloc[-1]
            
            # Trend direction
            if current_price > ma_short_current > ma_long_current:
                trend = "uptrend"
                # Strength based on angle of MA
                ma_slope = (ma_short_current - ma_short.iloc[-5]) / 5
                strength = min(abs(ma_slope) / (current_price * 0.01), 1.0)
            elif current_price < ma_short_current < ma_long_current:
                trend = "downtrend"
                ma_slope = (ma_short_current - ma_short.iloc[-5]) / 5
                strength = min(abs(ma_slope) / (current_price * 0.01), 1.0)
            else:
                trend = "sideways"
                # Measure consolidation tightness
                recent_range = close_prices.tail(window).max() - close_prices.tail(window).min()
                strength = 1.0 - min(recent_range / current_price, 0.5) * 2
            
            return trend, strength
            
        except Exception as e:
            logger.error(f"Error identifying trend: {str(e)}")
            return "sideways", 0.0
    
    @staticmethod
    def find_support_resistance(data: pd.DataFrame, lookback: int = 50) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels"""
        try:
            if len(data) < lookback:
                return [], []
            
            high_prices = data['High'].tail(lookback)
            low_prices = data['Low'].tail(lookback)
            
            # Find local maxima (resistance) and minima (support)
            resistance_levels = []
            support_levels = []
            
            # Simple peak/trough detection
            for i in range(2, len(high_prices) - 2):
                # Resistance: local maximum
                if (high_prices.iloc[i] > high_prices.iloc[i-1] and 
                    high_prices.iloc[i] > high_prices.iloc[i-2] and
                    high_prices.iloc[i] > high_prices.iloc[i+1] and 
                    high_prices.iloc[i] > high_prices.iloc[i+2]):
                    resistance_levels.append(high_prices.iloc[i])
                
                # Support: local minimum
                if (low_prices.iloc[i] < low_prices.iloc[i-1] and 
                    low_prices.iloc[i] < low_prices.iloc[i-2] and
                    low_prices.iloc[i] < low_prices.iloc[i+1] and 
                    low_prices.iloc[i] < low_prices.iloc[i+2]):
                    support_levels.append(low_prices.iloc[i])
            
            # Remove duplicates and sort
            resistance_levels = sorted(list(set([round(r, 2) for r in resistance_levels])), reverse=True)[:3]
            support_levels = sorted(list(set([round(s, 2) for s in support_levels])))[-3:]
            
            return support_levels, resistance_levels
            
        except Exception as e:
            logger.error(f"Error finding support/resistance: {str(e)}")
            return [], []

class VolumeAnalyzer:
    """Analyze volume patterns across timeframes"""
    
    @staticmethod
    def analyze_volume_trend(data: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        """Analyze volume trends and patterns"""
        try:
            if len(data) < window:
                return {"trend": "neutral", "strength": 0.0, "anomalies": []}
            
            volume = data['Volume']
            close_prices = data['Close']
            
            # Volume moving average
            volume_ma = volume.rolling(window=window).mean()
            current_volume = volume.iloc[-1]
            current_volume_ma = volume_ma.iloc[-1]
            
            # Volume trend
            recent_volume_ma = volume_ma.tail(5).mean()
            older_volume_ma = volume_ma.tail(window).head(5).mean()
            
            if recent_volume_ma > older_volume_ma * 1.1:
                volume_trend = "increasing"
            elif recent_volume_ma < older_volume_ma * 0.9:
                volume_trend = "decreasing"
            else:
                volume_trend = "neutral"
            
            # Volume strength (current vs average)
            volume_ratio = current_volume / current_volume_ma if current_volume_ma > 0 else 1.0
            
            # Price-volume divergence
            price_change = (close_prices.iloc[-1] - close_prices.iloc[-5]) / close_prices.iloc[-5]
            volume_change = (recent_volume_ma - older_volume_ma) / older_volume_ma
            
            # Detect volume anomalies (spikes)
            volume_threshold = current_volume_ma * 2
            volume_spikes = []
            
            for i in range(-10, 0):
                if i < -len(volume):
                    continue
                if volume.iloc[i] > volume_threshold:
                    volume_spikes.append({
                        "date": data.index[i],
                        "volume": volume.iloc[i],
                        "ratio": volume.iloc[i] / current_volume_ma
                    })
            
            return {
                "trend": volume_trend,
                "strength": min(abs(volume_ratio - 1), 2.0),
                "current_ratio": volume_ratio,
                "price_volume_divergence": abs(price_change - volume_change),
                "volume_spikes": volume_spikes,
                "average_volume": current_volume_ma
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume: {str(e)}")
            return {"trend": "neutral", "strength": 0.0, "anomalies": []}

class MultiTimeframeAnalyzer:
    """Comprehensive multi-timeframe analysis system"""
    
    def __init__(self):
        self.market_fetcher = MarketDataFetcher()
        self.rsi_strategy = RSIStrategy()
        self.macd_strategy = MACDStrategy()
        self.bollinger_strategy = BollingerBandsStrategy()
        self.trend_analyzer = TrendAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        
        # Timeframe configurations
        self.timeframe_configs = {
            "1d": {"period": "6mo", "window": 20},
            "1wk": {"period": "2y", "window": 12},  
            "1mo": {"period": "5y", "window": 6}
        }
    
    def analyze_timeframe(self, symbol: str, timeframe: str) -> TimeframeSignal:
        """Analyze a single timeframe"""
        try:
            config = self.timeframe_configs.get(timeframe, {"period": "6mo", "window": 20})
            
            # Get market data
            data = self.market_fetcher.get_stock_data(symbol, period=config["period"], interval=timeframe)
            if data.empty:
                raise ValueError(f"No data available for {symbol} on {timeframe}")
            
            current_price = data['Close'].iloc[-1]
            
            # Technical indicator signals
            rsi_signal = self.rsi_strategy.analyze(data)
            macd_signal = self.macd_strategy.analyze(data)
            bollinger_signal = self.bollinger_strategy.analyze(data)
            
            # Trend analysis
            trend, trend_strength = self.trend_analyzer.identify_trend(data, config["window"])
            
            # Support/Resistance levels
            support_levels, resistance_levels = self.trend_analyzer.find_support_resistance(data)
            support_level = support_levels[-1] if support_levels else None
            resistance_level = resistance_levels[0] if resistance_levels else None
            
            # Volume analysis
            volume_analysis = self.volume_analyzer.analyze_volume_trend(data, config["window"])
            
            # Combine signals
            signals = [rsi_signal, macd_signal, bollinger_signal]
            buy_signals = sum(1 for s in signals if s.get('signal') == 'buy')
            sell_signals = sum(1 for s in signals if s.get('signal') == 'sell')
            
            if buy_signals >= 2:
                consensus_signal = 'buy'
                confidence = (buy_signals / len(signals)) * trend_strength
            elif sell_signals >= 2:
                consensus_signal = 'sell'
                confidence = (sell_signals / len(signals)) * trend_strength
            else:
                consensus_signal = 'hold'
                confidence = 0.5 * trend_strength
            
            # Collect indicators
            indicators = {
                'rsi': rsi_signal.get('rsi', 50),
                'macd': macd_signal.get('macd', 0),
                'macd_signal': macd_signal.get('signal', 0),
                'bb_position': bollinger_signal.get('bb_position', 0.5),
                'trend_strength': trend_strength,
                'volume_ratio': volume_analysis['current_ratio']
            }
            
            return TimeframeSignal(
                timeframe=timeframe,
                signal=consensus_signal,
                confidence=confidence,
                price=current_price,
                indicators=indicators,
                trend=trend,
                support_level=support_level,
                resistance_level=resistance_level,
                volume_trend=volume_analysis['trend']
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} on {timeframe}: {str(e)}")
            return TimeframeSignal(
                timeframe=timeframe,
                signal='hold',
                confidence=0.0,
                price=0.0,
                indicators={},
                trend='sideways',
                volume_trend='neutral'
            )
    
    def multi_timeframe_analysis(self, symbol: str, timeframes: List[str] = None) -> MultiTimeframeAnalysis:
        """Comprehensive multi-timeframe analysis"""
        try:
            if timeframes is None:
                timeframes = ["1d", "1wk", "1mo"]
            
            # Analyze each timeframe
            timeframe_signals = {}
            for tf in timeframes:
                signal = self.analyze_timeframe(symbol, tf)
                timeframe_signals[tf] = signal
            
            # Generate consensus signal
            consensus_signal, consensus_confidence = self._generate_consensus(timeframe_signals)
            
            # Trend alignment analysis
            trend_alignment = {tf: signal.trend for tf, signal in timeframe_signals.items()}
            
            # Collect key levels across timeframes
            key_levels = {}
            for tf, signal in timeframe_signals.items():
                levels = []
                if signal.support_level:
                    levels.append(signal.support_level)
                if signal.resistance_level:
                    levels.append(signal.resistance_level)
                key_levels[tf] = levels
            
            # Volume analysis summary
            volume_analysis = {
                tf: {
                    'trend': signal.volume_trend,
                    'volume_ratio': signal.indicators.get('volume_ratio', 1.0)
                }
                for tf, signal in timeframe_signals.items()
            }
            
            # Risk assessment
            risk_assessment = self._assess_risk(timeframe_signals, consensus_signal)
            
            return MultiTimeframeAnalysis(
                symbol=symbol,
                timeframe_signals=timeframe_signals,
                consensus_signal=consensus_signal,
                consensus_confidence=consensus_confidence,
                trend_alignment=trend_alignment,
                key_levels=key_levels,
                volume_analysis=volume_analysis,
                risk_assessment=risk_assessment,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis for {symbol}: {str(e)}")
            raise
    
    def _generate_consensus(self, timeframe_signals: Dict[str, TimeframeSignal]) -> Tuple[str, float]:
        """Generate consensus signal across timeframes"""
        try:
            if not timeframe_signals:
                return 'hold', 0.0
            
            # Weight signals by timeframe importance
            timeframe_weights = {"1mo": 0.4, "1wk": 0.35, "1d": 0.25}
            
            weighted_scores = {"buy": 0, "sell": 0, "hold": 0}
            total_weight = 0
            
            for tf, signal in timeframe_signals.items():
                weight = timeframe_weights.get(tf, 0.1)
                confidence_weight = weight * signal.confidence
                
                weighted_scores[signal.signal] += confidence_weight
                total_weight += weight
            
            if total_weight == 0:
                return 'hold', 0.0
            
            # Normalize scores
            for signal in weighted_scores:
                weighted_scores[signal] /= total_weight
            
            # Find consensus
            consensus_signal = max(weighted_scores.items(), key=lambda x: x[1])[0]
            consensus_confidence = weighted_scores[consensus_signal]
            
            return consensus_signal, consensus_confidence
            
        except Exception as e:
            logger.error(f"Error generating consensus: {str(e)}")
            return 'hold', 0.0
    
    def _assess_risk(self, timeframe_signals: Dict[str, TimeframeSignal], consensus_signal: str) -> Dict[str, Any]:
        """Assess trading risk across timeframes"""
        try:
            # Signal alignment (all timeframes agreeing)
            signals = [signal.signal for signal in timeframe_signals.values()]
            signal_agreement = signals.count(consensus_signal) / len(signals)
            
            # Trend alignment
            trends = [signal.trend for signal in timeframe_signals.values()]
            uptrend_count = trends.count('uptrend')
            downtrend_count = trends.count('downtrend')
            sideways_count = trends.count('sideways')
            
            trend_consistency = max(uptrend_count, downtrend_count, sideways_count) / len(trends)
            
            # Volatility risk (based on trend strength variance)
            trend_strengths = [signal.indicators.get('trend_strength', 0) for signal in timeframe_signals.values()]
            volatility_risk = np.std(trend_strengths) if trend_strengths else 0
            
            # Volume confirmation
            volume_trends = [signal.volume_trend for signal in timeframe_signals.values()]
            volume_confirmation = volume_trends.count('increasing') / len(volume_trends)
            
            # Overall risk score (lower is better)
            risk_factors = [
                1 - signal_agreement,  # Low agreement = high risk
                1 - trend_consistency,  # Trend conflict = high risk
                volatility_risk,  # High volatility = high risk
                1 - volume_confirmation if consensus_signal == 'buy' else volume_confirmation  # Volume divergence
            ]
            
            overall_risk = np.mean(risk_factors)
            
            # Risk level
            if overall_risk < 0.3:
                risk_level = "low"
            elif overall_risk < 0.6:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            return {
                'overall_risk': overall_risk,
                'risk_level': risk_level,
                'signal_agreement': signal_agreement,
                'trend_consistency': trend_consistency,
                'volatility_risk': volatility_risk,
                'volume_confirmation': volume_confirmation,
                'risk_factors': {
                    'signal_conflict': 1 - signal_agreement,
                    'trend_divergence': 1 - trend_consistency,
                    'high_volatility': volatility_risk,
                    'volume_divergence': abs(0.5 - volume_confirmation)
                }
            }
            
        except Exception as e:
            logger.error(f"Error assessing risk: {str(e)}")
            return {'overall_risk': 0.5, 'risk_level': 'medium'}
    
    def get_timeframe_comparison(self, symbol: str) -> Dict[str, Any]:
        """Get detailed comparison across timeframes"""
        try:
            analysis = self.multi_timeframe_analysis(symbol)
            
            # Create comparison matrix
            comparison = {
                'symbol': symbol,
                'consensus': {
                    'signal': analysis.consensus_signal,
                    'confidence': analysis.consensus_confidence
                },
                'timeframes': {},
                'alignment_score': 0,
                'key_insights': []
            }
            
            # Timeframe details
            for tf, signal in analysis.timeframe_signals.items():
                comparison['timeframes'][tf] = {
                    'signal': signal.signal,
                    'confidence': signal.confidence,
                    'trend': signal.trend,
                    'price': signal.price,
                    'support': signal.support_level,
                    'resistance': signal.resistance_level,
                    'rsi': signal.indicators.get('rsi', 50),
                    'volume_trend': signal.volume_trend
                }
            
            # Calculate alignment score
            signals = [signal.signal for signal in analysis.timeframe_signals.values()]
            alignment_score = max(signals.count('buy'), signals.count('sell'), signals.count('hold')) / len(signals)
            comparison['alignment_score'] = alignment_score
            
            # Generate insights
            insights = []
            
            if alignment_score >= 0.8:
                insights.append(f"Strong consensus across timeframes: {analysis.consensus_signal}")
            
            if analysis.risk_assessment['risk_level'] == 'low':
                insights.append("Low risk opportunity with good signal alignment")
            elif analysis.risk_assessment['risk_level'] == 'high':
                insights.append("High risk due to conflicting signals across timeframes")
            
            # Trend insights
            trends = list(analysis.trend_alignment.values())
            if all(t == 'uptrend' for t in trends):
                insights.append("All timeframes showing uptrend - strong bullish momentum")
            elif all(t == 'downtrend' for t in trends):
                insights.append("All timeframes showing downtrend - strong bearish pressure")
            
            comparison['key_insights'] = insights
            comparison['timestamp'] = analysis.timestamp.isoformat()
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error getting timeframe comparison for {symbol}: {str(e)}")
            raise
