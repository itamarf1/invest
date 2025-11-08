import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
from datetime import datetime

from src.trading.strategies.moving_average import MovingAverageStrategy, SignalType
from src.data.news import NewsFetcher, NewsArticle
from src.analysis.sentiment import SentimentAnalyzer, SentimentImpactCalculator

logger = logging.getLogger(__name__)

class SentimentEnhancedStrategy(MovingAverageStrategy):
    """Moving average strategy enhanced with news sentiment analysis"""
    
    def __init__(self, short_window: int = 20, long_window: int = 50, 
                 sentiment_weight: float = 0.3, news_days_back: int = 7):
        super().__init__(short_window, long_window)
        
        self.sentiment_weight = sentiment_weight  # How much sentiment affects signals
        self.news_days_back = news_days_back
        
        self.news_fetcher = NewsFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.impact_calculator = SentimentImpactCalculator()
        
        self.sentiment_cache = {}  # Cache sentiment results
    
    def get_latest_signal(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Get trading signal enhanced with sentiment analysis"""
        # Get base technical signal
        base_signal = super().get_latest_signal(data, symbol)
        
        # Get sentiment analysis
        sentiment_data = self._get_sentiment_analysis(symbol)
        
        # Enhance signal with sentiment
        enhanced_signal = self._enhance_signal_with_sentiment(base_signal, sentiment_data, data)
        
        return enhanced_signal
    
    def _get_sentiment_analysis(self, symbol: str) -> Dict:
        """Get or fetch sentiment analysis for symbol"""
        cache_key = f"{symbol}_{datetime.now().strftime('%Y-%m-%d-%H')}"
        
        # Check cache (refresh hourly)
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        try:
            # Fetch news articles
            articles = self.news_fetcher.fetch_news_for_symbol(symbol, days_back=self.news_days_back)
            
            if not articles:
                logger.warning(f"No news articles found for {symbol}")
                return self._get_neutral_sentiment()
            
            # Analyze sentiment
            sentiment_result = self.sentiment_analyzer.analyze_multiple_articles(articles)
            
            # Calculate impact
            stock_volatility = self._estimate_volatility(symbol)
            impact_analysis = self.impact_calculator.calculate_impact_score(
                sentiment_result, stock_volatility
            )
            
            # Combine results
            combined_result = {
                'sentiment_score': sentiment_result['overall_sentiment'],
                'sentiment_label': sentiment_result['sentiment_label'],
                'confidence': sentiment_result['confidence'],
                'article_count': sentiment_result['article_count'],
                'impact_score': impact_analysis['impact_score'],
                'estimated_price_impact': impact_analysis['estimated_price_impact_pct'],
                'articles': sentiment_result['articles'][:5]  # Top 5 articles
            }
            
            # Cache result
            self.sentiment_cache[cache_key] = combined_result
            
            logger.info(f"Sentiment analysis for {symbol}: {sentiment_result['sentiment_label']} "
                       f"({sentiment_result['overall_sentiment']:.3f})")
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {str(e)}")
            return self._get_neutral_sentiment()
    
    def _enhance_signal_with_sentiment(self, base_signal: Dict, sentiment_data: Dict, 
                                     price_data: pd.DataFrame) -> Dict:
        """Enhance technical signal with sentiment analysis"""
        
        original_confidence = base_signal.get('confidence', 0.5)
        sentiment_score = sentiment_data['sentiment_score']
        sentiment_confidence = sentiment_data['confidence']
        impact_score = sentiment_data['impact_score']
        
        # Calculate sentiment influence
        sentiment_influence = sentiment_score * self.sentiment_weight * sentiment_confidence
        
        # Adjust signal based on sentiment
        enhanced_action = self._determine_enhanced_action(
            base_signal['action'], sentiment_score, impact_score
        )
        
        # Calculate enhanced confidence
        # Sentiment can either boost or reduce confidence
        confidence_adjustment = 0
        if base_signal['action'] == 'BUY' and sentiment_score > 0:
            confidence_adjustment = sentiment_influence * 0.5
        elif base_signal['action'] == 'SELL' and sentiment_score < 0:
            confidence_adjustment = abs(sentiment_influence) * 0.5
        elif base_signal['action'] == 'BUY' and sentiment_score < -0.2:
            confidence_adjustment = sentiment_influence * 0.7  # Negative adjustment
        elif base_signal['action'] == 'SELL' and sentiment_score > 0.2:
            confidence_adjustment = sentiment_influence * 0.7  # Negative adjustment
        
        enhanced_confidence = min(1.0, max(0.0, original_confidence + confidence_adjustment))
        
        # Create enhanced signal
        enhanced_signal = base_signal.copy()
        enhanced_signal.update({
            'action': enhanced_action,
            'confidence': enhanced_confidence,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_data['sentiment_label'],
            'sentiment_confidence': sentiment_confidence,
            'sentiment_influence': sentiment_influence,
            'impact_score': impact_score,
            'estimated_price_impact': sentiment_data['estimated_price_impact'],
            'article_count': sentiment_data['article_count'],
            'strategy_type': 'sentiment_enhanced',
            'base_action': base_signal['action'],
            'base_confidence': original_confidence,
            'news_articles': sentiment_data['articles']
        })
        
        return enhanced_signal
    
    def _determine_enhanced_action(self, base_action: str, sentiment_score: float, 
                                 impact_score: float) -> str:
        """Determine final action considering sentiment"""
        
        # Strong sentiment can override weak technical signals
        if impact_score > 0.3:  # High impact news
            if sentiment_score > 0.3:
                return 'BUY'
            elif sentiment_score < -0.3:
                return 'SELL'
        
        # Moderate sentiment reinforces existing signals
        if base_action == 'BUY' and sentiment_score < -0.4:
            return 'HOLD'  # Strong negative sentiment overrides buy signal
        elif base_action == 'SELL' and sentiment_score > 0.4:
            return 'HOLD'  # Strong positive sentiment overrides sell signal
        
        return base_action  # Keep original signal
    
    def _estimate_volatility(self, symbol: str, window: int = 30) -> float:
        """Estimate stock volatility for impact calculation"""
        try:
            # This is a simplified estimation
            # In a real system, you'd calculate historical volatility
            volatility_map = {
                'AAPL': 0.25, 'MSFT': 0.22, 'GOOGL': 0.28, 'AMZN': 0.32,
                'TSLA': 0.45, 'META': 0.35, 'NVDA': 0.40, 'NFLX': 0.38
            }
            return volatility_map.get(symbol.upper(), 0.25)
        except:
            return 0.25  # Default volatility
    
    def _get_neutral_sentiment(self) -> Dict:
        """Return neutral sentiment data when analysis fails"""
        return {
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'confidence': 0.0,
            'article_count': 0,
            'impact_score': 0.0,
            'estimated_price_impact': 0.0,
            'articles': []
        }
    
    def get_portfolio_summary(self) -> Dict:
        """Get enhanced portfolio summary including sentiment"""
        base_summary = super().get_portfolio_summary()
        base_summary.update({
            'strategy': 'Sentiment Enhanced Moving Average',
            'sentiment_weight': self.sentiment_weight,
            'news_analysis_days': self.news_days_back,
            'cache_size': len(self.sentiment_cache)
        })
        return base_summary
    
    def backtest_with_sentiment(self, data: pd.DataFrame, symbol: str, 
                              initial_capital: float = 10000, 
                              use_historical_sentiment: bool = False) -> Dict:
        """
        Backtest strategy with sentiment analysis
        Note: Historical sentiment is not available, so this simulates the enhanced strategy
        """
        if not use_historical_sentiment:
            # Use base moving average backtest
            return super().backtest(data, initial_capital)
        
        # TODO: Implement historical sentiment backtesting
        # This would require historical news data and sentiment analysis
        logger.warning("Historical sentiment backtesting not yet implemented")
        return super().backtest(data, initial_capital)

class NewsEventDetector:
    """Detect significant news events that might affect stock prices"""
    
    def __init__(self):
        self.news_fetcher = NewsFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Keywords that indicate significant events
        self.significant_keywords = {
            'earnings': ['earnings', 'quarterly results', 'q1', 'q2', 'q3', 'q4'],
            'merger_acquisition': ['merger', 'acquisition', 'takeover', 'buyout'],
            'regulatory': ['sec filing', 'investigation', 'lawsuit', 'regulation'],
            'product': ['product launch', 'new product', 'recall', 'patent'],
            'management': ['ceo', 'resignation', 'appointed', 'leadership change'],
            'financial': ['dividend', 'buyback', 'debt', 'bankruptcy', 'ipo']
        }
    
    def detect_events(self, symbol: str, days_back: int = 3) -> List[Dict]:
        """Detect significant news events for a symbol"""
        try:
            articles = self.news_fetcher.fetch_news_for_symbol(symbol, days_back)
            events = []
            
            for article in articles:
                event_types = self._classify_event(article)
                
                if event_types:
                    # Analyze sentiment
                    sentiment = self.sentiment_analyzer.analyze_article(article)
                    
                    event = {
                        'title': article.title,
                        'event_types': event_types,
                        'sentiment_score': sentiment['score'],
                        'sentiment_label': sentiment['label'],
                        'published_date': article.published_date.isoformat() if article.published_date else None,
                        'source': article.source,
                        'relevance_score': article.relevance_score,
                        'url': article.url
                    }
                    events.append(event)
            
            # Sort by relevance and recency
            events.sort(key=lambda x: (x['relevance_score'], x['published_date']), reverse=True)
            
            return events[:10]  # Return top 10 events
            
        except Exception as e:
            logger.error(f"Error detecting events for {symbol}: {str(e)}")
            return []
    
    def _classify_event(self, article: NewsArticle) -> List[str]:
        """Classify the type of news event"""
        content = (article.title + " " + article.summary).lower()
        event_types = []
        
        for event_type, keywords in self.significant_keywords.items():
            for keyword in keywords:
                if keyword in content:
                    event_types.append(event_type)
                    break
        
        return list(set(event_types))  # Remove duplicates
