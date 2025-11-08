from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Tuple, Optional
import re
import logging
from datetime import datetime, timedelta
import statistics
from src.data.news import NewsArticle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Financial keywords that modify sentiment
        self.positive_financial_terms = {
            'beat expectations', 'exceeded', 'strong earnings', 'profit surge', 
            'revenue growth', 'bullish', 'upgrade', 'outperform', 'buy rating',
            'increased guidance', 'positive outlook', 'strong performance',
            'record high', 'breakthrough', 'expansion', 'acquisition',
            'dividend increase', 'share buyback', 'ipo success'
        }
        
        self.negative_financial_terms = {
            'missed expectations', 'below estimates', 'earnings miss', 'loss',
            'revenue decline', 'bearish', 'downgrade', 'underperform', 'sell rating',
            'lowered guidance', 'negative outlook', 'poor performance',
            'record low', 'bankruptcy', 'layoffs', 'investigation',
            'dividend cut', 'debt concerns', 'regulatory issues'
        }
    
    def analyze_article(self, article: NewsArticle) -> Dict[str, float]:
        """Analyze sentiment of a single article"""
        text = f"{article.title} {article.summary}"
        
        # Clean and prepare text
        cleaned_text = self._preprocess_text(text)
        
        # Get sentiment from multiple models
        textblob_sentiment = self._textblob_sentiment(cleaned_text)
        vader_sentiment = self._vader_sentiment(cleaned_text)
        
        # Apply financial keyword adjustments
        financial_adjustment = self._financial_keyword_adjustment(cleaned_text)
        
        # Combine sentiments with weights
        combined_score = (
            textblob_sentiment * 0.3 +
            vader_sentiment['compound'] * 0.5 +
            financial_adjustment * 0.2
        )
        
        # Normalize to [-1, 1] range
        combined_score = max(-1.0, min(1.0, combined_score))
        
        # Determine label
        if combined_score >= 0.1:
            label = 'positive'
        elif combined_score <= -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Store results in article
        article.sentiment_score = combined_score
        article.sentiment_label = label
        
        return {
            'score': combined_score,
            'label': label,
            'textblob_score': textblob_sentiment,
            'vader_score': vader_sentiment['compound'],
            'financial_adjustment': financial_adjustment,
            'confidence': abs(combined_score)
        }
    
    def analyze_multiple_articles(self, articles: List[NewsArticle]) -> Dict[str, any]:
        """Analyze sentiment for multiple articles and generate summary"""
        if not articles:
            return {
                'overall_sentiment': 0.0,
                'sentiment_label': 'neutral',
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'confidence': 0.0,
                'articles': []
            }
        
        sentiments = []
        article_results = []
        
        for article in articles:
            result = self.analyze_article(article)
            sentiments.append(result['score'])
            article_results.append({
                'title': article.title,
                'source': article.source,
                'sentiment_score': result['score'],
                'sentiment_label': result['label'],
                'published_date': article.published_date.isoformat() if article.published_date else None,
                'relevance_score': article.relevance_score
            })
        
        # Calculate weighted average (more recent articles have higher weight)
        weighted_sentiments = self._calculate_weighted_sentiment(articles, sentiments)
        
        overall_sentiment = weighted_sentiments['weighted_average']
        
        # Determine overall label
        if overall_sentiment >= 0.1:
            overall_label = 'positive'
        elif overall_sentiment <= -0.1:
            overall_label = 'negative'
        else:
            overall_label = 'neutral'
        
        # Count sentiments
        positive_count = sum(1 for s in sentiments if s >= 0.1)
        negative_count = sum(1 for s in sentiments if s <= -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        # Calculate confidence based on consensus and sample size
        confidence = self._calculate_confidence(sentiments, len(articles))
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_label': overall_label,
            'article_count': len(articles),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'confidence': confidence,
            'sentiment_distribution': {
                'mean': statistics.mean(sentiments),
                'median': statistics.median(sentiments),
                'std_dev': statistics.stdev(sentiments) if len(sentiments) > 1 else 0
            },
            'time_decay_factor': weighted_sentiments['time_decay_factor'],
            'articles': sorted(article_results, key=lambda x: x['published_date'], reverse=True)
        }
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of raw text (for social media posts)"""
        if not text or not text.strip():
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
        
        # Clean and prepare text
        cleaned_text = self._preprocess_text(text)
        
        # Get sentiment from multiple models
        textblob_sentiment = self._textblob_sentiment(cleaned_text)
        vader_sentiment = self._vader_sentiment(cleaned_text)
        
        # Apply financial keyword adjustments
        financial_adjustment = self._financial_keyword_adjustment(cleaned_text)
        
        # Combine sentiments with weights
        combined_score = (
            textblob_sentiment * 0.3 +
            vader_sentiment['compound'] * 0.5 +
            financial_adjustment * 0.2
        )
        
        # Normalize to [-1, 1] range
        combined_score = max(-1.0, min(1.0, combined_score))
        
        # Return VADER-style output with combined score
        return {
            'compound': combined_score,
            'pos': vader_sentiment['pos'],
            'neu': vader_sentiment['neu'],
            'neg': vader_sentiment['neg'],
            'textblob_score': textblob_sentiment,
            'financial_adjustment': financial_adjustment
        }
    
    def get_sentiment_trend(self, articles: List[NewsArticle], days_back: int = 7) -> Dict[str, any]:
        """Analyze sentiment trend over time"""
        if not articles:
            return {'trend': 'neutral', 'daily_sentiments': []}
        
        # Group articles by date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_articles = [a for a in articles if a.published_date >= cutoff_date]
        
        daily_sentiments = {}
        
        for article in recent_articles:
            date_key = article.published_date.date()
            
            if date_key not in daily_sentiments:
                daily_sentiments[date_key] = []
            
            # Analyze if not already done
            if article.sentiment_score is None:
                self.analyze_article(article)
            
            daily_sentiments[date_key].append(article.sentiment_score)
        
        # Calculate daily averages
        daily_data = []
        for date, sentiments in sorted(daily_sentiments.items()):
            daily_avg = statistics.mean(sentiments)
            daily_data.append({
                'date': date.isoformat(),
                'sentiment': daily_avg,
                'article_count': len(sentiments)
            })
        
        # Determine trend
        if len(daily_data) >= 2:
            recent_avg = statistics.mean([d['sentiment'] for d in daily_data[-2:]])
            older_avg = statistics.mean([d['sentiment'] for d in daily_data[:-2]]) if len(daily_data) > 2 else daily_data[0]['sentiment']
            
            if recent_avg > older_avg + 0.1:
                trend = 'improving'
            elif recent_avg < older_avg - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'daily_sentiments': daily_data,
            'total_articles': len(recent_articles)
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase for analysis
        return text.lower()
    
    def _textblob_sentiment(self, text: str) -> float:
        """Get sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def _vader_sentiment(self, text: str) -> Dict[str, float]:
        """Get sentiment using VADER"""
        try:
            return self.vader_analyzer.polarity_scores(text)
        except:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
    
    def _financial_keyword_adjustment(self, text: str) -> float:
        """Adjust sentiment based on financial keywords"""
        adjustment = 0.0
        
        # Check positive financial terms
        for term in self.positive_financial_terms:
            if term in text:
                adjustment += 0.2
        
        # Check negative financial terms
        for term in self.negative_financial_terms:
            if term in text:
                adjustment -= 0.2
        
        # Normalize adjustment
        return max(-1.0, min(1.0, adjustment))
    
    def _calculate_weighted_sentiment(self, articles: List[NewsArticle], sentiments: List[float]) -> Dict[str, float]:
        """Calculate weighted sentiment giving more weight to recent and relevant articles"""
        if not articles or not sentiments:
            return {'weighted_average': 0.0, 'time_decay_factor': 0.0}
        
        now = datetime.now()
        weighted_scores = []
        total_weight = 0
        
        for article, sentiment in zip(articles, sentiments):
            # Time decay factor (more recent = higher weight)
            hours_old = (now - article.published_date).total_seconds() / 3600
            time_weight = max(0.1, 1.0 / (1.0 + hours_old / 24))  # Decay over days
            
            # Relevance weight
            relevance_weight = article.relevance_score if article.relevance_score > 0 else 0.5
            
            # Combined weight
            combined_weight = time_weight * relevance_weight
            
            weighted_scores.append(sentiment * combined_weight)
            total_weight += combined_weight
        
        weighted_average = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
        avg_time_weight = total_weight / len(articles) if articles else 0.0
        
        return {
            'weighted_average': weighted_average,
            'time_decay_factor': avg_time_weight
        }
    
    def _calculate_confidence(self, sentiments: List[float], article_count: int) -> float:
        """Calculate confidence in sentiment analysis"""
        if not sentiments:
            return 0.0
        
        # Sample size factor (more articles = higher confidence)
        sample_factor = min(1.0, article_count / 10.0)
        
        # Consensus factor (less variance = higher confidence)
        if len(sentiments) > 1:
            std_dev = statistics.stdev(sentiments)
            consensus_factor = max(0.0, 1.0 - std_dev)
        else:
            consensus_factor = 0.5
        
        # Magnitude factor (stronger sentiments = higher confidence)
        avg_magnitude = statistics.mean([abs(s) for s in sentiments])
        magnitude_factor = min(1.0, avg_magnitude * 2)
        
        # Combined confidence
        confidence = (sample_factor + consensus_factor + magnitude_factor) / 3.0
        
        return round(confidence, 2)

class SentimentImpactCalculator:
    """Calculate the potential impact of sentiment on stock price"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def calculate_impact_score(self, sentiment_analysis: Dict[str, any], 
                              stock_volatility: float = 0.02) -> Dict[str, float]:
        """Calculate potential impact of sentiment on stock price"""
        
        overall_sentiment = sentiment_analysis['overall_sentiment']
        confidence = sentiment_analysis['confidence']
        article_count = sentiment_analysis['article_count']
        
        # Base impact from sentiment strength
        base_impact = abs(overall_sentiment) * confidence
        
        # Volume amplifier (more articles = higher potential impact)
        volume_amplifier = min(1.5, 1.0 + (article_count - 5) * 0.1)
        
        # Volatility factor (more volatile stocks are more affected by sentiment)
        volatility_factor = min(2.0, 1.0 + stock_volatility * 10)
        
        # Calculate final impact score
        impact_score = base_impact * volume_amplifier * volatility_factor
        impact_score = min(1.0, impact_score)  # Cap at 1.0
        
        # Determine impact direction
        impact_direction = 1 if overall_sentiment > 0 else -1 if overall_sentiment < 0 else 0
        
        # Estimate potential price movement (as percentage)
        estimated_price_impact = impact_score * impact_direction * stock_volatility * 5
        
        return {
            'impact_score': round(impact_score, 3),
            'impact_direction': impact_direction,
            'estimated_price_impact_pct': round(estimated_price_impact * 100, 2),
            'confidence_level': 'high' if confidence > 0.7 else 'medium' if confidence > 0.4 else 'low',
            'volume_amplifier': round(volume_amplifier, 2),
            'volatility_factor': round(volatility_factor, 2)
        }
