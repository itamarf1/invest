import tweepy
import praw
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import re
from collections import Counter
import os
from dotenv import load_dotenv

from src.analysis.sentiment import SentimentAnalyzer

load_dotenv()
logger = logging.getLogger(__name__)

class TwitterSentimentFetcher:
    """Twitter sentiment analysis for stock symbols"""
    
    def __init__(self):
        # Twitter API v2 credentials
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.client = None
        
        if self.bearer_token:
            try:
                self.client = tweepy.Client(bearer_token=self.bearer_token)
                logger.info("Twitter API client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Twitter client: {str(e)}")
        else:
            logger.info("Twitter credentials not found. Will use enhanced news sentiment analysis.")
    
    def search_tweets(self, symbol: str, count: int = 100, days_back: int = 1) -> List[Dict]:
        """Search for tweets mentioning a stock symbol"""
        if not self.client:
            return self._get_mock_tweets(symbol, count)
        
        try:
            # Calculate time window
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            # Search queries for the stock
            queries = [
                f"${symbol} lang:en -is:retweet",
                f"{symbol} stock lang:en -is:retweet",
                f"{symbol} trading lang:en -is:retweet"
            ]
            
            all_tweets = []
            
            for query in queries:
                try:
                    tweets = tweepy.Paginator(
                        self.client.search_recent_tweets,
                        query=query,
                        tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations'],
                        max_results=min(100, count // len(queries)),
                        start_time=start_time,
                        end_time=end_time
                    ).flatten(limit=count // len(queries))
                    
                    for tweet in tweets:
                        all_tweets.append({
                            'id': tweet.id,
                            'text': tweet.text,
                            'created_at': tweet.created_at,
                            'author_id': tweet.author_id,
                            'retweet_count': tweet.public_metrics['retweet_count'],
                            'like_count': tweet.public_metrics['like_count'],
                            'reply_count': tweet.public_metrics['reply_count'],
                            'quote_count': tweet.public_metrics['quote_count'],
                            'platform': 'twitter',
                            'symbol': symbol
                        })
                        
                except Exception as e:
                    logger.warning(f"Error fetching tweets for query '{query}': {str(e)}")
                    continue
            
            return all_tweets[:count]
            
        except Exception as e:
            logger.error(f"Error searching Twitter for {symbol}: {str(e)}")
            return self._get_mock_tweets(symbol, count)
    
    def _get_mock_tweets(self, symbol: str, count: int) -> List[Dict]:
        """Generate mock tweets for testing"""
        mock_tweets = [
            f"${symbol} looking strong today! Great earnings report 📈",
            f"Thinking of buying ${symbol} on this dip. Good entry point?",
            f"${symbol} technical analysis shows bullish pattern forming",
            f"Not sure about ${symbol} at these levels. Seems overvalued",
            f"${symbol} breaking resistance! Target $300 🚀",
            f"Sold my ${symbol} position. Taking profits while I can",
            f"${symbol} quarterly results better than expected. Bullish!",
            f"${symbol} facing headwinds in this market. Staying cautious"
        ]
        
        tweets = []
        for i in range(min(count, len(mock_tweets))):
            tweets.append({
                'id': f"mock_{i}",
                'text': mock_tweets[i],
                'created_at': datetime.now() - timedelta(hours=i),
                'author_id': f'mock_user_{i}',
                'retweet_count': np.random.randint(0, 50),
                'like_count': np.random.randint(0, 200),
                'reply_count': np.random.randint(0, 20),
                'quote_count': np.random.randint(0, 10),
                'platform': 'twitter',
                'symbol': symbol
            })
        
        return tweets

class RedditSentimentFetcher:
    """Reddit sentiment analysis for stock symbols"""
    
    def __init__(self):
        # Reddit API credentials
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = os.getenv('REDDIT_USER_AGENT', 'InvestmentSystem/1.0')
        
        self.reddit = None
        
        if self.client_id and self.client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                logger.info("Reddit API client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Reddit client: {str(e)}")
        else:
            logger.warning("Reddit credentials not found. Using mock data.")
    
    def search_reddit_posts(self, symbol: str, count: int = 50, subreddits: List[str] = None) -> List[Dict]:
        """Search for Reddit posts mentioning a stock symbol"""
        if not self.reddit:
            return self._get_mock_reddit_posts(symbol, count)
        
        if subreddits is None:
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis', 'ValueInvesting']
        
        all_posts = []
        
        try:
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for posts
                    search_queries = [f"{symbol}", f"${symbol}"]
                    
                    for query in search_queries:
                        posts = subreddit.search(
                            query,
                            limit=count // (len(subreddits) * len(search_queries)),
                            time_filter='day'  # Last 24 hours
                        )
                        
                        for post in posts:
                            all_posts.append({
                                'id': post.id,
                                'title': post.title,
                                'text': post.selftext,
                                'score': post.score,
                                'upvote_ratio': post.upvote_ratio,
                                'num_comments': post.num_comments,
                                'created_utc': datetime.fromtimestamp(post.created_utc),
                                'subreddit': subreddit_name,
                                'author': str(post.author) if post.author else 'deleted',
                                'platform': 'reddit',
                                'symbol': symbol
                            })
                            
                except Exception as e:
                    logger.warning(f"Error fetching from r/{subreddit_name}: {str(e)}")
                    continue
            
            return all_posts[:count]
            
        except Exception as e:
            logger.error(f"Error searching Reddit for {symbol}: {str(e)}")
            return self._get_mock_reddit_posts(symbol, count)
    
    def _get_mock_reddit_posts(self, symbol: str, count: int) -> List[Dict]:
        """Generate mock Reddit posts for testing"""
        mock_posts = [
            {
                'title': f"${symbol} DD: Why this is the next big play",
                'text': f"Deep dive analysis on {symbol}. Fundamentals look solid and technicals are aligning for a breakout."
            },
            {
                'title': f"Thoughts on ${symbol} earnings?",
                'text': f"What's everyone's take on {symbol} earnings coming up? Expecting a beat or miss?"
            },
            {
                'title': f"${symbol} to the moon! 🚀🚀🚀",
                'text': f"{symbol} breaking all resistance levels. This is going parabolic!"
            },
            {
                'title': f"Should I sell my ${symbol} bags?",
                'text': f"Been holding {symbol} for months. Down 20%. Cut losses or hold?"
            },
            {
                'title': f"${symbol} technical analysis - bullish divergence",
                'text': f"RSI showing bullish divergence on {symbol} daily chart. Could see a bounce soon."
            }
        ]
        
        posts = []
        for i in range(min(count, len(mock_posts))):
            post_data = mock_posts[i]
            posts.append({
                'id': f"mock_reddit_{i}",
                'title': post_data['title'],
                'text': post_data['text'],
                'score': np.random.randint(1, 500),
                'upvote_ratio': np.random.uniform(0.6, 0.95),
                'num_comments': np.random.randint(0, 100),
                'created_utc': datetime.now() - timedelta(hours=i),
                'subreddit': 'wallstreetbets',
                'author': f'mock_user_{i}',
                'platform': 'reddit',
                'symbol': symbol
            })
        
        return posts

class SocialSentimentAnalyzer:
    """Analyze sentiment from social media data"""
    
    def __init__(self):
        self.twitter_fetcher = TwitterSentimentFetcher()
        self.reddit_fetcher = RedditSentimentFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Social sentiment keywords and their weights
        self.bullish_keywords = {
            'moon': 2.0, 'rocket': 1.8, 'bullish': 1.5, 'buy': 1.2, 'pump': 1.8,
            'breakout': 1.4, 'rally': 1.3, 'squeeze': 1.6, 'diamond hands': 1.7,
            'hodl': 1.3, 'to the moon': 2.0, 'stonks': 1.1, 'calls': 1.2
        }
        
        self.bearish_keywords = {
            'crash': 2.0, 'dump': 1.8, 'bearish': 1.5, 'sell': 1.2, 'puts': 1.4,
            'short': 1.3, 'drop': 1.1, 'fall': 1.1, 'bear market': 1.6,
            'bubble': 1.5, 'overvalued': 1.4, 'correction': 1.2
        }
    
    def get_social_sentiment(self, symbol: str, tweet_count: int = 100, 
                           reddit_count: int = 50, days_back: int = 1) -> Dict:
        """Get comprehensive social media sentiment for a symbol"""
        
        # Fetch social media data
        tweets = self.twitter_fetcher.search_tweets(symbol, tweet_count, days_back)
        reddit_posts = self.reddit_fetcher.search_reddit_posts(symbol, reddit_count)
        
        # Combine all social media content
        all_social_data = tweets + reddit_posts
        
        if not all_social_data:
            return {
                'symbol': symbol,
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'total_posts': 0,
                'platform_breakdown': {},
                'trending_keywords': [],
                'volume_score': 0.0,
                'engagement_score': 0.0
            }
        
        # Analyze sentiment for each post
        sentiment_scores = []
        weighted_scores = []
        platform_data = {'twitter': [], 'reddit': []}
        all_text = []
        
        for post in all_social_data:
            # Get text content
            if post['platform'] == 'twitter':
                text = post['text']
                engagement = post['like_count'] + post['retweet_count']
            else:  # reddit
                text = f"{post['title']} {post['text']}"
                engagement = post['score'] + post['num_comments']
            
            all_text.append(text)
            
            # Basic sentiment analysis
            base_sentiment = self.sentiment_analyzer.analyze_text(text)
            base_score = base_sentiment['compound']
            
            # Apply social media keyword weighting
            social_weight = self._calculate_social_weight(text)
            weighted_score = base_score + social_weight
            weighted_score = max(-1.0, min(1.0, weighted_score))  # Clamp to [-1, 1]
            
            # Weight by engagement
            engagement_weight = min(2.0, 1.0 + np.log10(max(1, engagement)) / 3)
            final_score = weighted_score * engagement_weight
            
            sentiment_scores.append(base_score)
            weighted_scores.append(final_score)
            platform_data[post['platform']].append({
                'sentiment': final_score,
                'engagement': engagement,
                'text': text[:100] + '...' if len(text) > 100 else text
            })
        
        # Calculate overall metrics
        avg_sentiment = np.mean(weighted_scores) if weighted_scores else 0.0
        sentiment_std = np.std(weighted_scores) if len(weighted_scores) > 1 else 0.0
        
        # Determine sentiment label
        if avg_sentiment > 0.1:
            sentiment_label = 'bullish'
        elif avg_sentiment < -0.1:
            sentiment_label = 'bearish'
        else:
            sentiment_label = 'neutral'
        
        # Calculate confidence (inverse of standard deviation)
        confidence = max(0.0, min(1.0, 1.0 - sentiment_std))
        
        # Calculate volume and engagement scores
        volume_score = min(1.0, len(all_social_data) / 100.0)  # Normalize to [0, 1]
        
        total_engagement = sum([
            post.get('like_count', 0) + post.get('retweet_count', 0) + 
            post.get('score', 0) + post.get('num_comments', 0)
            for post in all_social_data
        ])
        engagement_score = min(1.0, total_engagement / 10000.0)  # Normalize to [0, 1]
        
        # Extract trending keywords
        trending_keywords = self._extract_trending_keywords(' '.join(all_text))
        
        # Platform breakdown
        platform_breakdown = {}
        for platform, posts in platform_data.items():
            if posts:
                platform_breakdown[platform] = {
                    'count': len(posts),
                    'avg_sentiment': np.mean([p['sentiment'] for p in posts]),
                    'total_engagement': sum([p['engagement'] for p in posts])
                }
        
        return {
            'symbol': symbol,
            'sentiment_score': avg_sentiment,
            'sentiment_label': sentiment_label,
            'confidence': confidence,
            'total_posts': len(all_social_data),
            'platform_breakdown': platform_breakdown,
            'trending_keywords': trending_keywords,
            'volume_score': volume_score,
            'engagement_score': engagement_score,
            'raw_scores': weighted_scores,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_social_weight(self, text: str) -> float:
        """Calculate additional sentiment weight based on social media keywords"""
        text_lower = text.lower()
        weight = 0.0
        
        # Check bullish keywords
        for keyword, multiplier in self.bullish_keywords.items():
            if keyword in text_lower:
                weight += 0.1 * multiplier
        
        # Check bearish keywords
        for keyword, multiplier in self.bearish_keywords.items():
            if keyword in text_lower:
                weight -= 0.1 * multiplier
        
        return max(-0.5, min(0.5, weight))  # Clamp to reasonable range
    
    def _extract_trending_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """Extract trending keywords from social media text"""
        # Clean text
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'[^a-zA-Z\s$]', '', text)
        
        # Extract words (including $ symbols for tickers)
        words = re.findall(r'\$[A-Z]{2,5}|\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'use', 'man', 'new', 'now', 'way', 'may', 'say'}
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count and return top keywords
        word_counts = Counter(words)
        return word_counts.most_common(top_n)
    
    def get_social_sentiment_history(self, symbol: str, days: int = 7) -> List[Dict]:
        """Get historical social sentiment for trending analysis"""
        history = []
        
        for day in range(days):
            # For now, return mock historical data
            # In production, you'd fetch and store historical data
            sentiment_score = np.random.uniform(-0.5, 0.5) + np.sin(day * 0.5) * 0.3
            
            history.append({
                'date': (datetime.now() - timedelta(days=day)).date().isoformat(),
                'sentiment_score': sentiment_score,
                'sentiment_label': 'bullish' if sentiment_score > 0.1 else 'bearish' if sentiment_score < -0.1 else 'neutral',
                'post_count': np.random.randint(20, 200),
                'engagement_score': np.random.uniform(0.1, 0.9)
            })
        
        return list(reversed(history))
