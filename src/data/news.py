import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
import re
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsArticle:
    def __init__(self, title: str, summary: str, url: str, published_date: datetime, 
                 source: str, relevance_score: float = 0.0):
        self.title = title
        self.summary = summary
        self.url = url
        self.published_date = published_date
        self.source = source
        self.relevance_score = relevance_score
        self.sentiment_score = None
        self.sentiment_label = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'summary': self.summary,
            'url': self.url,
            'published_date': self.published_date.isoformat() if self.published_date else None,
            'source': self.source,
            'relevance_score': self.relevance_score,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label
        }

class NewsFetcher:
    def __init__(self):
        self.newsapi_key = os.getenv('NEWS_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        
        # RSS feeds for financial news (free sources)
        self.rss_feeds = {
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
            'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
            'cnbc_finance': 'https://feeds.cnbc.com/cnbc/id/100003114/device/rss/rss.html',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss'
        }
    
    def fetch_news_for_symbol(self, symbol: str, days_back: int = 7) -> List[NewsArticle]:
        """Fetch news articles for a specific stock symbol"""
        articles = []
        
        # Try multiple sources
        if self.newsapi_key:
            articles.extend(self._fetch_from_newsapi(symbol, days_back))
        
        if self.alpha_vantage_key:
            articles.extend(self._fetch_from_alpha_vantage(symbol))
        
        # Always try RSS feeds (free)
        articles.extend(self._fetch_from_rss_feeds(symbol, days_back))
        
        # Remove duplicates and sort by relevance
        unique_articles = self._deduplicate_articles(articles)
        
        # Score relevance and sort
        for article in unique_articles:
            article.relevance_score = self._calculate_relevance_score(article, symbol)
        
        unique_articles.sort(key=lambda x: (x.relevance_score, x.published_date), reverse=True)
        
        return unique_articles[:20]  # Return top 20 most relevant articles
    
    def fetch_general_market_news(self, days_back: int = 3) -> List[NewsArticle]:
        """Fetch general market and economic news"""
        articles = []
        
        # RSS feeds for general financial news
        for source_name, feed_url in self.rss_feeds.items():
            try:
                feed_articles = self._parse_rss_feed(feed_url, source_name, days_back)
                articles.extend(feed_articles)
            except Exception as e:
                logger.warning(f"Failed to fetch from {source_name}: {str(e)}")
        
        # Remove duplicates and sort by date
        unique_articles = self._deduplicate_articles(articles)
        unique_articles.sort(key=lambda x: x.published_date, reverse=True)
        
        return unique_articles[:15]  # Return 15 most recent articles
    
    def _fetch_from_newsapi(self, symbol: str, days_back: int) -> List[NewsArticle]:
        """Fetch news from NewsAPI (requires API key)"""
        if not self.newsapi_key:
            return []
        
        articles = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        try:
            # Get company name for better search results
            company_name = self._get_company_name(symbol)
            query = f'"{symbol}" OR "{company_name}"'
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'sortBy': 'relevancy',
                'language': 'en',
                'apiKey': self.newsapi_key,
                'pageSize': 50
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            for item in data.get('articles', []):
                if item.get('title') and item.get('description'):
                    published_date = datetime.fromisoformat(
                        item['publishedAt'].replace('Z', '+00:00')
                    ) if item.get('publishedAt') else datetime.now()
                    
                    article = NewsArticle(
                        title=item['title'],
                        summary=item['description'],
                        url=item.get('url', ''),
                        published_date=published_date,
                        source=item.get('source', {}).get('name', 'NewsAPI'),
                        relevance_score=0.8  # High relevance for targeted search
                    )
                    articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from NewsAPI for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {str(e)}")
        
        return articles
    
    def _fetch_from_alpha_vantage(self, symbol: str) -> List[NewsArticle]:
        """Fetch news from Alpha Vantage News API"""
        if not self.alpha_vantage_key:
            return []
        
        articles = []
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.alpha_vantage_key,
                'limit': 20
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            for item in data.get('feed', []):
                published_date = datetime.strptime(
                    item['time_published'], '%Y%m%dT%H%M%S'
                ) if item.get('time_published') else datetime.now()
                
                # Extract relevance score for the specific symbol
                relevance_score = 0.5
                for ticker_data in item.get('ticker_sentiment', []):
                    if ticker_data.get('ticker') == symbol:
                        relevance_score = float(ticker_data.get('relevance_score', 0.5))
                        break
                
                article = NewsArticle(
                    title=item.get('title', ''),
                    summary=item.get('summary', ''),
                    url=item.get('url', ''),
                    published_date=published_date,
                    source=item.get('source', 'Alpha Vantage'),
                    relevance_score=relevance_score
                )
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from Alpha Vantage for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching from Alpha Vantage: {str(e)}")
        
        return articles
    
    def _fetch_from_rss_feeds(self, symbol: str, days_back: int) -> List[NewsArticle]:
        """Fetch news from RSS feeds and filter for symbol relevance"""
        articles = []
        
        for source_name, feed_url in self.rss_feeds.items():
            try:
                feed_articles = self._parse_rss_feed(feed_url, source_name, days_back)
                
                # Filter articles that mention the symbol
                relevant_articles = []
                for article in feed_articles:
                    if self._is_article_relevant(article, symbol):
                        relevant_articles.append(article)
                
                articles.extend(relevant_articles)
                
            except Exception as e:
                logger.warning(f"Failed to fetch from {source_name}: {str(e)}")
        
        return articles
    
    def _parse_rss_feed(self, feed_url: str, source_name: str, days_back: int) -> List[NewsArticle]:
        """Parse RSS feed and return articles"""
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries:
                # Parse published date
                published_date = datetime.now()
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_date = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    published_date = datetime(*entry.updated_parsed[:6])
                
                # Skip old articles
                if published_date < cutoff_date:
                    continue
                
                # Get summary/description
                summary = ''
                if hasattr(entry, 'summary'):
                    summary = self._clean_html(entry.summary)
                elif hasattr(entry, 'description'):
                    summary = self._clean_html(entry.description)
                
                article = NewsArticle(
                    title=getattr(entry, 'title', ''),
                    summary=summary,
                    url=getattr(entry, 'link', ''),
                    published_date=published_date,
                    source=source_name.replace('_', ' ').title()
                )
                articles.append(article)
            
        except Exception as e:
            logger.error(f"Error parsing RSS feed {feed_url}: {str(e)}")
        
        return articles
    
    def _is_article_relevant(self, article: NewsArticle, symbol: str) -> bool:
        """Check if an article is relevant to a stock symbol"""
        content = (article.title + " " + article.summary).lower()
        symbol_lower = symbol.lower()
        
        # Direct symbol mention
        if symbol_lower in content:
            return True
        
        # Company name mention (basic check)
        company_name = self._get_company_name(symbol).lower()
        if company_name and company_name in content:
            return True
        
        return False
    
    def _calculate_relevance_score(self, article: NewsArticle, symbol: str) -> float:
        """Calculate relevance score for an article"""
        content = (article.title + " " + article.summary).lower()
        symbol_lower = symbol.lower()
        score = 0.0
        
        # Symbol mentions
        title_mentions = content.count(symbol_lower)
        score += title_mentions * 0.3
        
        # Title vs summary weight
        if symbol_lower in article.title.lower():
            score += 0.5
        
        # Company name mentions
        company_name = self._get_company_name(symbol).lower()
        if company_name and company_name in content:
            score += 0.4
        
        # Financial keywords
        financial_keywords = ['earnings', 'revenue', 'profit', 'loss', 'stock', 'shares', 
                             'dividend', 'acquisition', 'merger', 'ipo', 'sec filing']
        for keyword in financial_keywords:
            if keyword in content:
                score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name for symbol (simplified mapping)"""
        company_map = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta',
            'NVDA': 'NVIDIA',
            'JPM': 'JPMorgan',
            'V': 'Visa',
            'JNJ': 'Johnson & Johnson'
        }
        return company_map.get(symbol.upper(), '')
    
    def _clean_html(self, html_content: str) -> str:
        """Remove HTML tags and clean text"""
        if not html_content:
            return ''
        
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity"""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Normalize title for comparison
            normalized_title = re.sub(r'[^\w\s]', '', article.title.lower())
            normalized_title = re.sub(r'\s+', ' ', normalized_title).strip()
            
            if normalized_title not in seen_titles and len(normalized_title) > 10:
                seen_titles.add(normalized_title)
                unique_articles.append(article)
        
        return unique_articles
