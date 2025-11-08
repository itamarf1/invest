# Investment System 📈

A modular investment system built for rapid iteration with minimal costs. Features both CLI and web interfaces for comprehensive market analysis.

## Features

- **Market Data**: Real-time and historical stock data via yfinance
- **News & Sentiment Analysis**: Multi-source news aggregation with AI sentiment analysis
- **Trading Strategies**: Moving average crossover enhanced with sentiment signals
- **Technical Indicators**: SMA, EMA, MACD, RSI, Volatility
- **Web Dashboard**: Interactive web interface with news sentiment and real-time charts
- **CLI Interface**: Easy-to-use command line tools with sentiment commands
- **Risk Analysis**: Portfolio analysis with news impact scoring
- **Event Detection**: Significant news event identification and categorization
- **API Endpoints**: RESTful API for integration

## Quick Start

1. **Setup Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Start Web Dashboard** (Recommended)
   ```bash
   # Start the web dashboard
   python run_dashboard.py
   
   # Open browser to http://localhost:8000
   ```

3. **Or Use CLI Commands**
   ```bash
   # Get stock quote
   python main.py quote --symbol AAPL

   # Get trading signals
   python main.py signal --symbol AAPL --period 6mo

   # Run backtest
   python main.py backtest --symbol AAPL --period 1y --capital 10000

   # Analyze portfolio
   python main.py portfolio --symbols AAPL MSFT GOOGL TSLA
   ```

## Web Dashboard 🌐

The web dashboard provides an intuitive interface for market analysis:

### Features
- **Stock Analysis**: Real-time price data with technical indicators
- **Interactive Charts**: Price movements with SMA overlays
- **Trading Signals**: Buy/sell/hold signals with confidence levels
- **Portfolio Analysis**: Multi-stock analysis with signal summary
- **Strategy Backtesting**: Visual performance comparison
- **Responsive Design**: Works on desktop and mobile

### Pages
- **Dashboard** (`/`): Individual stock analysis with charts
- **Portfolio** (`/portfolio`): Multi-stock portfolio analysis
- **Backtest** (`/backtest`): Strategy backtesting interface
- **News & Sentiment** (`/news`): News analysis and sentiment tracking

### API Endpoints
- `POST /api/stock-data`: Get stock data and technical indicators
- `POST /api/trading-signal`: Get buy/sell/hold signals
- `POST /api/sentiment-analysis`: Get sentiment-enhanced trading signals
- `POST /api/news-events`: Detect significant news events
- `GET /api/market-news`: Get general market news with sentiment
- `POST /api/portfolio-analysis`: Analyze multiple stocks
- `POST /api/backtest`: Run strategy backtests
- `GET /health`: Health check

## CLI Commands

### Stock Quote
```bash
python main.py quote --symbol AAPL --period 1mo
```

### Trading Signal
```bash
python main.py signal --symbol AAPL --period 6mo
```

### Backtest Strategy
```bash
python main.py backtest --symbol AAPL --period 1y --capital 10000
```

### Portfolio Analysis
```bash
python main.py portfolio --symbols AAPL MSFT GOOGL --period 1mo
```

### News Sentiment Analysis
```bash
python main.py sentiment --symbol AAPL --period 7d
```

### News Events Detection
```bash
python main.py events --symbol TSLA --period 7d
```

### Market Sentiment Overview
```bash
python main.py market-sentiment
```

## Time Periods
- `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

## Project Structure

```
invest/
├── src/
│   ├── data/
│   │   └── market.py          # Market data fetcher
│   ├── analysis/
│   │   └── __init__.py
│   ├── trading/
│   │   └── strategies/
│   │       └── moving_average.py  # MA crossover strategy
│   ├── api/
│   │   └── main.py           # FastAPI web application
│   └── utils/
├── templates/                 # HTML templates
│   ├── base.html
│   ├── dashboard.html
│   ├── portfolio.html
│   └── backtest.html
├── static/                    # CSS, JS, images
│   ├── css/
│   │   └── main.css
│   └── js/
│       └── main.js
├── tests/
├── configs/
├── main.py                    # CLI entry point
├── run_dashboard.py           # Web dashboard launcher
├── requirements.txt
└── README.md
```

## Trading Strategies

### 1. Moving Average Crossover (Base)
- **Short MA**: 20-day moving average
- **Long MA**: 50-day moving average  
- **Buy Signal**: When short MA crosses above long MA
- **Sell Signal**: When short MA crosses below long MA

### 2. Sentiment-Enhanced Strategy
- Combines technical analysis with news sentiment
- **News Sources**: RSS feeds from major financial outlets
- **Sentiment Analysis**: TextBlob + VADER with financial keyword weighting
- **Signal Enhancement**: Sentiment can boost/reduce technical signal confidence
- **Impact Scoring**: Estimates potential price movement from news sentiment

## News & Sentiment Features

### Data Sources
- **RSS Feeds**: Yahoo Finance, MarketWatch, Reuters, CNBC, Bloomberg (free)
- **NewsAPI**: Premium news aggregation (requires API key)
- **Alpha Vantage**: Financial news with sentiment scores (requires API key)

### Sentiment Analysis
- **Multi-Model Approach**: Combines TextBlob and VADER sentiment analyzers
- **Financial Context**: Custom keyword weighting for financial terms
- **Confidence Scoring**: Measures reliability of sentiment analysis
- **Time Decay**: Recent news weighted more heavily
- **Impact Estimation**: Predicts potential price movement from sentiment

### Event Detection
- **Earnings**: Quarterly results and guidance updates
- **M&A**: Merger and acquisition announcements
- **Regulatory**: SEC filings, investigations, regulations
- **Product**: Launches, recalls, patents
- **Management**: Leadership changes, appointments
- **Financial**: Dividends, buybacks, debt issues

## Next Steps

1. ✅ News sentiment analysis (Complete!)
2. Implement paper trading connections
3. Add more sophisticated strategies
4. Connect to live trading APIs
5. Add social media sentiment (Twitter, Reddit)

## Cost Structure

- **Development**: $0 (free tier APIs)
- **Running Costs**: $0-10/month
- **Data**: Free (yfinance)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - see LICENSE file for details