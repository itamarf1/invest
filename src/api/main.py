from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.market import MarketDataFetcher
from src.trading.strategies.moving_average import MovingAverageStrategy
from src.trading.strategies.sentiment_enhanced import SentimentEnhancedStrategy, NewsEventDetector
from src.trading.strategies.rsi_strategy import RSIStrategy
from src.trading.strategies.macd_strategy import MACDStrategy
from src.trading.strategies.bollinger_strategy import BollingerBandsStrategy
from src.trading.strategies.ensemble_strategy import EnsembleStrategy
from src.data.news import NewsFetcher
from src.analysis.sentiment import SentimentAnalyzer
from src.data.social_sentiment import SocialSentimentAnalyzer
from src.trading.paper_trader import PaperTrader, TradingBot, OrderSide, OrderType
from src.trading.portfolio_manager import PortfolioManager
from src.data.realtime_stream import RealTimeDataStream, AlertManager, Alert, PriceUpdate
from src.trading.portfolio_optimizer import PortfolioOptimizer, PortfolioRebalancer
from src.crypto.crypto_data import CryptoDataFetcher
from src.crypto.crypto_strategies import CryptoMomentumStrategy, CryptoMeanReversionStrategy, CryptoCorrelationStrategy, CryptoDCAStrategy, CryptoVolatilityStrategy
from src.ml.price_prediction import MLPricePredictor
from src.ml.deep_learning import LSTMPredictor, SimpleNeuralNetwork, TENSORFLOW_AVAILABLE
from src.analysis.multi_timeframe import MultiTimeframeAnalyzer
from src.risk.risk_management import RiskCalculator, PositionSizer, PortfolioRiskManager, RiskMonitor
from src.options.options_data import OptionsDataFetcher, OptionType
from src.options.options_strategies import OptionsStrategist, StrategyType
from src.options.advanced_options_strategies import AdvancedOptionsStrategist, AdvancedStrategyType
from src.backtesting.options_backtesting import OptionsBacktester, BacktestResults
from src.bonds.bonds_data import BondsDataFetcher
from src.commodities.commodities_data import CommoditiesDataFetcher
from src.forex.forex_data import ForexDataFetcher
from src.bots.trading_bots import BotManager, BotConfiguration, BotType, BotStatus
from src.data.global_markets import GlobalMarketsDataFetcher, MarketRegion
from src.auth import init_auth_manager, auth_router, user_portfolio_router
from src.analysis.algorithm_dashboard_routes import router as algorithm_dashboard_router
# from src.api.broker_routes import router as broker_router


def safe_float(value):
    """Convert value to float, replacing NaN/inf with None"""
    if pd.isna(value) or np.isinf(value):
        return None
    return float(value)

app = FastAPI(title="Investment Dashboard", description="Real-time investment analysis dashboard")

# Initialize authentication system
auth_manager = init_auth_manager(google_client_id=os.getenv("GOOGLE_CLIENT_ID", "your-google-client-id.apps.googleusercontent.com"))

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Include authentication routes
app.include_router(auth_router)
app.include_router(user_portfolio_router)
app.include_router(algorithm_dashboard_router)
# app.include_router(broker_router)

fetcher = MarketDataFetcher()
strategy = MovingAverageStrategy()
sentiment_strategy = SentimentEnhancedStrategy()
news_fetcher = NewsFetcher()
sentiment_analyzer = SentimentAnalyzer()
event_detector = NewsEventDetector()
paper_trader = PaperTrader()
portfolio_manager = PortfolioManager(paper_trader)
trading_bot = TradingBot(paper_trader)

# Advanced trading strategies
rsi_strategy = RSIStrategy()
macd_strategy = MACDStrategy()
bollinger_strategy = BollingerBandsStrategy()
ensemble_strategy = EnsembleStrategy()

# Social sentiment analyzer
social_sentiment = SocialSentimentAnalyzer()

# Real-time streaming and alerts
realtime_stream = RealTimeDataStream()
alert_manager = AlertManager()

# Portfolio optimization
portfolio_optimizer = PortfolioOptimizer(fetcher)
portfolio_rebalancer = PortfolioRebalancer(portfolio_manager, portfolio_optimizer)

# Cryptocurrency components
crypto_fetcher = CryptoDataFetcher()
crypto_momentum_strategy = CryptoMomentumStrategy()
crypto_mean_reversion_strategy = CryptoMeanReversionStrategy()
crypto_correlation_strategy = CryptoCorrelationStrategy()
crypto_volatility_strategy = CryptoVolatilityStrategy()

# Machine Learning components
ml_predictor = MLPricePredictor()
if TENSORFLOW_AVAILABLE:
    lstm_predictor = LSTMPredictor()
else:
    simple_nn = SimpleNeuralNetwork()

# Multi-timeframe analysis
multi_timeframe_analyzer = MultiTimeframeAnalyzer()

# Risk management components
risk_calculator = RiskCalculator()
portfolio_risk_manager = PortfolioRiskManager()

# Options trading components
options_fetcher = OptionsDataFetcher()
options_strategist = OptionsStrategist()
advanced_options_strategist = AdvancedOptionsStrategist()

# Additional asset class components
bonds_fetcher = BondsDataFetcher()
commodities_fetcher = CommoditiesDataFetcher()
forex_fetcher = ForexDataFetcher()

# Global markets data fetcher
global_markets_fetcher = GlobalMarketsDataFetcher()

# Trading bots manager
bot_manager = BotManager()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.symbol_subscriptions: Dict[WebSocket, set] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.symbol_subscriptions[websocket] = set()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.symbol_subscriptions:
            del self.symbol_subscriptions[websocket]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)

    async def broadcast_to_symbol_subscribers(self, message: str, symbol: str):
        disconnected = []
        for connection in self.active_connections:
            if symbol in self.symbol_subscriptions.get(connection, set()):
                try:
                    await connection.send_text(message)
                except:
                    disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    def subscribe_to_symbol(self, websocket: WebSocket, symbol: str):
        if websocket in self.symbol_subscriptions:
            self.symbol_subscriptions[websocket].add(symbol.upper())
            realtime_stream.subscribe_symbol(symbol)

    def unsubscribe_from_symbol(self, websocket: WebSocket, symbol: str):
        if websocket in self.symbol_subscriptions:
            self.symbol_subscriptions[websocket].discard(symbol.upper())

connection_manager = ConnectionManager()

# Price update queue for WebSocket broadcasting (thread-safe)
import queue
price_update_queue = queue.Queue()
alert_queue = queue.Queue()

# Set up real-time data callbacks
def on_price_update(price_update: PriceUpdate):
    """Callback for price updates - uses thread-safe queue"""
    message = {
        "type": "price_update",
        "data": {
            "symbol": price_update.symbol,
            "price": price_update.price,
            "change": price_update.change,
            "change_percent": price_update.change_percent,
            "volume": price_update.volume,
            "timestamp": price_update.timestamp.isoformat(),
            "bid": price_update.bid,
            "ask": price_update.ask,
            "bid_size": price_update.bid_size,
            "ask_size": price_update.ask_size
        }
    }
    
    # Put message in queue for processing
    try:
        price_update_queue.put_nowait((price_update.symbol, json.dumps(message)))
    except queue.Full:
        pass  # Drop message if queue is full

def on_alert_triggered(alert: Alert):
    """Callback for triggered alerts - uses thread-safe queue"""
    message = {
        "type": "alert",
        "data": {
            "id": alert.id,
            "symbol": alert.symbol,
            "alert_type": alert.alert_type.value,
            "message": alert.message,
            "current_value": alert.current_value,
            "target_value": alert.target_value,
            "timestamp": alert.timestamp.isoformat()
        }
    }
    
    # Put message in queue for processing
    try:
        alert_queue.put_nowait((alert.symbol, json.dumps(message)))
    except queue.Full:
        pass  # Drop message if queue is full

# Background task to process queued messages
async def process_websocket_messages():
    """Process queued messages and broadcast to WebSocket clients"""
    while True:
        try:
            # Process price updates
            while not price_update_queue.empty():
                try:
                    symbol, message = price_update_queue.get_nowait()
                    await connection_manager.broadcast_to_symbol_subscribers(message, symbol)
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error broadcasting price update: {str(e)}")
            
            # Process alerts
            while not alert_queue.empty():
                try:
                    symbol, message = alert_queue.get_nowait()
                    await connection_manager.broadcast_to_symbol_subscribers(message, symbol)
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error broadcasting alert: {str(e)}")
                    
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            
        except Exception as e:
            logger.error(f"Error in WebSocket message processor: {str(e)}")
            await asyncio.sleep(1)

realtime_stream.add_price_callback(on_price_update)
realtime_stream.alert_manager.add_alert_callback(on_alert_triggered)

# Start the background task
import atexit

async def start_background_tasks():
    """Start background tasks"""
    asyncio.create_task(process_websocket_messages())

@app.on_event("startup")
async def startup_event():
    """Start background tasks when app starts"""
    await start_background_tasks()


class StockRequest(BaseModel):
    symbol: str
    period: str = "6mo"

class StockRequestCustom(BaseModel):
    symbol: str
    period: str = "6mo"
    interval: str = "1d"


class PortfolioRequest(BaseModel):
    symbols: List[str]
    period: str = "6mo"


class BacktestRequest(BaseModel):
    symbol: str
    period: str = "1y"
    capital: float = 10000
    short_window: int = 20
    long_window: int = 50


class OrderRequest(BaseModel):
    symbol: str
    quantity: int
    side: str  # 'buy' or 'sell'
    order_type: str = "market"  # 'market' or 'limit'
    limit_price: Optional[float] = None


class AutoTradeRequest(BaseModel):
    symbol: str
    period: str = "6mo"


class AdvancedSignalRequest(BaseModel):
    symbol: str
    period: str = "6mo"
    strategy: str = "ensemble"  # 'rsi', 'macd', 'bollinger', 'ensemble'


class MultiStrategyBacktestRequest(BaseModel):
    symbol: str
    period: str = "1y"
    capital: float = 10000
    strategies: List[str] = ["moving_average", "rsi", "macd", "bollinger", "ensemble"]


class SocialSentimentRequest(BaseModel):
    symbol: str
    tweet_count: int = 100
    reddit_count: int = 50
    days_back: int = 1


class PortfolioOptimizationRequest(BaseModel):
    symbols: List[str]
    optimization_type: str = "max_sharpe"  # 'max_sharpe', 'min_variance', 'target_return', 'risk_parity'
    target_return: Optional[float] = None
    period: str = "1y"


class RebalancingRequest(BaseModel):
    target_weights: Dict[str, float]
    rebalance_threshold: float = 0.05
    max_trading_cost: float = 0.005


class RebalancingScheduleRequest(BaseModel):
    target_weights: Dict[str, float] 
    frequency: str = "monthly"  # 'monthly', 'quarterly', 'annually'
    rebalance_threshold: float = 0.05


class CryptoAnalysisRequest(BaseModel):
    symbol: str
    strategy: str = "momentum"  # 'momentum', 'mean_reversion', 'correlation', 'volatility'


class CryptoPortfolioRequest(BaseModel):
    symbols: List[str]
    

class CryptoDCARequest(BaseModel):
    target_allocation: Dict[str, float]  # {'BTC': 0.4, 'ETH': 0.3, 'ADA': 0.3}
    portfolio_value: float
    current_holdings: Dict[str, float] = {}  # {'BTC': 0.5, 'ETH': 10}
    frequency: str = "weekly"


class MLPredictionRequest(BaseModel):
    symbol: str
    horizon: int = 5  # Days ahead to predict
    model_type: str = "ensemble"  # 'ensemble', 'random_forest', 'gradient_boost', 'linear'


class MLTrainingRequest(BaseModel):
    symbol: str
    horizon: int = 5
    period: str = "2y"  # Training data period


class LSTMPredictionRequest(BaseModel):
    symbol: str
    sequence_length: int = 60
    prediction_horizon: int = 5


class BatchPredictionRequest(BaseModel):
    symbols: List[str]
    horizon: int = 5
    model_type: str = "ensemble"


class MultiTimeframeRequest(BaseModel):
    symbol: str
    timeframes: Optional[List[str]] = ["1d", "1wk", "1mo"]


class TimeframeComparisonRequest(BaseModel):
    symbols: List[str]
    timeframes: Optional[List[str]] = ["1d", "1wk", "1mo"]


class RiskAnalysisRequest(BaseModel):
    symbol: str
    period: str = "1y"
    benchmark: str = "SPY"


class PositionSizingRequest(BaseModel):
    symbol: str
    entry_price: float
    stop_loss_price: float
    target_price: Optional[float] = None
    portfolio_value: float = 100000
    max_risk_per_trade: float = 0.02


class PortfolioRiskRequest(BaseModel):
    positions: Dict[str, float]  # symbol -> weight
    portfolio_value: float = 100000


class RiskMonitoringRequest(BaseModel):
    positions: Dict[str, float]  # symbol -> weight  
    portfolio_value: float = 100000


class AlertRequest(BaseModel):
    symbol: str
    alert_type: str  # 'price', 'volume', 'breakout'
    target_value: float
    message: Optional[str] = None
    direction: Optional[str] = "above"  # 'above', 'below' for price alerts
    volume_multiplier: Optional[float] = 2.0  # for volume alerts
    breakout_type: Optional[str] = "resistance"  # 'resistance', 'support' for breakout alerts


class OptionsChainRequest(BaseModel):
    symbol: str
    expiration_date: Optional[str] = None


class CoveredCallRequest(BaseModel):
    symbol: str
    shares_owned: int
    target_strike_pct: float = 0.05


class ProtectivePutRequest(BaseModel):
    symbol: str
    shares_owned: int
    protection_level: float = 0.05


class BullCallSpreadRequest(BaseModel):
    symbol: str
    risk_amount: float = 1000
    spread_width: float = 5.0


class IronCondorRequest(BaseModel):
    symbol: str
    risk_amount: float = 2000
    wing_width: float = 5.0


class LongStraddleRequest(BaseModel):
    symbol: str
    risk_amount: float = 1000


class StrategyAnalysisRequest(BaseModel):
    symbol: str
    strategy_name: str
    strategy_data: Dict[str, Any]


class ButterflyRequest(BaseModel):
    symbol: str
    center_strike: Optional[float] = None
    wing_width: float = 5.0
    option_type: str = "call"  # "call" or "put"


class CalendarSpreadRequest(BaseModel):
    symbol: str
    strike: Optional[float] = None
    option_type: str = "call"
    short_dte: int = 30
    long_dte: int = 60


class JadeLizardRequest(BaseModel):
    symbol: str
    risk_amount: float = 2000


class RatioSpreadRequest(BaseModel):
    symbol: str
    ratio: int = 2
    risk_amount: float = 1500


class SyntheticStockRequest(BaseModel):
    symbol: str
    strike: Optional[float] = None


class AdvancedStrategyAnalysisRequest(BaseModel):
    symbol: str
    strategy_data: Dict[str, Any]
    current_price: float


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(request: Request):
    return templates.TemplateResponse("portfolio.html", {"request": request})


@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    return templates.TemplateResponse("backtest.html", {"request": request})


@app.get("/news", response_class=HTMLResponse)
async def news_page(request: Request):
    return templates.TemplateResponse("news.html", {"request": request})


@app.get("/trading", response_class=HTMLResponse)
async def trading_page(request: Request):
    return templates.TemplateResponse("trading.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/algorithms", response_class=HTMLResponse)
async def algorithm_dashboard_page(request: Request):
    return templates.TemplateResponse("algorithm_dashboard.html", {"request": request})


@app.post("/api/stock-data")
async def get_stock_data(request: StockRequest):
    """Get stock data with data source indicators - TEMPORARY FALLBACK TO ORIGINAL FETCHER"""
    try:
        # Use original fetcher for now while we debug hybrid pipeline
        if request.period == "1mo":
            data = fetcher.get_stock_data(request.symbol, period="1mo", interval="1h")
        elif request.period == "3mo":
            data = fetcher.get_stock_data(request.symbol, period="3mo", interval="1h") 
        elif request.period == "6mo":
            data = fetcher.get_stock_data(request.symbol, period="6mo", interval="1d")
        elif request.period in ["1y", "2y"]:
            data = fetcher.get_stock_data(request.symbol, period=request.period, interval="1d")
        else:
            data = fetcher.get_stock_data(request.symbol, period="6mo", interval="1d")
            
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")

        data_with_indicators = fetcher.get_technical_indicators(data)
        data_with_returns = fetcher.calculate_returns(data_with_indicators)
        
        latest = data_with_returns.iloc[-1]
        current_price = fetcher.get_current_price(request.symbol)
        
        # Enhanced current price handling - prioritize real-time data
        final_current_price = current_price if current_price is not None else latest['Close']
        
        # Better daily change calculation
        daily_change = 0.0
        if len(data_with_returns) >= 2:
            try:
                prev_close = data_with_returns.iloc[-2]['Close']
                if current_price is not None and prev_close != 0:
                    daily_change = ((current_price - prev_close) / prev_close * 100)
                elif prev_close != 0:
                    daily_change = ((latest['Close'] - prev_close) / prev_close * 100)
            except (IndexError, ZeroDivisionError):
                daily_change = 0.0
        
        # Enhanced RSI handling
        rsi_value = latest.get('RSI')
        if pd.isna(rsi_value) or rsi_value is None:
            rsi_value = 50.0
        
        # Determine if this is a warrant (needed for timezone fix)
        is_warrant = request.symbol.endswith('W')
        
        chart_data = []
        # Show appropriate amount of data based on period
        tail_count = 200 if request.period in ["1mo", "3mo"] else 100
        
        for i, row in data_with_returns.tail(tail_count).iterrows():
            close_price = current_price if (i == data_with_returns.index[-1] and current_price is not None) else row['Close']
            
            # Fix timezone issue for warrants - Yahoo Finance returns wrong year for some warrants
            display_index = i
            if is_warrant and hasattr(i, 'year') and i.year > 2024:
                # Likely a timezone bug, subtract 1 year
                display_index = i.replace(year=i.year - 1)
            
            # Format timestamp based on data frequency
            if hasattr(display_index, 'hour'):
                date_str = display_index.strftime("%Y-%m-%d %H:%M")
            else:
                date_str = display_index.strftime("%Y-%m-%d")
            
            chart_data.append({
                "date": date_str,
                "close": safe_float(close_price),
                "sma_20": safe_float(row.get('SMA_20')),
                "sma_50": safe_float(row.get('SMA_50')),
                "volume": int(row['Volume']) if pd.notna(row['Volume']) else 0
            })
        
        # Get stock info
        stock_info = fetcher.get_stock_info(request.symbol)
        company_name = stock_info.get('longName', stock_info.get('shortName', request.symbol))
        
        # Mock data source indicators for now - will be replaced with hybrid pipeline
        is_warrant = request.symbol.endswith('W')
        mock_source = "stooq" if is_warrant else "yahoo"
        mock_quality = 85.0 if is_warrant else 95.0
        
        return {
            "symbol": request.symbol,
            "company_name": company_name,
            "current_price": safe_float(final_current_price),
            "latest_close": safe_float(latest['Close']),
            "daily_change": safe_float(daily_change),
            "sma_20": safe_float(latest.get('SMA_20')),
            "sma_50": safe_float(latest.get('SMA_50')),
            "rsi": safe_float(rsi_value),
            "volume": int(latest['Volume']) if pd.notna(latest['Volume']) else 0,
            "chart_data": chart_data,
            "data_quality": {
                "is_real_time": current_price is not None,
                "has_recent_volume": int(latest['Volume']) > 0 if pd.notna(latest['Volume']) else False,
                "data_points": len(data_with_returns),
                "granularity": "hourly" if request.period in ["1mo", "3mo", "6mo"] else "daily",
                "refresh_rate": "5 minutes",
                "security_type": "warrant" if is_warrant else "stock",
                "last_updated": data_with_returns.index[-1].isoformat()
            },
            "data_source": mock_source,
            "quality_score": mock_quality,
            "data_source_priority": [mock_source, "yahoo", "alpha_vantage"] if is_warrant else ["yahoo", "stooq", "alpha_vantage"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trading-signal")
async def get_trading_signal(request: StockRequest):
    try:
        # Use same enhanced data logic as stock-data endpoint
        if request.period == "1mo":
            data = fetcher.get_stock_data(request.symbol, period="1mo", interval="1h")
        elif request.period == "3mo":
            data = fetcher.get_stock_data(request.symbol, period="3mo", interval="1h") 
        elif request.period == "6mo":
            data = fetcher.get_stock_data(request.symbol, period="6mo", interval="1h")
        elif request.period in ["1y", "2y"]:
            data = fetcher.get_stock_data(request.symbol, period=request.period, interval="1d")
        else:
            data = fetcher.get_stock_data(request.symbol, period="6mo", interval="1h")
            
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        signal = strategy.get_latest_signal(data, request.symbol)
        
        # Get current price for more accurate signal pricing
        current_price = fetcher.get_current_price(request.symbol)
        signal_price = current_price if current_price is not None else signal.get('price', data['Close'].iloc[-1])
        
        return {
            "symbol": request.symbol,
            "action": signal['action'],
            "price": safe_float(signal_price),  # Use current price
            "confidence": safe_float(signal['confidence']),
            "timestamp": signal['timestamp'].isoformat() if signal['timestamp'] else None,
            "sma_short": safe_float(signal.get('sma_short')),
            "sma_long": safe_float(signal.get('sma_long'))
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/portfolio-analysis")
async def analyze_portfolio(request: PortfolioRequest):
    try:
        results = []
        total_signals = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for symbol in request.symbols:
            try:
                data = fetcher.get_stock_data(symbol, period=request.period)
                if not data.empty:
                    signal = strategy.get_latest_signal(data, symbol)
                    current_price = fetcher.get_current_price(symbol)
                    
                    results.append({
                        "symbol": symbol,
                        "action": signal['action'],
                        "price": safe_float(current_price or signal['price']),
                        "confidence": safe_float(signal['confidence']),
                        "status": "success"
                    })
                    
                    total_signals[signal['action']] += 1
                else:
                    results.append({
                        "symbol": symbol,
                        "action": "HOLD",
                        "price": None,
                        "confidence": 0,
                        "status": "no_data"
                    })
            except Exception as e:
                results.append({
                    "symbol": symbol,
                    "action": "HOLD",
                    "price": None,
                    "confidence": 0,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "portfolio": results,
            "summary": total_signals,
            "total_stocks": len(request.symbols)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest")
async def run_backtest(request: BacktestRequest):
    try:
        data = fetcher.get_stock_data(request.symbol, period=request.period)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        custom_strategy = MovingAverageStrategy(
            short_window=request.short_window,
            long_window=request.long_window
        )
        
        results = custom_strategy.backtest(data, initial_capital=request.capital)
        
        if 'error' in results:
            raise HTTPException(status_code=400, detail=results['error'])
        
        portfolio_chart = []
        for item in results['portfolio_values'][-100:]:  # Last 100 data points
            portfolio_chart.append({
                "date": item['date'].strftime("%Y-%m-%d"),
                "portfolio_value": round(item['portfolio_value'], 2),
                "stock_price": round(item['stock_price'], 2)
            })
        
        recent_trades = []
        for trade in results['trades'][-10:]:  # Last 10 trades
            recent_trades.append({
                "date": trade['date'].strftime("%Y-%m-%d"),
                "action": trade['action'],
                "shares": trade['shares'],
                "price": round(trade['price'], 2),
                "value": round(trade.get('cost', trade.get('revenue', 0)), 2)
            })
        
        return {
            "symbol": request.symbol,
            "period": request.period,
            "initial_capital": results['initial_capital'],
            "final_value": results['final_value'],
            "total_return_pct": results['total_return_pct'],
            "buy_hold_return_pct": results['buy_hold_return_pct'],
            "outperformance": results['outperformance'],
            "num_trades": results['num_trades'],
            "portfolio_chart": portfolio_chart,
            "recent_trades": recent_trades
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sentiment-analysis")
async def get_sentiment_analysis(request: StockRequest):
    try:
        data = fetcher.get_stock_data(request.symbol, period=request.period)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        signal = sentiment_strategy.get_latest_signal(data, request.symbol)
        
        return {
            "symbol": request.symbol,
            "sentiment_score": safe_float(signal.get('sentiment_score')),
            "sentiment_label": signal.get('sentiment_label', 'neutral'),
            "sentiment_confidence": safe_float(signal.get('sentiment_confidence')),
            "impact_score": safe_float(signal.get('impact_score')),
            "estimated_price_impact": safe_float(signal.get('estimated_price_impact')),
            "article_count": signal.get('article_count', 0),
            "enhanced_action": signal.get('action', 'HOLD'),
            "enhanced_confidence": safe_float(signal.get('confidence')),
            "base_action": signal.get('base_action', 'HOLD'),
            "base_confidence": safe_float(signal.get('base_confidence')),
            "sentiment_influence": safe_float(signal.get('sentiment_influence')),
            "news_articles": signal.get('news_articles', [])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/news-events")
async def get_news_events(request: StockRequest):
    try:
        days_back = min(7, max(1, int(request.period.replace('mo', '30').replace('d', '1').replace('y', '365')[:2])))
        events = event_detector.detect_events(request.symbol, days_back=days_back)
        
        return {
            "symbol": request.symbol,
            "events": events,
            "event_count": len(events),
            "days_analyzed": days_back
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market-news")
async def get_market_news():
    try:
        articles = news_fetcher.fetch_general_market_news(days_back=3)
        
        # Analyze sentiment for market news
        if articles:
            sentiment_result = sentiment_analyzer.analyze_multiple_articles(articles)
            
            return {
                "articles": [article.to_dict() for article in articles],
                "overall_sentiment": safe_float(sentiment_result['overall_sentiment']),
                "sentiment_label": sentiment_result['sentiment_label'],
                "article_count": len(articles),
                "sentiment_distribution": {
                    "positive": sentiment_result['positive_count'],
                    "negative": sentiment_result['negative_count'],
                    "neutral": sentiment_result['neutral_count']
                }
            }
        else:
            return {
                "articles": [],
                "overall_sentiment": 0.0,
                "sentiment_label": "neutral",
                "article_count": 0,
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0}
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio-status")
async def get_portfolio_status():
    """Get current portfolio status"""
    try:
        connected = paper_trader.is_connected()
        account_info = paper_trader.get_account_info()
        positions = portfolio_manager.get_current_positions()
        summary = portfolio_manager.get_portfolio_summary()
        
        return {
            "connected": connected,
            "connection_type": "Alpaca Paper Trading" if connected else "Simulated Mode",
            "account_value": safe_float(account_info.get('portfolio_value', 0)),
            "cash_balance": safe_float(account_info.get('cash', 0)),
            "buying_power": safe_float(account_info.get('buying_power', 0)),
            "day_change": safe_float(summary.get('day_change', 0)),
            "day_change_pct": safe_float(summary.get('day_change_pct', 0)),
            "total_positions": len(positions),
            "positions": [
                {
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "avg_entry_price": safe_float(pos.avg_entry_price),
                    "current_price": safe_float(pos.current_price),
                    "market_value": safe_float(pos.market_value),
                    "unrealized_pnl": safe_float(pos.unrealized_pnl),
                    "unrealized_pnl_pct": safe_float(pos.unrealized_pnl_pct),
                    "side": pos.side
                } for pos in positions
            ],
            "allocations": summary.get('allocations', [])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/place-order")
async def place_order(request: OrderRequest):
    """Place a trading order"""
    try:
        # Validate inputs
        if request.side.lower() not in ['buy', 'sell']:
            raise HTTPException(status_code=400, detail="Invalid side. Must be 'buy' or 'sell'")
        
        if request.quantity <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be positive")
        
        # Convert to enums
        order_side = OrderSide.BUY if request.side.lower() == 'buy' else OrderSide.SELL
        order_type = OrderType.MARKET if request.order_type.lower() == 'market' else OrderType.LIMIT
        
        # Submit order
        order = paper_trader.submit_order(
            symbol=request.symbol,
            qty=request.quantity,
            side=order_side,
            order_type=order_type,
            limit_price=request.limit_price
        )
        
        if order:
            # Add to trade history if filled
            if order['status'] == 'filled':
                portfolio_manager.add_trade(order)
            
            return {
                "success": True,
                "order_id": order['id'],
                "status": order['status'],
                "symbol": order['symbol'],
                "quantity": order['qty'],
                "side": order['side'],
                "order_type": order['order_type'],
                "avg_fill_price": safe_float(order.get('avg_fill_price')),
                "message": f"{request.side.upper()} order for {request.quantity} {request.symbol} submitted successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Order failed to submit")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orders")
async def get_orders(status: str = "all", limit: int = 20):
    """Get order history"""
    try:
        status_filter = None if status == "all" else status
        orders = paper_trader.get_orders(status=status_filter, limit=limit)
        
        return {
            "orders": [
                {
                    "id": order['id'],
                    "symbol": order['symbol'],
                    "side": order['side'],
                    "quantity": order['qty'],
                    "filled_qty": order['filled_qty'],
                    "order_type": order['order_type'],
                    "status": order['status'],
                    "submitted_at": order['submitted_at'],
                    "filled_at": order['filled_at'],
                    "avg_fill_price": safe_float(order.get('avg_fill_price')),
                    "limit_price": safe_float(order.get('limit_price'))
                } for order in orders
            ],
            "total_orders": len(orders)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auto-trade")
async def execute_auto_trade(request: AutoTradeRequest):
    """Execute automated trading based on signals"""
    try:
        # Get enhanced signal
        data = fetcher.get_stock_data(request.symbol, period=request.period)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        signal = sentiment_strategy.get_latest_signal(data, request.symbol)
        
        # Execute signal if auto-trading enabled
        result = {
            "symbol": request.symbol,
            "signal": signal['action'],
            "confidence": safe_float(signal.get('confidence', 0)),
            "sentiment_score": safe_float(signal.get('sentiment_score', 0)),
            "sentiment_label": signal.get('sentiment_label', 'neutral'),
            "auto_trading_enabled": trading_bot.auto_trading_enabled
        }
        
        if trading_bot.auto_trading_enabled:
            order = trading_bot.execute_signal(signal)
            
            if order:
                portfolio_manager.add_trade(order)
                result.update({
                    "order_executed": True,
                    "order_id": order['id'],
                    "order_side": order['side'],
                    "quantity": order['qty'],
                    "message": f"{order['side'].upper()} order executed for {order['qty']} {order['symbol']}"
                })
            else:
                result.update({
                    "order_executed": False,
                    "message": "No order placed (low confidence or other constraints)"
                })
        else:
            result.update({
                "order_executed": False,
                "message": "Auto-trading disabled. Set ENABLE_AUTO_TRADING=true to enable."
            })
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/performance-metrics")
async def get_performance_metrics():
    """Get portfolio performance metrics"""
    try:
        metrics = portfolio_manager.calculate_performance_metrics()
        
        return {
            "total_return": safe_float(metrics.total_return),
            "total_return_pct": safe_float(metrics.total_return_pct),
            "annualized_return": safe_float(metrics.annualized_return),
            "volatility": safe_float(metrics.volatility),
            "sharpe_ratio": safe_float(metrics.sharpe_ratio),
            "max_drawdown": safe_float(metrics.max_drawdown),
            "win_rate": safe_float(metrics.win_rate),
            "profit_factor": safe_float(metrics.profit_factor),
            "total_trades": metrics.total_trades,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "avg_win": safe_float(metrics.avg_win),
            "avg_loss": safe_float(metrics.avg_loss),
            "largest_win": safe_float(metrics.largest_win),
            "largest_loss": safe_float(metrics.largest_loss)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/position-analysis/{symbol}")
async def get_position_analysis(symbol: str):
    """Get detailed analysis for a specific position"""
    try:
        analysis = portfolio_manager.get_position_analysis(symbol)
        
        if 'error' in analysis:
            raise HTTPException(status_code=404, detail=analysis['error'])
        
        return analysis
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sector-allocation")
async def get_sector_allocation():
    """Get portfolio sector allocation"""
    try:
        sectors = portfolio_manager.get_sector_allocation()
        
        return {
            "sectors": [
                {
                    "name": sector,
                    "allocation_pct": data['allocation_pct'],
                    "market_value": data['market_value'],
                    "positions": data['positions']
                }
                for sector, data in sectors.items()
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/advanced-signal")
async def get_advanced_signal(request: AdvancedSignalRequest):
    """Get advanced trading signal from specified strategy"""
    try:
        data = fetcher.get_stock_data(request.symbol, period=request.period)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Select strategy
        strategy_map = {
            'rsi': rsi_strategy,
            'macd': macd_strategy,
            'bollinger': bollinger_strategy,
            'ensemble': ensemble_strategy
        }
        
        selected_strategy = strategy_map.get(request.strategy.lower())
        if not selected_strategy:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")
        
        signal = selected_strategy.get_latest_signal(data, request.symbol)
        
        # Clean up numpy types for JSON serialization
        def clean_signal(obj):
            if isinstance(obj, dict):
                return {k: clean_signal(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_signal(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif obj is None:
                return None
            else:
                try:
                    if pd.isna(obj) or np.isinf(obj):
                        return None
                except (TypeError, ValueError):
                    pass
                return obj
        
        cleaned_signal = clean_signal(signal)
        return cleaned_signal
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/multi-strategy-backtest")
async def multi_strategy_backtest(request: MultiStrategyBacktestRequest):
    """Compare multiple trading strategies"""
    try:
        data = fetcher.get_stock_data(request.symbol, period=request.period)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        strategy_map = {
            'moving_average': strategy,
            'sentiment_enhanced': sentiment_strategy,
            'rsi': rsi_strategy,
            'macd': macd_strategy,
            'bollinger': bollinger_strategy,
            'ensemble': ensemble_strategy
        }
        
        results = {}
        
        for strategy_name in request.strategies:
            if strategy_name not in strategy_map:
                continue
                
            try:
                selected_strategy = strategy_map[strategy_name]
                result = selected_strategy.backtest(data, initial_capital=request.capital)
                
                if 'error' not in result:
                    # Clean and summarize results
                    results[strategy_name] = {
                        'total_return_pct': safe_float(result['total_return_pct']),
                        'buy_hold_return_pct': safe_float(result['buy_hold_return_pct']),
                        'outperformance': safe_float(result['outperformance']),
                        'num_trades': result['num_trades'],
                        'strategy': result.get('strategy', strategy_name),
                        'final_value': safe_float(result['final_value']),
                        'parameters': result.get('parameters', {})
                    }
                    
                    # Add strategy-specific metrics
                    if 'ensemble_metrics' in result:
                        results[strategy_name]['ensemble_metrics'] = result['ensemble_metrics']
                        
                else:
                    results[strategy_name] = {'error': result['error']}
                    
            except Exception as e:
                results[strategy_name] = {'error': str(e)}
        
        # Rank strategies by performance
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            ranked_strategies = sorted(
                valid_results.items(),
                key=lambda x: x[1]['total_return_pct'] or 0,
                reverse=True
            )
            best_strategy = ranked_strategies[0]
        else:
            best_strategy = None
        
        return {
            'symbol': request.symbol,
            'period': request.period,
            'initial_capital': request.capital,
            'results': results,
            'best_strategy': {
                'name': best_strategy[0] if best_strategy else None,
                'performance': best_strategy[1] if best_strategy else None
            } if best_strategy else None,
            'comparison_metrics': {
                'strategies_tested': len(request.strategies),
                'successful_tests': len(valid_results),
                'failed_tests': len(results) - len(valid_results)
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategy-comparison/{symbol}")
async def strategy_comparison(symbol: str, period: str = "6mo"):
    """Get real-time signals from all strategies for comparison"""
    try:
        data = fetcher.get_stock_data(symbol, period=period)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        strategy_map = {
            'moving_average': strategy,
            'sentiment_enhanced': sentiment_strategy,
            'rsi': rsi_strategy,
            'macd': macd_strategy,
            'bollinger': bollinger_strategy,
            'ensemble': ensemble_strategy
        }
        
        signals = {}
        
        for strategy_name, selected_strategy in strategy_map.items():
            try:
                signal = selected_strategy.get_latest_signal(data, symbol)
                
                # Clean and summarize signal
                signals[strategy_name] = {
                    'action': signal.get('action', 'HOLD'),
                    'confidence': safe_float(signal.get('confidence', 0.0)),
                    'strategy': signal.get('strategy', strategy_name),
                    'signal_reason': signal.get('signal_reason', ''),
                    'price': safe_float(signal.get('price', 0))
                }
                
                # Add strategy-specific indicators
                if 'rsi' in signal:
                    signals[strategy_name]['rsi'] = safe_float(signal['rsi'])
                if 'macd' in signal:
                    signals[strategy_name]['macd'] = safe_float(signal['macd'])
                if 'bb_width' in signal:
                    signals[strategy_name]['bb_width'] = safe_float(signal['bb_width'])
                if 'sentiment_score' in signal:
                    signals[strategy_name]['sentiment_score'] = safe_float(signal['sentiment_score'])
                if 'agreement_metrics' in signal:
                    signals[strategy_name]['agreement_ratio'] = safe_float(
                        signal['agreement_metrics'].get('agreement_ratio', 0)
                    )
                
            except Exception as e:
                signals[strategy_name] = {
                    'action': 'ERROR',
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        # Calculate consensus
        valid_signals = {k: v for k, v in signals.items() if v['action'] != 'ERROR'}
        action_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for signal in valid_signals.values():
            action = signal['action']
            if action in action_counts:
                action_counts[action] += 1
        
        total_strategies = len(valid_signals)
        consensus_action = max(action_counts.items(), key=lambda x: x[1])[0] if total_strategies > 0 else 'HOLD'
        consensus_strength = action_counts[consensus_action] / total_strategies if total_strategies > 0 else 0
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'signals': signals,
            'consensus': {
                'action': consensus_action,
                'strength': consensus_strength,
                'vote_counts': action_counts,
                'total_strategies': total_strategies
            },
            'market_data': {
                'current_price': safe_float(data['Close'].iloc[-1]),
                'daily_change_pct': safe_float(
                    ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
                ) if len(data) > 1 else 0
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/social-sentiment")
async def get_social_sentiment(request: SocialSentimentRequest):
    """Get social media sentiment analysis for a stock"""
    try:
        result = social_sentiment.get_social_sentiment(
            symbol=request.symbol,
            tweet_count=request.tweet_count,
            reddit_count=request.reddit_count,
            days_back=request.days_back
        )
        
        # Clean result for JSON serialization
        cleaned_result = {}
        for key, value in result.items():
            if key == 'raw_scores':
                # Convert numpy arrays to lists
                cleaned_result[key] = [float(x) for x in value] if value else []
            elif isinstance(value, (int, float, str, bool, type(None))):
                cleaned_result[key] = value
            elif isinstance(value, dict):
                cleaned_result[key] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in value.items()}
            elif isinstance(value, list):
                cleaned_result[key] = value
            else:
                cleaned_result[key] = str(value)
        
        return cleaned_result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/social-sentiment-history/{symbol}")
async def get_social_sentiment_history(symbol: str, days: int = 7):
    """Get historical social sentiment for trending analysis"""
    try:
        history = social_sentiment.get_social_sentiment_history(symbol, days)
        return {
            'symbol': symbol,
            'history': history,
            'days_analyzed': days
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/enhanced-signal/{symbol}")
async def get_enhanced_signal_with_social(symbol: str, period: str = "6mo"):
    """Get trading signal enhanced with social sentiment"""
    try:
        # Get market data
        data = fetcher.get_stock_data(symbol, period=period)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Get signals from multiple sources
        technical_signal = ensemble_strategy.get_latest_signal(data, symbol)
        social_result = social_sentiment.get_social_sentiment(symbol, tweet_count=50, reddit_count=25)
        
        # Combine technical and social signals
        technical_score = technical_signal.get('confidence', 0.5) * (
            1 if technical_signal.get('action') == 'BUY' else
            -1 if technical_signal.get('action') == 'SELL' else 0
        )
        
        social_score = social_result.get('sentiment_score', 0.0)
        social_confidence = social_result.get('confidence', 0.0)
        
        # Weight combination (70% technical, 30% social)
        combined_score = (technical_score * 0.7) + (social_score * 0.3)
        
        # Determine enhanced action
        if combined_score > 0.2:
            enhanced_action = 'BUY'
            enhanced_confidence = min(0.95, abs(combined_score) + (social_confidence * 0.1))
        elif combined_score < -0.2:
            enhanced_action = 'SELL'  
            enhanced_confidence = min(0.95, abs(combined_score) + (social_confidence * 0.1))
        else:
            enhanced_action = 'HOLD'
            enhanced_confidence = 0.5 + (social_confidence * 0.2)
        
        return {
            'symbol': symbol,
            'enhanced_action': enhanced_action,
            'enhanced_confidence': enhanced_confidence,
            'combined_score': combined_score,
            'technical_signal': {
                'action': technical_signal.get('action'),
                'confidence': safe_float(technical_signal.get('confidence')),
                'agreement_ratio': safe_float(
                    technical_signal.get('agreement_metrics', {}).get('agreement_ratio', 0)
                ) if 'agreement_metrics' in technical_signal else 0
            },
            'social_sentiment': {
                'sentiment_score': safe_float(social_result.get('sentiment_score')),
                'sentiment_label': social_result.get('sentiment_label'),
                'confidence': safe_float(social_result.get('confidence')),
                'total_posts': social_result.get('total_posts', 0),
                'volume_score': safe_float(social_result.get('volume_score')),
                'engagement_score': safe_float(social_result.get('engagement_score')),
                'trending_keywords': social_result.get('trending_keywords', [])[:5]
            },
            'market_data': {
                'current_price': safe_float(data['Close'].iloc[-1]),
                'daily_change_pct': safe_float(
                    ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
                ) if len(data) > 1 else 0
            },
            'timestamp': datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/realtime/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time data streaming"""
    await connection_manager.connect(websocket)
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["action"] == "subscribe":
                symbol = message["symbol"].upper()
                connection_manager.subscribe_to_symbol(websocket, symbol)
                
                # Send confirmation
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "subscription_confirmed",
                        "symbol": symbol,
                        "message": f"Subscribed to {symbol} real-time data"
                    }),
                    websocket
                )
                
            elif message["action"] == "unsubscribe":
                symbol = message["symbol"].upper()
                connection_manager.unsubscribe_from_symbol(websocket, symbol)
                
                # Send confirmation
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "unsubscription_confirmed",
                        "symbol": symbol,
                        "message": f"Unsubscribed from {symbol} real-time data"
                    }),
                    websocket
                )
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


@app.post("/api/alerts")
async def create_alert(alert_request: AlertRequest):
    """Create a new alert"""
    try:
        symbol = alert_request.symbol.upper()
        alert_type = alert_request.alert_type.lower()
        
        if alert_type == "price":
            alert_id = realtime_stream.add_price_alert(
                symbol=symbol,
                target_price=alert_request.target_value,
                direction=alert_request.direction,
                message=alert_request.message
            )
            
        elif alert_type == "volume":
            alert_id = realtime_stream.add_volume_alert(
                symbol=symbol,
                volume_multiplier=alert_request.volume_multiplier,
                message=alert_request.message
            )
            
        elif alert_type == "breakout":
            alert_id = realtime_stream.add_breakout_alert(
                symbol=symbol,
                level=alert_request.target_value,
                breakout_type=alert_request.breakout_type,
                message=alert_request.message
            )
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown alert type: {alert_type}")
        
        return {
            "success": True,
            "alert_id": alert_id,
            "symbol": symbol,
            "alert_type": alert_type,
            "target_value": alert_request.target_value,
            "message": f"Alert created for {symbol}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/alerts")
async def get_alerts(symbol: Optional[str] = None, status: str = "active"):
    """Get alerts (active or triggered)"""
    try:
        if status == "active":
            alerts = realtime_stream.alert_manager.get_active_alerts(symbol)
        elif status == "triggered":
            alerts = realtime_stream.alert_manager.get_triggered_alerts(symbol, hours_back=24)
        else:
            raise HTTPException(status_code=400, detail="Status must be 'active' or 'triggered'")
        
        return {
            "alerts": [
                {
                    "id": alert.id,
                    "symbol": alert.symbol,
                    "alert_type": alert.alert_type.value,
                    "message": alert.message,
                    "current_value": alert.current_value,
                    "target_value": alert.target_value,
                    "timestamp": alert.timestamp.isoformat(),
                    "triggered": alert.triggered,
                    "metadata": alert.metadata
                }
                for alert in alerts
            ],
            "count": len(alerts),
            "status": status,
            "symbol": symbol
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/alerts/{alert_id}")
async def delete_alert(alert_id: str):
    """Delete an alert"""
    try:
        success = realtime_stream.alert_manager.remove_alert(alert_id)
        
        if success:
            return {
                "success": True,
                "message": f"Alert {alert_id} deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/realtime/status")
async def get_realtime_status():
    """Get real-time streaming status"""
    try:
        current_prices = realtime_stream.get_current_prices()
        
        return {
            "status": "active" if realtime_stream.is_running else "inactive",
            "subscribed_symbols": list(realtime_stream.subscribed_symbols),
            "active_connections": len(connection_manager.active_connections),
            "current_prices": current_prices,
            "use_simulation": realtime_stream.use_simulation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/realtime/subscribe")
async def subscribe_to_symbol(symbol_request: StockRequest):
    """Subscribe to real-time data for a symbol"""
    try:
        symbol = symbol_request.symbol.upper()
        realtime_stream.subscribe_symbol(symbol)
        
        return {
            "success": True,
            "symbol": symbol,
            "message": f"Subscribed to real-time data for {symbol}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/realtime/subscribe/{symbol}")
async def unsubscribe_from_symbol(symbol: str):
    """Unsubscribe from real-time data for a symbol"""
    try:
        symbol = symbol.upper()
        realtime_stream.unsubscribe_symbol(symbol)
        
        return {
            "success": True,
            "symbol": symbol,
            "message": f"Unsubscribed from real-time data for {symbol}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/portfolio-optimization")
async def optimize_portfolio(request: PortfolioOptimizationRequest):
    """Optimize portfolio using Modern Portfolio Theory"""
    try:
        result = portfolio_optimizer.optimize_portfolio(
            symbols=request.symbols,
            optimization_type=request.optimization_type,
            target_return=request.target_return
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Portfolio optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/portfolio-metrics")
async def calculate_portfolio_metrics(symbols: List[str], weights: List[float], period: str = "1y"):
    """Calculate portfolio risk and return metrics for given weights"""
    try:
        if len(symbols) != len(weights):
            raise HTTPException(status_code=400, detail="Number of symbols must match number of weights")
        
        if abs(sum(weights) - 1.0) > 0.001:
            raise HTTPException(status_code=400, detail="Weights must sum to 1.0")
        
        result = portfolio_optimizer.calculate_portfolio_metrics(symbols, weights, period)
        
        return result
        
    except Exception as e:
        logger.error(f"Portfolio metrics calculation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rebalancing-analysis")
async def analyze_rebalancing(request: RebalancingRequest):
    """Analyze portfolio rebalancing needs"""
    try:
        result = portfolio_rebalancer.analyze_rebalancing_need(
            target_weights=request.target_weights,
            rebalance_threshold=request.rebalance_threshold
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Rebalancing analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/execute-rebalancing")
async def execute_rebalancing(request: RebalancingRequest):
    """Execute portfolio rebalancing"""
    try:
        result = portfolio_rebalancer.execute_rebalancing(
            target_weights=request.target_weights,
            rebalance_threshold=request.rebalance_threshold,
            max_trading_cost=request.max_trading_cost
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Rebalancing execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/schedule-rebalancing")
async def schedule_rebalancing(request: RebalancingScheduleRequest):
    """Schedule automatic portfolio rebalancing"""
    try:
        result = portfolio_rebalancer.schedule_rebalancing(
            target_weights=request.target_weights,
            frequency=request.frequency,
            rebalance_threshold=request.rebalance_threshold
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Rebalancing scheduling error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/crypto-price/{symbol}")
async def get_crypto_price(symbol: str):
    """Get current cryptocurrency price"""
    try:
        price_data = crypto_fetcher.get_crypto_price(symbol)
        if not price_data:
            raise HTTPException(status_code=404, detail=f"Crypto symbol {symbol} not found")
        
        return {
            "symbol": price_data.symbol,
            "price": price_data.price,
            "change_24h": price_data.change_24h,
            "change_24h_pct": price_data.change_24h_pct,
            "volume_24h": price_data.volume_24h,
            "market_cap": price_data.market_cap,
            "timestamp": price_data.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Crypto price fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/crypto-portfolio")
async def get_crypto_portfolio(request: CryptoPortfolioRequest):
    """Get portfolio data for multiple cryptocurrencies"""
    try:
        crypto_data = crypto_fetcher.get_multiple_crypto_prices(request.symbols)
        
        portfolio = []
        total_market_cap = 0
        
        for symbol, data in crypto_data.items():
            portfolio.append({
                "symbol": data.symbol,
                "price": data.price,
                "change_24h": data.change_24h,
                "change_24h_pct": data.change_24h_pct,
                "volume_24h": data.volume_24h,
                "market_cap": data.market_cap
            })
            total_market_cap += data.market_cap
        
        return {
            "portfolio": portfolio,
            "total_market_cap": total_market_cap,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Crypto portfolio error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/crypto-market-overview")
async def get_crypto_market_overview():
    """Get overall cryptocurrency market data"""
    try:
        market_data = crypto_fetcher.get_crypto_market_overview()
        return market_data
        
    except Exception as e:
        logger.error(f"Crypto market overview error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/crypto-top/{limit}")
async def get_top_cryptocurrencies(limit: int = 50):
    """Get top cryptocurrencies by market cap"""
    try:
        top_cryptos = crypto_fetcher.get_top_cryptocurrencies(limit)
        return {"cryptocurrencies": top_cryptos, "limit": limit}
        
    except Exception as e:
        logger.error(f"Top cryptocurrencies error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/crypto-analysis")
async def analyze_crypto(request: CryptoAnalysisRequest):
    """Analyze cryptocurrency using specified strategy"""
    try:
        symbol = request.symbol
        strategy = request.strategy.lower()
        
        if strategy == "momentum":
            signal = crypto_momentum_strategy.analyze_momentum(symbol)
        elif strategy == "mean_reversion":
            signal = crypto_mean_reversion_strategy.analyze_mean_reversion(symbol)
        elif strategy == "correlation":
            signal = crypto_correlation_strategy.analyze_correlation_signal(symbol)
        elif strategy == "volatility":
            signal = crypto_volatility_strategy.analyze_volatility_signal(symbol)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")
        
        return {
            "symbol": signal.symbol,
            "strategy": strategy,
            "signal": signal.signal.value,
            "confidence": signal.confidence,
            "price": signal.price,
            "target_price": signal.target_price,
            "stop_loss": signal.stop_loss,
            "reasoning": signal.reasoning,
            "timestamp": signal.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Crypto analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/crypto-dca")
async def crypto_dca_analysis(request: CryptoDCARequest):
    """Get Dollar Cost Averaging signals for crypto portfolio"""
    try:
        dca_strategy = CryptoDCAStrategy(request.target_allocation, request.frequency)
        signals = dca_strategy.get_dca_signals(request.portfolio_value, request.current_holdings)
        
        result_signals = []
        for signal in signals:
            result_signals.append({
                "symbol": signal.symbol,
                "signal": signal.signal.value,
                "confidence": signal.confidence,
                "price": signal.price,
                "reasoning": signal.reasoning,
                "timestamp": signal.timestamp.isoformat()
            })
        
        return {
            "target_allocation": request.target_allocation,
            "portfolio_value": request.portfolio_value,
            "frequency": request.frequency,
            "signals": result_signals,
            "total_signals": len(result_signals)
        }
        
    except Exception as e:
        logger.error(f"Crypto DCA analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/crypto-historical/{symbol}")
async def get_crypto_historical(symbol: str, days: int = 30):
    """Get historical price data for a cryptocurrency"""
    try:
        historical_data = crypto_fetcher.get_crypto_historical_data(symbol, days)
        
        if historical_data.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")
        
        # Convert to list format for API response
        data_points = []
        for timestamp, row in historical_data.iterrows():
            data_points.append({
                "timestamp": timestamp.isoformat(),
                "price": float(row['price']),
                "volume": float(row.get('volume', 0))
            })
        
        return {
            "symbol": symbol.upper(),
            "days": days,
            "data_points": len(data_points),
            "data": data_points
        }
        
    except Exception as e:
        logger.error(f"Crypto historical data error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml-train")
async def train_ml_model(request: MLTrainingRequest):
    """Train machine learning model for price prediction"""
    try:
        result = ml_predictor.train_model(
            symbol=request.symbol,
            horizon=request.horizon
        )
        
        return result
        
    except Exception as e:
        logger.error(f"ML training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml-predict")
async def predict_ml_price(request: MLPredictionRequest):
    """Make ML price prediction"""
    try:
        prediction = ml_predictor.predict_price(
            symbol=request.symbol,
            horizon=request.horizon
        )
        
        return {
            "symbol": prediction.symbol,
            "predictions": prediction.predictions,
            "confidence_score": prediction.confidence_score,
            "model_type": prediction.model_type,
            "prediction_horizon": prediction.prediction_horizon,
            "current_price": prediction.current_price,
            "predicted_direction": prediction.predicted_direction,
            "probability_up": prediction.probability_up,
            "probability_down": prediction.probability_down,
            "target_prices": prediction.target_prices,
            "timestamp": prediction.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"ML prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml-batch-predict")
async def batch_predict_ml(request: BatchPredictionRequest):
    """Make ML predictions for multiple symbols"""
    try:
        results = ml_predictor.batch_predict(
            symbols=request.symbols,
            horizon=request.horizon
        )
        
        response = {}
        for symbol, prediction in results.items():
            response[symbol] = {
                "predictions": prediction.predictions,
                "confidence_score": prediction.confidence_score,
                "model_type": prediction.model_type,
                "current_price": prediction.current_price,
                "predicted_direction": prediction.predicted_direction,
                "target_prices": prediction.target_prices,
                "timestamp": prediction.timestamp.isoformat()
            }
        
        return {
            "predictions": response,
            "symbols_processed": len(results),
            "horizon": request.horizon
        }
        
    except Exception as e:
        logger.error(f"Batch ML prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml-performance/{symbol}")
async def get_ml_performance(symbol: str, horizon: int = 5):
    """Get ML model performance metrics"""
    try:
        performance = ml_predictor.get_model_performance(symbol, horizon)
        return performance
        
    except Exception as e:
        logger.error(f"ML performance error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/lstm-predict")
async def predict_lstm_price(request: LSTMPredictionRequest):
    """Make LSTM deep learning price prediction"""
    try:
        if not TENSORFLOW_AVAILABLE:
            raise HTTPException(status_code=501, detail="TensorFlow not available for LSTM predictions")
        
        lstm_predictor_instance = LSTMPredictor(
            sequence_length=request.sequence_length,
            prediction_horizon=request.prediction_horizon
        )
        
        prediction = lstm_predictor_instance.predict_lstm(request.symbol)
        
        return {
            "symbol": prediction.symbol,
            "predictions": prediction.predictions,
            "confidence_intervals": prediction.confidence_intervals,
            "model_accuracy": prediction.model_accuracy,
            "sequence_length": prediction.sequence_length,
            "prediction_horizon": prediction.prediction_horizon,
            "features_used": prediction.features_used,
            "timestamp": prediction.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"LSTM prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/lstm-train")
async def train_lstm_model(request: LSTMPredictionRequest):
    """Train LSTM model"""
    try:
        if not TENSORFLOW_AVAILABLE:
            raise HTTPException(status_code=501, detail="TensorFlow not available for LSTM training")
        
        lstm_predictor_instance = LSTMPredictor(
            sequence_length=request.sequence_length,
            prediction_horizon=request.prediction_horizon
        )
        
        result = lstm_predictor_instance.train_lstm_model(request.symbol)
        
        return result
        
    except Exception as e:
        logger.error(f"LSTM training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml-models-available")
async def get_available_ml_models():
    """Get information about available ML models"""
    return {
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "ml_models": [
            "ensemble",
            "random_forest",
            "gradient_boost",
            "linear"
        ],
        "deep_learning_models": [
            "lstm" if TENSORFLOW_AVAILABLE else "simple_nn"
        ],
        "features": {
            "technical_indicators": ["RSI", "MACD", "Bollinger Bands", "Moving Averages"],
            "price_patterns": ["Momentum", "Volatility", "Price Ranges"],
            "volume_analysis": ["Volume Ratios", "Price-Volume"],
            "lag_features": ["Historical Prices", "Historical Returns"]
        }
    }


@app.post("/api/multi-timeframe-analysis")
async def multi_timeframe_analysis(request: MultiTimeframeRequest):
    """Comprehensive multi-timeframe analysis"""
    try:
        analysis = multi_timeframe_analyzer.multi_timeframe_analysis(
            symbol=request.symbol,
            timeframes=request.timeframes
        )
        
        # Convert TimeframeSignal objects to dictionaries
        timeframe_signals = {}
        for tf, signal in analysis.timeframe_signals.items():
            timeframe_signals[tf] = {
                "timeframe": signal.timeframe,
                "signal": signal.signal,
                "confidence": signal.confidence,
                "price": signal.price,
                "indicators": signal.indicators,
                "trend": signal.trend,
                "support_level": signal.support_level,
                "resistance_level": signal.resistance_level,
                "volume_trend": signal.volume_trend
            }
        
        return {
            "symbol": analysis.symbol,
            "timeframe_signals": timeframe_signals,
            "consensus_signal": analysis.consensus_signal,
            "consensus_confidence": analysis.consensus_confidence,
            "trend_alignment": analysis.trend_alignment,
            "key_levels": analysis.key_levels,
            "volume_analysis": analysis.volume_analysis,
            "risk_assessment": analysis.risk_assessment,
            "timestamp": analysis.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Multi-timeframe analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/timeframe-comparison/{symbol}")
async def timeframe_comparison(symbol: str, timeframes: Optional[str] = "1d,1wk,1mo"):
    """Get timeframe comparison for a symbol"""
    try:
        timeframe_list = timeframes.split(",") if timeframes else ["1d", "1wk", "1mo"]
        comparison = multi_timeframe_analyzer.get_timeframe_comparison(symbol)
        
        return comparison
        
    except Exception as e:
        logger.error(f"Timeframe comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch-timeframe-analysis")
async def batch_timeframe_analysis(request: TimeframeComparisonRequest):
    """Get multi-timeframe analysis for multiple symbols"""
    try:
        results = {}
        
        for symbol in request.symbols:
            try:
                analysis = multi_timeframe_analyzer.multi_timeframe_analysis(
                    symbol=symbol,
                    timeframes=request.timeframes
                )
                
                results[symbol] = {
                    "consensus_signal": analysis.consensus_signal,
                    "consensus_confidence": analysis.consensus_confidence,
                    "risk_level": analysis.risk_assessment["risk_level"],
                    "trend_alignment": analysis.trend_alignment,
                    "timeframe_signals": {
                        tf: {
                            "signal": signal.signal,
                            "confidence": signal.confidence,
                            "trend": signal.trend
                        }
                        for tf, signal in analysis.timeframe_signals.items()
                    }
                }
                
            except Exception as e:
                logger.warning(f"Error analyzing {symbol}: {str(e)}")
                results[symbol] = {"error": str(e)}
        
        return {
            "results": results,
            "symbols_processed": len([k for k, v in results.items() if "error" not in v]),
            "symbols_failed": len([k for k, v in results.items() if "error" in v]),
            "timeframes": request.timeframes
        }
        
    except Exception as e:
        logger.error(f"Batch timeframe analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/timeframe-signals/{symbol}/{timeframe}")
async def get_timeframe_signal(symbol: str, timeframe: str):
    """Get detailed signal for specific symbol and timeframe"""
    try:
        signal = multi_timeframe_analyzer.analyze_timeframe(symbol, timeframe)
        
        return {
            "symbol": symbol,
            "timeframe": signal.timeframe,
            "signal": signal.signal,
            "confidence": signal.confidence,
            "price": signal.price,
            "indicators": signal.indicators,
            "trend": signal.trend,
            "support_level": signal.support_level,
            "resistance_level": signal.resistance_level,
            "volume_trend": signal.volume_trend,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Timeframe signal error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/available-timeframes")
async def get_available_timeframes():
    """Get list of available timeframes for analysis"""
    return {
        "timeframes": [
            {"code": "1d", "name": "Daily", "description": "Daily charts for swing trading"},
            {"code": "1wk", "name": "Weekly", "description": "Weekly charts for medium-term trends"},
            {"code": "1mo", "name": "Monthly", "description": "Monthly charts for long-term trends"}
        ],
        "supported_intervals": ["1d", "1wk", "1mo"],
        "recommended_combinations": [
            ["1d", "1wk", "1mo"],
            ["1d", "1wk"], 
            ["1wk", "1mo"]
        ]
    }


@app.post("/api/risk-analysis")
async def analyze_risk(request: RiskAnalysisRequest):
    """Get comprehensive risk analysis for a symbol"""
    try:
        risk_metrics = risk_calculator.calculate_risk_metrics(
            symbol=request.symbol,
            period=request.period,
            benchmark=request.benchmark
        )
        
        return {
            "symbol": request.symbol,
            "var_95": risk_metrics.var_95,
            "var_99": risk_metrics.var_99,
            "expected_shortfall_95": risk_metrics.expected_shortfall_95,
            "expected_shortfall_99": risk_metrics.expected_shortfall_99,
            "max_drawdown": risk_metrics.max_drawdown,
            "volatility": risk_metrics.volatility,
            "sharpe_ratio": risk_metrics.sharpe_ratio,
            "sortino_ratio": risk_metrics.sortino_ratio,
            "beta": risk_metrics.beta,
            "correlation_to_market": risk_metrics.correlation_to_market,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Risk analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/position-sizing")
async def calculate_position_size(request: PositionSizingRequest):
    """Calculate optimal position size based on risk management"""
    try:
        position_sizer = PositionSizer(
            portfolio_value=request.portfolio_value,
            max_risk_per_trade=request.max_risk_per_trade
        )
        
        sizing = position_sizer.calculate_risk_based_position_size(
            symbol=request.symbol,
            entry_price=request.entry_price,
            stop_loss_price=request.stop_loss_price,
            target_price=request.target_price
        )
        
        return {
            "symbol": sizing.symbol,
            "recommended_position_size": sizing.recommended_position_size,
            "max_position_size": sizing.max_position_size,
            "kelly_criterion": sizing.kelly_criterion,
            "risk_adjusted_size": sizing.risk_adjusted_size,
            "stop_loss_level": sizing.stop_loss_level,
            "take_profit_level": sizing.take_profit_level,
            "risk_reward_ratio": sizing.risk_reward_ratio,
            "position_value": sizing.position_value,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Position sizing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/portfolio-risk")
async def assess_portfolio_risk(request: PortfolioRiskRequest):
    """Comprehensive portfolio risk assessment"""
    try:
        risk_assessment = portfolio_risk_manager.assess_portfolio_risk(
            positions=request.positions,
            portfolio_value=request.portfolio_value
        )
        
        return {
            "total_var_95": risk_assessment.total_var_95,
            "total_var_99": risk_assessment.total_var_99,
            "diversification_ratio": risk_assessment.diversification_ratio,
            "portfolio_beta": risk_assessment.portfolio_beta,
            "correlation_matrix": risk_assessment.correlation_matrix,
            "concentration_risk": risk_assessment.concentration_risk,
            "stress_test_scenarios": risk_assessment.stress_test_scenarios,
            "risk_budget_allocation": risk_assessment.risk_budget_allocation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Portfolio risk assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/risk-monitoring")
async def monitor_risk_limits(request: RiskMonitoringRequest):
    """Monitor portfolio against risk limits and generate alerts"""
    try:
        risk_monitor = RiskMonitor(portfolio_value=request.portfolio_value)
        risk_check = risk_monitor.check_risk_limits(request.positions)
        
        return risk_check
        
    except Exception as e:
        logger.error(f"Risk monitoring error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio-var/{portfolio_id}")
async def get_portfolio_var(portfolio_id: str, confidence_level: float = 0.95):
    """Get Portfolio Value at Risk for existing portfolio"""
    try:
        # This would typically fetch portfolio data from database
        # For demo, using sample data
        sample_positions = {
            'AAPL': 0.30,
            'GOOGL': 0.25,
            'MSFT': 0.25,
            'TSLA': 0.20
        }
        
        var_results = portfolio_risk_manager.calculate_portfolio_var(
            positions=sample_positions,
            confidence_level=confidence_level
        )
        
        return {
            "portfolio_id": portfolio_id,
            "confidence_level": confidence_level,
            "portfolio_var": var_results["portfolio_var"],
            "individual_vars": var_results["individual_vars"],
            "diversification_benefit": var_results.get("diversification_benefit", 0),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Portfolio VaR error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/correlation-matrix")
async def get_correlation_matrix(symbols: str = "AAPL,GOOGL,MSFT,TSLA", period: str = "1y"):
    """Get correlation matrix for specified symbols"""
    try:
        symbol_list = symbols.split(",")
        correlation_matrix = portfolio_risk_manager.calculate_correlation_matrix(symbol_list, period)
        
        if correlation_matrix.empty:
            return {"error": "Unable to calculate correlation matrix"}
        
        # Convert to dict format
        correlation_dict = {}
        for symbol1 in correlation_matrix.index:
            correlation_dict[symbol1] = correlation_matrix.loc[symbol1].to_dict()
        
        return {
            "symbols": symbol_list,
            "correlation_matrix": correlation_dict,
            "period": period,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Correlation matrix error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk-limits")
async def get_risk_limits():
    """Get standard risk management limits and parameters"""
    return {
        "var_limits": {
            "daily_95": 0.05,  # 5% daily VaR limit
            "daily_99": 0.08,  # 8% daily VaR limit at 99%
        },
        "position_limits": {
            "max_single_position": 0.20,  # 20% max single position
            "max_sector_concentration": 0.40,  # 40% max sector exposure
            "max_beta": 1.5,  # Maximum portfolio beta
        },
        "correlation_limits": {
            "max_avg_correlation": 0.7,  # Maximum average correlation
            "min_diversification_ratio": 1.2,  # Minimum diversification benefit
        },
        "risk_parameters": {
            "default_max_risk_per_trade": 0.02,  # 2% max risk per trade
            "kelly_fraction_limit": 0.25,  # 25% max Kelly fraction
            "confidence_levels": [0.90, 0.95, 0.99],
        },
        "rebalancing_triggers": {
            "var_breach": "Immediate rebalancing required",
            "concentration_breach": "Consider position reduction",
            "correlation_increase": "Add diversifying assets",
        }
    }


# Options Trading Endpoints

@app.post("/api/options-chain")
async def get_options_chain(request: OptionsChainRequest):
    """Get options chain data for a symbol"""
    try:
        chain = options_fetcher.get_options_chain(
            symbol=request.symbol,
            expiration_date=request.expiration_date
        )
        
        if not chain:
            raise HTTPException(status_code=404, detail=f"No options data found for {request.symbol}")
        
        # Convert to serializable format
        calls_data = {}
        for strike, contract in chain.calls.items():
            calls_data[str(strike)] = {
                "symbol": contract.symbol,
                "strike": contract.strike,
                "expiration": contract.expiration.isoformat(),
                "price": safe_float(contract.price),
                "bid": safe_float(contract.bid),
                "ask": safe_float(contract.ask),
                "volume": contract.volume,
                "open_interest": contract.open_interest,
                "implied_volatility": safe_float(contract.implied_volatility),
                "delta": safe_float(contract.delta),
                "gamma": safe_float(contract.gamma),
                "theta": safe_float(contract.theta),
                "vega": safe_float(contract.vega),
                "rho": safe_float(contract.rho)
            }
        
        puts_data = {}
        for strike, contract in chain.puts.items():
            puts_data[str(strike)] = {
                "symbol": contract.symbol,
                "strike": contract.strike,
                "expiration": contract.expiration.isoformat(),
                "price": safe_float(contract.price),
                "bid": safe_float(contract.bid),
                "ask": safe_float(contract.ask),
                "volume": contract.volume,
                "open_interest": contract.open_interest,
                "implied_volatility": safe_float(contract.implied_volatility),
                "delta": safe_float(contract.delta),
                "gamma": safe_float(contract.gamma),
                "theta": safe_float(contract.theta),
                "vega": safe_float(contract.vega),
                "rho": safe_float(contract.rho)
            }
        
        return {
            "symbol": chain.underlying_symbol,
            "underlying_price": safe_float(chain.underlying_price),
            "expiration_dates": [dt.isoformat() for dt in chain.expiration_dates],
            "calls": calls_data,
            "puts": puts_data,
            "timestamp": chain.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Options chain error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/covered-call")
async def create_covered_call(request: CoveredCallRequest):
    """Create a covered call strategy"""
    try:
        strategy = options_strategist.create_covered_call(
            symbol=request.symbol,
            shares_owned=request.shares_owned,
            target_strike_pct=request.target_strike_pct
        )
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Unable to create covered call for {request.symbol}")
        
        return {
            "strategy_name": strategy.name,
            "description": strategy.description,
            "max_profit": safe_float(strategy.max_profit),
            "max_loss": safe_float(strategy.max_loss),
            "breakeven_points": [safe_float(bp) for bp in strategy.breakeven_points],
            "total_premium": safe_float(strategy.total_premium),
            "risk_reward_ratio": safe_float(strategy.risk_reward_ratio),
            "legs": len(strategy.legs),
            "recommendation": "Conservative income strategy - suitable for stable stocks"
        }
        
    except Exception as e:
        logger.error(f"Covered call error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/protective-put")
async def create_protective_put(request: ProtectivePutRequest):
    """Create a protective put strategy"""
    try:
        strategy = options_strategist.create_protective_put(
            symbol=request.symbol,
            shares_owned=request.shares_owned,
            protection_level=request.protection_level
        )
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Unable to create protective put for {request.symbol}")
        
        return {
            "strategy_name": strategy.name,
            "description": strategy.description,
            "max_profit": safe_float(strategy.max_profit),
            "max_loss": safe_float(strategy.max_loss),
            "breakeven_points": [safe_float(bp) for bp in strategy.breakeven_points],
            "total_premium": safe_float(strategy.total_premium),
            "risk_reward_ratio": safe_float(strategy.risk_reward_ratio),
            "legs": len(strategy.legs),
            "recommendation": "Protective strategy - insurance against downside risk"
        }
        
    except Exception as e:
        logger.error(f"Protective put error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bull-call-spread")
async def create_bull_call_spread(request: BullCallSpreadRequest):
    """Create a bull call spread strategy"""
    try:
        strategy = options_strategist.create_bull_call_spread(
            symbol=request.symbol,
            risk_amount=request.risk_amount,
            spread_width=request.spread_width
        )
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Unable to create bull call spread for {request.symbol}")
        
        return {
            "strategy_name": strategy.name,
            "description": strategy.description,
            "max_profit": safe_float(strategy.max_profit),
            "max_loss": safe_float(strategy.max_loss),
            "breakeven_points": [safe_float(bp) for bp in strategy.breakeven_points],
            "total_premium": safe_float(strategy.total_premium),
            "risk_reward_ratio": safe_float(strategy.risk_reward_ratio),
            "legs": len(strategy.legs),
            "recommendation": "Bullish strategy with limited risk - good for moderately bullish outlook"
        }
        
    except Exception as e:
        logger.error(f"Bull call spread error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/iron-condor")
async def create_iron_condor(request: IronCondorRequest):
    """Create an iron condor strategy"""
    try:
        strategy = options_strategist.create_iron_condor(
            symbol=request.symbol,
            risk_amount=request.risk_amount,
            wing_width=request.wing_width
        )
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Unable to create iron condor for {request.symbol}")
        
        return {
            "strategy_name": strategy.name,
            "description": strategy.description,
            "max_profit": safe_float(strategy.max_profit),
            "max_loss": safe_float(strategy.max_loss),
            "breakeven_points": [safe_float(bp) for bp in strategy.breakeven_points],
            "total_premium": safe_float(strategy.total_premium),
            "risk_reward_ratio": safe_float(strategy.risk_reward_ratio),
            "legs": len(strategy.legs),
            "recommendation": "Neutral strategy - profits from low volatility and sideways movement"
        }
        
    except Exception as e:
        logger.error(f"Iron condor error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/long-straddle")
async def create_long_straddle(request: LongStraddleRequest):
    """Create a long straddle strategy"""
    try:
        strategy = options_strategist.create_long_straddle(
            symbol=request.symbol,
            risk_amount=request.risk_amount
        )
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Unable to create long straddle for {request.symbol}")
        
        return {
            "strategy_name": strategy.name,
            "description": strategy.description,
            "max_profit": safe_float(strategy.max_profit),
            "max_loss": safe_float(strategy.max_loss),
            "breakeven_points": [safe_float(bp) for bp in strategy.breakeven_points],
            "total_premium": safe_float(strategy.total_premium),
            "risk_reward_ratio": safe_float(strategy.risk_reward_ratio),
            "legs": len(strategy.legs),
            "recommendation": "Volatility strategy - profits from large price movements in either direction"
        }
        
    except Exception as e:
        logger.error(f"Long straddle error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/options-volatility")
async def get_options_volatility(request: StockRequest):
    """Get historical and implied volatility data"""
    try:
        # Get historical volatility
        hist_vol = options_fetcher.get_historical_volatility(request.symbol, days=30)
        
        # Get options chain to calculate average implied volatility
        chain = options_fetcher.get_options_chain(request.symbol)
        
        avg_call_iv = 0
        avg_put_iv = 0
        
        if chain:
            call_ivs = [contract.implied_volatility for contract in chain.calls.values() 
                       if contract.implied_volatility > 0]
            put_ivs = [contract.implied_volatility for contract in chain.puts.values() 
                      if contract.implied_volatility > 0]
            
            avg_call_iv = np.mean(call_ivs) if call_ivs else 0
            avg_put_iv = np.mean(put_ivs) if put_ivs else 0
        
        # Volatility analysis
        iv_rank = 0
        if avg_call_iv > 0 and hist_vol > 0:
            iv_rank = avg_call_iv / hist_vol
        
        volatility_regime = "normal"
        if avg_call_iv > hist_vol * 1.25:
            volatility_regime = "high_iv"
        elif avg_call_iv < hist_vol * 0.75:
            volatility_regime = "low_iv"
        
        return {
            "symbol": request.symbol,
            "historical_volatility_30d": safe_float(hist_vol),
            "average_call_iv": safe_float(avg_call_iv),
            "average_put_iv": safe_float(avg_put_iv),
            "iv_rank": safe_float(iv_rank),
            "volatility_regime": volatility_regime,
            "recommendations": {
                "high_iv": "Consider selling options (iron condors, covered calls)",
                "low_iv": "Consider buying options (straddles, protective puts)",
                "normal": "Neutral strategies appropriate"
            }.get(volatility_regime, "Monitor volatility changes"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Options volatility error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/liquid-options")
async def find_liquid_options(request: StockRequest):
    """Find liquid options for trading"""
    try:
        chain = options_fetcher.get_options_chain(request.symbol)
        if not chain:
            raise HTTPException(status_code=404, detail=f"No options data found for {request.symbol}")
        
        liquid_options = options_fetcher.find_liquid_options(
            chain, 
            min_volume=10,
            min_open_interest=50
        )
        
        # Format liquid calls
        liquid_calls = []
        for contract in liquid_options["calls"]:
            liquid_calls.append({
                "strike": contract.strike,
                "price": safe_float(contract.price),
                "bid": safe_float(contract.bid),
                "ask": safe_float(contract.ask),
                "volume": contract.volume,
                "open_interest": contract.open_interest,
                "implied_volatility": safe_float(contract.implied_volatility),
                "delta": safe_float(contract.delta)
            })
        
        # Format liquid puts
        liquid_puts = []
        for contract in liquid_options["puts"]:
            liquid_puts.append({
                "strike": contract.strike,
                "price": safe_float(contract.price),
                "bid": safe_float(contract.bid),
                "ask": safe_float(contract.ask),
                "volume": contract.volume,
                "open_interest": contract.open_interest,
                "implied_volatility": safe_float(contract.implied_volatility),
                "delta": safe_float(contract.delta)
            })
        
        return {
            "symbol": request.symbol,
            "underlying_price": safe_float(chain.underlying_price),
            "liquid_calls": liquid_calls,
            "liquid_puts": liquid_puts,
            "total_liquid_calls": len(liquid_calls),
            "total_liquid_puts": len(liquid_puts),
            "liquidity_score": min(len(liquid_calls) + len(liquid_puts), 100) / 100,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Liquid options error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Advanced Options Trading Endpoints

@app.post("/api/butterfly-spread")
async def create_butterfly_spread(request: ButterflyRequest):
    """Create a butterfly spread strategy"""
    try:
        option_type = OptionType.CALL if request.option_type.lower() == "call" else OptionType.PUT
        
        strategy = advanced_options_strategist.create_long_butterfly(
            symbol=request.symbol,
            center_strike=request.center_strike,
            wing_width=request.wing_width,
            option_type=option_type
        )
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Unable to create butterfly spread for {request.symbol}")
        
        return {
            "strategy_name": strategy.name,
            "description": strategy.description,
            "max_profit": safe_float(strategy.max_profit),
            "max_loss": safe_float(strategy.max_loss),
            "breakeven_points": [safe_float(bp) for bp in strategy.breakeven_points],
            "risk_reward_ratio": safe_float(strategy.risk_reward_ratio),
            "total_premium": safe_float(strategy.total_premium),
            "legs": len(strategy.legs),
            "strategy_type": "neutral",
            "recommendation": "Best for low volatility environments - profits when stock stays near center strike"
        }
        
    except Exception as e:
        logger.error(f"Butterfly spread error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/iron-butterfly")
async def create_iron_butterfly(request: ButterflyRequest):
    """Create an iron butterfly strategy"""
    try:
        strategy = advanced_options_strategist.create_iron_butterfly(
            symbol=request.symbol,
            center_strike=request.center_strike,
            wing_width=request.wing_width
        )
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Unable to create iron butterfly for {request.symbol}")
        
        return {
            "strategy_name": strategy.name,
            "description": strategy.description,
            "max_profit": safe_float(strategy.max_profit),
            "max_loss": safe_float(strategy.max_loss),
            "breakeven_points": [safe_float(bp) for bp in strategy.breakeven_points],
            "risk_reward_ratio": safe_float(strategy.risk_reward_ratio),
            "total_premium": safe_float(strategy.total_premium),
            "legs": len(strategy.legs),
            "strategy_type": "neutral",
            "recommendation": "High probability income strategy - collect premium when stock stays near center strike"
        }
        
    except Exception as e:
        logger.error(f"Iron butterfly error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calendar-spread")
async def create_calendar_spread(request: CalendarSpreadRequest):
    """Create a calendar spread strategy"""
    try:
        option_type = OptionType.CALL if request.option_type.lower() == "call" else OptionType.PUT
        
        strategy = advanced_options_strategist.create_calendar_spread(
            symbol=request.symbol,
            strike=request.strike,
            option_type=option_type,
            short_dte=request.short_dte,
            long_dte=request.long_dte
        )
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Unable to create calendar spread for {request.symbol}")
        
        return {
            "strategy_name": strategy.name,
            "description": strategy.description,
            "max_profit": safe_float(strategy.max_profit),
            "max_loss": safe_float(strategy.max_loss),
            "breakeven_points": [safe_float(bp) for bp in strategy.breakeven_points],
            "risk_reward_ratio": safe_float(strategy.risk_reward_ratio),
            "total_premium": safe_float(strategy.total_premium),
            "legs": len(strategy.legs),
            "strategy_type": "time_decay",
            "recommendation": "Benefits from time decay - best when stock stays near strike price"
        }
        
    except Exception as e:
        logger.error(f"Calendar spread error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/jade-lizard")
async def create_jade_lizard(request: JadeLizardRequest):
    """Create a jade lizard strategy"""
    try:
        strategy = advanced_options_strategist.create_jade_lizard(
            symbol=request.symbol,
            risk_amount=request.risk_amount
        )
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Unable to create jade lizard for {request.symbol}")
        
        return {
            "strategy_name": strategy.name,
            "description": strategy.description,
            "max_profit": safe_float(strategy.max_profit),
            "max_loss": safe_float(strategy.max_loss) if strategy.max_loss != float('inf') else None,
            "unlimited_downside_risk": strategy.max_loss == float('inf'),
            "breakeven_points": [safe_float(bp) for bp in strategy.breakeven_points],
            "risk_reward_ratio": safe_float(strategy.risk_reward_ratio) if strategy.risk_reward_ratio != float('inf') else None,
            "total_premium": safe_float(strategy.total_premium),
            "legs": len(strategy.legs),
            "strategy_type": "income",
            "recommendation": "High probability income strategy - but unlimited downside risk requires careful management"
        }
        
    except Exception as e:
        logger.error(f"Jade lizard error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ratio-spread")
async def create_ratio_spread(request: RatioSpreadRequest):
    """Create a ratio call spread strategy"""
    try:
        strategy = advanced_options_strategist.create_ratio_call_spread(
            symbol=request.symbol,
            ratio=request.ratio,
            risk_amount=request.risk_amount
        )
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Unable to create ratio spread for {request.symbol}")
        
        return {
            "strategy_name": strategy.name,
            "description": strategy.description,
            "max_profit": safe_float(strategy.max_profit),
            "max_loss": safe_float(strategy.max_loss) if strategy.max_loss != float('inf') else None,
            "unlimited_upside_risk": strategy.max_loss == float('inf'),
            "breakeven_points": [safe_float(bp) for bp in strategy.breakeven_points],
            "risk_reward_ratio": safe_float(strategy.risk_reward_ratio) if strategy.risk_reward_ratio != float('inf') else None,
            "total_premium": safe_float(strategy.total_premium),
            "legs": len(strategy.legs),
            "strategy_type": "income_with_risk",
            "recommendation": f"Income strategy with {request.ratio}:1 ratio - monitor upside risk carefully"
        }
        
    except Exception as e:
        logger.error(f"Ratio spread error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/synthetic-stock")
async def create_synthetic_stock(request: SyntheticStockRequest):
    """Create a synthetic stock position"""
    try:
        strategy = advanced_options_strategist.create_synthetic_stock(
            symbol=request.symbol,
            strike=request.strike
        )
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Unable to create synthetic stock for {request.symbol}")
        
        return {
            "strategy_name": strategy.name,
            "description": strategy.description,
            "max_profit": None,  # Unlimited like stock
            "max_loss": None,    # Unlimited like stock
            "breakeven_points": [safe_float(bp) for bp in strategy.breakeven_points],
            "total_premium": safe_float(strategy.total_premium),
            "legs": len(strategy.legs),
            "strategy_type": "synthetic",
            "stock_equivalent": "100_shares",
            "recommendation": "Replicates 100 shares of stock using options - same risk/reward profile"
        }
        
    except Exception as e:
        logger.error(f"Synthetic stock error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/advanced-strategy-analysis")
async def analyze_advanced_strategy(request: AdvancedStrategyAnalysisRequest):
    """Perform advanced analysis of an options strategy"""
    try:
        # This would typically reconstruct the strategy from the data
        # For now, we'll provide a comprehensive analysis framework
        
        return {
            "symbol": request.symbol,
            "current_price": request.current_price,
            "analysis": {
                "greeks_summary": {
                    "delta_exposure": "Position directional sensitivity",
                    "gamma_risk": "Delta change sensitivity", 
                    "theta_impact": "Time decay effect",
                    "vega_sensitivity": "Volatility impact"
                },
                "risk_factors": [
                    "Monitor assignment risk on short options",
                    "Track implied volatility changes",
                    "Manage time decay exposure",
                    "Watch for early exercise risk"
                ],
                "optimal_conditions": [
                    "Best market environment for this strategy",
                    "Ideal volatility range",
                    "Preferred time to expiration"
                ],
                "exit_strategies": [
                    "Profit-taking levels",
                    "Stop-loss conditions", 
                    "Time-based exits",
                    "Volatility-based adjustments"
                ]
            },
            "recommendations": [
                "Monitor position Greeks daily",
                "Set profit targets at 50-75% of maximum",
                "Have exit plan for adverse scenarios",
                "Consider portfolio impact"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Advanced strategy analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/options-strategy-comparison/{symbol}")
async def compare_options_strategies(symbol: str):
    """Compare different options strategies for a symbol"""
    try:
        strategies_comparison = {}
        
        # Get basic strategies
        covered_call = options_strategist.create_covered_call(symbol, 500)
        bull_call_spread = options_strategist.create_bull_call_spread(symbol, 1000)
        iron_condor = options_strategist.create_iron_condor(symbol, 2000)
        
        # Get advanced strategies
        butterfly = advanced_options_strategist.create_long_butterfly(symbol)
        iron_butterfly = advanced_options_strategist.create_iron_butterfly(symbol)
        calendar = advanced_options_strategist.create_calendar_spread(symbol)
        
        strategies = {
            "covered_call": covered_call,
            "bull_call_spread": bull_call_spread,
            "iron_condor": iron_condor,
            "butterfly": butterfly,
            "iron_butterfly": iron_butterfly,
            "calendar_spread": calendar
        }
        
        for name, strategy in strategies.items():
            if strategy:
                strategies_comparison[name] = {
                    "name": strategy.name,
                    "description": strategy.description,
                    "max_profit": safe_float(strategy.max_profit) if strategy.max_profit != float('inf') else None,
                    "max_loss": safe_float(strategy.max_loss) if strategy.max_loss != float('inf') else None,
                    "risk_reward_ratio": safe_float(strategy.risk_reward_ratio) if strategy.risk_reward_ratio != float('inf') else None,
                    "breakeven_points": [safe_float(bp) for bp in strategy.breakeven_points],
                    "complexity": len(strategy.legs),
                    "strategy_category": {
                        "covered_call": "Income",
                        "bull_call_spread": "Bullish", 
                        "iron_condor": "Neutral",
                        "butterfly": "Neutral",
                        "iron_butterfly": "Neutral",
                        "calendar_spread": "Time Decay"
                    }.get(name, "Advanced")
                }
        
        return {
            "symbol": symbol,
            "total_strategies": len(strategies_comparison),
            "strategies": strategies_comparison,
            "recommendations": {
                "bullish_outlook": ["bull_call_spread", "covered_call"],
                "neutral_outlook": ["iron_condor", "butterfly", "iron_butterfly"],
                "income_generation": ["covered_call", "iron_condor", "iron_butterfly"],
                "low_volatility": ["butterfly", "iron_butterfly", "calendar_spread"],
                "high_volatility": ["straddle", "strangle"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Strategy comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "investment-dashboard"}


# Options Backtesting Endpoints

class BacktestRequest(BaseModel):
    symbol: str
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: Optional[float] = 50000
    strategy_params: Optional[Dict[str, Any]] = None

class MultiStrategyBacktestRequest(BaseModel):
    symbol: str
    strategies: List[str]
    start_date: str
    end_date: str
    initial_capital: Optional[float] = 50000


@app.post("/api/backtest-strategy")
async def backtest_options_strategy(request: BacktestRequest):
    """Run backtest for a specific options strategy"""
    try:
        backtester = OptionsBacktester(initial_capital=request.initial_capital)
        
        # Parse dates
        start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
        
        # Run backtest
        results = backtester.run_strategy_backtest(
            symbol=request.symbol,
            strategy_name=request.strategy_name,
            start_date=start_date,
            end_date=end_date,
            strategy_params=request.strategy_params or {}
        )
        
        # Convert results to serializable format
        return {
            "backtest_summary": {
                "symbol": request.symbol,
                "strategy": request.strategy_name,
                "start_date": results.start_date.isoformat(),
                "end_date": results.end_date.isoformat(),
                "initial_capital": results.initial_capital,
                "final_capital": results.final_capital,
                "total_return": safe_float(results.total_return),
                "total_return_pct": safe_float(results.total_return_pct),
                "max_drawdown": safe_float(results.max_drawdown),
                "max_drawdown_pct": safe_float(results.max_drawdown_pct),
                "sharpe_ratio": safe_float(results.sharpe_ratio),
                "win_rate": safe_float(results.win_rate),
                "profit_factor": safe_float(results.profit_factor)
            },
            "trading_statistics": {
                "total_trades": results.total_trades,
                "winning_trades": results.winning_trades,
                "losing_trades": results.losing_trades,
                "avg_win": safe_float(results.avg_win),
                "avg_loss": safe_float(results.avg_loss),
                "best_trade": safe_float(results.best_trade),
                "worst_trade": safe_float(results.worst_trade),
                "avg_days_held": safe_float(results.avg_days_held)
            },
            "trades_summary": [
                {
                    "entry_date": trade.entry_date.isoformat(),
                    "exit_date": trade.exit_date.isoformat() if trade.exit_date else None,
                    "pnl": safe_float(trade.pnl),
                    "pnl_pct": safe_float(trade.pnl_pct),
                    "days_held": trade.days_held,
                    "exit_reason": trade.exit_reason,
                    "underlying_entry": safe_float(trade.underlying_entry),
                    "underlying_exit": safe_float(trade.underlying_exit) if trade.underlying_exit else None
                }
                for trade in results.trades[:10]  # Return first 10 trades
            ],
            "total_trades_count": len(results.trades),
            "equity_curve_points": len(results.equity_curve),
            "strategy_breakdown": results.strategy_breakdown,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest-multi-strategy")
async def backtest_multiple_strategies(request: MultiStrategyBacktestRequest):
    """Compare multiple options strategies via backtesting"""
    try:
        # Parse dates
        start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
        
        # Run multi-strategy backtest
        backtester = OptionsBacktester(initial_capital=request.initial_capital)
        results = backtester.run_multi_strategy_backtest(
            symbol=request.symbol,
            strategies=request.strategies,
            start_date=start_date,
            end_date=end_date
        )
        
        # Format results for comparison
        strategy_comparison = {}
        for strategy_name, result in results.items():
            strategy_comparison[strategy_name] = {
                "total_return_pct": safe_float(result.total_return_pct),
                "max_drawdown_pct": safe_float(result.max_drawdown_pct),
                "sharpe_ratio": safe_float(result.sharpe_ratio),
                "win_rate": safe_float(result.win_rate),
                "profit_factor": safe_float(result.profit_factor),
                "total_trades": result.total_trades,
                "avg_days_held": safe_float(result.avg_days_held),
                "best_trade": safe_float(result.best_trade),
                "worst_trade": safe_float(result.worst_trade)
            }
        
        # Find best performing strategy
        best_strategy = max(
            strategy_comparison.keys(),
            key=lambda s: strategy_comparison[s]["total_return_pct"]
        ) if strategy_comparison else None
        
        return {
            "symbol": request.symbol,
            "strategies_tested": request.strategies,
            "backtest_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": (end_date - start_date).days
            },
            "strategy_comparison": strategy_comparison,
            "best_strategy": {
                "name": best_strategy,
                "return_pct": strategy_comparison[best_strategy]["total_return_pct"] if best_strategy else 0,
                "win_rate": strategy_comparison[best_strategy]["win_rate"] if best_strategy else 0,
                "sharpe_ratio": strategy_comparison[best_strategy]["sharpe_ratio"] if best_strategy else 0
            } if best_strategy else None,
            "performance_ranking": sorted(
                strategy_comparison.keys(),
                key=lambda s: strategy_comparison[s]["total_return_pct"],
                reverse=True
            ),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Multi-strategy backtest error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/available-strategies")
async def get_available_strategies():
    """Get list of available options strategies for backtesting"""
    return {
        "basic_strategies": [
            {
                "name": "covered_call",
                "display_name": "Covered Call",
                "description": "Conservative income strategy - sell calls against stock holdings",
                "complexity": "Beginner",
                "market_outlook": "Neutral to slightly bullish"
            },
            {
                "name": "protective_put",
                "display_name": "Protective Put", 
                "description": "Insurance strategy - buy puts to protect stock holdings",
                "complexity": "Beginner",
                "market_outlook": "Bullish but risk-averse"
            },
            {
                "name": "bull_call_spread",
                "display_name": "Bull Call Spread",
                "description": "Limited risk bullish strategy using call options",
                "complexity": "Intermediate",
                "market_outlook": "Moderately bullish"
            },
            {
                "name": "iron_condor",
                "display_name": "Iron Condor",
                "description": "Neutral strategy for range-bound markets",
                "complexity": "Intermediate",
                "market_outlook": "Neutral - low volatility"
            },
            {
                "name": "long_straddle",
                "display_name": "Long Straddle",
                "description": "Volatility strategy - profits from large moves in either direction",
                "complexity": "Intermediate",
                "market_outlook": "High volatility expected"
            }
        ],
        "advanced_strategies": [
            {
                "name": "long_butterfly",
                "display_name": "Long Butterfly",
                "description": "Low volatility strategy - profits when stock stays near center strike",
                "complexity": "Advanced",
                "market_outlook": "Very neutral - low volatility"
            },
            {
                "name": "calendar_spread",
                "display_name": "Calendar Spread",
                "description": "Time decay strategy using different expirations",
                "complexity": "Advanced",
                "market_outlook": "Neutral with time decay benefit"
            },
            {
                "name": "jade_lizard",
                "display_name": "Jade Lizard",
                "description": "High probability income strategy with unlimited downside risk",
                "complexity": "Expert",
                "market_outlook": "Neutral to bullish - income focused"
            }
        ],
        "backtesting_parameters": {
            "recommended_period": "6-12 months",
            "min_capital": 25000,
            "max_capital": 500000,
            "supported_symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "SPY", "QQQ"]
        }
    }


@app.get("/api/backtest-defaults/{strategy}")
async def get_strategy_defaults(strategy: str):
    """Get default parameters for a specific strategy"""
    try:
        backtester = OptionsBacktester()
        defaults = backtester._get_default_strategy_params(strategy)
        
        strategy_info = {
            "covered_call": {
                "description": "Conservative income strategy",
                "typical_holding_period": "30-45 days",
                "success_factors": ["Low volatility", "Sideways to slightly bullish market"]
            },
            "iron_condor": {
                "description": "Neutral income strategy",
                "typical_holding_period": "21-35 days", 
                "success_factors": ["Range-bound market", "Low implied volatility"]
            },
            "long_butterfly": {
                "description": "Low volatility strategy",
                "typical_holding_period": "14-30 days",
                "success_factors": ["Very low volatility", "Price stays near center strike"]
            }
        }
        
        return {
            "strategy": strategy,
            "default_parameters": defaults,
            "strategy_info": strategy_info.get(strategy, {
                "description": "Advanced options strategy",
                "typical_holding_period": "21-45 days",
                "success_factors": ["Market dependent"]
            }),
            "backtesting_notes": [
                "Historical results may not predict future performance",
                "Consider transaction costs and slippage",
                "Test multiple market conditions",
                "Validate with paper trading before live implementation"
            ]
        }
        
    except Exception as e:
        logger.error(f"Strategy defaults error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Bonds Endpoints

class BondRequest(BaseModel):
    symbol: str

class BondAllocationRequest(BaseModel):
    risk_tolerance: str  # conservative, moderate, aggressive
    time_horizon: str   # short, long
    portfolio_size: Optional[float] = 100000


@app.post("/api/bonds")
async def get_bond_data(request: BondRequest):
    """Get bond ETF data"""
    try:
        bond_data = bonds_fetcher.get_bond_data(request.symbol)
        
        if not bond_data:
            raise HTTPException(status_code=404, detail=f"Bond data not found for {request.symbol}")
        
        # Calculate additional metrics
        metrics = bonds_fetcher.calculate_bond_metrics(bond_data)
        
        return {
            "symbol": bond_data.symbol,
            "name": bond_data.name,
            "price": safe_float(bond_data.price),
            "yield_to_maturity": safe_float(bond_data.yield_to_maturity),
            "duration": safe_float(bond_data.duration),
            "bond_type": bond_data.bond_type.value,
            "volume": bond_data.volume,
            "metrics": {
                "modified_duration": safe_float(metrics.get('modified_duration')),
                "duration_risk_1pct": safe_float(metrics.get('duration_risk_1pct')),
                "estimated_volatility": safe_float(metrics.get('estimated_volatility')),
                "interest_rate_risk": metrics.get('interest_rate_risk', 'Unknown')
            },
            "timestamp": bond_data.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Bond data error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bonds/multiple")
async def get_multiple_bonds(symbols: str = "TLT,IEF,LQD,HYG,MUB"):
    """Get multiple bond ETF data"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        bonds_data = bonds_fetcher.get_multiple_bonds(symbol_list)
        
        results = {}
        for symbol, bond_data in bonds_data.items():
            metrics = bonds_fetcher.calculate_bond_metrics(bond_data)
            results[symbol] = {
                "name": bond_data.name,
                "price": safe_float(bond_data.price),
                "yield_to_maturity": safe_float(bond_data.yield_to_maturity),
                "duration": safe_float(bond_data.duration),
                "bond_type": bond_data.bond_type.value,
                "estimated_volatility": safe_float(metrics.get('estimated_volatility')),
                "interest_rate_risk": metrics.get('interest_rate_risk', 'Unknown')
            }
        
        return {
            "bonds": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Multiple bonds error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bonds/yield-curve")
async def get_treasury_yield_curve():
    """Get US Treasury yield curve"""
    try:
        yield_curve = bonds_fetcher.get_treasury_yield_curve()
        curve_analysis = bonds_fetcher.analyze_yield_curve_shape(yield_curve)
        
        curve_data = [
            {
                "maturity": point.maturity,
                "maturity_years": point.maturity_years,
                "yield_rate": safe_float(point.yield_rate)
            }
            for point in yield_curve
        ]
        
        return {
            "yield_curve": curve_data,
            "analysis": curve_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Yield curve error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bonds/allocation")
async def get_bond_allocation(request: BondAllocationRequest):
    """Get bond allocation recommendations"""
    try:
        recommendations = bonds_fetcher.get_bond_allocation_recommendations(
            request.risk_tolerance, 
            request.time_horizon
        )
        
        # Add current prices to recommendations
        for etf, details in recommendations.get('allocation', {}).items():
            bond_data = bonds_fetcher.get_bond_data(etf)
            if bond_data:
                details['current_price'] = safe_float(bond_data.price)
                details['yield'] = safe_float(bond_data.yield_to_maturity)
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Bond allocation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bonds/categories")
async def get_bond_categories():
    """Get bond categories and available ETFs"""
    try:
        categories = bonds_fetcher.get_bond_categories()
        return {
            "categories": categories,
            "total_etfs": sum(len(etfs) for etfs in categories.values()),
            "description": {
                "treasury": "US government bonds - lowest credit risk",
                "corporate": "Corporate bonds - higher yield, higher credit risk",
                "municipal": "Municipal bonds - tax advantages",
                "international": "International bonds - currency and credit risk",
                "etf": "Diversified bond ETFs"
            }
        }
        
    except Exception as e:
        logger.error(f"Bond categories error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Commodities Endpoints

class CommodityRequest(BaseModel):
    symbol: str

class CommodityAllocationRequest(BaseModel):
    portfolio_size: float
    risk_level: str = "moderate"  # conservative, moderate, aggressive


@app.post("/api/commodities")
async def get_commodity_data(request: CommodityRequest):
    """Get commodity ETF data"""
    try:
        commodity_data = commodities_fetcher.get_commodity_data(request.symbol)
        
        if not commodity_data:
            raise HTTPException(status_code=404, detail=f"Commodity data not found for {request.symbol}")
        
        return {
            "symbol": commodity_data.symbol,
            "name": commodity_data.name,
            "price": safe_float(commodity_data.price),
            "currency": commodity_data.currency,
            "unit": commodity_data.unit,
            "category": commodity_data.category.value,
            "change_24h": safe_float(commodity_data.change_24h),
            "change_24h_pct": safe_float(commodity_data.change_24h_pct),
            "volume": commodity_data.volume,
            "high_52w": safe_float(commodity_data.high_52w),
            "low_52w": safe_float(commodity_data.low_52w),
            "volatility": safe_float(commodity_data.volatility),
            "seasonality_factor": commodity_data.seasonality_factor,
            "timestamp": commodity_data.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Commodity data error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/commodities/multiple")
async def get_multiple_commodities(symbols: str = "GLD,USO,CORN,CPER,SLV"):
    """Get multiple commodity ETF data"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        commodities_data = commodities_fetcher.get_multiple_commodities(symbol_list)
        
        results = {}
        for symbol, commodity_data in commodities_data.items():
            results[symbol] = {
                "name": commodity_data.name,
                "price": safe_float(commodity_data.price),
                "category": commodity_data.category.value,
                "change_24h_pct": safe_float(commodity_data.change_24h_pct),
                "volatility": safe_float(commodity_data.volatility),
                "unit": commodity_data.unit,
                "seasonality_factor": commodity_data.seasonality_factor
            }
        
        return {
            "commodities": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Multiple commodities error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/commodities/trends")
async def get_commodity_trends(symbols: Optional[str] = None, period: str = "1y"):
    """Analyze commodity market trends"""
    try:
        symbol_list = symbols.split(",") if symbols else None
        trends_analysis = commodities_fetcher.analyze_commodity_trends(symbol_list, period)
        
        return trends_analysis
        
    except Exception as e:
        logger.error(f"Commodity trends error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/commodities/allocation")
async def get_commodity_allocation(request: CommodityAllocationRequest):
    """Get commodity allocation strategy"""
    try:
        allocation_strategy = commodities_fetcher.get_commodity_allocation_strategy(
            request.portfolio_size,
            request.risk_level
        )
        
        return allocation_strategy
        
    except Exception as e:
        logger.error(f"Commodity allocation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/commodities/categories") 
async def get_commodity_categories():
    """Get commodity categories and available ETFs"""
    try:
        categories = commodities_fetcher.get_commodity_categories()
        return {
            "categories": categories,
            "total_etfs": sum(len(commodities) for commodities in categories.values()),
            "category_descriptions": {
                "energy": "Oil, natural gas, and energy-related commodities",
                "precious_metals": "Gold, silver, platinum, palladium",
                "industrial_metals": "Copper, aluminum, tin, nickel",
                "agriculture": "Corn, soybeans, wheat, and agricultural products",
                "soft_commodities": "Coffee, cocoa, sugar, cotton",
                "livestock": "Live cattle, lean hogs"
            }
        }
        
    except Exception as e:
        logger.error(f"Commodity categories error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Forex Endpoints

class ForexRequest(BaseModel):
    symbol: str

class ForexRecommendationsRequest(BaseModel):
    risk_level: str = "moderate"  # conservative, moderate, aggressive
    trading_style: str = "swing"  # scalp, day, swing, position

class GlobalMarketRequest(BaseModel):
    symbol: str
    market: Optional[str] = None

class GlobalMarketSearchRequest(BaseModel):
    query: str
    limit: int = 10


@app.post("/api/forex")
async def get_forex_pair(request: ForexRequest):
    """Get forex pair data via currency ETF"""
    try:
        forex_data = forex_fetcher.get_forex_pair(request.symbol)
        
        if not forex_data:
            raise HTTPException(status_code=404, detail=f"Forex data not found for {request.symbol}")
        
        return {
            "symbol": forex_data.symbol,
            "name": forex_data.name,
            "base_currency": forex_data.base_currency,
            "quote_currency": forex_data.quote_currency,
            "price": safe_float(forex_data.price),
            "bid": safe_float(forex_data.bid),
            "ask": safe_float(forex_data.ask),
            "spread": safe_float(forex_data.spread),
            "change_24h": safe_float(forex_data.change_24h),
            "change_24h_pct": safe_float(forex_data.change_24h_pct),
            "currency_class": forex_data.currency_class.value,
            "volatility": safe_float(forex_data.volatility),
            "high_52w": safe_float(forex_data.high_52w),
            "low_52w": safe_float(forex_data.low_52w),
            "central_bank_rate": safe_float(forex_data.central_bank_rate),
            "timestamp": forex_data.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Forex data error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forex/multiple")
async def get_multiple_forex_pairs(symbols: str = "FXE,FXY,FXB,FXA,FXC"):
    """Get multiple forex pairs data"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        forex_data = forex_fetcher.get_multiple_forex_pairs(symbol_list)
        
        results = {}
        for symbol, pair_data in forex_data.items():
            results[symbol] = {
                "name": pair_data.name,
                "pair": f"{pair_data.base_currency}/{pair_data.quote_currency}",
                "price": safe_float(pair_data.price),
                "change_24h_pct": safe_float(pair_data.change_24h_pct),
                "volatility": safe_float(pair_data.volatility),
                "currency_class": pair_data.currency_class.value,
                "central_bank_rate": safe_float(pair_data.central_bank_rate)
            }
        
        return {
            "forex_pairs": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Multiple forex error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forex/currency-strength")
async def get_currency_strength(currencies: Optional[str] = None):
    """Analyze currency strength"""
    try:
        currency_list = currencies.split(",") if currencies else None
        strength_analysis = forex_fetcher.get_currency_strength_analysis(currency_list)
        
        return strength_analysis
        
    except Exception as e:
        logger.error(f"Currency strength error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forex/economic-calendar")
async def get_economic_calendar():
    """Get upcoming economic events affecting currencies"""
    try:
        calendar_data = forex_fetcher.get_economic_calendar_impact()
        
        return {
            "economic_events": calendar_data,
            "total_events": sum(len(events) for events in calendar_data.values()) if 'error' not in calendar_data else 0,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Economic calendar error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/forex/recommendations")
async def get_forex_recommendations(request: ForexRecommendationsRequest):
    """Get forex trading recommendations"""
    try:
        recommendations = forex_fetcher.get_forex_trading_recommendations(
            request.risk_level,
            request.trading_style
        )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Forex recommendations error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/asset-classes")
async def get_available_asset_classes():
    """Get all available asset classes and their characteristics"""
    return {
        "asset_classes": {
            "stocks": {
                "description": "Equity securities of public companies",
                "risk_level": "Medium to High",
                "liquidity": "High",
                "examples": ["AAPL", "GOOGL", "MSFT", "TSLA"]
            },
            "bonds": {
                "description": "Fixed-income securities (government and corporate)",
                "risk_level": "Low to Medium", 
                "liquidity": "High",
                "examples": ["TLT", "IEF", "LQD", "HYG", "MUB"]
            },
            "commodities": {
                "description": "Physical goods and raw materials",
                "risk_level": "Medium to High",
                "liquidity": "Medium",
                "examples": ["GLD", "USO", "CORN", "CPER", "SLV"]
            },
            "forex": {
                "description": "Foreign exchange currency pairs",
                "risk_level": "High",
                "liquidity": "Very High",
                "examples": ["FXE", "FXY", "FXB", "FXA", "FXC"]
            },
            "cryptocurrencies": {
                "description": "Digital currencies and tokens",
                "risk_level": "Very High",
                "liquidity": "Medium to High",
                "examples": ["BTC", "ETH", "BNB", "ADA", "SOL"]
            },
            "options": {
                "description": "Derivatives contracts on underlying assets",
                "risk_level": "Medium to Very High",
                "liquidity": "Medium",
                "examples": ["Call options", "Put options", "Spreads", "Straddles"]
            }
        },
        "diversification_benefits": [
            "Different asset classes have varying correlation levels",
            "Bonds often provide stability during stock market volatility",
            "Commodities can hedge against inflation",
            "Forex allows exposure to different economies",
            "Options provide hedging and income generation opportunities"
        ],
        "recommended_allocations": {
            "conservative": {
                "stocks": "30%",
                "bonds": "60%", 
                "commodities": "5%",
                "cash/other": "5%"
            },
            "moderate": {
                "stocks": "60%",
                "bonds": "30%",
                "commodities": "5%",
                "alternatives": "5%"
            },
            "aggressive": {
                "stocks": "80%",
                "bonds": "10%",
                "commodities": "5%",
                "alternatives": "5%"
            }
        }
    }


# Global Markets Endpoints

@app.post("/api/global-markets/stock")
async def get_global_stock_data(request: GlobalMarketRequest):
    """Get stock data from global markets"""
    try:
        stock_data = global_markets_fetcher.get_global_stock_data(
            symbol=request.symbol,
            market=request.market
        )
        
        if not stock_data:
            raise HTTPException(status_code=404, detail=f"Stock data not found for {request.symbol}")
        
        return {
            "symbol": stock_data.symbol,
            "exchange": stock_data.exchange,
            "country": stock_data.country,
            "company_name": stock_data.company_name,
            "currency": stock_data.currency,
            "current_price": safe_float(stock_data.current_price),
            "local_price": safe_float(stock_data.local_price),
            "daily_change": safe_float(stock_data.daily_change),
            "daily_change_pct": safe_float(stock_data.daily_change_pct),
            "volume": stock_data.volume,
            "market_cap": safe_float(stock_data.market_cap),
            "pe_ratio": safe_float(stock_data.pe_ratio),
            "session_status": stock_data.session_status.value,
            "local_time": stock_data.local_time,
            "week_52_high": safe_float(stock_data.week_52_high),
            "week_52_low": safe_float(stock_data.week_52_low),
            "avg_volume": stock_data.avg_volume,
            "dividend_yield": safe_float(stock_data.dividend_yield),
            "last_updated": stock_data.last_updated.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Global stock data error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/global-markets/overview")
async def get_global_market_overview():
    """Get overview of all global markets"""
    try:
        overview = global_markets_fetcher.get_global_market_overview()
        return overview
        
    except Exception as e:
        logger.error(f"Global markets overview error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/global-markets/{market_code}")
async def get_market_info(market_code: str):
    """Get detailed information about a specific market"""
    try:
        market_info = global_markets_fetcher.get_market_info(market_code)
        
        if not market_info:
            raise HTTPException(status_code=404, detail=f"Market not found: {market_code}")
        
        # Get market indices
        indices_data = global_markets_fetcher.get_market_indices(market_code)
        
        # Get sample stocks
        sample_stocks = global_markets_fetcher.get_sample_stocks_by_market(market_code, limit=5)
        
        return {
            "market_info": {
                "exchange": market_info.exchange,
                "country": market_info.country,
                "region": market_info.region.value,
                "currency": market_info.currency,
                "timezone": market_info.timezone,
                "open_time": market_info.open_time,
                "close_time": market_info.close_time,
                "lunch_break": market_info.lunch_break,
                "market_cap_usd_trillions": market_info.market_cap_usd,
                "major_indices": market_info.major_indices
            },
            "indices": indices_data,
            "sample_stocks": [
                {
                    "symbol": stock.symbol,
                    "company_name": stock.company_name,
                    "current_price": safe_float(stock.current_price),
                    "local_price": safe_float(stock.local_price),
                    "daily_change_pct": safe_float(stock.daily_change_pct),
                    "currency": stock.currency
                }
                for stock in sample_stocks
            ]
        }
        
    except Exception as e:
        logger.error(f"Market info error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/global-markets/regions/{region}")
async def get_markets_by_region(region: str):
    """Get markets filtered by region"""
    try:
        # Map string to enum
        region_map = {
            'north_america': MarketRegion.NORTH_AMERICA,
            'europe': MarketRegion.EUROPE,
            'asia_pacific': MarketRegion.ASIA_PACIFIC,
            'emerging_markets': MarketRegion.EMERGING_MARKETS
        }
        
        region_enum = region_map.get(region.lower())
        if not region_enum:
            raise HTTPException(status_code=400, detail=f"Invalid region: {region}")
        
        markets = global_markets_fetcher.get_markets_by_region(region_enum)
        
        result = {}
        for market_code, market_info in markets.items():
            result[market_code] = {
                "exchange": market_info.exchange,
                "country": market_info.country,
                "currency": market_info.currency,
                "market_cap_usd_trillions": market_info.market_cap_usd,
                "timezone": market_info.timezone
            }
        
        return {
            "region": region,
            "markets": result,
            "count": len(result)
        }
        
    except Exception as e:
        logger.error(f"Markets by region error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/global-markets/search")
async def search_global_stocks(request: GlobalMarketSearchRequest):
    """Search for stocks across global markets"""
    try:
        results = global_markets_fetcher.search_global_stocks(
            query=request.query,
            limit=request.limit
        )
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Global stock search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/global-markets")
async def get_all_markets():
    """Get all supported global markets"""
    try:
        all_markets = global_markets_fetcher.get_all_markets()
        
        result = {}
        for market_code, market_info in all_markets.items():
            result[market_code] = {
                "exchange": market_info.exchange,
                "country": market_info.country,
                "region": market_info.region.value,
                "currency": market_info.currency,
                "market_cap_usd_trillions": market_info.market_cap_usd,
                "sample_tickers": market_info.sample_tickers[:3]  # Show first 3 as preview
            }
        
        return {
            "markets": result,
            "total_markets": len(result),
            "regions": list(set(info.region.value for info in all_markets.values())),
            "total_market_cap_usd_trillions": sum(info.market_cap_usd for info in all_markets.values())
        }
        
    except Exception as e:
        logger.error(f"All markets error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Trading Bots Endpoints

class CreateBotRequest(BaseModel):
    name: str
    bot_type: str  # trend_following, sentiment_based, etc.
    symbols: List[str]
    max_positions: int = 3
    max_risk_per_trade: float = 2.0  # Percentage
    initial_capital: float = 10000
    rebalance_frequency: int = 6  # Hours
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 10.0
    strategy_params: Optional[Dict[str, Any]] = None

class BotActionRequest(BaseModel):
    bot_id: str


@app.post("/api/bots/create")
async def create_trading_bot(request: CreateBotRequest):
    """Create a new trading bot"""
    try:
        # Generate unique bot ID
        bot_id = f"bot_{int(time.time())}"
        
        # Map string to enum
        bot_type_map = {
            'trend_following': BotType.TREND_FOLLOWING,
            'sentiment_based': BotType.SENTIMENT_BASED,
            'mean_reversion': BotType.MEAN_REVERSION,
            'momentum': BotType.MOMENTUM,
            'multi_strategy': BotType.MULTI_STRATEGY
        }
        
        bot_type = bot_type_map.get(request.bot_type.lower())
        if not bot_type:
            raise HTTPException(status_code=400, detail=f"Unsupported bot type: {request.bot_type}")
        
        # Create bot configuration
        config = BotConfiguration(
            bot_id=bot_id,
            name=request.name,
            bot_type=bot_type,
            symbols=request.symbols,
            max_positions=request.max_positions,
            max_risk_per_trade=request.max_risk_per_trade,
            initial_capital=request.initial_capital,
            rebalance_frequency=request.rebalance_frequency,
            stop_loss_pct=request.stop_loss_pct,
            take_profit_pct=request.take_profit_pct,
            strategy_params=request.strategy_params or {}
        )
        
        # Add bot to manager
        success = bot_manager.add_bot(config)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create bot")
        
        return {
            "bot_id": bot_id,
            "name": request.name,
            "type": request.bot_type,
            "status": "created",
            "symbols": request.symbols,
            "initial_capital": request.initial_capital,
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating bot: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bots/start")
async def start_bot(request: BotActionRequest):
    """Start a trading bot"""
    try:
        success = bot_manager.start_bot(request.bot_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Bot {request.bot_id} not found")
        
        return {
            "bot_id": request.bot_id,
            "status": "started",
            "started_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting bot: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bots/stop")
async def stop_bot(request: BotActionRequest):
    """Stop a trading bot"""
    try:
        success = bot_manager.stop_bot(request.bot_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Bot {request.bot_id} not found")
        
        return {
            "bot_id": request.bot_id,
            "status": "stopped",
            "stopped_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error stopping bot: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/bots/{bot_id}")
async def delete_bot(bot_id: str):
    """Delete a trading bot"""
    try:
        success = bot_manager.remove_bot(bot_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Bot {bot_id} not found")
        
        return {
            "bot_id": bot_id,
            "status": "deleted",
            "deleted_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error deleting bot: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bots")
async def get_all_bots():
    """Get status of all trading bots"""
    try:
        bots_status = bot_manager.get_all_bots_status()
        
        return {
            "bots": bots_status,
            "total_bots": len(bots_status),
            "active_bots": len([b for b in bots_status.values() if b['status'] == 'running']),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting bots status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bots/{bot_id}")
async def get_bot_details(bot_id: str):
    """Get detailed information about a specific bot"""
    try:
        performance = bot_manager.get_bot_performance(bot_id)
        if not performance:
            raise HTTPException(status_code=404, detail=f"Bot {bot_id} not found")
        
        # Get bot configuration
        bot = bot_manager.bots.get(bot_id)
        if not bot:
            raise HTTPException(status_code=404, detail=f"Bot {bot_id} not found")
        
        return {
            "bot_id": bot_id,
            "configuration": {
                "name": bot.config.name,
                "type": bot.config.bot_type.value,
                "symbols": bot.config.symbols,
                "max_positions": bot.config.max_positions,
                "max_risk_per_trade": bot.config.max_risk_per_trade,
                "initial_capital": bot.config.initial_capital,
                "rebalance_frequency": bot.config.rebalance_frequency,
                "stop_loss_pct": bot.config.stop_loss_pct,
                "take_profit_pct": bot.config.take_profit_pct,
                "enabled": bot.config.enabled
            },
            "status": bot.status.value,
            "performance": {
                "total_trades": performance.total_trades,
                "winning_trades": performance.winning_trades,
                "losing_trades": performance.losing_trades,
                "win_rate": (performance.winning_trades / performance.total_trades * 100) if performance.total_trades > 0 else 0,
                "total_pnl": safe_float(performance.total_pnl),
                "total_return_pct": safe_float(performance.total_return_pct),
                "max_drawdown": safe_float(performance.max_drawdown),
                "sharpe_ratio": safe_float(performance.sharpe_ratio)
            },
            "current_positions": [
                {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "quantity": safe_float(pos.quantity),
                    "entry_price": safe_float(pos.entry_price),
                    "current_price": safe_float(pos.current_price),
                    "unrealized_pnl": safe_float(pos.unrealized_pnl),
                    "entry_time": pos.entry_time.isoformat(),
                    "stop_loss": safe_float(pos.stop_loss),
                    "take_profit": safe_float(pos.take_profit)
                }
                for pos in performance.current_positions
            ],
            "recent_trades": [
                {
                    "timestamp": trade['timestamp'].isoformat(),
                    "symbol": trade['symbol'],
                    "action": trade['action'],
                    "price": safe_float(trade['price']),
                    "quantity": safe_float(trade.get('quantity', 0)),
                    "pnl": safe_float(trade.get('pnl', 0)),
                    "reason": trade.get('reason', '')
                }
                for trade in bot.trades_log[-10:]  # Last 10 trades
            ],
            "last_updated": performance.last_updated.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting bot details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bots/types")
async def get_bot_types():
    """Get available bot types and their descriptions"""
    return {
        "bot_types": {
            "trend_following": {
                "name": "Trend Following",
                "description": "Follows market trends using moving averages and momentum indicators",
                "suitable_for": "Bull and bear markets with clear trends",
                "risk_level": "Medium",
                "recommended_symbols": ["AAPL", "GOOGL", "MSFT", "SPY"],
                "parameters": {
                    "short_window": "Short-term moving average period (default: 20)",
                    "long_window": "Long-term moving average period (default: 50)",
                    "min_confidence": "Minimum confidence threshold (default: 0.7)"
                }
            },
            "sentiment_based": {
                "name": "Sentiment Trading",
                "description": "Makes trading decisions based on news sentiment and social media analysis",
                "suitable_for": "Volatile markets with high news flow",
                "risk_level": "Medium-High",
                "recommended_symbols": ["AAPL", "TSLA", "AMZN", "META"],
                "parameters": {
                    "min_confidence": "Minimum sentiment confidence (default: 0.6)",
                    "sentiment_threshold": "Sentiment score threshold (default: 0.1)"
                }
            },
            "mean_reversion": {
                "name": "Mean Reversion",
                "description": "Buys oversold assets and sells overbought assets",
                "suitable_for": "Range-bound markets with low volatility",
                "risk_level": "Medium",
                "recommended_symbols": ["SPY", "QQQ", "IWM"],
                "parameters": {
                    "rsi_oversold": "RSI oversold threshold (default: 30)",
                    "rsi_overbought": "RSI overbought threshold (default: 70)"
                }
            },
            "momentum": {
                "name": "Momentum Trading",
                "description": "Trades based on price and volume momentum",
                "suitable_for": "High volatility markets with strong moves",
                "risk_level": "High",
                "recommended_symbols": ["QQQ", "ARKK", "SOXL"],
                "parameters": {
                    "momentum_period": "Momentum calculation period (default: 20)",
                    "volume_threshold": "Volume threshold multiplier (default: 1.5)"
                }
            }
        },
        "risk_levels": {
            "Low": "Conservative approach with strict risk controls",
            "Medium": "Balanced risk-reward with moderate position sizes",
            "Medium-High": "More aggressive with higher potential returns",
            "High": "Aggressive approach with higher volatility tolerance"
        },
        "recommended_settings": {
            "conservative": {
                "max_risk_per_trade": 1.0,
                "stop_loss_pct": 3.0,
                "take_profit_pct": 6.0,
                "max_positions": 2
            },
            "moderate": {
                "max_risk_per_trade": 2.0,
                "stop_loss_pct": 5.0,
                "take_profit_pct": 10.0,
                "max_positions": 3
            },
            "aggressive": {
                "max_risk_per_trade": 5.0,
                "stop_loss_pct": 8.0,
                "take_profit_pct": 15.0,
                "max_positions": 5
            }
        }
    }


@app.post("/api/bots/run-cycle")
async def run_bot_cycle():
    """Manually trigger a trading cycle for all active bots"""
    try:
        active_bots = [bot for bot in bot_manager.bots.values() if bot.status == BotStatus.RUNNING]
        
        if not active_bots:
            return {
                "message": "No active bots to run",
                "active_bots": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Run trading cycles
        tasks = [bot.run_trading_cycle() for bot in active_bots]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and errors
        successes = len([r for r in results if not isinstance(r, Exception)])
        errors = len([r for r in results if isinstance(r, Exception)])
        
        return {
            "message": "Trading cycle completed",
            "active_bots": len(active_bots),
            "successful_cycles": successes,
            "failed_cycles": errors,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error running bot cycle: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stock-data-custom")
async def get_stock_data_custom(request: StockRequestCustom):
    """Get stock data with custom interval support"""
    try:
        # Use the custom interval directly
        data = fetcher.get_stock_data(request.symbol, period=request.period, interval=request.interval)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        data_with_indicators = fetcher.get_technical_indicators(data)
        data_with_returns = fetcher.calculate_returns(data_with_indicators)
        
        latest = data_with_returns.iloc[-1]
        current_price = fetcher.get_current_price(request.symbol)
        
        # Enhanced current price handling - prioritize real-time data
        final_current_price = current_price if current_price is not None else latest['Close']
        
        # Better daily change calculation
        daily_change = 0.0
        if len(data_with_returns) >= 2:
            try:
                prev_close = data_with_returns.iloc[-2]['Close']
                if current_price is not None and prev_close != 0:
                    daily_change = ((current_price - prev_close) / prev_close * 100)
                elif prev_close != 0:
                    daily_change = ((latest['Close'] - prev_close) / prev_close * 100)
            except (IndexError, ZeroDivisionError):
                daily_change = 0.0
        
        # Enhanced RSI handling
        rsi_value = latest.get('RSI')
        if pd.isna(rsi_value) or rsi_value is None:
            rsi_value = 50.0
        
        chart_data = []
        # Adjust data points based on interval
        if request.interval in ['1m', '5m']:
            tail_count = 100  # Fewer points for minute data
        elif request.interval in ['15m', '30m', '1h']:
            tail_count = 200  # More points for hourly
        elif request.interval == '1wk':
            tail_count = 300  # Even more for weekly (5 years)
        else:
            tail_count = 250  # Standard for daily
        
        for i, row in data_with_returns.tail(tail_count).iterrows():
            close_price = current_price if (i == data_with_returns.index[-1] and current_price is not None) else row['Close']
            
            # Format timestamp based on interval
            if request.interval in ['1m', '5m', '15m', '30m', '1h']:
                date_str = i.strftime("%Y-%m-%d %H:%M")
            elif request.interval == '1wk':
                date_str = i.strftime("%Y-%m-%d")
            else:
                date_str = i.strftime("%Y-%m-%d")
            
            chart_data.append({
                "date": date_str,
                "close": safe_float(close_price),
                "sma_20": safe_float(row.get('SMA_20')),
                "sma_50": safe_float(row.get('SMA_50')),
                "volume": int(row['Volume']) if pd.notna(row['Volume']) else 0
            })
        
        # Get stock info
        stock_info = fetcher.get_stock_info(request.symbol)
        company_name = stock_info.get('longName', stock_info.get('shortName', request.symbol))
        
        return {
            "symbol": request.symbol,
            "company_name": company_name,
            "current_price": safe_float(final_current_price),
            "latest_close": safe_float(latest['Close']),
            "daily_change": safe_float(daily_change),
            "sma_20": safe_float(latest.get('SMA_20')),
            "sma_50": safe_float(latest.get('SMA_50')),
            "rsi": safe_float(rsi_value),
            "volume": int(latest['Volume']) if pd.notna(latest['Volume']) else 0,
            "chart_data": chart_data,
            "data_quality": {
                "is_real_time": current_price is not None,
                "has_recent_volume": int(latest['Volume']) > 0 if pd.notna(latest['Volume']) else False,
                "data_points": len(data_with_returns),
                "granularity": request.interval,
                "refresh_rate": "5 minutes",
                "security_type": "warrant" if request.symbol.endswith('W') else "stock",
                "last_updated": data_with_returns.index[-1].isoformat(),
                "period": request.period,
                "interval": request.interval
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-quality/{symbol}")
async def get_data_quality_report(symbol: str):
    """Get data quality assessment for a stock symbol"""
    try:
        symbol = symbol.upper()
        report = fetcher.get_data_quality_report(symbol)
        
        return {
            "symbol": symbol,
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Historical Data Collection Endpoints
historical_collector = None
hybrid_pipeline = None

def get_historical_collector():
    """Get or create historical data collector instance"""
    global historical_collector
    if historical_collector is None:
        from src.data.historical_data_collector import HistoricalDataCollector
        historical_collector = HistoricalDataCollector()
    return historical_collector

def get_hybrid_pipeline():
    """Get or create hybrid data pipeline instance"""
    global hybrid_pipeline
    if hybrid_pipeline is None:
        from src.data.hybrid_data_pipeline import HybridDataPipeline
        hybrid_pipeline = HybridDataPipeline()
    return hybrid_pipeline


@app.post("/api/historical-data/collect")
async def start_historical_data_collection(
    years_back: int = 10,
    test_mode: bool = False,
    batch_size: int = 50,
    max_workers: int = 10
):
    """Start historical data collection for NYSE stocks"""
    try:
        collector = get_historical_collector()
        
        # Get symbol list
        symbols = await collector.get_nyse_stock_list()
        
        # In test mode, limit to first 10 symbols
        if test_mode:
            symbols = symbols[:10]
            logger.info(f"Test mode: collecting data for {len(symbols)} symbols")
        
        # Start collection
        result = await collector.collect_historical_data(
            symbols=symbols,
            years_back=years_back,
            batch_size=batch_size,
            max_workers=max_workers
        )
        
        return {
            "status": "completed",
            "result": result,
            "test_mode": test_mode,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in historical data collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/historical-data/summary")
async def get_collection_summary():
    """Get summary of collected historical data"""
    try:
        collector = get_historical_collector()
        summary = collector.generate_collection_summary()
        
        return {
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/historical-data/symbols")
async def get_available_symbols(security_type: str = None, sector: str = None):
    """Get list of symbols with available historical data"""
    try:
        collector = get_historical_collector()
        symbols = collector.get_available_symbols(security_type=security_type, sector=sector)
        
        return {
            "symbols": symbols,
            "count": len(symbols),
            "filters": {
                "security_type": security_type,
                "sector": sector
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/historical-data/{symbol}")
async def get_historical_data(
    symbol: str, 
    start_date: str = None, 
    end_date: str = None
):
    """Get historical data for a specific symbol"""
    try:
        collector = get_historical_collector()
        symbol = symbol.upper()
        
        data = collector.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if data.empty:
            return {
                "symbol": symbol,
                "data": [],
                "message": "No historical data available for this symbol",
                "timestamp": datetime.now().isoformat()
            }
        
        # Convert to JSON-friendly format
        data_records = []
        for idx, row in data.iterrows():
            data_records.append({
                "date": idx.strftime('%Y-%m-%d'),
                "open": float(row.get('Open', 0)),
                "high": float(row.get('High', 0)),
                "low": float(row.get('Low', 0)),
                "close": float(row.get('Close', 0)),
                "adj_close": float(row.get('Adj Close', 0)),
                "volume": int(row.get('Volume', 0)),
                "dividends": float(row.get('Dividends', 0)),
                "stock_splits": float(row.get('Stock Splits', 0))
            })
        
        return {
            "symbol": symbol,
            "data": data_records,
            "count": len(data_records),
            "date_range": {
                "start": data.index[0].strftime('%Y-%m-%d') if len(data) > 0 else None,
                "end": data.index[-1].strftime('%Y-%m-%d') if len(data) > 0 else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stock-data-hybrid")
async def get_stock_data_hybrid(request: StockRequestCustom):
    """Get stock data using hybrid pipeline with intelligent source selection"""
    try:
        pipeline = get_hybrid_pipeline()
        
        # Convert period to date range
        end_date = datetime.now()
        
        period_map = {
            "1d": 1,
            "5d": 5, 
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825
        }
        
        days_back = period_map.get(request.period, 180)
        start_date = end_date - timedelta(days=days_back)
        
        # Create pipeline request
        from src.data.hybrid_data_pipeline import DataRequest
        data_request = DataRequest(
            symbol=request.symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval=request.interval,
            include_validation=False  # Disable cross-validation for speed (can enable later)
        )
        
        # Get data using hybrid pipeline
        response = await pipeline.get_data(data_request)
        
        if response.data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Calculate technical indicators using existing fetcher logic
        data_with_indicators = fetcher.get_technical_indicators(response.data)
        data_with_returns = fetcher.calculate_returns(data_with_indicators)
        
        latest = data_with_returns.iloc[-1]
        
        # Create chart data
        chart_data = []
        for idx, row in data_with_returns.iterrows():
            chart_data.append({
                "date": idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'hour') else idx.strftime('%Y-%m-%d'),
                "close": float(row['close']),
                "volume": int(row.get('volume', 0)),
                "sma_20": float(row.get('sma_20', 0)) if pd.notna(row.get('sma_20', 0)) else None,
                "sma_50": float(row.get('sma_50', 0)) if pd.notna(row.get('sma_50', 0)) else None
            })
        
        # Enhanced data quality information
        data_quality = {
            "source": response.source,
            "quality_score": response.quality_score,
            "is_real_time": response.source in ["yahoo", "alpha_vantage"],
            "granularity": "hourly" if request.interval in ["1h", "5m", "15m", "30m"] else "daily", 
            "data_points": len(response.data),
            "refresh_rate": "5 minutes" if response.quality_score > 90 else "30 minutes",
            "security_type": "warrant" if request.symbol.endswith('W') else "stock",
            "has_recent_volume": latest.get('volume', 0) > 0,
            "validation_available": response.validation_result is not None
        }
        
        # Add validation results if available
        if response.validation_result:
            data_quality["validation"] = {
                "sources_compared": response.validation_result.sources_compared,
                "price_correlations": response.validation_result.price_correlation,
                "recommended_source": response.validation_result.recommended_source,
                "confidence": response.validation_result.confidence
            }
        
        return {
            "symbol": request.symbol,
            "current_price": float(latest['close']),
            "daily_change": float(latest.get('daily_return', 0) * 100),
            "volume": int(latest.get('volume', 0)),
            "sma_20": float(latest.get('sma_20', 0)) if pd.notna(latest.get('sma_20', 0)) else None,
            "sma_50": float(latest.get('sma_50', 0)) if pd.notna(latest.get('sma_50', 0)) else None,
            "rsi": float(latest.get('rsi', 50)),
            "chart_data": chart_data,
            "data_quality": data_quality,
            "metadata": {
                "pipeline_stats": pipeline.get_pipeline_stats(),
                "request_params": {
                    "period": request.period,
                    "interval": request.interval
                },
                "last_updated": response.timestamp
            }
        }
        
    except Exception as e:
        logger.error(f"Error in hybrid stock data endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)