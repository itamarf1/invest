import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import time
from abc import ABC, abstractmethod

from src.data.market import MarketDataFetcher
from src.trading.paper_trader import PaperTrader, OrderSide, OrderType
from src.trading.strategies.moving_average import MovingAverageStrategy
from src.trading.strategies.rsi_strategy import RSIStrategy
from src.trading.strategies.sentiment_enhanced import SentimentEnhancedStrategy
from src.risk.risk_management import RiskCalculator, PositionSizer
from src.analysis.multi_timeframe import MultiTimeframeAnalyzer
from src.data.enhanced_social_sentiment import EnhancedSocialSentimentAnalyzer

logger = logging.getLogger(__name__)


class BotStatus(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class BotType(Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    SENTIMENT_BASED = "sentiment_based"
    MULTI_STRATEGY = "multi_strategy"
    RISK_PARITY = "risk_parity"
    MOMENTUM = "momentum"
    CONTRARIAN = "contrarian"


@dataclass
class BotConfiguration:
    bot_id: str
    name: str
    bot_type: BotType
    symbols: List[str]
    max_positions: int
    max_risk_per_trade: float  # Percentage
    initial_capital: float
    rebalance_frequency: int  # Hours
    stop_loss_pct: float
    take_profit_pct: float
    enabled: bool = True
    
    # Strategy-specific parameters
    strategy_params: Dict[str, Any] = None
    
    # Risk management
    max_drawdown_pct: float = 20.0
    daily_loss_limit: float = 2.0  # Percentage
    position_size_method: str = "fixed"  # "fixed", "kelly", "volatility"
    
    def __post_init__(self):
        if self.strategy_params is None:
            self.strategy_params = {}


@dataclass
class BotPosition:
    symbol: str
    side: str  # "long" or "short"
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class BotPerformance:
    bot_id: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_return_pct: float
    max_drawdown: float
    sharpe_ratio: float
    current_positions: List[BotPosition]
    daily_returns: List[float]
    last_updated: datetime


class TradingBot(ABC):
    """Base class for all trading bots"""
    
    def __init__(self, config: BotConfiguration):
        self.config = config
        self.status = BotStatus.STOPPED
        self.paper_trader = PaperTrader()
        self.market_data = MarketDataFetcher()
        self.risk_calculator = RiskCalculator()
        self.position_sizer = PositionSizer(config.initial_capital, config.max_risk_per_trade)
        self.multi_timeframe = MultiTimeframeAnalyzer()
        
        # Performance tracking
        self.trades_log = []
        self.positions = {}
        self.daily_pnl = []
        self.last_rebalance = None
        self.start_time = None
        self.current_equity = config.initial_capital
        
        logger.info(f"Trading bot {config.name} ({config.bot_type.value}) initialized")
    
    @abstractmethod
    def analyze_market(self, symbol: str) -> Dict[str, Any]:
        """Analyze market conditions for a symbol"""
        pass
    
    @abstractmethod
    def generate_signals(self, symbol: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on analysis"""
        pass
    
    def should_rebalance(self) -> bool:
        """Check if it's time to rebalance"""
        if not self.last_rebalance:
            return True
        
        hours_since_rebalance = (datetime.now() - self.last_rebalance).total_seconds() / 3600
        return hours_since_rebalance >= self.config.rebalance_frequency
    
    def check_risk_limits(self) -> bool:
        """Check if risk limits are breached"""
        try:
            # Check daily loss limit
            if self.daily_pnl:
                daily_loss_pct = (self.daily_pnl[-1] / self.config.initial_capital) * 100
                if daily_loss_pct <= -self.config.daily_loss_limit:
                    logger.warning(f"Bot {self.config.bot_id} hit daily loss limit: {daily_loss_pct:.2f}%")
                    return False
            
            # Check maximum drawdown
            if len(self.daily_pnl) > 1:
                peak = max(self.daily_pnl)
                current = self.daily_pnl[-1]
                drawdown_pct = ((current - peak) / peak) * 100
                if abs(drawdown_pct) >= self.config.max_drawdown_pct:
                    logger.warning(f"Bot {self.config.bot_id} hit max drawdown limit: {drawdown_pct:.2f}%")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
            return False
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """Calculate appropriate position size"""
        try:
            if self.config.position_size_method == "fixed":
                return self.config.initial_capital * (self.config.max_risk_per_trade / 100) / entry_price
            
            elif self.config.position_size_method == "kelly":
                # Kelly criterion-based sizing
                sizing = self.position_sizer.calculate_risk_based_position_size(
                    symbol, entry_price, stop_loss
                )
                return sizing.kelly_criterion
            
            elif self.config.position_size_method == "volatility":
                # Volatility-based sizing
                try:
                    risk_metrics = self.risk_calculator.calculate_risk_metrics(symbol)
                    volatility_adj = min(1.0, 0.2 / risk_metrics.volatility)  # Inverse volatility scaling
                    base_size = self.config.initial_capital * (self.config.max_risk_per_trade / 100) / entry_price
                    return base_size * volatility_adj
                except:
                    return self.config.initial_capital * (self.config.max_risk_per_trade / 100) / entry_price
            
            else:
                return self.config.initial_capital * (self.config.max_risk_per_trade / 100) / entry_price
                
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0
    
    def execute_trade(self, symbol: str, signal: Dict[str, Any]) -> bool:
        """Execute a trade based on signal"""
        try:
            if signal['action'] == 'none':
                return False
            
            current_data = self.market_data.get_stock_data(symbol)
            if not current_data or 'Close' not in current_data or current_data['Close'].empty:
                logger.warning(f"No price data available for {symbol}")
                return False
            
            current_price = float(current_data['Close'].iloc[-1])
            
            # Calculate stop loss and take profit levels
            if signal['action'] == 'buy':
                stop_loss = current_price * (1 - self.config.stop_loss_pct / 100)
                take_profit = current_price * (1 + self.config.take_profit_pct / 100)
                side = OrderSide.BUY
            else:  # sell
                stop_loss = current_price * (1 + self.config.stop_loss_pct / 100)
                take_profit = current_price * (1 - self.config.take_profit_pct / 100)
                side = OrderSide.SELL
            
            # Calculate position size
            quantity = self.calculate_position_size(symbol, current_price, stop_loss)
            
            if quantity <= 0:
                logger.warning(f"Invalid position size calculated for {symbol}: {quantity}")
                return False
            
            # Execute the trade
            order_result = self.paper_trader.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity
            )
            
            if order_result:
                # Record the position
                position = BotPosition(
                    symbol=symbol,
                    side='long' if side == OrderSide.BUY else 'short',
                    quantity=quantity,
                    entry_price=current_price,
                    entry_time=datetime.now(),
                    current_price=current_price,
                    unrealized_pnl=0.0,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                self.positions[symbol] = position
                
                # Log the trade
                self.trades_log.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': signal['action'],
                    'price': current_price,
                    'quantity': quantity,
                    'confidence': signal.get('confidence', 0),
                    'reason': signal.get('reason', 'Signal generated')
                })
                
                logger.info(f"Bot {self.config.bot_id} executed {signal['action']} for {symbol} at ${current_price:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {str(e)}")
            return False
    
    def update_positions(self):
        """Update current positions with latest prices and P&L"""
        try:
            for symbol, position in self.positions.items():
                current_data = self.market_data.get_stock_data(symbol)
                if current_data and 'Close' in current_data and not current_data['Close'].empty:
                    current_price = float(current_data['Close'].iloc[-1])
                    position.current_price = current_price
                    
                    if position.side == 'long':
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    else:
                        position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                    
                    # Check stop loss and take profit
                    should_close = False
                    close_reason = ""
                    
                    if position.side == 'long':
                        if position.stop_loss and current_price <= position.stop_loss:
                            should_close = True
                            close_reason = "stop_loss"
                        elif position.take_profit and current_price >= position.take_profit:
                            should_close = True
                            close_reason = "take_profit"
                    else:  # short
                        if position.stop_loss and current_price >= position.stop_loss:
                            should_close = True
                            close_reason = "stop_loss"
                        elif position.take_profit and current_price <= position.take_profit:
                            should_close = True
                            close_reason = "take_profit"
                    
                    if should_close:
                        self.close_position(symbol, close_reason)
                        
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
    
    def close_position(self, symbol: str, reason: str = "manual"):
        """Close a position"""
        try:
            if symbol not in self.positions:
                return False
            
            position = self.positions[symbol]
            
            # Execute closing trade
            side = OrderSide.SELL if position.side == 'long' else OrderSide.BUY
            order_result = self.paper_trader.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=position.quantity
            )
            
            if order_result:
                # Log the closing trade
                self.trades_log.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'close',
                    'price': position.current_price,
                    'quantity': position.quantity,
                    'pnl': position.unrealized_pnl,
                    'reason': reason
                })
                
                # Update equity
                self.current_equity += position.unrealized_pnl
                
                # Remove position
                del self.positions[symbol]
                
                logger.info(f"Bot {self.config.bot_id} closed {symbol} position: P&L ${position.unrealized_pnl:.2f} ({reason})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {str(e)}")
            return False
    
    def get_performance_metrics(self) -> BotPerformance:
        """Calculate and return performance metrics"""
        try:
            total_trades = len([t for t in self.trades_log if t['action'] in ['buy', 'sell']])
            winning_trades = len([t for t in self.trades_log if t.get('pnl', 0) > 0])
            losing_trades = len([t for t in self.trades_log if t.get('pnl', 0) < 0])
            
            total_pnl = sum(position.unrealized_pnl for position in self.positions.values())
            total_pnl += sum(t.get('pnl', 0) for t in self.trades_log if 'pnl' in t)
            
            total_return_pct = (total_pnl / self.config.initial_capital) * 100
            
            # Calculate max drawdown
            equity_curve = [self.config.initial_capital]
            running_pnl = 0
            for trade in self.trades_log:
                if 'pnl' in trade:
                    running_pnl += trade['pnl']
                    equity_curve.append(self.config.initial_capital + running_pnl)
            
            if len(equity_curve) > 1:
                peak = max(equity_curve)
                trough = min(equity_curve[equity_curve.index(peak):])
                max_drawdown = ((trough - peak) / peak) * 100
            else:
                max_drawdown = 0
            
            # Simple Sharpe ratio calculation
            if len(self.daily_pnl) > 1:
                returns = np.diff(self.daily_pnl) / self.config.initial_capital
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            return BotPerformance(
                bot_id=self.config.bot_id,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                total_pnl=total_pnl,
                total_return_pct=total_return_pct,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                current_positions=list(self.positions.values()),
                daily_returns=self.daily_pnl,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return BotPerformance(
                bot_id=self.config.bot_id,
                total_trades=0, winning_trades=0, losing_trades=0,
                total_pnl=0, total_return_pct=0, max_drawdown=0,
                sharpe_ratio=0, current_positions=[], daily_returns=[],
                last_updated=datetime.now()
            )
    
    async def run_trading_cycle(self):
        """Execute one trading cycle"""
        try:
            if not self.config.enabled or not self.check_risk_limits():
                return
            
            # Update existing positions
            self.update_positions()
            
            # Check if it's time to rebalance
            if not self.should_rebalance():
                return
            
            # Analyze each symbol and generate signals
            for symbol in self.config.symbols:
                try:
                    # Skip if we already have a position (unless it's multi-strategy)
                    if symbol in self.positions and self.config.bot_type != BotType.MULTI_STRATEGY:
                        continue
                    
                    # Skip if we've reached max positions
                    if len(self.positions) >= self.config.max_positions:
                        break
                    
                    # Analyze market
                    analysis = self.analyze_market(symbol)
                    if not analysis:
                        continue
                    
                    # Generate signals
                    signal = self.generate_signals(symbol, analysis)
                    if not signal or signal.get('action') == 'none':
                        continue
                    
                    # Execute trade if signal is strong enough
                    min_confidence = self.config.strategy_params.get('min_confidence', 0.6)
                    if signal.get('confidence', 0) >= min_confidence:
                        self.execute_trade(symbol, signal)
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    continue
            
            self.last_rebalance = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")
            self.status = BotStatus.ERROR


class TrendFollowingBot(TradingBot):
    """Trend following trading bot using moving averages and momentum"""
    
    def __init__(self, config: BotConfiguration):
        super().__init__(config)
        self.ma_strategy = MovingAverageStrategy()
        
        # Default parameters
        self.config.strategy_params.setdefault('short_window', 20)
        self.config.strategy_params.setdefault('long_window', 50)
        self.config.strategy_params.setdefault('min_confidence', 0.7)
    
    def analyze_market(self, symbol: str) -> Dict[str, Any]:
        """Analyze market using trend indicators"""
        try:
            # Get price data
            data = self.market_data.get_stock_data(symbol, period="6mo")
            if not data or data.empty:
                return {}
            
            # Calculate moving averages
            short_ma = data['Close'].rolling(self.config.strategy_params['short_window']).mean()
            long_ma = data['Close'].rolling(self.config.strategy_params['long_window']).mean()
            
            # Calculate momentum
            momentum = data['Close'].pct_change(20)  # 20-day momentum
            
            # Multi-timeframe analysis
            mtf_analysis = self.multi_timeframe.multi_timeframe_analysis(symbol, ['1d', '1wk'])
            
            return {
                'current_price': float(data['Close'].iloc[-1]),
                'short_ma': float(short_ma.iloc[-1]) if not pd.isna(short_ma.iloc[-1]) else None,
                'long_ma': float(long_ma.iloc[-1]) if not pd.isna(long_ma.iloc[-1]) else None,
                'momentum': float(momentum.iloc[-1]) if not pd.isna(momentum.iloc[-1]) else 0,
                'volume': float(data['Volume'].iloc[-1]),
                'mtf_consensus': mtf_analysis.consensus_signal if mtf_analysis else 'neutral',
                'mtf_confidence': mtf_analysis.consensus_confidence if mtf_analysis else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return {}
    
    def generate_signals(self, symbol: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trend-following signals"""
        try:
            if not analysis.get('short_ma') or not analysis.get('long_ma'):
                return {'action': 'none', 'confidence': 0}
            
            current_price = analysis['current_price']
            short_ma = analysis['short_ma']
            long_ma = analysis['long_ma']
            momentum = analysis.get('momentum', 0)
            mtf_consensus = analysis.get('mtf_consensus', 'neutral')
            mtf_confidence = analysis.get('mtf_confidence', 0)
            
            # Trend following logic
            signals = []
            confidence_factors = []
            
            # Moving average crossover
            if short_ma > long_ma:
                signals.append('buy')
                ma_strength = (short_ma - long_ma) / long_ma
                confidence_factors.append(min(1.0, ma_strength * 10))
            elif short_ma < long_ma:
                signals.append('sell')
                ma_strength = (long_ma - short_ma) / long_ma
                confidence_factors.append(min(1.0, ma_strength * 10))
            
            # Price above/below moving averages
            if current_price > short_ma > long_ma:
                signals.append('buy')
                confidence_factors.append(0.8)
            elif current_price < short_ma < long_ma:
                signals.append('sell')
                confidence_factors.append(0.8)
            
            # Momentum confirmation
            if momentum > 0.02:  # 2% positive momentum
                signals.append('buy')
                confidence_factors.append(min(1.0, momentum * 10))
            elif momentum < -0.02:  # 2% negative momentum
                signals.append('sell')
                confidence_factors.append(min(1.0, abs(momentum) * 10))
            
            # Multi-timeframe confirmation
            if mtf_consensus == 'bullish' and mtf_confidence > 0.6:
                signals.append('buy')
                confidence_factors.append(mtf_confidence)
            elif mtf_consensus == 'bearish' and mtf_confidence > 0.6:
                signals.append('sell')
                confidence_factors.append(mtf_confidence)
            
            # Determine final signal
            if not signals:
                return {'action': 'none', 'confidence': 0}
            
            # Count signal votes
            buy_votes = signals.count('buy')
            sell_votes = signals.count('sell')
            
            if buy_votes > sell_votes:
                action = 'buy'
                confidence = np.mean(confidence_factors) if confidence_factors else 0
            elif sell_votes > buy_votes:
                action = 'sell'
                confidence = np.mean(confidence_factors) if confidence_factors else 0
            else:
                action = 'none'
                confidence = 0
            
            return {
                'action': action,
                'confidence': confidence,
                'reason': f"Trend following: MA({short_ma:.2f}/{long_ma:.2f}), Momentum({momentum:.3f}), MTF({mtf_consensus})"
            }
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return {'action': 'none', 'confidence': 0}


class SentimentBot(TradingBot):
    """Sentiment-based trading bot using news and social media sentiment"""
    
    def __init__(self, config: BotConfiguration):
        super().__init__(config)
        self.sentiment_analyzer = EnhancedSocialSentimentAnalyzer()
        self.sentiment_strategy = SentimentEnhancedStrategy()
        
        # Default parameters
        self.config.strategy_params.setdefault('min_confidence', 0.6)
        self.config.strategy_params.setdefault('sentiment_threshold', 0.1)
    
    def analyze_market(self, symbol: str) -> Dict[str, Any]:
        """Analyze market using sentiment data"""
        try:
            # Get sentiment analysis
            sentiment_data = self.sentiment_analyzer.analyze_stock_sentiment(symbol)
            
            # Get price data for momentum
            price_data = self.market_data.get_stock_data(symbol, period="1mo")
            current_price = float(price_data['Close'].iloc[-1]) if not price_data.empty else 0
            
            # Calculate price momentum
            momentum = price_data['Close'].pct_change(5).iloc[-1] if len(price_data) > 5 else 0
            
            return {
                'current_price': current_price,
                'overall_sentiment': sentiment_data.get('overall_assessment', {}).get('sentiment_score', 0),
                'sentiment_confidence': sentiment_data.get('overall_assessment', {}).get('confidence', 0),
                'sentiment_label': sentiment_data.get('overall_assessment', {}).get('sentiment_label', 'Neutral'),
                'news_sentiment': sentiment_data.get('news_sentiment', {}).get('overall_score', 0),
                'article_count': sentiment_data.get('news_sentiment', {}).get('article_count', 0),
                'trading_signal': sentiment_data.get('trading_implications', {}).get('primary_signal', 'hold'),
                'momentum': float(momentum) if not pd.isna(momentum) else 0,
                'market_context': sentiment_data.get('market_context', {})
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
            return {}
    
    def generate_signals(self, symbol: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sentiment-based trading signals"""
        try:
            sentiment_score = analysis.get('overall_sentiment', 0)
            sentiment_confidence = analysis.get('sentiment_confidence', 0)
            trading_signal = analysis.get('trading_signal', 'hold')
            news_sentiment = analysis.get('news_sentiment', 0)
            article_count = analysis.get('article_count', 0)
            momentum = analysis.get('momentum', 0)
            
            threshold = self.config.strategy_params['sentiment_threshold']
            min_confidence = self.config.strategy_params['min_confidence']
            
            # Sentiment-based signals
            signals = []
            confidence_factors = []
            
            # Overall sentiment signal
            if sentiment_score > threshold and sentiment_confidence > min_confidence:
                signals.append('buy')
                confidence_factors.append(sentiment_confidence)
            elif sentiment_score < -threshold and sentiment_confidence > min_confidence:
                signals.append('sell')
                confidence_factors.append(sentiment_confidence)
            
            # Trading implications signal
            if trading_signal == 'bullish':
                signals.append('buy')
                confidence_factors.append(0.7)
            elif trading_signal == 'bearish':
                signals.append('sell')
                confidence_factors.append(0.7)
            
            # News sentiment confirmation
            if news_sentiment > threshold and article_count >= 3:
                signals.append('buy')
                confidence_factors.append(min(1.0, article_count / 10))
            elif news_sentiment < -threshold and article_count >= 3:
                signals.append('sell')
                confidence_factors.append(min(1.0, article_count / 10))
            
            # Momentum confirmation
            if abs(momentum) > 0.01:  # 1% momentum threshold
                if momentum > 0 and sentiment_score > 0:
                    signals.append('buy')
                    confidence_factors.append(0.6)
                elif momentum < 0 and sentiment_score < 0:
                    signals.append('sell')
                    confidence_factors.append(0.6)
            
            # Determine final signal
            if not signals:
                return {'action': 'none', 'confidence': 0}
            
            buy_votes = signals.count('buy')
            sell_votes = signals.count('sell')
            
            if buy_votes > sell_votes:
                action = 'buy'
                confidence = np.mean(confidence_factors)
            elif sell_votes > buy_votes:
                action = 'sell'
                confidence = np.mean(confidence_factors)
            else:
                action = 'none'
                confidence = 0
            
            return {
                'action': action,
                'confidence': confidence,
                'reason': f"Sentiment: {sentiment_score:.3f}, Signal: {trading_signal}, Articles: {article_count}"
            }
            
        except Exception as e:
            logger.error(f"Error generating sentiment signals for {symbol}: {str(e)}")
            return {'action': 'none', 'confidence': 0}


class BotManager:
    """Manages multiple trading bots"""
    
    def __init__(self):
        self.bots: Dict[str, TradingBot] = {}
        self.running = False
        self.update_interval = 300  # 5 minutes
    
    def add_bot(self, config: BotConfiguration) -> bool:
        """Add a new trading bot"""
        try:
            if config.bot_type == BotType.TREND_FOLLOWING:
                bot = TrendFollowingBot(config)
            elif config.bot_type == BotType.SENTIMENT_BASED:
                bot = SentimentBot(config)
            else:
                logger.error(f"Unsupported bot type: {config.bot_type}")
                return False
            
            self.bots[config.bot_id] = bot
            logger.info(f"Added bot: {config.name} ({config.bot_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding bot: {str(e)}")
            return False
    
    def remove_bot(self, bot_id: str) -> bool:
        """Remove a trading bot"""
        try:
            if bot_id in self.bots:
                bot = self.bots[bot_id]
                bot.status = BotStatus.STOPPED
                
                # Close all positions
                for symbol in list(bot.positions.keys()):
                    bot.close_position(symbol, "bot_removal")
                
                del self.bots[bot_id]
                logger.info(f"Removed bot: {bot_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing bot: {str(e)}")
            return False
    
    def start_bot(self, bot_id: str) -> bool:
        """Start a specific bot"""
        try:
            if bot_id in self.bots:
                self.bots[bot_id].status = BotStatus.RUNNING
                self.bots[bot_id].start_time = datetime.now()
                logger.info(f"Started bot: {bot_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error starting bot: {str(e)}")
            return False
    
    def stop_bot(self, bot_id: str) -> bool:
        """Stop a specific bot"""
        try:
            if bot_id in self.bots:
                self.bots[bot_id].status = BotStatus.STOPPED
                logger.info(f"Stopped bot: {bot_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error stopping bot: {str(e)}")
            return False
    
    def get_bot_performance(self, bot_id: str) -> Optional[BotPerformance]:
        """Get performance metrics for a bot"""
        try:
            if bot_id in self.bots:
                return self.bots[bot_id].get_performance_metrics()
            return None
        except Exception as e:
            logger.error(f"Error getting bot performance: {str(e)}")
            return None
    
    def get_all_bots_status(self) -> Dict[str, Any]:
        """Get status of all bots"""
        try:
            bots_status = {}
            for bot_id, bot in self.bots.items():
                performance = bot.get_performance_metrics()
                bots_status[bot_id] = {
                    'name': bot.config.name,
                    'type': bot.config.bot_type.value,
                    'status': bot.status.value,
                    'symbols': bot.config.symbols,
                    'positions': len(bot.positions),
                    'total_return_pct': performance.total_return_pct,
                    'total_trades': performance.total_trades,
                    'win_rate': (performance.winning_trades / performance.total_trades * 100) if performance.total_trades > 0 else 0,
                    'last_updated': performance.last_updated.isoformat()
                }
            
            return bots_status
        except Exception as e:
            logger.error(f"Error getting bots status: {str(e)}")
            return {}
    
    async def run_all_bots(self):
        """Run all active bots"""
        try:
            self.running = True
            logger.info("Starting bot manager...")
            
            while self.running:
                # Run trading cycles for all active bots
                tasks = []
                for bot_id, bot in self.bots.items():
                    if bot.status == BotStatus.RUNNING:
                        tasks.append(bot.run_trading_cycle())
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Wait for next cycle
                await asyncio.sleep(self.update_interval)
                
        except Exception as e:
            logger.error(f"Error in bot manager main loop: {str(e)}")
            self.running = False
    
    def stop_all_bots(self):
        """Stop all bots and the manager"""
        try:
            self.running = False
            for bot_id, bot in self.bots.items():
                bot.status = BotStatus.STOPPED
            logger.info("Stopped all bots")
        except Exception as e:
            logger.error(f"Error stopping all bots: {str(e)}")


