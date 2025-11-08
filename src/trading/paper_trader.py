import alpaca_trade_api as tradeapi
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    DONE_FOR_DAY = "done_for_day"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REPLACED = "replaced"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    ACCEPTED = "accepted"
    PENDING_NEW = "pending_new"
    ACCEPTED_FOR_BIDDING = "accepted_for_bidding"
    STOPPED = "stopped"
    REJECTED = "rejected"
    SUSPENDED = "suspended"
    CALCULATED = "calculated"

class PaperTrader:
    """Paper trading integration with Alpaca"""
    
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.paper_trading = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'
        
        # Risk management settings
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', 0.05))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', 0.02))
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', 0.05))
        self.min_cash_balance = float(os.getenv('MIN_CASH_BALANCE', 1000))
        
        self.api = None
        self._initialize_api()
    
    def _initialize_api(self):
        """Initialize Alpaca API connection"""
        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca API credentials not found. Paper trading will be simulated.")
            return
        
        try:
            base_url = 'https://paper-api.alpaca.markets' if self.paper_trading else 'https://api.alpaca.markets'
            
            self.api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=base_url,
                api_version='v2'
            )
            
            # Test connection
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca {'Paper' if self.paper_trading else 'Live'} Trading")
            logger.info(f"Account Status: {account.status}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {str(e)}")
            self.api = None
    
    def is_connected(self) -> bool:
        """Check if connected to Alpaca API"""
        return self.api is not None
    
    def get_account_info(self) -> Dict:
        """Get account information and buying power"""
        if not self.is_connected():
            return self._get_simulated_account()
        
        try:
            account = self.api.get_account()
            
            return {
                'account_id': account.id,
                'status': account.status,
                'currency': account.currency,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'day_trade_buying_power': float(account.day_trade_buying_power),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'daytrade_count': account.daytrade_count,
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked,
                'created_at': account.created_at
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        if not self.is_connected():
            return []
        
        try:
            positions = self.api.list_positions()
            
            position_data = []
            for position in positions:
                position_data.append({
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'side': 'long' if float(position.qty) > 0 else 'short',
                    'market_value': float(position.market_value),
                    'avg_entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price) if position.current_price else 0,
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'cost_basis': float(position.cost_basis),
                    'change_today': float(position.change_today) if position.change_today else 0
                })
            
            return position_data
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def submit_order(self, symbol: str, qty: int, side: OrderSide, 
                    order_type: OrderType = OrderType.MARKET, 
                    time_in_force: str = 'day', 
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None) -> Optional[Dict]:
        """Submit an order"""
        if not self.is_connected():
            return self._simulate_order(symbol, qty, side, order_type, limit_price, stop_price)
        
        try:
            # Risk check before placing order
            if not self._risk_check_order(symbol, qty, side):
                logger.warning(f"Order for {symbol} failed risk check")
                return None
            
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side.value,
                type=order_type.value,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price
            )
            
            order_data = {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'filled_qty': float(order.filled_qty),
                'side': order.side,
                'order_type': order.order_type,
                'time_in_force': order.time_in_force,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
                'status': order.status,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'avg_fill_price': float(order.filled_avg_price) if order.filled_avg_price else None
            }
            
            logger.info(f"Order submitted: {side.value.upper()} {qty} {symbol} - Order ID: {order.id}")
            
            return order_data
            
        except Exception as e:
            logger.error(f"Error submitting order: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.is_connected():
            logger.info(f"Simulated: Cancel order {order_id}")
            return True
        
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False
    
    def get_orders(self, status: Optional[str] = None, 
                  limit: int = 50, nested: bool = True) -> List[Dict]:
        """Get orders"""
        if not self.is_connected():
            return []
        
        try:
            orders = self.api.list_orders(
                status=status,
                limit=limit,
                nested=nested
            )
            
            order_data = []
            for order in orders:
                order_data.append({
                    'id': order.id,
                    'symbol': order.symbol,
                    'qty': float(order.qty),
                    'filled_qty': float(order.filled_qty),
                    'side': order.side,
                    'order_type': order.order_type,
                    'time_in_force': order.time_in_force,
                    'limit_price': float(order.limit_price) if order.limit_price else None,
                    'stop_price': float(order.stop_price) if order.stop_price else None,
                    'status': order.status,
                    'submitted_at': order.submitted_at,
                    'filled_at': order.filled_at,
                    'avg_fill_price': float(order.filled_avg_price) if order.filled_avg_price else None
                })
            
            return order_data
            
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            return []
    
    def get_portfolio_history(self, period: str = '1M', 
                            timeframe: str = '1Day') -> Dict:
        """Get portfolio history"""
        if not self.is_connected():
            return {}
        
        try:
            history = self.api.get_portfolio_history(
                period=period,
                timeframe=timeframe
            )
            
            return {
                'timestamp': history.timestamp,
                'equity': history.equity,
                'profit_loss': history.profit_loss,
                'profit_loss_pct': history.profit_loss_pct,
                'base_value': history.base_value,
                'timeframe': timeframe
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio history: {str(e)}")
            return {}
    
    def calculate_position_size(self, symbol: str, current_price: float) -> int:
        """Calculate appropriate position size based on risk management"""
        account_info = self.get_account_info()
        
        if not account_info or 'portfolio_value' not in account_info:
            return 0
        
        portfolio_value = account_info['portfolio_value']
        max_position_value = portfolio_value * self.max_position_size
        
        # Calculate quantity
        qty = int(max_position_value / current_price)
        
        # Ensure we have enough buying power
        buying_power = account_info.get('buying_power', 0)
        max_qty_by_power = int(buying_power / current_price)
        
        return min(qty, max_qty_by_power)
    
    def _risk_check_order(self, symbol: str, qty: int, side: OrderSide) -> bool:
        """Perform risk checks before placing order"""
        account_info = self.get_account_info()
        
        if not account_info:
            return False
        
        # Check if account is blocked
        if account_info.get('trading_blocked', False):
            logger.error("Trading is blocked on this account")
            return False
        
        # Check minimum cash balance
        if account_info.get('cash', 0) < self.min_cash_balance:
            logger.error(f"Cash balance below minimum: ${account_info.get('cash', 0)}")
            return False
        
        # For buy orders, check buying power
        if side == OrderSide.BUY:
            # Get current price estimate (simplified)
            try:
                from src.data.market import MarketDataFetcher
                fetcher = MarketDataFetcher()
                current_price = fetcher.get_current_price(symbol)
                
                if current_price:
                    order_value = qty * current_price
                    
                    if order_value > account_info.get('buying_power', 0):
                        logger.error(f"Insufficient buying power for order: ${order_value}")
                        return False
                    
                    # Check position size limit
                    portfolio_value = account_info.get('portfolio_value', 0)
                    if order_value > portfolio_value * self.max_position_size:
                        logger.error(f"Order exceeds maximum position size limit")
                        return False
            
            except Exception as e:
                logger.warning(f"Could not perform price check: {str(e)}")
        
        return True
    
    def _get_simulated_account(self) -> Dict:
        """Return simulated account info when API is not connected"""
        return {
            'account_id': 'SIMULATED',
            'status': 'ACTIVE',
            'currency': 'USD',
            'buying_power': 100000.0,
            'cash': 100000.0,
            'portfolio_value': 100000.0,
            'equity': 100000.0,
            'last_equity': 100000.0,
            'day_trade_buying_power': 100000.0,
            'initial_margin': 0.0,
            'maintenance_margin': 0.0,
            'daytrade_count': 0,
            'pattern_day_trader': False,
            'trading_blocked': False,
            'transfers_blocked': False,
            'account_blocked': False,
            'created_at': datetime.now().isoformat()
        }
    
    def _simulate_order(self, symbol: str, qty: int, side: OrderSide, 
                       order_type: OrderType, limit_price: Optional[float], 
                       stop_price: Optional[float]) -> Dict:
        """Simulate order when API is not connected"""
        import uuid
        
        order_id = str(uuid.uuid4())
        
        logger.info(f"SIMULATED ORDER: {side.value.upper()} {qty} {symbol}")
        
        return {
            'id': order_id,
            'symbol': symbol,
            'qty': float(qty),
            'filled_qty': float(qty),  # Assume filled immediately
            'side': side.value,
            'order_type': order_type.value,
            'time_in_force': 'day',
            'limit_price': limit_price,
            'stop_price': stop_price,
            'status': 'filled',
            'submitted_at': datetime.now().isoformat(),
            'filled_at': datetime.now().isoformat(),
            'avg_fill_price': limit_price if limit_price else 100.0  # Simulated price
        }

class TradingBot:
    """Automated trading bot that executes signals"""
    
    def __init__(self, paper_trader: PaperTrader):
        self.paper_trader = paper_trader
        self.auto_trading_enabled = os.getenv('ENABLE_AUTO_TRADING', 'false').lower() == 'true'
        self.trading_hours_only = os.getenv('TRADING_HOURS_ONLY', 'true').lower() == 'true'
    
    def execute_signal(self, signal: Dict) -> Optional[Dict]:
        """Execute a trading signal"""
        if not self.auto_trading_enabled:
            logger.info(f"Auto-trading disabled. Signal: {signal['action']} {signal['symbol']}")
            return None
        
        symbol = signal['symbol']
        action = signal['action']
        confidence = signal.get('confidence', 0)
        
        # Only execute high-confidence signals
        if confidence < 0.7:
            logger.info(f"Signal confidence too low ({confidence:.2f}) for {symbol}")
            return None
        
        # Check market hours if required
        if self.trading_hours_only and not self._is_market_hours():
            logger.info(f"Outside trading hours. Signal for {symbol} queued.")
            return None
        
        try:
            current_price = self._get_current_price(symbol)
            if not current_price:
                logger.error(f"Could not get current price for {symbol}")
                return None
            
            if action == 'BUY':
                return self._execute_buy_signal(symbol, current_price, confidence)
            elif action == 'SELL':
                return self._execute_sell_signal(symbol, current_price, confidence)
            else:
                logger.info(f"No action for HOLD signal: {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing signal for {symbol}: {str(e)}")
            return None
    
    def _execute_buy_signal(self, symbol: str, current_price: float, confidence: float) -> Optional[Dict]:
        """Execute a buy signal"""
        # Check if we already have a position
        positions = self.paper_trader.get_positions()
        existing_position = next((p for p in positions if p['symbol'] == symbol), None)
        
        if existing_position and existing_position['qty'] > 0:
            logger.info(f"Already have position in {symbol}, skipping buy signal")
            return None
        
        # Calculate position size
        qty = self.paper_trader.calculate_position_size(symbol, current_price)
        
        if qty <= 0:
            logger.warning(f"Position size too small for {symbol}")
            return None
        
        # Submit buy order
        order = self.paper_trader.submit_order(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )
        
        if order:
            logger.info(f"BUY order executed: {qty} shares of {symbol} at ${current_price}")
            
            # Set stop loss order
            stop_price = current_price * (1 - self.paper_trader.stop_loss_pct)
            self._set_stop_loss(symbol, qty, stop_price)
        
        return order
    
    def _execute_sell_signal(self, symbol: str, current_price: float, confidence: float) -> Optional[Dict]:
        """Execute a sell signal"""
        # Check if we have a position to sell
        positions = self.paper_trader.get_positions()
        existing_position = next((p for p in positions if p['symbol'] == symbol), None)
        
        if not existing_position or existing_position['qty'] <= 0:
            logger.info(f"No position in {symbol} to sell")
            return None
        
        qty = int(existing_position['qty'])
        
        # Submit sell order
        order = self.paper_trader.submit_order(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET
        )
        
        if order:
            logger.info(f"SELL order executed: {qty} shares of {symbol} at ${current_price}")
        
        return order
    
    def _set_stop_loss(self, symbol: str, qty: int, stop_price: float):
        """Set a stop loss order"""
        try:
            stop_order = self.paper_trader.submit_order(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                order_type=OrderType.STOP,
                stop_price=stop_price
            )
            
            if stop_order:
                logger.info(f"Stop loss set for {symbol}: ${stop_price:.2f}")
        
        except Exception as e:
            logger.error(f"Error setting stop loss for {symbol}: {str(e)}")
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            from src.data.market import MarketDataFetcher
            fetcher = MarketDataFetcher()
            return fetcher.get_current_price(symbol)
        except:
            return None
    
    def _is_market_hours(self) -> bool:
        """Check if it's currently market hours"""
        # Simplified market hours check (9:30 AM - 4:00 PM EST, Mon-Fri)
        now = datetime.now()
        
        # Skip weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check time (simplified - doesn't account for holidays or timezone properly)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
