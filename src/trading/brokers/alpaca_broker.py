"""
Alpaca broker integration
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

from .base import (
    BaseBroker, BrokerType, Position, AccountInfo, Order, Trade,
    OrderSide, OrderType, OrderStatus, TimeInForce
)

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None

logger = logging.getLogger(__name__)

class AlpacaBroker(BaseBroker):
    """Alpaca Markets broker integration"""
    
    def _get_broker_type(self) -> BrokerType:
        return BrokerType.ALPACA
    
    def connect(self) -> bool:
        """Connect to Alpaca API"""
        if tradeapi is None:
            self.logger.error("alpaca_trade_api not installed. Run: pip install alpaca_trade_api")
            return False
        
        # Get credentials from config or environment
        api_key = self.config.get('api_key') or os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID')
        secret_key = self.config.get('secret_key') or os.getenv('ALPACA_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY')
        
        if not api_key or not secret_key:
            self.logger.error("Alpaca API credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
            return False
        
        try:
            base_url = 'https://paper-api.alpaca.markets' if self.is_paper_trading else 'https://api.alpaca.markets'
            
            self._client = tradeapi.REST(
                key_id=api_key,
                secret_key=secret_key,
                base_url=base_url,
                api_version='v2'
            )
            
            # Test connection
            account = self._client.get_account()
            self.logger.info(f"Connected to Alpaca {'Paper' if self.is_paper_trading else 'Live'} Trading")
            self.logger.info(f"Account Status: {account.status}")
            
            self._is_connected = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca API: {str(e)}")
            self._client = None
            self._is_connected = False
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Alpaca API"""
        self._client = None
        self._is_connected = False
        return True
    
    def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information"""
        if not self.is_connected():
            return None
        
        try:
            account = self._client.get_account()
            
            return AccountInfo(
                account_id=account.id,
                broker_type=self.broker_type,
                status=account.status,
                currency=account.currency,
                buying_power=float(account.buying_power),
                cash=float(account.cash),
                portfolio_value=float(account.portfolio_value),
                equity=float(account.equity),
                last_equity=float(account.last_equity),
                day_trade_buying_power=float(account.day_trade_buying_power),
                initial_margin=float(account.initial_margin),
                maintenance_margin=float(account.maintenance_margin),
                daytrade_count=account.daytrade_count,
                pattern_day_trader=account.pattern_day_trader,
                trading_blocked=account.trading_blocked,
                transfers_blocked=account.transfers_blocked,
                account_blocked=account.account_blocked,
                created_at=account.created_at,
                additional_info={
                    'regt_buying_power': float(account.regt_buying_power),
                    'crypto_status': getattr(account, 'crypto_status', None)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            return None
    
    def get_positions(self) -> List[Position]:
        """Get current positions"""
        if not self.is_connected():
            return []
        
        try:
            positions = self._client.list_positions()
            
            position_list = []
            for pos in positions:
                position_list.append(Position(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    side='long' if float(pos.qty) > 0 else 'short',
                    avg_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price) if pos.current_price else 0,
                    market_value=float(pos.market_value),
                    unrealized_pnl=float(pos.unrealized_pl),
                    unrealized_pnl_pct=float(pos.unrealized_plpc),
                    cost_basis=float(pos.cost_basis),
                    change_today=float(pos.change_today) if pos.change_today else 0
                ))
            
            return position_list
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def submit_order(self, symbol: str, quantity: float, side: OrderSide,
                    order_type: OrderType = OrderType.MARKET,
                    time_in_force: TimeInForce = TimeInForce.DAY,
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    **kwargs) -> Optional[Order]:
        """Submit an order"""
        if not self.is_connected():
            return None
        
        # Validate order first
        current_price = self.get_current_price(symbol) if side == OrderSide.BUY else None
        is_valid, message = self.validate_order(symbol, quantity, side, current_price)
        
        if not is_valid:
            self.logger.warning(f"Order validation failed: {message}")
            return None
        
        try:
            alpaca_order = self._client.submit_order(
                symbol=symbol,
                qty=int(quantity),
                side=side.value,
                type=order_type.value,
                time_in_force=time_in_force.value,
                limit_price=limit_price,
                stop_price=stop_price,
                **kwargs
            )
            
            order = Order(
                id=alpaca_order.id,
                symbol=alpaca_order.symbol,
                quantity=float(alpaca_order.qty),
                filled_quantity=float(alpaca_order.filled_qty),
                side=OrderSide(alpaca_order.side),
                order_type=OrderType(alpaca_order.order_type),
                status=self._map_alpaca_status(alpaca_order.status),
                time_in_force=TimeInForce(alpaca_order.time_in_force),
                limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                submitted_at=alpaca_order.submitted_at,
                filled_at=alpaca_order.filled_at,
                avg_fill_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                broker_order_id=alpaca_order.id
            )
            
            self.logger.info(f"Order submitted: {side.value.upper()} {quantity} {symbol} - Order ID: {alpaca_order.id}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error submitting order: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.is_connected():
            return False
        
        try:
            self._client.cancel_order(order_id)
            self.logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False
    
    def get_orders(self, status: Optional[str] = None, limit: int = 50) -> List[Order]:
        """Get orders"""
        if not self.is_connected():
            return []
        
        try:
            orders = self._client.list_orders(
                status=status,
                limit=limit,
                nested=True
            )
            
            order_list = []
            for alpaca_order in orders:
                order_list.append(Order(
                    id=alpaca_order.id,
                    symbol=alpaca_order.symbol,
                    quantity=float(alpaca_order.qty),
                    filled_quantity=float(alpaca_order.filled_qty),
                    side=OrderSide(alpaca_order.side),
                    order_type=OrderType(alpaca_order.order_type),
                    status=self._map_alpaca_status(alpaca_order.status),
                    time_in_force=TimeInForce(alpaca_order.time_in_force),
                    limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                    stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                    submitted_at=alpaca_order.submitted_at,
                    filled_at=alpaca_order.filled_at,
                    avg_fill_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                    broker_order_id=alpaca_order.id
                ))
            
            return order_list
            
        except Exception as e:
            self.logger.error(f"Error getting orders: {str(e)}")
            return []
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of a specific order"""
        if not self.is_connected():
            return None
        
        try:
            alpaca_order = self._client.get_order(order_id)
            
            return Order(
                id=alpaca_order.id,
                symbol=alpaca_order.symbol,
                quantity=float(alpaca_order.qty),
                filled_quantity=float(alpaca_order.filled_qty),
                side=OrderSide(alpaca_order.side),
                order_type=OrderType(alpaca_order.order_type),
                status=self._map_alpaca_status(alpaca_order.status),
                time_in_force=TimeInForce(alpaca_order.time_in_force),
                limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                submitted_at=alpaca_order.submitted_at,
                filled_at=alpaca_order.filled_at,
                avg_fill_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                broker_order_id=alpaca_order.id
            )
            
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {str(e)}")
            return None
    
    def get_portfolio_history(self, period: str = '1M') -> Dict:
        """Get portfolio history"""
        if not self.is_connected():
            return {}
        
        try:
            history = self._client.get_portfolio_history(
                period=period,
                timeframe='1Day'
            )
            
            return {
                'timestamp': history.timestamp,
                'equity': history.equity,
                'profit_loss': history.profit_loss,
                'profit_loss_pct': history.profit_loss_pct,
                'base_value': history.base_value,
                'timeframe': '1Day'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio history: {str(e)}")
            return {}
    
    def _map_alpaca_status(self, alpaca_status: str) -> OrderStatus:
        """Map Alpaca order status to standard status"""
        status_mapping = {
            'new': OrderStatus.NEW,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'done_for_day': OrderStatus.FILLED,  # Map to filled
            'canceled': OrderStatus.CANCELED,
            'expired': OrderStatus.EXPIRED,
            'replaced': OrderStatus.CANCELED,  # Old order is cancelled
            'pending_cancel': OrderStatus.PENDING,
            'pending_replace': OrderStatus.PENDING,
            'accepted': OrderStatus.ACCEPTED,
            'pending_new': OrderStatus.PENDING,
            'accepted_for_bidding': OrderStatus.ACCEPTED,
            'stopped': OrderStatus.CANCELED,
            'rejected': OrderStatus.REJECTED,
            'suspended': OrderStatus.PENDING,
            'calculated': OrderStatus.FILLED
        }
        
        return status_mapping.get(alpaca_status, OrderStatus.PENDING)