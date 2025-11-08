"""
Interactive Brokers (IBKR) broker integration
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional
import logging
import threading

from .base import (
    BaseBroker, BrokerType, Position, AccountInfo, Order, Trade,
    OrderSide, OrderType, OrderStatus, TimeInForce
)

try:
    from ib_insync import IB, Stock, Order as IBOrder, MarketOrder, LimitOrder, StopOrder
    from ib_insync import Contract, PortfolioItem
except ImportError:
    IB = None

logger = logging.getLogger(__name__)

class IBKRBroker(BaseBroker):
    """Interactive Brokers broker integration using ib_insync"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.ib = None
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 7497 if self.is_paper_trading else 7496)
        self.client_id = config.get('client_id', 1)
        self._order_counter = 0
        self._order_lock = threading.Lock()
    
    def _get_broker_type(self) -> BrokerType:
        return BrokerType.INTERACTIVE_BROKERS
    
    def connect(self) -> bool:
        """Connect to Interactive Brokers TWS or IB Gateway"""
        if IB is None:
            self.logger.error("ib_insync not installed. Run: pip install ib_insync")
            return False
        
        try:
            self.ib = IB()
            
            # Connect to TWS/IB Gateway
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            
            # Wait a moment for connection to stabilize
            time.sleep(2)
            
            if self.ib.isConnected():
                self.logger.info(f"Connected to Interactive Brokers TWS/Gateway on {self.host}:{self.port}")
                self.logger.info(f"Client ID: {self.client_id}, Paper Trading: {self.is_paper_trading}")
                self._is_connected = True
                return True
            else:
                self.logger.error("Failed to connect to Interactive Brokers")
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to Interactive Brokers: {str(e)}")
            self.ib = None
            self._is_connected = False
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Interactive Brokers"""
        try:
            if self.ib and self.ib.isConnected():
                self.ib.disconnect()
            self._is_connected = False
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting: {str(e)}")
            return False
    
    def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information"""
        if not self.is_connected():
            return None
        
        try:
            # Get account values
            account_values = self.ib.accountValues()
            account_summary = self.ib.accountSummary()
            
            # Convert to dictionary for easier access
            values_dict = {av.tag: float(av.value) if av.value.replace('.', '').replace('-', '').isdigit() else av.value 
                          for av in account_values if av.currency == 'USD'}
            summary_dict = {av.tag: float(av.value) if av.value.replace('.', '').replace('-', '').isdigit() else av.value 
                          for av in account_summary if av.currency == 'USD'}
            
            # Get account name/ID
            account_id = values_dict.get('AccountCode', 'IBKR_ACCOUNT')
            
            return AccountInfo(
                account_id=account_id,
                broker_type=self.broker_type,
                status='ACTIVE',  # IB doesn't provide explicit status in the same way
                currency='USD',
                buying_power=summary_dict.get('BuyingPower', 0.0),
                cash=summary_dict.get('TotalCashValue', 0.0),
                portfolio_value=summary_dict.get('NetLiquidation', 0.0),
                equity=summary_dict.get('NetLiquidation', 0.0),
                last_equity=summary_dict.get('NetLiquidation', 0.0),
                day_trade_buying_power=summary_dict.get('DayTradesRemaining', 0),
                initial_margin=summary_dict.get('InitMarginReq', 0.0),
                maintenance_margin=summary_dict.get('MaintMarginReq', 0.0),
                pattern_day_trader=summary_dict.get('DayTradesRemaining', 4) < 4,
                additional_info={
                    'gross_position_value': summary_dict.get('GrossPositionValue', 0.0),
                    'unrealized_pnl': summary_dict.get('UnrealizedPnL', 0.0),
                    'realized_pnl': summary_dict.get('RealizedPnL', 0.0),
                    'available_funds': summary_dict.get('AvailableFunds', 0.0)
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
            portfolio_items = self.ib.portfolio()
            
            position_list = []
            for item in portfolio_items:
                if item.position != 0:  # Only include non-zero positions
                    position_list.append(Position(
                        symbol=item.contract.symbol,
                        quantity=float(item.position),
                        side='long' if item.position > 0 else 'short',
                        avg_entry_price=float(item.averageCost),
                        current_price=float(item.marketPrice) if item.marketPrice else 0,
                        market_value=float(item.marketValue),
                        unrealized_pnl=float(item.unrealizedPNL),
                        unrealized_pnl_pct=float(item.unrealizedPNL / abs(item.marketValue * 100)) if item.marketValue else 0,
                        cost_basis=float(abs(item.position * item.averageCost)),
                        sector=getattr(item.contract, 'primaryExchange', 'Unknown')
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
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Create order based on type
            if order_type == OrderType.MARKET:
                ib_order = MarketOrder(side.value.upper(), int(quantity))
            elif order_type == OrderType.LIMIT:
                if limit_price is None:
                    raise ValueError("Limit price required for limit orders")
                ib_order = LimitOrder(side.value.upper(), int(quantity), limit_price)
            elif order_type == OrderType.STOP:
                if stop_price is None:
                    raise ValueError("Stop price required for stop orders")
                ib_order = StopOrder(side.value.upper(), int(quantity), stop_price)
            else:
                raise ValueError(f"Order type {order_type} not yet implemented for IBKR")
            
            # Set time in force
            if time_in_force == TimeInForce.GTC:
                ib_order.tif = 'GTC'
            elif time_in_force == TimeInForce.IOC:
                ib_order.tif = 'IOC'
            elif time_in_force == TimeInForce.FOK:
                ib_order.tif = 'FOK'
            else:
                ib_order.tif = 'DAY'
            
            # Submit order
            trade = self.ib.placeOrder(contract, ib_order)
            
            # Wait briefly for order to be acknowledged
            self.ib.sleep(1)
            
            with self._order_lock:
                self._order_counter += 1
                internal_id = f"IBKR_{self._order_counter}_{int(time.time())}"
            
            order = Order(
                id=internal_id,
                symbol=symbol,
                quantity=float(quantity),
                filled_quantity=float(trade.orderStatus.filled),
                side=side,
                order_type=order_type,
                status=self._map_ibkr_status(trade.orderStatus.status),
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                submitted_at=datetime.now(),
                broker_order_id=str(trade.order.orderId),
                additional_info={
                    'ib_trade': trade,
                    'contract': contract
                }
            )
            
            self.logger.info(f"Order submitted: {side.value.upper()} {quantity} {symbol} - Order ID: {trade.order.orderId}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error submitting order: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.is_connected():
            return False
        
        try:
            # For IBKR, we need to find the trade by order ID
            trades = self.ib.trades()
            
            target_trade = None
            for trade in trades:
                if str(trade.order.orderId) == order_id or order_id in str(trade):
                    target_trade = trade
                    break
            
            if target_trade:
                self.ib.cancelOrder(target_trade.order)
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                self.logger.error(f"Order not found: {order_id}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False
    
    def get_orders(self, status: Optional[str] = None, limit: int = 50) -> List[Order]:
        """Get orders"""
        if not self.is_connected():
            return []
        
        try:
            trades = self.ib.trades()
            
            order_list = []
            count = 0
            
            for trade in reversed(trades):  # Most recent first
                if count >= limit:
                    break
                
                # Filter by status if specified
                if status and not self._status_matches(trade.orderStatus.status, status):
                    continue
                
                order = Order(
                    id=f"IBKR_{trade.order.orderId}",
                    symbol=trade.contract.symbol,
                    quantity=float(trade.order.totalQuantity),
                    filled_quantity=float(trade.orderStatus.filled),
                    side=OrderSide.BUY if trade.order.action == 'BUY' else OrderSide.SELL,
                    order_type=self._map_ibkr_order_type(trade.order.orderType),
                    status=self._map_ibkr_status(trade.orderStatus.status),
                    time_in_force=self._map_ibkr_tif(trade.order.tif),
                    limit_price=float(trade.order.lmtPrice) if hasattr(trade.order, 'lmtPrice') and trade.order.lmtPrice else None,
                    stop_price=float(trade.order.auxPrice) if hasattr(trade.order, 'auxPrice') and trade.order.auxPrice else None,
                    avg_fill_price=float(trade.orderStatus.avgFillPrice) if trade.orderStatus.avgFillPrice else None,
                    broker_order_id=str(trade.order.orderId),
                    additional_info={'ib_trade': trade}
                )
                
                order_list.append(order)
                count += 1
            
            return order_list
            
        except Exception as e:
            self.logger.error(f"Error getting orders: {str(e)}")
            return []
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of a specific order"""
        if not self.is_connected():
            return None
        
        try:
            trades = self.ib.trades()
            
            for trade in trades:
                if str(trade.order.orderId) == order_id or order_id in str(trade):
                    return Order(
                        id=f"IBKR_{trade.order.orderId}",
                        symbol=trade.contract.symbol,
                        quantity=float(trade.order.totalQuantity),
                        filled_quantity=float(trade.orderStatus.filled),
                        side=OrderSide.BUY if trade.order.action == 'BUY' else OrderSide.SELL,
                        order_type=self._map_ibkr_order_type(trade.order.orderType),
                        status=self._map_ibkr_status(trade.orderStatus.status),
                        time_in_force=self._map_ibkr_tif(trade.order.tif),
                        limit_price=float(trade.order.lmtPrice) if hasattr(trade.order, 'lmtPrice') and trade.order.lmtPrice else None,
                        stop_price=float(trade.order.auxPrice) if hasattr(trade.order, 'auxPrice') and trade.order.auxPrice else None,
                        avg_fill_price=float(trade.orderStatus.avgFillPrice) if trade.orderStatus.avgFillPrice else None,
                        broker_order_id=str(trade.order.orderId),
                        additional_info={'ib_trade': trade}
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {str(e)}")
            return None
    
    def get_portfolio_history(self, period: str = '1M') -> Dict:
        """Get portfolio history - limited implementation for IBKR"""
        if not self.is_connected():
            return {}
        
        try:
            # IBKR doesn't provide historical portfolio data in the same way as Alpaca
            # This is a simplified implementation
            account_info = self.get_account_info()
            
            if account_info:
                return {
                    'current_value': account_info.portfolio_value,
                    'cash': account_info.cash,
                    'equity': account_info.equity,
                    'timestamp': datetime.now().isoformat(),
                    'note': 'IBKR historical portfolio data requires additional implementation'
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio history: {str(e)}")
            return {}
    
    def _map_ibkr_status(self, ibkr_status: str) -> OrderStatus:
        """Map IBKR order status to standard status"""
        status_mapping = {
            'PendingSubmit': OrderStatus.PENDING,
            'PreSubmitted': OrderStatus.ACCEPTED,
            'Submitted': OrderStatus.NEW,
            'Filled': OrderStatus.FILLED,
            'Cancelled': OrderStatus.CANCELED,
            'Inactive': OrderStatus.REJECTED,
            'PartiallyFilled': OrderStatus.PARTIALLY_FILLED,
            'ApiCancelled': OrderStatus.CANCELED,
            'PendingCancel': OrderStatus.PENDING
        }
        
        return status_mapping.get(ibkr_status, OrderStatus.PENDING)
    
    def _map_ibkr_order_type(self, ibkr_type: str) -> OrderType:
        """Map IBKR order type to standard type"""
        type_mapping = {
            'MKT': OrderType.MARKET,
            'LMT': OrderType.LIMIT,
            'STP': OrderType.STOP,
            'STP LMT': OrderType.STOP_LIMIT,
            'TRAIL': OrderType.TRAILING_STOP
        }
        
        return type_mapping.get(ibkr_type, OrderType.MARKET)
    
    def _map_ibkr_tif(self, ibkr_tif: str) -> TimeInForce:
        """Map IBKR time in force to standard TIF"""
        tif_mapping = {
            'DAY': TimeInForce.DAY,
            'GTC': TimeInForce.GTC,
            'IOC': TimeInForce.IOC,
            'FOK': TimeInForce.FOK
        }
        
        return tif_mapping.get(ibkr_tif, TimeInForce.DAY)
    
    def _status_matches(self, ibkr_status: str, filter_status: str) -> bool:
        """Check if IBKR status matches filter"""
        if filter_status == 'open':
            return ibkr_status in ['PendingSubmit', 'PreSubmitted', 'Submitted', 'PartiallyFilled']
        elif filter_status == 'filled':
            return ibkr_status == 'Filled'
        elif filter_status == 'closed':
            return ibkr_status in ['Filled', 'Cancelled', 'Inactive', 'ApiCancelled']
        else:
            return True  # 'all' or any other value