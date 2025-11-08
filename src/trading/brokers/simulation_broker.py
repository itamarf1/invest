"""
Simulation broker for testing and development
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional
import logging

from .base import (
    BaseBroker, BrokerType, Position, AccountInfo, Order, Trade,
    OrderSide, OrderType, OrderStatus, TimeInForce
)

logger = logging.getLogger(__name__)

class SimulationBroker(BaseBroker):
    """Simulated broker for testing and development"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.initial_balance = config.get('initial_balance', 100000.0)
        self.commission = config.get('commission', 0.0)
        
        # Simulated account state
        self.cash = self.initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
        
        # Always consider simulation connected
        self._is_connected = True
    
    def _get_broker_type(self) -> BrokerType:
        return BrokerType.SIMULATION
    
    def connect(self) -> bool:
        """Connect to simulation - always succeeds"""
        self._is_connected = True
        self.logger.info("Connected to simulation broker")
        return True
    
    def disconnect(self) -> bool:
        """Disconnect from simulation"""
        self._is_connected = False
        return True
    
    def get_account_info(self) -> Optional[AccountInfo]:
        """Get simulated account information"""
        if not self.is_connected():
            return None
        
        try:
            # Calculate portfolio value
            portfolio_value = self.cash
            for position in self.positions.values():
                portfolio_value += position.market_value
            
            return AccountInfo(
                account_id='SIM_ACCOUNT',
                broker_type=self.broker_type,
                status='ACTIVE',
                currency='USD',
                buying_power=self.cash * 4,  # Simulated 4:1 margin
                cash=self.cash,
                portfolio_value=portfolio_value,
                equity=portfolio_value,
                last_equity=portfolio_value,
                day_trade_buying_power=self.cash * 4,
                additional_info={
                    'initial_balance': self.initial_balance,
                    'commission': self.commission,
                    'positions_count': len(self.positions)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            return None
    
    def get_positions(self) -> List[Position]:
        """Get current simulated positions"""
        if not self.is_connected():
            return []
        
        try:
            # Update position prices with current market data
            self._update_position_prices()
            return list(self.positions.values())
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def submit_order(self, symbol: str, quantity: float, side: OrderSide,
                    order_type: OrderType = OrderType.MARKET,
                    time_in_force: TimeInForce = TimeInForce.DAY,
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    **kwargs) -> Optional[Order]:
        """Submit a simulated order"""
        if not self.is_connected():
            return None
        
        try:
            # Generate order ID
            self.order_counter += 1
            order_id = f"SIM_{self.order_counter}_{int(datetime.now().timestamp())}"
            
            # Get current price for simulation
            current_price = self.get_current_price(symbol)
            if not current_price:
                current_price = 100.0  # Fallback price for simulation
                self.logger.warning(f"Using fallback price of ${current_price} for {symbol}")
            
            # Determine execution price
            if order_type == OrderType.MARKET:
                exec_price = current_price
            elif order_type == OrderType.LIMIT and limit_price:
                # For simulation, assume limit orders execute if price is favorable
                if side == OrderSide.BUY and limit_price >= current_price:
                    exec_price = current_price
                elif side == OrderSide.SELL and limit_price <= current_price:
                    exec_price = current_price
                else:
                    # Order doesn't execute immediately
                    exec_price = None
            else:
                exec_price = current_price
            
            # Create order
            order = Order(
                id=order_id,
                symbol=symbol,
                quantity=float(quantity),
                filled_quantity=0.0,
                side=side,
                order_type=order_type,
                status=OrderStatus.NEW,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                submitted_at=datetime.now(),
                broker_order_id=order_id
            )
            
            # Store order
            self.orders[order_id] = order
            
            # Try to execute order immediately for market orders
            if order_type == OrderType.MARKET or exec_price:
                self._execute_order(order_id, exec_price or current_price)
            
            self.logger.info(f"Simulated order submitted: {side.value.upper()} {quantity} {symbol} - Order ID: {order_id}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error submitting order: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a simulated order"""
        if not self.is_connected():
            return False
        
        try:
            if order_id in self.orders:
                order = self.orders[order_id]
                if order.status in [OrderStatus.NEW, OrderStatus.ACCEPTED, OrderStatus.PENDING]:
                    order.status = OrderStatus.CANCELED
                    self.logger.info(f"Simulated order cancelled: {order_id}")
                    return True
                else:
                    self.logger.warning(f"Cannot cancel order {order_id} in status {order.status}")
                    return False
            else:
                self.logger.error(f"Order not found: {order_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False
    
    def get_orders(self, status: Optional[str] = None, limit: int = 50) -> List[Order]:
        """Get simulated orders"""
        if not self.is_connected():
            return []
        
        try:
            orders = list(self.orders.values())
            
            # Filter by status if specified
            if status:
                if status == 'open':
                    orders = [o for o in orders if o.status in [OrderStatus.NEW, OrderStatus.ACCEPTED, OrderStatus.PENDING]]
                elif status == 'filled':
                    orders = [o for o in orders if o.status == OrderStatus.FILLED]
                elif status == 'closed':
                    orders = [o for o in orders if o.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]]
            
            # Sort by submission time (newest first) and limit
            orders.sort(key=lambda x: x.submitted_at, reverse=True)
            return orders[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting orders: {str(e)}")
            return []
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of a specific simulated order"""
        if not self.is_connected():
            return None
        
        return self.orders.get(order_id)
    
    def get_portfolio_history(self, period: str = '1M') -> Dict:
        """Get simulated portfolio history"""
        if not self.is_connected():
            return {}
        
        try:
            account_info = self.get_account_info()
            
            if account_info:
                return {
                    'current_value': account_info.portfolio_value,
                    'cash': account_info.cash,
                    'equity': account_info.equity,
                    'timestamp': datetime.now().isoformat(),
                    'note': 'Simulation broker - limited historical data'
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio history: {str(e)}")
            return {}
    
    def _execute_order(self, order_id: str, price: float):
        """Execute a simulated order"""
        try:
            order = self.orders.get(order_id)
            if not order:
                return
            
            quantity = order.quantity
            symbol = order.symbol
            side = order.side
            
            # Calculate total cost including commission
            total_cost = quantity * price + self.commission
            
            # Check if we have enough cash for buy orders
            if side == OrderSide.BUY:
                if total_cost > self.cash:
                    order.status = OrderStatus.REJECTED
                    self.logger.warning(f"Order rejected: insufficient cash. Need ${total_cost:.2f}, have ${self.cash:.2f}")
                    return
            
            # Check if we have enough shares for sell orders
            elif side == OrderSide.SELL:
                existing_position = self.positions.get(symbol)
                if not existing_position or existing_position.quantity < quantity:
                    order.status = OrderStatus.REJECTED
                    self.logger.warning(f"Order rejected: insufficient shares. Need {quantity}, have {existing_position.quantity if existing_position else 0}")
                    return
            
            # Execute the order
            if side == OrderSide.BUY:
                self._execute_buy(symbol, quantity, price)
            else:
                self._execute_sell(symbol, quantity, price)
            
            # Update order status
            order.filled_quantity = quantity
            order.avg_fill_price = price
            order.filled_at = datetime.now()
            order.status = OrderStatus.FILLED
            
            self.logger.info(f"Simulated order executed: {side.value.upper()} {quantity} {symbol} at ${price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error executing order {order_id}: {str(e)}")
    
    def _execute_buy(self, symbol: str, quantity: float, price: float):
        """Execute a buy order"""
        total_cost = quantity * price + self.commission
        self.cash -= total_cost
        
        if symbol in self.positions:
            # Add to existing position
            position = self.positions[symbol]
            total_shares = position.quantity + quantity
            total_cost_basis = position.cost_basis + (quantity * price)
            new_avg_price = total_cost_basis / total_shares
            
            position.quantity = total_shares
            position.avg_entry_price = new_avg_price
            position.cost_basis = total_cost_basis
        else:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                side='long',
                avg_entry_price=price,
                current_price=price,
                market_value=quantity * price,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                cost_basis=quantity * price,
                created_at=datetime.now()
            )
    
    def _execute_sell(self, symbol: str, quantity: float, price: float):
        """Execute a sell order"""
        total_proceeds = quantity * price - self.commission
        self.cash += total_proceeds
        
        if symbol in self.positions:
            position = self.positions[symbol]
            position.quantity -= quantity
            
            if position.quantity <= 0:
                # Close position completely
                del self.positions[symbol]
            else:
                # Update remaining position
                position.cost_basis = position.quantity * position.avg_entry_price
    
    def _update_position_prices(self):
        """Update position prices with current market data"""
        try:
            for symbol, position in self.positions.items():
                current_price = self.get_current_price(symbol)
                if current_price:
                    position.current_price = current_price
                    position.market_value = position.quantity * current_price
                    position.unrealized_pnl = position.market_value - position.cost_basis
                    position.unrealized_pnl_pct = (position.unrealized_pnl / position.cost_basis) * 100 if position.cost_basis else 0
                    
        except Exception as e:
            self.logger.error(f"Error updating position prices: {str(e)}")
    
    def reset_simulation(self):
        """Reset simulation to initial state"""
        self.cash = self.initial_balance
        self.positions.clear()
        self.orders.clear()
        self.order_counter = 0
        self.logger.info("Simulation broker reset to initial state")