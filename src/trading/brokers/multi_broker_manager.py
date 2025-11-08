"""
Multi-broker manager for handling multiple broker accounts
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime

from .base import BaseBroker, BrokerType, Position, AccountInfo, Order, OrderSide, OrderType, TimeInForce
from .factory import BrokerFactory

logger = logging.getLogger(__name__)

@dataclass
class BrokerAccount:
    """Represents a broker account"""
    name: str
    broker: BaseBroker
    is_default: bool = False
    is_active: bool = True
    weight: float = 1.0  # For portfolio allocation

class MultiBrokerManager:
    """Manager for multiple broker accounts"""
    
    def __init__(self):
        self.accounts: Dict[str, BrokerAccount] = {}
        self.default_account: Optional[str] = None
        self.logger = logging.getLogger(__name__)
    
    def add_broker(self, name: str, broker: BaseBroker, is_default: bool = False, weight: float = 1.0) -> bool:
        """Add a broker account"""
        try:
            account = BrokerAccount(
                name=name,
                broker=broker,
                is_default=is_default,
                weight=weight
            )
            
            self.accounts[name] = account
            
            if is_default or not self.default_account:
                self.default_account = name
            
            self.logger.info(f"Added broker account: {name} ({broker.broker_type.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding broker {name}: {str(e)}")
            return False
    
    def remove_broker(self, name: str) -> bool:
        """Remove a broker account"""
        try:
            if name in self.accounts:
                account = self.accounts[name]
                
                # Disconnect broker
                if account.broker.is_connected():
                    account.broker.disconnect()
                
                del self.accounts[name]
                
                # Update default if needed
                if self.default_account == name:
                    self.default_account = next(iter(self.accounts.keys())) if self.accounts else None
                
                self.logger.info(f"Removed broker account: {name}")
                return True
            else:
                self.logger.error(f"Broker account not found: {name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error removing broker {name}: {str(e)}")
            return False
    
    def get_broker(self, name: Optional[str] = None) -> Optional[BaseBroker]:
        """Get broker by name, or default broker"""
        try:
            if name is None:
                name = self.default_account
            
            if name and name in self.accounts:
                return self.accounts[name].broker
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting broker {name}: {str(e)}")
            return None
    
    def get_account(self, name: Optional[str] = None) -> Optional[BrokerAccount]:
        """Get broker account by name"""
        try:
            if name is None:
                name = self.default_account
            
            return self.accounts.get(name)
            
        except Exception as e:
            self.logger.error(f"Error getting account {name}: {str(e)}")
            return None
    
    def connect_all(self) -> Dict[str, bool]:
        """Connect all broker accounts"""
        results = {}
        
        for name, account in self.accounts.items():
            try:
                if not account.broker.is_connected():
                    success = account.broker.connect()
                    results[name] = success
                    
                    if success:
                        self.logger.info(f"Connected to {name}")
                    else:
                        self.logger.error(f"Failed to connect to {name}")
                        account.is_active = False
                else:
                    results[name] = True
                    
            except Exception as e:
                self.logger.error(f"Error connecting to {name}: {str(e)}")
                results[name] = False
                account.is_active = False
        
        return results
    
    def disconnect_all(self) -> Dict[str, bool]:
        """Disconnect all broker accounts"""
        results = {}
        
        for name, account in self.accounts.items():
            try:
                success = account.broker.disconnect()
                results[name] = success
                account.is_active = False
                
            except Exception as e:
                self.logger.error(f"Error disconnecting from {name}: {str(e)}")
                results[name] = False
        
        return results
    
    def get_all_positions(self) -> Dict[str, List[Position]]:
        """Get positions from all connected brokers"""
        all_positions = {}
        
        for name, account in self.accounts.items():
            if account.is_active and account.broker.is_connected():
                try:
                    positions = account.broker.get_positions()
                    all_positions[name] = positions
                    
                except Exception as e:
                    self.logger.error(f"Error getting positions from {name}: {str(e)}")
                    all_positions[name] = []
        
        return all_positions
    
    def get_consolidated_positions(self) -> Dict[str, Position]:
        """Get consolidated positions across all brokers"""
        consolidated = {}
        
        all_positions = self.get_all_positions()
        
        for broker_name, positions in all_positions.items():
            for position in positions:
                symbol = position.symbol
                
                if symbol in consolidated:
                    # Consolidate with existing position
                    existing = consolidated[symbol]
                    
                    # Calculate weighted average entry price
                    total_quantity = existing.quantity + position.quantity
                    if total_quantity != 0:
                        weighted_avg_price = (
                            (existing.quantity * existing.avg_entry_price) + 
                            (position.quantity * position.avg_entry_price)
                        ) / total_quantity
                    else:
                        weighted_avg_price = existing.avg_entry_price
                    
                    # Update consolidated position
                    existing.quantity = total_quantity
                    existing.avg_entry_price = weighted_avg_price
                    existing.market_value += position.market_value
                    existing.unrealized_pnl += position.unrealized_pnl
                    existing.cost_basis += position.cost_basis
                    
                    # Recalculate percentage
                    if existing.cost_basis != 0:
                        existing.unrealized_pnl_pct = (existing.unrealized_pnl / existing.cost_basis) * 100
                    
                else:
                    # Create new consolidated position
                    consolidated[symbol] = Position(
                        symbol=position.symbol,
                        quantity=position.quantity,
                        side=position.side,
                        avg_entry_price=position.avg_entry_price,
                        current_price=position.current_price,
                        market_value=position.market_value,
                        unrealized_pnl=position.unrealized_pnl,
                        unrealized_pnl_pct=position.unrealized_pnl_pct,
                        cost_basis=position.cost_basis,
                        change_today=position.change_today,
                        sector=position.sector
                    )
        
        return consolidated
    
    def get_total_account_value(self) -> float:
        """Get total account value across all brokers"""
        total_value = 0.0
        
        for name, account in self.accounts.items():
            if account.is_active and account.broker.is_connected():
                try:
                    account_info = account.broker.get_account_info()
                    if account_info:
                        total_value += account_info.portfolio_value
                        
                except Exception as e:
                    self.logger.error(f"Error getting account value from {name}: {str(e)}")
        
        return total_value
    
    def submit_order_distributed(self, symbol: str, total_quantity: float, side: OrderSide,
                               order_type: OrderType = OrderType.MARKET,
                               time_in_force: TimeInForce = TimeInForce.DAY,
                               exclude_brokers: Set[str] = None,
                               **kwargs) -> Dict[str, Optional[Order]]:
        """Submit order distributed across multiple brokers based on weights"""
        if exclude_brokers is None:
            exclude_brokers = set()
        
        results = {}
        
        # Get active brokers and their weights
        active_brokers = [
            (name, account) for name, account in self.accounts.items()
            if account.is_active and account.broker.is_connected() and name not in exclude_brokers
        ]
        
        if not active_brokers:
            self.logger.error("No active brokers available for order distribution")
            return results
        
        # Calculate total weight
        total_weight = sum(account.weight for name, account in active_brokers)
        
        # Distribute quantity based on weights
        remaining_quantity = total_quantity
        
        for i, (name, account) in enumerate(active_brokers):
            if i == len(active_brokers) - 1:
                # Last broker gets remaining quantity to avoid rounding issues
                broker_quantity = remaining_quantity
            else:
                broker_quantity = int((total_quantity * account.weight) / total_weight)
                remaining_quantity -= broker_quantity
            
            if broker_quantity > 0:
                try:
                    order = account.broker.submit_order(
                        symbol=symbol,
                        quantity=broker_quantity,
                        side=side,
                        order_type=order_type,
                        time_in_force=time_in_force,
                        **kwargs
                    )
                    results[name] = order
                    
                    if order:
                        self.logger.info(f"Submitted {broker_quantity} shares to {name}")
                    else:
                        self.logger.error(f"Failed to submit order to {name}")
                        
                except Exception as e:
                    self.logger.error(f"Error submitting order to {name}: {str(e)}")
                    results[name] = None
        
        return results
    
    def get_broker_status(self) -> Dict[str, Dict]:
        """Get status of all brokers"""
        status = {}
        
        for name, account in self.accounts.items():
            broker = account.broker
            account_info = None
            
            if account.is_active and broker.is_connected():
                try:
                    account_info = broker.get_account_info()
                except:
                    pass
            
            status[name] = {
                'broker_type': broker.broker_type.value,
                'is_connected': broker.is_connected(),
                'is_active': account.is_active,
                'is_default': account.is_default,
                'weight': account.weight,
                'paper_trading': broker.is_paper_trading,
                'portfolio_value': account_info.portfolio_value if account_info else 0.0,
                'cash': account_info.cash if account_info else 0.0
            }
        
        return status
    
    def set_default_broker(self, name: str) -> bool:
        """Set default broker"""
        if name in self.accounts:
            # Update previous default
            if self.default_account:
                self.accounts[self.default_account].is_default = False
            
            # Set new default
            self.accounts[name].is_default = True
            self.default_account = name
            
            self.logger.info(f"Set default broker to: {name}")
            return True
        else:
            self.logger.error(f"Broker not found: {name}")
            return False
    
    def list_brokers(self) -> Dict[str, str]:
        """List all configured brokers"""
        return {
            name: f"{account.broker.broker_type.value} ({'Default' if account.is_default else 'Active' if account.is_active else 'Inactive'})"
            for name, account in self.accounts.items()
        }