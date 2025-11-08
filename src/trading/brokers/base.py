"""
Base broker interface and common data structures
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class BrokerType(Enum):
    """Supported broker types"""
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "ibkr"
    TD_AMERITRADE = "td_ameritrade"
    ETRADE = "etrade"
    CHARLES_SCHWAB = "schwab"
    SIMULATION = "simulation"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderStatus(Enum):
    """Order statuses"""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    PENDING = "pending"
    ACCEPTED = "accepted"

class TimeInForce(Enum):
    """Time in force options"""
    DAY = "day"
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill

@dataclass
class Position:
    """Portfolio position"""
    symbol: str
    quantity: float
    side: str  # "long" or "short"
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    cost_basis: float
    change_today: float = 0.0
    sector: str = "Unknown"
    created_at: Optional[datetime] = None

@dataclass  
class AccountInfo:
    """Account information"""
    account_id: str
    broker_type: BrokerType
    status: str
    currency: str = "USD"
    buying_power: float = 0.0
    cash: float = 0.0
    portfolio_value: float = 0.0
    equity: float = 0.0
    last_equity: float = 0.0
    day_trade_buying_power: float = 0.0
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    daytrade_count: int = 0
    pattern_day_trader: bool = False
    trading_blocked: bool = False
    transfers_blocked: bool = False
    account_blocked: bool = False
    created_at: Optional[datetime] = None
    additional_info: Dict = None

    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}

@dataclass
class Order:
    """Order information"""
    id: str
    symbol: str
    quantity: float
    filled_quantity: float
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    time_in_force: TimeInForce
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    avg_fill_price: Optional[float] = None
    broker_order_id: Optional[str] = None  # Broker-specific order ID
    additional_info: Dict = None

    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}

@dataclass
class Trade:
    """Executed trade information"""
    id: str
    symbol: str
    quantity: float
    side: OrderSide
    price: float
    timestamp: datetime
    order_id: str
    commission: float = 0.0
    additional_info: Dict = None

    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}

class BaseBroker(ABC):
    """
    Abstract base class for all broker integrations
    
    This defines the common interface that all broker implementations must follow.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.broker_type = self._get_broker_type()
        self.is_paper_trading = config.get('paper_trading', True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Risk management settings
        self.max_position_size = config.get('max_position_size', 0.05)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)
        self.max_daily_loss = config.get('max_daily_loss', 0.05)
        self.min_cash_balance = config.get('min_cash_balance', 1000)
        
        self._client = None
        self._is_connected = False
    
    @abstractmethod
    def _get_broker_type(self) -> BrokerType:
        """Return the broker type"""
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the broker API"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the broker API"""
        pass
    
    def is_connected(self) -> bool:
        """Check if connected to broker"""
        return self._is_connected
    
    @abstractmethod
    def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information"""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get current positions"""
        pass
    
    @abstractmethod
    def submit_order(self, symbol: str, quantity: float, side: OrderSide,
                    order_type: OrderType = OrderType.MARKET,
                    time_in_force: TimeInForce = TimeInForce.DAY,
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    **kwargs) -> Optional[Order]:
        """Submit an order"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    def get_orders(self, status: Optional[str] = None, 
                  limit: int = 50) -> List[Order]:
        """Get orders"""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of a specific order"""
        pass
    
    @abstractmethod
    def get_portfolio_history(self, period: str = '1M') -> Dict:
        """Get portfolio history"""
        pass
    
    def calculate_position_size(self, symbol: str, current_price: float) -> int:
        """Calculate appropriate position size based on risk management"""
        account_info = self.get_account_info()
        
        if not account_info:
            return 0
        
        portfolio_value = account_info.portfolio_value
        max_position_value = portfolio_value * self.max_position_size
        
        # Calculate quantity
        qty = int(max_position_value / current_price)
        
        # Ensure we have enough buying power
        buying_power = account_info.buying_power
        max_qty_by_power = int(buying_power / current_price)
        
        return min(qty, max_qty_by_power)
    
    def validate_order(self, symbol: str, quantity: float, side: OrderSide,
                      current_price: Optional[float] = None) -> Tuple[bool, str]:
        """Validate order before submission"""
        account_info = self.get_account_info()
        
        if not account_info:
            return False, "Unable to get account information"
        
        # Check if account is blocked
        if account_info.trading_blocked:
            return False, "Trading is blocked on this account"
        
        # Check minimum cash balance
        if account_info.cash < self.min_cash_balance:
            return False, f"Cash balance below minimum: ${account_info.cash}"
        
        # For buy orders, check buying power
        if side == OrderSide.BUY and current_price:
            order_value = quantity * current_price
            
            if order_value > account_info.buying_power:
                return False, f"Insufficient buying power for order: ${order_value}"
            
            # Check position size limit
            if order_value > account_info.portfolio_value * self.max_position_size:
                return False, "Order exceeds maximum position size limit"
        
        return True, "Order validation passed"
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol - should be implemented by broker or use market data"""
        # This is a placeholder - brokers should implement their own price fetching
        # or use the market data fetcher
        try:
            from ...data.market import MarketDataFetcher
            fetcher = MarketDataFetcher()
            return fetcher.get_current_price(symbol)
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    def __str__(self):
        return f"{self.broker_type.value.upper()} Broker ({'Paper' if self.is_paper_trading else 'Live'})"
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(type={self.broker_type.value}, paper={self.is_paper_trading})>"