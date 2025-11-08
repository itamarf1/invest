import asyncio
import websocket
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Set
import threading
import time
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd

from src.data.market import MarketDataFetcher

logger = logging.getLogger(__name__)

class AlertType(Enum):
    PRICE_ALERT = "price_alert"
    VOLUME_SPIKE = "volume_spike"
    BREAKOUT = "breakout"
    TECHNICAL_SIGNAL = "technical_signal"
    SENTIMENT_CHANGE = "sentiment_change"
    NEWS_EVENT = "news_event"

@dataclass
class Alert:
    id: str
    symbol: str
    alert_type: AlertType
    message: str
    current_value: float
    target_value: Optional[float]
    timestamp: datetime
    triggered: bool = False
    metadata: Optional[Dict] = None

@dataclass
class PriceUpdate:
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None

class AlertManager:
    """Manages trading alerts and notifications"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.active_alerts: Dict[str, List[Alert]] = {}  # symbol -> alerts
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.alert_counter = 0
        
    def add_alert(self, symbol: str, alert_type: AlertType, target_value: float, 
                 message: str, metadata: Optional[Dict] = None) -> str:
        """Add a new alert"""
        alert_id = f"{symbol}_{alert_type.value}_{self.alert_counter}"
        self.alert_counter += 1
        
        alert = Alert(
            id=alert_id,
            symbol=symbol.upper(),
            alert_type=alert_type,
            message=message,
            current_value=0.0,
            target_value=target_value,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.alerts[alert_id] = alert
        
        if symbol not in self.active_alerts:
            self.active_alerts[symbol] = []
        self.active_alerts[symbol].append(alert)
        
        logger.info(f"Alert added: {alert_id} for {symbol}")
        return alert_id
    
    def remove_alert(self, alert_id: str) -> bool:
        """Remove an alert"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            del self.alerts[alert_id]
            
            # Remove from active alerts
            if alert.symbol in self.active_alerts:
                self.active_alerts[alert.symbol] = [
                    a for a in self.active_alerts[alert.symbol] if a.id != alert_id
                ]
                
            logger.info(f"Alert removed: {alert_id}")
            return True
        return False
    
    def check_price_alerts(self, price_update: PriceUpdate):
        """Check if any price alerts should be triggered"""
        symbol = price_update.symbol
        
        if symbol not in self.active_alerts:
            return
        
        triggered_alerts = []
        
        for alert in self.active_alerts[symbol]:
            if alert.triggered:
                continue
                
            alert.current_value = price_update.price
            
            should_trigger = False
            
            if alert.alert_type == AlertType.PRICE_ALERT:
                # Price target alert
                direction = alert.metadata.get('direction', 'above')
                if direction == 'above' and price_update.price >= alert.target_value:
                    should_trigger = True
                elif direction == 'below' and price_update.price <= alert.target_value:
                    should_trigger = True
                    
            elif alert.alert_type == AlertType.VOLUME_SPIKE:
                # Volume spike alert
                avg_volume = alert.metadata.get('average_volume', 1000000)
                spike_threshold = alert.target_value  # multiplier
                if price_update.volume >= avg_volume * spike_threshold:
                    should_trigger = True
                    
            elif alert.alert_type == AlertType.BREAKOUT:
                # Breakout alert (price breaking resistance/support)
                breakout_type = alert.metadata.get('breakout_type', 'resistance')
                if breakout_type == 'resistance' and price_update.price > alert.target_value:
                    should_trigger = True
                elif breakout_type == 'support' and price_update.price < alert.target_value:
                    should_trigger = True
            
            if should_trigger:
                alert.triggered = True
                alert.timestamp = datetime.now()
                triggered_alerts.append(alert)
                
        # Notify callbacks
        for alert in triggered_alerts:
            self._notify_alert(alert)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback function for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def _notify_alert(self, alert: Alert):
        """Notify all callbacks about a triggered alert"""
        logger.info(f"ALERT TRIGGERED: {alert.message}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")
    
    def get_active_alerts(self, symbol: Optional[str] = None) -> List[Alert]:
        """Get active (non-triggered) alerts"""
        if symbol:
            return [a for a in self.active_alerts.get(symbol.upper(), []) if not a.triggered]
        else:
            all_alerts = []
            for alerts in self.active_alerts.values():
                all_alerts.extend([a for a in alerts if not a.triggered])
            return all_alerts
    
    def get_triggered_alerts(self, symbol: Optional[str] = None, hours_back: int = 24) -> List[Alert]:
        """Get recently triggered alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        if symbol:
            return [a for a in self.active_alerts.get(symbol.upper(), []) 
                   if a.triggered and a.timestamp >= cutoff_time]
        else:
            all_alerts = []
            for alerts in self.active_alerts.values():
                all_alerts.extend([a for a in alerts 
                                 if a.triggered and a.timestamp >= cutoff_time])
            return all_alerts

class RealTimeDataStream:
    """Real-time data streaming using WebSockets"""
    
    def __init__(self):
        self.subscribed_symbols: Set[str] = set()
        self.price_callbacks: List[Callable[[PriceUpdate], None]] = []
        self.alert_manager = AlertManager()
        self.market_fetcher = MarketDataFetcher()
        
        # WebSocket connection
        self.ws = None
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        # Simulated data for when real WebSocket isn't available
        self.simulation_thread = None
        self.use_simulation = True  # Default to simulation
        
    def subscribe_symbol(self, symbol: str):
        """Subscribe to real-time updates for a symbol"""
        symbol = symbol.upper()
        self.subscribed_symbols.add(symbol)
        logger.info(f"Subscribed to real-time data for {symbol}")
        
        if self.use_simulation and not self.simulation_thread:
            self._start_simulation()
    
    def unsubscribe_symbol(self, symbol: str):
        """Unsubscribe from updates for a symbol"""
        symbol = symbol.upper()
        self.subscribed_symbols.discard(symbol)
        logger.info(f"Unsubscribed from real-time data for {symbol}")
    
    def add_price_callback(self, callback: Callable[[PriceUpdate], None]):
        """Add callback for price updates"""
        self.price_callbacks.append(callback)
    
    def _start_simulation(self):
        """Start simulated real-time data feed"""
        if self.simulation_thread and self.simulation_thread.is_alive():
            return
            
        self.is_running = True
        self.simulation_thread = threading.Thread(target=self._simulate_real_time_data, daemon=True)
        self.simulation_thread.start()
        logger.info("Started simulated real-time data feed")
    
    def _simulate_real_time_data(self):
        """Simulate real-time price updates"""
        base_prices = {}
        
        while self.is_running and self.subscribed_symbols:
            try:
                for symbol in list(self.subscribed_symbols):
                    # Get or initialize base price
                    if symbol not in base_prices:
                        try:
                            data = self.market_fetcher.get_stock_data(symbol, period="1d")
                            if not data.empty:
                                base_prices[symbol] = float(data['Close'].iloc[-1])
                            else:
                                base_prices[symbol] = 100.0  # Fallback
                        except:
                            base_prices[symbol] = 100.0
                    
                    # Generate simulated price movement
                    base_price = base_prices[symbol]
                    
                    # Random walk with slight upward bias
                    change_percent = (random.random() - 0.48) * 2  # -0.96% to +1.04%
                    new_price = base_price * (1 + change_percent / 100)
                    
                    # Update base price for next iteration
                    base_prices[symbol] = new_price
                    
                    # Generate volume (random between 50K and 2M)
                    volume = random.randint(50000, 2000000)
                    
                    price_update = PriceUpdate(
                        symbol=symbol,
                        price=new_price,
                        change=new_price - base_price,
                        change_percent=change_percent,
                        volume=volume,
                        timestamp=datetime.now(),
                        bid=new_price - 0.01,
                        ask=new_price + 0.01,
                        bid_size=random.randint(100, 1000),
                        ask_size=random.randint(100, 1000)
                    )
                    
                    # Notify callbacks
                    self._notify_price_update(price_update)
                    
                    # Check alerts
                    self.alert_manager.check_price_alerts(price_update)
                
                # Sleep for a short interval (simulate real-time updates)
                time.sleep(random.uniform(1, 3))  # 1-3 seconds between updates
                
            except Exception as e:
                logger.error(f"Error in simulation: {str(e)}")
                time.sleep(5)
    
    def _notify_price_update(self, price_update: PriceUpdate):
        """Notify all callbacks about price update"""
        for callback in self.price_callbacks:
            try:
                callback(price_update)
            except Exception as e:
                logger.error(f"Error in price callback: {str(e)}")
    
    def stop(self):
        """Stop the real-time data stream"""
        self.is_running = False
        if self.ws:
            self.ws.close()
        logger.info("Real-time data stream stopped")
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for subscribed symbols"""
        prices = {}
        for symbol in self.subscribed_symbols:
            try:
                data = self.market_fetcher.get_stock_data(symbol, period="1d")
                if not data.empty:
                    prices[symbol] = float(data['Close'].iloc[-1])
            except:
                pass
        return prices
    
    # Alert convenience methods
    def add_price_alert(self, symbol: str, target_price: float, 
                       direction: str = 'above', message: Optional[str] = None) -> str:
        """Add a price alert"""
        if not message:
            message = f"{symbol} reached ${target_price:.2f}"
        
        return self.alert_manager.add_alert(
            symbol=symbol,
            alert_type=AlertType.PRICE_ALERT,
            target_value=target_price,
            message=message,
            metadata={'direction': direction}
        )
    
    def add_volume_alert(self, symbol: str, volume_multiplier: float = 2.0,
                        message: Optional[str] = None) -> str:
        """Add a volume spike alert"""
        if not message:
            message = f"{symbol} volume spike detected ({volume_multiplier}x normal)"
        
        # Get average volume
        try:
            data = self.market_fetcher.get_stock_data(symbol, period="1mo")
            avg_volume = data['Volume'].mean() if not data.empty else 1000000
        except:
            avg_volume = 1000000
        
        return self.alert_manager.add_alert(
            symbol=symbol,
            alert_type=AlertType.VOLUME_SPIKE,
            target_value=volume_multiplier,
            message=message,
            metadata={'average_volume': avg_volume}
        )
    
    def add_breakout_alert(self, symbol: str, level: float, 
                          breakout_type: str = 'resistance',
                          message: Optional[str] = None) -> str:
        """Add a breakout alert"""
        if not message:
            message = f"{symbol} {breakout_type} breakout at ${level:.2f}"
        
        return self.alert_manager.add_alert(
            symbol=symbol,
            alert_type=AlertType.BREAKOUT,
            target_value=level,
            message=message,
            metadata={'breakout_type': breakout_type}
        )

# Import random for simulation
import random

class TechnicalAlertService:
    """Advanced technical analysis alerts"""
    
    def __init__(self, stream: RealTimeDataStream):
        self.stream = stream
        self.market_fetcher = MarketDataFetcher()
        
        # Technical indicators cache
        self.indicator_cache: Dict[str, Dict] = {}
        self.cache_expiry = timedelta(minutes=5)
        
    def add_rsi_alert(self, symbol: str, oversold_threshold: float = 30, 
                     overbought_threshold: float = 70) -> List[str]:
        """Add RSI-based alerts"""
        alert_ids = []
        
        # Oversold alert
        alert_ids.append(
            self.stream.alert_manager.add_alert(
                symbol=symbol,
                alert_type=AlertType.TECHNICAL_SIGNAL,
                target_value=oversold_threshold,
                message=f"{symbol} RSI oversold (below {oversold_threshold})",
                metadata={'indicator': 'RSI', 'condition': 'oversold'}
            )
        )
        
        # Overbought alert
        alert_ids.append(
            self.stream.alert_manager.add_alert(
                symbol=symbol,
                alert_type=AlertType.TECHNICAL_SIGNAL,
                target_value=overbought_threshold,
                message=f"{symbol} RSI overbought (above {overbought_threshold})",
                metadata={'indicator': 'RSI', 'condition': 'overbought'}
            )
        )
        
        return alert_ids
    
    def add_moving_average_crossover_alert(self, symbol: str, 
                                         short_period: int = 20, 
                                         long_period: int = 50) -> str:
        """Add moving average crossover alert"""
        return self.stream.alert_manager.add_alert(
            symbol=symbol,
            alert_type=AlertType.TECHNICAL_SIGNAL,
            target_value=0,  # Will be calculated dynamically
            message=f"{symbol} MA crossover: {short_period} crosses {long_period}",
            metadata={
                'indicator': 'MA_CROSSOVER',
                'short_period': short_period,
                'long_period': long_period
            }
        )
    
    def check_technical_alerts(self, symbol: str):
        """Check technical indicator alerts for a symbol"""
        # This would be called periodically or on price updates
        # Implementation would calculate current technical indicators
        # and check against alert conditions
        pass
