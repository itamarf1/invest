from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    @abstractmethod
    def get_latest_signal(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Get the latest trading signal for a given symbol
        
        Args:
            data: Historical price data
            symbol: Stock symbol
            
        Returns:
            Dict containing signal information with keys:
            - symbol: Stock symbol
            - action: 'BUY', 'SELL', or 'HOLD'
            - confidence: Float between 0 and 1
            - price: Current price
            - timestamp: Signal timestamp
            - strategy: Strategy name
        """
        pass
    
    @abstractmethod
    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000) -> Dict[str, Any]:
        """
        Backtest the strategy on historical data
        
        Args:
            data: Historical price data
            initial_capital: Starting capital for backtest
            
        Returns:
            Dict containing backtest results with keys:
            - initial_capital: Starting capital
            - final_value: Final portfolio value
            - total_return_pct: Total return percentage
            - num_trades: Number of trades executed
            - trades: List of trade records
        """
        pass
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for all data points (optional override)
        
        Args:
            data: Historical price data
            
        Returns:
            DataFrame with added signal columns
        """
        # Default implementation - can be overridden by strategies that need it
        return data.copy()