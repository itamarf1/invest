import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class MovingAverageStrategy:
    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window
        self.positions = {}
        self.signals_history = []
    
    def calculate_signals(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(data) < self.long_window:
            logger.warning(f"Not enough data for {symbol}. Need at least {self.long_window} days")
            return data
        
        data_copy = data.copy()
        
        data_copy['SMA_Short'] = data_copy['Close'].rolling(window=self.short_window).mean()
        data_copy['SMA_Long'] = data_copy['Close'].rolling(window=self.long_window).mean()
        
        data_copy['Signal'] = 0
        data_copy.iloc[self.short_window:, data_copy.columns.get_loc('Signal')] = np.where(
            data_copy['SMA_Short'].iloc[self.short_window:] > data_copy['SMA_Long'].iloc[self.short_window:], 1, 0
        )
        
        data_copy['Position'] = data_copy['Signal'].diff()
        
        data_copy['Action'] = 'HOLD'
        data_copy.loc[data_copy['Position'] == 1, 'Action'] = 'BUY'
        data_copy.loc[data_copy['Position'] == -1, 'Action'] = 'SELL'
        
        return data_copy
    
    def get_latest_signal(self, data: pd.DataFrame, symbol: str) -> Dict:
        signals_data = self.calculate_signals(data, symbol)
        
        if signals_data.empty or 'Action' not in signals_data.columns:
            return {
                'symbol': symbol,
                'action': SignalType.HOLD.value,
                'price': data['Close'].iloc[-1] if not data.empty else None,
                'timestamp': data.index[-1] if not data.empty else None,
                'confidence': 0.0
            }
        
        latest = signals_data.iloc[-1]
        
        signal = {
            'symbol': symbol,
            'action': latest['Action'],
            'price': latest['Close'],
            'timestamp': latest.name,
            'sma_short': latest.get('SMA_Short'),
            'sma_long': latest.get('SMA_Long'),
            'confidence': self._calculate_confidence(signals_data)
        }
        
        return signal
    
    def _calculate_confidence(self, data: pd.DataFrame) -> float:
        if len(data) < 2:
            return 0.5
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        if latest['Action'] == 'HOLD':
            return 0.5
        
        price_momentum = (latest['Close'] - prev['Close']) / prev['Close']
        sma_spread = abs(latest['SMA_Short'] - latest['SMA_Long']) / latest['SMA_Long']
        
        confidence = min(0.5 + abs(price_momentum) * 10 + sma_spread * 5, 1.0)
        
        return round(confidence, 2)
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        signals_data = self.calculate_signals(data, "BACKTEST")
        
        if signals_data.empty:
            return {'error': 'No data for backtesting'}
        
        capital = initial_capital
        shares = 0
        trades = []
        portfolio_values = []
        
        for i, row in signals_data.iterrows():
            if pd.isna(row['Position']) or row['Position'] == 0:
                portfolio_value = capital + shares * row['Close']
                portfolio_values.append({
                    'date': i,
                    'portfolio_value': portfolio_value,
                    'stock_price': row['Close']
                })
                continue
            
            if row['Position'] == 1 and capital > 0:  # Buy signal
                shares_to_buy = capital // row['Close']
                if shares_to_buy > 0:
                    cost = shares_to_buy * row['Close']
                    capital -= cost
                    shares += shares_to_buy
                    
                    trades.append({
                        'date': i,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': row['Close'],
                        'cost': cost
                    })
            
            elif row['Position'] == -1 and shares > 0:  # Sell signal
                revenue = shares * row['Close']
                capital += revenue
                
                trades.append({
                    'date': i,
                    'action': 'SELL',
                    'shares': shares,
                    'price': row['Close'],
                    'revenue': revenue
                })
                
                shares = 0
            
            portfolio_value = capital + shares * row['Close']
            portfolio_values.append({
                'date': i,
                'portfolio_value': portfolio_value,
                'stock_price': row['Close']
            })
        
        final_value = capital + shares * signals_data['Close'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        buy_hold_return = (signals_data['Close'].iloc[-1] - signals_data['Close'].iloc[0]) / signals_data['Close'].iloc[0] * 100
        
        return {
            'initial_capital': initial_capital,
            'final_value': round(final_value, 2),
            'total_return_pct': round(total_return, 2),
            'buy_hold_return_pct': round(buy_hold_return, 2),
            'outperformance': round(total_return - buy_hold_return, 2),
            'num_trades': len(trades),
            'trades': trades,
            'portfolio_values': portfolio_values
        }
    
    def get_portfolio_summary(self) -> Dict:
        return {
            'strategy': 'Moving Average Crossover',
            'short_window': self.short_window,
            'long_window': self.long_window,
            'active_positions': len(self.positions),
            'total_signals': len(self.signals_history)
        }
