import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging

from src.trading.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class RSIStrategy(BaseStrategy):
    """RSI-based trading strategy with divergence detection"""
    
    def __init__(self, rsi_period: int = 14, overbought: float = 70, oversold: float = 30,
                 divergence_lookback: int = 20, min_confidence: float = 0.6):
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.divergence_lookback = divergence_lookback
        self.min_confidence = min_confidence
        
    def calculate_rsi(self, data: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate RSI with improved accuracy"""
        if period is None:
            period = self.rsi_period
            
        close = data['Close']
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def detect_bullish_divergence(self, data: pd.DataFrame, rsi: pd.Series) -> List[int]:
        """Detect bullish divergence: price makes lower lows while RSI makes higher lows"""
        divergences = []
        
        if len(data) < self.divergence_lookback * 2:
            return divergences
            
        for i in range(self.divergence_lookback, len(data) - self.divergence_lookback):
            # Look for local lows in price and RSI
            price_window = data['Low'].iloc[i-self.divergence_lookback:i+self.divergence_lookback+1]
            rsi_window = rsi.iloc[i-self.divergence_lookback:i+self.divergence_lookback+1]
            
            # Check if current point is a local low
            if (data['Low'].iloc[i] == price_window.min() and 
                len(price_window[price_window == price_window.min()]) == 1):
                
                # Look for previous low
                for j in range(max(0, i - self.divergence_lookback * 2), i - self.divergence_lookback):
                    prev_window = data['Low'].iloc[max(0, j-5):j+6]
                    
                    if (j < len(data) and data['Low'].iloc[j] == prev_window.min() and
                        len(prev_window[prev_window == prev_window.min()]) == 1):
                        
                        # Check for bullish divergence
                        if (data['Low'].iloc[i] < data['Low'].iloc[j] and
                            rsi.iloc[i] > rsi.iloc[j] and
                            rsi.iloc[i] < self.oversold + 10):  # Near oversold
                            divergences.append(i)
                            break
                            
        return divergences
    
    def detect_bearish_divergence(self, data: pd.DataFrame, rsi: pd.Series) -> List[int]:
        """Detect bearish divergence: price makes higher highs while RSI makes lower highs"""
        divergences = []
        
        if len(data) < self.divergence_lookback * 2:
            return divergences
            
        for i in range(self.divergence_lookback, len(data) - self.divergence_lookback):
            # Look for local highs in price and RSI
            price_window = data['High'].iloc[i-self.divergence_lookback:i+self.divergence_lookback+1]
            rsi_window = rsi.iloc[i-self.divergence_lookback:i+self.divergence_lookback+1]
            
            # Check if current point is a local high
            if (data['High'].iloc[i] == price_window.max() and 
                len(price_window[price_window == price_window.max()]) == 1):
                
                # Look for previous high
                for j in range(max(0, i - self.divergence_lookback * 2), i - self.divergence_lookback):
                    prev_window = data['High'].iloc[max(0, j-5):j+6]
                    
                    if (j < len(data) and data['High'].iloc[j] == prev_window.max() and
                        len(prev_window[prev_window == prev_window.max()]) == 1):
                        
                        # Check for bearish divergence
                        if (data['High'].iloc[i] > data['High'].iloc[j] and
                            rsi.iloc[i] < rsi.iloc[j] and
                            rsi.iloc[i] > self.overbought - 10):  # Near overbought
                            divergences.append(i)
                            break
                            
        return divergences
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI-based trading signals with divergence detection"""
        if len(data) < self.rsi_period + self.divergence_lookback:
            raise ValueError(f"Insufficient data. Need at least {self.rsi_period + self.divergence_lookback} periods")
        
        data_copy = data.copy()
        
        # Calculate RSI
        data_copy['RSI'] = self.calculate_rsi(data_copy)
        
        # Initialize signals
        data_copy['Signal'] = 0
        data_copy['RSI_Signal'] = 'HOLD'
        data_copy['Confidence'] = 0.5
        data_copy['Signal_Reason'] = ''
        
        # Basic RSI signals
        oversold_mask = data_copy['RSI'] < self.oversold
        overbought_mask = data_copy['RSI'] > self.overbought
        
        # Detect divergences
        bullish_divergences = self.detect_bullish_divergence(data_copy, data_copy['RSI'])
        bearish_divergences = self.detect_bearish_divergence(data_copy, data_copy['RSI'])
        
        # Generate signals
        for i in range(len(data_copy)):
            current_rsi = data_copy['RSI'].iloc[i]
            
            if pd.isna(current_rsi):
                continue
                
            # High confidence divergence signals
            if i in bullish_divergences:
                data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = 1
                data_copy.iloc[i, data_copy.columns.get_loc('RSI_Signal')] = 'BUY'
                data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = 0.85
                data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'Bullish Divergence'
                
            elif i in bearish_divergences:
                data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = -1
                data_copy.iloc[i, data_copy.columns.get_loc('RSI_Signal')] = 'SELL'
                data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = 0.85
                data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'Bearish Divergence'
                
            # Medium confidence basic RSI signals
            elif current_rsi < self.oversold:
                # Additional confirmation: check if RSI is trending up
                if i > 0 and data_copy['RSI'].iloc[i] > data_copy['RSI'].iloc[i-1]:
                    data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = 1
                    data_copy.iloc[i, data_copy.columns.get_loc('RSI_Signal')] = 'BUY'
                    data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = 0.7
                    data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'RSI Oversold Recovery'
                    
            elif current_rsi > self.overbought:
                # Additional confirmation: check if RSI is trending down
                if i > 0 and data_copy['RSI'].iloc[i] < data_copy['RSI'].iloc[i-1]:
                    data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = -1
                    data_copy.iloc[i, data_copy.columns.get_loc('RSI_Signal')] = 'SELL'
                    data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = 0.7
                    data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'RSI Overbought Decline'
            
            # Low confidence mean reversion signals
            elif 45 < current_rsi < 55:  # Neutral zone
                data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = 0
                data_copy.iloc[i, data_copy.columns.get_loc('RSI_Signal')] = 'HOLD'
                data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = 0.5
                data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'RSI Neutral'
        
        return data_copy
    
    def get_latest_signal(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Get the latest trading signal"""
        try:
            signals = self.generate_signals(data)
            latest = signals.iloc[-1]
            
            return {
                'symbol': symbol,
                'action': latest['RSI_Signal'],
                'confidence': latest['Confidence'],
                'price': latest['Close'],
                'rsi': latest['RSI'],
                'signal_reason': latest['Signal_Reason'],
                'timestamp': latest.name,
                'strategy': 'RSI',
                'parameters': {
                    'rsi_period': self.rsi_period,
                    'overbought': self.overbought,
                    'oversold': self.oversold
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating RSI signal for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'confidence': 0.0,
                'price': data['Close'].iloc[-1] if not data.empty else 0,
                'error': str(e),
                'strategy': 'RSI'
            }
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """Backtest RSI strategy"""
        try:
            signals = self.generate_signals(data)
            
            # Filter signals by minimum confidence
            valid_signals = signals[signals['Confidence'] >= self.min_confidence]
            
            if valid_signals.empty:
                return {'error': 'No valid signals generated with minimum confidence'}
            
            capital = initial_capital
            position = 0
            trades = []
            portfolio_values = []
            
            for i, row in valid_signals.iterrows():
                current_price = row['Close']
                signal = row['Signal']
                
                portfolio_value = capital + (position * current_price)
                portfolio_values.append({
                    'date': i,
                    'portfolio_value': portfolio_value,
                    'stock_price': current_price,
                    'position': position,
                    'cash': capital
                })
                
                if signal == 1 and position == 0:  # Buy signal
                    shares_to_buy = int(capital * 0.95 / current_price)  # Use 95% of capital
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        capital -= cost
                        position = shares_to_buy
                        
                        trades.append({
                            'date': i,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': cost,
                            'reason': row['Signal_Reason']
                        })
                        
                elif signal == -1 and position > 0:  # Sell signal
                    revenue = position * current_price
                    capital += revenue
                    
                    trades.append({
                        'date': i,
                        'action': 'SELL',
                        'shares': position,
                        'price': current_price,
                        'revenue': revenue,
                        'reason': row['Signal_Reason']
                    })
                    
                    position = 0
            
            # Final portfolio value
            final_price = data['Close'].iloc[-1]
            final_value = capital + (position * final_price)
            
            # Calculate returns
            total_return = final_value - initial_capital
            total_return_pct = (total_return / initial_capital) * 100
            
            # Buy and hold comparison
            buy_hold_return = ((final_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
            
            return {
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'buy_hold_return_pct': buy_hold_return,
                'outperformance': total_return_pct - buy_hold_return,
                'num_trades': len(trades),
                'trades': trades,
                'portfolio_values': portfolio_values,
                'strategy': 'RSI',
                'parameters': {
                    'rsi_period': self.rsi_period,
                    'overbought': self.overbought,
                    'oversold': self.oversold,
                    'min_confidence': self.min_confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Error in RSI backtest: {str(e)}")
            return {'error': str(e)}
