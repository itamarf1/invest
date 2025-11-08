import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging

from src.trading.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MACDStrategy(BaseStrategy):
    """MACD (Moving Average Convergence Divergence) trading strategy"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
                 min_confidence: float = 0.6, histogram_threshold: float = 0.0):
        self.fast_period = fast_period
        self.slow_period = slow_period  
        self.signal_period = signal_period
        self.min_confidence = min_confidence
        self.histogram_threshold = histogram_threshold
        
    def calculate_macd(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD line, signal line, and histogram"""
        close = data['Close']
        
        # Calculate exponential moving averages
        ema_fast = close.ewm(span=self.fast_period).mean()
        ema_slow = close.ewm(span=self.slow_period).mean()
        
        # MACD line = Fast EMA - Slow EMA
        macd_line = ema_fast - ema_slow
        
        # Signal line = EMA of MACD line
        signal_line = macd_line.ewm(span=self.signal_period).mean()
        
        # Histogram = MACD line - Signal line
        histogram = macd_line - signal_line
        
        return {
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram,
            'EMA_Fast': ema_fast,
            'EMA_Slow': ema_slow
        }
    
    def detect_zero_line_cross(self, macd: pd.Series) -> Dict[str, List[int]]:
        """Detect MACD zero line crossovers"""
        bullish_crosses = []
        bearish_crosses = []
        
        for i in range(1, len(macd)):
            if pd.isna(macd.iloc[i]) or pd.isna(macd.iloc[i-1]):
                continue
                
            # Bullish: MACD crosses above zero
            if macd.iloc[i-1] <= 0 and macd.iloc[i] > 0:
                bullish_crosses.append(i)
                
            # Bearish: MACD crosses below zero  
            elif macd.iloc[i-1] >= 0 and macd.iloc[i] < 0:
                bearish_crosses.append(i)
        
        return {
            'bullish': bullish_crosses,
            'bearish': bearish_crosses
        }
    
    def detect_signal_line_cross(self, macd: pd.Series, signal: pd.Series) -> Dict[str, List[int]]:
        """Detect MACD signal line crossovers"""
        bullish_crosses = []
        bearish_crosses = []
        
        for i in range(1, len(macd)):
            if pd.isna(macd.iloc[i]) or pd.isna(signal.iloc[i]) or pd.isna(macd.iloc[i-1]) or pd.isna(signal.iloc[i-1]):
                continue
                
            # Bullish: MACD crosses above signal line
            if macd.iloc[i-1] <= signal.iloc[i-1] and macd.iloc[i] > signal.iloc[i]:
                bullish_crosses.append(i)
                
            # Bearish: MACD crosses below signal line
            elif macd.iloc[i-1] >= signal.iloc[i-1] and macd.iloc[i] < signal.iloc[i]:
                bearish_crosses.append(i)
        
        return {
            'bullish': bullish_crosses,
            'bearish': bearish_crosses
        }
    
    def detect_histogram_reversal(self, histogram: pd.Series, lookback: int = 3) -> Dict[str, List[int]]:
        """Detect histogram trend reversals"""
        bullish_reversals = []
        bearish_reversals = []
        
        for i in range(lookback, len(histogram)):
            if any(pd.isna(histogram.iloc[i-j]) for j in range(lookback + 1)):
                continue
                
            # Check for bullish reversal (histogram was declining, now increasing)
            recent_values = histogram.iloc[i-lookback:i+1].values
            
            # Bullish: histogram is negative but increasing (momentum slowing)
            if (histogram.iloc[i] < 0 and 
                all(recent_values[j] < recent_values[j+1] for j in range(len(recent_values)-1)) and
                len(recent_values) >= 2):
                bullish_reversals.append(i)
                
            # Bearish: histogram is positive but decreasing (momentum slowing)
            elif (histogram.iloc[i] > 0 and 
                  all(recent_values[j] > recent_values[j+1] for j in range(len(recent_values)-1)) and
                  len(recent_values) >= 2):
                bearish_reversals.append(i)
        
        return {
            'bullish': bullish_reversals,
            'bearish': bearish_reversals
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MACD-based trading signals"""
        if len(data) < self.slow_period + self.signal_period:
            raise ValueError(f"Insufficient data. Need at least {self.slow_period + self.signal_period} periods")
        
        data_copy = data.copy()
        
        # Calculate MACD components
        macd_data = self.calculate_macd(data_copy)
        data_copy['MACD'] = macd_data['MACD']
        data_copy['MACD_Signal'] = macd_data['Signal']
        data_copy['MACD_Histogram'] = macd_data['Histogram']
        data_copy['EMA_Fast'] = macd_data['EMA_Fast']
        data_copy['EMA_Slow'] = macd_data['EMA_Slow']
        
        # Initialize signal columns
        data_copy['Signal'] = 0
        data_copy['Action'] = 'HOLD'
        data_copy['Confidence'] = 0.5
        data_copy['Signal_Reason'] = ''
        
        # Detect crossovers and reversals
        zero_crosses = self.detect_zero_line_cross(data_copy['MACD'])
        signal_crosses = self.detect_signal_line_cross(data_copy['MACD'], data_copy['MACD_Signal'])
        histogram_reversals = self.detect_histogram_reversal(data_copy['MACD_Histogram'])
        
        # Generate signals based on different patterns
        for i in range(len(data_copy)):
            if pd.isna(data_copy['MACD'].iloc[i]):
                continue
                
            # High confidence: MACD zero line crossover
            if i in zero_crosses['bullish']:
                data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = 1
                data_copy.iloc[i, data_copy.columns.get_loc('Action')] = 'BUY'
                data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = 0.85
                data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'MACD Bullish Zero Cross'
                
            elif i in zero_crosses['bearish']:
                data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = -1
                data_copy.iloc[i, data_copy.columns.get_loc('Action')] = 'SELL'
                data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = 0.85
                data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'MACD Bearish Zero Cross'
                
            # Medium confidence: Signal line crossover
            elif i in signal_crosses['bullish']:
                # Extra confirmation: histogram should be increasing
                conf = 0.75 if i in histogram_reversals['bullish'] else 0.65
                
                data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = 1
                data_copy.iloc[i, data_copy.columns.get_loc('Action')] = 'BUY'
                data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = conf
                data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'MACD Signal Line Cross Up'
                
            elif i in signal_crosses['bearish']:
                # Extra confirmation: histogram should be decreasing
                conf = 0.75 if i in histogram_reversals['bearish'] else 0.65
                
                data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = -1
                data_copy.iloc[i, data_copy.columns.get_loc('Action')] = 'SELL'
                data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = conf
                data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'MACD Signal Line Cross Down'
                
            # Lower confidence: Histogram reversal only
            elif i in histogram_reversals['bullish']:
                data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = 1
                data_copy.iloc[i, data_copy.columns.get_loc('Action')] = 'BUY'
                data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = 0.55
                data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'MACD Histogram Reversal Up'
                
            elif i in histogram_reversals['bearish']:
                data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = -1
                data_copy.iloc[i, data_copy.columns.get_loc('Action')] = 'SELL'
                data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = 0.55
                data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'MACD Histogram Reversal Down'
        
        return data_copy
    
    def get_latest_signal(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Get the latest trading signal"""
        try:
            signals = self.generate_signals(data)
            latest = signals.iloc[-1]
            
            return {
                'symbol': symbol,
                'action': latest['Action'],
                'confidence': latest['Confidence'],
                'price': latest['Close'],
                'macd': latest['MACD'],
                'macd_signal': latest['MACD_Signal'],
                'macd_histogram': latest['MACD_Histogram'],
                'signal_reason': latest['Signal_Reason'],
                'timestamp': latest.name,
                'strategy': 'MACD',
                'parameters': {
                    'fast_period': self.fast_period,
                    'slow_period': self.slow_period,
                    'signal_period': self.signal_period
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating MACD signal for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'confidence': 0.0,
                'price': data['Close'].iloc[-1] if not data.empty else 0,
                'error': str(e),
                'strategy': 'MACD'
            }
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """Backtest MACD strategy"""
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
                    'cash': capital,
                    'macd': row['MACD'],
                    'macd_signal': row['MACD_Signal']
                })
                
                if signal == 1 and position == 0:  # Buy signal
                    shares_to_buy = int(capital * 0.95 / current_price)
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
                            'reason': row['Signal_Reason'],
                            'macd': row['MACD']
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
                        'reason': row['Signal_Reason'],
                        'macd': row['MACD']
                    })
                    
                    position = 0
            
            # Final portfolio value
            final_price = data['Close'].iloc[-1]
            final_value = capital + (position * final_price)
            
            # Calculate performance metrics
            total_return = final_value - initial_capital
            total_return_pct = (total_return / initial_capital) * 100
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
                'strategy': 'MACD',
                'parameters': {
                    'fast_period': self.fast_period,
                    'slow_period': self.slow_period,
                    'signal_period': self.signal_period,
                    'min_confidence': self.min_confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Error in MACD backtest: {str(e)}")
            return {'error': str(e)}
