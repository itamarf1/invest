import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging

from src.trading.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands trading strategy with squeeze detection"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, squeeze_threshold: float = 0.1,
                 min_confidence: float = 0.6, volume_confirm: bool = True):
        self.period = period
        self.std_dev = std_dev
        self.squeeze_threshold = squeeze_threshold
        self.min_confidence = min_confidence
        self.volume_confirm = volume_confirm
        
    def calculate_bollinger_bands(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        close = data['Close']
        
        # Simple moving average
        sma = close.rolling(window=self.period).mean()
        
        # Standard deviation
        std = close.rolling(window=self.period).std()
        
        # Upper and lower bands
        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)
        
        # Band width (for squeeze detection)
        band_width = (upper_band - lower_band) / sma
        
        # %B (position within bands)
        percent_b = (close - lower_band) / (upper_band - lower_band)
        
        return {
            'SMA': sma,
            'Upper_Band': upper_band,
            'Lower_Band': lower_band,
            'Band_Width': band_width,
            'Percent_B': percent_b,
            'Standard_Dev': std
        }
    
    def detect_squeeze(self, band_width: pd.Series, lookback: int = 20) -> List[int]:
        """Detect Bollinger Band squeeze (low volatility periods)"""
        squeezes = []
        
        for i in range(lookback, len(band_width)):
            if pd.isna(band_width.iloc[i]):
                continue
                
            # Check if current band width is near lowest in lookback period
            recent_widths = band_width.iloc[i-lookback:i+1]
            min_width = recent_widths.min()
            
            if (band_width.iloc[i] <= min_width * (1 + self.squeeze_threshold) and
                band_width.iloc[i] < recent_widths.mean() * 0.8):  # 20% below average
                squeezes.append(i)
        
        return squeezes
    
    def detect_band_touches(self, data: pd.DataFrame, upper_band: pd.Series, 
                           lower_band: pd.Series) -> Dict[str, List[int]]:
        """Detect price touches/penetrations of bands"""
        upper_touches = []
        lower_touches = []
        
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        for i in range(len(data)):
            if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                continue
                
            # Upper band touch (resistance)
            if (high.iloc[i] >= upper_band.iloc[i] * 0.995 or  # Within 0.5% of band
                close.iloc[i] >= upper_band.iloc[i]):
                upper_touches.append(i)
                
            # Lower band touch (support)
            elif (low.iloc[i] <= lower_band.iloc[i] * 1.005 or  # Within 0.5% of band
                  close.iloc[i] <= lower_band.iloc[i]):
                lower_touches.append(i)
        
        return {
            'upper': upper_touches,
            'lower': lower_touches
        }
    
    def detect_band_walk(self, close: pd.Series, upper_band: pd.Series, 
                        lower_band: pd.Series, min_periods: int = 3) -> Dict[str, List[int]]:
        """Detect band walking (strong trend continuation)"""
        upper_walks = []
        lower_walks = []
        
        for i in range(min_periods, len(close)):
            if any(pd.isna(val) for val in [close.iloc[i-j] for j in range(min_periods)] +
                  [upper_band.iloc[i-j] for j in range(min_periods)] +
                  [lower_band.iloc[i-j] for j in range(min_periods)]):
                continue
                
            # Upper band walk (bullish trend)
            if all(close.iloc[i-j] >= upper_band.iloc[i-j] * 0.98 
                  for j in range(min_periods)):
                upper_walks.append(i)
                
            # Lower band walk (bearish trend)  
            elif all(close.iloc[i-j] <= lower_band.iloc[i-j] * 1.02
                    for j in range(min_periods)):
                lower_walks.append(i)
        
        return {
            'upper': upper_walks,
            'lower': lower_walks
        }
    
    def confirm_with_volume(self, data: pd.DataFrame, index: int, 
                          signal_type: str) -> float:
        """Confirm signals with volume analysis"""
        if not self.volume_confirm or 'Volume' not in data.columns:
            return 1.0
            
        try:
            # Calculate average volume over last 20 periods
            volume_sma = data['Volume'].rolling(window=20).mean()
            current_volume = data['Volume'].iloc[index]
            avg_volume = volume_sma.iloc[index]
            
            if pd.isna(avg_volume) or avg_volume == 0:
                return 1.0
                
            volume_ratio = current_volume / avg_volume
            
            # Higher volume increases confidence
            if volume_ratio > 1.5:  # 50% above average
                return 1.2
            elif volume_ratio > 1.2:  # 20% above average  
                return 1.1
            elif volume_ratio < 0.5:  # Below average volume
                return 0.8
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger Bands trading signals"""
        if len(data) < self.period + 10:
            raise ValueError(f"Insufficient data. Need at least {self.period + 10} periods")
        
        data_copy = data.copy()
        
        # Calculate Bollinger Bands
        bb_data = self.calculate_bollinger_bands(data_copy)
        data_copy['BB_SMA'] = bb_data['SMA']
        data_copy['BB_Upper'] = bb_data['Upper_Band']
        data_copy['BB_Lower'] = bb_data['Lower_Band']
        data_copy['BB_Width'] = bb_data['Band_Width']
        data_copy['BB_PercentB'] = bb_data['Percent_B']
        
        # Initialize signal columns
        data_copy['Signal'] = 0
        data_copy['Action'] = 'HOLD'
        data_copy['Confidence'] = 0.5
        data_copy['Signal_Reason'] = ''
        
        # Detect patterns
        squeezes = self.detect_squeeze(data_copy['BB_Width'])
        band_touches = self.detect_band_touches(data_copy, data_copy['BB_Upper'], data_copy['BB_Lower'])
        band_walks = self.detect_band_walk(data_copy['Close'], data_copy['BB_Upper'], data_copy['BB_Lower'])
        
        # Generate signals
        for i in range(len(data_copy)):
            if pd.isna(data_copy['BB_Upper'].iloc[i]):
                continue
                
            current_price = data_copy['Close'].iloc[i]
            upper_band = data_copy['BB_Upper'].iloc[i]
            lower_band = data_copy['BB_Lower'].iloc[i]
            sma = data_copy['BB_SMA'].iloc[i]
            percent_b = data_copy['BB_PercentB'].iloc[i]
            
            volume_multiplier = self.confirm_with_volume(data_copy, i, 'any')
            
            # High confidence: Band walk (trend continuation)
            if i in band_walks['upper']:
                confidence = min(0.85 * volume_multiplier, 0.95)
                data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = 1
                data_copy.iloc[i, data_copy.columns.get_loc('Action')] = 'BUY'
                data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = confidence
                data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'Upper Band Walk (Trend)'
                
            elif i in band_walks['lower']:
                confidence = min(0.85 * volume_multiplier, 0.95)
                data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = -1
                data_copy.iloc[i, data_copy.columns.get_loc('Action')] = 'SELL'
                data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = confidence
                data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'Lower Band Walk (Trend)'
                
            # Medium confidence: Mean reversion after band touch
            elif i in band_touches['lower'] and percent_b < 0.1:  # Oversold
                # Check if price is moving back toward mean
                if i > 0 and data_copy['Close'].iloc[i] > data_copy['Close'].iloc[i-1]:
                    confidence = min(0.72 * volume_multiplier, 0.90)
                    data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = 1
                    data_copy.iloc[i, data_copy.columns.get_loc('Action')] = 'BUY'
                    data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = confidence
                    data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'Lower Band Bounce (Mean Reversion)'
                    
            elif i in band_touches['upper'] and percent_b > 0.9:  # Overbought
                # Check if price is moving back toward mean
                if i > 0 and data_copy['Close'].iloc[i] < data_copy['Close'].iloc[i-1]:
                    confidence = min(0.72 * volume_multiplier, 0.90)
                    data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = -1
                    data_copy.iloc[i, data_copy.columns.get_loc('Action')] = 'SELL'
                    data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = confidence
                    data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'Upper Band Rejection (Mean Reversion)'
            
            # Lower confidence: Squeeze breakout
            elif i in squeezes:
                # Wait for breakout direction
                if current_price > sma:
                    confidence = min(0.62 * volume_multiplier, 0.80)
                    data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = 1
                    data_copy.iloc[i, data_copy.columns.get_loc('Action')] = 'BUY'
                    data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = confidence
                    data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'Squeeze Bullish Breakout'
                    
                elif current_price < sma:
                    confidence = min(0.62 * volume_multiplier, 0.80)
                    data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = -1
                    data_copy.iloc[i, data_copy.columns.get_loc('Action')] = 'SELL'
                    data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = confidence
                    data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'Squeeze Bearish Breakout'
            
            # Very low confidence: %B extreme readings
            elif not pd.isna(percent_b):
                if percent_b < -0.1:  # Well below lower band
                    confidence = min(0.55 * volume_multiplier, 0.70)
                    data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = 1
                    data_copy.iloc[i, data_copy.columns.get_loc('Action')] = 'BUY'
                    data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = confidence
                    data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'Extreme Oversold'
                    
                elif percent_b > 1.1:  # Well above upper band
                    confidence = min(0.55 * volume_multiplier, 0.70)
                    data_copy.iloc[i, data_copy.columns.get_loc('Signal')] = -1
                    data_copy.iloc[i, data_copy.columns.get_loc('Action')] = 'SELL'
                    data_copy.iloc[i, data_copy.columns.get_loc('Confidence')] = confidence
                    data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = 'Extreme Overbought'
        
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
                'bb_upper': latest['BB_Upper'],
                'bb_lower': latest['BB_Lower'],
                'bb_sma': latest['BB_SMA'],
                'bb_width': latest['BB_Width'],
                'percent_b': latest['BB_PercentB'],
                'signal_reason': latest['Signal_Reason'],
                'timestamp': latest.name,
                'strategy': 'Bollinger_Bands',
                'parameters': {
                    'period': self.period,
                    'std_dev': self.std_dev,
                    'squeeze_threshold': self.squeeze_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating Bollinger Bands signal for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'confidence': 0.0,
                'price': data['Close'].iloc[-1] if not data.empty else 0,
                'error': str(e),
                'strategy': 'Bollinger_Bands'
            }
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """Backtest Bollinger Bands strategy"""
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
                    'bb_upper': row['BB_Upper'],
                    'bb_lower': row['BB_Lower'],
                    'percent_b': row['BB_PercentB']
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
                            'percent_b': row['BB_PercentB']
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
                        'percent_b': row['BB_PercentB']
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
                'strategy': 'Bollinger_Bands',
                'parameters': {
                    'period': self.period,
                    'std_dev': self.std_dev,
                    'squeeze_threshold': self.squeeze_threshold,
                    'min_confidence': self.min_confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Bollinger Bands backtest: {str(e)}")
            return {'error': str(e)}
