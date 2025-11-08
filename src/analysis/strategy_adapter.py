import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class StrategyAdapter:
    """Adapter to convert strategy outputs to standard format for performance tracking"""
    
    @staticmethod
    def adapt_signals(strategy_output: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """Convert strategy DataFrame output to list of signal dictionaries"""
        try:
            signals = []
            
            if strategy_output is None or strategy_output.empty:
                return signals
            
            # Check if we have the expected columns
            required_cols = ['Signal', 'Strength']
            if not all(col in strategy_output.columns for col in required_cols):
                logger.warning(f"Strategy output missing required columns: {required_cols}")
                return signals
            
            for idx, row in strategy_output.iterrows():
                signal_value = row.get('Signal', 0)
                strength = row.get('Strength', 0.5)
                
                # Convert numeric signals to action strings
                if signal_value > 0:
                    action = 'buy'
                elif signal_value < 0:
                    action = 'sell'
                else:
                    action = 'hold'
                
                # Only include actual buy/sell signals
                if action in ['buy', 'sell']:
                    signals.append({
                        'date': idx,
                        'timestamp': idx,
                        'symbol': symbol,
                        'action': action,
                        'signal': action,
                        'strength': abs(strength),
                        'confidence': abs(strength),
                        'signal_value': signal_value,
                        'price': row.get('Close', 0),
                        'volume': row.get('Volume', 0)
                    })
            
            logger.info(f"Adapted {len(signals)} signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"Error adapting signals for {symbol}: {str(e)}")
            return []
    
    @staticmethod
    def create_mock_signals(data: pd.DataFrame, symbol: str, strategy_name: str) -> List[Dict[str, Any]]:
        """Create mock signals when strategy fails - for demo purposes"""
        try:
            signals = []
            
            if len(data) < 20:
                return signals
            
            # Create some realistic mock signals based on price movements
            closes = data['Close'].values
            
            # Generate signals every 10-20 days with some randomness
            np.random.seed(hash(strategy_name + symbol) % 1000)  # Consistent randomness per strategy/symbol
            
            signal_frequency = 15  # Average days between signals
            last_signal_idx = 0
            position = 0  # 0 = no position, 1 = long, -1 = short
            
            for i in range(20, len(data) - 1, signal_frequency):
                if i - last_signal_idx < 5:  # Minimum gap between signals
                    continue
                
                current_price = closes[i]
                prev_price = closes[i-10] if i >= 10 else closes[0]
                price_change = (current_price - prev_price) / prev_price
                
                # Simple strategy logic for realistic signals
                if strategy_name == 'rsi':
                    # RSI-like behavior - contrarian signals
                    if price_change < -0.05 and position <= 0:  # Oversold
                        action = 'buy'
                        strength = min(0.9, abs(price_change) * 10)
                    elif price_change > 0.05 and position > 0:  # Overbought
                        action = 'sell'
                        strength = min(0.9, abs(price_change) * 10)
                    else:
                        continue
                        
                elif strategy_name == 'macd':
                    # MACD-like behavior - trend following
                    if price_change > 0.02 and position <= 0:  # Bullish momentum
                        action = 'buy'
                        strength = min(0.8, abs(price_change) * 8)
                    elif price_change < -0.02 and position > 0:  # Bearish momentum
                        action = 'sell'
                        strength = min(0.8, abs(price_change) * 8)
                    else:
                        continue
                        
                elif strategy_name == 'bollinger_bands':
                    # Bollinger-like behavior - mean reversion
                    if price_change < -0.03 and position <= 0:  # Below lower band
                        action = 'buy'
                        strength = min(0.7, abs(price_change) * 12)
                    elif price_change > 0.03 and position > 0:  # Above upper band
                        action = 'sell'
                        strength = min(0.7, abs(price_change) * 12)
                    else:
                        continue
                        
                elif strategy_name == 'moving_average':
                    # MA crossover behavior
                    short_ma = np.mean(closes[max(0, i-5):i+1])
                    long_ma = np.mean(closes[max(0, i-20):i+1])
                    
                    if short_ma > long_ma and position <= 0:
                        action = 'buy'
                        strength = 0.6
                    elif short_ma < long_ma and position > 0:
                        action = 'sell'
                        strength = 0.6
                    else:
                        continue
                        
                else:  # ensemble or other
                    # Mixed approach
                    if price_change > 0.01 and np.random.random() > 0.7 and position <= 0:
                        action = 'buy'
                        strength = np.random.uniform(0.5, 0.8)
                    elif price_change < -0.01 and np.random.random() > 0.7 and position > 0:
                        action = 'sell'
                        strength = np.random.uniform(0.5, 0.8)
                    else:
                        continue
                
                # Update position
                if action == 'buy':
                    position = 1
                elif action == 'sell':
                    position = 0
                
                date_idx = data.index[i]
                signals.append({
                    'date': date_idx,
                    'timestamp': date_idx,
                    'symbol': symbol,
                    'action': action,
                    'signal': action,
                    'strength': strength,
                    'confidence': strength,
                    'signal_value': 1 if action == 'buy' else -1,
                    'price': current_price,
                    'volume': data.iloc[i].get('Volume', 0)
                })
                
                last_signal_idx = i
            
            logger.info(f"Created {len(signals)} mock signals for {strategy_name} on {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"Error creating mock signals for {symbol}: {str(e)}")
            return []