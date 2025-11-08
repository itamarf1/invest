import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

from src.trading.strategies.base_strategy import BaseStrategy
from src.trading.strategies.moving_average import MovingAverageStrategy
from src.trading.strategies.rsi_strategy import RSIStrategy
from src.trading.strategies.macd_strategy import MACDStrategy
from src.trading.strategies.bollinger_strategy import BollingerBandsStrategy
from src.trading.strategies.sentiment_enhanced import SentimentEnhancedStrategy

logger = logging.getLogger(__name__)

class EnsembleStrategy(BaseStrategy):
    """Ensemble strategy combining multiple technical and sentiment signals"""
    
    def __init__(self, 
                 weights: Optional[Dict[str, float]] = None,
                 min_strategies_agree: int = 2,
                 min_ensemble_confidence: float = 0.65,
                 use_sentiment: bool = True):
        
        # Initialize individual strategies
        self.strategies = {
            'moving_average': MovingAverageStrategy(short_window=20, long_window=50),
            'rsi': RSIStrategy(rsi_period=14, overbought=70, oversold=30),
            'macd': MACDStrategy(fast_period=12, slow_period=26, signal_period=9),
            'bollinger': BollingerBandsStrategy(period=20, std_dev=2.0),
        }
        
        if use_sentiment:
            self.strategies['sentiment'] = SentimentEnhancedStrategy()
        
        # Default weights (can be optimized)
        self.weights = weights or {
            'moving_average': 0.2,
            'rsi': 0.25,
            'macd': 0.25,
            'bollinger': 0.2,
            'sentiment': 0.1 if use_sentiment else 0.0
        }
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        self.min_strategies_agree = min_strategies_agree
        self.min_ensemble_confidence = min_ensemble_confidence
        self.use_sentiment = use_sentiment
        
    def get_individual_signals(self, data: pd.DataFrame, symbol: str) -> Dict[str, Dict]:
        """Get signals from all individual strategies"""
        signals = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                signal = strategy.get_latest_signal(data, symbol)
                signals[strategy_name] = signal
                
            except Exception as e:
                logger.warning(f"Error getting {strategy_name} signal for {symbol}: {str(e)}")
                signals[strategy_name] = {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        return signals
    
    def calculate_ensemble_score(self, signals: Dict[str, Dict]) -> Tuple[float, Dict]:
        """Calculate weighted ensemble score and agreement metrics"""
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        
        strategy_votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        strategy_confidences = []
        
        for strategy_name, signal in signals.items():
            if 'error' in signal:
                continue
                
            action = signal.get('action', 'HOLD')
            confidence = signal.get('confidence', 0.0)
            weight = self.weights.get(strategy_name, 0.0)
            
            weighted_confidence = confidence * weight
            strategy_confidences.append(confidence)
            strategy_votes[action] += 1
            
            if action == 'BUY':
                buy_score += weighted_confidence
            elif action == 'SELL':
                sell_score += weighted_confidence
            else:  # HOLD
                hold_score += weighted_confidence
        
        # Calculate agreement metrics
        total_strategies = len([s for s in signals.values() if 'error' not in s])
        max_votes = max(strategy_votes.values()) if strategy_votes else 0
        agreement_ratio = max_votes / total_strategies if total_strategies > 0 else 0
        
        avg_confidence = np.mean(strategy_confidences) if strategy_confidences else 0.0
        
        agreement_metrics = {
            'strategy_votes': strategy_votes,
            'agreement_ratio': agreement_ratio,
            'avg_confidence': avg_confidence,
            'total_strategies': total_strategies
        }
        
        return (buy_score, sell_score, hold_score), agreement_metrics
    
    def determine_ensemble_action(self, scores: Tuple[float, float, float], 
                                 agreement_metrics: Dict) -> Tuple[str, float, str]:
        """Determine final ensemble action and confidence"""
        buy_score, sell_score, hold_score = scores
        
        # Find dominant action
        if buy_score > sell_score and buy_score > hold_score:
            base_action = 'BUY'
            base_confidence = buy_score
        elif sell_score > buy_score and sell_score > hold_score:
            base_action = 'SELL'
            base_confidence = sell_score
        else:
            base_action = 'HOLD'
            base_confidence = hold_score
        
        # Adjust confidence based on agreement
        agreement_bonus = agreement_metrics['agreement_ratio'] * 0.2  # Up to 20% bonus
        final_confidence = min(base_confidence + agreement_bonus, 1.0)
        
        # Check minimum agreement requirement
        max_votes = max(agreement_metrics['strategy_votes'].values())
        if max_votes < self.min_strategies_agree:
            final_action = 'HOLD'
            final_confidence = 0.5
            reason = f"Insufficient agreement ({max_votes}/{agreement_metrics['total_strategies']} strategies)"
        else:
            final_action = base_action
            reason = f"Ensemble {base_action} ({max_votes}/{agreement_metrics['total_strategies']} agree)"
        
        # Apply minimum confidence filter
        if final_confidence < self.min_ensemble_confidence and final_action != 'HOLD':
            final_action = 'HOLD'
            final_confidence = 0.5
            reason += " - Low confidence"
        
        return final_action, final_confidence, reason
    
    def get_latest_signal(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Get ensemble trading signal"""
        try:
            # Get individual signals
            individual_signals = self.get_individual_signals(data, symbol)
            
            # Calculate ensemble scores
            scores, agreement_metrics = self.calculate_ensemble_score(individual_signals)
            
            # Determine final action
            action, confidence, reason = self.determine_ensemble_action(scores, agreement_metrics)
            
            # Get current price
            current_price = data['Close'].iloc[-1] if not data.empty else 0
            
            # Calculate additional metrics
            signal_strength = max(scores) - min(scores)  # Spread between highest and lowest
            conviction = confidence * agreement_metrics['agreement_ratio']
            
            return {
                'symbol': symbol,
                'action': action,
                'confidence': confidence,
                'conviction': conviction,
                'price': current_price,
                'signal_reason': reason,
                'signal_strength': signal_strength,
                'timestamp': data.index[-1] if not data.empty else datetime.now(),
                'strategy': 'Ensemble',
                'individual_signals': individual_signals,
                'ensemble_scores': {
                    'buy_score': scores[0],
                    'sell_score': scores[1], 
                    'hold_score': scores[2]
                },
                'agreement_metrics': agreement_metrics,
                'parameters': {
                    'weights': self.weights,
                    'min_strategies_agree': self.min_strategies_agree,
                    'min_ensemble_confidence': self.min_ensemble_confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating ensemble signal for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'confidence': 0.0,
                'price': data['Close'].iloc[-1] if not data.empty else 0,
                'error': str(e),
                'strategy': 'Ensemble'
            }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ensemble signals for backtesting"""
        if len(data) < 50:  # Need sufficient data for all strategies
            raise ValueError("Insufficient data for ensemble strategy")
        
        data_copy = data.copy()
        
        # Initialize ensemble columns
        data_copy['Ensemble_Action'] = 'HOLD'
        data_copy['Ensemble_Confidence'] = 0.5
        data_copy['Ensemble_Signal'] = 0
        data_copy['Signal_Reason'] = ''
        data_copy['Buy_Score'] = 0.0
        data_copy['Sell_Score'] = 0.0
        data_copy['Agreement_Ratio'] = 0.0
        
        # Generate signals for recent periods (last 100 periods for efficiency)
        start_idx = max(0, len(data_copy) - 100)
        
        for i in range(start_idx, len(data_copy)):
            try:
                # Get data up to current point
                current_data = data_copy.iloc[:i+1]
                
                if len(current_data) < 50:  # Still not enough data
                    continue
                    
                # Get ensemble signal for current point
                signal_result = self.get_latest_signal(current_data, 'BACKTEST')
                
                # Update dataframe
                data_copy.iloc[i, data_copy.columns.get_loc('Ensemble_Action')] = signal_result['action']
                data_copy.iloc[i, data_copy.columns.get_loc('Ensemble_Confidence')] = signal_result['confidence']
                data_copy.iloc[i, data_copy.columns.get_loc('Signal_Reason')] = signal_result['signal_reason']
                data_copy.iloc[i, data_copy.columns.get_loc('Buy_Score')] = signal_result['ensemble_scores']['buy_score']
                data_copy.iloc[i, data_copy.columns.get_loc('Sell_Score')] = signal_result['ensemble_scores']['sell_score']
                data_copy.iloc[i, data_copy.columns.get_loc('Agreement_Ratio')] = signal_result['agreement_metrics']['agreement_ratio']
                
                if signal_result['action'] == 'BUY':
                    data_copy.iloc[i, data_copy.columns.get_loc('Ensemble_Signal')] = 1
                elif signal_result['action'] == 'SELL':
                    data_copy.iloc[i, data_copy.columns.get_loc('Ensemble_Signal')] = -1
                else:
                    data_copy.iloc[i, data_copy.columns.get_loc('Ensemble_Signal')] = 0
                    
            except Exception as e:
                logger.warning(f"Error generating ensemble signal at index {i}: {str(e)}")
                continue
        
        return data_copy
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """Backtest ensemble strategy"""
        try:
            signals = self.generate_signals(data)
            
            # Filter for actual trading signals
            trading_signals = signals[
                (signals['Ensemble_Signal'] != 0) & 
                (signals['Ensemble_Confidence'] >= self.min_ensemble_confidence)
            ]
            
            if trading_signals.empty:
                return {'error': 'No valid ensemble signals generated'}
            
            capital = initial_capital
            position = 0
            trades = []
            portfolio_values = []
            
            for i, row in trading_signals.iterrows():
                current_price = row['Close']
                signal = row['Ensemble_Signal']
                
                portfolio_value = capital + (position * current_price)
                portfolio_values.append({
                    'date': i,
                    'portfolio_value': portfolio_value,
                    'stock_price': current_price,
                    'position': position,
                    'cash': capital,
                    'buy_score': row['Buy_Score'],
                    'sell_score': row['Sell_Score'],
                    'agreement_ratio': row['Agreement_Ratio']
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
                            'confidence': row['Ensemble_Confidence'],
                            'agreement_ratio': row['Agreement_Ratio']
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
                        'confidence': row['Ensemble_Confidence'],
                        'agreement_ratio': row['Agreement_Ratio']
                    })
                    
                    position = 0
            
            # Final portfolio value
            final_price = data['Close'].iloc[-1]
            final_value = capital + (position * final_price)
            
            # Calculate performance metrics
            total_return = final_value - initial_capital
            total_return_pct = (total_return / initial_capital) * 100
            buy_hold_return = ((final_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
            
            # Calculate additional ensemble metrics
            avg_confidence = trading_signals['Ensemble_Confidence'].mean()
            avg_agreement = trading_signals['Agreement_Ratio'].mean()
            
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
                'strategy': 'Ensemble',
                'ensemble_metrics': {
                    'avg_confidence': avg_confidence,
                    'avg_agreement_ratio': avg_agreement,
                    'total_signals': len(trading_signals),
                    'strategies_used': list(self.strategies.keys())
                },
                'parameters': {
                    'weights': self.weights,
                    'min_strategies_agree': self.min_strategies_agree,
                    'min_ensemble_confidence': self.min_ensemble_confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble backtest: {str(e)}")
            return {'error': str(e)}
