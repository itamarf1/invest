import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from src.crypto.crypto_data import CryptoDataFetcher, CryptoPriceData

logger = logging.getLogger(__name__)

class CryptoSignal(Enum):
    BUY = "buy"
    SELL = "sell" 
    HOLD = "hold"

@dataclass
class CryptoTradingSignal:
    symbol: str
    signal: CryptoSignal
    confidence: float  # 0-1
    price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class CryptoMomentumStrategy:
    """Momentum-based crypto trading strategy"""
    
    def __init__(self, lookback_days: int = 14, momentum_threshold: float = 0.05):
        self.lookback_days = lookback_days
        self.momentum_threshold = momentum_threshold
        self.crypto_fetcher = CryptoDataFetcher()
    
    def analyze_momentum(self, symbol: str) -> CryptoTradingSignal:
        """Analyze momentum for a cryptocurrency"""
        try:
            # Get historical data
            historical_data = self.crypto_fetcher.get_crypto_historical_data(symbol, self.lookback_days)
            current_price_data = self.crypto_fetcher.get_crypto_price(symbol)
            
            if historical_data.empty or not current_price_data:
                return CryptoTradingSignal(
                    symbol=symbol,
                    signal=CryptoSignal.HOLD,
                    confidence=0.0,
                    price=0,
                    reasoning="Insufficient data"
                )
            
            current_price = current_price_data.price
            
            # Calculate momentum indicators
            prices = historical_data['price']
            
            # Price momentum (% change from N days ago)
            price_momentum = (current_price - prices.iloc[0]) / prices.iloc[0]
            
            # Volume momentum (if available)
            volume_momentum = 0
            if 'volume' in historical_data.columns:
                recent_volume = historical_data['volume'].tail(3).mean()
                past_volume = historical_data['volume'].head(3).mean()
                if past_volume > 0:
                    volume_momentum = (recent_volume - past_volume) / past_volume
            
            # RSI calculation
            rsi = self._calculate_rsi(prices)
            
            # Moving average trend
            ma_short = prices.tail(5).mean()
            ma_long = prices.tail(14).mean()
            ma_trend = (ma_short - ma_long) / ma_long
            
            # Combine signals
            momentum_score = (
                price_momentum * 0.4 +
                volume_momentum * 0.2 +
                ma_trend * 0.3 +
                (50 - rsi) / 100 * 0.1  # RSI divergence from 50
            )
            
            confidence = min(abs(momentum_score) * 2, 1.0)
            
            # Generate signal
            if momentum_score > self.momentum_threshold:
                signal = CryptoSignal.BUY
                target_price = current_price * 1.1
                stop_loss = current_price * 0.95
                reasoning = f"Strong upward momentum ({momentum_score:.3f}), RSI: {rsi:.1f}"
            elif momentum_score < -self.momentum_threshold:
                signal = CryptoSignal.SELL
                target_price = current_price * 0.9
                stop_loss = current_price * 1.05
                reasoning = f"Strong downward momentum ({momentum_score:.3f}), RSI: {rsi:.1f}"
            else:
                signal = CryptoSignal.HOLD
                target_price = None
                stop_loss = None
                reasoning = f"Neutral momentum ({momentum_score:.3f}), RSI: {rsi:.1f}"
            
            return CryptoTradingSignal(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error analyzing momentum for {symbol}: {str(e)}")
            return CryptoTradingSignal(
                symbol=symbol,
                signal=CryptoSignal.HOLD,
                confidence=0.0,
                price=0,
                reasoning=f"Error: {str(e)}"
            )
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0

class CryptoMeanReversionStrategy:
    """Mean reversion strategy for crypto trading"""
    
    def __init__(self, lookback_days: int = 20, std_threshold: float = 2.0):
        self.lookback_days = lookback_days
        self.std_threshold = std_threshold
        self.crypto_fetcher = CryptoDataFetcher()
    
    def analyze_mean_reversion(self, symbol: str) -> CryptoTradingSignal:
        """Analyze mean reversion opportunity"""
        try:
            historical_data = self.crypto_fetcher.get_crypto_historical_data(symbol, self.lookback_days)
            current_price_data = self.crypto_fetcher.get_crypto_price(symbol)
            
            if historical_data.empty or not current_price_data:
                return CryptoTradingSignal(
                    symbol=symbol,
                    signal=CryptoSignal.HOLD,
                    confidence=0.0,
                    price=0,
                    reasoning="Insufficient data"
                )
            
            current_price = current_price_data.price
            prices = historical_data['price']
            
            # Calculate Bollinger Bands
            mean_price = prices.mean()
            std_price = prices.std()
            
            upper_band = mean_price + (self.std_threshold * std_price)
            lower_band = mean_price - (self.std_threshold * std_price)
            
            # Z-score (how many std devs from mean)
            z_score = (current_price - mean_price) / std_price
            
            confidence = min(abs(z_score) / self.std_threshold, 1.0)
            
            # Generate signal
            if current_price <= lower_band:
                # Price is oversold, expect reversion upward
                signal = CryptoSignal.BUY
                target_price = mean_price
                stop_loss = current_price * 0.95
                reasoning = f"Oversold - price below lower band (z-score: {z_score:.2f})"
            elif current_price >= upper_band:
                # Price is overbought, expect reversion downward
                signal = CryptoSignal.SELL
                target_price = mean_price
                stop_loss = current_price * 1.05
                reasoning = f"Overbought - price above upper band (z-score: {z_score:.2f})"
            else:
                signal = CryptoSignal.HOLD
                target_price = None
                stop_loss = None
                reasoning = f"Price within normal range (z-score: {z_score:.2f})"
            
            return CryptoTradingSignal(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error analyzing mean reversion for {symbol}: {str(e)}")
            return CryptoTradingSignal(
                symbol=symbol,
                signal=CryptoSignal.HOLD,
                confidence=0.0,
                price=0,
                reasoning=f"Error: {str(e)}"
            )

class CryptoCorrelationStrategy:
    """Strategy based on crypto correlations and market leadership"""
    
    def __init__(self):
        self.crypto_fetcher = CryptoDataFetcher()
        self.major_cryptos = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']
    
    def analyze_correlation_signal(self, symbol: str) -> CryptoTradingSignal:
        """Analyze signal based on major crypto correlations"""
        try:
            # Get price data for major cryptos
            crypto_data = self.crypto_fetcher.get_multiple_crypto_prices(self.major_cryptos + [symbol])
            
            if symbol.upper() not in crypto_data:
                return CryptoTradingSignal(
                    symbol=symbol,
                    signal=CryptoSignal.HOLD,
                    confidence=0.0,
                    price=0,
                    reasoning="Symbol data not available"
                )
            
            current_crypto = crypto_data[symbol.upper()]
            
            # Calculate market momentum (based on major cryptos)
            market_changes = []
            for major_symbol in self.major_cryptos:
                if major_symbol in crypto_data:
                    market_changes.append(crypto_data[major_symbol].change_24h_pct)
            
            if not market_changes:
                market_momentum = 0
            else:
                market_momentum = np.mean(market_changes)
            
            # Compare symbol performance to market
            symbol_performance = current_crypto.change_24h_pct
            relative_performance = symbol_performance - market_momentum
            
            # Bitcoin dominance analysis
            btc_change = crypto_data.get('BTC', CryptoPriceData('BTC', 0, 0, 0, 0, 0, datetime.now())).change_24h_pct
            
            confidence = min(abs(relative_performance) / 5, 1.0)  # Scale by 5% difference
            
            # Generate signal
            if market_momentum > 2 and relative_performance > 1:
                # Market is up and symbol is outperforming
                signal = CryptoSignal.BUY
                target_price = current_crypto.price * 1.08
                stop_loss = current_crypto.price * 0.95
                reasoning = f"Market bullish ({market_momentum:.1f}%), outperforming by {relative_performance:.1f}%"
            elif market_momentum < -2 and relative_performance < -1:
                # Market is down and symbol is underperforming
                signal = CryptoSignal.SELL
                target_price = current_crypto.price * 0.92
                stop_loss = current_crypto.price * 1.05
                reasoning = f"Market bearish ({market_momentum:.1f}%), underperforming by {relative_performance:.1f}%"
            elif abs(relative_performance) > 5:
                # Strong relative performance (divergence)
                if relative_performance > 0:
                    signal = CryptoSignal.BUY
                    reasoning = f"Strong outperformance (+{relative_performance:.1f}% vs market)"
                else:
                    signal = CryptoSignal.SELL
                    reasoning = f"Strong underperformance ({relative_performance:.1f}% vs market)"
                target_price = None
                stop_loss = None
            else:
                signal = CryptoSignal.HOLD
                target_price = None
                stop_loss = None
                reasoning = f"Following market trend (market: {market_momentum:.1f}%, relative: {relative_performance:.1f}%)"
            
            return CryptoTradingSignal(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                price=current_crypto.price,
                target_price=target_price,
                stop_loss=stop_loss,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error analyzing correlation signal for {symbol}: {str(e)}")
            return CryptoTradingSignal(
                symbol=symbol,
                signal=CryptoSignal.HOLD,
                confidence=0.0,
                price=0,
                reasoning=f"Error: {str(e)}"
            )

class CryptoDCAStrategy:
    """Dollar Cost Averaging strategy for crypto"""
    
    def __init__(self, target_allocation: Dict[str, float], frequency: str = "weekly"):
        self.target_allocation = target_allocation  # {'BTC': 0.4, 'ETH': 0.3, 'ADA': 0.3}
        self.frequency = frequency
        self.crypto_fetcher = CryptoDataFetcher()
    
    def get_dca_signals(self, portfolio_value: float, current_holdings: Dict[str, float]) -> List[CryptoTradingSignal]:
        """Generate DCA buy/sell signals to maintain target allocation"""
        try:
            signals = []
            
            # Get current prices
            symbols = list(self.target_allocation.keys())
            crypto_data = self.crypto_fetcher.get_multiple_crypto_prices(symbols)
            
            # Calculate current allocation
            total_current_value = 0
            current_values = {}
            
            for symbol in symbols:
                if symbol in current_holdings and symbol in crypto_data:
                    current_value = current_holdings[symbol] * crypto_data[symbol].price
                    current_values[symbol] = current_value
                    total_current_value += current_value
                else:
                    current_values[symbol] = 0
            
            # Add cash to total portfolio value
            total_portfolio = total_current_value + portfolio_value
            
            # Generate rebalancing signals
            for symbol, target_weight in self.target_allocation.items():
                if symbol not in crypto_data:
                    continue
                
                crypto = crypto_data[symbol]
                current_value = current_values[symbol]
                target_value = total_portfolio * target_weight
                value_difference = target_value - current_value
                
                # Only trade if difference is significant (> 5% of target)
                if abs(value_difference) > target_value * 0.05:
                    if value_difference > 0:
                        # Need to buy more
                        signal = CryptoTradingSignal(
                            symbol=symbol,
                            signal=CryptoSignal.BUY,
                            confidence=0.8,
                            price=crypto.price,
                            reasoning=f"DCA rebalancing - need ${value_difference:.0f} more to reach target allocation"
                        )
                    else:
                        # Need to sell some
                        signal = CryptoTradingSignal(
                            symbol=symbol,
                            signal=CryptoSignal.SELL,
                            confidence=0.8,
                            price=crypto.price,
                            reasoning=f"DCA rebalancing - need to sell ${abs(value_difference):.0f} to reach target allocation"
                        )
                    
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating DCA signals: {str(e)}")
            return []

class CryptoVolatilityStrategy:
    """Strategy based on volatility analysis"""
    
    def __init__(self, lookback_days: int = 30, vol_threshold_high: float = 0.8, vol_threshold_low: float = 0.3):
        self.lookback_days = lookback_days
        self.vol_threshold_high = vol_threshold_high  # High volatility threshold
        self.vol_threshold_low = vol_threshold_low    # Low volatility threshold
        self.crypto_fetcher = CryptoDataFetcher()
    
    def analyze_volatility_signal(self, symbol: str) -> CryptoTradingSignal:
        """Generate signal based on volatility analysis"""
        try:
            historical_data = self.crypto_fetcher.get_crypto_historical_data(symbol, self.lookback_days)
            current_price_data = self.crypto_fetcher.get_crypto_price(symbol)
            
            if historical_data.empty or not current_price_data:
                return CryptoTradingSignal(
                    symbol=symbol,
                    signal=CryptoSignal.HOLD,
                    confidence=0.0,
                    price=0,
                    reasoning="Insufficient data"
                )
            
            current_price = current_price_data.price
            prices = historical_data['price']
            
            # Calculate daily returns
            returns = prices.pct_change().dropna()
            
            # Calculate volatility metrics
            volatility = returns.std() * np.sqrt(365)  # Annualized volatility
            recent_volatility = returns.tail(7).std() * np.sqrt(365)  # Recent 7-day volatility
            volatility_trend = recent_volatility - volatility
            
            # Calculate VIX-like indicator for crypto
            high_low_spread = (prices.rolling(window=14).max() - prices.rolling(window=14).min()) / prices.rolling(window=14).mean()
            current_spread = high_low_spread.iloc[-1]
            
            confidence = min(abs(volatility - 0.5) * 2, 1.0)
            
            # Generate signal based on volatility regime
            if volatility > self.vol_threshold_high and volatility_trend > 0:
                # High and increasing volatility - expect mean reversion
                signal = CryptoSignal.SELL if current_price_data.change_24h_pct > 5 else CryptoSignal.BUY
                reasoning = f"High volatility regime ({volatility:.1%}) - contrarian signal"
                target_price = current_price * (0.95 if signal == CryptoSignal.SELL else 1.05)
                stop_loss = current_price * (1.1 if signal == CryptoSignal.SELL else 0.9)
            elif volatility < self.vol_threshold_low:
                # Low volatility - expect breakout
                signal = CryptoSignal.BUY if current_price_data.change_24h_pct > 2 else CryptoSignal.HOLD
                reasoning = f"Low volatility ({volatility:.1%}) - potential breakout setup"
                target_price = current_price * 1.1
                stop_loss = current_price * 0.95
            else:
                # Normal volatility
                signal = CryptoSignal.HOLD
                reasoning = f"Normal volatility regime ({volatility:.1%})"
                target_price = None
                stop_loss = None
            
            return CryptoTradingSignal(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error analyzing volatility for {symbol}: {str(e)}")
            return CryptoTradingSignal(
                symbol=symbol,
                signal=CryptoSignal.HOLD,
                confidence=0.0,
                price=0,
                reasoning=f"Error: {str(e)}"
            )
