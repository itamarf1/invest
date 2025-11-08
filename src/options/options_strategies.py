import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import math

from src.options.options_data import OptionContract, OptionsChain, OptionType, OptionStrategy, OptionsDataFetcher

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Types of options strategies"""
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    COLLAR = "collar"
    CALENDAR_SPREAD = "calendar_spread"

@dataclass
class StrategyLeg:
    contract: OptionContract
    quantity: int  # Positive for buy, negative for sell
    action: str   # "buy" or "sell"
    cost: float   # Total cost (negative for premium received)

@dataclass
class StrategyAnalysis:
    strategy: OptionStrategy
    current_pnl: float
    break_even_probability: float
    max_profit_probability: float
    risk_metrics: Dict[str, float]
    recommendations: List[str]
    exit_signals: List[str]

class OptionsStrategist:
    """Options strategies analyzer and builder"""
    
    def __init__(self):
        self.options_fetcher = OptionsDataFetcher()
        self.commission_per_contract = 0.65  # Standard broker commission
    
    def create_covered_call(self, symbol: str, shares_owned: int, 
                          target_strike_pct: float = 0.05) -> Optional[OptionStrategy]:
        """Create a covered call strategy"""
        try:
            options_chain = self.options_fetcher.get_options_chain(symbol)
            if not options_chain:
                return None
            
            current_price = options_chain.underlying_price
            target_strike = current_price * (1 + target_strike_pct)
            
            # Find call option closest to target strike
            best_call = None
            min_diff = float('inf')
            
            for strike, contract in options_chain.calls.items():
                if strike >= target_strike:
                    diff = abs(strike - target_strike)
                    if diff < min_diff:
                        min_diff = diff
                        best_call = contract
            
            if not best_call:
                logger.error("No suitable call option found for covered call")
                return None
            
            # Calculate number of contracts (1 contract = 100 shares)
            contracts = min(shares_owned // 100, 10)  # Limit to 10 contracts
            if contracts == 0:
                logger.error("Insufficient shares for covered call (need at least 100)")
                return None
            
            # Strategy legs
            legs = [
                {
                    "contract": best_call,
                    "quantity": -contracts,  # Sell calls
                    "action": "sell",
                    "cost": -best_call.price * contracts * 100  # Premium received
                }
            ]
            
            # Calculate P&L at different prices
            pnl_range = {}
            prices = np.linspace(current_price * 0.8, current_price * 1.2, 50)
            
            for price in prices:
                # Stock P&L (on covered shares)
                stock_pnl = (price - current_price) * contracts * 100
                
                # Option P&L
                if price <= best_call.strike:
                    option_pnl = best_call.price * contracts * 100  # Keep premium
                else:
                    option_pnl = best_call.price * contracts * 100 - (price - best_call.strike) * contracts * 100
                
                total_pnl = stock_pnl + option_pnl - (self.commission_per_contract * contracts)
                pnl_range[price] = total_pnl
            
            # Calculate metrics
            max_profit = best_call.price * contracts * 100 + (best_call.strike - current_price) * contracts * 100
            max_loss = float('-inf')  # Theoretically unlimited loss (if stock goes to 0)
            breakeven = current_price - best_call.price
            
            return OptionStrategy(
                name="Covered Call",
                description=f"Sell {contracts} call contracts at ${best_call.strike:.2f} strike",
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                profit_loss_range=pnl_range,
                total_premium=best_call.price * contracts * 100,
                risk_reward_ratio=float('inf') if max_profit > 0 else 0
            )
            
        except Exception as e:
            logger.error(f"Error creating covered call for {symbol}: {str(e)}")
            return None
    
    def create_protective_put(self, symbol: str, shares_owned: int,
                            protection_level: float = 0.05) -> Optional[OptionStrategy]:
        """Create a protective put strategy"""
        try:
            options_chain = self.options_fetcher.get_options_chain(symbol)
            if not options_chain:
                return None
            
            current_price = options_chain.underlying_price
            target_strike = current_price * (1 - protection_level)
            
            # Find put option closest to target strike
            best_put = None
            min_diff = float('inf')
            
            for strike, contract in options_chain.puts.items():
                if strike <= target_strike:
                    diff = abs(strike - target_strike)
                    if diff < min_diff:
                        min_diff = diff
                        best_put = contract
            
            if not best_put:
                logger.error("No suitable put option found for protective put")
                return None
            
            # Calculate number of contracts
            contracts = min(shares_owned // 100, 10)
            if contracts == 0:
                logger.error("Insufficient shares for protective put (need at least 100)")
                return None
            
            # Strategy legs
            legs = [
                {
                    "contract": best_put,
                    "quantity": contracts,  # Buy puts
                    "action": "buy", 
                    "cost": best_put.price * contracts * 100  # Premium paid
                }
            ]
            
            # Calculate P&L at different prices
            pnl_range = {}
            prices = np.linspace(current_price * 0.6, current_price * 1.4, 50)
            
            for price in prices:
                # Stock P&L
                stock_pnl = (price - current_price) * contracts * 100
                
                # Put P&L
                if price >= best_put.strike:
                    put_pnl = -best_put.price * contracts * 100  # Lose premium
                else:
                    put_pnl = (best_put.strike - price) * contracts * 100 - best_put.price * contracts * 100
                
                total_pnl = stock_pnl + put_pnl - (self.commission_per_contract * contracts)
                pnl_range[price] = total_pnl
            
            # Calculate metrics
            max_profit = float('inf')  # Unlimited upside
            max_loss = (current_price - best_put.strike) * contracts * 100 + best_put.price * contracts * 100
            breakeven = current_price + best_put.price
            
            return OptionStrategy(
                name="Protective Put",
                description=f"Buy {contracts} put contracts at ${best_put.strike:.2f} strike for protection",
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                profit_loss_range=pnl_range,
                total_premium=-best_put.price * contracts * 100,  # Cost (negative)
                risk_reward_ratio=float('inf')
            )
            
        except Exception as e:
            logger.error(f"Error creating protective put for {symbol}: {str(e)}")
            return None
    
    def create_bull_call_spread(self, symbol: str, risk_amount: float = 1000,
                               spread_width: float = 5.0) -> Optional[OptionStrategy]:
        """Create a bull call spread strategy"""
        try:
            options_chain = self.options_fetcher.get_options_chain(symbol)
            if not options_chain:
                return None
            
            current_price = options_chain.underlying_price
            
            # Find ATM or slightly OTM calls
            long_strike = None
            short_strike = None
            
            strikes = sorted(options_chain.calls.keys())
            
            # Find long strike (near ATM)
            for strike in strikes:
                if strike >= current_price:
                    long_strike = strike
                    break
            
            if not long_strike:
                return None
            
            # Find short strike (long_strike + spread_width)
            target_short = long_strike + spread_width
            for strike in strikes:
                if strike >= target_short:
                    short_strike = strike
                    break
            
            if not short_strike:
                return None
            
            long_call = options_chain.calls[long_strike]
            short_call = options_chain.calls[short_strike]
            
            # Calculate number of contracts based on risk amount
            net_premium = long_call.price - short_call.price
            contracts = max(1, int(risk_amount / (net_premium * 100)))
            
            # Strategy legs
            legs = [
                {
                    "contract": long_call,
                    "quantity": contracts,
                    "action": "buy",
                    "cost": long_call.price * contracts * 100
                },
                {
                    "contract": short_call,
                    "quantity": -contracts,
                    "action": "sell",
                    "cost": -short_call.price * contracts * 100
                }
            ]
            
            # Calculate P&L at different prices
            pnl_range = {}
            prices = np.linspace(current_price * 0.8, current_price * 1.2, 50)
            
            for price in prices:
                long_value = max(0, price - long_strike)
                short_value = max(0, price - short_strike)
                
                total_value = (long_value - short_value) * contracts * 100
                total_pnl = total_value - net_premium * contracts * 100 - (self.commission_per_contract * 2 * contracts)
                pnl_range[price] = total_pnl
            
            # Calculate metrics
            max_profit = (short_strike - long_strike - net_premium) * contracts * 100
            max_loss = net_premium * contracts * 100
            breakeven = long_strike + net_premium
            
            return OptionStrategy(
                name="Bull Call Spread",
                description=f"Buy {contracts} calls at ${long_strike:.2f}, sell {contracts} calls at ${short_strike:.2f}",
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                profit_loss_range=pnl_range,
                total_premium=-net_premium * contracts * 100,
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0
            )
            
        except Exception as e:
            logger.error(f"Error creating bull call spread for {symbol}: {str(e)}")
            return None
    
    def create_iron_condor(self, symbol: str, risk_amount: float = 2000,
                          wing_width: float = 5.0) -> Optional[OptionStrategy]:
        """Create an iron condor strategy"""
        try:
            options_chain = self.options_fetcher.get_options_chain(symbol)
            if not options_chain:
                return None
            
            current_price = options_chain.underlying_price
            strikes = sorted(options_chain.calls.keys())
            
            # Find strikes around current price
            atm_strike = min(strikes, key=lambda x: abs(x - current_price))
            atm_index = strikes.index(atm_strike)
            
            if atm_index < 2 or atm_index > len(strikes) - 3:
                logger.error("Insufficient strikes available for iron condor")
                return None
            
            # Define the four strikes
            put_long_strike = strikes[max(0, atm_index - 2)]
            put_short_strike = strikes[max(0, atm_index - 1)]
            call_short_strike = strikes[min(len(strikes) - 1, atm_index + 1)]
            call_long_strike = strikes[min(len(strikes) - 1, atm_index + 2)]
            
            # Get contracts
            put_long = options_chain.puts.get(put_long_strike)
            put_short = options_chain.puts.get(put_short_strike)
            call_short = options_chain.calls.get(call_short_strike)
            call_long = options_chain.calls.get(call_long_strike)
            
            if not all([put_long, put_short, call_short, call_long]):
                logger.error("Missing option contracts for iron condor")
                return None
            
            # Calculate net credit
            net_credit = (put_short.price + call_short.price) - (put_long.price + call_long.price)
            contracts = max(1, int(risk_amount / (wing_width * 100)))
            
            # Strategy legs
            legs = [
                {"contract": put_long, "quantity": contracts, "action": "buy", "cost": put_long.price * contracts * 100},
                {"contract": put_short, "quantity": -contracts, "action": "sell", "cost": -put_short.price * contracts * 100},
                {"contract": call_short, "quantity": -contracts, "action": "sell", "cost": -call_short.price * contracts * 100},
                {"contract": call_long, "quantity": contracts, "action": "buy", "cost": call_long.price * contracts * 100}
            ]
            
            # Calculate P&L at different prices
            pnl_range = {}
            prices = np.linspace(current_price * 0.8, current_price * 1.2, 50)
            
            for price in prices:
                put_spread_value = 0
                call_spread_value = 0
                
                # Put spread value
                if price <= put_long_strike:
                    put_spread_value = put_short_strike - put_long_strike
                elif price <= put_short_strike:
                    put_spread_value = put_short_strike - price
                
                # Call spread value  
                if price >= call_long_strike:
                    call_spread_value = call_long_strike - call_short_strike
                elif price >= call_short_strike:
                    call_spread_value = price - call_short_strike
                
                total_value = (put_spread_value + call_spread_value) * contracts * 100
                total_pnl = net_credit * contracts * 100 - total_value - (self.commission_per_contract * 4 * contracts)
                pnl_range[price] = total_pnl
            
            # Calculate metrics
            max_profit = net_credit * contracts * 100
            max_loss = (wing_width - net_credit) * contracts * 100
            breakeven_lower = put_short_strike - net_credit
            breakeven_upper = call_short_strike + net_credit
            
            return OptionStrategy(
                name="Iron Condor",
                description=f"Sell put spread {put_short_strike:.0f}/{put_long_strike:.0f}, sell call spread {call_short_strike:.0f}/{call_long_strike:.0f}",
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven_lower, breakeven_upper],
                profit_loss_range=pnl_range,
                total_premium=net_credit * contracts * 100,
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0
            )
            
        except Exception as e:
            logger.error(f"Error creating iron condor for {symbol}: {str(e)}")
            return None
    
    def create_long_straddle(self, symbol: str, risk_amount: float = 1000) -> Optional[OptionStrategy]:
        """Create a long straddle strategy"""
        try:
            options_chain = self.options_fetcher.get_options_chain(symbol)
            if not options_chain:
                return None
            
            current_price = options_chain.underlying_price
            
            # Find ATM strike
            atm_strike = min(options_chain.calls.keys(), key=lambda x: abs(x - current_price))
            
            atm_call = options_chain.calls.get(atm_strike)
            atm_put = options_chain.puts.get(atm_strike)
            
            if not atm_call or not atm_put:
                logger.error("Missing ATM options for straddle")
                return None
            
            # Calculate contracts based on risk amount
            total_premium = atm_call.price + atm_put.price
            contracts = max(1, int(risk_amount / (total_premium * 100)))
            
            # Strategy legs
            legs = [
                {"contract": atm_call, "quantity": contracts, "action": "buy", "cost": atm_call.price * contracts * 100},
                {"contract": atm_put, "quantity": contracts, "action": "buy", "cost": atm_put.price * contracts * 100}
            ]
            
            # Calculate P&L at different prices
            pnl_range = {}
            prices = np.linspace(current_price * 0.7, current_price * 1.3, 50)
            
            for price in prices:
                call_value = max(0, price - atm_strike)
                put_value = max(0, atm_strike - price)
                total_value = (call_value + put_value) * contracts * 100
                total_pnl = total_value - total_premium * contracts * 100 - (self.commission_per_contract * 2 * contracts)
                pnl_range[price] = total_pnl
            
            # Calculate metrics
            max_profit = float('inf')  # Unlimited in theory
            max_loss = total_premium * contracts * 100
            breakeven_lower = atm_strike - total_premium
            breakeven_upper = atm_strike + total_premium
            
            return OptionStrategy(
                name="Long Straddle",
                description=f"Buy {contracts} calls and {contracts} puts at ${atm_strike:.2f} strike",
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven_lower, breakeven_upper],
                profit_loss_range=pnl_range,
                total_premium=-total_premium * contracts * 100,
                risk_reward_ratio=float('inf')
            )
            
        except Exception as e:
            logger.error(f"Error creating long straddle for {symbol}: {str(e)}")
            return None
    
    def analyze_strategy(self, strategy: OptionStrategy, 
                        current_underlying_price: float) -> StrategyAnalysis:
        """Analyze an existing option strategy"""
        try:
            # Calculate current P&L
            current_pnl = strategy.profit_loss_range.get(current_underlying_price, 0)
            
            # Find prices and probabilities
            prices = list(strategy.profit_loss_range.keys())
            pnls = list(strategy.profit_loss_range.values())
            
            # Estimate probabilities (simplified normal distribution assumption)
            price_mean = current_underlying_price
            price_std = current_underlying_price * 0.2  # Assume 20% annual volatility
            
            # Calculate break-even probability
            profitable_prices = [p for p, pnl in strategy.profit_loss_range.items() if pnl > 0]
            break_even_prob = len(profitable_prices) / len(prices) if prices else 0
            
            # Calculate max profit probability
            max_profit_threshold = strategy.max_profit * 0.8 if strategy.max_profit != float('inf') else max(pnls) * 0.8
            max_profit_prices = [p for p, pnl in strategy.profit_loss_range.items() if pnl >= max_profit_threshold]
            max_profit_prob = len(max_profit_prices) / len(prices) if prices else 0
            
            # Risk metrics
            risk_metrics = {
                "current_delta": self._calculate_strategy_delta(strategy, current_underlying_price),
                "current_gamma": self._calculate_strategy_gamma(strategy, current_underlying_price),
                "current_theta": self._calculate_strategy_theta(strategy),
                "profit_probability": break_even_prob,
                "risk_reward_ratio": strategy.risk_reward_ratio
            }
            
            # Generate recommendations
            recommendations = self._generate_strategy_recommendations(strategy, current_pnl, break_even_prob)
            
            # Generate exit signals
            exit_signals = self._generate_exit_signals(strategy, current_pnl, current_underlying_price)
            
            return StrategyAnalysis(
                strategy=strategy,
                current_pnl=current_pnl,
                break_even_probability=break_even_prob,
                max_profit_probability=max_profit_prob,
                risk_metrics=risk_metrics,
                recommendations=recommendations,
                exit_signals=exit_signals
            )
            
        except Exception as e:
            logger.error(f"Error analyzing strategy: {str(e)}")
            return StrategyAnalysis(
                strategy=strategy,
                current_pnl=0,
                break_even_probability=0,
                max_profit_probability=0,
                risk_metrics={},
                recommendations=["Error analyzing strategy"],
                exit_signals=[]
            )
    
    def _calculate_strategy_delta(self, strategy: OptionStrategy, current_price: float) -> float:
        """Calculate total delta of the strategy"""
        total_delta = 0
        for leg in strategy.legs:
            if leg["contract"].delta:
                total_delta += leg["quantity"] * leg["contract"].delta
        return total_delta
    
    def _calculate_strategy_gamma(self, strategy: OptionStrategy, current_price: float) -> float:
        """Calculate total gamma of the strategy"""
        total_gamma = 0
        for leg in strategy.legs:
            if leg["contract"].gamma:
                total_gamma += leg["quantity"] * leg["contract"].gamma
        return total_gamma
    
    def _calculate_strategy_theta(self, strategy: OptionStrategy) -> float:
        """Calculate total theta of the strategy"""
        total_theta = 0
        for leg in strategy.legs:
            if leg["contract"].theta:
                total_theta += leg["quantity"] * leg["contract"].theta
        return total_theta
    
    def _generate_strategy_recommendations(self, strategy: OptionStrategy, 
                                         current_pnl: float, break_even_prob: float) -> List[str]:
        """Generate recommendations for the strategy"""
        recommendations = []
        
        if current_pnl > strategy.max_profit * 0.75 if strategy.max_profit != float('inf') else False:
            recommendations.append("Consider taking profits - strategy near maximum profit")
        
        if current_pnl < strategy.max_loss * 0.5 if strategy.max_loss != float('-inf') else False:
            recommendations.append("Consider closing position - approaching maximum loss")
        
        if break_even_prob < 0.3:
            recommendations.append("Low probability of profit - consider adjusting or closing")
        elif break_even_prob > 0.7:
            recommendations.append("High probability strategy - monitor for profit taking opportunities")
        
        # Time-based recommendations
        days_to_expiry = (strategy.legs[0]["contract"].expiration - datetime.now()).days
        if days_to_expiry < 7:
            recommendations.append("Close to expiration - monitor for assignment risk")
        elif days_to_expiry < 30:
            recommendations.append("Theta decay accelerating - consider profit taking or adjustment")
        
        return recommendations if recommendations else ["Strategy within normal parameters"]
    
    def _generate_exit_signals(self, strategy: OptionStrategy, current_pnl: float, 
                             current_price: float) -> List[str]:
        """Generate exit signals for the strategy"""
        exit_signals = []
        
        # Profit target signals
        if strategy.max_profit != float('inf') and current_pnl > strategy.max_profit * 0.5:
            exit_signals.append("Profit target: 50% of max profit achieved")
        
        # Loss limit signals
        if strategy.max_loss != float('-inf') and current_pnl < -abs(strategy.max_loss) * 0.5:
            exit_signals.append("Stop loss: 50% of max loss reached")
        
        # Technical signals based on underlying price
        if strategy.breakeven_points:
            if len(strategy.breakeven_points) == 1:
                breakeven = strategy.breakeven_points[0]
                if "Bull" in strategy.name and current_price < breakeven * 0.95:
                    exit_signals.append("Underlying below breakeven - consider exit")
                elif "Bear" in strategy.name and current_price > breakeven * 1.05:
                    exit_signals.append("Underlying above breakeven - consider exit")
        
        # Volatility signals (simplified)
        for leg in strategy.legs:
            if hasattr(leg["contract"], 'implied_volatility') and leg["contract"].implied_volatility:
                if leg["contract"].implied_volatility > 0.5:  # 50% IV
                    exit_signals.append("High implied volatility - consider profit taking")
                elif leg["contract"].implied_volatility < 0.15:  # 15% IV
                    exit_signals.append("Low implied volatility - consider closing short positions")
        
        return exit_signals
