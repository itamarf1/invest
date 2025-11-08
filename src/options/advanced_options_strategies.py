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

class AdvancedStrategyType(Enum):
    """Advanced options strategy types"""
    BUTTERFLY = "butterfly"
    IRON_BUTTERFLY = "iron_butterfly"
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"
    CONDOR = "condor"
    JADE_LIZARD = "jade_lizard"
    RATIO_SPREAD = "ratio_spread"
    CHRISTMAS_TREE = "christmas_tree"
    SYNTHETIC_STOCK = "synthetic_stock"
    COLLAR = "collar"
    STRAP = "strap"
    STRIP = "strip"
    GUT_STRANGLE = "gut_strangle"
    REVERSE_IRON_CONDOR = "reverse_iron_condor"

@dataclass
class AdvancedStrategyAnalysis:
    strategy: OptionStrategy
    greeks_analysis: Dict[str, float]
    volatility_impact: Dict[str, float]
    time_decay_analysis: Dict[str, float]
    profit_zones: List[Tuple[float, float]]  # (min_price, max_price) tuples
    optimal_exit_conditions: List[str]
    risk_warnings: List[str]
    market_outlook_required: str  # bullish, bearish, neutral, high_volatility, low_volatility

class AdvancedOptionsStrategist:
    """Advanced options strategies builder and analyzer"""
    
    def __init__(self):
        self.options_fetcher = OptionsDataFetcher()
        self.commission_per_contract = 0.65
        self.assignment_risk_threshold = 0.05  # 5 cents ITM
    
    def create_long_butterfly(self, symbol: str, center_strike: Optional[float] = None,
                             wing_width: float = 5.0, option_type: OptionType = OptionType.CALL) -> Optional[OptionStrategy]:
        """Create a long butterfly spread (low volatility strategy)"""
        try:
            options_chain = self.options_fetcher.get_options_chain(symbol)
            if not options_chain:
                return None
            
            current_price = options_chain.underlying_price
            
            # Set center strike near current price if not provided
            if center_strike is None:
                center_strike = self._find_nearest_strike(options_chain, current_price, option_type)
            
            # Calculate strikes
            lower_strike = center_strike - wing_width
            upper_strike = center_strike + wing_width
            
            # Get option contracts
            options_dict = options_chain.calls if option_type == OptionType.CALL else options_chain.puts
            
            lower_option = options_dict.get(lower_strike)
            center_option = options_dict.get(center_strike)
            upper_option = options_dict.get(upper_strike)
            
            if not all([lower_option, center_option, upper_option]):
                logger.error(f"Missing options for butterfly spread at strikes {lower_strike}, {center_strike}, {upper_strike}")
                return None
            
            # Strategy legs: Buy 1, Sell 2, Buy 1
            legs = [
                {"contract": lower_option, "quantity": 1, "action": "buy", "cost": lower_option.price * 100},
                {"contract": center_option, "quantity": -2, "action": "sell", "cost": -center_option.price * 2 * 100},
                {"contract": upper_option, "quantity": 1, "action": "buy", "cost": upper_option.price * 100}
            ]
            
            # Calculate P&L at different prices
            pnl_range = {}
            prices = np.linspace(current_price * 0.8, current_price * 1.2, 100)
            
            for price in prices:
                if option_type == OptionType.CALL:
                    lower_value = max(0, price - lower_strike)
                    center_value = max(0, price - center_strike)
                    upper_value = max(0, price - upper_strike)
                else:
                    lower_value = max(0, lower_strike - price)
                    center_value = max(0, center_strike - price)
                    upper_value = max(0, upper_strike - price)
                
                total_value = lower_value - (2 * center_value) + upper_value
                net_premium = lower_option.price + upper_option.price - (2 * center_option.price)
                total_pnl = (total_value * 100) - (net_premium * 100) - (self.commission_per_contract * 4)
                pnl_range[price] = total_pnl
            
            # Calculate metrics
            net_debit = (lower_option.price + upper_option.price - (2 * center_option.price)) * 100
            max_profit = (wing_width - abs(net_debit/100)) * 100
            max_loss = abs(net_debit)
            
            breakeven_lower = lower_strike + abs(net_debit/100)
            breakeven_upper = upper_strike - abs(net_debit/100)
            
            strategy_name = f"Long {option_type.value.title()} Butterfly"
            
            return OptionStrategy(
                name=strategy_name,
                description=f"Buy butterfly spread at {center_strike} center strike with {wing_width} wing width",
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven_lower, breakeven_upper],
                profit_loss_range=pnl_range,
                total_premium=net_debit,
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0
            )
            
        except Exception as e:
            logger.error(f"Error creating butterfly spread for {symbol}: {str(e)}")
            return None
    
    def create_iron_butterfly(self, symbol: str, center_strike: Optional[float] = None,
                             wing_width: float = 5.0) -> Optional[OptionStrategy]:
        """Create an iron butterfly spread (neutral strategy)"""
        try:
            options_chain = self.options_fetcher.get_options_chain(symbol)
            if not options_chain:
                return None
            
            current_price = options_chain.underlying_price
            
            # Set center strike near current price if not provided
            if center_strike is None:
                center_strike = self._find_nearest_strike(options_chain, current_price, OptionType.CALL)
            
            # Calculate strikes
            lower_strike = center_strike - wing_width
            upper_strike = center_strike + wing_width
            
            # Get option contracts
            call_lower = options_chain.calls.get(lower_strike)
            call_center = options_chain.calls.get(center_strike)
            put_center = options_chain.puts.get(center_strike)
            put_upper = options_chain.puts.get(upper_strike)
            
            if not all([call_lower, call_center, put_center, put_upper]):
                logger.error(f"Missing options for iron butterfly")
                return None
            
            # Strategy legs: Buy call wing, Sell call center, Sell put center, Buy put wing
            legs = [
                {"contract": call_lower, "quantity": 1, "action": "buy", "cost": call_lower.price * 100},
                {"contract": call_center, "quantity": -1, "action": "sell", "cost": -call_center.price * 100},
                {"contract": put_center, "quantity": -1, "action": "sell", "cost": -put_center.price * 100},
                {"contract": put_upper, "quantity": 1, "action": "buy", "cost": put_upper.price * 100}
            ]
            
            # Calculate P&L
            pnl_range = {}
            prices = np.linspace(current_price * 0.8, current_price * 1.2, 100)
            
            net_credit = (call_center.price + put_center.price) - (call_lower.price + put_upper.price)
            
            for price in prices:
                call_spread_value = 0
                put_spread_value = 0
                
                # Call spread (short)
                if price > center_strike:
                    if price >= upper_strike:
                        call_spread_value = -(upper_strike - center_strike)
                    else:
                        call_spread_value = -(price - center_strike)
                
                # Put spread (short)
                if price < center_strike:
                    if price <= lower_strike:
                        put_spread_value = -(center_strike - lower_strike)
                    else:
                        put_spread_value = -(center_strike - price)
                
                total_value = call_spread_value + put_spread_value
                total_pnl = (net_credit * 100) + (total_value * 100) - (self.commission_per_contract * 4)
                pnl_range[price] = total_pnl
            
            # Calculate metrics
            max_profit = net_credit * 100
            max_loss = (wing_width - net_credit) * 100
            
            breakeven_lower = center_strike - net_credit
            breakeven_upper = center_strike + net_credit
            
            return OptionStrategy(
                name="Iron Butterfly",
                description=f"Iron butterfly centered at {center_strike} with {wing_width} wing width",
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven_lower, breakeven_upper],
                profit_loss_range=pnl_range,
                total_premium=net_credit * 100,
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0
            )
            
        except Exception as e:
            logger.error(f"Error creating iron butterfly for {symbol}: {str(e)}")
            return None
    
    def create_calendar_spread(self, symbol: str, strike: Optional[float] = None,
                              option_type: OptionType = OptionType.CALL,
                              short_dte: int = 30, long_dte: int = 60) -> Optional[OptionStrategy]:
        """Create a calendar spread (time decay strategy)"""
        try:
            options_chain = self.options_fetcher.get_options_chain(symbol)
            if not options_chain:
                return None
            
            current_price = options_chain.underlying_price
            
            # Use ATM strike if not provided
            if strike is None:
                strike = self._find_nearest_strike(options_chain, current_price, option_type)
            
            # For simplicity, we'll use the same expiration but simulate different DTEs
            # In production, you'd need multiple expiration chains
            
            options_dict = options_chain.calls if option_type == OptionType.CALL else options_chain.puts
            option_contract = options_dict.get(strike)
            
            if not option_contract:
                logger.error(f"Missing option contract for calendar spread at strike {strike}")
                return None
            
            # Simulate different time values for near/far month
            # Near month (sell) - higher time decay
            near_month_price = option_contract.price * 0.6  # Simulate lower price for shorter expiry
            far_month_price = option_contract.price       # Full price for longer expiry
            
            # Strategy legs: Sell near month, Buy far month
            legs = [
                {"contract": option_contract, "quantity": -1, "action": "sell", "cost": -near_month_price * 100},
                {"contract": option_contract, "quantity": 1, "action": "buy", "cost": far_month_price * 100}
            ]
            
            # Calculate P&L (simplified - at near expiration)
            pnl_range = {}
            prices = np.linspace(current_price * 0.85, current_price * 1.15, 50)
            
            net_debit = (far_month_price - near_month_price) * 100
            
            for price in prices:
                if option_type == OptionType.CALL:
                    near_value = max(0, price - strike)
                    # Far month retains some time value
                    far_value = max(price - strike, 0) + max(0, (strike - price) * 0.1) if price < strike else max(0, price - strike)
                else:
                    near_value = max(0, strike - price)
                    far_value = max(strike - price, 0) + max(0, (price - strike) * 0.1) if price > strike else max(0, strike - price)
                
                position_value = (far_value - near_value) * 100
                total_pnl = position_value - net_debit - (self.commission_per_contract * 2)
                pnl_range[price] = total_pnl
            
            # Calculate metrics
            max_profit = (option_contract.price * 0.3) * 100  # Estimated max profit from time decay
            max_loss = net_debit
            
            return OptionStrategy(
                name=f"{option_type.value.title()} Calendar Spread",
                description=f"Calendar spread at {strike} strike ({short_dte}d short / {long_dte}d long)",
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[strike],  # Simplified - max profit at strike
                profit_loss_range=pnl_range,
                total_premium=-net_debit,  # Net debit
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0
            )
            
        except Exception as e:
            logger.error(f"Error creating calendar spread for {symbol}: {str(e)}")
            return None
    
    def create_jade_lizard(self, symbol: str, risk_amount: float = 2000) -> Optional[OptionStrategy]:
        """Create a jade lizard strategy (high probability income strategy)"""
        try:
            options_chain = self.options_fetcher.get_options_chain(symbol)
            if not options_chain:
                return None
            
            current_price = options_chain.underlying_price
            strikes = sorted(options_chain.calls.keys())
            
            # Find strikes: ATM put (short), OTM call spread
            atm_put_strike = self._find_nearest_strike(options_chain, current_price, OptionType.PUT)
            
            # Call spread strikes (above current price)
            call_strikes = [s for s in strikes if s > current_price]
            if len(call_strikes) < 2:
                return None
            
            short_call_strike = call_strikes[0]  # First OTM call
            long_call_strike = call_strikes[min(1, len(call_strikes) - 1)]  # Next strike out
            
            # Get contracts
            short_put = options_chain.puts.get(atm_put_strike)
            short_call = options_chain.calls.get(short_call_strike)
            long_call = options_chain.calls.get(long_call_strike)
            
            if not all([short_put, short_call, long_call]):
                return None
            
            # Calculate position size
            net_credit = short_put.price + short_call.price - long_call.price
            contracts = max(1, int(risk_amount / (net_credit * 100)))
            
            # Strategy legs: Short put, short call, long call
            legs = [
                {"contract": short_put, "quantity": -contracts, "action": "sell", "cost": -short_put.price * contracts * 100},
                {"contract": short_call, "quantity": -contracts, "action": "sell", "cost": -short_call.price * contracts * 100},
                {"contract": long_call, "quantity": contracts, "action": "buy", "cost": long_call.price * contracts * 100}
            ]
            
            # Calculate P&L
            pnl_range = {}
            prices = np.linspace(current_price * 0.7, current_price * 1.3, 100)
            
            for price in prices:
                put_value = max(0, atm_put_strike - price) if price < atm_put_strike else 0
                
                call_spread_value = 0
                if price > short_call_strike:
                    if price >= long_call_strike:
                        call_spread_value = long_call_strike - short_call_strike
                    else:
                        call_spread_value = price - short_call_strike
                
                total_value = put_value + call_spread_value
                total_pnl = (net_credit * contracts * 100) - (total_value * contracts * 100) - (self.commission_per_contract * 3 * contracts)
                pnl_range[price] = total_pnl
            
            # Calculate metrics
            max_profit = net_credit * contracts * 100
            max_loss_call_side = (long_call_strike - short_call_strike - net_credit) * contracts * 100
            max_loss_put_side = float('inf')  # Unlimited downside risk
            
            return OptionStrategy(
                name="Jade Lizard",
                description=f"Short {atm_put_strike} put, short {short_call_strike}/{long_call_strike} call spread",
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss_call_side,  # Limited upside risk only
                breakeven_points=[atm_put_strike - net_credit, short_call_strike + net_credit],
                profit_loss_range=pnl_range,
                total_premium=net_credit * contracts * 100,
                risk_reward_ratio=max_profit / max_loss_call_side if max_loss_call_side > 0 else float('inf')
            )
            
        except Exception as e:
            logger.error(f"Error creating jade lizard for {symbol}: {str(e)}")
            return None
    
    def create_ratio_call_spread(self, symbol: str, ratio: int = 2, risk_amount: float = 1500) -> Optional[OptionStrategy]:
        """Create a ratio call spread (1 buy : N sell ratio)"""
        try:
            options_chain = self.options_fetcher.get_options_chain(symbol)
            if not options_chain:
                return None
            
            current_price = options_chain.underlying_price
            strikes = sorted(options_chain.calls.keys())
            
            # Find long and short strikes
            long_strike = None
            short_strike = None
            
            for strike in strikes:
                if strike >= current_price and long_strike is None:
                    long_strike = strike
                elif long_strike is not None and strike > long_strike and short_strike is None:
                    short_strike = strike
                    break
            
            if not long_strike or not short_strike:
                return None
            
            long_call = options_chain.calls.get(long_strike)
            short_call = options_chain.calls.get(short_strike)
            
            if not long_call or not short_call:
                return None
            
            # Calculate position size
            net_debit = long_call.price - (ratio * short_call.price)
            if net_debit > 0:
                contracts = max(1, int(risk_amount / (net_debit * 100)))
            else:
                contracts = 1  # Net credit
            
            # Strategy legs: Buy 1, Sell ratio
            legs = [
                {"contract": long_call, "quantity": contracts, "action": "buy", "cost": long_call.price * contracts * 100},
                {"contract": short_call, "quantity": -ratio * contracts, "action": "sell", "cost": -short_call.price * ratio * contracts * 100}
            ]
            
            # Calculate P&L
            pnl_range = {}
            prices = np.linspace(current_price * 0.8, current_price * 1.4, 100)
            
            for price in prices:
                long_value = max(0, price - long_strike)
                short_value = max(0, price - short_strike)
                
                total_value = (long_value - (ratio * short_value)) * contracts * 100
                total_pnl = total_value - (net_debit * contracts * 100) - (self.commission_per_contract * (1 + ratio) * contracts)
                pnl_range[price] = total_pnl
            
            # Calculate metrics
            max_profit_strike = short_strike
            max_profit = ((short_strike - long_strike) - net_debit) * contracts * 100
            
            # Max loss calculation depends on whether it's net debit or credit
            if net_debit > 0:
                max_loss = net_debit * contracts * 100
            else:
                max_loss = float('inf')  # Unlimited upside risk for net credit ratio spreads
            
            upside_breakeven = short_strike + (short_strike - long_strike) / (ratio - 1)
            
            return OptionStrategy(
                name=f"Ratio Call Spread (1:{ratio})",
                description=f"Buy {contracts} calls at {long_strike}, sell {ratio * contracts} calls at {short_strike}",
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[long_strike + net_debit, upside_breakeven],
                profit_loss_range=pnl_range,
                total_premium=-net_debit * contracts * 100,
                risk_reward_ratio=max_profit / max_loss if max_loss != float('inf') and max_loss > 0 else float('inf')
            )
            
        except Exception as e:
            logger.error(f"Error creating ratio call spread for {symbol}: {str(e)}")
            return None
    
    def create_synthetic_stock(self, symbol: str, strike: Optional[float] = None) -> Optional[OptionStrategy]:
        """Create synthetic stock position using options"""
        try:
            options_chain = self.options_fetcher.get_options_chain(symbol)
            if not options_chain:
                return None
            
            current_price = options_chain.underlying_price
            
            # Use ATM strike if not provided
            if strike is None:
                strike = self._find_nearest_strike(options_chain, current_price, OptionType.CALL)
            
            call_option = options_chain.calls.get(strike)
            put_option = options_chain.puts.get(strike)
            
            if not call_option or not put_option:
                return None
            
            # Synthetic long stock: Buy call, Sell put
            legs = [
                {"contract": call_option, "quantity": 1, "action": "buy", "cost": call_option.price * 100},
                {"contract": put_option, "quantity": -1, "action": "sell", "cost": -put_option.price * 100}
            ]
            
            # Calculate P&L
            pnl_range = {}
            prices = np.linspace(current_price * 0.6, current_price * 1.4, 100)
            
            net_debit = call_option.price - put_option.price
            
            for price in prices:
                call_value = max(0, price - strike)
                put_value = max(0, strike - price)
                
                position_value = call_value - put_value
                # Synthetic stock P&L = (Current Price - Strike) - Net Premium
                synthetic_pnl = ((price - strike) - net_debit) * 100
                total_pnl = synthetic_pnl - (self.commission_per_contract * 2)
                pnl_range[price] = total_pnl
            
            return OptionStrategy(
                name="Synthetic Long Stock",
                description=f"Long call + short put at {strike} strike (mimics 100 shares)",
                legs=legs,
                max_profit=float('inf'),  # Unlimited upside like stock
                max_loss=float('inf'),    # Unlimited downside like stock
                breakeven_points=[strike + net_debit],
                profit_loss_range=pnl_range,
                total_premium=-net_debit * 100,
                risk_reward_ratio=float('inf')
            )
            
        except Exception as e:
            logger.error(f"Error creating synthetic stock for {symbol}: {str(e)}")
            return None
    
    def _find_nearest_strike(self, options_chain: OptionsChain, target_price: float, 
                           option_type: OptionType) -> Optional[float]:
        """Find the strike closest to target price"""
        try:
            options_dict = options_chain.calls if option_type == OptionType.CALL else options_chain.puts
            strikes = list(options_dict.keys())
            
            if not strikes:
                return None
            
            return min(strikes, key=lambda x: abs(x - target_price))
            
        except Exception as e:
            logger.error(f"Error finding nearest strike: {str(e)}")
            return None
    
    def analyze_advanced_strategy(self, strategy: OptionStrategy, 
                                current_underlying_price: float) -> AdvancedStrategyAnalysis:
        """Advanced analysis of options strategy"""
        try:
            # Calculate current Greeks
            total_delta = sum(leg["quantity"] * (leg["contract"].delta or 0) for leg in strategy.legs)
            total_gamma = sum(leg["quantity"] * (leg["contract"].gamma or 0) for leg in strategy.legs)
            total_theta = sum(leg["quantity"] * (leg["contract"].theta or 0) for leg in strategy.legs)
            total_vega = sum(leg["quantity"] * (leg["contract"].vega or 0) for leg in strategy.legs)
            
            greeks_analysis = {
                "delta": total_delta,
                "gamma": total_gamma,
                "theta": total_theta,
                "vega": total_vega,
                "delta_neutral": abs(total_delta) < 0.1
            }
            
            # Volatility impact analysis
            volatility_impact = {
                "vega_risk": abs(total_vega),
                "volatility_sensitivity": "High" if abs(total_vega) > 50 else "Medium" if abs(total_vega) > 20 else "Low",
                "benefits_from_vol_increase": total_vega > 0,
                "benefits_from_vol_decrease": total_vega < 0
            }
            
            # Time decay analysis
            time_decay_analysis = {
                "theta_risk": total_theta,
                "benefits_from_time_decay": total_theta > 0,
                "time_decay_rate": "High" if abs(total_theta) > 10 else "Medium" if abs(total_theta) > 5 else "Low"
            }
            
            # Find profit zones
            profit_zones = self._find_profit_zones(strategy.profit_loss_range)
            
            # Generate optimal exit conditions
            optimal_exit_conditions = self._generate_exit_conditions(strategy, greeks_analysis)
            
            # Generate risk warnings
            risk_warnings = self._generate_risk_warnings(strategy, greeks_analysis, volatility_impact)
            
            # Determine required market outlook
            market_outlook = self._determine_market_outlook(strategy, greeks_analysis, volatility_impact)
            
            return AdvancedStrategyAnalysis(
                strategy=strategy,
                greeks_analysis=greeks_analysis,
                volatility_impact=volatility_impact,
                time_decay_analysis=time_decay_analysis,
                profit_zones=profit_zones,
                optimal_exit_conditions=optimal_exit_conditions,
                risk_warnings=risk_warnings,
                market_outlook_required=market_outlook
            )
            
        except Exception as e:
            logger.error(f"Error in advanced strategy analysis: {str(e)}")
            return AdvancedStrategyAnalysis(
                strategy=strategy,
                greeks_analysis={},
                volatility_impact={},
                time_decay_analysis={},
                profit_zones=[],
                optimal_exit_conditions=["Unable to analyze exit conditions"],
                risk_warnings=["Analysis error - monitor position closely"],
                market_outlook_required="unknown"
            )
    
    def _find_profit_zones(self, pnl_range: Dict[float, float]) -> List[Tuple[float, float]]:
        """Find profitable price ranges"""
        try:
            profit_zones = []
            prices = sorted(pnl_range.keys())
            
            zone_start = None
            for price in prices:
                pnl = pnl_range[price]
                
                if pnl > 0 and zone_start is None:
                    zone_start = price
                elif pnl <= 0 and zone_start is not None:
                    profit_zones.append((zone_start, price))
                    zone_start = None
            
            # Close final zone if still profitable
            if zone_start is not None:
                profit_zones.append((zone_start, prices[-1]))
            
            return profit_zones
            
        except Exception as e:
            logger.error(f"Error finding profit zones: {str(e)}")
            return []
    
    def _generate_exit_conditions(self, strategy: OptionStrategy, 
                                greeks: Dict[str, float]) -> List[str]:
        """Generate optimal exit conditions"""
        exit_conditions = []
        
        try:
            # Profit-based exits
            if strategy.max_profit != float('inf'):
                exit_conditions.append(f"Take profits at 50-75% of max profit (${strategy.max_profit * 0.5:.0f} - ${strategy.max_profit * 0.75:.0f})")
            
            # Loss-based exits
            if strategy.max_loss != float('inf'):
                exit_conditions.append(f"Stop loss at 50% of max loss (${strategy.max_loss * 0.5:.0f})")
            
            # Time-based exits
            if greeks.get("theta", 0) > 0:
                exit_conditions.append("Close with 7-14 days to expiration to maximize time decay")
            else:
                exit_conditions.append("Monitor closely approaching expiration due to theta risk")
            
            # Volatility-based exits
            if abs(greeks.get("vega", 0)) > 20:
                exit_conditions.append("Consider closing if implied volatility changes by >25%")
            
            # Delta-based exits
            if abs(greeks.get("delta", 0)) > 0.5:
                exit_conditions.append("Monitor delta closely - position becomes more directional")
            
            return exit_conditions if exit_conditions else ["Monitor position based on market conditions"]
            
        except Exception as e:
            logger.error(f"Error generating exit conditions: {str(e)}")
            return ["Monitor position manually"]
    
    def _generate_risk_warnings(self, strategy: OptionStrategy, greeks: Dict[str, float],
                               vol_impact: Dict[str, Any]) -> List[str]:
        """Generate risk warnings"""
        warnings = []
        
        try:
            # Assignment risk
            for leg in strategy.legs:
                if leg["quantity"] < 0 and leg["contract"].option_type == OptionType.CALL:
                    # Short calls
                    warnings.append("Assignment risk on short calls if stock price rises significantly")
                elif leg["quantity"] < 0 and leg["contract"].option_type == OptionType.PUT:
                    # Short puts
                    warnings.append("Assignment risk on short puts if stock price falls significantly")
            
            # High vega risk
            if abs(greeks.get("vega", 0)) > 50:
                warnings.append("High volatility sensitivity - significant P&L swings possible")
            
            # Gamma risk
            if abs(greeks.get("gamma", 0)) > 0.1:
                warnings.append("High gamma - delta will change rapidly with price movements")
            
            # Time decay risk
            if greeks.get("theta", 0) < -10:
                warnings.append("High time decay risk - position loses value over time")
            
            # Unlimited loss risk
            if strategy.max_loss == float('inf'):
                warnings.append("UNLIMITED LOSS POTENTIAL - Monitor position closely")
            
            # Liquidity warnings
            total_contracts = sum(abs(leg["quantity"]) for leg in strategy.legs)
            if total_contracts > 4:
                warnings.append("Complex strategy - ensure all legs have adequate liquidity")
            
            return warnings if warnings else ["Standard options risks apply"]
            
        except Exception as e:
            logger.error(f"Error generating risk warnings: {str(e)}")
            return ["Unable to assess risks - trade with caution"]
    
    def _determine_market_outlook(self, strategy: OptionStrategy, greeks: Dict[str, float],
                                vol_impact: Dict[str, Any]) -> str:
        """Determine required market outlook for strategy"""
        try:
            delta = greeks.get("delta", 0)
            vega = greeks.get("vega", 0)
            
            # Directional bias
            if delta > 0.3:
                base_outlook = "bullish"
            elif delta < -0.3:
                base_outlook = "bearish"
            else:
                base_outlook = "neutral"
            
            # Volatility bias
            if abs(vega) > 30:
                if vega > 0:
                    vol_outlook = "high_volatility"
                else:
                    vol_outlook = "low_volatility"
                
                # Combine outlooks
                if base_outlook == "neutral":
                    return vol_outlook
                else:
                    return f"{base_outlook}_{vol_outlook}"
            
            return base_outlook
            
        except Exception as e:
            logger.error(f"Error determining market outlook: {str(e)}")
            return "unknown"
