import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import math
from scipy.stats import norm
import yfinance as yf

logger = logging.getLogger(__name__)

class OptionType(Enum):
    """Option types"""
    CALL = "call"
    PUT = "put"

@dataclass
class OptionContract:
    symbol: str
    strike: float
    expiration: datetime
    option_type: OptionType
    price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

@dataclass
class OptionsChain:
    underlying_symbol: str
    underlying_price: float
    expiration_dates: List[datetime]
    calls: Dict[float, OptionContract]  # strike -> contract
    puts: Dict[float, OptionContract]   # strike -> contract
    timestamp: datetime

@dataclass
class OptionStrategy:
    name: str
    description: str
    legs: List[Dict[str, Any]]  # Each leg: {contract, quantity, action}
    max_profit: Optional[float]
    max_loss: Optional[float]
    breakeven_points: List[float]
    profit_loss_range: Dict[float, float]  # price -> P&L
    total_premium: float
    risk_reward_ratio: float

class BlackScholesCalculator:
    """Black-Scholes options pricing and Greeks calculator"""
    
    @staticmethod
    def calculate_option_price(S: float, K: float, T: float, r: float, sigma: float, 
                             option_type: OptionType = OptionType.CALL) -> float:
        """Calculate option price using Black-Scholes formula"""
        try:
            if T <= 0:
                # At expiration
                if option_type == OptionType.CALL:
                    return max(S - K, 0)
                else:
                    return max(K - S, 0)
            
            # Black-Scholes formula
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            if option_type == OptionType.CALL:
                price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return max(price, 0)
            
        except Exception as e:
            logger.error(f"Error calculating option price: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float,
                        option_type: OptionType = OptionType.CALL) -> Dict[str, float]:
        """Calculate option Greeks"""
        try:
            if T <= 0:
                return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
            
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            # Delta
            if option_type == OptionType.CALL:
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
            
            # Gamma (same for calls and puts)
            gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
            
            # Theta
            if option_type == OptionType.CALL:
                theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - 
                        r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
            else:
                theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + 
                        r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
            
            # Vega (same for calls and puts)
            vega = S * norm.pdf(d1) * math.sqrt(T) / 100
            
            # Rho
            if option_type == OptionType.CALL:
                rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
            else:
                rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
            
            return {
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "rho": rho
            }
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {str(e)}")
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
    
    @staticmethod
    def calculate_implied_volatility(market_price: float, S: float, K: float, T: float, 
                                   r: float, option_type: OptionType = OptionType.CALL,
                                   max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        try:
            # Initial guess
            sigma = 0.3
            
            for i in range(max_iterations):
                # Calculate option price and vega with current sigma
                price = BlackScholesCalculator.calculate_option_price(S, K, T, r, sigma, option_type)
                greeks = BlackScholesCalculator.calculate_greeks(S, K, T, r, sigma, option_type)
                vega = greeks["vega"] * 100  # Convert back from percentage
                
                # Newton-Raphson update
                if abs(vega) < 1e-10:  # Avoid division by zero
                    break
                
                price_diff = price - market_price
                if abs(price_diff) < tolerance:
                    break
                
                sigma = sigma - price_diff / vega
                sigma = max(0.01, min(sigma, 5.0))  # Keep sigma in reasonable bounds
            
            return sigma
            
        except Exception as e:
            logger.error(f"Error calculating implied volatility: {str(e)}")
            return 0.3  # Default 30%

class OptionsDataFetcher:
    """Fetch options data from various sources"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.bs_calculator = BlackScholesCalculator()
    
    def get_options_chain(self, symbol: str, expiration_date: Optional[str] = None) -> Optional[OptionsChain]:
        """Get options chain data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current price
            hist = ticker.history(period="1d")
            if hist.empty:
                logger.error(f"No price data for {symbol}")
                return None
            
            underlying_price = hist['Close'].iloc[-1]
            
            # Get available expiration dates
            try:
                expirations = ticker.options
                if not expirations:
                    logger.error(f"No options available for {symbol}")
                    return None
            except Exception:
                logger.error(f"Error getting expiration dates for {symbol}")
                return None
            
            # Use first expiration if not specified
            if expiration_date is None:
                expiration_date = expirations[0]
            elif expiration_date not in expirations:
                logger.warning(f"Expiration {expiration_date} not available, using {expirations[0]}")
                expiration_date = expirations[0]
            
            # Get options chain
            try:
                opt_chain = ticker.option_chain(expiration_date)
                calls_df = opt_chain.calls
                puts_df = opt_chain.puts
            except Exception:
                logger.error(f"Error getting options chain for {symbol} on {expiration_date}")
                return None
            
            # Parse expiration date
            exp_date = datetime.strptime(expiration_date, "%Y-%m-%d")
            time_to_expiry = (exp_date - datetime.now()).days / 365.0
            
            # Process calls
            calls = {}
            for _, row in calls_df.iterrows():
                try:
                    strike = float(row['strike'])
                    
                    # Calculate Greeks if not provided
                    greeks = self.bs_calculator.calculate_greeks(
                        S=underlying_price,
                        K=strike,
                        T=time_to_expiry,
                        r=self.risk_free_rate,
                        sigma=row.get('impliedVolatility', 0.3),
                        option_type=OptionType.CALL
                    )
                    
                    calls[strike] = OptionContract(
                        symbol=row['contractSymbol'],
                        strike=strike,
                        expiration=exp_date,
                        option_type=OptionType.CALL,
                        price=row['lastPrice'],
                        bid=row['bid'],
                        ask=row['ask'],
                        volume=row.get('volume', 0),
                        open_interest=row.get('openInterest', 0),
                        implied_volatility=row.get('impliedVolatility', 0.3),
                        delta=greeks['delta'],
                        gamma=greeks['gamma'],
                        theta=greeks['theta'],
                        vega=greeks['vega'],
                        rho=greeks['rho']
                    )
                except Exception as e:
                    logger.warning(f"Error processing call option at strike {row['strike']}: {str(e)}")
                    continue
            
            # Process puts
            puts = {}
            for _, row in puts_df.iterrows():
                try:
                    strike = float(row['strike'])
                    
                    # Calculate Greeks if not provided
                    greeks = self.bs_calculator.calculate_greeks(
                        S=underlying_price,
                        K=strike,
                        T=time_to_expiry,
                        r=self.risk_free_rate,
                        sigma=row.get('impliedVolatility', 0.3),
                        option_type=OptionType.PUT
                    )
                    
                    puts[strike] = OptionContract(
                        symbol=row['contractSymbol'],
                        strike=strike,
                        expiration=exp_date,
                        option_type=OptionType.PUT,
                        price=row['lastPrice'],
                        bid=row['bid'],
                        ask=row['ask'],
                        volume=row.get('volume', 0),
                        open_interest=row.get('openInterest', 0),
                        implied_volatility=row.get('impliedVolatility', 0.3),
                        delta=greeks['delta'],
                        gamma=greeks['gamma'],
                        theta=greeks['theta'],
                        vega=greeks['vega'],
                        rho=greeks['rho']
                    )
                except Exception as e:
                    logger.warning(f"Error processing put option at strike {row['strike']}: {str(e)}")
                    continue
            
            # Convert expiration dates
            exp_dates = [datetime.strptime(exp, "%Y-%m-%d") for exp in expirations]
            
            return OptionsChain(
                underlying_symbol=symbol,
                underlying_price=underlying_price,
                expiration_dates=exp_dates,
                calls=calls,
                puts=puts,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {str(e)}")
            return None
    
    def get_historical_volatility(self, symbol: str, days: int = 30) -> float:
        """Calculate historical volatility"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days*2}d")  # Get extra data
            
            if len(hist) < days:
                logger.warning(f"Insufficient data for volatility calculation: {len(hist)} days")
                return 0.3  # Default 30%
            
            # Calculate daily returns
            returns = hist['Close'].pct_change().dropna().tail(days)
            
            # Annualized volatility
            volatility = returns.std() * math.sqrt(252)
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating historical volatility for {symbol}: {str(e)}")
            return 0.3
    
    def find_liquid_options(self, options_chain: OptionsChain, min_volume: int = 10, 
                           min_open_interest: int = 50) -> Dict[str, List[OptionContract]]:
        """Find liquid options from the chain"""
        try:
            liquid_calls = []
            liquid_puts = []
            
            # Filter calls
            for contract in options_chain.calls.values():
                if contract.volume >= min_volume and contract.open_interest >= min_open_interest:
                    liquid_calls.append(contract)
            
            # Filter puts
            for contract in options_chain.puts.values():
                if contract.volume >= min_volume and contract.open_interest >= min_open_interest:
                    liquid_puts.append(contract)
            
            return {
                "calls": sorted(liquid_calls, key=lambda x: x.volume, reverse=True),
                "puts": sorted(liquid_puts, key=lambda x: x.volume, reverse=True)
            }
            
        except Exception as e:
            logger.error(f"Error finding liquid options: {str(e)}")
            return {"calls": [], "puts": []}
    
    def get_earnings_date(self, symbol: str) -> Optional[datetime]:
        """Get next earnings date (simplified implementation)"""
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            
            if calendar is not None and not calendar.empty:
                # Get the next earnings date
                next_earnings = calendar.index[0]
                return next_earnings
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not get earnings date for {symbol}: {str(e)}")
            return None
