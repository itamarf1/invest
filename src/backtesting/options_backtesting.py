import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from src.options.options_data import OptionType, OptionsDataFetcher, OptionContract, OptionStrategy
from src.options.options_strategies import OptionsStrategist
from src.options.advanced_options_strategies import AdvancedOptionsStrategist

logger = logging.getLogger(__name__)

class BacktestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class BacktestTrade:
    """Individual trade in backtest"""
    strategy_name: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    pnl: float
    pnl_pct: float
    max_profit: float
    max_loss: float
    days_held: int
    exit_reason: str
    underlying_entry: float
    underlying_exit: Optional[float]
    greeks_entry: Dict[str, float]
    legs: List[Dict[str, Any]]

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_days_held: float
    trades: List[BacktestTrade]
    equity_curve: pd.Series
    monthly_returns: pd.Series
    strategy_breakdown: Dict[str, Dict[str, float]]

class OptionsBacktester:
    """Advanced options strategies backtesting engine"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.options_data_fetcher = OptionsDataFetcher()
        self.basic_strategist = OptionsStrategist()
        self.advanced_strategist = AdvancedOptionsStrategist()
        
        # Backtesting parameters
        self.commission_per_contract = 0.65
        self.slippage_pct = 0.01  # 1% slippage assumption
        self.interest_rate = 0.05  # Risk-free rate for options pricing
        
    def run_strategy_backtest(self, symbol: str, strategy_name: str, 
                             start_date: datetime, end_date: datetime,
                             strategy_params: Dict[str, Any] = None) -> BacktestResults:
        """Run backtest for a specific options strategy"""
        try:
            logger.info(f"Starting backtest for {strategy_name} on {symbol}")
            
            # Get historical price data
            historical_data = self._get_historical_data(symbol, start_date, end_date)
            if historical_data.empty:
                raise ValueError(f"No historical data available for {symbol}")
            
            # Initialize tracking variables
            trades = []
            equity_curve = [self.initial_capital]
            current_positions = []
            
            # Strategy parameters
            if strategy_params is None:
                strategy_params = self._get_default_strategy_params(strategy_name)
            
            # Run backtest day by day
            for date in pd.date_range(start=start_date, end=end_date, freq='D'):
                if date.weekday() >= 5:  # Skip weekends
                    continue
                    
                try:
                    current_price = historical_data.loc[date, 'Close'] if date in historical_data.index else None
                    if current_price is None:
                        continue
                    
                    # Check exit conditions for existing positions
                    trades_to_close = []
                    for i, position in enumerate(current_positions):
                        exit_signal, exit_reason = self._check_exit_conditions(
                            position, current_price, date
                        )
                        if exit_signal:
                            trades_to_close.append((i, exit_reason))
                    
                    # Close positions
                    for i, exit_reason in reversed(trades_to_close):
                        position = current_positions.pop(i)
                        trade = self._close_position(position, current_price, date, exit_reason)
                        trades.append(trade)
                        self.current_capital += trade.pnl
                    
                    # Check entry conditions
                    entry_signal = self._check_entry_conditions(
                        symbol, strategy_name, current_price, date, strategy_params
                    )
                    
                    if entry_signal and len(current_positions) < 5:  # Limit concurrent positions
                        new_position = self._enter_position(
                            symbol, strategy_name, current_price, date, strategy_params
                        )
                        if new_position:
                            current_positions.append(new_position)
                    
                    # Update equity curve
                    portfolio_value = self.current_capital + sum(
                        self._calculate_position_value(pos, current_price, date)
                        for pos in current_positions
                    )
                    equity_curve.append(portfolio_value)
                    
                except Exception as e:
                    logger.warning(f"Error processing date {date}: {str(e)}")
                    continue
            
            # Close any remaining positions
            final_price = historical_data.iloc[-1]['Close']
            final_date = historical_data.index[-1]
            for position in current_positions:
                trade = self._close_position(position, final_price, final_date, "backtest_end")
                trades.append(trade)
                self.current_capital += trade.pnl
            
            # Calculate final results
            results = self._calculate_backtest_results(
                trades, equity_curve, start_date, end_date, strategy_name
            )
            
            logger.info(f"Backtest completed: {len(trades)} trades, {results.total_return_pct:.2f}% return")
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise
    
    def run_multi_strategy_backtest(self, symbol: str, strategies: List[str],
                                   start_date: datetime, end_date: datetime) -> Dict[str, BacktestResults]:
        """Run backtest comparing multiple strategies"""
        results = {}
        
        for strategy in strategies:
            try:
                # Reset capital for each strategy
                self.current_capital = self.initial_capital
                
                result = self.run_strategy_backtest(symbol, strategy, start_date, end_date)
                results[strategy] = result
                
                logger.info(f"Strategy {strategy}: {result.total_return_pct:.2f}% return, "
                          f"{result.win_rate:.1f}% win rate")
                
            except Exception as e:
                logger.error(f"Error backtesting strategy {strategy}: {str(e)}")
                continue
        
        return results
    
    def _get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical price data for backtesting"""
        try:
            # Add buffer for options expiration
            buffer_start = start_date - timedelta(days=60)
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=buffer_start, end=end_date + timedelta(days=1))
            
            if data.empty:
                raise ValueError(f"No historical data for {symbol}")
            
            # Calculate additional metrics
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _get_default_strategy_params(self, strategy_name: str) -> Dict[str, Any]:
        """Get default parameters for each strategy"""
        defaults = {
            'covered_call': {
                'target_strike_pct': 0.05,
                'min_premium': 1.0,
                'dte_range': (30, 45)
            },
            'iron_condor': {
                'wing_width': 10.0,
                'risk_amount': 2000,
                'min_credit': 0.3
            },
            'long_butterfly': {
                'wing_width': 5.0,
                'option_type': OptionType.CALL
            },
            'calendar_spread': {
                'short_dte': 30,
                'long_dte': 60
            },
            'jade_lizard': {
                'risk_amount': 2000
            },
            'protective_put': {
                'protection_pct': 0.05,
                'max_cost_pct': 0.02
            }
        }
        
        return defaults.get(strategy_name, {})
    
    def _check_entry_conditions(self, symbol: str, strategy_name: str, 
                               current_price: float, date: datetime, 
                               params: Dict[str, Any]) -> bool:
        """Check if conditions are met to enter a new position"""
        try:
            # Basic entry conditions based on market regime
            returns_window = 20
            
            # Get recent price action (simplified for backtesting)
            volatility_threshold = 0.25  # 25% annualized
            trend_threshold = 0.02      # 2% trend strength
            
            # Strategy-specific entry logic
            if strategy_name in ['covered_call', 'jade_lizard']:
                # Enter in neutral to slightly bullish markets
                return np.random.random() < 0.3  # 30% entry probability
            
            elif strategy_name in ['iron_condor', 'long_butterfly']:
                # Enter in low volatility, range-bound markets
                return np.random.random() < 0.25  # 25% entry probability
            
            elif strategy_name == 'calendar_spread':
                # Enter when expecting time decay with stable price
                return np.random.random() < 0.2  # 20% entry probability
            
            elif strategy_name == 'protective_put':
                # Enter during market uncertainty
                return np.random.random() < 0.15  # 15% entry probability
            
            else:
                return np.random.random() < 0.2
            
        except Exception as e:
            logger.warning(f"Error checking entry conditions: {str(e)}")
            return False
    
    def _enter_position(self, symbol: str, strategy_name: str, 
                       current_price: float, date: datetime, 
                       params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enter a new options position"""
        try:
            # Create strategy based on name
            strategy = self._create_strategy(symbol, strategy_name, params)
            if not strategy:
                return None
            
            # Calculate position sizing
            position_risk = min(self.current_capital * 0.05, 5000)  # 5% max risk per trade
            contracts = max(1, int(position_risk / abs(strategy.total_premium)))
            
            # Create position record
            position = {
                'strategy': strategy,
                'strategy_name': strategy_name,
                'symbol': symbol,
                'entry_date': date,
                'entry_price': strategy.total_premium,
                'contracts': contracts,
                'underlying_entry': current_price,
                'max_profit_target': strategy.max_profit * 0.5,  # Take 50% of max profit
                'max_loss_limit': strategy.max_loss * 0.5,       # Stop at 50% of max loss
                'days_target': 21,  # Target 21 days holding period
                'greeks_entry': self._calculate_position_greeks(strategy)
            }
            
            # Deduct premium cost/credit from capital
            premium_impact = strategy.total_premium * contracts
            commission_cost = self.commission_per_contract * len(strategy.legs) * contracts
            
            self.current_capital -= (premium_impact + commission_cost)
            
            logger.debug(f"Entered {strategy_name} position: {contracts} contracts, "
                        f"premium ${premium_impact:.2f}")
            
            return position
            
        except Exception as e:
            logger.error(f"Error entering position: {str(e)}")
            return None
    
    def _create_strategy(self, symbol: str, strategy_name: str, 
                        params: Dict[str, Any]) -> Optional[OptionStrategy]:
        """Create options strategy based on name and parameters"""
        try:
            if strategy_name == 'covered_call':
                return self.basic_strategist.create_covered_call(
                    symbol, 100, params.get('target_strike_pct', 0.05)
                )
            
            elif strategy_name == 'iron_condor':
                return self.basic_strategist.create_iron_condor(
                    symbol, params.get('risk_amount', 2000), 
                    params.get('wing_width', 10.0)
                )
            
            elif strategy_name == 'long_butterfly':
                return self.advanced_strategist.create_long_butterfly(
                    symbol, wing_width=params.get('wing_width', 5.0)
                )
            
            elif strategy_name == 'calendar_spread':
                return self.advanced_strategist.create_calendar_spread(symbol)
            
            elif strategy_name == 'jade_lizard':
                return self.advanced_strategist.create_jade_lizard(
                    symbol, params.get('risk_amount', 2000)
                )
            
            elif strategy_name == 'protective_put':
                return self.basic_strategist.create_protective_put(
                    symbol, 100, params.get('protection_pct', 0.05)
                )
            
            else:
                logger.warning(f"Unknown strategy: {strategy_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating strategy {strategy_name}: {str(e)}")
            return None
    
    def _check_exit_conditions(self, position: Dict[str, Any], 
                              current_price: float, date: datetime) -> Tuple[bool, str]:
        """Check if position should be exited"""
        try:
            days_held = (date - position['entry_date']).days
            current_value = self._calculate_position_value(position, current_price, date)
            entry_cost = abs(position['entry_price']) * position['contracts']
            current_pnl = current_value - entry_cost
            
            # Time-based exit
            if days_held >= position['days_target']:
                return True, "time_target"
            
            # Profit target
            if current_pnl >= position['max_profit_target'] * position['contracts']:
                return True, "profit_target"
            
            # Stop loss
            if current_pnl <= -position['max_loss_limit'] * position['contracts']:
                return True, "stop_loss"
            
            # Strategy-specific exits
            strategy_name = position['strategy_name']
            
            if strategy_name in ['iron_condor', 'long_butterfly']:
                # Exit if underlying moves too far from entry
                price_change_pct = abs((current_price - position['underlying_entry']) / position['underlying_entry'])
                if price_change_pct > 0.15:  # 15% move
                    return True, "underlying_breakout"
            
            elif strategy_name == 'covered_call':
                # Exit if stock drops significantly
                price_change_pct = (current_price - position['underlying_entry']) / position['underlying_entry']
                if price_change_pct < -0.10:  # 10% drop
                    return True, "stock_decline"
            
            return False, "hold"
            
        except Exception as e:
            logger.warning(f"Error checking exit conditions: {str(e)}")
            return False, "error"
    
    def _close_position(self, position: Dict[str, Any], current_price: float, 
                       exit_date: datetime, exit_reason: str) -> BacktestTrade:
        """Close a position and calculate P&L"""
        try:
            exit_value = self._calculate_position_value(position, current_price, exit_date)
            entry_cost = abs(position['entry_price']) * position['contracts']
            
            # Calculate P&L including commissions
            commission_cost = self.commission_per_contract * len(position['strategy'].legs) * position['contracts']
            pnl = exit_value - entry_cost - commission_cost
            pnl_pct = (pnl / entry_cost) * 100 if entry_cost > 0 else 0
            
            days_held = (exit_date - position['entry_date']).days
            
            # Create trade record
            trade = BacktestTrade(
                strategy_name=position['strategy_name'],
                entry_date=position['entry_date'],
                exit_date=exit_date,
                entry_price=position['entry_price'],
                exit_price=exit_value / position['contracts'],
                quantity=position['contracts'],
                pnl=pnl,
                pnl_pct=pnl_pct,
                max_profit=position['strategy'].max_profit * position['contracts'],
                max_loss=position['strategy'].max_loss * position['contracts'],
                days_held=days_held,
                exit_reason=exit_reason,
                underlying_entry=position['underlying_entry'],
                underlying_exit=current_price,
                greeks_entry=position['greeks_entry'],
                legs=[{
                    'action': leg['action'],
                    'quantity': leg['quantity'],
                    'contract_type': leg['contract'].option_type.value,
                    'strike': leg['contract'].strike,
                    'premium': leg['contract'].price
                } for leg in position['strategy'].legs]
            )
            
            logger.debug(f"Closed {position['strategy_name']}: P&L ${pnl:.2f} ({pnl_pct:.1f}%)")
            
            return trade
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return BacktestTrade(
                strategy_name=position['strategy_name'],
                entry_date=position['entry_date'],
                exit_date=exit_date,
                entry_price=0, exit_price=0, quantity=0,
                pnl=0, pnl_pct=0, max_profit=0, max_loss=0,
                days_held=0, exit_reason="error",
                underlying_entry=0, underlying_exit=current_price,
                greeks_entry={}, legs=[]
            )
    
    def _calculate_position_value(self, position: Dict[str, Any], 
                                 current_price: float, date: datetime) -> float:
        """Calculate current value of options position"""
        try:
            # Simplified position valuation
            # In production, would use Black-Scholes with current volatility and time decay
            
            strategy = position['strategy']
            days_held = (date - position['entry_date']).days
            time_decay_factor = max(0.1, 1.0 - (days_held / 45))  # Simplified time decay
            
            # Calculate intrinsic value
            total_intrinsic = 0
            for leg in strategy.legs:
                if leg['contract'].option_type == OptionType.CALL:
                    intrinsic = max(0, current_price - leg['contract'].strike)
                else:
                    intrinsic = max(0, leg['contract'].strike - current_price)
                
                total_intrinsic += intrinsic * leg['quantity']
            
            # Add time value (simplified)
            time_value = abs(strategy.total_premium) * time_decay_factor
            
            # Total position value
            position_value = (total_intrinsic + time_value) * position['contracts'] * 100
            
            return max(0, position_value)
            
        except Exception as e:
            logger.warning(f"Error calculating position value: {str(e)}")
            return 0
    
    def _calculate_position_greeks(self, strategy: OptionStrategy) -> Dict[str, float]:
        """Calculate position Greeks (simplified)"""
        try:
            total_delta = sum(leg['quantity'] * (leg['contract'].delta or 0) for leg in strategy.legs)
            total_gamma = sum(leg['quantity'] * (leg['contract'].gamma or 0) for leg in strategy.legs)
            total_theta = sum(leg['quantity'] * (leg['contract'].theta or 0) for leg in strategy.legs)
            total_vega = sum(leg['quantity'] * (leg['contract'].vega or 0) for leg in strategy.legs)
            
            return {
                'delta': total_delta,
                'gamma': total_gamma,
                'theta': total_theta,
                'vega': total_vega
            }
            
        except Exception as e:
            logger.warning(f"Error calculating Greeks: {str(e)}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    def _calculate_backtest_results(self, trades: List[BacktestTrade], 
                                   equity_curve: List[float], 
                                   start_date: datetime, end_date: datetime,
                                   strategy_name: str) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        try:
            if not trades:
                return BacktestResults(
                    start_date=start_date, end_date=end_date,
                    initial_capital=self.initial_capital, final_capital=self.current_capital,
                    total_return=0, total_return_pct=0, max_drawdown=0, max_drawdown_pct=0,
                    sharpe_ratio=0, win_rate=0, profit_factor=0, total_trades=0,
                    winning_trades=0, losing_trades=0, avg_win=0, avg_loss=0,
                    best_trade=0, worst_trade=0, avg_days_held=0,
                    trades=[], equity_curve=pd.Series(), monthly_returns=pd.Series(),
                    strategy_breakdown={}
                )
            
            # Basic statistics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.pnl > 0])
            losing_trades = total_trades - winning_trades
            
            total_return = self.current_capital - self.initial_capital
            total_return_pct = (total_return / self.initial_capital) * 100
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # P&L statistics
            wins = [t.pnl for t in trades if t.pnl > 0]
            losses = [t.pnl for t in trades if t.pnl < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            best_trade = max([t.pnl for t in trades]) if trades else 0
            worst_trade = min([t.pnl for t in trades]) if trades else 0
            
            # Profit factor
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Time statistics
            avg_days_held = np.mean([t.days_held for t in trades]) if trades else 0
            
            # Drawdown analysis
            equity_series = pd.Series(equity_curve)
            running_max = equity_series.expanding().max()
            drawdown = equity_series - running_max
            max_drawdown = drawdown.min()
            max_drawdown_pct = (max_drawdown / running_max.loc[drawdown.idxmin()]) * 100
            
            # Sharpe ratio (simplified)
            returns = equity_series.pct_change().dropna()
            if len(returns) > 1 and returns.std() > 0:
                sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            else:
                sharpe_ratio = 0
            
            # Monthly returns
            monthly_returns = pd.Series(dtype=float)
            if len(equity_curve) > 30:
                monthly_data = equity_series[::30]  # Approximate monthly sampling
                monthly_returns = monthly_data.pct_change().dropna()
            
            # Strategy breakdown
            strategy_breakdown = {
                strategy_name: {
                    'trades': total_trades,
                    'win_rate': win_rate,
                    'avg_pnl': np.mean([t.pnl for t in trades]),
                    'total_pnl': sum([t.pnl for t in trades])
                }
            }
            
            return BacktestResults(
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
                final_capital=self.current_capital,
                total_return=total_return,
                total_return_pct=total_return_pct,
                max_drawdown=max_drawdown,
                max_drawdown_pct=max_drawdown_pct,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                avg_win=avg_win,
                avg_loss=avg_loss,
                best_trade=best_trade,
                worst_trade=worst_trade,
                avg_days_held=avg_days_held,
                trades=trades,
                equity_curve=equity_series,
                monthly_returns=monthly_returns,
                strategy_breakdown=strategy_breakdown
            )
            
        except Exception as e:
            logger.error(f"Error calculating backtest results: {str(e)}")
            raise
