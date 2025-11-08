import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from src.data.market import MarketDataFetcher
from src.trading.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Modern Portfolio Theory (MPT) based portfolio optimization"""
    
    def __init__(self, market_fetcher: MarketDataFetcher):
        self.market_fetcher = market_fetcher
        
    def calculate_portfolio_metrics(self, symbols: List[str], weights: List[float],
                                  period: str = "1y") -> Dict[str, float]:
        """Calculate portfolio risk and return metrics"""
        try:
            # Get historical data
            returns_data = self._get_returns_matrix(symbols, period)
            if returns_data.empty:
                return {"error": "Unable to fetch data for symbols"}
            
            weights = np.array(weights)
            
            # Calculate portfolio returns
            portfolio_returns = (returns_data * weights).sum(axis=1)
            
            # Calculate metrics
            annual_return = portfolio_returns.mean() * 252  # Annualized
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(portfolio_returns, 5)
            
            return {
                "annual_return": annual_return,
                "annual_volatility": annual_volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "var_95": var_95,
                "portfolio_returns": portfolio_returns.tolist(),
                "symbols": symbols,
                "weights": weights.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {"error": str(e)}
    
    def optimize_portfolio(self, symbols: List[str], 
                          optimization_type: str = "max_sharpe",
                          target_return: Optional[float] = None,
                          risk_tolerance: float = 0.15) -> Dict[str, Any]:
        """Optimize portfolio using Modern Portfolio Theory"""
        try:
            # Get returns data
            returns_data = self._get_returns_matrix(symbols)
            if returns_data.empty:
                return {"error": "Unable to fetch data for optimization"}
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_data.mean() * 252  # Annualized
            cov_matrix = returns_data.cov() * 252  # Annualized
            
            n_assets = len(symbols)
            
            # Optimization constraints
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
            bounds = tuple((0, 1) for _ in range(n_assets))  # Long-only positions
            
            # Initial guess (equal weights)
            initial_guess = np.array([1/n_assets] * n_assets)
            
            if optimization_type == "max_sharpe":
                # Maximize Sharpe ratio
                def negative_sharpe(weights):
                    portfolio_return = np.sum(expected_returns * weights)
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    return -portfolio_return / portfolio_vol if portfolio_vol > 0 else -999
                
                result = minimize(negative_sharpe, initial_guess, method='SLSQP',
                                bounds=bounds, constraints=constraints)
                
            elif optimization_type == "min_variance":
                # Minimize variance
                def portfolio_variance(weights):
                    return np.dot(weights.T, np.dot(cov_matrix, weights))
                
                result = minimize(portfolio_variance, initial_guess, method='SLSQP',
                                bounds=bounds, constraints=constraints)
                
            elif optimization_type == "target_return" and target_return:
                # Minimize variance subject to target return
                def portfolio_variance(weights):
                    return np.dot(weights.T, np.dot(cov_matrix, weights))
                
                # Add target return constraint
                constraints.append({
                    'type': 'eq', 
                    'fun': lambda x: np.sum(expected_returns * x) - target_return
                })
                
                result = minimize(portfolio_variance, initial_guess, method='SLSQP',
                                bounds=bounds, constraints=constraints)
                
            elif optimization_type == "risk_parity":
                # Risk parity optimization
                def risk_parity_objective(weights):
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                    contrib = weights * marginal_contrib
                    target_contrib = portfolio_vol / n_assets
                    return np.sum((contrib - target_contrib) ** 2)
                
                result = minimize(risk_parity_objective, initial_guess, method='SLSQP',
                                bounds=bounds, constraints=constraints)
            
            else:
                return {"error": f"Unknown optimization type: {optimization_type}"}
            
            if not result.success:
                return {"error": f"Optimization failed: {result.message}"}
            
            # Extract optimal weights
            optimal_weights = result.x
            
            # Calculate optimized portfolio metrics
            portfolio_metrics = self.calculate_portfolio_metrics(symbols, optimal_weights.tolist())
            
            # Generate efficient frontier for visualization
            efficient_frontier = self._generate_efficient_frontier(symbols, returns_data, 
                                                                 expected_returns, cov_matrix)
            
            return {
                "optimization_type": optimization_type,
                "symbols": symbols,
                "optimal_weights": optimal_weights.tolist(),
                "portfolio_metrics": portfolio_metrics,
                "efficient_frontier": efficient_frontier,
                "optimization_success": True
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            return {"error": str(e)}
    
    def _get_returns_matrix(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """Get returns matrix for multiple symbols"""
        price_data = {}
        
        for symbol in symbols:
            try:
                data = self.market_fetcher.get_stock_data(symbol, period=period)
                if not data.empty:
                    price_data[symbol] = data['Close']
            except Exception as e:
                logger.warning(f"Could not fetch data for {symbol}: {str(e)}")
                continue
        
        if not price_data:
            return pd.DataFrame()
        
        # Create price DataFrame
        prices_df = pd.DataFrame(price_data)
        
        # Calculate returns
        returns_df = prices_df.pct_change().dropna()
        
        return returns_df
    
    def _generate_efficient_frontier(self, symbols: List[str], returns_data: pd.DataFrame,
                                   expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                                   num_points: int = 50) -> List[Dict]:
        """Generate efficient frontier points"""
        try:
            n_assets = len(symbols)
            
            # Calculate return range
            min_ret = expected_returns.min()
            max_ret = expected_returns.max()
            target_returns = np.linspace(min_ret, max_ret, num_points)
            
            efficient_portfolios = []
            
            for target_ret in target_returns:
                try:
                    # Minimize variance subject to target return
                    constraints = [
                        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                        {'type': 'eq', 'fun': lambda x: np.sum(expected_returns * x) - target_ret}
                    ]
                    bounds = tuple((0, 1) for _ in range(n_assets))
                    initial_guess = np.array([1/n_assets] * n_assets)
                    
                    def portfolio_variance(weights):
                        return np.dot(weights.T, np.dot(cov_matrix, weights))
                    
                    result = minimize(portfolio_variance, initial_guess, method='SLSQP',
                                    bounds=bounds, constraints=constraints)
                    
                    if result.success:
                        weights = result.x
                        portfolio_return = np.sum(expected_returns * weights)
                        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
                        
                        efficient_portfolios.append({
                            "return": portfolio_return,
                            "volatility": portfolio_vol,
                            "sharpe_ratio": sharpe_ratio,
                            "weights": weights.tolist()
                        })
                        
                except:
                    continue
            
            return efficient_portfolios
            
        except Exception as e:
            logger.error(f"Error generating efficient frontier: {str(e)}")
            return []

class PortfolioRebalancer:
    """Portfolio rebalancing strategies"""
    
    def __init__(self, portfolio_manager: PortfolioManager, optimizer: PortfolioOptimizer):
        self.portfolio_manager = portfolio_manager
        self.optimizer = optimizer
        
    def analyze_rebalancing_need(self, target_weights: Dict[str, float],
                               rebalance_threshold: float = 0.05) -> Dict[str, Any]:
        """Analyze if portfolio needs rebalancing"""
        try:
            # Get current portfolio
            current_positions = self.portfolio_manager.get_current_positions()
            if not current_positions:
                return {"rebalancing_needed": False, "reason": "No current positions"}
            
            # Calculate current weights
            total_value = sum(pos.market_value for pos in current_positions)
            current_weights = {pos.symbol: pos.market_value / total_value 
                             for pos in current_positions}
            
            # Calculate weight differences
            weight_differences = {}
            max_deviation = 0
            
            for symbol, target_weight in target_weights.items():
                current_weight = current_weights.get(symbol, 0)
                difference = abs(current_weight - target_weight)
                weight_differences[symbol] = {
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "difference": difference,
                    "deviation_pct": difference / target_weight if target_weight > 0 else 0
                }
                max_deviation = max(max_deviation, difference)
            
            rebalancing_needed = max_deviation > rebalance_threshold
            
            # Calculate rebalancing trades
            rebalancing_trades = []
            if rebalancing_needed:
                rebalancing_trades = self._calculate_rebalancing_trades(
                    current_weights, target_weights, total_value)
            
            return {
                "rebalancing_needed": rebalancing_needed,
                "max_deviation": max_deviation,
                "threshold": rebalance_threshold,
                "weight_differences": weight_differences,
                "total_portfolio_value": total_value,
                "rebalancing_trades": rebalancing_trades,
                "estimated_trading_cost": self._estimate_trading_cost(rebalancing_trades)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing rebalancing need: {str(e)}")
            return {"error": str(e)}
    
    def execute_rebalancing(self, target_weights: Dict[str, float],
                          rebalance_threshold: float = 0.05,
                          max_trading_cost: float = 0.005) -> Dict[str, Any]:
        """Execute portfolio rebalancing"""
        try:
            # Analyze rebalancing need
            analysis = self.analyze_rebalancing_need(target_weights, rebalance_threshold)
            
            if "error" in analysis:
                return analysis
                
            if not analysis["rebalancing_needed"]:
                return {
                    "rebalancing_executed": False,
                    "reason": "Portfolio within rebalancing threshold",
                    "analysis": analysis
                }
            
            # Check if trading cost is acceptable
            if analysis["estimated_trading_cost"] > max_trading_cost:
                return {
                    "rebalancing_executed": False,
                    "reason": f"Trading cost ({analysis['estimated_trading_cost']:.3f}) exceeds maximum ({max_trading_cost:.3f})",
                    "analysis": analysis
                }
            
            # Execute trades
            executed_trades = []
            failed_trades = []
            
            for trade in analysis["rebalancing_trades"]:
                try:
                    if trade["action"] == "buy":
                        result = self.portfolio_manager.paper_trader.place_order(
                            symbol=trade["symbol"],
                            quantity=trade["quantity"],
                            side="buy",
                            order_type="market"
                        )
                    elif trade["action"] == "sell":
                        result = self.portfolio_manager.paper_trader.place_order(
                            symbol=trade["symbol"],
                            quantity=abs(trade["quantity"]),
                            side="sell",
                            order_type="market"
                        )
                    
                    if result.get("success", False):
                        executed_trades.append({
                            "symbol": trade["symbol"],
                            "action": trade["action"],
                            "quantity": trade["quantity"],
                            "estimated_value": trade["estimated_value"],
                            "order_id": result.get("order_id")
                        })
                    else:
                        failed_trades.append({
                            "symbol": trade["symbol"],
                            "action": trade["action"],
                            "reason": result.get("error", "Unknown error")
                        })
                        
                except Exception as e:
                    failed_trades.append({
                        "symbol": trade["symbol"],
                        "action": trade["action"],
                        "reason": str(e)
                    })
            
            return {
                "rebalancing_executed": True,
                "executed_trades": executed_trades,
                "failed_trades": failed_trades,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error executing rebalancing: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_rebalancing_trades(self, current_weights: Dict[str, float],
                                    target_weights: Dict[str, float],
                                    total_value: float) -> List[Dict]:
        """Calculate trades needed for rebalancing"""
        trades = []
        
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            weight_change = target_weight - current_weight
            
            if abs(weight_change) > 0.001:  # Only trade if change is significant
                value_change = weight_change * total_value
                
                # Get current price to calculate quantity
                try:
                    current_price = self.optimizer.market_fetcher.get_current_price(symbol)
                    if current_price:
                        quantity = int(abs(value_change) / current_price)
                        if quantity > 0:
                            trades.append({
                                "symbol": symbol,
                                "action": "buy" if weight_change > 0 else "sell",
                                "quantity": quantity,
                                "current_weight": current_weight,
                                "target_weight": target_weight,
                                "weight_change": weight_change,
                                "estimated_value": value_change,
                                "current_price": current_price
                            })
                except:
                    continue
        
        return trades
    
    def _estimate_trading_cost(self, trades: List[Dict]) -> float:
        """Estimate trading cost as percentage of portfolio value"""
        if not trades:
            return 0.0
        
        total_trade_value = sum(abs(trade["estimated_value"]) for trade in trades)
        # Assume 0.1% trading cost (commission + spread)
        trading_cost_rate = 0.001
        
        return total_trade_value * trading_cost_rate
    
    def schedule_rebalancing(self, target_weights: Dict[str, float],
                           frequency: str = "monthly",
                           rebalance_threshold: float = 0.05) -> Dict[str, Any]:
        """Schedule automatic rebalancing"""
        try:
            # Calculate next rebalancing date
            now = datetime.now()
            
            if frequency == "monthly":
                if now.month == 12:
                    next_rebalance = now.replace(year=now.year + 1, month=1, day=1)
                else:
                    next_rebalance = now.replace(month=now.month + 1, day=1)
            elif frequency == "quarterly":
                next_quarter_month = ((now.month - 1) // 3 + 1) * 3 + 1
                if next_quarter_month > 12:
                    next_rebalance = now.replace(year=now.year + 1, month=1, day=1)
                else:
                    next_rebalance = now.replace(month=next_quarter_month, day=1)
            elif frequency == "annually":
                next_rebalance = now.replace(year=now.year + 1, month=1, day=1)
            else:
                return {"error": f"Unknown frequency: {frequency}"}
            
            return {
                "rebalancing_scheduled": True,
                "target_weights": target_weights,
                "frequency": frequency,
                "threshold": rebalance_threshold,
                "next_rebalance_date": next_rebalance.isoformat(),
                "current_analysis": self.analyze_rebalancing_need(target_weights, rebalance_threshold)
            }
            
        except Exception as e:
            logger.error(f"Error scheduling rebalancing: {str(e)}")
            return {"error": str(e)}
