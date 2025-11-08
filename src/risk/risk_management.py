import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats
from sklearn.covariance import EmpiricalCovariance
import warnings
warnings.filterwarnings('ignore')

from src.data.market import MarketDataFetcher

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    var_95: float  # Value at Risk at 95% confidence
    var_99: float  # Value at Risk at 99% confidence
    expected_shortfall_95: float  # Conditional VaR (Expected Shortfall)
    expected_shortfall_99: float
    max_drawdown: float
    volatility: float  # Annualized volatility
    sharpe_ratio: float
    sortino_ratio: float
    beta: Optional[float] = None  # Beta vs market
    correlation_to_market: Optional[float] = None

@dataclass
class PositionSizing:
    symbol: str
    recommended_position_size: float  # As percentage of portfolio
    max_position_size: float
    kelly_criterion: float
    risk_adjusted_size: float
    stop_loss_level: float
    take_profit_level: float
    risk_reward_ratio: float
    position_value: float  # Dollar amount

@dataclass
class PortfolioRisk:
    total_var_95: float
    total_var_99: float
    diversification_ratio: float
    portfolio_beta: float
    correlation_matrix: Dict[str, Dict[str, float]]
    concentration_risk: Dict[str, float]  # Risk contribution by asset
    stress_test_scenarios: Dict[str, float]
    risk_budget_allocation: Dict[str, float]

class RiskCalculator:
    """Calculate various risk metrics for assets and portfolios"""
    
    def __init__(self):
        self.market_fetcher = MarketDataFetcher()
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk using historical simulation"""
        try:
            if len(returns) < 30:
                return 0.0
            
            return np.percentile(returns, (1 - confidence_level) * 100)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0
    
    def calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            var = self.calculate_var(returns, confidence_level)
            return returns[returns <= var].mean()
            
        except Exception as e:
            logger.error(f"Error calculating Expected Shortfall: {str(e)}")
            return 0.0
    
    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + prices.pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            excess_returns = returns.mean() - risk_free_rate / 252  # Daily risk-free rate
            return excess_returns / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (using downside deviation)"""
        try:
            excess_returns = returns.mean() - risk_free_rate / 252
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
            return excess_returns / downside_std * np.sqrt(252) if downside_std > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0
    
    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> Tuple[float, float]:
        """Calculate beta and correlation with market"""
        try:
            if len(asset_returns) != len(market_returns) or len(asset_returns) < 30:
                return 0.0, 0.0
            
            # Align indices
            aligned_data = pd.DataFrame({'asset': asset_returns, 'market': market_returns}).dropna()
            
            if len(aligned_data) < 30:
                return 0.0, 0.0
            
            covariance = aligned_data['asset'].cov(aligned_data['market'])
            market_variance = aligned_data['market'].var()
            beta = covariance / market_variance if market_variance > 0 else 0
            
            correlation = aligned_data['asset'].corr(aligned_data['market'])
            
            return beta, correlation
            
        except Exception as e:
            logger.error(f"Error calculating beta: {str(e)}")
            return 0.0, 0.0
    
    def calculate_risk_metrics(self, symbol: str, period: str = "1y", 
                             benchmark: str = "SPY") -> RiskMetrics:
        """Calculate comprehensive risk metrics for a symbol"""
        try:
            # Get asset data
            asset_data = self.market_fetcher.get_stock_data(symbol, period=period)
            if asset_data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            asset_returns = asset_data['Close'].pct_change().dropna()
            
            # Get benchmark data for beta calculation
            beta, correlation = 0.0, 0.0
            try:
                benchmark_data = self.market_fetcher.get_stock_data(benchmark, period=period)
                if not benchmark_data.empty:
                    benchmark_returns = benchmark_data['Close'].pct_change().dropna()
                    beta, correlation = self.calculate_beta(asset_returns, benchmark_returns)
            except:
                pass
            
            return RiskMetrics(
                var_95=self.calculate_var(asset_returns, 0.95),
                var_99=self.calculate_var(asset_returns, 0.99),
                expected_shortfall_95=self.calculate_expected_shortfall(asset_returns, 0.95),
                expected_shortfall_99=self.calculate_expected_shortfall(asset_returns, 0.99),
                max_drawdown=self.calculate_max_drawdown(asset_data['Close']),
                volatility=asset_returns.std() * np.sqrt(252),  # Annualized
                sharpe_ratio=self.calculate_sharpe_ratio(asset_returns),
                sortino_ratio=self.calculate_sortino_ratio(asset_returns),
                beta=beta,
                correlation_to_market=correlation
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics for {symbol}: {str(e)}")
            raise

class PositionSizer:
    """Calculate optimal position sizes based on risk management principles"""
    
    def __init__(self, portfolio_value: float, max_risk_per_trade: float = 0.02):
        self.portfolio_value = portfolio_value
        self.max_risk_per_trade = max_risk_per_trade  # 2% max risk per trade
        self.risk_calculator = RiskCalculator()
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion for position sizing"""
        try:
            if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
                return 0.0
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            b = avg_win / abs(avg_loss)
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Cap Kelly at reasonable levels (25% max)
            return max(0, min(kelly_fraction, 0.25))
            
        except Exception as e:
            logger.error(f"Error calculating Kelly criterion: {str(e)}")
            return 0.0
    
    def calculate_risk_based_position_size(self, symbol: str, entry_price: float,
                                         stop_loss_price: float, 
                                         target_price: Optional[float] = None) -> PositionSizing:
        """Calculate position size based on risk management"""
        try:
            if stop_loss_price <= 0 or entry_price <= 0:
                raise ValueError("Invalid prices")
            
            # Risk per share
            risk_per_share = abs(entry_price - stop_loss_price)
            
            # Maximum dollar risk for this trade
            max_dollar_risk = self.portfolio_value * self.max_risk_per_trade
            
            # Position size based on risk
            risk_based_shares = max_dollar_risk / risk_per_share if risk_per_share > 0 else 0
            risk_based_value = risk_based_shares * entry_price
            risk_based_percentage = risk_based_value / self.portfolio_value
            
            # Get risk metrics for the symbol
            try:
                risk_metrics = self.risk_calculator.calculate_risk_metrics(symbol)
                volatility = risk_metrics.volatility
            except:
                volatility = 0.20  # Default 20% volatility
            
            # Volatility-adjusted position size
            base_volatility = 0.15  # 15% base volatility
            volatility_adjustment = base_volatility / max(volatility, 0.05)
            volatility_adjusted_percentage = risk_based_percentage * volatility_adjustment
            
            # Calculate Kelly Criterion (simplified)
            # Assuming 55% win rate, 1.5:1 reward:risk for estimation
            win_rate = 0.55
            if target_price:
                reward = abs(target_price - entry_price)
                risk = risk_per_share
                reward_risk_ratio = reward / risk if risk > 0 else 1.5
            else:
                reward_risk_ratio = 1.5  # Default assumption
            
            kelly_percentage = self.calculate_kelly_criterion(win_rate, reward_risk_ratio, 1.0)
            
            # Final position size (most conservative approach)
            recommended_percentage = min(
                risk_based_percentage,
                volatility_adjusted_percentage,
                kelly_percentage,
                0.20  # Max 20% of portfolio in single position
            )
            
            # Calculate final values
            position_value = self.portfolio_value * recommended_percentage
            max_position_value = self.portfolio_value * min(0.20, kelly_percentage * 1.5)
            
            # Take profit level (if not provided)
            if not target_price:
                target_price = entry_price * (1 + (risk_per_share / entry_price) * 2)  # 2:1 R:R
            
            return PositionSizing(
                symbol=symbol,
                recommended_position_size=recommended_percentage,
                max_position_size=max_position_value / self.portfolio_value,
                kelly_criterion=kelly_percentage,
                risk_adjusted_size=volatility_adjusted_percentage,
                stop_loss_level=stop_loss_price,
                take_profit_level=target_price,
                risk_reward_ratio=abs(target_price - entry_price) / risk_per_share if risk_per_share > 0 else 0,
                position_value=position_value
            )
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {str(e)}")
            raise
    
    def calculate_portfolio_position_sizes(self, positions: Dict[str, Dict]) -> Dict[str, PositionSizing]:
        """Calculate position sizes for multiple positions"""
        try:
            results = {}
            
            for symbol, position_info in positions.items():
                try:
                    sizing = self.calculate_risk_based_position_size(
                        symbol=symbol,
                        entry_price=position_info['entry_price'],
                        stop_loss_price=position_info['stop_loss'],
                        target_price=position_info.get('target_price')
                    )
                    results[symbol] = sizing
                except Exception as e:
                    logger.warning(f"Error calculating position size for {symbol}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating portfolio position sizes: {str(e)}")
            return {}

class PortfolioRiskManager:
    """Comprehensive portfolio risk management"""
    
    def __init__(self):
        self.risk_calculator = RiskCalculator()
    
    def calculate_portfolio_var(self, positions: Dict[str, float], 
                              confidence_level: float = 0.95,
                              period: str = "1y") -> Dict[str, float]:
        """Calculate portfolio VaR using Monte Carlo simulation"""
        try:
            if not positions:
                return {"portfolio_var": 0.0, "individual_vars": {}}
            
            # Get returns for all positions
            returns_data = {}
            for symbol, weight in positions.items():
                try:
                    data = self.risk_calculator.market_fetcher.get_stock_data(symbol, period=period)
                    if not data.empty:
                        returns_data[symbol] = data['Close'].pct_change().dropna()
                except:
                    continue
            
            if not returns_data:
                return {"portfolio_var": 0.0, "individual_vars": {}}
            
            # Align all return series
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if returns_df.empty:
                return {"portfolio_var": 0.0, "individual_vars": {}}
            
            # Calculate individual VaRs
            individual_vars = {}
            for symbol in returns_df.columns:
                if symbol in positions:
                    var = self.risk_calculator.calculate_var(returns_df[symbol], confidence_level)
                    individual_vars[symbol] = var * positions[symbol]
            
            # Calculate portfolio returns
            portfolio_returns = pd.Series(index=returns_df.index, dtype=float)
            for symbol, weight in positions.items():
                if symbol in returns_df.columns:
                    portfolio_returns += returns_df[symbol] * weight
            
            # Portfolio VaR
            portfolio_var = self.risk_calculator.calculate_var(portfolio_returns, confidence_level)
            
            return {
                "portfolio_var": portfolio_var,
                "individual_vars": individual_vars,
                "diversification_benefit": sum(individual_vars.values()) - abs(portfolio_var)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {str(e)}")
            return {"portfolio_var": 0.0, "individual_vars": {}}
    
    def calculate_correlation_matrix(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """Calculate correlation matrix for portfolio assets"""
        try:
            returns_data = {}
            
            for symbol in symbols:
                try:
                    data = self.risk_calculator.market_fetcher.get_stock_data(symbol, period=period)
                    if not data.empty:
                        returns_data[symbol] = data['Close'].pct_change().dropna()
                except:
                    continue
            
            if len(returns_data) < 2:
                return pd.DataFrame()
            
            returns_df = pd.DataFrame(returns_data).dropna()
            return returns_df.corr()
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            return pd.DataFrame()
    
    def assess_portfolio_risk(self, positions: Dict[str, float], 
                            portfolio_value: float = 100000) -> PortfolioRisk:
        """Comprehensive portfolio risk assessment"""
        try:
            symbols = list(positions.keys())
            
            # Portfolio VaR
            var_results = self.calculate_portfolio_var(positions, 0.95)
            var_99_results = self.calculate_portfolio_var(positions, 0.99)
            
            # Correlation matrix
            correlation_matrix = self.calculate_correlation_matrix(symbols)
            correlation_dict = {}
            if not correlation_matrix.empty:
                for symbol1 in correlation_matrix.index:
                    correlation_dict[symbol1] = correlation_matrix.loc[symbol1].to_dict()
            
            # Portfolio beta (weighted average)
            portfolio_beta = 0.0
            for symbol, weight in positions.items():
                try:
                    risk_metrics = self.risk_calculator.calculate_risk_metrics(symbol)
                    if risk_metrics.beta:
                        portfolio_beta += risk_metrics.beta * weight
                except:
                    continue
            
            # Concentration risk (risk contribution by asset)
            concentration_risk = {}
            total_var = abs(var_results["portfolio_var"])
            
            for symbol, weight in positions.items():
                # Simple concentration measure: weight * individual variance
                individual_var = var_results["individual_vars"].get(symbol, 0)
                concentration_risk[symbol] = abs(individual_var) / total_var if total_var > 0 else 0
            
            # Diversification ratio
            individual_vol_sum = 0
            portfolio_vol = 0
            
            for symbol, weight in positions.items():
                try:
                    risk_metrics = self.risk_calculator.calculate_risk_metrics(symbol)
                    individual_vol_sum += weight * risk_metrics.volatility
                    portfolio_vol += (weight ** 2) * (risk_metrics.volatility ** 2)
                    
                    # Add correlation effects (simplified)
                    for other_symbol, other_weight in positions.items():
                        if symbol != other_symbol and not correlation_matrix.empty:
                            if symbol in correlation_matrix.index and other_symbol in correlation_matrix.columns:
                                corr = correlation_matrix.loc[symbol, other_symbol]
                                other_risk = self.risk_calculator.calculate_risk_metrics(other_symbol)
                                portfolio_vol += 2 * weight * other_weight * risk_metrics.volatility * other_risk.volatility * corr
                except:
                    continue
            
            portfolio_vol = np.sqrt(portfolio_vol) if portfolio_vol > 0 else 0.1
            diversification_ratio = individual_vol_sum / portfolio_vol if portfolio_vol > 0 else 1.0
            
            # Stress test scenarios
            stress_scenarios = {
                "market_crash_20pct": var_results["portfolio_var"] * 3,  # 3x normal VaR
                "high_correlation_scenario": var_results["portfolio_var"] * 2,
                "interest_rate_shock": var_results["portfolio_var"] * 1.5,
                "sector_specific_shock": max(concentration_risk.values()) * portfolio_value if concentration_risk else 0
            }
            
            # Risk budget allocation
            risk_budget = {}
            for symbol, contribution in concentration_risk.items():
                risk_budget[symbol] = contribution
            
            return PortfolioRisk(
                total_var_95=var_results["portfolio_var"],
                total_var_99=var_99_results["portfolio_var"],
                diversification_ratio=diversification_ratio,
                portfolio_beta=portfolio_beta,
                correlation_matrix=correlation_dict,
                concentration_risk=concentration_risk,
                stress_test_scenarios=stress_scenarios,
                risk_budget_allocation=risk_budget
            )
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {str(e)}")
            raise

class RiskMonitor:
    """Real-time risk monitoring and alerts"""
    
    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value
        self.risk_manager = PortfolioRiskManager()
        self.position_sizer = PositionSizer(portfolio_value)
        
    def check_risk_limits(self, positions: Dict[str, float]) -> Dict[str, Any]:
        """Check if portfolio exceeds risk limits"""
        try:
            risk_assessment = self.risk_manager.assess_portfolio_risk(positions, self.portfolio_value)
            
            alerts = []
            risk_status = "normal"
            
            # VaR limits (5% of portfolio)
            var_limit = 0.05
            if abs(risk_assessment.total_var_95) > var_limit:
                alerts.append({
                    "type": "var_exceeded",
                    "message": f"Portfolio VaR ({abs(risk_assessment.total_var_95):.1%}) exceeds limit ({var_limit:.1%})",
                    "severity": "high"
                })
                risk_status = "high"
            
            # Concentration limits (no single asset > 20%)
            max_concentration = max(positions.values()) if positions else 0
            if max_concentration > 0.20:
                alerts.append({
                    "type": "concentration_risk",
                    "message": f"Single asset concentration ({max_concentration:.1%}) exceeds 20% limit",
                    "severity": "medium"
                })
                risk_status = "medium" if risk_status == "normal" else risk_status
            
            # Correlation risk (average correlation > 0.7)
            if risk_assessment.correlation_matrix:
                correlations = []
                symbols = list(risk_assessment.correlation_matrix.keys())
                for i, symbol1 in enumerate(symbols):
                    for symbol2 in symbols[i+1:]:
                        if symbol2 in risk_assessment.correlation_matrix[symbol1]:
                            corr = risk_assessment.correlation_matrix[symbol1][symbol2]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                
                if correlations and np.mean(correlations) > 0.7:
                    alerts.append({
                        "type": "high_correlation",
                        "message": f"Average portfolio correlation ({np.mean(correlations):.2f}) indicates low diversification",
                        "severity": "medium"
                    })
            
            # Beta risk (portfolio beta > 1.5 or < 0)
            if risk_assessment.portfolio_beta > 1.5:
                alerts.append({
                    "type": "high_beta",
                    "message": f"Portfolio beta ({risk_assessment.portfolio_beta:.2f}) indicates high market sensitivity",
                    "severity": "low"
                })
            
            return {
                "risk_status": risk_status,
                "alerts": alerts,
                "risk_metrics": {
                    "var_95": risk_assessment.total_var_95,
                    "portfolio_beta": risk_assessment.portfolio_beta,
                    "diversification_ratio": risk_assessment.diversification_ratio,
                    "max_concentration": max_concentration
                },
                "recommendations": self._generate_risk_recommendations(alerts, risk_assessment)
            }
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
            return {"risk_status": "unknown", "alerts": [], "error": str(e)}
    
    def _generate_risk_recommendations(self, alerts: List[Dict], risk_assessment: PortfolioRisk) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        for alert in alerts:
            if alert["type"] == "var_exceeded":
                recommendations.append("Consider reducing position sizes or adding hedging instruments")
            elif alert["type"] == "concentration_risk":
                recommendations.append("Diversify portfolio by reducing large positions")
            elif alert["type"] == "high_correlation":
                recommendations.append("Add assets from different sectors or asset classes")
            elif alert["type"] == "high_beta":
                recommendations.append("Consider adding defensive stocks or bonds to reduce market sensitivity")
        
        # General recommendations based on metrics
        if risk_assessment.diversification_ratio < 1.2:
            recommendations.append("Portfolio lacks diversification benefits - consider broader asset allocation")
        
        if not recommendations:
            recommendations.append("Portfolio risk metrics are within acceptable ranges")
        
        return recommendations
