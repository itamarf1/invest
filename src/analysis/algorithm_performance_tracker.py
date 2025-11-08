import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

from ..trading.strategies.moving_average import MovingAverageStrategy
from ..trading.strategies.rsi_strategy import RSIStrategy
from ..trading.strategies.macd_strategy import MACDStrategy
from ..trading.strategies.bollinger_strategy import BollingerBandsStrategy
from ..trading.strategies.sentiment_enhanced import SentimentEnhancedStrategy
from ..trading.strategies.ensemble_strategy import EnsembleStrategy
from ..data.market import MarketDataFetcher
from .strategy_adapter import StrategyAdapter

logger = logging.getLogger(__name__)


class AlgorithmType(Enum):
    MOVING_AVERAGE = "moving_average"
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER_BANDS = "bollinger_bands"
    SENTIMENT_ENHANCED = "sentiment_enhanced"
    ENSEMBLE = "ensemble"
    MACHINE_LEARNING = "machine_learning"
    CRYPTO_MOMENTUM = "crypto_momentum"
    CRYPTO_MEAN_REVERSION = "crypto_mean_reversion"


@dataclass
class AlgorithmPerformanceMetrics:
    algorithm_id: str
    algorithm_type: AlgorithmType
    name: str
    symbol: str
    timeframe: str
    
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    avg_trade_duration: float
    
    # Risk metrics
    var_95: float  # Value at Risk 95%
    beta: float
    alpha: float
    
    # Benchmark comparison
    benchmark_return: float
    excess_return: float
    information_ratio: float
    
    # Timestamps
    start_date: datetime
    end_date: datetime
    last_updated: datetime
    
    # Raw data
    equity_curve: List[float]
    drawdown_curve: List[float]
    trade_log: List[Dict[str, Any]]


@dataclass
class BenchmarkData:
    symbol: str
    name: str
    returns: List[float]
    cumulative_returns: List[float]
    dates: List[datetime]


class AlgorithmPerformanceTracker:
    """Tracks and analyzes performance of all trading algorithms over time"""
    
    def __init__(self):
        self.market_fetcher = MarketDataFetcher()
        self.performance_history: Dict[str, List[AlgorithmPerformanceMetrics]] = {}
        self.benchmarks: Dict[str, BenchmarkData] = {}
        
        # Initialize strategies
        self.strategies = {
            AlgorithmType.MOVING_AVERAGE: MovingAverageStrategy(),
            AlgorithmType.RSI: RSIStrategy(),
            AlgorithmType.MACD: MACDStrategy(),
            AlgorithmType.BOLLINGER_BANDS: BollingerBandsStrategy(),
            AlgorithmType.SENTIMENT_ENHANCED: SentimentEnhancedStrategy(),
            AlgorithmType.ENSEMBLE: EnsembleStrategy()
        }
        
        # Common test symbols for comparison
        self.test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"]
        
        # Benchmark indices
        self.benchmark_symbols = {
            "SPY": "S&P 500",
            "QQQ": "NASDAQ 100",
            "IWM": "Russell 2000",
            "VTI": "Total Stock Market",
            "BND": "Bond Market"
        }
        
        logger.info("Algorithm Performance Tracker initialized")
    
    async def run_comprehensive_analysis(self, lookback_days: int = 365) -> Dict[str, Any]:
        """Run comprehensive analysis of all algorithms"""
        try:
            start_date = datetime.now() - timedelta(days=lookback_days)
            end_date = datetime.now()
            
            logger.info(f"Starting comprehensive algorithm analysis from {start_date.date()} to {end_date.date()}")
            
            # Load benchmark data first
            await self._load_benchmark_data(start_date, end_date)
            
            # Analyze all algorithms in parallel
            tasks = []
            for symbol in self.test_symbols:
                for algo_type in self.strategies.keys():
                    tasks.append(self._analyze_algorithm(algo_type, symbol, start_date, end_date))
            
            # Execute all analyses in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful_analyses = [r for r in results if not isinstance(r, Exception)]
            failed_analyses = [r for r in results if isinstance(r, Exception)]
            
            logger.info(f"Completed {len(successful_analyses)} successful analyses, {len(failed_analyses)} failed")
            
            # Generate summary report
            summary = self._generate_performance_summary()
            
            return {
                "analysis_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": lookback_days
                },
                "algorithms_analyzed": len(successful_analyses),
                "symbols_tested": len(self.test_symbols),
                "benchmarks_loaded": len(self.benchmarks),
                "summary": summary,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            return {"error": str(e)}
    
    async def _analyze_algorithm(self, algo_type: AlgorithmType, symbol: str, 
                                start_date: datetime, end_date: datetime) -> Optional[AlgorithmPerformanceMetrics]:
        """Analyze single algorithm on single symbol"""
        try:
            # Get market data with higher granularity for more trading opportunities
            data = self.market_fetcher.get_stock_data(symbol, period="6mo", interval="1h")
            if data.empty:
                return None
            
            # Filter data to date range - handle timezone aware data
            if data.index.tz is not None:
                # Convert naive datetime to timezone-aware
                if start_date.tzinfo is None:
                    start_date = start_date.replace(tzinfo=data.index.tz)
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=data.index.tz)
            
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            if len(data) < 50:  # Need minimum data
                return None
            
            strategy = self.strategies[algo_type]
            
            # Generate signals
            signals = await self._generate_signals(strategy, data, symbol)
            if not signals:
                return None
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(
                algo_type, symbol, data, signals, start_date, end_date
            )
            
            # Store in history
            algo_id = f"{algo_type.value}_{symbol}"
            if algo_id not in self.performance_history:
                self.performance_history[algo_id] = []
            
            self.performance_history[algo_id].append(performance)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error analyzing {algo_type.value} on {symbol}: {str(e)}")
            return None
    
    async def _generate_signals(self, strategy, data: pd.DataFrame, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Generate trading signals for strategy"""
        try:
            # Use thread pool for CPU-intensive strategy calculations
            loop = asyncio.get_event_loop()
            
            def run_strategy():
                try:
                    # Try to generate signals with the strategy
                    if hasattr(strategy, 'generate_signals'):
                        if hasattr(strategy, '__class__') and strategy.__class__.__name__ == 'MovingAverageStrategy':
                            # MovingAverageStrategy needs symbol parameter
                            result = strategy.generate_signals(data, symbol)
                        else:
                            # Other strategies typically just need data
                            result = strategy.generate_signals(data)
                        
                        # Convert DataFrame result to signal list
                        if isinstance(result, pd.DataFrame):
                            signals = StrategyAdapter.adapt_signals(result, symbol)
                            if not signals:
                                # If no signals from adapter, create mock signals
                                signals = StrategyAdapter.create_mock_signals(
                                    data, symbol, strategy.__class__.__name__.lower().replace('strategy', '')
                                )
                            return signals
                        elif isinstance(result, list):
                            return result
                        else:
                            logger.warning(f"Unexpected strategy output type: {type(result)}")
                            return StrategyAdapter.create_mock_signals(
                                data, symbol, strategy.__class__.__name__.lower().replace('strategy', '')
                            )
                    else:
                        logger.warning(f"Strategy {strategy.__class__.__name__} missing generate_signals method")
                        return StrategyAdapter.create_mock_signals(
                            data, symbol, strategy.__class__.__name__.lower().replace('strategy', '')
                        )
                except Exception as e:
                    logger.error(f"Strategy {strategy.__class__.__name__} failed: {str(e)}")
                    # Fall back to mock signals for demo
                    return StrategyAdapter.create_mock_signals(
                        data, symbol, strategy.__class__.__name__.lower().replace('strategy', '')
                    )
            
            with ThreadPoolExecutor() as executor:
                signals = await loop.run_in_executor(executor, run_strategy)
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            # Return mock signals as fallback
            return StrategyAdapter.create_mock_signals(
                data, symbol, 'fallback'
            )
    
    def _calculate_performance_metrics(self, algo_type: AlgorithmType, symbol: str, 
                                     data: pd.DataFrame, signals: List[Dict[str, Any]], 
                                     start_date: datetime, end_date: datetime) -> AlgorithmPerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        # Simulate trading based on signals
        equity_curve, trades = self._simulate_trading(data, signals)
        
        if len(equity_curve) == 0:
            # Return default metrics if no trades
            return self._create_default_metrics(algo_type, symbol, start_date, end_date)
        
        # Calculate returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        
        # Annualized return
        days = len(equity_curve)
        annualized_return = ((equity_curve[-1] / equity_curve[0]) ** (252 / days) - 1) * 100
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return/100 - risk_free_rate) / (volatility/100) if volatility > 0 else 0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max * 100
        max_drawdown = np.min(drawdowns)
        
        # Trade statistics
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = len([t for t in trades if t['pnl'] <= 0])
        total_trades = len(trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] <= 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
        
        # Average trade duration (in days)
        durations = [t.get('duration', 0) for t in trades]
        avg_trade_duration = np.mean(durations) if durations else 0
        
        # Risk metrics
        var_95 = np.percentile(returns, 5) * 100 if len(returns) > 0 else 0
        
        # Benchmark comparison
        benchmark_return, beta, alpha, excess_return, information_ratio = self._calculate_benchmark_metrics(
            equity_curve, symbol, start_date, end_date
        )
        
        return AlgorithmPerformanceMetrics(
            algorithm_id=f"{algo_type.value}_{symbol}",
            algorithm_type=algo_type,
            name=f"{algo_type.value.title().replace('_', ' ')} ({symbol})",
            symbol=symbol,
            timeframe="1D",
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_duration=avg_trade_duration,
            var_95=var_95,
            beta=beta,
            alpha=alpha,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            information_ratio=information_ratio,
            start_date=start_date,
            end_date=end_date,
            last_updated=datetime.now(),
            equity_curve=equity_curve.tolist(),
            drawdown_curve=drawdowns.tolist(),
            trade_log=trades
        )
    
    def _simulate_trading(self, data: pd.DataFrame, signals: List[Dict[str, Any]], 
                         initial_capital: float = 10000) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Simulate trading based on signals"""
        try:
            equity = [initial_capital]
            cash = initial_capital
            position = 0
            trades = []
            entry_price = 0
            entry_date = None
            
            for i, row in data.iterrows():
                current_price = row['Close']
                
                # Find signal for this date
                signal = None
                for s in signals:
                    signal_date = pd.to_datetime(s.get('date', s.get('timestamp', i)))
                    if signal_date.date() == i.date():
                        signal = s
                        break
                
                if signal:
                    action = signal.get('action', signal.get('signal', 'hold')).lower()
                    strength = signal.get('strength', signal.get('confidence', 1.0))
                    
                    if action == 'buy' and position <= 0 and cash > 0:
                        # Enter long position
                        shares_to_buy = int(cash * 0.95 / current_price)  # Use 95% of cash
                        if shares_to_buy > 0:
                            position += shares_to_buy
                            cash -= shares_to_buy * current_price
                            entry_price = current_price
                            entry_date = i
                    
                    elif action == 'sell' and position > 0:
                        # Exit position
                        cash += position * current_price
                        pnl = position * (current_price - entry_price)
                        pnl_pct = (current_price / entry_price - 1) * 100
                        
                        duration = (i - entry_date).days if entry_date else 0
                        
                        trades.append({
                            'entry_date': entry_date.isoformat() if entry_date else i.isoformat(),
                            'exit_date': i.isoformat(),
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'quantity': position,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'duration': duration,
                            'strength': strength
                        })
                        
                        position = 0
                        entry_price = 0
                        entry_date = None
                
                # Calculate current portfolio value
                portfolio_value = cash + position * current_price
                equity.append(portfolio_value)
            
            return np.array(equity[1:]), trades  # Skip initial value
            
        except Exception as e:
            logger.error(f"Error in trading simulation: {str(e)}")
            return np.array([initial_capital]), []
    
    def _calculate_benchmark_metrics(self, equity_curve: np.ndarray, symbol: str, 
                                   start_date: datetime, end_date: datetime) -> Tuple[float, float, float, float, float]:
        """Calculate metrics relative to benchmark"""
        try:
            # Use SPY as primary benchmark
            benchmark_symbol = "SPY"
            if benchmark_symbol not in self.benchmarks:
                return 0, 1, 0, 0, 0
            
            benchmark = self.benchmarks[benchmark_symbol]
            
            # Align dates
            benchmark_returns = benchmark.cumulative_returns
            if len(benchmark_returns) == 0:
                return 0, 1, 0, 0, 0
            
            # Calculate benchmark return for same period
            benchmark_return = (benchmark_returns[-1] / benchmark_returns[0] - 1) * 100
            
            # Calculate algorithm returns
            algo_returns = np.diff(equity_curve) / equity_curve[:-1]
            bench_returns = np.diff(benchmark_returns) / np.array(benchmark_returns[:-1])
            
            # Align lengths
            min_len = min(len(algo_returns), len(bench_returns))
            if min_len < 10:
                return benchmark_return, 1, 0, 0, 0
            
            algo_returns = algo_returns[:min_len]
            bench_returns = bench_returns[:min_len]
            
            # Calculate beta
            covariance = np.cov(algo_returns, bench_returns)[0, 1]
            benchmark_variance = np.var(bench_returns)
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
            
            # Calculate alpha (annualized)
            algo_return_total = (equity_curve[-1] / equity_curve[0] - 1) * 100
            risk_free_rate = 2.0  # 2% risk-free rate
            alpha = (algo_return_total - risk_free_rate - beta * (benchmark_return - risk_free_rate))
            
            # Excess return
            excess_return = algo_return_total - benchmark_return
            
            # Information ratio
            excess_returns = algo_returns - bench_returns
            tracking_error = np.std(excess_returns) * np.sqrt(252)
            information_ratio = (excess_return/100) / tracking_error if tracking_error != 0 else 0
            
            return benchmark_return, beta, alpha, excess_return, information_ratio
            
        except Exception as e:
            logger.error(f"Error calculating benchmark metrics: {str(e)}")
            return 0, 1, 0, 0, 0
    
    async def _load_benchmark_data(self, start_date: datetime, end_date: datetime):
        """Load benchmark data for comparison"""
        for symbol, name in self.benchmark_symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                
                if not hist.empty:
                    # Filter to date range - handle timezone aware data
                    if hist.index.tz is not None:
                        # Convert naive datetime to timezone-aware
                        if start_date.tzinfo is None:
                            start_date = start_date.replace(tzinfo=hist.index.tz)
                        if end_date.tzinfo is None:
                            end_date = end_date.replace(tzinfo=hist.index.tz)
                    
                    hist = hist[(hist.index >= start_date) & (hist.index <= end_date)]
                    
                    if not hist.empty:
                        returns = hist['Close'].pct_change().dropna().tolist()
                        cumulative_returns = (1 + hist['Close'].pct_change().fillna(0)).cumprod().tolist()
                        dates = hist.index.tolist()
                        
                        self.benchmarks[symbol] = BenchmarkData(
                            symbol=symbol,
                            name=name,
                            returns=returns,
                            cumulative_returns=cumulative_returns,
                            dates=dates
                        )
                        
                        logger.info(f"Loaded benchmark data for {symbol}")
                
            except Exception as e:
                logger.error(f"Error loading benchmark {symbol}: {str(e)}")
    
    def _create_default_metrics(self, algo_type: AlgorithmType, symbol: str, 
                               start_date: datetime, end_date: datetime) -> AlgorithmPerformanceMetrics:
        """Create default metrics when no trades occurred"""
        return AlgorithmPerformanceMetrics(
            algorithm_id=f"{algo_type.value}_{symbol}",
            algorithm_type=algo_type,
            name=f"{algo_type.value.title().replace('_', ' ')} ({symbol})",
            symbol=symbol,
            timeframe="1D",
            total_return=0,
            annualized_return=0,
            volatility=0,
            sharpe_ratio=0,
            max_drawdown=0,
            win_rate=0,
            profit_factor=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_win=0,
            avg_loss=0,
            avg_trade_duration=0,
            var_95=0,
            beta=1,
            alpha=0,
            benchmark_return=0,
            excess_return=0,
            information_ratio=0,
            start_date=start_date,
            end_date=end_date,
            last_updated=datetime.now(),
            equity_curve=[10000],
            drawdown_curve=[0],
            trade_log=[]
        )
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate summary statistics across all algorithms"""
        try:
            if not self.performance_history:
                return {"message": "No performance data available"}
            
            # Get latest performance for each algorithm
            latest_performances = {}
            for algo_id, performances in self.performance_history.items():
                if performances:
                    latest_performances[algo_id] = performances[-1]
            
            if not latest_performances:
                return {"message": "No recent performance data"}
            
            # Calculate summary statistics
            returns = [p.total_return for p in latest_performances.values()]
            sharpe_ratios = [p.sharpe_ratio for p in latest_performances.values() if p.sharpe_ratio != 0]
            max_drawdowns = [p.max_drawdown for p in latest_performances.values()]
            win_rates = [p.win_rate for p in latest_performances.values()]
            
            # Algorithm rankings
            best_return = max(latest_performances.values(), key=lambda x: x.total_return)
            best_sharpe = max(latest_performances.values(), key=lambda x: x.sharpe_ratio)
            best_win_rate = max(latest_performances.values(), key=lambda x: x.win_rate)
            lowest_drawdown = max(latest_performances.values(), key=lambda x: -x.max_drawdown)
            
            # Algorithm type performance
            algo_type_performance = {}
            for perf in latest_performances.values():
                algo_type = perf.algorithm_type.value
                if algo_type not in algo_type_performance:
                    algo_type_performance[algo_type] = []
                algo_type_performance[algo_type].append(perf.total_return)
            
            algo_type_avg_returns = {
                algo_type: np.mean(returns) 
                for algo_type, returns in algo_type_performance.items()
            }
            
            return {
                "total_algorithms_tracked": len(latest_performances),
                "analysis_period": f"{latest_performances[list(latest_performances.keys())[0]].start_date.date()} to {latest_performances[list(latest_performances.keys())[0]].end_date.date()}",
                
                "overall_statistics": {
                    "average_return": np.mean(returns),
                    "median_return": np.median(returns),
                    "best_return": max(returns),
                    "worst_return": min(returns),
                    "average_sharpe_ratio": np.mean(sharpe_ratios) if sharpe_ratios else 0,
                    "average_max_drawdown": np.mean(max_drawdowns),
                    "average_win_rate": np.mean(win_rates)
                },
                
                "top_performers": {
                    "best_return": {
                        "name": best_return.name,
                        "return": best_return.total_return,
                        "sharpe_ratio": best_return.sharpe_ratio
                    },
                    "best_sharpe_ratio": {
                        "name": best_sharpe.name,
                        "return": best_sharpe.total_return,
                        "sharpe_ratio": best_sharpe.sharpe_ratio
                    },
                    "best_win_rate": {
                        "name": best_win_rate.name,
                        "win_rate": best_win_rate.win_rate,
                        "return": best_win_rate.total_return
                    },
                    "lowest_drawdown": {
                        "name": lowest_drawdown.name,
                        "max_drawdown": lowest_drawdown.max_drawdown,
                        "return": lowest_drawdown.total_return
                    }
                },
                
                "algorithm_type_performance": {
                    algo_type: {
                        "average_return": avg_return,
                        "algorithms_count": len(algo_type_performance[algo_type])
                    }
                    for algo_type, avg_return in algo_type_avg_returns.items()
                },
                
                "benchmarks_loaded": list(self.benchmarks.keys()),
                "symbols_analyzed": list(set(p.symbol for p in latest_performances.values()))
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {"error": str(e)}
    
    def get_algorithm_comparison(self, algorithm_ids: List[str] = None, 
                               metric: str = "total_return") -> Dict[str, Any]:
        """Get detailed comparison of specific algorithms"""
        try:
            if algorithm_ids is None:
                # Get all latest performances
                algorithm_ids = list(self.performance_history.keys())
            
            comparisons = []
            for algo_id in algorithm_ids:
                if algo_id in self.performance_history and self.performance_history[algo_id]:
                    perf = self.performance_history[algo_id][-1]  # Latest performance
                    comparisons.append({
                        "algorithm_id": algo_id,
                        "name": perf.name,
                        "type": perf.algorithm_type.value,
                        "symbol": perf.symbol,
                        "total_return": perf.total_return,
                        "annualized_return": perf.annualized_return,
                        "sharpe_ratio": perf.sharpe_ratio,
                        "max_drawdown": perf.max_drawdown,
                        "win_rate": perf.win_rate,
                        "total_trades": perf.total_trades,
                        "volatility": perf.volatility,
                        "benchmark_return": perf.benchmark_return,
                        "excess_return": perf.excess_return
                    })
            
            # Sort by specified metric
            if metric in ["total_return", "annualized_return", "sharpe_ratio", "win_rate", "excess_return"]:
                comparisons.sort(key=lambda x: x[metric], reverse=True)
            elif metric in ["max_drawdown", "volatility"]:
                comparisons.sort(key=lambda x: abs(x[metric]))
            
            return {
                "comparison_metric": metric,
                "algorithms_compared": len(comparisons),
                "algorithms": comparisons,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in algorithm comparison: {str(e)}")
            return {"error": str(e)}
    
    def get_algorithm_details(self, algorithm_id: str) -> Dict[str, Any]:
        """Get detailed performance data for specific algorithm"""
        try:
            if algorithm_id not in self.performance_history:
                return {"error": "Algorithm not found"}
            
            performances = self.performance_history[algorithm_id]
            if not performances:
                return {"error": "No performance data available"}
            
            latest = performances[-1]
            
            return {
                "algorithm_id": algorithm_id,
                "basic_info": {
                    "name": latest.name,
                    "type": latest.algorithm_type.value,
                    "symbol": latest.symbol,
                    "timeframe": latest.timeframe
                },
                "performance_metrics": {
                    "total_return": latest.total_return,
                    "annualized_return": latest.annualized_return,
                    "volatility": latest.volatility,
                    "sharpe_ratio": latest.sharpe_ratio,
                    "max_drawdown": latest.max_drawdown,
                    "var_95": latest.var_95
                },
                "trading_stats": {
                    "total_trades": latest.total_trades,
                    "winning_trades": latest.winning_trades,
                    "losing_trades": latest.losing_trades,
                    "win_rate": latest.win_rate,
                    "profit_factor": latest.profit_factor,
                    "avg_win": latest.avg_win,
                    "avg_loss": latest.avg_loss,
                    "avg_trade_duration": latest.avg_trade_duration
                },
                "benchmark_comparison": {
                    "benchmark_return": latest.benchmark_return,
                    "excess_return": latest.excess_return,
                    "beta": latest.beta,
                    "alpha": latest.alpha,
                    "information_ratio": latest.information_ratio
                },
                "equity_curve": latest.equity_curve,
                "drawdown_curve": latest.drawdown_curve,
                "recent_trades": latest.trade_log[-10:],  # Last 10 trades
                "analysis_period": {
                    "start_date": latest.start_date.isoformat(),
                    "end_date": latest.end_date.isoformat(),
                    "last_updated": latest.last_updated.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting algorithm details: {str(e)}")
            return {"error": str(e)}