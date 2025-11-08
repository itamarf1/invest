from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from .algorithm_performance_tracker import AlgorithmPerformanceTracker
from ..auth import require_user, get_current_user
from ..auth.auth_manager import User, UserRole

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/algorithm-dashboard", tags=["algorithm-dashboard"])

# Global performance tracker instance
performance_tracker = AlgorithmPerformanceTracker()


class AnalysisRequest(BaseModel):
    lookback_days: int = 365
    symbols: Optional[List[str]] = None


class ComparisonRequest(BaseModel):
    algorithm_ids: Optional[List[str]] = None
    metric: str = "total_return"


@router.get("/overview")
async def get_dashboard_overview():
    """Get algorithm performance dashboard overview"""
    try:
        # Run quick analysis if no data exists
        if not performance_tracker.performance_history:
            logger.info("No performance data found, running initial analysis...")
            result = await performance_tracker.run_comprehensive_analysis(lookback_days=180)
            logger.info(f"Initial analysis completed: {result}")
        
        summary = performance_tracker._generate_performance_summary()
        
        return {
            "dashboard_overview": summary,
            "data_freshness": "live" if performance_tracker.performance_history else "initializing",
            "algorithms_count": len(performance_tracker.performance_history),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard overview: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load dashboard overview: {str(e)}"
        )


@router.post("/run-analysis")
async def run_comprehensive_analysis(
    background_tasks: BackgroundTasks,
    request: AnalysisRequest,
    user: User = Depends(require_user)
):
    """Run comprehensive algorithm analysis"""
    try:
        # Run analysis in background for better performance
        background_tasks.add_task(
            performance_tracker.run_comprehensive_analysis,
            request.lookback_days
        )
        
        return {
            "message": "Comprehensive analysis started",
            "analysis_parameters": {
                "lookback_days": request.lookback_days,
                "symbols": request.symbols or performance_tracker.test_symbols
            },
            "estimated_completion": "2-5 minutes",
            "status": "running"
        }
        
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start analysis"
        )


@router.post("/refresh-data")
async def refresh_market_data(
    background_tasks: BackgroundTasks,
    user: User = Depends(require_user)
):
    """Force refresh of market data and recalculate algorithms"""
    try:
        # Clear cache to force fresh data fetch
        performance_tracker.market_fetcher.cache.clear()
        performance_tracker.market_fetcher.cache_timestamps.clear()
        
        # Run fresh analysis with current data
        background_tasks.add_task(
            performance_tracker.run_comprehensive_analysis,
            90  # Use shorter lookback for faster refresh
        )
        
        return {
            "message": "Market data refresh initiated",
            "cache_cleared": True,
            "fresh_analysis_started": True,
            "estimated_completion": "1-2 minutes",
            "status": "refreshing"
        }
        
    except Exception as e:
        logger.error(f"Error refreshing data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh market data"
        )


@router.get("/algorithms/comparison")
async def get_algorithm_comparison(
    metric: str = "total_return",
    limit: int = 20
):
    """Get algorithm performance comparison"""
    try:
        valid_metrics = [
            "total_return", "annualized_return", "sharpe_ratio", 
            "max_drawdown", "win_rate", "volatility", "excess_return"
        ]
        
        if metric not in valid_metrics:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metric. Must be one of: {', '.join(valid_metrics)}"
            )
        
        comparison = performance_tracker.get_algorithm_comparison(metric=metric)
        
        # Limit results
        if "algorithms" in comparison:
            comparison["algorithms"] = comparison["algorithms"][:limit]
        
        return comparison
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting algorithm comparison: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get algorithm comparison"
        )


@router.get("/algorithms/{algorithm_id}")
async def get_algorithm_details(
    algorithm_id: str,
    user: User = Depends(require_user)
):
    """Get detailed performance data for specific algorithm"""
    try:
        details = performance_tracker.get_algorithm_details(algorithm_id)
        
        if "error" in details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=details["error"]
            )
        
        return details
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting algorithm details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get algorithm details"
        )


@router.get("/algorithms")
async def list_algorithms(
    algorithm_type: Optional[str] = None,
    symbol: Optional[str] = None,
    user: User = Depends(require_user)
):
    """List all tracked algorithms with optional filtering"""
    try:
        algorithms = []
        
        for algo_id, performances in performance_tracker.performance_history.items():
            if not performances:
                continue
            
            latest = performances[-1]
            
            # Apply filters
            if algorithm_type and latest.algorithm_type.value != algorithm_type:
                continue
            if symbol and latest.symbol.upper() != symbol.upper():
                continue
            
            algorithms.append({
                "algorithm_id": algo_id,
                "name": latest.name,
                "type": latest.algorithm_type.value,
                "symbol": latest.symbol,
                "total_return": latest.total_return,
                "sharpe_ratio": latest.sharpe_ratio,
                "max_drawdown": latest.max_drawdown,
                "win_rate": latest.win_rate,
                "total_trades": latest.total_trades,
                "last_updated": latest.last_updated.isoformat()
            })
        
        # Sort by total return
        algorithms.sort(key=lambda x: x["total_return"], reverse=True)
        
        return {
            "algorithms": algorithms,
            "total_count": len(algorithms),
            "filters_applied": {
                "algorithm_type": algorithm_type,
                "symbol": symbol
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing algorithms: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list algorithms"
        )


@router.get("/benchmarks")
async def get_benchmark_data(
    user: User = Depends(require_user)
):
    """Get benchmark performance data"""
    try:
        benchmarks = {}
        
        for symbol, benchmark in performance_tracker.benchmarks.items():
            if benchmark.cumulative_returns:
                total_return = (benchmark.cumulative_returns[-1] / benchmark.cumulative_returns[0] - 1) * 100
                
                benchmarks[symbol] = {
                    "name": benchmark.name,
                    "symbol": symbol,
                    "total_return": total_return,
                    "data_points": len(benchmark.cumulative_returns),
                    "date_range": {
                        "start": benchmark.dates[0].isoformat() if benchmark.dates else None,
                        "end": benchmark.dates[-1].isoformat() if benchmark.dates else None
                    }
                }
        
        return {
            "benchmarks": benchmarks,
            "primary_benchmark": "SPY",
            "benchmarks_count": len(benchmarks)
        }
        
    except Exception as e:
        logger.error(f"Error getting benchmark data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get benchmark data"
        )


@router.get("/performance-matrix")
async def get_performance_matrix(
    user: User = Depends(require_user)
):
    """Get performance matrix for heat map visualization"""
    try:
        matrix_data = {}
        algorithm_types = set()
        symbols = set()
        
        for algo_id, performances in performance_tracker.performance_history.items():
            if not performances:
                continue
            
            latest = performances[-1]
            algo_type = latest.algorithm_type.value
            symbol = latest.symbol
            
            algorithm_types.add(algo_type)
            symbols.add(symbol)
            
            if algo_type not in matrix_data:
                matrix_data[algo_type] = {}
            
            matrix_data[algo_type][symbol] = {
                "total_return": latest.total_return,
                "sharpe_ratio": latest.sharpe_ratio,
                "max_drawdown": latest.max_drawdown,
                "win_rate": latest.win_rate,
                "volatility": latest.volatility,
                "total_trades": latest.total_trades
            }
        
        return {
            "matrix_data": matrix_data,
            "algorithm_types": sorted(list(algorithm_types)),
            "symbols": sorted(list(symbols)),
            "metrics_available": [
                "total_return", "sharpe_ratio", "max_drawdown", 
                "win_rate", "volatility", "total_trades"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting performance matrix: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get performance matrix"
        )


@router.get("/top-performers")
async def get_top_performers(
    metric: str = "total_return",
    count: int = 10,
    user: User = Depends(require_user)
):
    """Get top performing algorithms by specific metric"""
    try:
        valid_metrics = [
            "total_return", "sharpe_ratio", "win_rate", "profit_factor"
        ]
        
        if metric not in valid_metrics:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metric. Must be one of: {', '.join(valid_metrics)}"
            )
        
        performers = []
        
        for algo_id, performances in performance_tracker.performance_history.items():
            if not performances:
                continue
            
            latest = performances[-1]
            metric_value = getattr(latest, metric, 0)
            
            performers.append({
                "algorithm_id": algo_id,
                "name": latest.name,
                "type": latest.algorithm_type.value,
                "symbol": latest.symbol,
                "metric_value": metric_value,
                "total_return": latest.total_return,
                "sharpe_ratio": latest.sharpe_ratio,
                "max_drawdown": latest.max_drawdown,
                "total_trades": latest.total_trades
            })
        
        # Sort by metric
        performers.sort(key=lambda x: x["metric_value"], reverse=True)
        
        return {
            "metric": metric,
            "top_performers": performers[:count],
            "total_algorithms": len(performers)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting top performers: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get top performers"
        )


@router.get("/risk-analysis")
async def get_risk_analysis(
    user: User = Depends(require_user)
):
    """Get risk analysis across all algorithms"""
    try:
        risk_data = []
        
        for algo_id, performances in performance_tracker.performance_history.items():
            if not performances:
                continue
            
            latest = performances[-1]
            
            risk_data.append({
                "algorithm_id": algo_id,
                "name": latest.name,
                "type": latest.algorithm_type.value,
                "symbol": latest.symbol,
                "volatility": latest.volatility,
                "max_drawdown": latest.max_drawdown,
                "var_95": latest.var_95,
                "beta": latest.beta,
                "sharpe_ratio": latest.sharpe_ratio,
                "total_return": latest.total_return
            })
        
        # Calculate risk metrics
        if risk_data:
            volatilities = [r["volatility"] for r in risk_data]
            drawdowns = [r["max_drawdown"] for r in risk_data]
            vars = [r["var_95"] for r in risk_data]
            
            risk_summary = {
                "average_volatility": sum(volatilities) / len(volatilities),
                "average_max_drawdown": sum(drawdowns) / len(drawdowns),
                "average_var_95": sum(vars) / len(vars),
                "highest_risk": max(risk_data, key=lambda x: x["volatility"]),
                "lowest_risk": min(risk_data, key=lambda x: x["volatility"]),
                "worst_drawdown": min(risk_data, key=lambda x: x["max_drawdown"])
            }
        else:
            risk_summary = {}
        
        return {
            "risk_analysis": risk_data,
            "risk_summary": risk_summary,
            "algorithms_analyzed": len(risk_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting risk analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get risk analysis"
        )


# Admin endpoints
@router.get("/admin/system-status")
async def get_system_status(
    user: User = Depends(require_user)
):
    """Get algorithm tracking system status (admin only)"""
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        total_algorithms = len(performance_tracker.performance_history)
        total_data_points = sum(
            len(performances) for performances in performance_tracker.performance_history.values()
        )
        
        # Get latest analysis times
        latest_updates = []
        for algo_id, performances in performance_tracker.performance_history.items():
            if performances:
                latest_updates.append(performances[-1].last_updated)
        
        oldest_update = min(latest_updates) if latest_updates else None
        newest_update = max(latest_updates) if latest_updates else None
        
        return {
            "system_status": {
                "total_algorithms_tracked": total_algorithms,
                "total_performance_records": total_data_points,
                "benchmarks_loaded": len(performance_tracker.benchmarks),
                "test_symbols": performance_tracker.test_symbols,
                "strategies_available": list(performance_tracker.strategies.keys()),
                "oldest_data": oldest_update.isoformat() if oldest_update else None,
                "newest_data": newest_update.isoformat() if newest_update else None
            },
            "system_health": "operational" if total_algorithms > 0 else "initializing"
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system status"
        )