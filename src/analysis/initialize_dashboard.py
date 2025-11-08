import asyncio
import logging
from datetime import datetime, timedelta
from .algorithm_performance_tracker import AlgorithmPerformanceTracker

logger = logging.getLogger(__name__)


async def initialize_algorithm_dashboard():
    """Initialize the algorithm dashboard with performance data"""
    try:
        logger.info("Initializing Algorithm Performance Dashboard...")
        
        # Create tracker instance
        tracker = AlgorithmPerformanceTracker()
        
        # Run comprehensive analysis with shorter lookback for faster initialization
        result = await tracker.run_comprehensive_analysis(lookback_days=180)
        
        if 'error' not in result:
            logger.info(f"Dashboard initialized successfully: {result['algorithms_analyzed']} algorithms analyzed")
            return tracker, result
        else:
            logger.error(f"Dashboard initialization failed: {result['error']}")
            return None, result
            
    except Exception as e:
        logger.error(f"Error initializing dashboard: {str(e)}")
        return None, {"error": str(e)}


def initialize_dashboard_sync():
    """Synchronous wrapper for dashboard initialization"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(initialize_algorithm_dashboard())
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Error in sync initialization: {str(e)}")
        return None, {"error": str(e)}