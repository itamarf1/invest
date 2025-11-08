from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging

from .middleware import require_user, require_admin, get_auth_manager
from .auth_manager import User, UserRole
from .user_portfolio_manager import multi_user_portfolio_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/user/portfolio", tags=["user-portfolio"])


class PortfolioSettingsRequest(BaseModel):
    settings: Dict[str, Any]


class TradeRequest(BaseModel):
    symbol: str
    action: str  # buy, sell
    quantity: int
    order_type: str = "market"  # market, limit
    price: Optional[float] = None


@router.get("/summary")
async def get_portfolio_summary(
    user: User = Depends(require_user)
):
    """Get current user's portfolio summary"""
    try:
        summary = multi_user_portfolio_manager.get_user_portfolio_summary(user)
        return summary
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get portfolio summary"
        )


@router.get("/positions")
async def get_user_positions(
    user: User = Depends(require_user)
):
    """Get current user's positions"""
    try:
        portfolio_manager = multi_user_portfolio_manager.get_user_portfolio_manager(user)
        positions = portfolio_manager.get_positions()
        
        return {
            "positions": [
                {
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "average_price": pos.average_price,
                    "current_price": pos.current_price,
                    "market_value": pos.quantity * pos.current_price if pos.current_price else 0,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                    "side": pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
                }
                for pos in positions
            ],
            "total_positions": len(positions)
        }
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get positions"
        )


@router.get("/performance")
async def get_portfolio_performance(
    user: User = Depends(require_user)
):
    """Get current user's portfolio performance metrics"""
    try:
        portfolio_manager = multi_user_portfolio_manager.get_user_portfolio_manager(user)
        performance = portfolio_manager.get_performance_metrics()
        portfolio_value = portfolio_manager.get_portfolio_value()
        
        return {
            "portfolio_value": portfolio_value,
            "performance": performance
        }
    except Exception as e:
        logger.error(f"Error getting performance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get portfolio performance"
        )


@router.get("/orders")
async def get_user_orders(
    user: User = Depends(require_user)
):
    """Get current user's order history"""
    try:
        paper_trader = multi_user_portfolio_manager.get_user_paper_trader(user)
        orders = paper_trader.get_orders()
        
        return {
            "orders": [
                {
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
                    "quantity": order.quantity,
                    "order_type": order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
                    "price": order.price,
                    "filled_price": order.filled_price,
                    "filled_quantity": order.filled_quantity,
                    "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                    "created_at": order.created_at.isoformat() if order.created_at else None,
                    "filled_at": order.filled_at.isoformat() if order.filled_at else None
                }
                for order in orders
            ],
            "total_orders": len(orders)
        }
    except Exception as e:
        logger.error(f"Error getting orders: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get orders"
        )


@router.post("/trade")
async def execute_trade(
    trade_request: TradeRequest,
    user: User = Depends(require_user)
):
    """Execute a trade for the current user"""
    try:
        paper_trader = multi_user_portfolio_manager.get_user_paper_trader(user)
        
        # Check user's trading permissions
        settings = multi_user_portfolio_manager.get_user_settings(user)
        trading_prefs = settings.get("trading_preferences", {})
        
        # Basic validation (you can extend this)
        if trade_request.quantity <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Quantity must be positive"
            )
        
        # Execute the trade (simplified - you may want to add more validation)
        from ..trading.paper_trader import OrderSide, OrderType
        
        side = OrderSide.BUY if trade_request.action.lower() == "buy" else OrderSide.SELL
        order_type = OrderType.MARKET if trade_request.order_type.lower() == "market" else OrderType.LIMIT
        
        order = paper_trader.create_order(
            symbol=trade_request.symbol,
            side=side,
            quantity=trade_request.quantity,
            order_type=order_type,
            price=trade_request.price
        )
        
        return {
            "success": True,
            "order": {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": side.value,
                "quantity": order.quantity,
                "order_type": order_type.value,
                "price": order.price,
                "status": order.status.value,
                "created_at": order.created_at.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute trade"
        )


@router.get("/settings")
async def get_portfolio_settings(
    user: User = Depends(require_user)
):
    """Get current user's portfolio settings"""
    try:
        settings = multi_user_portfolio_manager.get_user_settings(user)
        return {
            "user_id": user.user_id,
            "settings": settings
        }
    except Exception as e:
        logger.error(f"Error getting settings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get portfolio settings"
        )


@router.put("/settings")
async def update_portfolio_settings(
    settings_request: PortfolioSettingsRequest,
    user: User = Depends(require_user)
):
    """Update current user's portfolio settings"""
    try:
        success = multi_user_portfolio_manager.update_user_settings(
            user, 
            settings_request.settings
        )
        
        if success:
            return {
                "success": True,
                "message": "Portfolio settings updated successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update settings"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update portfolio settings"
        )


# Admin endpoints
@router.get("/admin/stats")
async def get_portfolio_system_stats(
    admin_user: User = Depends(require_user)
):
    """Get portfolio system statistics (admin only)"""
    if admin_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        stats = multi_user_portfolio_manager.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system statistics"
        )


@router.post("/admin/cleanup")
async def cleanup_inactive_portfolios(
    days_inactive: int = 90,
    admin_user: User = Depends(require_user)
):
    """Clean up inactive portfolios (admin only)"""
    if admin_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        count = multi_user_portfolio_manager.cleanup_inactive_portfolios(days_inactive)
        return {
            "success": True,
            "message": f"Cleaned up {count} inactive portfolios"
        }
    except Exception as e:
        logger.error(f"Error cleaning up portfolios: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clean up portfolios"
        )