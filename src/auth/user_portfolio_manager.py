from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from dataclasses import dataclass

from ..trading.portfolio_manager import PortfolioManager
from ..trading.paper_trader import PaperTrader
from .auth_manager import User

logger = logging.getLogger(__name__)


@dataclass
class UserPortfolio:
    user_id: str
    portfolio_manager: PortfolioManager
    paper_trader: PaperTrader
    created_at: datetime
    last_updated: datetime
    settings: Dict[str, Any]


class MultiUserPortfolioManager:
    """Manages portfolios for multiple users with isolation"""
    
    def __init__(self):
        self.user_portfolios: Dict[str, UserPortfolio] = {}
        logger.info("Multi-user portfolio manager initialized")
    
    def get_or_create_user_portfolio(self, user: User) -> UserPortfolio:
        """Get existing portfolio or create new one for user"""
        if user.user_id not in self.user_portfolios:
            # Create new portfolio for user
            paper_trader = PaperTrader()
            portfolio_manager = PortfolioManager(paper_trader)
            
            # Initialize with default settings based on user role
            default_settings = self._get_default_settings(user)
            
            user_portfolio = UserPortfolio(
                user_id=user.user_id,
                portfolio_manager=portfolio_manager,
                paper_trader=paper_trader,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                settings=default_settings
            )
            
            self.user_portfolios[user.user_id] = user_portfolio
            logger.info(f"Created new portfolio for user {user.user_id}")
        
        return self.user_portfolios[user.user_id]
    
    def get_user_portfolio_manager(self, user: User) -> PortfolioManager:
        """Get portfolio manager for specific user"""
        user_portfolio = self.get_or_create_user_portfolio(user)
        user_portfolio.last_updated = datetime.utcnow()
        return user_portfolio.portfolio_manager
    
    def get_user_paper_trader(self, user: User) -> PaperTrader:
        """Get paper trader for specific user"""
        user_portfolio = self.get_or_create_user_portfolio(user)
        return user_portfolio.paper_trader
    
    def update_user_settings(self, user: User, settings: Dict[str, Any]) -> bool:
        """Update user portfolio settings"""
        try:
            user_portfolio = self.get_or_create_user_portfolio(user)
            user_portfolio.settings.update(settings)
            user_portfolio.last_updated = datetime.utcnow()
            logger.info(f"Updated settings for user {user.user_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating user settings: {str(e)}")
            return False
    
    def get_user_settings(self, user: User) -> Dict[str, Any]:
        """Get user portfolio settings"""
        user_portfolio = self.get_or_create_user_portfolio(user)
        return user_portfolio.settings.copy()
    
    def get_user_portfolio_summary(self, user: User) -> Dict[str, Any]:
        """Get summary of user's portfolio"""
        try:
            user_portfolio = self.get_or_create_user_portfolio(user)
            portfolio_manager = user_portfolio.portfolio_manager
            
            # Get portfolio data
            positions = portfolio_manager.get_positions()
            portfolio_value = portfolio_manager.get_portfolio_value()
            performance = portfolio_manager.get_performance_metrics()
            
            return {
                "user_id": user.user_id,
                "created_at": user_portfolio.created_at.isoformat(),
                "last_updated": user_portfolio.last_updated.isoformat(),
                "total_positions": len(positions),
                "portfolio_value": portfolio_value,
                "performance": performance,
                "settings": user_portfolio.settings,
                "positions_summary": [
                    {
                        "symbol": pos.symbol,
                        "quantity": pos.quantity,
                        "current_value": pos.quantity * pos.current_price if pos.current_price else 0,
                        "unrealized_pnl": pos.unrealized_pnl
                    }
                    for pos in positions[:10]  # Limit to top 10 positions
                ]
            }
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {str(e)}")
            return {"error": str(e)}
    
    def delete_user_portfolio(self, user_id: str) -> bool:
        """Delete user portfolio (for GDPR compliance)"""
        try:
            if user_id in self.user_portfolios:
                del self.user_portfolios[user_id]
                logger.info(f"Deleted portfolio for user {user_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting user portfolio: {str(e)}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide portfolio statistics"""
        try:
            total_users = len(self.user_portfolios)
            active_users = len([
                p for p in self.user_portfolios.values() 
                if (datetime.utcnow() - p.last_updated).days < 7
            ])
            
            total_positions = sum(
                len(p.portfolio_manager.get_positions()) 
                for p in self.user_portfolios.values()
            )
            
            total_portfolio_value = sum(
                p.portfolio_manager.get_portfolio_value() 
                for p in self.user_portfolios.values()
            )
            
            return {
                "total_users": total_users,
                "active_users_7d": active_users,
                "total_positions": total_positions,
                "total_portfolio_value": total_portfolio_value,
                "average_positions_per_user": total_positions / total_users if total_users > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {"error": str(e)}
    
    def _get_default_settings(self, user: User) -> Dict[str, Any]:
        """Get default settings based on user role"""
        base_settings = {
            "initial_capital": 10000.0,
            "max_positions": 10,
            "max_position_size": 0.1,  # 10% of portfolio
            "risk_tolerance": "moderate",
            "auto_rebalancing": False,
            "notifications": {
                "email": True,
                "push": False,
                "sms": False
            },
            "trading_preferences": {
                "allow_options": False,
                "allow_crypto": False,
                "allow_forex": False,
                "allow_margin": False
            }
        }
        
        # Adjust based on user role
        if user.role.value == "premium":
            base_settings.update({
                "initial_capital": 50000.0,
                "max_positions": 25,
                "trading_preferences": {
                    "allow_options": True,
                    "allow_crypto": True,
                    "allow_forex": True,
                    "allow_margin": False
                }
            })
        elif user.role.value == "admin":
            base_settings.update({
                "initial_capital": 100000.0,
                "max_positions": 50,
                "trading_preferences": {
                    "allow_options": True,
                    "allow_crypto": True,
                    "allow_forex": True,
                    "allow_margin": True
                }
            })
        
        return base_settings
    
    def cleanup_inactive_portfolios(self, days_inactive: int = 90) -> int:
        """Clean up portfolios inactive for specified days"""
        try:
            cutoff_date = datetime.utcnow().replace(day=datetime.utcnow().day - days_inactive)
            inactive_users = [
                user_id for user_id, portfolio in self.user_portfolios.items()
                if portfolio.last_updated < cutoff_date
            ]
            
            count = 0
            for user_id in inactive_users:
                if self.delete_user_portfolio(user_id):
                    count += 1
            
            logger.info(f"Cleaned up {count} inactive portfolios")
            return count
        except Exception as e:
            logger.error(f"Error cleaning up portfolios: {str(e)}")
            return 0


# Global instance
multi_user_portfolio_manager = MultiUserPortfolioManager()