from .auth_manager import AuthManager, User, Session, UserRole
from .middleware import (
    init_auth_manager, get_auth_manager, 
    require_user, require_authentication, require_mfa_verified,
    require_role, require_admin, require_premium,
    get_current_user, get_current_session
)
from .auth_routes import router as auth_router
from .user_portfolio_routes import router as user_portfolio_router

__all__ = [
    'AuthManager',
    'User', 
    'Session',
    'UserRole',
    'init_auth_manager',
    'get_auth_manager',
    'require_user',
    'require_authentication', 
    'require_mfa_verified',
    'require_role',
    'require_admin',
    'require_premium',
    'get_current_user',
    'get_current_session',
    'auth_router',
    'user_portfolio_router'
]