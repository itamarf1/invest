from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List
import logging
from .auth_manager import AuthManager, User, Session, UserRole

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)

# Global auth manager instance (will be initialized in main.py)
auth_manager: Optional[AuthManager] = None


def init_auth_manager(google_client_id: str = None) -> AuthManager:
    """Initialize the global auth manager"""
    global auth_manager
    auth_manager = AuthManager(google_client_id=google_client_id)
    return auth_manager


def get_auth_manager() -> AuthManager:
    """Get the auth manager instance"""
    if auth_manager is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Auth manager not initialized"
        )
    return auth_manager


async def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    # Check for forwarded IP headers (for reverse proxies)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    return request.client.host if request.client else "unknown"


async def get_user_agent(request: Request) -> str:
    """Extract user agent from request"""
    return request.headers.get("User-Agent", "unknown")


async def get_current_session(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Session]:
    """Get current session from Bearer token"""
    if not credentials:
        return None
    
    auth_mgr = get_auth_manager()
    session = auth_mgr.validate_session(credentials.credentials)
    
    return session


async def get_current_user(
    session: Optional[Session] = Depends(get_current_session)
) -> Optional[User]:
    """Get current user from session"""
    if not session:
        return None
    
    auth_mgr = get_auth_manager()
    return auth_mgr.get_user(session.user_id)


async def require_authentication(
    session: Optional[Session] = Depends(get_current_session)
) -> Session:
    """Require valid authentication"""
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return session


async def require_user(
    user: Optional[User] = Depends(get_current_user)
) -> User:
    """Require valid user"""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    return user


async def require_mfa_verified(
    session: Session = Depends(require_authentication)
) -> Session:
    """Require MFA verification for sensitive operations"""
    if not session.mfa_verified:
        auth_mgr = get_auth_manager()
        user = auth_mgr.get_user(session.user_id)
        
        if user and user.mfa_enabled:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="MFA verification required for this operation"
            )
    
    return session


def require_role(required_roles: List[UserRole]):
    """Decorator factory for role-based access control"""
    async def role_checker(user: User = Depends(require_user)) -> User:
        if user.role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {[r.value for r in required_roles]}"
            )
        return user
    
    return role_checker


# Convenience dependencies for different roles
require_admin = Depends(require_role([UserRole.ADMIN]))
require_premium = Depends(require_role([UserRole.PREMIUM, UserRole.ADMIN]))


async def check_rate_limit(
    request: Request,
    auth_mgr: AuthManager = Depends(get_auth_manager)
) -> bool:
    """Check rate limiting"""
    client_ip = await get_client_ip(request)
    
    if not auth_mgr.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed authentication attempts. Please try again later."
        )
    
    return True


class AuthenticationError(HTTPException):
    """Custom authentication error"""
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"}
        )


class AuthorizationError(HTTPException):
    """Custom authorization error"""
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )