from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import qrcode
import io
import base64
import logging
from datetime import datetime

from .auth_manager import AuthManager, User, Session, UserRole
from .middleware import (
    get_auth_manager, get_client_ip, get_user_agent, 
    require_user, require_authentication, check_rate_limit,
    get_current_user, get_current_session
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])


# Request Models
class GoogleLoginRequest(BaseModel):
    id_token: str


class MFATokenRequest(BaseModel):
    token: str


class MFASetupVerifyRequest(BaseModel):
    token: str


class SessionListResponse(BaseModel):
    sessions: List[Dict[str, Any]]


# Authentication Endpoints

@router.post("/login/google")
async def google_login(
    request: Request,
    login_request: GoogleLoginRequest,
    auth_mgr: AuthManager = Depends(get_auth_manager),
    rate_limit_check: bool = Depends(check_rate_limit)
):
    """Login with Google OAuth"""
    try:
        client_ip = await get_client_ip(request)
        user_agent = await get_user_agent(request)
        
        # Verify Google token
        google_info = auth_mgr.verify_google_token(login_request.id_token)
        if not google_info:
            auth_mgr.record_failed_attempt(client_ip)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Google token"
            )
        
        if not google_info.get('email_verified', False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email not verified with Google"
            )
        
        # Create or update user
        user = auth_mgr.create_or_update_user(google_info)
        
        # Create session
        session = auth_mgr.create_session(user.user_id, client_ip, user_agent)
        
        # Reset failed attempts on successful login
        auth_mgr.reset_failed_attempts(client_ip)
        
        # Check if MFA is required
        requires_mfa = user.mfa_enabled and not session.mfa_verified
        
        return {
            "success": True,
            "session_token": session.session_id,
            "user": {
                "user_id": user.user_id,
                "email": user.email,
                "name": user.name,
                "picture": user.picture,
                "role": user.role.value,
                "mfa_enabled": user.mfa_enabled
            },
            "requires_mfa": requires_mfa,
            "expires_at": session.expires_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Google login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/verify-mfa")
async def verify_mfa(
    mfa_request: MFATokenRequest,
    session: Session = Depends(require_authentication),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """Verify MFA token for current session"""
    try:
        user = auth_mgr.get_user(session.user_id)
        if not user or not user.mfa_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA not enabled for this user"
            )
        
        # Verify MFA token
        if auth_mgr.verify_mfa_token(user.user_id, mfa_request.token):
            session.mfa_verified = True
            logger.info(f"MFA verified for user {user.user_id}")
            
            return {
                "success": True,
                "message": "MFA verification successful"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid MFA token"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA verification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA verification failed"
        )


@router.post("/setup-mfa")
async def setup_mfa(
    user: User = Depends(require_user),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """Setup MFA for current user"""
    try:
        if user.mfa_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA is already enabled for this user"
            )
        
        # Generate QR code URL
        qr_url = auth_mgr.setup_mfa(user.user_id)
        if not qr_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to setup MFA"
            )
        
        # Generate QR code image
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Convert to base64
        qr_code_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return {
            "success": True,
            "qr_code_url": qr_url,
            "qr_code_image": f"data:image/png;base64,{qr_code_base64}",
            "secret": user.mfa_secret,  # For manual entry
            "message": "Scan the QR code with your authenticator app and verify with a token"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA setup error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA setup failed"
        )


@router.post("/setup-mfa/verify")
async def verify_mfa_setup(
    verify_request: MFASetupVerifyRequest,
    user: User = Depends(require_user),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """Verify and complete MFA setup"""
    try:
        if user.mfa_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA is already enabled"
            )
        
        # Verify the token
        if auth_mgr.verify_mfa_setup(user.user_id, verify_request.token):
            return {
                "success": True,
                "message": "MFA has been successfully enabled for your account"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token. Please try again."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA setup verification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA setup verification failed"
        )


@router.delete("/disable-mfa")
async def disable_mfa(
    user: User = Depends(require_user),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """Disable MFA for current user"""
    try:
        if not user.mfa_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA is not enabled for this user"
            )
        
        if auth_mgr.disable_mfa(user.user_id):
            # Invalidate all sessions to force re-authentication
            auth_mgr.invalidate_all_user_sessions(user.user_id)
            
            return {
                "success": True,
                "message": "MFA has been disabled. Please log in again."
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to disable MFA"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA disable error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to disable MFA"
        )


@router.get("/profile")
async def get_user_profile(
    user: User = Depends(require_user)
):
    """Get current user profile"""
    return {
        "user_id": user.user_id,
        "email": user.email,
        "name": user.name,
        "picture": user.picture,
        "role": user.role.value,
        "mfa_enabled": user.mfa_enabled,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat(),
        "is_active": user.is_active,
        "preferences": user.preferences
    }


@router.get("/sessions")
async def get_user_sessions(
    user: User = Depends(require_user),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """Get all active sessions for current user"""
    sessions = auth_mgr.get_user_sessions(user.user_id)
    
    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "created_at": s.created_at.isoformat(),
                "last_activity": s.last_activity.isoformat(),
                "expires_at": s.expires_at.isoformat(),
                "ip_address": s.ip_address,
                "user_agent": s.user_agent,
                "is_current": s.session_id == (await get_current_session()).session_id if await get_current_session() else False
            }
            for s in sessions
        ]
    }


@router.delete("/sessions/{session_id}")
async def revoke_session(
    session_id: str,
    user: User = Depends(require_user),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """Revoke a specific session"""
    # Verify the session belongs to the current user
    user_sessions = auth_mgr.get_user_sessions(user.user_id)
    session_ids = [s.session_id for s in user_sessions]
    
    if session_id not in session_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    if auth_mgr.invalidate_session(session_id):
        return {
            "success": True,
            "message": "Session revoked successfully"
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke session"
        )


@router.delete("/sessions")
async def revoke_all_sessions(
    user: User = Depends(require_user),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """Revoke all sessions for current user"""
    count = auth_mgr.invalidate_all_user_sessions(user.user_id)
    
    return {
        "success": True,
        "message": f"Revoked {count} sessions. Please log in again."
    }


@router.post("/logout")
async def logout(
    session: Session = Depends(require_authentication),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """Logout current session"""
    if auth_mgr.invalidate_session(session.session_id):
        return {
            "success": True,
            "message": "Logged out successfully"
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/status")
async def auth_status(
    current_user: Optional[User] = Depends(get_current_user),
    current_session: Optional[Session] = Depends(get_current_session)
):
    """Check authentication status"""
    if not current_user or not current_session:
        return {
            "authenticated": False,
            "user": None,
            "session": None
        }
    
    return {
        "authenticated": True,
        "user": {
            "user_id": current_user.user_id,
            "email": current_user.email,
            "name": current_user.name,
            "role": current_user.role.value,
            "mfa_enabled": current_user.mfa_enabled
        },
        "session": {
            "session_id": current_session.session_id,
            "expires_at": current_session.expires_at.isoformat(),
            "mfa_verified": current_session.mfa_verified
        }
    }


# Admin endpoints
@router.get("/admin/stats")
async def get_auth_stats(
    admin_user: User = Depends(require_user),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """Get authentication system statistics (admin only)"""
    if admin_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    stats = auth_mgr.get_system_stats()
    return stats


@router.post("/admin/cleanup")
async def cleanup_expired_sessions(
    admin_user: User = Depends(require_user),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """Clean up expired sessions (admin only)"""
    if admin_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    count = auth_mgr.cleanup_expired_sessions()
    return {
        "success": True,
        "message": f"Cleaned up {count} expired sessions"
    }