import secrets
import time
import hashlib
import hmac
import pyotp
from typing import Dict, Optional, Set, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from google.auth.transport import requests
from google.oauth2 import id_token
import logging
import json

logger = logging.getLogger(__name__)


class UserRole(Enum):
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"


@dataclass
class User:
    user_id: str
    email: str
    name: str
    picture: Optional[str]
    role: UserRole
    google_sub: str
    mfa_enabled: bool
    mfa_secret: Optional[str]
    created_at: datetime
    last_login: datetime
    is_active: bool = True
    preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}


@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    mfa_verified: bool = False


class AuthManager:
    """Google OAuth + MFA Authentication Manager with in-memory storage"""
    
    def __init__(self, google_client_id: str = None):
        # In-memory storage (in production, use Redis or database)
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> set of session_ids
        self.failed_attempts: Dict[str, int] = {}  # IP -> failed count
        
        # Configuration
        self.google_client_id = google_client_id or "your-google-client-id.apps.googleusercontent.com"
        self.session_timeout = timedelta(hours=24)
        self.mfa_timeout = timedelta(minutes=30)
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
        
        logger.info("AuthManager initialized with in-memory storage")
    
    def verify_google_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify Google OAuth token and return user info"""
        try:
            # Verify the token with Google
            idinfo = id_token.verify_oauth2_token(
                token, 
                requests.Request(), 
                self.google_client_id
            )
            
            # Verify issuer
            if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
                raise ValueError('Wrong issuer.')
            
            return {
                'sub': idinfo['sub'],
                'email': idinfo['email'],
                'name': idinfo['name'],
                'picture': idinfo.get('picture'),
                'email_verified': idinfo.get('email_verified', False)
            }
            
        except ValueError as e:
            logger.error(f"Google token verification failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Google token verification error: {str(e)}")
            return None
    
    def create_or_update_user(self, google_info: Dict[str, Any]) -> User:
        """Create new user or update existing user from Google info"""
        google_sub = google_info['sub']
        email = google_info['email']
        
        # Check if user exists by Google sub or email
        existing_user = None
        for user in self.users.values():
            if user.google_sub == google_sub or user.email == email:
                existing_user = user
                break
        
        if existing_user:
            # Update existing user
            existing_user.name = google_info['name']
            existing_user.picture = google_info.get('picture')
            existing_user.last_login = datetime.utcnow()
            logger.info(f"Updated existing user: {email}")
            return existing_user
        else:
            # Create new user
            user_id = self._generate_user_id()
            user = User(
                user_id=user_id,
                email=email,
                name=google_info['name'],
                picture=google_info.get('picture'),
                role=UserRole.USER,
                google_sub=google_sub,
                mfa_enabled=False,
                mfa_secret=None,
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow()
            )
            
            self.users[user_id] = user
            self.user_sessions[user_id] = set()
            logger.info(f"Created new user: {email}")
            return user
    
    def create_session(self, user_id: str, ip_address: str, user_agent: str) -> Session:
        """Create a new session for authenticated user"""
        session_id = self._generate_session_id()
        now = datetime.utcnow()
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            expires_at=now + self.session_timeout,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        self.user_sessions[user_id].add(session_id)
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session
    
    def validate_session(self, session_id: str, update_activity: bool = True) -> Optional[Session]:
        """Validate session and optionally update last activity"""
        if not session_id or session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        now = datetime.utcnow()
        
        # Check if session is expired or inactive
        if not session.is_active or now > session.expires_at:
            self.invalidate_session(session_id)
            return None
        
        if update_activity:
            session.last_activity = now
            session.expires_at = now + self.session_timeout
        
        return session
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a specific session"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.is_active = False
        
        # Remove from user sessions
        if session.user_id in self.user_sessions:
            self.user_sessions[session.user_id].discard(session_id)
        
        logger.info(f"Invalidated session {session_id}")
        return True
    
    def invalidate_all_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user"""
        if user_id not in self.user_sessions:
            return 0
        
        session_ids = list(self.user_sessions[user_id])
        count = 0
        
        for session_id in session_ids:
            if self.invalidate_session(session_id):
                count += 1
        
        logger.info(f"Invalidated {count} sessions for user {user_id}")
        return count
    
    def setup_mfa(self, user_id: str) -> Optional[str]:
        """Setup MFA for user and return QR code data"""
        user = self.get_user(user_id)
        if not user:
            return None
        
        # Generate MFA secret
        secret = pyotp.random_base32()
        user.mfa_secret = secret
        
        # Generate QR code URL
        totp = pyotp.TOTP(secret)
        qr_url = totp.provisioning_uri(
            name=user.email,
            issuer_name="Investment Platform"
        )
        
        logger.info(f"MFA setup initiated for user {user_id}")
        return qr_url
    
    def verify_mfa_setup(self, user_id: str, token: str) -> bool:
        """Verify MFA token during setup"""
        user = self.get_user(user_id)
        if not user or not user.mfa_secret:
            return False
        
        totp = pyotp.TOTP(user.mfa_secret)
        if totp.verify(token):
            user.mfa_enabled = True
            logger.info(f"MFA enabled for user {user_id}")
            return True
        
        return False
    
    def verify_mfa_token(self, user_id: str, token: str) -> bool:
        """Verify MFA token for login"""
        user = self.get_user(user_id)
        if not user or not user.mfa_enabled or not user.mfa_secret:
            return False
        
        totp = pyotp.TOTP(user.mfa_secret)
        return totp.verify(token)
    
    def disable_mfa(self, user_id: str) -> bool:
        """Disable MFA for user"""
        user = self.get_user(user_id)
        if not user:
            return False
        
        user.mfa_enabled = False
        user.mfa_secret = None
        logger.info(f"MFA disabled for user {user_id}")
        return True
    
    def check_rate_limit(self, ip_address: str) -> bool:
        """Check if IP is rate limited"""
        failed_count = self.failed_attempts.get(ip_address, 0)
        return failed_count < self.max_failed_attempts
    
    def record_failed_attempt(self, ip_address: str) -> None:
        """Record failed login attempt"""
        self.failed_attempts[ip_address] = self.failed_attempts.get(ip_address, 0) + 1
        logger.warning(f"Failed login attempt from {ip_address}")
    
    def reset_failed_attempts(self, ip_address: str) -> None:
        """Reset failed attempts for IP"""
        if ip_address in self.failed_attempts:
            del self.failed_attempts[ip_address]
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        now = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if not session.is_active or now > session.expires_at:
                expired_sessions.append(session_id)
        
        count = 0
        for session_id in expired_sessions:
            if self.invalidate_session(session_id):
                count += 1
        
        if count > 0:
            logger.info(f"Cleaned up {count} expired sessions")
        
        return count
    
    def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get active sessions for user"""
        if user_id not in self.user_sessions:
            return []
        
        sessions = []
        for session_id in self.user_sessions[user_id]:
            session = self.validate_session(session_id, update_activity=False)
            if session:
                sessions.append(session)
        
        return sessions
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get authentication system statistics"""
        active_sessions = len([s for s in self.sessions.values() if s.is_active])
        active_users = len(set(s.user_id for s in self.sessions.values() if s.is_active))
        mfa_enabled_users = len([u for u in self.users.values() if u.mfa_enabled])
        
        return {
            "total_users": len(self.users),
            "active_users": active_users,
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "mfa_enabled_users": mfa_enabled_users,
            "rate_limited_ips": len(self.failed_attempts)
        }
    
    def _generate_user_id(self) -> str:
        """Generate unique user ID"""
        return f"user_{int(time.time())}_{secrets.token_hex(8)}"
    
    def _generate_session_id(self) -> str:
        """Generate secure session ID"""
        return secrets.token_urlsafe(32)
    
    def export_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Export user data (for GDPR compliance)"""
        user = self.get_user(user_id)
        if not user:
            return None
        
        user_dict = asdict(user)
        user_dict['role'] = user.role.value
        user_dict['created_at'] = user.created_at.isoformat()
        user_dict['last_login'] = user.last_login.isoformat()
        
        # Add session info
        sessions = self.get_user_sessions(user_id)
        user_dict['active_sessions'] = [
            {
                'session_id': s.session_id,
                'created_at': s.created_at.isoformat(),
                'last_activity': s.last_activity.isoformat(),
                'ip_address': s.ip_address,
                'user_agent': s.user_agent
            }
            for s in sessions
        ]
        
        return user_dict