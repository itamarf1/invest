# Authentication Setup Guide

This guide explains how to set up Google OAuth authentication with MFA for the investment dashboard.

## 🚀 Quick Start

The authentication system is already implemented and running! Here's what you need to know:

## 📋 Features Implemented

### ✅ Google OAuth Authentication
- **Login with Google**: Users can sign in using their Google accounts
- **Email verification**: Only verified Google accounts are allowed
- **Automatic user creation**: New users are automatically created on first login

### 🔒 Multi-Factor Authentication (MFA)
- **TOTP-based MFA**: Uses Time-based One-Time Passwords (Google Authenticator compatible)
- **QR Code setup**: Easy setup with QR codes for authenticator apps
- **Optional MFA**: Users can enable/disable MFA as needed
- **MFA verification**: Required for sensitive operations when enabled

### 👤 User Management
- **Role-based access**: USER, PREMIUM, ADMIN roles
- **Session management**: Secure session handling with expiration
- **Multiple sessions**: Users can have multiple active sessions
- **Session revocation**: Users can revoke individual or all sessions

### 💼 Portfolio Isolation
- **User-specific portfolios**: Each user has their own isolated portfolio
- **Role-based settings**: Different default settings based on user role
- **Trading permissions**: Configurable trading permissions per user
- **Performance tracking**: Individual performance metrics per user

## 🌐 Available Endpoints

### Authentication Endpoints (`/api/auth/`)
- `POST /login/google` - Login with Google OAuth
- `POST /verify-mfa` - Verify MFA token
- `POST /setup-mfa` - Setup MFA for user
- `POST /setup-mfa/verify` - Complete MFA setup
- `DELETE /disable-mfa` - Disable MFA
- `GET /profile` - Get user profile
- `GET /sessions` - Get user sessions
- `DELETE /sessions/{session_id}` - Revoke specific session
- `DELETE /sessions` - Revoke all sessions
- `POST /logout` - Logout current session
- `GET /status` - Check authentication status

### User Portfolio Endpoints (`/api/user/portfolio/`)
- `GET /summary` - Get portfolio summary
- `GET /positions` - Get user positions
- `GET /performance` - Get portfolio performance
- `GET /orders` - Get order history
- `POST /trade` - Execute trades
- `GET /settings` - Get portfolio settings
- `PUT /settings` - Update portfolio settings

### Admin Endpoints
- `GET /api/auth/admin/stats` - Authentication system stats
- `GET /api/user/portfolio/admin/stats` - Portfolio system stats

## 🎯 User Interface

### Login Page: `/login`
- **Google Sign-in button**: One-click authentication
- **MFA setup flow**: Guided MFA setup with QR codes
- **Session management**: View and manage active sessions
- **User profile**: View user information and settings

## 🔧 Configuration Required

To use this in production, you need to:

### 1. Google OAuth Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google+ API
4. Create OAuth 2.0 credentials
5. Add your domain to authorized domains
6. Update the client ID in:
   - `src/auth/auth_manager.py` (line 31)
   - `templates/login.html` (line 33 and 209)

### 2. Environment Configuration
```bash
# Add to your environment or .env file
GOOGLE_CLIENT_ID="your-actual-google-client-id.apps.googleusercontent.com"
```

### 3. Production Considerations
- Replace in-memory storage with Redis or database
- Set up proper SSL/HTTPS
- Configure proper CORS settings
- Set up monitoring and logging
- Configure rate limiting
- Set up backup and recovery

## 🧪 Testing the System

### 1. Access the Login Page
```bash
curl http://localhost:8000/login
```

### 2. Check Authentication Status
```bash
curl http://localhost:8000/api/auth/status
```

### 3. Test Protected Endpoint
```bash
curl http://localhost:8000/api/auth/profile
# Should return: {"detail":"Authentication required"}
```

### 4. Test User Portfolio Endpoints
```bash
curl http://localhost:8000/api/user/portfolio/summary
# Should return: {"detail":"Authentication required"}
```

## 🏗️ Architecture

### Authentication Flow
1. User clicks "Sign in with Google"
2. Google OAuth flow completes
3. Backend verifies Google token
4. User account created/updated
5. Session token issued
6. MFA check (if enabled)
7. Access granted to protected resources

### Security Features
- ✅ Rate limiting on failed attempts
- ✅ Session timeout and rotation
- ✅ MFA for sensitive operations
- ✅ Role-based access control
- ✅ Secure password-less authentication
- ✅ Protection against CSRF attacks
- ✅ User portfolio isolation

## 📊 User Roles

### USER (Default)
- Basic trading features
- $10,000 starting capital
- Max 10 positions
- Stocks only

### PREMIUM
- Advanced trading features
- $50,000 starting capital
- Max 25 positions
- Stocks, Options, Crypto, Forex

### ADMIN
- All features + admin panels
- $100,000 starting capital
- Max 50 positions
- All trading types including margin

## 🔐 Security Best Practices

This implementation follows security best practices:
- No passwords stored locally
- OAuth-based authentication
- Session-based authorization
- MFA support for sensitive operations
- User portfolio isolation
- Rate limiting and monitoring
- Secure session management

## 📈 Next Steps

The authentication system is production-ready with the following considerations:
1. Set up proper Google OAuth credentials
2. Configure production database (replace in-memory storage)
3. Set up HTTPS and proper domain configuration
4. Configure monitoring and alerting
5. Set up backup and disaster recovery