"""
Authentication manager for brokers
"""

import os
import json
import keyring
import logging
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from cryptography.fernet import Fernet
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class BrokerAuthManager:
    """Manages authentication credentials for brokers"""
    
    def __init__(self, config_dir: str = "~/.invest"):
        self.config_dir = Path(config_dir).expanduser()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.credentials_file = self.config_dir / "broker_credentials.json"
        self.key_file = self.config_dir / "broker.key"
        
        # Load environment variables
        load_dotenv()
        
        # Initialize encryption
        self._init_encryption()
        
        # Load cached credentials
        self.credentials = self._load_credentials()
    
    def _init_encryption(self):
        """Initialize encryption for storing sensitive data"""
        try:
            if self.key_file.exists():
                with open(self.key_file, 'rb') as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                # Set restrictive permissions
                os.chmod(self.key_file, 0o600)
            
            self.cipher = Fernet(key)
            
        except Exception as e:
            logger.error(f"Error initializing encryption: {str(e)}")
            self.cipher = None
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            if self.cipher:
                return self.cipher.encrypt(data.encode()).decode()
            else:
                logger.warning("Encryption not available, storing data in plain text")
                return data
        except Exception as e:
            logger.error(f"Error encrypting data: {str(e)}")
            return data
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            if self.cipher and encrypted_data:
                return self.cipher.decrypt(encrypted_data.encode()).decode()
            else:
                return encrypted_data
        except Exception as e:
            logger.error(f"Error decrypting data: {str(e)}")
            return encrypted_data
    
    def _load_credentials(self) -> Dict:
        """Load credentials from file"""
        try:
            if self.credentials_file.exists():
                with open(self.credentials_file, 'r') as f:
                    encrypted_creds = json.load(f)
                
                # Decrypt credentials
                credentials = {}
                for broker, creds in encrypted_creds.items():
                    credentials[broker] = {}
                    for key, value in creds.items():
                        if key.endswith('_encrypted'):
                            # Decrypt sensitive fields
                            original_key = key.replace('_encrypted', '')
                            credentials[broker][original_key] = self._decrypt_data(value)
                        else:
                            credentials[broker][key] = value
                
                return credentials
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error loading credentials: {str(e)}")
            return {}
    
    def _save_credentials(self):
        """Save credentials to file"""
        try:
            # Encrypt sensitive credentials
            encrypted_creds = {}
            sensitive_fields = {
                'api_key', 'secret_key', 'access_token', 'refresh_token',
                'consumer_key', 'consumer_secret', 'access_secret',
                'client_secret', 'password'
            }
            
            for broker, creds in self.credentials.items():
                encrypted_creds[broker] = {}
                for key, value in creds.items():
                    if key in sensitive_fields and value:
                        # Encrypt sensitive fields
                        encrypted_creds[broker][f"{key}_encrypted"] = self._encrypt_data(str(value))
                    else:
                        encrypted_creds[broker][key] = value
            
            with open(self.credentials_file, 'w') as f:
                json.dump(encrypted_creds, f, indent=2)
            
            # Set restrictive permissions
            os.chmod(self.credentials_file, 0o600)
            
        except Exception as e:
            logger.error(f"Error saving credentials: {str(e)}")
    
    def get_credentials(self, broker_name: str) -> Dict:
        """Get credentials for a broker"""
        # First try cached credentials
        if broker_name in self.credentials:
            return self.credentials[broker_name].copy()
        
        # Then try environment variables
        env_creds = self._get_credentials_from_env(broker_name)
        if env_creds:
            return env_creds
        
        # Finally try system keyring
        keyring_creds = self._get_credentials_from_keyring(broker_name)
        if keyring_creds:
            return keyring_creds
        
        return {}
    
    def set_credentials(self, broker_name: str, credentials: Dict):
        """Set credentials for a broker"""
        try:
            self.credentials[broker_name] = credentials.copy()
            self._save_credentials()
            logger.info(f"Saved credentials for {broker_name}")
            
        except Exception as e:
            logger.error(f"Error setting credentials for {broker_name}: {str(e)}")
    
    def remove_credentials(self, broker_name: str):
        """Remove credentials for a broker"""
        try:
            if broker_name in self.credentials:
                del self.credentials[broker_name]
                self._save_credentials()
                logger.info(f"Removed credentials for {broker_name}")
            
            # Also remove from keyring if present
            self._remove_from_keyring(broker_name)
            
        except Exception as e:
            logger.error(f"Error removing credentials for {broker_name}: {str(e)}")
    
    def _get_credentials_from_env(self, broker_name: str) -> Dict:
        """Get credentials from environment variables"""
        credentials = {}
        
        # Common environment variable patterns
        env_mappings = {
            'alpaca': {
                'api_key': ['ALPACA_API_KEY', 'APCA_API_KEY_ID'],
                'secret_key': ['ALPACA_SECRET_KEY', 'APCA_API_SECRET_KEY'],
                'paper_trading': ['ALPACA_PAPER_TRADING', 'ALPACA_PAPER']
            },
            'ibkr': {
                'host': ['IBKR_HOST'],
                'port': ['IBKR_PORT'],
                'client_id': ['IBKR_CLIENT_ID']
            },
            'td_ameritrade': {
                'client_id': ['TD_CLIENT_ID', 'TDA_CLIENT_ID'],
                'refresh_token': ['TD_REFRESH_TOKEN', 'TDA_REFRESH_TOKEN'],
                'account_id': ['TD_ACCOUNT_ID', 'TDA_ACCOUNT_ID'],
                'redirect_uri': ['TD_REDIRECT_URI', 'TDA_REDIRECT_URI']
            },
            'etrade': {
                'consumer_key': ['ETRADE_CONSUMER_KEY'],
                'consumer_secret': ['ETRADE_CONSUMER_SECRET'],
                'access_token': ['ETRADE_ACCESS_TOKEN'],
                'access_secret': ['ETRADE_ACCESS_SECRET']
            },
            'schwab': {
                'client_id': ['SCHWAB_CLIENT_ID'],
                'client_secret': ['SCHWAB_CLIENT_SECRET'],
                'access_token': ['SCHWAB_ACCESS_TOKEN'],
                'refresh_token': ['SCHWAB_REFRESH_TOKEN'],
                'redirect_uri': ['SCHWAB_REDIRECT_URI']
            }
        }
        
        # Get broker-specific mappings
        broker_key = broker_name.lower().replace('_', '').replace('-', '')
        for key, patterns in env_mappings.items():
            if key in broker_key or broker_key in key:
                for cred_name, env_vars in patterns.items():
                    for env_var in env_vars:
                        value = os.getenv(env_var)
                        if value:
                            credentials[cred_name] = value
                            break
                break
        
        return credentials
    
    def _get_credentials_from_keyring(self, broker_name: str) -> Dict:
        """Get credentials from system keyring"""
        try:
            service_name = f"invest-system-{broker_name}"
            username = "default"
            
            stored_creds = keyring.get_password(service_name, username)
            if stored_creds:
                return json.loads(stored_creds)
            
            return {}
            
        except Exception as e:
            logger.debug(f"Could not retrieve credentials from keyring for {broker_name}: {str(e)}")
            return {}
    
    def save_to_keyring(self, broker_name: str, credentials: Dict):
        """Save credentials to system keyring"""
        try:
            service_name = f"invest-system-{broker_name}"
            username = "default"
            
            keyring.set_password(service_name, username, json.dumps(credentials))
            logger.info(f"Saved credentials to keyring for {broker_name}")
            
        except Exception as e:
            logger.error(f"Error saving to keyring for {broker_name}: {str(e)}")
    
    def _remove_from_keyring(self, broker_name: str):
        """Remove credentials from system keyring"""
        try:
            service_name = f"invest-system-{broker_name}"
            username = "default"
            
            keyring.delete_password(service_name, username)
            
        except Exception as e:
            logger.debug(f"Could not remove credentials from keyring for {broker_name}: {str(e)}")
    
    def list_stored_brokers(self) -> List[str]:
        """List brokers with stored credentials"""
        return list(self.credentials.keys())
    
    def validate_credentials(self, broker_name: str, credentials: Dict) -> Tuple[bool, str]:
        """Validate broker credentials"""
        try:
            # Basic validation based on broker type
            broker_key = broker_name.lower()
            
            if 'alpaca' in broker_key:
                required = ['api_key', 'secret_key']
                if all(credentials.get(field) for field in required):
                    return True, "Alpaca credentials look valid"
                else:
                    return False, f"Missing required fields: {[f for f in required if not credentials.get(f)]}"
            
            elif 'ibkr' in broker_key or 'interactive' in broker_key:
                # IBKR just needs host and port typically
                return True, "IBKR credentials look valid"
            
            elif 'td' in broker_key or 'ameritrade' in broker_key:
                required = ['client_id']
                if all(credentials.get(field) for field in required):
                    return True, "TD Ameritrade credentials look valid"
                else:
                    return False, f"Missing required fields: {[f for f in required if not credentials.get(f)]}"
            
            elif 'etrade' in broker_key:
                required = ['consumer_key', 'consumer_secret']
                if all(credentials.get(field) for field in required):
                    return True, "E*TRADE credentials look valid"
                else:
                    return False, f"Missing required fields: {[f for f in required if not credentials.get(f)]}"
            
            elif 'schwab' in broker_key:
                required = ['client_id', 'client_secret']
                if all(credentials.get(field) for field in required):
                    return True, "Schwab credentials look valid"
                else:
                    return False, f"Missing required fields: {[f for f in required if not credentials.get(f)]}"
            
            else:
                return True, "Unknown broker type, cannot validate"
                
        except Exception as e:
            return False, f"Error validating credentials: {str(e)}"
    
    def create_sample_env_file(self, file_path: str = ".env.sample"):
        """Create a sample environment file"""
        sample_env = """
# Investment System - Broker Credentials
# Copy to .env and fill in your actual credentials

# General Settings
BROKER_TYPE=alpaca
PAPER_TRADING=true
MAX_POSITION_SIZE=0.05
STOP_LOSS_PCT=0.02

# Alpaca Markets
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_PAPER_TRADING=true

# Alternative Alpaca format
# APCA_API_KEY_ID=your_alpaca_api_key_here
# APCA_API_SECRET_KEY=your_alpaca_secret_key_here

# Interactive Brokers
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# TD Ameritrade
TD_CLIENT_ID=your_td_client_id_here
TD_REFRESH_TOKEN=your_td_refresh_token_here
TD_ACCOUNT_ID=your_td_account_id_here
TD_REDIRECT_URI=https://localhost

# E*TRADE
ETRADE_CONSUMER_KEY=your_etrade_consumer_key_here
ETRADE_CONSUMER_SECRET=your_etrade_consumer_secret_here
ETRADE_ACCESS_TOKEN=your_etrade_access_token_here
ETRADE_ACCESS_SECRET=your_etrade_access_secret_here

# Charles Schwab
SCHWAB_CLIENT_ID=your_schwab_client_id_here
SCHWAB_CLIENT_SECRET=your_schwab_client_secret_here
SCHWAB_ACCESS_TOKEN=your_schwab_access_token_here
SCHWAB_REFRESH_TOKEN=your_schwab_refresh_token_here
SCHWAB_REDIRECT_URI=https://localhost
"""
        
        try:
            with open(file_path, 'w') as f:
                f.write(sample_env.strip())
            
            logger.info(f"Created sample environment file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error creating sample env file: {str(e)}")