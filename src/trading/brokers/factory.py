"""
Broker factory and configuration management
"""

import os
import json
import logging
from typing import Dict, Optional, Type
from pathlib import Path

from .base import BaseBroker, BrokerType
from .alpaca_broker import AlpacaBroker
from .ibkr_broker import IBKRBroker
from .td_ameritrade_broker import TDAmeritradeBroker
from .etrade_broker import ETradeBroker
from .schwab_broker import SchwabBroker
from .simulation_broker import SimulationBroker

logger = logging.getLogger(__name__)

class BrokerFactory:
    """Factory for creating broker instances"""
    
    # Registry of available brokers
    _brokers: Dict[BrokerType, Type[BaseBroker]] = {
        BrokerType.ALPACA: AlpacaBroker,
        BrokerType.INTERACTIVE_BROKERS: IBKRBroker,
        BrokerType.TD_AMERITRADE: TDAmeritradeBroker,
        BrokerType.ETRADE: ETradeBroker,
        BrokerType.CHARLES_SCHWAB: SchwabBroker,
        BrokerType.SIMULATION: SimulationBroker
    }
    
    @classmethod
    def create_broker(cls, broker_type: BrokerType, config: Optional[Dict] = None) -> Optional[BaseBroker]:
        """Create a broker instance"""
        if config is None:
            config = {}
        
        try:
            if broker_type not in cls._brokers:
                logger.error(f"Unsupported broker type: {broker_type}")
                return None
            
            broker_class = cls._brokers[broker_type]
            broker = broker_class(config)
            
            logger.info(f"Created {broker_type.value} broker instance")
            return broker
            
        except Exception as e:
            logger.error(f"Error creating {broker_type.value} broker: {str(e)}")
            return None
    
    @classmethod
    def create_from_config_file(cls, config_path: str, broker_name: str = None) -> Optional[BaseBroker]:
        """Create broker from configuration file"""
        try:
            config_manager = BrokerConfigManager(config_path)
            
            if broker_name:
                broker_config = config_manager.get_broker_config(broker_name)
                if not broker_config:
                    logger.error(f"Broker '{broker_name}' not found in config")
                    return None
                
                broker_type = BrokerType(broker_config['type'])
                return cls.create_broker(broker_type, broker_config)
            else:
                # Use default broker
                default_config = config_manager.get_default_broker_config()
                if not default_config:
                    logger.error("No default broker configuration found")
                    return None
                
                broker_type = BrokerType(default_config['type'])
                return cls.create_broker(broker_type, default_config)
                
        except Exception as e:
            logger.error(f"Error creating broker from config file: {str(e)}")
            return None
    
    @classmethod
    def create_from_env(cls, broker_type: Optional[BrokerType] = None) -> Optional[BaseBroker]:
        """Create broker from environment variables"""
        try:
            if broker_type is None:
                # Try to determine broker type from environment
                broker_type_str = os.getenv('BROKER_TYPE', 'alpaca').lower()
                try:
                    broker_type = BrokerType(broker_type_str)
                except ValueError:
                    logger.error(f"Invalid broker type in environment: {broker_type_str}")
                    return None
            
            # Build config from environment variables
            config = cls._build_config_from_env(broker_type)
            return cls.create_broker(broker_type, config)
            
        except Exception as e:
            logger.error(f"Error creating broker from environment: {str(e)}")
            return None
    
    @classmethod
    def _build_config_from_env(cls, broker_type: BrokerType) -> Dict:
        """Build broker configuration from environment variables"""
        config = {
            'paper_trading': os.getenv('PAPER_TRADING', 'true').lower() == 'true',
            'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '0.05')),
            'stop_loss_pct': float(os.getenv('STOP_LOSS_PCT', '0.02')),
            'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '0.05')),
            'min_cash_balance': float(os.getenv('MIN_CASH_BALANCE', '1000'))
        }
        
        if broker_type == BrokerType.ALPACA:
            config.update({
                'api_key': os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID'),
                'secret_key': os.getenv('ALPACA_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY')
            })
        
        elif broker_type == BrokerType.INTERACTIVE_BROKERS:
            config.update({
                'host': os.getenv('IBKR_HOST', '127.0.0.1'),
                'port': int(os.getenv('IBKR_PORT', '7497' if config['paper_trading'] else '7496')),
                'client_id': int(os.getenv('IBKR_CLIENT_ID', '1'))
            })
        
        elif broker_type == BrokerType.TD_AMERITRADE:
            config.update({
                'client_id': os.getenv('TD_CLIENT_ID'),
                'redirect_uri': os.getenv('TD_REDIRECT_URI', 'https://localhost'),
                'refresh_token': os.getenv('TD_REFRESH_TOKEN'),
                'account_id': os.getenv('TD_ACCOUNT_ID')
            })
        
        elif broker_type == BrokerType.ETRADE:
            config.update({
                'consumer_key': os.getenv('ETRADE_CONSUMER_KEY'),
                'consumer_secret': os.getenv('ETRADE_CONSUMER_SECRET'),
                'access_token': os.getenv('ETRADE_ACCESS_TOKEN'),
                'access_secret': os.getenv('ETRADE_ACCESS_SECRET')
            })
        
        elif broker_type == BrokerType.CHARLES_SCHWAB:
            config.update({
                'client_id': os.getenv('SCHWAB_CLIENT_ID'),
                'client_secret': os.getenv('SCHWAB_CLIENT_SECRET'),
                'redirect_uri': os.getenv('SCHWAB_REDIRECT_URI', 'https://localhost'),
                'access_token': os.getenv('SCHWAB_ACCESS_TOKEN'),
                'refresh_token': os.getenv('SCHWAB_REFRESH_TOKEN')
            })
        
        return config
    
    @classmethod
    def get_available_brokers(cls) -> Dict[str, str]:
        """Get list of available broker types"""
        return {
            broker_type.value: broker_class.__name__ 
            for broker_type, broker_class in cls._brokers.items()
        }

class BrokerConfigManager:
    """Manages broker configurations from file"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}")
                return {}
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loaded broker configuration from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
            return {}
    
    def get_broker_config(self, broker_name: str) -> Optional[Dict]:
        """Get configuration for a specific broker"""
        brokers = self.config.get('brokers', {})
        return brokers.get(broker_name)
    
    def get_default_broker_config(self) -> Optional[Dict]:
        """Get default broker configuration"""
        default_name = self.config.get('default_broker')
        if default_name:
            return self.get_broker_config(default_name)
        
        # If no default specified, return first available broker
        brokers = self.config.get('brokers', {})
        if brokers:
            return next(iter(brokers.values()))
        
        return None
    
    def save_config(self, config: Dict):
        """Save configuration to file"""
        try:
            self.config = config
            
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved broker configuration to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config file: {str(e)}")
    
    def add_broker(self, name: str, broker_type: BrokerType, broker_config: Dict):
        """Add a broker configuration"""
        if 'brokers' not in self.config:
            self.config['brokers'] = {}
        
        self.config['brokers'][name] = {
            'type': broker_type.value,
            **broker_config
        }
        
        # Set as default if it's the first broker
        if len(self.config['brokers']) == 1:
            self.config['default_broker'] = name
        
        self.save_config(self.config)
    
    def set_default_broker(self, broker_name: str):
        """Set default broker"""
        if broker_name in self.config.get('brokers', {}):
            self.config['default_broker'] = broker_name
            self.save_config(self.config)
        else:
            logger.error(f"Broker '{broker_name}' not found")
    
    def list_brokers(self) -> Dict[str, str]:
        """List all configured brokers"""
        brokers = self.config.get('brokers', {})
        return {name: config['type'] for name, config in brokers.items()}
    
    def create_sample_config(self):
        """Create a sample configuration file"""
        sample_config = {
            "default_broker": "alpaca_paper",
            "brokers": {
                "alpaca_paper": {
                    "type": "alpaca",
                    "paper_trading": True,
                    "api_key": "YOUR_ALPACA_API_KEY",
                    "secret_key": "YOUR_ALPACA_SECRET_KEY",
                    "max_position_size": 0.05,
                    "stop_loss_pct": 0.02
                },
                "alpaca_live": {
                    "type": "alpaca", 
                    "paper_trading": False,
                    "api_key": "YOUR_ALPACA_API_KEY",
                    "secret_key": "YOUR_ALPACA_SECRET_KEY",
                    "max_position_size": 0.03,
                    "stop_loss_pct": 0.015
                },
                "ibkr": {
                    "type": "ibkr",
                    "paper_trading": True,
                    "host": "127.0.0.1",
                    "port": 7497,
                    "client_id": 1,
                    "max_position_size": 0.05
                },
                "td_ameritrade": {
                    "type": "td_ameritrade",
                    "paper_trading": False,
                    "client_id": "YOUR_TD_CLIENT_ID",
                    "refresh_token": "YOUR_TD_REFRESH_TOKEN",
                    "account_id": "YOUR_TD_ACCOUNT_ID"
                },
                "simulation": {
                    "type": "simulation",
                    "paper_trading": True,
                    "initial_balance": 100000,
                    "commission": 0.0
                }
            }
        }
        
        self.save_config(sample_config)
        logger.info(f"Created sample configuration at {self.config_path}")
        return sample_config