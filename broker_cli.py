#!/usr/bin/env python3
"""
Broker management CLI tool
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from trading.brokers.factory import BrokerFactory, BrokerConfigManager
from trading.brokers.multi_broker_manager import MultiBrokerManager
from trading.brokers.auth_manager import BrokerAuthManager
from trading.brokers.base import BrokerType, OrderSide, OrderType

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class BrokerCLI:
    """CLI tool for managing brokers"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".invest"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.auth_manager = BrokerAuthManager(str(self.config_dir))
        self.config_manager = BrokerConfigManager(str(self.config_dir / "brokers.json"))
        self.multi_broker = MultiBrokerManager()
    
    def list_available_brokers(self):
        """List all available broker types"""
        print("Available broker types:")
        available = BrokerFactory.get_available_brokers()
        
        for broker_type, class_name in available.items():
            print(f"  {broker_type:<20} - {class_name}")
    
    def list_configured_brokers(self):
        """List configured brokers"""
        print("Configured brokers:")
        brokers = self.config_manager.list_brokers()
        
        if not brokers:
            print("  No brokers configured")
        else:
            for name, broker_type in brokers.items():
                default = " (DEFAULT)" if name == self.config_manager.config.get('default_broker') else ""
                print(f"  {name:<20} - {broker_type}{default}")
    
    def create_sample_config(self):
        """Create sample configuration files"""
        print("Creating sample configuration files...")
        
        # Create sample broker config
        config_path = self.config_dir / "brokers.json"
        self.config_manager.create_sample_config()
        print(f"✅ Created sample broker config: {config_path}")
        
        # Create sample env file
        env_path = Path.cwd() / ".env.sample"
        self.auth_manager.create_sample_env_file(str(env_path))
        print(f"✅ Created sample environment file: {env_path}")
        
        print("\nNext steps:")
        print("1. Copy .env.sample to .env and fill in your credentials")
        print("2. Edit the broker configuration file with your settings")
        print("3. Test connection with: broker_cli.py test <broker_name>")
    
    def add_broker_interactive(self):
        """Interactively add a broker"""
        print("Add new broker configuration")
        print("-" * 30)
        
        # Get broker name
        name = input("Broker name: ").strip()
        if not name:
            print("❌ Broker name is required")
            return
        
        # Get broker type
        print("\nAvailable broker types:")
        broker_types = list(BrokerType)
        for i, bt in enumerate(broker_types, 1):
            print(f"  {i}. {bt.value}")
        
        try:
            choice = int(input("\nSelect broker type (number): "))
            if choice < 1 or choice > len(broker_types):
                raise ValueError()
            broker_type = broker_types[choice - 1]
        except (ValueError, IndexError):
            print("❌ Invalid choice")
            return
        
        # Get broker-specific configuration
        config = {'paper_trading': True}  # Default to paper trading
        
        if broker_type == BrokerType.ALPACA:
            config['api_key'] = input("Alpaca API Key: ").strip()
            config['secret_key'] = input("Alpaca Secret Key: ").strip()
            
        elif broker_type == BrokerType.INTERACTIVE_BROKERS:
            config['host'] = input("TWS/Gateway Host (127.0.0.1): ").strip() or "127.0.0.1"
            config['port'] = int(input("TWS/Gateway Port (7497): ") or "7497")
            config['client_id'] = int(input("Client ID (1): ") or "1")
            
        elif broker_type == BrokerType.TD_AMERITRADE:
            config['client_id'] = input("TD Ameritrade Client ID: ").strip()
            config['refresh_token'] = input("TD Ameritrade Refresh Token: ").strip()
            config['account_id'] = input("TD Ameritrade Account ID: ").strip()
            
        elif broker_type == BrokerType.SIMULATION:
            config['initial_balance'] = float(input("Initial Balance (100000): ") or "100000")
            config['commission'] = float(input("Commission per trade (0): ") or "0")
        
        # Paper trading option
        if broker_type != BrokerType.SIMULATION:
            paper = input("Paper trading? (y/N): ").strip().lower()
            config['paper_trading'] = paper.startswith('y')
        
        # Risk management settings
        config['max_position_size'] = float(input("Max position size (0.05): ") or "0.05")
        config['stop_loss_pct'] = float(input("Stop loss percentage (0.02): ") or "0.02")
        
        # Save configuration
        self.config_manager.add_broker(name, broker_type, config)
        print(f"✅ Added broker: {name}")
        
        # Ask if this should be the default
        if len(self.config_manager.list_brokers()) == 1:
            print(f"Set as default broker (first broker)")
        else:
            default = input("Set as default broker? (y/N): ").strip().lower()
            if default.startswith('y'):
                self.config_manager.set_default_broker(name)
    
    def test_broker(self, broker_name: str):
        """Test connection to a broker"""
        print(f"Testing connection to broker: {broker_name}")
        
        try:
            # Create broker from config
            broker_config = self.config_manager.get_broker_config(broker_name)
            if not broker_config:
                print(f"❌ Broker '{broker_name}' not found in configuration")
                return
            
            broker_type = BrokerType(broker_config['type'])
            
            # Merge with credentials
            auth_creds = self.auth_manager.get_credentials(broker_name)
            if auth_creds:
                broker_config.update(auth_creds)
            
            # Create and test broker
            broker = BrokerFactory.create_broker(broker_type, broker_config)
            if not broker:
                print(f"❌ Failed to create broker instance")
                return
            
            print(f"Created {broker}")
            
            # Test connection
            if broker.connect():
                print("✅ Connection successful!")
                
                # Get account info
                account_info = broker.get_account_info()
                if account_info:
                    print(f"\nAccount Info:")
                    print(f"  Account ID: {account_info.account_id}")
                    print(f"  Status: {account_info.status}")
                    print(f"  Portfolio Value: ${account_info.portfolio_value:,.2f}")
                    print(f"  Cash: ${account_info.cash:,.2f}")
                    print(f"  Buying Power: ${account_info.buying_power:,.2f}")
                
                # Get positions
                positions = broker.get_positions()
                if positions:
                    print(f"\nPositions ({len(positions)}):")
                    for pos in positions[:5]:  # Show first 5
                        print(f"  {pos.symbol}: {pos.quantity} shares @ ${pos.avg_entry_price:.2f}")
                else:
                    print("\nNo positions found")
                
                # Disconnect
                broker.disconnect()
                print("\n✅ Test completed successfully")
            else:
                print("❌ Connection failed")
                
        except Exception as e:
            print(f"❌ Error testing broker: {str(e)}")
    
    def status_all_brokers(self):
        """Show status of all configured brokers"""
        print("Loading all configured brokers...")
        
        brokers = self.config_manager.list_brokers()
        if not brokers:
            print("No brokers configured")
            return
        
        # Load all brokers
        for name, broker_type_str in brokers.items():
            try:
                broker_config = self.config_manager.get_broker_config(name)
                broker_type = BrokerType(broker_type_str)
                
                # Merge credentials
                auth_creds = self.auth_manager.get_credentials(name)
                if auth_creds:
                    broker_config.update(auth_creds)
                
                broker = BrokerFactory.create_broker(broker_type, broker_config)
                if broker:
                    is_default = name == self.config_manager.config.get('default_broker')
                    self.multi_broker.add_broker(name, broker, is_default=is_default)
                    
            except Exception as e:
                print(f"⚠️  Could not load {name}: {str(e)}")
        
        # Connect and show status
        print("\nConnecting to brokers...")
        results = self.multi_broker.connect_all()
        
        print("\nBroker Status:")
        print("-" * 80)
        print(f"{'Name':<15} {'Type':<12} {'Connected':<10} {'Paper':<6} {'Portfolio':<12} {'Cash':<12}")
        print("-" * 80)
        
        status = self.multi_broker.get_broker_status()
        for name, info in status.items():
            connected = "✅ Yes" if info['is_connected'] else "❌ No"
            paper = "Yes" if info['paper_trading'] else "No"
            portfolio = f"${info['portfolio_value']:,.0f}"
            cash = f"${info['cash']:,.0f}"
            
            print(f"{name:<15} {info['broker_type']:<12} {connected:<10} {paper:<6} {portfolio:<12} {cash:<12}")
            
            if info['is_default']:
                print(f"{'':>15} ⭐ DEFAULT BROKER")
        
        # Show totals
        total_value = self.multi_broker.get_total_account_value()
        print("-" * 80)
        print(f"{'TOTAL':<15} {'':>32} ${total_value:,.2f}")
    
    def set_credentials(self, broker_name: str):
        """Set credentials for a broker"""
        print(f"Setting credentials for broker: {broker_name}")
        
        # Determine broker type
        broker_config = self.config_manager.get_broker_config(broker_name)
        if not broker_config:
            print(f"❌ Broker '{broker_name}' not found")
            return
        
        broker_type = broker_config['type']
        credentials = {}
        
        if broker_type == 'alpaca':
            credentials['api_key'] = input("Alpaca API Key: ").strip()
            credentials['secret_key'] = input("Alpaca Secret Key: ").strip()
            
        elif broker_type == 'td_ameritrade':
            credentials['client_id'] = input("TD Client ID: ").strip()
            credentials['refresh_token'] = input("TD Refresh Token: ").strip()
            credentials['account_id'] = input("TD Account ID (optional): ").strip()
            
        elif broker_type == 'etrade':
            credentials['consumer_key'] = input("E*TRADE Consumer Key: ").strip()
            credentials['consumer_secret'] = input("E*TRADE Consumer Secret: ").strip()
            credentials['access_token'] = input("E*TRADE Access Token: ").strip()
            credentials['access_secret'] = input("E*TRADE Access Secret: ").strip()
            
        else:
            print(f"Interactive credential setting not supported for {broker_type}")
            return
        
        # Validate and save
        is_valid, message = self.auth_manager.validate_credentials(broker_name, credentials)
        if is_valid:
            self.auth_manager.set_credentials(broker_name, credentials)
            print("✅ Credentials saved successfully")
        else:
            print(f"❌ Validation failed: {message}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Broker Management CLI")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List commands
    subparsers.add_parser('list-types', help='List available broker types')
    subparsers.add_parser('list-brokers', help='List configured brokers')
    subparsers.add_parser('status', help='Show status of all brokers')
    
    # Configuration commands
    subparsers.add_parser('init', help='Create sample configuration files')
    subparsers.add_parser('add', help='Add a new broker interactively')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test broker connection')
    test_parser.add_argument('broker_name', help='Broker name to test')
    
    # Credentials command  
    creds_parser = subparsers.add_parser('set-creds', help='Set broker credentials')
    creds_parser.add_argument('broker_name', help='Broker name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = BrokerCLI()
    
    try:
        if args.command == 'list-types':
            cli.list_available_brokers()
        elif args.command == 'list-brokers':
            cli.list_configured_brokers()
        elif args.command == 'status':
            cli.status_all_brokers()
        elif args.command == 'init':
            cli.create_sample_config()
        elif args.command == 'add':
            cli.add_broker_interactive()
        elif args.command == 'test':
            cli.test_broker(args.broker_name)
        elif args.command == 'set-creds':
            cli.set_credentials(args.broker_name)
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()