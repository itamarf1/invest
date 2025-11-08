#!/usr/bin/env python3
"""
Multi-broker trading system example
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from trading.brokers.factory import BrokerFactory, BrokerConfigManager
from trading.brokers.multi_broker_manager import MultiBrokerManager
from trading.brokers.auth_manager import BrokerAuthManager
from trading.brokers.base import BrokerType, OrderSide, OrderType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Example of using multiple brokers"""
    
    print("🚀 Multi-Broker Trading System Example")
    print("=" * 50)
    
    # Initialize managers
    auth_manager = BrokerAuthManager()
    multi_broker = MultiBrokerManager()
    
    # Example 1: Add brokers from environment variables
    print("\n1. Adding brokers from environment...")
    
    # Add Alpaca broker (paper trading)
    try:
        alpaca_broker = BrokerFactory.create_from_env(BrokerType.ALPACA)
        if alpaca_broker:
            multi_broker.add_broker("alpaca_paper", alpaca_broker, is_default=True)
            print("✅ Added Alpaca paper trading broker")
        else:
            print("⚠️  Could not create Alpaca broker (check credentials)")
    except Exception as e:
        print(f"❌ Error adding Alpaca: {str(e)}")
    
    # Add simulation broker
    try:
        sim_config = {
            'paper_trading': True,
            'initial_balance': 100000.0,
            'commission': 1.0
        }
        sim_broker = BrokerFactory.create_broker(BrokerType.SIMULATION, sim_config)
        if sim_broker:
            multi_broker.add_broker("simulation", sim_broker, weight=0.5)
            print("✅ Added simulation broker")
    except Exception as e:
        print(f"❌ Error adding simulation broker: {str(e)}")
    
    # Add Interactive Brokers (if TWS/Gateway is running)
    try:
        ibkr_config = {
            'paper_trading': True,
            'host': '127.0.0.1',
            'port': 7497,
            'client_id': 1
        }
        ibkr_broker = BrokerFactory.create_broker(BrokerType.INTERACTIVE_BROKERS, ibkr_config)
        if ibkr_broker:
            multi_broker.add_broker("ibkr_paper", ibkr_broker, weight=0.3)
            print("✅ Added IBKR paper trading broker")
    except Exception as e:
        print(f"⚠️  Could not add IBKR broker (TWS/Gateway may not be running): {str(e)}")
    
    # Example 2: Connect to all brokers
    print("\n2. Connecting to all brokers...")
    connection_results = multi_broker.connect_all()
    
    for broker_name, connected in connection_results.items():
        status = "✅ Connected" if connected else "❌ Failed"
        print(f"  {broker_name}: {status}")
    
    # Example 3: Show broker status
    print("\n3. Broker Status:")
    status = multi_broker.get_broker_status()
    for broker_name, info in status.items():
        print(f"  {broker_name}:")
        print(f"    Type: {info['broker_type']}")
        print(f"    Connected: {info['is_connected']}")
        print(f"    Active: {info['is_active']}")
        print(f"    Paper Trading: {info['paper_trading']}")
        print(f"    Portfolio Value: ${info['portfolio_value']:,.2f}")
        print(f"    Cash: ${info['cash']:,.2f}")
        print(f"    Weight: {info['weight']}")
        if info['is_default']:
            print(f"    ⭐ DEFAULT BROKER")
        print()
    
    # Example 4: Get consolidated portfolio
    print("4. Consolidated Portfolio:")
    total_value = multi_broker.get_total_account_value()
    print(f"Total Portfolio Value: ${total_value:,.2f}")
    
    positions = multi_broker.get_consolidated_positions()
    if positions:
        print("\nConsolidated Positions:")
        for symbol, position in positions.items():
            print(f"  {symbol}: {position.quantity} shares @ ${position.avg_entry_price:.2f}")
            print(f"    Current: ${position.current_price:.2f}")
            print(f"    P&L: ${position.unrealized_pnl:.2f} ({position.unrealized_pnl_pct:.1f}%)")
    else:
        print("No positions found")
    
    # Example 5: Demonstrate order distribution
    print("\n5. Example: Distributed Order Submission")
    print("This would distribute a 100-share AAPL order across connected brokers:")
    
    # Don't actually submit orders in this example
    print("  - Simulation broker (weight 0.5): ~50 shares")
    print("  - IBKR broker (weight 0.3): ~30 shares")  
    print("  - Alpaca broker (weight 1.0): ~100 shares if only one connected")
    print("  (Orders not actually submitted in this example)")
    
    # Uncomment to actually submit orders (be careful!)
    # results = multi_broker.submit_order_distributed(
    #     symbol="AAPL",
    #     total_quantity=100,
    #     side=OrderSide.BUY,
    #     order_type=OrderType.MARKET
    # )
    
    # Example 6: Authentication management
    print("\n6. Authentication Management Example:")
    
    # Show stored credentials (without sensitive data)
    stored_brokers = auth_manager.list_stored_brokers()
    print(f"Brokers with stored credentials: {stored_brokers}")
    
    # Example of setting credentials (for demo purposes)
    print("\nExample credential storage:")
    demo_creds = {
        'api_key': 'demo_key_12345',
        'secret_key': 'demo_secret_67890',
        'paper_trading': True
    }
    
    # Validate credentials
    is_valid, message = auth_manager.validate_credentials('alpaca', demo_creds)
    print(f"Demo credentials validation: {message}")
    
    # Example 7: Configuration management
    print("\n7. Configuration Management:")
    config_path = Path.home() / ".invest" / "brokers.json"
    config_manager = BrokerConfigManager(str(config_path))
    
    if not config_path.exists():
        print("Creating sample broker configuration...")
        sample_config = config_manager.create_sample_config()
        print(f"Sample config created at: {config_path}")
        print("Edit this file with your actual broker credentials")
    else:
        print(f"Existing config found at: {config_path}")
        brokers = config_manager.list_brokers()
        print(f"Configured brokers: {brokers}")
    
    # Clean up
    print("\n8. Cleanup:")
    disconnect_results = multi_broker.disconnect_all()
    for broker_name, disconnected in disconnect_results.items():
        status = "✅ Disconnected" if disconnected else "❌ Error"
        print(f"  {broker_name}: {status}")
    
    print("\n✨ Multi-broker example completed!")
    print("\nNext steps:")
    print("1. Configure your actual broker credentials in environment variables or config file")
    print("2. For IBKR: Start TWS or IB Gateway")
    print("3. For TD Ameritrade: Complete OAuth flow to get refresh token")
    print("4. For E*TRADE/Schwab: Implement full OAuth integration")
    print("5. Test with small positions first!")

if __name__ == "__main__":
    main()