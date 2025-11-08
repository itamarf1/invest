# Multi-Broker Trading System

## Overview

This investment system now supports **multiple brokers** through a unified abstraction layer. You can connect to and trade across multiple broker accounts simultaneously, with automatic order distribution, consolidated portfolio tracking, and unified risk management.

## 🚀 Supported Brokers

| Broker | Status | Features | Authentication |
|--------|--------|----------|----------------|
| **Alpaca Markets** | ✅ Full Support | All trading operations | API Keys |
| **Interactive Brokers** | ✅ Full Support | All trading operations | TWS/Gateway |
| **TD Ameritrade** | ⚠️ OAuth Required | All trading operations | OAuth 2.0 |
| **E*TRADE** | 🚧 Placeholder | Basic structure | OAuth 1.0 |
| **Charles Schwab** | 🚧 Placeholder | Basic structure | OAuth 2.0 |
| **Simulation** | ✅ Full Support | Paper trading | None |

## 📋 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize Configuration

```bash
python broker_cli.py init
```

This creates:
- `~/.invest/brokers.json` - Broker configurations
- `.env.sample` - Environment variable template

### 3. Configure Your First Broker

#### Option A: Using Environment Variables
```bash
cp .env.sample .env
# Edit .env with your broker credentials
```

#### Option B: Using Configuration File
```bash
python broker_cli.py add
# Follow interactive prompts
```

#### Option C: Using CLI
```bash
python broker_cli.py set-creds alpaca_paper
```

### 4. Test Connection

```bash
python broker_cli.py test alpaca_paper
```

### 5. View Status

```bash
python broker_cli.py status
```

## 🏗️ Architecture

### Broker Abstraction Layer

```
┌─────────────────┐
│   Your App      │
├─────────────────┤
│ MultiBrokerMgr  │ ← Manages multiple brokers
├─────────────────┤
│  BrokerFactory  │ ← Creates broker instances
├─────────────────┤
│   BaseBroker    │ ← Common interface
├─────────────────┤
│ AlpacaBroker    │ │ IBKRBroker  │ │ TDBroker │
│ SchwabBroker    │ │ ETradeBroker│ │ SimBroker│
└─────────────────┘
```

### Key Components

1. **BaseBroker** - Abstract base class defining common interface
2. **BrokerFactory** - Creates broker instances from configuration
3. **MultiBrokerManager** - Manages multiple broker accounts
4. **AuthManager** - Handles secure credential storage
5. **ConfigManager** - Manages broker configurations

## 📖 Usage Examples

### Basic Usage

```python
from src.trading.brokers.factory import BrokerFactory
from src.trading.brokers.base import BrokerType, OrderSide, OrderType

# Create broker
broker = BrokerFactory.create_from_env(BrokerType.ALPACA)

# Connect and trade
if broker.connect():
    # Get account info
    account = broker.get_account_info()
    print(f"Portfolio Value: ${account.portfolio_value:,.2f}")
    
    # Submit order
    order = broker.submit_order(
        symbol="AAPL",
        quantity=10,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET
    )
    
    # Get positions
    positions = broker.get_positions()
    for pos in positions:
        print(f"{pos.symbol}: {pos.quantity} shares")
```

### Multi-Broker Usage

```python
from src.trading.brokers.multi_broker_manager import MultiBrokerManager
from src.trading.brokers.factory import BrokerFactory

# Create manager
manager = MultiBrokerManager()

# Add brokers
alpaca = BrokerFactory.create_from_env(BrokerType.ALPACA)
sim = BrokerFactory.create_broker(BrokerType.SIMULATION, {
    'initial_balance': 100000
})

manager.add_broker("alpaca", alpaca, is_default=True)
manager.add_broker("simulation", sim, weight=0.5)

# Connect all
manager.connect_all()

# Distribute order across brokers
results = manager.submit_order_distributed(
    symbol="TSLA",
    total_quantity=100,
    side=OrderSide.BUY
)

# Get consolidated positions
positions = manager.get_consolidated_positions()
```

## 🔐 Security & Authentication

### Credential Storage

The system supports multiple credential storage methods:

1. **Environment Variables** (recommended for production)
2. **Encrypted JSON Files** (local development)
3. **System Keyring** (secure local storage)

### Setting Up Credentials

#### Alpaca Markets
```bash
export ALPACA_API_KEY="your_key_here"
export ALPACA_SECRET_KEY="your_secret_here" 
export ALPACA_PAPER_TRADING="true"
```

#### Interactive Brokers
```bash
export IBKR_HOST="127.0.0.1"
export IBKR_PORT="7497"  # 7497 for paper, 7496 for live
export IBKR_CLIENT_ID="1"
```

#### TD Ameritrade
```bash
export TD_CLIENT_ID="your_client_id"
export TD_REFRESH_TOKEN="your_refresh_token"
export TD_ACCOUNT_ID="your_account_id"
```

### OAuth Setup

#### TD Ameritrade OAuth Flow
```python
from src.trading.brokers.td_ameritrade_broker import TDAmeritradeBroker

broker = TDAmeritradeBroker({
    'client_id': 'your_client_id',
    'redirect_uri': 'https://localhost'
})

# Start OAuth flow
auth_url = broker.authenticate_manually()
print(f"Visit: {auth_url}")

# After getting authorization code
broker.complete_authentication("authorization_code_from_callback")
```

## 📊 Configuration

### Broker Configuration File

```json
{
  "default_broker": "alpaca_paper",
  "brokers": {
    "alpaca_paper": {
      "type": "alpaca",
      "paper_trading": true,
      "max_position_size": 0.05,
      "stop_loss_pct": 0.02
    },
    "ibkr": {
      "type": "ibkr",
      "paper_trading": true,
      "host": "127.0.0.1",
      "port": 7497,
      "client_id": 1
    },
    "simulation": {
      "type": "simulation",
      "initial_balance": 100000,
      "commission": 0.0
    }
  }
}
```

### Risk Management Settings

Each broker can have individual risk settings:

```json
{
  "max_position_size": 0.05,      // 5% max position size
  "stop_loss_pct": 0.02,          // 2% stop loss
  "max_daily_loss": 0.05,         // 5% max daily loss
  "min_cash_balance": 1000        // Minimum cash balance
}
```

## 🔧 CLI Tools

### Broker Management CLI

```bash
# List available broker types
python broker_cli.py list-types

# List configured brokers
python broker_cli.py list-brokers

# Show status of all brokers
python broker_cli.py status

# Add new broker interactively
python broker_cli.py add

# Test broker connection
python broker_cli.py test alpaca_paper

# Set credentials for broker
python broker_cli.py set-creds alpaca_paper
```

### Main Investment CLI (Updated)

```bash
# Use specific broker for trading
python main.py --broker alpaca_paper buy --symbol AAPL --quantity 10

# Show multi-broker portfolio
python main.py multi-portfolio

# Distribute order across brokers
python main.py distributed-buy --symbol TSLA --quantity 100
```

## 🧪 Testing

### Run Broker Tests

```bash
# Run all broker tests
python tests/test_brokers.py

# Run specific test
python -m unittest tests.test_brokers.TestSimulationBroker

# Run with verbose output
python tests/test_brokers.py -v
```

### Integration Testing

```bash
# Test with real broker (paper trading)
python examples/multi_broker_example.py
```

## 📋 Broker-Specific Setup

### Alpaca Markets
1. Sign up at [alpaca.markets](https://alpaca.markets)
2. Generate API keys from dashboard
3. Set environment variables or add to config

### Interactive Brokers
1. Install TWS or IB Gateway
2. Enable API connections in settings
3. Configure socket port (7497 for paper, 7496 for live)
4. Set client ID and connection details

### TD Ameritrade
1. Create developer account at [developer.tdameritrade.com](https://developer.tdameritrade.com)
2. Register your application
3. Complete OAuth flow to get refresh token
4. Set client ID and tokens

### E*TRADE (Placeholder)
1. Register developer account
2. Complete OAuth 1.0 flow
3. Implementation needs completion

### Charles Schwab (Placeholder)
1. Register for API access (acquired TD Ameritrade)
2. Similar to TD Ameritrade process
3. Implementation needs completion

## 🚨 Important Notes

### Security
- **Never commit API keys** to version control
- Use environment variables in production
- Enable 2FA on all broker accounts
- Start with paper trading accounts

### Risk Management
- Set position size limits appropriately
- Use stop losses
- Monitor daily loss limits
- Test thoroughly with small amounts

### Production Considerations
- Use paper trading initially
- Implement proper logging
- Set up monitoring and alerts
- Have rollback procedures

## 🔄 Migration from Single Broker

### From Existing Alpaca Integration

```python
# Old way
from src.trading.paper_trader import PaperTrader
trader = PaperTrader()

# New way
from src.trading.brokers.factory import BrokerFactory
from src.trading.brokers.base import BrokerType

broker = BrokerFactory.create_from_env(BrokerType.ALPACA)
```

### Configuration Migration

The old `PaperTrader` class is still supported, but new code should use the broker abstraction layer.

## 🛠️ Development

### Adding a New Broker

1. **Create broker class** inheriting from `BaseBroker`
2. **Implement required methods**: `connect()`, `submit_order()`, etc.
3. **Add to factory** in `BrokerFactory._brokers`
4. **Update authentication** patterns in `AuthManager`
5. **Add tests** for the new broker

### Example Broker Implementation

```python
class MyBroker(BaseBroker):
    def _get_broker_type(self) -> BrokerType:
        return BrokerType.MY_BROKER
    
    def connect(self) -> bool:
        # Implement connection logic
        pass
    
    def submit_order(self, symbol, quantity, side, **kwargs):
        # Implement order submission
        pass
    
    # ... implement other required methods
```

## 📞 Support

### Getting Help
- Check logs in `~/.invest/logs/`
- Run broker tests to verify setup
- Use simulation broker for testing
- Check broker-specific documentation

### Common Issues
- **Connection Failed**: Check credentials and network
- **Orders Rejected**: Verify account permissions and balance
- **OAuth Issues**: Check redirect URIs and token expiration

## 🎯 Roadmap

- [ ] Complete E*TRADE OAuth implementation
- [ ] Complete Schwab API integration
- [ ] Add Robinhood support
- [ ] Add Fidelity support
- [ ] WebSocket real-time data feeds
- [ ] Advanced order types (bracket, OCO)
- [ ] Multi-broker arbitrage detection
- [ ] Portfolio rebalancing across brokers

---

**⚠️ Disclaimer**: This is trading software that can place real orders. Always test with paper trading accounts first. Use at your own risk. The authors are not responsible for any financial losses.