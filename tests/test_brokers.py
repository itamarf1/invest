#!/usr/bin/env python3
"""
Tests for broker integrations
"""

import unittest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from trading.brokers.base import BrokerType, OrderSide, OrderType, TimeInForce
from trading.brokers.factory import BrokerFactory
from trading.brokers.simulation_broker import SimulationBroker
from trading.brokers.multi_broker_manager import MultiBrokerManager
from trading.brokers.auth_manager import BrokerAuthManager

class TestBrokerBase(unittest.TestCase):
    """Test base broker functionality"""
    
    def setUp(self):
        self.config = {
            'paper_trading': True,
            'initial_balance': 100000.0,
            'commission': 1.0,
            'max_position_size': 0.05
        }
    
    def test_broker_types(self):
        """Test broker type enumeration"""
        broker_types = [bt.value for bt in BrokerType]
        expected = ['alpaca', 'ibkr', 'td_ameritrade', 'etrade', 'schwab', 'simulation']
        
        for expected_type in expected:
            self.assertIn(expected_type, broker_types)

class TestSimulationBroker(unittest.TestCase):
    """Test simulation broker"""
    
    def setUp(self):
        self.config = {
            'paper_trading': True,
            'initial_balance': 100000.0,
            'commission': 1.0
        }
        self.broker = SimulationBroker(self.config)
    
    def test_connection(self):
        """Test broker connection"""
        self.assertTrue(self.broker.connect())
        self.assertTrue(self.broker.is_connected())
        
        self.assertTrue(self.broker.disconnect())
        self.assertFalse(self.broker.is_connected())
    
    def test_account_info(self):
        """Test getting account information"""
        self.broker.connect()
        
        account_info = self.broker.get_account_info()
        self.assertIsNotNone(account_info)
        self.assertEqual(account_info.broker_type, BrokerType.SIMULATION)
        self.assertEqual(account_info.cash, 100000.0)
        self.assertEqual(account_info.portfolio_value, 100000.0)
    
    def test_order_submission(self):
        """Test order submission"""
        self.broker.connect()
        
        # Submit a buy order
        order = self.broker.submit_order(
            symbol='AAPL',
            quantity=10,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )
        
        self.assertIsNotNone(order)
        self.assertEqual(order.symbol, 'AAPL')
        self.assertEqual(order.quantity, 10.0)
        self.assertEqual(order.side, OrderSide.BUY)
    
    def test_position_tracking(self):
        """Test position tracking after orders"""
        self.broker.connect()
        
        # Submit a buy order
        buy_order = self.broker.submit_order(
            symbol='AAPL',
            quantity=10,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )
        
        self.assertIsNotNone(buy_order)
        
        # Check positions
        positions = self.broker.get_positions()
        self.assertEqual(len(positions), 1)
        
        position = positions[0]
        self.assertEqual(position.symbol, 'AAPL')
        self.assertEqual(position.quantity, 10.0)
        self.assertEqual(position.side, 'long')
        
        # Submit a sell order
        sell_order = self.broker.submit_order(
            symbol='AAPL',
            quantity=5,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET
        )
        
        self.assertIsNotNone(sell_order)
        
        # Check updated positions
        positions = self.broker.get_positions()
        self.assertEqual(len(positions), 1)
        
        position = positions[0]
        self.assertEqual(position.quantity, 5.0)  # 10 - 5 = 5
    
    def test_order_history(self):
        """Test order history tracking"""
        self.broker.connect()
        
        # Submit multiple orders
        for i in range(3):
            self.broker.submit_order(
                symbol='TSLA',
                quantity=5,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET
            )
        
        # Get order history
        orders = self.broker.get_orders()
        self.assertEqual(len(orders), 3)
        
        # Check order details
        for order in orders:
            self.assertEqual(order.symbol, 'TSLA')
            self.assertEqual(order.quantity, 5.0)
            self.assertEqual(order.side, OrderSide.BUY)
    
    def test_order_cancellation(self):
        """Test order cancellation"""
        self.broker.connect()
        
        # Submit a limit order (won't execute immediately)
        order = self.broker.submit_order(
            symbol='MSFT',
            quantity=10,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=1.0  # Very low price, won't execute
        )
        
        self.assertIsNotNone(order)
        
        # Cancel the order
        success = self.broker.cancel_order(order.id)
        self.assertTrue(success)
        
        # Check order status
        updated_order = self.broker.get_order_status(order.id)
        self.assertIsNotNone(updated_order)
    
    def test_reset_simulation(self):
        """Test simulation reset"""
        self.broker.connect()
        
        # Make some trades
        self.broker.submit_order('AAPL', 10, OrderSide.BUY)
        self.broker.submit_order('TSLA', 5, OrderSide.BUY)
        
        # Check we have positions and orders
        positions = self.broker.get_positions()
        orders = self.broker.get_orders()
        
        self.assertGreater(len(positions), 0)
        self.assertGreater(len(orders), 0)
        
        # Reset simulation
        self.broker.reset_simulation()
        
        # Check everything is cleared
        positions = self.broker.get_positions()
        orders = self.broker.get_orders()
        account_info = self.broker.get_account_info()
        
        self.assertEqual(len(positions), 0)
        self.assertEqual(len(orders), 0)
        self.assertEqual(account_info.cash, 100000.0)

class TestBrokerFactory(unittest.TestCase):
    """Test broker factory"""
    
    def test_create_simulation_broker(self):
        """Test creating simulation broker"""
        config = {
            'paper_trading': True,
            'initial_balance': 50000.0
        }
        
        broker = BrokerFactory.create_broker(BrokerType.SIMULATION, config)
        
        self.assertIsNotNone(broker)
        self.assertIsInstance(broker, SimulationBroker)
        self.assertEqual(broker.broker_type, BrokerType.SIMULATION)
    
    def test_get_available_brokers(self):
        """Test getting available brokers"""
        available = BrokerFactory.get_available_brokers()
        
        self.assertIsInstance(available, dict)
        self.assertIn('simulation', available)
        self.assertIn('alpaca', available)
    
    def test_create_from_env(self):
        """Test creating broker from environment"""
        # Set environment variables for simulation
        os.environ['BROKER_TYPE'] = 'simulation'
        os.environ['PAPER_TRADING'] = 'true'
        
        try:
            broker = BrokerFactory.create_from_env()
            self.assertIsNotNone(broker)
            self.assertEqual(broker.broker_type, BrokerType.SIMULATION)
        finally:
            # Clean up environment
            if 'BROKER_TYPE' in os.environ:
                del os.environ['BROKER_TYPE']
            if 'PAPER_TRADING' in os.environ:
                del os.environ['PAPER_TRADING']

class TestMultiBrokerManager(unittest.TestCase):
    """Test multi-broker manager"""
    
    def setUp(self):
        self.manager = MultiBrokerManager()
        
        # Create test brokers
        self.sim1 = SimulationBroker({'initial_balance': 100000, 'commission': 1.0})
        self.sim2 = SimulationBroker({'initial_balance': 50000, 'commission': 0.5})
    
    def test_add_broker(self):
        """Test adding brokers"""
        success1 = self.manager.add_broker('sim1', self.sim1, is_default=True)
        success2 = self.manager.add_broker('sim2', self.sim2, weight=0.5)
        
        self.assertTrue(success1)
        self.assertTrue(success2)
        
        # Check default broker
        default_broker = self.manager.get_broker()
        self.assertEqual(default_broker, self.sim1)
        
        # Check specific broker
        sim2_broker = self.manager.get_broker('sim2')
        self.assertEqual(sim2_broker, self.sim2)
    
    def test_connect_all(self):
        """Test connecting all brokers"""
        self.manager.add_broker('sim1', self.sim1)
        self.manager.add_broker('sim2', self.sim2)
        
        results = self.manager.connect_all()
        
        self.assertEqual(len(results), 2)
        self.assertTrue(results['sim1'])
        self.assertTrue(results['sim2'])
    
    def test_get_status(self):
        """Test getting broker status"""
        self.manager.add_broker('sim1', self.sim1)
        self.manager.connect_all()
        
        status = self.manager.get_broker_status()
        
        self.assertIn('sim1', status)
        self.assertEqual(status['sim1']['broker_type'], 'simulation')
        self.assertTrue(status['sim1']['is_connected'])
    
    def test_consolidated_positions(self):
        """Test consolidated position tracking"""
        self.manager.add_broker('sim1', self.sim1)
        self.manager.add_broker('sim2', self.sim2)
        self.manager.connect_all()
        
        # Make trades on both brokers
        self.sim1.submit_order('AAPL', 10, OrderSide.BUY)
        self.sim2.submit_order('AAPL', 5, OrderSide.BUY)
        
        # Get consolidated positions
        consolidated = self.manager.get_consolidated_positions()
        
        self.assertIn('AAPL', consolidated)
        aapl_position = consolidated['AAPL']
        self.assertEqual(aapl_position.quantity, 15.0)  # 10 + 5

class TestAuthManager(unittest.TestCase):
    """Test authentication manager"""
    
    def setUp(self):
        # Use temporary directory for tests
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.auth_manager = BrokerAuthManager(self.temp_dir)
    
    def test_credential_validation(self):
        """Test credential validation"""
        # Valid Alpaca credentials
        alpaca_creds = {
            'api_key': 'test_key_123',
            'secret_key': 'test_secret_456'
        }
        
        is_valid, message = self.auth_manager.validate_credentials('alpaca', alpaca_creds)
        self.assertTrue(is_valid)
        
        # Invalid credentials (missing secret)
        invalid_creds = {
            'api_key': 'test_key_123'
        }
        
        is_valid, message = self.auth_manager.validate_credentials('alpaca', invalid_creds)
        self.assertFalse(is_valid)
    
    def test_set_and_get_credentials(self):
        """Test setting and getting credentials"""
        test_creds = {
            'api_key': 'test_api_key',
            'secret_key': 'test_secret_key'
        }
        
        # Set credentials
        self.auth_manager.set_credentials('test_broker', test_creds)
        
        # Get credentials
        retrieved_creds = self.auth_manager.get_credentials('test_broker')
        
        self.assertEqual(retrieved_creds['api_key'], 'test_api_key')
        self.assertEqual(retrieved_creds['secret_key'], 'test_secret_key')
    
    def tearDown(self):
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

def run_tests():
    """Run all tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBrokerBase,
        TestSimulationBroker,
        TestBrokerFactory,
        TestMultiBrokerManager,
        TestAuthManager
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)