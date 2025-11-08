"""
TD Ameritrade broker integration
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from urllib.parse import quote, urlencode
import webbrowser
import time

from .base import (
    BaseBroker, BrokerType, Position, AccountInfo, Order, Trade,
    OrderSide, OrderType, OrderStatus, TimeInForce
)

logger = logging.getLogger(__name__)

class TDAmeritradeBroker(BaseBroker):
    """TD Ameritrade broker integration using their API"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.client_id = config.get('client_id')  # Your app's consumer key
        self.redirect_uri = config.get('redirect_uri', 'https://localhost')
        self.refresh_token = config.get('refresh_token')
        self.access_token = None
        self.account_id = config.get('account_id')
        
        self.base_url = 'https://api.tdameritrade.com/v1'
        self.token_endpoint = 'https://api.tdameritrade.com/v1/oauth2/token'
        
        # Session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
    
    def _get_broker_type(self) -> BrokerType:
        return BrokerType.TD_AMERITRADE
    
    def connect(self) -> bool:
        """Connect to TD Ameritrade API"""
        if not self.client_id:
            self.logger.error("TD Ameritrade client_id (consumer key) not provided")
            return False
        
        try:
            # Try to get access token using refresh token if available
            if self.refresh_token:
                if self._refresh_access_token():
                    self._is_connected = True
                    self.logger.info("Connected to TD Ameritrade API using refresh token")
                    return True
            
            # If no refresh token or refresh failed, need manual authentication
            self.logger.warning("TD Ameritrade requires manual authentication flow")
            self.logger.info("Call authenticate_manually() to start OAuth flow")
            return False
            
        except Exception as e:
            self.logger.error(f"Error connecting to TD Ameritrade: {str(e)}")
            return False
    
    def authenticate_manually(self) -> str:
        """Start manual OAuth authentication flow"""
        try:
            # Step 1: Generate authorization URL
            auth_url = (
                f"https://auth.tdameritrade.com/auth?"
                f"response_type=code&"
                f"redirect_uri={quote(self.redirect_uri)}&"
                f"client_id={quote(self.client_id)}%40AMER.OAUTHAP"
            )
            
            self.logger.info("Opening TD Ameritrade authorization page...")
            self.logger.info(f"Authorization URL: {auth_url}")
            
            # Try to open browser automatically
            try:
                webbrowser.open(auth_url)
            except:
                pass
            
            return auth_url
            
        except Exception as e:
            self.logger.error(f"Error starting authentication: {str(e)}")
            return ""
    
    def complete_authentication(self, authorization_code: str) -> bool:
        """Complete authentication with authorization code"""
        try:
            # Exchange authorization code for refresh token
            token_data = {
                'grant_type': 'authorization_code',
                'refresh_token': '',
                'access_type': 'offline',
                'code': authorization_code,
                'client_id': self.client_id,
                'redirect_uri': self.redirect_uri
            }
            
            response = requests.post(
                self.token_endpoint,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                data=urlencode(token_data)
            )
            
            if response.status_code == 200:
                token_response = response.json()
                self.access_token = token_response.get('access_token')
                self.refresh_token = token_response.get('refresh_token')
                
                self.logger.info("Successfully authenticated with TD Ameritrade")
                self.logger.info(f"Save this refresh token: {self.refresh_token}")
                
                self._is_connected = True
                return True
            else:
                self.logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error completing authentication: {str(e)}")
            return False
    
    def _refresh_access_token(self) -> bool:
        """Refresh access token using refresh token"""
        try:
            token_data = {
                'grant_type': 'refresh_token',
                'refresh_token': self.refresh_token,
                'client_id': self.client_id
            }
            
            response = requests.post(
                self.token_endpoint,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                data=urlencode(token_data)
            )
            
            if response.status_code == 200:
                token_response = response.json()
                self.access_token = token_response.get('access_token')
                
                # Update session headers
                self.session.headers.update({
                    'Authorization': f'Bearer {self.access_token}'
                })
                
                return True
            else:
                self.logger.error(f"Token refresh failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error refreshing token: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from TD Ameritrade API"""
        self.access_token = None
        self._is_connected = False
        return True
    
    def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information"""
        if not self.is_connected():
            return None
        
        try:
            if not self.account_id:
                # Get account IDs first
                accounts = self._get_accounts()
                if accounts and len(accounts) > 0:
                    self.account_id = accounts[0]['securitiesAccount']['accountId']
                else:
                    self.logger.error("No accounts found")
                    return None
            
            url = f"{self.base_url}/accounts/{self.account_id}"
            response = self.session.get(url)
            
            if response.status_code == 200:
                account_data = response.json()
                account = account_data['securitiesAccount']
                
                return AccountInfo(
                    account_id=account['accountId'],
                    broker_type=self.broker_type,
                    status='ACTIVE',  # TD Ameritrade doesn't provide explicit status
                    currency='USD',
                    buying_power=float(account['currentBalances'].get('buyingPower', 0)),
                    cash=float(account['currentBalances'].get('cashBalance', 0)),
                    portfolio_value=float(account['currentBalances'].get('liquidationValue', 0)),
                    equity=float(account['currentBalances'].get('equity', 0)),
                    last_equity=float(account['currentBalances'].get('equity', 0)),
                    day_trade_buying_power=float(account['currentBalances'].get('dayTradingBuyingPower', 0)),
                    pattern_day_trader=account.get('isDayTrader', False),
                    additional_info={
                        'account_type': account.get('type', ''),
                        'long_market_value': float(account['currentBalances'].get('longMarketValue', 0)),
                        'short_market_value': float(account['currentBalances'].get('shortMarketValue', 0)),
                        'money_market_fund': float(account['currentBalances'].get('moneyMarketFund', 0))
                    }
                )
            else:
                self.logger.error(f"Error getting account info: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            return None
    
    def get_positions(self) -> List[Position]:
        """Get current positions"""
        if not self.is_connected():
            return []
        
        try:
            url = f"{self.base_url}/accounts/{self.account_id}"
            params = {'fields': 'positions'}
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                account_data = response.json()
                positions_data = account_data['securitiesAccount'].get('positions', [])
                
                position_list = []
                for pos in positions_data:
                    if pos['longQuantity'] != 0 or pos['shortQuantity'] != 0:
                        quantity = pos['longQuantity'] - pos['shortQuantity']
                        
                        position_list.append(Position(
                            symbol=pos['instrument']['symbol'],
                            quantity=float(quantity),
                            side='long' if quantity > 0 else 'short',
                            avg_entry_price=float(pos['averagePrice']),
                            current_price=float(pos['marketValue'] / abs(quantity)) if quantity != 0 else 0,
                            market_value=float(pos['marketValue']),
                            unrealized_pnl=float(pos.get('currentDayProfitLoss', 0)),
                            unrealized_pnl_pct=float(pos.get('currentDayProfitLossPercentage', 0)),
                            cost_basis=float(abs(quantity * pos['averagePrice'])),
                            sector=pos['instrument'].get('assetType', 'Unknown')
                        ))
                
                return position_list
            else:
                self.logger.error(f"Error getting positions: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def submit_order(self, symbol: str, quantity: float, side: OrderSide,
                    order_type: OrderType = OrderType.MARKET,
                    time_in_force: TimeInForce = TimeInForce.DAY,
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    **kwargs) -> Optional[Order]:
        """Submit an order"""
        if not self.is_connected():
            return None
        
        # Validate order first
        current_price = self.get_current_price(symbol) if side == OrderSide.BUY else None
        is_valid, message = self.validate_order(symbol, quantity, side, current_price)
        
        if not is_valid:
            self.logger.warning(f"Order validation failed: {message}")
            return None
        
        try:
            # Build order payload
            order_data = {
                "orderType": self._map_to_td_order_type(order_type),
                "session": "NORMAL",
                "duration": self._map_to_td_duration(time_in_force),
                "orderStrategyType": "SINGLE",
                "orderLegCollection": [
                    {
                        "instruction": "BUY" if side == OrderSide.BUY else "SELL",
                        "quantity": int(quantity),
                        "instrument": {
                            "symbol": symbol,
                            "assetType": "EQUITY"
                        }
                    }
                ]
            }
            
            # Add price if limit order
            if order_type == OrderType.LIMIT and limit_price:
                order_data["price"] = limit_price
            elif order_type == OrderType.STOP and stop_price:
                order_data["stopPrice"] = stop_price
            
            url = f"{self.base_url}/accounts/{self.account_id}/orders"
            response = self.session.post(url, json=order_data)
            
            if response.status_code == 201:
                # Get order ID from Location header
                location = response.headers.get('Location', '')
                order_id = location.split('/')[-1] if location else str(int(time.time()))
                
                order = Order(
                    id=f"TDA_{order_id}",
                    symbol=symbol,
                    quantity=float(quantity),
                    filled_quantity=0.0,
                    side=side,
                    order_type=order_type,
                    status=OrderStatus.NEW,
                    time_in_force=time_in_force,
                    limit_price=limit_price,
                    stop_price=stop_price,
                    submitted_at=datetime.now(),
                    broker_order_id=order_id
                )
                
                self.logger.info(f"Order submitted: {side.value.upper()} {quantity} {symbol} - Order ID: {order_id}")
                
                return order
            else:
                self.logger.error(f"Error submitting order: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error submitting order: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.is_connected():
            return False
        
        try:
            # Extract TD order ID
            td_order_id = order_id.replace('TDA_', '')
            
            url = f"{self.base_url}/accounts/{self.account_id}/orders/{td_order_id}"
            response = self.session.delete(url)
            
            if response.status_code == 200:
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                self.logger.error(f"Error cancelling order: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False
    
    def get_orders(self, status: Optional[str] = None, limit: int = 50) -> List[Order]:
        """Get orders"""
        if not self.is_connected():
            return []
        
        try:
            url = f"{self.base_url}/accounts/{self.account_id}/orders"
            params = {
                'maxResults': limit,
                'fromEnteredTime': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            }
            
            if status:
                params['status'] = status.upper()
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                orders_data = response.json()
                
                order_list = []
                for order_data in orders_data:
                    if order_data.get('orderLegCollection'):
                        leg = order_data['orderLegCollection'][0]
                        
                        order = Order(
                            id=f"TDA_{order_data['orderId']}",
                            symbol=leg['instrument']['symbol'],
                            quantity=float(leg['quantity']),
                            filled_quantity=float(order_data.get('filledQuantity', 0)),
                            side=OrderSide.BUY if leg['instruction'] == 'BUY' else OrderSide.SELL,
                            order_type=self._map_from_td_order_type(order_data['orderType']),
                            status=self._map_td_status(order_data['status']),
                            time_in_force=self._map_from_td_duration(order_data['duration']),
                            limit_price=float(order_data['price']) if order_data.get('price') else None,
                            stop_price=float(order_data['stopPrice']) if order_data.get('stopPrice') else None,
                            submitted_at=datetime.fromisoformat(order_data['enteredTime'].replace('Z', '+00:00')),
                            avg_fill_price=float(order_data.get('price', 0)) if order_data.get('filledQuantity', 0) > 0 else None,
                            broker_order_id=str(order_data['orderId'])
                        )
                        
                        order_list.append(order)
                
                return order_list
            else:
                self.logger.error(f"Error getting orders: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting orders: {str(e)}")
            return []
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of a specific order"""
        if not self.is_connected():
            return None
        
        try:
            td_order_id = order_id.replace('TDA_', '')
            
            url = f"{self.base_url}/accounts/{self.account_id}/orders/{td_order_id}"
            response = self.session.get(url)
            
            if response.status_code == 200:
                order_data = response.json()
                
                if order_data.get('orderLegCollection'):
                    leg = order_data['orderLegCollection'][0]
                    
                    return Order(
                        id=f"TDA_{order_data['orderId']}",
                        symbol=leg['instrument']['symbol'],
                        quantity=float(leg['quantity']),
                        filled_quantity=float(order_data.get('filledQuantity', 0)),
                        side=OrderSide.BUY if leg['instruction'] == 'BUY' else OrderSide.SELL,
                        order_type=self._map_from_td_order_type(order_data['orderType']),
                        status=self._map_td_status(order_data['status']),
                        time_in_force=self._map_from_td_duration(order_data['duration']),
                        limit_price=float(order_data['price']) if order_data.get('price') else None,
                        stop_price=float(order_data['stopPrice']) if order_data.get('stopPrice') else None,
                        submitted_at=datetime.fromisoformat(order_data['enteredTime'].replace('Z', '+00:00')),
                        avg_fill_price=float(order_data.get('price', 0)) if order_data.get('filledQuantity', 0) > 0 else None,
                        broker_order_id=str(order_data['orderId'])
                    )
                
                return None
            else:
                self.logger.error(f"Error getting order status: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {str(e)}")
            return None
    
    def get_portfolio_history(self, period: str = '1M') -> Dict:
        """Get portfolio history - simplified for TD Ameritrade"""
        if not self.is_connected():
            return {}
        
        try:
            account_info = self.get_account_info()
            
            if account_info:
                return {
                    'current_value': account_info.portfolio_value,
                    'cash': account_info.cash,
                    'equity': account_info.equity,
                    'timestamp': datetime.now().isoformat(),
                    'note': 'TD Ameritrade historical data requires additional implementation'
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio history: {str(e)}")
            return {}
    
    def _get_accounts(self) -> List[Dict]:
        """Get list of accounts"""
        try:
            url = f"{self.base_url}/accounts"
            response = self.session.get(url)
            
            if response.status_code == 200:
                return response.json()
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting accounts: {str(e)}")
            return []
    
    def _map_to_td_order_type(self, order_type: OrderType) -> str:
        """Map standard order type to TD Ameritrade format"""
        mapping = {
            OrderType.MARKET: 'MARKET',
            OrderType.LIMIT: 'LIMIT',
            OrderType.STOP: 'STOP',
            OrderType.STOP_LIMIT: 'STOP_LIMIT'
        }
        return mapping.get(order_type, 'MARKET')
    
    def _map_from_td_order_type(self, td_type: str) -> OrderType:
        """Map TD Ameritrade order type to standard format"""
        mapping = {
            'MARKET': OrderType.MARKET,
            'LIMIT': OrderType.LIMIT,
            'STOP': OrderType.STOP,
            'STOP_LIMIT': OrderType.STOP_LIMIT
        }
        return mapping.get(td_type, OrderType.MARKET)
    
    def _map_to_td_duration(self, time_in_force: TimeInForce) -> str:
        """Map standard TIF to TD Ameritrade format"""
        mapping = {
            TimeInForce.DAY: 'DAY',
            TimeInForce.GTC: 'GTC',
            TimeInForce.IOC: 'IOC',
            TimeInForce.FOK: 'FOK'
        }
        return mapping.get(time_in_force, 'DAY')
    
    def _map_from_td_duration(self, td_duration: str) -> TimeInForce:
        """Map TD Ameritrade duration to standard TIF"""
        mapping = {
            'DAY': TimeInForce.DAY,
            'GTC': TimeInForce.GTC,
            'IOC': TimeInForce.IOC,
            'FOK': TimeInForce.FOK
        }
        return mapping.get(td_duration, TimeInForce.DAY)
    
    def _map_td_status(self, td_status: str) -> OrderStatus:
        """Map TD Ameritrade order status to standard status"""
        mapping = {
            'AWAITING_PARENT_ORDER': OrderStatus.PENDING,
            'AWAITING_CONDITION': OrderStatus.PENDING,
            'AWAITING_MANUAL_REVIEW': OrderStatus.PENDING,
            'ACCEPTED': OrderStatus.ACCEPTED,
            'AWAITING_UR_OUT': OrderStatus.PENDING,
            'PENDING_ACTIVATION': OrderStatus.PENDING,
            'QUEUED': OrderStatus.NEW,
            'WORKING': OrderStatus.NEW,
            'REJECTED': OrderStatus.REJECTED,
            'PENDING_CANCEL': OrderStatus.PENDING,
            'CANCELED': OrderStatus.CANCELED,
            'PENDING_REPLACE': OrderStatus.PENDING,
            'REPLACED': OrderStatus.CANCELED,
            'FILLED': OrderStatus.FILLED,
            'EXPIRED': OrderStatus.EXPIRED
        }
        return mapping.get(td_status, OrderStatus.PENDING)