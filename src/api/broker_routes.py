"""
FastAPI routes for broker management
"""

from fastapi import APIRouter, Request, HTTPException, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.trading.brokers.factory import BrokerFactory, BrokerConfigManager
from src.trading.brokers.multi_broker_manager import MultiBrokerManager
from src.trading.brokers.auth_manager import BrokerAuthManager
from src.trading.brokers.base import BrokerType, OrderSide, OrderType, TimeInForce
from src.auth import get_current_user

logger = logging.getLogger(__name__)

# Initialize managers
config_dir = Path.home() / ".invest"
auth_manager = BrokerAuthManager(str(config_dir))
config_manager = BrokerConfigManager(str(config_dir / "brokers.json"))
multi_broker = MultiBrokerManager()

templates = Jinja2Templates(directory="templates")
router = APIRouter()

class BrokerCredentials(BaseModel):
    broker_name: str
    broker_type: str
    credentials: Dict[str, Any]
    paper_trading: bool = True
    max_position_size: float = 0.05
    stop_loss_pct: float = 0.02

class OrderRequest(BaseModel):
    symbol: str
    quantity: float
    side: str
    order_type: str = "market"
    limit_price: Optional[float] = None
    broker_name: Optional[str] = None

@router.get("/brokers", response_class=HTMLResponse)
async def brokers_page(request: Request, user = Depends(get_current_user)):
    """Broker management page"""
    try:
        # Get broker status
        configured_brokers = config_manager.list_brokers()
        broker_types = BrokerFactory.get_available_brokers()
        
        # Load brokers into multi_broker manager
        for name, broker_type_str in configured_brokers.items():
            try:
                broker_config = config_manager.get_broker_config(name)
                if broker_config:
                    broker_type = BrokerType(broker_type_str)
                    
                    # Merge credentials
                    auth_creds = auth_manager.get_credentials(name)
                    if auth_creds:
                        broker_config.update(auth_creds)
                    
                    broker = BrokerFactory.create_broker(broker_type, broker_config)
                    if broker and name not in [acc.name for acc in multi_broker.accounts.values()]:
                        is_default = name == config_manager.config.get('default_broker')
                        multi_broker.add_broker(name, broker, is_default=is_default)
            except Exception as e:
                logger.error(f"Error loading broker {name}: {str(e)}")
        
        # Get status of all brokers
        connection_results = multi_broker.connect_all()
        broker_status = multi_broker.get_broker_status()
        
        return templates.TemplateResponse("brokers/index.html", {
            "request": request,
            "user": user,
            "configured_brokers": configured_brokers,
            "broker_types": broker_types,
            "broker_status": broker_status,
            "connection_results": connection_results
        })
        
    except Exception as e:
        logger.error(f"Error loading brokers page: {str(e)}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@router.get("/brokers/add", response_class=HTMLResponse)
async def add_broker_page(request: Request, user = Depends(get_current_user)):
    """Add broker page"""
    broker_types = list(BrokerType)
    return templates.TemplateResponse("brokers/add.html", {
        "request": request,
        "user": user,
        "broker_types": broker_types
    })

@router.post("/brokers/add")
async def add_broker(
    request: Request,
    broker_name: str = Form(...),
    broker_type: str = Form(...),
    paper_trading: bool = Form(False),
    max_position_size: float = Form(0.05),
    stop_loss_pct: float = Form(0.02),
    # Alpaca fields
    alpaca_api_key: Optional[str] = Form(None),
    alpaca_secret_key: Optional[str] = Form(None),
    # IBKR fields
    ibkr_host: Optional[str] = Form("127.0.0.1"),
    ibkr_port: Optional[int] = Form(7497),
    ibkr_client_id: Optional[int] = Form(1),
    # TD Ameritrade fields
    td_client_id: Optional[str] = Form(None),
    td_refresh_token: Optional[str] = Form(None),
    td_account_id: Optional[str] = Form(None),
    # Simulation fields
    sim_initial_balance: Optional[float] = Form(100000),
    sim_commission: Optional[float] = Form(0),
    user = Depends(get_current_user)
):
    """Add new broker configuration"""
    try:
        # Build configuration
        config = {
            'paper_trading': paper_trading,
            'max_position_size': max_position_size,
            'stop_loss_pct': stop_loss_pct
        }
        
        # Add broker-specific configuration
        if broker_type == 'alpaca':
            if alpaca_api_key and alpaca_secret_key:
                config.update({
                    'api_key': alpaca_api_key,
                    'secret_key': alpaca_secret_key
                })
            else:
                raise HTTPException(status_code=400, detail="Alpaca API key and secret key are required")
                
        elif broker_type == 'ibkr':
            config.update({
                'host': ibkr_host,
                'port': ibkr_port,
                'client_id': ibkr_client_id
            })
            
        elif broker_type == 'td_ameritrade':
            if td_client_id:
                config.update({
                    'client_id': td_client_id,
                    'refresh_token': td_refresh_token,
                    'account_id': td_account_id
                })
            else:
                raise HTTPException(status_code=400, detail="TD Ameritrade client ID is required")
                
        elif broker_type == 'simulation':
            config.update({
                'initial_balance': sim_initial_balance,
                'commission': sim_commission
            })
        
        # Save configuration
        broker_type_enum = BrokerType(broker_type)
        config_manager.add_broker(broker_name, broker_type_enum, config)
        
        # Store sensitive credentials separately
        sensitive_creds = {}
        if broker_type == 'alpaca' and alpaca_api_key and alpaca_secret_key:
            sensitive_creds = {
                'api_key': alpaca_api_key,
                'secret_key': alpaca_secret_key
            }
        elif broker_type == 'td_ameritrade' and td_refresh_token:
            sensitive_creds = {
                'refresh_token': td_refresh_token
            }
        
        if sensitive_creds:
            auth_manager.set_credentials(broker_name, sensitive_creds)
        
        return RedirectResponse(url="/brokers?success=Broker added successfully", status_code=302)
        
    except Exception as e:
        logger.error(f"Error adding broker: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/brokers/{broker_name}/test")
async def test_broker_connection(broker_name: str, user = Depends(get_current_user)):
    """Test broker connection"""
    try:
        # Get broker configuration
        broker_config = config_manager.get_broker_config(broker_name)
        if not broker_config:
            raise HTTPException(status_code=404, detail="Broker not found")
        
        broker_type = BrokerType(broker_config['type'])
        
        # Merge with credentials
        auth_creds = auth_manager.get_credentials(broker_name)
        if auth_creds:
            broker_config.update(auth_creds)
        
        # Create and test broker
        broker = BrokerFactory.create_broker(broker_type, broker_config)
        if not broker:
            raise HTTPException(status_code=500, detail="Failed to create broker instance")
        
        # Test connection
        connection_success = broker.connect()
        
        if connection_success:
            # Get account info
            account_info = broker.get_account_info()
            positions = broker.get_positions()
            
            # Disconnect
            broker.disconnect()
            
            return {
                "success": True,
                "message": "Connection successful",
                "account_info": {
                    "account_id": account_info.account_id if account_info else None,
                    "portfolio_value": account_info.portfolio_value if account_info else 0,
                    "cash": account_info.cash if account_info else 0,
                    "buying_power": account_info.buying_power if account_info else 0
                } if account_info else None,
                "positions_count": len(positions) if positions else 0
            }
        else:
            return {
                "success": False,
                "message": "Connection failed"
            }
            
    except Exception as e:
        logger.error(f"Error testing broker {broker_name}: {str(e)}")
        return {
            "success": False,
            "message": str(e)
        }

@router.delete("/brokers/{broker_name}")
async def delete_broker(broker_name: str, user = Depends(get_current_user)):
    """Delete broker configuration"""
    try:
        # Remove from config
        if broker_name in config_manager.config.get('brokers', {}):
            del config_manager.config['brokers'][broker_name]
            
            # Update default if needed
            if config_manager.config.get('default_broker') == broker_name:
                remaining_brokers = list(config_manager.config.get('brokers', {}).keys())
                config_manager.config['default_broker'] = remaining_brokers[0] if remaining_brokers else None
            
            config_manager.save_config(config_manager.config)
            
            # Remove credentials
            auth_manager.remove_credentials(broker_name)
            
            # Remove from multi_broker manager
            multi_broker.remove_broker(broker_name)
            
            return {"success": True, "message": f"Broker {broker_name} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Broker not found")
            
    except Exception as e:
        logger.error(f"Error deleting broker {broker_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/brokers/portfolio", response_class=HTMLResponse)
async def multi_broker_portfolio(request: Request, user = Depends(get_current_user)):
    """Multi-broker portfolio view"""
    try:
        # Load all configured brokers
        configured_brokers = config_manager.list_brokers()
        
        for name, broker_type_str in configured_brokers.items():
            try:
                broker_config = config_manager.get_broker_config(name)
                if broker_config:
                    broker_type = BrokerType(broker_type_str)
                    
                    # Merge credentials
                    auth_creds = auth_manager.get_credentials(name)
                    if auth_creds:
                        broker_config.update(auth_creds)
                    
                    broker = BrokerFactory.create_broker(broker_type, broker_config)
                    if broker and name not in [acc.name for acc in multi_broker.accounts.values()]:
                        is_default = name == config_manager.config.get('default_broker')
                        multi_broker.add_broker(name, broker, is_default=is_default)
            except Exception as e:
                logger.error(f"Error loading broker {name}: {str(e)}")
        
        # Connect to brokers
        multi_broker.connect_all()
        
        # Get consolidated data
        all_positions = multi_broker.get_all_positions()
        consolidated_positions = multi_broker.get_consolidated_positions()
        total_value = multi_broker.get_total_account_value()
        broker_status = multi_broker.get_broker_status()
        
        return templates.TemplateResponse("brokers/portfolio.html", {
            "request": request,
            "user": user,
            "all_positions": all_positions,
            "consolidated_positions": consolidated_positions,
            "total_value": total_value,
            "broker_status": broker_status
        })
        
    except Exception as e:
        logger.error(f"Error loading portfolio: {str(e)}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@router.post("/brokers/order")
async def submit_order(order_request: OrderRequest, user = Depends(get_current_user)):
    """Submit order through broker"""
    try:
        side = OrderSide.BUY if order_request.side.upper() == 'BUY' else OrderSide.SELL
        order_type = OrderType.LIMIT if order_request.order_type.upper() == 'LIMIT' else OrderType.MARKET
        
        if order_request.broker_name:
            # Submit to specific broker
            broker = multi_broker.get_broker(order_request.broker_name)
            if not broker:
                raise HTTPException(status_code=404, detail="Broker not found")
            
            if not broker.is_connected():
                broker.connect()
            
            order = broker.submit_order(
                symbol=order_request.symbol,
                quantity=order_request.quantity,
                side=side,
                order_type=order_type,
                limit_price=order_request.limit_price
            )
            
            return {
                "success": True,
                "order_id": order.id if order else None,
                "broker": order_request.broker_name
            }
        else:
            # Distribute across brokers
            results = multi_broker.submit_order_distributed(
                symbol=order_request.symbol,
                total_quantity=order_request.quantity,
                side=side,
                order_type=order_type
            )
            
            return {
                "success": True,
                "results": {name: order.id if order else None for name, order in results.items()}
            }
            
    except Exception as e:
        logger.error(f"Error submitting order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/brokers/status")
async def api_broker_status(user = Depends(get_current_user)):
    """API endpoint for broker status"""
    try:
        # Load brokers if not already loaded
        configured_brokers = config_manager.list_brokers()
        
        for name, broker_type_str in configured_brokers.items():
            try:
                if name not in [acc.name for acc in multi_broker.accounts.values()]:
                    broker_config = config_manager.get_broker_config(name)
                    if broker_config:
                        broker_type = BrokerType(broker_type_str)
                        
                        # Merge credentials
                        auth_creds = auth_manager.get_credentials(name)
                        if auth_creds:
                            broker_config.update(auth_creds)
                        
                        broker = BrokerFactory.create_broker(broker_type, broker_config)
                        if broker:
                            is_default = name == config_manager.config.get('default_broker')
                            multi_broker.add_broker(name, broker, is_default=is_default)
            except Exception as e:
                logger.error(f"Error loading broker {name}: {str(e)}")
        
        # Get status
        broker_status = multi_broker.get_broker_status()
        total_value = multi_broker.get_total_account_value()
        
        return {
            "brokers": broker_status,
            "total_portfolio_value": total_value,
            "connected_count": sum(1 for status in broker_status.values() if status['is_connected']),
            "total_count": len(broker_status)
        }
        
    except Exception as e:
        logger.error(f"Error getting broker status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))