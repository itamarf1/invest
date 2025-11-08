import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from enum import Enum

from src.trading.paper_trader import PaperTrader, OrderSide
from src.data.market import MarketDataFetcher

logger = logging.getLogger(__name__)

@dataclass
class Position:
    symbol: str
    quantity: int
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    cost_basis: float
    side: str  # 'long' or 'short'
    entry_date: datetime
    last_updated: datetime

@dataclass
class Trade:
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    price: float
    timestamp: datetime
    order_type: str
    commission: float = 0.0
    pnl: Optional[float] = None

@dataclass
class PerformanceMetrics:
    total_return: float
    total_return_pct: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

class PortfolioManager:
    """Manages portfolio positions, tracks P&L, and calculates performance metrics"""
    
    def __init__(self, paper_trader: PaperTrader):
        self.paper_trader = paper_trader
        self.market_fetcher = MarketDataFetcher()
        
        # Performance tracking
        self.trades_history: List[Trade] = []
        self.daily_values: List[Tuple[datetime, float]] = []
        self.benchmark_symbol = 'SPY'  # S&P 500 as benchmark
        
    def get_current_positions(self) -> List[Position]:
        """Get current portfolio positions with updated prices"""
        alpaca_positions = self.paper_trader.get_positions()
        positions = []
        
        for pos_data in alpaca_positions:
            try:
                # Get current market price
                current_price = self.market_fetcher.get_current_price(pos_data['symbol'])
                if not current_price:
                    current_price = pos_data['current_price']
                
                position = Position(
                    symbol=pos_data['symbol'],
                    quantity=int(pos_data['qty']),
                    avg_entry_price=pos_data['avg_entry_price'],
                    current_price=current_price,
                    market_value=pos_data['market_value'],
                    unrealized_pnl=pos_data['unrealized_pl'],
                    unrealized_pnl_pct=pos_data['unrealized_plpc'],
                    cost_basis=pos_data['cost_basis'],
                    side=pos_data['side'],
                    entry_date=datetime.now() - timedelta(days=30),  # Simplified
                    last_updated=datetime.now()
                )
                positions.append(position)
                
            except Exception as e:
                logger.error(f"Error processing position {pos_data['symbol']}: {str(e)}")
        
        return positions
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary with key metrics"""
        account_info = self.paper_trader.get_account_info()
        positions = self.get_current_positions()
        
        if not account_info:
            return {}
        
        # Calculate position summaries
        total_market_value = sum(pos.market_value for pos in positions)
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
        total_cost_basis = sum(pos.cost_basis for pos in positions)
        
        # Calculate portfolio allocation
        allocations = []
        for position in positions:
            allocation_pct = (position.market_value / total_market_value * 100) if total_market_value > 0 else 0
            allocations.append({
                'symbol': position.symbol,
                'allocation_pct': round(allocation_pct, 2),
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': position.unrealized_pnl_pct
            })
        
        # Sort by allocation percentage
        allocations.sort(key=lambda x: x['allocation_pct'], reverse=True)
        
        return {
            'account_value': account_info.get('portfolio_value', 0),
            'cash_balance': account_info.get('cash', 0),
            'buying_power': account_info.get('buying_power', 0),
            'total_positions': len(positions),
            'total_market_value': total_market_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_cost_basis': total_cost_basis,
            'day_change': account_info.get('portfolio_value', 0) - account_info.get('last_equity', 0),
            'day_change_pct': ((account_info.get('portfolio_value', 0) - account_info.get('last_equity', 0)) / account_info.get('last_equity', 1)) * 100,
            'allocations': allocations,
            'largest_position': allocations[0] if allocations else None,
            'cash_allocation_pct': (account_info.get('cash', 0) / account_info.get('portfolio_value', 1)) * 100
        }
    
    def calculate_performance_metrics(self, start_date: Optional[datetime] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)  # Default to 1 year
        
        # Get portfolio history
        portfolio_history = self.paper_trader.get_portfolio_history(period='1Y', timeframe='1Day')
        
        if not portfolio_history or not portfolio_history.get('equity'):
            # Fallback to basic metrics if no history available
            return self._calculate_basic_metrics()
        
        equity_values = portfolio_history['equity']
        timestamps = portfolio_history['timestamp']
        
        if len(equity_values) < 2:
            return self._calculate_basic_metrics()
        
        # Convert to returns
        returns = pd.Series(equity_values).pct_change().dropna()
        
        if len(returns) == 0:
            return self._calculate_basic_metrics()
        
        # Calculate metrics
        total_return = (equity_values[-1] - equity_values[0])
        total_return_pct = (total_return / equity_values[0]) * 100
        
        # Annualized return
        days_elapsed = len(equity_values)
        years_elapsed = days_elapsed / 252  # Trading days per year
        annualized_return = ((equity_values[-1] / equity_values[0]) ** (1 / years_elapsed) - 1) * 100 if years_elapsed > 0 else 0
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_return = (annualized_return / 100) - risk_free_rate
        sharpe_ratio = (excess_return / (volatility / 100)) if volatility > 0 else 0
        
        # Maximum drawdown
        equity_series = pd.Series(equity_values)
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        # Trade-based metrics
        trades_metrics = self._calculate_trades_metrics()
        
        return PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=trades_metrics['win_rate'],
            profit_factor=trades_metrics['profit_factor'],
            total_trades=trades_metrics['total_trades'],
            winning_trades=trades_metrics['winning_trades'],
            losing_trades=trades_metrics['losing_trades'],
            avg_win=trades_metrics['avg_win'],
            avg_loss=trades_metrics['avg_loss'],
            largest_win=trades_metrics['largest_win'],
            largest_loss=trades_metrics['largest_loss']
        )
    
    def _calculate_basic_metrics(self) -> PerformanceMetrics:
        """Calculate basic metrics when portfolio history is not available"""
        trades_metrics = self._calculate_trades_metrics()
        
        return PerformanceMetrics(
            total_return=0.0,
            total_return_pct=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=trades_metrics['win_rate'],
            profit_factor=trades_metrics['profit_factor'],
            total_trades=trades_metrics['total_trades'],
            winning_trades=trades_metrics['winning_trades'],
            losing_trades=trades_metrics['losing_trades'],
            avg_win=trades_metrics['avg_win'],
            avg_loss=trades_metrics['avg_loss'],
            largest_win=trades_metrics['largest_win'],
            largest_loss=trades_metrics['largest_loss']
        )
    
    def _calculate_trades_metrics(self) -> Dict:
        """Calculate trade-based performance metrics"""
        if not self.trades_history:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        # Group trades by symbol to calculate P&L
        symbol_trades = {}
        for trade in self.trades_history:
            if trade.symbol not in symbol_trades:
                symbol_trades[trade.symbol] = []
            symbol_trades[trade.symbol].append(trade)
        
        realized_pnls = []
        
        # Calculate realized P&L for each symbol
        for symbol, trades in symbol_trades.items():
            position_qty = 0
            position_cost = 0
            
            for trade in sorted(trades, key=lambda t: t.timestamp):
                if trade.side == 'buy':
                    position_qty += trade.quantity
                    position_cost += trade.quantity * trade.price
                elif trade.side == 'sell' and position_qty > 0:
                    # Calculate P&L for sold quantity
                    sold_qty = min(trade.quantity, position_qty)
                    avg_cost = position_cost / position_qty if position_qty > 0 else 0
                    pnl = sold_qty * (trade.price - avg_cost)
                    realized_pnls.append(pnl)
                    
                    # Update position
                    position_qty -= sold_qty
                    position_cost = position_cost * (position_qty / (position_qty + sold_qty)) if position_qty > 0 else 0
        
        if not realized_pnls:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': len(self.trades_history),
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        # Calculate metrics
        winning_trades = [pnl for pnl in realized_pnls if pnl > 0]
        losing_trades = [pnl for pnl in realized_pnls if pnl < 0]
        
        total_trades = len(realized_pnls)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        
        win_rate = (winning_count / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = sum(winning_trades) / winning_count if winning_count > 0 else 0
        avg_loss = sum(losing_trades) / losing_count if losing_count > 0 else 0
        
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if sum(losing_trades) < 0 else 0
        
        largest_win = max(winning_trades) if winning_trades else 0
        largest_loss = min(losing_trades) if losing_trades else 0
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': winning_count,
            'losing_trades': losing_count,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }
    
    def add_trade(self, order_data: Dict):
        """Add a completed trade to history"""
        try:
            trade = Trade(
                trade_id=order_data['id'],
                symbol=order_data['symbol'],
                side=order_data['side'],
                quantity=int(order_data['filled_qty']),
                price=order_data['avg_fill_price'] or 0.0,
                timestamp=datetime.fromisoformat(order_data['filled_at'].replace('Z', '+00:00')) if order_data['filled_at'] else datetime.now(),
                order_type=order_data['order_type'],
                commission=0.0  # Alpaca has no commissions
            )
            
            self.trades_history.append(trade)
            logger.info(f"Trade added to history: {trade.side.upper()} {trade.quantity} {trade.symbol} @ ${trade.price}")
            
        except Exception as e:
            logger.error(f"Error adding trade to history: {str(e)}")
    
    def get_position_analysis(self, symbol: str) -> Dict:
        """Get detailed analysis for a specific position"""
        positions = self.get_current_positions()
        position = next((p for p in positions if p.symbol == symbol), None)
        
        if not position:
            return {'error': f'No position found for {symbol}'}
        
        # Get historical data for the position
        try:
            stock_data = self.market_fetcher.get_stock_data(symbol, period="1mo")
            if not stock_data.empty:
                # Calculate some basic metrics
                current_price = stock_data['Close'].iloc[-1]
                entry_price = position.avg_entry_price
                
                # Price performance since entry (simplified)
                days_held = 30  # Simplified
                daily_return = (current_price / entry_price) ** (1/days_held) - 1 if days_held > 0 else 0
                
                return {
                    'symbol': symbol,
                    'position': asdict(position),
                    'current_price': current_price,
                    'entry_price': entry_price,
                    'days_held_estimate': days_held,
                    'daily_return_estimate': daily_return * 100,
                    'price_change_since_entry': ((current_price - entry_price) / entry_price) * 100,
                    'risk_reward_ratio': abs(position.unrealized_pnl / position.cost_basis) if position.cost_basis > 0 else 0
                }
        except Exception as e:
            logger.error(f"Error analyzing position {symbol}: {str(e)}")
        
        return {'symbol': symbol, 'position': asdict(position)}
    
    def get_sector_allocation(self) -> Dict:
        """Get portfolio allocation by sector (simplified)"""
        positions = self.get_current_positions()
        
        # Simplified sector mapping
        sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary', 'META': 'Technology', 'NVDA': 'Technology', 'NFLX': 'Communication Services',
            'JPM': 'Financials', 'V': 'Financials', 'JNJ': 'Healthcare', 'PG': 'Consumer Staples',
            'SPY': 'ETF', 'QQQ': 'ETF', 'VTI': 'ETF'
        }
        
        sector_allocation = {}
        total_value = sum(pos.market_value for pos in positions)
        
        for position in positions:
            sector = sector_map.get(position.symbol, 'Other')
            
            if sector not in sector_allocation:
                sector_allocation[sector] = {
                    'allocation_pct': 0,
                    'market_value': 0,
                    'positions': []
                }
            
            sector_allocation[sector]['market_value'] += position.market_value
            sector_allocation[sector]['positions'].append(position.symbol)
        
        # Calculate percentages
        for sector in sector_allocation:
            sector_allocation[sector]['allocation_pct'] = (
                sector_allocation[sector]['market_value'] / total_value * 100
            ) if total_value > 0 else 0
        
        return sector_allocation
    
    def export_trades_csv(self, filename: str = None) -> str:
        """Export trades history to CSV"""
        if not filename:
            filename = f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        if not self.trades_history:
            logger.warning("No trades to export")
            return filename
        
        # Convert trades to DataFrame
        trades_data = [asdict(trade) for trade in self.trades_history]
        df = pd.DataFrame(trades_data)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        logger.info(f"Trades exported to {filename}")
        
        return filename
