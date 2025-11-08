#!/usr/bin/env python3

import argparse
import sys
import os
from datetime import datetime
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.market import MarketDataFetcher
from src.trading.strategies.moving_average import MovingAverageStrategy
from src.trading.strategies.sentiment_enhanced import SentimentEnhancedStrategy, NewsEventDetector
from src.trading.paper_trader import PaperTrader, TradingBot, OrderSide, OrderType
from src.trading.portfolio_manager import PortfolioManager
from src.data.news import NewsFetcher
from src.analysis.sentiment import SentimentAnalyzer


class InvestmentCLI:
    def __init__(self):
        self.fetcher = MarketDataFetcher()
        self.strategy = MovingAverageStrategy()
        self.sentiment_strategy = SentimentEnhancedStrategy()
        self.news_fetcher = NewsFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.event_detector = NewsEventDetector()
        self.paper_trader = PaperTrader()
        self.portfolio_manager = PortfolioManager(self.paper_trader)
        self.trading_bot = TradingBot(self.paper_trader)
    
    def get_stock_data(self, symbol: str, period: str = "1mo"):
        print(f"\n📊 Fetching data for {symbol}...")
        data = self.fetcher.get_stock_data(symbol, period=period)
        
        if data.empty:
            print(f"❌ No data found for {symbol}")
            return
        
        data_with_indicators = self.fetcher.get_technical_indicators(data)
        latest = data_with_indicators.iloc[-1]
        
        print(f"\n--- {symbol} Summary ---")
        print(f"Period: {period}")
        print(f"Latest Close: ${latest['Close']:.2f}")
        print(f"Daily Change: {((latest['Close'] - data_with_indicators.iloc[-2]['Close']) / data_with_indicators.iloc[-2]['Close'] * 100):.2f}%")
        
        if pd.notna(latest['SMA_20']):
            print(f"SMA 20: ${latest['SMA_20']:.2f}")
        if pd.notna(latest['SMA_50']):
            print(f"SMA 50: ${latest['SMA_50']:.2f}")
        if pd.notna(latest['RSI']):
            print(f"RSI: {latest['RSI']:.2f}")
        
        current_price = self.fetcher.get_current_price(symbol)
        if current_price:
            print(f"Live Price: ${current_price:.2f}")
    
    def get_trading_signal(self, symbol: str, period: str = "6mo"):
        print(f"\n🎯 Analyzing trading signals for {symbol}...")
        data = self.fetcher.get_stock_data(symbol, period=period)
        
        if data.empty:
            print(f"❌ No data found for {symbol}")
            return
        
        signal = self.strategy.get_latest_signal(data, symbol)
        
        print(f"\n--- Trading Signal for {symbol} ---")
        print(f"Action: {signal['action']}")
        print(f"Price: ${signal['price']:.2f}")
        print(f"Confidence: {signal['confidence']*100:.0f}%")
        print(f"Timestamp: {signal['timestamp']}")
        
        if 'sma_short' in signal and pd.notna(signal['sma_short']):
            print(f"SMA Short: ${signal['sma_short']:.2f}")
            print(f"SMA Long: ${signal['sma_long']:.2f}")
    
    def run_backtest(self, symbol: str, period: str = "1y", capital: float = 10000):
        print(f"\n📈 Running backtest for {symbol}...")
        data = self.fetcher.get_stock_data(symbol, period=period)
        
        if data.empty:
            print(f"❌ No data found for {symbol}")
            return
        
        results = self.strategy.backtest(data, initial_capital=capital)
        
        if 'error' in results:
            print(f"❌ {results['error']}")
            return
        
        print(f"\n--- Backtest Results for {symbol} ---")
        print(f"Period: {period}")
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Buy & Hold Return: {results['buy_hold_return_pct']:.2f}%")
        print(f"Outperformance: {results['outperformance']:.2f}%")
        print(f"Number of Trades: {results['num_trades']}")
        
        if results['num_trades'] > 0:
            print(f"\nRecent Trades:")
            for trade in results['trades'][-3:]:  # Show last 3 trades
                print(f"  {trade['date'].strftime('%Y-%m-%d')}: {trade['action']} {trade['shares']} shares at ${trade['price']:.2f}")
    
    def analyze_portfolio(self, symbols: list, period: str = "1mo"):
        print(f"\n📋 Portfolio Analysis ({len(symbols)} stocks)")
        print("="*50)
        
        portfolio_data = {}
        total_signals = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for symbol in symbols:
            data = self.fetcher.get_stock_data(symbol, period=period)
            if not data.empty:
                signal = self.strategy.get_latest_signal(data, symbol)
                portfolio_data[symbol] = signal
                total_signals[signal['action']] += 1
                
                print(f"{symbol:6} | {signal['action']:4} | ${signal['price']:8.2f} | {signal['confidence']*100:3.0f}%")
        
        print("="*50)
        print(f"Summary: {total_signals['BUY']} BUY, {total_signals['SELL']} SELL, {total_signals['HOLD']} HOLD")
    
    def analyze_sentiment(self, symbol: str, period: str = "7d"):
        print(f"\n📰 Analyzing news sentiment for {symbol}...")
        
        try:
            data = self.fetcher.get_stock_data(symbol, period="1mo")  # Need price data for strategy
            if data.empty:
                print(f"❌ No price data found for {symbol}")
                return
            
            signal = self.sentiment_strategy.get_latest_signal(data, symbol)
            
            print(f"\n--- Sentiment Analysis for {symbol} ---")
            print(f"Sentiment Score: {signal.get('sentiment_score', 0):.3f}")
            print(f"Sentiment Label: {signal.get('sentiment_label', 'neutral').upper()}")
            print(f"Sentiment Confidence: {signal.get('sentiment_confidence', 0)*100:.0f}%")
            print(f"Impact Score: {signal.get('impact_score', 0):.3f}")
            print(f"Est. Price Impact: {signal.get('estimated_price_impact', 0):.2f}%")
            print(f"Articles Analyzed: {signal.get('article_count', 0)}")
            
            print(f"\n--- Signal Comparison ---")
            print(f"Base Technical Signal: {signal.get('base_action', 'N/A')} (Confidence: {signal.get('base_confidence', 0)*100:.0f}%)")
            print(f"Enhanced Signal: {signal.get('action', 'N/A')} (Confidence: {signal.get('confidence', 0)*100:.0f}%)")
            print(f"Sentiment Influence: {signal.get('sentiment_influence', 0):.3f}")
            
            # Show recent news articles
            articles = signal.get('news_articles', [])
            if articles:
                print(f"\n--- Recent News Articles ---")
                for i, article in enumerate(articles[:5], 1):
                    print(f"{i}. {article['title'][:70]}...")
                    print(f"   Sentiment: {article['sentiment_score']:.3f} ({article['sentiment_label']})")
                    print(f"   Source: {article['source']} | Date: {article.get('published_date', 'N/A')[:10]}")
        
        except Exception as e:
            print(f"❌ Error analyzing sentiment: {str(e)}")
    
    def show_news_events(self, symbol: str, days_back: int = 7):
        print(f"\n📅 Detecting significant news events for {symbol}...")
        
        try:
            events = self.event_detector.detect_events(symbol, days_back=days_back)
            
            if not events:
                print(f"No significant events found for {symbol} in the last {days_back} days")
                return
            
            print(f"\n--- Significant Events for {symbol} ---")
            for i, event in enumerate(events[:10], 1):
                print(f"\n{i}. {event['title']}")
                print(f"   Event Types: {', '.join(event['event_types'])}")
                print(f"   Sentiment: {event['sentiment_score']:.3f} ({event['sentiment_label']})")
                print(f"   Source: {event['source']}")
                print(f"   Date: {event['published_date'][:10] if event['published_date'] else 'N/A'}")
                print(f"   Relevance: {event['relevance_score']*100:.0f}%")
        
        except Exception as e:
            print(f"❌ Error detecting events: {str(e)}")
    
    def show_market_sentiment(self):
        print(f"\n🌐 Analyzing general market sentiment...")
        
        try:
            articles = self.news_fetcher.fetch_general_market_news(days_back=3)
            
            if not articles:
                print("No market news articles found")
                return
            
            sentiment_result = self.sentiment_analyzer.analyze_multiple_articles(articles)
            
            print(f"\n--- Market Sentiment Summary ---")
            print(f"Overall Sentiment: {sentiment_result['overall_sentiment']:.3f} ({sentiment_result['sentiment_label'].upper()})")
            print(f"Confidence: {sentiment_result['confidence']*100:.0f}%")
            print(f"Articles Analyzed: {sentiment_result['article_count']}")
            print(f"Distribution: {sentiment_result['positive_count']} positive, {sentiment_result['negative_count']} negative, {sentiment_result['neutral_count']} neutral")
            
            # Show some recent articles
            print(f"\n--- Recent Market Articles ---")
            for i, article in enumerate(articles[:5], 1):
                print(f"{i}. {article.title[:70]}...")
                print(f"   Source: {article.source} | Date: {article.published_date.strftime('%Y-%m-%d') if article.published_date else 'N/A'}")
        
        except Exception as e:
            print(f"❌ Error analyzing market sentiment: {str(e)}")
    
    def show_portfolio(self):
        print(f"\n💼 Portfolio Summary")
        print("="*60)
        
        try:
            # Check connection status
            connected = self.paper_trader.is_connected()
            print(f"Connection: {'✅ Alpaca Paper Trading' if connected else '⚠️  Simulated Mode'}")
            
            # Get account info
            account = self.paper_trader.get_account_info()
            if account:
                print(f"Account Value: ${account.get('portfolio_value', 0):,.2f}")
                print(f"Cash Balance: ${account.get('cash', 0):,.2f}")
                print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
                
                day_change = account.get('portfolio_value', 0) - account.get('last_equity', 0)
                day_change_pct = (day_change / account.get('last_equity', 1)) * 100
                change_symbol = "+" if day_change >= 0 else ""
                print(f"Today's Change: {change_symbol}${day_change:.2f} ({change_symbol}{day_change_pct:.2f}%)")
            
            # Get positions
            positions = self.portfolio_manager.get_current_positions()
            
            if positions:
                print(f"\n--- Positions ({len(positions)}) ---")
                print(f"{'Symbol':<8} {'Qty':<6} {'Avg Price':<12} {'Current':<12} {'P&L':<12} {'P&L %':<8}")
                print("-" * 65)
                
                for pos in positions:
                    pnl_symbol = "+" if pos.unrealized_pnl >= 0 else ""
                    pnl_pct_symbol = "+" if pos.unrealized_pnl_pct >= 0 else ""
                    
                    print(f"{pos.symbol:<8} {pos.quantity:<6} ${pos.avg_entry_price:<11.2f} "
                          f"${pos.current_price:<11.2f} {pnl_symbol}${pos.unrealized_pnl:<11.2f} "
                          f"{pnl_pct_symbol}{pos.unrealized_pnl_pct:<7.2f}%")
            else:
                print("\n--- No Current Positions ---")
            
            # Show performance metrics
            print(f"\n--- Performance Metrics ---")
            metrics = self.portfolio_manager.calculate_performance_metrics()
            print(f"Total Return: {metrics.total_return_pct:.2f}%")
            print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {metrics.max_drawdown:.2f}%")
            print(f"Win Rate: {metrics.win_rate:.1f}%")
            print(f"Total Trades: {metrics.total_trades}")
            
        except Exception as e:
            print(f"❌ Error getting portfolio: {str(e)}")
    
    def place_order(self, symbol: str, quantity: int, side: str, order_type: str = "market"):
        print(f"\n📋 Placing {side.upper()} order for {quantity} shares of {symbol}")
        
        try:
            # Validate inputs
            if side.lower() not in ['buy', 'sell']:
                print("❌ Invalid side. Use 'buy' or 'sell'")
                return
            
            if quantity <= 0:
                print("❌ Quantity must be positive")
                return
            
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            order_type_enum = OrderType.MARKET if order_type.lower() == 'market' else OrderType.LIMIT
            
            # Get current price for reference
            current_price = self.fetcher.get_current_price(symbol)
            if current_price:
                estimated_cost = quantity * current_price
                print(f"Current Price: ${current_price:.2f}")
                print(f"Estimated {'Cost' if side.lower() == 'buy' else 'Value'}: ${estimated_cost:,.2f}")
                
                # Confirm order
                confirm = input(f"Confirm {side.upper()} {quantity} {symbol}? (y/N): ")
                if confirm.lower() != 'y':
                    print("Order cancelled")
                    return
            
            # Submit order
            order = self.paper_trader.submit_order(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                order_type=order_type_enum
            )
            
            if order:
                print(f"✅ Order submitted successfully!")
                print(f"Order ID: {order['id']}")
                print(f"Status: {order['status']}")
                
                # Add to trade history if filled
                if order['status'] == 'filled':
                    self.portfolio_manager.add_trade(order)
            else:
                print("❌ Order failed")
                
        except Exception as e:
            print(f"❌ Error placing order: {str(e)}")
    
    def show_orders(self, status: str = "all"):
        print(f"\n📊 Orders ({status.upper()})")
        print("="*80)
        
        try:
            status_filter = None if status == "all" else status
            orders = self.paper_trader.get_orders(status=status_filter, limit=20)
            
            if not orders:
                print("No orders found")
                return
            
            print(f"{'Order ID':<12} {'Symbol':<8} {'Side':<5} {'Qty':<6} {'Type':<8} "
                  f"{'Status':<12} {'Submitted':<12}")
            print("-" * 80)
            
            for order in orders:
                submitted_time = order['submitted_at'][:10] if order['submitted_at'] else 'N/A'
                
                print(f"{order['id'][:12]:<12} {order['symbol']:<8} {order['side']:<5} "
                      f"{order['qty']:<6.0f} {order['order_type']:<8} {order['status']:<12} "
                      f"{submitted_time:<12}")
                      
        except Exception as e:
            print(f"❌ Error getting orders: {str(e)}")
    
    def auto_trade(self, symbol: str, period: str = "6mo"):
        print(f"\n🤖 Auto-trading signal for {symbol}")
        print("="*50)
        
        try:
            # Get enhanced signal
            data = self.fetcher.get_stock_data(symbol, period="3mo")  # Need enough data
            if data.empty:
                print(f"❌ No data available for {symbol}")
                return
                
            signal = self.sentiment_strategy.get_latest_signal(data, symbol)
            
            print(f"Signal: {signal['action']}")
            print(f"Confidence: {signal.get('confidence', 0)*100:.0f}%")
            print(f"Sentiment: {signal.get('sentiment_score', 0):.3f} ({signal.get('sentiment_label', 'neutral')})")
            
            # Execute signal if auto-trading enabled
            if self.trading_bot.auto_trading_enabled:
                print("\nExecuting signal...")
                order = self.trading_bot.execute_signal(signal)
                
                if order:
                    print(f"✅ Order executed: {order['side'].upper()} {order['qty']} {order['symbol']}")
                    self.portfolio_manager.add_trade(order)
                else:
                    print("ℹ️ No order placed (low confidence or other constraints)")
            else:
                print("\n⚠️ Auto-trading disabled. Set ENABLE_AUTO_TRADING=true to enable.")
                print(f"Recommendation: {signal['action']} {symbol}")
                
        except Exception as e:
            print(f"❌ Error in auto-trading: {str(e)}")
    
    def show_performance(self):
        print(f"\n📈 Performance Analysis")
        print("="*60)
        
        try:
            metrics = self.portfolio_manager.calculate_performance_metrics()
            
            print(f"--- Returns ---")
            print(f"Total Return: {metrics.total_return_pct:.2f}%")
            print(f"Annualized Return: {metrics.annualized_return:.2f}%")
            print(f"Volatility: {metrics.volatility:.2f}%")
            print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"Maximum Drawdown: {metrics.max_drawdown:.2f}%")
            
            print(f"\n--- Trading Statistics ---")
            print(f"Total Trades: {metrics.total_trades}")
            print(f"Win Rate: {metrics.win_rate:.1f}%")
            print(f"Profit Factor: {metrics.profit_factor:.2f}")
            print(f"Winning Trades: {metrics.winning_trades}")
            print(f"Losing Trades: {metrics.losing_trades}")
            
            if metrics.total_trades > 0:
                print(f"Average Win: ${metrics.avg_win:.2f}")
                print(f"Average Loss: ${metrics.avg_loss:.2f}")
                print(f"Largest Win: ${metrics.largest_win:.2f}")
                print(f"Largest Loss: ${metrics.largest_loss:.2f}")
            
            # Show sector allocation
            print(f"\n--- Sector Allocation ---")
            sectors = self.portfolio_manager.get_sector_allocation()
            for sector, data in sectors.items():
                print(f"{sector}: {data['allocation_pct']:.1f}% (${data['market_value']:,.2f})")
                
        except Exception as e:
            print(f"❌ Error getting performance: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Investment System CLI")
    parser.add_argument('command', choices=['quote', 'signal', 'backtest', 'portfolio', 'sentiment', 'events', 'market-sentiment', 
                                       'trading-portfolio', 'buy', 'sell', 'orders', 'auto-trade', 'performance'], 
                       help='Command to execute')
    parser.add_argument('--symbol', '-s', type=str, help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--symbols', nargs='+', help='Multiple stock symbols')
    parser.add_argument('--period', '-p', default='1mo', 
                       help='Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)')
    parser.add_argument('--capital', '-c', type=float, default=10000,
                       help='Initial capital for backtesting')
    parser.add_argument('--quantity', '-q', type=int, default=10,
                       help='Number of shares for buy/sell orders')
    parser.add_argument('--side', type=str, choices=['buy', 'sell'],
                       help='Order side for manual orders')
    parser.add_argument('--order-type', type=str, choices=['market', 'limit'], default='market',
                       help='Order type for manual orders')
    parser.add_argument('--status', type=str, choices=['all', 'open', 'closed', 'filled'], default='all',
                       help='Order status filter')
    
    args = parser.parse_args()
    
    cli = InvestmentCLI()
    
    if args.command == 'quote':
        if not args.symbol:
            print("Error: --symbol required for quote command")
            return
        cli.get_stock_data(args.symbol, args.period)
    
    elif args.command == 'signal':
        if not args.symbol:
            print("Error: --symbol required for signal command")
            return
        cli.get_trading_signal(args.symbol, args.period)
    
    elif args.command == 'backtest':
        if not args.symbol:
            print("Error: --symbol required for backtest command")
            return
        cli.run_backtest(args.symbol, args.period, args.capital)
    
    elif args.command == 'portfolio':
        symbols = args.symbols or ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ', 'VTI', 'BTC-USD']
        cli.analyze_portfolio(symbols, args.period)
    
    elif args.command == 'sentiment':
        if not args.symbol:
            print("Error: --symbol required for sentiment command")
            return
        cli.analyze_sentiment(args.symbol, args.period)
    
    elif args.command == 'events':
        if not args.symbol:
            print("Error: --symbol required for events command")
            return
        days_back = min(30, max(1, int(args.period.replace('mo', '30').replace('d', '1').replace('y', '365')[:2])))
        cli.show_news_events(args.symbol, days_back)
    
    elif args.command == 'market-sentiment':
        cli.show_market_sentiment()
    
    elif args.command == 'trading-portfolio':
        cli.show_portfolio()
    
    elif args.command == 'buy':
        if not args.symbol:
            print("Error: --symbol required for buy command")
            return
        cli.place_order(args.symbol, args.quantity, 'buy', args.order_type)
    
    elif args.command == 'sell':
        if not args.symbol:
            print("Error: --symbol required for sell command")
            return
        cli.place_order(args.symbol, args.quantity, 'sell', args.order_type)
    
    elif args.command == 'orders':
        cli.show_orders(args.status)
    
    elif args.command == 'auto-trade':
        if not args.symbol:
            print("Error: --symbol required for auto-trade command")
            return
        cli.auto_trade(args.symbol, args.period)
    
    elif args.command == 'performance':
        cli.show_performance()


if __name__ == "__main__":
    main()