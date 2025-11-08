import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Set
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sqlite3
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SecurityInfo:
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    exchange: str
    security_type: str  # stock, etf, warrant, etc.
    is_active: bool
    last_updated: datetime


@dataclass
class HistoricalDataRecord:
    symbol: str
    date: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    adj_close: float
    volume: int
    splits: float
    dividends: float
    

class HistoricalDataCollector:
    """
    Comprehensive system to collect and store historical data for all NYSE stocks and options
    from the past decade with efficient storage and retrieval capabilities.
    """
    
    def __init__(self, data_dir: str = "data/historical"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Database for metadata and fast queries
        self.db_path = self.data_dir / "securities.db"
        self.init_database()
        
        # Rate limiting to respect API limits
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
        
        # NYSE exchanges to cover
        self.nyse_exchanges = ['NYSE', 'NASDAQ', 'AMEX', 'BATS']
        
        logger.info(f"Historical data collector initialized with data dir: {self.data_dir}")
    
    def init_database(self):
        """Initialize SQLite database for metadata storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Securities metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS securities (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,
                    exchange TEXT,
                    security_type TEXT,
                    is_active INTEGER,
                    last_updated TEXT,
                    data_start_date TEXT,
                    data_end_date TEXT,
                    total_records INTEGER
                )
            ''')
            
            # Data collection status table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS collection_status (
                    symbol TEXT,
                    period_start TEXT,
                    period_end TEXT,
                    status TEXT,
                    error_message TEXT,
                    last_attempt TEXT,
                    records_collected INTEGER,
                    PRIMARY KEY (symbol, period_start)
                )
            ''')
            
            # Create indexes for fast queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_securities_exchange ON securities(exchange)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_securities_sector ON securities(sector)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_securities_active ON securities(is_active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_collection_status ON collection_status(status)')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    async def get_nyse_stock_list(self) -> List[str]:
        """Get comprehensive list of NYSE-traded securities"""
        try:
            logger.info("Fetching NYSE stock list...")
            
            # Get major indices and their components
            major_indices = ['SPY', 'QQQ', 'IWM', 'DIA']  # S&P 500, NASDAQ, Russell 2000, Dow
            all_symbols = set()
            
            # Method 1: Get from major ETF holdings
            for index_symbol in major_indices:
                try:
                    ticker = yf.Ticker(index_symbol)
                    # Try to get institutional holders or major holdings
                    info = ticker.info
                    if 'holdings' in info:
                        holdings = info['holdings']
                        for holding in holdings[:50]:  # Top 50 holdings
                            if 'symbol' in holding:
                                all_symbols.add(holding['symbol'])
                except Exception as e:
                    logger.warning(f"Could not get holdings for {index_symbol}: {str(e)}")
            
            # Method 2: Common NYSE symbols by market cap ranges
            # Add major blue-chip stocks
            blue_chip_stocks = [
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK.A', 'BRK.B',
                'UNH', 'JNJ', 'JPM', 'V', 'PG', 'HD', 'MA', 'PYPL', 'DIS', 'ADBE', 'CRM', 'NFLX',
                'XOM', 'VZ', 'KO', 'PFE', 'CSCO', 'PEP', 'T', 'ABT', 'AVGO', 'TMO', 'COST', 'NKE',
                'DHR', 'MRK', 'WMT', 'LLY', 'NEE', 'ACN', 'ORCL', 'CVX', 'MDT', 'TXN', 'QCOM', 'UNP',
                'LIN', 'PM', 'RTX', 'LOW', 'SPGI', 'HON', 'IBM', 'AMGN', 'CAT', 'GS', 'AXP', 'BA',
                'BLK', 'GILD', 'SYK', 'BKNG', 'ADP', 'MDLZ', 'VRTX', 'TJX', 'AMT', 'C', 'CVS', 'MO',
                'ISRG', 'ZTS', 'MMM', 'PLD', 'TMUS', 'DUK', 'SO', 'FIS', 'CME', 'CL', 'EMR', 'BSX',
                'EL', 'SHW', 'ITW', 'AON', 'GE', 'D', 'MMC', 'NSC', 'ATVI', 'HCA', 'REGN', 'ECL'
            ]
            all_symbols.update(blue_chip_stocks)
            
            # Method 3: Add sector representatives
            sector_representatives = {
                'Technology': ['CRM', 'ORCL', 'IBM', 'HPQ', 'DELL', 'VMW', 'INTU', 'ADSK'],
                'Healthcare': ['PFE', 'MRK', 'ABT', 'BMY', 'GILD', 'BIIB', 'CELG', 'AMGN'],
                'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'BK'],
                'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'KMI', 'WMB', 'OXY'],
                'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'TGT', 'HD', 'LOW', 'NKE', 'SBUX'],
                'Industrial': ['GE', 'CAT', 'BA', 'MMM', 'UNP', 'RTX', 'LMT', 'HON', 'EMR'],
                'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'DOW', 'PPG'],
                'Utilities': ['NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP', 'XEL', 'SRE'],
                'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'WELL', 'PSA', 'O'],
                'Telecom': ['VZ', 'T', 'TMUS', 'CTL', 'USM', 'CHTR']
            }
            
            for sector, symbols in sector_representatives.items():
                all_symbols.update(symbols)
            
            # Method 4: Add popular ETFs and index funds
            popular_etfs = [
                'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VEA', 'VWO', 'AGG', 'BND', 'TLT',
                'GLD', 'SLV', 'USO', 'UNG', 'XLF', 'XLK', 'XLE', 'XLI', 'XLV', 'XLP',
                'XLU', 'XLB', 'XLRE', 'XLY', 'VIG', 'VXUS', 'IEFA', 'IEMG', 'IJH', 'IJR'
            ]
            all_symbols.update(popular_etfs)
            
            # Method 5: Generate systematic symbols (common patterns)
            # Add common single/double/triple letter symbols
            common_patterns = []
            
            # Single letters
            for i in range(26):
                common_patterns.append(chr(ord('A') + i))
            
            # Popular two-letter combinations
            popular_two_letter = ['AA', 'AI', 'AT', 'BA', 'CA', 'CI', 'DD', 'EA', 'ET', 'GE', 'GM', 'HP', 'IBM', 'IT']
            common_patterns.extend(popular_two_letter)
            
            all_symbols.update(common_patterns)
            
            # Convert to sorted list
            symbol_list = sorted(list(all_symbols))
            
            logger.info(f"Collected {len(symbol_list)} NYSE symbols for historical data collection")
            return symbol_list
            
        except Exception as e:
            logger.error(f"Error getting NYSE stock list: {str(e)}")
            # Fallback to a smaller but reliable list
            fallback_list = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH',
                'PG', 'HD', 'V', 'MA', 'DIS', 'PYPL', 'NFLX', 'CRM', 'ADBE', 'XOM', 'VZ', 'KO',
                'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'GLD', 'TLT', 'XLF'
            ]
            logger.info(f"Using fallback list with {len(fallback_list)} symbols")
            return fallback_list
    
    async def collect_historical_data(self, symbols: List[str], years_back: int = 10, 
                                    batch_size: int = 50, max_workers: int = 10):
        """Collect historical data for all symbols"""
        try:
            logger.info(f"Starting historical data collection for {len(symbols)} symbols, {years_back} years back")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years_back * 365)
            
            logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
            
            # Process in batches to manage memory and API limits
            total_success = 0
            total_failed = 0
            
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(symbols) + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_symbols)} symbols)")
                
                # Collect data for this batch
                batch_results = await self._collect_batch_data(
                    batch_symbols, start_date, end_date, max_workers
                )
                
                # Update counters
                batch_success = sum(1 for r in batch_results if r['status'] == 'success')
                batch_failed = len(batch_results) - batch_success
                
                total_success += batch_success
                total_failed += batch_failed
                
                logger.info(f"Batch {batch_num} completed: {batch_success} success, {batch_failed} failed")
                
                # Brief pause between batches to be respectful to APIs
                await asyncio.sleep(1)
            
            logger.info(f"Historical data collection completed: {total_success} successful, {total_failed} failed")
            
            # Generate summary report
            summary = self.generate_collection_summary()
            return {
                'total_symbols': len(symbols),
                'successful': total_success, 
                'failed': total_failed,
                'date_range': f"{start_date.date()} to {end_date.date()}",
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Error in historical data collection: {str(e)}")
            raise
    
    async def _collect_batch_data(self, symbols: List[str], start_date: datetime, 
                                 end_date: datetime, max_workers: int) -> List[Dict]:
        """Collect data for a batch of symbols using thread pool"""
        results = []
        
        def collect_single_symbol(symbol: str) -> Dict:
            """Collect data for a single symbol"""
            try:
                # Rate limiting
                current_time = time.time()
                if current_time - self.last_request_time < self.rate_limit_delay:
                    time.sleep(self.rate_limit_delay - (current_time - self.last_request_time))
                self.last_request_time = time.time()
                
                # Get ticker data
                ticker = yf.Ticker(symbol)
                
                # Get historical data for the full period
                hist_data = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d',
                    actions=True  # Include dividends and splits
                )
                
                if hist_data.empty:
                    return {
                        'symbol': symbol,
                        'status': 'failed',
                        'error': 'No historical data available',
                        'records': 0
                    }
                
                # Get basic info about the security
                info = {}
                try:
                    info = ticker.info
                except:
                    pass  # Info might not be available for all symbols
                
                # Store the data
                records_stored = self._store_historical_data(symbol, hist_data, info)
                
                return {
                    'symbol': symbol,
                    'status': 'success',
                    'records': records_stored,
                    'start_date': hist_data.index[0].strftime('%Y-%m-%d'),
                    'end_date': hist_data.index[-1].strftime('%Y-%m-%d')
                }
                
            except Exception as e:
                logger.warning(f"Failed to collect data for {symbol}: {str(e)}")
                return {
                    'symbol': symbol,
                    'status': 'failed', 
                    'error': str(e),
                    'records': 0
                }
        
        # Use ThreadPoolExecutor for concurrent requests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(collect_single_symbol, symbol) for symbol in symbols]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing future: {str(e)}")
                    results.append({
                        'symbol': 'unknown',
                        'status': 'failed',
                        'error': str(e),
                        'records': 0
                    })
        
        return results
    
    def _store_historical_data(self, symbol: str, hist_data: pd.DataFrame, 
                              info: Dict) -> int:
        """Store historical data for a symbol"""
        try:
            # Create parquet file for efficient storage of price data
            parquet_path = self.data_dir / f"{symbol}.parquet"
            
            # Prepare data for storage
            storage_data = hist_data.copy()
            storage_data = storage_data.reset_index()
            storage_data['Symbol'] = symbol
            
            # Store as parquet for efficient access
            storage_data.to_parquet(parquet_path, index=False)
            
            # Update metadata in database
            self._update_security_metadata(symbol, hist_data, info)
            
            logger.debug(f"Stored {len(storage_data)} records for {symbol}")
            return len(storage_data)
            
        except Exception as e:
            logger.error(f"Error storing data for {symbol}: {str(e)}")
            return 0
    
    def _update_security_metadata(self, symbol: str, hist_data: pd.DataFrame, info: Dict):
        """Update security metadata in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract info safely
            name = info.get('longName', info.get('shortName', symbol))
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            market_cap = info.get('marketCap', 0)
            exchange = info.get('exchange', 'Unknown')
            
            # Determine security type
            security_type = 'stock'
            if symbol.endswith('W'):
                security_type = 'warrant'
            elif any(x in symbol for x in ['ETF', 'FUND']):
                security_type = 'etf'
            elif len(symbol) <= 3 and symbol.isupper():
                security_type = 'etf'  # Many ETFs are short symbols
            
            cursor.execute('''
                INSERT OR REPLACE INTO securities (
                    symbol, name, sector, industry, market_cap, exchange, security_type,
                    is_active, last_updated, data_start_date, data_end_date, total_records
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, name, sector, industry, market_cap, exchange, security_type,
                1, datetime.now().isoformat(),
                hist_data.index[0].strftime('%Y-%m-%d'),
                hist_data.index[-1].strftime('%Y-%m-%d'),
                len(hist_data)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating metadata for {symbol}: {str(e)}")
    
    def generate_collection_summary(self) -> Dict:
        """Generate summary of collected data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get basic stats
            cursor.execute('SELECT COUNT(*) FROM securities')
            total_securities = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(total_records) FROM securities')
            total_records = cursor.fetchone()[0] or 0
            
            # Get by security type
            cursor.execute('SELECT security_type, COUNT(*) FROM securities GROUP BY security_type')
            by_type = dict(cursor.fetchall())
            
            # Get by sector
            cursor.execute('SELECT sector, COUNT(*) FROM securities WHERE sector != "Unknown" GROUP BY sector ORDER BY COUNT(*) DESC LIMIT 10')
            by_sector = dict(cursor.fetchall())
            
            # Get date range
            cursor.execute('SELECT MIN(data_start_date), MAX(data_end_date) FROM securities')
            date_range = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_securities': total_securities,
                'total_records': total_records,
                'by_security_type': by_type,
                'top_sectors': by_sector,
                'date_range': {
                    'start': date_range[0],
                    'end': date_range[1]
                },
                'avg_records_per_security': total_records / total_securities if total_securities > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {}
    
    def get_available_symbols(self, security_type: str = None, sector: str = None) -> List[str]:
        """Get list of symbols with available historical data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT symbol FROM securities WHERE is_active = 1"
            params = []
            
            if security_type:
                query += " AND security_type = ?"
                params.append(security_type)
                
            if sector:
                query += " AND sector = ?"
                params.append(sector)
            
            cursor.execute(query, params)
            symbols = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error getting available symbols: {str(e)}")
            return []
    
    def get_historical_data(self, symbol: str, start_date: str = None, 
                          end_date: str = None) -> pd.DataFrame:
        """Retrieve historical data for a symbol"""
        try:
            parquet_path = self.data_dir / f"{symbol}.parquet"
            
            if not parquet_path.exists():
                logger.warning(f"No historical data file found for {symbol}")
                return pd.DataFrame()
            
            # Load data
            data = pd.read_parquet(parquet_path)
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
            
            # Filter by date range if specified
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {str(e)}")
            return pd.DataFrame()