import requests
import zipfile
import pandas as pd
import sqlite3
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os
import shutil
import time
from dataclasses import dataclass
import glob

logger = logging.getLogger(__name__)

@dataclass
class StooqDataSet:
    name: str
    url: str
    description: str
    expected_files: int
    file_pattern: str  # glob pattern to find files after extraction

class StooqDownloader:
    """
    Downloads and processes bulk historical stock data from Stooq
    """
    
    # Stooq individual download URL template
    STOOQ_CSV_URL = "https://stooq.pl/q/d/l/?s={symbol}&d1={start_date}&d2={end_date}&i=d"
    
    # Known Stooq dataset URLs (as of 2025) - Note: bulk downloads may require manual access
    DATASETS = {
        'us_stocks': StooqDataSet(
            name='US Stocks Daily',
            url='https://stooq.com/db/h/',  # Manual bulk download page
            description='Daily US stock data including NYSE, NASDAQ',
            expected_files=3600,
            file_pattern='**/*.us.txt'
        ),
        'us_etf': StooqDataSet(
            name='US ETF Daily', 
            url='https://stooq.com/db/h/',  # Manual bulk download page
            description='Daily US ETF data',
            expected_files=2700,
            file_pattern='**/*.us.txt'
        )
    }
    
    def __init__(self, data_dir: str = "data/stooq"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Database for metadata and fast symbol lookup
        self.db_path = self.data_dir / "stooq_metadata.db"
        self.init_database()
        
        logger.info(f"Stooq downloader initialized: {self.data_dir}")
    
    def init_database(self):
        """Initialize SQLite database for metadata"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stooq_symbols (
                    symbol TEXT PRIMARY KEY,
                    filename TEXT,
                    exchange TEXT,
                    security_type TEXT,
                    first_date TEXT,
                    last_date TEXT,
                    total_records INTEGER,
                    file_size INTEGER,
                    last_updated TEXT,
                    data_quality_score REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS download_log (
                    dataset_name TEXT,
                    download_date TEXT,
                    file_size INTEGER,
                    files_extracted INTEGER,
                    success INTEGER,
                    error_message TEXT
                )
            ''')
            
            # Indexes for fast queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON stooq_symbols(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_exchange ON stooq_symbols(exchange)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_type ON stooq_symbols(security_type)')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def download_symbol_from_stooq(self, symbol: str, start_date: str = "20000101", 
                                   end_date: str = None) -> pd.DataFrame:
        """Download individual symbol data from Stooq"""
        try:
            if not end_date:
                end_date = datetime.now().strftime("%Y%m%d")
            
            # Format symbol for Stooq (lowercase with .us suffix)
            stooq_symbol = f"{symbol.lower()}.us"
            
            url = self.STOOQ_CSV_URL.format(
                symbol=stooq_symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            logger.debug(f"Downloading {symbol} from Stooq: {url}")
            
            # Download with headers to appear like a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Check if we got actual CSV data or an error page
            if len(response.text) < 50:
                logger.warning(f"No data or invalid response for {symbol} from Stooq")
                return pd.DataFrame()
            
            # Parse CSV data
            from io import StringIO
            csv_data = StringIO(response.text)
            
            df = pd.read_csv(csv_data)
            
            if df.empty:
                logger.warning(f"Empty dataset for {symbol} from Stooq")
                return pd.DataFrame()
            
            # Stooq uses Polish column names, map to English
            column_mapping = {
                'Data': 'date',
                'Otwarcie': 'open', 
                'Najwyzszy': 'high',
                'Najnizszy': 'low',
                'Zamkniecie': 'close',
                'Wolumen': 'volume'
            }
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Ensure we have the expected columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing columns in {symbol} data. Got: {list(df.columns)}")
                return pd.DataFrame()
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            logger.info(f"Downloaded {len(df)} records for {symbol} from Stooq")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading {symbol} from Stooq: {str(e)}")
            return pd.DataFrame()
    
    def download_symbols_list(self, symbols: List[str], start_date: str = "20200101") -> Dict[str, pd.DataFrame]:
        """Download multiple symbols from Stooq individually"""
        results = {}
        
        logger.info(f"Downloading {len(symbols)} symbols from Stooq...")
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"Downloading {symbol} ({i+1}/{len(symbols)})")
                
                data = self.download_symbol_from_stooq(symbol, start_date)
                
                if not data.empty:
                    results[symbol] = data
                    
                    # Save to processed directory
                    processed_file = self.processed_dir / f"{symbol}.parquet"
                    data.to_parquet(processed_file, index=False)
                    
                    # Update database
                    self._update_symbol_metadata(symbol, data, 'stooq')
                
                # Rate limiting - be polite to Stooq
                if i < len(symbols) - 1:
                    time.sleep(1)  # 1 second between requests
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        logger.info(f"Completed downloading {len(results)} symbols from Stooq")
        return results
    
    def _update_symbol_metadata(self, symbol: str, df: pd.DataFrame, source: str):
        """Update symbol metadata in database"""
        try:
            if df.empty:
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Determine security type and exchange
            exchange = 'US_MARKET'
            security_type = 'stock'
            
            if symbol.endswith('W'):
                security_type = 'warrant'
            elif len(symbol) <= 4 and symbol.upper() in ['SPY', 'QQQ', 'IWM', 'VTI', 'GLD', 'TLT']:
                security_type = 'etf'
            
            quality_score = self._calculate_quality_score(df)
            
            cursor.execute('''
                INSERT OR REPLACE INTO stooq_symbols 
                (symbol, filename, exchange, security_type, first_date, last_date, 
                 total_records, file_size, last_updated, data_quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, f"{symbol.lower()}.us.txt", exchange, security_type,
                df.iloc[0]['date'].strftime('%Y-%m-%d'),
                df.iloc[-1]['date'].strftime('%Y-%m-%d'),
                len(df), len(df) * 50,  # Rough file size estimate
                datetime.now().isoformat(), quality_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating metadata for {symbol}: {str(e)}")
    
    def download_dataset(self, dataset_key: str, force_redownload: bool = False) -> bool:
        """Download and extract a Stooq dataset"""
        
        if dataset_key not in self.DATASETS:
            logger.error(f"Unknown dataset: {dataset_key}")
            return False
        
        dataset = self.DATASETS[dataset_key]
        zip_path = self.raw_dir / f"{dataset_key}.zip"
        extract_dir = self.raw_dir / dataset_key
        
        try:
            # Check if already downloaded
            if zip_path.exists() and not force_redownload:
                logger.info(f"Dataset {dataset_key} already downloaded, use force_redownload=True to re-download")
            else:
                logger.info(f"Downloading {dataset.name} from {dataset.url}")
                
                # Download with progress tracking
                response = requests.get(dataset.url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            if total_size > 0:
                                progress = downloaded_size / total_size * 100
                                if downloaded_size % (1024 * 1024) == 0:  # Log every MB
                                    logger.info(f"Downloaded {downloaded_size / (1024*1024):.1f} MB ({progress:.1f}%)")
                
                logger.info(f"Download completed: {zip_path} ({downloaded_size / (1024*1024):.1f} MB)")
            
            # Extract the zip file
            if extract_dir.exists() and not force_redownload:
                logger.info(f"Dataset {dataset_key} already extracted")
            else:
                logger.info(f"Extracting {dataset.name}...")
                
                # Remove existing extraction directory
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Count extracted files
                extracted_files = list(extract_dir.glob(dataset.file_pattern))
                logger.info(f"Extracted {len(extracted_files)} files from {dataset.name}")
            
            # Log successful download
            self._log_download(dataset_key, zip_path.stat().st_size, len(list(extract_dir.glob(dataset.file_pattern))), True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_key}: {str(e)}")
            self._log_download(dataset_key, 0, 0, False, str(e))
            return False
    
    def _log_download(self, dataset_name: str, file_size: int, files_extracted: int, 
                      success: bool, error_message: str = None):
        """Log download attempt"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO download_log 
                (dataset_name, download_date, file_size, files_extracted, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (dataset_name, datetime.now().isoformat(), file_size, files_extracted, 
                  1 if success else 0, error_message))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging download: {str(e)}")
    
    def process_stooq_files(self, dataset_key: str) -> int:
        """Process extracted Stooq files into our database"""
        
        if dataset_key not in self.DATASETS:
            logger.error(f"Unknown dataset: {dataset_key}")
            return 0
        
        dataset = self.DATASETS[dataset_key]
        extract_dir = self.raw_dir / dataset_key
        
        if not extract_dir.exists():
            logger.error(f"Dataset {dataset_key} not found. Download it first.")
            return 0
        
        try:
            # Find all data files
            data_files = list(extract_dir.glob(dataset.file_pattern))
            logger.info(f"Processing {len(data_files)} files from {dataset.name}")
            
            processed_count = 0
            conn = sqlite3.connect(self.db_path)
            
            for i, file_path in enumerate(data_files):
                try:
                    # Extract symbol from filename (e.g., "aapl.us.txt" -> "AAPL")
                    filename = file_path.name
                    symbol = filename.split('.')[0].upper()
                    
                    if not symbol:
                        continue
                    
                    # Read the CSV file
                    # Stooq format: Date,Open,High,Low,Close,Volume
                    df = pd.read_csv(file_path, names=['date', 'open', 'high', 'low', 'close', 'volume'])
                    
                    if df.empty:
                        continue
                    
                    # Parse dates and sort
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df.dropna(subset=['date']).sort_values('date')
                    
                    if len(df) == 0:
                        continue
                    
                    # Determine exchange and security type
                    exchange = 'NYSE'  # Default, could be enhanced
                    if '.us' in filename:
                        exchange = 'US_MARKET'
                    
                    security_type = 'stock'
                    if symbol.endswith('W'):
                        security_type = 'warrant'
                    elif len(symbol) <= 4 and any(etf_indicator in filename.lower() 
                                                for etf_indicator in ['etf', 'spy', 'qqq', 'iwm']):
                        security_type = 'etf'
                    
                    # Calculate data quality score
                    quality_score = self._calculate_quality_score(df)
                    
                    # Save processed data as parquet for efficient access
                    processed_file = self.processed_dir / f"{symbol}.parquet"
                    df.to_parquet(processed_file, index=False)
                    
                    # Update metadata database
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO stooq_symbols 
                        (symbol, filename, exchange, security_type, first_date, last_date, 
                         total_records, file_size, last_updated, data_quality_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, filename, exchange, security_type,
                        df.iloc[0]['date'].strftime('%Y-%m-%d'),
                        df.iloc[-1]['date'].strftime('%Y-%m-%d'),
                        len(df), file_path.stat().st_size,
                        datetime.now().isoformat(), quality_score
                    ))
                    
                    processed_count += 1
                    
                    # Progress logging
                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count}/{len(data_files)} files...")
                    
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {str(e)}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully processed {processed_count} symbols from {dataset.name}")
            return processed_count
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_key}: {str(e)}")
            return 0
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate a data quality score (0-100) for a dataset"""
        try:
            if df.empty:
                return 0.0
            
            score = 100.0
            
            # Penalize missing data
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            score -= missing_ratio * 30
            
            # Penalize zero volumes
            zero_volume_ratio = len(df[df['volume'] == 0]) / len(df)
            score -= zero_volume_ratio * 20
            
            # Penalize identical consecutive prices (stale data)
            if len(df) > 1:
                price_changes = df['close'].diff().fillna(0)
                no_change_ratio = len(price_changes[price_changes == 0]) / len(df)
                if no_change_ratio > 0.5:  # More than 50% no price change
                    score -= (no_change_ratio - 0.1) * 30
            
            # Penalize extreme price jumps (>50% daily changes)
            if len(df) > 1:
                price_changes = df['close'].pct_change().fillna(0)
                extreme_changes = len(price_changes[abs(price_changes) > 0.5])
                if extreme_changes > 0:
                    score -= min(extreme_changes / len(df) * 100, 20)
            
            # Bonus for longer time series
            if len(df) > 1000:  # More than ~4 years of data
                score += 5
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {str(e)}")
            return 50.0  # Default score
    
    def get_symbol_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get historical data for a symbol"""
        try:
            symbol = symbol.upper()
            processed_file = self.processed_dir / f"{symbol}.parquet"
            
            if not processed_file.exists():
                logger.warning(f"No Stooq data found for {symbol}")
                return pd.DataFrame()
            
            df = pd.read_parquet(processed_file)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Filter by date range
            if start_date:
                df = df[df['date'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['date'] <= pd.to_datetime(end_date)]
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading Stooq data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_available_symbols(self, exchange: str = None, security_type: str = None, 
                            min_quality_score: float = 0.0) -> List[Dict]:
        """Get list of available symbols with metadata"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM stooq_symbols WHERE data_quality_score >= ?"
            params = [min_quality_score]
            
            if exchange:
                query += " AND exchange = ?"
                params.append(exchange)
            
            if security_type:
                query += " AND security_type = ?"
                params.append(security_type)
            
            query += " ORDER BY data_quality_score DESC, total_records DESC"
            
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error getting available symbols: {str(e)}")
            return []
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics about the Stooq data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM stooq_symbols")
            total_symbols = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(total_records) FROM stooq_symbols")
            total_records = cursor.fetchone()[0] or 0
            
            # By security type
            cursor.execute("SELECT security_type, COUNT(*) FROM stooq_symbols GROUP BY security_type")
            by_type = dict(cursor.fetchall())
            
            # By exchange
            cursor.execute("SELECT exchange, COUNT(*) FROM stooq_symbols GROUP BY exchange")
            by_exchange = dict(cursor.fetchall())
            
            # Quality distribution
            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN data_quality_score >= 90 THEN 1 END) as excellent,
                    COUNT(CASE WHEN data_quality_score >= 70 AND data_quality_score < 90 THEN 1 END) as good,
                    COUNT(CASE WHEN data_quality_score >= 50 AND data_quality_score < 70 THEN 1 END) as fair,
                    COUNT(CASE WHEN data_quality_score < 50 THEN 1 END) as poor
                FROM stooq_symbols
            """)
            quality_dist = cursor.fetchone()
            
            # Date ranges
            cursor.execute("SELECT MIN(first_date), MAX(last_date) FROM stooq_symbols")
            date_range = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_symbols': total_symbols,
                'total_records': total_records,
                'by_security_type': by_type,
                'by_exchange': by_exchange,
                'quality_distribution': {
                    'excellent_90+': quality_dist[0],
                    'good_70-89': quality_dist[1], 
                    'fair_50-69': quality_dist[2],
                    'poor_<50': quality_dist[3]
                },
                'date_range': {
                    'earliest': date_range[0],
                    'latest': date_range[1]
                },
                'avg_records_per_symbol': total_records / total_symbols if total_symbols > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting summary stats: {str(e)}")
            return {}
    
    def download_and_process_all(self, force_redownload: bool = False) -> Dict:
        """Download and process all available datasets"""
        results = {}
        
        logger.info("Starting complete Stooq data download and processing...")
        
        for dataset_key in self.DATASETS:
            logger.info(f"Processing dataset: {dataset_key}")
            
            # Download
            download_success = self.download_dataset(dataset_key, force_redownload)
            if not download_success:
                results[dataset_key] = {'download': False, 'processed': 0}
                continue
            
            # Process
            processed_count = self.process_stooq_files(dataset_key)
            results[dataset_key] = {'download': True, 'processed': processed_count}
            
            logger.info(f"Completed {dataset_key}: {processed_count} symbols processed")
        
        # Generate final summary
        summary = self.get_summary_stats()
        results['summary'] = summary
        
        logger.info(f"Stooq download and processing completed!")
        logger.info(f"Total symbols: {summary.get('total_symbols', 0)}")
        logger.info(f"Total records: {summary.get('total_records', 0):,}")
        
        return results