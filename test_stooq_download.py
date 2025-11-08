#!/usr/bin/env python3

"""
Test script to download and process Stooq historical data
"""

import sys
import os
import logging
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.stooq_downloader import StooqDownloader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test Stooq data download and processing"""
    
    print("📥 Testing Stooq Historical Data Download")
    print("=" * 50)
    
    # Initialize downloader
    downloader = StooqDownloader()
    
    # Check current data status
    print("\n📊 Current data status...")
    summary = downloader.get_summary_stats()
    
    if summary.get('total_symbols', 0) > 0:
        print(f"  Existing data found:")
        print(f"    Total symbols: {summary['total_symbols']:,}")
        print(f"    Total records: {summary['total_records']:,}")
        print(f"    Date range: {summary['date_range']['earliest']} to {summary['date_range']['latest']}")
        print(f"    Quality distribution:")
        for quality, count in summary['quality_distribution'].items():
            print(f"      {quality}: {count}")
    else:
        print("  No existing Stooq data found")
    
    # Ask user if they want to proceed with download
    print(f"\n⚠️  Note: This will download ~350MB of data from Stooq")
    print(f"   US Stocks dataset: ~3,600 symbols")
    print(f"   US ETF dataset: ~2,700 symbols")
    
    # For testing, let's just download the US stocks dataset
    print(f"\n🚀 Starting download of US stocks dataset...")
    
    # Download US stocks
    success = downloader.download_dataset('us_stocks', force_redownload=False)
    
    if success:
        print(f"✅ Download completed successfully!")
        
        # Process the downloaded data
        print(f"\n🔄 Processing downloaded files...")
        processed_count = downloader.process_stooq_files('us_stocks')
        
        print(f"✅ Processed {processed_count} symbols")
        
        # Get updated summary
        print(f"\n📈 Updated data summary...")
        summary = downloader.get_summary_stats()
        
        print(f"  Total symbols: {summary.get('total_symbols', 0):,}")
        print(f"  Total records: {summary.get('total_records', 0):,}")
        print(f"  Average records per symbol: {summary.get('avg_records_per_symbol', 0):.0f}")
        
        print(f"\n  By security type:")
        for sec_type, count in summary.get('by_security_type', {}).items():
            print(f"    {sec_type}: {count:,}")
        
        print(f"\n  Data quality distribution:")
        for quality, count in summary.get('quality_distribution', {}).items():
            print(f"    {quality}: {count:,}")
        
        # Test retrieving some specific symbols
        print(f"\n🔍 Testing data retrieval...")
        
        test_symbols = ['AAPL', 'MSFT', 'SPY', 'ARQQW']
        
        for symbol in test_symbols:
            data = downloader.get_symbol_data(symbol)
            
            if not data.empty:
                print(f"  {symbol}: {len(data):,} records "
                      f"({data.iloc[0]['date'].strftime('%Y-%m-%d')} to "
                      f"{data.iloc[-1]['date'].strftime('%Y-%m-%d')})")
                
                # Check ARQQW data quality specifically
                if symbol == 'ARQQW':
                    recent_data = data.tail(10)
                    unique_prices = recent_data['close'].nunique()
                    avg_volume = recent_data['volume'].mean()
                    
                    print(f"    ARQQW recent analysis:")
                    print(f"      Unique prices in last 10 days: {unique_prices}")
                    print(f"      Average volume: {avg_volume:,.0f}")
                    print(f"      Price range: ${recent_data['close'].min():.3f} - ${recent_data['close'].max():.3f}")
            else:
                print(f"  {symbol}: No data found")
        
        # Get high-quality symbols
        print(f"\n⭐ High-quality symbols (score >= 90):")
        high_quality = downloader.get_available_symbols(min_quality_score=90.0)[:10]
        
        for symbol_info in high_quality:
            print(f"  {symbol_info['symbol']}: {symbol_info['data_quality_score']:.1f} "
                  f"({symbol_info['total_records']:,} records, {symbol_info['security_type']})")
        
        print(f"\n✅ Stooq data download and processing test completed!")
        
    else:
        print(f"❌ Download failed. Check network connection and try again.")
        return

if __name__ == "__main__":
    main()