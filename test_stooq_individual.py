#!/usr/bin/env python3

"""
Test script to download individual symbols from Stooq
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
    """Test individual symbol downloads from Stooq"""
    
    print("🎯 Testing Individual Stooq Symbol Downloads")
    print("=" * 50)
    
    # Initialize downloader
    downloader = StooqDownloader()
    
    # Test symbols including our problematic ARQQW
    test_symbols = ['AAPL', 'MSFT', 'SPY', 'ARQQW', 'GOOGL']
    
    print(f"📥 Testing download of {len(test_symbols)} symbols from Stooq...")
    print(f"Symbols: {', '.join(test_symbols)}")
    
    # Test single symbol first
    print(f"\n🔍 Testing single symbol download: AAPL")
    
    aapl_data = downloader.download_symbol_from_stooq('AAPL', start_date="20200101")
    
    if not aapl_data.empty:
        print(f"✅ AAPL: {len(aapl_data)} records downloaded")
        print(f"   Date range: {aapl_data.iloc[0]['date'].strftime('%Y-%m-%d')} to {aapl_data.iloc[-1]['date'].strftime('%Y-%m-%d')}")
        print(f"   Columns: {list(aapl_data.columns)}")
        print(f"   Recent prices: ${aapl_data.tail(3)['close'].tolist()}")
    else:
        print(f"❌ Failed to download AAPL data")
    
    # Test ARQQW specifically
    print(f"\n🚨 Testing problematic symbol: ARQQW")
    
    arqqw_data = downloader.download_symbol_from_stooq('ARQQW', start_date="20240101")
    
    if not arqqw_data.empty:
        print(f"✅ ARQQW: {len(arqqw_data)} records downloaded")
        print(f"   Date range: {arqqw_data.iloc[0]['date'].strftime('%Y-%m-%d')} to {arqqw_data.iloc[-1]['date'].strftime('%Y-%m-%d')}")
        
        # Analyze ARQQW data quality
        recent_data = arqqw_data.tail(10)
        unique_prices = recent_data['close'].nunique()
        avg_volume = recent_data['volume'].mean()
        price_range = (recent_data['close'].min(), recent_data['close'].max())
        
        print(f"   Recent analysis (last 10 days):")
        print(f"     Unique prices: {unique_prices}")
        print(f"     Price range: ${price_range[0]:.3f} - ${price_range[1]:.3f}")
        print(f"     Average volume: {avg_volume:,.0f}")
        print(f"     Last 5 closes: {recent_data['close'].tail(5).tolist()}")
        
        # Compare with what we see from Yahoo
        print(f"\n   🔄 Quality assessment:")
        quality_score = downloader._calculate_quality_score(arqqw_data)
        print(f"     Quality score: {quality_score:.1f}/100")
        
        if unique_prices <= 3:
            print(f"     ⚠️  WARNING: Very few unique prices - likely stale data")
        if avg_volume < 1000:
            print(f"     ⚠️  WARNING: Very low volume - may be illiquid")
        
    else:
        print(f"❌ Failed to download ARQQW data")
    
    # Test batch download
    print(f"\n📦 Testing batch download...")
    
    batch_symbols = ['AAPL', 'MSFT', 'SPY']  # Smaller batch for testing
    results = downloader.download_symbols_list(batch_symbols, start_date="20240101")
    
    print(f"Batch download completed: {len(results)} symbols successful")
    
    for symbol, data in results.items():
        print(f"  {symbol}: {len(data)} records")
    
    # Check database and get summary
    print(f"\n📊 Database summary after downloads...")
    summary = downloader.get_summary_stats()
    
    if summary:
        print(f"  Total symbols in database: {summary.get('total_symbols', 0)}")
        print(f"  Total records: {summary.get('total_records', 0):,}")
        
        if 'quality_distribution' in summary:
            print(f"  Quality distribution:")
            for quality, count in summary['quality_distribution'].items():
                print(f"    {quality}: {count}")
    
    # Get high-quality symbols
    available_symbols = downloader.get_available_symbols(min_quality_score=80.0)
    
    if available_symbols:
        print(f"\n⭐ High-quality symbols downloaded:")
        for symbol_info in available_symbols[:5]:  # Top 5
            print(f"  {symbol_info['symbol']}: {symbol_info['data_quality_score']:.1f}/100 "
                  f"({symbol_info['total_records']:,} records)")
    
    print(f"\n✅ Stooq individual download test completed!")
    
    # Final comparison suggestion
    if len(results) > 0:
        print(f"\n💡 Next steps:")
        print(f"   1. Compare this Stooq data with Yahoo Finance using our validation system")
        print(f"   2. For ARQQW: Check if Stooq has more recent/accurate data than Yahoo")
        print(f"   3. Set up Alpha Vantage API for third-source validation")

if __name__ == "__main__":
    main()