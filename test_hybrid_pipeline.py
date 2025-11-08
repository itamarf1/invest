#!/usr/bin/env python3

"""
Test the hybrid data pipeline with all three sources
"""

import sys
import os
import logging
import asyncio
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.hybrid_data_pipeline import HybridDataPipeline, DataRequest, DataSourcePriority

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Test the hybrid data pipeline"""
    
    print("🚀 Testing Hybrid Data Pipeline")
    print("=" * 50)
    print("This tests intelligent data source selection with automatic quality validation")
    
    # Initialize pipeline
    print("\n📦 Initializing pipeline...")
    pipeline = HybridDataPipeline()
    
    # Show initial stats
    stats = pipeline.get_pipeline_stats()
    print(f"✅ Pipeline initialized!")
    print(f"   Source availability:")
    for source, available in stats['source_availability'].items():
        print(f"     {source}: {'✓' if available else '✗'}")
    
    # Test 1: High-quality stock (AAPL)
    print(f"\n🍎 Test 1: High-quality stock (AAPL)")
    
    aapl_request = DataRequest(
        symbol="AAPL",
        start_date="2024-10-01",
        end_date="2024-11-08",
        interval="1d",
        include_validation=True
    )
    
    aapl_response = await pipeline.get_data(aapl_request)
    
    print(f"   Result: {len(aapl_response.data)} records from {aapl_response.source}")
    print(f"   Quality score: {aapl_response.quality_score:.1f}/100")
    
    if aapl_response.validation_result:
        correlations = aapl_response.validation_result.price_correlation
        if correlations:
            print(f"   Cross-validation correlations:")
            for pair, corr in correlations.items():
                print(f"     {pair}: {corr:.4f}")
    
    # Test 2: Problematic warrant (ARQQW)
    print(f"\n🚨 Test 2: Problematic warrant (ARQQW)")
    
    arqqw_request = DataRequest(
        symbol="ARQQW",
        start_date="2024-10-01", 
        end_date="2024-11-08",
        interval="1d",
        include_validation=True
    )
    
    arqqw_response = await pipeline.get_data(arqqw_request)
    
    print(f"   Result: {len(arqqw_response.data)} records from {arqqw_response.source}")
    print(f"   Quality score: {arqqw_response.quality_score:.1f}/100")
    
    if not arqqw_response.data.empty:
        recent_data = arqqw_response.data.tail(5)
        recent_prices = recent_data['close'].tolist()
        unique_prices = recent_data['close'].nunique()
        avg_volume = recent_data['volume'].mean()
        
        print(f"   Recent analysis:")
        print(f"     Last 5 closes: {recent_prices}")
        print(f"     Unique prices: {unique_prices}")
        print(f"     Average volume: {avg_volume:,.0f}")
        
        if unique_prices == 1:
            print(f"     ⚠️  WARNING: Detected stale data!")
        else:
            print(f"     ✅ Good: Active price movement detected")
    
    # Test 3: Batch processing
    print(f"\n📦 Test 3: Batch processing")
    
    batch_symbols = ["AAPL", "MSFT", "SPY", "GOOGL", "ARQQW"]
    batch_requests = []
    
    for symbol in batch_symbols:
        request = DataRequest(
            symbol=symbol,
            start_date="2024-11-01",
            end_date="2024-11-08",
            interval="1d",
            include_validation=False  # Skip validation for speed
        )
        batch_requests.append(request)
    
    print(f"   Processing {len(batch_requests)} symbols...")
    batch_responses = await pipeline.batch_get_data(batch_requests)
    
    print(f"   Batch results:")
    for response in batch_responses:
        status = "✅" if not response.data.empty else "❌"
        print(f"     {status} {response.symbol}: {len(response.data)} records from {response.source} "
              f"(quality: {response.quality_score:.1f})")
    
    # Test 4: Source priority customization
    print(f"\n⚙️  Test 4: Custom source priority")
    
    # Configure to prefer Stooq for warrants
    custom_config = DataSourcePriority(
        primary_source="stooq",
        fallback_sources=["yahoo", "alpha_vantage"],
        quality_threshold=80.0,
        enable_cross_validation=True
    )
    
    pipeline.configure(custom_config)
    
    # Test ARQQW again with custom config
    arqqw_custom_response = await pipeline.get_data(arqqw_request)
    
    print(f"   ARQQW with Stooq priority:")
    print(f"     Source used: {arqqw_custom_response.source}")
    print(f"     Quality: {arqqw_custom_response.quality_score:.1f}/100")
    print(f"     Records: {len(arqqw_custom_response.data)}")
    
    # Test 5: Pipeline statistics
    print(f"\n📊 Test 5: Pipeline performance statistics")
    
    final_stats = pipeline.get_pipeline_stats()
    
    print(f"   Configuration:")
    print(f"     Primary source: {final_stats['configuration']['primary_source']}")
    print(f"     Quality threshold: {final_stats['configuration']['quality_threshold']}")
    print(f"     Cross-validation: {final_stats['configuration']['enable_cross_validation']}")
    
    print(f"   Source reliability:")
    for source, reliability in final_stats.get('source_reliability', {}).items():
        if reliability['samples'] > 0:
            print(f"     {source}: {reliability['average_quality']:.1f}/100 "
                  f"(recent: {reliability['recent_quality']:.1f}, "
                  f"samples: {reliability['samples']})")
    
    if 'alpha_vantage_usage' in final_stats:
        av_usage = final_stats['alpha_vantage_usage']
        print(f"   Alpha Vantage usage:")
        print(f"     Requests today: {av_usage['total_requests_today']}/{av_usage['daily_limit']}")
        print(f"     Remaining: {av_usage['remaining_requests']}")
    
    # Test 6: Error handling
    print(f"\n🔧 Test 6: Error handling")
    
    invalid_request = DataRequest(
        symbol="INVALID_SYMBOL_123",
        start_date="2024-11-01",
        end_date="2024-11-08"
    )
    
    error_response = await pipeline.get_data(invalid_request)
    
    print(f"   Invalid symbol result:")
    print(f"     Source: {error_response.source}")
    print(f"     Quality: {error_response.quality_score}")
    print(f"     Records: {len(error_response.data)}")
    print(f"     Error handled: {'✅' if 'error' in error_response.metadata else '❌'}")
    
    # Final summary
    print(f"\n🎯 Summary & Recommendations:")
    
    # Analyze which source worked best for different symbol types
    source_usage = {}
    quality_by_source = {}
    
    all_responses = batch_responses + [aapl_response, arqqw_response, arqqw_custom_response]
    
    for response in all_responses:
        if response.source not in source_usage:
            source_usage[response.source] = 0
            quality_by_source[response.source] = []
        
        source_usage[response.source] += 1
        quality_by_source[response.source].append(response.quality_score)
    
    print(f"   Source usage in tests:")
    for source, count in source_usage.items():
        if source in quality_by_source and quality_by_source[source]:
            avg_quality = sum(quality_by_source[source]) / len(quality_by_source[source])
            print(f"     {source}: {count} requests, avg quality {avg_quality:.1f}/100")
    
    print(f"\n✅ Hybrid pipeline testing completed!")
    print(f"   Your system now has intelligent multi-source data with quality validation!")
    print(f"   Key benefits:")
    print(f"     🎯 Automatic source selection based on symbol characteristics")
    print(f"     📊 Quality scoring and validation")
    print(f"     🔄 Smart fallback when primary sources have issues")
    print(f"     ⚡ Efficient batch processing")
    print(f"     📈 Performance monitoring and optimization")

if __name__ == "__main__":
    asyncio.run(main())