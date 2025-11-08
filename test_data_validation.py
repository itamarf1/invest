#!/usr/bin/env python3

"""
Test script to validate data sources and compare their quality
"""

import sys
import os
import logging
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.data_source_validator import DataSourceValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test data source validation"""
    
    # Initialize validator (Alpha Vantage key optional for now)
    validator = DataSourceValidator()
    
    # Test symbols - mix of popular stocks and potential problem cases
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'ARQQW']
    
    print("🔍 Testing Data Source Validation System")
    print("=" * 50)
    
    # Test individual symbol comparison
    print(f"\n📊 Testing individual comparison for AAPL...")
    result = validator.compare_sources('AAPL', test_period_days=90)
    
    print(f"Sources found: {result.sources_compared}")
    print(f"Recommended source: {result.recommended_source}")
    print(f"Confidence: {result.confidence:.2%}")
    
    if result.price_correlation:
        print("Price correlations:")
        for pair, corr in result.price_correlation.items():
            print(f"  {pair}: {corr:.4f}")
    
    if result.quality_scores:
        print("\nQuality Scores:")
        for source, quality in result.quality_scores.items():
            print(f"  {source}:")
            print(f"    Records: {quality.total_records}")
            print(f"    Completeness: {quality.data_completeness:.1f}%")
            print(f"    Reliability: {quality.reliability_score:.1f}")
            print(f"    Zero volume days: {quality.zero_volume_days}")
    
    # Test batch validation with subset
    print(f"\n📈 Testing batch validation...")
    print(f"Symbols: {test_symbols}")
    
    batch_results = validator.batch_validate_symbols(test_symbols[:3], test_period_days=90)
    
    print(f"\nBatch Results Summary:")
    for symbol, result in batch_results.items():
        print(f"  {symbol}: {len(result.sources_compared)} sources, "
              f"recommended: {result.recommended_source}, "
              f"confidence: {result.confidence:.1%}")
    
    # Generate comprehensive report
    print(f"\n📋 Generating validation report...")
    report = validator.generate_validation_report(batch_results)
    
    print("\nValidation Report Summary:")
    if 'summary' in report:
        summary = report['summary']
        print(f"  Symbols tested: {summary['total_symbols_tested']}")
        print(f"  Data coverage: {summary['data_coverage_rate']:.1f}%")
    
    if 'source_statistics' in report:
        stats = report['source_statistics']
        print(f"  Source availability: {stats['availability']}")
        print(f"  Average reliability: {stats['average_reliability']}")
        print(f"  Recommendations: {stats['recommendations']}")
    
    if 'correlation_analysis' in report:
        corr = report['correlation_analysis']
        print(f"  Avg price correlation: {corr['average_price_correlation']:.4f}")
        print(f"  Avg volume correlation: {corr['average_volume_correlation']:.4f}")
    
    print("\n✅ Data validation test completed!")
    
    # Test ARQQW specifically (our problem case)
    print(f"\n🔍 Special test for ARQQW (known data quality issues)...")
    arqqw_result = validator.compare_sources('ARQQW', test_period_days=30)
    
    print(f"ARQQW Sources: {arqqw_result.sources_compared}")
    print(f"ARQQW Recommended: {arqqw_result.recommended_source}")
    
    for source, quality in arqqw_result.quality_scores.items():
        print(f"  {source} quality: {quality.reliability_score:.1f}, "
              f"records: {quality.total_records}, "
              f"price range: ${quality.price_range[0]:.3f} - ${quality.price_range[1]:.3f}")
    
    # Show any major discrepancies
    total_discrepancies = sum(len(discs) for discs in arqqw_result.discrepancies.values())
    if total_discrepancies > 0:
        print(f"  Major price discrepancies found: {total_discrepancies}")
        for source, discs in arqqw_result.discrepancies.items():
            if discs:
                print(f"    {source}: {len(discs)} discrepancies")
                for disc in discs[:3]:  # Show first 3
                    print(f"      {disc['date']}: {disc['price_diff_pct']:.1f}% diff "
                          f"(${disc['price_1']:.3f} vs ${disc['price_2']:.3f})")

if __name__ == "__main__":
    main()