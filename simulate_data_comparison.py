#!/usr/bin/env python3

"""
Simulate data source comparison to show what real validation would look like
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.data_source_validator import DataSourceValidator, DataQuality, ValidationResult

def simulate_alpha_vantage_data(yahoo_data: pd.DataFrame, add_noise: bool = True) -> pd.DataFrame:
    """Simulate what Alpha Vantage data might look like based on Yahoo data"""
    if yahoo_data.empty:
        return pd.DataFrame()
    
    simulated = yahoo_data.copy()
    
    if add_noise:
        # Add small random variations to simulate different data sources
        # Typical inter-source differences are 0.01-0.1%
        noise_level = 0.001  # 0.1% noise
        
        for col in ['open', 'high', 'low', 'close']:
            noise = np.random.normal(0, noise_level, len(simulated))
            simulated[col] = simulated[col] * (1 + noise)
        
        # Volume might have slightly different values
        volume_noise = np.random.normal(0, 0.02, len(simulated))  # 2% noise
        simulated['volume'] = simulated['volume'] * (1 + volume_noise)
        simulated['volume'] = simulated['volume'].round().astype(int)
    
    simulated['source'] = 'alpha_vantage_sim'
    return simulated

def simulate_stooq_data(yahoo_data: pd.DataFrame, add_quality_issues: bool = False) -> pd.DataFrame:
    """Simulate Stooq data with potential quality issues"""
    if yahoo_data.empty:
        return pd.DataFrame()
    
    simulated = yahoo_data.copy()
    
    if add_quality_issues:
        # Simulate some common data quality issues
        
        # 1. Some missing days (5% random missing)
        missing_mask = np.random.random(len(simulated)) < 0.05
        simulated = simulated[~missing_mask]
        
        # 2. Some zero volume days (2% random)
        zero_vol_mask = np.random.random(len(simulated)) < 0.02
        simulated.loc[zero_vol_mask, 'volume'] = 0
        
        # 3. Occasional stale prices (prices don't change for 2-3 days)
        if len(simulated) > 10:
            stale_start = np.random.randint(5, len(simulated) - 5)
            stale_end = min(stale_start + 3, len(simulated))
            stale_price = simulated.iloc[stale_start]['close']
            simulated.iloc[stale_start:stale_end, simulated.columns.get_loc('close')] = stale_price
    
    simulated['source'] = 'stooq_sim'
    return simulated

def create_problematic_data(yahoo_data: pd.DataFrame) -> pd.DataFrame:
    """Create data with significant quality issues (like ARQQW)"""
    if yahoo_data.empty:
        return pd.DataFrame()
    
    # Start with real data but introduce major issues
    problematic = yahoo_data.copy()
    
    # Make price completely stale for last 30% of period
    stale_point = int(len(problematic) * 0.7)
    stale_price = problematic.iloc[stale_point]['close']
    
    # Set all prices to the same value
    problematic.iloc[stale_point:, problematic.columns.get_loc('open')] = stale_price
    problematic.iloc[stale_point:, problematic.columns.get_loc('high')] = stale_price
    problematic.iloc[stale_point:, problematic.columns.get_loc('low')] = stale_price
    problematic.iloc[stale_point:, problematic.columns.get_loc('close')] = stale_price
    
    # Zero volume for stale period
    problematic.iloc[stale_point:, problematic.columns.get_loc('volume')] = 0
    
    problematic['source'] = 'yahoo_problematic'
    return problematic

def main():
    """Simulate comprehensive data source comparison"""
    
    print("🎭 SIMULATED Data Source Comparison")
    print("=" * 50)
    print("This shows what real multi-source validation would look like")
    print()
    
    validator = DataSourceValidator()
    
    # Test with AAPL (should be high quality)
    print("📊 Simulating AAPL comparison (high-quality stock)...")
    
    # Get real Yahoo data
    yahoo_data = validator.get_yahoo_data('AAPL', '2025-09-01', '2025-11-08')
    
    if not yahoo_data.empty:
        # Simulate other sources
        av_data = simulate_alpha_vantage_data(yahoo_data, add_noise=True)
        stooq_data = simulate_stooq_data(yahoo_data, add_quality_issues=False)
        
        # Assess quality of each
        yahoo_quality = validator.assess_data_quality(yahoo_data, 'AAPL', 'yahoo')
        av_quality = validator.assess_data_quality(av_data, 'AAPL', 'alpha_vantage')
        stooq_quality = validator.assess_data_quality(stooq_data, 'AAPL', 'stooq')
        
        print(f"\nSource Quality Comparison for AAPL:")
        print(f"  Yahoo Finance:")
        print(f"    Records: {yahoo_quality.total_records}")
        print(f"    Reliability: {yahoo_quality.reliability_score:.1f}/100")
        print(f"    Completeness: {yahoo_quality.data_completeness:.1f}%")
        print(f"    Price range: ${yahoo_quality.price_range[0]:.2f} - ${yahoo_quality.price_range[1]:.2f}")
        
        print(f"  Alpha Vantage (simulated):")
        print(f"    Records: {av_quality.total_records}")
        print(f"    Reliability: {av_quality.reliability_score:.1f}/100")
        print(f"    Completeness: {av_quality.data_completeness:.1f}%")
        print(f"    Price range: ${av_quality.price_range[0]:.2f} - ${av_quality.price_range[1]:.2f}")
        
        print(f"  Stooq (simulated):")
        print(f"    Records: {stooq_quality.total_records}")
        print(f"    Reliability: {stooq_quality.reliability_score:.1f}/100")
        print(f"    Completeness: {stooq_quality.data_completeness:.1f}%")
        print(f"    Price range: ${stooq_quality.price_range[0]:.2f} - ${stooq_quality.price_range[1]:.2f}")
        
        # Calculate correlations
        common_dates = yahoo_data.index.intersection(av_data.index)
        if len(common_dates) > 10:
            price_corr = yahoo_data.loc[common_dates, 'close'].corr(av_data.loc[common_dates, 'close'])
            volume_corr = yahoo_data.loc[common_dates, 'volume'].corr(av_data.loc[common_dates, 'volume'])
            
            print(f"\nCorrelation Analysis:")
            print(f"  Yahoo vs Alpha Vantage price correlation: {price_corr:.4f}")
            print(f"  Yahoo vs Alpha Vantage volume correlation: {volume_corr:.4f}")
            
            # Look for discrepancies
            price_diff = (yahoo_data.loc[common_dates, 'close'] - av_data.loc[common_dates, 'close']).abs() / yahoo_data.loc[common_dates, 'close']
            large_diffs = price_diff[price_diff > 0.005]  # >0.5% difference
            
            print(f"  Price discrepancies (>0.5%): {len(large_diffs)}")
            if len(large_diffs) > 0:
                print(f"    Max discrepancy: {large_diffs.max()*100:.2f}%")
    
    # Test with problematic case (like ARQQW)
    print(f"\n🚨 Simulating ARQQW comparison (problematic data)...")
    
    # Get real ARQQW data
    arqqw_yahoo = validator.get_yahoo_data('ARQQW', '2025-10-01', '2025-11-08')
    
    if not arqqw_yahoo.empty:
        # Create a "better" version simulating what Alpha Vantage might have
        better_data = arqqw_yahoo.copy()
        
        # Simulate Alpha Vantage having more recent/accurate prices
        if len(better_data) > 10:
            # Make the last 50% have slightly higher, more realistic prices
            recent_point = int(len(better_data) * 0.5)
            base_price = better_data.iloc[recent_point-1]['close']
            
            # Create more realistic price movement (similar to user's observation that real price should be higher)
            dates_to_fix = better_data.index[recent_point:]
            price_trend = np.linspace(base_price, base_price * 1.58, len(dates_to_fix))  # Trend up to ~$0.585
            
            for i, date in enumerate(dates_to_fix):
                new_price = price_trend[i] + np.random.normal(0, 0.01)  # Add small random variation
                better_data.loc[date, 'close'] = new_price
                better_data.loc[date, 'open'] = new_price * (1 + np.random.normal(0, 0.005))
                better_data.loc[date, 'high'] = max(better_data.loc[date, 'open'], new_price) * (1 + abs(np.random.normal(0, 0.01)))
                better_data.loc[date, 'low'] = min(better_data.loc[date, 'open'], new_price) * (1 - abs(np.random.normal(0, 0.01)))
                better_data.loc[date, 'volume'] = max(1000, int(np.random.normal(5000, 2000)))
        
        better_data['source'] = 'alpha_vantage_better'
        
        # Assess both datasets
        yahoo_arqqw_quality = validator.assess_data_quality(arqqw_yahoo, 'ARQQW', 'yahoo')
        better_arqqw_quality = validator.assess_data_quality(better_data, 'ARQQW', 'alpha_vantage')
        
        print(f"\nSource Quality Comparison for ARQQW:")
        print(f"  Yahoo Finance (actual):")
        print(f"    Records: {yahoo_arqqw_quality.total_records}")
        print(f"    Reliability: {yahoo_arqqw_quality.reliability_score:.1f}/100")
        print(f"    Zero volume days: {yahoo_arqqw_quality.zero_volume_days}")
        print(f"    Price range: ${yahoo_arqqw_quality.price_range[0]:.3f} - ${yahoo_arqqw_quality.price_range[1]:.3f}")
        
        print(f"  Alpha Vantage (simulated better data):")
        print(f"    Records: {better_arqqw_quality.total_records}")
        print(f"    Reliability: {better_arqqw_quality.reliability_score:.1f}/100")
        print(f"    Zero volume days: {better_arqqw_quality.zero_volume_days}")
        print(f"    Price range: ${better_arqqw_quality.price_range[0]:.3f} - ${better_arqqw_quality.price_range[1]:.3f}")
        
        # Calculate discrepancies
        common_dates = arqqw_yahoo.index.intersection(better_data.index)
        if len(common_dates) > 5:
            price_diff = (arqqw_yahoo.loc[common_dates, 'close'] - better_data.loc[common_dates, 'close']).abs() / arqqw_yahoo.loc[common_dates, 'close']
            large_diffs = price_diff[price_diff > 0.05]  # >5% difference
            
            print(f"\nDiscrepancy Analysis:")
            print(f"  Large price differences (>5%): {len(large_diffs)}")
            if len(large_diffs) > 0:
                print(f"    Max discrepancy: {large_diffs.max()*100:.1f}%")
                print(f"    Average of large discrepancies: {large_diffs.mean()*100:.1f}%")
                
                # Show some specific examples
                print(f"    Recent examples:")
                for date in large_diffs.index[-3:]:  # Last 3 discrepancies
                    yahoo_price = arqqw_yahoo.loc[date, 'close']
                    better_price = better_data.loc[date, 'close']
                    diff_pct = abs(yahoo_price - better_price) / yahoo_price * 100
                    print(f"      {date.strftime('%Y-%m-%d')}: Yahoo ${yahoo_price:.3f} vs Better ${better_price:.3f} ({diff_pct:.1f}% diff)")
        
        print(f"\n💡 Recommendation:")
        if better_arqqw_quality.reliability_score > yahoo_arqqw_quality.reliability_score:
            print(f"  Use Alpha Vantage for ARQQW (reliability: {better_arqqw_quality.reliability_score:.1f} vs {yahoo_arqqw_quality.reliability_score:.1f})")
            print(f"  Yahoo Finance appears to have stale data for this warrant")
        else:
            print(f"  Yahoo Finance is adequate for ARQQW")
    
    print(f"\n📋 Summary & Next Steps:")
    print(f"✅ For popular stocks (AAPL, MSFT, etc.): Yahoo Finance is very reliable")
    print(f"⚠️  For warrants/options (ARQQW, etc.): Need multiple sources for validation")
    print(f"🔄 Recommended approach:")
    print(f"   1. Use Yahoo Finance as primary source")
    print(f"   2. Get Alpha Vantage API key for cross-validation")
    print(f"   3. Download Stooq bulk data for historical backfill")
    print(f"   4. Flag symbols with low reliability scores for manual review")

if __name__ == "__main__":
    main()