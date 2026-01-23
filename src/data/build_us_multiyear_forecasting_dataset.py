"""
Build US Multi-Year Forecasting Datasets (Year t → Year t+n).

This script creates forecasting datasets for multiple time horizons:
- 1-year: t → t+1 (short-term, high accuracy)
- 2-year: t → t+2 (medium-term)
- 3-year: t → t+3 (long-term)
- 4-year: t → t+4 (extended long-term)

This enables reliable long-range forecasting by training separate models
for each time horizon instead of extrapolating beyond training distribution.

Outputs:
- datasets/us_forecasting_1yr.csv
- datasets/us_forecasting_2yr.csv
- datasets/us_forecasting_3yr.csv
- datasets/us_forecasting_4yr.csv

Usage:
    python src/data/build_us_multiyear_forecasting_dataset.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Input path
PANEL_PATH = Path("datasets/us_panel_dataset.csv")

# Output directory
OUTPUT_DIR = Path("datasets")

# Features to use for forecasting (from year t)
FEATURE_COLUMNS = [
    'n_returns',
    'total_wages',
    'n_wages',
    'total_interest',
    'n_interest',
    'total_cap_gains',
    'n_cap_gains',
    'total_business',
    'n_business',
    'avg_agi',
    'pct_with_wages',
    'avg_wage_per_earner',
    'pct_with_business',
    'pct_with_cap_gains',
    'log_returns'
]

# Target variable
TARGET_COLUMN = 'avg_agi'


def build_forecasting_pairs_with_gap(panel_df: pd.DataFrame, year_gap: int) -> pd.DataFrame:
    """
    Create (t, t+n) pairs for each ZIP code.
    
    Args:
        panel_df: Panel dataset with ZIP-year observations
        year_gap: Number of years to forecast ahead (1, 2, 3, or 4)
    
    Returns:
        DataFrame with features from year t and target from year t+n
    """
    print(f"\nBuilding {year_gap}-year forecasting pairs...")
    
    # Sort by ZIP and year
    panel_df = panel_df.sort_values(['zip', 'year']).reset_index(drop=True)
    
    # Get unique ZIPs
    zips = panel_df['zip'].unique()
    print(f"Processing {len(zips):,} unique ZIP codes")
    
    forecast_pairs = []
    pairs_by_transition = {}
    
    for zip_code in zips:
        # Get all years for this ZIP
        zip_data = panel_df[panel_df['zip'] == zip_code].sort_values('year')
        years = zip_data['year'].values
        
        # Create pairs: (year_t, year_t+n)
        for i in range(len(zip_data)):
            year_t = years[i]
            year_t_plus_n = year_t + year_gap
            
            # Check if we have data for year_t+n
            if year_t_plus_n in years:
                row_t = zip_data[zip_data['year'] == year_t].iloc[0]
                row_t_plus_n = zip_data[zip_data['year'] == year_t_plus_n].iloc[0]
                
                pair = {
                    'zip': zip_code,
                    'state': row_t['state'],
                    'from_year': int(year_t),
                    'to_year': int(year_t_plus_n),
                    'year_gap': year_gap,
                }
                
                # Add features from year t
                for col in FEATURE_COLUMNS:
                    if col in row_t.index:
                        pair[f'{col}_t'] = row_t[col]
                
                # Add target from year t+n
                if TARGET_COLUMN in row_t_plus_n.index:
                    pair['target_income_tplusn'] = row_t_plus_n[TARGET_COLUMN]
                
                forecast_pairs.append(pair)
                
                # Track transitions
                transition = f"{year_t}→{year_t_plus_n}"
                pairs_by_transition[transition] = pairs_by_transition.get(transition, 0) + 1
    
    forecast_df = pd.DataFrame(forecast_pairs)
    
    print(f"Created {len(forecast_df):,} forecasting pairs")
    print(f"\nTransitions:")
    for transition, count in sorted(pairs_by_transition.items()):
        print(f"  {transition}: {count:,} pairs")
    
    return forecast_df


def compute_time_based_splits(forecast_df: pd.DataFrame, year_gap: int):
    """
    Compute time-based train/test splits.
    
    Strategy: Hold out the most recent transition for testing
    
    For 1-year gaps:
        Train: 2019→2020, 2020→2021
        Test:  2021→2022
    
    For 2-year gaps:
        Train: 2019→2021, 2020→2022
        Test:  Would be 2021→2023 (but we only have up to 2022)
        So use: Train: 2019→2021, Test: 2020→2022
    
    For 3-year gaps:
        Train: 2019→2022
        Test:  Would need 2020→2023 or later (not available)
        Use all for training for now
    """
    print(f"\nComputing time-based splits for {year_gap}-year gap...")
    
    # Get unique to_years
    to_years = sorted(forecast_df['to_year'].unique())
    print(f"Target years available: {to_years}")
    
    if len(to_years) < 2:
        # Not enough data for train/test split
        print(f"[WARNING] Only {len(to_years)} target year(s), using all data for training")
        forecast_df['split'] = 'train'
        test_count = 0
        train_count = len(forecast_df)
    else:
        # Use most recent target year for testing
        test_year = max(to_years)
        train_df = forecast_df[forecast_df['to_year'] < test_year].copy()
        test_df = forecast_df[forecast_df['to_year'] == test_year].copy()
        
        train_df['split'] = 'train'
        test_df['split'] = 'test'
        
        forecast_df = pd.concat([train_df, test_df], ignore_index=True)
        
        train_count = len(train_df)
        test_count = len(test_df)
        
        print(f"Train set: {train_count:,} pairs (target years: {sorted(train_df['to_year'].unique())})")
        print(f"Test set:  {test_count:,} pairs (target year: {test_year})")
    
    return forecast_df, train_count, test_count


def save_metadata(output_path: Path, metadata: dict):
    """Save metadata JSON file alongside dataset."""
    metadata_path = output_path.with_suffix('.metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")


def main():
    print("="*70)
    print("BUILDING US MULTI-YEAR FORECASTING DATASETS")
    print("="*70)
    
    # Check if panel dataset exists
    if not PANEL_PATH.exists():
        print(f"\n[ERROR] Panel dataset not found: {PANEL_PATH}")
        print("Please run: python src/data/build_us_panel_dataset.py")
        sys.exit(1)
    
    # Load panel dataset
    print(f"\nLoading panel dataset...")
    panel_df = pd.read_csv(PANEL_PATH)
    
    print(f"Panel dataset: {panel_df.shape}")
    print(f"Years: {sorted(panel_df['year'].unique())}")
    print(f"ZIP codes: {panel_df['zip'].nunique():,}")
    
    # Build datasets for each time horizon
    year_gaps = [1, 2, 3, 4]
    
    results_summary = []
    
    for year_gap in year_gaps:
        print("\n" + "="*70)
        print(f"BUILDING {year_gap}-YEAR FORECASTING DATASET")
        print("="*70)
        
        # Build pairs
        forecast_df = build_forecasting_pairs_with_gap(panel_df, year_gap)
        
        if len(forecast_df) == 0:
            print(f"[WARNING] No pairs created for {year_gap}-year gap")
            continue
        
        # Compute splits
        forecast_df, train_count, test_count = compute_time_based_splits(forecast_df, year_gap)
        
        # Validate
        print("\nValidation:")
        print(f"  Total pairs: {len(forecast_df):,}")
        print(f"  Missing target values: {forecast_df['target_income_tplusn'].isna().sum()}")
        print(f"  Year gap consistency: {(forecast_df['to_year'] - forecast_df['from_year'] == year_gap).all()}")
        
        # Save
        output_path = OUTPUT_DIR / f"us_forecasting_{year_gap}yr.csv"
        forecast_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved: {output_path}")
        print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Save metadata
        metadata = {
            'year_gap': int(year_gap),
            'total_pairs': int(len(forecast_df)),
            'train_pairs': int(train_count),
            'test_pairs': int(test_count),
            'num_zips': int(forecast_df['zip'].nunique()),
            'num_states': int(forecast_df['state'].nunique()),
            'from_years': [int(y) for y in sorted(forecast_df['from_year'].unique())],
            'to_years': [int(y) for y in sorted(forecast_df['to_year'].unique())],
            'features': [col for col in forecast_df.columns if col.endswith('_t')],
            'target_column': 'target_income_tplusn',
            'file_size_mb': round(output_path.stat().st_size / 1024 / 1024, 2)
        }
        save_metadata(output_path, metadata)
        
        # Store summary
        results_summary.append({
            'year_gap': year_gap,
            'total_pairs': len(forecast_df),
            'train_pairs': train_count,
            'test_pairs': test_count,
            'file': output_path.name
        })
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Gap':<8} {'Total Pairs':<15} {'Train':<15} {'Test':<15} {'File':<30}")
    print("-" * 80)
    for result in results_summary:
        print(f"{result['year_gap']}-year  {result['total_pairs']:>12,}  {result['train_pairs']:>12,}  {result['test_pairs']:>12,}  {result['file']:<30}")
    
    print("\n✅ All multi-year forecasting datasets created successfully!")
    print("\nNext steps:")
    print("  1. Train models: python src/training/train_us_multiyear_forecast_models.py")
    print("  2. Update backend to use appropriate model based on year gap")
    print("  3. Update frontend to allow predictions up to 2030")


if __name__ == "__main__":
    main()
