"""
Build US Forecasting Dataset (Year t → Year t+1).

This script transforms the panel dataset into a forecasting format where:
- Features come from year t
- Target (income) comes from year t+1

This enables true future-year prediction.

Output: datasets/us_forecasting_dataset.csv

Usage:
    python src/data/build_us_forecasting_dataset.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Input path
PANEL_PATH = Path("datasets/us_panel_dataset.csv")

# Output path
OUTPUT_PATH = Path("datasets/us_forecasting_dataset.csv")

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

# Target variable (from year t+1)
TARGET_COLUMN = 'avg_agi'


def build_forecasting_pairs(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create (t, t+1) pairs for each ZIP code.
    
    For each ZIP:
        - Features from year t
        - Target from year t+1
    """
    print("\nBuilding forecasting pairs...")
    
    # Sort by ZIP and year
    panel_df = panel_df.sort_values(['zip', 'year']).reset_index(drop=True)
    
    # Get unique ZIPs
    zips = panel_df['zip'].unique()
    print(f"Processing {len(zips):,} unique ZIP codes")
    
    forecast_pairs = []
    
    for zip_code in zips:
        # Get all years for this ZIP
        zip_data = panel_df[panel_df['zip'] == zip_code].sort_values('year')
        
        # Create pairs: (year_t, year_t+1)
        for i in range(len(zip_data) - 1):
            row_t = zip_data.iloc[i]
            row_t_plus_1 = zip_data.iloc[i + 1]
            
            year_t = row_t['year']
            year_t_plus_1 = row_t_plus_1['year']
            
            # Only create pair if years are consecutive
            if year_t_plus_1 == year_t + 1:
                pair = {
                    'zip': zip_code,
                    'state': row_t['state'],
                    'from_year': int(year_t),
                    'to_year': int(year_t_plus_1),
                }
                
                # Add features from year t
                for col in FEATURE_COLUMNS:
                    if col in row_t.index:
                        pair[f'{col}_t'] = row_t[col]
                
                # Add target from year t+1
                if TARGET_COLUMN in row_t_plus_1.index:
                    pair['target_income_tplus1'] = row_t_plus_1[TARGET_COLUMN]
                
                forecast_pairs.append(pair)
    
    forecast_df = pd.DataFrame(forecast_pairs)
    
    print(f"Created {len(forecast_df):,} forecasting pairs")
    
    return forecast_df


def validate_no_leakage(forecast_df: pd.DataFrame):
    """
    Validate that there's no data leakage.
    
    Ensures:
    - All features are from year t
    - Target is from year t+1
    - to_year > from_year
    """
    print("\nValidating no data leakage...")
    
    # Check year ordering
    invalid_years = forecast_df[forecast_df['to_year'] <= forecast_df['from_year']]
    if len(invalid_years) > 0:
        print(f"[ERROR] Found {len(invalid_years)} records with invalid year ordering")
        return False
    
    # Check that all feature columns end with _t
    feature_cols = [col for col in forecast_df.columns if col not in ['zip', 'state', 'from_year', 'to_year', 'target_income_tplus1']]
    non_t_features = [col for col in feature_cols if not col.endswith('_t')]
    
    if non_t_features:
        print(f"[WARNING] Found features without _t suffix: {non_t_features}")
    
    # Check for missing values
    missing_target = forecast_df['target_income_tplus1'].isna().sum()
    if missing_target > 0:
        print(f"[WARNING] {missing_target} records missing target values")
    
    print("[OK] No data leakage detected")
    print(f"[OK] All {len(forecast_df)} pairs are from_year < to_year")
    
    return True


def compute_time_based_splits(forecast_df: pd.DataFrame):
    """
    Compute time-based train/test splits.
    
    Train: 2019→2020, 2020→2021, 2021→2022 transitions
    Test:  Final year only (2022→2023 would be, but we use 2021→2022 as test)
    
    Actually for true evaluation:
    Train: 2019→2020, 2020→2021  
    Test:  2021→2022
    """
    print("\nComputing time-based splits...")
    
    # Count pairs by transition
    transition_counts = forecast_df.groupby(['from_year', 'to_year']).size()
    print("\nTransitions available:")
    print(transition_counts)
    
    # Define train/test split
    train_df = forecast_df[forecast_df['to_year'] < 2022].copy()
    test_df = forecast_df[forecast_df['to_year'] == 2022].copy()
    
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    
    print(f"\nTrain set: {len(train_df):,} pairs")
    print(f"  Transitions: {train_df['from_year'].min()}→{train_df['from_year'].max()+1}")
    print(f"Test set:  {len(test_df):,} pairs")
    print(f"  Transition: 2021→2022")
    
    # Combine
    forecast_df = pd.concat([train_df, test_df], ignore_index=True)
    
    return forecast_df


def main():
    print("="*70)
    print("BUILDING US FORECASTING DATASET")
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
    
    # Build forecasting pairs
    forecast_df = build_forecasting_pairs(panel_df)
    
    # Validate
    if not validate_no_leakage(forecast_df):
        print("\n[ERROR] Data leakage validation failed")
        sys.exit(1)
    
    # Compute splits
    forecast_df = compute_time_based_splits(forecast_df)
    
    # Summary statistics
    print("\n" + "="*70)
    print("FORECASTING DATASET SUMMARY")
    print("="*70)
    
    print(f"\nShape: {forecast_df.shape}")
    print(f"Features: {len([c for c in forecast_df.columns if c.endswith('_t')])}")
    
    print("\nTarget statistics:")
    print(forecast_df['target_income_tplus1'].describe())
    
    print("\nFeature availability:")
    feature_cols = [col for col in forecast_df.columns if col.endswith('_t')]
    missing_pct = (forecast_df[feature_cols].isna().sum() / len(forecast_df) * 100).sort_values(ascending=False)
    print(missing_pct.head(10))
    
    print("\nSplit distribution:")
    print(forecast_df['split'].value_counts())
    
    print("\nState distribution:")
    print(forecast_df['state'].value_counts().head(10))
    
    # Save
    print("\n" + "="*70)
    print("SAVING FORECASTING DATASET")
    print("="*70)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Size: {OUTPUT_PATH.stat().st_size / (1024**2):.1f} MB")
    
    # Save column list
    col_list_path = OUTPUT_PATH.with_suffix('.columns.txt')
    with open(col_list_path, 'w') as f:
        for col in forecast_df.columns:
            f.write(f"{col}\n")
    print(f"Column list: {col_list_path}")
    
    # Save metadata
    transitions_dict = forecast_df.groupby(['from_year', 'to_year']).size().to_dict()
    transitions_dict_str_keys = {f"{k[0]}→{k[1]}": v for k, v in transitions_dict.items()}
    
    metadata = {
        'total_pairs': len(forecast_df),
        'train_pairs': len(forecast_df[forecast_df['split'] == 'train']),
        'test_pairs': len(forecast_df[forecast_df['split'] == 'test']),
        'unique_zips': forecast_df['zip'].nunique(),
        'unique_states': forecast_df['state'].nunique(),
        'transitions': transitions_dict_str_keys,
        'feature_columns': feature_cols,
        'target_column': 'target_income_tplus1',
        'no_leakage_validated': True,
        'time_based_split': True
    }
    
    import json
    metadata_path = OUTPUT_PATH.with_suffix('.meta.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata: {metadata_path}")
    
    print("\n" + "="*70)
    print("FORECASTING DATASET COMPLETE")
    print("="*70)
    print("\n✅ Dataset ready for time-series forecasting")
    print("✅ No data leakage: features from year t, target from t+1")
    print("✅ Time-based split: train on earlier years, test on 2021→2022")
    
    print("\nNext steps:")
    print("  python src/training/train_us_forecast_stat.py")
    print("  python src/training/train_us_forecast_ml.py")
    print("  python src/training/train_us_forecast_hybrid.py")


if __name__ == '__main__':
    main()
