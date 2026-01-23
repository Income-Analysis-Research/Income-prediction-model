"""
Build US Panel Dataset from Multi-Year IRS Data.

This script loads IRS ZIP code data for years 2018-2022, standardizes columns,
aggregates to ZIP level, and creates a panel dataset with time dimension.

Output: datasets/us_panel_dataset.csv

Usage:
    python src/data/build_us_panel_dataset.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Input directory
MULTI_YEAR_DIR = Path("datasets/us_multi_year")

# Output path
OUTPUT_PATH = Path("datasets/us_panel_dataset.csv")

# Years to process
YEARS = [2018, 2019, 2020, 2021, 2022]

# Column mapping (old name → new name) for standardization
# After lowercasing, columns are like: zipcode, state, agi_stub, n1, a00100, etc.
COLUMN_MAPPING = {
    'zipcode': 'zip',
    'state': 'state',
    'agi_stub': 'agi_bracket',
    'n1': 'n_returns',
    'a00100': 'total_agi',
    'n00200': 'n_wages',
    'a00200': 'total_wages',
    'n00300': 'n_interest',
    'a00300': 'total_interest',
    'n00650': 'n_cap_gains',
    'a00650': 'total_cap_gains',
    'n00900': 'n_business',
    'a00900': 'total_business',
    'n01400': 'n_taxable_refunds',
    'a01400': 'total_taxable_refunds',
    'n01700': 'n_pensions',
    'a01700': 'total_pensions',
    # Add more mappings as needed
}


def load_year(year: int) -> pd.DataFrame:
    """Load and preprocess data for a single year."""
    file_path = MULTI_YEAR_DIR / f"{year}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"\nLoading {year} data...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Convert column names to lowercase for consistency
    df.columns = df.columns.str.lower()
    
    print(f"  Loaded {len(df):,} records")
    print(f"  Columns: {len(df.columns)}")
    
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names across years."""
    # Rename columns according to mapping
    existing_cols = {old: new for old, new in COLUMN_MAPPING.items() if old in df.columns}
    df = df.rename(columns=existing_cols)
    
    return df


def aggregate_to_zip_level(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Aggregate data to ZIP level (across all AGI brackets).
    
    IRS data has separate rows for each ZIP × AGI_bracket combination.
    We need to sum across AGI brackets to get ZIP-level totals.
    """
    print(f"  Aggregating to ZIP level...")
    
    # Group by ZIP and sum numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove agi_bracket from numeric cols (we don't want to sum it)
    if 'agi_bracket' in numeric_cols:
        numeric_cols.remove('agi_bracket')
    
    # Aggregate
    agg_dict = {col: 'sum' for col in numeric_cols}
    agg_dict['state'] = 'first'  # Keep state (same for all rows of a ZIP)
    
    df_agg = df.groupby('zip', as_index=False).agg(agg_dict)
    
    # Add year column
    df_agg['year'] = year
    
    print(f"  Aggregated to {len(df_agg):,} ZIP codes")
    
    return df_agg


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from raw columns."""
    print(f"  Computing derived features...")
    
    # Average AGI per return
    df['avg_agi'] = df['total_agi'] / df['n_returns'].replace(0, np.nan)
    
    # Percentage with wages
    df['pct_with_wages'] = df['n_wages'] / df['n_returns'].replace(0, np.nan)
    
    # Average wage per earner
    df['avg_wage_per_earner'] = df['total_wages'] / df['n_wages'].replace(0, np.nan)
    
    # Percentage with business income
    df['pct_with_business'] = df['n_business'] / df['n_returns'].replace(0, np.nan)
    
    # Percentage with capital gains
    df['pct_with_cap_gains'] = df['n_cap_gains'] / df['n_returns'].replace(0, np.nan)
    
    # Percentage with pension income
    df['pct_with_pension'] = df['n_pensions'] / df['n_returns'].replace(0, np.nan)
    
    # Log of returns (for skewness)
    df['log_returns'] = np.log1p(df['n_returns'])
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter data."""
    print(f"  Cleaning data...")
    
    initial_count = len(df)
    
    # Remove ZIP codes with very few returns (IRS already handles most privacy)
    # Lower threshold to 10 instead of 100 to keep more ZIPs
    df = df[df['n_returns'] >= 10]
    
    # Remove invalid ZIP codes
    df = df[df['zip'].notna()]
    # Don't filter by length - ZIPs are stored as integers and lose leading zeros
    # ZIP 00601 becomes 601, etc. Keep all valid ZIP integers.
    df = df[df['zip'] > 0]  # Just remove ZIP 0 if it exists
    
    # Remove outliers in income (> $10M average or < $1K)
    df = df[(df['avg_agi'] > 1) & (df['avg_agi'] < 10000)]
    
    print(f"  Removed {initial_count - len(df):,} records")
    print(f"  Remaining: {len(df):,} records")
    
    return df


def main():
    print("="*70)
    print("BUILDING US PANEL DATASET")
    print("="*70)
    
    # Check if multi-year directory exists
    if not MULTI_YEAR_DIR.exists():
        print(f"\nERROR: Directory not found: {MULTI_YEAR_DIR}")
        print("Please run: python scripts/download_us_irs_zip_data.py --all")
        sys.exit(1)
    
    # Process each year
    dfs = []
    
    for year in YEARS:
        try:
            # Load
            df = load_year(year)
            
            # Standardize
            df = standardize_columns(df)
            
            # Aggregate to ZIP level
            df = aggregate_to_zip_level(df, year)
            
            # Compute features
            df = compute_features(df)
            
            # Clean
            df = clean_data(df)
            
            dfs.append(df)
            
        except FileNotFoundError as e:
            print(f"\n[ERROR] {e}")
            print(f"Skipping year {year}")
            continue
        except Exception as e:
            print(f"\n[ERROR] Failed to process {year}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not dfs:
        print("\n[ERROR] No data processed successfully")
        sys.exit(1)
    
    # Concatenate all years
    print("\n" + "="*70)
    print("CONCATENATING YEARS")
    print("="*70)
    
    panel_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by ZIP and year
    panel_df = panel_df.sort_values(['zip', 'year']).reset_index(drop=True)
    
    print(f"\nPanel dataset shape: {panel_df.shape}")
    print(f"Years: {sorted(panel_df['year'].unique())}")
    print(f"Unique ZIP codes: {panel_df['zip'].nunique():,}")
    print(f"Total records: {len(panel_df):,}")
    
    # Summary statistics
    print("\nPanel structure:")
    print(panel_df.groupby('year').size())
    
    # Check for missing years per ZIP
    zip_year_counts = panel_df.groupby('zip')['year'].count()
    print(f"\nZIPs with all {len(YEARS)} years: {(zip_year_counts == len(YEARS)).sum():,}")
    print(f"ZIPs with partial coverage: {(zip_year_counts < len(YEARS)).sum():,}")
    
    # Save
    print("\n" + "="*70)
    print("SAVING PANEL DATASET")
    print("="*70)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Size: {OUTPUT_PATH.stat().st_size / (1024**2):.1f} MB")
    
    # Save column list for reference
    col_list_path = OUTPUT_PATH.with_suffix('.columns.txt')
    with open(col_list_path, 'w') as f:
        for col in panel_df.columns:
            f.write(f"{col}\n")
    print(f"Column list: {col_list_path}")
    
    print("\n" + "="*70)
    print("PANEL DATASET COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  python src/data/build_us_forecasting_dataset.py")


if __name__ == '__main__':
    main()
