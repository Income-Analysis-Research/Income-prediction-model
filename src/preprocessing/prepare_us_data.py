"""
US IRS Data Preprocessing Script
=================================

Converts the raw IRS 22zpallagi.csv file into the format expected by train_us.py.

Input: datasets/22zpallagi.csv
Output: datasets/us_irs_zipcode_data.csv

The IRS data has 6 income brackets (agi_stub 1-6) per ZIP code.
We need to aggregate them into a single row per ZIP code.
"""

import pandas as pd
import numpy as np

def preprocess_us_irs_data():
    print("=" * 80)
    print("US IRS 2022 DATA PREPROCESSING")
    print("=" * 80)
    
    # Load raw IRS data
    input_path = "datasets/22zpallagi.csv"
    print(f"\n📂 Loading: {input_path}")
    
    try:
        df = pd.read_csv(input_path, low_memory=False)
        print(f"✅ Loaded {len(df):,} records")
        print(f"   Columns: {len(df.columns)}")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {input_path}")
        return
    
    print("\n" + "=" * 80)
    print("DATA STRUCTURE ANALYSIS")
    print("=" * 80)
    
    # Understand the AGI stub structure
    print(f"\n💰 Income Brackets (agi_stub):")
    print(f"   1: $1 under $25,000")
    print(f"   2: $25,000 under $50,000")
    print(f"   3: $50,000 under $75,000")
    print(f"   4: $75,000 under $100,000")
    print(f"   5: $100,000 under $200,000")
    print(f"   6: $200,000 or more")
    
    # Filter out state-level aggregates (zipcode == 0)
    print(f"\n🔍 Filtering to actual ZIP codes...")
    print(f"   Before: {len(df):,} records")
    df = df[df['zipcode'] != 0].copy()
    print(f"   After: {len(df):,} records")
    print(f"   Unique ZIP codes: {df['zipcode'].nunique():,}")
    
    # Key columns we need
    print(f"\n📊 Aggregating data by ZIP code...")
    print(f"   Key columns:")
    print(f"   - N1: Number of returns")
    print(f"   - A00100: Adjusted Gross Income")
    print(f"   - N00200: Number with wages")
    print(f"   - A00200: Total wages")
    
    # Aggregate by ZIP code (sum across all income brackets)
    agg_dict = {
        'STATE': 'first',  # State stays the same
        'N1': 'sum',  # Total returns across all brackets
        'A00100': 'sum',  # Total AGI across all brackets
        'N00200': 'sum',  # Total with wages
        'A00200': 'sum',  # Total wages
    }
    
    output_df = df.groupby('zipcode').agg(agg_dict).reset_index()
    
    print(f"✅ Aggregated to {len(output_df):,} ZIP codes")
    
    # Rename columns to match expected format
    output_df = output_df.rename(columns={
        'N1': 'num_returns',
        'A00100': 'agi_amount',
        'N00200': 'num_wage_returns',
        'A00200': 'total_wages',
        'STATE': 'state'
    })
    
    # Convert zipcode to string with leading zeros
    output_df['zipcode'] = output_df['zipcode'].astype(str).str.zfill(5)
    
    # Calculate derived metrics
    print(f"\n🔢 Calculating derived metrics...")
    
    # Average AGI per return
    output_df['avg_agi'] = output_df['agi_amount'] / output_df['num_returns']
    
    # Average wage per wage earner
    output_df['avg_wage'] = np.where(
        output_df['num_wage_returns'] > 0,
        output_df['total_wages'] / output_df['num_wage_returns'],
        0
    )
    
    # Reorder columns
    output_df = output_df[[
        'zipcode', 'state', 'num_returns', 'agi_amount', 
        'avg_agi', 'num_wage_returns', 'total_wages', 'avg_wage'
    ]]
    
    # ========================================================================
    # DATA CLEANING
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("DATA CLEANING")
    print("=" * 80)
    
    print(f"\n📊 Before cleaning: {len(output_df):,} ZIP codes")
    
    # Remove ZIP codes with too few returns (privacy/statistical significance)
    min_returns = 20
    before_count = len(output_df)
    output_df = output_df[output_df['num_returns'] >= min_returns].copy()
    removed = before_count - len(output_df)
    
    if removed > 0:
        print(f"⚠️  Removed {removed:,} ZIP codes with < {min_returns} returns (privacy threshold)")
    
    # Remove outliers (extremely unrealistic values)
    # AGI should be positive and reasonable
    output_df = output_df[output_df['agi_amount'] > 0].copy()
    output_df = output_df[output_df['avg_agi'] > 0].copy()
    output_df = output_df[output_df['avg_agi'] < 10000000].copy()  # $10M per return seems extreme
    
    # Check for missing values
    missing = output_df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️  Missing values found:")
        print(missing[missing > 0])
        output_df = output_df.dropna()
    
    print(f"✅ After cleaning: {len(output_df):,} ZIP codes")
    
    # ========================================================================
    # VALIDATION & STATISTICS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    
    print(f"\n📊 Total ZIP codes: {len(output_df):,}")
    print(f"📍 States covered: {output_df['state'].nunique()}")
    print(f"📋 Total tax returns: {output_df['num_returns'].sum():,.0f}")
    print(f"💰 Total AGI: ${output_df['agi_amount'].sum() / 1000:,.0f} thousand")
    
    print(f"\n🔢 Feature Statistics:")
    stats_cols = ['num_returns', 'agi_amount', 'avg_agi', 'total_wages', 'avg_wage']
    for col in stats_cols:
        print(f"   {col:20s}: Mean = ${output_df[col].mean():>15,.2f}, "
              f"Median = ${output_df[col].median():>15,.2f}")
    
    print(f"\n📍 Top 10 States by ZIP Code Count:")
    top_states = output_df['state'].value_counts().head(10)
    for state, count in top_states.items():
        print(f"   {state}: {count:,} ZIP codes")
    
    # ========================================================================
    # SAVE OUTPUT
    # ========================================================================
    
    output_path = "datasets/us_irs_zipcode_data.csv"
    output_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print("✅ SUCCESS")
    print("=" * 80)
    print(f"\n💾 Saved to: {output_path}")
    print(f"📊 Final dataset: {len(output_df):,} ZIP codes × {len(output_df.columns)} features")
    
    print(f"\n🚀 NEXT STEP:")
    print(f"   Run: python train_us.py")
    print(f"   Expected: Model will train on {len(output_df):,} ZIP codes")
    
    print("\n" + "=" * 80)
    
    # Display sample rows
    print("\n📋 SAMPLE DATA (first 10 ZIP codes):")
    print("=" * 80)
    print(output_df.head(10).to_string(index=False))
    
    return output_df


if __name__ == "__main__":
    df = preprocess_us_irs_data()
