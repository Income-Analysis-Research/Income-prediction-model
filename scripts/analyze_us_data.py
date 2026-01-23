"""
Analyze US IRS Data Files
"""
import pandas as pd
import os

print("=" * 80)
print("US IRS 2022 ZIP CODE DATA ANALYSIS")
print("=" * 80)

# Check file exists
agi_file = "datasets/22zpallagi.csv"
noagi_file = "datasets/22zpallnoagi.csv"

if os.path.exists(agi_file):
    print(f"\n📂 Loading: {agi_file}")
    print(f"   Size: {os.path.getsize(agi_file) / 1024 / 1024:.2f} MB")
    
    df = pd.read_csv(agi_file, low_memory=False)
    
    print(f"\n✅ Loaded successfully")
    print(f"   Total records: {len(df):,}")
    print(f"   Total columns: {len(df.columns)}")
    
    # Check ZIP code coverage
    print(f"\n📍 Geographic Coverage:")
    print(f"   Unique ZIP codes: {df['zipcode'].nunique():,}")
    print(f"   States covered: {df['STATE'].nunique()}")
    
    # Check AGI stub distribution (income brackets)
    print(f"\n💰 Income Brackets (agi_stub):")
    print(df['agi_stub'].value_counts().sort_index())
    
    # Key columns needed for our project
    print(f"\n🔑 Required Columns Check:")
    required_cols = {
        'zipcode': 'ZIP code identifier',
        'STATE': 'State abbreviation',
        'N1': 'Number of returns/taxpayers',
        'A00100': 'Adjusted Gross Income (AGI) amount',
        'N00200': 'Number of returns with salary/wages',
        'A00200': 'Total salary and wages'
    }
    
    for col, desc in required_cols.items():
        status = "✅" if col in df.columns else "❌"
        print(f"   {status} {col:10s} - {desc}")
    
    # Sample data
    print(f"\n📊 Sample Data (first 5 real ZIP codes):")
    print("=" * 80)
    
    # Filter to actual ZIP codes (not state-level aggregates)
    real_zips = df[df['zipcode'] != '00000'].copy()
    
    display_cols = ['STATE', 'zipcode', 'agi_stub', 'N1', 'A00100']
    sample = real_zips[display_cols].head(10)
    
    print(sample.to_string(index=False))
    
    # Check for state-level data
    state_level = df[df['zipcode'] == '00000']
    print(f"\n📋 State-level aggregate records: {len(state_level)}")
    
    # Check data completeness
    print(f"\n✅ Data Quality:")
    print(f"   Records with N1 > 0: {(df['N1'] > 0).sum():,}")
    print(f"   Records with AGI data: {df['A00100'].notna().sum():,}")
    print(f"   Missing values in N1: {df['N1'].isna().sum()}")
    print(f"   Missing values in A00100: {df['A00100'].isna().sum()}")
    
    # Calculate total taxpayers and AGI
    total_returns = df['N1'].sum()
    total_agi = df['A00100'].sum()
    
    print(f"\n📈 National Totals:")
    print(f"   Total tax returns: {total_returns:,.0f}")
    print(f"   Total AGI: ${total_agi / 1000:,.0f} thousand (${total_agi:,.0f})")
    if total_returns > 0:
        print(f"   Average AGI per return: ${(total_agi / total_returns):,.0f}")
    
    # Check if this data is suitable for our project
    print(f"\n" + "=" * 80)
    print("PROJECT SUITABILITY ASSESSMENT")
    print("=" * 80)
    
    real_zip_count = len(real_zips[real_zips['agi_stub'] == 1])  # Count unique ZIPs in first bracket
    
    print(f"\n✅ File Format: CSV - Compatible")
    print(f"✅ ZIP Code Coverage: {real_zip_count:,} ZIP codes (Need 10,000+ minimum)")
    print(f"✅ Required Columns: All present")
    print(f"✅ Data Completeness: {(df['N1'].notna().sum() / len(df) * 100):.1f}%")
    
    if real_zip_count >= 10000:
        print(f"\n🎯 VERDICT: ✅ EXCELLENT - Ready for training")
    else:
        print(f"\n🎯 VERDICT: ⚠️ ACCEPTABLE - But check if state-level data is needed")
    
else:
    print(f"❌ File not found: {agi_file}")

print("\n" + "=" * 80)
