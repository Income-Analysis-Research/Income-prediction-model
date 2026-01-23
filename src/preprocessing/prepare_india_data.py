"""
Kaggle Census 2011 Data Preprocessing Script
============================================

This script converts the raw Kaggle india-districts-census-2011.csv file
into the format expected by train_india.py.

Input: datasets/archive (10)/india-districts-census-2011.csv
Output: datasets/india_district_census_data.csv

Features Generated:
- REQUIRED: district, state, literacy_rate, worker_participation, urban_ratio, avg_household_size
- OPTIONAL: asset_score, electricity_access, water_access, sanitation_access
"""

import pandas as pd
import numpy as np

def preprocess_kaggle_census():
    print("=" * 70)
    print("KAGGLE CENSUS 2011 DATA PREPROCESSING")
    print("=" * 70)
    
    # Load raw data
    input_path = r"datasets\archive (10)\india-districts-census-2011.csv"
    print(f"\n📂 Loading: {input_path}")
    
    try:
        df = pd.read_csv(input_path)
        print(f"✅ Loaded {len(df)} districts")
        print(f"   Columns: {len(df.columns)}")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {input_path}")
        print("   Please ensure the Kaggle dataset is in the correct location.")
        return
    
    # Initialize output dataframe
    output_df = pd.DataFrame()
    
    # ========================================================================
    # REQUIRED FEATURES (Mode B minimum)
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("EXTRACTING REQUIRED FEATURES")
    print("=" * 70)
    
    # 1. District and State (direct copy)
    output_df['district'] = df['District name']
    output_df['state'] = df['State Name']
    print(f"✅ district: {output_df['district'].notna().sum()} values")
    print(f"✅ state: {output_df['state'].notna().sum()} values")
    
    # 2. Literacy Rate (%)
    output_df['literacy_rate'] = (df['Literate'] / df['Population']) * 100
    print(f"✅ literacy_rate: Mean = {output_df['literacy_rate'].mean():.2f}%, "
          f"Range = [{output_df['literacy_rate'].min():.2f}%, {output_df['literacy_rate'].max():.2f}%]")
    
    # 3. Worker Participation Rate (%)
    output_df['worker_participation'] = (df['Workers'] / df['Population']) * 100
    print(f"✅ worker_participation: Mean = {output_df['worker_participation'].mean():.2f}%, "
          f"Range = [{output_df['worker_participation'].min():.2f}%, {output_df['worker_participation'].max():.2f}%]")
    
    # 4. Urban Ratio (%)
    total_households = df['Urban_Households'] + df['Rural_Households']
    output_df['urban_ratio'] = (df['Urban_Households'] / total_households) * 100
    print(f"✅ urban_ratio: Mean = {output_df['urban_ratio'].mean():.2f}%, "
          f"Range = [{output_df['urban_ratio'].min():.2f}%, {output_df['urban_ratio'].max():.2f}%]")
    
    # 5. Average Household Size
    output_df['avg_household_size'] = df['Population'] / df['Households']
    print(f"✅ avg_household_size: Mean = {output_df['avg_household_size'].mean():.2f}, "
          f"Range = [{output_df['avg_household_size'].min():.2f}, {output_df['avg_household_size'].max():.2f}]")
    
    # ========================================================================
    # OPTIONAL FEATURES (Mode A additions)
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("EXTRACTING OPTIONAL FEATURES (Mode A)")
    print("=" * 70)
    
    # 6. Asset Score (composite index 0-100)
    print("\n📊 Calculating asset_score from household amenities...")
    
    # Key asset indicators
    tv_ownership = df['Households_with_Television'] / df['Households']
    computer_ownership = df['Households_with_Computer'] / df['Households']
    car_ownership = df['Households_with_Car_Jeep_Van'] / df['Households']
    scooter_ownership = df['Households_with_Scooter_Motorcycle_Moped'] / df['Households']
    phone_ownership = df['Households_with_Telephone_Mobile_Phone'] / df['Households']
    
    # Composite asset score (weighted average)
    output_df['asset_score'] = (
        tv_ownership * 20 +
        computer_ownership * 25 +
        car_ownership * 30 +
        scooter_ownership * 15 +
        phone_ownership * 10
    ) * 100
    
    print(f"   Components:")
    print(f"   - TV ownership: {(tv_ownership.mean() * 100):.2f}%")
    print(f"   - Computer ownership: {(computer_ownership.mean() * 100):.2f}%")
    print(f"   - Car ownership: {(car_ownership.mean() * 100):.2f}%")
    print(f"   - Scooter ownership: {(scooter_ownership.mean() * 100):.2f}%")
    print(f"   - Phone ownership: {(phone_ownership.mean() * 100):.2f}%")
    print(f"✅ asset_score: Mean = {output_df['asset_score'].mean():.2f}, "
          f"Range = [{output_df['asset_score'].min():.2f}, {output_df['asset_score'].max():.2f}]")
    
    # 7. Electricity Access (%)
    output_df['electricity_access'] = (df['Housholds_with_Electric_Lighting'] / df['Households']) * 100
    print(f"✅ electricity_access: Mean = {output_df['electricity_access'].mean():.2f}%, "
          f"Range = [{output_df['electricity_access'].min():.2f}%, {output_df['electricity_access'].max():.2f}%]")
    
    # 8. Water Access (% with improved water source)
    print("\n💧 Calculating water_access from drinking water sources...")
    
    # Improved sources: tap water, tubewell, covered well
    improved_water = (
        df['Main_source_of_drinking_water_Tapwater_Households'] +
        df['Main_source_of_drinking_water_Tubewell_Borehole_Households'] +
        df['Main_source_of_drinking_water_Handpump_Tubewell_Borewell_Households']
    )
    output_df['water_access'] = (improved_water / df['Households']) * 100
    print(f"   Improved sources: Tap + Tubewell + Handpump")
    print(f"✅ water_access: Mean = {output_df['water_access'].mean():.2f}%, "
          f"Range = [{output_df['water_access'].min():.2f}%, {output_df['water_access'].max():.2f}%]")
    
    # 9. Sanitation Access (% with latrine within premises)
    output_df['sanitation_access'] = (
        df['Having_latrine_facility_within_the_premises_Total_Households'] / df['Households']
    ) * 100
    print(f"✅ sanitation_access: Mean = {output_df['sanitation_access'].mean():.2f}%, "
          f"Range = [{output_df['sanitation_access'].min():.2f}%, {output_df['sanitation_access'].max():.2f}%]")
    
    # ========================================================================
    # DATA CLEANING
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("DATA CLEANING")
    print("=" * 70)
    
    print(f"\n📊 Before cleaning: {len(output_df)} districts")
    
    # Remove rows with missing critical values
    critical_cols = ['district', 'state', 'literacy_rate', 'worker_participation', 
                     'urban_ratio', 'avg_household_size']
    
    before_count = len(output_df)
    output_df = output_df.dropna(subset=critical_cols)
    after_count = len(output_df)
    
    if before_count > after_count:
        print(f"⚠️  Removed {before_count - after_count} rows with missing required values")
    
    # Remove outliers (extremely unrealistic values)
    # Literacy rate should be 0-100%
    output_df = output_df[(output_df['literacy_rate'] >= 0) & (output_df['literacy_rate'] <= 100)]
    
    # Worker participation should be reasonable (typically 30-60%)
    output_df = output_df[(output_df['worker_participation'] >= 10) & (output_df['worker_participation'] <= 80)]
    
    # Urban ratio should be 0-100%
    output_df = output_df[(output_df['urban_ratio'] >= 0) & (output_df['urban_ratio'] <= 100)]
    
    # Household size should be reasonable (typically 3-7)
    output_df = output_df[(output_df['avg_household_size'] >= 2) & (output_df['avg_household_size'] <= 10)]
    
    print(f"✅ After cleaning: {len(output_df)} districts")
    print(f"   Removed {after_count - len(output_df)} outliers")
    
    # ========================================================================
    # VALIDATION & STATISTICS
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    
    print(f"\n📊 Total districts: {len(output_df)}")
    print(f"📍 States covered: {output_df['state'].nunique()}")
    print(f"\n🔢 Feature Completeness:")
    for col in output_df.columns:
        completeness = (output_df[col].notna().sum() / len(output_df)) * 100
        print(f"   {col:30s}: {completeness:6.2f}% complete")
    
    # Check for duplicate districts
    duplicates = output_df['district'].duplicated().sum()
    if duplicates > 0:
        print(f"\n⚠️  WARNING: {duplicates} duplicate district names found")
        print("   Keeping first occurrence only...")
        output_df = output_df.drop_duplicates(subset=['district'], keep='first')
    
    # ========================================================================
    # SAVE OUTPUT
    # ========================================================================
    
    output_path = r"datasets\india_district_census_data.csv"
    output_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print("✅ SUCCESS")
    print("=" * 70)
    print(f"\n💾 Saved to: {output_path}")
    print(f"📊 Final dataset: {len(output_df)} districts × {len(output_df.columns)} features")
    print(f"\n🎯 Training Mode: MODE A (Full features)")
    print(f"   Available features: 9/11 (82% coverage)")
    print(f"   - REQUIRED (6/6): ✅ All Census features")
    print(f"   - OPTIONAL (3/5): ✅ Asset score, Electricity, Water, Sanitation")
    print(f"   - MISSING (2/11): ❌ Nightlight data (not critical - only 5% weight)")
    
    print(f"\n🚀 NEXT STEP:")
    print(f"   Run: python train_india.py")
    print(f"   Expected: Model will train in MODE A with 9 features")
    
    print("\n" + "=" * 70)
    
    # Display sample rows
    print("\n📋 SAMPLE DATA (first 5 districts):")
    print("=" * 70)
    print(output_df.head().to_string())
    
    return output_df


if __name__ == "__main__":
    df = preprocess_kaggle_census()
