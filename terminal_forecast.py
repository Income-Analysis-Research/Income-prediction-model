"""
Terminal-based interactive forecasting tool
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path


def load_models():
    """Load the multi-year forecasting models."""
    models_dir = Path("models/us_forecast_multiyear")
    
    models = {}
    metadata = {}
    
    for year_gap in [1, 2, 3]:
        model_path = models_dir / f"rf_model_{year_gap}yr.pkl"
        meta_path = models_dir / f"rf_model_{year_gap}yr.metadata.json"
        
        if model_path.exists():
            models[year_gap] = joblib.load(model_path)
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    metadata[year_gap] = json.load(f)
            print(f"✓ Loaded {year_gap}-year model")
    
    return models, metadata


def get_zip_features(zipcode, base_year):
    """Extract features for a ZIP code from the panel dataset."""
    panel_path = Path("datasets/us_panel_dataset.csv")
    
    if not panel_path.exists():
        print(f"❌ Panel dataset not found at {panel_path}")
        return None
    
    df = pd.read_csv(panel_path)
    
    # Find ZIP data for base year
    zip_data = df[(df['zip'] == int(zipcode)) & (df['year'] == base_year)]
    
    if zip_data.empty:
        print(f"❌ No data found for ZIP {zipcode} in year {base_year}")
        return None
    
    # Feature columns (without _t suffix)
    feature_cols = [
        'n_returns', 'total_wages', 'n_wages', 'total_interest',
        'n_interest', 'total_cap_gains', 'n_cap_gains', 
        'total_business', 'n_business', 'avg_agi',
        'pct_with_wages', 'avg_wage_per_earner', 
        'pct_with_business', 'pct_with_cap_gains', 'log_returns'
    ]
    
    features = []
    for col in feature_cols:
        if col in zip_data.columns:
            features.append(zip_data.iloc[0][col])
        else:
            features.append(0.0)
    
    return np.array([features])


def get_actual_income(zipcode, target_year):
    """Get actual income if available."""
    panel_path = Path("datasets/us_panel_dataset.csv")
    df = pd.read_csv(panel_path)
    
    actual_data = df[(df['zip'] == int(zipcode)) & (df['year'] == target_year)]
    
    if not actual_data.empty and 'avg_agi' in actual_data.columns:
        return float(actual_data.iloc[0]['avg_agi']) * 1000
    return None


def get_available_years():
    """Get available base years from the dataset."""
    panel_path = Path("datasets/us_panel_dataset.csv")
    df = pd.read_csv(panel_path)
    return sorted(df['year'].unique())


def get_sample_zips(base_year, n=20):
    """Get sample ZIP codes for a given year."""
    panel_path = Path("datasets/us_panel_dataset.csv")
    df = pd.read_csv(panel_path)
    year_data = df[df['year'] == base_year]
    sample = year_data.sample(n=min(n, len(year_data)))
    return [str(z).zfill(5) for z in sample['zip'].tolist()]


def main():
    print("=" * 60)
    print("  📊 TERMINAL INCOME FORECASTING SYSTEM")
    print("=" * 60)
    print()
    
    # Load models
    print("Loading models...")
    models, metadata = load_models()
    
    if not models:
        print("❌ No models loaded. Please train models first.")
        return
    
    print(f"✓ Loaded {len(models)} forecasting models (1yr, 2yr, 3yr)")
    print()
    
    # Get available years
    available_years = get_available_years()
    print(f"📅 Available base years: {', '.join(map(str, available_years))}")
    print()
    
    # Get base year
    while True:
        base_year_input = input("Enter base year (2019-2022) [default: 2022]: ").strip()
        if not base_year_input:
            base_year = 2022
        else:
            try:
                base_year = int(base_year_input)
                if base_year not in available_years:
                    print(f"❌ Year must be one of: {available_years}")
                    continue
            except ValueError:
                print("❌ Invalid year. Please enter a number.")
                continue
        break
    
    print(f"\n✓ Using base year: {base_year}")
    print()
    
    # Show sample ZIPs
    print("📍 Sample ZIP codes available:")
    sample_zips = get_sample_zips(base_year, 20)
    for i in range(0, len(sample_zips), 5):
        print("  " + "  ".join(sample_zips[i:i+5]))
    print()
    
    # Get ZIP code
    while True:
        zipcode = input("Enter 5-digit ZIP code: ").strip()
        if len(zipcode) != 5 or not zipcode.isdigit():
            print("❌ Please enter a valid 5-digit ZIP code.")
            continue
        
        # Check if ZIP exists
        features = get_zip_features(zipcode, base_year)
        if features is None:
            retry = input("Try another ZIP? (y/n): ").strip().lower()
            if retry != 'y':
                return
            continue
        break
    
    print(f"\n✓ ZIP code {zipcode} found in {base_year} data")
    print()
    
    # Get target year
    max_year = min(base_year + 3, 2025)
    print(f"📅 You can forecast up to {max_year} (max 3 years ahead)")
    print()
    
    while True:
        target_year_input = input(f"Enter target year ({base_year+1}-{max_year}): ").strip()
        try:
            target_year = int(target_year_input)
            if target_year <= base_year:
                print(f"❌ Target year must be greater than {base_year}")
                continue
            if target_year > max_year:
                print(f"❌ Maximum target year is {max_year}")
                continue
            break
        except ValueError:
            print("❌ Invalid year. Please enter a number.")
            continue
    
    year_gap = target_year - base_year
    print(f"\n✓ Forecasting {year_gap}-year horizon: {base_year} → {target_year}")
    print()
    
    # Make prediction
    if year_gap not in models:
        print(f"❌ No model available for {year_gap}-year forecast")
        return
    
    print("🔮 Generating prediction...")
    print()
    
    model = models[year_gap]
    prediction = model.predict(features)[0]
    predicted_income = prediction * 1000
    
    # Get actual if available
    actual_income = get_actual_income(zipcode, target_year)
    
    # Display results
    print("=" * 60)
    print("  FORECAST RESULTS")
    print("=" * 60)
    print()
    print(f"  ZIP Code:        {zipcode}")
    print(f"  Base Year:       {base_year}")
    print(f"  Target Year:     {target_year}")
    print(f"  Forecast Horizon: {year_gap} year(s)")
    print()
    print(f"  💰 Predicted Income: ${predicted_income:,.2f}")
    
    if actual_income:
        print(f"  📊 Actual Income:    ${actual_income:,.2f}")
        error = predicted_income - actual_income
        error_pct = (error / actual_income) * 100
        print(f"  📈 Prediction Error: ${error:,.2f} ({error_pct:+.1f}%)")
    
    print()
    
    # Model info
    if year_gap in metadata:
        meta = metadata[year_gap]
        if 'test_performance' in meta:
            perf = meta['test_performance']
            print(f"  Model Performance:")
            print(f"    R² Score: {perf['r2']:.4f}")
            print(f"    RMSE:     ${perf['rmse']*1000:,.2f}")
    
    print()
    print("=" * 60)
    print()
    
    # Ask to continue
    again = input("Make another prediction? (y/n): ").strip().lower()
    if again == 'y':
        print("\n" * 2)
        main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
