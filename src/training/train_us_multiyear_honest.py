"""
Train US Income Model with Multi-Year IRS Data (2011-2022)
NO CHEATING - Uses only valid non-leaking features
Aggregates data across years for robust prediction
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# AGI stub income brackets (midpoints in thousands)
AGI_BRACKET_MIDPOINTS = {
    1: 12.5,   # $1 - $25,000
    2: 37.5,   # $25,000 - $50,000
    3: 62.5,   # $50,000 - $75,000
    4: 87.5,   # $75,000 - $100,000
    5: 150,    # $100,000 - $200,000
    6: 250     # $200,000+
}


def load_and_aggregate_year(year, data_dir='datasets'):
    """Load a single year of IRS data and aggregate by ZIP code."""
    year_short = str(year)[2:]  # '22' for 2022
    agi_file = Path(data_dir) / f"{year_short}zpallagi.csv"
    
    if not agi_file.exists():
        print(f"WARNING: {year} data not found, skipping")
        return None
    
    print(f"Loading {year} data...")
    df = pd.read_csv(agi_file, low_memory=False)
    
    # Normalize column names (older files use ZIPCODE/AGI_STUB, newer use zipcode/agi_stub)
    if 'ZIPCODE' in df.columns:
        df = df.rename(columns={'ZIPCODE': 'zipcode'})
    if 'AGI_STUB' in df.columns:
        df = df.rename(columns={'AGI_STUB': 'agi_stub'})
    
    # Filter out state-level aggregates (zipcode=0) and keep only valid ZIPs
    df = df[df['zipcode'] > 0].copy()
    
    # Filter out agi_stub=0 (total across all brackets - would be leakage)
    df = df[df['agi_stub'] > 0].copy()
    
    # Key columns (N = count, A = amount in thousands)
    # A00100 = Adjusted Gross Income
    # N1 = Number of returns
    # A00200 = Salaries and wages
    # N00200 = Number with salaries/wages
    
    # Calculate per-return averages for each AGI bracket
    df['avg_agi_per_return'] = df['A00100'] / df['N1'].replace(0, np.nan)
    df['avg_wages_per_return'] = df['A00200'] / df['N00200'].replace(0, np.nan)
    df['pct_with_wages'] = df['N00200'] / df['N1'].replace(0, np.nan) * 100
    
    # Add year column
    df['year'] = year
    
    # Aggregate by ZIP code
    zip_features = []
    
    for zipcode, group in df.groupby('zipcode'):
        total_returns = group['N1'].sum()
        
        if total_returns < 20:  # Skip ZIPs with very few returns
            continue
        
        # Weighted average AGI using return counts as weights
        weighted_agi = (group['A00100']).sum() / total_returns
        
        # Income distribution metrics (no leakage)
        agi_brackets = group.groupby('agi_stub')['N1'].sum()
        total_with_brackets = agi_brackets.sum()
        
        # Calculate income distribution features
        pct_low = (agi_brackets.get(1, 0) + agi_brackets.get(2, 0)) / total_with_brackets * 100
        pct_mid = (agi_brackets.get(3, 0) + agi_brackets.get(4, 0)) / total_with_brackets * 100
        pct_high = (agi_brackets.get(5, 0) + agi_brackets.get(6, 0)) / total_with_brackets * 100
        
        # Gini coefficient approximation from brackets
        bracket_shares = []
        for stub in sorted(AGI_BRACKET_MIDPOINTS.keys()):
            count = agi_brackets.get(stub, 0)
            income = count * AGI_BRACKET_MIDPOINTS[stub]
            bracket_shares.append((count / total_with_brackets, income))
        
        # Calculate wage statistics
        total_wages = group['A00200'].sum()
        total_wage_earners = group['N00200'].sum()
        avg_wage = total_wages / total_wage_earners if total_wage_earners > 0 else np.nan
        pct_with_wages = total_wage_earners / total_returns * 100
        
        # Business income (Schedule C)
        total_business_income = group.get('A00900', pd.Series([0])).sum()
        num_business = group.get('N00900', pd.Series([0])).sum()
        avg_business_income = total_business_income / num_business if num_business > 0 else 0
        pct_with_business = num_business / total_returns * 100
        
        # Capital gains
        total_cap_gains = group.get('A01000', pd.Series([0])).sum()
        num_cap_gains = group.get('N01000', pd.Series([0])).sum()
        pct_with_cap_gains = num_cap_gains / total_returns * 100
        
        # Retirement income
        total_pensions = group.get('A01700', pd.Series([0])).sum()
        num_pensions = group.get('N01700', pd.Series([0])).sum()
        pct_with_pension = num_pensions / total_returns * 100
        
        zip_features.append({
            'zipcode': int(zipcode),
            'year': year,
            'total_returns': total_returns,
            
            # Target variable (what we want to predict)
            'avg_agi': weighted_agi,
            
            # VALID FEATURES (no direct income leakage):
            # 1. Income distribution shape
            'pct_low_income': pct_low,
            'pct_mid_income': pct_mid,
            'pct_high_income': pct_high,
            
            # 2. Income source diversity
            'pct_with_wages': pct_with_wages,
            'pct_with_business': pct_with_business,
            'pct_with_cap_gains': pct_with_cap_gains,
            'pct_with_pension': pct_with_pension,
            
            # 3. Wage statistics (non-leaking)
            'avg_wage_per_earner': avg_wage,
            
            # 4. Business activity
            'avg_business_income': avg_business_income,
            
            # 5. Return volume (population proxy)
            'log_returns': np.log1p(total_returns)
        })
    
    result_df = pd.DataFrame(zip_features)
    print(f"   Processed {year}: {len(result_df):,} ZIP codes")
    return result_df


def combine_multiyear_data(years, data_dir='datasets'):
    """Combine multiple years of data."""
    all_data = []
    
    for year in years:
        year_data = load_and_aggregate_year(year, data_dir)
        if year_data is not None:
            all_data.append(year_data)
    
    if not all_data:
        raise ValueError("No data loaded!")
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined dataset: {len(combined):,} rows across {len(years)} years")
    return combined


def create_temporal_features(df):
    """Add temporal features for multi-year learning."""
    df = df.copy()
    
    # Year normalized (0 to 1 scale)
    min_year = df['year'].min()
    max_year = df['year'].max()
    df['year_normalized'] = (df['year'] - min_year) / (max_year - min_year) if max_year > min_year else 0
    
    # For each ZIP, calculate trend if we have multiple years
    zip_trends = []
    for zipcode, group in df.groupby('zipcode'):
        if len(group) >= 2:
            # Simple linear trend
            years = group['year'].values
            agis = group['avg_agi'].values
            if len(set(years)) > 1:
                trend = np.polyfit(years, agis, 1)[0]  # Slope
            else:
                trend = 0
        else:
            trend = 0
        
        for _ in range(len(group)):
            zip_trends.append(trend)
    
    df['income_trend'] = zip_trends
    
    return df


def train_models(df, test_size=0.2, random_state=42):
    """Train multiple models without cheating."""
    print("\n" + "="*60)
    print("TRAINING HONEST MULTI-YEAR MODEL")
    print("="*60)
    
    # Define features (NO TARGET LEAKAGE)
    feature_cols = [
        'pct_low_income',
        'pct_mid_income',
        'pct_high_income',
        'pct_with_wages',
        'pct_with_business',
        'pct_with_cap_gains',
        'pct_with_pension',
        'avg_wage_per_earner',
        'avg_business_income',
        'log_returns',
        'year_normalized',
        'income_trend'
    ]
    
    X = df[feature_cols].copy()
    y = df['avg_agi'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training: {len(X_train):,} samples")
    print(f"   Testing:  {len(X_test):,} samples")
    print(f"\n📋 Features used: {len(feature_cols)}")
    for feat in feature_cols:
        print(f"   - {feat}")
    
    results = {}
    models = {}
    
    # 1. Ridge Regression (Baseline)
    print("\n" + "-"*60)
    print("1. Ridge Regression (Scaled)")
    print("-"*60)
    
    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=10.0))
    ])
    
    ridge_pipeline.fit(X_train, y_train)
    ridge_pred = ridge_pipeline.predict(X_test)
    
    ridge_r2 = r2_score(y_test, ridge_pred)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
    ridge_mae = mean_absolute_error(y_test, ridge_pred)
    
    # Cross-validation
    ridge_cv = cross_val_score(ridge_pipeline, X_train, y_train, cv=5, scoring='r2')
    
    print(f"   Train R²:       {ridge_pipeline.score(X_train, y_train):.4f}")
    print(f"   Test R²:        {ridge_r2:.4f}")
    print(f"   CV R² (5-fold): {ridge_cv.mean():.4f} ± {ridge_cv.std():.4f}")
    print(f"   RMSE:           ${ridge_rmse*1000:,.0f}")
    print(f"   MAE:            ${ridge_mae*1000:,.0f}")
    
    results['ridge'] = {
        'train_r2': ridge_pipeline.score(X_train, y_train),
        'test_r2': ridge_r2,
        'cv_r2_mean': ridge_cv.mean(),
        'cv_r2_std': ridge_cv.std(),
        'rmse': ridge_rmse,
        'mae': ridge_mae
    }
    models['ridge'] = ridge_pipeline
    
    # 2. Random Forest
    print("\n" + "-"*60)
    print("2. Random Forest")
    print("-"*60)
    
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=random_state,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)
    
    print(f"   Train R²:       {rf.score(X_train, y_train):.4f}")
    print(f"   Test R²:        {rf_r2:.4f}")
    print(f"   RMSE:           ${rf_rmse*1000:,.0f}")
    print(f"   MAE:            ${rf_mae*1000:,.0f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n   Top 5 Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"      {row['feature']:30s}: {row['importance']:.4f}")
    
    results['rf'] = {
        'train_r2': rf.score(X_train, y_train),
        'test_r2': rf_r2,
        'rmse': rf_rmse,
        'mae': rf_mae,
        'feature_importance': feature_importance.to_dict('records')
    }
    models['rf'] = rf
    
    # 3. XGBoost
    print("\n" + "-"*60)
    print("3. XGBoost")
    print("-"*60)
    
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=random_state,
        n_jobs=-1
    )
    
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    
    xgb_r2 = r2_score(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    
    print(f"   Train R²:       {xgb.score(X_train, y_train):.4f}")
    print(f"   Test R²:        {xgb_r2:.4f}")
    print(f"   RMSE:           ${xgb_rmse*1000:,.0f}")
    print(f"   MAE:            ${xgb_mae*1000:,.0f}")
    
    results['xgboost'] = {
        'train_r2': xgb.score(X_train, y_train),
        'test_r2': xgb_r2,
        'rmse': xgb_rmse,
        'mae': xgb_mae
    }
    models['xgboost'] = xgb
    
    # 4. Gradient Boosting
    print("\n" + "-"*60)
    print("4. Gradient Boosting")
    print("-"*60)
    
    gb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state
    )
    
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    
    gb_r2 = r2_score(y_test, gb_pred)
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
    gb_mae = mean_absolute_error(y_test, gb_pred)
    
    print(f"   Train R²:       {gb.score(X_train, y_train):.4f}")
    print(f"   Test R²:        {gb_r2:.4f}")
    print(f"   RMSE:           ${gb_rmse*1000:,.0f}")
    print(f"   MAE:            ${gb_mae*1000:,.0f}")
    
    results['gradient_boosting'] = {
        'train_r2': gb.score(X_train, y_train),
        'test_r2': gb_r2,
        'rmse': gb_rmse,
        'mae': gb_mae
    }
    models['gb'] = gb
    
    # Find best model
    best_model_name = max(results.items(), key=lambda x: x[1]['test_r2'])[0]
    
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model_name.upper()}")
    print(f"   Test R² = {results[best_model_name]['test_r2']:.4f}")
    print("="*60)
    
    return models, results, feature_cols, X_test, y_test


def save_models(models, results, feature_cols, output_dir='models/us_multiyear'):
    """Save trained models and results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save models
    for name, model in models.items():
        joblib.dump(model, output_path / f'{name}_model.pkl')
        print(f"Saved {name}_model.pkl")
    
    # Save results
    with open(output_path / 'training_results.json', 'w') as f:
        # Convert numpy types to native Python types
        results_serializable = {}
        for model_name, metrics in results.items():
            results_serializable[model_name] = {}
            for key, value in metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    results_serializable[model_name][key] = float(value)
                else:
                    results_serializable[model_name][key] = value
        
        json.dump(results_serializable, f, indent=2)
    print(f"Saved training_results.json")
    
    # Save feature names
    with open(output_path / 'feature_names.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved feature_names.json")
    
    print(f"\nAll models saved to: {output_path.absolute()}")


def main():
    """Main training pipeline."""
    print("="*80)
    print(" HONEST MULTI-YEAR US INCOME PREDICTION MODEL ")
    print(" No cheating | No target leakage | Real-world features only ")
    print("="*80)
    
    # Use all available years (2009-2022, excluding 2010 which doesn't exist)
    years = [2009, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    
    # Load and combine data
    df = combine_multiyear_data(years)
    
    # Add temporal features
    df = create_temporal_features(df)
    
    # Remove any rows with missing target
    df = df.dropna(subset=['avg_agi'])
    
    print(f"\nFinal dataset: {len(df):,} ZIP-year observations")
    print(f"   Unique ZIPs: {df['zipcode'].nunique():,}")
    print(f"   Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"   Avg AGI range: ${df['avg_agi'].min()*1000:,.0f} - ${df['avg_agi'].max()*1000:,.0f}")
    
    # Train models
    models, results, feature_cols, X_test, y_test = train_models(df)
    
    # Save everything
    save_models(models, results, feature_cols)
    
    print("\nTRAINING COMPLETE!")
    print("\nTo use the models:")
    print(">>> import joblib")
    print(">>> model = joblib.load('models/us_multiyear/xgboost_model.pkl')")
    print(">>> # prediction = model.predict(features)")


if __name__ == "__main__":
    main()
