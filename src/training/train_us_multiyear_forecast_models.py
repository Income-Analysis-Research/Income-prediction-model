"""
Train Multi-Year Forecasting Models.

Trains separate RandomForest models for each time horizon:
- 1-year: t → t+1 (high accuracy, short-term)
- 2-year: t → t+2 (medium-term)
- 3-year: t → t+3 (long-term)

This enables reliable long-range forecasting by learning patterns specific
to each time horizon instead of extrapolating beyond training distribution.

Usage:
    python src/training/train_us_multiyear_forecast_models.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json

# Paths
DATASETS_DIR = Path("datasets")
MODELS_DIR = Path("models/us_forecast_multiyear")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model hyperparameters (tuned for forecasting)
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}


def load_forecasting_dataset(year_gap: int):
    """Load forecasting dataset for specific year gap."""
    dataset_path = DATASETS_DIR / f"us_forecasting_{year_gap}yr.csv"
    metadata_path = DATASETS_DIR / f"us_forecasting_{year_gap}yr.metadata.json"
    
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return None, None
    
    df = pd.read_csv(dataset_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return df, metadata


def prepare_features_and_target(df: pd.DataFrame):
    """Extract features and target from forecasting dataset."""
    # Features: all columns ending with _t
    feature_cols = [col for col in df.columns if col.endswith('_t')]
    
    # Target
    target_col = 'target_income_tplusn'
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle any missing values
    X = X.fillna(X.median())
    
    return X, y, feature_cols


def train_model(year_gap: int):
    """Train RandomForest model for specific year gap."""
    print(f"\n{'='*70}")
    print(f"TRAINING {year_gap}-YEAR FORECASTING MODEL")
    print(f"{'='*70}\n")
    
    # Load dataset
    df, metadata = load_forecasting_dataset(year_gap)
    
    if df is None:
        print(f"Skipping {year_gap}-year model (no data)")
        return None
    
    print(f"Dataset: {len(df):,} pairs")
    print(f"Train: {metadata['train_pairs']:,} pairs")
    print(f"Test: {metadata['test_pairs']:,} pairs")
    
    # Prepare data
    X, y, feature_cols = prepare_features_and_target(df)
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target: {y.name}")
    
    # Split train/test
    if 'split' in df.columns:
        train_mask = df['split'] == 'train'
        test_mask = df['split'] == 'test'
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
    else:
        # If no split column (e.g., 3-year with only training data)
        print("[WARNING] No test split available, using all data for training")
        X_train, X_test = X, X[:0]  # Empty test set
        y_train, y_test = y, y[:0]
    
    print(f"\nTrain samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    
    # Train model
    print("\nTraining RandomForest...")
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    print(f"\n📊 Training Performance:")
    print(f"   R² Score: {train_r2:.4f}")
    print(f"   RMSE:     ${train_rmse:,.0f}")
    print(f"   MAE:      ${train_mae:,.0f}")
    
    # Evaluate on test set (if available)
    if len(X_test) > 0:
        y_test_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print(f"\n🎯 Test Performance:")
        print(f"   R² Score: {test_r2:.4f}")
        print(f"   RMSE:     ${test_rmse:,.0f}")
        print(f"   MAE:      ${test_mae:,.0f}")
    else:
        test_r2 = None
        test_rmse = None
        test_mae = None
        print(f"\n⚠️ No test set available for validation")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 5 Most Important Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    model_path = MODELS_DIR / f"rf_model_{year_gap}yr.pkl"
    joblib.dump(model, model_path)
    print(f"\n💾 Saved model: {model_path}")
    
    # Save metadata
    model_metadata = {
        'year_gap': year_gap,
        'model_type': 'RandomForest',
        'n_estimators': RF_PARAMS['n_estimators'],
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'features': feature_cols,
        'performance': {
            'train': {
                'r2': float(train_r2),
                'rmse': float(train_rmse),
                'mae': float(train_mae)
            }
        }
    }
    
    if test_r2 is not None:
        model_metadata['performance']['test'] = {
            'r2': float(test_r2),
            'rmse': float(test_rmse),
            'mae': float(test_mae)
        }
    
    metadata_path = MODELS_DIR / f"rf_model_{year_gap}yr.metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    print(f"💾 Saved metadata: {metadata_path}")
    
    # Save feature importance
    importance_path = MODELS_DIR / f"feature_importance_{year_gap}yr.csv"
    feature_importance.to_csv(importance_path, index=False)
    
    return model_metadata


def main():
    print("="*70)
    print("TRAINING MULTI-YEAR FORECASTING MODELS")
    print("="*70)
    
    # Train models for each time horizon
    year_gaps = [1, 2, 3]
    
    results = []
    
    for year_gap in year_gaps:
        try:
            metadata = train_model(year_gap)
            if metadata:
                results.append(metadata)
        except Exception as e:
            print(f"\n❌ Error training {year_gap}-year model: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"\n{'Gap':<10} {'Train R²':<12} {'Test R²':<12} {'Test RMSE':<15} {'Samples':<15}")
    print("-" * 70)
    
    for result in results:
        gap = f"{result['year_gap']}-year"
        train_r2 = f"{result['performance']['train']['r2']:.4f}"
        
        if 'test' in result['performance']:
            test_r2 = f"{result['performance']['test']['r2']:.4f}"
            test_rmse = f"${result['performance']['test']['rmse']:,.0f}"
        else:
            test_r2 = "N/A"
            test_rmse = "N/A"
        
        samples = f"{result['train_samples']:,}"
        
        print(f"{gap:<10} {train_r2:<12} {test_r2:<12} {test_rmse:<15} {samples:<15}")
    
    print("\n✅ All models trained successfully!")
    print(f"\nModels saved to: {MODELS_DIR}")
    print("\nNext steps:")
    print("  1. Update backend to load and use multi-year models")
    print("  2. Implement model selection based on year gap")
    print("  3. Update frontend to allow predictions up to 2028-2030")


if __name__ == "__main__":
    main()
