"""
Train US Statistical Model (Ridge Regression)

Baseline statistical model with interpretable coefficients for research validation.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from src.hybrid.us_hybrid_model import USStatisticalModel


def load_and_prepare_data(data_path="datasets/us_irs_zipcode_data.csv"):
    """Load and prepare US dataset"""
    print("\n" + "="*60)
    print("LOADING US DATA")
    print("="*60)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} ZIP codes")
    
    # Calculate target
    df['avg_income'] = df['agi_amount'] / df['num_returns'].replace(0, np.nan)
    df = df[df['avg_income'].notna()]
    df = df[df['avg_income'] > 0]
    df = df[df['avg_income'] < 1e7]
    
    print(f"Valid records: {len(df):,}")
    
    # Engineer features (NO LEAKAGE)
    features = []
    
    if 'avg_wage' in df.columns:
        features.append('avg_wage')
    
    if 'total_wages' in df.columns:
        df['log_total_wages'] = np.log1p(df['total_wages'])
        features.append('log_total_wages')
    
    if 'num_wage_returns' in df.columns:
        df['log_wage_returns'] = np.log1p(df['num_wage_returns'])
        features.append('log_wage_returns')
        features.append('num_wage_returns')
    
    if 'total_wages' in df.columns and 'num_wage_returns' in df.columns:
        df['wage_per_return'] = df['total_wages'] / df['num_wage_returns'].replace(0, np.nan)
        df['wage_per_return'] = df['wage_per_return'].fillna(df['wage_per_return'].median())
        features.append('wage_per_return')
    
    if 'total_wages' in df.columns:
        features.append('total_wages')
    
    # State encoding
    if 'state' in df.columns:
        state_income = df.groupby('state')['avg_income'].mean()
        df['state_avg_income'] = df['state'].map(state_income)
        features.append('state_avg_income')
    
    print(f"Features: {len(features)}")
    for i, feat in enumerate(features, 1):
        print(f"  {i}. {feat}")
    
    X = df[features].fillna(df[features].median()).values
    y = df['avg_income'].values
    
    return X, y, features


def train_statistical_model():
    """Train and evaluate statistical model"""
    print("\n" + "="*60)
    print("US STATISTICAL MODEL TRAINING")
    print("="*60)
    
    # Load data
    X, y, feature_names = load_and_prepare_data()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING RIDGE REGRESSION")
    print("="*60)
    
    model = USStatisticalModel(alpha=1.0)
    model.fit(X_train, y_train, feature_names)
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(
        model.model, model.scaler.transform(X_train), y_train,
        cv=5, scoring='r2', n_jobs=-1
    )
    
    print(f"CV R² scores: {cv_scores}")
    print(f"CV R² mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print("\n" + "="*60)
    print("STATISTICAL MODEL RESULTS")
    print("="*60)
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²:  {test_r2:.4f}")
    print(f"Test RMSE: ${test_rmse/1000:.1f}k")
    print(f"Test MAE:  ${test_mae/1000:.1f}k")
    
    # Coefficient analysis
    coef_df = model.get_coefficients()
    print("\nTop Feature Coefficients:")
    print(coef_df.head(10).to_string(index=False))
    
    # Save model
    models_dir = Path("models/us")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    import joblib
    joblib.dump(model, models_dir / "stat_model.pkl")
    
    # Save results
    results = {
        'model_type': 'Ridge Regression',
        'training_date': datetime.now().isoformat(),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'cv_r2_mean': float(cv_scores.mean()),
        'cv_r2_std': float(cv_scores.std()),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'alpha': 1.0
    }
    
    with open(models_dir / "results_stat.txt", 'w') as f:
        f.write("US STATISTICAL MODEL - RIDGE REGRESSION\\n")
        f.write("="*60 + "\\n\\n")
        f.write(f"Training Date: {results['training_date']}\\n")
        f.write(f"Model: Ridge Regression (alpha={results['alpha']})\\n")
        f.write(f"Features: {results['n_features']}\\n\\n")
        
        f.write("FEATURE LIST:\\n")
        for i, feat in enumerate(feature_names, 1):
            f.write(f"  {i}. {feat}\\n")
        f.write("\\n")
        
        f.write("PERFORMANCE:\\n")
        f.write(f"  5-Fold CV R²: {results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}\\n")
        f.write(f"  Train R²: {results['train_r2']:.4f}\\n")
        f.write(f"  Test R²: {results['test_r2']:.4f}\\n")
        f.write(f"  Test RMSE: ${results['test_rmse']/1000:.1f}k\\n")
        f.write(f"  Test MAE: ${results['test_mae']/1000:.1f}k\\n\\n")
        
        f.write("TOP COEFFICIENTS:\\n")
        for _, row in coef_df.head(10).iterrows():
            f.write(f"  {row['feature']}: {row['coefficient']:.6f}\\n")
    
    with open(models_dir / "metadata_stat.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel saved to: {models_dir}")
    print("✓ stat_model.pkl")
    print("✓ results_stat.txt")
    print("✓ metadata_stat.json")


if __name__ == "__main__":
    train_statistical_model()
