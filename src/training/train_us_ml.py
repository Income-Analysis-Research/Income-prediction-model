"""
Train US ML Model (Random Forest + XGBoost Ensemble)

Powerful ML ensemble for prediction accuracy.
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

from src.hybrid.us_hybrid_model import USMLModel


def load_and_prepare_data(data_path="datasets/us_irs_zipcode_data.csv"):
    """Load and prepare US dataset"""
    df = pd.read_csv(data_path)
    
    # Target
    df['avg_income'] = df['agi_amount'] / df['num_returns'].replace(0, np.nan)
    df = df[df['avg_income'].notna()]
    df = df[df['avg_income'] > 0]
    df = df[df['avg_income'] < 1e7]
    
    # Features (non-leaking)
    features = []
    if 'avg_wage' in df.columns:
        features.append('avg_wage')
    
    if 'total_wages' in df.columns:
        df['log_total_wages'] = np.log1p(df['total_wages'])
        features.append('log_total_wages')
        features.append('total_wages')
    
    if 'num_wage_returns' in df.columns:
        df['log_wage_returns'] = np.log1p(df['num_wage_returns'])
        features.append('log_wage_returns')
        features.append('num_wage_returns')
    
    if 'total_wages' in df.columns and 'num_wage_returns' in df.columns:
        df['wage_per_return'] = df['total_wages'] / df['num_wage_returns'].replace(0, np.nan)
        df['wage_per_return'] = df['wage_per_return'].fillna(df['wage_per_return'].median())
        features.append('wage_per_return')
    
    if 'state' in df.columns:
        state_income = df.groupby('state')['avg_income'].mean()
        df['state_avg_income'] = df['state'].map(state_income)
        features.append('state_avg_income')
    
    X = df[features].fillna(df[features].median()).values
    y = df['avg_income'].values
    
    return X, y, features


def train_ml_model():
    """Train ML ensemble"""
    print("\n" + "="*60)
    print("US ML MODEL TRAINING")
    print("="*60)
    
    X, y, features = load_and_prepare_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {len(features)}")
    
    # Train model
    print("\nTraining ML Ensemble...")
    model = USMLModel(use_xgboost=True)
    model.fit(X_train, y_train)
    
    # Evaluate
    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print("\n" + "="*60)
    print("ML MODEL RESULTS")
    print("="*60)
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²:  {test_r2:.4f}")
    print(f"Test RMSE: ${test_rmse/1000:.1f}k")
    print(f"Test MAE:  ${test_mae/1000:.1f}k")
    
    # Save
    import joblib
    models_dir = Path("models/us")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "ml_model.pkl")
    
    results = {
        'model_type': 'RF + XGBoost Ensemble',
        'training_date': datetime.now().isoformat(),
        'n_features': len(features),
        'feature_names': features,
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae)
    }
    
    with open(models_dir / "results_ml.txt", 'w') as f:
        f.write("US ML MODEL - RF + XGBOOST ENSEMBLE\n")
        f.write("="*60 + "\n\n")
        f.write(f"Training Date: {results['training_date']}\n")
        f.write(f"Test R²: {test_r2:.4f}\n")
        f.write(f"Test RMSE: ${test_rmse/1000:.1f}k\n")
        f.write(f"Test MAE: ${test_mae/1000:.1f}k\n")
    
    with open(models_dir / "metadata_ml.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to: {models_dir}")


if __name__ == "__main__":
    train_ml_model()
