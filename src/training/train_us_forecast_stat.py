"""
Train US Income Forecasting Model - Statistical Approach.

Uses Ridge and ElasticNet regression with cross-validation.
Features: Year t statistics
Target: Year t+1 average income

Usage:
    python src/training/train_us_forecast_stat.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
FORECAST_DATA_PATH = Path("datasets/us_forecasting_dataset.csv")
OUTPUT_DIR = Path("models/us_forecast")
MODEL_NAME = "stat_model"

# Features to use
FEATURE_COLUMNS = [
    'n_returns_t',
    'total_wages_t',
    'n_wages_t',
    'total_interest_t',
    'n_interest_t',
    'total_cap_gains_t',
    'n_cap_gains_t',
    'total_business_t',
    'n_business_t',
    'avg_agi_t',
    'pct_with_wages_t',
    'avg_wage_per_earner_t',
    'pct_with_business_t',
    'pct_with_cap_gains_t',
    'log_returns_t'
]

TARGET_COLUMN = 'target_income_tplus1'


def load_data():
    """Load and split forecasting dataset."""
    print("Loading forecasting dataset...")
    df = pd.read_csv(FORECAST_DATA_PATH)
    
    # Split by pre-defined split column
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    print(f"Train: {len(train_df):,} pairs ({train_df['from_year'].min()}→{train_df['to_year'].max()})")
    print(f"Test:  {len(test_df):,} pairs ({test_df['from_year'].min()}→{test_df['to_year'].max()})")
    
    # Extract features and target
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMN].values
    
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COLUMN].values
    
    return X_train, X_test, y_train, y_test, train_df, test_df


def train_ridge(X_train, y_train):
    """Train Ridge regression with cross-validation."""
    print("\nTraining Ridge Regression...")
    
    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
    }
    
    ridge = Ridge(random_state=42)
    grid_search = GridSearchCV(
        ridge,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best alpha: {grid_search.best_params_['alpha']}")
    print(f"Best CV RMSE: ${np.sqrt(-grid_search.best_score_):,.0f}")
    
    return grid_search.best_estimator_


def train_elasticnet(X_train, y_train):
    """Train ElasticNet with cross-validation."""
    print("\nTraining ElasticNet...")
    
    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    
    elasticnet = ElasticNet(random_state=42, max_iter=10000)
    grid_search = GridSearchCV(
        elasticnet,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best alpha: {grid_search.best_params_['alpha']}")
    print(f"Best l1_ratio: {grid_search.best_params_['l1_ratio']}")
    print(f"Best CV RMSE: ${np.sqrt(-grid_search.best_score_):,.0f}")
    
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n{model_name} Test Results:")
    print(f"  R²:   {r2:.4f}")
    print(f"  RMSE: ${rmse:,.0f}")
    print(f"  MAE:  ${mae:,.0f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape)
    }


def main():
    print("="*70)
    print("US INCOME FORECASTING - STATISTICAL MODELS")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test, train_df, test_df = load_data()
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Ridge
    ridge_model = train_ridge(X_train_scaled, y_train)
    ridge_metrics = evaluate_model(ridge_model, X_test_scaled, y_test, "Ridge")
    
    # Train ElasticNet
    elasticnet_model = train_elasticnet(X_train_scaled, y_train)
    elasticnet_metrics = evaluate_model(elasticnet_model, X_test_scaled, y_test, "ElasticNet")
    
    # Select best model
    if ridge_metrics['r2'] >= elasticnet_metrics['r2']:
        best_model = ridge_model
        best_metrics = ridge_metrics
        best_name = "Ridge"
    else:
        best_model = elasticnet_model
        best_metrics = elasticnet_metrics
        best_name = "ElasticNet"
    
    print("\n" + "="*70)
    print(f"BEST MODEL: {best_name}")
    print("="*70)
    print(f"R²:   {best_metrics['r2']:.4f}")
    print(f"RMSE: ${best_metrics['rmse']:,.0f}")
    print(f"MAE:  ${best_metrics['mae']:,.0f}")
    
    # Save model and scaler
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = OUTPUT_DIR / f"{MODEL_NAME}.pkl"
    scaler_path = OUTPUT_DIR / f"{MODEL_NAME}_scaler.pkl"
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel saved: {model_path}")
    print(f"Scaler saved: {scaler_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'statistical',
        'algorithm': best_name,
        'features': FEATURE_COLUMNS,
        'target': TARGET_COLUMN,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_years': f"{train_df['from_year'].min()}→{train_df['to_year'].max()}",
        'test_year': f"{test_df['from_year'].min()}→{test_df['to_year'].max()}",
        'metrics': best_metrics,
        'ridge_metrics': ridge_metrics,
        'elasticnet_metrics': elasticnet_metrics,
        'scaler': 'StandardScaler',
        'ridge_params': {
            'alpha': float(ridge_model.alpha)
        },
        'elasticnet_params': {
            'alpha': float(elasticnet_model.alpha),
            'l1_ratio': float(elasticnet_model.l1_ratio)
        }
    }
    
    metadata_path = OUTPUT_DIR / f"{MODEL_NAME}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved: {metadata_path}")
    
    # Save predictions
    y_pred = best_model.predict(X_test_scaled)
    predictions_df = test_df[['zip', 'state', 'from_year', 'to_year']].copy()
    predictions_df['actual'] = y_test
    predictions_df['predicted'] = y_pred
    predictions_df['error'] = y_test - y_pred
    predictions_df['abs_error'] = np.abs(predictions_df['error'])
    predictions_df['pct_error'] = (predictions_df['error'] / y_test) * 100
    
    predictions_path = OUTPUT_DIR / f"{MODEL_NAME}_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved: {predictions_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\n✅ Statistical forecasting model ready")
    print(f"✅ Test R²: {best_metrics['r2']:.4f}")
    print(f"✅ Test RMSE: ${best_metrics['rmse']:,.0f}")


if __name__ == '__main__':
    main()
