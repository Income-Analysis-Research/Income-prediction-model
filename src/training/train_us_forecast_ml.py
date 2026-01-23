"""
Train US Income Forecasting Model - Machine Learning Approach.

Uses XGBoost and Random Forest with hyperparameter tuning.
Features: Year t statistics
Target: Year t+1 average income

Usage:
    python src/training/train_us_forecast_ml.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, will use RandomForest only")

# Paths
FORECAST_DATA_PATH = Path("datasets/us_forecasting_dataset.csv")
OUTPUT_DIR = Path("models/us_forecast")
MODEL_NAME_XGB = "xgb_forecast"
MODEL_NAME_RF = "rf_forecast"
MODEL_NAME_ML = "ml_model"

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


def train_xgboost(X_train, y_train):
    """Train XGBoost with hyperparameter tuning."""
    if not XGBOOST_AVAILABLE:
        return None
    
    print("\nTraining XGBoost...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    
    xgb = XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist')
    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV RMSE: ${np.sqrt(-grid_search.best_score_):,.0f}")
    
    return grid_search.best_estimator_


def train_random_forest(X_train, y_train):
    """Train Random Forest with hyperparameter tuning."""
    print("\nTraining Random Forest...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best params: {grid_search.best_params_}")
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


def get_feature_importance(model, feature_names, model_name):
    """Extract feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n{model_name} Top 10 Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:25s}: {row['importance']:.4f}")
        
        return importance_df.to_dict('records')
    
    return None


def main():
    print("="*70)
    print("US INCOME FORECASTING - MACHINE LEARNING MODELS")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test, train_df, test_df = load_data()
    
    models = {}
    metrics = {}
    
    # Train XGBoost
    if XGBOOST_AVAILABLE:
        xgb_model = train_xgboost(X_train, y_train)
        if xgb_model:
            models['XGBoost'] = xgb_model
            metrics['XGBoost'] = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
            xgb_importance = get_feature_importance(xgb_model, FEATURE_COLUMNS, "XGBoost")
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    models['RandomForest'] = rf_model
    metrics['RandomForest'] = evaluate_model(rf_model, X_test, y_test, "RandomForest")
    rf_importance = get_feature_importance(rf_model, FEATURE_COLUMNS, "RandomForest")
    
    # Select best model
    best_name = max(metrics, key=lambda k: metrics[k]['r2'])
    best_model = models[best_name]
    best_metrics = metrics[best_name]
    
    print("\n" + "="*70)
    print(f"BEST MODEL: {best_name}")
    print("="*70)
    print(f"R²:   {best_metrics['r2']:.4f}")
    print(f"RMSE: ${best_metrics['rmse']:,.0f}")
    print(f"MAE:  ${best_metrics['mae']:,.0f}")
    
    # Save best model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = OUTPUT_DIR / f"{MODEL_NAME_ML}.pkl"
    joblib.dump(best_model, model_path)
    print(f"\nModel saved: {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'machine_learning',
        'algorithm': best_name,
        'features': FEATURE_COLUMNS,
        'target': TARGET_COLUMN,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_years': f"{train_df['from_year'].min()}→{train_df['to_year'].max()}",
        'test_year': f"{test_df['from_year'].min()}→{test_df['to_year'].max()}",
        'metrics': best_metrics,
        'all_metrics': metrics,
        'feature_importance': xgb_importance if best_name == 'XGBoost' else rf_importance,
        'xgboost_available': XGBOOST_AVAILABLE
    }
    
    if best_name == 'XGBoost' and XGBOOST_AVAILABLE:
        metadata['model_params'] = {
            'n_estimators': int(best_model.n_estimators),
            'max_depth': int(best_model.max_depth),
            'learning_rate': float(best_model.learning_rate),
            'subsample': float(best_model.subsample)
        }
    elif best_name == 'RandomForest':
        metadata['model_params'] = {
            'n_estimators': int(best_model.n_estimators),
            'max_depth': int(best_model.max_depth) if best_model.max_depth else None,
            'min_samples_split': int(best_model.min_samples_split),
            'min_samples_leaf': int(best_model.min_samples_leaf)
        }
    
    metadata_path = OUTPUT_DIR / f"{MODEL_NAME_ML}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")
    
    # Save predictions
    y_pred = best_model.predict(X_test)
    predictions_df = test_df[['zip', 'state', 'from_year', 'to_year']].copy()
    predictions_df['actual'] = y_test
    predictions_df['predicted'] = y_pred
    predictions_df['error'] = y_test - y_pred
    predictions_df['abs_error'] = np.abs(predictions_df['error'])
    predictions_df['pct_error'] = (predictions_df['error'] / y_test) * 100
    
    predictions_path = OUTPUT_DIR / f"{MODEL_NAME_ML}_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved: {predictions_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\n✅ Machine learning forecasting model ready")
    print(f"✅ Test R²: {best_metrics['r2']:.4f}")
    print(f"✅ Test RMSE: ${best_metrics['rmse']:,.0f}")


if __name__ == '__main__':
    main()
