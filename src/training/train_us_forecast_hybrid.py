"""
Train US Income Forecasting Model - Hybrid Approach.

Combines statistical and ML models using stacking ensemble.
Meta-learner learns optimal weights for base models.

Usage:
    python src/training/train_us_forecast_hybrid.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
FORECAST_DATA_PATH = Path("datasets/us_forecasting_dataset.csv")
OUTPUT_DIR = Path("models/us_forecast")
MODEL_NAME = "hybrid_model"

# Pre-trained models
STAT_MODEL_PATH = OUTPUT_DIR / "stat_model.pkl"
STAT_SCALER_PATH = OUTPUT_DIR / "stat_model_scaler.pkl"
ML_MODEL_PATH = OUTPUT_DIR / "ml_model.pkl"

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


def load_base_models():
    """Load pre-trained base models."""
    print("\nLoading pre-trained base models...")
    
    models = {}
    
    # Load statistical model
    if STAT_MODEL_PATH.exists():
        stat_model = joblib.load(STAT_MODEL_PATH)
        stat_scaler = joblib.load(STAT_SCALER_PATH)
        models['stat'] = (stat_model, stat_scaler)
        print(f"✓ Loaded: {STAT_MODEL_PATH.name}")
    else:
        print(f"✗ Not found: {STAT_MODEL_PATH}")
    
    # Load ML model
    if ML_MODEL_PATH.exists():
        ml_model = joblib.load(ML_MODEL_PATH)
        models['ml'] = ml_model
        print(f"✓ Loaded: {ML_MODEL_PATH.name}")
    else:
        print(f"✗ Not found: {ML_MODEL_PATH}")
    
    if len(models) < 2:
        raise FileNotFoundError("Need both stat and ML models. Train them first.")
    
    return models


def create_hybrid_features(X, base_models, scaler_stat):
    """Create hybrid features by adding base model predictions."""
    # Statistical model predictions (needs scaling)
    X_stat_scaled = scaler_stat.transform(X)
    stat_preds = base_models['stat'][0].predict(X_stat_scaled)
    
    # ML model predictions (no scaling needed)
    ml_preds = base_models['ml'].predict(X)
    
    # Stack predictions as new features
    X_hybrid = np.column_stack([X, stat_preds, ml_preds])
    
    return X_hybrid


def train_hybrid_model(X_train, y_train, base_models):
    """Train hybrid stacking model."""
    print("\nTraining hybrid stacking model...")
    
    # Create hybrid features
    scaler_stat = base_models['stat'][1]
    X_train_hybrid = create_hybrid_features(X_train, base_models, scaler_stat)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Hybrid features: {X_train_hybrid.shape[1]} (original + 2 predictions)")
    
    # Simple Ridge meta-learner
    meta_learner = Ridge(alpha=10.0)
    meta_learner.fit(X_train_hybrid, y_train)
    
    # Get meta-learner weights for base model predictions
    # Last 2 features are stat and ml predictions
    weights = meta_learner.coef_[-2:]
    print(f"\nMeta-learner weights:")
    print(f"  Statistical model: {weights[0]:.4f}")
    print(f"  ML model:          {weights[1]:.4f}")
    
    return meta_learner, weights


def evaluate_model(model, X_test, y_test, base_models, scaler_stat):
    """Evaluate hybrid model on test set."""
    # Create hybrid features
    X_test_hybrid = create_hybrid_features(X_test, base_models, scaler_stat)
    
    y_pred = model.predict(X_test_hybrid)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\nHybrid Test Results:")
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
    print("US INCOME FORECASTING - HYBRID MODEL")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test, train_df, test_df = load_data()
    
    # Load base models
    base_models = load_base_models()
    
    # Train hybrid model
    scaler_stat = base_models['stat'][1]
    hybrid_model, weights = train_hybrid_model(X_train, y_train, base_models)
    
    # Evaluate
    metrics = evaluate_model(hybrid_model, X_test, y_test, base_models, scaler_stat)
    
    print("\n" + "="*70)
    print("HYBRID MODEL PERFORMANCE")
    print("="*70)
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"RMSE: ${metrics['rmse']:,.0f}")
    print(f"MAE:  ${metrics['mae']:,.0f}")
    
    # Save model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = OUTPUT_DIR / f"{MODEL_NAME}.pkl"
    joblib.dump(hybrid_model, model_path)
    print(f"\nModel saved: {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'hybrid',
        'algorithm': 'stacking_ensemble',
        'base_models': ['stat_model', 'ml_model'],
        'meta_learner': 'Ridge',
        'meta_learner_weights': {
            'stat': float(weights[0]),
            'ml': float(weights[1])
        },
        'features': FEATURE_COLUMNS,
        'target': TARGET_COLUMN,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_years': f"{train_df['from_year'].min()}→{train_df['to_year'].max()}",
        'test_year': f"{test_df['from_year'].min()}→{test_df['to_year'].max()}",
        'metrics': metrics
    }
    
    metadata_path = OUTPUT_DIR / f"{MODEL_NAME}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")
    
    # Save predictions
    X_test_hybrid = create_hybrid_features(X_test, base_models, scaler_stat)
    y_pred = hybrid_model.predict(X_test_hybrid)
    
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
    print("\n✅ Hybrid forecasting model ready")
    print(f"✅ Test R²: {metrics['r2']:.4f}")
    print(f"✅ Test RMSE: ${metrics['rmse']:,.0f}")
    
    print("\n📊 Model Comparison Available:")
    print("  - stat_model: Ridge/ElasticNet")
    print("  - ml_model: RandomForest/XGBoost")
    print("  - hybrid_model: Stacking Ensemble")


if __name__ == '__main__':
    main()
