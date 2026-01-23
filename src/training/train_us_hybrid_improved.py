"""
US Hybrid Model Training - IMPROVED VERSION
Implements stacked meta-learner approach for true hybrid fusion.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import joblib
from datetime import datetime
import json

class ImprovedUSHybridModel:
    """Stacked hybrid model using meta-learner for optimal fusion."""
    
    def __init__(self):
        # Base models with scaling
        self.stat_model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0))
        ])
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            random_state=42
        )
        # Meta-learner (stacking)
        self.meta_learner = Ridge(alpha=0.1)
        self.feature_names = []
    
    def fit(self, X_train, y_train, X_val, y_val):
        """Train base models + meta-learner."""
        print("\nTraining Base Models...")
        
        # Train base models on training set
        print("  - Ridge (with scaling)...")
        self.stat_model.fit(X_train, y_train)
        
        print("  - Random Forest...")
        self.rf_model.fit(X_train, y_train)
        
        print("  - XGBoost...")
        self.xgb_model.fit(X_train, y_train)
        
        # Generate predictions on validation set for meta-learner
        print("\nGenerating Validation Predictions for Meta-Learner...")
        val_pred_stat = self.stat_model.predict(X_val)
        val_pred_rf = self.rf_model.predict(X_val)
        val_pred_xgb = self.xgb_model.predict(X_val)
        
        # Stack predictions as features for meta-learner
        meta_features = np.column_stack([val_pred_stat, val_pred_rf, val_pred_xgb])
        
        print("Training Meta-Learner (Stacking)...")
        self.meta_learner.fit(meta_features, y_val)
        
        print(f"Meta-learner coefficients: {self.meta_learner.coef_}")
    
    def predict(self, X):
        """Generate stacked prediction."""
        pred_stat = self.stat_model.predict(X)
        pred_rf = self.rf_model.predict(X)
        pred_xgb = self.xgb_model.predict(X)
        
        meta_features = np.column_stack([pred_stat, pred_rf, pred_xgb])
        return self.meta_learner.predict(meta_features)
    
    def get_base_predictions(self, X):
        """Return individual base model predictions for analysis."""
        return {
            'stat': self.stat_model.predict(X),
            'rf': self.rf_model.predict(X),
            'xgb': self.xgb_model.predict(X)
        }
    
    def save(self, directory):
        """Save all models."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.stat_model, f"{directory}/stat_model_scaled.pkl")
        joblib.dump(self.rf_model, f"{directory}/rf_model.pkl")
        joblib.dump(self.xgb_model, f"{directory}/xgb_model.pkl")
        joblib.dump(self.meta_learner, f"{directory}/meta_learner.pkl")
        joblib.dump(self.feature_names, f"{directory}/feature_names.pkl")

def load_us_data():
    """Load and preprocess US data."""
    df = pd.read_csv('data_processed/us/processed_data.csv')
    
    # Non-leaking features only
    feature_cols = [
        'avg_wage', 'log_total_wages', 'log_wage_returns',
        'num_wage_returns', 'wage_per_return', 'total_wages',
        'state_avg_income'
    ]
    
    X = df[feature_cols].values
    y = df['avg_income'].values
    
    return X, y, feature_cols

def ablation_study(X_train, y_train, X_val, y_val, X_test, y_test, model):
    """Compare all model variants."""
    print("\n" + "="*60)
    print("ABLATION STUDY - MODEL COMPARISON")
    print("="*60)
    
    results = {}
    
    # 1. Statistical only
    stat_pred = model.stat_model.predict(X_test)
    stat_r2 = r2_score(y_test, stat_pred)
    stat_rmse = np.sqrt(mean_squared_error(y_test, stat_pred))
    stat_mae = mean_absolute_error(y_test, stat_pred)
    
    print(f"\n1. Statistical (Ridge + Scaling):")
    print(f"   R² = {stat_r2:.4f}")
    print(f"   RMSE = ${stat_rmse*1000:.1f}")
    print(f"   MAE = ${stat_mae*1000:.1f}")
    
    results['statistical'] = {
        'r2': stat_r2, 'rmse': stat_rmse, 'mae': stat_mae
    }
    
    # 2. RF only
    rf_pred = model.rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)
    
    print(f"\n2. Random Forest:")
    print(f"   R² = {rf_r2:.4f}")
    print(f"   RMSE = ${rf_rmse*1000:.1f}")
    print(f"   MAE = ${rf_mae*1000:.1f}")
    
    results['random_forest'] = {
        'r2': rf_r2, 'rmse': rf_rmse, 'mae': rf_mae
    }
    
    # 3. XGBoost only
    xgb_pred = model.xgb_model.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    
    print(f"\n3. XGBoost:")
    print(f"   R² = {xgb_r2:.4f}")
    print(f"   RMSE = ${xgb_rmse*1000:.1f}")
    print(f"   MAE = ${xgb_mae*1000:.1f}")
    
    results['xgboost'] = {
        'r2': xgb_r2, 'rmse': xgb_rmse, 'mae': xgb_mae
    }
    
    # 4. Ensemble (simple average)
    ensemble_pred = (rf_pred + xgb_pred) / 2
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    
    print(f"\n4. ML Ensemble (RF + XGBoost avg):")
    print(f"   R² = {ensemble_r2:.4f}")
    print(f"   RMSE = ${ensemble_rmse*1000:.1f}")
    print(f"   MAE = ${ensemble_mae*1000:.1f}")
    
    results['ml_ensemble'] = {
        'r2': ensemble_r2, 'rmse': ensemble_rmse, 'mae': ensemble_mae
    }
    
    # 5. Stacked Hybrid (our improved approach)
    hybrid_pred = model.predict(X_test)
    hybrid_r2 = r2_score(y_test, hybrid_pred)
    hybrid_rmse = np.sqrt(mean_squared_error(y_test, hybrid_pred))
    hybrid_mae = mean_absolute_error(y_test, hybrid_pred)
    
    print(f"\n5. ⭐ STACKED HYBRID (Meta-Learner):")
    print(f"   R² = {hybrid_r2:.4f}")
    print(f"   RMSE = ${hybrid_rmse*1000:.1f}")
    print(f"   MAE = ${hybrid_mae*1000:.1f}")
    print(f"   Meta-learner weights: {model.meta_learner.coef_}")
    
    results['stacked_hybrid'] = {
        'r2': hybrid_r2, 'rmse': hybrid_rmse, 'mae': hybrid_mae,
        'meta_weights': model.meta_learner.coef_.tolist()
    }
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"\n✅ Best Model: {best_model[0].upper()} (R² = {best_model[1]['r2']:.4f})")
    
    return results

def generate_statistical_diagnostics(model, X_test, y_test, feature_names, output_dir):
    """Generate statistical diagnostics for Ridge model."""
    print("\nGenerating Statistical Diagnostics...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get Ridge coefficients (from pipeline)
    ridge = model.stat_model.named_steps['ridge']
    coefficients = ridge.coef_
    
    # Coefficient table
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    coef_path = output_dir / 'ridge_coefficients.csv'
    coef_df.to_csv(coef_path, index=False)
    print(f"  ✓ Coefficients saved: {coef_path}")
    
    # Residual analysis
    predictions = model.stat_model.predict(X_test)
    residuals = y_test - predictions
    
    residual_stats = {
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'min': float(np.min(residuals)),
        'max': float(np.max(residuals)),
        'median': float(np.median(residuals)),
        'q25': float(np.percentile(residuals, 25)),
        'q75': float(np.percentile(residuals, 75))
    }
    
    stats_path = output_dir / 'residual_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(residual_stats, f, indent=2)
    print(f"  ✓ Residual stats saved: {stats_path}")
    
    return coef_df, residual_stats

def main():
    print("="*60)
    print("US IMPROVED HYBRID MODEL TRAINING")
    print("Stacked Meta-Learner Approach")
    print("="*60)
    
    # Load data
    print("\nLoading US data...")
    X, y, feature_names = load_us_data()
    print(f"Total samples: {len(X)}")
    print(f"Features: {len(feature_names)}")
    
    # Split: train (60%), validation (20%), test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 of 80% = 20% total
    )
    
    print(f"\nTrain: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Train improved hybrid model
    model = ImprovedUSHybridModel()
    model.feature_names = feature_names
    model.fit(X_train, y_train, X_val, y_val)
    
    # Ablation study
    results = ablation_study(X_train, y_train, X_val, y_val, X_test, y_test, model)
    
    # Statistical diagnostics
    coef_df, residual_stats = generate_statistical_diagnostics(
        model, X_test, y_test, feature_names, 'reports/us'
    )
    
    # Save models
    print("\nSaving models...")
    model.save('models/us')
    print("  ✓ All models saved to models/us/")
    
    # Save ablation results
    results_path = Path('models/us/ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'train_samples': int(len(X_train)),
            'val_samples': int(len(X_val)),
            'test_samples': int(len(X_test)),
            'features': feature_names,
            'results': results
        }, f, indent=2)
    print(f"  ✓ Ablation results: {results_path}")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Run: python generate_report_plots.py")
    print(f"2. Check: reports/us/ for diagnostic plots")
    print(f"3. Review: models/us/ablation_results.json")

if __name__ == "__main__":
    main()
