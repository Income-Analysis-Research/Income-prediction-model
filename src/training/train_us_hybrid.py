"""
Train US Hybrid Model (Statistical + ML Fusion)

Combines Ridge regression and ML ensemble with confidence-weighted fusion.
"""

import os, json, pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
from src.hybrid.us_hybrid_model import USHybridModel


def load_data():
    df = pd.read_csv("datasets/us_irs_zipcode_data.csv")
    df['avg_income'] = df['agi_amount'] / df['num_returns'].replace(0, np.nan)
    df = df[df['avg_income'].notna() & (df['avg_income'] > 0) & (df['avg_income'] < 1e7)]
    
    features = []
    if 'avg_wage' in df.columns: features.append('avg_wage')
    if 'total_wages' in df.columns:
        df['log_total_wages'] = np.log1p(df['total_wages'])
        features.extend(['log_total_wages', 'total_wages'])
    if 'num_wage_returns' in df.columns:
        df['log_wage_returns'] = np.log1p(df['num_wage_returns'])
        features.extend(['log_wage_returns', 'num_wage_returns'])
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


print("\\n" + "="*60)
print("US HYBRID MODEL TRAINING")
print("="*60)

X, y, features = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training hybrid model with {len(features)} features...")
model = USHybridModel(alpha=1.0, use_xgboost=True)
model.fit(X_train, y_train, features)

# Evaluate
result = model.predict(X_test, return_components=True)
hybrid_pred = result['prediction']
test_r2 = r2_score(y_test, hybrid_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, hybrid_pred))
test_mae = mean_absolute_error(y_test, hybrid_pred)

print(f"\\nHybrid Test R²: {test_r2:.4f}")
print(f"Test RMSE: ${test_rmse/1000:.1f}k")
print(f"Fusion weights: Stat={model.base_stat_weight:.2f}, ML={model.base_ml_weight:.2f}")

# Save
import joblib
models_dir = Path("models/us")
models_dir.mkdir(parents=True, exist_ok=True)
model.save(str(models_dir))

with open(models_dir / "results_hybrid.txt", 'w') as f:
    f.write(f"US HYBRID MODEL\\nTest R²: {test_r2:.4f}\\nRMSE: ${test_rmse/1000:.1f}k\\nWeights: Stat={model.base_stat_weight:.2f}, ML={model.base_ml_weight:.2f}\\n")

print("Saved to models/us/")
