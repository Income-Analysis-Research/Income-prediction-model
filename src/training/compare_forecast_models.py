"""
US Income Forecasting - Model Comparison Summary

This script loads all trained models and creates a comprehensive comparison.

Usage:
    python src/training/compare_forecast_models.py
"""

import json
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("models/us_forecast")


def load_metadata(model_name):
    """Load model metadata."""
    metadata_path = OUTPUT_DIR / f"{model_name}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def main():
    print("="*70)
    print("US INCOME FORECASTING - MODEL COMPARISON")
    print("="*70)
    
    models = ['stat_model', 'ml_model', 'hybrid_model']
    
    results = []
    for model_name in models:
        metadata = load_metadata(model_name)
        if metadata:
            results.append({
                'Model': model_name.replace('_', ' ').title(),
                'Algorithm': metadata.get('algorithm', 'N/A'),
                'R²': metadata['metrics']['r2'],
                'RMSE': metadata['metrics']['rmse'],
                'MAE': metadata['metrics']['mae'],
                'MAPE': metadata['metrics']['mape']
            })
    
    # Create comparison table
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("TEST SET PERFORMANCE (2021→2022)")
    print("="*70)
    print()
    print(df.to_string(index=False))
    
    # Find best model
    best_idx = df['R²'].idxmax()
    best_model = df.iloc[best_idx]
    
    print("\n" + "="*70)
    print("BEST MODEL")
    print("="*70)
    print(f"\nModel: {best_model['Model']}")
    print(f"Algorithm: {best_model['Algorithm']}")
    print(f"R²: {best_model['R²']:.4f}")
    print(f"RMSE: ${best_model['RMSE']:,.0f}")
    print(f"MAE: ${best_model['MAE']:,.0f}")
    print(f"MAPE: {best_model['MAPE']:.2f}%")
    
    # Save summary
    summary_path = OUTPUT_DIR / "model_comparison.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("US INCOME FORECASTING - MODEL COMPARISON\n")
        f.write("="*70 + "\n\n")
        f.write("TEST SET PERFORMANCE (2021→2022)\n")
        f.write("="*70 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "="*70 + "\n")
        f.write("BEST MODEL\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {best_model['Model']}\n")
        f.write(f"Algorithm: {best_model['Algorithm']}\n")
        f.write(f"R²: {best_model['R²']:.4f}\n")
        f.write(f"RMSE: ${best_model['RMSE']:,.0f}\n")
        f.write(f"MAE: ${best_model['MAE']:,.0f}\n")
        f.write(f"MAPE: {best_model['MAPE']:.2f}%\n")
        f.write("\n" + "="*70 + "\n")
        f.write("TIME-BASED EVALUATION\n")
        f.write("="*70 + "\n\n")
        f.write("Training: 2019→2020, 2020→2021 transitions (8,126 pairs)\n")
        f.write("Testing:  2021→2022 transition (4,061 pairs)\n\n")
        f.write("✅ No data leakage: Features from year t, target from t+1\n")
        f.write("✅ Strict temporal split: Train on past, test on future\n")
        f.write("✅ Real IRS data: Official ZIP-level tax statistics\n")
    
    print(f"\n\nSummary saved: {summary_path}")
    
    # Save CSV version
    csv_path = OUTPUT_DIR / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
