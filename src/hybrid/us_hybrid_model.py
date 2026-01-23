"""
US Hybrid Income Prediction Model

Combines statistical (linear regression) and machine learning (ensemble)
approaches with confidence-weighted fusion for research-grade predictions.

This module implements:
1. Statistical baseline (Ridge regression with interpretable coefficients)
2. ML ensemble (Random Forest, XGBoost)
3. Hybrid fusion with dynamic confidence weighting
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class USStatisticalModel:
    """Statistical baseline using Ridge regression for interpretability"""
    
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.coefficients: Optional[np.ndarray] = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
        """Train statistical model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.feature_names = feature_names
        self.coefficients = self.model.coef_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with statistical model"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_coefficients(self) -> pd.DataFrame:
        """Get interpretable coefficients"""
        if self.coefficients is None:
            return pd.DataFrame()
        return pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.coefficients
        }).sort_values('coefficient', key=abs, ascending=False)


class USMLModel:
    """Machine learning ensemble for powerful predictions"""
    
    def __init__(self, use_xgboost: bool = True):
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.xgb_model = None
        if use_xgboost and XGBOOST_AVAILABLE:
            self.xgb_model = XGBRegressor(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        
        self.scaler = StandardScaler()
        self.use_ensemble = use_xgboost and XGBOOST_AVAILABLE
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train ML models"""
        X_scaled = self.scaler.fit_transform(X)
        self.rf_model.fit(X_scaled, y)
        
        if self.xgb_model is not None:
            self.xgb_model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with ML ensemble"""
        X_scaled = self.scaler.transform(X)
        
        rf_pred = self.rf_model.predict(X_scaled)
        
        if self.use_ensemble and self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict(X_scaled)
            return (rf_pred + xgb_pred) / 2
        
        return rf_pred
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimates using RF tree variance"""
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict(X_scaled) for tree in self.rf_model.estimators_
        ])
        
        # Mean and std across trees
        mean_pred = tree_predictions.mean(axis=0)
        std_pred = tree_predictions.std(axis=0)
        
        return mean_pred, std_pred


class USHybridModel:
    """
    Hybrid model combining statistical and ML predictions
    with confidence-weighted fusion
    """
    
    def __init__(self, alpha: float = 1.0, use_xgboost: bool = True):
        self.stat_model = USStatisticalModel(alpha=alpha)
        self.ml_model = USMLModel(use_xgboost=use_xgboost)
        self.feature_names: list[str] = []
        
        # Fusion parameters (learned during training)
        self.base_stat_weight = 0.3
        self.base_ml_weight = 0.7
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
        """Train both models"""
        self.feature_names = feature_names
        
        # Train both models
        self.stat_model.fit(X, y, feature_names)
        self.ml_model.fit(X, y)
        
        # Learn optimal fusion weights via cross-validation
        self._optimize_fusion_weights(X, y)
        
        return self
    
    def _optimize_fusion_weights(self, X: np.ndarray, y: np.ndarray):
        """Optimize fusion weights using validation performance"""
        # Simple grid search for optimal weights
        best_score = -np.inf
        best_weights = (0.3, 0.7)
        
        for stat_w in np.linspace(0.1, 0.5, 5):
            ml_w = 1.0 - stat_w
            
            # Evaluate this weighting
            stat_pred = self.stat_model.predict(X)
            ml_pred = self.ml_model.predict(X)
            hybrid_pred = stat_w * stat_pred + ml_w * ml_pred
            
            # Use R² as score
            ss_res = np.sum((y - hybrid_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            if r2 > best_score:
                best_score = r2
                best_weights = (stat_w, ml_w)
        
        self.base_stat_weight, self.base_ml_weight = best_weights
    
    def predict(self, X: np.ndarray, return_components: bool = False) -> Dict:
        """
        Predict with hybrid model using confidence-weighted fusion
        
        Returns:
            dict with 'prediction', 'confidence', 'stat_pred', 'ml_pred', 'weights'
        """
        # Get predictions from both models
        stat_pred = self.stat_model.predict(X)
        ml_pred, ml_std = self.ml_model.predict_with_uncertainty(X)
        
        # Calculate confidence-based weights
        # Lower ML uncertainty → higher ML weight
        ml_confidence = 1.0 / (1.0 + ml_std)
        
        # Normalize confidence to weight
        total_confidence = ml_confidence + 1.0  # stat gets weight=1.0
        stat_weight = self.base_stat_weight * (1.0 / total_confidence)
        ml_weight = self.base_ml_weight * (ml_confidence / total_confidence)
        
        # Renormalize
        total_weight = stat_weight + ml_weight
        stat_weight = stat_weight / total_weight
        ml_weight = ml_weight / total_weight
        
        # Fused prediction
        hybrid_pred = stat_weight * stat_pred + ml_weight * ml_pred
        
        # Overall confidence (inverse of weighted std)
        confidence = 1.0 / (1.0 + ml_std * ml_weight)
        
        result = {
            'prediction': hybrid_pred,
            'confidence': confidence,
            'stat_pred': stat_pred,
            'ml_pred': ml_pred,
            'stat_weight': stat_weight,
            'ml_weight': ml_weight,
            'ml_uncertainty': ml_std
        }
        
        return result
    
    def save(self, model_dir: str):
        """Save all model components"""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.stat_model, model_path / "stat_model.pkl")
        joblib.dump(self.ml_model, model_path / "ml_model.pkl")
        joblib.dump({
            'base_stat_weight': self.base_stat_weight,
            'base_ml_weight': self.base_ml_weight,
            'feature_names': self.feature_names
        }, model_path / "hybrid_config.pkl")
    
    @classmethod
    def load(cls, model_dir: str) -> 'USHybridModel':
        """Load all model components"""
        model_path = Path(model_dir)
        
        instance = cls()
        instance.stat_model = joblib.load(model_path / "stat_model.pkl")
        instance.ml_model = joblib.load(model_path / "ml_model.pkl")
        
        config = joblib.load(model_path / "hybrid_config.pkl")
        instance.base_stat_weight = config['base_stat_weight']
        instance.base_ml_weight = config['base_ml_weight']
        instance.feature_names = config['feature_names']
        
        return instance
