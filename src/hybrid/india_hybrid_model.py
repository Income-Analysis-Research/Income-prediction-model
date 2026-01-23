"""
India Hybrid Income Proxy Model

Combines statistical and ML approaches for district-level income proxy estimation.
Uses socioeconomic indicators to create a relative income index (0-100).

Components:
1. Statistical model (Ridge regression with domain knowledge)
2. ML ensemble (Random Forest with uncertainty)
3. Hybrid fusion with confidence weighting
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional


class IndiaStatisticalModel:
    """Statistical baseline for income proxy prediction"""
    
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
        """Predict income proxy index"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return np.clip(predictions, 0, 100)
    
    def get_coefficients(self) -> pd.DataFrame:
        """Get interpretable coefficients"""
        if self.coefficients is None:
            return pd.DataFrame()
        return pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.coefficients
        }).sort_values('coefficient', key=abs, ascending=False)


class IndiaMLModel:
    """ML ensemble for income proxy prediction"""
    
    def __init__(self):
        self.reg_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.class_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        self.category_bins = [0, 40, 70, 100]
        self.category_labels = ['Low', 'Middle', 'Upper-Middle']
        
    def fit(self, X: np.ndarray, y_reg: np.ndarray, y_class: Optional[np.ndarray] = None):
        """Train regression and classification models"""
        X_scaled = self.scaler.fit_transform(X)
        
        # Train regression
        self.reg_model.fit(X_scaled, y_reg)
        
        # Train classification if labels provided
        if y_class is not None:
            self.class_model.fit(X_scaled, y_class)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict income proxy index"""
        X_scaled = self.scaler.transform(X)
        predictions = self.reg_model.predict(X_scaled)
        return np.clip(predictions, 0, 100)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty using tree variance"""
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict(X_scaled) for tree in self.reg_model.estimators_
        ])
        
        mean_pred = tree_predictions.mean(axis=0)
        std_pred = tree_predictions.std(axis=0)
        
        # Clip predictions to valid range
        mean_pred = np.clip(mean_pred, 0, 100)
        
        return mean_pred, std_pred
    
    def predict_category(self, X: np.ndarray) -> np.ndarray:
        """Predict income category"""
        X_scaled = self.scaler.transform(X)
        return self.class_model.predict(X_scaled)


class IndiaHybridModel:
    """
    Hybrid model for India district income proxy prediction
    Combines statistical interpretability with ML power
    """
    
    def __init__(self, alpha: float = 1.0):
        self.stat_model = IndiaStatisticalModel(alpha=alpha)
        self.ml_model = IndiaMLModel()
        self.feature_names: list[str] = []
        
        # Fusion parameters
        self.base_stat_weight = 0.4
        self.base_ml_weight = 0.6
        
    def fit(self, X: np.ndarray, y_reg: np.ndarray, y_class: Optional[np.ndarray], 
            feature_names: list[str]):
        """Train both models"""
        self.feature_names = feature_names
        
        # Train both models
        self.stat_model.fit(X, y_reg, feature_names)
        self.ml_model.fit(X, y_reg, y_class)
        
        # Optimize fusion weights
        self._optimize_fusion_weights(X, y_reg)
        
        return self
    
    def _optimize_fusion_weights(self, X: np.ndarray, y: np.ndarray):
        """Optimize fusion weights"""
        best_score = -np.inf
        best_weights = (0.4, 0.6)
        
        for stat_w in np.linspace(0.2, 0.6, 5):
            ml_w = 1.0 - stat_w
            
            stat_pred = self.stat_model.predict(X)
            ml_pred = self.ml_model.predict(X)
            hybrid_pred = stat_w * stat_pred + ml_w * ml_pred
            
            ss_res = np.sum((y - hybrid_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            if r2 > best_score:
                best_score = r2
                best_weights = (stat_w, ml_w)
        
        self.base_stat_weight, self.base_ml_weight = best_weights
    
    def predict(self, X: np.ndarray, return_components: bool = False) -> Dict:
        """
        Predict with hybrid model
        
        Returns:
            dict with prediction, confidence, category, and component predictions
        """
        # Get predictions
        stat_pred = self.stat_model.predict(X)
        ml_pred, ml_std = self.ml_model.predict_with_uncertainty(X)
        
        # Confidence-based weighting
        ml_confidence = 1.0 / (1.0 + ml_std)
        total_confidence = ml_confidence + 1.0
        
        stat_weight = self.base_stat_weight * (1.0 / total_confidence)
        ml_weight = self.base_ml_weight * (ml_confidence / total_confidence)
        
        # Normalize
        total_weight = stat_weight + ml_weight
        stat_weight = stat_weight / total_weight
        ml_weight = ml_weight / total_weight
        
        # Fused prediction
        hybrid_pred = stat_weight * stat_pred + ml_weight * ml_pred
        hybrid_pred = np.clip(hybrid_pred, 0, 100)
        
        # Category prediction
        category = self.ml_model.predict_category(X)
        
        # Confidence
        confidence = 1.0 / (1.0 + ml_std * ml_weight)
        
        result = {
            'prediction': hybrid_pred,
            'confidence': confidence,
            'category': category,
            'stat_pred': stat_pred,
            'ml_pred': ml_pred,
            'stat_weight': stat_weight,
            'ml_weight': ml_weight,
            'ml_uncertainty': ml_std
        }
        
        return result
    
    def save(self, model_dir: str):
        """Save all components"""
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
    def load(cls, model_dir: str) -> 'IndiaHybridModel':
        """Load all components"""
        model_path = Path(model_dir)
        
        instance = cls()
        instance.stat_model = joblib.load(model_path / "stat_model.pkl")
        instance.ml_model = joblib.load(model_path / "ml_model.pkl")
        
        config = joblib.load(model_path / "hybrid_config.pkl")
        instance.base_stat_weight = config['base_stat_weight']
        instance.base_ml_weight = config['base_ml_weight']
        instance.feature_names = config['feature_names']
        
        return instance
