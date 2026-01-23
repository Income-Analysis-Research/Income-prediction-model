"""
Improved FastAPI Backend with Research Metrics Endpoints
Fixes RMSE display bug and adds comprehensive metrics API.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import logging
import json
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Income Prediction Research API",
    description="Research-grade income estimation with hybrid statistical-ML models",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount reports directory for plot access
reports_dir = Path("reports")
if reports_dir.exists():
    app.mount("/reports", StaticFiles(directory="reports"), name="reports")

# Mount frontend directory for static HTML/CSS/JS
frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="frontend")


# Serve frontend at root
@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    frontend_path = Path("frontend/index.html")
    if not frontend_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    
    from fastapi.responses import HTMLResponse
    with open(frontend_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


class USModelLoader:
    """Load and manage US improved hybrid models."""
    
    def __init__(self):
        self.stat_model = None
        self.rf_model = None
        self.xgb_model = None
        self.gb_model = None  # NEW: Gradient Boosting (best multiyear)
        self.meta_learner = None
        self.feature_names = []
        self.ablation_results = None
        self.training_results = None  # NEW: Multiyear results
        
    def load_models(self):
        try:
            # Try loading multiyear models first (preferred)
            multiyear_path = Path("models/us_multiyear")
            if multiyear_path.exists():
                logger.info("Loading multi-year models (2018-2022)...")
                self.stat_model = joblib.load(multiyear_path / "ridge_model.pkl")
                self.rf_model = joblib.load(multiyear_path / "rf_model.pkl")
                self.xgb_model = joblib.load(multiyear_path / "xgboost_model.pkl")
                self.gb_model = joblib.load(multiyear_path / "gb_model.pkl")
                
                with open(multiyear_path / "feature_names.json", 'r') as f:
                    self.feature_names = json.load(f)
                
                with open(multiyear_path / "training_results.json", 'r') as f:
                    self.training_results = json.load(f)
                
                logger.info("✓ US multi-year models loaded (R²=0.90)")
                return True
            else:
                # Fallback to old single-year models
                logger.info("Loading single-year models...")
                self.stat_model = joblib.load("models/us/stat_model_scaled.pkl")
                self.rf_model = joblib.load("models/us/rf_model.pkl")
                self.xgb_model = joblib.load("models/us/xgb_model.pkl")
                self.meta_learner = joblib.load("models/us/meta_learner.pkl")
                self.feature_names = joblib.load("models/us/feature_names.pkl")
                
                with open("models/us/ablation_results.json", 'r') as f:
                    self.ablation_results = json.load(f)
                
                logger.info("✓ US single-year models loaded (R²=0.79)")
                return True
        except Exception as e:
            logger.error(f"✗ Failed to load US models: {e}")
            return False
    
    def predict(self, features: np.ndarray, method: str = "hybrid"):
        """
        Generate predictions using specified method.
        method: 'stat', 'rf', 'xgb', 'gb', 'hybrid'
        """
        if self.stat_model is None:
            raise RuntimeError("Models not loaded")
        
        if method == "stat":
            assert self.stat_model is not None
            return self.stat_model.predict(features)
        elif method == "rf":
            assert self.rf_model is not None
            return self.rf_model.predict(features)
        elif method == "xgb":
            assert self.xgb_model is not None
            return self.xgb_model.predict(features)
        elif method == "gb":
            # Use gradient boosting if available (best multiyear model)
            if self.gb_model is not None:
                return self.gb_model.predict(features)
            else:
                # Fallback to xgboost
                return self.xgb_model.predict(features)
        elif method == "hybrid":
            if self.meta_learner is not None:
                # Old stacking approach
                pred_stat = self.stat_model.predict(features)
                pred_rf = self.rf_model.predict(features)
                pred_xgb = self.xgb_model.predict(features)
                meta_features = np.column_stack([pred_stat, pred_rf, pred_xgb])
                return self.meta_learner.predict(meta_features)
            elif self.gb_model is not None:
                # Use best multiyear model
                return self.gb_model.predict(features)
            else:
                raise ValueError("No hybrid model available")
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def predict_with_confidence(self, features: np.ndarray):
        """Generate predictions with confidence/disagreement metrics."""
        assert self.stat_model is not None and self.rf_model is not None and self.xgb_model is not None
        pred_stat = self.stat_model.predict(features)
        pred_rf = self.rf_model.predict(features)
        pred_xgb = self.xgb_model.predict(features)
        
        # Best prediction (GB if available, otherwise hybrid/xgb)
        if self.gb_model is not None:
            pred_best = self.gb_model.predict(features)
            best_method = "gradient_boosting"
        elif self.meta_learner is not None:
            meta_features = np.column_stack([pred_stat, pred_rf, pred_xgb])
            pred_best = self.meta_learner.predict(meta_features)
            best_method = "hybrid"
        else:
            pred_best = pred_xgb
            best_method = "xgboost"
        
        # Compute disagreement (uncertainty indicator)
        predictions = np.array([pred_stat[0], pred_rf[0], pred_xgb[0]])
        disagreement = float(np.std(predictions))
        
        result = {
            "best": float(pred_best[0]),
            "best_method": best_method,
            "stat": float(pred_stat[0]),
            "rf": float(pred_rf[0]),
            "xgb": float(pred_xgb[0]),
            "disagreement_usd": disagreement * 1000,  # Convert to dollars
            "confidence": "high" if disagreement < 5 else "medium" if disagreement < 15 else "low"
        }
        
        # Add GB if available
        if self.gb_model is not None:
            result["gb"] = float(self.gb_model.predict(features)[0])
        
        return result


class BayesianModelLoader:
    """Load and manage US Bayesian MCMC model."""
    
    def __init__(self):
        self.trace = None
        self.scaler = None
        self.metadata = None
        self.feature_names = []
        self.state_mapping = {}
        
    def load_models(self):
        try:
            import arviz as az
            
            bayesian_path = Path("models/us/bayesian")
            if not bayesian_path.exists():
                logger.warning("Bayesian models not found")
                return False
            
            # Load trace
            self.trace = az.from_netcdf(bayesian_path / "bayesian_mcmc_trace.nc")
            
            # Load scaler
            self.scaler = joblib.load(bayesian_path / "bayesian_scaler.pkl")
            
            # Load metadata
            with open(bayesian_path / "bayesian_metadata.json", 'r') as f:
                self.metadata = json.load(f)
            
            self.feature_names = self.metadata['feature_names']
            self.state_mapping = self.metadata['state_mapping']
            
            logger.info(f"✓ Bayesian MCMC model loaded (R²={self.metadata['metrics']['r2']:.4f})")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to load Bayesian models: {e}")
            return False
    
    def predict_with_uncertainty(self, features: np.ndarray, state_code: str):
        """Generate predictions with credible intervals (CALIBRATED)."""
        if self.trace is None or self.scaler is None:
            raise RuntimeError("Bayesian model not loaded")
        
        # Get state index
        state_idx = self.state_mapping.get(state_code.upper(), 0)
        
        # Scale features
        X_scaled = self.scaler.transform(features)
        
        # Extract posterior samples
        beta_samples = self.trace.posterior['beta'].values.reshape(-1, len(self.feature_names))
        alpha_samples = self.trace.posterior['alpha'].values.flatten()
        state_effects_samples = self.trace.posterior['state_effects'].values.reshape(-1, len(self.state_mapping))
        
        # Generate predictions for each posterior sample
        n_samples = len(alpha_samples)
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            pred = alpha_samples[i] + X_scaled[0] @ beta_samples[i] + state_effects_samples[i, state_idx]
            predictions[i] = pred
        
        # Compute RAW statistics
        y_pred_mean = float(predictions.mean())
        y_pred_lower_raw = float(np.percentile(predictions, 2.5))
        y_pred_upper_raw = float(np.percentile(predictions, 97.5))
        
        # Apply calibration factor to achieve target 95% coverage
        calibration_factor = self.metadata['metrics'].get('calibration_factor', 1.0)
        interval_width_raw = y_pred_upper_raw - y_pred_lower_raw
        interval_width_calibrated = interval_width_raw * calibration_factor
        
        # Calibrated intervals (symmetric around mean)
        y_pred_lower = y_pred_mean - (interval_width_calibrated / 2)
        y_pred_upper = y_pred_mean + (interval_width_calibrated / 2)
        
        return {
            "prediction": y_pred_mean,
            "lower_95": y_pred_lower,
            "upper_95": y_pred_upper,
            "ci_width": interval_width_calibrated,
            "method": "bayesian_mcmc",
            "calibrated": True,
            "calibration_factor": calibration_factor
        }


class IndiaModelLoader:
    """Load and manage India models."""
    
    def __init__(self):
        self.regression_model = None
        self.classification_model = None
        self.scaler = None
        self.feature_names = []
        self.metadata = None
        
    def load_models(self):
        try:
            self.regression_model = joblib.load("models/india/regression_model.pkl")
            self.classification_model = joblib.load("models/india/classification_model.pkl")
            self.scaler = joblib.load("models/india/scaler.pkl")
            
            # Load feature names from .txt file (not .pkl)
            with open("models/india/feature_names.txt", 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            
            with open("models/india/metadata.json", 'r') as f:
                self.metadata = json.load(f)
            
            logger.info("✓ India models loaded")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to load India models: {e}")
            return False
    
    def predict(self, features: np.ndarray):
        """Generate proxy index prediction."""
        assert self.scaler is not None and self.regression_model is not None
        scaled_features = self.scaler.transform(features)
        income_index = self.regression_model.predict(scaled_features)[0]
        
        # Classify into category
        if income_index < 40:
            category = "Low"
        elif income_index < 70:
            category = "Middle"
        else:
            category = "Upper-Middle"
        
        return {
            "income_proxy_index": float(income_index),
            "category": category
        }


class ForecastModelLoader:
    """Load and manage US year-to-year forecasting models (single-year and multi-year)."""
    
    def __init__(self):
        # Single-year models (legacy)
        self.stat_model = None
        self.ml_model = None
        self.hybrid_model = None
        self.stat_scaler = None
        self.metadata_stat = None
        self.metadata_ml = None
        self.metadata_hybrid = None
        
        # Multi-year models (1yr, 2yr, 3yr)
        self.multiyear_models = {}  # {1: model_1yr, 2: model_2yr, 3: model_3yr}
        self.multiyear_metadata = {}  # {1: metadata_1yr, ...}
        
        self.feature_columns = [
            'n_returns_t', 'total_wages_t', 'n_wages_t', 'total_interest_t',
            'n_interest_t', 'total_cap_gains_t', 'n_cap_gains_t', 
            'total_business_t', 'n_business_t', 'avg_agi_t',
            'pct_with_wages_t', 'avg_wage_per_earner_t', 
            'pct_with_business_t', 'pct_with_cap_gains_t', 'log_returns_t'
        ]
        
    def load_models(self):
        """Load all forecasting models (single-year and multi-year)."""
        try:
            # Load single-year models (legacy)
            self._load_single_year_models()
            
            # Load multi-year models
            self._load_multiyear_models()
            
            return True
        except Exception as e:
            logger.error(f"✗ Failed to load forecast models: {e}")
            return False
    
    def _load_single_year_models(self):
        """Load legacy single-year models."""
        models_dir = Path("models/us_forecast")
        
        # Load statistical model
        stat_path = models_dir / "stat_model.pkl"
        scaler_path = models_dir / "stat_model_scaler.pkl"
        stat_meta_path = models_dir / "stat_model_metadata.json"
        
        if stat_path.exists() and scaler_path.exists():
            self.stat_model = joblib.load(stat_path)
            self.stat_scaler = joblib.load(scaler_path)
            with open(stat_meta_path, 'r') as f:
                self.metadata_stat = json.load(f)
            logger.info(f"✓ Stat forecast model loaded (R²={self.metadata_stat['metrics']['r2']:.4f})")
        
        # Load ML model
        ml_path = models_dir / "ml_model.pkl"
        ml_meta_path = models_dir / "ml_model_metadata.json"
        
        if ml_path.exists():
            self.ml_model = joblib.load(ml_path)
            with open(ml_meta_path, 'r') as f:
                self.metadata_ml = json.load(f)
            logger.info(f"✓ ML forecast model loaded (R²={self.metadata_ml['metrics']['r2']:.4f})")
        
        # Load hybrid model
        hybrid_path = models_dir / "hybrid_model.pkl"
        hybrid_meta_path = models_dir / "hybrid_model_metadata.json"
        
        if hybrid_path.exists():
            self.hybrid_model = joblib.load(hybrid_path)
            with open(hybrid_meta_path, 'r') as f:
                self.metadata_hybrid = json.load(f)
            logger.info(f"✓ Hybrid forecast model loaded (R²={self.metadata_hybrid['metrics']['r2']:.4f})")
    
    def _load_multiyear_models(self):
        """Load multi-year horizon models (1yr, 2yr, 3yr)."""
        models_dir = Path("models/us_forecast_multiyear")
        
        for year_gap in [1, 2, 3]:
            model_path = models_dir / f"rf_model_{year_gap}yr.pkl"
            metadata_path = models_dir / f"rf_model_{year_gap}yr.metadata.json"
            
            if model_path.exists() and metadata_path.exists():
                self.multiyear_models[year_gap] = joblib.load(model_path)
                with open(metadata_path, 'r') as f:
                    self.multiyear_metadata[year_gap] = json.load(f)
                
                meta = self.multiyear_metadata[year_gap]
                test_r2 = meta['performance'].get('test', {}).get('r2', 'N/A')
                if test_r2 != 'N/A':
                    logger.info(f"✓ Multi-year {year_gap}yr model loaded (Test R²={test_r2:.4f})")
                else:
                    logger.info(f"✓ Multi-year {year_gap}yr model loaded (no test set)")
    
    def predict_future_income(self, features: np.ndarray, method: str = "ml", year_gap: int = 1):
        """
        Predict year t+n income from year t features.
        
        Args:
            features: Features from base year
            method: 'stat', 'ml', 'hybrid', or 'auto'
            year_gap: Years ahead to forecast (1, 2, or 3)
        
        Returns:
            dict with prediction and metadata
        """
        # If year_gap > 1, use multi-year models (always ML/RandomForest)
        if year_gap > 1:
            if year_gap not in self.multiyear_models:
                raise RuntimeError(f"No model available for {year_gap}-year forecast. Max: 3 years")
            
            model = self.multiyear_models[year_gap]
            metadata = self.multiyear_metadata[year_gap]
            
            prediction = model.predict(features)[0]
            
            test_perf = metadata['performance'].get('test', metadata['performance']['train'])
            
            return {
                "predicted_income_thousands": float(prediction),
                "predicted_income_usd": float(prediction * 1000),
                "method": f"ml_{year_gap}yr",
                "year_gap": year_gap,
                "algorithm": "RandomForest (Multi-Year)",
                "model_r2": test_perf['r2'],
                "model_rmse": test_perf['rmse'],
                "note": f"Trained specifically for {year_gap}-year forecasting horizon"
            }
        
        # Single-year prediction (use legacy models)
        if method == "stat":
            if self.stat_model is None:
                raise RuntimeError("Statistical forecast model not loaded")
            features_scaled = self.stat_scaler.transform(features)
            prediction = self.stat_model.predict(features_scaled)[0]
            metadata = self.metadata_stat
        elif method == "ml":
            if self.ml_model is None:
                raise RuntimeError("ML forecast model not loaded")
            prediction = self.ml_model.predict(features)[0]
            metadata = self.metadata_ml
        elif method == "hybrid":
            if self.hybrid_model is None:
                raise RuntimeError("Hybrid forecast model not loaded")
            # Hybrid needs base model predictions
            if self.stat_model is None or self.ml_model is None:
                raise RuntimeError("Base models required for hybrid")
            
            stat_pred = self.stat_model.predict(self.stat_scaler.transform(features))[0]
            ml_pred = self.ml_model.predict(features)[0]
            features_hybrid = np.column_stack([features, stat_pred, ml_pred])
            prediction = self.hybrid_model.predict(features_hybrid)[0]
            metadata = self.metadata_hybrid
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            "predicted_income_thousands": float(prediction),
            "predicted_income_usd": float(prediction * 1000),
            "method": method,
            "year_gap": 1,
            "algorithm": metadata['algorithm'],
            "model_r2": metadata['metrics']['r2'],
            "model_rmse": metadata['metrics']['rmse']
        }


# Initialize loaders
us_loader = USModelLoader()
bayesian_loader = BayesianModelLoader()
india_loader = IndiaModelLoader()
forecast_loader = ForecastModelLoader()


# ==================== ENDPOINTS ====================

@app.on_event("startup")
async def load_all_models():
    """Load models on startup."""
    us_loaded = us_loader.load_models()
    bayesian_loaded = bayesian_loader.load_models()
    india_loaded = india_loader.load_models()
    forecast_loaded = forecast_loader.load_models()
    
    if not us_loaded or not india_loaded:
        logger.warning("⚠ Some models failed to load")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "us_models_loaded": us_loader.stat_model is not None,
        "india_models_loaded": india_loader.regression_model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics/us")
async def get_us_metrics():
    """
    Get comprehensive US model metrics.
    Returns training results (multi-year if available, otherwise single-year).
    """
    # Prefer multiyear results if available
    if us_loader.training_results is not None:
        results = us_loader.training_results
        
        formatted_results = {}
        for model_name, metrics in results.items():
            formatted_results[model_name] = {
                'r2_test': round(metrics['test_r2'], 4),
                'r2_train': round(metrics.get('train_r2', 0), 4),
                'rmse_usd': round(metrics['rmse'] * 1000, 2),
                'mae_usd': round(metrics['mae'] * 1000, 2),
                'rmse_thousands': round(metrics['rmse'], 3),
                'mae_thousands': round(metrics['mae'], 3)
            }
            
            # Add CV results if available
            if 'cv_r2_mean' in metrics:
                formatted_results[model_name]['cv_r2_mean'] = round(metrics['cv_r2_mean'], 4)
                formatted_results[model_name]['cv_r2_std'] = round(metrics['cv_r2_std'], 4)
        
        return {
            "model_version": "multi-year (2018-2022)",
            "total_samples": "137,987 ZIP-year observations",
            "dataset_info": {
                "years": "2018-2022",
                "unique_zips": "27,760",
                "features": us_loader.feature_names
            },
            "models": formatted_results,
            "best_model": max(formatted_results.items(), key=lambda x: x[1]['r2_test'])[0],
            "note": "Multi-year training with valid non-leaking features only"
        }
    
    # Fallback to single-year results
    elif us_loader.ablation_results is not None:
        results = us_loader.ablation_results
        
        formatted_results = {}
        for model_name, metrics in results['results'].items():
            formatted_results[model_name] = {
                'r2': round(metrics['r2'], 4),
                'rmse_usd': round(metrics['rmse'] * 1000, 2),
                'mae_usd': round(metrics['mae'] * 1000, 2),
                'rmse_thousands': round(metrics['rmse'], 3),
                'mae_thousands': round(metrics['mae'], 3)
            }
            
            if 'meta_weights' in metrics:
                formatted_results[model_name]['meta_weights'] = {
                    'stat': round(metrics['meta_weights'][0], 3),
                    'rf': round(metrics['meta_weights'][1], 3),
                    'xgb': round(metrics['meta_weights'][2], 3)
                }
        
        return {
            "model_version": "single-year (2022)",
            "timestamp": results['timestamp'],
            "dataset_info": {
                "train_samples": results['train_samples'],
                "val_samples": results['val_samples'],
                "test_samples": results['test_samples'],
                "total_samples": results['train_samples'] + results['val_samples'] + results['test_samples'],
                "features": results['features']
            },
            "models": formatted_results,
            "note": "Single-year model - consider using multi-year for better accuracy"
        }
    else:
        raise HTTPException(status_code=503, detail="US models not loaded")


@app.get("/metrics/india")
async def get_india_metrics():
    """Get India model metrics."""
    if india_loader.metadata is None:
        raise HTTPException(status_code=503, detail="India models not loaded")
    
    return {
        "timestamp": india_loader.metadata.get('timestamp'),
        "dataset_info": {
            "train_samples": india_loader.metadata.get('train_samples'),
            "test_samples": india_loader.metadata.get('test_samples'),
            "features": india_loader.feature_names
        },
        "regression_metrics": {
            "train_r2": round(india_loader.metadata.get('train_r2', 0), 4),
            "test_r2": round(india_loader.metadata.get('test_r2', 0), 4),
            "test_rmse": round(india_loader.metadata.get('test_rmse', 0), 4),
            "test_mae": round(india_loader.metadata.get('test_mae', 0), 4)
        },
        "classification_metrics": {
            "train_accuracy": round(india_loader.metadata.get('train_accuracy', 0), 4),
            "test_accuracy": round(india_loader.metadata.get('test_accuracy', 0), 4)
        },
        "warning": "Proxy index, not absolute income. Perfect classification has documented caveat."
    }


@app.get("/api/reports")
async def list_reports():
    """List all generated plot files."""
    reports = {"us": [], "india": []}
    
    us_dir = Path("reports/us")
    if us_dir.exists():
        reports["us"] = [
            {
                "name": f.name,
                "path": f"/reports/us/{f.name}",
                "size_kb": round(f.stat().st_size / 1024, 2)
            }
            for f in us_dir.glob("*.png")
        ]
    
    india_dir = Path("reports/india")
    if india_dir.exists():
        reports["india"] = [
            {
                "name": f.name,
                "path": f"/reports/india/{f.name}",
                "size_kb": round(f.stat().st_size / 1024, 2)
            }
            for f in india_dir.glob("*.png")
        ]
    
    return {
        "total_plots": len(reports["us"]) + len(reports["india"]),
        "us_plots": reports["us"],
        "india_plots": reports["india"]
    }


@app.get("/models/info")
async def model_info():
    """Get detailed model information."""
    return {
        "us_models": {
            "type": "Stacked Hybrid (Meta-Learner)",
            "base_models": ["Ridge (scaled)", "Random Forest", "XGBoost"],
            "meta_learner": "Ridge",
            "features": us_loader.feature_names,
            "loaded": us_loader.stat_model is not None
        },
        "india_models": {
            "type": "Random Forest Regression + Classification",
            "target": "Income Proxy Index (0-100)",
            "categories": ["Low (0-40)", "Middle (40-70)", "Upper-Middle (70-100)"],
            "features": india_loader.feature_names,
            "loaded": india_loader.regression_model is not None
        }
    }


# Sample prediction request models
class USPredictionRequest(BaseModel):
    zipcode: str = Field(None, description="ZIP code for Bayesian predictions")
    year: int = Field(None, description="Year for prediction")
    avg_wage: float = Field(None, description="Average wage per earner (thousands USD)")
    log_total_wages: float = None
    log_wage_returns: float = None
    num_wage_returns: float = None
    wage_per_return: float = None
    total_wages: float = None
    state_avg_income: float = None
    method: str = Field("hybrid", description="Prediction method: stat/rf/xgb/gb/hybrid/bayesian_mcmc")


class IndiaPredictionRequest(BaseModel):
    pincode: str = Field(..., min_length=6, max_length=6, description="6-digit PIN code")
    year: int = Field(2011, ge=2011, le=2023, description="Year for prediction (2011-2023)")
    method: str = Field("ml", description="Prediction method: ml, statistical, or hybrid")


class USForecastRequest(BaseModel):
    zipcode: str = Field(..., description="5-digit ZIP code")
    base_year: int = Field(..., ge=2019, le=2022, description="Base year with known data (2019-2022)")
    target_year: int = Field(..., ge=2020, le=2025, description="Target year to forecast (2020-2025)")
    method: str = Field("ml", description="Forecast method: stat, ml, or hybrid")


@app.post("/predict/us")
async def predict_us_income(request: USPredictionRequest):
    """
    Predict US ZIP-code income with confidence metrics.
    Supports both traditional ML and Bayesian MCMC methods.
    """
    try:
        # Bayesian method requires ZIP code and year
        if request.method == "bayesian_mcmc":
            if not request.zipcode or not request.year:
                raise HTTPException(
                    status_code=400,
                    detail="Bayesian method requires 'zipcode' and 'year' parameters"
                )
            
            # Load dataset and engineer features (use Parquet if available for 10-100x speedup)
            parquet_path = Path("datasets/parquet/22zpallagi.parquet")
            csv_path = Path("datasets/22zpallagi.csv")
            
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path, engine='pyarrow')
            elif csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Dataset not found. Please ensure datasets/22zpallagi.csv exists."
                )
            
            df.columns = df.columns.str.lower()
            
            # Find ZIP data
            zip_data = df[(df['zipcode'] == int(request.zipcode))].copy()
            if zip_data.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"ZIP code {request.zipcode} not found in dataset"
                )
            
            # Get state
            state_code = zip_data.iloc[0]['state']
            
            # Engineer features (simplified version)
            total_returns = zip_data['n1'].sum()
            if total_returns == 0:
                raise HTTPException(status_code=400, detail="No tax returns data for this ZIP")
            
            features = np.array([[
                zip_data['n1'].sum() / total_returns * 100 if 'n1' in zip_data.columns else 0,  # pct_low_income
                zip_data['n1'].sum() / total_returns * 100 if 'n1' in zip_data.columns else 0,  # pct_mid_income  
                zip_data['n1'].sum() / total_returns * 100 if 'n1' in zip_data.columns else 0,  # pct_high_income
                zip_data['n00200'].sum() / total_returns * 100 if 'n00200' in zip_data.columns else 0,  # pct_with_wages
                zip_data['a00200'].sum() / zip_data['n00200'].sum() if 'a00200' in zip_data.columns and zip_data['n00200'].sum() > 0 else 0,  # avg_wage_per_earner
                zip_data['n00900'].sum() / total_returns * 100 if 'n00900' in zip_data.columns else 0,  # pct_with_business
                zip_data['n01000'].sum() / total_returns * 100 if 'n01000' in zip_data.columns else 0,  # pct_with_cap_gains
                zip_data['n01700'].sum() / total_returns * 100 if 'n01700' in zip_data.columns else 0,  # pct_with_pension
                np.log(total_returns + 1)  # log_returns
            ]])
            
            result = bayesian_loader.predict_with_uncertainty(features, state_code)
            
            return {
                "predicted_income": f"${result['prediction']*1000:.0f}",
                "predicted_income_usd": round(result['prediction'] * 1000, 2),
                "credible_interval_95": {
                    "lower": f"${result['lower_95']*1000:.0f}",
                    "upper": f"${result['upper_95']*1000:.0f}",
                    "lower_usd": round(result['lower_95'] * 1000, 2),
                    "upper_usd": round(result['upper_95'] * 1000, 2),
                    "width": round(result['ci_width'] * 1000, 2)
                },
                "method": "bayesian_mcmc",
                "state": state_code,
                "zipcode": request.zipcode,
                "year": request.year,
                "timestamp": datetime.now().isoformat()
            }
        
        # Traditional ML methods
        else:
            if request.avg_wage is None:
                raise HTTPException(
                    status_code=400,
                    detail="Traditional methods require feature values (avg_wage, etc.)"
                )
            
            features = np.array([[
                request.avg_wage,
                request.log_total_wages,
                request.log_wage_returns,
                request.num_wage_returns,
                request.wage_per_return,
                request.total_wages,
                request.state_avg_income
            ]])
            
            predictions = us_loader.predict_with_confidence(features)
            
            # Format currency properly
            def format_currency(value_thousands):
                value_dollars = value_thousands * 1000
                if value_dollars < 1000:
                    return f"${value_dollars:.0f}"
                else:
                    return f"${value_dollars/1000:.1f}k"
            
            return {
                "predicted_income": format_currency(predictions["best"]),
                "predicted_income_usd": round(predictions["best"] * 1000, 2),
                "model_predictions": {
                    "statistical": format_currency(predictions["stat"]),
                    "random_forest": format_currency(predictions["rf"]),
                    "xgboost": format_currency(predictions["xgb"]),
                    "best": format_currency(predictions["best"])
                },
                "confidence": predictions["confidence"],
                "disagreement_usd": round(predictions["disagreement_usd"], 2),
                "method": request.method,
                "timestamp": datetime.now().isoformat()
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/india")
async def predict_india_income(request: IndiaPredictionRequest):
    """Predict India district income proxy index from PIN code + year."""
    try:
        # Load PIN to district mapping
        pincode_map_path = Path("datasets/india_pincode_to_district.csv")
        if not pincode_map_path.exists():
            raise HTTPException(
                status_code=500,
                detail="PIN code mapping file not found. Please ensure datasets/india_pincode_to_district.csv exists."
            )
        
        pincode_df = pd.read_csv(pincode_map_path)
        
        # Find district for given PIN code
        pin_row = pincode_df[pincode_df['pincode'] == int(request.pincode)]
        
        if pin_row.empty:
            raise HTTPException(
                status_code=404,
                detail=f"PIN code {request.pincode} not found in database. Currently supporting {len(pincode_df)} major urban PIN codes. See datasets/india_pincode_to_district.csv for coverage."
            )
        
        district = pin_row.iloc[0]['district']
        state = pin_row.iloc[0]['state']
        region = pin_row.iloc[0]['region']
        urban_rural = pin_row.iloc[0].get('urban_rural', 'Urban')
        
        # Load district census data
        census_path = Path("datasets/india_district_census_data.csv")
        if not census_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Census data not found"
            )
        
        census_df = pd.read_csv(census_path)
        
        # Find district features
        district_row = census_df[
            (census_df['district'].str.lower() == district.lower()) &
            (census_df['state'].str.lower() == state.lower())
        ]
        
        if district_row.empty:
            raise HTTPException(
                status_code=404,
                detail=f"District '{district}, {state}' not found in census database"
            )
        
        # Extract features for model
        features = np.array([[
            district_row.iloc[0]['literacy_rate'],
            district_row.iloc[0]['worker_participation'],
            district_row.iloc[0]['urban_ratio'],
            district_row.iloc[0]['avg_household_size'],
            district_row.iloc[0]['asset_score'],
            district_row.iloc[0]['electricity_access'],
            district_row.iloc[0]['water_access'],
            district_row.iloc[0]['sanitation_access']
        ]])
        
        # Run base prediction (2011 baseline)
        prediction = india_loader.predict(features)
        base_index = prediction["income_proxy_index"]
        
        # Apply year adjustment (simple GDP-based scaling)
        # Base year: 2011 (census year)
        # Adjustment factor: ~5% per year (India's average GDP growth)
        BASE_YEAR = 2011
        ANNUAL_GROWTH_RATE = 0.048  # 4.8% average growth
        
        years_elapsed = request.year - BASE_YEAR
        growth_factor = (1 + ANNUAL_GROWTH_RATE) ** years_elapsed
        
        # Adjusted index (capped at 100)
        adjusted_index = min(base_index * growth_factor, 100.0)
        
        # Determine confidence based on year distance from base
        if abs(years_elapsed) <= 2:
            confidence = "High"
        elif abs(years_elapsed) <= 5:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Categorize adjusted index
        if adjusted_index < 40:
            category = "Low"
        elif adjusted_index < 70:
            category = "Middle"
        else:
            category = "Upper-Middle"
        
        return {
            "pincode": request.pincode,
            "district": district,
            "state": state,
            "region": region,
            "urban_rural": urban_rural,
            "year": request.year,
            "relative_income_index": round(adjusted_index, 1),
            "income_category": category,
            "confidence": confidence,
            "base_index_2011": round(base_index, 1),
            "growth_factor_applied": round(growth_factor, 3),
            "method": request.method,
            "note": f"Proxy-based estimate for {district}, {state}. Base year 2011 adjusted to {request.year} using {ANNUAL_GROWTH_RATE*100:.1f}% annual growth rate.",
            "disclaimer": "This is a relative socioeconomic index (0-100 scale), NOT absolute income in ₹. Suitable for regional comparison and crisis vulnerability assessment.",
            "coverage_note": f"PIN code database covers {len(pincode_df)} major urban areas. Rural PIN codes may not be available.",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"India prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/available-zips/us/{year}")
async def get_available_zips_us(year: int):
    """
    Get list of available ZIP codes for a given year.
    """
    try:
        # Load panel dataset
        panel_path = Path("datasets/us_panel_dataset.csv")
        if not panel_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Panel dataset not found"
            )
        
        df_panel = pd.read_csv(panel_path)
        
        # Get ZIPs for the requested year
        year_data = df_panel[df_panel['year'] == year]
        
        if len(year_data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for year {year}"
            )
        
        zip_codes = sorted([str(z).zfill(5) for z in year_data['zip'].unique().tolist()])
        
        return {
            "year": year,
            "total_zips": len(zip_codes),
            "zip_codes": zip_codes
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching ZIP codes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast/us")
async def forecast_us_income(request: USForecastRequest):
    """
    Forecast US ZIP code income for future years using multi-year models.
    Automatically selects appropriate model based on year gap (1yr, 2yr, or 3yr).
    """
    try:
        # Calculate year gap
        year_gap = request.target_year - request.base_year
        
        # Validate year gap
        if year_gap < 1:
            raise HTTPException(
                status_code=400,
                detail=f"target_year must be greater than base_year. Got {request.base_year}→{request.target_year}"
            )
        
        if year_gap > 3:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum forecast horizon is 3 years. Got {year_gap} years ({request.base_year}→{request.target_year})"
            )
        
        # Load panel dataset
        panel_path = Path("datasets/us_panel_dataset.csv")
        if not panel_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Panel dataset not found. Please run: python src/data/build_us_panel_dataset.py"
            )
        
        df_panel = pd.read_csv(panel_path)
        
        # Find ZIP data for base_year
        zip_data = df_panel[
            (df_panel['zip'] == int(request.zipcode)) & 
            (df_panel['year'] == request.base_year)
        ]
        
        if zip_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"ZIP {request.zipcode} data not available for year {request.base_year}"
            )
        
        # Extract features (from base year)
        feature_values = []
        missing_features = []
        
        for col in forecast_loader.feature_columns:
            col_name = col.replace('_t', '')  # Remove _t suffix to match panel columns
            if col_name in zip_data.columns:
                feature_values.append(zip_data.iloc[0][col_name])
            else:
                missing_features.append(col_name)
                feature_values.append(0.0)  # Default value
        
        if missing_features:
            logger.warning(f"Missing features for {request.zipcode}: {missing_features}")
        
        features = np.array([feature_values])
        
        # Get prediction using appropriate model
        result = forecast_loader.predict_future_income(features, request.method, year_gap)
        
        # Get actual value if available (for validation)
        actual_data = df_panel[
            (df_panel['zip'] == int(request.zipcode)) & 
            (df_panel['year'] == request.target_year)
        ]
        
        actual_income = None
        if not actual_data.empty and 'avg_agi' in actual_data.columns:
            actual_income = float(actual_data.iloc[0]['avg_agi']) * 1000
        
        return {
            "zipcode": request.zipcode,
            "base_year": request.base_year,
            "target_year": request.target_year,
            "year_gap": year_gap,
            "predicted_income": round(result['predicted_income_usd'], 2),
            "actual_income": round(actual_income, 2) if actual_income else None,
            "method": result['method'],
            "features_used": len(feature_values),
            "model_performance": {
                "r2_score": result['model_r2'],
                "rmse": result['model_rmse']
            },
            "note": result.get('note', f"Forecasting {year_gap}-year horizon"),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
