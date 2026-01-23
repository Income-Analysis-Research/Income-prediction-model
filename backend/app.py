"""India-US Income Model API Backend"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Income Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class USModelLoader:
    """Load and manage US income prediction model"""
    
    def __init__(self):
        self.model: Optional[object] = None
        self.scaler: Optional[object] = None
        self.feature_names: list[str] = []
        
    def load_models(self):
        try:
            self.model = joblib.load("models/us/best_model.pkl")
            self.scaler = joblib.load("models/us/scaler.pkl")
            with open("models/us/feature_names.txt", 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            logger.info("US models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load US models: {e}")
            return False
            
    def predict(self, features: dict) -> float:
        if self.model is None:
            raise RuntimeError("Models not loaded")
        
        feature_vector = np.array([[
            features.get('num_returns', 0),
            features.get('agi_amount', 0),
            features.get('num_wage_returns', 0),
            features.get('total_wages', 0),
            features.get('avg_wage', 0),
            features.get('state_encoded', 0),
            features.get('population_density', 0),
            features.get('wage_return_ratio', 0),
            features.get('per_capita_agi', 0)
        ]])
        
        scaled_features = self.scaler.transform(feature_vector)
        prediction = self.model.predict(scaled_features)
        return float(prediction[0])


class IndiaModelLoader:
    """Load and manage India income prediction models"""
    
    def __init__(self):
        self.regression_model: Optional[object] = None
        self.classification_model: Optional[object] = None
        self.scaler: Optional[object] = None
        self.feature_names: list[str] = []
        
    def load_models(self):
        try:
            self.regression_model = joblib.load("models/india/regression_model.pkl")
            self.classification_model = joblib.load("models/india/classification_model.pkl")
            self.scaler = joblib.load("models/india/scaler.pkl")
            with open("models/india/feature_names.txt", 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            logger.info("India models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load India models: {e}")
            return False
            
    def predict(self, features: dict) -> dict:
        if self.regression_model is None:
            raise RuntimeError("Models not loaded")
        
        feature_vector = np.array([[
            features.get('literacy_rate', 0),
            features.get('worker_participation', 0),
            features.get('urban_ratio', 0),
            features.get('avg_household_size', 0),
            features.get('asset_score', 0),
            features.get('electricity_access', 0),
            features.get('water_access', 0),
            features.get('sanitation_access', 0),
            features.get('state_encoded', 0)
        ]])
        
        scaled_features = self.scaler.transform(feature_vector)
        
        income_index = self.regression_model.predict(scaled_features)[0]
        category_id = self.classification_model.predict(scaled_features)[0]
        
        categories = ["Low", "Lower-Middle", "Upper-Middle", "High"]
        category = categories[int(category_id)] if 0 <= category_id < len(categories) else "Unknown"
        
        return {
            "income_index": float(income_index),
            "category": category,
            "category_id": int(category_id)
        }


us_loader = USModelLoader()
india_loader = IndiaModelLoader()


class USPredictionRequest(BaseModel):
    zipcode: str = Field(..., description="5-digit US ZIP code")
    num_returns: float = Field(..., gt=0)
    agi_amount: float = Field(..., gt=0)
    num_wage_returns: float = Field(..., gt=0)
    total_wages: float = Field(..., gt=0)
    avg_wage: float = Field(..., gt=0)
    state_encoded: float
    population_density: float
    wage_return_ratio: float
    per_capita_agi: float


class IndiaPredictionRequest(BaseModel):
    district: str = Field(..., description="District name")
    state: str = Field(..., description="State name")
    literacy_rate: float = Field(..., ge=0, le=100)
    worker_participation: float = Field(..., ge=0, le=100)
    urban_ratio: float = Field(..., ge=0, le=100)
    avg_household_size: float = Field(..., gt=0)
    asset_score: float = Field(..., ge=0)
    electricity_access: float = Field(..., ge=0, le=100)
    water_access: float = Field(..., ge=0, le=100)
    sanitation_access: float = Field(..., ge=0, le=100)
    state_encoded: float


@app.on_event("startup")
async def load_all_models():
    us_loaded = us_loader.load_models()
    india_loaded = india_loader.load_models()
    
    if not us_loaded or not india_loaded:
        logger.warning("Some models failed to load")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "us_model_loaded": us_loader.model is not None,
        "india_model_loaded": india_loader.regression_model is not None
    }


@app.post("/predict/us")
async def predict_us_income(request: USPredictionRequest):
    try:
        prediction = us_loader.predict(request.dict())
        return {
            "zipcode": request.zipcode,
            "predicted_avg_agi": round(prediction, 2),
            "model_type": "US Average AGI Prediction",
            "unit": "USD (thousands)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/india")
async def predict_india_income(request: IndiaPredictionRequest):
    try:
        prediction = india_loader.predict(request.dict())
        return {
            "district": request.district,
            "state": request.state,
            "income_index": round(prediction["income_index"], 2),
            "category": prediction["category"],
            "model_type": "India Income Proxy"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/info")
async def model_info():
    return {
        "us_model": {
            "loaded": us_loader.model is not None,
            "features": us_loader.feature_names if us_loader.feature_names else []
        },
        "india_model": {
            "loaded": india_loader.regression_model is not None,
            "features": india_loader.feature_names if india_loader.feature_names else []
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
