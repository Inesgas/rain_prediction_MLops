import datetime
from pathlib import Path
from typing import List, Optional
from catboost import Pool
from fastapi import FastAPI, HTTPException, Request, status, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
import logging
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge, REGISTRY

from fastapi.security import HTTPBasic, HTTPBasicCredentials

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instanziieren Sie HTTPBasic für die Swagger-UI-Anmeldung
security = HTTPBasic()

# ========== BENUTZER & ROLLEN (CRUCIAL FIX) ==========
USERS = {
    "andrey": {"role": "admin"},
    "ines": {"role": "admin"},
    "gunter": {"role": "admin"},
    "admin": {"role": "user"}
}

def get_current_user_from_nginx(request: Request, credentials: Optional[HTTPBasicCredentials] = Depends(security)):
    # 1. Versuch: Den Namen des authentifizierten Benutzers direkt aus dem Nginx-Header lesen
    username = request.headers.get("X-Forwarded-User")
    
    # 2. Versuch (Fallback): Wenn kein Header da ist (Direktaufruf der Swagger-UI im Browser)
    if not username and credentials:
        username = credentials.username
        
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Basic"} # Zwingt die Swagger-UI zum Anmeldedialog
        )
        
    user = USERS.get(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
        
    return {"username": username, "role": user["role"]}

def require_admin(current_user: dict = Depends(get_current_user_from_nginx)):
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

def require_user(current_user: dict = Depends(get_current_user_from_nginx)):
    return current_user


# ========== FASTAPI APP ==========
app = FastAPI(
    title="Rain Prediction API",
    version="2.0.0",
    description="Rain Prediction API with Nginx authentication",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ========== PROMETHEUS METRICS ==========
# Automatische Infrastruktur-Metriken
instrumentator = Instrumentator(
    excluded_handlers=["/metrics", "/health"],
    env_var_name="ENABLE_METRICS",
)
instrumentator.instrument(app).expose(app, endpoint="/metrics")

# ========== FACHLICHE METRIKEN (MLOps) ==========
# Vorhersagen pro Location und Ergebnis
predictions_total = Counter(
    'rain_predictions_total',
    'Total number of predictions',
    ['location', 'rain_tomorrow']
)

# Confidence Histogramm
prediction_confidence = Histogram(
    'rain_prediction_confidence',
    'Confidence of predictions',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
)

# Aktuelle Confidence als Gauge
latest_confidence = Gauge(
    'rain_latest_confidence',
    'Confidence of the latest prediction'
)

# Input Werte für Data Drift Überwachung
input_humidity = Gauge(
    'rain_input_humidity',
    'Incoming humidity_3pm values',
    ['location']
)

input_rainfall = Gauge(
    'rain_input_rainfall',
    'Incoming rainfall values',
    ['location']
)

input_pressure = Gauge(
    'rain_input_pressure',
    'Incoming pressure_3pm values',
    ['location']
)

# Modell-Ladezeit (Startup)
model_load_time = Gauge(
    'rain_model_load_time_seconds',
    'Time taken to load the model'
)

# Fehlerzähler für spezifische ML-Fehler
prediction_errors = Counter(
    'rain_prediction_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== MODEL LOADING ==========
MODEL_PATH = Path("models/final_winner/winner_model.joblib")
model = None
MODEL_FEATURE_ORDER = []
NUMERIC_FILL_VALUES = {}
CATEGORICAL_FEATURES = ["location"]

@app.on_event("startup")
def load_model():
    global model, MODEL_FEATURE_ORDER, NUMERIC_FILL_VALUES, CATEGORICAL_FEATURES
    import time
    start_time = time.time()
    
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH.resolve()}")
    try:
        loaded_artifact = joblib.load(MODEL_PATH)
        if isinstance(loaded_artifact, dict):
            model = loaded_artifact.get("model")
            MODEL_FEATURE_ORDER = loaded_artifact.get("features", [])
            NUMERIC_FILL_VALUES = loaded_artifact.get("numeric_fill_values", {})
            CATEGORICAL_FEATURES = loaded_artifact.get("categorical_features", ["location"])
        else:
            model = loaded_artifact
        if not MODEL_FEATURE_ORDER and hasattr(model, "feature_names_"):
            MODEL_FEATURE_ORDER = list(model.feature_names_)
        if not MODEL_FEATURE_ORDER:
            raise AttributeError("No feature names found")
        
        load_time = time.time() - start_time
        model_load_time.set(load_time)
        
        logger.info(f"✓ Model loaded from {MODEL_PATH} in {load_time:.2f}s")
        logger.info(f"✓ Features required: {len(MODEL_FEATURE_ORDER)}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        prediction_errors.labels(error_type="model_load").inc()
        raise RuntimeError(f"Failed to load model: {e}")

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Application shutting down")

# ========== DATA LOGGING SETUP ==========
# This folder is mounted via docker-compose volume: - ./data:/app/data
LOG_DIR = Path("data/monitoring/predictions")

def log_predictions_to_csv(records: List[dict]):
    """
    Appends a list of prediction records to the daily CSV file.
    Runs asynchronously via background tasks to protect API request latency.
    """
    try:
        if not records:
            return
            
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        file_path = LOG_DIR / f"predictions_{today_str}.csv"
        
        new_data = pd.DataFrame(records)
        file_exists = file_path.is_file()
        new_data.to_csv(file_path, mode='a', header=not file_exists, index=False)
        logger.info(f"Logged {len(records)} prediction(s) to {file_path.name}")
    except Exception as e:
        logger.error(f"Failed to log predictions to CSV: {e}")
        prediction_errors.labels(error_type="csv_logging_failed").inc()

# ========== PYDANTIC MODELS ==========
class InferencePayload(BaseModel):
    location: str = Field(..., description="Location name")
    humidity_3pm: float = Field(..., ge=0, le=100)
    rain_today: str = Field(..., description="'Yes' or 'No'")
    wind_gust_speed: float = Field(..., ge=0)
    rainfall: float = Field(..., ge=0)
    pressure_3pm: float = Field(..., ge=800, le=1100)
    humidity_9am: float = Field(..., ge=0, le=100)
    
    @field_validator('rain_today')
    @classmethod
    def validate_rain_today(cls, v: str) -> str:
        if v.lower() not in ['yes', 'no']:
            raise ValueError('rain_today must be "Yes" or "No"')
        return v

class BatchInferencePayload(BaseModel):
    samples: List[InferencePayload]

class PredictionResponse(BaseModel):
    prediction: int
    rain_tomorrow: str
    confidence: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

# ========== SUPPORTED LOCATIONS ==========
SUPPORTED_LOCATIONS = [
    "Adelaide", "Albany", "Albury", "AliceSprings", "BadgerysCreek",
    "Badgingarra", "Balladonia", "Ballarat", "Bendigo", "Bridgetown",
    "Brisbane", "Broome", "Bunbury", "Cairns", "Canberra", "Cobar",
    "CoffsHarbour", "Dartmoor", "Darwin", "Devonport", "Esperance",
    "Geraldton", "GoldCoast", "Hobart", "Kalgoorlie", "Launceston",
    "Meekatharra", "Melbourne", "MelbourneAirport", "Mildura", "Moree",
    "MountGambier", "MountGinini", "Newcastle", "Nhil", "NorahHead",
    "NorfolkIsland", "Nuriootpa", "PearceRAAF", "Perth", "PerthAirport",
    "Portmacquarie", "Sale", "SalmonGums", "SunshineCoast", "Sydney",
    "SydneyAirport", "Townsville", "Tuggeranong", "Uluru", "WaggaWagga",
    "Walpole", "Watsonia", "Woomera"
]

# ========== PUBLIC ENDPOINTS ==========
@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_required": len(MODEL_FEATURE_ORDER),
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/locations", tags=["Locations"])
def get_all_locations():
    return {
        "total": len(SUPPORTED_LOCATIONS),
        "locations": sorted(SUPPORTED_LOCATIONS)
    }

# ========== PROTECTED ENDPOINTS ==========
@app.post("/predict", 
          response_model=PredictionResponse,
          tags=["Prediction"])
def predict(payload: InferencePayload, background_tasks: BackgroundTasks, current_user: dict = Depends(require_user)):
    if model is None:
        prediction_errors.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    background_tasks.add_task(logger.info, f"User {current_user['username']} prediction request: {payload.location}")
    
    try:
        input_data = payload.model_dump()
        input_data["rain_today"] = 1.0 if input_data["rain_today"].strip().lower() in ["yes", "1", "1.0"] else 0.0
        
        for feature in MODEL_FEATURE_ORDER:
            if feature not in input_data:
                if feature in CATEGORICAL_FEATURES:
                    input_data[feature] = payload.location if feature == "location" else "Unknown"
                else:
                    input_data[feature] = NUMERIC_FILL_VALUES.get(feature) or 0.0
                    
        df_input = pd.DataFrame([input_data])
        df_input = df_input[MODEL_FEATURE_ORDER]
        eval_pool = Pool(df_input, cat_features=CATEGORICAL_FEATURES)
        
        prediction = model.predict(eval_pool)
        prediction_proba = model.predict_proba(eval_pool) if hasattr(model, "predict_proba") else None
        pred_value = int(prediction) if isinstance(prediction, (np.ndarray, list)) else int(prediction)
        confidence = float(np.max(prediction_proba)) if prediction_proba is not None else None
        
        # ========== PROMETHEUS METRICS ==========
        predictions_total.labels(
            location=payload.location,
            rain_tomorrow="Yes" if pred_value == 1 else "No"
        ).inc()
        
        if confidence is not None:
            prediction_confidence.observe(confidence)
            latest_confidence.set(confidence)
        
        # Input values for data drift monitoring
        input_humidity.labels(location=payload.location).set(payload.humidity_3pm)
        input_rainfall.labels(location=payload.location).set(payload.rainfall)
        input_pressure.labels(location=payload.location).set(payload.pressure_3pm)
        
        # ========== BACKGROUND CSV LOGGING ==========
        log_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "location": payload.location,
            "predicted_rain": pred_value,
            "confidence": confidence
        }
        background_tasks.add_task(log_predictions_to_csv, [log_record])
        
        return PredictionResponse(
            prediction=pred_value,
            rain_tomorrow="Yes" if pred_value == 1 else "No",
            confidence=confidence,
            timestamp=datetime.datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        prediction_errors.labels(error_type="prediction_failed").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(
    payload: BatchInferencePayload, 
    background_tasks: BackgroundTasks, 
    current_user: dict = Depends(require_user)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    results = []
    log_records_batch = []
    
    for sample in payload.samples:
        try:
            input_data = sample.model_dump()
            input_data["rain_today"] = 1.0 if input_data["rain_today"].strip().lower() in ["yes", "1", "1.0"] else 0.0
            
            for feature in MODEL_FEATURE_ORDER:
                if feature not in input_data:
                    if feature in CATEGORICAL_FEATURES:
                        input_data[feature] = sample.location if feature == "location" else "Unknown"
                    else:
                        input_data[feature] = NUMERIC_FILL_VALUES.get(feature) or 0.0
            
            df_input = pd.DataFrame([input_data])
            df_input = df_input[MODEL_FEATURE_ORDER]
            eval_pool = Pool(df_input, cat_features=CATEGORICAL_FEATURES)
            
            prediction = model.predict(eval_pool)
            prediction_proba = model.predict_proba(eval_pool) if hasattr(model, "predict_proba") else None
            pred_value = int(prediction) if isinstance(prediction, (np.ndarray, list)) else int(prediction)
            confidence = float(np.max(prediction_proba)) if prediction_proba is not None else None
            
            # Batch Metrics for Prometheus
            predictions_total.labels(
                location=sample.location,
                rain_tomorrow="Yes" if pred_value == 1 else "No"
            ).inc()
            
            results.append({
                "location": sample.location,
                "prediction": pred_value,
                "rain_tomorrow": "Yes" if pred_value == 1 else "No",
                "confidence": confidence
            })
            
            # Collect data for the daily batch log file
            log_records_batch.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "location": sample.location,
                "predicted_rain": pred_value,
                "confidence": confidence
            })
            
        except Exception as e:
            results.append({"location": sample.location, "error": str(e)})
            prediction_errors.labels(error_type="batch_sample_failed").inc()
            
    # Trigger background logging task for the entire batch list
    if log_records_batch:
        background_tasks.add_task(log_predictions_to_csv, log_records_batch)
    
    return {"total_samples": len(payload.samples), "results": results}

# ========== ADMIN-ONLY ENDPOINTS ==========
@app.get("/model/info", tags=["Metadata"])
def get_model_info(current_user: dict = Depends(require_admin)):
    return {
        "model_loaded": model is not None,
        "features_count": len(MODEL_FEATURE_ORDER),
        "features_preview": MODEL_FEATURE_ORDER[:20],
        "categorical_features": CATEGORICAL_FEATURES,
        "model_path": str(MODEL_PATH)
    }

@app.get("/model/features", tags=["Metadata"])
def get_features(current_user: dict = Depends(require_admin)):
    return {
        "total_features": len(MODEL_FEATURE_ORDER),
        "all_features": MODEL_FEATURE_ORDER,
        "categorical": CATEGORICAL_FEATURES,
        "numeric_fill_values": NUMERIC_FILL_VALUES
    }