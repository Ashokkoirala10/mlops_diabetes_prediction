from fastapi import FastAPI
from pydantic import Field, create_model
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import redis
import json
import hashlib
import mlflow
from mlflow.tracking import MlflowClient
from xgboost import XGBClassifier
import os
import tempfile
from pydantic import BaseModel, Field

# ------------------ CONFIG ------------------
BASE_DIR = "/home/ashok/mlops_project" # chnage this according to your dir

# MLflow config
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_NAME = "Diabetes_XGB_Model"

# Redis config
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_DB = 0
CACHE_TTL_SECONDS = 24 * 3600
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# Database config
DB_URL = "mysql+pymysql://ashok:Docker%4012345%21@localhost:3306/diabetes_ml" # chnage this with your db connection credential
engine = create_engine(DB_URL)

# Prediction table name
TABLE_NAME = "predictions_auto"

# ------------------ UTILS ------------------
def make_cache_key(payload: dict) -> str:
    s = json.dumps(payload, sort_keys=True)
    return "pred:" + hashlib.sha256(s.encode()).hexdigest()

def load_latest_model_and_features(model_name: str):
    """Load the latest Production model and its features from MLflow"""
    client = MlflowClient()

    # client.transition_model_version_stage(
    # name="Diabetes_XGB_Model",
    # version=2,
    # stage="Production",
    # archive_existing_versions=True)  # to select he model for production

    # Get latest Production version
    prod_versions = client.get_latest_versions(name=model_name, stages=["Production"])

    if not prod_versions:
        raise RuntimeError(f"No Production versions found for {model_name}")
    latest_version = prod_versions[0]
    run_id = latest_version.run_id

    # Load model
    model_uri = f"models:/{model_name}/Production"
    model: XGBClassifier = mlflow.sklearn.load_model(model_uri)

    # Load selected features artifact
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, 
            artifact_path="features/selected_features.json",  
        )

        with open(artifact_path) as f:
            features_info = json.load(f)
    feature_cols = features_info["features"]
    feature_types = features_info["types"]

    return model, feature_cols, feature_types

# ------------------ MODEL & FEATURES LOADING ------------------
try:
    print("Loading latest Production model and features from MLflow...")
    model, feature_cols, feature_types = load_latest_model_and_features(MODEL_NAME)
    print(f"Model loaded successfully with {len(feature_cols)} features.")
except Exception as e:
    raise RuntimeError(f"Failed to load model/features from MLflow: {e}")

# # Dynamically create Pydantic request model
# fields_dict = {}
# for col in feature_cols:
#     ftype = float if feature_types[col] == "float" else int
#     if ftype == int:
#         fields_dict[col] = (ftype, Field(..., ge=0, le=1))
#     else:
#         fields_dict[col] = (ftype, Field(...))

# PredictRequest = create_model("PredictRequest", **fields_dict)


class PredictRequest(BaseModel):
    HighBP: int = Field(..., ge=0, le=1, description="0 = no high BP, 1 = high BP")
    HighChol: int = Field(..., ge=0, le=1, description="0 = no high cholesterol, 1 = high cholesterol")
    CholCheck: int = Field(..., ge=0, le=1, description="0 = no cholesterol check in 5 years, 1 = yes")
    BMI: float = Field(..., ge=0, description="Body Mass Index")
    Smoker: int = Field(..., ge=0, le=1, description="0 = no, 1 = yes, smoked at least 100 cigarettes")
    Stroke: int = Field(..., ge=0, le=1, description="0 = no, 1 = yes")
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1, description="0 = no, 1 = yes")
    PhysActivity: int = Field(..., ge=0, le=1, description="0 = no, 1 = yes")
    HvyAlcoholConsump: int = Field(..., ge=0, le=1, description="0 = no, 1 = yes")
    GenHlth: int = Field(..., ge=1, le=5, description="General health scale 1=excellent ... 5=poor")
    MentHlth: int = Field(..., ge=0, le=30, description="Days in past 30 days mental health not good")
    PhysHlth: int = Field(..., ge=0, le=30, description="Days in past 30 days physical health not good")
    DiffWalk: int = Field(..., ge=0, le=1, description="0 = no serious difficulty walking, 1 = yes")
    Age: int = Field(..., ge=1, le=13, description="Age category 1-13 (_AGEG5YR)")
    Education: int = Field(..., ge=1, le=6, description="Education level scale 1-6 (EDUCA)")
    Income: int = Field(..., ge=1, le=8, description="Income scale 1-8 (INCOME2)")

    class Config:
        schema_extra = {
            "example": {
                "HighBP": 1,
                "HighChol": 0,
                "CholCheck": 1,
                "BMI": 28.5,
                "Smoker": 0,
                "Stroke": 0,
                "HeartDiseaseorAttack": 0,
                "PhysActivity": 1,
                "HvyAlcoholConsump": 0,
                "GenHlth": 3,
                "MentHlth": 2,
                "PhysHlth": 1,
                "DiffWalk": 0,
                "Age": 5,
                "Education": 4,
                "Income": 3
            }
        }


# ------------------ FASTAPI ------------------
app = FastAPI(title="Diabetes Predictor", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500","http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ ROUTES ------------------
@app.get("/")
def root():
    return {"message": "Diabetes Predictor API running. Use /docs to explore."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/test-model")
def test_model():
    try:
        if model is None:
            return {"status": "Model not loaded"}
        return {
            "status": "Model loaded successfully",
            "model_type": str(type(model)),
            "features_count": len(feature_cols)
        }
    except Exception as e:
        return {"status": "Error loading model", "error": str(e)}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        payload = req.dict()
        key = make_cache_key(payload)

        # Check Redis cache
        cached = r.get(key)
        if cached:
            result = json.loads(cached)
            result["cached"] = True
            return result

        # Prepare DataFrame
        df = pd.DataFrame([payload])
        df.columns = df.columns.str.strip()
        df = df[feature_cols]

        # Cast numeric columns to float
        numeric_cols = [col for col, typ in feature_types.items() if typ == "float"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = df[c].astype(float)

        # Validate features against trained model
        trained_features = model.get_booster().feature_names
        missing = set(trained_features) - set(df.columns)
        extra = set(df.columns) - set(trained_features)
        if missing:
            return {"error": f"Missing features for prediction: {missing}"}
        if extra:
            return {"error": f"Extra features in input: {extra}"}

        # Predict
        prob = float(model.predict_proba(df)[:, 1][0])
        pred = int(prob >= 0.5)

        result = {"prediction": pred, "probability_diabetes": prob, "cached": False}

        # Cache result
        r.setex(key, CACHE_TTL_SECONDS, json.dumps(result))

        # Insert prediction into DB
        db_data = {**payload, "prediction": pred, "probability": prob}
        cols = ", ".join(list(db_data.keys()))
        values = ", ".join([f":{k}" for k in db_data.keys()])
        insert_sql = text(f"""
            INSERT INTO {TABLE_NAME} ({cols}, created_at)
            VALUES ({values}, NOW())
        """)
        try:
            with engine.begin() as conn:
                conn.execute(insert_sql, db_data)
        except SQLAlchemyError as e:
            print(f"Database insert failed: {e}")

        return result

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.get("/features")
def get_features():
    """Return selected features with type, validation ranges, and descriptions"""
    feature_info = {
        "HighBP": {"type":"int", "min":0, "max":1, "default":0, "description":"0 = no high BP, 1 = high BP"},
        "HighChol": {"type":"int", "min":0, "max":1, "default":0, "description":"0 = no high cholesterol, 1 = high cholesterol"},
        "CholCheck": {"type":"int", "min":0, "max":1, "default":0, "description":"0 = no cholesterol check in 5 years, 1 = yes"},
        "BMI": {"type":"float", "min":0, "max":100, "default":25, "description":"Body Mass Index"},
        "Smoker": {"type":"int", "min":0, "max":1, "default":0, "description":"0 = no, 1 = yes, smoked at least 100 cigarettes"},
        "Stroke": {"type":"int", "min":0, "max":1, "default":0, "description":"0 = no, 1 = yes"},
        "HeartDiseaseorAttack": {"type":"int", "min":0, "max":1, "default":0, "description":"0 = no, 1 = yes"},
        "PhysActivity": {"type":"int", "min":0, "max":1, "default":0, "description":"0 = no, 1 = yes"},
        "HvyAlcoholConsump": {"type":"int", "min":0, "max":1, "default":0, "description":"0 = no, 1 = yes"},
        "GenHlth": {"type":"int", "min":1, "max":5, "default":3, "description":"General health scale 1 = excellent ... 5 = poor"},
        "MentHlth": {"type":"int", "min":0, "max":30, "default":0, "description":"Days in past 30 days mental health not good"},
        "PhysHlth": {"type":"int", "min":0, "max":30, "default":0, "description":"Days in past 30 days physical health not good"},
        "DiffWalk": {"type":"int", "min":0, "max":1, "default":0, "description":"0 = no serious difficulty walking, 1 = yes"},
        "Age": {"type":"int", "min":1, "max":13, "default":5, "description":"Age category 1-13 (_AGEG5YR)"},
        "Education": {"type":"int", "min":1, "max":6, "default":3, "description":"Education level scale 1-6 (EDUCA)"},
        "Income": {"type":"int", "min":1, "max":8, "default":3, "description":"Income scale 1-8 (INCOME2)"},
    }
    return {"feature_info": feature_info}

