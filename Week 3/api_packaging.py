"""
AIML-4 — Week 3: Model Packaging for API
AI Mental Health UK · May 2026 Sprint

Export final model as pickle + ONNX (if available).
Write inference.py wrapper for FastAPI consumption.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Check for ONNX
try:
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    print("Installing onnx...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "skl2onnx", "onnx"])
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)

# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(".").resolve()
DATA_PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
API_DIR = ROOT / "api"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
API_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model():
    """Load the trained ensemble model."""
    model_path = MODEL_DIR / "ensemble_v3.pkl"
    
    if not model_path.exists():
        log.error("Ensemble model not found. Run AIML-2 first.")
        return None
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    log.info(f"Loaded model from {model_path}")
    return model

# ============================================================================
# LOAD FEATURE DATA
# ============================================================================

def get_feature_names():
    """Get feature names for the model."""
    feature_set_v2 = DATA_PROCESSED / "feature_set_v2.csv"
    score_features = DATA_PROCESSED / "score_features.csv"
    
    df1 = pd.read_csv(feature_set_v2)
    df2 = pd.read_csv(score_features)
    
    min_rows = min(len(df1), len(df2))
    df1 = df1.iloc[:min_rows].reset_index(drop=True)
    df2 = df2.iloc[:min_rows].reset_index(drop=True)
    
    X = pd.concat([df1, df2], axis=1)
    X = X.select_dtypes(include=[np.number]).fillna(0)
    
    return X.columns.tolist()

# ============================================================================
# EXPORT ONNX
# ============================================================================

def export_onnx(model, feature_names, output_dir):
    """Export model to ONNX format."""
    if not ONNX_AVAILABLE:
        log.warning("ONNX not available, skipping...")
        return None
    
    log.info("Exporting to ONNX...")
    
    try:
        # Create initial types
        initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
        
        # Convert
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            options={'zipmap': False}
        )
        
        # Save
        onnx_path = output_dir / "model.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        log.info(f"ONNX model saved → {onnx_path}")
        return onnx_path
    
    except Exception as e:
        log.warning(f"ONNX export failed: {e}")
        return None

# ============================================================================
# CREATE INFERENCE WRAPPER
# ============================================================================

def create_inference_wrapper(output_dir):
    """Create FastAPI inference wrapper."""
    
    code = '''"""
FastAPI Inference Wrapper
AI Mental Health UK — Mental Health Risk Prediction API

Usage:
    uvicorn inference:app --reload

Endpoints:
    POST /predict     - Predict high risk
    GET  /health      - Health check
    GET  /model-info  - Model information
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ensemble_v3.pkl"
FEATURE_NAMES = %s

# ============================================================================
# LOAD MODEL
# ============================================================================

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="AI Mental Health UK API",
    description="Mental Health Risk Prediction Model",
    version="1.0.0"
)

# ============================================================================
# INPUT MODEL
# ============================================================================

class PredictionInput(BaseModel):
    features: list = Field(
        ...,
        description="List of feature values",
        example=[0.5] * len(FEATURE_NAMES)
    )

class PredictionOutput(BaseModel):
    prediction: int = Field(..., description="0=Low Risk, 1=High Risk")
    probability: float = Field(..., description="Probability of high risk")
    confidence: str = Field(..., description="Risk level")

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {"message": "AI Mental Health UK API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "loaded"}

@app.get("/model-info")
def model_info():
    return {
        "model_type": "VotingClassifier",
        "n_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Validate input
        if len(input_data.features) != len(FEATURE_NAMES):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(FEATURE_NAMES)} features, got {len(input_data.features)}"
            )
        
        # Make prediction
        X = np.array(input_data.features).astype(np.float32).reshape(1, -1)
        
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[:, 1][0]
        
        # Determine confidence
        if prob > 0.7:
            confidence = "HIGH"
        elif prob > 0.3:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return PredictionOutput(
            prediction=int(pred),
            probability=float(prob),
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
''' % str(get_feature_names())
    
    # Save wrapper
    wrapper_path = output_dir / "inference.py"
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(code)
    
    log.info(f"Inference wrapper saved → {wrapper_path}")
    
    return wrapper_path

# ============================================================================
# CREATE REQUIREMENTS
# ============================================================================

def create_requirements(output_dir):
    """Create requirements.txt for API."""
    
    requirements = """# AI Mental Health UK API Requirements
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
""".strip()
    
    req_path = output_dir / "requirements.txt"
    with open(req_path, "w") as f:
        f.write(requirements)
    
    log.info(f"Requirements saved → {req_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    log.info("--- AIML-4 API PACKAGING STARTED ---")
    
    # Load model
    model = load_model()
    
    if model is None:
        log.error("Failed to load model")
        return
    
    # Get feature names
    feature_names = get_feature_names()
    log.info(f"Features: {len(feature_names)}")
    
    # Export ONNX
    onnx_path = export_onnx(model, feature_names, MODEL_DIR)
    
    # Create inference wrapper
    inference_path = create_inference_wrapper(API_DIR)
    
    # Create requirements
    create_requirements(API_DIR)
    
    # Summary
    log.info("=" * 50)
    log.info("API PACKAGING COMPLETE")
    log.info("=" * 50)
    log.info(f"Model: {MODEL_DIR / 'ensemble_v3.pkl'}")
    if onnx_path:
        log.info(f"ONNX:  {onnx_path}")
    log.info(f"Inference: {inference_path}")
    log.info(f"Requirements: {API_DIR / 'requirements.txt'}")
    
    log.info("\nTo run API:")
    log.info("  pip install -r api/requirements.txt")
    log.info("  uvicorn api/inference:app --reload")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()