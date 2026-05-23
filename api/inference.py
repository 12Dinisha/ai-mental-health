"""
FastAPI - FIXED
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ensemble_v3.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model: {MODEL_PATH}")
except Exception as e:
    print(f"Error: {e}")
    model = None

app = FastAPI(title="AI MH UK", version="1.0.0")

class PredictionInput(BaseModel):
    features: list = Field(..., description="16 features")

class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    confidence: str

@app.get("/")
def root():
    return {"msg": "AI MH UK"}

@app.get("/health")
def health():
    return {"status": "ok", "model": model is not None}

@app.post("/predict", response_model=PredictionOutput)
def predict(i: PredictionInput):
    if model is None:
        raise HTTPException(500, "No model")
    
    f = i.features
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and len(f) != expected:
        raise HTTPException(400, f"Expected {expected} features, got {len(f)}")

    X = np.array(f, dtype=np.float32).reshape(1, -1)
    p = model.predict(X)[0]
    pr = model.predict_proba(X)[0, 1]
    c = "HIGH" if pr > 0.7 else "MEDIUM" if pr > 0.3 else "LOW"
    return PredictionOutput(prediction=int(p), probability=float(pr), confidence=c)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)