"""
AIML-1 — Week 4: Model Validation on MHSDS
AI Mental Health UK · May 2026 Sprint

Validate ensemble model against MHSDS monthly dataset.
Report generalisation metrics.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

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

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "ensemble_v3.pkl"
VALIDATION_REPORT = REPORT_DIR / "validation_report.txt"

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model():
    """Load trained ensemble model."""
    if not MODEL_PATH.exists():
        log.error(f"Model not found: {MODEL_PATH}")
        return None
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    log.info(f"Loaded model: {MODEL_PATH}")
    return model

# ============================================================================
# CREATE VALIDATION DATA
# ============================================================================

def create_validation_data(n_samples=200):
    """Create synthetic validation dataset (simulating MHSDS data)."""
    
    log.info(f"Creating {n_samples} validation samples...")
    
    rng = np.random.default_rng(42)
    
    # Generate slightly different distribution than training
    # (simulating real-world distribution shift)
    
    phq9 = rng.normal(12, 7, size=n_samples).clip(0, 27).astype(int)
    gad7 = rng.normal(10, 6, size=n_samples).clip(0, 21).astype(int)
    
    df = pd.DataFrame({
        "phq9_score": phq9,
        "gad7_score": gad7,
    })
    
    # Create features matching training set
    df["age_band_code"] = rng.integers(0, 6, size=n_samples)
    df["imd_proxy"] = rng.integers(1, 6, size=n_samples)
    df["high_deprivation"] = (df["imd_proxy"] <= 2).astype(int)
    df["flag_depression"] = (phq9 >= 10).astype(int)
    df["flag_anxiety"] = (gad7 >= 8).astype(int)
    df["flag_comorbid_anx_dep"] = ((phq9 >= 10) & (gad7 >= 8)).astype(int)
    df["comorbidity_count"] = df["flag_depression"] + df["flag_anxiety"] + df["flag_comorbid_anx_dep"]
    
    # Tier codes
    df["phq9_tier_code"] = pd.cut(phq9, bins=[-1,4,9,14,19,27], labels=[0,1,2,3,4]).astype(int)
    df["gad7_tier_code"] = pd.cut(gad7, bins=[-1,4,9,14,21], labels=[0,1,2,3]).astype(int)
    df["severity_index"] = ((phq9 + gad7) / 48).round(4)
    
    # Target - different threshold for validation
    df["high_risk"] = ((phq9 >= 15) | (gad7 >= 15)).astype(int)
    
    log.info(f"Validation data created: {df.shape}")
    log.info(f"Target distribution: {df['high_risk'].value_counts().to_dict()}")
    
    return df

# ============================================================================
# RUN VALIDATION
# ============================================================================

def run_validation(model, df):
    """Run validation metrics."""
    
    log.info("Running validation...")
    
    # Get features matching the validation dataset and exclude the target.
    feature_cols = [
        "age_band_code", "imd_proxy", "high_deprivation",
        "flag_depression", "flag_anxiety", "flag_comorbid_anx_dep", "comorbidity_count",
        "phq9_score", "gad7_score", "phq9_tier_code", "gad7_tier_code",
        "severity_index"
    ]

    avail_cols = [c for c in feature_cols if c in df.columns]
    if not avail_cols:
        raise ValueError("No validation feature columns available")

    X = df[avail_cols].values.astype(np.float32)
    y = df["high_risk"].values.astype(int)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "avg_precision": average_precision_score(y_test, y_prob),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    log.info("=" * 50)
    log.info("VALIDATION METRICS")
    log.info("=" * 50)
    for k, v in metrics.items():
        log.info(f"  {k}: {v:.4f}")
    
    log.info(f"\nConfusion Matrix:\n{cm}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk'])}")
    
    return metrics, cm, X_test, y_test, y_prob

# ============================================================================
# PLOT ROC CURVE
# ============================================================================

def plot_roc(y_test, y_prob, output_dir):
    """Plot ROC curve."""
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(fpr, tpr, "b-", lw=2, label="ROC curve")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Validation Set")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out = output_dir / "validation_roc.png"
    plt.savefig(out, dpi=150)
    plt.close()
    
    log.info(f"ROC plot → {out}")

# ============================================================================
# PLOT CONFUSION MATRIX
# ============================================================================

def plot_confusion(cm, output_dir):
    """Plot confusion matrix."""
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low Risk", "High Risk"])
    ax.set_yticklabels(["Low Risk", "High Risk"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    
    # Add text
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=20)
    
    plt.tight_layout()
    out = output_dir / "validation_confusion.png"
    plt.savefig(out, dpi=150)
    plt.close()
    
    log.info(f"Confusion plot → {out}")

# ============================================================================
# WRITE REPORT
# ============================================================================

def write_report(metrics, output_dir):
    """Write validation report."""
    
    path = output_dir / "validation_report.txt"
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("WEEK 4 — AIML-1: MODEL VALIDATION REPORT\n")
        f.write("AI Mental Health UK Project\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("--- VALIDATION DATASET ---\n")
        f.write("  Source: MHSDS Monthly (simulated)\n")
        f.write("  Sample Size: 200\n\n")
        
        f.write("--- GENERALISATION METRICS ---\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        
        f.write("\n--- INTERPRETATION ---\n")
        
        if metrics["roc_auc"] >= 0.9:
            f.write("  ✅ Excellent generalisation (AUC >= 0.9)\n")
        elif metrics["roc_auc"] >= 0.8:
            f.write("  ✅ Good generalisation (AUC >= 0.8)\n")
        elif metrics["roc_auc"] >= 0.7:
            f.write("  ⚠️  Moderate generalisation (AUC >= 0.7)\n")
        else:
            f.write("  ❌ Poor generalisation (AUC < 0.7)\n")
        
        if metrics["f1"] >= 0.8:
            f.write("  ✅ Good F1 score (>= 0.8)\n")
        else:
            f.write("  ⚠️  F1 needs improvement\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    log.info(f"Report → {path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    log.info("--- AIML-1 VALIDATION STARTED ---")
    
    # Load model
    model = load_model()
    if model is None:
        log.error("Failed to load model")
        return
    
    # Create validation data
    df = create_validation_data(n_samples=200)
    
    # Run validation
    metrics, cm, X_test, y_test, y_prob = run_validation(model, df)
    
    # Plots
    plot_roc(y_test, y_prob, REPORT_DIR)
    plot_confusion(cm, REPORT_DIR)
    
    # Report
    write_report(metrics, REPORT_DIR)
    
    log.info("--- VALIDATION COMPLETE ---")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()