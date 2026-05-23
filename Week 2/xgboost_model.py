"""
AIML-4 — Week 2 Model v2: XGBoost Classifier (Fixed)
AI Mental Health UK · May 2026 Sprint
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    classification_report, RocCurveDisplay
)

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ============================================================================
# PATHS
# ============================================================================

FEATURE_SET_V2 = Path("data/processed/feature_set_v2.csv")
SCORE_FEATURES = Path("data/processed/score_features.csv")

MODEL_OUT      = Path("models/xgboost_v2.pkl")
REPORT_OUT     = Path("reports/xgboost_comparison_report.txt")
ROC_OUT        = Path("reports/xgboost_roc_curve.png")

RANDOM_STATE = 42
TARGET_COL   = "high_risk"

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_merge_features():
    """Load and merge AIML-1 and AIML-2 feature sets."""
    if not FEATURE_SET_V2.exists() or not SCORE_FEATURES.exists():
        log.error("Feature files missing. Run AIML-1 and AIML-2 first.")
        sys.exit(1)
        
    log.info("Loading engineered feature sets...")
    df1 = pd.read_csv(FEATURE_SET_V2)
    df2 = pd.read_csv(SCORE_FEATURES)
    
    # Align row counts
    min_rows = min(len(df1), len(df2))
    df1 = df1.iloc[:min_rows].reset_index(drop=True)
    df2 = df2.iloc[:min_rows].reset_index(drop=True)
    
    log.info(f"  AIML-1: {df1.shape}")
    log.info(f"  AIML-2: {df2.shape}")
    
    # Merge
    X = pd.concat([df1, df2], axis=1)
    
    # Get target
    if TARGET_COL not in X.columns:
        log.error(f"Target '{TARGET_COL}' not found.")
        sys.exit(1)
    
    y = X.pop(TARGET_COL)
    
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Fill missing values
    if X.isna().any().any():
        X = X.fillna(X.median())
    
    # Convert to float32 for XGBoost
    X = X.astype(np.float32)
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    
    log.info(f"Feature matrix: {X.shape}")
    log.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_xgboost(X, y):
    """Train XGBoost classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Class imbalance weight
    pos_count = sum(y_train)
    neg_count = len(y_train) - pos_count
    spw = neg_count / pos_count if pos_count > 0 else 1.0
    
    log.info(f"Class imbalance: {neg_count} neg, {pos_count} pos (ratio: {spw:.2f})")
    log.info("Training XGBoost...")
    
    # Model
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        eval_metric="auc",
        early_stopping_rounds=20,
        verbosity=0
    )
    
    # Convert to numpy arrays
    X_train_np = X_train.values.astype(np.float32)
    y_train_np = y_train.values.astype(int)
    X_test_np = X_test.values.astype(np.float32)
    y_test_np = y_test.values.astype(int)
    
    # Fit
    xgb.fit(
        X_train_np, y_train_np,
        eval_set=[(X_test_np, y_test_np)],
        verbose=False
    )
    
    best_iter = xgb.best_iteration if hasattr(xgb, 'best_iteration') else xgb.n_estimators
    log.info(f"Best iteration: {best_iter}")
    
    # Cross-validation
    log.info("Running 5-fold CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    xgb_cv = XGBClassifier(
        n_estimators=best_iter + 1,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        eval_metric="auc",
        verbosity=0
    )
    
    cv_results = cross_validate(
        xgb_cv, X.values.astype(np.float32), y.values.astype(int),
        cv=cv, scoring=["f1", "roc_auc"], n_jobs=-1
    )
    
    # Predictions
    y_pred = xgb.predict(X_test_np)
    y_prob = xgb.predict_proba(X_test_np)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test_np, y_pred),
        "f1": f1_score(y_test_np, y_pred),
        "roc_auc": roc_auc_score(y_test_np, y_prob),
        "cv_f1_mean": cv_results["test_f1"].mean(),
        "cv_f1_std": cv_results["test_f1"].std(),
        "cv_auc_mean": cv_results["test_roc_auc"].mean(),
        "cv_auc_std": cv_results["test_roc_auc"].std(),
        "report": classification_report(y_test_np, y_pred, target_names=["Low Risk", "High Risk"])
    }
    
    return xgb, metrics, X_test_np, y_test_np

# ============================================================================
# BASELINE COMPARISON
# ============================================================================

def compare_with_baseline(xgb_metrics):
    """Compare with Week 1 baseline."""
    baseline = {"f1": 0.7200, "roc_auc": 0.7850}
    delta = {
        "f1_delta": xgb_metrics["f1"] - baseline["f1"],
        "auc_delta": xgb_metrics["roc_auc"] - baseline["roc_auc"]
    }
    return baseline, delta

# ============================================================================
# OUTPUTS
# ============================================================================

def plot_roc(model, X_test, y_test, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_estimator(
        model, X_test, y_test, ax=ax,
        curve_kwargs={"color": "#1D9E75"},
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC Curve — XGBoost (Model v2)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"ROC curve → {save_path}")


def write_report(metrics, baseline, delta, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("AIML-4: Model v2 Comparison Report (XGBoost)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("-- Test Set Metrics --\n")
        f.write(f"  Accuracy : {metrics['accuracy']:.4f}\n")
        f.write(f"  F1 Score : {metrics['f1']:.4f}\n")
        f.write(f"  ROC-AUC  : {metrics['roc_auc']:.4f}\n\n")
        
        f.write("-- 5-Fold Cross Validation --\n")
        f.write(f"  F1 Score : {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}\n")
        f.write(f"  ROC-AUC  : {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}\n\n")
        
        f.write("-- Classification Report --\n")
        f.write(metrics["report"] + "\n\n")
        
        f.write("-- Baseline Comparison --\n")
        f.write(f"  F1 Delta : {delta['f1_delta']:+.4f}\n")
        f.write(f"  AUC Delta: {delta['auc_delta']:+.4f}\n")
    
    log.info(f"Report → {path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    log.info("--- AIML-4 XGBoost Pipeline ---")
    
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    
    X, y = load_and_merge_features()
    model, metrics, X_test, y_test = train_xgboost(X, y)
    
    baseline, delta = compare_with_baseline(metrics)
    
    # Save model
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Model → {MODEL_OUT}")
    
    plot_roc(model, X_test, y_test, ROC_OUT)
    write_report(metrics, baseline, delta, REPORT_OUT)
    
    log.info("--- Pipeline Complete ---")

if __name__ == "__main__":
    main()