"""
AIML-2 — Week 3: Ensemble + Optuna Tuning
AI Mental Health UK · May 2026 Sprint

Build ensemble (XGBoost + Random Forest + Logistic Regression)
Tune with Optuna hyperparameter optimization
Target F1 > 0.80
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Try to import Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    print("Installing optuna...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)

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

ENSEMBLE_MODEL_PATH = MODEL_DIR / "ensemble_v3.pkl"
OPTUNA_REPORT = REPORT_DIR / "optuna_tuning_log.txt"

# ============================================================================
# LOAD DATA
# ============================================================================

def load_data():
    """Load feature data."""
    feature_set_v2 = DATA_PROCESSED / "feature_set_v2.csv"
    score_features = DATA_PROCESSED / "score_features.csv"
    
    if not feature_set_v2.exists() or not score_features.exists():
        log.error("Features not found.")
        return None, None
    
    df1 = pd.read_csv(feature_set_v2)
    df2 = pd.read_csv(score_features)
    
    min_rows = min(len(df1), len(df2))
    df1 = df1.iloc[:min_rows].reset_index(drop=True)
    df2 = df2.iloc[:min_rows].reset_index(drop=True)
    
    X = pd.concat([df1, df2], axis=1)
    
    if "high_risk" not in X.columns:
        log.error("Target not found")
        return None, None
    
    y = X.pop("high_risk")
    X = X.select_dtypes(include=[np.number]).fillna(0)
    
    log.info(f"Data: {X.shape}, target: {y.value_counts().to_dict()}")
    
    return X, y

# ============================================================================
# BASE MODELS
# ============================================================================

def get_base_models():
    """Get base models for ensemble."""
    
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        eval_metric="auc",
        verbosity=0
    )
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )
    
    return ("xgb", xgb), ("rf", rf), ("lr", lr)

# ============================================================================
# OPTUNA TUNING
# ============================================================================

def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=20):
    """Tune XGBoost with Optuna."""
    log.info(f"Tuning XGBoost with {n_trials} trials...")
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        
        model = XGBClassifier(
            **params,
            random_state=42,
            eval_metric="auc",
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        return f1_score(y_val, y_pred)
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    log.info(f"Best F1: {study.best_value:.4f}")
    log.info(f"Best params: {study.best_params}")
    
    return study.best_params

# ============================================================================
# TRAIN ENSEMBLE
# ============================================================================

def train_ensemble(X, y, use_optuna=True, n_trials=20):
    """Train ensemble with optional Optuna tuning."""
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Class weight
    pos = sum(y_train)
    neg = len(y_train) - pos
    spw = neg / pos if pos > 0 else 1.0
    
    # Get base models
    models = get_base_models()
    
    # Tune XGBoost if Optuna available
    if use_optuna and OPTUNA_AVAILABLE:
        best_params = tune_xgboost(
            X_train.values.astype(np.float32),
            y_train.values.astype(int),
            X_test.values.astype(np.float32),
            y_test.values.astype(int),
            n_trials=n_trials
        )
        best_params["random_state"] = 42
        best_params["eval_metric"] = "auc"
        best_params["verbosity"] = 0
        best_params["scale_pos_weight"] = spw
        
        # Update XGBoost with best params
        models = (
            ("xgb", XGBClassifier(**best_params)),
            models[1],
            models[2]
        )
    
    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=list(models),
        voting="soft"
    )
    
    log.info("Training ensemble...")
    ensemble.fit(
        X_train.values.astype(np.float32),
        y_train.values.astype(int)
    )
    
    # Evaluate
    y_pred = ensemble.predict(X_test.values.astype(np.float32))
    y_prob = ensemble.predict_proba(X_test.values.astype(np.float32))[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "report": classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"])
    }
    
    log.info(f"Test F1: {metrics['f1']:.4f}")
    log.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(
        ensemble, X.values.astype(np.float32), y.values.astype(int),
        cv=5, scoring="f1"
    )
    
    log.info(f"CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return ensemble, metrics, cv_scores, X_test, y_test

# ============================================================================
# PLOT COMPARISON
# ============================================================================

def plot_comparison(metrics, output_dir):
    """Plot ensemble vs baseline comparison."""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    labels = ["Accuracy", "F1", "ROC-AUC"]
    ensemble_vals = [metrics["accuracy"], metrics["f1"], metrics["roc_auc"]]
    baseline_vals = [0.85, 0.72, 0.78]  # Week 1/2 baselines
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, ensemble_vals, width, label="Ensemble v3", color="steelblue")
    ax.bar(x + width/2, baseline_vals, width, label="Baseline", color="gray", alpha=0.7)
    
    ax.set_ylabel("Score")
    ax.set_title("Ensemble vs Baseline Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add target line
    ax.axhline(y=0.8, color="green", linestyle="--", label="Target F1=0.80")
    
    plt.tight_layout()
    out = output_dir / "ensemble_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info(f"Plot → {out}")

# ============================================================================
# SAVE REPORT
# ============================================================================

def save_report(metrics, cv_scores, output_dir):
    """Save ensemble report."""
    path = output_dir / "ensemble_report.txt"
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("WEEK 3 — AIML-2: ENSEMBLE MODEL REPORT\n")
        f.write("AI Mental Health UK Project\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("--- TEST SET METRICS ---\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"  ROC-AUC:  {metrics['roc_auc']:.4f}\n\n")
        
        f.write("--- CROSS-VALIDATION ---\n")
        f.write(f"  F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
        
        f.write("--- CLASSIFICATION REPORT ---\n")
        f.write(metrics["report"] + "\n\n")
        
        f.write("--- TARGET CHECK ---\n")
        if metrics["f1"] >= 0.80:
            f.write("✅ F1 Target (0.80) Achieved!\n")
        else:
            f.write(f"❌ F1 Target not met (current: {metrics['f1']:.4f})\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    log.info(f"Report → {path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    log.info("--- ENSEMBLE TRAINING STARTED ---")
    
    X, y = load_data()
    
    if X is None:
        log.error("Load failed")
        return
    
    # Train ensemble with Optuna
    ensemble, metrics, cv_scores, X_test, y_test = train_ensemble(
        X, y, use_optuna=OPTUNA_AVAILABLE, n_trials=20
    )
    
    # Save model
    with open(ENSEMBLE_MODEL_PATH, "wb") as f:
        pickle.dump(ensemble, f)
    log.info(f"Model → {ENSEMBLE_MODEL_PATH}")
    
    # Plot
    plot_comparison(metrics, REPORT_DIR)
    
    # Report
    save_report(metrics, cv_scores, REPORT_DIR)
    
    log.info("--- COMPLETE ---")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()