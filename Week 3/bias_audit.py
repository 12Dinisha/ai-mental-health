"""
AIML-1 — Week 3 Bias Audit
AI Mental Health UK · May 2026 Sprint

Evaluates XGBoost model across demographic groups.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

MODEL_PATH = MODEL_DIR / "xgboost_v2.pkl"

# ============================================================================
# LOAD DATA
# ============================================================================

def load_data_and_model():
    """Load feature data and trained model."""
    feature_set_v2 = DATA_PROCESSED / "feature_set_v2.csv"
    score_features = DATA_PROCESSED / "score_features.csv"
    
    if not feature_set_v2.exists() or not score_features.exists():
        log.error("Features not found.")
        return None, None, None
    
    df1 = pd.read_csv(feature_set_v2)
    df2 = pd.read_csv(score_features)
    
    min_rows = min(len(df1), len(df2))
    df1 = df1.iloc[:min_rows].reset_index(drop=True)
    df2 = df2.iloc[:min_rows].reset_index(drop=True)
    
    X = pd.concat([df1, df2], axis=1)
    
    if "high_risk" not in X.columns:
        log.error("Target not found")
        return None, None, None
    
    y = X.pop("high_risk")
    X = X.select_dtypes(include=[np.number]).fillna(0)
    
    if not MODEL_PATH.exists():
        log.error("Model not found.")
        return None, None, None
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    log.info(f"Loaded: {X.shape}, target: {y.value_counts().to_dict()}")
    
    return X, y, model

# ============================================================================
# CREATE SENSITIVE FEATURES
# ============================================================================

def create_sensitive_attributes(df):
    """Create synthetic sensitive attributes."""
    n = len(df)
    rng = np.random.default_rng(42)
    
    # Age groups
    if "age_band_code" in df.columns:
        age = df["age_band_code"].fillna(2).astype(int)
    else:
        age = rng.integers(0, 5, size=n)
    
    # Gender
    if "flag_anxiety" in df.columns:
        gender = (df["flag_anxiety"] > 0).astype(int)
    else:
        gender = rng.integers(0, 2, size=n)
    
    # Deprivation
    if "imd_proxy" in df.columns:
        deprivation = df["imd_proxy"].fillna(3).astype(int)
    else:
        deprivation = rng.integers(1, 6, size=n)
    
    return pd.DataFrame({
        "age_group": age,
        "gender": gender,
        "deprivation": deprivation
    })

# ============================================================================
# EVALUATE FAIRNESS
# ============================================================================

def evaluate_fairness(X, y, model, sensitive_features):
    """Evaluate fairness across groups."""
    log.info("Evaluating fairness...")
    
    # Convert to numpy
    X_np = X.values.astype(np.float32)
    y_np = y.values.astype(int)
    sf_np = sensitive_features.values
    
    # Split indices
    idx = np.arange(len(y_np))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42, 
        stratify=y_np
    )
    
    X_test = X_np[test_idx]
    y_test = y_np[test_idx]
    sf_test = sf_np[test_idx]
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Overall metrics
    overall = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
    }
    
    log.info(f"Overall: {overall}")
    
    # Per-group metrics
    results = []
    sf_cols = sensitive_features.columns
    
    for i, attr in enumerate(sf_cols):
        log.info(f"\n--- {attr.upper()} ---")
        
        unique_vals = sorted(np.unique(sf_test[:, i]))
        
        for val in unique_vals:
            mask = sf_test[:, i] == val
            y_t = y_test[mask]
            y_p = y_pred[mask]
            
            m = {
                "attribute": attr,
                "value": int(val),
                "n": int(mask.sum()),
                "accuracy": accuracy_score(y_t, y_p),
                "f1": f1_score(y_t, y_p, zero_division=0),
                "precision": precision_score(y_t, y_p, zero_division=0),
                "recall": recall_score(y_t, y_p, zero_division=0),
                "selection_rate": y_p.mean(),
            }
            results.append(m)
            log.info(f"  {val}: acc={m['accuracy']:.3f}, F1={m['f1']:.3f}")
    
    results_df = pd.DataFrame(results)
    
    # Disparity
    disparity = {}
    for metric in ["accuracy", "f1", "selection_rate"]:
        if metric in results_df.columns:
            disparity[metric] = results_df[metric].max() - results_df[metric].min()
    
    log.info(f"\n--- DISPARITY ---")
    for m, d in disparity.items():
        flag = "⚠️" if d > 0.1 else "✅"
        log.info(f"  {m}: {d:.4f} {flag}")
    
    return overall, results_df, disparity

# ============================================================================
# PLOT
# ============================================================================

def plot_fairness(results_df, output_dir):
    """Plot fairness metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    attrs = ["age_group", "gender", "deprivation"]
    
    for idx, attr in enumerate(attrs):
        ax = axes[idx]
        data = results_df[results_df["attribute"] == attr]
        
        # Use numeric x positions
        x_pos = np.arange(len(data))
        ax.bar(x_pos, data["f1"], color="steelblue")
        
        # Set labels as strings
        ax.set_xticks(x_pos)
        ax.set_xticklabels(data["value"].astype(str).tolist())
        
        ax.set_title(attr.replace("_", " ").title())
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    out = output_dir / "bias_audit_plot.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info(f"Plot → {out}")

# ============================================================================
# WRITE REPORT
# ============================================================================

def write_report(overall, results_df, disparity, output_dir):
    """Write bias audit report."""
    path = output_dir / "bias_audit_report.txt"
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("WEEK 3 — AIML-1: BIAS AUDIT REPORT\n")
        f.write("AI Mental Health UK Project\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("--- OVERALL ---\n")
        for k, v in overall.items():
            f.write(f"  {k}: {v:.4f}\n")
        
        f.write("\n--- DISPARITY (max-min) ---\n")
        for k, d in disparity.items():
            status = "WARNING" if d > 0.1 else "OK"
            f.write(f"  {k}: {d:.4f} [{status}]\n")
        
        f.write("\n--- PER-GROUP ---\n")
        f.write(results_df.to_string(index=False))
        
        f.write("\n\n--- RECOMMENDATIONS ---\n")
        if disparity.get("selection_rate", 0) > 0.1:
            f.write("- Selection rate disparity > 10%: Consider threshold adjustment\n")
        if disparity.get("f1", 0) > 0.1:
            f.write("- F1 disparity > 10%: Review group-level training\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    log.info(f"Report → {path}")
    
    results_df.to_csv(output_dir / "disparity_metrics.csv", index=False)
    log.info(f"CSV → {output_dir / 'disparity_metrics.csv'}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    log.info("--- BIAS AUDIT STARTED ---")
    
    X, y, model = load_data_and_model()
    
    if X is None:
        log.error("Load failed")
        return
    
    sf = create_sensitive_attributes(X)
    
    overall, results, disparity = evaluate_fairness(X, y, model, sf)
    
    plot_fairness(results, REPORT_DIR)
    write_report(overall, results, disparity, REPORT_DIR)
    
    log.info("--- COMPLETE ---")

if __name__ == "__main__":
    main()