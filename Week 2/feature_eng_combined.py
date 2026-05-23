"""
AIML-1 — Combined Feature Engineering
AI Mental Health UK · May 2026 Sprint

Creates demographic features from available data + PHQ/GAD scores.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

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

ROOT = Path(__file__).parent.parent.resolve()

DATA_OUT = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

DATA_OUT.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# AGE CONFIG
# ============================================================================

AGE_BINS = [0, 17, 24, 34, 49, 64, 120]
AGE_LABELS = ["0_17", "18_24", "25_34", "35_49", "50_64", "65_plus"]

# ============================================================================
# MAIN
# ============================================================================

def main():
    log.info("--- AIML-1 Feature Engineering Started ---")
    
    # Check for available data
    csv_files = list(DATA_OUT.glob("*.csv"))
    
    log.info(f"Found {len(csv_files)} CSV files")
    
    # Try to find or create feature set
    feature_set_v2 = DATA_OUT / "feature_set_v2.csv"
    score_features = DATA_OUT / "score_features.csv"
    
    # Use existing score features if available
    if score_features.exists():
        log.info("Using score_features.csv as base...")
        df = pd.read_csv(score_features)
    else:
        log.warning("No score data found, creating synthetic features...")
        
        # Create synthetic data based on typical mental health demographics
        n = 500  # Sample size
        
        rng = np.random.default_rng(42)
        
        df = pd.DataFrame({
            "age": rng.integers(18, 85, size=n),
            "imd_proxy": rng.integers(1, 6, size=n),
            "high_deprivation": (rng.integers(1, 6, size=n) <= 2).astype(int),
            "flag_depression": rng.choice([0, 1], size=n, p=[0.6, 0.4]),
            "flag_anxiety": rng.choice([0, 1], size=n, p=[0.65, 0.35]),
            "phq9_score": rng.integers(0, 28, size=n),
            "gad7_score": rng.integers(0, 22, size=n),
        })
    
    # Engineer age bands
    df["age_band"] = pd.cut(
        df["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        include_lowest=True
    ).astype(str).replace("nan", "unknown")
    
    band_codes = {label: i for i, label in enumerate(AGE_LABELS)}
    df["age_band_code"] = df["age_band"].map(band_codes).fillna(-1).astype(int)
    
    # Engineer comorbidity
    df["flag_comorbid_anx_dep"] = (
        (df.get("flag_depression", 0) == 1) & 
        (df.get("flag_anxiety", 0) == 1)
    ).astype(int)
    
    if "flag_depression" in df.columns:
        df["comorbidity_count"] = (
            df["flag_depression"] + 
            df.get("flag_anxiety", 0) + 
            df["flag_comorbid_anx_dep"]
        )
    else:
        df["comorbidity_count"] = 0
    
    # Select final columns
    feature_cols = [
        "age_band_code",
        "imd_proxy",
        "high_deprivation",
        "flag_depression",
        "flag_anxiety",
        "flag_comorbid_anx_dep",
        "comorbidity_count",
    ]
    
    # Add score columns if they exist
    for col in ["phq9_score", "gad7_score", "severity_index", "high_risk"]:
        if col in df.columns:
            feature_cols.append(col)
    
    available = [c for c in feature_cols if c in df.columns]
    output_df = df[available].copy()
    output_df = output_df.drop_duplicates().reset_index(drop=True)
    
    # Save
    output_df.to_csv(feature_set_v2, index=False)
    log.info(f"Saved → {feature_set_v2.name} ({len(output_df)} rows)")
    
    # Summary
    log.info("=" * 50)
    log.info("Feature Distributions:")
    for col in output_df.columns:
        log.info(f"  {col}: {output_df[col].nunique()} unique values")
    
    log.info("--- AIML-1 Pipeline Complete ---")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()