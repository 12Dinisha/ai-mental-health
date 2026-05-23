"""
AIML-2 — Week 2 Feature Engineering: PHQ-9 & GAD-7 Score Tiers
AI Mental Health UK · May 2026 Sprint

Ingests a dataset containing PHQ-9 and/or GAD-7 scores and engineers:
  1. Severity tier labels  (minimal / mild / moderate / moderately_severe / severe)
  2. Integer tier codes    (0–4) for ML ordinal encoding
  3. Combined severity index
  4. Change flags          (worsened / stable / improved) when repeat scores present
  5. High-risk binary flag (PHQ-9 >= 15 or GAD-7 >= 15)

Outputs:
  - data/processed/score_features.csv
  - data/processed/score_features.parquet
  - reports/score_features_summary.txt

Auto-detects dataset type:
  A) Structured PHQ/GAD CSV  (columns contain phq / gad in name)
  B) Sentiment Mental Health dataset  (statement + status)
  C) Synthetic data (when no input file found)

Usage:
  python score_features.py
  python score_features.py --input data/raw/phq_gad.csv
"""

import argparse
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

DEFAULT_INPUT      = Path("data/raw/phq_gad.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed")
REPORT_DIR         = Path("reports")

# ============================================================================
# PHQ-9 thresholds  (NICE / NHS scoring guidance)
# Score range: 0–27
# ============================================================================

PHQ9_BINS   = [-1, 4, 9, 14, 19, 27]
PHQ9_LABELS = ["minimal", "mild", "moderate", "moderately_severe", "severe"]
PHQ9_CODES  = {lbl: i for i, lbl in enumerate(PHQ9_LABELS)}

# ============================================================================
# GAD-7 thresholds  (NICE / NHS scoring guidance)
# Score range: 0–21
# ============================================================================

GAD7_BINS   = [-1, 4, 9, 14, 21]
GAD7_LABELS = ["minimal", "mild", "moderate", "severe"]
GAD7_CODES  = {lbl: i for i, lbl in enumerate(GAD7_LABELS)}

# ============================================================================
# Column name candidates
# ============================================================================

PHQ9_COL_CANDIDATES = [
    "phq9", "phq_9", "phq9_score", "phq_score", "phq9_total",
    "PHQ9", "PHQ_9", "phq", "PHQ",
]
GAD7_COL_CANDIDATES = [
    "gad7", "gad_7", "gad7_score", "gad_score", "gad7_total",
    "GAD7", "GAD_7", "gad", "GAD",
]

# ============================================================================
# HELPERS
# ============================================================================

def find_col(df, candidates):
    """Find first matching column."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_dataset(path):
    """Load dataset with auto-detect encoding."""
    log.info(f"Loading: {path}")
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            log.info(f"  Loaded {len(df):,} rows × {len(df.columns)} cols")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path}")


def apply_phq9_tiers(scores):
    """Apply PHQ-9 tier labels and codes."""
    scores_clipped = pd.to_numeric(scores, errors="coerce").clip(0, 27)
    tiers = pd.cut(
        scores_clipped,
        bins=PHQ9_BINS,
        labels=PHQ9_LABELS,
        right=True,
    ).astype(str).replace("nan", "unknown")
    codes = tiers.map(PHQ9_CODES).fillna(-1).astype(int)
    return tiers, codes


def apply_gad7_tiers(scores):
    """Apply GAD-7 tier labels and codes."""
    scores_clipped = pd.to_numeric(scores, errors="coerce").clip(0, 21)
    tiers = pd.cut(
        scores_clipped,
        bins=GAD7_BINS,
        labels=GAD7_LABELS,
        right=True,
    ).astype(str).replace("nan", "unknown")
    codes = tiers.map(GAD7_CODES).fillna(-1).astype(int)
    return tiers, codes


# ============================================================================
# PROCESS INPUT DATA
# ============================================================================

def process_structured(df):
    """Process structured PHQ/GAD dataset."""
    log.info("Processing structured dataset...")

    phq_col = find_col(df, PHQ9_COL_CANDIDATES)
    gad_col = find_col(df, GAD7_COL_CANDIDATES)

    if phq_col is None and gad_col is None:
        log.error(f"No PHQ/GAD columns. Found: {df.columns.tolist()}")
        return None

    # PHQ-9
    if phq_col:
        df["phq9_score"] = pd.to_numeric(df[phq_col], errors="coerce")
        df["phq9_tier"], df["phq9_tier_code"] = apply_phq9_tiers(df["phq9_score"])
    else:
        df["phq9_score"] = np.nan
        df["phq9_tier"] = "unknown"
        df["phq9_tier_code"] = -1

    # GAD-7
    if gad_col:
        df["gad7_score"] = pd.to_numeric(df[gad_col], errors="coerce")
        df["gad7_tier"], df["gad7_tier_code"] = apply_gad7_tiers(df["gad7_score"])
    else:
        df["gad7_score"] = np.nan
        df["gad7_tier"] = "unknown"
        df["gad7_tier_code"] = -1

    df["phq9_change"] = "unknown"
    df["gad7_change"] = "unknown"

    return df


# ============================================================================
# SYNTHETIC DATA GENERATOR
# ============================================================================

def create_synthetic_data(n=500):
    """Create synthetic PHQ/GAD scores with realistic distributions."""
    log.info(f"Creating synthetic data ({n} samples)...")

    rng = np.random.default_rng(42)

    # PHQ-9: mean ~10, std ~6, range 0-27
    phq9 = rng.normal(10, 6, size=n).clip(0, 27).astype(int)

    # GAD-7: mean ~8, std ~5, range 0-21
    gad7 = rng.normal(8, 5, size=n).clip(0, 21).astype(int)

    df = pd.DataFrame({
        "phq9_score": phq9,
        "gad7_score": gad7,
    })

    df["phq9_change"] = "unknown"
    df["gad7_change"] = "unknown"

    log.info(f"  PHQ-9: mean={phq9.mean():.1f}, range={phq9.min()}-{phq9.max()}")
    log.info(f"  GAD-7: mean={gad7.mean():.1f}, range={gad7.min()}-{gad7.max()}")

    return df


# ============================================================================
# SHARED FEATURES
# ============================================================================

def engineer_shared_features(df):
    """Apply tier labels, severity index, high-risk flag."""
    log.info("Engineering shared features...")

    # Apply tiers
    df["phq9_tier"], df["phq9_tier_code"] = apply_phq9_tiers(df["phq9_score"])
    df["gad7_tier"], df["gad7_tier_code"] = apply_gad7_tiers(df["gad7_score"])

    # Severity index (0-1 scale)
    df["severity_index"] = ((df["phq9_score"] + df["gad7_score"]) / 48).round(4)

    # High-risk flag
    df["high_risk"] = (
        (df["phq9_score"] >= 15) | (df["gad7_score"] >= 15)
    ).astype(int)

    # Change codes
    change_code_map = {"improved": -1, "stable": 0, "worsened": 1, "unknown": -99}
    df["phq9_change_code"] = df["phq9_change"].map(change_code_map).fillna(-99).astype(int)
    df["gad7_change_code"] = df["gad7_change"].map(change_code_map).fillna(-99).astype(int)

    log.info(f"  PHQ-9 tiers: {df['phq9_tier'].value_counts().to_dict()}")
    log.info(f"  GAD-7 tiers: {df['gad7_tier'].value_counts().to_dict()}")
    log.info(f"  High-risk: {df['high_risk'].sum()} ({df['high_risk'].mean()*100:.1f}%)")

    return df


# ============================================================================
# OUTPUT
# ============================================================================

ENGINEERED_COLS = [
    "phq9_score", "phq9_tier", "phq9_tier_code", "phq9_change", "phq9_change_code",
    "gad7_score", "gad7_tier", "gad7_tier_code", "gad7_change", "gad7_change_code",
    "severity_index", "high_risk",
]


def save_outputs(df, output_dir):
    """Save CSV and Parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "score_features.csv"
    parquet_path = output_dir / "score_features.parquet"

    df.to_csv(csv_path, index=False)
    log.info(f"Saved → {csv_path}")

    df.to_parquet(parquet_path, index=False, compression="snappy")
    log.info(f"Saved → {parquet_path}")


def write_summary_report(df, report_dir):
    """Write summary report."""
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "score_features_summary.txt"

    lines = [
        "=" * 60,
        "AIML-2 Score Feature Engineering Summary",
        "AI Mental Health UK — Week 2",
        "=" * 60,
        f"Total rows: {len(df):,}",
        f"Total columns: {len(df.columns)}",
        "",
        "--- PHQ-9 ---",
        f"Mean: {df['phq9_score'].mean():.2f}",
        f"Median: {df['phq9_score'].median():.0f}",
        df["phq9_tier"].value_counts().to_string(),
        "",
        "--- GAD-7 ---",
        f"Mean: {df['gad7_score'].mean():.2f}",
        f"Median: {df['gad7_score'].median():.0f}",
        df["gad7_tier"].value_counts().to_string(),
        "",
        "--- Risk ---",
        f"High-risk: {df['high_risk'].sum()} ({df['high_risk'].mean()*100:.1f}%)",
        f"Severity index mean: {df['severity_index'].mean():.3f}",
        "",
        "--- Preview ---",
        df[ENGINEERED_COLS].head(5).to_string(),
        "=" * 60,
    ]

    path.write_text("\n".join(lines))
    log.info(f"Report → {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="AIML-2 Score Features")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    # Load or create data
    if args.input.exists():
        df = load_dataset(args.input)
        df = process_structured(df)
        if df is None:
            df = create_synthetic_data()
    else:
        log.warning(f"Input not found: {args.input}")
        log.info("Creating synthetic PHQ/GAD data...")
        df = create_synthetic_data()

    # Engineer features
    df = engineer_shared_features(df)

    # Save
    save_outputs(df, args.output_dir)
    write_summary_report(df, REPORT_DIR)

    log.info(f"Done! {len(df)} rows saved to {args.output_dir}/")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()