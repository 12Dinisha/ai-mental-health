"""
AIML-2 | Week 1 | AI Mental Health UK Project
Task   : Data cleaning & normalisation — APMS 2014
Dates  : 2–4 May 2026
Output : data/processed/apms_2014_cleaned.csv
Depends: aiml_w1_01_ingestion.py (run first to produce apms_2014_raw_loaded.csv)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT     = Path(__file__).parent.resolve()
DATA_IN  = ROOT / "data" / "processed" / "apms_2014_raw_loaded.csv"
DATA_OUT = ROOT / "data" / "processed" / "apms_2014_cleaned.csv"
DATA_OUT.parent.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

# Columns with known coded missing values in NHS survey files
CODED_MISSING = {
    -9: np.nan,   # "Not applicable"
    -8: np.nan,   # "Don't know"
    -1: np.nan,   # "Refused / not answered"
    99: np.nan,   # Common survey sentinel
}

# Numeric columns to normalise (MinMax to [0,1])
MINMAX_COLS = ["age", "weight"]

# Numeric columns to standardise (Z-score for ML)
ZSCORE_COLS: list[str] = []   # extend if PHQ/GAD scale columns are present

# Categorical columns to label-encode
CATEGORICAL_COLS = ["sex", "ethnicity", "imd_quintile", "region"]

# Binary outcome columns (should only contain 0/1 after cleaning)
BINARY_COLS = ["any_cmd", "depression", "anxiety_gad", "phobia", "ocd", "cpd"]

# Imputation strategy per column type
IMPUTE_NUMERIC_STRATEGY   = "median"   # robust to skew in health data
IMPUTE_CATEGORICAL_STRATEGY = "mode"


# ─────────────────────────────────────────────
# Step 1 — Load
# ─────────────────────────────────────────────
def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    log.info("Loaded %d rows × %d cols from %s", *df.shape, path.name)
    return df


# ─────────────────────────────────────────────
# Step 2 — Replace coded missing values
# ─────────────────────────────────────────────
def replace_coded_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace survey-specific sentinel values with NaN."""
    total_replaced = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        replaced = df[col].isin(CODED_MISSING.keys()).sum()
        if replaced:
            df[col] = df[col].replace(CODED_MISSING)
            total_replaced += replaced
    log.info("Replaced %d coded-missing values with NaN.", total_replaced)
    return df


# ─────────────────────────────────────────────
# Step 3 — Drop duplicate rows
# ─────────────────────────────────────────────
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    log.info("Dropped %d duplicate rows (%d remain).", removed, len(df))
    return df


# ─────────────────────────────────────────────
# Step 4 — Impute missing values
# ─────────────────────────────────────────────
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
      - Numeric  → median (per column)
      - Categorical → mode (per column)
      - Binary outcome columns → mode (since true NaN means unanswered, not 0)
    """
    numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols   = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in numeric_cols:
        n_missing = df[col].isnull().sum()
        if n_missing:
            fill_val = df[col].median() if IMPUTE_NUMERIC_STRATEGY == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_val)
            log.info("Imputed %-20s  %d nulls → median %.4f", col, n_missing, fill_val)

    for col in object_cols:
        n_missing = df[col].isnull().sum()
        if n_missing:
            fill_val = df[col].mode()[0]
            df[col] = df[col].fillna(fill_val)
            log.info("Imputed %-20s  %d nulls → mode '%s'", col, n_missing, fill_val)

    remaining = df.isnull().sum().sum()
    log.info("Post-imputation null count: %d", remaining)
    return df


# ─────────────────────────────────────────────
# Step 5 — Encode categoricals
# ─────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.
    Returns the modified DataFrame and a mapping dict for reference.
    """
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            log.warning("Column '%s' not found — skipping encoding.", col)
            continue
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        log.info("Label-encoded '%s' → %d classes: %s", col, len(le.classes_), list(le.classes_[:5]))

    return df, {col: list(le.classes_) for col, le in encoders.items()}


# ─────────────────────────────────────────────
# Step 6 — Normalise numeric fields
# ─────────────────────────────────────────────
def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    MinMax scale continuous fields to [0, 1].
    Z-score standardise any scale-score columns.
    """
    mm = MinMaxScaler()
    for col in MINMAX_COLS:
        if col in df.columns:
            df[col] = mm.fit_transform(df[[col]])
            log.info("MinMax normalised '%s' → [0.0, 1.0]", col)

    if ZSCORE_COLS:
        ss = StandardScaler()
        present = [c for c in ZSCORE_COLS if c in df.columns]
        df[present] = ss.fit_transform(df[present])
        log.info("Z-score standardised: %s", present)

    return df


# ─────────────────────────────────────────────
# Step 7 — Validate binary outcome columns
# ─────────────────────────────────────────────
def validate_binary(df: pd.DataFrame) -> None:
    """Assert binary outcome columns contain only 0 and 1."""
    issues = []
    for col in BINARY_COLS:
        if col not in df.columns:
            continue
        unique_vals = set(df[col].dropna().unique())
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            issues.append(f"{col}: unexpected values {unique_vals - {0,1,0.0,1.0}}")

    if issues:
        log.warning("Binary column issues found:\n  %s", "\n  ".join(issues))
    else:
        log.info("✓ All binary outcome columns validated (0/1 only).")


# ─────────────────────────────────────────────
# Step 8 — Save
# ─────────────────────────────────────────────
def save(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    log.info("Cleaned dataset saved → %s  (%d rows × %d cols)", path, *df.shape)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main() -> pd.DataFrame:
    log.info("─── AIML-2: APMS 2014 Cleaning & Normalisation ───")

    df = load(DATA_IN)

    # Track shape at each stage
    def _log_shape(label: str):
        log.info("Shape after %-30s: %d × %d", label, *df.shape)

    df = replace_coded_missing(df);  _log_shape("replace coded missing")
    df = drop_duplicates(df);        _log_shape("drop duplicates")
    df = impute_missing(df);         _log_shape("impute missing")
    df, enc_map = encode_categoricals(df); _log_shape("encode categoricals")
    df = normalise(df);              _log_shape("normalise")

    validate_binary(df)

    save(df, DATA_OUT)

    # Print a final preview
    log.info("\nCleaned dataset preview:\n%s", df.head(5).to_string())
    log.info("─── Cleaning complete ───")

    return df


if __name__ == "__main__":
    main()
