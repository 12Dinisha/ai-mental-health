"""
AIML-1 | Week 1 | AI Mental Health UK Project
Task   : Env setup & data ingestion — APMS 2014 (NHS Digital)
Dates  : 1–2 May 2026
Output : Ingestion notebook / script + schema validation report
Dataset: Adult Psychiatric Morbidity Survey 2014
         https://digital.nhs.uk/data-and-information/publications/statistical/
                 adult-psychiatric-morbidity-survey
"""

import os
import hashlib
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import requests

# ─────────────────────────────────────────────
# 0.  Logging setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1.  Paths & config
# ─────────────────────────────────────────────
ROOT       = Path(__file__).parent.resolve()
DATA_RAW   = ROOT / "data" / "raw"
DATA_OUT   = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

for d in [DATA_RAW, DATA_OUT, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# APMS 2014 public download (CSV version via UK Data Service / NHS Digital)
# Replace with actual URL once access is confirmed.
APMS_URL      = "https://digital.nhs.uk/data-and-information/publications/statistical/adult-psychiatric-morbidity-survey"
APMS_FILENAME = "apms_2014.csv"
APMS_PATH     = DATA_RAW / APMS_FILENAME

# ─────────────────────────────────────────────
# 2.  Download helper
# ─────────────────────────────────────────────
def download_dataset(url: str, dest: Path) -> None:
    """Download dataset file with progress logging. Skips if already present."""
    if dest.exists():
        log.info("Dataset already exists at %s — skipping download.", dest)
        return

    log.info("Downloading dataset from %s …", url)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)

    log.info("Saved %s (%.1f KB).", dest.name, downloaded / 1024)


def file_md5(path: Path) -> str:
    """Return MD5 hex-digest for a file (integrity check)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────
# 3.  Load dataset
# ─────────────────────────────────────────────
def load_apms(path: Path) -> pd.DataFrame:
    """
    Load APMS 2014 CSV into a DataFrame.
    Handles common encoding issues in NHS-published files.
    """
    log.info("Loading dataset from %s …", path)
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        log.warning("UTF-8 failed — retrying with latin-1.")
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    log.info("Loaded %d rows × %d columns.", *df.shape)
    return df


# ─────────────────────────────────────────────
# 4.  Schema validation
# ─────────────────────────────────────────────
# Expected columns based on APMS 2014 data dictionary.
# Adjust to the actual column names in the downloaded file.
EXPECTED_COLUMNS = {
    # Demographics
    "age",          # respondent age
    "sex",          # 1=Male, 2=Female
    "ethnicity",    # ethnic group code
    "imd_quintile", # index of multiple deprivation
    "region",       # NHS region code
    # Mental health disorder flags (ICD-10 criteria)
    "any_cmd",      # common mental disorder (1=yes)
    "depression",
    "anxiety_gad",
    "phobia",
    "ocd",
    "cpd",          # complex PTSD / PTSD flag
    # Survey weights
    "weight",
}

def validate_schema(df: pd.DataFrame) -> dict:
    """
    Check that expected columns are present and report coverage.
    Returns a dict of validation results.
    """
    present  = set(df.columns.str.lower())
    missing  = EXPECTED_COLUMNS - present
    extra    = present - EXPECTED_COLUMNS

    results = {
        "total_columns"   : len(df.columns),
        "expected_present": len(EXPECTED_COLUMNS - missing),
        "missing_columns" : sorted(missing),
        "extra_columns"   : sorted(extra),
        "schema_ok"       : len(missing) == 0,
    }

    if results["schema_ok"]:
        log.info("✓ Schema validation passed — all expected columns present.")
    else:
        log.warning("✗ Schema validation: %d expected columns missing: %s",
                    len(missing), missing)
    return results


# ─────────────────────────────────────────────
# 5.  Null / missing-value checks
# ─────────────────────────────────────────────
NULL_RATE_THRESHOLD = 0.30  # flag columns with >30% nulls

def null_check(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising null counts and rates per column."""
    null_counts = df.isnull().sum()
    null_rates  = null_counts / len(df)

    summary = pd.DataFrame({
        "column"    : df.columns,
        "dtype"     : df.dtypes.values,
        "null_count": null_counts.values,
        "null_rate" : null_rates.values,
        "flagged"   : null_rates.values > NULL_RATE_THRESHOLD,
    }).sort_values("null_rate", ascending=False).reset_index(drop=True)

    flagged_n = summary["flagged"].sum()
    log.info("Null check complete. %d / %d columns flagged (>%.0f%% missing).",
             flagged_n, len(df.columns), NULL_RATE_THRESHOLD * 100)
    return summary


# ─────────────────────────────────────────────
# 6.  Row-level sanity checks
# ─────────────────────────────────────────────
def row_checks(df: pd.DataFrame) -> dict:
    """Basic row-level integrity checks."""
    results = {}

    # Duplicate rows
    dup_count = df.duplicated().sum()
    results["duplicate_rows"] = int(dup_count)
    if dup_count:
        log.warning("%d duplicate rows detected.", dup_count)
    else:
        log.info("✓ No duplicate rows.")

    # Age range (APMS covers 16–74)
    if "age" in df.columns:
        out_of_range = df[(df["age"] < 16) | (df["age"] > 74)].shape[0]
        results["age_out_of_range"] = int(out_of_range)
        if out_of_range:
            log.warning("Age out of expected range (16–74): %d rows.", out_of_range)
        else:
            log.info("✓ All age values within expected range 16–74.")

    # Survey weight non-positive
    if "weight" in df.columns:
        bad_weights = df[df["weight"] <= 0].shape[0]
        results["invalid_weights"] = int(bad_weights)
        if bad_weights:
            log.warning("%d rows with non-positive survey weight.", bad_weights)

    return results


# ─────────────────────────────────────────────
# 7.  Quick descriptive summary
# ─────────────────────────────────────────────
def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return describe() for numeric columns, plus unique counts for categoricals."""
    numeric_desc = df.describe(include=[np.number]).T
    numeric_desc.index.name = "column"
    return numeric_desc


# ─────────────────────────────────────────────
# 8.  Save outputs
# ─────────────────────────────────────────────
def save_ingestion_report(
    schema_results : dict,
    null_summary   : pd.DataFrame,
    row_results    : dict,
    df             : pd.DataFrame,
    md5            : str,
) -> Path:
    """Write a plain-text ingestion report."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"ingestion_report_{ts}.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("APMS 2014 - Ingestion Report\n")
        f.write(f"Generated : {datetime.now()}\n")
        f.write(f"File MD5   : {md5}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Rows    : {len(df):,}\n")
        f.write(f"Columns : {len(df.columns)}\n\n")

        f.write("-- Schema Validation --\n")
        f.write(f"  Schema OK      : {schema_results['schema_ok']}\n")
        f.write(f"  Missing cols   : {schema_results['missing_columns']}\n\n")

        f.write("-- Null Rates (top 10 worst) --\n")
        top_nulls = null_summary.head(10)[["column", "null_rate", "flagged"]]
        f.write(top_nulls.to_string(index=False) + "\n\n")

        f.write("-- Row-level Checks --\n")
        for k, v in row_results.items():
            f.write(f"  {k:<25}: {v}\n")

    log.info("Ingestion report saved → %s", path)
    return path


# ─────────────────────────────────────────────
# 9.  Main pipeline
# ─────────────────────────────────────────────
def main():
    log.info("─── AIML-1: APMS 2014 Ingestion Pipeline ───")

    # 9a. Download (skip if present)
    # Uncomment when the direct CSV download URL is confirmed:
    # download_dataset(APMS_URL, APMS_PATH)

    # For dev / offline testing, generate a synthetic APMS-like dataset
    if not APMS_PATH.exists():
        log.warning("APMS file not found at %s — generating synthetic sample.", APMS_PATH)
        _generate_synthetic_apms(APMS_PATH, n=5000)

    md5 = file_md5(APMS_PATH)
    log.info("File MD5: %s", md5)

    # 9b. Load
    df = load_apms(APMS_PATH)

    # 9c. Validate schema
    schema_results = validate_schema(df)

    # 9d. Null checks
    null_summary = null_check(df)

    # 9e. Row checks
    row_results = row_checks(df)

    # 9f. Describe
    desc = describe_dataset(df)
    log.info("\n%s", desc.to_string())

    # 9g. Save outputs
    out_csv  = DATA_OUT / "apms_2014_raw_loaded.csv"
    null_csv = REPORT_DIR / "null_summary.csv"
    df.to_csv(out_csv, index=False)
    null_summary.to_csv(null_csv, index=False)
    log.info("Raw dataset saved → %s", out_csv)
    log.info("Null summary saved → %s", null_csv)

    report_path = save_ingestion_report(schema_results, null_summary, row_results, df, md5)
    log.info("─── Ingestion complete. Report → %s ───", report_path)


# ─────────────────────────────────────────────
# 10. Synthetic data helper (offline / CI use)
# ─────────────────────────────────────────────
def _generate_synthetic_apms(path: Path, n: int = 5000) -> None:
    """
    Generate a synthetic APMS-like CSV for development and testing.
    Distributions are loosely based on published APMS 2014 prevalence figures.
    NOT for clinical use.
    """
    rng = np.random.default_rng(42)

    age        = rng.integers(16, 75, size=n)
    sex        = rng.choice([1, 2], size=n, p=[0.48, 0.52])
    ethnicity  = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.85, 0.04, 0.04, 0.04, 0.03])
    imd        = rng.integers(1, 6, size=n)
    region     = rng.choice(["E12000001","E12000002","E12000003","E12000004",
                              "E12000005","E12000006","E12000007","E12000008","E12000009"], size=n)

    # Disorder prevalence roughly matching APMS 2014 national estimates
    any_cmd    = rng.choice([0, 1], size=n, p=[0.83, 0.17])
    depression = rng.choice([0, 1], size=n, p=[0.921, 0.079])
    anxiety    = rng.choice([0, 1], size=n, p=[0.924, 0.076])
    phobia     = rng.choice([0, 1], size=n, p=[0.974, 0.026])
    ocd        = rng.choice([0, 1], size=n, p=[0.976, 0.024])
    cpd        = rng.choice([0, 1], size=n, p=[0.967, 0.033])
    weight     = rng.uniform(0.5, 3.0, size=n).round(4)

    # Inject ~5% nulls randomly across some columns
    for arr in [depression, anxiety, phobia]:
        idx = rng.choice(n, size=int(n * 0.05), replace=False)
        arr = arr.astype(float)
        arr[idx] = np.nan

    df = pd.DataFrame({
        "age"        : age,
        "sex"        : sex,
        "ethnicity"  : ethnicity,
        "imd_quintile": imd,
        "region"     : region,
        "any_cmd"    : any_cmd,
        "depression" : depression,
        "anxiety_gad": anxiety,
        "phobia"     : phobia,
        "ocd"        : ocd,
        "cpd"        : cpd,
        "weight"     : weight,
    })

    df.to_csv(path, index=False)
    log.info("Synthetic APMS dataset (%d rows) saved → %s", n, path)


if __name__ == "__main__":
    main()