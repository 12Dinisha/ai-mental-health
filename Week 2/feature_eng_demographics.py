"""
AIML-1 — ONS Dataset Processing (Simple Version)
Process all ONS tables without filtering
"""

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# --- PATHS ---
ROOT = Path(__file__).parent.resolve()
DATA_RAW = ROOT / "data" / "raw"
DATA_OUT = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

for d in [DATA_RAW, DATA_OUT, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- FUNCTIONS ---

def clean_dataset(df):
    """Basic cleaning."""
    df.columns = df.columns.astype(str).str.strip().str.lower()
    df = df.dropna(how="all")
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df.copy()

def process_workbook_simple(path):
    """Process ALL sheets without filtering."""
    log.info(f"Processing → {path.name}")
    
    workbook = pd.ExcelFile(path, engine="openpyxl")
    results = []
    
    for sheet in workbook.sheet_names[:10]:  # Limit to first 10 sheets for speed
        try:
            df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
            
            if df.empty or df.shape[1] <= 1:
                continue
            
            df = clean_dataset(df)
            
            # Save CSV
            safe_name = sheet.replace(" ", "_").replace("/", "_")[:50]
            out_path = DATA_OUT / f"{path.stem}_{safe_name}.csv"
            df.to_csv(out_path, index=False)
            
            log.info(f"  Saved → {out_path.name} ({len(df)} rows)")
            results.append({"sheet": sheet, "rows": len(df)})
            
        except Exception as e:
            log.warning(f"  Failed → {sheet}: {e}")
    
    return {"file": path.name, "results": results}

# --- MAIN ---
def main():
    log.info("--- SIMPLE ONS PIPELINE ---")
    
    # Get Excel files
    excel_files = list(DATA_RAW.glob("*.xlsx"))
    
    if not excel_files:
        log.error("No Excel files in data/raw")
        return
    
    log.info(f"Found {len(excel_files)} workbooks")
    
    all_results = []
    for f in excel_files[:2]:  # Process first 2 files
        result = process_workbook_simple(f)
        all_results.append(result)
    
    # Summary
    log.info("=" * 50)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 50)
    
    # List output files
    csv_count = len(list(DATA_OUT.glob("*.csv")))
    log.info(f"Total CSV files created: {csv_count}")

if __name__ == "__main__":
    main()