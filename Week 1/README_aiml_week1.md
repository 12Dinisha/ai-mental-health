# AI/ML Week 1 — APMS 2014 Pipeline
## AI Mental Health UK Project · May 2026

### Scripts (run in order)

| # | File | Intern | Task | Deliverable |
|---|------|--------|------|-------------|
| 1 | `aiml_w1_01_ingestion.py` | AIML-1 | Env setup & data ingestion | Ingestion notebook + null/schema report |
| 2 | `aiml_w1_02_cleaning.py` | AIML-2 | Data cleaning & normalisation | `apms_2014_cleaned.csv` |
| 3 | `aiml_w1_03_feature_exploration.py` | AIML-3 | Feature exploration & correlation | Heatmap PNG + feature report |
| 4 | `aiml_w1_04_baseline_model.py` | AIML-4 | Baseline LR model | Model pkl + ROC/CM plots + report |

---

### Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy requests
```

### Run pipeline

```bash
python aiml_w1_01_ingestion.py       # downloads/generates APMS data
python aiml_w1_02_cleaning.py        # cleans & normalises
python aiml_w1_03_feature_exploration.py   # produces correlation heatmap
python aiml_w1_04_baseline_model.py  # trains LR, saves model + report
```

### Output structure

```
data/
  raw/         apms_2014.csv
  processed/   apms_2014_raw_loaded.csv
               apms_2014_cleaned.csv
models/
               baseline_lr.pkl
reports/
               ingestion_report_<ts>.txt
               null_summary.csv
               correlation_heatmap.png
               feature_target_bars.png
               feature_report.txt
               baseline_model_report.txt
               roc_curve.png
               confusion_matrix.png
```

### Dataset
- **APMS 2014** (Adult Psychiatric Morbidity Survey, NHS Digital)
- Public dataset, no patient identifiers
- URL: https://digital.nhs.uk/data-and-information/publications/statistical/adult-psychiatric-morbidity-survey

> **Note:** If the APMS CSV URL is not directly accessible, download manually from the NHS Digital link above and place at `data/raw/apms_2014.csv`. Script 01 will skip download if the file already exists.

### Week 2 next steps
- AIML-1/2: Feature engineering on CSEW + PHQ/GAD datasets
- AIML-3: XGBoost model training
- AIML-4: Ensemble model building
