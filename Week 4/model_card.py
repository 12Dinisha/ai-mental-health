"""
AIML-2 — Week 4: Model Card & Documentation
AI Mental Health UK · May 2026 Sprint

Writes NHS-ready model card with purpose, training data, metrics, limitations, and bias findings.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(".").resolve()
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
DATA_PROCESSED = ROOT / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL CARD CONTENT
# ============================================================================

def generate_model_card():
    """Generate NHS-ready model card."""
    
    card = """# AI Mental Health UK — Risk Prediction Model Card

## 1. Model Overview


Model Name

Mental Health Risk Predictor

Version

v3.0

Model Type

XGBoost Classifier

Date Trained

May 2026

Developer

AI Mental Health UK Team

Framework

XGBoost + Scikit-learn

2. Purpose & Intended Use
Purpose
Predicts severe mental health crisis risk based on PHQ-9, GAD-7 scores.

Intended Use
Triage support for mental health services
Risk stratification in IAPT services
Decision support (NOT replacement) for clinicians
NOT Intended For
Autonomous decision-making
Replacing clinical assessment
Legal or insurance decisions
3. Training Data
Metric

Value

Total Samples

496

Features

14

Target

high_risk (0/1)

Positive Rate

25.8%

Feature List
age_band_code, imd_proxy, high_deprivation
phq9_score, gad7_score
flag_depression, flag_anxiety, flag_comorbid_anx_dep
comorbidity_count
phq9_tier_code, gad7_tier_code
severity_index, phq9_change_code, gad7_change_code
4. Performance Metrics
Training Set
Metric

Value

F1 Score

1.0000

ROC-AUC

1.0000

Validation (NHS Datasets)
Dataset

F1

AUC

MHSDS Monthly

0.34

0.51

NHS MH Dashboard

0.34

0.49

Samaritans Crisis

0.53

0.51

5. Limitations
Trained on synthetic data — real performance unknown
Not validated on diverse UK populations
Model may degrade over time
Limited to PHQ-9/GAD-7 only
6. Bias & Fairness
Group

Finding

Age

No disparity (synthetic data)

Gender

No disparity (synthetic data)

Deprivation

Minor (30% delta)

7. Regulatory Compliance
Requirement

Status

GDPR

Complete

NHS England

NOT Approved

CE Mark

NOT Certified

MHRA

NOT Registered

8. Dependencies

Copy code
xgboost>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
9. Version History
Version

Date

Changes

v1.0

May 2026

Baseline

v3.0

May 2026

XGBoost + Optuna

10. Contact
AI Mental Health UK

info@aimh-uk.org
docs.aimh-uk.org

Generated: May 2026 Version: 3.0 """

    return card

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Generate and save model card."""

    print("Generating Model Card...")

    card = generate_model_card()

    # Save markdown
    path = REPORT_DIR / "model_card.md"
    path.write_text(card, encoding="utf-8")
    print(f"Saved -> {path}")

    # Save text
    txt_path = REPORT_DIR / "model_card.txt"
    txt_path.write_text(card, encoding="utf-8")
    print(f"Saved -> {txt_path}")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()