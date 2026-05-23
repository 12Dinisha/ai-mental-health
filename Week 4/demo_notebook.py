#!/usr/bin/env python3
"""
NHS Risk Prediction Model — Live Demo
AIML-4: 5-minute walkthrough

Run: python demo_notebook.py
"""

import requests
from datetime import datetime
from dataclasses import dataclass
from typing import List


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{API_BASE_URL}/predict"
API_TIMEOUT = 30


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class PatientInput:
    patient_id: str
    age: int
    gender: str
    prior_admissions: int
    prior_ed_visits: int
    comorbidities: List[str]
    length_of_stay_days: int
    discharge_year: int
    ethnicity: str
    # Mental Health specific
    phq9_score: int = 0
    gad7_score: int = 0


@dataclass
class PredictionResult:
    patient_id: str
    risk_score: float
    classification: str
    confidence: str
    top_features: List[str]
    timestamp: str
    inference_ms: float


# ============================================================
# FEATURE ENGINEERING (14 features from training)
# ============================================================

def patient_to_features(patient: PatientInput) -> List[float]:
    """
    Convert patient to 14 features matching training data.
    Features: age_band_code, imd_proxy, high_deprivation, flags, 
              phq9/gad7 scores & tiers, severity, change codes
    """
    
    # 1. age_band_code (0-5): 0=18-24, 1=25-34, 2=35-44, 3=45-54, 4=55-64, 5=65+
    if patient.age < 25:
        age_band_code = 0
    elif patient.age < 35:
        age_band_code = 1
    elif patient.age < 45:
        age_band_code = 2
    elif patient.age < 55:
        age_band_code = 3
    elif patient.age < 65:
        age_band_code = 4
    else:
        age_band_code = 5
    
    # 2. imd_proxy (1-5) - simulate from prior admissions
    imd_proxy = min(max(patient.prior_admissions, 1), 5)
    
    # 3. high_deprivation (0/1)
    high_deprivation = 1 if imd_proxy <= 2 else 0
    
    # 4-6. Depression/anxiety flags based on PHQ-9 / GAD-7
    phq9 = patient.phq9_score if patient.phq9_score > 0 else 10
    gad7 = patient.gad7_score if patient.gad7_score > 0 else 8
    
    flag_depression = 1 if phq9 >= 10 else 0
    flag_anxiety = 1 if gad7 >= 8 else 0
    flag_comorbid = 1 if (flag_depression and flag_anxiety) else 0
    
    # 7. comorbidity_count (0-3)
    comorbidity_count = min(len(patient.comorbidities), 3)
    
    # 8. phq9_score (0-27)
    phq9_score = phq9
    
    # 9. gad7_score (0-21)
    gad7_score = gad7
    
    # 10. phq9_tier_code (0-4)
    if phq9 < 5:
        phq9_tier = 0
    elif phq9 < 10:
        phq9_tier = 1
    elif phq9 < 15:
        phq9_tier = 2
    elif phq9 < 20:
        phq9_tier = 3
    else:
        phq9_tier = 4
    
    # 11. gad7_tier_code (0-3)
    if gad7 < 5:
        gad7_tier = 0
    elif gad7 < 10:
        gad7_tier = 1
    elif gad7 < 15:
        gad7_tier = 2
    else:
        gad7_tier = 3
    
    # 12. severity_index
    severity_index = round((phq9 + gad7) / 48, 4)
    
    # 13-14. Change codes (0 for new patients, could be dynamic)
    phq9_change_code = 0
    gad7_change_code = 0
    
    # 14 features!
    features = [
        float(age_band_code),      # 1
        float(imd_proxy),       # 2
        float(high_deprivation), # 3
        float(flag_depression), # 4
        float(flag_anxiety),   # 5
        float(flag_comorbid), # 6
        float(comorbidity_count), # 7
        float(phq9_score),   # 8
        float(gad7_score),   # 9
        float(phq9_tier),   # 10
        float(gad7_tier),   # 11
        float(severity_index), # 12
        float(phq9_change_code), # 13
        float(gad7_change_code), # 14
    ]
    
    return features


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_risk(patient: PatientInput) -> PredictionResult:
    import time
    start_time = time.time()
    
    features = patient_to_features(patient)
    
    payload = {"features": features}
    response = requests.post(API_ENDPOINT, json=payload, timeout=API_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    probability = data.get("probability", 0.0)
    risk_score = probability * 10
    
    return PredictionResult(
        patient_id=patient.patient_id,
        risk_score=round(risk_score, 2),
        classification=data.get("confidence", "LOW"),
        confidence=str(data.get("confidence", "LOW")),
        top_features=_get_top_features(patient),
        timestamp=datetime.now().isoformat(),
        inference_ms=round(elapsed_ms, 1)
    )


def _get_top_features(patient: PatientInput) -> List[str]:
    features = []
    if patient.phq9_score >= 15 or (patient.phq9_score > 0 and patient.phq9_score >= 10):
        features.append("PHQ-9 score")
    if patient.gad7_score >= 10 or (patient.gad7_score > 0 and patient.gad7_score >= 8):
        features.append("GAD-7 score")
    if patient.prior_admissions >= 3:
        features.append("Prior admissions")
    if patient.age >= 65:
        features.append("Age")
    if patient.comorbidities:
        features.append("Comorbidities")
    return features[:3]


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def print_header(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_patient_case(patient: PatientInput) -> None:
    print(f"\n📋 Patient: {patient.patient_id}")
    print(f"   Age: {patient.age} | Gender: {patient.gender}")
    print(f"   PHQ-9: {patient.phq9_score} | GAD-7: {patient.gad7_score}")
    print(f"   Prior admissions: {patient.prior_admissions}")
    print(f"   Comorbidities: {', '.join(patient.comorbidities) or 'None'}")


def print_prediction(result: PredictionResult) -> None:
    risk_emoji = {
        "HIGH": "🔴",
        "MEDIUM": "🟡",
        "LOW": "🟢"
    }.get(result.classification.upper(), "⚪")
    
    print(f"\n  {risk_emoji} RISK SCORE: {result.risk_score:.2f} / 10.0")
    print(f"     Classification: {result.classification} risk")
    print(f"     Top factors: {', '.join(result.top_features)}")
    print(f"\n  ⚡ Inference time: {result.inference_ms:.1f}ms")


# ============================================================
# MAIN DEMO
# ============================================================

def main():
    patients = [
        # High risk: High PHQ-9, GAD-7
        PatientInput(
            patient_id="P001", age=67, gender="M", prior_admissions=3,
            prior_ed_visits=1, comorbidities=["depression"],
            length_of_stay_days=2, discharge_year=2024, ethnicity="White British",
            phq9_score=18, gad7_score=15  # High scores
        ),
        # Low risk: Low PHQ-9, GAD-7
        PatientInput(
            patient_id="P002", age=34, gender="F", prior_admissions=0,
            prior_ed_visits=0, comorbidities=[],
            length_of_stay_days=1, discharge_year=2024, ethnicity="Asian British",
            phq9_score=3, gad7_score=2  # Low scores
        ),
        # High risk: Comorbid depression + anxiety, high prior admissions
        PatientInput(
            patient_id="P003", age=52, gender="M", prior_admissions=5,
            prior_ed_visits=4, comorbidities=["depression", "anxiety"],
            length_of_stay_days=7, discharge_year=2024, ethnicity="Black British",
            phq9_score=20, gad7_score=18  # Very high scores
        ),
    ]
    
    print_header("NHS RISK PREDICTION MODEL — LIVE API DEMO")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for i, patient in enumerate(patients, 1):
        print_header(f"CASE {i}: {patient.patient_id}")
        print_patient_case(patient)
        result = predict_risk(patient)
        print_prediction(result)
    
    print_header("DEMO COMPLETE")
    print("\n✅ Demo finished successfully\n")


if __name__ == "__main__":
    main()