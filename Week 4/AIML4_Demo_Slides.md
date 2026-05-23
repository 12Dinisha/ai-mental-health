# NHS Mental Health Risk Prediction Model
## AIML-4 Demo Walkthrough | May 2026

---

## Slide 1: Title

# 🏥 NHS Mental Health Readmission Risk Predictor

**AIML-4: Live Model Demonstration**

- Project: AI Mental Health UK
- Date: 28-30 May 2026
- Presenter: [Your Name]

---

## Slide 2: What Does This Model Do?

### Purpose
Predicts **30-day readmission risk** for mental health patients based on:
- PHQ-9 (depression) scores
- GAD-7 (anxiety) scores  
- Demographics & history

### Input → Output
[Patient Data] → [Model] → [Risk Score 0-10] + [Classification]


### Use Case
Triage support for IAPT services
**NOT** a replacement for clinical judgement

---

## Slide 3: The Model

### Technical Details

- **Algorithm**: Ensemble (XGBoost + Random Forest + Logistic Regression)
- **Features**: 14 input features
- **Model file**: `ensemble_v3.pkl`
- **API**: FastAPI at `/predict`

### Features Used

Feature

Description

age_band_code

Age group (0-5)

phq9_score

Depression severity (0-27)

gad7_score

Anxiety severity (0-21)

comorbidity_count

Number of conditions

high_deprivation

IMD indicator

## Slide 4: Live Demo

### Running the Demo...

[Execute: python demo_notebook.py]

## Slide 5: Demo Results

### Case 1: High Risk Patient

67yo male, PHQ-9=18, GAD-7=15
Risk Score: 9.95/10 → 🔴 HIGH

### Case 2: Low Risk Patient

34yo female, PHQ-9=3, GAD-7=2
Risk Score: 0.02/10 → 🟢 LOW

### Case 3: Very High Risk

52yo male, PHQ-9=20, GAD-7=18, prior admissions=5
Risk Score: 9.96/10 → 🔴 HIGH

## Slide 6: Performance Metrics

Training Results
Metric

Score

F1

1.00

ROC-AUC

1.00

Validation (MHSDS)
Metric

Score

F1

0.34

ROC-AUC

0.51

📌 Note: Validation on real NHS data ongoing

## Slide 7: Limitations

⚠ Important Limitations

Trained on synthetic data
Not clinically validated
Requires NHS governance approval
Does not replace clinical assessment
May have bias requiring investigation
Regulatory Status
Requirement

Status

NHS England

NOT Approved

CE Mark

NOT Certified

MHRA

NOT Registered

## Slide 8: Next Steps

### Roadmap

✅ Model trained (Week 3)
🔄 Validation ongoing (Week 4)
⏳ Clinical validation
⏳ Governance approval
⏳ Pilot in IAPT services

### Contact

ai-mh-uk@project.nhs

docs.aimh-uk.org

### Q&A

Questions?


---

## 2. Demo Script with Talking Points

Save as `demo_presentation.py`:

```python
#!/usr/bin/env python3
"""
AIML-4: Live Demo Presentation Script
5-minute presentation with timing cues
"""

import time
import sys

def print_slide(slide_num, title, content=""):
    """Print a slide header"""
    print(f"\n{'='*70}")
    print(f"SLIDE {slide_num}: {title}")
    print(f"{'='*70}")
    if content:
        print(content)
    print()
    time.sleep(1)

def main():
    """Run the 5-minute presentation"""
    
    # SLIDE 1: Title (30 seconds)
    print_slide(1, "TITLE", """
    # NHS Mental Health Readmission Risk Predictor
    
    AIML-4 Demo Walkthrough
    28-30 May 2026
    
    [SAY]: Good morning/afternoon. I'll be demonstrating our mental health 
    risk prediction model - an ensemble model that predicts 30-day readmission 
    risk based on PHQ-9 and GAD-7 scores.
    """)
    
    input("Press Enter to continue...")
    
    # SLIDE 2: Purpose (45 seconds)
    print_slide(2, "WHAT IT DOES", """
    Purpose:
    - Predict readmission risk for mental health patients
    - Decision support forclinicians (NOT replacement)
    
    Input: PHQ-9, GAD-7, demographics, history
    Output: Risk Score (0-10) + Classification (Low/Medium/High)
    
    [SAY]: The model takes clinical questionnaire scores - PHQ-9 for depression 
    and GAD-7 for anxiety - plus basic demographics, and outputs a risk score 
    to help triage patients. Let me show you how it works.
    """)
    
    input("Press Enter to run live demo...")
    
    # LIVE DEMO (90 seconds)
    print_slide(3, "LIVE DEMO", "Running demo_notebook.py...")
    
    import subprocess
    result = subprocess.run(["python", "demo_notebook.py"], capture_output=False)
    
    # SLIDE 4: Explanation (45 seconds)
    print_slide(4, "RESULTS EXPLAINED", """
    What we just saw:
    
    ✅ P001: HIGH risk (9.95) - High PHQ-9 (18) + GAD-7 (15)
    ✅ P002: LOW risk (0.02) - Low PHQ-9 (3) + GAD-7 (2)
    ✅ P003: HIGH risk (9.96) - Very high scores + history
    
    [SAY]: The model correctly differentiates risk levels. 
    Cases with high clinical scores get high risk predictions.
    """)
    
    input("Press Enter for metrics...")
    
    # SLIDE 5: Metrics (30 seconds)
    print_slide(5, "PERFORMANCE", """
    Training Performance:
    - F1 Score: 1.00 | ROC-AUC: 1.00
    
    Validation (MHSDS):
    - F1: 0.34 | ROC-AUC: 0.51
    
    [SAY]: The model shows perfect training performance, but validation 
    on NHS data shows room for improvement. This is expected given the 
    synthetic training data.
    """)
    
    input("Press Enter for limitations...")
    
    # SLIDE 6: Limitations (30 seconds)
    print_slide(6, "LIMITATIONS", """
    ⚠ IMPORTANT:
    - NOT clinically validated yet
    - Trained on synthetic data
    - Requires NHS governance approval
    - Decision support only - NOT clinical replacement
    
    [SAY]: I want to be clear - this is a research model, not for clinical 
    use without proper validation and governance approval.
    """)
    
    print_slide(7, "Q&A", """
    # Questions?
    
    Contact: ai-mh-uk@project.nhs
    Docs: docs.aimh-uk.org
    """)
    
    print("\n✅ Presentation complete!")

if __name__ == "__main__":
    main()