# AI Mental Health UK - Risk Prediction Model

Mental Health Readmission Risk Prediction for NHS Services

## Project Overview

- **Purpose**: Predict 30-day readmission risk for mental health patients
- **Model**: Ensemble (XGBoost + Random Forest + Logistic Regression)
- **Target**: F1 > 0.80
- **Team**: Dinisha Jain, Mohit

## Project Structure
├── api/ # FastAPI inference server ├── data/ # Data files ├── models/ # Trained 
models ├── reports/ # Analysis & reports ├── Week 1/ # Data Collection & EDA ├── 
Week 2/ # Baseline Models ├── Week 3/ # Ensemble + Tuning ├── Week 4/ 
# Validation + Demo └── full_presentation.html # Presentation


## Quick Start

### Run API
```bash
uvicorn api.inference:app --reload --port 8000



### Run Demo

python Week 4/demo_notebook.py
Weeks Summary
Week

Content

1

Data Collection & EDA

2

Baseline Models

3

Ensemble + Optuna Tuning

4

Validation + Live Demo

License
NHS Internal Use Only

May 2026 EOF


Copy code

---

## Commit the README

```bash
git add README.md
git commit -m "Add README"
git push
