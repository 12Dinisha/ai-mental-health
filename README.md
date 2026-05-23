# AI Mental Health UK - Risk Prediction Model

Mental Health Readmission Risk Prediction for NHS Services

## Project Overview

- **Purpose**: Predict 30-day readmission risk for mental health patients
- **Model**: Ensemble (XGBoost + Random Forest + Logistic Regression)
- **Target**: F1 > 0.80
- **Team**: Dinisha Jain, Mohit

## Project Structure

- api/ - FastAPI inference server
- data/ - Data files
- models/ - Trained models
- reports/ - Analysis and reports
- Week 1/ - Data Collection and EDA
- Week 2/ - Baseline Models
- Week 3/ - Ensemble and Tuning
- Week 4/ - Validation and Demo
- full_presentation.html - Presentation

## Quick Start

Run API:
uvicorn api.inference:app --reload --port 8000

Run Demo:
python Week 4/demo_notebook.py


Copy code

## License

NHS Internal Use Only

May 2026
