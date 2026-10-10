# Churn Risk Advisor

A machine-learning web app that estimates customer churn risk and helps a telecom retention team prioritize follow-up.

## Features

- Predicts churn probability for one customer.
- Assigns LOW, WATCH, or HIGH risk bands.
- Compares alternative customer profiles using what-if analysis.
- Scores a batch of customers from a CSV file.
- Allows users to download scored results.

## Model Card: Churn Risk Advisor v1.0

**Intended use:** Rank existing telecom customers by churn risk so a retention team can prioritize calls. This is decision support, not automated decision-making.

**Not for:** Credit decisions, pricing decisions, or denying a customer a service.

**Dataset:** IBM Telco Customer Churn, 7,043 customers, approximately 26.5% churn.

**Model:** XGBoost (`XGBClassifier`), selected using five-fold cross-validation. The model uses 30 encoded feature columns.

**Performance:**
- Five-fold cross-validation AUC: 0.8502 ± 0.0117
- Test AUC: 0.8483
- Test recall: 0.521
- Test precision: 0.659

**Decision Threshold:** 0.30, selected to prioritize identifying customers at risk of churn. At this threshold, the model flags 544 customers, with 51.8% precision and 75.4% recall. The lower threshold helps identify more potential churners, although it also flags more customers who may not churn.

**Limitations:** The model uses one US telecom dataset from one period. It has not been validated on Pakistani telecom customers. Model associations do not prove that a particular intervention will prevent churn. Performance may change as customer behavior and service plans change.

**Fairness:** Further evaluation should compare recall across gender and senior-citizen groups.

**Owner:** M.Abuzar Shoukat  
**Version:** 1.0  
**Date:** 11 October 2026

## How to Run Locally

1. Install Python and open this project folder in VS Code.
2. Create and activate a virtual environment.
3. Install dependencies:

   `python -m pip install -r requirements.txt`

4. Start the app:

   `python -m streamlit run app.py`

## Files

- `app.py` — Streamlit application
- `churn_model.json` — trained XGBoost model
- `model_meta.json` — model metadata
- `sample_customers.csv` — sample customers for batch scoring
- `requirements.txt` — Python dependencies

## Live App

https://churn-risk-advisor-vbxmq6uvufhuw27njonzbr.streamlit.app/

## App Screenshots

### Single Customer Result
C:\Users\Abuzar Bhatti\Desktop\churn-risk-advisor\screenshot-one-customer.png.jpg

### Batch Scoring Results
C:\Users\Abuzar Bhatti\Desktop\churn-risk-advisor\screenshot-batch.png.jpg

## Earlier Project Work

Week 1 = https://www.kaggle.com/code/abuzarbhatti068/week-1-customer-churn-eda

Week 2 = https://www.kaggle.com/code/abuzarbhatti068/week-2-building-ml-models

Weel 3 = https://lnkd.in/dwBG4K9P
