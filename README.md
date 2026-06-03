# Fraud Detection using XGBoost & FastAPI

## Overview

A machine learning project for detecting fraudulent credit card transactions using XGBoost. The project includes data preprocessing, handling class imbalance using SMOTE, model training, evaluation, explainability using SHAP, and deployment-ready REST APIs built with FastAPI.

## Features

- Exploratory Data Analysis (EDA)
- Data preprocessing pipeline
- Stratified train-test split
- Feature scaling using StandardScaler
- Class imbalance handling using SMOTE
- Fraud detection using XGBoost
- Model evaluation using Precision, Recall, F1-Score, and ROC-AUC
- SHAP-based model explainability
- FastAPI inference service
- Batch prediction API
- Explainability API

---

## Tech Stack

### Machine Learning
- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Imbalanced-Learn (SMOTE)
- SHAP

### Backend
- FastAPI
- Pydantic
- Uvicorn

### Utilities
- Joblib
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Dataset

Credit Card Fraud Detection Dataset:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the dataset at:

```text
data/raw/creditcard.csv
```

---

## Project Structure

```text
fraud-detection/
│
├── backend/
│   ├── app.py
│   ├── schemas.py
│   └── model/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── ml/
│   ├── config.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── feature_importance.py
│   └── shap_analysis.py
│
├── notebooks/
│   └── EDA.ipynb
│
├── results/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Model Performance

Evaluation on the test dataset:

| Metric | Score |
|----------|----------|
| Precision | 0.74 |
| Recall | 0.86 |
| F1-Score | 0.79 |
| ROC-AUC | 0.983 |

### Confusion Matrix

```text
[[56834    30]
 [   14    84]]
```

### Results Summary

- Correctly detected fraudulent transactions: **84**
- Missed fraudulent transactions: **14**
- Legitimate transactions incorrectly flagged as fraud: **30**
- Correctly classified legitimate transactions: **56,834**

---

## Machine Learning Pipeline

```text
Raw Dataset
      ↓
Preprocessing
      ↓
Train/Test Split
      ↓
Feature Scaling
      ↓
SMOTE
      ↓
XGBoost Training
      ↓
Evaluation
      ↓
SHAP Explainability
      ↓
FastAPI Inference Service
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

---

### Model Information

```http
GET /model-info
```

Response:

```json
{
  "model": "XGBoost",
  "roc_auc": 0.983,
  "precision": 0.74,
  "recall": 0.86
}
```

---

### Single Prediction

```http
POST /predict
```

Response:

```json
{
  "prediction": "Legitimate",
  "fraud_probability": 0.0002
}
```

---

### Batch Prediction

```http
POST /predict-batch
```

Response:

```json
{
  "predictions": [
    "Legitimate",
    "Fraud"
  ],
  "fraud_probabilities": [
    0.0002,
    0.9831
  ]
}
```

---

### Explain Prediction

```http
POST /explain
```

Response:

```json
{
  "prediction": "Fraud",
  "fraud_probability": 0.9831,
  "top_features": [
    {
      "feature": "V14",
      "impact": -2.314
    },
    {
      "feature": "V17",
      "impact": 1.876
    },
    {
      "feature": "V10",
      "impact": -1.223
    }
  ]
}
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Rohit-166/Fraud_detection.git
cd fraud-detection
```

Create a conda environment:

```bash
conda create -n fraud-detection python=3.11
conda activate fraud-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### 1. Preprocess Dataset

```bash
python ml/preprocess.py
```

### 2. Train Model

```bash
python ml/train.py
```

### 3. Evaluate Model

```bash
python ml/evaluate.py
```

### 4. Run Feature Importance Analysis

```bash
python ml/feature_importance.py
```

### 5. Run SHAP Analysis

```bash
python ml/shap_analysis.py
```

### 6. Start FastAPI Server

```bash
uvicorn backend.app:app --reload
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Explainability

The project uses SHAP (SHapley Additive exPlanations) to interpret model predictions.

Features:
- Global feature importance analysis
- Local prediction explanations
- Top contributing features for each transaction
- Explainability endpoint exposed through FastAPI

