# Fraud Detection using XGBoost

## Overview

A machine learning project for detecting fraudulent credit card transactions using XGBoost. The project includes data preprocessing, handling class imbalance with SMOTE, model training, and evaluation on a highly imbalanced real-world dataset.

## Features

- Data preprocessing pipeline
- Stratified train-test split
- Feature scaling using StandardScaler
- Class imbalance handling using SMOTE
- Fraud detection using XGBoost
- Model evaluation using Precision, Recall, F1-Score, and ROC-AUC

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Imbalanced-Learn (SMOTE)
- Joblib
- FastAPI (In Progress)

## Dataset

Download the Credit Card Fraud Detection dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the dataset at:

```text
data/raw/creditcard.csv
```

## Project Structure

```text
fraud-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── EDA.ipynb
│
├── ml/
│   ├── config.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── backend/
│   └── model/
│
├── requirements.txt
├── .gitignore
└── README.md
```

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

## Installation

Clone the repository:

```bash
git clone https://github.com/Rohit-166/Fraud_detection.git
cd fraud-detection
```

Create and activate a Conda environment:

```bash
conda create -n fraud-detection python=3.11
conda activate fraud-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Preprocess Data

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

## API Endpoints

### Health Check

```http
GET /health
```

### Model Information

```http
GET /model-info
```

### Fraud Prediction

```http
POST /predict
```

Example Response:

```json
{
  "prediction": "Legitimate",
  "fraud_probability": 0.0002
}
```
## License

This project is intended for educational and portfolio purposes.

