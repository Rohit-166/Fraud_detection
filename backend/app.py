from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import joblib
import pandas as pd

from backend.schemas import (
    Transaction,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse
)

app = FastAPI(
    title="Fraud Detection API",
    description="Credit Card Fraud Detection using XGBoost",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(
    "backend/model/fraud_model.pkl"
)

scaler = joblib.load(
    "backend/model/scaler.pkl"
)


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API Running"
    }


@app.get(
    "/health",
    response_model=HealthResponse
)
def health():
    return {
        "status": "healthy"
    }


@app.get(
    "/model-info",
    response_model=ModelInfoResponse
)
def model_info():
    return {
        "model": "XGBoost",
        "roc_auc": 0.983,
        "precision": 0.74,
        "recall": 0.86
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(transaction: Transaction):

    data = transaction.model_dump()

    df = pd.DataFrame([data])

    df["Amount"] = scaler.transform(
        df[["Amount"]]
    )

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return {
        "prediction":
            "Fraud"
            if prediction == 1
            else "Legitimate",

        "fraud_probability":
            round(float(probability), 4)
    }