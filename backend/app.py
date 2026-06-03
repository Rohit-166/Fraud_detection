from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import joblib
import pandas as pd

import shap

from backend.schemas import (
    Transaction,
    PredictionResponse,
    BatchPredictionResponse,
    ExplainResponse,
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

explainer = shap.TreeExplainer(model)

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


@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse
)
def predict_batch(
    transactions: List[Transaction]
):

    data = [
        transaction.model_dump()
        for transaction in transactions
    ]

    df = pd.DataFrame(data)

    df["Amount"] = scaler.transform(
        df[["Amount"]]
    )

    predictions = model.predict(df)

    probabilities = model.predict_proba(df)[:, 1]

    return {
        "predictions": [
            "Fraud"
            if pred == 1
            else "Legitimate"
            for pred in predictions
        ],

        "fraud_probabilities": [
            round(float(prob), 4)
            for prob in probabilities
        ]
    }

@app.post(
    "/explain",
    response_model=ExplainResponse
)
def explain(
    transaction: Transaction
):

    data = transaction.model_dump()

    df = pd.DataFrame([data])

    df["Amount"] = scaler.transform(
        df[["Amount"]]
    )

    prediction = model.predict(df)[0]

    probability = (
        model.predict_proba(df)[0][1]
    )

    shap_explanation = explainer(df)

    shap_values = (
        shap_explanation.values[0]
    )

    feature_impacts = []

    for feature, impact in zip(
        df.columns,
        shap_values
    ):

        feature_impacts.append(
            {
                "feature": feature,
                "impact": round(
                    float(impact),
                    4
                )
            }
        )

    feature_impacts.sort(
        key=lambda x:
            abs(x["impact"]),
        reverse=True
    )

    return {
        "prediction":
            "Fraud"
            if prediction == 1
            else "Legitimate",

        "fraud_probability":
            round(
                float(probability),
                4
            ),

        "top_features":
            feature_impacts[:5]
    }