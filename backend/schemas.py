from typing import List

from pydantic import BaseModel



class Transaction(BaseModel):
    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


class PredictionResponse(BaseModel):
    prediction: str
    fraud_probability: float


class BatchPredictionResponse(BaseModel):
    predictions: List[str]
    fraud_probabilities: List[float]


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model: str
    roc_auc: float
    precision: float
    recall: float

class FeatureImpact(BaseModel):
    feature: str
    impact: float


class ExplainResponse(BaseModel):
    prediction: str
    fraud_probability: float
    top_features: List[FeatureImpact]