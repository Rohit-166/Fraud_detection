import os
import joblib
import pandas as pd

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def load_processed_data():

    X_train = pd.read_csv(
        "data/processed/X_train.csv"
    )

    X_test = pd.read_csv(
        "data/processed/X_test.csv"
    )

    y_train = pd.read_csv(
        "data/processed/y_train.csv"
    )

    y_test = pd.read_csv(
        "data/processed/y_test.csv"
    )

    return (
        X_train,
        X_test,
        y_train.squeeze(),
        y_test.squeeze()
    )

def apply_smote(
    X_train,
    y_train
):

    smote = SMOTE(
        random_state=42
    )

    X_resampled, y_resampled = (
        smote.fit_resample(
            X_train,
            y_train
        )
    )

    return X_resampled, y_resampled

def train_model(
    X_train,
    y_train
):

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(
        X_train,
        y_train
    )

    return model

def save_model(model):

    os.makedirs(
        "backend/model",
        exist_ok=True
    )

    joblib.dump(
        model,
        "backend/model/fraud_model.pkl"
    )

def main():

    print("Loading data...")

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_processed_data()

    print("Applying SMOTE...")

    X_train,y_train = apply_smote(
        X_train,
        y_train
    )

    print("Training XGBoost...")

    model = train_model(
        X_train,
        y_train
    )

    save_model(model)

    print("Model Saved!")

if __name__ == "__main__":
    main()