import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

def load_data():

    X_test = pd.read_csv(
        "data/processed/X_test.csv"
    )

    y_test = pd.read_csv(
        "data/processed/y_test.csv"
    ).squeeze()

    return X_test, y_test

def load_model():

    return joblib.load(
        "backend/model/fraud_model.pkl"
    )

def evaluate():

    model = load_model()

    X_test, y_test = load_data()

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print(
        "ROC AUC:",
        roc_auc_score(
            y_test,
            probabilities
        )
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

if __name__ == "__main__":
    evaluate()