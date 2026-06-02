import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import *

def load_data():
    return pd.read_csv(RAW_DATA_PATH)

def split_features_target(df):

    X = df.drop("Class", axis=1)

    y = df["Class"]

    return X, y


def create_train_test_split(X, y):

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

def scale_amount(
    X_train,
    X_test
):

    scaler = StandardScaler()

    X_train["Amount"] = scaler.fit_transform(
        X_train[["Amount"]]
    )

    X_test["Amount"] = scaler.transform(
        X_test[["Amount"]]
    )

    return X_train, X_test, scaler

def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test
):

    os.makedirs(
        PROCESSED_DATA_DIR,
        exist_ok=True
    )

    X_train.to_csv(
        f"{PROCESSED_DATA_DIR}/X_train.csv",
        index=False
    )

    X_test.to_csv(
        f"{PROCESSED_DATA_DIR}/X_test.csv",
        index=False
    )

    y_train.to_csv(
        f"{PROCESSED_DATA_DIR}/y_train.csv",
        index=False
    )

    y_test.to_csv(
        f"{PROCESSED_DATA_DIR}/y_test.csv",
        index=False
    )


def main():

    print("Loading dataset...")

    df = load_data()

    X, y = split_features_target(df)

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = create_train_test_split(
        X,
        y
    )

    X_train, X_test, scaler = scale_amount(
        X_train,
        X_test
    )
    
    import joblib
    import os
    os.makedirs(
    "backend/model",
    exist_ok=True
     )

    joblib.dump(
    scaler,
    "backend/model/scaler.pkl"
    )

    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("Preprocessing complete!")

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)


if __name__ == "__main__":
    main()
