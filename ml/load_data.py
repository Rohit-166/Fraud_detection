import pandas as pd


def load_data(path):
    df = pd.read_csv(path)

    print("Dataset Shape:", df.shape)

    return df


if __name__ == "__main__":
    df = load_data("data/raw/creditcard.csv")
    print(df.head())