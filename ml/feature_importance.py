import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load(
    "backend/model/fraud_model.pkl"
)

X_train = pd.read_csv(
    "data/processed/X_train.csv"
)

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importance
})

feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)

print(feature_importance.head(15))

plt.figure(figsize=(10,6))

plt.barh(
    feature_importance["Feature"][:15],
    feature_importance["Importance"][:15]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Top 15 Feature Importances"
)

plt.tight_layout()
import os

os.makedirs("results", exist_ok=True)

plt.savefig(
    "results/feature_importance.png",
    bbox_inches="tight"
)

print("Plot saved to results/feature_importance.png")