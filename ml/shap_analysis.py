import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load(
    "backend/model/fraud_model.pkl"
)

X_test = pd.read_csv(
    "data/processed/X_test.csv"
)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_test[:100]
)

os.makedirs("results", exist_ok=True)

shap.summary_plot(
    shap_values,
    X_test[:100],
    show=False
)

plt.tight_layout()

plt.savefig(
    "results/shap_summary.png",
    bbox_inches="tight",
    dpi=300
)

print("Saved to results/shap_summary.png")