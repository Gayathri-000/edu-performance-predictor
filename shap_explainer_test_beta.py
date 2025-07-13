#issue - the left window text got clipped

import pandas as pd
import shap
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

# === Load saved model and feature names ===
model = joblib.load("grade_predictor.pkl")
features = joblib.load("feature_names.pkl")

# === Load the cleaned dataset and prepare input ===
df = pd.read_csv("oulad_cleaned.csv")
df = df.fillna(0)

# Convert final_result to binary again (same logic as training)
df["final_result"] = df["final_result"].map({
    "Pass": 1, "Distinction": 1,
    "Fail": 0, "Withdrawn": 0
})

# Encode the categorical columns again (to match model training)
from sklearn.preprocessing import LabelEncoder
label_cols = ["code_module", "code_presentation", "age_band", "disability"]
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df[features]
y = df["final_result"]

# === Use SHAP to explain predictions ===
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# === Plot SHAP explanation for one student (row 0) ===
print("🎯 Prediction:", model.predict(X.iloc[[0]])[0])
print("✅ True label:", y.iloc[0])

# Waterfall plot
shap.plots.waterfall(shap_values[0], max_display=6)
