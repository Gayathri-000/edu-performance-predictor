import pandas as pd
import shap
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# === Load saved model and feature list ===
model = joblib.load("grade_predictor.pkl")
features = joblib.load("feature_names.pkl")

# === Load dataset used for training ===
df = pd.read_csv("oulad_cleaned.csv").fillna(0)

# === Re-map target to binary (same as training) ===
df["final_result"] = df["final_result"].map({
    "Pass": 1,    
    "Distinction": 1,
    "Fail": 0,
    "Withdrawn": 0
})

# === Re-encode categorical variables ===
label_cols = ["code_module", "code_presentation", "age_band", "disability"]
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# === Prepare X and y ===
X = df[features]
y = df["final_result"]

# === Create SHAP explainer ===
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# === Pick a row to explain (you can change 0 to any index) ===
row_idx = 0
print("🎯 Prediction:", model.predict(X.iloc[[row_idx]])[0])
print("✅ True label:", y.iloc[row_idx])

# === Plot SHAP waterfall (with no label clipping) ===
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values[row_idx], max_display=8, show=False)
plt.tight_layout(pad=3.0)
plt.show()
