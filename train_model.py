import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# === Load cleaned data ===
df = pd.read_csv("oulad_cleaned.csv")

# === Convert final_result to binary (1 = Pass/Distinction, 0 = Fail/Withdrawn) ===
df["final_result"] = df["final_result"].map({
    "Pass": 1,
    "Distinction": 1,
    "Fail": 0,
    "Withdrawn": 0
})

# === Handle missing values (should be none, but just in case) ===
df = df.fillna(0)

# === Encode categorical columns ===
label_cols = ["code_module", "code_presentation", "age_band", "disability"]
for col in label_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

# === Define features and target ===
features = [
    "code_module", "code_presentation", "age_band",
    "num_of_prev_attempts", "studied_credits", "disability",
    "avg_score", "total_clicks", "active_days"
]
X = df[features]
y = df["final_result"]

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# === Train XGBoost classifier ===
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy: {acc:.4f}")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# === Save model and feature names ===
joblib.dump(model, "grade_predictor.pkl")
joblib.dump(features, "feature_names.pkl")
print("💾 Model and feature list saved!")
