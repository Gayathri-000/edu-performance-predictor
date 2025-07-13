import streamlit as st
import pandas as pd
import joblib
import shap
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# === Load model, features, and SHAP explainer ===
model = joblib.load("grade_predictor.pkl")
features = joblib.load("feature_names.pkl")

# === Dummy dataframe to encode values ===
df_raw = pd.read_csv("oulad_cleaned.csv").fillna(0)
label_cols = ["code_module", "code_presentation", "age_band", "disability"]

encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df_raw[col] = le.fit_transform(df_raw[col])
    encoders[col] = le

# === Streamlit UI ===
st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Grade Predictor + SHAP Explainer")
st.markdown("Enter student details to predict academic outcome and get personalized advice.")

# === Input form ===
with st.form("student_form"):
    age_band = st.selectbox("Age Band", encoders["age_band"].classes_)
    disability = st.selectbox("Disability", encoders["disability"].classes_)
    num_of_prev_attempts = st.slider("Number of Previous Attempts", 0, 10, 0)
    studied_credits = st.slider("Studied Credits", 0, 120, 60, step=10)
    avg_score = st.slider("Average Assessment Score (%)", 0, 100, 60)
    total_clicks = st.slider("Total LMS Clicks", 0, 50000, 4000, step=500)
    active_days = st.slider("Days Active on LMS", 0, 250, 30)
    code_module = st.selectbox("Module Code", encoders["code_module"].classes_)
    code_presentation = st.selectbox("Presentation Session", encoders["code_presentation"].classes_)

    submitted = st.form_submit_button("🔮 Predict")

# === Process input and predict ===
if submitted:
    # Encode categorical values
    input_dict = {
        "code_module": encoders["code_module"].transform([code_module])[0],
        "code_presentation": encoders["code_presentation"].transform([code_presentation])[0],
        "age_band": encoders["age_band"].transform([age_band])[0],
        "num_of_prev_attempts": num_of_prev_attempts,
        "studied_credits": studied_credits,
        "disability": encoders["disability"].transform([disability])[0],
        "avg_score": avg_score,
        "total_clicks": total_clicks,
        "active_days": active_days
    }

    input_df = pd.DataFrame([input_dict])[features]

    # Prediction
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][pred]
    outcome = "✅ Likely to PASS" if pred == 1 else "❌ At Risk of Failing"

    st.subheader(f"🎯 Prediction: {outcome}")
    st.write(f"📊 Confidence: **{proba * 100:.2f}%**")

    # SHAP Explanation
    explainer = shap.Explainer(model, df_raw[features])
    shap_values = explainer(input_df)

    st.subheader("🔍 Why? SHAP Explanation")
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.plots.waterfall(shap_values[0], max_display=7, show=False)
    st.pyplot()

    # Recommendations
    st.subheader("✅ Personalized Recommendations")
    impact = shap_values.values[0]
    recs = []

    for i, value in enumerate(impact):
        if value < -0.3:
            feat = features[i]
            if feat == "avg_score":
                recs.append("📌 Improve quiz and assignment scores with revision.")
            elif feat == "total_clicks":
                recs.append("📌 Engage more with the LMS materials.")
            elif feat == "active_days":
                recs.append("📌 Spread your study time across more days.")
            elif feat == "num_of_prev_attempts":
                recs.append("📌 Review feedback from previous attempts.")
            elif feat == "disability":
                recs.append("📌 Consider accessing support services if needed.")

    if recs:
        for r in recs:
            st.markdown(r)
    else:
        st.success("🎉 No major issues detected. Keep up the good work!")

