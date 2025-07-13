import pandas as pd
import os

# === Load all required CSVs ===
base_path = r"C:\Users\gayat\Documents\edu app v1\oulad_project\data"
info = pd.read_csv(os.path.join(base_path, "studentInfo.csv"))
assessment = pd.read_csv(os.path.join(base_path, "studentAssessment.csv"))
vle = pd.read_csv(os.path.join(base_path, "studentVle.csv"))

# === Clean studentInfo and keep only useful columns ===
info = info[[
    "id_student", "code_module", "code_presentation",
    "age_band", "num_of_prev_attempts", "studied_credits",
    "disability", "final_result"
]]

# === Aggregate assessment scores ===
assessment_agg = (
    assessment
    .groupby("id_student")["score"]
    .mean()
    .reset_index()
    .rename(columns={"score": "avg_score"})
)

# === Aggregate LMS activity ===
vle_agg = (
    vle
    .groupby("id_student")
    .agg({
        "sum_click": "sum",
        "date": "nunique"
    })
    .reset_index()
    .rename(columns={
        "sum_click": "total_clicks",
        "date": "active_days"
    })
)

# === Merge all into one final DataFrame ===
df = info.merge(assessment_agg, on="id_student", how="left")
df = df.merge(vle_agg, on="id_student", how="left")

# === Clean and save ===
df = df.dropna(subset=["final_result"])   # Remove rows with missing target
df.to_csv("oulad_cleaned.csv", index=False)
print("✅ Saved cleaned merged data to oulad_cleaned.csv")
