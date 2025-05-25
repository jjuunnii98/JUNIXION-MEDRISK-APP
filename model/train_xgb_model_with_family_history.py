import pandas as pd
import xgboost as xgb

# ✅ 학습용 가상 데이터 생성 (family_history 포함)
df = pd.DataFrame({
    "avg_days": [7, 14, 21, 28, 35, 14],
    "is_inpatient": [0, 1, 1, 0, 1, 0],
    "patient_count": [3000, 6000, 8000, 11000, 15000, 5000],
    "family_history": [0, 1, 0, 1, 1, 0],
    "target_cost": [500000, 900000, 1100000, 800000, 1400000, 600000]
})

X = df[["avg_days", "is_inpatient", "patient_count", "family_history"]]
y = df["target_cost"]

dtrain = xgb.DMatrix(X, label=y)
params = {"objective": "reg:squarederror"}

model = xgb.train(params, dtrain, num_boost_round=30)
model.save_model("xgb_model.json")

print("✅ xgb_model.json (with family_history) 저장 완료")
