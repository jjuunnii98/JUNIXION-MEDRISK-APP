# train_model.py

import pandas as pd
import xgboost as xgb

# 예시 학습 데이터 생성
df = pd.DataFrame({
    "avg_days": [5, 10, 20, 30, 40],
    "is_inpatient": [0, 1, 1, 0, 1],
    "patient_count": [5000, 8000, 10000, 12000, 15000],
    "target_cost": [500000, 900000, 1200000, 750000, 1300000]
})

X = df[["avg_days", "is_inpatient", "patient_count"]]
y = df["target_cost"]

# XGBoost 모델 학습
dtrain = xgb.DMatrix(X, label=y)
params = {"objective": "reg:squarederror"}
model = xgb.train(params, dtrain, num_boost_round=20)

# 모델 저장
model.save_model("/Users/junyeong/Desktop/JUNIXION/2.Contest_list/2.Healthcare_Startup/data_analysis/junixion_medrisk_app/model/xgb_model.json")

print("✅ xgb_model.json 저장 완료")