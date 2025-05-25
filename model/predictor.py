# 📁 predictor.py (가족력 포함 + SHAP booster 반환 + 사용자 이력 누적 저장 포함)

import numpy as np
import pandas as pd
import xgboost as xgb
import traceback
import os
from datetime import datetime

# ✅ 히트맵 누적 저장용 CSV 경로
RISK_LOG_PATH = "./user_logs/risk_history.csv"
os.makedirs(os.path.dirname(RISK_LOG_PATH), exist_ok=True)

def predict_medical_cost(user_input: dict, df_hospital: pd.DataFrame, model_path: str) -> dict:
    """
    사용자 입력과 XGBoost Booster 모델을 기반으로 진료비 예측 및 위험등급 판단

    Parameters:
    - user_input: dict, 사용자 입력값
    - df_hospital: DataFrame, 병원유형별 보정계수 포함
    - model_path: str, 학습된 Booster 모델 (.json)

    Returns:
    - dict: 예측 결과 + booster 객체 포함
    """
    try:
        # ✅ 사용자 입력 추출
        age_group = user_input.get("age_group")
        avg_days = float(user_input.get("avg_days", 0))
        is_inpatient = int(user_input.get("is_inpatient", 0))
        patient_count = int(user_input.get("patient_count", 0))
        hospital_type = user_input.get("hospital_type")
        annual_income = float(user_input.get("annual_income"))
        cancer_name = user_input.get("cancer_name", "미지정 암종")
        region = user_input.get("region", "기타")  # 히트맵용 지역 필드
        family_history = int(user_input.get("family_history", 0))  # ✅ 가족력

        if annual_income <= 0:
            raise ValueError("연소득은 0보다 커야 합니다.")

        # ✅ 모델 입력값
        X = pd.DataFrame([{
            "avg_days": avg_days,
            "is_inpatient": is_inpatient,
            "patient_count": patient_count,
            "family_history": family_history  # ✅ 포함
        }])
        dmatrix = xgb.DMatrix(X)

        # ✅ Booster 모델 로드
        booster = xgb.Booster()
        booster.load_model(model_path)
        base_cost = float(booster.predict(dmatrix)[0])

        # ✅ 병원유형 보정계수 적용
        matched = df_hospital[df_hospital["hospital_type"] == hospital_type]
        if matched.empty:
            raise ValueError(f"'{hospital_type}'에 대한 보정계수 정보가 없습니다.")
        correction_factor = matched["correction_factor"].values[0]
        corrected_cost = round(base_cost * correction_factor)

        # ✅ 부담률 계산
        burden_ratio = corrected_cost / annual_income
        burden_ratio_pct = round(burden_ratio * 100, 2)

        # ✅ 위험등급 판단
        if burden_ratio <= 0.05:
            risk_level = "매우 낮음"
        elif burden_ratio <= 0.1:
            risk_level = "낮음"
        elif burden_ratio <= 0.2:
            risk_level = "보통"
        elif burden_ratio <= 0.3:
            risk_level = "높음"
        else:
            risk_level = "매우 높음"

        # ✅ 사용자 이력 누적 저장 (히트맵용)
        new_record = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "region": region,
            "age_group": age_group,
            "risk_score": round(burden_ratio * 100, 2)
        }])

        if os.path.exists(RISK_LOG_PATH):
            old = pd.read_csv(RISK_LOG_PATH)
            updated = pd.concat([old, new_record], ignore_index=True)
        else:
            updated = new_record

        updated.to_csv(RISK_LOG_PATH, index=False)

        return {
            "암종": cancer_name,
            "기본 예측 진료비(원)": round(base_cost),
            "병원유형": hospital_type,
            "보정계수": round(correction_factor, 3),
            "최종 보정 예측 진료비(원)": corrected_cost,
            "연령대": age_group,
            "연소득(원)": annual_income,
            "의료비 부담률": round(burden_ratio, 4),
            "의료비 부담률 (%)": burden_ratio_pct,
            "위험등급": risk_level,
            "raw_cost": corrected_cost,
            "raw_income": annual_income,
            "region": region,
            "booster": booster,
            "X_input": X
        }

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"[predict_medical_cost 오류] {e}")