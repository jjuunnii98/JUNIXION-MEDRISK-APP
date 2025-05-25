# 📁 app.py (최종본: 가족력 포함 + SHAP + 보험사 필터 개선 + 한글 폰트 대응)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import seaborn as sns
import shap
import os

from model.predictor import predict_medical_cost
from model.utils import (
    load_data_sources_safe,
    recommend_insurance_company,
    risk_score_map,
    log_risk_score
)

# ✅ 한글 폰트 설정 (Streamlit Cloud 대응)
font_path = "./fonts/NanumGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

# ✅ 기본 설정
st.set_page_config(page_title="JUNIXION - 의료비 예측", layout="wide")
st.title("JUNIXION MedRisk.AI")
st.caption("AI 기반 암환자 맞춤형 의료비 예측 및 보험사 추천 시스템")

@st.cache_data(show_spinner=True)
def load_all_data():
    return load_data_sources_safe()

df_t1, df_t2, df_t3, insurance_df = load_all_data()

# ✅ 전처리
df_t1.columns = df_t1.iloc[1]
df_t1 = df_t1[2:].copy()
df_t1 = df_t1.rename(columns={
    "명칭": "암종명", "인당진료비": "인당진료비",
    "인당입(내)원일수": "인당입원일수", "진료인원": "진료인원"
})
df_t1["암종명"] = df_t1["암종명"].astype(str)

if "correction_factor" not in df_t3.columns:
    df_t3["cost_per_person"] = pd.to_numeric(df_t3["내원일당진료비"], errors="coerce")
    df_t3["hospital_type"] = df_t3["구분"]
    df_t3["correction_factor"] = df_t3["cost_per_person"] / df_t3["cost_per_person"].mean()

insurance_df["인원수"] = pd.to_numeric(insurance_df.get("인원수", 0), errors="coerce").fillna(0)
insurance_df["보험사명"] = insurance_df.get("보험사명", "이름없음")

# ✅ 사용자 입력
with st.expander("📥 사용자 정보 입력", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        cancer_type = st.selectbox("암종 선택", df_t1["암종명"].dropna().unique())
        gender = st.radio("성별", ["남성", "여성"], horizontal=True)
        region = st.selectbox("거주 지역", ["서울", "경기", "인천", "부산", "기타"])
        age_group = st.selectbox("연령대", ["20대", "30대", "40대", "50대", "60대이상"])
        is_inpatient = st.radio("진료 유형", ["외래", "입원"], horizontal=True)
        is_inpatient = 1 if is_inpatient == "입원" else 0
        family_history = st.radio("가족력 여부", ["없음", "있음"], horizontal=True)
        family_history = 1 if family_history == "있음" else 0
        annual_income = st.number_input("예상 연소득 (원)", 1_000_000, 100_000_000, 30_000_000, step=500_000)
    with col2:
        hospital_type = st.selectbox("병원 유형", df_t3["hospital_type"].dropna().unique())
        cancer_row = df_t1[df_t1["암종명"] == cancer_type]
        avg_days = float(cancer_row["인당입원일수"].values[0]) if not cancer_row.empty else 7.0
        patient_count = int(cancer_row["진료인원"].values[0]) if not cancer_row.empty else 4000
        cancer_cost = int(str(cancer_row["인당진료비"].values[0]).replace(",", "").replace("원", "")) if not cancer_row.empty else 1_000_000

# ✅ 예측 버튼
if st.button("의료예측 및 보험 추천"):
    try:
        user_input = {
            "age_group": age_group,
            "avg_days": avg_days,
            "is_inpatient": is_inpatient,
            "patient_count": patient_count,
            "hospital_type": hospital_type,
            "annual_income": annual_income,
            "cancer_cost": cancer_cost,
            "cancer_name": cancer_type,
            "region": region,
            "family_history": family_history
        }

        st.markdown("입력값 확인:")
        st.json(user_input)

        result_dict = predict_medical_cost(user_input, df_t3, model_path="./model/xgb_model.json")
        booster = result_dict.pop("booster")
        X_input = result_dict.pop("X_input")

        score = risk_score_map.get(result_dict["위험등급"], 0)
        log_risk_score(region, age_group, score)

        st.subheader("예측 결과")
        st.table(pd.DataFrame(result_dict.items(), columns=["항목", "값"]))

        st.subheader("위험등급 게이지")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "위험등급 점수"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 20], 'color': "#66BB6A"},
                    {'range': [20, 40], 'color': "#9BE7C4"},
                    {'range': [40, 60], 'color': "#FFF176"},
                    {'range': [60, 80], 'color': "#FFB74D"},
                    {'range': [80, 100], 'color': "#EF5350"}
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'value': score}
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("SHAP 변수 기여도")
        try:
            explainer = shap.Explainer(booster)
            shap_values = explainer(X_input)
            fig_shap = plt.figure(figsize=(5, 3))
            shap.plots.bar(shap_values[0], show=False)
            st.pyplot(fig_shap)
        except Exception as e:
            st.warning(f"SHAP 시각화 실패: {e}")

        st.subheader("예측 요약 시각화")
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            fig1, ax1 = plt.subplots(figsize=(2.5, 2.2))
            labels = ["예측 진료비", "연소득"]
            values = [result_dict["raw_cost"], result_dict["raw_income"]]
            colors = ["#FF9999", "#99CCFF"]
            ax1.bar(labels, values, color=colors)
            ax1.set_ylabel("금액 (원)")
            for i, v in enumerate(values):
                ax1.text(i, v + v * 0.01, f"{v:,}", ha='center', fontsize=9)
            st.pyplot(fig1, use_container_width=False)

        with col_chart2:
            try:
                df_log = pd.read_csv("data/risk_history.csv")
                heatmap_data = df_log.pivot_table(index="age_group", columns="region", values="risk_score", aggfunc="mean")
                fig2, ax2 = plt.subplots(figsize=(2.5, 2.2))
                sns.heatmap(heatmap_data, cmap="Reds", annot=True, fmt=".1f", ax=ax2)
                st.pyplot(fig2, use_container_width=False)
            except Exception as e:
                st.warning(f"히트맵 생성 실패: {e}")

        st.subheader("추천 보험사 목록")
        try:
            recommended = recommend_insurance_company(result_dict["위험등급"], insurance_df)

            with st.expander("🔍 보험사 필터링 옵션"):
                colf1, colf2, colf3 = st.columns(3)
                with colf1:
                    selected_type = st.selectbox("보장유형", ["전체"] + list(recommended["보장유형"].dropna().unique()))
                with colf2:
                    selected_price = st.selectbox("보험료 수준", ["전체"] + list(recommended["평균보험료"].dropna().unique()))
                with colf3:
                    mobile_only = st.checkbox("모바일 가입 가능만 보기")

                if selected_type != "전체":
                    recommended = recommended[recommended["보장유형"] == selected_type]
                if selected_price != "전체":
                    recommended = recommended[recommended["평균보험료"] == selected_price]
                if mobile_only:
                    recommended = recommended[recommended["모바일가입"] == True]

            recommended["인원수"] = recommended["인원수"].astype(int).apply(lambda x: f"{x:,}")
            st.dataframe(recommended[["보험사명", "보장유형", "평균보험료", "모바일가입", "민원률", "보험사규모", "인원수"]]
                         .sort_values(by="인원수", ascending=False).reset_index(drop=True))

        except Exception as e:
            st.error(f"❌ 보험사 추천 오류: {e}")

    except Exception as e:
        st.error(f"❌ 예측 중 오류 발생: {e}")

# ✅ 사용자 피드백
st.markdown("---")
st.subheader("📣 사용자 피드백")
feedback = st.text_area("시스템에 대한 의견을 남겨주세요:", placeholder="예) UI가 보기 좋아요! 개선사항은...")
if st.button("피드백 제출"):
    with open("feedback_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{feedback}\n")
    st.success("감사합니다! 피드백이 제출되었습니다.")