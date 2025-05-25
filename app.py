import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import shap
import os
import plotly.graph_objects as go

from model.predictor import predict_medical_cost
from model.utils import (
    load_data_sources_safe,
    recommend_insurance_company,
    risk_score_map,
    log_risk_score
)

# ✅ Korean font (fallback)
try:
    font_path = "./fonts/NanumGothic.ttf"
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'
except Exception as e:
    st.warning(f"⚠️ Font error: {e}")

# ✅ Page setup
st.set_page_config(page_title="JUNIXION MedRisk.AI", layout="centered")
st.title("JUNIXION MedRisk.AI")
st.caption("AI-powered cancer cost prediction & insurance recommendation")

@st.cache_data
def load_all_data():
    return load_data_sources_safe()

df_t1, df_t2, df_t3, insurance_df = load_all_data()

# ✅ Preprocessing
df_t1.columns = df_t1.iloc[1]
df_t1 = df_t1[2:].copy()
df_t1 = df_t1.rename(columns={
    "명칭": "Cancer Type", "인당진료비": "Cost per Person",
    "인당입(내)원일수": "Days per Patient", "진료인원": "Patients"
})
df_t1["Cancer Type"] = df_t1["Cancer Type"].astype(str)

if "correction_factor" not in df_t3.columns:
    df_t3["cost_per_person"] = pd.to_numeric(df_t3["내원일당진료비"], errors="coerce")
    df_t3["hospital_type"] = df_t3["구분"]
    df_t3["correction_factor"] = df_t3["cost_per_person"] / df_t3["cost_per_person"].mean()

insurance_df["인원수"] = pd.to_numeric(insurance_df.get("인원수", 0), errors="coerce").fillna(0)
insurance_df["보험사명"] = insurance_df.get("보험사명", "Unnamed")

# ✅ User Input
with st.expander("📥 Enter your information", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        cancer_type = st.selectbox("Cancer Type", df_t1["Cancer Type"].dropna().unique())
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        region = st.selectbox("Region", ["Seoul", "Gyeonggi", "Incheon", "Busan", "Other"])
        age_group = st.selectbox("Age Group", ["20s", "30s", "40s", "50s", "60+"])
        is_inpatient = st.radio("Treatment Type", ["Outpatient", "Inpatient"], horizontal=True)
        is_inpatient = 1 if is_inpatient == "Inpatient" else 0
        family_history = st.radio("Family History", ["No", "Yes"], horizontal=True)
        family_history = 1 if family_history == "Yes" else 0
        annual_income = st.number_input("Annual Income (KRW)", 1_000_000, 100_000_000, 30_000_000, step=500_000)
    with col2:
        hospital_type = st.selectbox("Hospital Type", df_t3["hospital_type"].dropna().unique())
        cancer_row = df_t1[df_t1["Cancer Type"] == cancer_type]
        avg_days = float(cancer_row["Days per Patient"].values[0]) if not cancer_row.empty else 7.0
        patient_count = int(cancer_row["Patients"].values[0]) if not cancer_row.empty else 4000
        cancer_cost = int(str(cancer_row["Cost per Person"].values[0]).replace(",", "").replace("원", "")) if not cancer_row.empty else 1_000_000

# ✅ Prediction
if st.button("Predict Medical Cost & Recommend Insurance"):
    with st.spinner("🔍 Predicting..."):
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

            st.markdown("#### Input Summary")
            st.json(user_input)

            result_dict = predict_medical_cost(user_input, df_t3, model_path="./model/xgb_model.json")
            booster = result_dict.pop("booster")
            X_input = result_dict.pop("X_input")

            score = risk_score_map.get(result_dict["위험등급"], 0)
            log_risk_score(region, age_group, score)

            # ✅ Metric Cards
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            col_metric1.metric("💰 Estimated Cost", f"{result_dict['raw_cost']:,} KRW")
            col_metric2.metric("📉 Income", f"{result_dict['raw_income']:,} KRW")
            col_metric3.metric("⚠️ Burden (%)", f"{result_dict['의료비 부담률 (%)']}%")

            st.subheader("Prediction Details")
            st.table(pd.DataFrame(result_dict.items(), columns=["Metric", "Value"]))

            # ✅ Risk Gauge
            st.subheader("Risk Level")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': "Risk Score"},
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
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=False)

            # ✅ SHAP
            st.subheader("SHAP Feature Impact")
            try:
                explainer = shap.Explainer(booster)
                shap_values = explainer(X_input)
                shap_vals = shap_values.values[0]
                feature_names = X_input.columns.tolist()

                fig, ax = plt.subplots(figsize=(2.5, 1.5))
                colors = ['#FF6384' if val > 0 else '#36A2EB' for val in shap_vals]
                bars = ax.barh(feature_names, shap_vals, color=colors)
                ax.set_title("SHAP Impact", fontsize=9)
                ax.tick_params(labelsize=7)
                for i, (bar, val) in enumerate(zip(bars, shap_vals)):
                    xpos = bar.get_width()
                    ha = 'left' if xpos > 0 else 'right'
                    ax.text(xpos, bar.get_y() + bar.get_height()/2, f'{val:+.0f}', va='center', ha=ha, fontsize=6)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=False)

                # ✅ Explanation Text
                st.markdown("""
                💡 **Interpretation**: Positive SHAP values increase predicted cost, negative ones reduce it.
                Use this to understand what factors drive high medical costs for the patient.
                """)
            except Exception as e:
                st.warning(f"SHAP Error: {e}")

            # ✅ Summary Bar Chart
            st.subheader("Estimated Cost vs Income")
            fig1, ax1 = plt.subplots(figsize=(2.5, 1.8))
            labels = ["Estimated Cost", "Income"]
            values = [result_dict["raw_cost"], result_dict["raw_income"]]
            colors = ["#FF9999", "#99CCFF"]
            ax1.bar(labels, values, color=colors)
            for i, v in enumerate(values):
                ax1.text(i, v + v * 0.01, f"{v:,}", ha='center', fontsize=8)
            ax1.set_ylabel("KRW")
            st.pyplot(fig1, use_container_width=False)

            # ✅ Insurance Recommendation
            st.subheader("Recommended Insurance Companies")
            try:
                recommended = recommend_insurance_company(result_dict["위험등급"], insurance_df)
                if recommended.empty:
                    st.info("No matching insurance companies.")
                else:
                    with st.expander("🔍 Filter Options"):
                        colf1, colf2, colf3 = st.columns(3)
                        with colf1:
                            selected_type = st.selectbox("Type", ["All"] + list(recommended["보장유형"].dropna().astype(str).unique()))
                        with colf2:
                            selected_price = st.selectbox("Price Level", ["All"] + list(recommended["평균보험료"].dropna().astype(str).unique()))
                        with colf3:
                            mobile_only = st.checkbox("Mobile only")

                        if selected_type != "All":
                            recommended = recommended[recommended["보장유형"] == selected_type]
                        if selected_price != "All":
                            recommended = recommended[recommended["평균보험료"] == selected_price]
                        if mobile_only:
                            recommended = recommended[recommended["모바일가입"] == True]

                    recommended["인원수"] = pd.to_numeric(recommended["인원수"], errors="coerce").fillna(0).astype(int)
                    recommended["인원수"] = recommended["인원수"].apply(lambda x: f"{x:,}")
                    st.dataframe(recommended[["보험사명", "보장유형", "평균보험료", "모바일가입", "민원률", "보험사규모", "인원수"]]
                                 .sort_values(by="인원수", ascending=False).reset_index(drop=True))
            except Exception as e:
                st.error(f"Insurance recommendation error: {e}")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# ✅ Feedback
st.markdown("---")
st.subheader("📣 Feedback")
feedback = st.text_area("Please share your thoughts or suggestions:")
if st.button("Submit Feedback"):
    with open("feedback_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{feedback}\n")
    st.success("Thanks for your feedback!")