# 파일: app.py (JUNIXION MedRisk.AI 최종의 형)

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

# 한글 폰트 설정
try:
    font_path = "./fonts/NanumGothic.ttf"
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = font_prop.get_name()
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
except Exception as e:
    st.warning(f"\u26a0\ufe0f \ud3f0트 설정 오류: {e}")

# 기본 설정
st.set_page_config(page_title="JUNIXION - \uc758\ub8cc\ube44 \uc608\uce21", layout="wide")
st.title("JUNIXION MedRisk.AI")
st.caption("AI \uae30반 \uc554\ud559\uc790 \ub9de\ucda4\ud615 \uc758\ub8cc\ube44 \uc608\uce21 \ubc0f \ubcf4\ud5d8\uc0ac \ucd94\ucc9c \uc2dc\uc2a4\ud15c")

@st.cache_data(show_spinner=True)
def load_all_data():
    return load_data_sources_safe()

df_t1, df_t2, df_t3, insurance_df = load_all_data()

# 전처리
df_t1.columns = df_t1.iloc[1]
df_t1 = df_t1[2:].copy()
df_t1 = df_t1.rename(columns={"\uba85\ucc38": "\uc554\uc885\uba85", "\uc778\ub2f9\uc9c4\ub8cc\ube44": "\uc778\ub2f9\uc9c4\ub8cc\ube44", "\uc778\ub2f9\uc785(\ub0b4)\uc6d0\uc77c\uc218": "\uc778\ub2f9\uc785\uc6d0\uc77c\uc218", "\uc9c4\ub8cc\uc778\uc6d0": "\uc9c4\ub8cc\uc778\uc6d0"})
df_t1["\uc554\uc885\uba85"] = df_t1["\uc554\uc885\uba85"].astype(str)

if "correction_factor" not in df_t3.columns:
    df_t3["cost_per_person"] = pd.to_numeric(df_t3["\ub0b4\uc6d0\uc77c\ub2f9\uc9c4\ub8cc\ube44"], errors="coerce")
    df_t3["hospital_type"] = df_t3["\uad6c\ubcc4"]
    df_t3["correction_factor"] = df_t3["cost_per_person"] / df_t3["cost_per_person"].mean()

insurance_df["\uc778\uc6d0\uc218"] = pd.to_numeric(insurance_df.get("\uc778\uc6d0\uc218", 0), errors="coerce").fillna(0)
insurance_df["\ubcf4\ud5d8\uc0ac\uba85"] = insurance_df.get("\ubcf4\ud5d8\uc0ac\uba85", "\uc774\ub984\uc5c6\uc74c")

# 사용자 입력
with st.expander("\ud83d\udce5 \uc0ac용자 \uc815\ubcf4 \uc785력", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        cancer_type = st.selectbox("\uc554\uc885 \uc120\ud0dd", df_t1["\uc554\uc885\uba85"].dropna().unique())
        gender = st.radio("\uc131\ubcc4", ["\ub0a8\uc131", "\uc5ec\uc131"], horizontal=True)
        region = st.selectbox("\uac70\uc8fc \uc9c0\uc5ed", ["\uc11c\uc6b8", "\uacbd\uae30", "\uc778\ucc9c", "\ubd80\uc0b0", "\uae30\ud0c0"])
        age_group = st.selectbox("\uc5f0\ub839\ub300", ["20\ub300", "30\ub300", "40\ub300", "50\ub300", "60\ub300\uc774\uc0c1"])
        is_inpatient = st.radio("\uc9c4\ub8cc \uc720형", ["\uc678\ub840", "\uc785\uc6d0"], horizontal=True)
        is_inpatient = 1 if is_inpatient == "\uc785\uc6d0" else 0
        family_history = st.radio("\uac00족\ub825 \uc5ec\ubd80", ["\uc5c6\uc74c", "\uc788\uc74c"], horizontal=True)
        family_history = 1 if family_history == "\uc788\uc74c" else 0
        annual_income = st.number_input("\uc608산 \uc5f0소득 (\uc6d0)", 1_000_000, 100_000_000, 30_000_000, step=500_000)
    with col2:
        hospital_type = st.selectbox("\ubcd1\uc6d0 \uc720형", df_t3["hospital_type"].dropna().unique())
        cancer_row = df_t1[df_t1["\uc554\uc885\uba85"] == cancer_type]
        avg_days = float(cancer_row["\uc778\ub2f9\uc785\uc6d0\uc77c\uc218"].values[0]) if not cancer_row.empty else 7.0
        patient_count = int(cancer_row["\uc9c4\ub8cc\uc778\uc6d0"].values[0]) if not cancer_row.empty else 4000
        cancer_cost = int(str(cancer_row["\uc778\ub2f9\uc9c4\ub8cc\ube44"].values[0]).replace(",", "").replace("\uc6d0", "")) if not cancer_row.empty else 1_000_000

# 예\uce21 & \ubcf4\ud5d8사 \ucd94\ucc9c
if st.button("\uc758\ub8cc\uc608\uce21 \ubc0f \ubcf4\ud5d8 \ucd94\ucc9c"):
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

        st.markdown("\uc785\ub825\uac12 \ud655\uc778:")
        st.json(user_input)

        result_dict = predict_medical_cost(user_input, df_t3, model_path="./model/xgb_model.json")
        booster = result_dict.pop("booster")
        X_input = result_dict.pop("X_input")

        score = risk_score_map.get(result_dict["\uc704\ud5d8\ub4f1\uae09"], 0)
        log_risk_score(region, age_group, score)

        st.subheader("\uc608\uce21 \uacb0\uacfc")
        st.table(pd.DataFrame(result_dict.items(), columns=["\ud56d\ubaa9", "\uac12"]))

        st.subheader("\uc704\ud5d8\ub4f1\uae09 \uac8c이지")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "\uc704\ud5d8\ub4f1\uae09 \uc810\uc218"},
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

        st.subheader("SHAP \ubcc0\uc218 \uae30\uc5b4\ub3c4")
        try:
            explainer = shap.Explainer(booster)
            shap_values = explainer(X_input)
            shap_vals = shap_values.values[0]
            feature_names = X_input.columns.tolist()

            fig, ax = plt.subplots(figsize=(3.5, 2.2))
            colors = ['#FF6384' if val > 0 else '#36A2EB' for val in shap_vals]
            bars = ax.barh(feature_names, shap_vals, color=colors)
            ax.set_title("SHAP \ubcc0\uc218 \uc601\ud5a5\ub825", fontsize=10)
            ax.tick_params(axis='both', labelsize=8)
            for i, (bar, val) in enumerate(zip(bars, shap_vals)):
                xpos = bar.get_width()
                alignment = 'left' if xpos >= 0 else 'right'
                ax.text(xpos, bar.get_y() + bar.get_height() / 2, f'{val:+,.0f}',
                        ha=alignment, va='center', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"SHAP \uc2dc\uac01\ud654 \uc2e4\ud328: {e}")

        st.subheader("\uc608\uce21 \uc694약 \uc2dc\uac01\ud654")
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig1, ax1 = plt.subplots(figsize=(2.5, 2.2))
            labels = ["\uc608\uce21 \uc9c4\ub8cc\ube44", "\uc5f0\uc18c득"]
            values = [result_dict["raw_cost"], result_dict["raw_income"]]
            colors = ["#FF9999", "#99CCFF"]
            ax1.bar(labels, values, color=colors)
            ax1.set_ylabel("\uae08액 (\uc6d0)")
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
                st.warning(f"\ud788\ud2b8\ub9f5 \uc0dd\uc131 \uc2e4\ud328: {e}")

        st.subheader("\ucd94\ucc9c \ubcf4\ud5d8\uc0ac \ubaa9록")
        try:
            recommended = recommend_insurance_company(result_dict["\uc704\ud5d8\ub4f1\uae09"], insurance_df)

            if recommended.empty:
                st.info("\u2753 \ud574\ub2f9 \uc870\uac74에 \ub9de는 \ubcf4\ud5d8\uc0ac가 \uc5c6\uc2b5\ub2c8\ub2e4.")
            else:
                with st.expander("\ud83d\udd0d \ubcf4\ud5d8\uc0ac \ud544\ud130\ub9c1 \uc635션"):
                    colf1, colf2, colf3 = st.columns(3)
                    with colf1:
                        selected_type = st.selectbox("\ubcf4\uc7a5\uc720형", ["\uc804\uccb4"] + list(recommended["\ubcf4\uc7a5\uc720형"].dropna().astype(str).unique()))
                    with colf2:
                        selected_price = st.selectbox("\ubcf4\ud5d8\ub8cc \uc218\uc900", ["\uc804\uccb4"] + list(recommended["\ud3c9\uade0\ubcf4\ud5d8\ub8cc"].dropna().astype(str).unique()))
                    with colf3:
                        mobile_only = st.checkbox("\ubaa8\ubc14일 \uac00입 \uac00\ub2a5\ub9cc \ubcf4\uae30")

                    if selected_type != "\uc804\uccb4":
                        recommended = recommended[recommended["\ubcf4\uc7a5\uc720형"] == selected_type]
                    if selected_price != "\uc804\uccb4":
                        recommended = recommended[recommended["\ud3c9\uade0\ubcf4\ud5d8\ub8cc"] == selected_price]
                    if mobile_only:
                        recommended = recommended[recommended["\ubaa8\ubc14일\uac00\uc785"] == True]

                recommended["\uc778\uc6d0\uc218"] = pd.to_numeric(recommended["\uc778\uc6d0\uc218"], errors="coerce").fillna(0).astype(int)
                recommended["\uc778\uc6d0\uc218"] = recommended["\uc778\uc6d0\uc218"].apply(lambda x: f"{x:,}")

                st.dataframe(recommended[["\ubcf4\ud5d8\uc0ac\uba85", "\ubcf4\uc7a5\uc720형", "\ud3c9\uade0\ubcf4\ud5d8\ub8cc", "\ubaa8\ubc14일\uac00\uc785", "\ubbfc원\b960", "\ubcf4\ud5d8\uc0ac\uaddc\ubaa8", "\uc778\uc6d0\uc218"]]
                             .sort_values(by="\uc778\uc6d0\uc218", ascending=False).reset_index(drop=True))

        except Exception as e:
            st.error(f"\u274c \ubcf4\ud5d8\uc0ac \ucd94\ucc9c \uc624\ub958: {e}")

    except Exception as e:
        st.error(f"\u274c \uc608\uce21 \uc911 \uc624\ub958 \ubc1c\uc0dd: {