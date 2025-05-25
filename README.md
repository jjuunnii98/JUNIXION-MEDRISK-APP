# 🧠 JUNIXION MedRisk.AI

**AI 기반 암환자 맞춤형 의료비 예측 및 보험사 추천 시스템**  
**AI-Powered Medical Cost Prediction & Insurance Recommendation Platform for Cancer Patients**

![Streamlit](https://img.shields.io/badge/Deployed%20with-Streamlit-orange)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-blue)
![Status](https://img.shields.io/badge/Status-Ready%20for%20submission-green)

---

## 🔍 프로젝트 개요 | Project Overview

**JUNIXION MedRisk.AI**는 공공 의료데이터와 보험통계를 바탕으로  
사용자의 암종, 성별, 연령대, 가족력, 진료유형, 소득, 병원유형, 거주지역 등을 입력받아  
**AI 모델(XGBoost)**을 통해 예상 의료비를 예측하고  
연소득 대비 부담률을 기반으로 **5단계 위험등급**을 부여하며  
이에 따라 **사용자 맞춤형 보험사**를 추천하는  
**스마트 헬스케어 예측 및 추천 플랫폼**입니다.

**JUNIXION MedRisk.AI** is an AI-driven healthcare prediction platform  
that estimates expected medical costs for cancer patients using personal data  
(e.g., cancer type, gender, age group, family history, region, hospital type, income)  
via an **XGBoost model**, evaluates a 5-level **risk score** based on cost burden,  
and recommends suitable insurance companies accordingly.

---

## 📊 데이터 출처 | Data Sources

| 데이터 종류 (Data Type)         | 출처 (Source)                                       | 설명 (Description)                                     |
|------------------------------|----------------------------------------------------|------------------------------------------------------|
| 암종별 진료비/입원일수       | 건강보험심사평가원 (HIRA)                           | 암종별 인당 진료비 및 평균 입원일수 통계               |
| 병원유형별 진료비            | 건강보험심사평가원 (HIRA)                           | 병원 유형별 평균 단가 및 보정계수                      |
| 생명보험 가입자 통계         | 공공데이터포털 (금융위원회 제공)                   | 보험사별 가입자수, 보험사 규모, 보장 유형 등            |
| 사용자 이력 로그             | 앱 사용자의 입력 로그                              | 위험도 히스토리 누적 기록 및 시각화에 활용됨            |

All data is sourced from reliable public institutions in Korea  
(e.g., [HIRA](https://www.hira.or.kr), [Data.go.kr](https://www.data.go.kr)),  
and has been cleaned, transformed, and structured for model training and analytics.

---

## 🚀 배포 주소 | Demo App

🔗 [Click to Run on Streamlit](https://junixion-medrisk.streamlit.app/)

---

## ✅ 주요 기능 | Key Features

- 🎯 **AI 예측: XGBoost 기반 의료비 예측 (AI-Powered Medical Cost Estimation)**
- 🏥 병원유형 보정계수 적용 (Hospital Adjustment Factor Integration)
- 💰 부담률 계산 및 5단계 위험등급 산정 (Cost Burden Ratio & 5-Level Risk Grading)
- 📊 **SHAP 기반 변수 기여도 시각화** (SHAP Model Explainability Visualization)
- 🧬 입력 이력 기반 히트맵 생성 (Heatmap of Risk History by Region/Age)
- 🛡️ **위험도 기반 보험사 추천 알고리즘** (Risk-Aware Insurance Recommendation)
- 🎛️ 보험사 필터링: **보장유형, 보험료 수준, 모바일 가입 여부** 기준으로 조건 검색
- 📱 **모바일 반응형 UI 지원** (Responsive Design for Mobile Use)
- 💬 사용자 피드백 기록 기능 (Feedback Input Logging)
- 🔒 개인정보 저장 없이 예측만 제공 (No Personal Data Stored)

---

## 🛠 기술 스택 | Tech Stack

| 영역 (Category)   | 기술 스택 (Stack)                           |
|------------------|---------------------------------------------|
| Backend          | Python, pandas, XGBoost                     |
| Frontend         | Streamlit                                   |
| Visualization    | matplotlib, seaborn, plotly, SHAP           |
| Explainability   | SHAP for feature contribution analysis       |
| Deployment       | Streamlit Cloud                             |
| Version Control  | GitHub                                      |

---

## 📁 디렉토리 구조 | Project Structure

```bash
junixion_medrisk_app/
├── app.py                      # Main Streamlit App
├── requirements.txt            # Python dependencies
├── model/
│   └── xgb_model.json          # Trained XGBoost model
├── data/
│   ├── t1.xlsx                 # Cancer stats by type
│   ├── t2.xlsx                 # Age/Income data
│   ├── t3.xlsx                 # Hospital-type cost stats
│   └── life_insurance_general.json  # Insurance statistics
├── user_logs/
│   └── risk_history.csv        # Log of user risk score history
├── fonts/
│   └── NanumGothic.ttf         # Korean font for SHAP/Matplotlib
├── model/
│   └── predictor.py            # AI inference logic
│   └── utils.py                # Data loading, insurance filtering, logger


📮 문의 및 기여 | Contact & Contribution

이 프로젝트는 연구, 학술, 공모전 제출 목적의 비상업적 프로젝트입니다.
기술 기여, 피드백, 공동연구 또는 제품화를 위한 협업 제안은
GitHub Issue 또는 이메일을 통해 연락 주시기 바랍니다.

This project is non-commercial and intended for research & competition use.
We welcome any technical contributions, feedback, or collaborative inquiries.
Please reach out via GitHub Issues or email.

⸻
