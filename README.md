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

**JUNIXION MedRisk.AI** is an AI-powered platform  
that predicts expected medical costs for cancer patients  
based on user information (cancer type, age, gender, region, hospital type, income, etc.),  
calculates the cost burden ratio, assigns a **5-level risk grade**,  
and recommends **insurance companies** tailored to the user.

---

## 📊 데이터 출처 | Data Sources

| 데이터 종류 (Data Type)         | 출처 (Source)                                       | 설명 (Description)                                     |
|------------------------------|----------------------------------------------------|------------------------------------------------------|
| 암종별 진료비/입원일수       | 건강보험심사평가원 (HIRA)                           | 암종별 인당 진료비 및 평균 입원일수 통계               |
| 병원유형별 진료비            | 건강보험심사평가원 (HIRA)                           | 병원 유형별 평균 단가 및 보정계수                      |
| 생명보험 가입자 통계         | 공공데이터포털 (금융위원회 제공)                   | 보험사별 가입자수, 보험사 규모, 보장 유형 등            |
| 사용자 이력 로그             | 앱 사용자의 입력 로그                              | 위험도 히스토리 누적 기록 및 시각화에 활용됨            |

> ⚠️ **주의: 본 프로젝트는 실제 개인정보 없이 구성된 가상 데이터셋(모의 데이터)을 기반으로 개발되었습니다.**  
> 서비스에서 사용되는 모든 입력은 익명이며, 실제 의료진단이나 보험설계 대체 수단이 아닙니다.

> ⚠️ **Disclaimer: This project uses mock/virtual datasets without real personal data.**  
> The platform is for academic and demonstration purposes only and does not replace professional medical or insurance advice.

---

## 🚀 배포 주소 | Live App

🔗 [Click to Open the Streamlit App](https://junixion-medrisk.streamlit.app/)

---

## ✅ 주요 기능 | Key Features

- 🎯 **AI 예측: XGBoost 기반 의료비 예측 (AI-Powered Medical Cost Estimation)**
- 🏥 병원유형 보정계수 적용 (Hospital Cost Adjustment Integration)
- 💰 부담률 계산 및 5단계 위험등급 산정 (Cost Burden Ratio & 5-Level Risk Scoring)
- 📊 **SHAP 기반 변수 기여도 시각화** (Model Explainability via SHAP)
- 🧬 입력 이력 기반 히트맵 시각화 (Risk History Heatmap by Region & Age)
- 🛡️ **위험도 기반 보험사 추천 알고리즘** (Risk-Aware Insurance Matching)
- 🎛️ 보험사 필터링: **보장유형 / 보험료 수준 / 모바일 가입 여부** 검색 지원
- 📱 모바일 최적화 UI (Mobile-Responsive Streamlit App)
- 💬 피드백 수집 기능 포함 (User Feedback Logging Supported)
- 🔒 개인정보 저장 없음 (No Personal Data Stored or Tracked)

---

## 🛠 기술 스택 | Tech Stack

| 영역 (Category)   | 기술 스택 (Stack)                           |
|------------------|---------------------------------------------|
| Backend          | Python, pandas, XGBoost                     |
| Frontend         | Streamlit                                   |
| Visualization    | matplotlib, seaborn, plotly, SHAP           |
| Explainability   | SHAP for feature importance visualization   |
| Deployment       | Streamlit Cloud                             |
| Version Control  | Git + GitHub                                |

---

## 📁 디렉토리 구조 | Project Structure

```bash
junixion_medrisk_app/
├── app.py                      # 📌 Main Streamlit App (entry point)
├── requirements.txt            # 📦 Python dependencies
├── model/
│   ├── xgb_model.json          # 🎯 Trained XGBoost model
│   ├── predictor.py            # 🔍 Prediction logic
│   └── utils.py                # 🛠 Data loader & insurance logic
├── data/
│   ├── t1.xlsx                 # 📊 Cancer data by type
│   ├── t2.xlsx                 # 📊 Age & income stats
│   ├── t3.xlsx                 # 📊 Hospital-type cost data
│   └── life_insurance_general.json  # 🛡️ Insurance company data
├── user_logs/
│   └── risk_history.csv        # 🧬 Risk grade history log
├── fonts/
│   └── NanumGothic.ttf         # 🗂️ Korean font for plots

📮 문의 및 기여 | Contact & Contribution

본 프로젝트는 공모전 제출, 연구 발표, 프로토타입 용도로 개발된 비상업적 데모입니다.
기술 기여, 피드백, 공동연구 또는 제품화 제안은 아래 방법으로 연락 주세요:

This project is non-commercial and built for academic & competition purposes.
We welcome feedback, contributions, or productization discussions:
	•	GitHub Issue 등록
	•	이메일 문의 (✉️ 프로젝트 팀에 직접 연락)

⸻

🔒 Note: All user inputs are processed in-memory. No personally identifiable information (PII) is collected or stored.
🧪 Disclaimer: This is a prototype for demo/research use only — not a licensed medical tool.

✅ 최종 수정일: 2025-05-24
© 2025 JUNIXION Team. All rights reserved.
---

필요 시 `.md` 파일 형태로 저장해드릴 수도 있고, PDF/한글 버전도 제작 가능합니다. 원하시면 말씀해 주세요!
