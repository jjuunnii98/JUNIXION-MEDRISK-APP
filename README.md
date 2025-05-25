# 🧠 JUNIXION MedRisk.AI

**AI 기반 암환자 맞춤형 의료비 예측 및 보험사 추천 시스템**  
AI-Powered Medical Cost Prediction & Insurance Recommendation Platform for Cancer Patients

![Streamlit](https://img.shields.io/badge/Deployed%20with-Streamlit-orange)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-blue)
![Status](https://img.shields.io/badge/Status-Ready%20for%20submission-green)

---

## 🔍 프로젝트 개요 | Project Overview

**JUNIXION MedRisk.AI**는 공공 의료데이터와 보험통계를 기반으로  
개인의 암종, 성별, 연령, 소득, 병원유형, 거주지역 등의 입력을 통해  
예상 의료비를 **AI 모델(XGBoost)**로 예측하고,  
부담률 기반 위험등급을 도출하여 **적절한 보험사**를 추천하는  
**헬스케어 기반 AI 서비스 플랫폼**입니다.

**JUNIXION MedRisk.AI** is an AI-powered healthcare platform  
that predicts medical costs for cancer patients based on personal inputs  
(age, cancer type, income, region, hospital type) using **XGBoost**,  
and recommends insurance companies based on the risk level  
calculated from the burden-to-income ratio.

---

## 📊 데이터 출처 | Data Sources

| 데이터 종류 (Data Type)         | 출처 (Source)                                       | 설명 (Description)                                     |
|------------------------------|----------------------------------------------------|------------------------------------------------------|
| 암종별 진료비/입원일수       | 건강보험심사평가원 (HIRA)                           | 암종별 인당 진료비 및 입원일수 통계                    |
| 병원유형별 진료비            | 건강보험심사평가원 (HIRA)                           | 병원 유형에 따른 단가 보정계수                          |
| 생명보험 가입자 통계         | 공공데이터포털 (금융위원회 제공)                   | 보험사명, 가입자수, 보험사 구분 포함                   |
| 사용자 이력 로그             | Streamlit 상 사용자 입력값 기록                    | 위험등급 이력을 기반으로 히트맵 시각화에 활용            |

All data are sourced from **trusted public institutions** in Korea  
(e.g., [HIRA](https://www.hira.or.kr), [Data.go.kr](https://www.data.go.kr))  
and preprocessed for model learning and visualization.

---

## 🚀 배포 주소 | Demo App

🔗 [앱 실행 (Run on Streamlit)](https://junixion-medrisk.streamlit.app/)

---

## ✅ 주요 기능 | Key Features

- 🎯 **XGBoost 기반 의료비 예측 (XGBoost-Based Medical Cost Prediction)**
- 📊 병원 유형 보정계수 적용 (Hospital Cost Adjustment Factor)
- 💰 연소득 대비 부담률 계산 (Cost Burden Ratio Calculation)
- 🧭 5단계 위험등급 + 게이지 시각화 (5-Level Risk Grading + Gauge Visualization)
- 🔍 SHAP 모델 해석 시각화 (SHAP Explainability for AI Model)
- 🧬 사용자 입력 기반 히스토리 시각화 (User History Heatmap)
- 🛡️ 위험도 기반 보험사 추천 알고리즘 (Insurance Recommendation based on Risk Level)
- 💬 피드백 텍스트 저장 기능 (Feedback Input Logging)
- 📱 모바일 반응형 레이아웃 (Mobile Responsive UI)

---

## 🛠 기술 스택 | Tech Stack

| 영역 (Category)   | 기술 스택 (Stack)                           |
|------------------|---------------------------------------------|
| Backend          | Python, pandas, XGBoost                     |
| Frontend         | Streamlit                                   |
| Visualization    | matplotlib, seaborn, plotly, SHAP           |
| Model Explainability | SHAP                                   |
| Deployment       | Streamlit Cloud                             |
| Version Control  | GitHub                                      |

---

## 📁 디렉토리 구조 | Project Structure

```bash
junixion_medrisk_app/
├── app.py                    # 메인 Streamlit 앱 | Main application
├── requirements.txt          # 의존성 패키지 목록 | Python dependencies
├── model/
│   └── xgb_model.json        # 학습된 모델 | Trained model
├── data/
│   ├── t1.xlsx               # 암종별 통계 | Cancer cost data
│   ├── t2.xlsx               # 소득/연령 통계 | Income/age data
│   ├── t3.xlsx               # 병원 유형 데이터 | Hospital cost stats
│   └── life_insurance_general.json  # 보험사 가입자 통계 | Insurance statistics
├── user_logs/
│   └── risk_history.csv      # 사용자 위험 이력 | User log

📮 문의 및 기여 | Contact & Contribution

이 프로젝트는 연구 및 공모전 제출을 위한 목적으로 개발되었습니다.
프로젝트 개선, 제휴 또는 연구 협업 문의는 GitHub Issue 또는 이메일로 연락 바랍니다.

This project is designed for public health research and competition submission.
For collaborations or improvements, feel free to open an issue or contact the maintainer.

