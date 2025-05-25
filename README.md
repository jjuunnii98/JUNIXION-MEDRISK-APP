## 🧠 JUNIXION MedRisk.AI

AI 기반 암환자 맞춤형 의료비 예측 및 보험사 추천 시스템

![Streamlit](https://img.shields.io/badge/Deployed%20with-Streamlit-orange)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-blue)
![Status](https://img.shields.io/badge/Status-Ready%20for%20submission-green)

---

## 🔍 프로젝트 개요
**JUNIXION MedRisk.AI**는 암환자 개인의 기본정보를 바탕으로 예상 의료비를 예측하고, 소득 대비 의료비 부담률에 따른 위험등급을 산정하여, 사용자 맞춤형 보험사를 추천하는 **AI 기반 헬스케어 플랫폼**입니다.

## 🚀 배포 주소
👉 [실행하러 가기](https://junixion-medrisk.streamlit.app/)

## 📂 주요 기능
- ✅ XGBoost 기반 의료비 예측
- ✅ 병원유형 보정계수 적용
- ✅ 연소득 대비 의료비 부담률 계산
- ✅ 위험등급 5단계 분류 + 게이지 시각화
- ✅ SHAP 해석 시각화 (모델 기여도 분석)
- ✅ 사용자 입력 이력 누적 → 히트맵 시각화
- ✅ 위험도에 따른 보험사 추천 알고리즘
- ✅ 피드백 입력 기능
- ✅ 모바일 대응 레이아웃 (Streamlit Wide 설정)

## 📊 사용 기술 스택
- **Backend**: Python, XGBoost, pandas
- **Frontend**: Streamlit
- **Visualization**: matplotlib, seaborn, plotly, SHAP
- **ML Explainability**: SHAP
- **Deployment**: Streamlit Cloud
- **Version Control**: GitHub

## 📁 디렉토리 구조
```bash
junixion_medrisk_app/
├── app.py                  # 메인 Streamlit 앱
├── requirements.txt        # 라이브러리 목록
├── model/
│   ├── xgb_model.json      # 학습된 XGBoost 모델
├── user_logs/
│   ├── risk_history.csv    # 사용자 위험도 기록
├── data/
│   └── ...                 # 암종, 병원유형 등 데이터셋
