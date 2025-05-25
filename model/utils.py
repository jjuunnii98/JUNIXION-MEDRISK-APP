# 📁 utils.py (보험사 추천 개선 + 추천 필터 컬럼 자동 생성 + 누적 기록 기능)

import pandas as pd
import json
import os
import csv
from datetime import datetime

# ✅ 위험등급 점수 맵 (게이지 차트용)
risk_score_map = {
    "매우 낮음": 10,
    "낮음": 25,
    "보통": 50,
    "높음": 75,
    "매우 높음": 90
}

def load_data_sources_safe():
    """
    암종, 소득, 병원유형, 보험사 데이터 로드
    (예외처리 포함 + 추천정보 컬럼 자동 추가)
    """
    try:
        # ✅ 상대 경로로 로드
        df_t1 = pd.read_excel("data/t1.xlsx")
        df_t2 = pd.read_excel("data/t2.xlsx")
        df_t3 = pd.read_excel("data/t3.xlsx")

        with open("data/life_insurance_general.json", "r", encoding="utf-8") as f:
            raw_json = json.load(f)

        items = raw_json["response"]["body"]["tableList"][0]["items"]["item"]
        insurance_df = pd.DataFrame(items)

        # ✅ 컬럼명 정리 및 필터링
        insurance_df = insurance_df.rename(columns={
            "fncoNm": "보험사명",
            "xcsmPlnpnCnt": "인원수",
            "xcsmPlnpnDcdNm": "구분"
        })
        insurance_df = insurance_df[insurance_df["보험사명"].str.contains("보험", na=False)].copy()
        insurance_df["인원수"] = pd.to_numeric(insurance_df["인원수"], errors="coerce").fillna(0)

        # ✅ 보험사 규모 분류
        def assign_scale(count):
            if count >= 3000:
                return "대형"
            elif count >= 1000:
                return "중형"
            else:
                return "소형"

        insurance_df["보험사규모"] = insurance_df["인원수"].apply(assign_scale)
        insurance_df = insurance_df.drop_duplicates(subset=["보험사명"])

        # ✅ 추천 관련 컬럼이 없으면 생성
        if "보장유형" not in insurance_df.columns:
            insurance_df["보장유형"] = "암 전용"
        if "평균보험료" not in insurance_df.columns:
            insurance_df["평균보험료"] = "중간"
        if "모바일가입" not in insurance_df.columns:
            insurance_df["모바일가입"] = True
        if "민원률" not in insurance_df.columns:
            insurance_df["민원률"] = 1.2  # 기본 평균 민원률

        return df_t1, df_t2, df_t3, insurance_df

    except Exception as e:
        raise RuntimeError(f"[load_data_sources 오류] {e}")


def recommend_insurance_company(risk_level: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    위험등급에 따라 보험사 추천 목록 반환 (정렬 기준 포함)
    """
    if "보험사규모" not in df.columns:
        raise KeyError("❌ 보험사규모 컬럼이 없습니다. 실제 컬럼: " + ", ".join(df.columns))

    if risk_level in ["매우 낮음", "낮음"]:
        filtered = df[df["보험사규모"].isin(["소형", "중형"])].copy()
    elif risk_level in ["보통", "높음"]:
        filtered = df[df["보험사규모"].isin(["중형", "대형"])].copy()
    elif risk_level == "매우 높음":
        filtered = df[df["보험사규모"] == "대형"].copy()
    else:
        raise ValueError(f"❌ 알 수 없는 위험등급: '{risk_level}'")

    # ✅ 우선순위 정렬: 모바일가입 → 보험사규모 → 인원수
    filtered = filtered.sort_values(
        by=["모바일가입", "보험사규모", "인원수"],
        ascending=[False, False, False]
    )

    return filtered.reset_index(drop=True)


def log_risk_score(region: str, age_group: str, risk_score: int, path: str = "data/risk_history.csv"):
    """
    지역 + 연령대 기반 위험도 기록 (누적 저장)
    """
    if "/" in path:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    write_header = not os.path.exists(path)
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "region", "age_group", "risk_score"])
        writer.writerow([datetime.now().isoformat(), region, age_group, risk_score])