# 📁 utils.py (위험등급 5단계 + 게이지 점수 맵 + 보험사 추천 + 사용자 이력 저장 기능 추가)

import pandas as pd
import json
import os
import csv
from datetime import datetime

# ✅ 위험등급 점수 맵 (게이지 차트 시각화용)
risk_score_map = {
    "매우 낮음": 10,
    "낮음": 25,
    "보통": 50,
    "높음": 75,
    "매우 높음": 90
}

def load_data_sources_safe():
    """
    암종, 소득, 병원유형, 보험사 데이터 로드 (예외처리 강화)
    """
    try:
        df_t1 = pd.read_excel("/Volumes/JUNIXION/2.Contest_list/2.Healthcare_Startup/document/t1.xlsx")
        df_t2 = pd.read_excel("/Volumes/JUNIXION/2.Contest_list/2.Healthcare_Startup/document/df_t2.xlsx")
        df_t3 = pd.read_excel("/Volumes/JUNIXION/2.Contest_list/2.Healthcare_Startup/document/t3.xlsx")

        with open("/Volumes/JUNIXION/2.Contest_list/2.Healthcare_Startup/document/life_insurance_general.json", "r", encoding="utf-8") as f:
            raw_json = json.load(f)

        try:
            items = raw_json["response"]["body"]["tableList"][0]["items"]["item"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"[JSON 구조 오류] 예상 키 없음: {e}")

        insurance_df = pd.DataFrame(items)

        insurance_df = insurance_df.rename(columns={
            "fncoNm": "보험사명",
            "xcsmPlnpnCnt": "인원수",
            "xcsmPlnpnDcdNm": "구분"
        })

        insurance_df = insurance_df[insurance_df["보험사명"].str.contains("보험", na=False)].copy()
        insurance_df["인원수"] = pd.to_numeric(insurance_df["인원수"], errors="coerce").fillna(0)

        def assign_scale(count):
            if count >= 3000:
                return "대형"
            elif count >= 1000:
                return "중형"
            else:
                return "소형"

        insurance_df["보험사규모"] = insurance_df["인원수"].apply(assign_scale)
        insurance_df = insurance_df.drop_duplicates(subset=["보험사명"])

        return df_t1, df_t2, df_t3, insurance_df

    except Exception as e:
        raise RuntimeError(f"[load_data_sources 오류] {e}")

def recommend_insurance_company(risk_level: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    위험등급(5단계)에 따라 보험사 추천 목록 반환
    """
    if "보험사규모" not in df.columns:
        raise KeyError("❌ 보험사규모 컬럼이 없습니다. 실제 컬럼: " + ", ".join(df.columns))

    if risk_level in ["매우 낮음", "낮음"]:
        return df[df["보험사규모"].isin(["소형", "중형"])].copy()
    elif risk_level in ["보통", "높음"]:
        return df[df["보험사규모"].isin(["중형", "대형"])].copy()
    elif risk_level == "매우 높음":
        return df[df["보험사규모"] == "대형"].copy()
    else:
        raise ValueError(f"❌ 알 수 없는 위험등급: '{risk_level}'")


# ✅ 사용자 이력 위험도 누적 저장 함수
def log_risk_score(region: str, age_group: str, risk_score: int, path: str = "data/risk_history.csv"):
    """
    지역, 연령대, 위험 점수를 risk_history.csv에 누적 저장
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if "/" in path else None

    write_header = not os.path.exists(path)
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "region", "age_group", "risk_score"])
        writer.writerow([datetime.now().isoformat(), region, age_group, risk_score])