import os
import pandas as pd

# python -m aiportfolio.agents.prepare.Tier3_regime

# REGIME_QMS.csv 에서 사용할 RegimeName 고정 순서
REGIME_NAMES = ("DRAI_Multi", "MACRO_GROWTH_Multi", "MACRO_INFLATION_Multi")


def _load_regime_raw() -> pd.DataFrame:
    """
    REGIME_QMS.csv를 로드하고 날짜 컬럼을 datetime으로 변환한다.
    - BOM 대응: 첫 컬럼명이 '\ufeffID' 로 읽힐 수 있음
    """
    file_path = os.path.join("database", "REGIME_QMS.csv")
    df = pd.read_csv(file_path, encoding="utf-8-sig")

    # 컬럼명 정규화 (BOM 제거 & 공백 제거)
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]

    df["BaseDate"] = pd.to_datetime(df["BaseDate"])
    df["LookBackDate"] = pd.to_datetime(df["LookBackDate"])
    return df


def calculate_regime_indicator(end_date) -> dict:
    """
    end_date 시점의 시장 레짐 상태를 반환한다.

    필터링 규칙 (look-ahead 방지):
        1) LookBackDate <= end_date
        2) BaseDate      <= end_date 가 존재하면 그 중 최신, 없으면 전체에서 최신
        3) 각 RegimeName 별로 LookBackDate 가 end_date 에 가장 가까운 행을 선택

    Returns:
        {
            "date": "YYYY-MM-DD",
            "DRAI_Multi": {"state": -1|0|1, "prob_positive":..., "prob_neutral":..., "prob_negative":...},
            "MACRO_GROWTH_Multi": {...},
            "MACRO_INFLATION_Multi": {...}
        }
    """
    end_date = pd.to_datetime(end_date)
    df = _load_regime_raw()

    # BaseDate 선택: end_date 이하 중 최신, 없으면 전체에서 최신
    base_candidates = df.loc[df["BaseDate"] <= end_date, "BaseDate"]
    if len(base_candidates) > 0:
        chosen_base = base_candidates.max()
    else:
        chosen_base = df["BaseDate"].max()

    df = df[(df["BaseDate"] == chosen_base) & (df["LookBackDate"] <= end_date)]

    regime_data: dict = {
        "date": end_date.strftime("%Y-%m-%d"),
        "regime_as_of": chosen_base.strftime("%Y-%m-%d"),
    }

    for name in REGIME_NAMES:
        sub = df[df["RegimeName"] == name]
        if sub.empty:
            print(f"[경고] {name} 레짐 데이터가 {end_date.date()} 이전에 없습니다. N/A 반환.")
            regime_data[name] = {
                "state": "N/A",
                "prob_positive": "N/A",
                "prob_neutral": "N/A",
                "prob_negative": "N/A",
            }
            continue

        # LookBackDate 최신 1행
        row = sub.loc[sub["LookBackDate"].idxmax()]
        regime_data[name] = {
            "state": int(row["STATES"]),
            "prob_positive": round(float(row["PROB_POSITIVE"]), 4),
            "prob_neutral": round(float(row["PROB_NEUTRAL"]), 4),
            "prob_negative": round(float(row["PROB_NEGATIVE"]), 4),
            "lookback_date": row["LookBackDate"].strftime("%Y-%m-%d"),
        }

    return regime_data


if __name__ == "__main__":
    import json

    for test_date in ["2024-05-31", "2024-08-31", "2024-12-31"]:
        print(f"\n=== {test_date} ===")
        print(json.dumps(calculate_regime_indicator(test_date), indent=2, ensure_ascii=False))
