import pandas as pd
import numpy as np
import os

# ==========================================================
# 전역 경로 설정 (사용자 환경에 맞게 수정 필요)
# ==========================================================
BASE_PATH_COMPUSTAT = r"C:\Users\shins\OneDrive\문서"
BASE_PATH_REPO = r"C:\Users\shins\OneDrive\LABA_BL_AGENTS_F\LABA_BL_AGENTS_FINAL\database"

# ==========================================================
# 1) S&P500 플래그 추가
# ==========================================================
def add_sp500_flag(compustat_df: pd.DataFrame, base_path: str) -> pd.DataFrame:
    sp500_path = os.path.join(base_path, "sp500_ticker_start_end.csv")
    if not os.path.exists(sp500_path):
        print(f"[오류] S&P500 파일을 찾을 수 없습니다: {sp500_path}")
        return compustat_df

    sp500_df = pd.read_csv(sp500_path)
    sp500_df["start_date"] = pd.to_datetime(sp500_df["start_date"])
    sp500_df["end_date"]   = pd.to_datetime(sp500_df["end_date"])
    
    # end_date 없는 종목(현재 편입중) 처리
    sp500_df["end_date"] = sp500_df["end_date"].fillna(pd.Timestamp("2099-12-31"))

    compustat_df["Ticker"] = compustat_df["Ticker"].astype(str).str.upper().str.strip()
    sp500_df["Ticker"]     = sp500_df["Ticker"].astype(str).str.upper().str.strip()

    merged = compustat_df.merge(sp500_df, on="Ticker", how="left")
    merged["public_date"] = pd.to_datetime(merged["public_date"])

    merged["sp500"] = (
        (merged["public_date"] >= merged["start_date"]) &
        (merged["public_date"] <= merged["end_date"])
    ).astype(int)

    return merged.drop(columns=["start_date", "end_date"])

# ==========================================================
# 2) GICS 섹터 매핑
# ==========================================================
def add_gics_sector_for_sp500(df: pd.DataFrame, base_path: str):
    gics_path = os.path.join(base_path, "ticker_GICS.csv")
    if not os.path.exists(gics_path):
        print(f"[오류] GICS 파일을 찾을 수 없습니다: {gics_path}")
        return df, pd.DataFrame()

    gics = pd.read_csv(gics_path)
    gics["Ticker"]   = gics["Ticker"].astype(str).str.upper().str.strip()
    gics["datadate"] = pd.to_datetime(gics["datadate"])

    gics_sorted = gics.sort_values(["Ticker", "datadate"])
    gics_latest = gics_sorted.groupby("Ticker").tail(1).copy()

    df = df.copy()
    df["Ticker"]      = df["Ticker"].astype(str).str.upper().str.strip()
    df["public_date"] = pd.to_datetime(df["public_date"])

    df_merged = df.merge(
        gics_latest[["Ticker", "gsector"]],
        on="Ticker",
        how="left"
    )
    
    # 디버깅용 로그: 실제 섹터 값이 숫자인지 문자인지 확인
    unique_sectors = df_merged['gsector'].dropna().unique()
    print(f"[DEBUG] CSV에 있는 섹터 값 예시: {unique_sectors[:5]}")

    sp500_mask = df_merged["sp500"] == 1
    unmatched_sp500 = df_merged[sp500_mask & df_merged["gsector"].isna()].copy()

    return df_merged, unmatched_sp500

# ==========================================================
# 3) 섹터 × 연 × 월별 평균 계산
# ==========================================================
def calculate_sector_monthly_average(df: pd.DataFrame, metric_cols=None) -> pd.DataFrame:
    df = df.copy()
    filtered = df[(df["sp500"] == 1) & (df["gsector"].notna())].copy()

    if filtered.empty:
        return pd.DataFrame()

    if metric_cols is None:
        metric_cols = filtered.select_dtypes(include=["float32", "float64", "int32", "int64"]).columns.tolist()
        metric_cols = [c for c in metric_cols if c not in ["sp500", "year", "month"]]

    # public_date는 Compustat 공시일(filing date)이므로 1개월 안전 버퍼만 적용
    filtered = filtered.sort_values(["Ticker", "public_date"])
    for col in (metric_cols or []):
        if col in filtered.columns:
            filtered[col] = filtered.groupby("Ticker")[col].shift(1)

    # 모든 메트릭이 NaN이 된 행은 제거
    if metric_cols:
        filtered = filtered.dropna(subset=metric_cols, how="all")
    if filtered.empty:
        return pd.DataFrame()

    filtered["public_date"] = pd.to_datetime(filtered["public_date"])
    filtered["year"]  = filtered["public_date"].dt.year
    filtered["month"] = filtered["public_date"].dt.month

    grouped = (
        filtered
        .groupby(["gsector", "year", "month"])[metric_cols]
        .median()
        .reset_index()
        .sort_values(["gsector", "year", "month"])
    )
    return grouped

# ==========================================================
# 4) Prompt Maker용 통합 함수 (View 생성 및 Parquet 저장)
# ==========================================================
def calculate_accounting_indicator():
    print("[INFO] Tier 2 회계 지표 계산 시작...")
    
    # 1. Compustat 로드 (여러 후보 파일 중 존재하는 것 사용)
    comp_candidates = [
        os.path.join(BASE_PATH_COMPUSTAT, "compustat_2021.01_2024.12_company.csv"),
        os.path.join(BASE_PATH_COMPUSTAT, "compustat_2021.05_2024.12_company.csv"),
    ]
    comp_path = next((p for p in comp_candidates if os.path.exists(p)), None)
    if comp_path is None:
        print("[ERROR] Compustat 원본 파일을 찾지 못했습니다.")
        return pd.DataFrame()

    compustat_df = pd.read_csv(comp_path)
    if "Ticker" not in compustat_df.columns and "TICKER" in compustat_df.columns:
        compustat_df = compustat_df.rename(columns={"TICKER": "Ticker"})

    # 2. S&P500 및 GICS 매핑
    compustat_df = add_sp500_flag(compustat_df, BASE_PATH_REPO)
    df_with_gics, _ = add_gics_sector_for_sp500(compustat_df, BASE_PATH_REPO)

    # 3. 섹터 median 계산
    metric_cols = ["bm", "npm", "roe", "roa", "CAPEI", "GProf", "totdebt_invcap"]
    available_metrics = [c for c in metric_cols if c in df_with_gics.columns]
    
    df_avg = calculate_sector_monthly_average(df_with_gics, available_metrics)

    if df_avg.empty:
        return pd.DataFrame(columns=['date', 'gsector', 'metric', 'acct_level_lagged_avg'])

    # -----------------------------------------------------------
    # [수정] GICS 숫자 코드를 영어 이름으로 변환 (Mapping)
    # -----------------------------------------------------------
    gics_map = {
        10: "Energy", 
        15: "Materials", 
        20: "Industrials",
        25: "Consumer Discretionary", 
        30: "Consumer Staples",
        35: "Health Care", 
        40: "Financials", 
        45: "Information Technology",
        50: "Communication Services", 
        55: "Utilities", 
        60: "Real Estate"
    }
    
    # 섹터 컬럼이 숫자형(int, float)일 경우 매핑 적용
    if pd.api.types.is_numeric_dtype(df_avg['gsector']):
        print("[INFO] 섹터 코드를 이름으로 변환합니다 (예: 45 -> Information Technology)")
        df_avg['gsector'] = df_avg['gsector'].map(gics_map)
    else:
        print("[INFO] 섹터가 이미 문자열 형식이므로 매핑을 건너뜁니다.")

    # 4. Prompt Maker 호환 포맷 변환 (Wide -> Long)
    df_avg['date'] = pd.to_datetime(df_avg[['year', 'month']].assign(day=1)) + pd.offsets.MonthEnd(0)
    
    id_vars = ['date', 'gsector']
    value_vars = available_metrics
    
    df_long = df_avg.melt(
        id_vars=id_vars, 
        value_vars=value_vars, 
        var_name='metric', 
        value_name='acct_level_lagged_avg'
    )

    df_long['metric'] = df_long['metric'].astype(str) + '_Median'

    print(f"[INFO] Tier 2 계산 완료. 데이터 형태: {df_long.shape}")
    print(f"[DEBUG] 최종 데이터에 포함된 섹터 목록: {df_long['gsector'].unique()}")

    # -----------------------------------------------------------
    # [NEW] 최종 View를 Parquet 파일로 저장
    # -----------------------------------------------------------
    output_filename = "tier2_accounting_metrics.parquet"
    output_path = os.path.join(BASE_PATH_REPO, output_filename)
    
    try:
        # engine='pyarrow' 사용 (설치 필요: pip install pyarrow)
        df_long.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"[SAVED] Tier 2 View 저장 완료: {output_path}")
    except ImportError:
        print("[오류] pyarrow가 설치되어 있지 않습니다. Parquet 저장을 건너뜁니다.")
        print("설치 명령어: pip install pyarrow")
    except Exception as e:
        print(f"[오류] Parquet 저장 중 에러 발생: {e}")

    return df_long

if __name__ == "__main__":
    result = calculate_accounting_indicator()
    print(result.head())
