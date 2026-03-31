import pandas as pd

from aiportfolio.util.data_load.open_DTB3 import open_rf_rate
from aiportfolio.util.data_load.open_final_stock_daily import open_final_stock_daily
from aiportfolio.util.data_load.open_final_stock_months import open_final_stock_months

# python -m aiportfolio.backtest.preprocessing

# ---------- 1) 일별 무위험수익률 제작 ----------
# https://fred.stlouisfed.org/series/DTB3
# 3-Month Treasury Bill Secondary Market Rate, Discount Basis (DTB3) 사용
def preprocess_rf_rate():
    df_rf = open_rf_rate()

    # 컬럼명 변경
    df_rf.rename(columns={'observation_date': 'date', 'DTB3': 'rf_daily'}, inplace=True)

    # 날짜 형식 변환
    df_rf['date'] = pd.to_datetime(df_rf['date'])

    # 무위험 수익률 전처리
    df_rf['rf_daily'] = df_rf['rf_daily'].ffill() # 휴일 때문에 발생하는 결측치 바로 전날의 값으로 채움.
    df_rf['rf_daily'] = df_rf['rf_daily'] / 100  # % -> 소수점 변환
    df_rf['rf_daily'] = (1 + df_rf['rf_daily']) ** (1/252) - 1  # 연율 -> 일율 변환
    
    return df_rf

# --- 2) 포트폴리오 가중치 산출에 이용된 종목들만 필터링하기 위한 더미 데이터프레임 ---
def filtering_dummy():
    df_month = open_final_stock_months()

    df_month['date'] = pd.to_datetime(df_month['cyear'].astype(str) + '-' + df_month['cmonth'].astype(str)) + pd.offsets.MonthEnd(0)

    df_month = df_month.sort_values(['Ticker', 'date'])
    df_month['sp500_lag1'] = df_month.groupby('Ticker')['sp500'].shift(1) # 전 월 sp500 값

    df_sp = df_month[df_month['sp500_lag1']==1].copy()

    df_sp['month'] = df_sp['date'].dt.to_period('M')
    flag_tickers = df_sp[['month', 'Ticker']].drop_duplicates()

    return flag_tickers

# ---------- 3) 일별 s&p 500의 초과수익률 제작 ----------
def sector_daily_returns():
    df = open_final_stock_daily()

    df.rename(columns={'DlyCalDt': 'date'}, inplace=True) # 컬럼명 변경
    df["date"] = pd.to_datetime(df["date"]) # 날짜 형식 변환

    # 가중치 계산에 사용된 종목만 필터링(여기부터 결측치X)
    df['month'] = df['date'].dt.to_period('M')
    a = filtering_dummy()
    daily_filtered = df.merge(a, on=['month','Ticker'])

    # 초과수익률 계산
    df_rf = preprocess_rf_rate()
    merged_df = pd.merge(daily_filtered, df_rf, on='date', how='left') # 종목 데이터 기준으로 병합
    merged_df['excess_return'] = merged_df['DlyRet'] - merged_df['rf_daily'] # 일별 초과수익률 계산

    merged_df['prev_DlyCap'] = merged_df.groupby('Ticker')['DlyCap'].shift(1)

    # 가중수익률 계산
    merged_df["_ret_x_cap"] = merged_df["excess_return"] * merged_df["prev_DlyCap"]

    group_keys = ['date', 'gsector']
    agg = (
    merged_df.groupby(group_keys, dropna=False)
            .agg(sector_prevmktcap=("prev_DlyCap", "sum"),
                ret_x_cap_sum=("_ret_x_cap", "sum"),
                n_stocks=("Ticker", "count"))
            .reset_index())

    mask = agg["sector_prevmktcap"] != 0
    agg["sector_excess_return"] = agg["ret_x_cap_sum"].div(agg["sector_prevmktcap"]).where(mask)

    # 중간열 제거 및 정리
    agg = agg.drop(columns=["ret_x_cap_sum"])
    agg = agg.sort_values(['date', 'gsector']).reset_index(drop=True).copy()
    
    return agg

# ---------- 4) 일별 market의 초과수익률 제작 ----------
def total_daily_returns():
    df = open_final_stock_daily()

    df.rename(columns={'DlyCalDt': 'date'}, inplace=True) # 컬럼명 변경
    df["date"] = pd.to_datetime(df["date"]) # 날짜 형식 변환

    # 날짜 중복되는 종목 처리(시가총액이 가장 큰 것 선택)
    filtered_df = (
        df.sort_values(['Ticker', 'date', 'DlyCap'], ascending=[True, True, False])
        .drop_duplicates(subset=['Ticker', 'date'], keep='first')
        .sort_index()
        .reset_index(drop=True)
    )

    # 초과수익률 계산
    df_rf = preprocess_rf_rate()
    merged_df = pd.merge(filtered_df, df_rf, on='date', how='left') # 종목 데이터 기준으로 병합
    merged_df['excess_return'] = merged_df['DlyRet'] - merged_df['rf_daily'] # 일별 초과수익률 계산

    merged_df['prev_DlyCap'] = merged_df.groupby('Ticker')['DlyCap'].shift(1)

    merged_df["_ret_x_cap"] = merged_df["excess_return"] * merged_df["prev_DlyCap"]
    agg = merged_df.groupby("date").agg(
        total_mktcap=("prev_DlyCap", "sum"),
        total_ret_x_cap=("_ret_x_cap", "sum")
    ).reset_index()

    # 분모 0 방지
    mask = agg["total_mktcap"] != 0
    agg["total_excess_return"] = agg["total_ret_x_cap"].div(agg["total_mktcap"]).where(mask)

    return agg

# ---------- 5) abnormal return 제작 ----------
def final_abnormal_returns():
    a = sector_daily_returns()
    b = total_daily_returns()
    merged_df = pd.merge(a, b, on='date', how='inner')

    merged_df['abnormal_return'] = merged_df['sector_excess_return'] - merged_df['total_excess_return']

    pivoted_df = merged_df.pivot(index='date', columns='gsector', values='abnormal_return')
    df_reset = pivoted_df.reset_index()

    return df_reset