import pandas as pd
import os

# python -m aiportfolio.agents.prepare.Tier3_calculate

def calculate_macro_indicator():
    file_path = os.path.join('database','Tier3.csv')

    data = pd.read_csv(file_path)

    # 컬럼명 변경
    data = data.rename(columns={
        'observation_date': 'date',
        'G20 CLI(Amplitude adjusted, Long-term average = 100)': 'G20_CLI'
    })

    # datetime 형식으로 변환
    data['date'] = pd.to_datetime(data['date'])

    # 월의 말일 기준으로 변경
    data['date'] = data['date'] + pd.offsets.MonthEnd(0)

    # 지표별 발표 지연(publication lag) 차등 적용
    # - FEDFUNDS, T10Y2Y: 실시간 시장 데이터 → lag 0 (rolling offset 1개월로 충분)
    # - CPI: BLS 발표 ~2-3주 지연 → lag 1개월
    # - GPDIC1_PCA: BEA GDP 속보치 ~1개월 지연 → lag 1개월
    # - G20_CLI: OECD 발표 ~2개월 지연 → lag 2개월
    publication_lag = {
        'CPI': 1,
        'GPDIC1_PCA': 1,
        'G20_CLI': 2,
    }
    for col, lag in publication_lag.items():
        if col in data.columns:
            data[col] = data[col].shift(lag)

    return data