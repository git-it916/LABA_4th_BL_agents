import os
import glob
import pandas as pd
import json
import numpy as np

from aiportfolio.util.sector_mapping import map_gics_sector_to_code
from aiportfolio.util.making_rollingdate import get_rolling_dates, get_backtest_dates
from aiportfolio.BL_MVO.BL_params.market_params import Market_Params
from aiportfolio.BL_MVO.MVO_opt import MVO_Optimizer

# !!!!!!!!!! 일별데이터 전처리 완료되면 의존성 수정해야함
from aiportfolio.backtest.preprocessing import final_abnormal_returns

class backtest():
    def __init__(self, simul_name, Tier, forecast_period, backtest_days_count):
        self.simul_name = simul_name
        self.Tier = Tier
        self.forecast_period = forecast_period
        self.backtest_days_count = backtest_days_count

    def open_BL_MVO_log(self):
        """
        로그 디렉토리에서 BL_MVO.json 파일을 찾아,
        모든 월의 데이터를 포함하는 long-format DataFrame으로 반환합니다.
        """
        log_dir = f'database/logs/Tier{self.Tier}/result_of_BL-MVO'
        list_of_files = glob.glob(os.path.join(log_dir, f'{self.simul_name}.json'))

        if not list_of_files:
            print(f"오류: '{log_dir}' 디렉토리에서 로그 파일을 찾을 수 없습니다.")
            return None

        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"BL 로그 파일 사용: {latest_file}")

        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            all_data = []
            for record in data:
                # forecast_date를 그대로 사용 (날짜 변환 제거)
                forecast_date = pd.to_datetime(record['forecast_date'])
                weights = [float(w.strip('%')) / 100.0 for w in record['w_aiportfolio']]

                # BL 로그는 영어 섹터 이름을 포함
                sectors_english = record['SECTOR']

                # --- [수정된 부분] ---
                # 영어 섹터 이름을 GICS 숫자 코드로 변환
                try:
                    numeric_sectors = map_gics_sector_to_code(sectors_english)
                except KeyError as e:
                    # 맵핑 실패 시 오류 출력 후 해당 레코드를 건너뜀
                    print(f"  !오류! open_BL_MVO_log에서 {forecast_date.date()} BL 로그 GICS 맵핑 실패: {e}")
                    continue # 이 record 처리를 건너뛰고 다음 record로 이동
                # --------------------

                # 숫자 코드(numeric_sectors)를 사용
                # ForecastDate 컬럼으로 변경 (InvestmentMonth 대신)
                for sector, weight in zip(numeric_sectors, weights):
                    all_data.append({
                        'ForecastDate': forecast_date,  # 변경: forecast_date 그대로 사용
                        'SECTOR': sector, # 숫자 코드가 저장됨
                        'Weight': weight
                    })

            if not all_data:
                print("오류: open_BL_MVO_log에서 JSON 파일 내용은 있으나 처리된 데이터가 없습니다.")
                return None

            return pd.DataFrame(all_data)

        except Exception as e:
            print(f"open_BL_MVO_log 처리 중 오류 발생: {e}")
            return None

    def get_NONE_view_BL_weight(self):
        """
        Black-Litterman 프레임워크를 사용하되 뷰가 없는 상태(P=0)로 MVO와 동일한 결과를 산출합니다.

        이 함수는 BL 모델에서 P=0 (뷰 없음)으로 설정하여 μ_BL = π (시장 균형 수익률)을 얻고,
        이를 기반으로 Sharpe Ratio 최적화를 수행합니다. 수학적으로 순수 MVO와 동일한 결과를 보장합니다.

        Returns:
            pd.DataFrame: Long 형식 가중치
                - ForecastDate: 예측 기준일
                - SECTOR: GICS 섹터 코드
                - Weight: 가중치
        """
        forecast_dates = get_rolling_dates(self.forecast_period)

        all_data = []

        for period in forecast_dates:
            start_date = period['start_date']
            end_date = period['end_date']
            forecast_date = period['forecast_date']

            print(f"  - {forecast_date.date()} MVO 가중치 계산 중...")

            try:
                # BL 변수 생성
                market_params = Market_Params(start_date, end_date)
                Pi = market_params.making_pi()      # Equilibrium excess returns (π)
                sigma = market_params.making_sigma()  # Covariance matrix (Σ)
                sigma_for_optimize = market_params.making_sigma_for_optimize()
                sectors = sigma[1]
                num_sectors = len(sectors)

                P = np.zeros((1, num_sectors))
                Q = np.zeros((1, 1))
                Omega = np.eye(1)
                tau = 0.000000000000000001

                # --- Execute the Black-Litterman formula ---
                pi_np = (Pi.values.flatten() if isinstance(Pi, pd.DataFrame) else Pi.flatten()).reshape(-1, 1)
                sigma_np = sigma[0].values if isinstance(sigma[0], pd.DataFrame) else sigma[0]

                # Calculate intermediate terms
                tau_sigma_inv = np.linalg.inv(tau * sigma_np)
                omega_inv = np.linalg.inv(Omega)
                PT_omega_inv = P.T @ omega_inv

                # Term A: [ (τΣ)^(-1) + P^T·Ω^(-1)·P ]
                term_A = tau_sigma_inv + PT_omega_inv @ P

                # Term B: [ (τΣ)^(-1)·π + P^T·Ω^(-1)·Q ]
                term_B_part1 = tau_sigma_inv @ pi_np
                term_B_part2 = PT_omega_inv @ Q
                term_B = term_B_part1 + term_B_part2

                # Calculate posterior expected returns (μ_BL)
                mu_BL = np.linalg.inv(term_A) @ term_B

                # [검증] μ_BL ≈ π 확인
                diff = np.max(np.abs(mu_BL.flatten() - pi_np.flatten()))
                if diff > 0.001:
                    print(f"    [경고] μ_BL과 π의 차이: {diff:.6f}")

                # MVO 실행
                mvo = MVO_Optimizer(mu_BL.flatten(), sigma_for_optimize[0], sectors)
                w_tan, sectors = mvo.optimize_tangency_1()

                # w_tan을 1차원 배열로 변환
                w_tan_flat = w_tan.flatten()

                # Long 형식으로 변환
                for sector, weight in zip(sectors, w_tan_flat):
                    all_data.append({
                        'ForecastDate': forecast_date,
                        'SECTOR': sector,  # GICS 숫자 코드
                        'Weight': weight
                    })

            except Exception as e:
                print(f"  !오류! {forecast_date.date()} MVO 계산 실패: {e}")
                continue

        if not all_data:
            print("가중치 계산에 모두 실패했습니다.")
            return None

        return pd.DataFrame(all_data)

    def performance_of_portfolio(self, portfolio_weights, portfolio_name="Portfolio"):
        """
        여러 예측 기준일에 대해 포트폴리오의 백테스트 성과를 계산합니다.

        Args:
            portfolio_weights (pd.DataFrame): 포트폴리오 가중치
            portfolio_name (str): 포트폴리오 이름
                - 'AI_portfolio': LLM 뷰 + BL + MVO 최적화 결과
                - 'NONE_view': 뷰 없는 BL (P=0, 시장 균형 베이스라인)

        Returns:
            dict: 백테스트 성과 지표
        """
        # forecast_period를 리스트로 변환 (단일 날짜인 경우 대비)
        if not isinstance(self.forecast_period, list):
            forecast_period = [self.forecast_period]
        else:
            forecast_period = self.forecast_period

        # 일별 섹터별 초과수익률 데이터 로드 (한 번만 로드)
        daily_return_df = final_abnormal_returns()

        # date를 인덱스로 설정 (preprocessing.py는 'date' 컬럼 사용)
        if 'date' in daily_return_df.columns:
            daily_return_df = daily_return_df.set_index('date')
        elif 'DlyCalDt' in daily_return_df.columns:
            daily_return_df = daily_return_df.set_index('DlyCalDt')

        # 결과를 저장할 딕셔너리
        results = {}

        # 각 forecast_date에 대해 백테스트 수행
        for forecast_date in forecast_period:
            print(f"[{portfolio_name}] 백테스트 시작: {forecast_date}")

            try:
                # 백테스트 시작 날짜 계산
                backtest_start_date = get_backtest_dates(forecast_date)

                # 백테스트 시작일 이후의 데이터만 필터링
                available_dates = daily_return_df.index[daily_return_df.index >= backtest_start_date]

                if len(available_dates) == 0:
                    raise ValueError(f"[오류] {backtest_start_date.date()} 이후 데이터가 없습니다.")

                # backtest_days_count 만큼의 거래일 선택
                if len(available_dates) < self.backtest_days_count:
                    raise ValueError(
                        f"[오류] 요청한 {self.backtest_days_count}일보다 사용 가능한 데이터가 적습니다. "
                        f"(사용 가능한 데이터: {len(available_dates)}일)"
                    )

                # 백테스트 기간 데이터 추출
                backtest_period_dates = available_dates[:self.backtest_days_count]
                backtest_returns = daily_return_df.loc[backtest_period_dates]

                print(f"[알림] 백테스트 기간: {backtest_period_dates[0].date()} ~ {backtest_period_dates[-1].date()} ({len(backtest_period_dates)}일)")

                # forecast_date에 해당하는 포트폴리오 가중치 가져오기
                # 문자열인 경우 YY-MM-DD 형식으로 파싱
                if isinstance(forecast_date, str):
                    forecast_date_dt = pd.to_datetime(forecast_date, format='%y-%m-%d')
                else:
                    forecast_date_dt = pd.to_datetime(forecast_date)

                portfolio_weights_date = portfolio_weights[portfolio_weights['ForecastDate'] == forecast_date_dt]

                if portfolio_weights_date.empty:
                    print(f"[경고] {forecast_date_dt.date()}에 대한 포트폴리오 가중치가 없습니다. 건너뜁니다.")
                    continue

                # 가중치를 Series로 변환 (SECTOR를 인덱스로)
                weights = portfolio_weights_date.set_index('SECTOR')['Weight']

                # 포트폴리오 일별 수익률 계산
                portfolio_daily_returns = []

                for date in backtest_period_dates:
                    daily_sector_returns = backtest_returns.loc[date]

                    # 가중치와 수익률을 곱해서 포트폴리오 수익률 계산
                    portfolio_return = 0.0
                    for sector in weights.index:
                        if sector in daily_sector_returns.index:
                            portfolio_return += weights[sector] * daily_sector_returns[sector]

                    portfolio_daily_returns.append(portfolio_return)

                # 결과를 Series로 저장
                portfolio_returns_series = pd.Series(portfolio_daily_returns, index=backtest_period_dates)

                # 누적 수익률 계산
                cumulative_return = (1 + portfolio_returns_series).cumprod() - 1
                final_return = cumulative_return.iloc[-1]

                # 평균 일별 수익률
                avg_daily_return = portfolio_returns_series.mean()

                # 변동성 (표준편차)
                volatility = portfolio_returns_series.std()

                # 샤프 비율 (연율화, 무위험 수익률 0 가정)
                sharpe_ratio = (avg_daily_return / volatility) * (252 ** 0.5) if volatility != 0 else 0

                # 일별 누적 Sharpe Ratio 계산 (expanding window)
                # 각 영업일까지의 일별 수익률을 사용하여 Sharpe Ratio 계산
                cumulative_sharpe_ratios = []
                for i in range(1, len(portfolio_returns_series) + 1):
                    # i일까지의 일별 수익률
                    returns_so_far = portfolio_returns_series.iloc[:i]

                    # 평균과 표준편차 계산
                    mean_return = returns_so_far.mean()
                    std_return = returns_so_far.std()

                    # Sharpe Ratio 계산 (연율화)
                    if std_return != 0:
                        sharpe = (mean_return / std_return) * (252 ** 0.5)
                    else:
                        sharpe = 0.0

                    cumulative_sharpe_ratios.append(sharpe)

                # 결과 저장 (상세 정보 포함)
                # JSON 직렬화를 위해 키를 문자열로, pandas 객체를 리스트/문자열로 변환
                results[forecast_date_dt.strftime('%Y-%m-%d')] = {
                    'portfolio_name': portfolio_name,
                    'forecast_date': forecast_date_dt.strftime('%Y-%m-%d'),
                    'daily_returns': portfolio_returns_series.tolist(),
                    'cumulative_returns': cumulative_return.tolist(),
                    'cumulative_sharpe_ratios': cumulative_sharpe_ratios,
                    'final_return': float(final_return),
                    'avg_daily_return': float(avg_daily_return),
                    'volatility': float(volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'backtest_start': backtest_period_dates[0].strftime('%Y-%m-%d'),
                    'backtest_end': backtest_period_dates[-1].strftime('%Y-%m-%d'),
                    'backtest_days': len(backtest_period_dates)
                }

            except Exception as e:
                print(f"[오류] {forecast_date} 백테스트 실패: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"[{portfolio_name}] 백테스트 완료: {len(results)}/{len(forecast_period)}개 성공")
        print(f"{'='*60}\n")

        return results