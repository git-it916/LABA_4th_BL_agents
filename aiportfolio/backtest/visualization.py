import os
import json
import numpy as np

from aiportfolio.util.save_log_as_json import save_performance_as_json

def calculate_average_cumulative_returns(simul_name, Tier):
    """
    JSON 파일에서 백테스트 결과를 불러와 포트폴리오별 영업일별 평균 누적 수익률을 계산하고 시각화합니다.

    Args:
        simul_name (str): 시뮬레이션 이름
        Tier (int): 분석 단계 (1, 2, 3)

    Returns:
        dict: 포트폴리오별 평균 누적 수익률 정보
            - 'AI_portfolio': LLM 뷰 + BL + MVO 결과
            - 'NONE_view': 뷰 없는 BL (베이스라인) 결과
    """
    # 1. JSON 파일 경로 생성 및 로드
    base_log_dir = os.path.join("database", "logs")
    tier_dir_name = f"Tier{Tier}"
    save_dir = os.path.join(base_log_dir, tier_dir_name, 'result_of_test')
    filename = f'{simul_name}.json'
    filepath = os.path.join(save_dir, filename)

    print(f"\n{'='*80}")
    print(f"백테스트 결과 분석 시작: {simul_name} (Tier {Tier})")
    print(f"{'='*80}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[1/4] JSON 파일 로드 완료: {filepath}")
    except FileNotFoundError:
        print(f"[오류] 파일을 찾을 수 없습니다: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"[오류] JSON 파일 파싱 실패: {e}")
        return None

    # 리스트 형식 확인
    if not isinstance(data, list):
        data = [data]

    # 2. 포트폴리오별로 데이터 분리
    print(f"[2/4] 포트폴리오별 데이터 분리 중...")

    portfolios = {}  # {'AI_portfolio': {...}, 'NONE_view': {...}}

    for item in data:
        for date, result_data in item.items():
            portfolio_name = result_data.get('portfolio_name')

            # 하위 호환성: 'MVO' → 'NONE_view' 자동 변환
            if portfolio_name == 'MVO':
                portfolio_name = 'NONE_view'

            # 요약 데이터(portfolio_name 없음) 건너뛰기
            if portfolio_name is None:
                continue

            if portfolio_name not in portfolios:
                portfolios[portfolio_name] = {}

            portfolios[portfolio_name][date] = result_data

    print(f"      발견된 포트폴리오: {list(portfolios.keys())}")
    for pf_name, pf_data in portfolios.items():
        print(f"      - {pf_name}: {len(pf_data)}개 forecast_date")

    # 3. 각 포트폴리오별 영업일별 평균 계산
    print(f"[3/4] 포트폴리오별 영업일별 평균 누적 수익률 계산 중...")

    results = {}

    for portfolio_name, portfolio_data in portfolios.items():
        if not portfolio_data:
            print(f"      [경고] '{portfolio_name}' 데이터가 비어있습니다.")
            continue

        # cumulative_returns 및 cumulative_sharpe_ratios 수집
        all_cumulative_returns = []
        all_sharpe_ratios = []
        backtest_days_set = set()

        for forecast_date, result_data in portfolio_data.items():
            cumulative_returns = result_data['cumulative_returns']
            sharpe_ratios = result_data.get('cumulative_sharpe_ratios', [])  # 없으면 빈 리스트
            backtest_days = result_data['backtest_days']

            all_cumulative_returns.append(cumulative_returns)
            all_sharpe_ratios.append(sharpe_ratios)
            backtest_days_set.add(backtest_days)

        # 백테스트 기간 일치 확인
        if len(backtest_days_set) > 1:
            print(f"      [경고] '{portfolio_name}' 백테스트 기간 불일치: {backtest_days_set}")
            min_days = min(backtest_days_set)
            all_cumulative_returns = [returns[:min_days] for returns in all_cumulative_returns]
            all_sharpe_ratios = [sharpe[:min_days] for sharpe in all_sharpe_ratios if sharpe]
        else:
            min_days = list(backtest_days_set)[0]

        # 영업일별 평균 계산
        returns_array = np.array(all_cumulative_returns)  # (num_periods, backtest_days)
        avg_cumulative_returns = returns_array.mean(axis=0)  # (backtest_days,)

        # Sharpe Ratio 평균 계산 (데이터가 있는 경우만)
        avg_sharpe_ratios = []
        if all_sharpe_ratios and all(sharpe for sharpe in all_sharpe_ratios):
            sharpe_array = np.array(all_sharpe_ratios)  # (num_periods, backtest_days)
            avg_sharpe_ratios = sharpe_array.mean(axis=0).tolist()  # (backtest_days,)

        results[portfolio_name] = {
            'avg_cumulative_returns': avg_cumulative_returns.tolist(),
            'avg_sharpe_ratios': avg_sharpe_ratios,
            'num_periods': len(portfolio_data),
            'backtest_days': min_days,
            'final_avg_cumulative_return': float(avg_cumulative_returns[-1])
        }

        print(f"      ✓ {portfolio_name}: {len(portfolio_data)}개 기간, {min_days}일 평균 계산 완료")

    save_performance_as_json(results, simul_name, Tier)

    # 4. 결과 시각화
    print(f"\n[4/4] 결과 시각화")
    print(f"\n{'='*80}")
    print(f"포트폴리오별 영업일별 평균 누적 수익률")
    print(f"{'='*80}\n")

    # 포트폴리오별 요약
    for portfolio_name, result in results.items():
        print(f"■ {portfolio_name}")
        print(f"  ├─ 분석 기간 수: {result['num_periods']}개")
        print(f"  ├─ 백테스트 영업일: {result['backtest_days']}일")
        print(f"  └─ 평균 최종 누적 수익률: {result['final_avg_cumulative_return']*100:.2f}%\n")

    # 간단한 텍스트 차트
    if len(results) == 2:
        portfolio_names = list(results.keys())
        pf1_name, pf2_name = portfolio_names[0], portfolio_names[1]
        pf1_returns = results[pf1_name]['avg_cumulative_returns']
        pf2_returns = results[pf2_name]['avg_cumulative_returns']
        pf1_sharpe = results[pf1_name]['avg_sharpe_ratios']
        pf2_sharpe = results[pf2_name]['avg_sharpe_ratios']

        print(f"{'='*80}")
        print(f"영업일별 누적 수익률 비교 (매일)")
        print(f"{'='*80}")
        print(f"{'영업일':>6} | {pf1_name:>12} | {pf2_name:>12} | {'차이':>12}")
        print(f"{'-'*80}")

        backtest_days = results[pf1_name]['backtest_days']
        for day in range(backtest_days):
            pf1_val = pf1_returns[day] * 100
            pf2_val = pf2_returns[day] * 100
            diff = pf1_val - pf2_val

            print(f"{day+1:>6}일 | {pf1_val:>11.2f}% | {pf2_val:>11.2f}% | {diff:>+11.2f}%")

        # Sharpe Ratio 비교 (데이터가 있는 경우만)
        if pf1_sharpe and pf2_sharpe:
            print(f"\n{'='*80}")
            print(f"해당 영업일 까지의 Sharpe Ratio 비교 (매일, 연율화)")
            print(f"{'='*80}")
            print(f"{'영업일':>6} | {pf1_name:>12} | {pf2_name:>12} | {'차이':>12}")
            print(f"{'-'*80}")

            for day in range(backtest_days):
                pf1_sr = pf1_sharpe[day]
                pf2_sr = pf2_sharpe[day]
                diff_sr = pf1_sr - pf2_sr

                print(f"{day+1:>6}일 | {pf1_sr:>12.4f} | {pf2_sr:>12.4f} | {diff_sr:>+12.4f}")

    print(f"\n{'='*80}")
    print(f"분석 완료")
    print(f"{'='*80}\n")

    return results

