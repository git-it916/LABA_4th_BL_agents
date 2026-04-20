import os
import gc
import torch

from .BL_MVO.BL_opt import get_bl_outputs
from .BL_MVO.MVO_opt import MVO_Optimizer
from .util.making_rollingdate import get_rolling_dates
from .util.sector_mapping import map_code_to_gics_sector
from .util.save_log_as_json import save_BL_as_json, save_performance_as_json
from aiportfolio.backtest.calculating_performance import backtest
from aiportfolio.backtest.visualization import calculate_average_cumulative_returns


def _clear_gpu_memory():
    """각 LLM 호출 후 GPU 메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def scene(simul_name, Tier, tau, forecast_period, backtest_days_count, model='llama', tier3_mode='macro'):
    """
    전체 시뮬레이션 실행 함수

    tier3_mode: 'macro' (기본, Tier3.csv 거시 지표) | 'regime' (REGIME_QMS 시장 레짐)
                Tier==3 일 때만 의미 있음.
    """
    # 결과를 저장할 디렉토리 생성
    base_dir = os.path.join("database", "logs")
    os.makedirs(base_dir, exist_ok=True)
    tier_dirs = ['Tier1', 'Tier2', 'Tier3']
    subdirs = ['result_of_BL-MVO', 'LLM-view', 'result_of_test']
    for tier in tier_dirs:
        path = os.path.join(base_dir, tier)
        os.makedirs(path, exist_ok=True)
        for subdir in subdirs:
            sub_path = os.path.join(path, subdir)
            os.makedirs(sub_path, exist_ok=True)

    # 학습기간 설정        
    forecast_date = get_rolling_dates(forecast_period)

    results = []

    # 기간별 BL -> MVO 수행
    for i, period in enumerate(forecast_date):

        print(f"--- [{i+1}/{len(forecast_date)}] forecast_date: {period['forecast_date']} ---")
        start_date = period['start_date']
        end_date = period['end_date']

        # BL 실행 (LLM 호출 포함)
        BL = get_bl_outputs(tau, start_date=start_date, end_date=end_date, simul_name=simul_name, Tier=Tier, model=model, tier3_mode=tier3_mode)

        # LLM 호출 후 GPU 메모리 정리
        _clear_gpu_memory()

        # MVO 실행
        mvo = MVO_Optimizer(mu=BL[0], sigma=BL[1], sectors=BL[2])
        w_tan = mvo.optimize_tangency_1()[0]

        # w_tan을 1차원 배열로 변환
        w_tan_flat = w_tan.flatten()

        # 결과 저장
        scenario_result = {
            "forecast_date": period['forecast_date'],
            "w_aiportfolio": [f"{weight * 100:.4f}%" for weight in w_tan_flat],
            "SECTOR": map_code_to_gics_sector(BL[2])
        }
        results.append(scenario_result)

        # 각 기간 완료 후 메모리 상태 출력
        if torch.cuda.is_available():
            print(f"    [메모리] GPU 사용량: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    save_BL_as_json(results, simul_name, Tier)

    test = backtest(simul_name, Tier, forecast_period, backtest_days_count)
    BL_result = test.open_BL_MVO_log()
    none_view_result = test.get_NONE_view_BL_weight()

    BL_backtest_result = test.performance_of_portfolio(BL_result, portfolio_name='AI_portfolio')
    save_performance_as_json(BL_backtest_result, simul_name, Tier)

    none_view_backtest_result = test.performance_of_portfolio(none_view_result, portfolio_name='NONE_view')
    save_performance_as_json(none_view_backtest_result, simul_name, Tier)

    calculate_average_cumulative_returns(simul_name, Tier)

    return results