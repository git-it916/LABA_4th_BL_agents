from aiportfolio.scene import scene
from aiportfolio.agents.Llama_config import cleanup_pipeline

######################################
#            configuration           #
######################################

simul_name_base = '0331_rep_'

Tier1_repetition_count = 10
Tier2_repetition_count = 10
Tier3_repetition_count = 10

tau = 0.025
model = 'llama'  # 'llama' or 'gemini'

forecast_period = [
        "24-05-31",
        "24-06-30",
        "24-07-31",
        "24-08-31",
        "24-09-30",
        "24-10-31",
        "24-11-30",
        "24-12-31"
    ]

backtest_days_count = 19

######################################
#                run                 #
######################################

# GPU 메모리 정리는 scene.py 내부에서 각 LLM 호출마다 수행됨

failed_simulations = []  # 실패한 시뮬레이션 기록

if Tier1_repetition_count >= 1:
    for i in range(1, Tier1_repetition_count + 1):
        simul_name = simul_name_base + 'Tier1_' + f'{i}'
        try:
            scene(simul_name, 1, tau, forecast_period, backtest_days_count, model)
        except Exception as e:
            print(f"[오류] {simul_name} 실패: {e}")
            failed_simulations.append(simul_name)
            continue

if Tier2_repetition_count >= 1:
    for i in range(1, Tier2_repetition_count + 1):
        simul_name = simul_name_base + 'Tier2_' + f'{i}'
        try:
            scene(simul_name, 2, tau, forecast_period, backtest_days_count, model)
        except Exception as e:
            print(f"[오류] {simul_name} 실패: {e}")
            failed_simulations.append(simul_name)
            continue

if Tier3_repetition_count >= 1:
    for i in range(1, Tier3_repetition_count + 1):
        simul_name = simul_name_base + 'Tier3_' + f'{i}'
        try:
            scene(simul_name, 3, tau, forecast_period, backtest_days_count, model)
        except Exception as e:
            print(f"[오류] {simul_name} 실패: {e}")
            failed_simulations.append(simul_name)
            continue

# 모든 시뮬레이션 완료 후 파이프라인 완전 해제
cleanup_pipeline()

# 결과 요약
print("\n" + "="*50)
print("[완료] 모든 시뮬레이션 종료 및 메모리 해제 완료")
if failed_simulations:
    print(f"[경고] 실패한 시뮬레이션 ({len(failed_simulations)}개):")
    for name in failed_simulations:
        print(f"  - {name}")
else:
    print("[성공] 모든 시뮬레이션이 정상 완료되었습니다.")