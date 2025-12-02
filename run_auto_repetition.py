from aiportfolio.scene import scene

######################################
#            configuration           #
######################################

simul_name_base = 'gogoletsgo_'

Tier1_repetition_count = 30
Tier2_repetition_count = 30
Tier3_repetition_count = 30

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

if Tier1_repetition_count >= 1:
    for i in range(1,Tier1_repetition_count+1):
        simul_name = simul_name_base + 'Tier1_' + f'{i}'
        scene(simul_name, 1, tau, forecast_period, backtest_days_count, model)

if Tier2_repetition_count >= 1:
    for i in range(1,Tier2_repetition_count+1):
        simul_name = simul_name_base + 'Tier2_' + f'{i}'
        scene(simul_name, 2, tau, forecast_period, backtest_days_count, model)

if Tier3_repetition_count >= 1:
    for i in range(1,Tier3_repetition_count+1):
        simul_name = simul_name_base + 'Tier3_' + f'{i}'
        scene(simul_name, 3, tau, forecast_period, backtest_days_count, model)