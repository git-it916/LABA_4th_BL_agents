from aiportfolio.scene import scene

######################################
#            configuration           #
######################################

simul_name = '0331_1'
Tier = 3
tau = 0.025
model = 'llama'  # 'llama' or 'gemini'
tier3_mode = 'macro'  # 'macro' (Tier3.csv 거시 지표) | 'regime' (REGIME_QMS 시장 레짐)

'''
forecast_period = [
        "24-05-31",
        "24-06-30",
        "24-07-31"
]
'''
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

scene(simul_name, Tier, tau, forecast_period, backtest_days_count, model, tier3_mode=tier3_mode)

