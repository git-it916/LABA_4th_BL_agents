import numpy as np
import pandas as pd
from datetime import datetime

# Import parameters from the BL_params directory
from aiportfolio.BL_MVO.BL_params.market_params import Market_Params
from aiportfolio.BL_MVO.BL_params.view_params import get_view_params

def get_bl_outputs(tau, start_date, end_date, simul_name=None, Tier=None, model='llama', tier3_mode='macro'):
    """
    Execute the Black-Litterman model to compute posterior expected returns and covariance.

    Theoretical Foundation:
        Black & Litterman (1992) "Global Portfolio Optimization"
        He & Litterman (1999) "The Intuition Behind Black-Litterman Model Portfolios"

    Formula:
        μ_BL = [(τΣ)^(-1) + P^T·Ω^(-1)·P]^(-1) × [(τΣ)^(-1)·π + P^T·Ω^(-1)·Q]
        Σ_BL = [(τΣ)^(-1) + P^T·Ω^(-1)·P]^(-1)

    Args:  
        tau (float): Black-Litterman 불확실성 계수 (일반적으로 0.01~0.05)
        start_date (datetime): 시작 날짜
        end_date (datetime): 종료 날짜 (예측 기준일)
        simul_name (str, optional): 시뮬레이션 이름
        Tier (int, optional): 분석 단계 (1, 2, 3)

    Returns:
        tuple: (mu_BL, Sigma_BL, sectors)
            - mu_BL (np.ndarray): 사후 기대수익률 벡터 (N×1)
            - Sigma_BL (pd.DataFrame): 사후 공분산 행렬 (N×N)
            - sectors (list): 섹터 리스트
    """
    # BL 변수 생성
    market_params = Market_Params(start_date, end_date)
    Pi = market_params.making_pi()      # Equilibrium excess returns (π)
    sigma = market_params.making_sigma()  # Covariance matrix (Σ)
    sigma_for_optimize = market_params.making_sigma_for_optimize()

    P, Q, Omega = get_view_params(sigma[0], tau, end_date, simul_name, Tier, model, tier3_mode=tier3_mode)

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

    # --- Return the outputs for the MVO script ---
    sectors = sigma[1]

    print('P (Picking Matrix)')
    print(P)
    print('\nQ (View Vector)')
    print(Q)
    print('\nπ (Equilibrium Excess Returns)')
    print(Pi)
    print('\nμ_BL (Posterior Expected Returns)')
    print(mu_BL)

    return mu_BL.reshape(-1, 1), sigma_for_optimize[0], sectors