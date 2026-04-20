import numpy as np
import pandas as pd

from aiportfolio.agents.Llama_config import prepare_pipeline_obj
from aiportfolio.agents.Llama_view_generator import generate_sector_views
from aiportfolio.agents.converting_viewtomatrix import open_view_log, create_Q_vector, create_P_matrix

def get_view_params(sigma, tau, end_date, simul_name, Tier, model='llama', tier3_mode='macro'):
    """
    This function calculates and returns the view-related parameters P, Q, and Omega.

    Args:
        sigma (pd.DataFrame): The covariance matrix of asset returns.
        tau (float): A scalar indicating the uncertainty in the prior estimate.

    Returns:
        tuple: A tuple containing P, Q, and Omega.
    """
    # LLM으로 뷰 생성
    if model == 'llama':
        # GPU 사용 가능 여부 확인 (Llama만 필요)
        import torch
        if not torch.cuda.is_available():
            print("\n" + "="*80)
            print("[치명적 오류] GPU를 사용할 수 없습니다.")
            print("="*80)
            print("이 프로그램은 Llama 3 모델을 사용하여 섹터 뷰를 생성합니다.")
            print("Llama 3 8B 모델은 GPU 없이는 실행할 수 없습니다.")
            print("\n해결 방법:")
            print("1. NVIDIA GPU가 설치된 시스템에서 실행하세요.")
            print("2. CUDA가 올바르게 설치되었는지 확인하세요.")
            print("3. PyTorch가 CUDA 버전으로 설치되었는지 확인하세요:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("="*80 + "\n")
            raise RuntimeError("GPU를 사용할 수 없어 프로그램을 중단합니다.")

        pipeline_to_use = prepare_pipeline_obj()
    else:
        # Gemini는 pipeline 불필요
        pipeline_to_use = None

    generate_sector_views(pipeline_to_use, end_date, simul_name, Tier, model, tier3_mode=tier3_mode)
    views_data = open_view_log(simul_name=simul_name, Tier=Tier, end_date=end_date)

    if views_data is None:
        raise ValueError(f"Failed to load views data for end_date={end_date}")

    # --- Picking matrix (P) ---
    P = create_P_matrix(views_data)

    # --- View vector (Q) ---
    Q = create_Q_vector(views_data)

    # --- Omega matrix (Ω) ---
    # Calculate view uncertainty matrix (diagonal)
    # Formula: Ω_ii = τ × P_i × Σ × P_i^T
    # Reference: He & Litterman (1999)
    num_views = P.shape[0]
    Omega = np.zeros((num_views, num_views))
    sigma_np = sigma.values if isinstance(sigma, pd.DataFrame) else sigma

    for i in range(num_views):
        P_row = P[i, :]
        p_sigma_pT = P_row @ sigma_np @ P_row.T
        omega_i = tau * p_sigma_pT
        Omega[i, i] = omega_i
    Omega = Omega / 1

    print('\n=== View Parameters ===')
    print('P (Picking Matrix):')
    print(P)
    print('\nQ (View Vector):')
    print(Q)
    print('\nΩ (Omega - View Uncertainty Matrix):')
    print(Omega)

    return P, Q, Omega