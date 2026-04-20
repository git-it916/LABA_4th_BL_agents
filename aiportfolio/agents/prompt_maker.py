"""
개선된 프롬프트 생성 시스템

주요 개선사항:
1. 소수점 2자리로 제한 (가독성 향상)
2. Tier별 가변 프롬프트 (일관성 유지하면서 필요한 부분만 변경)
3. 불필요한 메타데이터 제거
4. 더 깔끔한 JSON 출력
5. [New] Tier 2 데이터 Parquet 로드 방식 적용 (속도 개선)
"""
# python -m aiportfolio.agents.prompt_maker

import pandas as pd
import numpy as np
import os
import json
from aiportfolio.agents.prepare.Tier1_calculate import indicator
# Tier2 계산 함수는 파일이 없을 때를 대비해 import 유지
from aiportfolio.agents.prepare.Tier2_calculate import calculate_accounting_indicator
from aiportfolio.agents.prepare.Tier3_calculate import calculate_macro_indicator
from aiportfolio.agents.prepare.Tier3_regime import calculate_regime_indicator

# ==========================================================
# [설정] 데이터베이스 경로
# ==========================================================
# 프로젝트 루트 디렉토리 기준 상대 경로 사용 (사용자 환경 독립적)
import os as _os
# __file__은 aiportfolio/agents/prompt_maker.py
# 프로젝트 루트는 3단계 위 (agents -> aiportfolio -> LABA_BL_AGENTS_FINAL)
_CURRENT_FILE_DIR = _os.path.dirname(_os.path.abspath(__file__))  # .../aiportfolio/agents
_AIPORTFOLIO_DIR = _os.path.dirname(_CURRENT_FILE_DIR)           # .../aiportfolio
_PROJECT_ROOT = _os.path.dirname(_AIPORTFOLIO_DIR)               # .../LABA_BL_AGENTS_FINAL
BASE_PATH_DB = _os.path.join(_PROJECT_ROOT, "database")
TIER2_PARQUET_FILE = "tier2_accounting_metrics.parquet"

def round_numeric_values(data, decimals=2):
    """
    딕셔너리의 모든 숫자 값을 지정된 소수점 자리로 반올림
    """
    if isinstance(data, dict):
        return {k: round_numeric_values(v, decimals) for k, v in data.items()}
    elif isinstance(data, list):
        return [round_numeric_values(item, decimals) for item in data]
    elif isinstance(data, (float, np.floating)):
        return round(float(data), decimals)
    elif isinstance(data, (int, np.integer)):
        return int(data)
    elif isinstance(data, str):
        try:
            if data.startswith('[') and data.endswith(']'):
                parsed = eval(data)
                if isinstance(parsed, list):
                    rounded = [round(float(x), decimals) if isinstance(x, (int, float)) else x for x in parsed]
                    return str(rounded)
        except:
            pass
        return data
    else:
        return data


def making_tier1_INPUT(end_date):
    """
    Tier 1 (기술적 지표) 데이터 생성
    """
    data = indicator()

    def safe_get_value(sector, column):
        filtered = data.loc[(data['date'] == end_date) & (data['gsector'] == sector), column]
        if len(filtered) == 0:
            print(f"[경고] {sector} 섹터의 {column} 데이터가 {end_date}에 없습니다. 'N/A'로 대체합니다.")
            return "N/A"
        value = filtered.iloc[0]

        if isinstance(value, list):
            return [round(float(x), 4) for x in value]
        elif isinstance(value, (int, float, np.number)):
            return round(float(value), 4)
        else:
            return value

    sectors = [
        "Energy", "Materials", "Industrials", "Consumer Discretionary",
        "Consumer Staples", "Health Care", "Financials", "Information Technology",
        "Communication Services", "Utilities", "Real Estate"
    ]

    sector_data_list = []
    for sector in sectors:
        sector_data_list.append({
            "sector": sector,
            "ttm_returns": safe_get_value(sector, 'return_list'),
            "z_score": safe_get_value(sector, 'z-score'),
            "volatility": safe_get_value(sector, 'volatility'),
            "trend_strength": safe_get_value(sector, 'trend_strength'),
            "cagr_3y": safe_get_value(sector, 'CAGR')
        })

    return sector_data_list


def making_tier2_INPUT(end_date):
    """
    Tier 2 (회계 지표) 데이터 생성
    [수정됨] 매번 계산하지 않고 Parquet 파일을 읽어서 처리
    """
    parquet_path = os.path.join(BASE_PATH_DB, TIER2_PARQUET_FILE)
    data = pd.DataFrame()

    # 1. Parquet 파일 로드 시도
    if os.path.exists(parquet_path):
        try:
            # print(f"[INFO] Tier 2 데이터를 파일에서 로드합니다: {parquet_path}") # 디버그용 출력 (필요시 주석 해제)
            data = pd.read_parquet(parquet_path)
            
            # 날짜 형식 보장 (문자열로 저장되었을 경우를 대비)
            if not pd.api.types.is_datetime64_any_dtype(data['date']):
                data['date'] = pd.to_datetime(data['date'])
                
        except Exception as e:
            print(f"[오류] Parquet 파일 로드 실패: {e}")
            data = pd.DataFrame() # 로드 실패 시 빈 DF
    else:
        print(f"[알림] Parquet 파일이 없습니다. ({parquet_path})")

    # 2. 파일이 없거나 로드 실패 시 직접 계산 (Fallback)
    if data.empty:
        print("[알림] Tier 2 데이터를 실시간으로 계산합니다...")
        data = calculate_accounting_indicator()

    # --- 이하 로직은 동일 ---
    sectors = [
        "Energy", "Materials", "Industrials", "Consumer Discretionary",
        "Consumer Staples", "Health Care", "Financials", "Information Technology",
        "Communication Services", "Utilities", "Real Estate"
    ]

    # [안전장치] 데이터가 여전히 비어있을 경우 예외 처리
    if data.empty:
        print("[오류] Tier 2 데이터 생성 실패. 모든 값을 'N/A'로 반환합니다.")
        return [{
            "sector": sector, "bm_Median": "N/A", "CAPEI_Median": "N/A", "GProf_Median": "N/A",
            "npm_Median": "N/A", "roa_Median": "N/A", "roe_Median": "N/A", "totdebt_invcap_Median": "N/A"
        } for sector in sectors]

    def safe_get_metric_value(sector, metric_name):
        """
        Long Format 데이터에서 값 추출
        data columns: ['date', 'gsector', 'metric', 'acct_level_lagged_avg']
        """
        filtered = data.loc[
            (data['date'] == end_date) &
            (data['gsector'] == sector) &
            (data['metric'] == metric_name),
            'acct_level_lagged_avg'
        ]

        if len(filtered) == 0:
            return "N/A"

        value = filtered.iloc[0]

        if isinstance(value, (int, float, np.number)):
            return round(float(value), 2)
        else:
            return value

    # 최종 리스트 생성 (Median 사용)
    sector_data_list = []
    for sector in sectors:
        sector_data_list.append({
            "sector": sector,
            "bm_Median": safe_get_metric_value(sector, 'bm_Median'),
            "CAPEI_Median": safe_get_metric_value(sector, 'CAPEI_Median'),
            "GProf_Median": safe_get_metric_value(sector, 'GProf_Median'),
            "npm_Median": safe_get_metric_value(sector, 'npm_Median'),
            "roa_Median": safe_get_metric_value(sector, 'roa_Median'),
            "roe_Median": safe_get_metric_value(sector, 'roe_Median'),
            "totdebt_invcap_Median": safe_get_metric_value(sector, 'totdebt_invcap_Median')
        })

    return sector_data_list


def making_tier3_INPUT(end_date):
    """
    Tier 3 (거시 지표) 데이터 생성
    """
    data = calculate_macro_indicator()

    def safe_get_value(column):
        filtered = data.loc[(data['date'] == end_date), column]
        if len(filtered) == 0:
            print(f"[경고] {column} 데이터가 {end_date}에 없습니다. 'N/A'로 대체합니다.")
            return "N/A"
        value = filtered.iloc[0]

        if isinstance(value, (int, float, np.number)):
            return round(float(value), 2)
        else:
            return value

    macro_data = {
        "date": str(end_date.date()) if hasattr(end_date, 'date') else str(end_date),
        "FEDFUNDS": safe_get_value('FEDFUNDS'),
        "CPI": safe_get_value('CPI'),
        "G20_CLI": safe_get_value('G20_CLI'),
        "T10Y2Y": safe_get_value('T10Y2Y'),
        "GPDIC1_PCA": safe_get_value('GPDIC1_PCA')
    }

    return macro_data


def making_tier3_regime_INPUT(end_date):
    """
    Tier 3 (시장 레짐) 데이터 생성 — REGIME_QMS.csv 기반
    macro 지표 대신 DRAI/GROWTH/INFLATION 3개 regime 의 STATES + 확률을 반환.
    """
    return calculate_regime_indicator(end_date)


def load_tier_guidelines(tier, tier3_mode='macro'):
    """
    Tier별 분석 가이드라인 로드 (누적 방식).
    tier3_mode='regime' 이면 Tier 3 섹션을 '3 GUIDELINES (REGIME VERSION)' 블록으로 교체한다.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, 'prompt_template', 'tier_guidelines.txt')

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        sections = content.split('### TIER ')
        included_tiers = list(range(1, tier + 1))
        tier_sections = []

        for target_tier in included_tiers:
            # Tier 3 은 모드별로 다른 헤더를 매칭 (파일 내 블록 배치 순서와 독립적)
            if target_tier == 3:
                if tier3_mode == 'regime':
                    target_header = '3 GUIDELINES (REGIME VERSION)'
                else:
                    target_header = '3 GUIDELINES (Technical'
            else:
                target_header = f'{target_tier} GUIDELINES'

            for section in sections:
                if section.startswith(target_header):
                    end_marker = '\n### TIER '
                    if end_marker in section:
                        section = section[:section.index(end_marker)]

                    if '### COMMON RULES' in section:
                        section = section[:section.index('### COMMON RULES')]

                    tier_sections.append(f'### TIER {section.strip()}')
                    break

        if not tier_sections:
            print(f"[경고] Tier {tier} 가이드라인을 찾을 수 없습니다.")
            return f"[Tier {tier} analysis guidelines not found]"

        combined_guidelines = '\n\n---\n\n'.join(tier_sections)

        if '### COMMON RULES' in content:
            common_start = content.index('### COMMON RULES')
            common_section = content[common_start:]
            combined_guidelines += '\n\n---\n\n' + common_section

        return combined_guidelines

    except FileNotFoundError:
        print(f"[오류] 가이드라인 파일을 찾을 수 없습니다: {file_path}")
        return f"[Tier {tier} guidelines file not found]"
    except Exception as e:
        print(f"[오류] 가이드라인 로드 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return f"[Error loading Tier {tier} guidelines]"


def making_system_prompt(tier, tier3_mode='macro'):
    """
    Tier별 시스템 프롬프트 생성
    tier3_mode: 'macro' (기본, 기존 거시 지표) | 'regime' (REGIME_QMS 사용)
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(base_path, 'prompt_template', 'system_prompt.txt')

    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()

        # regime 모드일 땐 역할 설명의 거시지표 예시도 레짐 예시로 치환
        if tier3_mode == 'regime':
            template = template.replace(
                "Macroeconomic conditions (e.g., FEDFUNDS, CPI, G20_CLI, T10Y2Y, GPDIC1_PCA)",
                "Market regime states (DRAI risk regime, MACRO growth regime, MACRO inflation regime, each with HMM posterior probabilities)"
            )

        tier_guidelines = load_tier_guidelines(tier, tier3_mode=tier3_mode)

        prompt = template.replace('{{TIER}}', str(tier))
        prompt = prompt.replace('{{TIER_SPECIFIC_GUIDELINES}}', tier_guidelines)

        return prompt

    except FileNotFoundError:
        print(f"[오류] 시스템 프롬프트 템플릿을 찾을 수 없습니다: {template_path}")
        return None
    except Exception as e:
        print(f"[오류] 시스템 프롬프트 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def making_user_prompt(end_date, tier, tier3_mode='macro'):
    """
    Tier별 사용자 프롬프트 생성
    tier3_mode: 'macro' (기본) | 'regime' (REGIME_QMS 기반)
    """
    # 날짜 타입 디버깅
    print(f"[디버그] making_user_prompt: end_date={end_date}, type={type(end_date)}, tier={tier}, tier3_mode={tier3_mode}")

    base_path = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(base_path, 'prompt_template', 'user_prompt.txt')

    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()

        data_blocks = []

        # Tier 1: 기술적 지표 (항상 포함)
        print(f"[디버그] Calling making_tier1_INPUT with end_date={end_date}")
        tier1_data = making_tier1_INPUT(end_date)
        tier1_json = json.dumps(tier1_data, indent=2, ensure_ascii=False)
        data_blocks.append(f"=== Technical Indicators (Tier 1) ===\n{tier1_json}")

        # Tier 2: 회계 지표 추가
        if tier >= 2:
            tier2_data = making_tier2_INPUT(end_date)
            tier2_json = json.dumps(tier2_data, indent=2, ensure_ascii=False)
            data_blocks.append(f"\n=== Accounting Indicators (Tier 2) ===\n{tier2_json}")

        # Tier 3: 거시 지표 or 시장 레짐
        if tier >= 3:
            if tier3_mode == 'regime':
                tier3_data = making_tier3_regime_INPUT(end_date)
                tier3_json = json.dumps(tier3_data, indent=2, ensure_ascii=False)
                data_blocks.append(f"\n=== Market Regime (Tier 3) ===\n{tier3_json}")
            else:
                tier3_data = making_tier3_INPUT(end_date)
                tier3_json = json.dumps(tier3_data, indent=2, ensure_ascii=False)
                data_blocks.append(f"\n=== Macro Indicators (Tier 3) ===\n{tier3_json}")

        combined_data = '\n'.join(data_blocks)

        prompt = template.replace('{{TIER}}', str(tier))
        prompt = prompt.replace('{{DATA_BLOCKS}}', combined_data)

        return prompt

    except FileNotFoundError:
        print(f"[오류] 사용자 프롬프트 템플릿을 찾을 수 없습니다: {template_path}")
        return None
    except Exception as e:
        print(f"[오류] 사용자 프롬프트 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


# 테스트 코드
if __name__ == "__main__":
    test_date = pd.Timestamp('2024-05-31')

    # 가이드라인 테스트
    tier_guidelines1 = load_tier_guidelines(tier=1)
    
    print('### tier1 Guidelines ###')
    print(tier_guidelines1[:200] + "...") 
    
    print("\n" + "="*80)
    print("Tier 1 프롬프트 생성 테스트")
    print("="*80)
    user_prompt = making_user_prompt(end_date=test_date, tier=1)
    if user_prompt:
        print(user_prompt[:500] + "...\n")

    print("\n" + "="*80)
    print("Tier 2 프롬프트 생성 테스트 (Parquet 파일 사용)")
    print("="*80)
    user_prompt_2 = making_user_prompt(end_date=test_date, tier=2)
    if user_prompt_2:
        start_idx = user_prompt_2.find("=== Accounting Indicators")
        if start_idx != -1:
            print(user_prompt_2[start_idx:start_idx+800] + "...\n")
        else:
            print("Accounting Indicators not found in prompt!")

    print("\n" + "="*80)
    print("Tier 3 프롬프트 생성 테스트")
    print("="*80)
    user_prompt_3 = making_user_prompt(end_date=test_date, tier=3)
    if user_prompt_3:
        print("Tier 3 generated successfully (length: {})".format(len(user_prompt_3)))