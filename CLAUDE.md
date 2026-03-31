# LABA_BL_AGENTS_FINAL

> LLM 에이전트를 Black-Litterman 모델의 투자자 견해(views)에 통합하는 포트폴리오 최적화 시스템
> 연구기관: LABA (Lab for Accounting Big Data & Artificial Intelligence)

## 프로젝트 개요

Llama 3 (또는 Gemini) LLM이 섹터 간 상대적 수익률 전망을 자동 생성하고, Black-Litterman 모델에 통합하여 포트폴리오를 최적화한다.

**포트폴리오 비교 구조:**
- **AI_portfolio**: LLM 뷰 + BL + MVO 최적화
- **NONE_view (베이스라인)**: 뷰 없는 BL (P=0, 시장 균형 수익률 π 그대로 사용) - 수학적으로 순수 MVO와 동등

**3단계 데이터 분석 (Tier):**
- Tier 1: 기술적 지표 (CAGR, 모멘텀, 변동성, 추세강도, z-score)
- Tier 2: 회계 지표 (B/M, ROE, ROA, npm, CAPEI, GProf, 부채비율)
- Tier 3: 거시 지표 (FEDFUNDS, CPI, G20_CLI, T10Y2Y, GPDIC1_PCA)

## 핵심 구조

```
aiportfolio/
  agents/
    prepare/
      Tier1_calculate.py        # 기술적 지표 계산
      Tier2_calculate.py        # 회계 지표 (Parquet 캐시 사용)
      Tier3_calculate.py        # 거시 지표 (database/Tier3.csv)
    Llama_config_수정중.py      # LLM 파이프라인 (Llama3 + Gemini)
    Llama_view_generator.py     # 뷰 생성 오케스트레이션
    prompt_maker_improved.py    # 프롬프트 생성 (실제 사용)
    converting_viewtomatrix.py  # 뷰 → P, Q 행렬 변환
    prompt_template/            # 시스템/사용자 프롬프트 템플릿
  BL_MVO/
    BL_params/
      market_params.py          # 시장 매개변수 (π, Σ, λ, w_mkt)
      view_params.py            # 뷰 매개변수 (P, Q, Ω) + GPU 필수 체크
      prepare/
        sector_excess_return.py # 초과수익률 전처리
    BL_opt.py                   # BL 사후 수익률 (μ_BL, Σ_BL)
    MVO_opt.py                  # Sharpe Ratio 최대화 (SLSQP)
  backtest/
    calculating_performance.py  # 백테스트 메인 클래스
    data_prepare.py             # 가중치 데이터 로드
    preprocessing_2차수정.py    # 일별 섹터 수익률
    visalization.py             # 평균 누적수익률 집계
  util/
    sector_mapping.py           # GICS 코드 ↔ 섹터명
    making_rollingdate.py       # 롤링 기간 생성
    save_log_as_json.py         # JSON 결과 저장
    data_load/                  # 데이터 로더 모듈
  scene.py                      # 메인 오케스트레이션

run_single.py                   # 단일 시뮬레이션 실행
run_auto_repetition.py          # 반복 시뮬레이션 실행
final_visualization.py          # 최종 결과 시각화
statistical_analysis.py         # 통계 분석
```

## 실행 흐름

```
scene(simul_name, Tier, tau, forecast_period, backtest_days_count, model)
  ├─ 기간별 반복:
  │   ├─ get_bl_outputs() → LLM 뷰 생성 → BL 공식 적용 → μ_BL
  │   ├─ MVO_Optimizer.optimize_tangency_1() → 최적 가중치
  │   └─ save_BL_as_json() → 결과 저장
  ├─ backtest() → AI_portfolio + NONE_view 성과 비교
  └─ calculate_average_cumulative_returns() → 시각화
```

## 핵심 수식

**BL 사후 수익률:**
```
μ_BL = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹ × [(τΣ)⁻¹ π + Pᵀ Ω⁻¹ Q]
Σ_BL = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹
```

- π = λ × Σ × w_mkt (균형 초과수익률, CAPM 역계산)
- λ = E[R_m - R_f] / Var(R_m) (시장 위험 회피도)
- Ω_ii = τ × P_i × Σ × P_iᵀ (뷰 불확실성, He & Litterman 1999)

**MVO:** max Sharpe Ratio = (wᵀμ) / √(wᵀΣw), s.t. Σw=1, w≥0

## 데이터베이스

| 파일 | 용도 |
|------|------|
| final_stock_months.parquet | 월별 주식 수익률 (CRSP) |
| final_stock_daily.parquet | 일일 주식 수익률 |
| tier2_accounting_metrics.parquet | Tier 2 회계 지표 (전처리 완료) |
| Tier3.csv | 거시 지표 (G20 CLI 등) |
| DTB3.csv | 3개월 T-Bill 수익률 |
| ticker_GICS.csv | 티커 → GICS 섹터 매핑 |
| sp500_ticker_start_end.csv | S&P500 편입 기간 |
| logs/Tier{1,2,3}/ | 시뮬레이션 결과 (BL-MVO, LLM-view, test) |

## 실행 방법

```bash
# 환경 설정
pip install -r requirements.txt   # torch+cu121, transformers, google-genai 등
huggingface-cli login              # Llama 3 모델 접근용

# 실행
python run_single.py               # 단일 시뮬레이션 (Tier, tau, forecast_period 설정)
python run_auto_repetition.py      # 반복 시뮬레이션
python final_visualization.py      # 결과 시각화
python statistical_analysis.py     # 통계 분석
```

**run_single.py 주요 파라미터:**
- `simul_name`: 시뮬레이션 식별자
- `Tier`: 1 (기술적), 2 (+회계), 3 (+거시)
- `tau`: BL 불확실성 계수 (기본 0.025)
- `model`: 'llama' 또는 'gemini'
- `forecast_period`: 예측 기간 리스트 (예: ["24-05-31", ..., "24-12-31"])
- `backtest_days_count`: 백테스트 거래일 수

## 설계 결정 사항

- **GPU 필수**: Llama 3 8B 4-bit 양자화 사용, view_params.py에서 CUDA 체크
- **Tier 2 데이터**: `tier2_accounting_metrics.parquet`로 캐시, Compustat 원본 없어도 동작
- **날짜 형식**: forecast_period는 `%y-%m-%d` (예: "24-05-31"), making_rollingdate.py에서 명시적 파싱
- **날짜 컬럼**: 모든 가중치 데이터는 `ForecastDate` 컬럼 사용 (통일됨)
- **CAR 계산**: 복리 `(1 + r).cumprod() - 1` 사용
- **MVO 가중치**: 소수점 3자리 반올림 후 정규화
- **LLM 출력**: JSON-only 강제, reasoning 필드를 JSON 내부에 포함
- **포트폴리오 명명**: AI_portfolio (LLM 뷰 + BL), NONE_view (뷰 없는 BL, 구 MVO)
- **프롬프트**: prompt_maker_improved.py + system_prompt_상윤_수정전.txt + tier_guidelines_상윤_v2.txt 사용

## 11개 GICS 섹터 (고정 순서)

Energy, Materials, Industrials, Consumer Discretionary, Consumer Staples,
Health Care, Financials, Information Technology, Communication Services,
Utilities, Real Estate

## 참고 문헌

- Black & Litterman (1992) "Global Portfolio Optimization"
- He & Litterman (1999) "The Intuition Behind Black-Litterman Model Portfolios"
- Idzorek (2005) "A step-by-step guide to the Black-Litterman model"
- Markowitz (1952) "Portfolio Selection"
