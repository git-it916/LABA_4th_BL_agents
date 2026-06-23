# -*- coding: utf-8 -*-
"""재실행 전 스모크 테스트: Tier1 1회 + Tier3 regime 1회. 파이프라인 검증 + 1회당 시간 측정."""
import time
from aiportfolio.scene import scene
from aiportfolio.agents.Llama_config import cleanup_pipeline

tau = 0.025
model = 'llama'
backtest_days_count = 19
forecast_period = [
    "24-05-31", "24-06-30", "24-07-31", "24-08-31",
    "24-09-30", "24-10-31", "24-11-30", "24-12-31",
]

results = {}
try:
    for label, (name, tier, kw) in {
        'Tier1': ('smoke_Tier1_1', 1, {}),
        'Tier3_regime': ('smoke_Tier3_regime_1', 3, {'tier3_mode': 'regime'}),
    }.items():
        t0 = time.perf_counter()
        print(f"\n{'#'*70}\n# SMOKE START {label} ({name})\n{'#'*70}")
        scene(name, tier, tau, forecast_period, backtest_days_count, model, **kw)
        dt = time.perf_counter() - t0
        results[label] = dt
        print(f"\n{'#'*70}\n# SMOKE DONE  {label}: {dt/60:.1f} min\n{'#'*70}")
finally:
    cleanup_pipeline()

print("\n===== SMOKE TIMING SUMMARY =====")
for k, v in results.items():
    print(f"  {k}: {v/60:.2f} min / rep")
if results:
    avg = sum(results.values()) / len(results)
    print(f"  => 50x3 reps ETA (rough): {avg*150/3600:.1f} h")
