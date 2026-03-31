"""
result_of_test JSON 파일을 시각화하여 database/logs/graph/ 에 저장하는 스크립트.

사용법:
    # 단일 시뮬레이션 (Tier1/2/3 폴더에 같은 이름 파일)
    python visualize_results.py 0331_1

    # 반복 시뮬레이션 (run_auto_repetition 결과)
    python visualize_results.py --rep 0331_rep_ 10

    # 자동 탐색
    python visualize_results.py
"""

import os
import re
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

BASE_DIR = os.path.join("database", "logs")
GRAPH_DIR = os.path.join(BASE_DIR, "graph")
TIERS = [1, 2, 3]
TIER_LABELS = {
    1: "Tier 1 (Technical)",
    2: "Tier 2 (+ Accounting)",
    3: "Tier 3 (+ Macro)",
}
COLORS = {
    "AI_portfolio": "#2563eb",
    "NONE_view": "#9ca3af",
    "Tier1": "#ef4444",
    "Tier2": "#22c55e",
    "Tier3": "#8b5cf6",
}


# ═══════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════

def discover_simul_names() -> list[str]:
    """Tier1/result_of_test 에 있는 JSON 파일명(확장자 제외) 목록 반환."""
    search_dir = os.path.join(BASE_DIR, "Tier1", "result_of_test")
    if not os.path.isdir(search_dir):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(search_dir)
        if f.endswith(".json")
    )


def load_result(simul_name: str, tier: int):
    """특정 Tier의 result_of_test JSON 로드. 없으면 None."""
    path = os.path.join(BASE_DIR, f"Tier{tier}", "result_of_test", f"{simul_name}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_repetition_results(base_name: str, tier: int, n_reps: int) -> list:
    """반복 시뮬레이션 결과 로드. {base_name}Tier{tier}_{i}.json 형식."""
    results = []
    for i in range(1, n_reps + 1):
        fname = f"{base_name}Tier{tier}_{i}"
        data = load_result(fname, tier)
        if data is not None:
            results.append(data)
    return results


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def _pct(arr: np.ndarray) -> np.ndarray:
    return arr * 100


# ═══════════════════════════════════════════════════════════════
#  Single-run charts (기존)
# ═══════════════════════════════════════════════════════════════

def plot_cross_tier(tier_data: dict[int, list], simul_name: str, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    none_view = None
    for tier in TIERS:
        data = tier_data.get(tier)
        if data is None:
            continue
        summary = data[2]
        ai = np.array(summary["AI_portfolio"]["avg_cumulative_returns"])
        days = np.arange(1, len(ai) + 1)
        ax.plot(days, _pct(ai), label=f"AI — {TIER_LABELS[tier]}",
                linewidth=2, color=COLORS[f"Tier{tier}"])
        if none_view is None:
            none_view = np.array(summary["NONE_view"]["avg_cumulative_returns"])
    if none_view is not None:
        days = np.arange(1, len(none_view) + 1)
        ax.plot(days, _pct(none_view), label="NONE_view (Baseline)",
                linewidth=2, linestyle="--", color=COLORS["NONE_view"])
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("Business Days")
    ax.set_ylabel("Avg Cumulative Return (%)")
    ax.set_title(f"Cross-Tier Comparison — {simul_name}")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _save(fig, os.path.join(out_dir, "1_cross_tier_avg_cumret.png"))


def plot_per_tier_periods(tier_data: dict[int, list], simul_name: str, out_dir: str) -> None:
    for tier in TIERS:
        data = tier_data.get(tier)
        if data is None:
            continue
        ai_periods: dict = data[0]
        none_periods: dict = data[1]
        dates = list(ai_periods.keys())
        n = len(dates)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        for idx, date in enumerate(dates):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            ai_cum = np.array(ai_periods[date]["cumulative_returns"])
            none_cum = np.array(none_periods[date]["cumulative_returns"])
            days = np.arange(1, len(ai_cum) + 1)
            ax.plot(days, _pct(ai_cum), color=COLORS["AI_portfolio"], linewidth=1.5, label="AI")
            ax.plot(days, _pct(none_cum), color=COLORS["NONE_view"], linewidth=1.5,
                    linestyle="--", label="NONE")
            ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
            ax.set_title(date, fontsize=10)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.2, linestyle="--")
            if idx == 0:
                ax.legend(fontsize=8)
        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].set_visible(False)
        fig.suptitle(f"{TIER_LABELS[tier]} — Period Cumulative Returns ({simul_name})",
                     fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, f"2_tier{tier}_period_cumret.png"))


def plot_period_bars(tier_data: dict[int, list], simul_name: str, out_dir: str) -> None:
    for tier in TIERS:
        data = tier_data.get(tier)
        if data is None:
            continue
        ai_periods: dict = data[0]
        none_periods: dict = data[1]
        dates = list(ai_periods.keys())
        short_dates = [d[2:] for d in dates]
        ai_returns = [ai_periods[d]["final_return"] * 100 for d in dates]
        none_returns = [none_periods[d]["final_return"] * 100 for d in dates]
        ai_sharpe = [ai_periods[d]["sharpe_ratio"] for d in dates]
        none_sharpe = [none_periods[d]["sharpe_ratio"] for d in dates]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(dates) * 1.2), 8))
        x = np.arange(len(dates))
        w = 0.35
        ax1.bar(x - w / 2, ai_returns, w, label="AI_portfolio", color=COLORS["AI_portfolio"], alpha=0.85)
        ax1.bar(x + w / 2, none_returns, w, label="NONE_view", color=COLORS["NONE_view"], alpha=0.85)
        ax1.axhline(0, color="black", linewidth=0.5)
        ax1.set_ylabel("Final Return (%)")
        ax1.set_title(f"{TIER_LABELS[tier]} — Final Return by Period")
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_dates, rotation=45, ha="right", fontsize=9)
        ax1.legend(fontsize=9)
        ax1.grid(axis="y", alpha=0.25, linestyle="--")
        ai_sharpe_clean = [s if np.isfinite(s) else 0 for s in ai_sharpe]
        none_sharpe_clean = [s if np.isfinite(s) else 0 for s in none_sharpe]
        ax2.bar(x - w / 2, ai_sharpe_clean, w, label="AI_portfolio", color=COLORS["AI_portfolio"], alpha=0.85)
        ax2.bar(x + w / 2, none_sharpe_clean, w, label="NONE_view", color=COLORS["NONE_view"], alpha=0.85)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_ylabel("Sharpe Ratio")
        ax2.set_title(f"{TIER_LABELS[tier]} — Sharpe Ratio by Period")
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_dates, rotation=45, ha="right", fontsize=9)
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.25, linestyle="--")
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, f"3_tier{tier}_bars.png"))


def plot_summary_table(tier_data: dict[int, list], simul_name: str, out_dir: str) -> None:
    rows_data = []
    for tier in TIERS:
        data = tier_data.get(tier)
        if data is None:
            continue
        summary = data[2]
        ai_info = summary["AI_portfolio"]
        none_info = summary["NONE_view"]
        ai_ret = ai_info["final_avg_cumulative_return"] * 100
        none_ret = none_info["final_avg_cumulative_return"] * 100
        ai_sharpe_list = [s for s in ai_info["avg_sharpe_ratios"] if np.isfinite(s)]
        none_sharpe_list = [s for s in none_info["avg_sharpe_ratios"] if np.isfinite(s)]
        ai_avg_sharpe = np.mean(ai_sharpe_list) if ai_sharpe_list else float("nan")
        none_avg_sharpe = np.mean(none_sharpe_list) if none_sharpe_list else float("nan")
        rows_data.append([
            TIER_LABELS[tier],
            f"{ai_ret:+.3f}%", f"{none_ret:+.3f}%", f"{ai_ret - none_ret:+.3f}%",
            f"{ai_avg_sharpe:.2f}", f"{none_avg_sharpe:.2f}",
            str(ai_info["num_periods"]),
        ])
    col_labels = ["Tier", "AI Return", "NONE Return", "Excess", "AI Sharpe", "NONE Sharpe", "Periods"]
    fig, ax = plt.subplots(figsize=(12, 1.2 + 0.5 * len(rows_data)))
    ax.axis("off")
    table = ax.table(cellText=rows_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#e2e8f0")
        table[0, j].set_text_props(fontweight="bold")
    ax.set_title(f"Performance Summary — {simul_name}", fontsize=14, fontweight="bold", pad=20)
    _save(fig, os.path.join(out_dir, "4_summary_table.png"))


# ═══════════════════════════════════════════════════════════════
#  Repetition charts (반복 실험용)
# ═══════════════════════════════════════════════════════════════

def _extract_final_returns(results: list) -> list[float]:
    """각 반복의 final_avg_cumulative_return 추출."""
    out = []
    for data in results:
        ret = data[2]["AI_portfolio"]["final_avg_cumulative_return"]
        out.append(ret * 100)
    return out


def _extract_avg_cumrets(results: list) -> np.ndarray:
    """각 반복의 avg_cumulative_returns를 (n_reps, n_days) 배열로."""
    arrays = []
    for data in results:
        arr = data[2]["AI_portfolio"]["avg_cumulative_returns"]
        arrays.append(arr)
    return np.array(arrays) * 100


def _extract_final_sharpes(results: list) -> list[float]:
    """각 반복의 최종 Sharpe ratio 추출."""
    out = []
    for data in results:
        ratios = data[2]["AI_portfolio"]["avg_sharpe_ratios"]
        finite = [s for s in ratios if np.isfinite(s)]
        out.append(finite[-1] if finite else float("nan"))
    return out


def plot_rep_cumret_bands(
    all_results: dict[int, list],
    base_name: str,
    out_dir: str,
) -> None:
    """Tier별 평균 누적수익률: mean line + ±1σ band + 개별 run (반투명)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    none_view = None
    for tier in TIERS:
        results = all_results.get(tier, [])
        if not results:
            continue
        mat = _extract_avg_cumrets(results)  # (n_reps, n_days)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        days = np.arange(1, len(mean) + 1)
        color = COLORS[f"Tier{tier}"]

        # 개별 run (반투명)
        for row in mat:
            ax.plot(days, row, color=color, alpha=0.12, linewidth=0.8)
        # mean ± std band
        ax.fill_between(days, mean - std, mean + std, color=color, alpha=0.18)
        ax.plot(days, mean, color=color, linewidth=2.5,
                label=f"{TIER_LABELS[tier]}  (n={len(results)})")

        if none_view is None:
            none_view = np.array(results[0][2]["NONE_view"]["avg_cumulative_returns"]) * 100

    if none_view is not None:
        days = np.arange(1, len(none_view) + 1)
        ax.plot(days, none_view, color=COLORS["NONE_view"], linewidth=2,
                linestyle="--", label="NONE_view (Baseline)")

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("Business Days")
    ax.set_ylabel("Avg Cumulative Return (%)")
    ax.set_title(f"Cross-Tier Comparison (mean ± 1σ) — {base_name}")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.25, linestyle="--")
    _save(fig, os.path.join(out_dir, "1_rep_cumret_bands.png"))


def plot_rep_boxplot(
    all_results: dict[int, list],
    base_name: str,
    out_dir: str,
) -> None:
    """Final Return & Sharpe Ratio box plot (Tier별)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    return_data, sharpe_data, labels, colors_list = [], [], [], []
    for tier in TIERS:
        results = all_results.get(tier, [])
        if not results:
            continue
        return_data.append(_extract_final_returns(results))
        sharpe_data.append(_extract_final_sharpes(results))
        labels.append(TIER_LABELS[tier])
        colors_list.append(COLORS[f"Tier{tier}"])

    # NONE_view 기준선 (어느 결과든 동일)
    none_ret = None
    for tier in TIERS:
        results = all_results.get(tier, [])
        if results:
            none_ret = results[0][2]["NONE_view"]["final_avg_cumulative_return"] * 100
            break

    # Final Return box plot
    bp1 = ax1.boxplot(return_data, labels=labels, patch_artist=True, widths=0.5)
    for patch, c in zip(bp1["boxes"], colors_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    if none_ret is not None:
        ax1.axhline(none_ret, color=COLORS["NONE_view"], linestyle="--", linewidth=1.5,
                    label=f"NONE_view ({none_ret:.3f}%)")
        ax1.legend(fontsize=9)
    ax1.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax1.set_ylabel("Final Avg Cumulative Return (%)")
    ax1.set_title("Final Return Distribution")
    ax1.grid(axis="y", alpha=0.25, linestyle="--")

    # Sharpe Ratio box plot
    bp2 = ax2.boxplot(sharpe_data, labels=labels, patch_artist=True, widths=0.5)
    for patch, c in zip(bp2["boxes"], colors_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax2.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Sharpe Ratio Distribution")
    ax2.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle(f"Repetition Results — {base_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "2_rep_boxplot.png"))


def plot_rep_summary_table(
    all_results: dict[int, list],
    base_name: str,
    out_dir: str,
) -> None:
    """반복 실험 통계 요약 테이블."""
    rows_data = []

    none_ret = None
    for tier in TIERS:
        results = all_results.get(tier, [])
        if results:
            none_ret = results[0][2]["NONE_view"]["final_avg_cumulative_return"] * 100
            break

    for tier in TIERS:
        results = all_results.get(tier, [])
        if not results:
            continue
        returns = _extract_final_returns(results)
        sharpes = _extract_final_sharpes(results)
        ret_arr = np.array(returns)
        sh_arr = np.array([s for s in sharpes if np.isfinite(s)])

        rows_data.append([
            TIER_LABELS[tier],
            str(len(results)),
            f"{ret_arr.mean():+.3f}%",
            f"{ret_arr.std():.3f}%",
            f"{ret_arr.min():+.3f}%",
            f"{ret_arr.max():+.3f}%",
            f"{ret_arr.mean() - (none_ret or 0):+.3f}%",
            f"{sh_arr.mean():.2f}" if len(sh_arr) else "N/A",
            f"{sh_arr.std():.2f}" if len(sh_arr) else "N/A",
        ])

    col_labels = [
        "Tier", "n", "Mean Ret", "Std Ret", "Min Ret", "Max Ret",
        "Excess vs NONE", "Mean Sharpe", "Std Sharpe",
    ]

    fig, ax = plt.subplots(figsize=(16, 1.4 + 0.5 * len(rows_data)))
    ax.axis("off")
    table = ax.table(cellText=rows_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#e2e8f0")
        table[0, j].set_text_props(fontweight="bold")

    title = f"Repetition Summary — {base_name}"
    if none_ret is not None:
        title += f"  (NONE_view: {none_ret:+.3f}%)"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    _save(fig, os.path.join(out_dir, "3_rep_summary_table.png"))


# ═══════════════════════════════════════════════════════════════
#  Entrypoints
# ═══════════════════════════════════════════════════════════════

def visualize(simul_name: str) -> None:
    """단일 시뮬레이션 시각화."""
    print(f"\n{'='*60}")
    print(f" Visualizing: {simul_name}")
    print(f"{'='*60}")

    tier_data: dict[int, list] = {}
    for tier in TIERS:
        result = load_result(simul_name, tier)
        if result is not None:
            tier_data[tier] = result
            print(f"  [OK] Tier {tier} loaded")
        else:
            print(f"  [--] Tier {tier} not found, skipping")

    if not tier_data:
        print("  No data found. Skipping.")
        return

    out_dir = os.path.join(GRAPH_DIR, simul_name)
    plot_cross_tier(tier_data, simul_name, out_dir)
    plot_per_tier_periods(tier_data, simul_name, out_dir)
    plot_period_bars(tier_data, simul_name, out_dir)
    plot_summary_table(tier_data, simul_name, out_dir)
    print(f"\n  All graphs saved to: {out_dir}/")


def visualize_repetitions(base_name: str, n_reps: int) -> None:
    """반복 시뮬레이션 시각화."""
    print(f"\n{'='*60}")
    print(f" Visualizing repetitions: {base_name} (×{n_reps})")
    print(f"{'='*60}")

    all_results: dict[int, list] = {}
    for tier in TIERS:
        results = load_repetition_results(base_name, tier, n_reps)
        if results:
            all_results[tier] = results
            print(f"  [OK] Tier {tier}: {len(results)}/{n_reps} loaded")
        else:
            print(f"  [--] Tier {tier}: no files found")

    if not all_results:
        print("  No data found.")
        return

    out_dir = os.path.join(GRAPH_DIR, f"{base_name}rep{n_reps}")

    plot_rep_cumret_bands(all_results, base_name, out_dir)
    plot_rep_boxplot(all_results, base_name, out_dir)
    plot_rep_summary_table(all_results, base_name, out_dir)

    print(f"\n  All graphs saved to: {out_dir}/")


def main() -> None:
    args = sys.argv[1:]

    # --rep 모드: python visualize_results.py --rep 0331_rep_ 10
    if args and args[0] == "--rep":
        if len(args) < 3:
            print("Usage: python visualize_results.py --rep <base_name> <n_reps>")
            return
        base_name = args[1]
        n_reps = int(args[2])
        visualize_repetitions(base_name, n_reps)
    elif args:
        for name in args:
            visualize(name)
    else:
        names = discover_simul_names()
        if not names:
            print("result_of_test 폴더에 JSON 파일이 없습니다.")
            return
        print(f"발견된 시뮬레이션: {names}")
        for name in names:
            visualize(name)

    print("\nDone.")


if __name__ == "__main__":
    main()
