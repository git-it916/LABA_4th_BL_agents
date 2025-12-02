from __future__ import annotations

import argparse
import pandas as pd

from aiportfolio.util.data_load.open_final_stock_months import open_final_stock_months

# Standard GICS sector mapping used across the project
GICS_SECTOR_MAP = {
    10: "Energy",
    15: "Materials",
    20: "Industrials",
    25: "Consumer Discretionary",
    30: "Consumer Staples",
    35: "Health Care",
    40: "Financials",
    45: "Information Technology",
    50: "Communication Services",
    55: "Utilities",
    60: "Real Estate",
}

GICS_SECTOR_ORDER = list(GICS_SECTOR_MAP.keys())


def calculate_gics_distribution(
    start_year: int | None = None,
    end_year: int | None = None,
    sp500_only: bool = True,
) -> pd.DataFrame:
    """
    Build Panel C style distribution stats for final_stock_months.parquet.

    Returns a DataFrame with columns:
    - sector_code: numeric GICS sector code
    - sector: human-readable sector name
    - observations: record count for the sector
    - pct_of_total: share of total observations
    - avg_per_month: observations divided by the number of months in-sample
    """
    df = open_final_stock_months().copy()

    if start_year is not None:
        df = df[df["cyear"] >= start_year]
    if end_year is not None:
        df = df[df["cyear"] <= end_year]

    if sp500_only and "sp500" in df.columns:
        df = df[df["sp500"] == 1]

    df = df[df["gsector"].notna()].copy()
    df["gsector"] = df["gsector"].astype(int)

    months_in_sample = df[["cyear", "cmonth"]].drop_duplicates().shape[0]
    total_obs = len(df)

    rows: list[dict[str, object]] = []
    for code in GICS_SECTOR_ORDER:
        sector_count = int((df["gsector"] == code).sum())
        pct_total = (sector_count / total_obs * 100) if total_obs else 0.0
        avg_month = (sector_count / months_in_sample) if months_in_sample else 0.0
        rows.append(
            {
                "sector_code": code,
                "sector": GICS_SECTOR_MAP.get(code, str(code)),
                "observations": sector_count,
                "pct_of_total": pct_total,
                "avg_per_month": avg_month,
            }
        )

    total_row = {
        "sector_code": None,
        "sector": "Total",
        "observations": total_obs,
        "pct_of_total": 100.0 if total_obs else 0.0,
        "avg_per_month": (total_obs / months_in_sample) if months_in_sample else 0.0,
    }
    rows.append(total_row)

    return pd.DataFrame(rows)


def format_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a pretty-printed version for quick inspection."""
    formatted = df.copy()
    formatted["sector_code"] = formatted["sector_code"].apply(
        lambda x: "" if pd.isna(x) else str(int(x)) if isinstance(x, (int, float)) else str(x)
    )
    formatted["observations"] = formatted["observations"].map(lambda x: f"{x:,}")
    formatted["pct_of_total"] = formatted["pct_of_total"].map(lambda x: f"{x:0.1f}%")
    formatted["avg_per_month"] = formatted["avg_per_month"].map(lambda x: f"{x:0.0f}")
    return formatted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Distribution by GICS Sector (In-Sample Period)."
    )
    parser.add_argument("--start-year", type=int, help="Inclusive calendar year filter.")
    parser.add_argument("--end-year", type=int, help="Inclusive calendar year filter.")
    parser.add_argument(
        "--all-stocks",
        action="store_true",
        help="Include all tickers instead of only rows flagged sp500==1.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = calculate_gics_distribution(
        start_year=args.start_year,
        end_year=args.end_year,
        sp500_only=not args.all_stocks,
    )
    pretty = format_distribution_table(summary)
    print("\nPanel C. Distribution by GICS Sector (In-Sample Period)\n")
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
