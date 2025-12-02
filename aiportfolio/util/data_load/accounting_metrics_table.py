from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path

# Mapping reused across the project
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

# Metric configuration: parquet key → display label → scale factor
METRIC_SPECS = [
    ("bm_Mean", "B/M", 1.0),
    ("CAPEI_Mean", "CAPEI", 1.0),
    ("roa_Mean", "ROA (%)", 100.0),
    ("roe_Mean", "ROE (%)", 100.0),
    ("GProf_Mean", "GPROF (%)", 100.0),
    ("npm_Mean", "NPM (%)", 100.0),
    ("totdebt_invcap_Mean", "D/C (%)", 100.0),
]


def _read_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def calculate_accounting_metrics_table(
    parquet_path: Path | str = Path("database/tier2_accounting_metrics.parquet"),
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """
    Aggregate accounting metrics by GICS sector (in-sample median across months).

    Returns a DataFrame with columns:
    - gics_code, sector_name and one column per configured metric label.
    """
    df = _read_data(Path(parquet_path))

    if start_year is not None:
        df = df[df["date"].dt.year >= start_year]
    if end_year is not None:
        df = df[df["date"].dt.year <= end_year]

    # Median by sector/metric over the filtered period
    grouped = (
        df.groupby(["gsector", "metric"])["acct_level_lagged_avg"]
        .median()
        .reset_index()
    )

    # Build lookup for quick access
    metric_lookup = {}
    for _, row in grouped.iterrows():
        metric_lookup[(row["gsector"], row["metric"])] = row["acct_level_lagged_avg"]

    rows: list[dict[str, object]] = []
    for code in GICS_SECTOR_ORDER:
        sector_name = GICS_SECTOR_MAP[code]
        row: dict[str, object] = {
            "gics_code": code,
            "sector_name": sector_name,
        }
        for key, label, scale in METRIC_SPECS:
            value = metric_lookup.get((sector_name, key))
            if value is None:
                row[label] = None
            else:
                scaled = value * scale
                row[label] = scaled
        rows.append(row)

    return pd.DataFrame(rows)


def format_table_for_display(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Return a string-formatted table for quick printing."""
    formatted = df.copy()
    formatted["gics_code"] = formatted["gics_code"].astype(int)
    for _, label, _ in METRIC_SPECS:
        formatted[label] = formatted[label].map(
            lambda x: "" if pd.isna(x) else f"{x:.{decimals}f}"
        )
    return formatted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Panel C accounting metrics table by GICS sector."
    )
    parser.add_argument("--start-year", type=int, help="Inclusive start year.")
    parser.add_argument("--end-year", type=int, help="Inclusive end year.")
    parser.add_argument(
        "--parquet-path",
        type=str,
        default="database/tier2_accounting_metrics.parquet",
        help="Path to tier2_accounting_metrics.parquet",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for display values.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    table = calculate_accounting_metrics_table(
        parquet_path=Path(args.parquet_path),
        start_year=args.start_year,
        end_year=args.end_year,
    )
    pretty = format_table_for_display(table, decimals=args.decimals)
    print("\nPanel C. Accounting Metrics by GICS Sector (In-Sample Period)\n")
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
