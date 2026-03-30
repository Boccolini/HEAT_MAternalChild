from __future__ import annotations

import argparse
import csv
from pathlib import Path

import duckdb


STAGE_COLUMNS = [
    "pregnancy_id",
    "municipality_code",
    "state_code",
    "gestational_week_completed",
    "preterm_event_flag",
    "tmax_weekly_mean",
    "maternal_age_years",
    "maternal_schooling_code",
    "parity_total_prior",
    "plurality_code",
    "newborn_sex_code",
    "maternal_race_color_code",
    "ga_days_approx_flag",
    "temperature_source_set",
    "climate_correction_path_set",
]

MINFIT_COLUMNS = [
    "preterm_event_flag",
    "tmax_weekly_mean",
    "maternal_age_years",
    "maternal_schooling_code",
    "parity_total_prior",
    "plurality_code",
    "newborn_sex_code",
    "maternal_race_color_code",
    "municipality_code",
    "state_code",
]

COLUMN_SETS = {
    "full_stage": STAGE_COLUMNS,
    "minfit": MINFIT_COLUMNS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble YEAR x UF staged E002 parquet blocks into a single yearly CSV on a remote host."
    )
    parser.add_argument("--input-glob", required=True, help="Glob for staged parquet inputs.")
    parser.add_argument("--output-csv", required=True, help="Path to the assembled yearly CSV.")
    parser.add_argument("--summary-csv", required=True, help="Path to the yearly assembly summary CSV.")
    parser.add_argument("--label", required=True, help="Human-readable label for this rung.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--temp-dir", required=True, help="DuckDB temp directory.")
    parser.add_argument(
        "--column-set",
        choices=sorted(COLUMN_SETS),
        default="full_stage",
        help="Projection to assemble from the staged parquet blocks.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def connect(temp_dir: Path, threads: int, memory_limit: str) -> duckdb.DuckDBPyConnection:
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute("PRAGMA max_temp_directory_size='150GiB'")
    con.execute(f"SET temp_directory = '{temp_dir.as_posix()}'")
    return con


def write_summary(path: Path, row: dict[str, str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_csv = Path(args.output_csv).expanduser().resolve()
    summary_csv = Path(args.summary_csv).expanduser().resolve()
    input_glob = Path(args.input_glob).expanduser().as_posix()
    temp_dir = Path(args.temp_dir).expanduser().resolve()

    ensure_parent(output_csv)
    ensure_parent(summary_csv)

    selected_columns = COLUMN_SETS[args.column_set]
    projection = ",\n            ".join(selected_columns)
    parquet_query = f"""
        SELECT
            {projection}
        FROM read_parquet('{input_glob}', union_by_name=true, hive_partitioning=true)
    """

    con = connect(temp_dir=temp_dir, threads=args.threads, memory_limit=args.memory_limit)
    try:
        summary = con.execute(
            f"""
            SELECT
                COUNT(*) AS row_count,
                SUM(CAST(preterm_event_flag AS BIGINT)) AS event_count,
                COUNT(DISTINCT municipality_code) AS municipality_count,
                COUNT(DISTINCT state_code) AS state_count
            FROM ({parquet_query})
            """
        ).fetchone()

        con.execute(
            f"""
            COPY (
                SELECT {projection}
                FROM ({parquet_query})
                ORDER BY state_code, municipality_code
            ) TO '{output_csv.as_posix()}' (HEADER, DELIMITER ',')
            """
        )
    finally:
        con.close()

    output_size_bytes = output_csv.stat().st_size if output_csv.exists() else 0
    row = {
        "label": args.label,
        "input_glob": input_glob,
        "output_csv": str(output_csv),
        "row_count": str(int(summary[0] or 0)),
        "event_count": str(int(summary[1] or 0)),
        "municipality_count": str(int(summary[2] or 0)),
        "state_count": str(int(summary[3] or 0)),
        "column_set": args.column_set,
        "output_size_bytes": str(output_size_bytes),
    }
    write_summary(summary_csv, row)
    print(row)


if __name__ == "__main__":
    main()
