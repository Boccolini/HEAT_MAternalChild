from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    ROOT
    / "artifacts/analysis/shared_marts/flagship_gestational_weekly_person_period_mart"
)
REGISTRY_DIR = ROOT / "artifacts/analysis/step9_rebuild/registries"
RESULTS_DIR = ROOT / "artifacts/analysis/step9_rebuild/results/remote_benchmarks/e002"
STAGE_DIR = RESULTS_DIR / "block_stage"
REVIEW_DIR = ROOT / "artifacts/analysis/step9_rebuild/reviews"

MANIFEST_PATH = REGISTRY_DIR / "e002_external_block_manifest.csv"
QC_PATH = REGISTRY_DIR / "e002_external_block_qc.csv"
STAGE_LOG_PATH = REGISTRY_DIR / "e002_external_block_stage_log.csv"
COMPARISON_PATH = RESULTS_DIR / "e002_remote_benchmark_comparison.csv"
COMPARISON_MEMO_PATH = REVIEW_DIR / "e002_remote_benchmark_comparison.md"

SIMPLE_RESULT_PATH = RESULTS_DIR / "year2020_states_12_14_16_remote_simple_result.csv"
STEPUP_RESULT_PATH = RESULTS_DIR / "year2020_states_12_14_16_remote_stepup_result.csv"

YEARS = [2020, 2021, 2022, 2023, 2024]
UF_CODES = [
    "11", "12", "13", "14", "15", "16", "17",
    "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "31", "32", "33", "35",
    "41", "42", "43",
    "50", "51", "52", "53",
]

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

MISSINGNESS_POLICY = """
    tmax_weekly_mean IS NOT NULL
    AND maternal_age_years IS NOT NULL
    AND maternal_schooling_code IS NOT NULL
    AND parity_total_prior IS NOT NULL
    AND plurality_code IS NOT NULL
    AND newborn_sex_code IS NOT NULL
"""


@dataclass(frozen=True)
class BlockSpec:
    year: int
    uf_code: str

    @property
    def block_id(self) -> str:
        return f"e002_block_{self.year}_UF{self.uf_code}"

    @property
    def year_dir(self) -> Path:
        return SOURCE_ROOT / f"YEAR={self.year}"

    @property
    def source_files(self) -> list[Path]:
        pattern = f"flagship_gestational_weekly_person_period_mart_{self.year}_UF{self.uf_code}_M*.parquet"
        return sorted(self.year_dir.glob(pattern))

    @property
    def stage_path(self) -> Path:
        return STAGE_DIR / f"YEAR={self.year}" / f"e002_stage_input_{self.year}_UF{self.uf_code}.parquet"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _connect() -> duckdb.DuckDBPyConnection:
    temp_dir = ROOT / "artifacts" / "tmp" / "duckdb" / "e002_external_blocks"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET memory_limit = '4GB'")
    con.execute("PRAGMA max_temp_directory_size='20GiB'")
    con.execute(f"SET temp_directory = '{temp_dir}'")
    return con


def _quoted_paths(paths: list[Path]) -> str:
    return ", ".join(f"'{path}'" for path in paths)


def _filtered_sql(paths: list[Path]) -> str:
    return f"""
        SELECT
            pregnancy_id,
            municipality_code,
            state_code,
            gestational_week_completed,
            CAST(preterm_event_flag AS INTEGER) AS preterm_event_flag,
            CAST(tmax_weekly_mean AS DOUBLE) AS tmax_weekly_mean,
            CAST(maternal_age_years AS DOUBLE) AS maternal_age_years,
            CAST(maternal_schooling_code AS VARCHAR) AS maternal_schooling_code,
            CAST(parity_total_prior AS VARCHAR) AS parity_total_prior,
            CAST(plurality_code AS VARCHAR) AS plurality_code,
            CAST(newborn_sex_code AS VARCHAR) AS newborn_sex_code,
            CAST(COALESCE(maternal_race_color_code, 'unknown') AS VARCHAR) AS maternal_race_color_code,
            ga_days_approx_flag,
            temperature_source_set,
            climate_correction_path_set
        FROM read_parquet([{_quoted_paths(paths)}])
        WHERE {MISSINGNESS_POLICY}
    """


def _stringify_levels(values: list[str]) -> str:
    return "|".join(values)


def _distinct_levels(
    con: duckdb.DuckDBPyConnection, filtered_sql: str, column: str
) -> list[str]:
    rows = con.execute(
        f"SELECT DISTINCT CAST({column} AS VARCHAR) AS value FROM ({filtered_sql}) WHERE {column} IS NOT NULL ORDER BY 1"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _read_single_csv(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else None


def _normalize_r_bool(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"true", "t", "1"}:
        return "TRUE"
    if normalized in {"false", "f", "0"}:
        return "FALSE"
    return ""


def _is_finite_number(value: str | None) -> bool:
    try:
        return math.isfinite(float(value or ""))
    except ValueError:
        return False


def _stepup_ready_for_ladder(stepup: dict[str, str] | None) -> bool:
    if not stepup:
        return False
    required_numeric = (
        "estimate",
        "std_error",
        "rr",
        "ci_low",
        "ci_high",
        "elapsed_seconds",
    )
    if any(not _is_finite_number(stepup.get(field)) for field in required_numeric):
        return False
    return _normalize_r_bool(stepup.get("singular_flag")) == "FALSE"


def build_blocks() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    manifest_rows: list[dict[str, str]] = []
    qc_rows: list[dict[str, str]] = []
    stage_rows: list[dict[str, str]] = []

    con = _connect()
    try:
        for year in YEARS:
            for uf_code in UF_CODES:
                spec = BlockSpec(year=year, uf_code=uf_code)
                source_files = spec.source_files
                observed_months = sorted(
                    path.stem.rsplit("_M", 1)[1] for path in source_files
                )
                missing_months = [
                    f"{month:02d}"
                    for month in range(1, 13)
                    if f"{month:02d}" not in observed_months
                ]
                filtered_sql = _filtered_sql(source_files)
                stage_path = spec.stage_path
                manifest_rows.append(
                    {
                        "block_id": spec.block_id,
                        "year": str(year),
                        "uf_code": uf_code,
                        "expected_month_count": "12",
                        "observed_month_count": str(len(source_files)),
                        "missing_months": "|".join(missing_months),
                        "source_glob": str(
                            spec.year_dir
                            / f"flagship_gestational_weekly_person_period_mart_{year}_UF{uf_code}_M*.parquet"
                        ),
                        "stage_artifact_path": _relative(stage_path),
                        "all_source_months_present": "yes" if len(source_files) == 12 else "no",
                    }
                )

                if not source_files:
                    qc_rows.append(
                        {
                            "block_id": spec.block_id,
                            "year": str(year),
                            "uf_code": uf_code,
                            "row_count_after_policy": "0",
                            "event_count_after_policy": "0",
                            "municipality_count": "0",
                            "state_count": "0",
                            "maternal_schooling_levels": "0",
                            "parity_levels": "0",
                            "plurality_levels": "0",
                            "newborn_sex_levels": "0",
                            "maternal_race_levels": "0",
                            "maternal_schooling_values": "",
                            "parity_values": "",
                            "plurality_values": "",
                            "newborn_sex_values": "",
                            "maternal_race_values": "",
                            "invariant_terms": "",
                            "plurality_code_invariant": "no",
                            "two_level_exact_fit_eligible": "no",
                        }
                    )
                    stage_rows.append(
                        {
                            "block_id": spec.block_id,
                            "year": str(year),
                            "uf_code": uf_code,
                            "extraction_status": "missing_source_files",
                            "qc_status": "missing_source_files",
                            "artifact_status": "not_written",
                            "stage_artifact_path": _relative(stage_path),
                            "stage_rows_written": "0",
                            "notes": "No monthly parquet source files were found for this YEAR x UF block.",
                        }
                    )
                    continue

                summary = con.execute(
                    f"""
                    SELECT
                        COUNT(*) AS row_count,
                        SUM(CAST(preterm_event_flag AS BIGINT)) AS event_count,
                        COUNT(DISTINCT municipality_code) AS municipality_count,
                        COUNT(DISTINCT state_code) AS state_count
                    FROM ({filtered_sql})
                    """
                ).fetchone()
                row_count = int(summary[0] or 0)
                event_count = int(summary[1] or 0)
                municipality_count = int(summary[2] or 0)
                state_count = int(summary[3] or 0)

                schooling_values = _distinct_levels(con, filtered_sql, "maternal_schooling_code")
                parity_values = _distinct_levels(con, filtered_sql, "parity_total_prior")
                plurality_values = _distinct_levels(con, filtered_sql, "plurality_code")
                sex_values = _distinct_levels(con, filtered_sql, "newborn_sex_code")
                race_values = _distinct_levels(con, filtered_sql, "maternal_race_color_code")

                invariants: list[str] = []
                if len(schooling_values) < 2:
                    invariants.append("maternal_schooling_code")
                if len(parity_values) < 2:
                    invariants.append("parity_total_prior")
                if len(plurality_values) < 2:
                    invariants.append("plurality_code")
                if len(sex_values) < 2:
                    invariants.append("newborn_sex_code")
                if len(race_values) < 2:
                    invariants.append("maternal_race_color_code")

                qc_rows.append(
                    {
                        "block_id": spec.block_id,
                        "year": str(year),
                        "uf_code": uf_code,
                        "row_count_after_policy": str(row_count),
                        "event_count_after_policy": str(event_count),
                        "municipality_count": str(municipality_count),
                        "state_count": str(state_count),
                        "maternal_schooling_levels": str(len(schooling_values)),
                        "parity_levels": str(len(parity_values)),
                        "plurality_levels": str(len(plurality_values)),
                        "newborn_sex_levels": str(len(sex_values)),
                        "maternal_race_levels": str(len(race_values)),
                        "maternal_schooling_values": _stringify_levels(schooling_values),
                        "parity_values": _stringify_levels(parity_values),
                        "plurality_values": _stringify_levels(plurality_values),
                        "newborn_sex_values": _stringify_levels(sex_values),
                        "maternal_race_values": _stringify_levels(race_values),
                        "invariant_terms": "|".join(invariants),
                        "plurality_code_invariant": "yes" if "plurality_code" in invariants else "no",
                        "two_level_exact_fit_eligible": "no",
                    }
                )

                _ensure_parent(stage_path)
                con.execute(
                    f"COPY ({filtered_sql}) TO '{stage_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
                )
                stage_rows.append(
                    {
                        "block_id": spec.block_id,
                        "year": str(year),
                        "uf_code": uf_code,
                        "extraction_status": "completed",
                        "qc_status": "completed",
                        "artifact_status": "written",
                        "stage_artifact_path": _relative(stage_path),
                        "stage_rows_written": str(row_count),
                        "notes": (
                            "YEAR x UF block staged from existing monthly gestational mart partitions "
                            "under the frozen E002 missingness policy."
                        ),
                    }
                )
    finally:
        con.close()

    return manifest_rows, qc_rows, stage_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_comparison() -> tuple[list[dict[str, str]], str]:
    simple = _read_single_csv(SIMPLE_RESULT_PATH)
    stepup = _read_single_csv(STEPUP_RESULT_PATH)
    comparison_status = "complete" if simple and stepup else "pending_stepup"
    gate_status = (
        "ready_for_external_ladder"
        if _stepup_ready_for_ladder(stepup)
        else "blocked_pending_stepup"
    )
    row = {
        "slice_name": "year2020_states_12_14_16_remote",
        "comparison_status": comparison_status,
        "simple_formula": "preterm_event_flag ~ tmax_weekly_mean + maternal_age_years + (1|state_code) + (1|municipality_code)",
        "stepup_formula": (
            "preterm_event_flag ~ tmax_weekly_mean + maternal_age_years + maternal_schooling_code + "
            "parity_total_prior + newborn_sex_code + maternal_race_color_code + "
            "(1|state_code) + (1|municipality_code)"
        ),
        "plurality_handling": (
            "plurality_code is structurally invariant in the singleton-restricted cohort and remains audited but excluded from the benchmark fit."
        ),
        "simple_row_count": simple["row_count"] if simple else "",
        "simple_event_count": simple["event_count"] if simple else "",
        "simple_rr": simple["rr"] if simple else "",
        "simple_ci_low": simple["ci_low"] if simple else "",
        "simple_ci_high": simple["ci_high"] if simple else "",
        "simple_singular_flag": _normalize_r_bool(simple["singular_flag"]) if simple else "",
        "simple_elapsed_seconds": simple["elapsed_seconds"] if simple else "",
        "stepup_row_count": stepup["row_count"] if stepup else "",
        "stepup_event_count": stepup["event_count"] if stepup else "",
        "stepup_rr": stepup["rr"] if stepup else "",
        "stepup_ci_low": stepup["ci_low"] if stepup else "",
        "stepup_ci_high": stepup["ci_high"] if stepup else "",
        "stepup_singular_flag": _normalize_r_bool(stepup["singular_flag"]) if stepup else "",
        "stepup_elapsed_seconds": stepup["elapsed_seconds"] if stepup else "",
        "warning_text": stepup["warning_text"] if stepup else "",
        "gate_status": gate_status,
    }
    memo = [
        "# E002 Remote Benchmark Comparison",
        "",
        f"- Comparison status: `{comparison_status}`",
        f"- Gate status: `{gate_status}`",
        "- Slice: `2020_states_12_14_16`",
        "- `plurality_code` remains audited but excluded from the benchmark fit because it is invariant in the singleton-restricted cohort.",
        "",
        "## Simple Benchmark",
    ]
    if simple:
        memo.extend(
            [
                f"- rows: `{simple['row_count']}`",
                f"- events: `{simple['event_count']}`",
                f"- RR: `{simple['rr']}`",
                f"- 95% CI: `{simple['ci_low']}` to `{simple['ci_high']}`",
                f"- singular: `{_normalize_r_bool(simple['singular_flag'])}`",
                f"- elapsed seconds: `{simple['elapsed_seconds']}`",
            ]
        )
    else:
        memo.append("- simple benchmark is missing locally.")
    memo.extend(["", "## Step-up Benchmark"])
    if stepup:
        memo.extend(
            [
                f"- rows: `{stepup['row_count']}`",
                f"- events: `{stepup['event_count']}`",
                f"- RR: `{stepup['rr']}`",
                f"- 95% CI: `{stepup['ci_low']}` to `{stepup['ci_high']}`",
                f"- singular: `{_normalize_r_bool(stepup['singular_flag'])}`",
                f"- elapsed seconds: `{stepup['elapsed_seconds']}`",
                f"- warnings: `{stepup['warning_text']}`",
            ]
        )
    else:
        memo.append("- step-up benchmark is still running remotely or has not yet been copied into the workspace.")
    return [row], "\n".join(memo) + "\n"


def main() -> None:
    manifest_rows, qc_rows, stage_rows = build_blocks()
    write_csv(MANIFEST_PATH, manifest_rows)
    write_csv(QC_PATH, qc_rows)
    write_csv(STAGE_LOG_PATH, stage_rows)

    comparison_rows, comparison_memo = build_comparison()
    write_csv(COMPARISON_PATH, comparison_rows)
    _ensure_parent(COMPARISON_MEMO_PATH)
    COMPARISON_MEMO_PATH.write_text(comparison_memo, encoding="utf-8")


if __name__ == "__main__":
    main()
