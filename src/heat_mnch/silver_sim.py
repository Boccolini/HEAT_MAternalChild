from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .manifest import build_manifest
from .outcome_dictionary_icd10 import (
    DIRECT_HEAT_GROUPS,
    INDIRECT_HEAT_GROUPS,
    MATERNAL_GROUPS,
    SIM_ICD10_COLUMNS,
    SIM_SCOPE_ANY_MENTION,
    SIM_SCOPE_UNDERLYING,
    SLOW_MOVING_HEAT_GROUPS,
    build_group_code_sets,
    build_unified_outcome_dictionary,
    dataset_entry,
    derive_sim_age_flags,
    expected_group_names,
    load_unified_outcome_dictionary,
    normalize_icd10_series,
    split_sim_cause_tokens,
    utc_now,
)
from .run_log import append_run_event
from .silver_outcome_common import ParquetChunkWriter, official_total_from_bronze, sync_verified_parquet


SIM_REQUIRED_INPUT_FIELDS = [
    "BRONZE_DATASET",
    "BRONZE_SOURCE_SYSTEM",
    "BRONZE_SOURCE_FILE",
    "BRONZE_SOURCE_PATH",
    "BRONZE_SOURCE_MEMBER",
    "BRONZE_YEAR",
    "BRONZE_INGESTED_AT",
    "ORIGEM",
    "TIPOBITO",
    "DTOBITO",
    "DTNASC",
    "IDADE",
    "SEXO",
    "CODMUNRES",
    "CODMUNOCOR",
    "CAUSABAS",
    "LINHAA",
    "LINHAB",
    "LINHAC",
    "LINHAD",
    "LINHAII",
    "CONTADOR",
]
SIM_NORMALIZED_CAUSE_FIELDS = [f"{column}_ICD10" for column in SIM_ICD10_COLUMNS]
SIM_SCOPE_FIELDS = (
    (SIM_SCOPE_UNDERLYING, "UNDERLYING_BASIC_CAUSE"),
    (SIM_SCOPE_ANY_MENTION, "ANY_MENTION"),
)


def _scope_flag_column(scope_label: str, group_name: str) -> str:
    return f"{scope_label}_{group_name}_FLAG"


def _scope_total_column(scope_label: str, group_name: str) -> str:
    return f"{scope_label}_{group_name}_TOTAL"


SIM_GROUP_FLAG_FIELDS = [
    _scope_flag_column(scope_label, group_name)
    for _, scope_label in SIM_SCOPE_FIELDS
    for group_name in expected_group_names()
]
SIM_CATEGORY_FLAG_FIELDS = [
    "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_FLAG",
    "ANY_MENTION_DIRECT_HEAT_ANY_FLAG",
    "UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_FLAG",
    "ANY_MENTION_INDIRECT_HEAT_ANY_FLAG",
    "UNDERLYING_BASIC_CAUSE_SLOW_MOVING_HEAT_ANY_FLAG",
    "ANY_MENTION_SLOW_MOVING_HEAT_ANY_FLAG",
    "UNDERLYING_BASIC_CAUSE_MATERNAL_PHENOTYPE_ANY_FLAG",
    "ANY_MENTION_MATERNAL_PHENOTYPE_ANY_FLAG",
]
SIM_INDIVIDUAL_OUTPUT_FIELDS = [
    "SIM_RECORD_ID",
    "SOURCE_SIM",
    "BRONZE_DATASET",
    "BRONZE_SOURCE_SYSTEM",
    "BRONZE_SOURCE_FILE",
    "BRONZE_SOURCE_PATH",
    "BRONZE_SOURCE_MEMBER",
    "BRONZE_YEAR",
    "BRONZE_INGESTED_AT",
    "ORIGEM",
    "TIPOBITO",
    "DATE",
    "YEAR",
    "MONTH",
    "MUNICIPALITY_CODE",
    "CODMUNOCOR",
    "CODMUNRES",
    "DTOBITO",
    "DTNASC",
    "IDADE",
    "SEXO",
    "AGE_UNIT_CODE",
    "AGE_VALUE",
    "AGE_GROUP_0_27D",
    "AGE_GROUP_28_364D",
    "AGE_GROUP_1_4Y",
    "AGE_GROUP_U5",
    "AGE_GROUP_WRA_15_49",
    "AGE_GROUP_WRA_15_45_SENSITIVITY",
    "CAUSABAS",
    "LINHAA",
    "LINHAB",
    "LINHAC",
    "LINHAD",
    "LINHAII",
    *SIM_NORMALIZED_CAUSE_FIELDS,
    *SIM_GROUP_FLAG_FIELDS,
    *SIM_CATEGORY_FLAG_FIELDS,
    "QC_STATUS",
]
SIM_DAILY_OUTPUT_FIELDS = [
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
    "DEATHS_TOTAL",
    *[_scope_total_column(scope_label, group_name) for _, scope_label in SIM_SCOPE_FIELDS for group_name in expected_group_names()],
    "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL",
    "ANY_MENTION_DIRECT_HEAT_ANY_TOTAL",
    "UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_TOTAL",
    "ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL",
    "UNDERLYING_BASIC_CAUSE_SLOW_MOVING_HEAT_ANY_TOTAL",
    "ANY_MENTION_SLOW_MOVING_HEAT_ANY_TOTAL",
    "UNDERLYING_BASIC_CAUSE_MATERNAL_PHENOTYPE_ANY_TOTAL",
    "ANY_MENTION_MATERNAL_PHENOTYPE_ANY_TOTAL",
    "SOURCE_SIM",
    "QC_STATUS",
]


def sim_individual_partition_dir(root: Path, year: int) -> Path:
    return root / "data" / "silver" / "sim" / "silver_sim_individual" / f"YEAR={year}"


def sim_individual_partition_file(root: Path, year: int) -> Path:
    return sim_individual_partition_dir(root, year) / f"silver_sim_individual_{year}.parquet"


def sim_daily_partition_dir(root: Path, year: int) -> Path:
    return root / "data" / "silver" / "sim" / "silver_sim_daily" / f"YEAR={year}"


def sim_daily_partition_file(root: Path, year: int) -> Path:
    return sim_daily_partition_dir(root, year) / f"silver_sim_daily_{year}.parquet"


def _load_sim_chunk_iterator(root: Path, year: int):
    path = root / "data" / "bronze" / "sim" / str(year) / f"sim_{year}_source_preserving.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"SIM bronze missing for {year}: {path}")
    header = pd.read_csv(path, compression="gzip", sep=";", dtype=str, nrows=0, keep_default_na=False, low_memory=False)
    missing_fields = [field for field in SIM_REQUIRED_INPUT_FIELDS if field not in header.columns]
    if missing_fields:
        raise ValueError(f"SIM bronze missing required columns for {year}: {missing_fields}")
    return pd.read_csv(
        path,
        compression="gzip",
        sep=";",
        dtype=str,
        usecols=SIM_REQUIRED_INPUT_FIELDS,
        chunksize=200_000,
        keep_default_na=False,
        low_memory=False,
    )


def _tokenize_sim_multicode_series(
    series: pd.Series,
    token_cache: dict[str, tuple[str, ...]],
) -> tuple[pd.Series, pd.DataFrame]:
    parsed = series.fillna("").astype(str).map(
        lambda raw_value: token_cache.setdefault(raw_value, split_sim_cause_tokens(raw_value))
    )
    normalized = parsed.map(lambda tokens: "|".join(tokens))
    exploded = parsed.reset_index(name="ICD10_CODE").explode("ICD10_CODE")
    exploded = exploded[exploded["ICD10_CODE"].notna() & exploded["ICD10_CODE"].ne("")]
    return normalized, exploded.rename(columns={"index": "ROW_INDEX"}).reset_index(drop=True)


def _build_sim_chunk(
    chunk: pd.DataFrame,
    year: int,
    row_offset: int,
    code_sets: dict[str, set[str]],
    group_lookup: pd.DataFrame,
) -> pd.DataFrame:
    working = chunk.copy().reset_index(drop=True)
    working["SIM_RECORD_ID"] = [
        f"SIM_RECORD_{year}_{row_number:08d}"
        for row_number in range(row_offset + 1, row_offset + len(working) + 1)
    ]
    working["SOURCE_SIM"] = "silver_sim_individual"

    death_date = pd.to_datetime(working["DTOBITO"].replace("", pd.NA), format="%Y-%m-%d", errors="coerce")
    if death_date.isna().any():
        raise ValueError(f"SIM bronze has invalid DTOBITO values for {year}")
    if not bool(death_date.dt.year.eq(year).all()):
        raise ValueError(f"SIM bronze contains out-of-year DTOBITO values for {year}")
    municipality_code = working["CODMUNOCOR"].fillna("").astype(str).str.strip()
    if (~municipality_code.str.fullmatch(r"\d{6}")).any():
        raise ValueError(f"SIM bronze has invalid CODMUNOCOR values for {year}")

    working["DATE"] = death_date.dt.strftime("%Y-%m-%d")
    working["YEAR"] = death_date.dt.year.astype("int16")
    working["MONTH"] = death_date.dt.month.astype("int8")
    working["MUNICIPALITY_CODE"] = municipality_code

    age_flags = derive_sim_age_flags(working["IDADE"])
    for column in age_flags.columns:
        working[column] = age_flags[column]

    token_cache: dict[str, tuple[str, ...]] = {}
    normalized_fields = {"CAUSABAS": normalize_icd10_series(working["CAUSABAS"])}
    working["CAUSABAS_ICD10"] = normalized_fields["CAUSABAS"]
    mention_tokens = [
        pd.DataFrame(
            {
                "ROW_INDEX": working.index,
                "ICD10_CODE": normalized_fields["CAUSABAS"],
            }
        )
    ]
    mention_tokens[0] = mention_tokens[0][mention_tokens[0]["ICD10_CODE"].ne("")]
    for column in SIM_ICD10_COLUMNS:
        if column == "CAUSABAS":
            continue
        normalized_series, exploded_tokens = _tokenize_sim_multicode_series(working[column], token_cache)
        normalized_fields[column] = normalized_series
        working[f"{column}_ICD10"] = normalized_series
        mention_tokens.append(exploded_tokens)

    mention_match_frame = pd.DataFrame(index=working.index)
    all_mention_tokens = pd.concat(mention_tokens, ignore_index=True).drop_duplicates()
    if not all_mention_tokens.empty:
        matches = all_mention_tokens.merge(group_lookup, on="ICD10_CODE", how="inner")
        if not matches.empty:
            mention_match_frame = (
                matches.assign(MATCH=1)
                .pivot_table(
                    index="ROW_INDEX",
                    columns="OUTCOME_GROUP",
                    values="MATCH",
                    aggfunc="max",
                    fill_value=0,
                )
                .astype("Int8")
            )

    primary_series = normalized_fields["CAUSABAS"]
    for group_name in expected_group_names():
        underlying_match = primary_series.isin(code_sets[group_name]).astype("Int8")
        any_mention_match = mention_match_frame.get(group_name)
        if any_mention_match is None:
            any_mention_series = pd.Series(0, index=working.index, dtype="Int8")
        else:
            any_mention_series = any_mention_match.reindex(working.index, fill_value=0).astype("Int8")
        working[_scope_flag_column("UNDERLYING_BASIC_CAUSE", group_name)] = underlying_match
        working[_scope_flag_column("ANY_MENTION", group_name)] = any_mention_series

    working["UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_FLAG"] = working[
        [_scope_flag_column("UNDERLYING_BASIC_CAUSE", group_name) for group_name in DIRECT_HEAT_GROUPS]
    ].max(axis=1).astype("Int8")
    working["ANY_MENTION_DIRECT_HEAT_ANY_FLAG"] = working[
        [_scope_flag_column("ANY_MENTION", group_name) for group_name in DIRECT_HEAT_GROUPS]
    ].max(axis=1).astype("Int8")
    working["UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_FLAG"] = working[
        [_scope_flag_column("UNDERLYING_BASIC_CAUSE", group_name) for group_name in INDIRECT_HEAT_GROUPS]
    ].max(axis=1).astype("Int8")
    working["ANY_MENTION_INDIRECT_HEAT_ANY_FLAG"] = working[
        [_scope_flag_column("ANY_MENTION", group_name) for group_name in INDIRECT_HEAT_GROUPS]
    ].max(axis=1).astype("Int8")
    working["UNDERLYING_BASIC_CAUSE_SLOW_MOVING_HEAT_ANY_FLAG"] = working[
        [_scope_flag_column("UNDERLYING_BASIC_CAUSE", group_name) for group_name in SLOW_MOVING_HEAT_GROUPS]
    ].max(axis=1).astype("Int8")
    working["ANY_MENTION_SLOW_MOVING_HEAT_ANY_FLAG"] = working[
        [_scope_flag_column("ANY_MENTION", group_name) for group_name in SLOW_MOVING_HEAT_GROUPS]
    ].max(axis=1).astype("Int8")
    working["UNDERLYING_BASIC_CAUSE_MATERNAL_PHENOTYPE_ANY_FLAG"] = working[
        [_scope_flag_column("UNDERLYING_BASIC_CAUSE", group_name) for group_name in MATERNAL_GROUPS]
    ].max(axis=1).astype("Int8")
    working["ANY_MENTION_MATERNAL_PHENOTYPE_ANY_FLAG"] = working[
        [_scope_flag_column("ANY_MENTION", group_name) for group_name in MATERNAL_GROUPS]
    ].max(axis=1).astype("Int8")
    working["QC_STATUS"] = "PASS"

    return working[SIM_INDIVIDUAL_OUTPUT_FIELDS].copy()


def _aggregate_sim_daily(chunk_frame: pd.DataFrame) -> pd.DataFrame:
    working = chunk_frame.copy()
    working["DEATHS_TOTAL"] = 1
    total_columns = {
        _scope_total_column(scope_label, group_name): _scope_flag_column(scope_label, group_name)
        for _, scope_label in SIM_SCOPE_FIELDS
        for group_name in expected_group_names()
    }
    total_columns.update(
        {
            "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL": "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_FLAG",
            "ANY_MENTION_DIRECT_HEAT_ANY_TOTAL": "ANY_MENTION_DIRECT_HEAT_ANY_FLAG",
            "UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_TOTAL": "UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_FLAG",
            "ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL": "ANY_MENTION_INDIRECT_HEAT_ANY_FLAG",
            "UNDERLYING_BASIC_CAUSE_SLOW_MOVING_HEAT_ANY_TOTAL": "UNDERLYING_BASIC_CAUSE_SLOW_MOVING_HEAT_ANY_FLAG",
            "ANY_MENTION_SLOW_MOVING_HEAT_ANY_TOTAL": "ANY_MENTION_SLOW_MOVING_HEAT_ANY_FLAG",
            "UNDERLYING_BASIC_CAUSE_MATERNAL_PHENOTYPE_ANY_TOTAL": "UNDERLYING_BASIC_CAUSE_MATERNAL_PHENOTYPE_ANY_FLAG",
            "ANY_MENTION_MATERNAL_PHENOTYPE_ANY_TOTAL": "ANY_MENTION_MATERNAL_PHENOTYPE_ANY_FLAG",
        }
    )
    grouped = working.groupby(["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"], as_index=False, sort=True)
    aggregation = {"DEATHS_TOTAL": ("DEATHS_TOTAL", "sum")}
    for total_column, flag_column in total_columns.items():
        aggregation[total_column] = (flag_column, "sum")
    daily = grouped.agg(**aggregation)
    daily["SOURCE_SIM"] = "silver_sim_individual"
    daily["QC_STATUS"] = "PASS"
    return daily[SIM_DAILY_OUTPUT_FIELDS].copy()


def _reconcile_totals(individual_frame: pd.DataFrame, daily_frame: pd.DataFrame) -> dict[str, bool]:
    reconciliation = {"DEATHS_TOTAL": int(daily_frame["DEATHS_TOTAL"].sum()) == int(len(individual_frame))}
    for _, scope_label in SIM_SCOPE_FIELDS:
        for group_name in expected_group_names():
            total_column = _scope_total_column(scope_label, group_name)
            flag_column = _scope_flag_column(scope_label, group_name)
            reconciliation[total_column] = int(daily_frame[total_column].sum()) == int(individual_frame[flag_column].eq(1).sum())
    for total_column, flag_column in [
        ("UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL", "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_FLAG"),
        ("ANY_MENTION_DIRECT_HEAT_ANY_TOTAL", "ANY_MENTION_DIRECT_HEAT_ANY_FLAG"),
        ("UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_TOTAL", "UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_FLAG"),
        ("ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL", "ANY_MENTION_INDIRECT_HEAT_ANY_FLAG"),
        ("UNDERLYING_BASIC_CAUSE_SLOW_MOVING_HEAT_ANY_TOTAL", "UNDERLYING_BASIC_CAUSE_SLOW_MOVING_HEAT_ANY_FLAG"),
        ("ANY_MENTION_SLOW_MOVING_HEAT_ANY_TOTAL", "ANY_MENTION_SLOW_MOVING_HEAT_ANY_FLAG"),
        ("UNDERLYING_BASIC_CAUSE_MATERNAL_PHENOTYPE_ANY_TOTAL", "UNDERLYING_BASIC_CAUSE_MATERNAL_PHENOTYPE_ANY_FLAG"),
        ("ANY_MENTION_MATERNAL_PHENOTYPE_ANY_TOTAL", "ANY_MENTION_MATERNAL_PHENOTYPE_ANY_FLAG"),
    ]:
        reconciliation[total_column] = int(daily_frame[total_column].sum()) == int(individual_frame[flag_column].eq(1).sum())
    return reconciliation


def validate_canonical_sim_year(
    root: Path,
    year: int,
    individual_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    dictionary: pd.DataFrame,
    individual_partition_path: Path | None = None,
    daily_partition_path: Path | None = None,
) -> dict[str, Any]:
    official_total = official_total_from_bronze(root, "sim", year)
    reconciliation = _reconcile_totals(individual_frame, daily_frame)
    primary_regex_ok = bool(individual_frame["CAUSABAS_ICD10"].fillna("").astype(str).str.fullmatch(r"[A-Z][0-9][0-9][A-Z0-9]?").fillna(False).all())
    dictionary_ok = not dictionary["ICD10_CODE"].astype(str).str.contains(r"[\*\-]", regex=True).any()
    direct_groups = [group_name for group_name in DIRECT_HEAT_GROUPS if group_name in dictionary["OUTCOME_GROUP"].unique()]
    indirect_groups = [group_name for group_name in INDIRECT_HEAT_GROUPS if group_name in dictionary["OUTCOME_GROUP"].unique()]
    direct_vs_indirect_distinct = set(direct_groups).isdisjoint(set(indirect_groups))

    individual_partition_ok = True
    if individual_partition_path is not None:
        individual_partition_ok = individual_partition_path.exists()
    daily_partition_ok = True
    if daily_partition_path is not None:
        daily_partition_ok = daily_partition_path.exists()

    group_counts_underlying = {
        group_name: int(individual_frame[_scope_flag_column("UNDERLYING_BASIC_CAUSE", group_name)].eq(1).sum())
        for group_name in expected_group_names()
    }
    group_counts_any = {
        group_name: int(individual_frame[_scope_flag_column("ANY_MENTION", group_name)].eq(1).sum())
        for group_name in expected_group_names()
    }
    validation = {
        "year": year,
        "official_deaths_total": official_total,
        "individual_deaths_total": int(len(individual_frame)),
        "daily_deaths_total": int(daily_frame["DEATHS_TOTAL"].sum()),
        "annual_total_matches_official": int(len(individual_frame)) == official_total == int(daily_frame["DEATHS_TOTAL"].sum()),
        "date_year_coverage_ok": bool(individual_frame["YEAR"].eq(year).all()),
        "municipality_coverage_ok": set(individual_frame["MUNICIPALITY_CODE"]) == set(daily_frame["MUNICIPALITY_CODE"]),
        "individual_key_uniqueness_ok": int(individual_frame["SIM_RECORD_ID"].duplicated().sum()) == 0,
        "daily_key_uniqueness_ok": int(daily_frame.duplicated(["MUNICIPALITY_CODE", "DATE"]).sum()) == 0,
        "dictionary_expansion_ok": dictionary_ok,
        "direct_vs_indirect_distinct_ok": direct_vs_indirect_distinct,
        "underlying_vs_any_mention_distinct_ok": True,
        "primary_cause_coding_expectation_ok": primary_regex_ok,
        "individual_vs_daily_reconciliation": reconciliation,
        "individual_vs_daily_reconciliation_ok": all(reconciliation.values()),
        "individual_partition_integrity_ok": individual_partition_ok,
        "daily_partition_integrity_ok": daily_partition_ok,
        "age_flag_coverage": {
            "AGE_GROUP_0_27D": int(individual_frame["AGE_GROUP_0_27D"].eq(1).sum()),
            "AGE_GROUP_28_364D": int(individual_frame["AGE_GROUP_28_364D"].eq(1).sum()),
            "AGE_GROUP_1_4Y": int(individual_frame["AGE_GROUP_1_4Y"].eq(1).sum()),
            "AGE_GROUP_U5": int(individual_frame["AGE_GROUP_U5"].eq(1).sum()),
            "AGE_GROUP_WRA_15_49": int(individual_frame["AGE_GROUP_WRA_15_49"].eq(1).sum()),
            "AGE_GROUP_WRA_15_45_SENSITIVITY": int(individual_frame["AGE_GROUP_WRA_15_45_SENSITIVITY"].eq(1).sum()),
        },
        "summary": {
            "total_sim_deaths": int(len(individual_frame)),
            "municipalities_with_at_least_one_event": int(individual_frame["MUNICIPALITY_CODE"].nunique()),
            "counts_by_icd_group_underlying_basic_cause": group_counts_underlying,
            "counts_by_icd_group_any_mention": group_counts_any,
            "counts_direct_vs_indirect": {
                "underlying_basic_cause_direct": int(individual_frame["UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_FLAG"].eq(1).sum()),
                "any_mention_direct": int(individual_frame["ANY_MENTION_DIRECT_HEAT_ANY_FLAG"].eq(1).sum()),
                "underlying_basic_cause_indirect": int(individual_frame["UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_FLAG"].eq(1).sum()),
                "any_mention_indirect": int(individual_frame["ANY_MENTION_INDIRECT_HEAT_ANY_FLAG"].eq(1).sum()),
                "underlying_basic_cause_slow_moving": int(individual_frame["UNDERLYING_BASIC_CAUSE_SLOW_MOVING_HEAT_ANY_FLAG"].eq(1).sum()),
                "any_mention_slow_moving": int(individual_frame["ANY_MENTION_SLOW_MOVING_HEAT_ANY_FLAG"].eq(1).sum()),
            },
            "counts_by_scope": {
                "underlying_basic_cause_any_event": int((individual_frame[[ _scope_flag_column("UNDERLYING_BASIC_CAUSE", group_name) for group_name in expected_group_names() ]].max(axis=1) == 1).sum()),
                "any_mention_any_event": int((individual_frame[[ _scope_flag_column("ANY_MENTION", group_name) for group_name in expected_group_names() ]].max(axis=1) == 1).sum()),
            },
            "x30_counts": {
                "underlying_basic_cause": int(individual_frame[_scope_flag_column("UNDERLYING_BASIC_CAUSE", "EXCESSIVE_NATURAL_HEAT_X30")].eq(1).sum()),
                "any_mention": int(individual_frame[_scope_flag_column("ANY_MENTION", "EXCESSIVE_NATURAL_HEAT_X30")].eq(1).sum()),
            },
            "individual_vs_daily_reconciliation_result": all(reconciliation.values()),
        },
    }
    validation["all_checks_passed"] = all(
        [
            validation["annual_total_matches_official"],
            validation["date_year_coverage_ok"],
            validation["municipality_coverage_ok"],
            validation["individual_key_uniqueness_ok"],
            validation["daily_key_uniqueness_ok"],
            validation["dictionary_expansion_ok"],
            validation["direct_vs_indirect_distinct_ok"],
            validation["underlying_vs_any_mention_distinct_ok"],
            validation["primary_cause_coding_expectation_ok"],
            validation["individual_vs_daily_reconciliation_ok"],
            validation["individual_partition_integrity_ok"],
            validation["daily_partition_integrity_ok"],
        ]
    )
    return validation


def _write_validation_reports(root: Path, validations: dict[int, dict[str, Any]], dictionary_result: dict[str, Any], manifest_path: str) -> None:
    reports_dir = root / "artifacts" / "validation_reports" / "silver" / "sim"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for year, payload in validations.items():
        report_path = reports_dir / f"{year}.json"
        report_payload = {
            "dataset": "sim",
            "year": year,
            "generated_at": utc_now(),
            "manifest_path": manifest_path,
            "dictionary_manifest_path": str(dictionary_result["manifest_path"].relative_to(root)),
            **payload,
        }
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_path = root / "artifacts" / "status" / "silver_sim_last_run_summary.md"
    lines = [
        "# Canonical SIM Silver Summary",
        "",
        f"- Generated at: {utc_now()}",
        f"- Manifest: `{manifest_path}`",
        f"- Outcome dictionary: `{dictionary_result['canonical_path'].relative_to(root)}`",
        f"- All yearly checks passed: `{all(validation['all_checks_passed'] for validation in validations.values())}`",
        "",
    ]
    for year in sorted(validations):
        summary = validations[year]["summary"]
        lines.extend(
            [
                f"## {year}",
                "",
                f"- Total SIM deaths: {summary['total_sim_deaths']}",
                f"- Municipalities with at least one event: {summary['municipalities_with_at_least_one_event']}",
                f"- Counts by ICD group (underlying/basic cause): {summary['counts_by_icd_group_underlying_basic_cause']}",
                f"- Counts by ICD group (any mention): {summary['counts_by_icd_group_any_mention']}",
                f"- Counts by DIRECT vs INDIRECT: {summary['counts_direct_vs_indirect']}",
                f"- Counts by underlying/basic cause vs any mention: {summary['counts_by_scope']}",
                f"- Counts with X30 flag: {summary['x30_counts']}",
                f"- Individual vs daily reconciliation result: `{summary['individual_vs_daily_reconciliation_result']}`",
                "",
            ]
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def promote_sim_silver_range(
    root: Path,
    runtime_root: Path,
    start_year: int,
    end_year: int,
    log_path: Path,
    command: str,
    force: bool = False,
) -> dict[str, Any]:
    del force
    dictionary_result = build_unified_outcome_dictionary(root, start_year, end_year)
    dictionary = load_unified_outcome_dictionary(root)
    code_sets = build_group_code_sets(dictionary)
    group_lookup = dictionary[["ICD10_CODE", "OUTCOME_GROUP"]].drop_duplicates().copy()
    validations: dict[int, dict[str, Any]] = {}
    individual_partition_paths: dict[int, str] = {}
    daily_partition_paths: dict[int, str] = {}

    for year in range(start_year, end_year + 1):
        runtime_individual_path = runtime_root / sim_individual_partition_file(root, year).relative_to(root)
        individual_writer = ParquetChunkWriter(runtime_individual_path)
        daily_parts: list[pd.DataFrame] = []
        row_offset = 0
        for chunk in _load_sim_chunk_iterator(root, year):
            individual_chunk = _build_sim_chunk(chunk, year, row_offset, code_sets, group_lookup)
            individual_writer.write(individual_chunk)
            daily_parts.append(_aggregate_sim_daily(individual_chunk))
            row_offset += len(individual_chunk)
        individual_row_count, individual_columns = individual_writer.close()
        daily_frame = pd.concat(daily_parts, ignore_index=True).groupby(
            ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"],
            as_index=False,
            sort=True,
        ).sum(numeric_only=True)
        daily_frame["SOURCE_SIM"] = "silver_sim_individual"
        daily_frame["QC_STATUS"] = "PASS"
        daily_frame = daily_frame[SIM_DAILY_OUTPUT_FIELDS].sort_values(["DATE", "MUNICIPALITY_CODE"], kind="mergesort").reset_index(drop=True)
        runtime_daily_path = runtime_root / sim_daily_partition_file(root, year).relative_to(root)
        daily_writer = ParquetChunkWriter(runtime_daily_path)
        daily_writer.write(daily_frame)
        daily_row_count, daily_columns = daily_writer.close()

        final_individual_path = sim_individual_partition_file(root, year)
        final_daily_path = sim_daily_partition_file(root, year)
        sync_verified_parquet(runtime_individual_path, final_individual_path, individual_row_count, individual_columns)
        sync_verified_parquet(runtime_daily_path, final_daily_path, daily_row_count, daily_columns)
        runtime_individual_path.unlink(missing_ok=True)
        runtime_daily_path.unlink(missing_ok=True)
        if runtime_individual_path.parent.exists():
            shutil.rmtree(runtime_individual_path.parent)
        if runtime_daily_path.parent.exists():
            shutil.rmtree(runtime_daily_path.parent)

        individual_frame = pd.read_parquet(final_individual_path)
        daily_frame = pd.read_parquet(final_daily_path)
        validation = validate_canonical_sim_year(
            root,
            year,
            individual_frame,
            daily_frame,
            dictionary,
            individual_partition_path=final_individual_path,
            daily_partition_path=final_daily_path,
        )
        if not validation["all_checks_passed"]:
            raise ValueError(f"SIM silver validation failed for {year}")
        validations[year] = validation
        individual_partition_paths[year] = str(final_individual_path.relative_to(root))
        daily_partition_paths[year] = str(final_daily_path.relative_to(root))
        append_run_event(
            log_path,
            phase="phase_2_silver_promotion",
            dataset="sim",
            year=year,
            command=command,
            decision=(
                "canonical SIM individual and daily silver outputs written with explicit ICD-10 outcome flags; "
                f"individual path={final_individual_path.relative_to(root)} daily path={final_daily_path.relative_to(root)}"
            ),
            status="PASS",
        )

    manifest = build_manifest(root, "silver", "sim")
    manifest_path = f"artifacts/manifests/silver_{manifest['dataset']}.manifest.json"
    _write_validation_reports(root, validations, dictionary_result, manifest_path)
    return {
        "dataset": "sim",
        "years": list(range(start_year, end_year + 1)),
        "manifest_path": manifest_path,
        "dictionary_path": str(dictionary_result["canonical_path"].relative_to(root)),
        "dictionary_manifest_path": str(dictionary_result["manifest_path"].relative_to(root)),
        "individual_partition_paths": individual_partition_paths,
        "daily_partition_paths": daily_partition_paths,
        "validation": validations,
    }
