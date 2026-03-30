from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .bronze import (
    load_datasus_popsvs_canonical_frame,
    official_population_codes,
    population_derivative_expected_codes,
)
from .manifest import build_manifest, count_rows, sha256_file
from .run_log import append_run_event
from .silver_outcome_common import ParquetChunkWriter, sync_verified_parquet


UNDER5_DATASET_NAME = "silver_pop_under5"
WRA_15_49_DATASET_NAME = "silver_pop_wra_15_49"
WRA_15_45_DATASET_NAME = "silver_pop_wra_15_45_sensitivity"
ANALYTICAL_DATASET = "analytical"
POPULATION_DATASET = "population"
RELEASE_MANIFEST_NAME = "silver_release_manifest.json"
RELEASE_REPORT_NAME = "silver_release_candidate_report.md"
RELEASE_MATRIX_NAME = "silver_release_candidate_matrix.csv"
READBACK_TOLERANCE = 1e-9

POPULATION_COMMON_FIELDS = [
    "SOURCE_POPULATION",
    "BRONZE_SOURCE_SYSTEM",
    "BRONZE_SOURCE_FILE",
    "BRONZE_SOURCE_PATH",
    "BRONZE_SOURCE_MEMBER",
    "BRONZE_YEAR",
    "BRONZE_INGESTED_AT",
    "MUNICIPALITY_CODE_IBGE7",
    "MUNICIPALITY_CODE",
    "YEAR",
    "DENOMINATOR_NAME",
    "DENOMINATOR_SCOPE",
]
POP_UNDER5_FIELDS = POPULATION_COMMON_FIELDS + ["POP_UNDER5", "QC_STATUS"]
POP_WRA_15_49_FIELDS = POPULATION_COMMON_FIELDS + ["POP_WRA_15_49", "QC_STATUS"]
POP_WRA_15_45_FIELDS = POPULATION_COMMON_FIELDS + ["POP_WRA_15_45_SENSITIVITY", "QC_STATUS"]

SIM_CHILD_FLAG_COLUMNS = [
    "UNDERLYING_BASIC_CAUSE_CHILD_HEAT_THERMAL_PRIMARY_FLAG",
    "UNDERLYING_BASIC_CAUSE_CHILD_DEHYDRATION_PRIMARY_FLAG",
    "ANY_MENTION_CHILD_FLUID_ELECTROLYTE_SENSITIVITY_FLAG",
    "UNDERLYING_BASIC_CAUSE_CHILD_DIARRHEA_GI_PRIMARY_FLAG",
    "UNDERLYING_BASIC_CAUSE_CHILD_RESPIRATORY_ACUTE_PRIMARY_FLAG",
    "ANY_MENTION_CHILD_ASTHMA_WHEEZE_SENSITIVITY_FLAG",
    "ANY_MENTION_CHILD_RENAL_ACUTE_SENSITIVITY_FLAG",
    "ANY_MENTION_CHILD_MALNUTRITION_SLOW_MOVING_SENSITIVITY_FLAG",
    "ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_FLAG",
    "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_FLAG",
    "ANY_MENTION_DIRECT_HEAT_ANY_FLAG",
    "UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_FLAG",
    "ANY_MENTION_INDIRECT_HEAT_ANY_FLAG",
    "UNDERLYING_BASIC_CAUSE_SLOW_MOVING_HEAT_ANY_FLAG",
    "ANY_MENTION_SLOW_MOVING_HEAT_ANY_FLAG",
]
SIH_CHILD_FLAG_COLUMNS = [
    "PRINCIPAL_DIAGNOSIS_CHILD_HEAT_THERMAL_PRIMARY_FLAG",
    "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_FLAG",
    "ANY_DIAGNOSIS_CHILD_FLUID_ELECTROLYTE_SENSITIVITY_FLAG",
    "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_FLAG",
    "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_FLAG",
    "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_FLAG",
    "ANY_DIAGNOSIS_CHILD_RENAL_ACUTE_SENSITIVITY_FLAG",
    "ANY_DIAGNOSIS_CHILD_MALNUTRITION_SLOW_MOVING_SENSITIVITY_FLAG",
    "EXTERNAL_CAUSE_FIELDS_EXCESSIVE_NATURAL_HEAT_X30_FLAG",
    "PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG",
    "ANY_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG",
    "PRINCIPAL_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG",
    "ANY_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG",
    "PRINCIPAL_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG",
    "ANY_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG",
]
SIH_MATERNAL_FLAG_COLUMNS = [
    "PRINCIPAL_DIAGNOSIS_HDP_ANY_PRIMARY_FLAG",
    "PRINCIPAL_DIAGNOSIS_HDP_SEVERE_PRIMARY_FLAG",
    "PRINCIPAL_DIAGNOSIS_PRETERM_LABOR_PRIMARY_FLAG",
    "ANY_DIAGNOSIS_THREATENED_PRETERM_LABOR_SENSITIVITY_FLAG",
    "PRINCIPAL_DIAGNOSIS_GDM_GESTATIONAL_PRIMARY_FLAG",
    "ANY_DIAGNOSIS_DIABETES_PREGNANCY_BROAD_SENSITIVITY_FLAG",
    "PRINCIPAL_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_FLAG",
    "ANY_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_FLAG",
]
SIM_CHILD_OUTPUT_FIELDS = [
    "AGE_SCOPE",
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
    "DEATHS_TOTAL",
    *[column.replace("_FLAG", "_TOTAL") for column in SIM_CHILD_FLAG_COLUMNS],
    "POP_UNDER5",
    "DENOMINATOR_NAME",
    "DENOMINATOR_VALUE",
    "SOURCE_ANALYTICAL_VIEW",
    "SOURCE_SIM",
    "QC_STATUS",
]
SIH_CHILD_OUTPUT_FIELDS = [
    "AGE_SCOPE",
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
    "COMPETENCE_YEAR",
    "COMPETENCE_MONTH",
    "ADMISSIONS_TOTAL",
    *[column.replace("_FLAG", "_TOTAL") for column in SIH_CHILD_FLAG_COLUMNS],
    "POP_UNDER5",
    "DENOMINATOR_NAME",
    "DENOMINATOR_VALUE",
    "SOURCE_ANALYTICAL_VIEW",
    "SOURCE_SIH",
    "QC_STATUS",
]
SIH_MATERNAL_15_49_OUTPUT_FIELDS = [
    "AGE_SCOPE",
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
    "COMPETENCE_YEAR",
    "COMPETENCE_MONTH",
    "ADMISSIONS_TOTAL",
    *[column.replace("_FLAG", "_TOTAL") for column in SIH_MATERNAL_FLAG_COLUMNS],
    "POP_WRA_15_49",
    "DENOMINATOR_NAME",
    "DENOMINATOR_VALUE",
    "SOURCE_ANALYTICAL_VIEW",
    "SOURCE_SIH",
    "QC_STATUS",
]
SIH_MATERNAL_15_45_OUTPUT_FIELDS = [
    "AGE_SCOPE",
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
    "COMPETENCE_YEAR",
    "COMPETENCE_MONTH",
    "ADMISSIONS_TOTAL",
    *[column.replace("_FLAG", "_TOTAL") for column in SIH_MATERNAL_FLAG_COLUMNS],
    "POP_WRA_15_45_SENSITIVITY",
    "DENOMINATOR_NAME",
    "DENOMINATOR_VALUE",
    "SOURCE_ANALYTICAL_VIEW",
    "SOURCE_SIH",
    "QC_STATUS",
]
CHILD_AGE_SCOPES = (
    ("AGE_0_27D", "AGE_GROUP_0_27D"),
    ("AGE_28_364D", "AGE_GROUP_28_364D"),
    ("AGE_1_4Y", "AGE_GROUP_1_4Y"),
    ("AGE_U5", "AGE_GROUP_U5"),
)
MATERNAL_15_49_SCOPE = (("AGE_WRA_15_49", "AGE_GROUP_WRA_15_49"),)
MATERNAL_15_45_SCOPE = (("AGE_WRA_15_45_SENSITIVITY", "AGE_GROUP_WRA_15_45_SENSITIVITY"),)
EXISTING_SILVER_DATASETS = (
    "climate",
    "sinasc",
    "sim",
    "sih",
    "stillbirth",
)
EXISTING_SILVER_VALIDATION_DATASETS = (
    "climate",
    "sinasc",
    "sim",
    "sih",
    "stillbirth",
)


@dataclass(frozen=True)
class PopulationSpec:
    dataset_name: str
    value_column: str
    denominator_name: str
    denominator_scope: str
    source_kind: str
    age_min: int | None = None
    age_max: int | None = None


POPULATION_SPECS = (
    PopulationSpec(
        dataset_name=UNDER5_DATASET_NAME,
        value_column="POP_UNDER5",
        denominator_name="POP_UNDER5",
        denominator_scope="AGE_U5",
        source_kind="bronze_pop_under5",
    ),
    PopulationSpec(
        dataset_name=WRA_15_49_DATASET_NAME,
        value_column="POP_WRA_15_49",
        denominator_name="POP_WRA_15_49",
        denominator_scope="AGE_WRA_15_49",
        source_kind="datasus_popsvs",
        age_min=15,
        age_max=49,
    ),
    PopulationSpec(
        dataset_name=WRA_15_45_DATASET_NAME,
        value_column="POP_WRA_15_45_SENSITIVITY",
        denominator_name="POP_WRA_15_45_SENSITIVITY",
        denominator_scope="AGE_WRA_15_45_SENSITIVITY",
        source_kind="datasus_popsvs",
        age_min=15,
        age_max=45,
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _population_partition_file(root: Path, dataset_name: str, year: int) -> Path:
    return root / "data" / "silver" / POPULATION_DATASET / dataset_name / f"YEAR={year}" / f"{dataset_name}_{year}.parquet"


def _analytical_partition_file(root: Path, dataset_name: str, year: int) -> Path:
    return root / "data" / "silver" / ANALYTICAL_DATASET / dataset_name / f"YEAR={year}" / f"{dataset_name}_{year}.parquet"


def _validation_report_dir(root: Path) -> Path:
    return root / "artifacts" / "validation_reports" / "silver" / "release_candidate"


def _build_release_manifest_path(root: Path) -> Path:
    return root / "artifacts" / "manifests" / RELEASE_MANIFEST_NAME


def _build_release_report_path(root: Path) -> Path:
    return root / "artifacts" / "status" / RELEASE_REPORT_NAME


def _build_release_matrix_path(root: Path) -> Path:
    return root / "artifacts" / "status" / RELEASE_MATRIX_NAME


def _remove_partition_if_exists(path: Path) -> None:
    if path.parent.exists():
        shutil.rmtree(path.parent)


def _population_output_fields(dataset_name: str) -> list[str]:
    if dataset_name == UNDER5_DATASET_NAME:
        return POP_UNDER5_FIELDS
    if dataset_name == WRA_15_49_DATASET_NAME:
        return POP_WRA_15_49_FIELDS
    if dataset_name == WRA_15_45_DATASET_NAME:
        return POP_WRA_15_45_FIELDS
    raise ValueError(f"Unsupported population dataset: {dataset_name}")


def _official_municipality_codes6(root: Path, year: int) -> set[str]:
    return {code[:6] for code in official_population_codes(root, year)}


def _blocked_analytical_validation(
    dataset_name: str,
    year: int,
    expected_age_scopes: list[str],
    denominator_column: str,
    blocking_issue: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "year": year,
        "row_count": 0,
        "age_scope_presence": {scope: False for scope in expected_age_scopes},
        "age_coverage_ok": False,
        "schema_contract_ok": False,
        "municipality_key_ok": False,
        "date_year_coverage_ok": False,
        "duplicates_zero": True,
        "duplicate_count": 0,
        "critical_missing_zero": True,
        "critical_missing_count": 0,
        "denominator_column": denominator_column,
        "denominator_missing_count": 0,
        "excluded_nonmunicipal_rows": 0,
        "internal_reconciliation_ok": False,
        "internal_reconciliation": {},
        "blocking_issue": blocking_issue,
        "all_checks_passed": False,
    }


def _read_parquet_or_none(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _iter_parquet_batches(path: Path, columns: list[str], batch_size: int = 250_000):
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def _normalize_population_keys(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["MUNICIPALITY_CODE_IBGE7"] = working["MUNICIPALITY_CODE_IBGE7"].astype(str).str.zfill(7)
    working["MUNICIPALITY_CODE"] = working["MUNICIPALITY_CODE_IBGE7"].str[:6]
    return working


def _validate_population_frame(
    root: Path,
    year: int,
    frame: pd.DataFrame,
    value_column: str,
    expected_fields: list[str],
) -> dict[str, Any]:
    expected_codes = set(population_derivative_expected_codes(root, year))
    observed_codes = set(frame["MUNICIPALITY_CODE_IBGE7"].tolist())
    missing_codes = sorted(expected_codes - observed_codes)
    unexpected_codes = sorted(observed_codes - expected_codes)
    duplicate_ibge7 = int(frame.duplicated(["MUNICIPALITY_CODE_IBGE7", "YEAR"]).sum())
    duplicate_6d = int(frame.duplicated(["MUNICIPALITY_CODE", "YEAR"]).sum())
    null_count = int(frame[["MUNICIPALITY_CODE_IBGE7", "MUNICIPALITY_CODE", "YEAR", value_column]].isna().sum().sum())
    negative_value_count = int(frame[value_column].fillna(-1).astype("float64").lt(0).sum())
    schema_contract_ok = list(frame.columns) == list(expected_fields)
    year_coverage_ok = bool(frame["YEAR"].notna().all() and frame["YEAR"].astype("int64").eq(year).all())
    municipality_key_ok = bool(
        frame["MUNICIPALITY_CODE_IBGE7"].astype(str).str.fullmatch(r"\d{7}").all()
        and frame["MUNICIPALITY_CODE"].astype(str).str.fullmatch(r"\d{6}").all()
    )
    return {
        "expected_municipality_years": int(len(expected_codes)),
        "observed_municipality_years": int(len(frame)),
        "municipalities_covered": int(frame["MUNICIPALITY_CODE_IBGE7"].nunique()),
        "schema_contract_ok": schema_contract_ok,
        "year_coverage_ok": year_coverage_ok,
        "municipality_key_ok": municipality_key_ok,
        "duplicate_ibge7_count": duplicate_ibge7,
        "duplicate_6d_count": duplicate_6d,
        "missing_code_count": int(len(missing_codes)),
        "missing_codes_sample": missing_codes[:10],
        "unexpected_code_count": int(len(unexpected_codes)),
        "unexpected_codes_sample": unexpected_codes[:10],
        "null_critical_count": null_count,
        "negative_value_count": negative_value_count,
        "duplicates_zero": duplicate_ibge7 == 0 and duplicate_6d == 0,
        "missing_municipality_years_zero": len(missing_codes) == 0,
        "critical_missing_zero": null_count == 0 and negative_value_count == 0,
        "all_checks_passed": (
            schema_contract_ok
            and year_coverage_ok
            and municipality_key_ok
            and duplicate_ibge7 == 0
            and duplicate_6d == 0
            and len(missing_codes) == 0
            and len(unexpected_codes) == 0
            and null_count == 0
            and negative_value_count == 0
        ),
    }


def _load_under5_bronze_frame(root: Path, year: int) -> pd.DataFrame:
    path = root / "data" / "bronze" / "population" / "under5" / str(year) / f"pop_under5_{year}_source_preserving.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Bronze pop_under5 missing for {year}: {path}")
    frame = pd.read_csv(path, compression="gzip", sep=";", dtype=str, keep_default_na=False, low_memory=False)
    required = [
        "BRONZE_SOURCE_SYSTEM",
        "BRONZE_SOURCE_FILE",
        "BRONZE_SOURCE_PATH",
        "BRONZE_SOURCE_MEMBER",
        "BRONZE_YEAR",
        "BRONZE_INGESTED_AT",
        "COD_MUNICIPIO",
        "ANO",
        "POP_UNDER5",
    ]
    missing = [field for field in required if field not in frame.columns]
    if missing:
        raise ValueError(f"Bronze pop_under5 missing required fields for {year}: {missing}")
    result = pd.DataFrame(
        {
            "SOURCE_POPULATION": UNDER5_DATASET_NAME,
            "BRONZE_SOURCE_SYSTEM": frame["BRONZE_SOURCE_SYSTEM"].astype(str),
            "BRONZE_SOURCE_FILE": frame["BRONZE_SOURCE_FILE"].astype(str),
            "BRONZE_SOURCE_PATH": frame["BRONZE_SOURCE_PATH"].astype(str),
            "BRONZE_SOURCE_MEMBER": frame["BRONZE_SOURCE_MEMBER"].astype(str),
            "BRONZE_YEAR": pd.to_numeric(frame["BRONZE_YEAR"], errors="coerce").astype("Int64"),
            "BRONZE_INGESTED_AT": frame["BRONZE_INGESTED_AT"].astype(str),
            "MUNICIPALITY_CODE_IBGE7": frame["COD_MUNICIPIO"].astype(str).str.zfill(7),
            "YEAR": pd.to_numeric(frame["ANO"], errors="coerce").astype("Int64"),
            "DENOMINATOR_NAME": "POP_UNDER5",
            "DENOMINATOR_SCOPE": "AGE_U5",
            "POP_UNDER5": pd.to_numeric(frame["POP_UNDER5"], errors="coerce").astype("Int64"),
            "QC_STATUS": "PASS",
        }
    )
    result = _normalize_population_keys(result)
    return result[POP_UNDER5_FIELDS].sort_values(["YEAR", "MUNICIPALITY_CODE_IBGE7"], kind="mergesort").reset_index(drop=True)


def _load_wra_frame_from_popsvs(root: Path, year: int, dataset_name: str, value_column: str, age_min: int, age_max: int) -> pd.DataFrame:
    source_frame, metadata, source_path = load_datasus_popsvs_canonical_frame(root, year)
    filtered = source_frame[
        source_frame["ANO_NUM"].eq(year)
        & source_frame["SEXO_NUM"].eq(2)
        & source_frame["IDADE_NUM"].between(age_min, age_max, inclusive="both")
    ].copy()
    aggregated = (
        filtered.groupby("COD_MUN", as_index=False)["POP_NUM"]
        .sum()
        .rename(columns={"COD_MUN": "MUNICIPALITY_CODE_IBGE7", "POP_NUM": value_column})
    )
    result = pd.DataFrame(
        {
            "SOURCE_POPULATION": dataset_name,
            "BRONZE_SOURCE_SYSTEM": metadata.get("BRONZE_SOURCE_SYSTEM", "datasus_popsvs"),
            "BRONZE_SOURCE_FILE": metadata.get("BRONZE_SOURCE_FILE", source_path.name),
            "BRONZE_SOURCE_PATH": metadata.get("BRONZE_SOURCE_PATH", str(source_path.relative_to(root))),
            "BRONZE_SOURCE_MEMBER": metadata.get("BRONZE_SOURCE_MEMBER", ""),
            "BRONZE_YEAR": year,
            "BRONZE_INGESTED_AT": metadata.get("BRONZE_INGESTED_AT", ""),
            "MUNICIPALITY_CODE_IBGE7": aggregated["MUNICIPALITY_CODE_IBGE7"].astype(str).str.zfill(7),
            "YEAR": year,
            "DENOMINATOR_NAME": value_column,
            "DENOMINATOR_SCOPE": "AGE_WRA_15_49" if value_column == "POP_WRA_15_49" else "AGE_WRA_15_45_SENSITIVITY",
            value_column: aggregated[value_column].astype("Int64"),
            "QC_STATUS": "PASS",
        }
    )
    result = _normalize_population_keys(result)
    fields = POP_WRA_15_49_FIELDS if value_column == "POP_WRA_15_49" else POP_WRA_15_45_FIELDS
    return result[fields].sort_values(["YEAR", "MUNICIPALITY_CODE_IBGE7"], kind="mergesort").reset_index(drop=True)


def _write_population_partition(
    root: Path,
    runtime_root: Path,
    dataset_name: str,
    year: int,
    frame: pd.DataFrame,
) -> str:
    runtime_path = runtime_root / _population_partition_file(root, dataset_name, year).relative_to(root)
    writer = ParquetChunkWriter(runtime_path)
    writer.write(frame)
    row_count, columns = writer.close()
    final_path = _population_partition_file(root, dataset_name, year)
    sync_verified_parquet(runtime_path, final_path, row_count, columns)
    runtime_path.unlink(missing_ok=True)
    if runtime_path.parent.exists():
        shutil.rmtree(runtime_path.parent)
    return str(final_path.relative_to(root))


def _load_population_lookup(root: Path, dataset_name: str, year: int, value_column: str) -> pd.DataFrame | None:
    path = _population_partition_file(root, dataset_name, year)
    if not path.exists():
        return None
    lookup = pd.read_parquet(path, columns=["MUNICIPALITY_CODE", "YEAR", value_column])
    return lookup.drop_duplicates(["MUNICIPALITY_CODE", "YEAR"], keep="first").reset_index(drop=True)


def _aggregate_daily_from_flags(
    frame: pd.DataFrame,
    age_scopes: tuple[tuple[str, str], ...],
    key_columns: list[str],
    count_column: str,
    flag_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    results: list[pd.DataFrame] = []
    summary: dict[str, dict[str, int]] = {}
    for age_scope, age_flag_column in age_scopes:
        subset = frame.loc[frame[age_flag_column].eq(1), key_columns + flag_columns].copy()
        if subset.empty:
            summary[age_scope] = {count_column: 0, **{flag.replace("_FLAG", "_TOTAL"): 0 for flag in flag_columns}}
            continue
        subset["AGE_SCOPE"] = age_scope
        subset[count_column] = 1
        groupby_columns = ["AGE_SCOPE", *key_columns]
        grouped = subset.groupby(groupby_columns, as_index=False, sort=True).sum(numeric_only=True)
        rename_map = {flag: flag.replace("_FLAG", "_TOTAL") for flag in flag_columns}
        grouped = grouped.rename(columns=rename_map)
        results.append(grouped)
        summary[age_scope] = {
            count_column: int(subset[count_column].sum()),
            **{rename_map[flag]: int(subset[flag].fillna(0).sum()) for flag in flag_columns},
        }
    if not results:
        empty_columns = ["AGE_SCOPE", *key_columns, count_column, *[flag.replace("_FLAG", "_TOTAL") for flag in flag_columns]]
        return pd.DataFrame(columns=empty_columns), summary
    return pd.concat(results, ignore_index=True), summary


def _compact_daily_parts(parts: list[pd.DataFrame], group_columns: list[str]) -> list[pd.DataFrame]:
    if len(parts) < 16:
        return parts
    combined = pd.concat(parts, ignore_index=True)
    numeric_columns = [column for column in combined.columns if column not in group_columns]
    compacted = combined.groupby(group_columns, as_index=False, sort=True)[numeric_columns].sum(numeric_only=True)
    return [compacted]


def _build_child_sim_daily_view(root: Path, runtime_root: Path, year: int) -> tuple[dict[str, Any], str]:
    path = root / "data" / "silver" / "sim" / "silver_sim_individual" / f"YEAR={year}" / f"silver_sim_individual_{year}.parquet"
    lookup = _load_population_lookup(root, UNDER5_DATASET_NAME, year, "POP_UNDER5")
    if lookup is None:
        _remove_partition_if_exists(_analytical_partition_file(root, "silver_child_u5_sim_daily", year))
        return (
            _blocked_analytical_validation(
                dataset_name="silver_child_u5_sim_daily",
                year=year,
                expected_age_scopes=[scope for scope, _ in CHILD_AGE_SCOPES],
                denominator_column="POP_UNDER5",
                blocking_issue=f"required denominator partition missing: {UNDER5_DATASET_NAME} {year}",
            ),
            "",
        )
    valid_municipality_codes = set(lookup["MUNICIPALITY_CODE"].astype(str))
    required_columns = ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", *[flag for _, flag in CHILD_AGE_SCOPES], *SIM_CHILD_FLAG_COLUMNS]
    daily_parts: list[pd.DataFrame] = []
    summary: dict[str, dict[str, int]] = {age_scope: {} for age_scope, _ in CHILD_AGE_SCOPES}
    excluded_nonmunicipal_rows = 0
    for batch in _iter_parquet_batches(path, required_columns):
        batch = batch.copy()
        batch["MUNICIPALITY_CODE"] = batch["MUNICIPALITY_CODE"].astype(str)
        valid_mask = batch["MUNICIPALITY_CODE"].isin(valid_municipality_codes)
        excluded_nonmunicipal_rows += int((~valid_mask).sum())
        batch = batch.loc[valid_mask].copy()
        if batch.empty:
            continue
        daily_batch, batch_summary = _aggregate_daily_from_flags(
            batch,
            CHILD_AGE_SCOPES,
            key_columns=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"],
            count_column="DEATHS_TOTAL",
            flag_columns=SIM_CHILD_FLAG_COLUMNS,
        )
        if not daily_batch.empty:
            daily_parts.append(daily_batch)
            daily_parts = _compact_daily_parts(daily_parts, ["AGE_SCOPE", "MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"])
        for age_scope, values in batch_summary.items():
            summary.setdefault(age_scope, {})
            for key, value in values.items():
                summary[age_scope][key] = summary[age_scope].get(key, 0) + value
    if daily_parts:
        daily = pd.concat(daily_parts, ignore_index=True).groupby(
            ["AGE_SCOPE", "MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"],
            as_index=False,
            sort=True,
        ).sum(numeric_only=True)
    else:
        daily = pd.DataFrame(columns=SIM_CHILD_OUTPUT_FIELDS)
    daily = daily.merge(lookup, on=["MUNICIPALITY_CODE", "YEAR"], how="left")
    daily["DENOMINATOR_NAME"] = pd.NA
    daily["DENOMINATOR_VALUE"] = pd.NA
    u5_mask = daily["AGE_SCOPE"].eq("AGE_U5")
    daily.loc[u5_mask, "DENOMINATOR_NAME"] = "POP_UNDER5"
    daily.loc[u5_mask, "DENOMINATOR_VALUE"] = daily.loc[u5_mask, "POP_UNDER5"]
    daily["SOURCE_ANALYTICAL_VIEW"] = "silver_child_u5_sim_daily"
    daily["SOURCE_SIM"] = "silver_sim_individual"
    daily["QC_STATUS"] = "PASS"
    daily = daily[SIM_CHILD_OUTPUT_FIELDS].sort_values(["AGE_SCOPE", "DATE", "MUNICIPALITY_CODE"], kind="mergesort").reset_index(drop=True)
    validation = _validate_analytical_daily_view(
        dataset_name="silver_child_u5_sim_daily",
        year=year,
        daily_frame=daily,
        expected_age_scopes=[scope for scope, _ in CHILD_AGE_SCOPES],
        summary_counts=summary,
        count_column="DEATHS_TOTAL",
        denominator_column="POP_UNDER5",
        denominator_required_age_scope="AGE_U5",
        expected_fields=SIM_CHILD_OUTPUT_FIELDS,
        excluded_nonmunicipal_rows=excluded_nonmunicipal_rows,
    )
    final_path = _analytical_partition_file(root, "silver_child_u5_sim_daily", year)
    if not validation["all_checks_passed"]:
        _remove_partition_if_exists(final_path)
        return validation, ""
    runtime_path = runtime_root / final_path.relative_to(root)
    writer = ParquetChunkWriter(runtime_path)
    writer.write(daily)
    row_count, columns = writer.close()
    sync_verified_parquet(runtime_path, final_path, row_count, columns)
    runtime_path.unlink(missing_ok=True)
    if runtime_path.parent.exists():
        shutil.rmtree(runtime_path.parent)
    return validation, str(final_path.relative_to(root))


def _build_child_sih_daily_view(root: Path, runtime_root: Path, year: int) -> tuple[dict[str, Any], str]:
    path = root / "data" / "silver" / "sih" / "silver_sih_individual" / f"YEAR={year}" / f"silver_sih_individual_{year}.parquet"
    lookup = _load_population_lookup(root, UNDER5_DATASET_NAME, year, "POP_UNDER5")
    if lookup is None:
        _remove_partition_if_exists(_analytical_partition_file(root, "silver_child_u5_sih_daily", year))
        return (
            _blocked_analytical_validation(
                dataset_name="silver_child_u5_sih_daily",
                year=year,
                expected_age_scopes=[scope for scope, _ in CHILD_AGE_SCOPES],
                denominator_column="POP_UNDER5",
                blocking_issue=f"required denominator partition missing: {UNDER5_DATASET_NAME} {year}",
            ),
            "",
        )
    valid_municipality_codes = set(lookup["MUNICIPALITY_CODE"].astype(str))
    lookup = lookup.rename(columns={"YEAR": "COMPETENCE_YEAR"})
    required_columns = [
        "MUNICIPALITY_CODE",
        "DATE",
        "YEAR",
        "MONTH",
        "COMPETENCE_YEAR",
        "COMPETENCE_MONTH",
        *[flag for _, flag in CHILD_AGE_SCOPES],
        *SIH_CHILD_FLAG_COLUMNS,
    ]
    daily_parts: list[pd.DataFrame] = []
    summary: dict[str, dict[str, int]] = {age_scope: {} for age_scope, _ in CHILD_AGE_SCOPES}
    excluded_nonmunicipal_rows = 0
    for batch in _iter_parquet_batches(path, required_columns):
        batch = batch.copy()
        batch["MUNICIPALITY_CODE"] = batch["MUNICIPALITY_CODE"].astype(str)
        valid_mask = batch["MUNICIPALITY_CODE"].isin(valid_municipality_codes)
        excluded_nonmunicipal_rows += int((~valid_mask).sum())
        batch = batch.loc[valid_mask].copy()
        if batch.empty:
            continue
        daily_batch, batch_summary = _aggregate_daily_from_flags(
            batch,
            CHILD_AGE_SCOPES,
            key_columns=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", "COMPETENCE_YEAR", "COMPETENCE_MONTH"],
            count_column="ADMISSIONS_TOTAL",
            flag_columns=SIH_CHILD_FLAG_COLUMNS,
        )
        if not daily_batch.empty:
            daily_parts.append(daily_batch)
            daily_parts = _compact_daily_parts(
                daily_parts,
                ["AGE_SCOPE", "MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", "COMPETENCE_YEAR", "COMPETENCE_MONTH"],
            )
        for age_scope, values in batch_summary.items():
            summary.setdefault(age_scope, {})
            for key, value in values.items():
                summary[age_scope][key] = summary[age_scope].get(key, 0) + value
    if daily_parts:
        daily = pd.concat(daily_parts, ignore_index=True).groupby(
            ["AGE_SCOPE", "MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", "COMPETENCE_YEAR", "COMPETENCE_MONTH"],
            as_index=False,
            sort=True,
        ).sum(numeric_only=True)
    else:
        daily = pd.DataFrame(columns=SIH_CHILD_OUTPUT_FIELDS)
    daily = daily.merge(lookup, on=["MUNICIPALITY_CODE", "COMPETENCE_YEAR"], how="left")
    daily["DENOMINATOR_NAME"] = pd.NA
    daily["DENOMINATOR_VALUE"] = pd.NA
    u5_mask = daily["AGE_SCOPE"].eq("AGE_U5")
    daily.loc[u5_mask, "DENOMINATOR_NAME"] = "POP_UNDER5"
    daily.loc[u5_mask, "DENOMINATOR_VALUE"] = daily.loc[u5_mask, "POP_UNDER5"]
    daily["SOURCE_ANALYTICAL_VIEW"] = "silver_child_u5_sih_daily"
    daily["SOURCE_SIH"] = "silver_sih_individual"
    daily["QC_STATUS"] = "PASS"
    daily = daily[SIH_CHILD_OUTPUT_FIELDS].sort_values(
        ["AGE_SCOPE", "DATE", "MUNICIPALITY_CODE", "COMPETENCE_YEAR", "COMPETENCE_MONTH"],
        kind="mergesort",
    ).reset_index(drop=True)
    validation = _validate_analytical_daily_view(
        dataset_name="silver_child_u5_sih_daily",
        year=year,
        daily_frame=daily,
        expected_age_scopes=[scope for scope, _ in CHILD_AGE_SCOPES],
        summary_counts=summary,
        count_column="ADMISSIONS_TOTAL",
        denominator_column="POP_UNDER5",
        denominator_required_age_scope="AGE_U5",
        expected_fields=SIH_CHILD_OUTPUT_FIELDS,
        excluded_nonmunicipal_rows=excluded_nonmunicipal_rows,
    )
    final_path = _analytical_partition_file(root, "silver_child_u5_sih_daily", year)
    if not validation["all_checks_passed"]:
        _remove_partition_if_exists(final_path)
        return validation, ""
    runtime_path = runtime_root / final_path.relative_to(root)
    writer = ParquetChunkWriter(runtime_path)
    writer.write(daily)
    row_count, columns = writer.close()
    sync_verified_parquet(runtime_path, final_path, row_count, columns)
    runtime_path.unlink(missing_ok=True)
    if runtime_path.parent.exists():
        shutil.rmtree(runtime_path.parent)
    return validation, str(final_path.relative_to(root))


def _build_maternal_sih_daily_view(
    root: Path,
    runtime_root: Path,
    year: int,
    dataset_name: str,
    age_scopes: tuple[tuple[str, str], ...],
    denominator_dataset_name: str,
    denominator_column: str,
    output_fields: list[str],
) -> tuple[dict[str, Any], str]:
    path = root / "data" / "silver" / "sih" / "silver_sih_individual" / f"YEAR={year}" / f"silver_sih_individual_{year}.parquet"
    lookup = _load_population_lookup(root, denominator_dataset_name, year, denominator_column)
    if lookup is None:
        _remove_partition_if_exists(_analytical_partition_file(root, dataset_name, year))
        return (
            _blocked_analytical_validation(
                dataset_name=dataset_name,
                year=year,
                expected_age_scopes=[scope for scope, _ in age_scopes],
                denominator_column=denominator_column,
                blocking_issue=f"required denominator partition missing: {denominator_dataset_name} {year}",
            ),
            "",
        )
    valid_municipality_codes = set(lookup["MUNICIPALITY_CODE"].astype(str))
    lookup = lookup.rename(columns={"YEAR": "COMPETENCE_YEAR"})
    required_columns = [
        "MUNICIPALITY_CODE",
        "DATE",
        "YEAR",
        "MONTH",
        "COMPETENCE_YEAR",
        "COMPETENCE_MONTH",
        *[flag for _, flag in age_scopes],
        *SIH_MATERNAL_FLAG_COLUMNS,
    ]
    daily_parts: list[pd.DataFrame] = []
    summary: dict[str, dict[str, int]] = {age_scope: {} for age_scope, _ in age_scopes}
    excluded_nonmunicipal_rows = 0
    for batch in _iter_parquet_batches(path, required_columns):
        batch = batch.copy()
        batch["MUNICIPALITY_CODE"] = batch["MUNICIPALITY_CODE"].astype(str)
        valid_mask = batch["MUNICIPALITY_CODE"].isin(valid_municipality_codes)
        excluded_nonmunicipal_rows += int((~valid_mask).sum())
        batch = batch.loc[valid_mask].copy()
        if batch.empty:
            continue
        daily_batch, batch_summary = _aggregate_daily_from_flags(
            batch,
            age_scopes,
            key_columns=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", "COMPETENCE_YEAR", "COMPETENCE_MONTH"],
            count_column="ADMISSIONS_TOTAL",
            flag_columns=SIH_MATERNAL_FLAG_COLUMNS,
        )
        if not daily_batch.empty:
            daily_parts.append(daily_batch)
            daily_parts = _compact_daily_parts(
                daily_parts,
                ["AGE_SCOPE", "MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", "COMPETENCE_YEAR", "COMPETENCE_MONTH"],
            )
        for age_scope, values in batch_summary.items():
            summary.setdefault(age_scope, {})
            for key, value in values.items():
                summary[age_scope][key] = summary[age_scope].get(key, 0) + value
    if daily_parts:
        daily = pd.concat(daily_parts, ignore_index=True).groupby(
            ["AGE_SCOPE", "MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", "COMPETENCE_YEAR", "COMPETENCE_MONTH"],
            as_index=False,
            sort=True,
        ).sum(numeric_only=True)
    else:
        daily = pd.DataFrame(columns=output_fields)
    daily = daily.merge(lookup, on=["MUNICIPALITY_CODE", "COMPETENCE_YEAR"], how="left")
    daily["DENOMINATOR_NAME"] = denominator_column
    daily["DENOMINATOR_VALUE"] = daily[denominator_column]
    daily["SOURCE_ANALYTICAL_VIEW"] = dataset_name
    daily["SOURCE_SIH"] = "silver_sih_individual"
    daily["QC_STATUS"] = "PASS"
    daily = daily[output_fields].sort_values(
        ["AGE_SCOPE", "DATE", "MUNICIPALITY_CODE", "COMPETENCE_YEAR", "COMPETENCE_MONTH"],
        kind="mergesort",
    ).reset_index(drop=True)
    validation = _validate_analytical_daily_view(
        dataset_name=dataset_name,
        year=year,
        daily_frame=daily,
        expected_age_scopes=[scope for scope, _ in age_scopes],
        summary_counts=summary,
        count_column="ADMISSIONS_TOTAL",
        denominator_column=denominator_column,
        denominator_required_age_scope=None,
        expected_fields=output_fields,
        excluded_nonmunicipal_rows=excluded_nonmunicipal_rows,
    )
    final_path = _analytical_partition_file(root, dataset_name, year)
    if not validation["all_checks_passed"]:
        _remove_partition_if_exists(final_path)
        return validation, ""
    runtime_path = runtime_root / final_path.relative_to(root)
    writer = ParquetChunkWriter(runtime_path)
    writer.write(daily)
    row_count, columns = writer.close()
    sync_verified_parquet(runtime_path, final_path, row_count, columns)
    runtime_path.unlink(missing_ok=True)
    if runtime_path.parent.exists():
        shutil.rmtree(runtime_path.parent)
    return validation, str(final_path.relative_to(root))


def _validate_analytical_daily_view(
    dataset_name: str,
    year: int,
    daily_frame: pd.DataFrame,
    expected_age_scopes: list[str],
    summary_counts: dict[str, dict[str, int]],
    count_column: str,
    denominator_column: str,
    denominator_required_age_scope: str | None,
    expected_fields: list[str],
    excluded_nonmunicipal_rows: int,
) -> dict[str, Any]:
    expected_total_columns = sorted({key for values in summary_counts.values() for key in values.keys()})
    reconciliation: dict[str, bool] = {}
    for age_scope, totals in summary_counts.items():
        scope_frame = daily_frame.loc[daily_frame["AGE_SCOPE"].eq(age_scope)]
        for total_column in expected_total_columns:
            expected_value = int(totals.get(total_column, 0))
            observed_value = int(scope_frame[total_column].fillna(0).sum()) if total_column in scope_frame.columns else 0
            reconciliation[f"{age_scope}:{total_column}"] = observed_value == expected_value
    age_scope_presence = {scope: bool(daily_frame["AGE_SCOPE"].eq(scope).any()) for scope in expected_age_scopes}
    schema_contract_ok = list(daily_frame.columns) == list(expected_fields)
    if daily_frame.empty:
        municipality_key_format_ok = True
        date_year_coverage_ok = False
    else:
        municipality_key_format_ok = bool(daily_frame["MUNICIPALITY_CODE"].astype(str).str.fullmatch(r"\d{6}").all())
        parsed_dates = pd.to_datetime(daily_frame["DATE"], errors="coerce")
        if "COMPETENCE_YEAR" in daily_frame.columns:
            date_year_coverage_ok = bool(
                parsed_dates.notna().all()
                and pd.to_numeric(daily_frame["COMPETENCE_YEAR"], errors="coerce").notna().all()
                and pd.to_numeric(daily_frame["COMPETENCE_YEAR"], errors="coerce").astype("int64").eq(year).all()
            )
        else:
            date_year_coverage_ok = bool(parsed_dates.notna().all() and parsed_dates.dt.year.eq(year).all())
    duplicates = int(
        daily_frame.duplicated(
            [column for column in daily_frame.columns if column in {"AGE_SCOPE", "MUNICIPALITY_CODE", "DATE", "COMPETENCE_YEAR", "COMPETENCE_MONTH"}]
        ).sum()
    )
    if denominator_required_age_scope is None:
        denominator_missing = int(daily_frame[denominator_column].isna().sum())
    else:
        denominator_missing = int(
            daily_frame.loc[daily_frame["AGE_SCOPE"].eq(denominator_required_age_scope), denominator_column].isna().sum()
        )
    critical_missing = int(daily_frame[["AGE_SCOPE", "MUNICIPALITY_CODE", "DATE", "YEAR"]].isna().sum().sum())
    return {
        "dataset": dataset_name,
        "year": year,
        "row_count": int(len(daily_frame)),
        "age_scope_presence": age_scope_presence,
        "age_coverage_ok": all(age_scope_presence.values()),
        "schema_contract_ok": schema_contract_ok,
        "municipality_key_ok": duplicates == 0 and municipality_key_format_ok,
        "date_year_coverage_ok": date_year_coverage_ok,
        "duplicates_zero": duplicates == 0,
        "duplicate_count": duplicates,
        "critical_missing_zero": critical_missing == 0,
        "critical_missing_count": critical_missing,
        "denominator_column": denominator_column,
        "denominator_missing_count": denominator_missing,
        "excluded_nonmunicipal_rows": excluded_nonmunicipal_rows,
        "internal_reconciliation_ok": all(reconciliation.values()),
        "internal_reconciliation": reconciliation,
        "blocking_issue": "" if denominator_missing == 0 else f"{denominator_column} missing on analytical rows",
        "all_checks_passed": (
            schema_contract_ok
            and duplicates == 0
            and municipality_key_format_ok
            and date_year_coverage_ok
            and critical_missing == 0
            and all(age_scope_presence.values())
            and all(reconciliation.values())
            and denominator_missing == 0
        ),
    }


def _existing_silver_validation_rows(root: Path, start_year: int, end_year: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in EXISTING_SILVER_VALIDATION_DATASETS:
        validation_dir = root / "artifacts" / "validation_reports" / "silver" / dataset
        manifest_path = root / "artifacts" / "manifests" / f"silver_{dataset}.manifest.json"
        for year in range(start_year, end_year + 1):
            report_path = validation_dir / f"{year}.json"
            report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
            if dataset == "climate":
                dataset_pass = bool(
                    report.get("zero_missing_ok")
                    and report.get("dense_panel_ok")
                    and report.get("calendar_coverage_ok")
                    and report.get("municipality_coverage_ok")
                    and report.get("partition_integrity_ok")
                    and report.get("key_uniqueness_ok")
                    and report.get("provenance_ok")
                    and report.get("row_level_provenance_ok")
                )
            else:
                dataset_pass = bool(report.get("all_checks_passed"))
            row_count = None
            for key in (
                "individual_row_count",
                "event_live_births_total",
                "individual_deaths_total",
                "individual_admissions_total",
                "observed_rows",
                "daily_row_count",
            ):
                if report.get(key) is not None:
                    row_count = report.get(key)
                    break
            rows.append(
                {
                    "dataset": f"silver_{dataset}",
                    "year": year,
                    "status": "PASS" if dataset_pass else "FAIL",
                    "manifest_ok": manifest_path.exists(),
                    "schema_contract_ok": report.get("dictionary_expansion_ok", report.get("codebook_adherence_ok", dataset_pass)),
                    "municipality_key_ok": report.get("municipality_coverage_ok", dataset_pass),
                    "date_coverage_ok": report.get("date_year_coverage_ok", report.get("event_year_coverage_ok", report.get("calendar_coverage_ok", dataset_pass))),
                    "age_coverage_ok": report.get("age_flag_coverage") is not None if dataset in {"sim", "sih"} else True,
                    "missing_critical_ok": report.get("zero_missing_ok", dataset_pass),
                    "internal_reconciliation_ok": report.get("individual_vs_daily_reconciliation_ok", dataset_pass),
                    "external_validation_ok": report.get("annual_total_matches_official", report.get("annual_raw_total_matches_official", dataset_pass)),
                    "row_count": row_count,
                    "blocking_issue": "" if dataset_pass else "existing silver validation report failed or missing",
                }
            )
    return rows


def _build_population_tables(
    root: Path,
    runtime_root: Path,
    start_year: int,
    end_year: int,
    log_path: Path,
    command: str,
) -> dict[str, dict[int, dict[str, Any]]]:
    results: dict[str, dict[int, dict[str, Any]]] = {spec.dataset_name: {} for spec in POPULATION_SPECS}
    for spec in POPULATION_SPECS:
        for year in range(start_year, end_year + 1):
            if spec.source_kind == "bronze_pop_under5":
                frame = _load_under5_bronze_frame(root, year)
            else:
                frame = _load_wra_frame_from_popsvs(root, year, spec.dataset_name, spec.value_column, spec.age_min or 0, spec.age_max or 0)
            validation = _validate_population_frame(root, year, frame, spec.value_column, _population_output_fields(spec.dataset_name))
            payload = {
                "dataset": spec.dataset_name,
                "year": year,
                "status": "PASS" if validation["all_checks_passed"] else "FAIL",
                "validation": validation,
                "row_count": int(len(frame)),
                "partition_path": None,
                "value_column": spec.value_column,
                "blocking_issue": "",
            }
            if validation["all_checks_passed"]:
                payload["partition_path"] = _write_population_partition(root, runtime_root, spec.dataset_name, year, frame)
            else:
                _remove_partition_if_exists(_population_partition_file(root, spec.dataset_name, year))
                payload["blocking_issue"] = (
                    f"missing municipality-years={validation['missing_code_count']} "
                    f"unexpected_codes={validation['unexpected_code_count']} "
                    f"duplicate_ibge7={validation['duplicate_ibge7_count']}"
                )
            append_run_event(
                log_path,
                phase="phase_2_silver_promotion",
                dataset=spec.dataset_name,
                year=year,
                command=command,
                decision=(
                    f"population denominator {spec.dataset_name} {'written' if payload['partition_path'] else 'blocked'}; "
                    f"observed={validation['observed_municipality_years']} expected={validation['expected_municipality_years']}"
                ),
                status=payload["status"],
            )
            results[spec.dataset_name][year] = payload
    build_manifest(root, "silver", POPULATION_DATASET)
    return results


def _build_analytical_views(
    root: Path,
    runtime_root: Path,
    start_year: int,
    end_year: int,
    log_path: Path,
    command: str,
    population_results: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, dict[int, dict[str, Any]]]:
    builders = [
        ("silver_child_u5_sim_daily", _build_child_sim_daily_view, UNDER5_DATASET_NAME, "POP_UNDER5", [scope for scope, _ in CHILD_AGE_SCOPES]),
        ("silver_child_u5_sih_daily", _build_child_sih_daily_view, UNDER5_DATASET_NAME, "POP_UNDER5", [scope for scope, _ in CHILD_AGE_SCOPES]),
        (
            "silver_maternal_wra_15_49_sih_daily",
            lambda root_arg, runtime_arg, year_arg: _build_maternal_sih_daily_view(
                root_arg,
                runtime_arg,
                year_arg,
                dataset_name="silver_maternal_wra_15_49_sih_daily",
                age_scopes=MATERNAL_15_49_SCOPE,
                denominator_dataset_name=WRA_15_49_DATASET_NAME,
                denominator_column="POP_WRA_15_49",
                output_fields=SIH_MATERNAL_15_49_OUTPUT_FIELDS,
            ),
            WRA_15_49_DATASET_NAME,
            "POP_WRA_15_49",
            [scope for scope, _ in MATERNAL_15_49_SCOPE],
        ),
        (
            "silver_maternal_wra_15_45_sensitivity_sih_daily",
            lambda root_arg, runtime_arg, year_arg: _build_maternal_sih_daily_view(
                root_arg,
                runtime_arg,
                year_arg,
                dataset_name="silver_maternal_wra_15_45_sensitivity_sih_daily",
                age_scopes=MATERNAL_15_45_SCOPE,
                denominator_dataset_name=WRA_15_45_DATASET_NAME,
                denominator_column="POP_WRA_15_45_SENSITIVITY",
                output_fields=SIH_MATERNAL_15_45_OUTPUT_FIELDS,
            ),
            WRA_15_45_DATASET_NAME,
            "POP_WRA_15_45_SENSITIVITY",
            [scope for scope, _ in MATERNAL_15_45_SCOPE],
        ),
    ]
    results: dict[str, dict[int, dict[str, Any]]] = {name: {} for name, *_ in builders}
    for dataset_name, builder, prerequisite_dataset, denominator_column, expected_age_scopes in builders:
        for year in range(start_year, end_year + 1):
            prerequisite_payload = population_results[prerequisite_dataset][year]
            if prerequisite_payload["status"] != "PASS":
                _remove_partition_if_exists(_analytical_partition_file(root, dataset_name, year))
                validation = _blocked_analytical_validation(
                    dataset_name=dataset_name,
                    year=year,
                    expected_age_scopes=expected_age_scopes,
                    denominator_column=denominator_column,
                    blocking_issue=f"required denominator dataset blocked: {prerequisite_dataset} {year}",
                )
                partition_path = ""
            else:
                validation, partition_path = builder(root, runtime_root, year)
            status = "PASS" if validation["all_checks_passed"] else "FAIL"
            append_run_event(
                log_path,
                phase="phase_2_silver_promotion",
                dataset=dataset_name,
                year=year,
                command=command,
                decision=(
                    f"analytical silver view {dataset_name} {'written' if partition_path else 'blocked'}; "
                    f"row_count={validation['row_count']} denominator_missing={validation['denominator_missing_count']}"
                ),
                status=status,
            )
            results[dataset_name][year] = {
                "dataset": dataset_name,
                "year": year,
                "status": status,
                "validation": validation,
                "partition_path": partition_path or None,
                "blocking_issue": validation.get("blocking_issue", ""),
            }
    build_manifest(root, "silver", ANALYTICAL_DATASET)
    return results


def _build_release_matrix_rows(
    population_results: dict[str, dict[int, dict[str, Any]]],
    analytical_results: dict[str, dict[int, dict[str, Any]]],
    existing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = existing_rows.copy()
    for dataset_name, yearly in population_results.items():
        for year, payload in yearly.items():
            validation = payload["validation"]
            rows.append(
                {
                    "dataset": dataset_name,
                    "year": year,
                    "status": payload["status"],
                    "manifest_ok": bool(payload["partition_path"]),
                    "schema_contract_ok": validation["schema_contract_ok"],
                    "municipality_key_ok": validation["municipality_key_ok"],
                    "date_coverage_ok": validation["year_coverage_ok"],
                    "age_coverage_ok": True,
                    "missing_critical_ok": validation["critical_missing_zero"],
                    "internal_reconciliation_ok": validation["duplicates_zero"],
                    "external_validation_ok": validation["missing_municipality_years_zero"] and validation["unexpected_code_count"] == 0,
                    "row_count": payload["row_count"],
                    "blocking_issue": "" if payload["status"] == "PASS" else payload["blocking_issue"],
                }
            )
    for dataset_name, yearly in analytical_results.items():
        for year, payload in yearly.items():
            validation = payload["validation"]
            rows.append(
                {
                    "dataset": dataset_name,
                    "year": year,
                    "status": payload["status"],
                    "manifest_ok": bool(payload["partition_path"]),
                    "schema_contract_ok": validation["schema_contract_ok"],
                    "municipality_key_ok": validation["municipality_key_ok"],
                    "date_coverage_ok": validation["date_year_coverage_ok"],
                    "age_coverage_ok": validation["age_coverage_ok"],
                    "missing_critical_ok": validation["critical_missing_zero"] and validation["denominator_missing_count"] == 0,
                    "internal_reconciliation_ok": validation["internal_reconciliation_ok"],
                    "external_validation_ok": validation["denominator_missing_count"] == 0,
                    "row_count": validation["row_count"],
                    "blocking_issue": "" if payload["status"] == "PASS" else payload["blocking_issue"],
                }
            )
    return rows


def _load_manifest_entry_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for dataset in [*EXISTING_SILVER_DATASETS, POPULATION_DATASET, ANALYTICAL_DATASET]:
        manifest_path = root / "artifacts" / "manifests" / f"silver_{dataset}.manifest.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            counts[f"silver_{dataset}"] = int(payload.get("entry_count", 0))
    return counts


def _summarize_flag_from_individual(
    path: Path,
    age_flag_column: str,
    outcome_flag_column: str,
    valid_municipality_codes: set[str],
) -> tuple[int | None, int | None]:
    if not path.exists():
        return None, None
    municipalities_with_event: set[str] = set()
    event_total = 0
    for batch in _iter_parquet_batches(path, ["MUNICIPALITY_CODE", age_flag_column, outcome_flag_column]):
        batch = batch.copy()
        batch["MUNICIPALITY_CODE"] = batch["MUNICIPALITY_CODE"].astype(str)
        batch = batch.loc[batch["MUNICIPALITY_CODE"].isin(valid_municipality_codes) & batch[age_flag_column].eq(1)].copy()
        if batch.empty:
            continue
        outcome = pd.to_numeric(batch[outcome_flag_column], errors="coerce").fillna(0)
        event_total += int(outcome.sum())
        municipalities_with_event.update(batch.loc[outcome.gt(0), "MUNICIPALITY_CODE"].astype(str).tolist())
    return int(len(municipalities_with_event)), int(event_total)


def _annual_outcome_summary(
    root: Path,
    start_year: int,
    end_year: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        valid_municipality_codes = _official_municipality_codes6(root, year)
        sinasc_daily = pd.read_parquet(
            root / "data" / "silver" / "sinasc" / "silver_sinasc_daily" / f"YEAR={year}" / f"silver_sinasc_daily_{year}.parquet"
        )
        stillbirth_daily = pd.read_parquet(
            root / "data" / "silver" / "stillbirth" / "silver_stillbirth_daily" / f"YEAR={year}" / f"silver_stillbirth_daily_{year}.parquet"
        )
        child_sim_daily = _read_parquet_or_none(_analytical_partition_file(root, "silver_child_u5_sim_daily", year))
        child_sih_daily = _read_parquet_or_none(_analytical_partition_file(root, "silver_child_u5_sih_daily", year))
        maternal_15_49_daily = _read_parquet_or_none(_analytical_partition_file(root, "silver_maternal_wra_15_49_sih_daily", year))
        if child_sim_daily is None:
            child_sim_municipality_count, child_sim_event_total = _summarize_flag_from_individual(
                root / "data" / "silver" / "sim" / "silver_sim_individual" / f"YEAR={year}" / f"silver_sim_individual_{year}.parquet",
                "AGE_GROUP_U5",
                "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_FLAG",
                valid_municipality_codes,
            )
        else:
            child_sim_municipality_count = int(
                child_sim_daily.loc[
                    (child_sim_daily["AGE_SCOPE"].eq("AGE_U5"))
                    & (child_sim_daily["UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL"].gt(0)),
                    "MUNICIPALITY_CODE",
                ].nunique()
            )
            child_sim_event_total = int(
                child_sim_daily.loc[
                    child_sim_daily["AGE_SCOPE"].eq("AGE_U5"),
                    "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL",
                ].sum()
            )
        if child_sih_daily is None:
            child_sih_municipality_count, child_sih_event_total = _summarize_flag_from_individual(
                root / "data" / "silver" / "sih" / "silver_sih_individual" / f"YEAR={year}" / f"silver_sih_individual_{year}.parquet",
                "AGE_GROUP_U5",
                "PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG",
                valid_municipality_codes,
            )
        else:
            child_sih_municipality_count = int(
                child_sih_daily.loc[
                    (child_sih_daily["AGE_SCOPE"].eq("AGE_U5"))
                    & (child_sih_daily["PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_TOTAL"].gt(0)),
                    "MUNICIPALITY_CODE",
                ].nunique()
            )
            child_sih_event_total = int(
                child_sih_daily.loc[
                    child_sih_daily["AGE_SCOPE"].eq("AGE_U5"),
                    "PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_TOTAL",
                ].sum()
            )
        if maternal_15_49_daily is None:
            maternal_15_49_municipality_count, maternal_15_49_event_total = _summarize_flag_from_individual(
                root / "data" / "silver" / "sih" / "silver_sih_individual" / f"YEAR={year}" / f"silver_sih_individual_{year}.parquet",
                "AGE_GROUP_WRA_15_49",
                "PRINCIPAL_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_FLAG",
                valid_municipality_codes,
            )
        else:
            maternal_15_49_municipality_count = int(
                maternal_15_49_daily.loc[
                    maternal_15_49_daily["PRINCIPAL_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_TOTAL"].gt(0),
                    "MUNICIPALITY_CODE",
                ].nunique()
            )
            maternal_15_49_event_total = int(
                maternal_15_49_daily["PRINCIPAL_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_TOTAL"].sum()
            )
        pop_under5_lookup = _load_population_lookup(root, UNDER5_DATASET_NAME, year, "POP_UNDER5")
        pop_wra_lookup = _load_population_lookup(root, WRA_15_49_DATASET_NAME, year, "POP_WRA_15_49")
        pop_under5_total = int(pop_under5_lookup["POP_UNDER5"].sum()) if pop_under5_lookup is not None else None
        pop_wra_total = int(pop_wra_lookup["POP_WRA_15_49"].sum()) if pop_wra_lookup is not None else None

        summary_specs = [
            {
                "dataset": "silver_sinasc_daily",
                "outcome": "NV_PRETERM_TOTAL",
                "municipality_count": int(sinasc_daily.loc[sinasc_daily["NV_PRETERM_TOTAL"].gt(0), "MUNICIPALITY_CODE"].nunique()),
                "event_total": int(sinasc_daily["NV_PRETERM_TOTAL"].sum()),
                "denominator_name": "LIVE_BIRTHS_TOTAL",
                "denominator_total": int(sinasc_daily["LIVE_BIRTHS_TOTAL"].sum()),
            },
            {
                "dataset": "silver_stillbirth_daily",
                "outcome": "STILLBIRTH_TOTAL",
                "municipality_count": int(stillbirth_daily.loc[stillbirth_daily["STILLBIRTH_TOTAL"].gt(0), "MUNICIPALITY_CODE"].nunique()),
                "event_total": int(stillbirth_daily["STILLBIRTH_TOTAL"].sum()),
                "denominator_name": "NASCIMENTOS_TOTAIS_INCLUINDO_FETAIS",
                "denominator_total": int(stillbirth_daily["NASCIMENTOS_TOTAIS_INCLUINDO_FETAIS"].sum()),
            },
            {
                "dataset": "silver_child_u5_sim_daily",
                "outcome": "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL",
                "municipality_count": child_sim_municipality_count,
                "event_total": child_sim_event_total,
                "denominator_name": "POP_UNDER5",
                "denominator_total": pop_under5_total,
            },
            {
                "dataset": "silver_child_u5_sih_daily",
                "outcome": "PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_TOTAL",
                "municipality_count": child_sih_municipality_count,
                "event_total": child_sih_event_total,
                "denominator_name": "POP_UNDER5",
                "denominator_total": pop_under5_total,
            },
            {
                "dataset": "silver_maternal_wra_15_49_sih_daily",
                "outcome": "PRINCIPAL_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_TOTAL",
                "municipality_count": maternal_15_49_municipality_count,
                "event_total": maternal_15_49_event_total,
                "denominator_name": "POP_WRA_15_49",
                "denominator_total": pop_wra_total,
            },
        ]
        for payload in summary_specs:
            denominator_total = payload["denominator_total"]
            payload["year"] = year
            payload["crude_rate_per_1000"] = (
                float(round(1000.0 * payload["event_total"] / denominator_total, 6))
                if payload["event_total"] is not None and denominator_total not in (None, 0)
                else None
            )
            rows.append(payload)
    return rows


def _write_release_outputs(
    root: Path,
    start_year: int,
    end_year: int,
    matrix_rows: list[dict[str, Any]],
    population_results: dict[str, dict[int, dict[str, Any]]],
    analytical_results: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    matrix_path = _build_release_matrix_path(root)
    report_path = _build_release_report_path(root)
    manifest_path = _build_release_manifest_path(root)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    matrix_frame = pd.DataFrame(matrix_rows).sort_values(["dataset", "year"], kind="mergesort").reset_index(drop=True)
    matrix_frame.to_csv(matrix_path, index=False, quoting=csv.QUOTE_MINIMAL)
    annual_summary_rows = _annual_outcome_summary(root, start_year, end_year)
    manifest_entry_counts = _load_manifest_entry_counts(root)

    denominator_summary_lines = []
    for dataset_name, yearly in population_results.items():
        pass_years = [year for year, payload in yearly.items() if payload["status"] == "PASS"]
        fail_years = [year for year, payload in yearly.items() if payload["status"] != "PASS"]
        municipality_years = sum(payload["validation"]["observed_municipality_years"] for year, payload in yearly.items() if payload["status"] == "PASS")
        municipalities_covered = sorted({payload["validation"]["municipalities_covered"] for payload in yearly.values() if payload["status"] == "PASS"})
        duplicates_zero = all(payload["validation"]["duplicates_zero"] for payload in yearly.values() if payload["status"] == "PASS")
        missing_zero = all(payload["validation"]["missing_municipality_years_zero"] for payload in yearly.values() if payload["status"] == "PASS")
        denominator_summary_lines.extend(
            [
                f"- {dataset_name}: municipality-years={municipality_years}, municipalities={municipalities_covered}, years={pass_years}, duplicates=0? `{duplicates_zero}`, missing municipality-years=0? `{missing_zero}`",
                f"  blocked years: {fail_years}" if fail_years else "",
            ]
        )
    denominator_summary_lines = [line for line in denominator_summary_lines if line]

    annual_summary_frame = pd.DataFrame(annual_summary_rows)
    annual_summary_md = []
    for year in range(start_year, end_year + 1):
        annual_summary_md.extend([f"## {year}", ""])
        year_frame = annual_summary_frame.loc[annual_summary_frame["year"].eq(year)].copy()
        for row in year_frame.to_dict(orient="records"):
            annual_summary_md.append(
                f"- {row['dataset']} / {row['outcome']}: municipalities with at least one event={row['municipality_count']}, "
                f"total events={row['event_total']}, denominator used={row['denominator_name']} ({row['denominator_total']}), "
                f"crude rate={row['crude_rate_per_1000']}"
            )
        annual_summary_md.append("")

    blocking_rows = matrix_frame.loc[matrix_frame["status"].ne("PASS")]
    overall_status = "PASS" if blocking_rows.empty else "FAIL"
    report_lines = [
        "# Silver Release Candidate Report",
        "",
        f"- Generated at: {utc_now()}",
        f"- Overall status: `{overall_status}`",
        f"- Manifest counts: {manifest_entry_counts}",
        f"- Matrix: `{matrix_path.relative_to(root)}`",
        "",
        "## Denominator Tables",
        "",
        *denominator_summary_lines,
        "",
        "## Blocking Issues",
        "",
    ]
    if blocking_rows.empty:
        report_lines.append("- None")
    else:
        for row in blocking_rows.to_dict(orient="records"):
            report_lines.append(f"- {row['dataset']} {row['year']}: {row['blocking_issue']}")
    report_lines.extend(
        [
            "",
            "## Year-by-Year Outcome Summary",
            "",
            *annual_summary_md,
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    manifest_payload = {
        "generated_at": utc_now(),
        "status": overall_status,
        "years": list(range(start_year, end_year + 1)),
        "matrix_path": str(matrix_path.relative_to(root)),
        "report_path": str(report_path.relative_to(root)),
        "included_manifests": {
            name: f"artifacts/manifests/{name}.manifest.json"
            for name in ["silver_climate", "silver_sinasc", "silver_sim", "silver_sih", "silver_stillbirth", "silver_population", "silver_analytical"]
            if (root / "artifacts" / "manifests" / f"{name}.manifest.json").exists()
        },
        "matrix_row_count": int(len(matrix_frame)),
        "blocking_issue_count": int(len(blocking_rows)),
        "artifacts": [
            {
                "path": str(matrix_path.relative_to(root)),
                "row_count": int(len(matrix_frame)),
                "sha256": sha256_file(matrix_path),
                "size_bytes": matrix_path.stat().st_size,
            },
            {
                "path": str(report_path.relative_to(root)),
                "sha256": sha256_file(report_path),
                "size_bytes": report_path.stat().st_size,
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": overall_status,
        "matrix_path": str(matrix_path.relative_to(root)),
        "report_path": str(report_path.relative_to(root)),
        "manifest_path": str(manifest_path.relative_to(root)),
        "blocking_issue_count": int(len(blocking_rows)),
    }


def build_silver_release_candidate(
    root: Path,
    runtime_root: Path,
    start_year: int,
    end_year: int,
    log_path: Path,
    command: str,
    force: bool = False,
) -> dict[str, Any]:
    del force
    population_results = _build_population_tables(root, runtime_root, start_year, end_year, log_path, command)
    analytical_results = _build_analytical_views(root, runtime_root, start_year, end_year, log_path, command, population_results)
    existing_rows = _existing_silver_validation_rows(root, start_year, end_year)
    matrix_rows = _build_release_matrix_rows(population_results, analytical_results, existing_rows)
    release_outputs = _write_release_outputs(root, start_year, end_year, matrix_rows, population_results, analytical_results)
    append_run_event(
        log_path,
        phase="phase_3_silver_release_candidate",
        dataset="silver_release_candidate",
        year=None,
        command=command,
        decision=(
            f"silver release candidate built; status={release_outputs['status']} "
            f"matrix={release_outputs['matrix_path']} report={release_outputs['report_path']}"
        ),
        status=release_outputs["status"],
    )
    return {
        "dataset": "silver_release_candidate",
        "years": list(range(start_year, end_year + 1)),
        "population_results": population_results,
        "analytical_results": analytical_results,
        "release_outputs": release_outputs,
    }
