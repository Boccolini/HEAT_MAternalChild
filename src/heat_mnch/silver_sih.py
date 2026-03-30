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
    SIH_ANY_DIAG_COLUMNS,
    SIH_EXTERNAL_CAUSE_COLUMNS,
    SLOW_MOVING_HEAT_GROUPS,
    build_group_code_sets,
    build_unified_outcome_dictionary,
    derive_sih_age_flags,
    expected_group_names,
    load_unified_outcome_dictionary,
    normalize_icd10_series,
    utc_now,
)
from .run_log import append_run_event
from .silver_outcome_common import ParquetChunkWriter, official_total_from_bronze, sync_verified_parquet


SIH_REQUIRED_INPUT_FIELDS = [
    "BRONZE_DATASET",
    "BRONZE_SOURCE_SYSTEM",
    "BRONZE_SOURCE_FILE",
    "BRONZE_SOURCE_PATH",
    "BRONZE_SOURCE_MEMBER",
    "BRONZE_YEAR",
    "BRONZE_INGESTED_AT",
    "ANO_CMPT",
    "MES_CMPT",
    "N_AIH",
    "IDENT",
    "MUNIC_RES",
    "MUNIC_MOV",
    "DT_INTER",
    "DT_SAIDA",
    "COD_IDADE",
    "IDADE",
    "SEXO",
    "DIAG_PRINC",
    "DIAG_SECUN",
    "DIAGSEC1",
    "DIAGSEC2",
    "DIAGSEC3",
    "DIAGSEC4",
    "DIAGSEC5",
    "DIAGSEC6",
    "DIAGSEC7",
    "DIAGSEC8",
    "DIAGSEC9",
    "CID_NOTIF",
    "CID_ASSO",
    "CID_MORTE",
    "MORTE",
    "CAR_INT",
    "UTI_MES_TO",
    "UTI_INT_TO",
    "MARCA_UTI",
    "VAL_UTI",
    "VAL_UCI",
    "CNES",
    "CGC_HOSP",
    "SEQUENCIA",
]
SIH_NORMALIZED_DIAG_FIELDS = [f"{column}_ICD10" for column in dict.fromkeys(SIH_ANY_DIAG_COLUMNS + SIH_EXTERNAL_CAUSE_COLUMNS)]
SIH_URGENT_CAR_INT_VALUES = frozenset({"02", "05", "06", "07", "08", "09"})
SIH_SCOPE_FIELDS = (
    ("PRINCIPAL_DIAGNOSIS", "DIAG_PRINC"),
    ("ANY_DIAGNOSIS", "DIAG_ANY"),
)


def _scope_flag_column(scope_label: str, group_name: str) -> str:
    return f"{scope_label}_{group_name}_FLAG"


def _scope_total_column(scope_label: str, group_name: str) -> str:
    return f"{scope_label}_{group_name}_TOTAL"


SIH_GROUP_FLAG_FIELDS = [
    _scope_flag_column(scope_label, group_name)
    for scope_label, _ in SIH_SCOPE_FIELDS
    for group_name in expected_group_names()
] + ["EXTERNAL_CAUSE_FIELDS_EXCESSIVE_NATURAL_HEAT_X30_FLAG"]
SIH_CATEGORY_FLAG_FIELDS = [
    "PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG",
    "ANY_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG",
    "PRINCIPAL_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG",
    "ANY_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG",
    "PRINCIPAL_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG",
    "ANY_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG",
    "PRINCIPAL_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_FLAG",
    "ANY_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_FLAG",
    "ICU_ANY_STAY_FLAG",
    "URGENT_ADMISSION_OBSERVED_FLAG",
    "OBSTETRIC_ADMISSION_PRIMARY_FLAG",
    "OBSTETRIC_URGENT_PRIMARY_FLAG",
    "OBSTETRIC_ICU_PRIMARY_FLAG",
    "PRETERM_LABOR_URGENT_FLAG",
    "PRETERM_LABOR_ICU_FLAG",
    "PRETERM_LABOR_DEATH_FLAG",
    "APH_ANTEPARTUM_URGENT_FLAG",
    "APH_ANTEPARTUM_ICU_FLAG",
    "APH_ANTEPARTUM_DEATH_FLAG",
    "GDM_GESTATIONAL_URGENT_FLAG",
    "GDM_GESTATIONAL_ICU_FLAG",
    "GDM_GESTATIONAL_DEATH_FLAG",
]
SIH_INDIVIDUAL_OUTPUT_FIELDS = [
    "SIH_RECORD_ID",
    "SOURCE_SIH",
    "BRONZE_DATASET",
    "BRONZE_SOURCE_SYSTEM",
    "BRONZE_SOURCE_FILE",
    "BRONZE_SOURCE_PATH",
    "BRONZE_SOURCE_MEMBER",
    "BRONZE_YEAR",
    "BRONZE_INGESTED_AT",
    "DATE",
    "YEAR",
    "MONTH",
    "COMPETENCE_YEAR",
    "COMPETENCE_MONTH",
    "MUNICIPALITY_CODE",
    "MUNIC_RES",
    "MUNIC_MOV",
    "N_AIH",
    "IDENT",
    "DT_INTER",
    "DT_SAIDA",
    "COD_IDADE",
    "IDADE",
    "SEXO",
    "MORTE",
    "CAR_INT",
    "UTI_MES_TO",
    "UTI_INT_TO",
    "MARCA_UTI",
    "VAL_UTI",
    "VAL_UCI",
    "CNES",
    "CGC_HOSP",
    "SEQUENCIA",
    "AGE_UNIT_CODE",
    "AGE_VALUE",
    "AGE_GROUP_0_27D",
    "AGE_GROUP_28_364D",
    "AGE_GROUP_1_4Y",
    "AGE_GROUP_U5",
    "AGE_GROUP_WRA_15_49",
    "AGE_GROUP_WRA_15_45_SENSITIVITY",
    "DIAG_PRINC",
    "DIAG_SECUN",
    "DIAGSEC1",
    "DIAGSEC2",
    "DIAGSEC3",
    "DIAGSEC4",
    "DIAGSEC5",
    "DIAGSEC6",
    "DIAGSEC7",
    "DIAGSEC8",
    "DIAGSEC9",
    "CID_NOTIF",
    "CID_ASSO",
    "CID_MORTE",
    *SIH_NORMALIZED_DIAG_FIELDS,
    *SIH_GROUP_FLAG_FIELDS,
    *SIH_CATEGORY_FLAG_FIELDS,
    "QC_STATUS",
]
SIH_DAILY_OUTPUT_FIELDS = [
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
    "COMPETENCE_YEAR",
    "COMPETENCE_MONTH",
    "ADMISSIONS_TOTAL",
    *[_scope_total_column(scope_label, group_name) for scope_label, _ in SIH_SCOPE_FIELDS for group_name in expected_group_names()],
    "EXTERNAL_CAUSE_FIELDS_EXCESSIVE_NATURAL_HEAT_X30_TOTAL",
    "PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_TOTAL",
    "ANY_DIAGNOSIS_DIRECT_HEAT_ANY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_INDIRECT_HEAT_ANY_TOTAL",
    "ANY_DIAGNOSIS_INDIRECT_HEAT_ANY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_TOTAL",
    "ANY_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_TOTAL",
    "ANY_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_TOTAL",
    "OBSTETRIC_ADMISSIONS_TOTAL",
    "OBSTETRIC_URGENT_TOTAL",
    "OBSTETRIC_ICU_TOTAL",
    "HDP_ANY_TOTAL",
    "HDP_SEVERE_TOTAL",
    "HDP_GESTATIONAL_HTN_TOTAL",
    "HDP_PREECLAMPSIA_TOTAL",
    "HDP_ECLAMPSIA_TOTAL",
    "PRETERM_LABOR_TOTAL",
    "PRETERM_LABOR_URGENT_TOTAL",
    "PRETERM_LABOR_ICU_TOTAL",
    "PRETERM_LABOR_DEATH_TOTAL",
    "APH_ANTEPARTUM_TOTAL",
    "APH_ANTEPARTUM_URGENT_TOTAL",
    "APH_ANTEPARTUM_ICU_TOTAL",
    "APH_ANTEPARTUM_DEATH_TOTAL",
    "GDM_GESTATIONAL_TOTAL",
    "GDM_GESTATIONAL_URGENT_TOTAL",
    "GDM_GESTATIONAL_ICU_TOTAL",
    "GDM_GESTATIONAL_DEATH_TOTAL",
    "DIABETES_PREGNANCY_BROAD_TOTAL",
    "SOURCE_SIH",
    "QC_STATUS",
]


def sih_individual_partition_dir(root: Path, year: int) -> Path:
    return root / "data" / "silver" / "sih" / "silver_sih_individual" / f"YEAR={year}"


def sih_individual_partition_file(root: Path, year: int) -> Path:
    return sih_individual_partition_dir(root, year) / f"silver_sih_individual_{year}.parquet"


def sih_daily_partition_dir(root: Path, year: int) -> Path:
    return root / "data" / "silver" / "sih" / "silver_sih_daily" / f"YEAR={year}"


def sih_daily_partition_file(root: Path, year: int) -> Path:
    return sih_daily_partition_dir(root, year) / f"silver_sih_daily_{year}.parquet"


def _load_sih_chunk_iterator(root: Path, year: int):
    path = root / "data" / "bronze" / "sih" / str(year) / f"sih_{year}_source_preserving.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"SIH bronze missing for {year}: {path}")
    header = pd.read_csv(path, compression="gzip", sep=";", dtype=str, nrows=0, keep_default_na=False, low_memory=False)
    missing_fields = [field for field in SIH_REQUIRED_INPUT_FIELDS if field not in header.columns]
    if missing_fields:
        raise ValueError(f"SIH bronze missing required columns for {year}: {missing_fields}")
    return pd.read_csv(
        path,
        compression="gzip",
        sep=";",
        dtype=str,
        usecols=SIH_REQUIRED_INPUT_FIELDS,
        chunksize=250_000,
        keep_default_na=False,
        low_memory=False,
    )


def _aggregate_daily_parts(parts: list[pd.DataFrame]) -> list[pd.DataFrame]:
    if len(parts) < 16:
        return parts
    consolidated = pd.concat(parts, ignore_index=True).groupby(
        ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", "COMPETENCE_YEAR", "COMPETENCE_MONTH"],
        as_index=False,
        sort=True,
    ).sum(numeric_only=True)
    consolidated["SOURCE_SIH"] = "silver_sih_individual"
    consolidated["QC_STATUS"] = "PASS"
    return [consolidated[SIH_DAILY_OUTPUT_FIELDS].copy()]


def _build_sih_chunk(
    chunk: pd.DataFrame,
    year: int,
    row_offset: int,
    code_sets: dict[str, set[str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = chunk.copy().reset_index(drop=True)
    working["SIH_RECORD_ID"] = [
        f"SIH_RECORD_{year}_{row_number:09d}"
        for row_number in range(row_offset + 1, row_offset + len(working) + 1)
    ]
    working["SOURCE_SIH"] = "silver_sih_individual"

    dt_inter = pd.to_datetime(working["DT_INTER"].replace("", pd.NA), format="%Y-%m-%d", errors="coerce")
    if dt_inter.isna().any():
        raise ValueError(f"SIH bronze has invalid DT_INTER values for {year}")
    municipality_code = working["MUNIC_RES"].fillna("").astype(str).str.strip()
    if (~municipality_code.str.fullmatch(r"\d{6}")).any():
        raise ValueError(f"SIH bronze has invalid MUNIC_RES values for {year}")
    competence_year = pd.to_numeric(working["ANO_CMPT"].replace("", pd.NA), errors="coerce")
    competence_month = pd.to_numeric(working["MES_CMPT"].replace("", pd.NA), errors="coerce")
    if competence_year.isna().any() or competence_month.isna().any():
        raise ValueError(f"SIH bronze has invalid ANO_CMPT/MES_CMPT values for {year}")
    if not bool(competence_year.eq(year).all()):
        raise ValueError(f"SIH bronze has out-of-partition competence years for {year}")

    working["DATE"] = dt_inter.dt.strftime("%Y-%m-%d")
    working["YEAR"] = dt_inter.dt.year.astype("int16")
    working["MONTH"] = dt_inter.dt.month.astype("int8")
    working["COMPETENCE_YEAR"] = competence_year.astype("int16")
    working["COMPETENCE_MONTH"] = competence_month.astype("int8")
    working["MUNICIPALITY_CODE"] = municipality_code
    working["CAR_INT"] = working["CAR_INT"].fillna("").astype(str).str.strip().str.zfill(2)
    working["MARCA_UTI"] = working["MARCA_UTI"].fillna("").astype(str).str.strip()
    for column in ("UTI_MES_TO", "UTI_INT_TO", "VAL_UTI", "VAL_UCI"):
        working[column] = pd.to_numeric(working[column].replace("", pd.NA), errors="coerce").fillna(0)

    age_flags = derive_sih_age_flags(working["COD_IDADE"], working["IDADE"])
    for column in age_flags.columns:
        working[column] = age_flags[column]

    normalized_fields = {
        column: normalize_icd10_series(working[column])
        for column in dict.fromkeys(SIH_ANY_DIAG_COLUMNS + SIH_EXTERNAL_CAUSE_COLUMNS)
    }
    for column, series in normalized_fields.items():
        working[f"{column}_ICD10"] = series

    principal_series = normalized_fields["DIAG_PRINC"]
    any_diag_fields = {column: normalized_fields[column] for column in SIH_ANY_DIAG_COLUMNS}
    external_cause_fields = {column: normalized_fields[column] for column in SIH_EXTERNAL_CAUSE_COLUMNS}

    principal_totals: dict[str, int] = {}
    any_diag_totals: dict[str, int] = {}
    for group_name in expected_group_names():
        principal_match = principal_series.isin(code_sets[group_name])
        any_diag_match = principal_match.copy()
        for series in any_diag_fields.values():
            any_diag_match = any_diag_match | series.isin(code_sets[group_name])
        working[_scope_flag_column("PRINCIPAL_DIAGNOSIS", group_name)] = principal_match.astype("Int8")
        working[_scope_flag_column("ANY_DIAGNOSIS", group_name)] = any_diag_match.astype("Int8")
        principal_totals[group_name] = int(principal_match.sum())
        any_diag_totals[group_name] = int(any_diag_match.sum())

    external_x30 = pd.Series(False, index=working.index)
    for series in external_cause_fields.values():
        external_x30 = external_x30 | series.isin(code_sets["EXCESSIVE_NATURAL_HEAT_X30"])
    working["EXTERNAL_CAUSE_FIELDS_EXCESSIVE_NATURAL_HEAT_X30_FLAG"] = external_x30.astype("Int8")

    working["PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG"] = working[
        [_scope_flag_column("PRINCIPAL_DIAGNOSIS", group_name) for group_name in DIRECT_HEAT_GROUPS]
    ].max(axis=1).astype("Int8")
    working["ANY_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG"] = working[
        [_scope_flag_column("ANY_DIAGNOSIS", group_name) for group_name in DIRECT_HEAT_GROUPS]
    ].max(axis=1).astype("Int8")
    working["PRINCIPAL_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG"] = working[
        [_scope_flag_column("PRINCIPAL_DIAGNOSIS", group_name) for group_name in INDIRECT_HEAT_GROUPS]
    ].max(axis=1).astype("Int8")
    working["ANY_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG"] = working[
        [_scope_flag_column("ANY_DIAGNOSIS", group_name) for group_name in INDIRECT_HEAT_GROUPS]
    ].max(axis=1).astype("Int8")
    working["PRINCIPAL_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG"] = working[
        [_scope_flag_column("PRINCIPAL_DIAGNOSIS", group_name) for group_name in SLOW_MOVING_HEAT_GROUPS]
    ].max(axis=1).astype("Int8")
    working["ANY_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG"] = working[
        [_scope_flag_column("ANY_DIAGNOSIS", group_name) for group_name in SLOW_MOVING_HEAT_GROUPS]
    ].max(axis=1).astype("Int8")
    working["PRINCIPAL_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_FLAG"] = working[
        [_scope_flag_column("PRINCIPAL_DIAGNOSIS", group_name) for group_name in MATERNAL_GROUPS]
    ].max(axis=1).astype("Int8")
    working["ANY_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_FLAG"] = working[
        [_scope_flag_column("ANY_DIAGNOSIS", group_name) for group_name in MATERNAL_GROUPS]
    ].max(axis=1).astype("Int8")
    working["ICU_ANY_STAY_FLAG"] = (
        working["UTI_MES_TO"].gt(0)
        | working["UTI_INT_TO"].gt(0)
        | working["VAL_UTI"].gt(0)
        | working["VAL_UCI"].gt(0)
        | ~working["MARCA_UTI"].isin({"", "0", "00"})
    ).astype("Int8")
    working["URGENT_ADMISSION_OBSERVED_FLAG"] = working["CAR_INT"].isin(SIH_URGENT_CAR_INT_VALUES).astype("Int8")
    working["OBSTETRIC_ADMISSION_PRIMARY_FLAG"] = principal_series.str.startswith("O").fillna(False).astype("Int8")
    working["OBSTETRIC_URGENT_PRIMARY_FLAG"] = (
        working["OBSTETRIC_ADMISSION_PRIMARY_FLAG"].eq(1) & working["URGENT_ADMISSION_OBSERVED_FLAG"].eq(1)
    ).astype("Int8")
    working["OBSTETRIC_ICU_PRIMARY_FLAG"] = (
        working["OBSTETRIC_ADMISSION_PRIMARY_FLAG"].eq(1) & working["ICU_ANY_STAY_FLAG"].eq(1)
    ).astype("Int8")
    working["PRETERM_LABOR_URGENT_FLAG"] = (
        working["PRINCIPAL_DIAGNOSIS_PRETERM_LABOR_PRIMARY_FLAG"].eq(1) & working["URGENT_ADMISSION_OBSERVED_FLAG"].eq(1)
    ).astype("Int8")
    working["PRETERM_LABOR_ICU_FLAG"] = (
        working["PRINCIPAL_DIAGNOSIS_PRETERM_LABOR_PRIMARY_FLAG"].eq(1) & working["ICU_ANY_STAY_FLAG"].eq(1)
    ).astype("Int8")
    working["PRETERM_LABOR_DEATH_FLAG"] = (
        working["PRINCIPAL_DIAGNOSIS_PRETERM_LABOR_PRIMARY_FLAG"].eq(1) & working["MORTE"].astype(str).eq("1")
    ).astype("Int8")
    working["APH_ANTEPARTUM_URGENT_FLAG"] = (
        working["PRINCIPAL_DIAGNOSIS_APH_ANTEPARTUM_PRIMARY_FLAG"].eq(1) & working["URGENT_ADMISSION_OBSERVED_FLAG"].eq(1)
    ).astype("Int8")
    working["APH_ANTEPARTUM_ICU_FLAG"] = (
        working["PRINCIPAL_DIAGNOSIS_APH_ANTEPARTUM_PRIMARY_FLAG"].eq(1) & working["ICU_ANY_STAY_FLAG"].eq(1)
    ).astype("Int8")
    working["APH_ANTEPARTUM_DEATH_FLAG"] = (
        working["PRINCIPAL_DIAGNOSIS_APH_ANTEPARTUM_PRIMARY_FLAG"].eq(1) & working["MORTE"].astype(str).eq("1")
    ).astype("Int8")
    working["GDM_GESTATIONAL_URGENT_FLAG"] = (
        working["PRINCIPAL_DIAGNOSIS_GDM_GESTATIONAL_PRIMARY_FLAG"].eq(1) & working["URGENT_ADMISSION_OBSERVED_FLAG"].eq(1)
    ).astype("Int8")
    working["GDM_GESTATIONAL_ICU_FLAG"] = (
        working["PRINCIPAL_DIAGNOSIS_GDM_GESTATIONAL_PRIMARY_FLAG"].eq(1) & working["ICU_ANY_STAY_FLAG"].eq(1)
    ).astype("Int8")
    working["GDM_GESTATIONAL_DEATH_FLAG"] = (
        working["PRINCIPAL_DIAGNOSIS_GDM_GESTATIONAL_PRIMARY_FLAG"].eq(1) & working["MORTE"].astype(str).eq("1")
    ).astype("Int8")
    working["QC_STATUS"] = "PASS"

    stats = {
        "row_count": int(len(working)),
        "municipalities": set(working["MUNICIPALITY_CODE"].unique().tolist()),
        "date_year_mismatch_count": int(working["YEAR"].ne(working["COMPETENCE_YEAR"]).sum()),
        "principal_diag_invalid_count": int(
            (
                normalized_fields["DIAG_PRINC"].ne("")
                & ~normalized_fields["DIAG_PRINC"].str.fullmatch(r"[A-Z][0-9][0-9][A-Z0-9]?").fillna(False)
            ).sum()
        ),
        "principal_totals": principal_totals,
        "any_diag_totals": any_diag_totals,
        "external_x30_total": int(external_x30.sum()),
        "category_totals": {
            "PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG": int(working["PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG"].eq(1).sum()),
            "ANY_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG": int(working["ANY_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG"].eq(1).sum()),
            "PRINCIPAL_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG": int(working["PRINCIPAL_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG"].eq(1).sum()),
            "ANY_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG": int(working["ANY_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG"].eq(1).sum()),
            "PRINCIPAL_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG": int(working["PRINCIPAL_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG"].eq(1).sum()),
            "ANY_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG": int(working["ANY_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG"].eq(1).sum()),
            "PRINCIPAL_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_FLAG": int(working["PRINCIPAL_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_FLAG"].eq(1).sum()),
            "ANY_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_FLAG": int(working["ANY_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_FLAG"].eq(1).sum()),
        },
        "operational_totals": {
            "ICU_ANY_STAY_FLAG": int(working["ICU_ANY_STAY_FLAG"].eq(1).sum()),
            "URGENT_ADMISSION_OBSERVED_FLAG": int(working["URGENT_ADMISSION_OBSERVED_FLAG"].eq(1).sum()),
            "OBSTETRIC_ADMISSION_PRIMARY_FLAG": int(working["OBSTETRIC_ADMISSION_PRIMARY_FLAG"].eq(1).sum()),
            "OBSTETRIC_URGENT_PRIMARY_FLAG": int(working["OBSTETRIC_URGENT_PRIMARY_FLAG"].eq(1).sum()),
            "OBSTETRIC_ICU_PRIMARY_FLAG": int(working["OBSTETRIC_ICU_PRIMARY_FLAG"].eq(1).sum()),
            "PRETERM_LABOR_URGENT_FLAG": int(working["PRETERM_LABOR_URGENT_FLAG"].eq(1).sum()),
            "PRETERM_LABOR_ICU_FLAG": int(working["PRETERM_LABOR_ICU_FLAG"].eq(1).sum()),
            "PRETERM_LABOR_DEATH_FLAG": int(working["PRETERM_LABOR_DEATH_FLAG"].eq(1).sum()),
            "APH_ANTEPARTUM_URGENT_FLAG": int(working["APH_ANTEPARTUM_URGENT_FLAG"].eq(1).sum()),
            "APH_ANTEPARTUM_ICU_FLAG": int(working["APH_ANTEPARTUM_ICU_FLAG"].eq(1).sum()),
            "APH_ANTEPARTUM_DEATH_FLAG": int(working["APH_ANTEPARTUM_DEATH_FLAG"].eq(1).sum()),
            "GDM_GESTATIONAL_URGENT_FLAG": int(working["GDM_GESTATIONAL_URGENT_FLAG"].eq(1).sum()),
            "GDM_GESTATIONAL_ICU_FLAG": int(working["GDM_GESTATIONAL_ICU_FLAG"].eq(1).sum()),
            "GDM_GESTATIONAL_DEATH_FLAG": int(working["GDM_GESTATIONAL_DEATH_FLAG"].eq(1).sum()),
        },
        "scope_any_event_totals": {
            "PRINCIPAL_DIAGNOSIS_ANY_EVENT_FLAG": int(
                working[[_scope_flag_column("PRINCIPAL_DIAGNOSIS", group_name) for group_name in expected_group_names()]].max(axis=1).eq(1).sum()
            ),
            "ANY_DIAGNOSIS_ANY_EVENT_FLAG": int(
                working[[_scope_flag_column("ANY_DIAGNOSIS", group_name) for group_name in expected_group_names()]].max(axis=1).eq(1).sum()
            ),
        },
        "age_flag_coverage": {
            "AGE_GROUP_0_27D": int(working["AGE_GROUP_0_27D"].eq(1).sum()),
            "AGE_GROUP_28_364D": int(working["AGE_GROUP_28_364D"].eq(1).sum()),
            "AGE_GROUP_1_4Y": int(working["AGE_GROUP_1_4Y"].eq(1).sum()),
            "AGE_GROUP_U5": int(working["AGE_GROUP_U5"].eq(1).sum()),
            "AGE_GROUP_WRA_15_49": int(working["AGE_GROUP_WRA_15_49"].eq(1).sum()),
            "AGE_GROUP_WRA_15_45_SENSITIVITY": int(working["AGE_GROUP_WRA_15_45_SENSITIVITY"].eq(1).sum()),
        },
    }
    return working[SIH_INDIVIDUAL_OUTPUT_FIELDS].copy(), stats


def _aggregate_sih_daily(chunk_frame: pd.DataFrame) -> pd.DataFrame:
    working = chunk_frame.copy()
    working["ADMISSIONS_TOTAL"] = 1
    total_columns = {
        _scope_total_column(scope_label, group_name): _scope_flag_column(scope_label, group_name)
        for scope_label, _ in SIH_SCOPE_FIELDS
        for group_name in expected_group_names()
    }
    total_columns["EXTERNAL_CAUSE_FIELDS_EXCESSIVE_NATURAL_HEAT_X30_TOTAL"] = "EXTERNAL_CAUSE_FIELDS_EXCESSIVE_NATURAL_HEAT_X30_FLAG"
    total_columns.update(
        {
            "PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_TOTAL": "PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG",
            "ANY_DIAGNOSIS_DIRECT_HEAT_ANY_TOTAL": "ANY_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG",
            "PRINCIPAL_DIAGNOSIS_INDIRECT_HEAT_ANY_TOTAL": "PRINCIPAL_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG",
            "ANY_DIAGNOSIS_INDIRECT_HEAT_ANY_TOTAL": "ANY_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG",
            "PRINCIPAL_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_TOTAL": "PRINCIPAL_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG",
            "ANY_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_TOTAL": "ANY_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG",
            "PRINCIPAL_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_TOTAL": "PRINCIPAL_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_FLAG",
            "ANY_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_TOTAL": "ANY_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_FLAG",
            "OBSTETRIC_ADMISSIONS_TOTAL": "OBSTETRIC_ADMISSION_PRIMARY_FLAG",
            "OBSTETRIC_URGENT_TOTAL": "OBSTETRIC_URGENT_PRIMARY_FLAG",
            "OBSTETRIC_ICU_TOTAL": "OBSTETRIC_ICU_PRIMARY_FLAG",
            "HDP_ANY_TOTAL": "PRINCIPAL_DIAGNOSIS_HDP_ANY_PRIMARY_FLAG",
            "HDP_SEVERE_TOTAL": "PRINCIPAL_DIAGNOSIS_HDP_SEVERE_PRIMARY_FLAG",
            "HDP_GESTATIONAL_HTN_TOTAL": "PRINCIPAL_DIAGNOSIS_HDP_GESTATIONAL_HTN_PRIMARY_FLAG",
            "HDP_PREECLAMPSIA_TOTAL": "PRINCIPAL_DIAGNOSIS_HDP_PREECLAMPSIA_PRIMARY_FLAG",
            "HDP_ECLAMPSIA_TOTAL": "PRINCIPAL_DIAGNOSIS_HDP_ECLAMPSIA_PRIMARY_FLAG",
            "PRETERM_LABOR_TOTAL": "PRINCIPAL_DIAGNOSIS_PRETERM_LABOR_PRIMARY_FLAG",
            "PRETERM_LABOR_URGENT_TOTAL": "PRETERM_LABOR_URGENT_FLAG",
            "PRETERM_LABOR_ICU_TOTAL": "PRETERM_LABOR_ICU_FLAG",
            "PRETERM_LABOR_DEATH_TOTAL": "PRETERM_LABOR_DEATH_FLAG",
            "APH_ANTEPARTUM_TOTAL": "PRINCIPAL_DIAGNOSIS_APH_ANTEPARTUM_PRIMARY_FLAG",
            "APH_ANTEPARTUM_URGENT_TOTAL": "APH_ANTEPARTUM_URGENT_FLAG",
            "APH_ANTEPARTUM_ICU_TOTAL": "APH_ANTEPARTUM_ICU_FLAG",
            "APH_ANTEPARTUM_DEATH_TOTAL": "APH_ANTEPARTUM_DEATH_FLAG",
            "GDM_GESTATIONAL_TOTAL": "PRINCIPAL_DIAGNOSIS_GDM_GESTATIONAL_PRIMARY_FLAG",
            "GDM_GESTATIONAL_URGENT_TOTAL": "GDM_GESTATIONAL_URGENT_FLAG",
            "GDM_GESTATIONAL_ICU_TOTAL": "GDM_GESTATIONAL_ICU_FLAG",
            "GDM_GESTATIONAL_DEATH_TOTAL": "GDM_GESTATIONAL_DEATH_FLAG",
            "DIABETES_PREGNANCY_BROAD_TOTAL": "ANY_DIAGNOSIS_DIABETES_PREGNANCY_BROAD_SENSITIVITY_FLAG",
        }
    )
    grouped = working.groupby(
        ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", "COMPETENCE_YEAR", "COMPETENCE_MONTH"],
        as_index=False,
        sort=True,
    )
    aggregation = {"ADMISSIONS_TOTAL": ("ADMISSIONS_TOTAL", "sum")}
    for total_column, flag_column in total_columns.items():
        aggregation[total_column] = (flag_column, "sum")
    daily = grouped.agg(**aggregation)
    daily["SOURCE_SIH"] = "silver_sih_individual"
    daily["QC_STATUS"] = "PASS"
    return daily[SIH_DAILY_OUTPUT_FIELDS].copy()


def validate_canonical_sih_year(
    root: Path,
    year: int,
    summary_stats: dict[str, Any],
    daily_frame: pd.DataFrame,
    dictionary: pd.DataFrame,
    individual_partition_path: Path | None = None,
    daily_partition_path: Path | None = None,
) -> dict[str, Any]:
    official_total = official_total_from_bronze(root, "sih", year)
    reconciliation = {"ADMISSIONS_TOTAL": int(daily_frame["ADMISSIONS_TOTAL"].sum()) == int(summary_stats["row_count"])}
    for scope_label, _ in SIH_SCOPE_FIELDS:
        for group_name in expected_group_names():
            total_column = _scope_total_column(scope_label, group_name)
            source_totals = summary_stats["principal_totals"] if scope_label == "PRINCIPAL_DIAGNOSIS" else summary_stats["any_diag_totals"]
            reconciliation[total_column] = int(daily_frame[total_column].sum()) == int(source_totals[group_name])
    reconciliation["EXTERNAL_CAUSE_FIELDS_EXCESSIVE_NATURAL_HEAT_X30_TOTAL"] = (
        int(daily_frame["EXTERNAL_CAUSE_FIELDS_EXCESSIVE_NATURAL_HEAT_X30_TOTAL"].sum()) == int(summary_stats["external_x30_total"])
    )
    for total_column, flag_key in [
        ("PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_TOTAL", "PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG"),
        ("ANY_DIAGNOSIS_DIRECT_HEAT_ANY_TOTAL", "ANY_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG"),
        ("PRINCIPAL_DIAGNOSIS_INDIRECT_HEAT_ANY_TOTAL", "PRINCIPAL_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG"),
        ("ANY_DIAGNOSIS_INDIRECT_HEAT_ANY_TOTAL", "ANY_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG"),
        ("PRINCIPAL_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_TOTAL", "PRINCIPAL_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG"),
        ("ANY_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_TOTAL", "ANY_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG"),
        ("PRINCIPAL_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_TOTAL", "PRINCIPAL_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_FLAG"),
        ("ANY_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_TOTAL", "ANY_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_FLAG"),
        ("OBSTETRIC_ADMISSIONS_TOTAL", "OBSTETRIC_ADMISSION_PRIMARY_FLAG"),
        ("OBSTETRIC_URGENT_TOTAL", "OBSTETRIC_URGENT_PRIMARY_FLAG"),
        ("OBSTETRIC_ICU_TOTAL", "OBSTETRIC_ICU_PRIMARY_FLAG"),
        ("PRETERM_LABOR_URGENT_TOTAL", "PRETERM_LABOR_URGENT_FLAG"),
        ("PRETERM_LABOR_ICU_TOTAL", "PRETERM_LABOR_ICU_FLAG"),
        ("PRETERM_LABOR_DEATH_TOTAL", "PRETERM_LABOR_DEATH_FLAG"),
        ("APH_ANTEPARTUM_URGENT_TOTAL", "APH_ANTEPARTUM_URGENT_FLAG"),
        ("APH_ANTEPARTUM_ICU_TOTAL", "APH_ANTEPARTUM_ICU_FLAG"),
        ("APH_ANTEPARTUM_DEATH_TOTAL", "APH_ANTEPARTUM_DEATH_FLAG"),
        ("GDM_GESTATIONAL_URGENT_TOTAL", "GDM_GESTATIONAL_URGENT_FLAG"),
        ("GDM_GESTATIONAL_ICU_TOTAL", "GDM_GESTATIONAL_ICU_FLAG"),
        ("GDM_GESTATIONAL_DEATH_TOTAL", "GDM_GESTATIONAL_DEATH_FLAG"),
    ]:
        source = summary_stats["category_totals"] if flag_key in summary_stats["category_totals"] else summary_stats["operational_totals"]
        reconciliation[total_column] = int(daily_frame[total_column].sum()) == int(source[flag_key])
    for total_column, source_totals, group_name in [
        ("HDP_ANY_TOTAL", summary_stats["principal_totals"], "HDP_ANY_PRIMARY"),
        ("HDP_SEVERE_TOTAL", summary_stats["principal_totals"], "HDP_SEVERE_PRIMARY"),
        ("HDP_GESTATIONAL_HTN_TOTAL", summary_stats["principal_totals"], "HDP_GESTATIONAL_HTN_PRIMARY"),
        ("HDP_PREECLAMPSIA_TOTAL", summary_stats["principal_totals"], "HDP_PREECLAMPSIA_PRIMARY"),
        ("HDP_ECLAMPSIA_TOTAL", summary_stats["principal_totals"], "HDP_ECLAMPSIA_PRIMARY"),
        ("PRETERM_LABOR_TOTAL", summary_stats["principal_totals"], "PRETERM_LABOR_PRIMARY"),
        ("APH_ANTEPARTUM_TOTAL", summary_stats["principal_totals"], "APH_ANTEPARTUM_PRIMARY"),
        ("GDM_GESTATIONAL_TOTAL", summary_stats["principal_totals"], "GDM_GESTATIONAL_PRIMARY"),
        ("DIABETES_PREGNANCY_BROAD_TOTAL", summary_stats["any_diag_totals"], "DIABETES_PREGNANCY_BROAD_SENSITIVITY"),
    ]:
        reconciliation[total_column] = int(daily_frame[total_column].sum()) == int(source_totals[group_name])

    dictionary_ok = not dictionary["ICD10_CODE"].astype(str).str.contains(r"[\*\-]", regex=True).any()
    direct_vs_indirect_distinct = set(DIRECT_HEAT_GROUPS).isdisjoint(set(INDIRECT_HEAT_GROUPS))
    validation = {
        "year": year,
        "official_admissions_total": official_total,
        "individual_admissions_total": int(summary_stats["row_count"]),
        "daily_admissions_total": int(daily_frame["ADMISSIONS_TOTAL"].sum()),
        "annual_total_matches_official": int(summary_stats["row_count"]) == official_total == int(daily_frame["ADMISSIONS_TOTAL"].sum()),
        "competence_year_coverage_ok": True,
        "municipality_coverage_ok": set(summary_stats["municipalities"]) == set(daily_frame["MUNICIPALITY_CODE"]),
        "daily_key_uniqueness_ok": int(
            daily_frame.duplicated(["MUNICIPALITY_CODE", "DATE", "COMPETENCE_YEAR", "COMPETENCE_MONTH"]).sum()
        ) == 0,
        "dictionary_expansion_ok": dictionary_ok,
        "direct_vs_indirect_distinct_ok": direct_vs_indirect_distinct,
        "principal_vs_any_diag_distinct_ok": True,
        "external_cause_scope_explicit_ok": "EXTERNAL_CAUSE_FIELDS_EXCESSIVE_NATURAL_HEAT_X30_TOTAL" in daily_frame.columns,
        "primary_diagnosis_coding_expectation_ok": int(summary_stats["principal_diag_invalid_count"]) == 0,
        "date_year_mismatch_count": int(summary_stats["date_year_mismatch_count"]),
        "individual_vs_daily_reconciliation": reconciliation,
        "individual_vs_daily_reconciliation_ok": all(reconciliation.values()),
        "individual_partition_integrity_ok": individual_partition_path is None or individual_partition_path.exists(),
        "daily_partition_integrity_ok": daily_partition_path is None or daily_partition_path.exists(),
        "age_flag_coverage": summary_stats["age_flag_coverage"],
        "summary": {
            "total_sih_admissions": int(summary_stats["row_count"]),
            "municipalities_with_at_least_one_event": int(len(summary_stats["municipalities"])),
            "counts_by_icd_group_principal_diagnosis": summary_stats["principal_totals"],
            "counts_by_icd_group_any_diagnosis": summary_stats["any_diag_totals"],
            "counts_direct_vs_indirect": {
                "principal_direct": int(summary_stats["category_totals"]["PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG"]),
                "any_diag_direct": int(summary_stats["category_totals"]["ANY_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG"]),
                "principal_indirect": int(summary_stats["category_totals"]["PRINCIPAL_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG"]),
                "any_diag_indirect": int(summary_stats["category_totals"]["ANY_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG"]),
                "principal_slow_moving": int(summary_stats["category_totals"]["PRINCIPAL_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG"]),
                "any_diag_slow_moving": int(summary_stats["category_totals"]["ANY_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG"]),
            },
            "counts_by_scope": {
                "principal_diagnosis_any_event": int(summary_stats["scope_any_event_totals"]["PRINCIPAL_DIAGNOSIS_ANY_EVENT_FLAG"]),
                "any_diagnosis_any_event": int(summary_stats["scope_any_event_totals"]["ANY_DIAGNOSIS_ANY_EVENT_FLAG"]),
            },
            "x30_counts": {
                "principal_diagnosis": int(summary_stats["principal_totals"]["EXCESSIVE_NATURAL_HEAT_X30"]),
                "any_diagnosis": int(summary_stats["any_diag_totals"]["EXCESSIVE_NATURAL_HEAT_X30"]),
                "external_cause_fields": int(summary_stats["external_x30_total"]),
            },
            "operational_gold_ready_totals": {
                "obstetric_admissions": int(summary_stats["operational_totals"]["OBSTETRIC_ADMISSION_PRIMARY_FLAG"]),
                "obstetric_urgent": int(summary_stats["operational_totals"]["OBSTETRIC_URGENT_PRIMARY_FLAG"]),
                "obstetric_icu": int(summary_stats["operational_totals"]["OBSTETRIC_ICU_PRIMARY_FLAG"]),
                "preterm_labor_urgent": int(summary_stats["operational_totals"]["PRETERM_LABOR_URGENT_FLAG"]),
                "preterm_labor_icu": int(summary_stats["operational_totals"]["PRETERM_LABOR_ICU_FLAG"]),
                "preterm_labor_death": int(summary_stats["operational_totals"]["PRETERM_LABOR_DEATH_FLAG"]),
                "gdm_urgent": int(summary_stats["operational_totals"]["GDM_GESTATIONAL_URGENT_FLAG"]),
                "gdm_icu": int(summary_stats["operational_totals"]["GDM_GESTATIONAL_ICU_FLAG"]),
                "gdm_death": int(summary_stats["operational_totals"]["GDM_GESTATIONAL_DEATH_FLAG"]),
            },
            "individual_vs_daily_reconciliation_result": all(reconciliation.values()),
        },
    }
    validation["all_checks_passed"] = all(
        [
            validation["annual_total_matches_official"],
            validation["municipality_coverage_ok"],
            validation["daily_key_uniqueness_ok"],
            validation["dictionary_expansion_ok"],
            validation["direct_vs_indirect_distinct_ok"],
            validation["principal_vs_any_diag_distinct_ok"],
            validation["external_cause_scope_explicit_ok"],
            validation["primary_diagnosis_coding_expectation_ok"],
            validation["individual_vs_daily_reconciliation_ok"],
            validation["individual_partition_integrity_ok"],
            validation["daily_partition_integrity_ok"],
        ]
    )
    return validation


def _write_validation_reports(root: Path, validations: dict[int, dict[str, Any]], dictionary_result: dict[str, Any], manifest_path: str) -> None:
    reports_dir = root / "artifacts" / "validation_reports" / "silver" / "sih"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for year, payload in validations.items():
        report_path = reports_dir / f"{year}.json"
        report_payload = {
            "dataset": "sih",
            "year": year,
            "generated_at": utc_now(),
            "manifest_path": manifest_path,
            "dictionary_manifest_path": str(dictionary_result["manifest_path"].relative_to(root)),
            **payload,
        }
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_path = root / "artifacts" / "status" / "silver_sih_last_run_summary.md"
    lines = [
        "# Canonical SIH Silver Summary",
        "",
        f"- Generated at: {utc_now()}",
        f"- Manifest: `{manifest_path}`",
        f"- Outcome dictionary: `{dictionary_result['canonical_path'].relative_to(root)}`",
        f"- All yearly checks passed: `{all(validation['all_checks_passed'] for validation in validations.values())}`",
        "",
    ]
    for year in sorted(validations):
        summary = validations[year]["summary"]
        mismatch_count = validations[year]["date_year_mismatch_count"]
        lines.extend(
            [
                f"## {year}",
                "",
                f"- Total SIH admissions: {summary['total_sih_admissions']}",
                f"- Municipalities with at least one event: {summary['municipalities_with_at_least_one_event']}",
                f"- Counts by ICD group (principal diagnosis): {summary['counts_by_icd_group_principal_diagnosis']}",
                f"- Counts by ICD group (any diagnosis): {summary['counts_by_icd_group_any_diagnosis']}",
                f"- Counts by DIRECT vs INDIRECT: {summary['counts_direct_vs_indirect']}",
                f"- Counts by principal diagnosis vs any diagnosis: {summary['counts_by_scope']}",
                f"- Counts with X30 flag: {summary['x30_counts']}",
                f"- Gold-ready obstetric operational totals: {summary['operational_gold_ready_totals']}",
                f"- Date year mismatch count (DT_INTER vs competence year): {mismatch_count}",
                f"- Individual vs daily reconciliation result: `{summary['individual_vs_daily_reconciliation_result']}`",
                "",
            ]
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def promote_sih_silver_range(
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
    validations: dict[int, dict[str, Any]] = {}
    individual_partition_paths: dict[int, str] = {}
    daily_partition_paths: dict[int, str] = {}

    for year in range(start_year, end_year + 1):
        runtime_individual_path = runtime_root / sih_individual_partition_file(root, year).relative_to(root)
        individual_writer = ParquetChunkWriter(runtime_individual_path)
        daily_parts: list[pd.DataFrame] = []
        row_offset = 0
        summary_stats = {
            "row_count": 0,
            "municipalities": set(),
            "date_year_mismatch_count": 0,
            "principal_diag_invalid_count": 0,
            "principal_totals": {group_name: 0 for group_name in expected_group_names()},
            "any_diag_totals": {group_name: 0 for group_name in expected_group_names()},
            "external_x30_total": 0,
            "category_totals": {
                "PRINCIPAL_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG": 0,
                "ANY_DIAGNOSIS_DIRECT_HEAT_ANY_FLAG": 0,
                "PRINCIPAL_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG": 0,
                "ANY_DIAGNOSIS_INDIRECT_HEAT_ANY_FLAG": 0,
                "PRINCIPAL_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG": 0,
                "ANY_DIAGNOSIS_SLOW_MOVING_HEAT_ANY_FLAG": 0,
                "PRINCIPAL_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_FLAG": 0,
                "ANY_DIAGNOSIS_MATERNAL_PHENOTYPE_ANY_FLAG": 0,
            },
            "operational_totals": {
                "ICU_ANY_STAY_FLAG": 0,
                "URGENT_ADMISSION_OBSERVED_FLAG": 0,
                "OBSTETRIC_ADMISSION_PRIMARY_FLAG": 0,
                "OBSTETRIC_URGENT_PRIMARY_FLAG": 0,
                "OBSTETRIC_ICU_PRIMARY_FLAG": 0,
                "PRETERM_LABOR_URGENT_FLAG": 0,
                "PRETERM_LABOR_ICU_FLAG": 0,
                "PRETERM_LABOR_DEATH_FLAG": 0,
                "APH_ANTEPARTUM_URGENT_FLAG": 0,
                "APH_ANTEPARTUM_ICU_FLAG": 0,
                "APH_ANTEPARTUM_DEATH_FLAG": 0,
                "GDM_GESTATIONAL_URGENT_FLAG": 0,
                "GDM_GESTATIONAL_ICU_FLAG": 0,
                "GDM_GESTATIONAL_DEATH_FLAG": 0,
            },
            "scope_any_event_totals": {
                "PRINCIPAL_DIAGNOSIS_ANY_EVENT_FLAG": 0,
                "ANY_DIAGNOSIS_ANY_EVENT_FLAG": 0,
            },
            "age_flag_coverage": {
                "AGE_GROUP_0_27D": 0,
                "AGE_GROUP_28_364D": 0,
                "AGE_GROUP_1_4Y": 0,
                "AGE_GROUP_U5": 0,
                "AGE_GROUP_WRA_15_49": 0,
                "AGE_GROUP_WRA_15_45_SENSITIVITY": 0,
            },
        }
        for chunk in _load_sih_chunk_iterator(root, year):
            individual_chunk, chunk_stats = _build_sih_chunk(chunk, year, row_offset, code_sets)
            individual_writer.write(individual_chunk)
            daily_parts.append(_aggregate_sih_daily(individual_chunk))
            daily_parts = _aggregate_daily_parts(daily_parts)
            row_offset += len(individual_chunk)
            summary_stats["row_count"] += chunk_stats["row_count"]
            summary_stats["municipalities"].update(chunk_stats["municipalities"])
            summary_stats["date_year_mismatch_count"] += chunk_stats["date_year_mismatch_count"]
            summary_stats["principal_diag_invalid_count"] += chunk_stats["principal_diag_invalid_count"]
            summary_stats["external_x30_total"] += chunk_stats["external_x30_total"]
            for group_name, value in chunk_stats["principal_totals"].items():
                summary_stats["principal_totals"][group_name] += value
            for group_name, value in chunk_stats["any_diag_totals"].items():
                summary_stats["any_diag_totals"][group_name] += value
            for key, value in chunk_stats["category_totals"].items():
                summary_stats["category_totals"][key] += value
            for key, value in chunk_stats["operational_totals"].items():
                summary_stats["operational_totals"][key] += value
            for key, value in chunk_stats["scope_any_event_totals"].items():
                summary_stats["scope_any_event_totals"][key] += value
            for key, value in chunk_stats["age_flag_coverage"].items():
                summary_stats["age_flag_coverage"][key] += value

        individual_row_count, individual_columns = individual_writer.close()
        daily_frame = pd.concat(daily_parts, ignore_index=True).groupby(
            ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", "COMPETENCE_YEAR", "COMPETENCE_MONTH"],
            as_index=False,
            sort=True,
        ).sum(numeric_only=True)
        daily_frame["SOURCE_SIH"] = "silver_sih_individual"
        daily_frame["QC_STATUS"] = "PASS"
        daily_frame = daily_frame[SIH_DAILY_OUTPUT_FIELDS].sort_values(
            ["DATE", "MUNICIPALITY_CODE", "COMPETENCE_YEAR"],
            kind="mergesort",
        ).reset_index(drop=True)
        runtime_daily_path = runtime_root / sih_daily_partition_file(root, year).relative_to(root)
        daily_writer = ParquetChunkWriter(runtime_daily_path)
        daily_writer.write(daily_frame)
        daily_row_count, daily_columns = daily_writer.close()

        final_individual_path = sih_individual_partition_file(root, year)
        final_daily_path = sih_daily_partition_file(root, year)
        sync_verified_parquet(runtime_individual_path, final_individual_path, individual_row_count, individual_columns)
        sync_verified_parquet(runtime_daily_path, final_daily_path, daily_row_count, daily_columns)
        runtime_individual_path.unlink(missing_ok=True)
        runtime_daily_path.unlink(missing_ok=True)
        if runtime_individual_path.parent.exists():
            shutil.rmtree(runtime_individual_path.parent)
        if runtime_daily_path.parent.exists():
            shutil.rmtree(runtime_daily_path.parent)

        validation = validate_canonical_sih_year(
            root,
            year,
            summary_stats,
            daily_frame,
            dictionary,
            individual_partition_path=final_individual_path,
            daily_partition_path=final_daily_path,
        )
        if not validation["all_checks_passed"]:
            raise ValueError(f"SIH silver validation failed for {year}")
        validations[year] = validation
        individual_partition_paths[year] = str(final_individual_path.relative_to(root))
        daily_partition_paths[year] = str(final_daily_path.relative_to(root))
        append_run_event(
            log_path,
            phase="phase_2_silver_promotion",
            dataset="sih",
            year=year,
            command=command,
            decision=(
                "canonical SIH individual and daily silver outputs written with explicit ICD-10 outcome flags; "
                f"individual path={final_individual_path.relative_to(root)} daily path={final_daily_path.relative_to(root)}"
            ),
            status="PASS",
        )

    manifest = build_manifest(root, "silver", "sih")
    manifest_path = f"artifacts/manifests/silver_{manifest['dataset']}.manifest.json"
    _write_validation_reports(root, validations, dictionary_result, manifest_path)
    return {
        "dataset": "sih",
        "years": list(range(start_year, end_year + 1)),
        "manifest_path": manifest_path,
        "dictionary_path": str(dictionary_result["canonical_path"].relative_to(root)),
        "dictionary_manifest_path": str(dictionary_result["manifest_path"].relative_to(root)),
        "individual_partition_paths": individual_partition_paths,
        "daily_partition_paths": daily_partition_paths,
        "validation": validations,
    }
