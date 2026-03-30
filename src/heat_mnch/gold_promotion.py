from __future__ import annotations

import calendar
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .gold_documentation import (
    DOCUMENTED_GOLD_ITEMS,
    build_gold_readiness_ledger,
    build_gold_table_summaries,
    build_gold_variable_dictionary,
    build_stage5_mini_report,
    build_stage5_table_year_matrix,
)
from .gold_materialization import (
    build_gold_analysis_child_u5_year,
    build_gold_analysis_wra_15_45_sens_year,
    build_gold_analysis_wra_15_49_year,
    build_gold_mortality_year,
    build_gold_panel_year,
    build_gold_perinatal_year,
)
from .manifest import build_manifest, sha256_file
from .outcome_dictionary_icd10 import load_unified_outcome_dictionary
from .run_log import append_run_event


REPORT_NAME = "gold_promotion_report.md"
MATRIX_NAME = "gold_promotion_matrix.csv"
LEDGER_NAME = "gold_promotion_ledger.csv"
BLOCKERS_NAME = "gold_promotion_blockers.md"
MINI_REPORT_NAME = "gold_promotion_mini_report.csv"
MANIFEST_NAME = "gold_release_manifest.json"
MANIFEST_PACKAGE_DIRNAME = "gold_promotion_manifests"
PERINATAL_STAGE_REPORT_NAME = "gold_stage1_perinatal_report.md"
PERINATAL_STAGE_MATRIX_NAME = "gold_stage1_perinatal_matrix.csv"
PERINATAL_STAGE_LEDGER_NAME = "gold_stage1_perinatal_ledger.csv"
PERINATAL_STAGE_BLOCKERS_NAME = "gold_stage1_perinatal_blockers.md"
PERINATAL_STAGE_MINI_REPORT_NAME = "gold_stage1_perinatal_mini_report.csv"
PERINATAL_STAGE_MANIFEST_NAME = "gold_stage1_perinatal_manifest.json"
PERINATAL_STAGE_PACKAGE_DIRNAME = "gold_stage1_perinatal_manifests"
MORTALITY_STAGE_REPORT_NAME = "gold_stage2_mortality_report.md"
MORTALITY_STAGE_MATRIX_NAME = "gold_stage2_mortality_matrix.csv"
MORTALITY_STAGE_LEDGER_NAME = "gold_stage2_mortality_ledger.csv"
MORTALITY_STAGE_BLOCKERS_NAME = "gold_stage2_mortality_blockers.md"
MORTALITY_STAGE_MINI_REPORT_NAME = "gold_stage2_mortality_mini_report.csv"
MORTALITY_STAGE_MANIFEST_NAME = "gold_stage2_mortality_manifest.json"
MORTALITY_STAGE_PACKAGE_DIRNAME = "gold_stage2_mortality_manifests"
PANEL_STAGE_REPORT_NAME = "gold_stage3_panel_report.md"
PANEL_STAGE_MATRIX_NAME = "gold_stage3_panel_matrix.csv"
PANEL_STAGE_LEDGER_NAME = "gold_stage3_panel_ledger.csv"
PANEL_STAGE_BLOCKERS_NAME = "gold_stage3_panel_blockers.md"
PANEL_STAGE_MINI_REPORT_NAME = "gold_stage3_panel_mini_report.csv"
PANEL_STAGE_MANIFEST_NAME = "gold_stage3_panel_manifest.json"
PANEL_STAGE_PACKAGE_DIRNAME = "gold_stage3_panel_manifests"
DERIVED_STAGE_REPORT_NAME = "gold_stage4_derived_report.md"
DERIVED_STAGE_MATRIX_NAME = "gold_stage4_derived_matrix.csv"
DERIVED_STAGE_LEDGER_NAME = "gold_stage4_derived_ledger.csv"
DERIVED_STAGE_BLOCKERS_NAME = "gold_stage4_derived_blockers.md"
DERIVED_STAGE_MINI_REPORT_NAME = "gold_stage4_derived_mini_report.csv"
DERIVED_STAGE_MANIFEST_NAME = "gold_stage4_derived_manifest.json"
DERIVED_STAGE_PACKAGE_DIRNAME = "gold_stage4_derived_manifests"
DICTIONARY_STAGE_REPORT_NAME = "gold_stage5_dictionary_report.md"
DICTIONARY_STAGE_MATRIX_NAME = "gold_stage5_dictionary_matrix.csv"
DICTIONARY_STAGE_LEDGER_NAME = "gold_stage5_readiness_ledger.csv"
DICTIONARY_STAGE_BLOCKERS_NAME = "gold_stage5_dictionary_blockers.md"
DICTIONARY_STAGE_MINI_REPORT_NAME = "gold_stage5_dictionary_mini_report.csv"
DICTIONARY_STAGE_VARIABLE_DICTIONARY_NAME = "gold_stage5_variable_dictionary.csv"
DICTIONARY_STAGE_MANIFEST_NAME = "gold_stage5_dictionary_manifest.json"
DICTIONARY_STAGE_PACKAGE_DIRNAME = "gold_stage5_dictionary_manifests"
RELEASE_STAGE_REPORT_NAME = "gold_release_candidate_report.md"
RELEASE_STAGE_MATRIX_NAME = "gold_release_candidate_matrix.csv"
RELEASE_STAGE_FAILURE_NOTES_NAME = "failure_notes.md"
RELEASE_STAGE_ENDPOINT_LEDGER_NAME = "endpoint_readiness_ledger.csv"
RELEASE_STAGE_MANIFEST_NAME = "gold_stage6_release_manifest.json"
RELEASE_STAGE_PACKAGE_DIRNAME = "gold_release_candidate_manifests"

CANONICAL_GOLD_TABLES = (
    "gold_mnch_perinatal",
    "gold_mnch_mortality",
    "gold_mnch_panel",
)
DERIVED_GOLD_VIEWS = (
    "gold_analysis_child_u5",
    "gold_analysis_wra_15_49",
    "gold_analysis_wra_15_45_sens",
)
SILVER_MANIFEST_DATASETS = (
    "climate",
    "sinasc",
    "sim",
    "sih",
    "stillbirth",
    "population",
    "analytical",
)

PERINATAL_REQUIRED_SINASC_COLUMNS = {
    "MUNICIPALITY_CODE",
    "CODMUNRES",
    "DATE",
    "YEAR",
    "MONTH",
    "DTNASC",
    "GA_WEEKS_CORRECTED_FINAL",
    "GA_DAYS_FINAL",
    "GA_VALID_FOR_ANALYSIS_FLAG",
    "PRETERM_LT37_FLAG",
    "PRETERM_LT32_FLAG",
    "PRETERM_DAYS_SHORT_OF_37W_FINAL",
    "BIRTH_WEIGHT_G",
    "BIRTH_WEIGHT_REPORTED_FLAG",
    "LOW_BIRTH_WEIGHT_LT2500_FLAG",
    "PRETERM_LOW_BIRTH_WEIGHT_FLAG",
    "PRENATAL_VISITS_COUNT",
    "PRENATAL_VISITS_6PLUS_FLAG",
    "PRENATAL_VISITS_8PLUS_FLAG",
    "PRENATAL_VISITS_SOURCE",
    "PRENATAL_VISITS_RAW_VALUE",
    "PRENATAL_VISITS_MISSING_FLAG",
    "MATERNAL_AGE_YEARS",
    "MATERNAL_AGE_MISSING_FLAG",
    "MATERNAL_SCHOOLING_CODE_FINAL",
    "MATERNAL_SCHOOLING_SOURCE",
    "MATERNAL_SCHOOLING_MISSING_FLAG",
    "PRIOR_VAGINAL_BIRTHS_COUNT",
    "PRIOR_CESAREAN_BIRTHS_COUNT",
    "PARITY_TOTAL_PRIOR",
    "PARITY_TOTAL_PRIOR_MISSING_FLAG",
    "PLURALITY_CODE",
    "PLURALITY_CATEGORY",
    "MULTIPLE_GESTATION_FLAG",
    "NEWBORN_SEX_CODE",
    "NEWBORN_SEX_CATEGORY",
    "MATERNAL_RACE_COLOR_CODE",
    "MATERNAL_RACE_COLOR_CATEGORY",
    "CONGENITAL_ANOMALY_FLAG",
    "CONGENITAL_ANOMALY_MISSING_FLAG",
    "SOURCE_SINASC",
}
PERINATAL_REQUIRED_STILLBIRTH_COLUMNS = {
    "MUNICIPALITY_CODE",
    "EVENT_DATE",
    "YEAR",
    "MONTH",
    "LIVE_BIRTHS_TOTAL",
    "STILLBIRTH_TOTAL_GTE28",
    "STILLBIRTH_TOTAL_GTE22_SENSITIVITY",
    "TOTAL_BIRTHS_PRIMARY",
    "TOTAL_BIRTHS_SENSITIVITY_GTE22",
    "OFFSET_LOG_TOTAL_BIRTHS_PRIMARY",
    "OFFSET_LOG_TOTAL_BIRTHS_SENSITIVITY_GTE22",
    "SOURCE_STILLBIRTH",
    "QC_STATUS",
}
PERINATAL_REQUIRED_STILLBIRTH_INDIVIDUAL_COLUMNS = {
    "MUNICIPALITY_CODE",
    "EVENT_DATE",
    "IDADEMAE",
    "ESCMAE2010",
    "ESCMAE",
    "SERIESCMAE",
    "GRAVIDEZ",
    "SEXO",
    "MATERNAL_AGE_YEARS",
    "MATERNAL_AGE_MISSING_FLAG",
    "MATERNAL_SCHOOLING_CODE_FINAL",
    "MATERNAL_SCHOOLING_SOURCE",
    "MATERNAL_SCHOOLING_MISSING_FLAG",
    "PLURALITY_CODE",
    "PLURALITY_CATEGORY",
    "MULTIPLE_GESTATION_FLAG",
    "FETAL_SEX_CODE",
    "FETAL_SEX_CATEGORY",
    "STILLBIRTH_GTE28_PRIMARY_FLAG",
    "STILLBIRTH_GTE22_SENSITIVITY_FLAG",
    "SOURCE_STILLBIRTH",
    "QC_STATUS",
}
PERINATAL_REQUIRED_CLIMATE_COLUMNS = {
    "COD_MUNICIPIO",
    "DATE",
    "TMAX_C",
    "TMIN_C",
    "TMEAN_C",
    "RH_MIN_PCT",
    "RH_MEAN_PCT",
    "HEATWAVE_LEN_P95",
    "HEAT_INDEX_C",
    "FONTE_CLIMA_FINAL",
    "FONTE_TMAX_C_FINAL",
    "FONTE_TMIN_C_FINAL",
    "FONTE_TMEAN_C_FINAL",
    "FONTE_RH_MIN_PCT_FINAL",
    "FONTE_RH_MEAN_PCT_FINAL",
    "IMPUTADO_POR_FALLBACK",
    "IMPUTADO_TMAX_C",
    "IMPUTADO_TMIN_C",
    "IMPUTADO_TMEAN_C",
    "IMPUTADO_RH_MIN_PCT",
    "IMPUTADO_RH_MEAN_PCT",
    "CLIMATE_UPSTREAM_QUARANTINE_FLAG",
    "CLIMATE_RECORD_ANOMALY_FLAG",
    "CLIMATE_CORRECTION_PATH",
    "HEAT_INDEX_METHOD",
    "HEAT_INDEX_SOURCE_PAIR",
    "HEAT_INDEX_INPUTS_ZERO_MISSING",
}
PERINATAL_REQUIRED_SIH_COLUMNS = {
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
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
}
PERINATAL_REQUIRED_DICTIONARY_GROUPS = {
    "HDP_ANY_PRIMARY",
    "HDP_SEVERE_PRIMARY",
    "PRETERM_LABOR_PRIMARY",
    "THREATENED_PRETERM_LABOR_SENSITIVITY",
    "APH_ANTEPARTUM_PRIMARY",
    "GDM_GESTATIONAL_PRIMARY",
    "DIABETES_PREGNANCY_BROAD_SENSITIVITY",
}
PERINATAL_SUBTYPE_GROUPS = {
    "HDP_GESTATIONAL_HTN_PRIMARY",
    "HDP_PREECLAMPSIA_PRIMARY",
    "HDP_ECLAMPSIA_PRIMARY",
}
MORTALITY_REQUIRED_SIM_COLUMNS = {
    "AGE_SCOPE",
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
    "DEATHS_TOTAL",
    "SOURCE_SIM",
    "QC_STATUS",
}
MORTALITY_REQUIRED_SIM_INDIVIDUAL_COLUMNS = {
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
    "AGE_GROUP_0_27D",
    "AGE_GROUP_28_364D",
    "AGE_GROUP_U5",
    "SOURCE_SIM",
    "QC_STATUS",
}
MORTALITY_REQUIRED_POP_COLUMNS = {
    "MUNICIPALITY_CODE",
    "YEAR",
    "POP_UNDER5",
    "SOURCE_POPULATION",
    "QC_STATUS",
}
MORTALITY_REQUIRED_SINASC_COLUMNS = {
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "LIVE_BIRTHS_TOTAL",
    "SOURCE_SINASC",
    "QC_STATUS",
}
PANEL_REQUIRED_PERINATAL_COLUMNS = {
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
    "TMAX_C",
    "TMIN_C",
    "TMEAN_C",
    "RH_MIN_PCT",
    "RH_MEAN_PCT",
    "HEATWAVE_LEN_P95",
    "HEAT_INDEX_APPROX_C",
    "FONTE_CLIMA_FINAL",
    "FONTE_TMAX_C_FINAL",
    "FONTE_TMIN_C_FINAL",
    "FONTE_TMEAN_C_FINAL",
    "FONTE_RH_MIN_PCT_FINAL",
    "FONTE_RH_MEAN_PCT_FINAL",
    "IMPUTADO_POR_FALLBACK",
    "IMPUTADO_TMAX_C",
    "IMPUTADO_TMIN_C",
    "IMPUTADO_TMEAN_C",
    "IMPUTADO_RH_MIN_PCT",
    "IMPUTADO_RH_MEAN_PCT",
    "CLIMATE_UPSTREAM_QUARANTINE_FLAG",
    "CLIMATE_RECORD_ANOMALY_FLAG",
    "CLIMATE_CORRECTION_PATH",
    "HEAT_INDEX_METHOD",
    "HEAT_INDEX_SOURCE_PAIR",
    "HEAT_INDEX_INPUTS_ZERO_MISSING",
    "PRETERM_FLAG",
    "STILLBIRTH_TOTAL_PRIMARY_GTE28",
    "LIVE_BIRTHS_TOTAL",
    "TOTAL_BIRTHS_PRIMARY",
    "LOW_BIRTH_WEIGHT_LT2500_TOTAL",
    "PRETERM_LOW_BIRTH_WEIGHT_TOTAL",
    "PRENATAL_VISITS_8PLUS_FLAG",
    "OBSTETRIC_ADMISSIONS_TOTAL",
    "HDP_ANY_TOTAL",
    "HDP_SEVERE_TOTAL",
    "PRETERM_LABOR_TOTAL",
    "APH_ANTEPARTUM_TOTAL",
    "GDM_GESTATIONAL_TOTAL",
    "DENOMINATOR_ZERO_FLAG",
    "DENOMINATOR_MISSING_FLAG",
    "RATE_NOT_COMPUTABLE_FLAG",
    "OFFSET_VALID_FLAG",
    "QC_STATUS",
}
PANEL_REQUIRED_MORTALITY_COLUMNS = {
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
    "DEATHS_NEONATAL",
    "DEATHS_U1",
    "DEATHS_U5",
    "LIVE_BIRTHS_TOTAL_ANNUAL",
    "POP_U5",
    "ACUTE_CARE_PROXY_TOTAL",
    "ACUTE_CARE_PROXY_LABEL",
    "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL",
    "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL",
    "DENOMINATOR_ZERO_FLAG",
    "DENOMINATOR_MISSING_FLAG",
    "RATE_NOT_COMPUTABLE_FLAG",
    "OFFSET_VALID_FLAG",
    "QC_STATUS",
}
PANEL_REQUIRED_WRA_COLUMNS = {
    "MUNICIPALITY_CODE",
    "YEAR",
    "POP_WRA_15_49",
}
DERIVED_CHILD_REQUIRED_COLUMNS = {
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
    "DEATHS_NEONATAL",
    "DEATHS_U1",
    "DEATHS_U5",
    "NEONATAL_DEATHS_TOTAL",
    "POSTNEONATAL_DEATHS_TOTAL",
    "INFANT_DEATHS_TOTAL",
    "UNDER5_DEATHS_TOTAL",
    "LIVE_BIRTHS_TOTAL_ANNUAL",
    "POP_U5",
    "ACUTE_CARE_PROXY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL",
    "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL",
    "TMAX_C",
    "TMIN_C",
    "TMEAN_C",
    "RH_MIN_PCT",
    "RH_MEAN_PCT",
    "HEATWAVE_LEN_P95",
    "HEAT_INDEX_APPROX_C",
    "CLIMATE_UPSTREAM_QUARANTINE_FLAG",
    "CLIMATE_RECORD_ANOMALY_FLAG",
    "CLIMATE_CORRECTION_PATH",
    "DENOMINATOR_ZERO_FLAG",
    "DENOMINATOR_MISSING_FLAG",
    "RATE_NOT_COMPUTABLE_FLAG",
    "OFFSET_VALID_FLAG",
    "QC_STATUS",
}
DERIVED_WRA_REQUIRED_COLUMNS = {
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
    "PRETERM_FLAG",
    "OBSTETRIC_ADMISSIONS_TOTAL",
    "HDP_ANY_TOTAL",
    "PRETERM_LABOR_TOTAL",
    "APH_ANTEPARTUM_TOTAL",
    "GDM_GESTATIONAL_TOTAL",
    "TMAX_C",
    "TMIN_C",
    "TMEAN_C",
    "RH_MIN_PCT",
    "RH_MEAN_PCT",
    "HEATWAVE_LEN_P95",
    "HEAT_INDEX_APPROX_C",
    "CLIMATE_UPSTREAM_QUARANTINE_FLAG",
    "CLIMATE_RECORD_ANOMALY_FLAG",
    "CLIMATE_CORRECTION_PATH",
    "DENOMINATOR_ZERO_FLAG",
    "DENOMINATOR_MISSING_FLAG",
    "RATE_NOT_COMPUTABLE_FLAG",
    "OFFSET_VALID_FLAG",
    "QC_STATUS",
}
DERIVED_WRA_15_45_POP_COLUMNS = {
    "MUNICIPALITY_CODE",
    "YEAR",
    "POP_WRA_15_45_SENSITIVITY",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_output_path(root: Path, filename: str) -> Path:
    return root / "artifacts" / "status" / filename


def _root_output_path(root: Path, filename: str) -> Path:
    return root / filename


def _stage_manifest_path(root: Path) -> Path:
    return root / "artifacts" / "manifests" / MANIFEST_NAME


def _package_dir(root: Path) -> Path:
    return root / MANIFEST_PACKAGE_DIRNAME


def _parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema.names)


def _partition_path(root: Path, *parts: str) -> Path:
    return root.joinpath(*parts)


def _gold_output_path(root: Path, table: str, year: int) -> Path:
    return _partition_path(root, "data", "gold", table, f"YEAR={year}", f"{table}_{year}.parquet")


def _perinatal_year_paths(root: Path, year: int) -> dict[str, Path]:
    return {
        "sinasc": _partition_path(
            root, "data", "silver", "sinasc", "silver_sinasc_event", f"YEAR={year}", f"silver_sinasc_event_{year}.parquet"
        ),
        "stillbirth": _partition_path(
            root, "data", "silver", "stillbirth", "silver_stillbirth_daily", f"YEAR={year}", f"silver_stillbirth_daily_{year}.parquet"
        ),
        "stillbirth_individual": _partition_path(
            root,
            "data",
            "silver",
            "stillbirth",
            "silver_stillbirth_individual",
            f"YEAR={year}",
            f"silver_stillbirth_individual_{year}.parquet",
        ),
        "sih": _partition_path(
            root, "data", "silver", "sih", "silver_sih_daily", f"YEAR={year}", f"silver_sih_daily_{year}.parquet"
        ),
        "climate": _partition_path(
            root, "data", "silver", "climate", f"YEAR={year}", f"climate_canonical_daily_{year}.parquet"
        ),
    }


def _mortality_year_paths(root: Path, year: int) -> dict[str, Path]:
    return {
        "sim_child": _partition_path(
            root, "data", "silver", "analytical", "silver_child_u5_sim_daily", f"YEAR={year}", f"silver_child_u5_sim_daily_{year}.parquet"
        ),
        "sim_individual": _partition_path(
            root, "data", "silver", "sim", "silver_sim_individual", f"YEAR={year}", f"silver_sim_individual_{year}.parquet"
        ),
        "population": _partition_path(
            root, "data", "silver", "population", "silver_pop_under5", f"YEAR={year}", f"silver_pop_under5_{year}.parquet"
        ),
        "sinasc": _partition_path(
            root, "data", "silver", "sinasc", "silver_sinasc_daily", f"YEAR={year}", f"silver_sinasc_daily_{year}.parquet"
        ),
        "climate": _partition_path(
            root, "data", "silver", "climate", f"YEAR={year}", f"climate_canonical_daily_{year}.parquet"
        ),
    }


def _panel_year_paths(root: Path, year: int) -> dict[str, Path]:
    return {
        "canonical_gold_perinatal": _gold_output_path(root, "gold_mnch_perinatal", year),
        "canonical_gold_mortality": _gold_output_path(root, "gold_mnch_mortality", year),
        "wra_population": _partition_path(
            root, "data", "silver", "population", "silver_pop_wra_15_49", f"YEAR={year}", f"silver_pop_wra_15_49_{year}.parquet"
        ),
    }


def _derived_year_paths(root: Path, year: int, table: str) -> dict[str, Path]:
    if table == "gold_analysis_child_u5":
        return {
            "canonical_gold_mortality": _gold_output_path(root, "gold_mnch_mortality", year),
        }
    if table == "gold_analysis_wra_15_49":
        return {
            "canonical_gold_perinatal": _gold_output_path(root, "gold_mnch_perinatal", year),
            "silver_population": _partition_path(root, "data", "silver", "population", "silver_pop_wra_15_49", f"YEAR={year}", f"silver_pop_wra_15_49_{year}.parquet"),
        }
    return {
        "canonical_gold_perinatal": _gold_output_path(root, "gold_mnch_perinatal", year),
        "silver_population": _partition_path(
            root, "data", "silver", "population", "silver_pop_wra_15_45_sensitivity", f"YEAR={year}", f"silver_pop_wra_15_45_sensitivity_{year}.parquet"
        ),
    }


def _copy_output_to_root(root: Path, path: Path) -> Path:
    destination = _root_output_path(root, path.name)
    shutil.copy2(path, destination)
    return destination


def _copy_manifest_package(root: Path, manifest_paths: list[Path]) -> Path:
    package_dir = _package_dir(root)
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    copied_entries: list[dict[str, Any]] = []
    for source_path in manifest_paths:
        destination = package_dir / source_path.name
        shutil.copy2(source_path, destination)
        copied_entries.append(
            {
                "filename": destination.name,
                "source_path": str(source_path.relative_to(root)),
                "package_path": str(destination.relative_to(root)),
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    index_path = package_dir / "manifest_index.json"
    index_path.write_text(
        json.dumps(
            {
                "generated_at": utc_now(),
                "entry_count": len(copied_entries),
                "entries": copied_entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return package_dir


def _refresh_manifests(root: Path) -> list[Path]:
    manifest_paths: list[Path] = []
    for dataset in SILVER_MANIFEST_DATASETS:
        build_manifest(root, "silver", dataset)
        manifest_paths.append(root / "artifacts" / "manifests" / f"silver_{dataset}.manifest.json")
    for dataset in [*CANONICAL_GOLD_TABLES, *DERIVED_GOLD_VIEWS]:
        build_manifest(root, "gold", dataset)
        manifest_paths.append(root / "artifacts" / "manifests" / f"gold_{dataset}.manifest.json")
    extra_manifests = (
        root / "artifacts" / "manifests" / "mnch_outcome_dictionary_icd10.manifest.json",
        root / "artifacts" / "manifests" / "silver_release_manifest.json",
    )
    manifest_paths.extend(path for path in extra_manifests if path.exists())
    return manifest_paths


def _refresh_perinatal_stage_manifests(root: Path) -> list[Path]:
    manifest_paths: list[Path] = []
    for dataset in ("climate", "sinasc", "sih", "stillbirth"):
        build_manifest(root, "silver", dataset)
        manifest_paths.append(root / "artifacts" / "manifests" / f"silver_{dataset}.manifest.json")
    build_manifest(root, "gold", "gold_mnch_perinatal")
    manifest_paths.append(root / "artifacts" / "manifests" / "gold_gold_mnch_perinatal.manifest.json")
    extra_manifests = (
        root / "artifacts" / "manifests" / "mnch_outcome_dictionary_icd10.manifest.json",
        root / "artifacts" / "manifests" / "silver_release_manifest.json",
    )
    manifest_paths.extend(path for path in extra_manifests if path.exists())
    return manifest_paths


def _refresh_mortality_stage_manifests(root: Path) -> list[Path]:
    manifest_paths: list[Path] = []
    for dataset in ("climate", "sinasc", "sim", "population", "analytical"):
        build_manifest(root, "silver", dataset)
        manifest_paths.append(root / "artifacts" / "manifests" / f"silver_{dataset}.manifest.json")
    build_manifest(root, "gold", "gold_mnch_mortality")
    manifest_paths.append(root / "artifacts" / "manifests" / "gold_gold_mnch_mortality.manifest.json")
    extra_manifests = (
        root / "artifacts" / "manifests" / "mnch_outcome_dictionary_icd10.manifest.json",
        root / "artifacts" / "manifests" / "silver_release_manifest.json",
        root / "artifacts" / "manifests" / PERINATAL_STAGE_MANIFEST_NAME,
    )
    manifest_paths.extend(path for path in extra_manifests if path.exists())
    return manifest_paths


def _refresh_panel_stage_manifests(root: Path) -> list[Path]:
    manifest_paths: list[Path] = []
    for dataset in ("climate", "population"):
        build_manifest(root, "silver", dataset)
        manifest_paths.append(root / "artifacts" / "manifests" / f"silver_{dataset}.manifest.json")
    for dataset in ("gold_mnch_perinatal", "gold_mnch_mortality", "gold_mnch_panel"):
        build_manifest(root, "gold", dataset)
        manifest_paths.append(root / "artifacts" / "manifests" / f"gold_{dataset}.manifest.json")
    extra_manifests = (
        root / "artifacts" / "manifests" / "silver_release_manifest.json",
        root / "artifacts" / "manifests" / PERINATAL_STAGE_MANIFEST_NAME,
        root / "artifacts" / "manifests" / MORTALITY_STAGE_MANIFEST_NAME,
    )
    manifest_paths.extend(path for path in extra_manifests if path.exists())
    return manifest_paths


def _refresh_derived_stage_manifests(root: Path) -> list[Path]:
    manifest_paths: list[Path] = []
    build_manifest(root, "silver", "population")
    manifest_paths.append(root / "artifacts" / "manifests" / "silver_population.manifest.json")
    for dataset in DERIVED_GOLD_VIEWS:
        build_manifest(root, "gold", dataset)
        manifest_paths.append(root / "artifacts" / "manifests" / f"gold_{dataset}.manifest.json")
    extra_manifests = (
        root / "artifacts" / "manifests" / "silver_release_manifest.json",
        root / "artifacts" / "manifests" / PERINATAL_STAGE_MANIFEST_NAME,
        root / "artifacts" / "manifests" / MORTALITY_STAGE_MANIFEST_NAME,
        root / "artifacts" / "manifests" / PANEL_STAGE_MANIFEST_NAME,
    )
    manifest_paths.extend(path for path in extra_manifests if path.exists())
    return manifest_paths


def _refresh_dictionary_stage_manifests(root: Path) -> list[Path]:
    manifest_paths: list[Path] = []
    for dataset in DOCUMENTED_GOLD_ITEMS:
        build_manifest(root, "gold", dataset)
        manifest_paths.append(root / "artifacts" / "manifests" / f"gold_{dataset}.manifest.json")
    extra_manifests = (
        root / "artifacts" / "manifests" / "silver_release_manifest.json",
        root / "artifacts" / "manifests" / PERINATAL_STAGE_MANIFEST_NAME,
        root / "artifacts" / "manifests" / MORTALITY_STAGE_MANIFEST_NAME,
        root / "artifacts" / "manifests" / PANEL_STAGE_MANIFEST_NAME,
        root / "artifacts" / "manifests" / DERIVED_STAGE_MANIFEST_NAME,
    )
    manifest_paths.extend(path for path in extra_manifests if path.exists())
    return manifest_paths


def _refresh_release_stage_manifests(root: Path) -> list[Path]:
    manifest_paths: list[Path] = []
    for dataset in DOCUMENTED_GOLD_ITEMS:
        build_manifest(root, "gold", dataset)
        manifest_paths.append(root / "artifacts" / "manifests" / f"gold_{dataset}.manifest.json")
    extra_manifests = (
        root / "artifacts" / "manifests" / "silver_release_manifest.json",
        root / "artifacts" / "manifests" / PERINATAL_STAGE_MANIFEST_NAME,
        root / "artifacts" / "manifests" / MORTALITY_STAGE_MANIFEST_NAME,
        root / "artifacts" / "manifests" / PANEL_STAGE_MANIFEST_NAME,
        root / "artifacts" / "manifests" / DERIVED_STAGE_MANIFEST_NAME,
        root / "artifacts" / "manifests" / DICTIONARY_STAGE_MANIFEST_NAME,
    )
    manifest_paths.extend(path for path in extra_manifests if path.exists())
    return manifest_paths


def _read_ledger_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _approved_silver_release(root: Path) -> dict[str, Any]:
    blockers_path = root / "silver_release_candidate_blockers.md"
    ledger_path = root / "silver_release_candidate_ledger.csv"
    package_index_path = root / "silver_release_candidate_manifests" / "manifest_index.json"
    report_path = root / "silver_release_candidate_report.md"
    matrix_path = root / "silver_release_candidate_matrix.csv"

    missing = [
        str(path.relative_to(root))
        for path in (blockers_path, ledger_path, package_index_path, report_path, matrix_path)
        if not path.exists()
    ]
    if missing:
        return {
            "status": "FAIL",
            "ok": False,
            "missing_artifacts": missing,
            "notes": f"silver handoff package incomplete: {missing}",
        }

    blockers_text = blockers_path.read_text(encoding="utf-8")
    ledger_rows = _read_ledger_rows(ledger_path)
    ledger_pass = bool(ledger_rows) and all(row.get("status") == "PASS" for row in ledger_rows)
    blockers_pass = "Overall status: `PASS`" in blockers_text and "\n- None" in blockers_text
    manifest_index = json.loads(package_index_path.read_text(encoding="utf-8"))
    return {
        "status": "PASS" if ledger_pass and blockers_pass else "FAIL",
        "ok": ledger_pass and blockers_pass,
        "missing_artifacts": [],
        "notes": "approved silver release package present and PASS" if ledger_pass and blockers_pass else "silver release package contains FAIL rows",
        "manifest_entry_count": int(manifest_index.get("entry_count", 0)),
        "ledger_row_count": len(ledger_rows),
    }


def _approved_stage1_perinatal(root: Path) -> dict[str, Any]:
    report_path = root / PERINATAL_STAGE_REPORT_NAME
    ledger_path = root / PERINATAL_STAGE_LEDGER_NAME
    manifest_path = root / "artifacts" / "manifests" / PERINATAL_STAGE_MANIFEST_NAME
    package_index_path = root / PERINATAL_STAGE_PACKAGE_DIRNAME / "manifest_index.json"

    missing = [
        str(path.relative_to(root))
        for path in (report_path, ledger_path, manifest_path, package_index_path)
        if not path.exists()
    ]
    if missing:
        return {
            "status": "FAIL",
            "ok": False,
            "missing_artifacts": missing,
            "notes": f"gold stage 1 perinatal package incomplete: {missing}",
        }

    ledger_rows = _read_ledger_rows(ledger_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger_pass = bool(ledger_rows) and all(row.get("status") == "PASS" for row in ledger_rows)
    manifest_pass = manifest.get("status") == "PASS"
    return {
        "status": "PASS" if ledger_pass and manifest_pass else "FAIL",
        "ok": ledger_pass and manifest_pass,
        "missing_artifacts": [],
        "notes": "approved gold stage 1 perinatal package present and PASS" if ledger_pass and manifest_pass else "gold stage 1 perinatal package contains FAIL rows",
        "ledger_row_count": len(ledger_rows),
        "manifest_year_count": len(manifest.get("years", [])),
    }


def _approved_stage2_mortality(root: Path) -> dict[str, Any]:
    report_path = root / MORTALITY_STAGE_REPORT_NAME
    ledger_path = root / MORTALITY_STAGE_LEDGER_NAME
    manifest_path = root / "artifacts" / "manifests" / MORTALITY_STAGE_MANIFEST_NAME
    package_index_path = root / MORTALITY_STAGE_PACKAGE_DIRNAME / "manifest_index.json"

    missing = [
        str(path.relative_to(root))
        for path in (report_path, ledger_path, manifest_path, package_index_path)
        if not path.exists()
    ]
    if missing:
        return {
            "status": "FAIL",
            "ok": False,
            "missing_artifacts": missing,
            "notes": f"gold stage 2 mortality package incomplete: {missing}",
        }

    ledger_rows = _read_ledger_rows(ledger_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger_pass = bool(ledger_rows) and all(row.get("status") == "PASS" for row in ledger_rows)
    manifest_pass = manifest.get("status") == "PASS"
    return {
        "status": "PASS" if ledger_pass and manifest_pass else "FAIL",
        "ok": ledger_pass and manifest_pass,
        "missing_artifacts": [],
        "notes": "approved gold stage 2 mortality package present and PASS" if ledger_pass and manifest_pass else "gold stage 2 mortality package contains FAIL rows",
        "ledger_row_count": len(ledger_rows),
        "manifest_year_count": len(manifest.get("years", [])),
    }


def _approved_stage3_panel(root: Path) -> dict[str, Any]:
    report_path = root / PANEL_STAGE_REPORT_NAME
    ledger_path = root / PANEL_STAGE_LEDGER_NAME
    manifest_path = root / "artifacts" / "manifests" / PANEL_STAGE_MANIFEST_NAME
    package_index_path = root / PANEL_STAGE_PACKAGE_DIRNAME / "manifest_index.json"

    missing = [
        str(path.relative_to(root))
        for path in (report_path, ledger_path, manifest_path, package_index_path)
        if not path.exists()
    ]
    if missing:
        return {
            "status": "FAIL",
            "ok": False,
            "missing_artifacts": missing,
            "notes": f"gold stage 3 panel package incomplete: {missing}",
        }

    ledger_rows = _read_ledger_rows(ledger_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger_pass = bool(ledger_rows) and all(row.get("status") == "PASS" for row in ledger_rows)
    manifest_pass = manifest.get("status") == "PASS"
    return {
        "status": "PASS" if ledger_pass and manifest_pass else "FAIL",
        "ok": ledger_pass and manifest_pass,
        "missing_artifacts": [],
        "notes": "approved gold stage 3 panel package present and PASS" if ledger_pass and manifest_pass else "gold stage 3 panel package contains FAIL rows",
        "ledger_row_count": len(ledger_rows),
        "manifest_year_count": len(manifest.get("years", [])),
    }


def _approved_stage4_derived(root: Path) -> dict[str, Any]:
    report_path = root / DERIVED_STAGE_REPORT_NAME
    ledger_path = root / DERIVED_STAGE_LEDGER_NAME
    manifest_path = root / "artifacts" / "manifests" / DERIVED_STAGE_MANIFEST_NAME
    package_index_path = root / DERIVED_STAGE_PACKAGE_DIRNAME / "manifest_index.json"

    missing = [
        str(path.relative_to(root))
        for path in (report_path, ledger_path, manifest_path, package_index_path)
        if not path.exists()
    ]
    if missing:
        return {
            "status": "FAIL",
            "ok": False,
            "missing_artifacts": missing,
            "notes": f"gold stage 4 derived package incomplete: {missing}",
        }

    ledger_rows = _read_ledger_rows(ledger_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger_pass = bool(ledger_rows) and all(row.get("status") == "PASS" for row in ledger_rows)
    manifest_pass = manifest.get("status") == "PASS"
    return {
        "status": "PASS" if ledger_pass and manifest_pass else "FAIL",
        "ok": ledger_pass and manifest_pass,
        "missing_artifacts": [],
        "notes": "approved gold stage 4 derived package present and PASS" if ledger_pass and manifest_pass else "gold stage 4 derived package contains FAIL rows",
        "ledger_row_count": len(ledger_rows),
        "manifest_year_count": len(manifest.get("years", [])),
    }


def _approved_stage5_dictionary(root: Path) -> dict[str, Any]:
    report_path = root / DICTIONARY_STAGE_REPORT_NAME
    ledger_path = root / DICTIONARY_STAGE_LEDGER_NAME
    manifest_path = root / "artifacts" / "manifests" / DICTIONARY_STAGE_MANIFEST_NAME
    package_index_path = root / DICTIONARY_STAGE_PACKAGE_DIRNAME / "manifest_index.json"

    missing = [
        str(path.relative_to(root))
        for path in (report_path, ledger_path, manifest_path, package_index_path)
        if not path.exists()
    ]
    if missing:
        return {
            "status": "FAIL",
            "ok": False,
            "missing_artifacts": missing,
            "notes": f"gold stage 5 dictionary package incomplete: {missing}",
        }

    ledger_rows = _read_ledger_rows(ledger_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    valid_statuses = {"Ready now", "Partially ready", "Blocked"}
    ledger_pass = bool(ledger_rows) and all(row.get("status") in valid_statuses for row in ledger_rows)
    manifest_pass = manifest.get("status") == "PASS"
    return {
        "status": "PASS" if ledger_pass and manifest_pass else "FAIL",
        "ok": ledger_pass and manifest_pass,
        "missing_artifacts": [],
        "notes": "approved gold stage 5 dictionary package present and PASS" if ledger_pass and manifest_pass else "gold stage 5 dictionary package contains invalid readiness rows",
        "ledger_row_count": len(ledger_rows),
        "manifest_year_count": len(manifest.get("years", [])),
    }


def _missing_columns(path: Path, required_columns: set[str]) -> tuple[bool, list[str], set[str]]:
    if not path.exists():
        return False, sorted(required_columns), set()
    columns = _parquet_columns(path)
    missing = sorted(required_columns - columns)
    return len(missing) == 0, missing, columns


def _perinatal_schema_blockers(sih_daily_columns: set[str], dictionary_groups: set[str]) -> list[str]:
    blockers: list[str] = []
    if not PERINATAL_REQUIRED_SIH_COLUMNS.issubset(sih_daily_columns):
        missing = sorted(PERINATAL_REQUIRED_SIH_COLUMNS - sih_daily_columns)
        blockers.append(f"approved silver SIH output is missing required maternal core columns: {missing}")
    missing_groups = sorted(PERINATAL_REQUIRED_DICTIONARY_GROUPS - dictionary_groups)
    if missing_groups:
        blockers.append(f"active unified ICD-10 dictionary is missing required maternal groups: {missing_groups}")
    missing_subtype_groups = sorted(PERINATAL_SUBTYPE_GROUPS - dictionary_groups)
    if missing_subtype_groups:
        blockers.append(
            "active unified ICD-10 dictionary does not version gestational-HTN, preeclampsia, and eclampsia subtype groups; "
            f"missing={missing_subtype_groups}"
        )
    return blockers


def _evaluate_perinatal_year(root: Path, year: int, silver_release_ok: bool) -> dict[str, Any]:
    paths = _perinatal_year_paths(root, year)
    sinasc_ok, sinasc_missing, _ = _missing_columns(paths["sinasc"], PERINATAL_REQUIRED_SINASC_COLUMNS)
    stillbirth_ok, stillbirth_missing, _ = _missing_columns(paths["stillbirth"], PERINATAL_REQUIRED_STILLBIRTH_COLUMNS)
    stillbirth_individual_ok, stillbirth_individual_missing, _ = _missing_columns(
        paths["stillbirth_individual"], PERINATAL_REQUIRED_STILLBIRTH_INDIVIDUAL_COLUMNS
    )
    climate_ok, climate_missing, _ = _missing_columns(paths["climate"], PERINATAL_REQUIRED_CLIMATE_COLUMNS)
    sih_ok, sih_missing, sih_columns = _missing_columns(paths["sih"], PERINATAL_REQUIRED_SIH_COLUMNS)

    try:
        dictionary_groups = set(load_unified_outcome_dictionary(root)["OUTCOME_GROUP"].astype(str).unique().tolist())
        dictionary_ok = True
        dictionary_issue = ""
    except FileNotFoundError:
        dictionary_groups = set()
        dictionary_ok = False
        dictionary_issue = "active unified ICD-10 dictionary missing"

    blockers: list[str] = []
    if not silver_release_ok:
        blockers.append("approved silver release gate is not PASS")
    if not sinasc_ok:
        blockers.append(f"silver_sinasc_event missing required columns: {sinasc_missing}")
    if not stillbirth_ok:
        blockers.append(f"silver_stillbirth_daily missing required columns: {stillbirth_missing}")
    if not stillbirth_individual_ok:
        blockers.append(f"silver_stillbirth_individual missing required columns: {stillbirth_individual_missing}")
    if not climate_ok:
        blockers.append(f"silver_climate missing required columns: {climate_missing}")
    if not sih_ok:
        blockers.append(f"silver_sih_daily missing required columns: {sih_missing}")
    if not dictionary_ok:
        blockers.append(dictionary_issue)
    blockers.extend(_perinatal_schema_blockers(sih_columns, dictionary_groups))
    blockers = list(dict.fromkeys(blockers))

    return {
        "stage": "canonical_gold",
        "table": "gold_mnch_perinatal",
        "year": year,
        "status": "PASS" if not blockers else "FAIL",
        "stage_order_ok": True,
        "silver_release_ok": silver_release_ok,
        "input_files_ok": all(path.exists() for path in paths.values()),
        "required_columns_ok": sinasc_ok and stillbirth_ok and stillbirth_individual_ok and climate_ok and sih_ok,
        "dictionary_ok": dictionary_ok,
        "climate_ready_ok": climate_ok,
        "denominator_ready_ok": sinasc_ok and stillbirth_ok,
        "materialized": False,
        "blocking_issue": " | ".join(blockers),
    }


def _evaluate_mortality_year(root: Path, year: int, silver_release_ok: bool, stage_order_ok: bool) -> dict[str, Any]:
    paths = _mortality_year_paths(root, year)
    sim_ok, sim_missing, _ = _missing_columns(paths["sim_child"], MORTALITY_REQUIRED_SIM_COLUMNS)
    sim_individual_ok, sim_individual_missing, _ = _missing_columns(paths["sim_individual"], MORTALITY_REQUIRED_SIM_INDIVIDUAL_COLUMNS)
    pop_ok, pop_missing, _ = _missing_columns(paths["population"], MORTALITY_REQUIRED_POP_COLUMNS)
    sinasc_ok, sinasc_missing, _ = _missing_columns(paths["sinasc"], MORTALITY_REQUIRED_SINASC_COLUMNS)
    climate_ok, climate_missing, _ = _missing_columns(paths["climate"], PERINATAL_REQUIRED_CLIMATE_COLUMNS)
    try:
        load_unified_outcome_dictionary(root)
        dictionary_ok = True
        dictionary_issue = ""
    except FileNotFoundError:
        dictionary_ok = False
        dictionary_issue = "active unified ICD-10 dictionary missing for mortality cause-group outputs"

    blockers: list[str] = []
    if not silver_release_ok:
        blockers.append("approved silver release gate is not PASS")
    if not stage_order_ok:
        blockers.append("stage-order stop: prior `gold_stage1_perinatal` is not PASS")
    if not sim_ok:
        blockers.append(f"silver_child_u5_sim_daily missing required columns: {sim_missing}")
    if not sim_individual_ok:
        blockers.append(f"silver_sim_individual missing required columns: {sim_individual_missing}")
    if not pop_ok:
        blockers.append(f"silver_pop_under5 missing required columns: {pop_missing}")
    if not sinasc_ok:
        blockers.append(f"silver_sinasc_daily missing required columns: {sinasc_missing}")
    if not climate_ok:
        blockers.append(f"silver_climate missing required columns: {climate_missing}")
    if not dictionary_ok:
        blockers.append(dictionary_issue)

    return {
        "stage": "canonical_gold",
        "table": "gold_mnch_mortality",
        "year": year,
        "status": "PASS" if not blockers else "FAIL",
        "stage_order_ok": stage_order_ok,
        "silver_release_ok": silver_release_ok,
        "input_files_ok": all(path.exists() for path in paths.values()),
        "required_columns_ok": sim_ok and sim_individual_ok and pop_ok and sinasc_ok and climate_ok,
        "dictionary_ok": dictionary_ok,
        "climate_ready_ok": climate_ok,
        "denominator_ready_ok": pop_ok and sinasc_ok,
        "materialized": False,
        "blocking_issue": " | ".join(blockers),
    }


def _evaluate_panel_year(
    root: Path,
    year: int,
    silver_release_ok: bool,
    perinatal_stage_ok: bool,
    mortality_stage_ok: bool,
) -> dict[str, Any]:
    paths = _panel_year_paths(root, year)
    perinatal_ok, perinatal_missing, _ = _missing_columns(paths["canonical_gold_perinatal"], PANEL_REQUIRED_PERINATAL_COLUMNS)
    mortality_ok, mortality_missing, _ = _missing_columns(paths["canonical_gold_mortality"], PANEL_REQUIRED_MORTALITY_COLUMNS)
    wra_ok, wra_missing, _ = _missing_columns(paths["wra_population"], PANEL_REQUIRED_WRA_COLUMNS)

    blockers: list[str] = []
    if not silver_release_ok:
        blockers.append("approved silver release gate is not PASS")
    if not perinatal_stage_ok:
        blockers.append("stage-order stop: prior `gold_stage1_perinatal` is not PASS")
    if not mortality_stage_ok:
        blockers.append("stage-order stop: prior `gold_stage2_mortality` is not PASS")
    if not perinatal_ok:
        blockers.append(f"gold_mnch_perinatal missing required columns: {perinatal_missing}")
    if not mortality_ok:
        blockers.append(f"gold_mnch_mortality missing required columns: {mortality_missing}")
    if not wra_ok:
        blockers.append(f"silver_pop_wra_15_49 missing required columns: {wra_missing}")

    stage_order_ok = perinatal_stage_ok and mortality_stage_ok
    return {
        "stage": "canonical_gold",
        "table": "gold_mnch_panel",
        "year": year,
        "status": "PASS" if not blockers else "FAIL",
        "stage_order_ok": stage_order_ok,
        "silver_release_ok": silver_release_ok,
        "input_files_ok": all(path.exists() for path in paths.values()),
        "required_columns_ok": perinatal_ok and mortality_ok and wra_ok,
        "dictionary_ok": True,
        "climate_ready_ok": perinatal_ok and mortality_ok,
        "denominator_ready_ok": perinatal_ok and mortality_ok and wra_ok,
        "materialized": False,
        "blocking_issue": " | ".join(blockers),
    }


def _evaluate_derived_year(
    root: Path,
    year: int,
    table: str,
    silver_release_ok: bool,
    panel_stage_ok: bool,
) -> dict[str, Any]:
    paths = _derived_year_paths(root, year, table)
    blockers: list[str] = []

    if not silver_release_ok:
        blockers.append("approved silver release gate is not PASS")
    if not panel_stage_ok:
        blockers.append("stage-order stop: prior `gold_stage3_panel` is not PASS")

    if table == "gold_analysis_child_u5":
        gold_ok, gold_missing, _ = _missing_columns(paths["canonical_gold_mortality"], DERIVED_CHILD_REQUIRED_COLUMNS)
        required_columns_ok = gold_ok
        climate_ready_ok = gold_ok
        denominator_ready_ok = gold_ok
        if not gold_ok:
            blockers.append(f"gold_mnch_mortality missing required columns: {gold_missing}")
    elif table == "gold_analysis_wra_15_49":
        gold_ok, gold_missing, _ = _missing_columns(paths["canonical_gold_perinatal"], DERIVED_WRA_REQUIRED_COLUMNS)
        pop_ok, pop_missing, _ = _missing_columns(paths["silver_population"], PANEL_REQUIRED_WRA_COLUMNS)
        required_columns_ok = gold_ok and pop_ok
        climate_ready_ok = gold_ok
        denominator_ready_ok = gold_ok and pop_ok
        if not gold_ok:
            blockers.append(f"gold_mnch_perinatal missing required columns: {gold_missing}")
        if not pop_ok:
            blockers.append(f"silver_pop_wra_15_49 missing required columns: {pop_missing}")
    else:
        gold_ok, gold_missing, _ = _missing_columns(paths["canonical_gold_perinatal"], DERIVED_WRA_REQUIRED_COLUMNS)
        pop_ok, pop_missing, _ = _missing_columns(paths["silver_population"], DERIVED_WRA_15_45_POP_COLUMNS)
        required_columns_ok = gold_ok and pop_ok
        climate_ready_ok = gold_ok
        denominator_ready_ok = gold_ok and pop_ok
        if not gold_ok:
            blockers.append(f"gold_mnch_perinatal missing required columns: {gold_missing}")
        if not pop_ok:
            blockers.append(f"silver_pop_wra_15_45_sensitivity missing required columns: {pop_missing}")

    return {
        "stage": "derived_gold",
        "table": table,
        "year": year,
        "status": "PASS" if not blockers else "FAIL",
        "stage_order_ok": panel_stage_ok,
        "silver_release_ok": silver_release_ok,
        "input_files_ok": all(path.exists() for path in paths.values()),
        "required_columns_ok": required_columns_ok,
        "dictionary_ok": True,
        "climate_ready_ok": climate_ready_ok,
        "denominator_ready_ok": denominator_ready_ok,
        "materialized": False,
        "blocking_issue": " | ".join(blockers),
    }


def _blocked_row(stage: str, table: str, year: int, silver_release_ok: bool, blocking_issue: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "table": table,
        "year": year,
        "status": "FAIL",
        "stage_order_ok": False,
        "silver_release_ok": silver_release_ok,
        "input_files_ok": False,
        "required_columns_ok": False,
        "dictionary_ok": False,
        "climate_ready_ok": False,
        "denominator_ready_ok": False,
        "materialized": False,
        "blocking_issue": blocking_issue,
    }


def _materialized_row(base_row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    row = dict(base_row)
    row["status"] = "PASS"
    row["materialized"] = True
    row["blocking_issue"] = ""
    row["internal_reconciliation_summary"] = result.get("internal_reconciliation_summary", "")
    row["external_validation_summary"] = result.get("external_validation_summary", "")
    return row


def _failed_materialization_row(base_row: dict[str, Any], issue: str) -> dict[str, Any]:
    row = dict(base_row)
    row["status"] = "FAIL"
    row["materialized"] = False
    row["blocking_issue"] = issue
    row["internal_reconciliation_summary"] = row.get("internal_reconciliation_summary", "")
    row["external_validation_summary"] = row.get("external_validation_summary", "")
    return row


def _input_paths_for_row(root: Path, table: str, year: int) -> list[Path]:
    if table == "gold_mnch_perinatal":
        return list(_perinatal_year_paths(root, year).values())
    if table == "gold_mnch_mortality":
        return list(_mortality_year_paths(root, year).values())
    if table == "gold_mnch_panel":
        return list(_panel_year_paths(root, year).values())
    return list(_derived_year_paths(root, year, table).values())


def _denominator_used(table: str) -> str:
    if table == "gold_mnch_perinatal":
        return "annual total births including fetal deaths"
    if table in {"gold_mnch_mortality", "gold_analysis_child_u5"}:
        return "LIVE_BIRTHS_TOTAL_ANNUAL for neonatal+infant; POP_U5 for under-5"
    if table == "gold_mnch_panel":
        return "outcome-specific canonical denominators: preterm->daily live births; stillbirth->daily total births; neonatal+infant->annual live births; under-5->POP_U5; maternal acute->POP_WRA_15_49"
    if table == "gold_analysis_wra_15_49":
        return "POP_WRA_15_49"
    return "POP_WRA_15_45_SENSITIVITY"


def _corrective_action(row: dict[str, Any]) -> str:
    if row["table"] == "gold_mnch_perinatal":
        return (
            "Refresh approved silver SIH daily outputs and the unified ICD-10 dictionary, then rerun canonical gold perinatal."
        )
    if row["table"] == "gold_mnch_mortality":
        return "Resolve stage-order or mortality silver input blockers, then rerun canonical gold mortality only."
    if row["table"] == "gold_mnch_panel":
        return "Resolve canonical perinatal and mortality first, then rerun panel promotion."
    return "Resolve canonical gold blockers first, then materialize derived analytical views from approved canonical gold artifacts."


def _output_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "rows": 0,
            "unique_municipalities": 0,
            "min_date": "",
            "max_date": "",
            "day_completeness": "NOT_MATERIALIZED",
            "month_completeness": "NOT_MATERIALIZED",
            "year_completeness": "NOT_MATERIALIZED",
            "municipalities_with_at_least_one_event": 0,
            "total_events": 0,
            "crude_rate_when_applicable": "N/A_OUTPUT_NOT_MATERIALIZED",
            "climate_linkage_completeness": "NOT_EVALUATED_OUTPUT_NOT_MATERIALIZED",
            "missingness_in_critical_fields": "NOT_EVALUATED_OUTPUT_NOT_MATERIALIZED",
            "DENOMINATOR_ZERO_FLAG_summary": "N/A_OUTPUT_NOT_MATERIALIZED",
            "DENOMINATOR_MISSING_FLAG_summary": "N/A_OUTPUT_NOT_MATERIALIZED",
            "RATE_NOT_COMPUTABLE_FLAG_summary": "N/A_OUTPUT_NOT_MATERIALIZED",
            "OFFSET_VALID_FLAG_summary": "N/A_OUTPUT_NOT_MATERIALIZED",
        }
    frame = pd.read_parquet(path)
    table_name = path.parents[1].name
    municipality_column = "MUNICIPALITY_CODE" if "MUNICIPALITY_CODE" in frame.columns else "COD_MUNICIPIO"
    date_column = "DATE" if "DATE" in frame.columns else ("EVENT_DATE" if "EVENT_DATE" in frame.columns else "")
    event_columns = [column for column in frame.columns if column.endswith("_TOTAL")]
    total_events = int(frame[event_columns].sum(numeric_only=True).sum()) if event_columns else 0
    critical_columns = [column for column in ("MUNICIPALITY_CODE", "COD_MUNICIPIO", "DATE", "EVENT_DATE", "YEAR", "QC_STATUS") if column in frame.columns]
    critical_missing = int(frame[critical_columns].isna().sum().sum()) if critical_columns else 0
    min_date = ""
    max_date = ""
    day_completeness = "NOT_EVALUATED"
    month_completeness = "NOT_EVALUATED"
    year_completeness = "NOT_EVALUATED"
    if date_column:
        parsed_dates = pd.to_datetime(frame[date_column], errors="coerce")
        if parsed_dates.notna().any():
            min_date = str(parsed_dates.min().date())
            max_date = str(parsed_dates.max().date())
            years = sorted(parsed_dates.dropna().dt.year.unique().tolist())
            year_value = int(years[0]) if years else None
            unique_municipalities = int(frame[municipality_column].astype(str).nunique()) if municipality_column in frame.columns else 0
            days_in_year = 366 if year_value is not None and calendar.isleap(year_value) else (365 if year_value is not None else 0)
            expected_rows = unique_municipalities * days_in_year if unique_municipalities and days_in_year else 0
            observed_rows = int(len(frame))
            day_completeness = (
                f"{observed_rows}/{expected_rows} ({observed_rows / expected_rows:.4f})" if expected_rows else "NOT_EVALUATED"
            )
            months = parsed_dates.dt.to_period("M").astype(str)
            observed_muni_months = int(frame.assign(_MONTH_PERIOD=months)[[municipality_column, "_MONTH_PERIOD"]].drop_duplicates().shape[0])
            expected_muni_months = unique_municipalities * 12 if unique_municipalities else 0
            month_completeness = (
                f"{observed_muni_months}/{expected_muni_months} ({observed_muni_months / expected_muni_months:.4f})"
                if expected_muni_months
                else "NOT_EVALUATED"
            )
            municipality_day_counts = frame.groupby(municipality_column)[date_column].nunique()
            municipalities_full_year = int(municipality_day_counts.eq(days_in_year).sum()) if days_in_year else 0
            year_completeness = (
                f"{municipalities_full_year}/{unique_municipalities} ({municipalities_full_year / unique_municipalities:.4f})"
                if unique_municipalities
                else "NOT_EVALUATED"
            )
    climate_linkage = "NOT_APPLICABLE"
    if {"TMAX_C", "TMIN_C", "TMEAN_C", "RH_MIN_PCT", "RH_MEAN_PCT", "FONTE_CLIMA_FINAL"}.issubset(frame.columns):
        complete = int(frame[["TMAX_C", "TMIN_C", "TMEAN_C", "RH_MIN_PCT", "RH_MEAN_PCT", "FONTE_CLIMA_FINAL"]].notna().all(axis=1).sum())
        climate_linkage = f"{complete}/{len(frame)} ({complete / len(frame):.4f})" if len(frame) else "NOT_EVALUATED"
    crude_rate = "N/A"
    if table_name == "gold_mnch_perinatal":
        preterm_column = "PRETERM_FLAG" if "PRETERM_FLAG" in frame.columns else ("NV_PRETERM_TOTAL" if "NV_PRETERM_TOTAL" in frame.columns else "")
        live_births_column = "NV_TOTAL" if "NV_TOTAL" in frame.columns else ("LIVE_BIRTHS_TOTAL" if "LIVE_BIRTHS_TOTAL" in frame.columns else "")
        stillbirth_column = "STILLBIRTH_TOTAL_PRIMARY_GTE28" if "STILLBIRTH_TOTAL_PRIMARY_GTE28" in frame.columns else ("STILLBIRTH_TOTAL" if "STILLBIRTH_TOTAL" in frame.columns else "")
        total_births_column = "TOTAL_BIRTHS_PRIMARY" if "TOTAL_BIRTHS_PRIMARY" in frame.columns else (
            "NASCIMENTOS_TOTAIS_INCLUINDO_FETAIS" if "NASCIMENTOS_TOTAIS_INCLUINDO_FETAIS" in frame.columns else ""
        )
        rate_parts: list[str] = []
        if preterm_column and live_births_column:
            live_births_total = int(frame[live_births_column].sum())
            if live_births_total > 0:
                rate_parts.append(f"preterm_per_live_birth={frame[preterm_column].sum() / live_births_total:.6f}")
        if stillbirth_column and total_births_column:
            total_births = int(frame[total_births_column].sum())
            if total_births > 0:
                rate_parts.append(f"stillbirth_per_1000_total_births={frame[stillbirth_column].sum() * 1000.0 / total_births:.6f}")
        crude_rate = "; ".join(rate_parts) if rate_parts else "N/A_DENOMINATOR_INVALID"
    elif table_name in {"gold_mnch_mortality", "gold_analysis_child_u5"}:
        live_births_column = "LIVE_BIRTHS_TOTAL_ANNUAL" if "LIVE_BIRTHS_TOTAL_ANNUAL" in frame.columns else "LIVE_BIRTHS_TOTAL"
        under5_column = "POP_U5" if "POP_U5" in frame.columns else "UNDER5_POPULATION"
        neonatal_column = "DEATHS_NEONATAL" if "DEATHS_NEONATAL" in frame.columns else "NEONATAL_DEATHS_TOTAL"
        infant_column = "DEATHS_U1" if "DEATHS_U1" in frame.columns else "INFANT_DEATHS_TOTAL"
        under5_deaths_column = "DEATHS_U5" if "DEATHS_U5" in frame.columns else "UNDER5_DEATHS_TOTAL"
        annual = frame[["MUNICIPALITY_CODE", "YEAR", live_births_column, under5_column]].drop_duplicates()
        live_births_total = float(pd.to_numeric(annual[live_births_column], errors="coerce").fillna(0).sum())
        under5_total = float(pd.to_numeric(annual[under5_column], errors="coerce").fillna(0).sum())
        neonatal_rate = frame[neonatal_column].sum() * 1000.0 / live_births_total if live_births_total > 0 else float("nan")
        infant_rate = frame[infant_column].sum() * 1000.0 / live_births_total if live_births_total > 0 else float("nan")
        under5_rate = frame[under5_deaths_column].sum() * 100000.0 / under5_total if under5_total > 0 else float("nan")
        crude_rate = (
            f"neonatal_per_1000_lb={neonatal_rate:.6f}; infant_per_1000_lb={infant_rate:.6f}; under5_per_100000_pop={under5_rate:.6f}"
            if live_births_total > 0 and under5_total > 0
            else "N/A_DENOMINATOR_INVALID"
        )
    elif table_name == "gold_mnch_panel":
        annual = frame[
            ["MUNICIPALITY_CODE", "YEAR", "REF_LIVE_BIRTHS_TOTAL_ANNUAL", "REF_POP_U5", "REF_POP_WRA_15_49"]
        ].drop_duplicates()
        rate_parts: list[str] = []
        live_births_daily_total = float(pd.to_numeric(frame["REF_LIVE_BIRTHS_DAILY"], errors="coerce").fillna(0).sum())
        total_births_daily_total = float(pd.to_numeric(frame["REF_TOTAL_BIRTHS_PRIMARY_DAILY"], errors="coerce").fillna(0).sum())
        annual_live_births_total = float(pd.to_numeric(annual["REF_LIVE_BIRTHS_TOTAL_ANNUAL"], errors="coerce").fillna(0).sum())
        annual_under5_total = float(pd.to_numeric(annual["REF_POP_U5"], errors="coerce").fillna(0).sum())
        annual_wra_total = float(pd.to_numeric(annual["REF_POP_WRA_15_49"], errors="coerce").fillna(0).sum())
        if live_births_daily_total > 0:
            rate_parts.append(f"preterm_per_live_birth={frame['PRETERM_BIRTHS_CANONICAL_TOTAL'].sum() / live_births_daily_total:.6f}")
        if total_births_daily_total > 0:
            rate_parts.append(
                f"stillbirth_per_1000_total_births={frame['STILLBIRTH_TOTAL_PRIMARY_GTE28'].sum() * 1000.0 / total_births_daily_total:.6f}"
            )
        if annual_live_births_total > 0:
            rate_parts.append(f"neonatal_per_1000_lb={frame['DEATHS_NEONATAL'].sum() * 1000.0 / annual_live_births_total:.6f}")
            rate_parts.append(f"infant_per_1000_lb={frame['DEATHS_U1'].sum() * 1000.0 / annual_live_births_total:.6f}")
        if annual_under5_total > 0:
            rate_parts.append(f"under5_per_100000_pop={frame['DEATHS_U5'].sum() * 100000.0 / annual_under5_total:.6f}")
        if annual_wra_total > 0:
            rate_parts.append(
                f"obstetric_admissions_per_100000_wra={frame['OBSTETRIC_ADMISSIONS_TOTAL'].sum() * 100000.0 / annual_wra_total:.6f}"
            )
        crude_rate = "; ".join(rate_parts) if rate_parts else "N/A_DENOMINATOR_INVALID"
    elif table_name.startswith("gold_analysis_wra_") and "OBSTETRIC_ADMISSIONS_TOTAL" in frame.columns:
        pop_columns = [column for column in frame.columns if column.startswith("POP_WRA_")]
        if pop_columns:
            denominator_total = int(frame[["MUNICIPALITY_CODE", "YEAR", pop_columns[0]]].drop_duplicates()[pop_columns[0]].sum())
            crude_rate = (
                f"obstetric_admissions_per_100000_wra={frame['OBSTETRIC_ADMISSIONS_TOTAL'].sum() * 100000.0 / denominator_total:.6f}"
                if denominator_total > 0
                else "N/A_DENOMINATOR_INVALID"
            )
    return {
        "rows": int(len(frame)),
        "unique_municipalities": int(frame[municipality_column].astype(str).nunique()) if municipality_column in frame.columns else 0,
        "min_date": min_date,
        "max_date": max_date,
        "day_completeness": day_completeness,
        "month_completeness": month_completeness,
        "year_completeness": year_completeness,
        "municipalities_with_at_least_one_event": int(
            frame.loc[frame[event_columns].sum(axis=1).gt(0), municipality_column].astype(str).nunique()
        )
        if municipality_column in frame.columns and event_columns
        else 0,
        "total_events": total_events,
        "crude_rate_when_applicable": crude_rate,
        "climate_linkage_completeness": climate_linkage,
        "missingness_in_critical_fields": f"critical_missing_count={critical_missing}",
        "DENOMINATOR_ZERO_FLAG_summary": (
            f"sum={int(frame['DENOMINATOR_ZERO_FLAG'].fillna(0).sum())}" if "DENOMINATOR_ZERO_FLAG" in frame.columns else "COLUMN_MISSING"
        ),
        "DENOMINATOR_MISSING_FLAG_summary": (
            f"sum={int(frame['DENOMINATOR_MISSING_FLAG'].fillna(0).sum())}" if "DENOMINATOR_MISSING_FLAG" in frame.columns else "COLUMN_MISSING"
        ),
        "RATE_NOT_COMPUTABLE_FLAG_summary": (
            f"sum={int(frame['RATE_NOT_COMPUTABLE_FLAG'].fillna(0).sum())}" if "RATE_NOT_COMPUTABLE_FLAG" in frame.columns else "COLUMN_MISSING"
        ),
        "OFFSET_VALID_FLAG_summary": (
            f"valid_sum={int(frame['OFFSET_VALID_FLAG'].fillna(0).sum())}" if "OFFSET_VALID_FLAG" in frame.columns else "COLUMN_MISSING"
        ),
    }


def _mini_report_rows(root: Path, rows: list[dict[str, Any]], manifest_refreshed: bool) -> list[dict[str, Any]]:
    report_rows: list[dict[str, Any]] = []
    for row in rows:
        output_path = _gold_output_path(root, str(row["table"]), int(row["year"]))
        metrics = _output_metrics(output_path)
        input_paths = _input_paths_for_row(root, str(row["table"]), int(row["year"]))
        internal_summary = row.get(
            "internal_reconciliation_summary",
            (
                f"stage_order_ok={row['stage_order_ok']}; silver_release_ok={row['silver_release_ok']}; "
                f"required_columns_ok={row['required_columns_ok']}; dictionary_ok={row['dictionary_ok']}; "
                f"denominator_ready_ok={row['denominator_ready_ok']}"
            ),
        )
        external_summary = row.get(
            "external_validation_summary",
            "NOT_RUN_OUTPUT_NOT_MATERIALIZED" if not output_path.exists() else "MATERIALIZED_CHECK_NOT_IMPLEMENTED",
        )
        report_rows.append(
            {
                "table": row["table"],
                "year": int(row["year"]),
                "STATUS": row["status"],
                "input_silver_paths": "; ".join(str(path.relative_to(root)) for path in input_paths),
                "output_gold_path": str(output_path.relative_to(root)),
                "rows": metrics["rows"],
                "unique_municipalities": metrics["unique_municipalities"],
                "min_date": metrics["min_date"],
                "max_date": metrics["max_date"],
                "day_completeness": metrics["day_completeness"],
                "month_completeness": metrics["month_completeness"],
                "year_completeness": metrics["year_completeness"],
                "municipalities_with_at_least_one_event": metrics["municipalities_with_at_least_one_event"],
                "total_events": metrics["total_events"],
                "denominator_used": _denominator_used(str(row["table"])),
                "crude_rate_when_applicable": metrics["crude_rate_when_applicable"],
                "climate_linkage_completeness": (
                    metrics["climate_linkage_completeness"] if output_path.exists() else f"input_ready={row['climate_ready_ok']}; output_not_materialized"
                ),
                "missingness_in_critical_fields": (
                    metrics["missingness_in_critical_fields"] if output_path.exists() else f"required_columns_ok={row['required_columns_ok']}; output_not_materialized"
                ),
                "DENOMINATOR_ZERO_FLAG_summary": metrics["DENOMINATOR_ZERO_FLAG_summary"],
                "DENOMINATOR_MISSING_FLAG_summary": metrics["DENOMINATOR_MISSING_FLAG_summary"],
                "RATE_NOT_COMPUTABLE_FLAG_summary": metrics["RATE_NOT_COMPUTABLE_FLAG_summary"],
                "OFFSET_VALID_FLAG_summary": metrics["OFFSET_VALID_FLAG_summary"],
                "internal_reconciliation_summary": internal_summary,
                "external_validation_summary": external_summary,
                "manifest_refreshed_yes_no": "yes" if manifest_refreshed else "no",
                "blocker_and_corrective_action_if_FAIL": (
                    ""
                    if row["status"] == "PASS"
                    else f"{row['blocking_issue']} || corrective_action={_corrective_action(row)}"
                ),
            }
        )
    return report_rows


def _write_outputs(root: Path, rows: list[dict[str, Any]], manifest_paths: list[Path], silver_release: dict[str, Any]) -> dict[str, Any]:
    status_dir = root / "artifacts" / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    report_path = _status_output_path(root, REPORT_NAME)
    matrix_path = _status_output_path(root, MATRIX_NAME)
    ledger_path = _status_output_path(root, LEDGER_NAME)
    blockers_path = _status_output_path(root, BLOCKERS_NAME)
    mini_report_path = _status_output_path(root, MINI_REPORT_NAME)
    stage_manifest_path = _stage_manifest_path(root)

    matrix_columns = [
        "stage",
        "table",
        "year",
        "status",
        "stage_order_ok",
        "silver_release_ok",
        "input_files_ok",
        "required_columns_ok",
        "dictionary_ok",
        "climate_ready_ok",
        "denominator_ready_ok",
        "materialized",
        "blocking_issue",
    ]
    ledger_columns = ["stage", "table", "year", "status", "decision", "blocking_issue"]
    matrix_frame = pd.DataFrame(rows)[matrix_columns].sort_values(["stage", "table", "year"], kind="mergesort").reset_index(drop=True)
    matrix_frame.to_csv(matrix_path, index=False, quoting=csv.QUOTE_MINIMAL)

    ledger_frame = matrix_frame[["stage", "table", "year", "status", "blocking_issue"]].copy()
    ledger_frame["decision"] = ledger_frame["status"].map(lambda status: "materialized" if status == "PASS" else "promotion_blocked")
    ledger_frame = ledger_frame[ledger_columns]
    ledger_frame.to_csv(ledger_path, index=False, quoting=csv.QUOTE_MINIMAL)

    mini_report_frame = (
        pd.DataFrame(_mini_report_rows(root, rows, manifest_refreshed=True))
        .sort_values(["table", "year"], kind="mergesort")
        .reset_index(drop=True)
    )
    mini_report_frame.to_csv(mini_report_path, index=False, quoting=csv.QUOTE_MINIMAL)

    blocking_rows = [row for row in rows if row["status"] != "PASS"]
    unique_blockers = list(dict.fromkeys(row["blocking_issue"] for row in blocking_rows if row["blocking_issue"]))
    overall_status = "PASS" if not blocking_rows else "FAIL"
    canonical_tables_pass = sum(
        all(row["status"] == "PASS" for row in rows if row["table"] == table_name)
        for table_name in CANONICAL_GOLD_TABLES
    )
    derived_tables_pass = sum(
        all(row["status"] == "PASS" for row in rows if row["table"] == table_name)
        for table_name in DERIVED_GOLD_VIEWS
    )

    blocker_lines = [
        "# Gold Promotion Blockers",
        "",
        f"- Generated at: {utc_now()}",
        f"- Overall status: `{overall_status}`",
        "",
    ]
    if not unique_blockers:
        blocker_lines.append("- None")
    else:
        for issue in unique_blockers:
            blocker_lines.append(f"- {issue}")
    blockers_path.write_text("\n".join(blocker_lines) + "\n", encoding="utf-8")

    table_summaries: list[str] = []
    for table_name in [*CANONICAL_GOLD_TABLES, *DERIVED_GOLD_VIEWS]:
        table_rows = [row for row in rows if row["table"] == table_name]
        pass_years = [row["year"] for row in table_rows if row["status"] == "PASS"]
        fail_years = [row["year"] for row in table_rows if row["status"] != "PASS"]
        table_summaries.append(f"- {table_name}: PASS years={pass_years} FAIL years={fail_years}")

    if overall_status == "PASS":
        decision_lines = [
            "- Canonical gold tables were materialized for 2020-2024 in stage order.",
            "- Derived analytical views were materialized only after canonical gold stabilized.",
            "- No benchmark modeling was run in this stage.",
        ]
    else:
        decision_lines = [
            "- Promotion stopped at the first failing stage in the current run order.",
            "- Downstream stages after the first failure were not materialized.",
            "- No model formulas were embedded and no benchmark modeling was run in this stage.",
        ]

    report_lines = [
        "# Gold Promotion Report",
        "",
        f"- Generated at: {utc_now()}",
        f"- Overall status: `{overall_status}`",
        f"- Silver release gate: `{silver_release['status']}`",
        f"- Canonical gold tables materialized: `{canonical_tables_pass}/{len(CANONICAL_GOLD_TABLES)}`",
        f"- Derived analytical views materialized: `{derived_tables_pass}/{len(DERIVED_GOLD_VIEWS)}`",
        f"- Final manifests package: `{MANIFEST_PACKAGE_DIRNAME}`",
        f"- Mini-report: `{mini_report_path.relative_to(root)}`",
        "",
        "## Decision",
        "",
        *decision_lines,
        "",
        "## Table-Year Summary",
        "",
        *table_summaries,
        "",
        "## Blockers",
        "",
    ]
    if not unique_blockers:
        report_lines.append("- None")
    else:
        for issue in unique_blockers:
            report_lines.append(f"- {issue}")
    report_lines.extend(
        [
            "",
            "## Model Readiness",
            "",
            f"- Acute daily DAG family: {'READY' if overall_status == 'PASS' else 'NOT READY'}",
            f"- Gestational-window DAG family: {'READY' if overall_status == 'PASS' else 'NOT READY'}",
            f"- Intervention/policy DAG family: {'READY' if overall_status == 'PASS' else 'NOT READY'}",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    stage_manifest = {
        "generated_at": utc_now(),
        "status": overall_status,
        "years": sorted({int(row["year"]) for row in rows}),
        "report_path": str(report_path.relative_to(root)),
        "matrix_path": str(matrix_path.relative_to(root)),
        "ledger_path": str(ledger_path.relative_to(root)),
        "blockers_path": str(blockers_path.relative_to(root)),
        "mini_report_path": str(mini_report_path.relative_to(root)),
        "manifest_package_dir": MANIFEST_PACKAGE_DIRNAME,
        "silver_release_status": silver_release["status"],
        "blocking_issue_count": len(unique_blockers),
        "artifacts": [
            {"path": str(report_path.relative_to(root)), "sha256": sha256_file(report_path), "size_bytes": report_path.stat().st_size},
            {"path": str(matrix_path.relative_to(root)), "sha256": sha256_file(matrix_path), "size_bytes": matrix_path.stat().st_size},
            {"path": str(ledger_path.relative_to(root)), "sha256": sha256_file(ledger_path), "size_bytes": ledger_path.stat().st_size},
            {"path": str(mini_report_path.relative_to(root)), "sha256": sha256_file(mini_report_path), "size_bytes": mini_report_path.stat().st_size},
            {"path": str(blockers_path.relative_to(root)), "sha256": sha256_file(blockers_path), "size_bytes": blockers_path.stat().st_size},
        ],
    }
    stage_manifest_path.write_text(json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    package_dir = _copy_manifest_package(root, [*manifest_paths, stage_manifest_path])
    copied_report = _copy_output_to_root(root, report_path)
    copied_matrix = _copy_output_to_root(root, matrix_path)
    copied_ledger = _copy_output_to_root(root, ledger_path)
    copied_mini_report = _copy_output_to_root(root, mini_report_path)
    copied_blockers = _copy_output_to_root(root, blockers_path)
    return {
        "status": overall_status,
        "report_path": str(copied_report.relative_to(root)),
        "matrix_path": str(copied_matrix.relative_to(root)),
        "ledger_path": str(copied_ledger.relative_to(root)),
        "mini_report_path": str(copied_mini_report.relative_to(root)),
        "blockers_path": str(copied_blockers.relative_to(root)),
        "manifest_package_dir": str(package_dir.relative_to(root)),
        "stage_manifest_path": str(stage_manifest_path.relative_to(root)),
        "blocking_issue_count": len(unique_blockers),
    }


def _perinatal_stage_manifest_path(root: Path) -> Path:
    return root / "artifacts" / "manifests" / PERINATAL_STAGE_MANIFEST_NAME


def _perinatal_stage_package_dir(root: Path) -> Path:
    return root / PERINATAL_STAGE_PACKAGE_DIRNAME


def _perinatal_stage_mini_report_rows(root: Path, rows: list[dict[str, Any]], manifest_refreshed: bool) -> list[dict[str, Any]]:
    report_rows: list[dict[str, Any]] = []
    for row in rows:
        output_path = _gold_output_path(root, "gold_mnch_perinatal", int(row["year"]))
        metrics = _output_metrics(output_path)
        input_paths = _input_paths_for_row(root, "gold_mnch_perinatal", int(row["year"]))
        summary = row.get("perinatal_summary", {}) if isinstance(row.get("perinatal_summary"), dict) else {}
        validations = row.get("perinatal_validation_flags", {}) if isinstance(row.get("perinatal_validation_flags"), dict) else {}
        report_rows.append(
            {
                "table": row["table"],
                "year": int(row["year"]),
                "STATUS": row["status"],
                "input_silver_paths": "; ".join(str(path.relative_to(root)) for path in input_paths),
                "output_gold_path": str(output_path.relative_to(root)),
                "rows": metrics["rows"],
                "unique_municipalities": metrics["unique_municipalities"],
                "min_date": metrics["min_date"],
                "max_date": metrics["max_date"],
                "day_completeness": metrics["day_completeness"],
                "month_completeness": metrics["month_completeness"],
                "year_completeness": metrics["year_completeness"],
                "municipalities_with_at_least_one_event": metrics["municipalities_with_at_least_one_event"],
                "total_events": metrics["total_events"],
                "denominator_used": "daily total births primary/sensitivity for stillbirth; maternal acute WRA denominator not materialized in stage 1",
                "crude_rate_when_applicable": metrics["crude_rate_when_applicable"],
                "climate_linkage_completeness": metrics["climate_linkage_completeness"],
                "missingness_in_critical_fields": metrics["missingness_in_critical_fields"],
                "DENOMINATOR_ZERO_FLAG_summary": metrics["DENOMINATOR_ZERO_FLAG_summary"],
                "DENOMINATOR_MISSING_FLAG_summary": metrics["DENOMINATOR_MISSING_FLAG_summary"],
                "RATE_NOT_COMPUTABLE_FLAG_summary": metrics["RATE_NOT_COMPUTABLE_FLAG_summary"],
                "OFFSET_VALID_FLAG_summary": metrics["OFFSET_VALID_FLAG_summary"],
                "internal_reconciliation_summary": row.get("internal_reconciliation_summary", ""),
                "external_validation_summary": row.get("external_validation_summary", ""),
                "manifest_refreshed_yes_no": "yes" if manifest_refreshed else "no",
                "blocker_and_corrective_action_if_FAIL": (
                    ""
                    if row["status"] == "PASS"
                    else f"{row['blocking_issue']} || corrective_action={_corrective_action(row)}"
                ),
                "live_births": summary.get("live_births", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "preterm_births": summary.get("preterm_births", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "very_preterm_births": summary.get("very_preterm_births", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "crude_preterm_rate": (
                    f"{summary['preterm_births'] / summary['live_births']:.6f}"
                    if summary.get("live_births")
                    else "N/A_OUTPUT_NOT_MATERIALIZED"
                ),
                "mean_gest_days_lost_below_37w_among_preterm": summary.get("mean_gest_days_lost_among_preterm", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "median_gest_days_lost_below_37w_among_preterm": summary.get("median_gest_days_lost_among_preterm", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "stillbirth_primary_count": summary.get("stillbirth_primary_count", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "stillbirth_sensitivity_count": summary.get("stillbirth_sensitivity_count", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "stillbirth_rate_per_1000_total_births": summary.get("stillbirth_rate_per_1000_total_births", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "hdp_total": summary.get("hdp_total", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "hdp_severe_total": summary.get("hdp_severe_total", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "municipalities_with_at_least_one_preterm_event": summary.get("municipalities_with_at_least_one_preterm_event", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "municipalities_with_at_least_one_stillbirth_event": summary.get("municipalities_with_at_least_one_stillbirth_event", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "municipalities_with_at_least_one_hdp_event": summary.get("municipalities_with_at_least_one_hdp_event", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "prenatal_count_completeness": summary.get("prenatal_count_completeness", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "prenatal_6plus_prevalence": summary.get("prenatal_6plus_prevalence", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "event_vs_daily_reconciliation_results": json.dumps(
                    summary.get("event_vs_daily_reconciliation", validations),
                    sort_keys=True,
                ),
            }
        )
    return report_rows


def _write_perinatal_stage_outputs(
    root: Path,
    rows: list[dict[str, Any]],
    manifest_paths: list[Path],
    silver_release: dict[str, Any],
) -> dict[str, Any]:
    status_dir = root / "artifacts" / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    report_path = _status_output_path(root, PERINATAL_STAGE_REPORT_NAME)
    matrix_path = _status_output_path(root, PERINATAL_STAGE_MATRIX_NAME)
    ledger_path = _status_output_path(root, PERINATAL_STAGE_LEDGER_NAME)
    blockers_path = _status_output_path(root, PERINATAL_STAGE_BLOCKERS_NAME)
    mini_report_path = _status_output_path(root, PERINATAL_STAGE_MINI_REPORT_NAME)
    stage_manifest_path = _perinatal_stage_manifest_path(root)

    matrix_columns = [
        "stage",
        "table",
        "year",
        "status",
        "stage_order_ok",
        "silver_release_ok",
        "input_files_ok",
        "required_columns_ok",
        "dictionary_ok",
        "climate_ready_ok",
        "denominator_ready_ok",
        "materialized",
        "blocking_issue",
    ]
    matrix_frame = pd.DataFrame(rows)[matrix_columns].sort_values(["table", "year"], kind="mergesort").reset_index(drop=True)
    matrix_frame.to_csv(matrix_path, index=False, quoting=csv.QUOTE_MINIMAL)

    ledger_frame = matrix_frame[["stage", "table", "year", "status", "blocking_issue"]].copy()
    ledger_frame["decision"] = ledger_frame["status"].map(lambda status: "materialized" if status == "PASS" else "promotion_blocked")
    ledger_frame = ledger_frame[["stage", "table", "year", "status", "decision", "blocking_issue"]]
    ledger_frame.to_csv(ledger_path, index=False, quoting=csv.QUOTE_MINIMAL)

    mini_report_frame = pd.DataFrame(_perinatal_stage_mini_report_rows(root, rows, manifest_refreshed=True))
    mini_report_frame = mini_report_frame.sort_values(["table", "year"], kind="mergesort").reset_index(drop=True)
    mini_report_frame.to_csv(mini_report_path, index=False, quoting=csv.QUOTE_MINIMAL)

    blocking_rows = [row for row in rows if row["status"] != "PASS"]
    unique_blockers = list(dict.fromkeys(row["blocking_issue"] for row in blocking_rows if row["blocking_issue"]))
    overall_status = "PASS" if not blocking_rows else "FAIL"

    blocker_lines = [
        "# Gold Stage 1 Perinatal Blockers",
        "",
        f"- Generated at: {utc_now()}",
        f"- Overall status: `{overall_status}`",
        "",
    ]
    if not unique_blockers:
        blocker_lines.append("- None")
    else:
        blocker_lines.extend(f"- {issue}" for issue in unique_blockers)
    blockers_path.write_text("\n".join(blocker_lines) + "\n", encoding="utf-8")

    pass_years = [row["year"] for row in rows if row["status"] == "PASS"]
    fail_years = [row["year"] for row in rows if row["status"] != "PASS"]
    report_lines = [
        "# Gold Stage 1 Perinatal Report",
        "",
        f"- Generated at: {utc_now()}",
        f"- Overall status: `{overall_status}`",
        f"- Silver release gate: `{silver_release['status']}`",
        "- Stage executed: `gold_mnch_perinatal` only",
        "- No downstream gold stage was materialized in this run.",
        f"- PASS years: `{pass_years}`",
        f"- FAIL years: `{fail_years}`",
        f"- Mini-report: `{mini_report_path.relative_to(root)}`",
        f"- Manifest package: `{PERINATAL_STAGE_PACKAGE_DIRNAME}`",
        "",
        "## Decision",
        "",
        "- Stage 1 built only the canonical perinatal gold mart.",
        "- Mortality, panel, and derived analytical views were intentionally not advanced in this run.",
        "- No model formulas were embedded and no benchmark modeling was executed in this stage.",
        "",
        "## Blockers",
        "",
    ]
    if not unique_blockers:
        report_lines.append("- None")
    else:
        report_lines.extend(f"- {issue}" for issue in unique_blockers)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    stage_manifest = {
        "generated_at": utc_now(),
        "status": overall_status,
        "stage": "gold_stage1_perinatal",
        "years": sorted({int(row["year"]) for row in rows}),
        "report_path": str(report_path.relative_to(root)),
        "matrix_path": str(matrix_path.relative_to(root)),
        "ledger_path": str(ledger_path.relative_to(root)),
        "blockers_path": str(blockers_path.relative_to(root)),
        "mini_report_path": str(mini_report_path.relative_to(root)),
        "manifest_package_dir": PERINATAL_STAGE_PACKAGE_DIRNAME,
        "silver_release_status": silver_release["status"],
        "blocking_issue_count": len(unique_blockers),
        "artifacts": [
            {"path": str(report_path.relative_to(root)), "sha256": sha256_file(report_path), "size_bytes": report_path.stat().st_size},
            {"path": str(matrix_path.relative_to(root)), "sha256": sha256_file(matrix_path), "size_bytes": matrix_path.stat().st_size},
            {"path": str(ledger_path.relative_to(root)), "sha256": sha256_file(ledger_path), "size_bytes": ledger_path.stat().st_size},
            {"path": str(mini_report_path.relative_to(root)), "sha256": sha256_file(mini_report_path), "size_bytes": mini_report_path.stat().st_size},
            {"path": str(blockers_path.relative_to(root)), "sha256": sha256_file(blockers_path), "size_bytes": blockers_path.stat().st_size},
        ],
    }
    stage_manifest_path.write_text(json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    package_dir = _perinatal_stage_package_dir(root)
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    copied_entries: list[dict[str, Any]] = []
    for source_path in [*manifest_paths, stage_manifest_path]:
        destination = package_dir / source_path.name
        shutil.copy2(source_path, destination)
        copied_entries.append(
            {
                "filename": destination.name,
                "source_path": str(source_path.relative_to(root)),
                "package_path": str(destination.relative_to(root)),
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    index_path = package_dir / "manifest_index.json"
    index_path.write_text(
        json.dumps({"generated_at": utc_now(), "entry_count": len(copied_entries), "entries": copied_entries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    copied_report = _copy_output_to_root(root, report_path)
    copied_matrix = _copy_output_to_root(root, matrix_path)
    copied_ledger = _copy_output_to_root(root, ledger_path)
    copied_mini_report = _copy_output_to_root(root, mini_report_path)
    copied_blockers = _copy_output_to_root(root, blockers_path)
    return {
        "status": overall_status,
        "report_path": str(copied_report.relative_to(root)),
        "matrix_path": str(copied_matrix.relative_to(root)),
        "ledger_path": str(copied_ledger.relative_to(root)),
        "mini_report_path": str(copied_mini_report.relative_to(root)),
        "blockers_path": str(copied_blockers.relative_to(root)),
        "manifest_package_dir": str(package_dir.relative_to(root)),
        "stage_manifest_path": str(stage_manifest_path.relative_to(root)),
        "blocking_issue_count": len(unique_blockers),
    }


def _mortality_stage_manifest_path(root: Path) -> Path:
    return root / "artifacts" / "manifests" / MORTALITY_STAGE_MANIFEST_NAME


def _mortality_stage_package_dir(root: Path) -> Path:
    return root / MORTALITY_STAGE_PACKAGE_DIRNAME


def _mortality_stage_mini_report_rows(root: Path, rows: list[dict[str, Any]], manifest_refreshed: bool) -> list[dict[str, Any]]:
    report_rows: list[dict[str, Any]] = []
    for row in rows:
        output_path = _gold_output_path(root, "gold_mnch_mortality", int(row["year"]))
        metrics = _output_metrics(output_path)
        input_paths = _input_paths_for_row(root, "gold_mnch_mortality", int(row["year"]))
        summary = row.get("mortality_summary", {}) if isinstance(row.get("mortality_summary"), dict) else {}
        validations = row.get("mortality_validation_flags", {}) if isinstance(row.get("mortality_validation_flags"), dict) else {}
        cause_counts = summary.get("cause_group_family_counts", {}) if isinstance(summary.get("cause_group_family_counts"), dict) else {}
        report_rows.append(
            {
                "table": row["table"],
                "year": int(row["year"]),
                "STATUS": row["status"],
                "input_silver_paths": "; ".join(str(path.relative_to(root)) for path in input_paths),
                "output_gold_path": str(output_path.relative_to(root)),
                "rows": metrics["rows"],
                "unique_municipalities": metrics["unique_municipalities"],
                "min_date": metrics["min_date"],
                "max_date": metrics["max_date"],
                "day_completeness": metrics["day_completeness"],
                "month_completeness": metrics["month_completeness"],
                "year_completeness": metrics["year_completeness"],
                "municipalities_with_at_least_one_event": metrics["municipalities_with_at_least_one_event"],
                "total_events": metrics["total_events"],
                "denominator_used": _denominator_used("gold_mnch_mortality"),
                "crude_rate_when_applicable": metrics["crude_rate_when_applicable"],
                "climate_linkage_completeness": metrics["climate_linkage_completeness"],
                "missingness_in_critical_fields": metrics["missingness_in_critical_fields"],
                "DENOMINATOR_ZERO_FLAG_summary": metrics["DENOMINATOR_ZERO_FLAG_summary"],
                "DENOMINATOR_MISSING_FLAG_summary": metrics["DENOMINATOR_MISSING_FLAG_summary"],
                "RATE_NOT_COMPUTABLE_FLAG_summary": metrics["RATE_NOT_COMPUTABLE_FLAG_summary"],
                "OFFSET_VALID_FLAG_summary": metrics["OFFSET_VALID_FLAG_summary"],
                "internal_reconciliation_summary": row.get("internal_reconciliation_summary", ""),
                "external_validation_summary": row.get("external_validation_summary", ""),
                "manifest_refreshed_yes_no": "yes" if manifest_refreshed else "no",
                "blocker_and_corrective_action_if_FAIL": (
                    ""
                    if row["status"] == "PASS"
                    else f"{row['blocking_issue']} || corrective_action={_corrective_action(row)}"
                ),
                "neonatal_deaths": summary.get("neonatal_deaths", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "infant_deaths": summary.get("infant_deaths", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "under5_deaths": summary.get("under5_deaths", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "municipalities_with_at_least_one_neonatal_death": summary.get(
                    "municipalities_with_at_least_one_neonatal_death",
                    "N/A_OUTPUT_NOT_MATERIALIZED",
                ),
                "municipalities_with_at_least_one_infant_death": summary.get(
                    "municipalities_with_at_least_one_infant_death",
                    "N/A_OUTPUT_NOT_MATERIALIZED",
                ),
                "municipalities_with_at_least_one_under5_death": summary.get(
                    "municipalities_with_at_least_one_under5_death",
                    "N/A_OUTPUT_NOT_MATERIALIZED",
                ),
                "crude_neonatal_rate": summary.get("crude_neonatal_rate", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "crude_infant_rate": summary.get("crude_infant_rate", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "crude_under5_rate": summary.get("crude_under5_rate", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "denominator_completeness": json.dumps(summary.get("denominator_completeness", {}), sort_keys=True),
                "cause_group_family_counts": json.dumps(cause_counts, sort_keys=True),
                "x30_flag_count": cause_counts.get("ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_TOTAL", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "age_reconciliation_results": json.dumps(summary.get("age_reconciliation", validations), sort_keys=True),
            }
        )
    return report_rows


def _write_mortality_stage_outputs(
    root: Path,
    rows: list[dict[str, Any]],
    manifest_paths: list[Path],
    silver_release: dict[str, Any],
    perinatal_stage: dict[str, Any],
) -> dict[str, Any]:
    status_dir = root / "artifacts" / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    report_path = _status_output_path(root, MORTALITY_STAGE_REPORT_NAME)
    matrix_path = _status_output_path(root, MORTALITY_STAGE_MATRIX_NAME)
    ledger_path = _status_output_path(root, MORTALITY_STAGE_LEDGER_NAME)
    blockers_path = _status_output_path(root, MORTALITY_STAGE_BLOCKERS_NAME)
    mini_report_path = _status_output_path(root, MORTALITY_STAGE_MINI_REPORT_NAME)
    stage_manifest_path = _mortality_stage_manifest_path(root)

    matrix_columns = [
        "stage",
        "table",
        "year",
        "status",
        "stage_order_ok",
        "silver_release_ok",
        "input_files_ok",
        "required_columns_ok",
        "dictionary_ok",
        "climate_ready_ok",
        "denominator_ready_ok",
        "materialized",
        "blocking_issue",
    ]
    matrix_frame = pd.DataFrame(rows)[matrix_columns].sort_values(["table", "year"], kind="mergesort").reset_index(drop=True)
    matrix_frame.to_csv(matrix_path, index=False, quoting=csv.QUOTE_MINIMAL)

    ledger_frame = matrix_frame[["stage", "table", "year", "status", "blocking_issue"]].copy()
    ledger_frame["decision"] = ledger_frame["status"].map(lambda status: "materialized" if status == "PASS" else "promotion_blocked")
    ledger_frame = ledger_frame[["stage", "table", "year", "status", "decision", "blocking_issue"]]
    ledger_frame.to_csv(ledger_path, index=False, quoting=csv.QUOTE_MINIMAL)

    mini_report_frame = pd.DataFrame(_mortality_stage_mini_report_rows(root, rows, manifest_refreshed=True))
    mini_report_frame = mini_report_frame.sort_values(["table", "year"], kind="mergesort").reset_index(drop=True)
    mini_report_frame.to_csv(mini_report_path, index=False, quoting=csv.QUOTE_MINIMAL)

    blocking_rows = [row for row in rows if row["status"] != "PASS"]
    unique_blockers = list(dict.fromkeys(row["blocking_issue"] for row in blocking_rows if row["blocking_issue"]))
    overall_status = "PASS" if not blocking_rows else "FAIL"

    blocker_lines = [
        "# Gold Stage 2 Mortality Blockers",
        "",
        f"- Generated at: {utc_now()}",
        f"- Overall status: `{overall_status}`",
        "",
    ]
    if not unique_blockers:
        blocker_lines.append("- None")
    else:
        blocker_lines.extend(f"- {issue}" for issue in unique_blockers)
    blockers_path.write_text("\n".join(blocker_lines) + "\n", encoding="utf-8")

    pass_years = [row["year"] for row in rows if row["status"] == "PASS"]
    fail_years = [row["year"] for row in rows if row["status"] != "PASS"]
    report_lines = [
        "# Gold Stage 2 Mortality Report",
        "",
        f"- Generated at: {utc_now()}",
        f"- Overall status: `{overall_status}`",
        f"- Silver release gate: `{silver_release['status']}`",
        f"- Gold Stage 1 Perinatal gate: `{perinatal_stage['status']}`",
        "- Stage executed: `gold_mnch_mortality` only",
        "- No downstream gold stage was materialized in this run.",
        f"- PASS years: `{pass_years}`",
        f"- FAIL years: `{fail_years}`",
        f"- Mini-report: `{mini_report_path.relative_to(root)}`",
        f"- Manifest package: `{MORTALITY_STAGE_PACKAGE_DIRNAME}`",
        "",
        "## Decision",
        "",
        "- Stage 2 built only the canonical mortality gold mart.",
        "- Panel and derived analytical views were intentionally not advanced in this run.",
        "- No model formulas were embedded and no benchmark modeling was executed in this stage.",
        "",
        "## Blockers",
        "",
    ]
    if not unique_blockers:
        report_lines.append("- None")
    else:
        report_lines.extend(f"- {issue}" for issue in unique_blockers)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    stage_manifest = {
        "generated_at": utc_now(),
        "status": overall_status,
        "stage": "gold_stage2_mortality",
        "years": sorted({int(row["year"]) for row in rows}),
        "report_path": str(report_path.relative_to(root)),
        "matrix_path": str(matrix_path.relative_to(root)),
        "ledger_path": str(ledger_path.relative_to(root)),
        "blockers_path": str(blockers_path.relative_to(root)),
        "mini_report_path": str(mini_report_path.relative_to(root)),
        "manifest_package_dir": MORTALITY_STAGE_PACKAGE_DIRNAME,
        "silver_release_status": silver_release["status"],
        "perinatal_stage_status": perinatal_stage["status"],
        "blocking_issue_count": len(unique_blockers),
        "artifacts": [
            {"path": str(report_path.relative_to(root)), "sha256": sha256_file(report_path), "size_bytes": report_path.stat().st_size},
            {"path": str(matrix_path.relative_to(root)), "sha256": sha256_file(matrix_path), "size_bytes": matrix_path.stat().st_size},
            {"path": str(ledger_path.relative_to(root)), "sha256": sha256_file(ledger_path), "size_bytes": ledger_path.stat().st_size},
            {"path": str(mini_report_path.relative_to(root)), "sha256": sha256_file(mini_report_path), "size_bytes": mini_report_path.stat().st_size},
            {"path": str(blockers_path.relative_to(root)), "sha256": sha256_file(blockers_path), "size_bytes": blockers_path.stat().st_size},
        ],
    }
    stage_manifest_path.write_text(json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    package_dir = _mortality_stage_package_dir(root)
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    copied_entries: list[dict[str, Any]] = []
    for source_path in [*manifest_paths, stage_manifest_path]:
        destination = package_dir / source_path.name
        shutil.copy2(source_path, destination)
        copied_entries.append(
            {
                "filename": destination.name,
                "source_path": str(source_path.relative_to(root)),
                "package_path": str(destination.relative_to(root)),
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    index_path = package_dir / "manifest_index.json"
    index_path.write_text(
        json.dumps(
            {
                "generated_at": utc_now(),
                "entry_count": len(copied_entries),
                "entries": copied_entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    copied_report = _copy_output_to_root(root, report_path)
    copied_matrix = _copy_output_to_root(root, matrix_path)
    copied_ledger = _copy_output_to_root(root, ledger_path)
    copied_mini_report = _copy_output_to_root(root, mini_report_path)
    copied_blockers = _copy_output_to_root(root, blockers_path)
    return {
        "status": overall_status,
        "report_path": str(copied_report.relative_to(root)),
        "matrix_path": str(copied_matrix.relative_to(root)),
        "ledger_path": str(copied_ledger.relative_to(root)),
        "mini_report_path": str(copied_mini_report.relative_to(root)),
        "blockers_path": str(copied_blockers.relative_to(root)),
        "manifest_package_dir": str(package_dir.relative_to(root)),
        "stage_manifest_path": str(stage_manifest_path.relative_to(root)),
        "blocking_issue_count": len(unique_blockers),
    }


def _panel_stage_manifest_path(root: Path) -> Path:
    return root / "artifacts" / "manifests" / PANEL_STAGE_MANIFEST_NAME


def _panel_stage_package_dir(root: Path) -> Path:
    return root / PANEL_STAGE_PACKAGE_DIRNAME


def _panel_stage_mini_report_rows(root: Path, rows: list[dict[str, Any]], manifest_refreshed: bool) -> list[dict[str, Any]]:
    report_rows: list[dict[str, Any]] = []
    for row in rows:
        output_path = _gold_output_path(root, "gold_mnch_panel", int(row["year"]))
        metrics = _output_metrics(output_path) if row.get("materialized") else _output_metrics(Path("__not_materialized__.parquet"))
        input_paths = _input_paths_for_row(root, "gold_mnch_panel", int(row["year"]))
        summary = row.get("panel_summary", {}) if isinstance(row.get("panel_summary"), dict) else {}
        validations = row.get("panel_validation_flags", {}) if isinstance(row.get("panel_validation_flags"), dict) else {}
        crude_rate_by_endpoint = summary.get("crude_rate_by_endpoint", {}) if isinstance(summary.get("crude_rate_by_endpoint"), dict) else {}
        report_rows.append(
            {
                "table": row["table"],
                "year": int(row["year"]),
                "STATUS": row["status"],
                "input_silver_paths": "; ".join(str(path.relative_to(root)) for path in input_paths),
                "output_gold_path": str(output_path.relative_to(root)),
                "rows": metrics["rows"],
                "unique_municipalities": metrics["unique_municipalities"],
                "min_date": metrics["min_date"],
                "max_date": metrics["max_date"],
                "day_completeness": metrics["day_completeness"],
                "month_completeness": metrics["month_completeness"],
                "year_completeness": metrics["year_completeness"],
                "municipalities_with_at_least_one_event": metrics["municipalities_with_at_least_one_event"],
                "total_events": metrics["total_events"],
                "denominator_used": _denominator_used("gold_mnch_panel"),
                "crude_rate_when_applicable": (
                    json.dumps(crude_rate_by_endpoint, sort_keys=True) if crude_rate_by_endpoint else metrics["crude_rate_when_applicable"]
                ),
                "climate_linkage_completeness": metrics["climate_linkage_completeness"],
                "missingness_in_critical_fields": metrics["missingness_in_critical_fields"],
                "DENOMINATOR_ZERO_FLAG_summary": metrics["DENOMINATOR_ZERO_FLAG_summary"],
                "DENOMINATOR_MISSING_FLAG_summary": metrics["DENOMINATOR_MISSING_FLAG_summary"],
                "RATE_NOT_COMPUTABLE_FLAG_summary": metrics["RATE_NOT_COMPUTABLE_FLAG_summary"],
                "OFFSET_VALID_FLAG_summary": metrics["OFFSET_VALID_FLAG_summary"],
                "internal_reconciliation_summary": row.get("internal_reconciliation_summary", ""),
                "external_validation_summary": row.get("external_validation_summary", ""),
                "manifest_refreshed_yes_no": "yes" if manifest_refreshed else "no",
                "blocker_and_corrective_action_if_FAIL": (
                    ""
                    if row["status"] == "PASS"
                    else f"{row['blocking_issue']} || corrective_action={_corrective_action(row)}"
                ),
                "total_municipality_days": summary.get("total_municipality_days", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "municipalities_with_full_daily_coverage": summary.get(
                    "municipalities_with_full_daily_coverage",
                    "N/A_OUTPUT_NOT_MATERIALIZED",
                ),
                "municipalities_with_at_least_one_event_by_endpoint": json.dumps(
                    summary.get("municipalities_with_event_by_endpoint", {}),
                    sort_keys=True,
                ),
                "total_events_by_endpoint": json.dumps(summary.get("total_events_by_endpoint", {}), sort_keys=True),
                "crude_rate_by_endpoint": json.dumps(crude_rate_by_endpoint, sort_keys=True),
                "DENOMINATOR_ZERO_FLAG_count": summary.get("DENOMINATOR_ZERO_FLAG_count", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "DENOMINATOR_MISSING_FLAG_count": summary.get("DENOMINATOR_MISSING_FLAG_count", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "RATE_NOT_COMPUTABLE_FLAG_count": summary.get("RATE_NOT_COMPUTABLE_FLAG_count", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "OFFSET_VALID_FLAG_count": summary.get("OFFSET_VALID_FLAG_count", "N/A_OUTPUT_NOT_MATERIALIZED"),
                "zero_missing_confirmation_required_climate_linkage_fields": json.dumps(
                    summary.get("zero_missing_confirmation_required_climate_linkage_fields", {}),
                    sort_keys=True,
                ),
                "climate_required_field_missing_counts": json.dumps(
                    summary.get("climate_required_field_missing_counts", {}),
                    sort_keys=True,
                ),
                "panel_validation_results": json.dumps(validations, sort_keys=True),
            }
        )
    return report_rows


def _write_panel_stage_outputs(
    root: Path,
    rows: list[dict[str, Any]],
    manifest_paths: list[Path],
    silver_release: dict[str, Any],
    perinatal_stage: dict[str, Any],
    mortality_stage: dict[str, Any],
) -> dict[str, Any]:
    status_dir = root / "artifacts" / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    report_path = _status_output_path(root, PANEL_STAGE_REPORT_NAME)
    matrix_path = _status_output_path(root, PANEL_STAGE_MATRIX_NAME)
    ledger_path = _status_output_path(root, PANEL_STAGE_LEDGER_NAME)
    blockers_path = _status_output_path(root, PANEL_STAGE_BLOCKERS_NAME)
    mini_report_path = _status_output_path(root, PANEL_STAGE_MINI_REPORT_NAME)
    stage_manifest_path = _panel_stage_manifest_path(root)

    matrix_columns = [
        "stage",
        "table",
        "year",
        "status",
        "stage_order_ok",
        "silver_release_ok",
        "input_files_ok",
        "required_columns_ok",
        "dictionary_ok",
        "climate_ready_ok",
        "denominator_ready_ok",
        "materialized",
        "blocking_issue",
    ]
    matrix_frame = pd.DataFrame(rows)[matrix_columns].sort_values(["table", "year"], kind="mergesort").reset_index(drop=True)
    matrix_frame.to_csv(matrix_path, index=False, quoting=csv.QUOTE_MINIMAL)

    ledger_frame = matrix_frame[["stage", "table", "year", "status", "blocking_issue"]].copy()
    ledger_frame["decision"] = ledger_frame["status"].map(lambda status: "materialized" if status == "PASS" else "promotion_blocked")
    ledger_frame = ledger_frame[["stage", "table", "year", "status", "decision", "blocking_issue"]]
    ledger_frame.to_csv(ledger_path, index=False, quoting=csv.QUOTE_MINIMAL)

    mini_report_frame = pd.DataFrame(_panel_stage_mini_report_rows(root, rows, manifest_refreshed=True))
    mini_report_frame = mini_report_frame.sort_values(["table", "year"], kind="mergesort").reset_index(drop=True)
    mini_report_frame.to_csv(mini_report_path, index=False, quoting=csv.QUOTE_MINIMAL)

    blocking_rows = [row for row in rows if row["status"] != "PASS"]
    unique_blockers = list(dict.fromkeys(row["blocking_issue"] for row in blocking_rows if row["blocking_issue"]))
    overall_status = "PASS" if not blocking_rows else "FAIL"

    blocker_lines = [
        "# Gold Stage 3 Panel Blockers",
        "",
        f"- Generated at: {utc_now()}",
        f"- Overall status: `{overall_status}`",
        "",
    ]
    if not unique_blockers:
        blocker_lines.append("- None")
    else:
        blocker_lines.extend(f"- {issue}" for issue in unique_blockers)
    blockers_path.write_text("\n".join(blocker_lines) + "\n", encoding="utf-8")

    pass_years = [row["year"] for row in rows if row["status"] == "PASS"]
    fail_years = [row["year"] for row in rows if row["status"] != "PASS"]
    report_lines = [
        "# Gold Stage 3 Panel Report",
        "",
        f"- Generated at: {utc_now()}",
        f"- Overall status: `{overall_status}`",
        f"- Silver release gate: `{silver_release['status']}`",
        f"- Gold Stage 1 Perinatal gate: `{perinatal_stage['status']}`",
        f"- Gold Stage 2 Mortality gate: `{mortality_stage['status']}`",
        "- Stage executed: `gold_mnch_panel` only",
        "- No downstream gold stage was materialized in this run.",
        f"- PASS years: `{pass_years}`",
        f"- FAIL years: `{fail_years}`",
        f"- Mini-report: `{mini_report_path.relative_to(root)}`",
        f"- Manifest package: `{PANEL_STAGE_PACKAGE_DIRNAME}`",
        "",
        "## Decision",
        "",
        "- Stage 3 built only the harmonized municipality-day canonical panel.",
        "- Derived analytical views were intentionally not advanced in this run.",
        "- No model formulas were embedded and no benchmark modeling was executed in this stage.",
        "",
        "## Blockers",
        "",
    ]
    if not unique_blockers:
        report_lines.append("- None")
    else:
        report_lines.extend(f"- {issue}" for issue in unique_blockers)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    stage_manifest = {
        "generated_at": utc_now(),
        "status": overall_status,
        "stage": "gold_stage3_panel",
        "years": sorted({int(row["year"]) for row in rows}),
        "report_path": str(report_path.relative_to(root)),
        "matrix_path": str(matrix_path.relative_to(root)),
        "ledger_path": str(ledger_path.relative_to(root)),
        "blockers_path": str(blockers_path.relative_to(root)),
        "mini_report_path": str(mini_report_path.relative_to(root)),
        "manifest_package_dir": PANEL_STAGE_PACKAGE_DIRNAME,
        "silver_release_status": silver_release["status"],
        "perinatal_stage_status": perinatal_stage["status"],
        "mortality_stage_status": mortality_stage["status"],
        "blocking_issue_count": len(unique_blockers),
        "artifacts": [
            {"path": str(report_path.relative_to(root)), "sha256": sha256_file(report_path), "size_bytes": report_path.stat().st_size},
            {"path": str(matrix_path.relative_to(root)), "sha256": sha256_file(matrix_path), "size_bytes": matrix_path.stat().st_size},
            {"path": str(ledger_path.relative_to(root)), "sha256": sha256_file(ledger_path), "size_bytes": ledger_path.stat().st_size},
            {"path": str(mini_report_path.relative_to(root)), "sha256": sha256_file(mini_report_path), "size_bytes": mini_report_path.stat().st_size},
            {"path": str(blockers_path.relative_to(root)), "sha256": sha256_file(blockers_path), "size_bytes": blockers_path.stat().st_size},
        ],
    }
    stage_manifest_path.write_text(json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    package_dir = _panel_stage_package_dir(root)
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    copied_entries: list[dict[str, Any]] = []
    for source_path in [*manifest_paths, stage_manifest_path]:
        destination = package_dir / source_path.name
        shutil.copy2(source_path, destination)
        copied_entries.append(
            {
                "filename": destination.name,
                "source_path": str(source_path.relative_to(root)),
                "package_path": str(destination.relative_to(root)),
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    index_path = package_dir / "manifest_index.json"
    index_path.write_text(
        json.dumps({"generated_at": utc_now(), "entry_count": len(copied_entries), "entries": copied_entries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    copied_report = _copy_output_to_root(root, report_path)
    copied_matrix = _copy_output_to_root(root, matrix_path)
    copied_ledger = _copy_output_to_root(root, ledger_path)
    copied_mini_report = _copy_output_to_root(root, mini_report_path)
    copied_blockers = _copy_output_to_root(root, blockers_path)
    return {
        "status": overall_status,
        "report_path": str(copied_report.relative_to(root)),
        "matrix_path": str(copied_matrix.relative_to(root)),
        "ledger_path": str(copied_ledger.relative_to(root)),
        "mini_report_path": str(copied_mini_report.relative_to(root)),
        "blockers_path": str(copied_blockers.relative_to(root)),
        "manifest_package_dir": str(package_dir.relative_to(root)),
        "stage_manifest_path": str(stage_manifest_path.relative_to(root)),
        "blocking_issue_count": len(unique_blockers),
    }


def promote_gold_panel_range(
    root: Path,
    runtime_root: Path,
    start_year: int,
    end_year: int,
    log_path: Path,
    command: str,
    force: bool = False,
) -> dict[str, Any]:
    del force
    silver_release = _approved_silver_release(root)
    perinatal_stage = _approved_stage1_perinatal(root)
    mortality_stage = _approved_stage2_mortality(root)

    rows: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        readiness = _evaluate_panel_year(root, year, silver_release["ok"], perinatal_stage["ok"], mortality_stage["ok"])
        if readiness["status"] == "PASS":
            try:
                result = build_gold_panel_year(root, runtime_root, year)
                payload = _materialized_row(readiness, result)
                payload["panel_summary"] = result.get("panel_summary", {})
                payload["panel_validation_flags"] = result.get("panel_validation_flags", {})
            except Exception as exc:  # pragma: no cover - exercised in stage runs
                payload = _failed_materialization_row(readiness, f"gold_mnch_panel stage 3 materialization failed: {exc}")
        else:
            payload = readiness
        rows.append(payload)
        append_run_event(
            log_path,
            phase="phase_4_gold_stage3_panel",
            dataset=payload["table"],
            year=year,
            command=command,
            decision="gold stage 3 panel blocked" if payload["status"] != "PASS" else "gold stage 3 panel written",
            status=payload["status"],
        )

    manifest_paths = _refresh_panel_stage_manifests(root)
    outputs = _write_panel_stage_outputs(root, rows, manifest_paths, silver_release, perinatal_stage, mortality_stage)
    append_run_event(
        log_path,
        phase="phase_4_gold_stage3_panel",
        dataset="gold_stage3_panel",
        year=None,
        command=command,
        decision=(
            f"gold stage 3 panel completed; status={outputs['status']} "
            f"report={outputs['report_path']} matrix={outputs['matrix_path']}"
        ),
        status=outputs["status"],
    )
    return {
        "status": outputs["status"],
        "silver_release": silver_release,
        "perinatal_stage": perinatal_stage,
        "mortality_stage": mortality_stage,
        "years": list(range(start_year, end_year + 1)),
        "rows": rows,
        "outputs": outputs,
    }


def _derived_stage_manifest_path(root: Path) -> Path:
    return root / "artifacts" / "manifests" / DERIVED_STAGE_MANIFEST_NAME


def _derived_stage_package_dir(root: Path) -> Path:
    return root / DERIVED_STAGE_PACKAGE_DIRNAME


def _derived_stage_mini_report_rows(root: Path, rows: list[dict[str, Any]], manifest_refreshed: bool) -> list[dict[str, Any]]:
    report_rows: list[dict[str, Any]] = []
    for row in rows:
        output_path = _gold_output_path(root, str(row["table"]), int(row["year"]))
        metrics = _output_metrics(output_path) if row.get("materialized") else _output_metrics(Path("__not_materialized__.parquet"))
        input_paths = _input_paths_for_row(root, str(row["table"]), int(row["year"]))
        summary = row.get("derived_summary", {}) if isinstance(row.get("derived_summary"), dict) else {}
        validations = row.get("derived_validation_flags", {}) if isinstance(row.get("derived_validation_flags"), dict) else {}
        report_rows.append(
            {
                "view": row["table"],
                "year": int(row["year"]),
                "STATUS": row["status"],
                "input_gold_paths": "; ".join(str(path.relative_to(root)) for path in input_paths),
                "output_view_path": str(output_path.relative_to(root)),
                "rows": metrics["rows"],
                "municipalities_covered": metrics["unique_municipalities"],
                "min_date": metrics["min_date"],
                "max_date": metrics["max_date"],
                "day_completeness": metrics["day_completeness"],
                "month_completeness": metrics["month_completeness"],
                "year_completeness": metrics["year_completeness"],
                "events_by_view_and_endpoint": json.dumps(summary.get("events_by_endpoint", {}), sort_keys=True),
                "reconciliation_against_canonical_gold_totals": json.dumps(
                    summary.get("reconciliation_against_canonical_gold_totals", validations),
                    sort_keys=True,
                ),
                "internal_reconciliation_summary": row.get("internal_reconciliation_summary", ""),
                "external_validation_summary": row.get("external_validation_summary", ""),
                "manifest_refreshed_yes_no": "yes" if manifest_refreshed else "no",
                "blocker_and_corrective_action_if_FAIL": (
                    ""
                    if row["status"] == "PASS"
                    else f"{row['blocking_issue']} || corrective_action={_corrective_action(row)}"
                ),
            }
        )
    return report_rows


def _write_derived_stage_outputs(
    root: Path,
    rows: list[dict[str, Any]],
    manifest_paths: list[Path],
    silver_release: dict[str, Any],
    panel_stage: dict[str, Any],
) -> dict[str, Any]:
    status_dir = root / "artifacts" / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    report_path = _status_output_path(root, DERIVED_STAGE_REPORT_NAME)
    matrix_path = _status_output_path(root, DERIVED_STAGE_MATRIX_NAME)
    ledger_path = _status_output_path(root, DERIVED_STAGE_LEDGER_NAME)
    blockers_path = _status_output_path(root, DERIVED_STAGE_BLOCKERS_NAME)
    mini_report_path = _status_output_path(root, DERIVED_STAGE_MINI_REPORT_NAME)
    stage_manifest_path = _derived_stage_manifest_path(root)

    matrix_columns = [
        "stage",
        "table",
        "year",
        "status",
        "stage_order_ok",
        "silver_release_ok",
        "input_files_ok",
        "required_columns_ok",
        "dictionary_ok",
        "climate_ready_ok",
        "denominator_ready_ok",
        "materialized",
        "blocking_issue",
    ]
    matrix_frame = pd.DataFrame(rows)[matrix_columns].sort_values(["table", "year"], kind="mergesort").reset_index(drop=True)
    matrix_frame.to_csv(matrix_path, index=False, quoting=csv.QUOTE_MINIMAL)

    ledger_frame = matrix_frame[["stage", "table", "year", "status", "blocking_issue"]].copy()
    ledger_frame["decision"] = ledger_frame["status"].map(lambda status: "materialized" if status == "PASS" else "promotion_blocked")
    ledger_frame = ledger_frame[["stage", "table", "year", "status", "decision", "blocking_issue"]]
    ledger_frame.to_csv(ledger_path, index=False, quoting=csv.QUOTE_MINIMAL)

    mini_report_frame = pd.DataFrame(_derived_stage_mini_report_rows(root, rows, manifest_refreshed=True))
    mini_report_frame = mini_report_frame.sort_values(["view", "year"], kind="mergesort").reset_index(drop=True)
    mini_report_frame.to_csv(mini_report_path, index=False, quoting=csv.QUOTE_MINIMAL)

    blocking_rows = [row for row in rows if row["status"] != "PASS"]
    unique_blockers = list(dict.fromkeys(row["blocking_issue"] for row in blocking_rows if row["blocking_issue"]))
    overall_status = "PASS" if not blocking_rows else "FAIL"

    blocker_lines = [
        "# Gold Stage 4 Derived View Blockers",
        "",
        f"- Generated at: {utc_now()}",
        f"- Overall status: `{overall_status}`",
        "",
    ]
    if not unique_blockers:
        blocker_lines.append("- None")
    else:
        blocker_lines.extend(f"- {issue}" for issue in unique_blockers)
    blockers_path.write_text("\n".join(blocker_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# Gold Stage 4 Derived View Report",
        "",
        f"- Generated at: {utc_now()}",
        f"- Overall status: `{overall_status}`",
        f"- Silver release gate: `{silver_release['status']}`",
        f"- Gold Stage 3 Panel gate: `{panel_stage['status']}`",
        "- Stage executed: derived analytical views only",
        "- Canonical gold tables were not modified in this run.",
        f"- Mini-report: `{mini_report_path.relative_to(root)}`",
        f"- Manifest package: `{DERIVED_STAGE_PACKAGE_DIRNAME}`",
        "",
        "## Decision",
        "",
        "- Stage 4 built only the analytical derived views from approved canonical gold inputs.",
        "- Canonical gold perinatal, mortality, and panel tables were left unchanged.",
        "- No model formulas were embedded and no benchmark modeling was executed in this stage.",
        "",
        "## Blockers",
        "",
    ]
    if not unique_blockers:
        report_lines.append("- None")
    else:
        report_lines.extend(f"- {issue}" for issue in unique_blockers)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    stage_manifest = {
        "generated_at": utc_now(),
        "status": overall_status,
        "stage": "gold_stage4_derived",
        "years": sorted({int(row["year"]) for row in rows}),
        "views": sorted({str(row["table"]) for row in rows}),
        "report_path": str(report_path.relative_to(root)),
        "matrix_path": str(matrix_path.relative_to(root)),
        "ledger_path": str(ledger_path.relative_to(root)),
        "blockers_path": str(blockers_path.relative_to(root)),
        "mini_report_path": str(mini_report_path.relative_to(root)),
        "manifest_package_dir": DERIVED_STAGE_PACKAGE_DIRNAME,
        "silver_release_status": silver_release["status"],
        "panel_stage_status": panel_stage["status"],
        "blocking_issue_count": len(unique_blockers),
        "artifacts": [
            {"path": str(report_path.relative_to(root)), "sha256": sha256_file(report_path), "size_bytes": report_path.stat().st_size},
            {"path": str(matrix_path.relative_to(root)), "sha256": sha256_file(matrix_path), "size_bytes": matrix_path.stat().st_size},
            {"path": str(ledger_path.relative_to(root)), "sha256": sha256_file(ledger_path), "size_bytes": ledger_path.stat().st_size},
            {"path": str(mini_report_path.relative_to(root)), "sha256": sha256_file(mini_report_path), "size_bytes": mini_report_path.stat().st_size},
            {"path": str(blockers_path.relative_to(root)), "sha256": sha256_file(blockers_path), "size_bytes": blockers_path.stat().st_size},
        ],
    }
    stage_manifest_path.write_text(json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    package_dir = _derived_stage_package_dir(root)
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    copied_entries: list[dict[str, Any]] = []
    for source_path in [*manifest_paths, stage_manifest_path]:
        destination = package_dir / source_path.name
        shutil.copy2(source_path, destination)
        copied_entries.append(
            {
                "filename": destination.name,
                "source_path": str(source_path.relative_to(root)),
                "package_path": str(destination.relative_to(root)),
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    index_path = package_dir / "manifest_index.json"
    index_path.write_text(
        json.dumps({"generated_at": utc_now(), "entry_count": len(copied_entries), "entries": copied_entries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    copied_report = _copy_output_to_root(root, report_path)
    copied_matrix = _copy_output_to_root(root, matrix_path)
    copied_ledger = _copy_output_to_root(root, ledger_path)
    copied_mini_report = _copy_output_to_root(root, mini_report_path)
    copied_blockers = _copy_output_to_root(root, blockers_path)
    return {
        "status": overall_status,
        "report_path": str(copied_report.relative_to(root)),
        "matrix_path": str(copied_matrix.relative_to(root)),
        "ledger_path": str(copied_ledger.relative_to(root)),
        "mini_report_path": str(copied_mini_report.relative_to(root)),
        "blockers_path": str(copied_blockers.relative_to(root)),
        "manifest_package_dir": str(package_dir.relative_to(root)),
        "stage_manifest_path": str(stage_manifest_path.relative_to(root)),
        "blocking_issue_count": len(unique_blockers),
    }


def promote_gold_derived_range(
    root: Path,
    runtime_root: Path,
    start_year: int,
    end_year: int,
    log_path: Path,
    command: str,
    force: bool = False,
) -> dict[str, Any]:
    del force
    silver_release = _approved_silver_release(root)
    panel_stage = _approved_stage3_panel(root)

    builders = {
        "gold_analysis_child_u5": build_gold_analysis_child_u5_year,
        "gold_analysis_wra_15_49": build_gold_analysis_wra_15_49_year,
        "gold_analysis_wra_15_45_sens": build_gold_analysis_wra_15_45_sens_year,
    }

    rows: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        for table in ("gold_analysis_child_u5", "gold_analysis_wra_15_49", "gold_analysis_wra_15_45_sens"):
            readiness = _evaluate_derived_year(root, year, table, silver_release["ok"], panel_stage["ok"])
            if readiness["status"] == "PASS":
                try:
                    result = builders[table](root, runtime_root, year)
                    payload = _materialized_row(readiness, result)
                    payload["derived_summary"] = result.get("derived_summary", {})
                    payload["derived_validation_flags"] = result.get("derived_validation_flags", {})
                except Exception as exc:  # pragma: no cover - exercised in stage runs
                    payload = _failed_materialization_row(readiness, f"{table} stage 4 materialization failed: {exc}")
            else:
                payload = readiness
            rows.append(payload)
            append_run_event(
                log_path,
                phase="phase_4_gold_stage4_derived_views",
                dataset=payload["table"],
                year=year,
                command=command,
                decision="derived analytical view blocked" if payload["status"] != "PASS" else "derived analytical view written",
                status=payload["status"],
            )

    manifest_paths = _refresh_derived_stage_manifests(root)
    outputs = _write_derived_stage_outputs(root, rows, manifest_paths, silver_release, panel_stage)
    append_run_event(
        log_path,
        phase="phase_4_gold_stage4_derived_views",
        dataset="gold_stage4_derived",
        year=None,
        command=command,
        decision=(
            f"gold stage 4 derived views completed; status={outputs['status']} "
            f"report={outputs['report_path']} matrix={outputs['matrix_path']}"
        ),
        status=outputs["status"],
    )
    return {
        "status": outputs["status"],
        "silver_release": silver_release,
        "panel_stage": panel_stage,
        "years": list(range(start_year, end_year + 1)),
        "rows": rows,
        "outputs": outputs,
    }


def _dictionary_stage_manifest_path(root: Path) -> Path:
    return root / "artifacts" / "manifests" / DICTIONARY_STAGE_MANIFEST_NAME


def _dictionary_stage_package_dir(root: Path) -> Path:
    return root / DICTIONARY_STAGE_PACKAGE_DIRNAME


def _write_dictionary_stage_outputs(
    root: Path,
    matrix: pd.DataFrame,
    variable_dictionary: pd.DataFrame,
    readiness_ledger: pd.DataFrame,
    mini_report: pd.DataFrame,
    manifest_paths: list[Path],
    silver_release: dict[str, Any],
    perinatal_stage: dict[str, Any],
    mortality_stage: dict[str, Any],
    panel_stage: dict[str, Any],
    derived_stage: dict[str, Any],
    required_documented: bool,
    missing_required_variables: list[str],
) -> dict[str, Any]:
    status_dir = root / "artifacts" / "status"
    dictionary_dir = root / "artifacts" / "dictionaries"
    status_dir.mkdir(parents=True, exist_ok=True)
    dictionary_dir.mkdir(parents=True, exist_ok=True)

    report_path = _status_output_path(root, DICTIONARY_STAGE_REPORT_NAME)
    matrix_path = _status_output_path(root, DICTIONARY_STAGE_MATRIX_NAME)
    ledger_path = _status_output_path(root, DICTIONARY_STAGE_LEDGER_NAME)
    mini_report_path = _status_output_path(root, DICTIONARY_STAGE_MINI_REPORT_NAME)
    blockers_path = _status_output_path(root, DICTIONARY_STAGE_BLOCKERS_NAME)
    variable_dictionary_path = dictionary_dir / DICTIONARY_STAGE_VARIABLE_DICTIONARY_NAME
    stage_manifest_path = _dictionary_stage_manifest_path(root)

    matrix.to_csv(matrix_path, index=False, quoting=csv.QUOTE_MINIMAL)
    readiness_ledger.to_csv(ledger_path, index=False, quoting=csv.QUOTE_MINIMAL)
    mini_report.to_csv(mini_report_path, index=False, quoting=csv.QUOTE_MINIMAL)
    variable_dictionary.to_csv(variable_dictionary_path, index=False, quoting=csv.QUOTE_MINIMAL)

    blocked_rows = readiness_ledger.loc[readiness_ledger["status"] == "Blocked"].copy()
    partial_rows = readiness_ledger.loc[readiness_ledger["status"] == "Partially ready"].copy()
    matrix_failures = matrix.loc[matrix["status"] != "PASS"].copy()
    unique_blockers = list(
        dict.fromkeys(
            [
                *[str(value) for value in blocked_rows["blocker"].tolist() if str(value).strip()],
                *[str(value) for value in matrix_failures["blocking_issue"].tolist() if str(value).strip()],
            ]
        )
    )
    stage_order_ok = all(
        stage.get("ok", False)
        for stage in (silver_release, perinatal_stage, mortality_stage, panel_stage, derived_stage)
    )
    overall_status = "PASS" if stage_order_ok and required_documented and matrix_failures.empty and blocked_rows.empty else "FAIL"

    blocker_lines = [
        "# Gold Stage 5 Dictionary And Readiness Blockers",
        "",
        f"- Generated at: {utc_now()}",
        f"- Overall status: `{overall_status}`",
        "",
    ]
    if not unique_blockers:
        blocker_lines.append("- None")
    else:
        blocker_lines.extend(f"- {issue}" for issue in unique_blockers)
    blockers_path.write_text("\n".join(blocker_lines) + "\n", encoding="utf-8")

    ready_now_count = int(readiness_ledger["status"].eq("Ready now").sum())
    partially_ready_count = int(readiness_ledger["status"].eq("Partially ready").sum())
    blocked_count = int(readiness_ledger["status"].eq("Blocked").sum())
    partial_endpoint_names = partial_rows["endpoint_name"].astype(str).tolist()

    report_lines = [
        "# Gold Stage 5 Variable Dictionary And Readiness Report",
        "",
        f"- Generated at: {utc_now()}",
        f"- Overall status: `{overall_status}`",
        f"- Silver release gate: `{silver_release['status']}`",
        f"- Gold Stage 1 Perinatal gate: `{perinatal_stage['status']}`",
        f"- Gold Stage 2 Mortality gate: `{mortality_stage['status']}`",
        f"- Gold Stage 3 Panel gate: `{panel_stage['status']}`",
        f"- Gold Stage 4 Derived gate: `{derived_stage['status']}`",
        f"- Variable dictionary rows: `{len(variable_dictionary)}`",
        f"- Endpoint readiness rows: `{len(readiness_ledger)}`",
        f"- Ready now endpoints: `{ready_now_count}`",
        f"- Partially ready endpoints: `{partially_ready_count}`",
        f"- Blocked endpoints: `{blocked_count}`",
        f"- Required documentation coverage complete: `{required_documented}`",
        f"- Missing required documentation variables: `{missing_required_variables}`",
        f"- Variable dictionary: `{variable_dictionary_path.relative_to(root)}`",
        f"- Readiness ledger: `{ledger_path.relative_to(root)}`",
        f"- Table-year matrix: `{matrix_path.relative_to(root)}`",
        f"- Mini-report: `{mini_report_path.relative_to(root)}`",
        f"- Manifest package: `{DICTIONARY_STAGE_PACKAGE_DIRNAME}`",
        "",
        "## Decision",
        "",
        "- Stage 5 refreshed the documentation layer only; no canonical gold table or derived view was modified.",
        "- The readiness ledger certifies endpoint status from the approved gold artifacts already on disk.",
        "- Partially ready endpoints remain approved outputs but carry explicit readiness cautions in the ledger.",
        "",
        "## Partially Ready Endpoints",
        "",
    ]
    if not partial_endpoint_names:
        report_lines.append("- None")
    else:
        report_lines.extend(f"- {endpoint}" for endpoint in partial_endpoint_names)
    report_lines.extend(["", "## Blockers", ""])
    if not unique_blockers:
        report_lines.append("- None")
    else:
        report_lines.extend(f"- {issue}" for issue in unique_blockers)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    stage_manifest = {
        "generated_at": utc_now(),
        "status": overall_status,
        "stage": "gold_stage5_dictionary_readiness",
        "years": sorted({int(value) for value in matrix["year"].tolist()}),
        "tables": sorted({str(value) for value in matrix["table"].tolist()}),
        "report_path": str(report_path.relative_to(root)),
        "matrix_path": str(matrix_path.relative_to(root)),
        "ledger_path": str(ledger_path.relative_to(root)),
        "variable_dictionary_path": str(variable_dictionary_path.relative_to(root)),
        "mini_report_path": str(mini_report_path.relative_to(root)),
        "blockers_path": str(blockers_path.relative_to(root)),
        "manifest_package_dir": DICTIONARY_STAGE_PACKAGE_DIRNAME,
        "silver_release_status": silver_release["status"],
        "perinatal_stage_status": perinatal_stage["status"],
        "mortality_stage_status": mortality_stage["status"],
        "panel_stage_status": panel_stage["status"],
        "derived_stage_status": derived_stage["status"],
        "required_documented": required_documented,
        "missing_required_variables": missing_required_variables,
        "ready_now_endpoint_count": ready_now_count,
        "partially_ready_endpoint_count": partially_ready_count,
        "blocked_endpoint_count": blocked_count,
        "blocking_issue_count": len(unique_blockers),
        "artifacts": [
            {"path": str(report_path.relative_to(root)), "sha256": sha256_file(report_path), "size_bytes": report_path.stat().st_size},
            {"path": str(matrix_path.relative_to(root)), "sha256": sha256_file(matrix_path), "size_bytes": matrix_path.stat().st_size},
            {"path": str(ledger_path.relative_to(root)), "sha256": sha256_file(ledger_path), "size_bytes": ledger_path.stat().st_size},
            {
                "path": str(variable_dictionary_path.relative_to(root)),
                "sha256": sha256_file(variable_dictionary_path),
                "size_bytes": variable_dictionary_path.stat().st_size,
            },
            {"path": str(mini_report_path.relative_to(root)), "sha256": sha256_file(mini_report_path), "size_bytes": mini_report_path.stat().st_size},
            {"path": str(blockers_path.relative_to(root)), "sha256": sha256_file(blockers_path), "size_bytes": blockers_path.stat().st_size},
        ],
    }
    stage_manifest_path.write_text(json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    package_dir = _dictionary_stage_package_dir(root)
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    copied_entries: list[dict[str, Any]] = []
    for source_path in [*manifest_paths, stage_manifest_path]:
        destination = package_dir / source_path.name
        shutil.copy2(source_path, destination)
        copied_entries.append(
            {
                "filename": destination.name,
                "source_path": str(source_path.relative_to(root)),
                "package_path": str(destination.relative_to(root)),
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    index_path = package_dir / "manifest_index.json"
    index_path.write_text(
        json.dumps({"generated_at": utc_now(), "entry_count": len(copied_entries), "entries": copied_entries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    copied_report = _copy_output_to_root(root, report_path)
    copied_matrix = _copy_output_to_root(root, matrix_path)
    copied_ledger = _copy_output_to_root(root, ledger_path)
    copied_dictionary = _copy_output_to_root(root, variable_dictionary_path)
    copied_mini_report = _copy_output_to_root(root, mini_report_path)
    copied_blockers = _copy_output_to_root(root, blockers_path)
    return {
        "status": overall_status,
        "report_path": str(copied_report.relative_to(root)),
        "matrix_path": str(copied_matrix.relative_to(root)),
        "ledger_path": str(copied_ledger.relative_to(root)),
        "variable_dictionary_path": str(copied_dictionary.relative_to(root)),
        "mini_report_path": str(copied_mini_report.relative_to(root)),
        "blockers_path": str(copied_blockers.relative_to(root)),
        "manifest_package_dir": str(package_dir.relative_to(root)),
        "stage_manifest_path": str(stage_manifest_path.relative_to(root)),
        "ready_now_endpoint_count": ready_now_count,
        "partially_ready_endpoint_count": partially_ready_count,
        "blocked_endpoint_count": blocked_count,
    }


def promote_gold_dictionary_range(
    root: Path,
    runtime_root: Path,
    start_year: int,
    end_year: int,
    log_path: Path,
    command: str,
    force: bool = False,
) -> dict[str, Any]:
    del runtime_root, force
    silver_release = _approved_silver_release(root)
    perinatal_stage = _approved_stage1_perinatal(root)
    mortality_stage = _approved_stage2_mortality(root)
    panel_stage = _approved_stage3_panel(root)
    derived_stage = _approved_stage4_derived(root)

    table_summaries = build_gold_table_summaries(root, start_year, end_year)
    dictionary_result = build_gold_variable_dictionary(root, start_year, end_year)
    readiness_ledger = build_gold_readiness_ledger(root, start_year, end_year, table_summaries)
    matrix = build_stage5_table_year_matrix(root, start_year, end_year, table_summaries, dictionary_result["dictionary"])
    mini_report = build_stage5_mini_report(matrix, readiness_ledger, manifest_refreshed=True)

    for row in matrix.to_dict(orient="records"):
        append_run_event(
            log_path,
            phase="phase_4_gold_stage5_dictionary_readiness",
            dataset=str(row["table"]),
            year=int(row["year"]),
            command=command,
            decision="gold stage 5 documentation blocked" if row["status"] != "PASS" else "gold stage 5 documentation refreshed",
            status=str(row["status"]),
        )

    manifest_paths = _refresh_dictionary_stage_manifests(root)
    outputs = _write_dictionary_stage_outputs(
        root=root,
        matrix=matrix,
        variable_dictionary=dictionary_result["dictionary"],
        readiness_ledger=readiness_ledger,
        mini_report=mini_report,
        manifest_paths=manifest_paths,
        silver_release=silver_release,
        perinatal_stage=perinatal_stage,
        mortality_stage=mortality_stage,
        panel_stage=panel_stage,
        derived_stage=derived_stage,
        required_documented=bool(dictionary_result["required_documented"]),
        missing_required_variables=list(dictionary_result["missing_required_variables"]),
    )
    append_run_event(
        log_path,
        phase="phase_4_gold_stage5_dictionary_readiness",
        dataset="gold_stage5_dictionary_readiness",
        year=None,
        command=command,
        decision=(
            f"gold stage 5 dictionary and readiness completed; status={outputs['status']} "
            f"report={outputs['report_path']} ledger={outputs['ledger_path']}"
        ),
        status=outputs["status"],
    )
    return {
        "status": outputs["status"],
        "silver_release": silver_release,
        "perinatal_stage": perinatal_stage,
        "mortality_stage": mortality_stage,
        "panel_stage": panel_stage,
        "derived_stage": derived_stage,
        "table_summaries": {
            table: {
                "variable_count": int(summary["variable_count"]),
                "missing_years": list(summary["missing_years"]),
                "schema_consistent": bool(summary["schema_consistent"]),
                "climate_complete": bool(summary["climate_complete"]),
                "internal_validation_pass": bool(summary["internal_validation_pass"]),
                "external_validation_pass": bool(summary["external_validation_pass"]),
            }
            for table, summary in table_summaries.items()
        },
        "dictionary_result": {
            "row_count": int(len(dictionary_result["dictionary"])),
            "table_variable_counts": dict(dictionary_result["table_variable_counts"]),
            "required_documented": bool(dictionary_result["required_documented"]),
            "missing_required_variables": list(dictionary_result["missing_required_variables"]),
        },
        "readiness_ledger": {
            "row_count": int(len(readiness_ledger)),
            "ready_now_count": int(readiness_ledger["status"].eq("Ready now").sum()),
            "partially_ready_count": int(readiness_ledger["status"].eq("Partially ready").sum()),
            "blocked_count": int(readiness_ledger["status"].eq("Blocked").sum()),
        },
        "matrix": {
            "row_count": int(len(matrix)),
            "fail_count": int(matrix["status"].ne("PASS").sum()),
        },
        "outputs": outputs,
    }


def _release_stage_manifest_path(root: Path) -> Path:
    return root / "artifacts" / "manifests" / RELEASE_STAGE_MANIFEST_NAME


def _release_stage_package_dir(root: Path) -> Path:
    return root / RELEASE_STAGE_PACKAGE_DIRNAME


def _load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _table_release_rows(
    root: Path,
    start_year: int,
    end_year: int,
    readiness_ledger: pd.DataFrame,
    failure_notes_text: str,
) -> list[dict[str, Any]]:
    years = list(range(start_year, end_year + 1))
    stage_manifest_map = {
        "gold_mnch_perinatal": root / "artifacts" / "manifests" / PERINATAL_STAGE_MANIFEST_NAME,
        "gold_mnch_mortality": root / "artifacts" / "manifests" / MORTALITY_STAGE_MANIFEST_NAME,
        "gold_mnch_panel": root / "artifacts" / "manifests" / PANEL_STAGE_MANIFEST_NAME,
        "gold_analysis_child_u5": root / "artifacts" / "manifests" / DERIVED_STAGE_MANIFEST_NAME,
        "gold_analysis_wra_15_49": root / "artifacts" / "manifests" / DERIVED_STAGE_MANIFEST_NAME,
        "gold_analysis_wra_15_45_sens": root / "artifacts" / "manifests" / DERIVED_STAGE_MANIFEST_NAME,
    }
    mini_report_map = {
        "gold_mnch_perinatal": root / "artifacts" / "status" / PERINATAL_STAGE_MINI_REPORT_NAME,
        "gold_mnch_mortality": root / "artifacts" / "status" / MORTALITY_STAGE_MINI_REPORT_NAME,
        "gold_mnch_panel": root / "artifacts" / "status" / PANEL_STAGE_MINI_REPORT_NAME,
        "gold_analysis_child_u5": root / "artifacts" / "status" / DERIVED_STAGE_MINI_REPORT_NAME,
        "gold_analysis_wra_15_49": root / "artifacts" / "status" / DERIVED_STAGE_MINI_REPORT_NAME,
        "gold_analysis_wra_15_45_sens": root / "artifacts" / "status" / DERIVED_STAGE_MINI_REPORT_NAME,
    }
    rows: list[dict[str, Any]] = []
    for table in DOCUMENTED_GOLD_ITEMS:
        manifest_path = stage_manifest_map[table]
        gold_manifest_path = root / "artifacts" / "manifests" / f"gold_{table}.manifest.json"
        stage_manifest = _load_json_file(manifest_path) if manifest_path.exists() else {}
        mini_report_path = mini_report_map[table]
        mini_report_exists = mini_report_path.exists()
        table_matrix_rows: list[dict[str, Any]] = []
        if mini_report_exists:
            mini_report = pd.read_csv(mini_report_path)
            key_column = "table" if "table" in mini_report.columns else "view"
            table_matrix_rows = mini_report.loc[mini_report[key_column].astype(str) == table].to_dict(orient="records")
        mini_report_issued = mini_report_exists and len(table_matrix_rows) == len(years) and all(
            str(row.get("STATUS", "")).upper() == "PASS" for row in table_matrix_rows
        )
        stage_status = str(stage_manifest.get("status", "FAIL"))
        years_covered = sorted(stage_manifest.get("years", [])) if isinstance(stage_manifest.get("years", []), list) else []
        manifest_refreshed = gold_manifest_path.exists()
        readiness_rows = readiness_ledger.loc[readiness_ledger["gold_table"].astype(str) == table].copy()
        readiness_assigned = not readiness_rows.empty and readiness_rows["status"].isin({"Ready now", "Partially ready", "Blocked"}).all()
        unresolved_rows = readiness_rows.loc[readiness_rows["status"].astype(str) != "Ready now"].copy()
        unresolved_item_count = int(len(unresolved_rows))
        unresolved_documented = unresolved_item_count == 0 or all(
            endpoint in failure_notes_text for endpoint in unresolved_rows["endpoint_name"].astype(str).tolist()
        )
        output_paths_exist = all(_gold_output_path(root, table, year).exists() for year in years)
        blocking_issue = ""
        if stage_status != "PASS":
            blocking_issue = "upstream stage manifest is not PASS"
        elif not manifest_refreshed:
            blocking_issue = "gold dataset manifest missing after refresh"
        elif not mini_report_issued:
            blocking_issue = "required mini-report rows missing or not PASS"
        elif not readiness_assigned:
            blocking_issue = "endpoint readiness statuses are incomplete for this table"
        elif not unresolved_documented:
            blocking_issue = "unresolved readiness items were not documented in failure_notes.md"
        elif not output_paths_exist:
            blocking_issue = "one or more gold parquet partitions are missing for the requested release range"
        rows.append(
            {
                "component_type": "canonical_gold" if table in CANONICAL_GOLD_TABLES else "derived_gold",
                "table_name": table,
                "years_covered": ";".join(str(year) for year in years_covered),
                "status": "PASS" if not blocking_issue else "FAIL",
                "stage_manifest_path": str(manifest_path.relative_to(root)),
                "gold_manifest_path": str(gold_manifest_path.relative_to(root)) if manifest_refreshed else "",
                "mini_report_path": str(mini_report_path.relative_to(root)),
                "manifest_refreshed_yes_no": "yes" if manifest_refreshed else "no",
                "mini_report_issued_yes_no": "yes" if mini_report_issued else "no",
                "readiness_statuses_assigned_yes_no": "yes" if readiness_assigned else "no",
                "unresolved_items_documented_yes_no": "yes" if unresolved_documented else "no",
                "unresolved_item_count": unresolved_item_count,
                "ready_now_endpoint_count": int(readiness_rows["status"].eq("Ready now").sum()) if not readiness_rows.empty else 0,
                "partially_ready_endpoint_count": int(readiness_rows["status"].eq("Partially ready").sum()) if not readiness_rows.empty else 0,
                "blocked_endpoint_count": int(readiness_rows["status"].eq("Blocked").sum()) if not readiness_rows.empty else 0,
                "blocking_issue": blocking_issue,
            }
        )
    return rows


def _write_release_stage_outputs(
    root: Path,
    matrix: pd.DataFrame,
    readiness_ledger: pd.DataFrame,
    failure_notes: str,
    manifest_paths: list[Path],
    start_year: int,
    end_year: int,
    silver_release: dict[str, Any],
    perinatal_stage: dict[str, Any],
    mortality_stage: dict[str, Any],
    panel_stage: dict[str, Any],
    derived_stage: dict[str, Any],
    dictionary_stage: dict[str, Any],
) -> dict[str, Any]:
    status_dir = root / "artifacts" / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    report_path = _status_output_path(root, RELEASE_STAGE_REPORT_NAME)
    matrix_path = _status_output_path(root, RELEASE_STAGE_MATRIX_NAME)
    endpoint_ledger_path = _status_output_path(root, RELEASE_STAGE_ENDPOINT_LEDGER_NAME)
    failure_notes_path = _status_output_path(root, RELEASE_STAGE_FAILURE_NOTES_NAME)
    release_manifest_path = _stage_manifest_path(root)
    stage_manifest_path = _release_stage_manifest_path(root)

    matrix.to_csv(matrix_path, index=False, quoting=csv.QUOTE_MINIMAL)
    readiness_ledger.to_csv(endpoint_ledger_path, index=False, quoting=csv.QUOTE_MINIMAL)
    failure_notes_path.write_text(failure_notes, encoding="utf-8")

    matrix_failures = matrix.loc[matrix["status"] != "PASS"].copy()
    overall_status = "PASS" if matrix_failures.empty else "FAIL"
    unresolved_count = int(readiness_ledger["status"].ne("Ready now").sum())
    ready_now_count = int(readiness_ledger["status"].eq("Ready now").sum())
    partially_ready_count = int(readiness_ledger["status"].eq("Partially ready").sum())
    blocked_count = int(readiness_ledger["status"].eq("Blocked").sum())

    report_lines = [
        "# Gold Release Candidate Report",
        "",
        f"- Generated at: {utc_now()}",
        f"- Overall status: `{overall_status}`",
        f"- Silver release gate: `{silver_release['status']}`",
        f"- Gold Stage 1 Perinatal gate: `{perinatal_stage['status']}`",
        f"- Gold Stage 2 Mortality gate: `{mortality_stage['status']}`",
        f"- Gold Stage 3 Panel gate: `{panel_stage['status']}`",
        f"- Gold Stage 4 Derived gate: `{derived_stage['status']}`",
        f"- Gold Stage 5 Dictionary/Readiness gate: `{dictionary_stage['status']}`",
        f"- Canonical gold tables PASS: `{int(matrix['component_type'].eq('canonical_gold').sum() - matrix.loc[(matrix['component_type'] == 'canonical_gold') & (matrix['status'] != 'PASS')].shape[0])}/{len(CANONICAL_GOLD_TABLES)}`",
        f"- Derived analytical views PASS: `{int(matrix['component_type'].eq('derived_gold').sum() - matrix.loc[(matrix['component_type'] == 'derived_gold') & (matrix['status'] != 'PASS')].shape[0])}/{len(DERIVED_GOLD_VIEWS)}`",
        f"- Ready now endpoints: `{ready_now_count}`",
        f"- Partially ready endpoints: `{partially_ready_count}`",
        f"- Blocked endpoints: `{blocked_count}`",
        f"- Unresolved items documented: `{unresolved_count}`",
        f"- Release matrix: `{matrix_path.relative_to(root)}`",
        f"- Endpoint readiness ledger: `{endpoint_ledger_path.relative_to(root)}`",
        f"- Failure notes: `{failure_notes_path.relative_to(root)}`",
        f"- Manifest package: `{RELEASE_STAGE_PACKAGE_DIRNAME}`",
        "",
        "## Modeling Handoff Rules",
        "",
        "- Neonatal and infant mortality models must use annual live births.",
        "- Under-5 mortality models must use under-5 population.",
        "- Stillbirth models must use total births.",
        "- Maternal acute outcome models must use women 15-49 population.",
        "- Never log(0).",
        "- Heat Index enters as `HEAT_INDEX_APPROX_C`, as a clearly labeled approximate composite sensitivity exposure.",
        "- Keep `PRETERM_FLAG` as the primary prematurity endpoint.",
        "- Use `GEST_DAYS_LOST_BELOW_37W` only as a secondary or sensitivity outcome.",
        "- Keep event or window logic canonical for gestational and perinatal outcomes.",
        "- Do not collapse gestational outcomes into same-day-only logic in the main modeling handoff.",
        "",
        "## Decision",
        "",
        "- Stage 6 froze the approved gold release candidate and did not rematerialize canonical or derived gold outputs.",
        "- Gold and later modeling should consume only the approved release artifacts in this package.",
        "",
        "## Unresolved Items",
        "",
    ]
    if unresolved_count == 0:
        report_lines.append("- None")
    else:
        unresolved_names = readiness_ledger.loc[readiness_ledger["status"].ne("Ready now"), "endpoint_name"].astype(str).tolist()
        report_lines.extend(f"- {name}" for name in unresolved_names)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    artifact_entries = [
        {"path": str(report_path.relative_to(root)), "sha256": sha256_file(report_path), "size_bytes": report_path.stat().st_size},
        {"path": str(matrix_path.relative_to(root)), "sha256": sha256_file(matrix_path), "size_bytes": matrix_path.stat().st_size},
        {"path": str(endpoint_ledger_path.relative_to(root)), "sha256": sha256_file(endpoint_ledger_path), "size_bytes": endpoint_ledger_path.stat().st_size},
        {"path": str(failure_notes_path.relative_to(root)), "sha256": sha256_file(failure_notes_path), "size_bytes": failure_notes_path.stat().st_size},
    ]
    release_manifest = {
        "generated_at": utc_now(),
        "status": overall_status,
        "stage": "gold_release_candidate",
        "years": list(range(start_year, end_year + 1)),
        "report_path": str(report_path.relative_to(root)),
        "matrix_path": str(matrix_path.relative_to(root)),
        "endpoint_readiness_ledger_path": str(endpoint_ledger_path.relative_to(root)),
        "failure_notes_path": str(failure_notes_path.relative_to(root)),
        "manifest_package_dir": RELEASE_STAGE_PACKAGE_DIRNAME,
        "silver_release_status": silver_release["status"],
        "perinatal_stage_status": perinatal_stage["status"],
        "mortality_stage_status": mortality_stage["status"],
        "panel_stage_status": panel_stage["status"],
        "derived_stage_status": derived_stage["status"],
        "dictionary_stage_status": dictionary_stage["status"],
        "ready_now_endpoint_count": ready_now_count,
        "partially_ready_endpoint_count": partially_ready_count,
        "blocked_endpoint_count": blocked_count,
        "unresolved_item_count": unresolved_count,
        "artifacts": artifact_entries,
    }
    release_manifest_path.write_text(json.dumps(release_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stage_manifest = dict(release_manifest)
    stage_manifest["stage"] = "gold_stage6_release_candidate"
    stage_manifest_path.write_text(json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    package_dir = _release_stage_package_dir(root)
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    copied_entries: list[dict[str, Any]] = []
    for source_path in [*manifest_paths, release_manifest_path, stage_manifest_path]:
        destination = package_dir / source_path.name
        shutil.copy2(source_path, destination)
        copied_entries.append(
            {
                "filename": destination.name,
                "source_path": str(source_path.relative_to(root)),
                "package_path": str(destination.relative_to(root)),
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    index_path = package_dir / "manifest_index.json"
    index_path.write_text(
        json.dumps({"generated_at": utc_now(), "entry_count": len(copied_entries), "entries": copied_entries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    copied_report = _copy_output_to_root(root, report_path)
    copied_matrix = _copy_output_to_root(root, matrix_path)
    copied_endpoint_ledger = _copy_output_to_root(root, endpoint_ledger_path)
    copied_failure_notes = _copy_output_to_root(root, failure_notes_path)
    return {
        "status": overall_status,
        "report_path": str(copied_report.relative_to(root)),
        "matrix_path": str(copied_matrix.relative_to(root)),
        "endpoint_readiness_ledger_path": str(copied_endpoint_ledger.relative_to(root)),
        "failure_notes_path": str(copied_failure_notes.relative_to(root)),
        "manifest_package_dir": str(package_dir.relative_to(root)),
        "release_manifest_path": str(release_manifest_path.relative_to(root)),
        "stage_manifest_path": str(stage_manifest_path.relative_to(root)),
        "ready_now_endpoint_count": ready_now_count,
        "partially_ready_endpoint_count": partially_ready_count,
        "blocked_endpoint_count": blocked_count,
        "unresolved_item_count": unresolved_count,
    }


def promote_gold_release_range(
    root: Path,
    runtime_root: Path,
    start_year: int,
    end_year: int,
    log_path: Path,
    command: str,
    force: bool = False,
) -> dict[str, Any]:
    del runtime_root, force
    silver_release = _approved_silver_release(root)
    perinatal_stage = _approved_stage1_perinatal(root)
    mortality_stage = _approved_stage2_mortality(root)
    panel_stage = _approved_stage3_panel(root)
    derived_stage = _approved_stage4_derived(root)
    dictionary_stage = _approved_stage5_dictionary(root)

    readiness_ledger_path = root / DICTIONARY_STAGE_LEDGER_NAME
    if readiness_ledger_path.exists():
        readiness_ledger = pd.read_csv(readiness_ledger_path)
    else:
        readiness_ledger = pd.DataFrame(
            columns=[
                "endpoint_name",
                "status",
                "gold_table",
                "canonical_source",
                "required_fields",
                "required_fields_present_yes_no",
                "denominator_rule",
                "denominator_valid_yes_no",
                "climate_enrichment_complete_yes_no",
                "internal_validation_pass_fail",
                "external_validation_pass_fail",
                "primary_or_sensitivity",
                "age_scope",
                "blocker",
                "next_corrective_action",
            ]
        )
    unresolved = readiness_ledger.loc[readiness_ledger["status"].astype(str) != "Ready now"].copy()
    failure_lines = [
        "# Failure Notes",
        "",
        f"- Generated at: {utc_now()}",
        f"- Overall release status basis: stages 1-5 plus readiness ledger from `{DICTIONARY_STAGE_LEDGER_NAME}`",
        "",
    ]
    if unresolved.empty:
        failure_lines.append("- None")
    else:
        for row in unresolved.to_dict(orient="records"):
            failure_lines.extend(
                [
                    f"## {row['endpoint_name']}",
                    "",
                    f"- Status: {row['status']}",
                    f"- Gold table: {row['gold_table']}",
                    f"- Blocker: {row['blocker']}",
                    f"- Next corrective action: {row['next_corrective_action']}",
                    "",
                ]
            )
    failure_notes = "\n".join(failure_lines).rstrip() + "\n"
    matrix_rows = _table_release_rows(root, start_year, end_year, readiness_ledger, failure_notes)
    matrix = pd.DataFrame(matrix_rows).sort_values(["component_type", "table_name"], kind="mergesort").reset_index(drop=True)

    for row in matrix.to_dict(orient="records"):
        append_run_event(
            log_path,
            phase="phase_4_gold_stage6_release_candidate",
            dataset=str(row["table_name"]),
            year=None,
            command=command,
            decision="gold release candidate blocked" if row["status"] != "PASS" else "gold release candidate component frozen",
            status=str(row["status"]),
        )

    manifest_paths = _refresh_release_stage_manifests(root)
    outputs = _write_release_stage_outputs(
        root=root,
        matrix=matrix,
        readiness_ledger=readiness_ledger,
        failure_notes=failure_notes,
        manifest_paths=manifest_paths,
        start_year=start_year,
        end_year=end_year,
        silver_release=silver_release,
        perinatal_stage=perinatal_stage,
        mortality_stage=mortality_stage,
        panel_stage=panel_stage,
        derived_stage=derived_stage,
        dictionary_stage=dictionary_stage,
    )
    append_run_event(
        log_path,
        phase="phase_4_gold_stage6_release_candidate",
        dataset="gold_stage6_release_candidate",
        year=None,
        command=command,
        decision=(
            f"gold release candidate completed; status={outputs['status']} "
            f"report={outputs['report_path']} matrix={outputs['matrix_path']}"
        ),
        status=outputs["status"],
    )
    return {
        "status": outputs["status"],
        "silver_release": silver_release,
        "perinatal_stage": perinatal_stage,
        "mortality_stage": mortality_stage,
        "panel_stage": panel_stage,
        "derived_stage": derived_stage,
        "dictionary_stage": dictionary_stage,
        "matrix": {
            "row_count": int(len(matrix)),
            "fail_count": int(matrix["status"].ne("PASS").sum()),
        },
        "readiness_ledger": {
            "row_count": int(len(readiness_ledger)),
            "ready_now_count": int(readiness_ledger["status"].eq("Ready now").sum()),
            "partially_ready_count": int(readiness_ledger["status"].eq("Partially ready").sum()),
            "blocked_count": int(readiness_ledger["status"].eq("Blocked").sum()),
        },
        "outputs": outputs,
    }


def promote_gold_mortality_range(
    root: Path,
    runtime_root: Path,
    start_year: int,
    end_year: int,
    log_path: Path,
    command: str,
    force: bool = False,
) -> dict[str, Any]:
    del force
    silver_release = _approved_silver_release(root)
    perinatal_stage = _approved_stage1_perinatal(root)

    rows: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        readiness = _evaluate_mortality_year(root, year, silver_release["ok"], perinatal_stage["ok"])
        if readiness["status"] == "PASS":
            try:
                result = build_gold_mortality_year(root, runtime_root, year)
                payload = _materialized_row(readiness, result)
                payload["mortality_summary"] = result.get("mortality_summary", {})
                payload["mortality_validation_flags"] = result.get("mortality_validation_flags", {})
            except Exception as exc:  # pragma: no cover - exercised in stage runs
                payload = _failed_materialization_row(readiness, f"gold_mnch_mortality stage 2 materialization failed: {exc}")
        else:
            payload = readiness
        rows.append(payload)
        append_run_event(
            log_path,
            phase="phase_4_gold_stage2_mortality",
            dataset=payload["table"],
            year=year,
            command=command,
            decision="gold stage 2 mortality blocked" if payload["status"] != "PASS" else "gold stage 2 mortality written",
            status=payload["status"],
        )

    manifest_paths = _refresh_mortality_stage_manifests(root)
    outputs = _write_mortality_stage_outputs(root, rows, manifest_paths, silver_release, perinatal_stage)
    append_run_event(
        log_path,
        phase="phase_4_gold_stage2_mortality",
        dataset="gold_stage2_mortality",
        year=None,
        command=command,
        decision=(
            f"gold stage 2 mortality completed; status={outputs['status']} "
            f"report={outputs['report_path']} matrix={outputs['matrix_path']}"
        ),
        status=outputs["status"],
    )
    return {
        "status": outputs["status"],
        "silver_release": silver_release,
        "perinatal_stage": perinatal_stage,
        "years": list(range(start_year, end_year + 1)),
        "rows": rows,
        "outputs": outputs,
    }


def promote_gold_perinatal_range(
    root: Path,
    runtime_root: Path,
    start_year: int,
    end_year: int,
    log_path: Path,
    command: str,
    force: bool = False,
) -> dict[str, Any]:
    del force
    silver_release = _approved_silver_release(root)

    rows: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        readiness = _evaluate_perinatal_year(root, year, silver_release["ok"])
        if readiness["status"] == "PASS":
            try:
                result = build_gold_perinatal_year(root, runtime_root, year)
                payload = _materialized_row(readiness, result)
                payload["perinatal_summary"] = result.get("perinatal_summary", {})
                payload["perinatal_validation_flags"] = result.get("perinatal_validation_flags", {})
            except Exception as exc:  # pragma: no cover - exercised in stage runs
                payload = _failed_materialization_row(readiness, f"gold_mnch_perinatal stage 1 materialization failed: {exc}")
        else:
            payload = readiness
        rows.append(payload)
        append_run_event(
            log_path,
            phase="phase_4_gold_stage1_perinatal",
            dataset=payload["table"],
            year=year,
            command=command,
            decision="gold stage 1 perinatal blocked" if payload["status"] != "PASS" else "gold stage 1 perinatal written",
            status=payload["status"],
        )

    manifest_paths = _refresh_perinatal_stage_manifests(root)
    outputs = _write_perinatal_stage_outputs(root, rows, manifest_paths, silver_release)
    append_run_event(
        log_path,
        phase="phase_4_gold_stage1_perinatal",
        dataset="gold_stage1_perinatal",
        year=None,
        command=command,
        decision=(
            f"gold stage 1 perinatal completed; status={outputs['status']} "
            f"report={outputs['report_path']} matrix={outputs['matrix_path']}"
        ),
        status=outputs["status"],
    )
    return {
        "status": outputs["status"],
        "silver_release": silver_release,
        "years": list(range(start_year, end_year + 1)),
        "rows": rows,
        "outputs": outputs,
    }


def promote_gold_range(
    root: Path,
    runtime_root: Path,
    start_year: int,
    end_year: int,
    log_path: Path,
    command: str,
    force: bool = False,
) -> dict[str, Any]:
    del force
    silver_release = _approved_silver_release(root)

    rows: list[dict[str, Any]] = []
    perinatal_rows: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        readiness = _evaluate_perinatal_year(root, year, silver_release["ok"])
        if readiness["status"] == "PASS":
            try:
                result = build_gold_perinatal_year(root, runtime_root, year)
                payload = _materialized_row(readiness, result)
            except Exception as exc:  # pragma: no cover - exercised in promotion runs
                payload = _failed_materialization_row(readiness, f"gold_mnch_perinatal materialization failed: {exc}")
        else:
            payload = readiness
        perinatal_rows.append(payload)
        rows.append(payload)
        append_run_event(
            log_path,
            phase="phase_4_gold_promotion",
            dataset=payload["table"],
            year=year,
            command=command,
            decision="canonical gold perinatal promotion blocked" if payload["status"] != "PASS" else "canonical gold perinatal written",
            status=payload["status"],
        )

    perinatal_failed = any(row["status"] != "PASS" for row in perinatal_rows)
    if not perinatal_failed:
        build_manifest(root, "gold", "gold_mnch_perinatal")
    for year in range(start_year, end_year + 1):
        readiness = _evaluate_mortality_year(root, year, silver_release["ok"], not perinatal_failed)
        if readiness["status"] == "PASS":
            try:
                result = build_gold_mortality_year(root, runtime_root, year)
                payload = _materialized_row(readiness, result)
            except Exception as exc:  # pragma: no cover - exercised in promotion runs
                payload = _failed_materialization_row(readiness, f"gold_mnch_mortality materialization failed: {exc}")
        else:
            payload = readiness
        rows.append(payload)
        append_run_event(
            log_path,
            phase="phase_4_gold_promotion",
            dataset=payload["table"],
            year=year,
            command=command,
            decision="canonical gold mortality promotion blocked" if payload["status"] != "PASS" else "canonical gold mortality written",
            status=payload["status"],
        )

    mortality_failed = any(row["status"] != "PASS" for row in rows if row["table"] == "gold_mnch_mortality")
    if not mortality_failed and not perinatal_failed:
        build_manifest(root, "gold", "gold_mnch_mortality")

    derived_blocker = "stage-order stop: derived analytical gold views require stable canonical gold tables first"
    panel_rows: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        readiness = _evaluate_panel_year(root, year, silver_release["ok"], not perinatal_failed, not mortality_failed)
        if readiness["status"] == "PASS":
            try:
                panel_row = _materialized_row(readiness, build_gold_panel_year(root, runtime_root, year))
            except Exception as exc:  # pragma: no cover - exercised in promotion runs
                panel_row = _failed_materialization_row(readiness, f"gold_mnch_panel materialization failed: {exc}")
        else:
            panel_row = readiness
        panel_rows.append(panel_row)
        rows.append(panel_row)
        append_run_event(
            log_path,
            phase="phase_4_gold_promotion",
            dataset=panel_row["table"],
            year=year,
            command=command,
            decision="canonical gold panel promotion blocked" if panel_row["status"] != "PASS" else "canonical gold panel written",
            status=panel_row["status"],
        )
    panel_failed = any(row["status"] != "PASS" for row in panel_rows)
    if not panel_failed and not perinatal_failed and not mortality_failed:
        build_manifest(root, "gold", "gold_mnch_panel")

    for year in range(start_year, end_year + 1):
        for derived_view, builder in (
            ("gold_analysis_child_u5", build_gold_analysis_child_u5_year),
            ("gold_analysis_wra_15_49", build_gold_analysis_wra_15_49_year),
            ("gold_analysis_wra_15_45_sens", build_gold_analysis_wra_15_45_sens_year),
        ):
            readiness = _evaluate_derived_year(
                root,
                year,
                derived_view,
                silver_release["ok"],
                not perinatal_failed and not mortality_failed and not panel_failed,
            )
            if readiness["status"] == "PASS":
                try:
                    derived_row = _materialized_row(readiness, builder(root, runtime_root, year))
                except Exception as exc:  # pragma: no cover - exercised in promotion runs
                    derived_row = _failed_materialization_row(readiness, f"{derived_view} materialization failed: {exc}")
            else:
                derived_row = readiness
            rows.append(derived_row)
            append_run_event(
                log_path,
                phase="phase_4_gold_promotion",
                dataset=derived_view,
                year=year,
                command=command,
                decision="derived analytical gold view blocked" if derived_row["status"] != "PASS" else "derived analytical gold view written",
                status=derived_row["status"],
            )

    manifest_paths = _refresh_manifests(root)
    outputs = _write_outputs(root, rows, manifest_paths, silver_release)
    append_run_event(
        log_path,
        phase="phase_4_gold_promotion",
        dataset="gold_release_gate",
        year=None,
        command=command,
        decision=(
            f"gold promotion gate completed; status={outputs['status']} "
            f"report={outputs['report_path']} matrix={outputs['matrix_path']}"
        ),
        status=outputs["status"],
    )
    return {
        "status": outputs["status"],
        "silver_release": silver_release,
        "years": list(range(start_year, end_year + 1)),
        "rows": rows,
        "outputs": outputs,
    }
