from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bronze import bronze_output_path
from .manifest import build_manifest
from .run_log import append_run_event
from .silver_outcome_common import ParquetChunkWriter, official_total_from_bronze, sync_verified_parquet
from .silver_sinasc import (
    _derive_maternal_age_fields,
    _derive_plurality_fields,
    _derive_schooling_fields,
    _derive_sex_fields,
    sinasc_daily_partition_file,
    sinasc_event_partition_file,
)


STILLBIRTH_REQUIRED_INPUT_FIELDS = [
    "BRONZE_SOURCE_SYSTEM",
    "BRONZE_SOURCE_FILE",
    "BRONZE_SOURCE_PATH",
    "BRONZE_SOURCE_MEMBER",
    "BRONZE_YEAR",
    "BRONZE_INGESTED_AT",
    "TIPOBITO",
    "DTOBITO",
    "DTNASC",
    "CODMUNRES",
    "CODMUNOCOR",
    "IDADEMAE",
    "ESCMAE2010",
    "ESCMAE",
    "SERIESCMAE",
    "SEMAGESTAC",
    "GESTACAO",
    "GRAVIDEZ",
    "PARTO",
    "OBITOPARTO",
    "PESO",
    "SEXO",
    "CONTADOR",
]
VALID_GESTACAO_CODES = {"1", "2", "3", "4", "5", "6"}
EVENT_FILENAME_TEMPLATE = "silver_stillbirth_individual_{year}.parquet"
DAILY_FILENAME_TEMPLATE = "silver_stillbirth_daily_{year}.parquet"
READBACK_TOLERANCE = 1e-9

INDIVIDUAL_OUTPUT_FIELDS = [
    "STILLBIRTH_RECORD_ID",
    "SOURCE_STILLBIRTH",
    "BRONZE_SOURCE_SYSTEM",
    "BRONZE_SOURCE_FILE",
    "BRONZE_SOURCE_PATH",
    "BRONZE_SOURCE_MEMBER",
    "BRONZE_YEAR",
    "BRONZE_INGESTED_AT",
    "TIPOBITO",
    "DTOBITO",
    "DTNASC",
    "CODMUNRES",
    "CODMUNOCOR",
    "IDADEMAE",
    "ESCMAE2010",
    "ESCMAE",
    "SERIESCMAE",
    "SEMAGESTAC",
    "GESTACAO",
    "GRAVIDEZ",
    "PARTO",
    "OBITOPARTO",
    "PESO",
    "SEXO",
    "FETAL_DEATH_RECORD_FLAG",
    "EVENT_DATE_RAW",
    "BIRTH_DATE_RAW",
    "MUNICIPALITY_RESIDENCE_RAW",
    "MUNICIPALITY_OCCURRENCE_RAW",
    "GA_WEEKS_RAW",
    "GA_BAND_RAW",
    "BIRTH_WEIGHT_G_RAW",
    "DEATH_RELATION_TO_PARTO_RAW",
    "EVENT_DATE",
    "YEAR",
    "MONTH",
    "MUNICIPALITY_CODE",
    "MUNICIPALITY_OCCURRENCE_CODE",
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
    "GA_WEEKS_RAW_VALID",
    "GA_WEEKS_COMPLETED_FINAL",
    "GA_SOURCE",
    "GA_VALID_FOR_ANALYSIS_FLAG",
    "STILLBIRTH_GTE28_PRIMARY_FLAG",
    "STILLBIRTH_GTE22_SENSITIVITY_FLAG",
    "STILLBIRTH_ANALYTIC_STATUS",
    "STILLBIRTH_DERIVATION_NOTE",
    "GA_QC_FETAL_RECORD_FLAG",
    "GA_QC_DTOBITO_EQUALS_DTNASC_FLAG",
    "GA_QC_PLAUSIBLE_RANGE_FLAG",
    "GA_QC_MISSING_FLAG",
    "GA_QC_BAND_ONLY_FLAG",
    "BIRTH_WEIGHT_REPORTED_FLAG",
    "QC_STATUS",
]
DAILY_OUTPUT_FIELDS = [
    "MUNICIPALITY_CODE",
    "EVENT_DATE",
    "YEAR",
    "MONTH",
    "LIVE_BIRTHS_TOTAL",
    "STILLBIRTH_TOTAL_RAW",
    "STILLBIRTH_TOTAL_GTE28",
    "STILLBIRTH_TOTAL_GTE22_SENSITIVITY",
    "STILLBIRTH_GA_REPORTED_TOTAL",
    "STILLBIRTH_GA_MISSING_TOTAL",
    "TOTAL_BIRTHS_PRIMARY",
    "TOTAL_BIRTHS_SENSITIVITY_GTE22",
    "OFFSET_LOG_TOTAL_BIRTHS_PRIMARY",
    "OFFSET_LOG_TOTAL_BIRTHS_SENSITIVITY_GTE22",
    "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_PRIMARY",
    "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_SENSITIVITY_GTE22",
    "STILLBIRTH_TOTAL",
    "NASCIMENTOS_TOTAIS_INCLUINDO_FETAIS",
    "OFFSET_LOG_NASCIMENTOS_TOTAIS_INCLUINDO_FETAIS",
    "LIVE_BIRTHS_SOURCE",
    "SOURCE_STILLBIRTH",
    "QC_STATUS",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stillbirth_individual_partition_dir(root: Path, year: int) -> Path:
    return root / "data" / "silver" / "stillbirth" / "silver_stillbirth_individual" / f"YEAR={year}"


def stillbirth_individual_partition_file(root: Path, year: int) -> Path:
    return stillbirth_individual_partition_dir(root, year) / EVENT_FILENAME_TEMPLATE.format(year=year)


def stillbirth_daily_partition_dir(root: Path, year: int) -> Path:
    return root / "data" / "silver" / "stillbirth" / "silver_stillbirth_daily" / f"YEAR={year}"


def stillbirth_daily_partition_file(root: Path, year: int) -> Path:
    return stillbirth_daily_partition_dir(root, year) / DAILY_FILENAME_TEMPLATE.format(year=year)


def _clean_text_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str).str.strip()


def _parse_iso_date_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _clean_text_series(frame, column).replace("", pd.NA)
    return pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _clean_text_series(frame, column).replace("", pd.NA)
    return pd.to_numeric(values, errors="coerce")


def _nullable_flag(index: pd.Index, true_mask: pd.Series, false_mask: pd.Series | None = None) -> pd.Series:
    result = pd.Series(pd.NA, index=index, dtype="Int8")
    if false_mask is not None:
        result.loc[false_mask] = 0
    result.loc[true_mask] = 1
    return result


def _classify_threshold(
    exact_weeks: pd.Series,
    valid_band: pd.Series,
    threshold_weeks: int,
    band_yes: set[str],
    band_no: set[str],
) -> pd.Series:
    result = pd.Series(pd.NA, index=exact_weeks.index, dtype="Int8")
    exact_available = exact_weeks.notna()
    result.loc[exact_available & exact_weeks.ge(threshold_weeks)] = 1
    result.loc[exact_available & exact_weeks.lt(threshold_weeks)] = 0
    band_only = (~exact_available) & valid_band.notna()
    result.loc[band_only & valid_band.isin(sorted(band_yes))] = 1
    result.loc[band_only & valid_band.isin(sorted(band_no))] = 0
    return result


def load_stillbirth_bronze_year(root: Path, year: int) -> pd.DataFrame:
    bronze_path = bronze_output_path(root, "stillbirth", year)
    if not bronze_path.exists():
        raise FileNotFoundError(f"Stillbirth bronze missing for {year}: {bronze_path}")
    dataframe = pd.read_csv(
        bronze_path,
        compression="gzip",
        sep=";",
        dtype=str,
        usecols=STILLBIRTH_REQUIRED_INPUT_FIELDS,
        keep_default_na=False,
        low_memory=False,
    )
    missing_fields = [field for field in STILLBIRTH_REQUIRED_INPUT_FIELDS if field not in dataframe.columns]
    if missing_fields:
        raise ValueError(f"Stillbirth bronze missing required columns for {year}: {missing_fields}")
    return dataframe


def build_canonical_stillbirth_individual_frame(root: Path, year: int) -> pd.DataFrame:
    working = load_stillbirth_bronze_year(root, year).copy().reset_index(drop=True)
    working["STILLBIRTH_RECORD_ID"] = [
        f"STILLBIRTH_RECORD_{year}_{row_number:06d}"
        for row_number in range(1, len(working) + 1)
    ]
    working["SOURCE_STILLBIRTH"] = "silver_stillbirth_individual"

    event_date = _parse_iso_date_series(working, "DTOBITO")
    if event_date.isna().any():
        raise ValueError(f"Stillbirth bronze has invalid DTOBITO values for {year}")
    if not bool(event_date.dt.year.eq(year).all()):
        raise ValueError(f"Stillbirth bronze contains out-of-year DTOBITO values for {year}")
    municipality_code = _clean_text_series(working, "CODMUNRES")
    municipality_occurrence_code = _clean_text_series(working, "CODMUNOCOR")
    if (~municipality_code.str.fullmatch(r"\d{6}")).any():
        raise ValueError(f"Stillbirth bronze has invalid CODMUNRES values for {year}")
    if (~municipality_occurrence_code.str.fullmatch(r"\d{6}")).any():
        raise ValueError(f"Stillbirth bronze has invalid CODMUNOCOR values for {year}")

    birth_date = _parse_iso_date_series(working, "DTNASC")
    tipobito = _clean_text_series(working, "TIPOBITO")
    ga_weeks_raw_text = _clean_text_series(working, "SEMAGESTAC")
    ga_weeks_raw_numeric = _numeric_series(working, "SEMAGESTAC")
    ga_weeks_raw_valid = ga_weeks_raw_numeric.where(ga_weeks_raw_numeric.between(1, 45, inclusive="both")).astype("Float64")
    ga_band_raw = _clean_text_series(working, "GESTACAO")
    ga_band_valid = ga_band_raw.where(ga_band_raw.isin(VALID_GESTACAO_CODES))
    birth_weight_raw = _clean_text_series(working, "PESO")
    birth_weight_g = _numeric_series(working, "PESO")

    for derived_fields in [
        _derive_maternal_age_fields(working),
        _derive_schooling_fields(working),
        _derive_plurality_fields(working),
        _derive_sex_fields(
            working,
            raw_column="SEXO",
            code_field="FETAL_SEX_CODE",
            category_field="FETAL_SEX_CATEGORY",
        ),
    ]:
        for field_name, values in derived_fields.items():
            working[field_name] = values

    working["FETAL_DEATH_RECORD_FLAG"] = tipobito.eq("1").astype("Int8")
    working["EVENT_DATE_RAW"] = _clean_text_series(working, "DTOBITO")
    working["BIRTH_DATE_RAW"] = _clean_text_series(working, "DTNASC")
    working["MUNICIPALITY_RESIDENCE_RAW"] = municipality_code
    working["MUNICIPALITY_OCCURRENCE_RAW"] = municipality_occurrence_code
    working["GA_WEEKS_RAW"] = ga_weeks_raw_text
    working["GA_BAND_RAW"] = ga_band_raw
    working["BIRTH_WEIGHT_G_RAW"] = birth_weight_raw
    working["DEATH_RELATION_TO_PARTO_RAW"] = _clean_text_series(working, "OBITOPARTO")
    working["EVENT_DATE"] = event_date.dt.strftime("%Y-%m-%d")
    working["YEAR"] = event_date.dt.year.astype("int16")
    working["MONTH"] = event_date.dt.month.astype("int8")
    working["MUNICIPALITY_CODE"] = municipality_code
    working["MUNICIPALITY_OCCURRENCE_CODE"] = municipality_occurrence_code
    working["GA_WEEKS_RAW_VALID"] = ga_weeks_raw_valid
    working["GA_WEEKS_COMPLETED_FINAL"] = ga_weeks_raw_valid
    working["GA_SOURCE"] = "MISSING"
    working.loc[ga_weeks_raw_valid.notna(), "GA_SOURCE"] = "SEMAGESTAC_EXACT_WEEKS"
    working.loc[ga_weeks_raw_valid.isna() & ga_band_valid.notna(), "GA_SOURCE"] = "GESTACAO_BAND_ONLY"
    working["GA_VALID_FOR_ANALYSIS_FLAG"] = working["GA_SOURCE"].ne("MISSING").astype("Int8")
    working["STILLBIRTH_GTE28_PRIMARY_FLAG"] = _classify_threshold(
        working["GA_WEEKS_COMPLETED_FINAL"],
        ga_band_valid,
        threshold_weeks=28,
        band_yes={"3", "4", "5", "6"},
        band_no={"1", "2"},
    )
    working["STILLBIRTH_GTE22_SENSITIVITY_FLAG"] = _classify_threshold(
        working["GA_WEEKS_COMPLETED_FINAL"],
        ga_band_valid,
        threshold_weeks=22,
        band_yes={"2", "3", "4", "5", "6"},
        band_no={"1"},
    )
    working["STILLBIRTH_ANALYTIC_STATUS"] = "UNKNOWN_GESTATIONAL_AGE"
    working.loc[working["STILLBIRTH_GTE22_SENSITIVITY_FLAG"].eq(0), "STILLBIRTH_ANALYTIC_STATUS"] = "BELOW_22_WEEKS"
    working.loc[
        working["STILLBIRTH_GTE28_PRIMARY_FLAG"].eq(0) & working["STILLBIRTH_GTE22_SENSITIVITY_FLAG"].eq(1),
        "STILLBIRTH_ANALYTIC_STATUS",
    ] = "SENSITIVITY_GTE22_ONLY"
    working.loc[working["STILLBIRTH_GTE28_PRIMARY_FLAG"].eq(1), "STILLBIRTH_ANALYTIC_STATUS"] = "PRIMARY_GTE28"
    working["STILLBIRTH_DERIVATION_NOTE"] = "MISSING_GESTATIONAL_AGE"
    working.loc[working["GA_SOURCE"].eq("GESTACAO_BAND_ONLY"), "STILLBIRTH_DERIVATION_NOTE"] = "GESTACAO_BAND_ONLY"
    working.loc[working["GA_SOURCE"].eq("SEMAGESTAC_EXACT_WEEKS"), "STILLBIRTH_DERIVATION_NOTE"] = "EXACT_WEEKS"

    working["GA_QC_FETAL_RECORD_FLAG"] = working["FETAL_DEATH_RECORD_FLAG"].astype("Int8")
    working["GA_QC_DTOBITO_EQUALS_DTNASC_FLAG"] = _nullable_flag(
        working.index,
        event_date.notna() & birth_date.notna() & event_date.eq(birth_date),
        event_date.notna() & birth_date.notna() & event_date.ne(birth_date),
    )
    working["GA_QC_PLAUSIBLE_RANGE_FLAG"] = _nullable_flag(
        working.index,
        working["GA_WEEKS_COMPLETED_FINAL"].notna() & working["GA_WEEKS_COMPLETED_FINAL"].between(1, 45, inclusive="both"),
        working["GA_WEEKS_COMPLETED_FINAL"].notna(),
    )
    working["GA_QC_MISSING_FLAG"] = working["GA_SOURCE"].eq("MISSING").astype("Int8")
    working["GA_QC_BAND_ONLY_FLAG"] = working["GA_SOURCE"].eq("GESTACAO_BAND_ONLY").astype("Int8")
    working["BIRTH_WEIGHT_REPORTED_FLAG"] = birth_weight_g.fillna(0).gt(0).astype("Int8")
    working["QC_STATUS"] = "PASS"

    return working[INDIVIDUAL_OUTPUT_FIELDS].sort_values(["EVENT_DATE", "MUNICIPALITY_CODE"], kind="mergesort").reset_index(drop=True)


def _load_canonical_live_births_residence_daily(root: Path, year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    sinasc_event_path = sinasc_event_partition_file(root, year)
    sinasc_daily_path = sinasc_daily_partition_file(root, year)
    if not sinasc_event_path.exists():
        raise FileNotFoundError(f"Canonical SINASC event parquet missing for {year}: {sinasc_event_path}")
    if not sinasc_daily_path.exists():
        raise FileNotFoundError(f"Canonical SINASC daily parquet missing for {year}: {sinasc_daily_path}")

    sinasc_event = pd.read_parquet(sinasc_event_path, columns=["DATE", "CODMUNRES"])
    municipality_code = sinasc_event["CODMUNRES"].fillna("").astype(str).str.strip()
    if (~municipality_code.str.fullmatch(r"\d{6}")).any():
        raise ValueError(f"Canonical SINASC event has invalid CODMUNRES values for {year}")
    event_date = pd.to_datetime(sinasc_event["DATE"], format="%Y-%m-%d", errors="coerce")
    if event_date.isna().any():
        raise ValueError(f"Canonical SINASC event has invalid DATE values for {year}")

    residence_daily = (
        pd.DataFrame(
            {
                "MUNICIPALITY_CODE": municipality_code,
                "EVENT_DATE": event_date.dt.strftime("%Y-%m-%d"),
                "YEAR": event_date.dt.year.astype("int16"),
                "MONTH": event_date.dt.month.astype("int8"),
                "LIVE_BIRTHS_TOTAL": 1,
            }
        )
        .groupby(["MUNICIPALITY_CODE", "EVENT_DATE", "YEAR", "MONTH"], as_index=False, sort=True)
        .sum(numeric_only=True)
    )
    official_daily_total = int(pd.read_parquet(sinasc_daily_path, columns=["LIVE_BIRTHS_TOTAL"])["LIVE_BIRTHS_TOTAL"].sum())
    event_total = int(len(sinasc_event))
    residence_total = int(residence_daily["LIVE_BIRTHS_TOTAL"].sum())
    if official_daily_total != event_total or residence_total != event_total:
        raise ValueError(
            f"Canonical SINASC live-birth totals disagree for {year}: daily={official_daily_total} event={event_total} residence_daily={residence_total}"
        )
    residence_daily["LIVE_BIRTHS_SOURCE"] = "silver_sinasc_event_residence_benchmarked_to_silver_sinasc_daily"
    return residence_daily, {
        "sinasc_daily_path": str(sinasc_daily_path.relative_to(root)),
        "sinasc_event_path": str(sinasc_event_path.relative_to(root)),
        "official_live_births_total": official_daily_total,
        "residence_live_births_total": residence_total,
        "annual_total_match": official_daily_total == residence_total,
    }


def build_canonical_stillbirth_daily_frame(
    root: Path,
    year: int,
    individual_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stillbirth_daily = (
        individual_frame.assign(
            STILLBIRTH_TOTAL_RAW=individual_frame["FETAL_DEATH_RECORD_FLAG"].fillna(0).astype("int16"),
            STILLBIRTH_TOTAL_GTE28=individual_frame["STILLBIRTH_GTE28_PRIMARY_FLAG"].fillna(0).astype("int16"),
            STILLBIRTH_TOTAL_GTE22_SENSITIVITY=individual_frame["STILLBIRTH_GTE22_SENSITIVITY_FLAG"].fillna(0).astype("int16"),
            STILLBIRTH_GA_REPORTED_TOTAL=individual_frame["GA_VALID_FOR_ANALYSIS_FLAG"].fillna(0).astype("int16"),
            STILLBIRTH_GA_MISSING_TOTAL=individual_frame["GA_SOURCE"].eq("MISSING").astype("int16"),
        )
        .groupby(["MUNICIPALITY_CODE", "EVENT_DATE", "YEAR", "MONTH"], as_index=False, sort=True)
        .agg(
            STILLBIRTH_TOTAL_RAW=("STILLBIRTH_TOTAL_RAW", "sum"),
            STILLBIRTH_TOTAL_GTE28=("STILLBIRTH_TOTAL_GTE28", "sum"),
            STILLBIRTH_TOTAL_GTE22_SENSITIVITY=("STILLBIRTH_TOTAL_GTE22_SENSITIVITY", "sum"),
            STILLBIRTH_GA_REPORTED_TOTAL=("STILLBIRTH_GA_REPORTED_TOTAL", "sum"),
            STILLBIRTH_GA_MISSING_TOTAL=("STILLBIRTH_GA_MISSING_TOTAL", "sum"),
        )
    )

    live_births_daily, benchmark = _load_canonical_live_births_residence_daily(root, year)
    daily = stillbirth_daily.merge(
        live_births_daily,
        on=["MUNICIPALITY_CODE", "EVENT_DATE", "YEAR", "MONTH"],
        how="outer",
    )
    fill_zero_columns = [
        "LIVE_BIRTHS_TOTAL",
        "STILLBIRTH_TOTAL_RAW",
        "STILLBIRTH_TOTAL_GTE28",
        "STILLBIRTH_TOTAL_GTE22_SENSITIVITY",
        "STILLBIRTH_GA_REPORTED_TOTAL",
        "STILLBIRTH_GA_MISSING_TOTAL",
    ]
    for column in fill_zero_columns:
        daily[column] = daily[column].fillna(0).astype("int32")
    daily["LIVE_BIRTHS_SOURCE"] = daily["LIVE_BIRTHS_SOURCE"].fillna(
        "silver_sinasc_event_residence_benchmarked_to_silver_sinasc_daily"
    )
    daily["TOTAL_BIRTHS_PRIMARY"] = daily["LIVE_BIRTHS_TOTAL"] + daily["STILLBIRTH_TOTAL_GTE28"]
    daily["TOTAL_BIRTHS_SENSITIVITY_GTE22"] = daily["LIVE_BIRTHS_TOTAL"] + daily["STILLBIRTH_TOTAL_GTE22_SENSITIVITY"]
    daily["OFFSET_LOG_TOTAL_BIRTHS_PRIMARY"] = daily["TOTAL_BIRTHS_PRIMARY"].map(lambda value: math.log(value) if value > 0 else np.nan)
    daily["OFFSET_LOG_TOTAL_BIRTHS_SENSITIVITY_GTE22"] = daily["TOTAL_BIRTHS_SENSITIVITY_GTE22"].map(
        lambda value: math.log(value) if value > 0 else np.nan
    )
    daily["STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_PRIMARY"] = np.where(
        daily["TOTAL_BIRTHS_PRIMARY"] > 0,
        1000.0 * daily["STILLBIRTH_TOTAL_GTE28"] / daily["TOTAL_BIRTHS_PRIMARY"],
        np.nan,
    )
    daily["STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_SENSITIVITY_GTE22"] = np.where(
        daily["TOTAL_BIRTHS_SENSITIVITY_GTE22"] > 0,
        1000.0 * daily["STILLBIRTH_TOTAL_GTE22_SENSITIVITY"] / daily["TOTAL_BIRTHS_SENSITIVITY_GTE22"],
        np.nan,
    )
    daily["STILLBIRTH_TOTAL"] = daily["STILLBIRTH_TOTAL_GTE28"]
    daily["NASCIMENTOS_TOTAIS_INCLUINDO_FETAIS"] = daily["TOTAL_BIRTHS_PRIMARY"]
    daily["OFFSET_LOG_NASCIMENTOS_TOTAIS_INCLUINDO_FETAIS"] = daily["OFFSET_LOG_TOTAL_BIRTHS_PRIMARY"]
    daily["SOURCE_STILLBIRTH"] = "silver_stillbirth_individual"
    daily["QC_STATUS"] = "PASS"
    return daily[DAILY_OUTPUT_FIELDS].sort_values(["EVENT_DATE", "MUNICIPALITY_CODE"], kind="mergesort").reset_index(drop=True), benchmark


def _reconcile_daily(individual_frame: pd.DataFrame, daily_frame: pd.DataFrame) -> dict[str, bool]:
    return {
        "STILLBIRTH_TOTAL_RAW": int(daily_frame["STILLBIRTH_TOTAL_RAW"].sum()) == int(individual_frame["FETAL_DEATH_RECORD_FLAG"].fillna(0).sum()),
        "STILLBIRTH_TOTAL_GTE28": int(daily_frame["STILLBIRTH_TOTAL_GTE28"].sum()) == int(individual_frame["STILLBIRTH_GTE28_PRIMARY_FLAG"].fillna(0).sum()),
        "STILLBIRTH_TOTAL_GTE22_SENSITIVITY": int(daily_frame["STILLBIRTH_TOTAL_GTE22_SENSITIVITY"].sum()) == int(individual_frame["STILLBIRTH_GTE22_SENSITIVITY_FLAG"].fillna(0).sum()),
    }


def validate_canonical_stillbirth_year(
    root: Path,
    year: int,
    individual_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    denominator_benchmark: dict[str, Any],
    individual_partition_path: Path | None = None,
    daily_partition_path: Path | None = None,
) -> dict[str, Any]:
    official_raw_total = official_total_from_bronze(root, "stillbirth", year)
    reconciliation = _reconcile_daily(individual_frame, daily_frame)
    band_only_mask = individual_frame["GA_SOURCE"].eq("GESTACAO_BAND_ONLY")
    band_not_pseudo_exact_ok = bool(individual_frame.loc[band_only_mask, "GA_WEEKS_COMPLETED_FINAL"].isna().all())
    denominator_primary_ok = bool(
        (daily_frame["TOTAL_BIRTHS_PRIMARY"] == daily_frame["LIVE_BIRTHS_TOTAL"] + daily_frame["STILLBIRTH_TOTAL_GTE28"]).all()
    )
    denominator_sensitivity_ok = bool(
        (daily_frame["TOTAL_BIRTHS_SENSITIVITY_GTE22"] == daily_frame["LIVE_BIRTHS_TOTAL"] + daily_frame["STILLBIRTH_TOTAL_GTE22_SENSITIVITY"]).all()
    )
    offset_primary_ok = bool(
        (
            (
                daily_frame["TOTAL_BIRTHS_PRIMARY"].gt(0)
                & (daily_frame["OFFSET_LOG_TOTAL_BIRTHS_PRIMARY"] - np.log(daily_frame["TOTAL_BIRTHS_PRIMARY"])).abs().le(READBACK_TOLERANCE)
            )
            | (daily_frame["TOTAL_BIRTHS_PRIMARY"].eq(0) & daily_frame["OFFSET_LOG_TOTAL_BIRTHS_PRIMARY"].isna())
        ).all()
    )
    offset_sensitivity_ok = bool(
        (
            (
                daily_frame["TOTAL_BIRTHS_SENSITIVITY_GTE22"].gt(0)
                & (
                    daily_frame["OFFSET_LOG_TOTAL_BIRTHS_SENSITIVITY_GTE22"]
                    - np.log(daily_frame["TOTAL_BIRTHS_SENSITIVITY_GTE22"])
                ).abs().le(READBACK_TOLERANCE)
            )
            | (daily_frame["TOTAL_BIRTHS_SENSITIVITY_GTE22"].eq(0) & daily_frame["OFFSET_LOG_TOTAL_BIRTHS_SENSITIVITY_GTE22"].isna())
        ).all()
    )
    rate_primary_ok = bool(
        (
            (
                daily_frame["TOTAL_BIRTHS_PRIMARY"].gt(0)
                & (
                    daily_frame["STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_PRIMARY"]
                    - (1000.0 * daily_frame["STILLBIRTH_TOTAL_GTE28"] / daily_frame["TOTAL_BIRTHS_PRIMARY"])
                ).abs().le(READBACK_TOLERANCE)
            )
            | (daily_frame["TOTAL_BIRTHS_PRIMARY"].eq(0) & daily_frame["STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_PRIMARY"].isna())
        ).all()
    )
    rate_sensitivity_ok = bool(
        (
            (
                daily_frame["TOTAL_BIRTHS_SENSITIVITY_GTE22"].gt(0)
                & (
                    daily_frame["STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_SENSITIVITY_GTE22"]
                    - (
                        1000.0
                        * daily_frame["STILLBIRTH_TOTAL_GTE22_SENSITIVITY"]
                        / daily_frame["TOTAL_BIRTHS_SENSITIVITY_GTE22"]
                    )
                ).abs().le(READBACK_TOLERANCE)
            )
            | (daily_frame["TOTAL_BIRTHS_SENSITIVITY_GTE22"].eq(0) & daily_frame["STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_SENSITIVITY_GTE22"].isna())
        ).all()
    )
    primary_not_all_raw_ok = True
    if bool(individual_frame["STILLBIRTH_GTE28_PRIMARY_FLAG"].fillna(0).sum()) == int(individual_frame["FETAL_DEATH_RECORD_FLAG"].sum()):
        qualification_expected = bool(
            individual_frame["GA_SOURCE"].eq("MISSING").any()
            or individual_frame["STILLBIRTH_GTE28_PRIMARY_FLAG"].fillna(0).lt(individual_frame["FETAL_DEATH_RECORD_FLAG"].fillna(0)).any()
        )
        primary_not_all_raw_ok = not qualification_expected

    individual_partition_ok = individual_partition_path.exists() if individual_partition_path is not None else True
    daily_partition_ok = daily_partition_path.exists() if daily_partition_path is not None else True
    ga_source_share = {
        source: float(round(share, 6))
        for source, share in (
            individual_frame["GA_SOURCE"].value_counts(normalize=True, dropna=False).sort_index().items()
        )
    }
    summary = {
        "raw_fetal_death_count": int(individual_frame["FETAL_DEATH_RECORD_FLAG"].sum()),
        "stillbirth_primary_gte28_count": int(individual_frame["STILLBIRTH_GTE28_PRIMARY_FLAG"].fillna(0).sum()),
        "stillbirth_sensitivity_gte22_count": int(individual_frame["STILLBIRTH_GTE22_SENSITIVITY_FLAG"].fillna(0).sum()),
        "municipalities_with_at_least_one_stillbirth": int(daily_frame.loc[daily_frame["STILLBIRTH_TOTAL_GTE28"].gt(0), "MUNICIPALITY_CODE"].nunique()),
        "total_births_primary": int(daily_frame["TOTAL_BIRTHS_PRIMARY"].sum()),
        "total_births_sensitivity": int(daily_frame["TOTAL_BIRTHS_SENSITIVITY_GTE22"].sum()),
        "stillbirth_rate_per_1000_total_births_primary": float(
            round(
                1000.0 * daily_frame["STILLBIRTH_TOTAL_GTE28"].sum() / daily_frame["TOTAL_BIRTHS_PRIMARY"].sum(),
                6,
            )
            if int(daily_frame["TOTAL_BIRTHS_PRIMARY"].sum()) > 0
            else np.nan
        ),
        "gestational_age_completeness": {
            "reported_count": int(individual_frame["GA_VALID_FOR_ANALYSIS_FLAG"].sum()),
            "reported_share": float(round(individual_frame["GA_VALID_FOR_ANALYSIS_FLAG"].mean(), 6)),
        },
        "ga_source_share": ga_source_share,
        "ga_qc_dtobito_not_equal_dtnasc_count": int(individual_frame["GA_QC_DTOBITO_EQUALS_DTNASC_FLAG"].eq(0).sum()),
        "completeness_maternal_age_years": float(individual_frame["MATERNAL_AGE_YEARS"].notna().mean()) if len(individual_frame) else 0.0,
        "completeness_maternal_schooling_code_final": float(individual_frame["MATERNAL_SCHOOLING_CODE_FINAL"].notna().mean()) if len(individual_frame) else 0.0,
        "completeness_plurality_code": float(individual_frame["PLURALITY_CODE"].notna().mean()) if len(individual_frame) else 0.0,
        "completeness_fetal_sex_code": float(individual_frame["FETAL_SEX_CODE"].notna().mean()) if len(individual_frame) else 0.0,
        "individual_vs_daily_reconciliation_result": all(reconciliation.values()),
        "denominator_benchmarking_summary": {
            "official_live_births_total": int(denominator_benchmark["official_live_births_total"]),
            "residence_live_births_total": int(denominator_benchmark["residence_live_births_total"]),
            "annual_total_match": bool(denominator_benchmark["annual_total_match"]),
            "sinasc_daily_path": denominator_benchmark["sinasc_daily_path"],
            "sinasc_event_path": denominator_benchmark["sinasc_event_path"],
        },
    }
    validation = {
        "year": year,
        "official_fetal_death_total": official_raw_total,
        "individual_row_count": int(len(individual_frame)),
        "daily_row_count": int(len(daily_frame)),
        "annual_raw_total_matches_official": int(individual_frame["FETAL_DEATH_RECORD_FLAG"].sum()) == official_raw_total == int(daily_frame["STILLBIRTH_TOTAL_RAW"].sum()),
        "event_year_coverage_ok": bool(pd.to_datetime(individual_frame["EVENT_DATE"], format="%Y-%m-%d", errors="coerce").dt.year.eq(year).all()),
        "maternal_residence_key_valid_ok": bool(individual_frame["MUNICIPALITY_CODE"].astype(str).str.fullmatch(r"\d{6}").all()),
        "municipality_coverage_ok": set(individual_frame["MUNICIPALITY_CODE"]) <= set(daily_frame["MUNICIPALITY_CODE"]),
        "individual_key_uniqueness_ok": int(individual_frame["STILLBIRTH_RECORD_ID"].duplicated().sum()) == 0,
        "daily_key_uniqueness_ok": int(daily_frame.duplicated(["MUNICIPALITY_CODE", "EVENT_DATE"]).sum()) == 0,
        "individual_vs_daily_reconciliation": reconciliation,
        "individual_vs_daily_reconciliation_ok": all(reconciliation.values()),
        "primary_denominator_arithmetic_ok": denominator_primary_ok,
        "sensitivity_denominator_arithmetic_ok": denominator_sensitivity_ok,
        "offset_primary_reproducibility_ok": offset_primary_ok,
        "offset_sensitivity_reproducibility_ok": offset_sensitivity_ok,
        "rate_primary_reproducibility_ok": rate_primary_ok,
        "rate_sensitivity_reproducibility_ok": rate_sensitivity_ok,
        "live_birth_merge_lineage_ok": bool(denominator_benchmark["annual_total_match"]),
        "gestational_band_not_mapped_to_pseudo_exact_weeks_ok": band_not_pseudo_exact_ok,
        "primary_not_equal_raw_without_qualification_ok": primary_not_all_raw_ok,
        "fetal_record_scope_ok": bool(individual_frame["GA_QC_FETAL_RECORD_FLAG"].eq(1).all()),
        "individual_partition_integrity_ok": individual_partition_ok,
        "daily_partition_integrity_ok": daily_partition_ok,
        "summary": summary,
    }
    validation["all_checks_passed"] = all(
        [
            validation["annual_raw_total_matches_official"],
            validation["event_year_coverage_ok"],
            validation["maternal_residence_key_valid_ok"],
            validation["municipality_coverage_ok"],
            validation["individual_key_uniqueness_ok"],
            validation["daily_key_uniqueness_ok"],
            validation["individual_vs_daily_reconciliation_ok"],
            validation["primary_denominator_arithmetic_ok"],
            validation["sensitivity_denominator_arithmetic_ok"],
            validation["offset_primary_reproducibility_ok"],
            validation["offset_sensitivity_reproducibility_ok"],
            validation["rate_primary_reproducibility_ok"],
            validation["rate_sensitivity_reproducibility_ok"],
            validation["live_birth_merge_lineage_ok"],
            validation["gestational_band_not_mapped_to_pseudo_exact_weeks_ok"],
            validation["primary_not_equal_raw_without_qualification_ok"],
            validation["fetal_record_scope_ok"],
            validation["individual_partition_integrity_ok"],
            validation["daily_partition_integrity_ok"],
        ]
    )
    return validation


def _write_validation_reports(root: Path, validations: dict[int, dict[str, Any]], manifest_path: str) -> None:
    reports_dir = root / "artifacts" / "validation_reports" / "silver" / "stillbirth"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for year, payload in validations.items():
        report_path = reports_dir / f"{year}.json"
        report_payload = {
            "dataset": "stillbirth",
            "year": year,
            "generated_at": utc_now(),
            "manifest_path": manifest_path,
            **payload,
        }
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_path = root / "artifacts" / "status" / "silver_stillbirth_last_run_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Canonical Stillbirth Silver Summary",
        "",
        f"- Generated at: {utc_now()}",
        f"- Manifest: `{manifest_path}`",
        f"- All yearly checks passed: `{all(validation['all_checks_passed'] for validation in validations.values())}`",
        "",
    ]
    for year in sorted(validations):
        summary = validations[year]["summary"]
        lines.extend(
            [
                f"## {year}",
                "",
                f"- Raw fetal-death count: {summary['raw_fetal_death_count']}",
                f"- Stillbirth primary count (>=28 weeks): {summary['stillbirth_primary_gte28_count']}",
                f"- Stillbirth sensitivity count (>=22 weeks): {summary['stillbirth_sensitivity_gte22_count']}",
                f"- Municipalities with at least one stillbirth: {summary['municipalities_with_at_least_one_stillbirth']}",
                f"- Total births primary: {summary['total_births_primary']}",
                f"- Total births sensitivity: {summary['total_births_sensitivity']}",
                f"- Stillbirth rate per 1,000 total births, primary definition: {summary['stillbirth_rate_per_1000_total_births_primary']}",
                f"- Gestational-age completeness: {summary['gestational_age_completeness']}",
                f"- Share of records by GA_SOURCE: {summary['ga_source_share']}",
                f"- Count with GA_QC_DTOBITO_EQUALS_DTNASC_FLAG == 0: {summary['ga_qc_dtobito_not_equal_dtnasc_count']}",
                f"- Completeness of MATERNAL_AGE_YEARS: {summary['completeness_maternal_age_years']:.6f}",
                f"- Completeness of MATERNAL_SCHOOLING_CODE_FINAL: {summary['completeness_maternal_schooling_code_final']:.6f}",
                f"- Completeness of PLURALITY_CODE: {summary['completeness_plurality_code']:.6f}",
                f"- Completeness of FETAL_SEX_CODE: {summary['completeness_fetal_sex_code']:.6f}",
                f"- Individual vs daily reconciliation result: `{summary['individual_vs_daily_reconciliation_result']}`",
                f"- Denominator benchmarking summary: {summary['denominator_benchmarking_summary']}",
                "",
            ]
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def promote_stillbirth_silver_range(
    root: Path,
    runtime_root: Path,
    start_year: int,
    end_year: int,
    log_path: Path,
    command: str,
    force: bool = False,
) -> dict[str, Any]:
    del force
    validations: dict[int, dict[str, Any]] = {}
    individual_partition_paths: dict[int, str] = {}
    daily_partition_paths: dict[int, str] = {}

    for year in range(start_year, end_year + 1):
        individual_frame = build_canonical_stillbirth_individual_frame(root, year)
        daily_frame, denominator_benchmark = build_canonical_stillbirth_daily_frame(root, year, individual_frame)

        runtime_individual_path = runtime_root / stillbirth_individual_partition_file(root, year).relative_to(root)
        runtime_daily_path = runtime_root / stillbirth_daily_partition_file(root, year).relative_to(root)
        individual_writer = ParquetChunkWriter(runtime_individual_path)
        individual_writer.write(individual_frame)
        individual_row_count, individual_columns = individual_writer.close()
        daily_writer = ParquetChunkWriter(runtime_daily_path)
        daily_writer.write(daily_frame)
        daily_row_count, daily_columns = daily_writer.close()

        final_individual_path = stillbirth_individual_partition_file(root, year)
        final_daily_path = stillbirth_daily_partition_file(root, year)
        sync_verified_parquet(runtime_individual_path, final_individual_path, individual_row_count, individual_columns)
        sync_verified_parquet(runtime_daily_path, final_daily_path, daily_row_count, daily_columns)
        runtime_individual_path.unlink(missing_ok=True)
        runtime_daily_path.unlink(missing_ok=True)
        if runtime_individual_path.parent.exists():
            shutil.rmtree(runtime_individual_path.parent)
        if runtime_daily_path.parent.exists():
            shutil.rmtree(runtime_daily_path.parent)

        validation = validate_canonical_stillbirth_year(
            root,
            year,
            individual_frame,
            daily_frame,
            denominator_benchmark,
            individual_partition_path=final_individual_path,
            daily_partition_path=final_daily_path,
        )
        if not validation["all_checks_passed"]:
            raise ValueError(f"Stillbirth silver validation failed for {year}")
        validations[year] = validation
        individual_partition_paths[year] = str(final_individual_path.relative_to(root))
        daily_partition_paths[year] = str(final_daily_path.relative_to(root))
        append_run_event(
            log_path,
            phase="phase_2_silver_promotion",
            dataset="stillbirth",
            year=year,
            command=command,
            decision=(
                "canonical stillbirth individual and daily silver outputs written with gestational-age-qualified "
                "stillbirth flags and total-birth denominators; "
                f"individual path={final_individual_path.relative_to(root)} daily path={final_daily_path.relative_to(root)}"
            ),
            status="PASS",
        )

    manifest = build_manifest(root, "silver", "stillbirth")
    manifest_path = f"artifacts/manifests/silver_{manifest['dataset']}.manifest.json"
    _write_validation_reports(root, validations, manifest_path)
    return {
        "dataset": "stillbirth",
        "years": list(range(start_year, end_year + 1)),
        "manifest_path": manifest_path,
        "individual_partition_paths": individual_partition_paths,
        "daily_partition_paths": daily_partition_paths,
        "validation": validations,
    }
