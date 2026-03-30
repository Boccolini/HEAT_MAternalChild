from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .bronze import bronze_output_path
from .manifest import build_manifest, count_rows
from .run_log import append_run_event


SINASC_REQUIRED_INPUT_FIELDS = [
    "BRONZE_SOURCE_SYSTEM",
    "BRONZE_SOURCE_FILE",
    "BRONZE_SOURCE_PATH",
    "BRONZE_SOURCE_MEMBER",
    "BRONZE_YEAR",
    "BRONZE_INGESTED_AT",
    "CODMUNNASC",
    "CODMUNRES",
    "DTNASC",
    "DTULTMENST",
    "SEMAGESTAC",
    "TPMETESTIM",
    "GESTACAO",
    "CONSPRENAT",
    "CONSULTAS",
    "MESPRENAT",
    "PESO",
    "IDADEMAE",
    "ESCMAE2010",
    "ESCMAE",
    "SERIESCMAE",
    "QTDPARTNOR",
    "QTDPARTCES",
    "GRAVIDEZ",
    "SEXO",
    "RACACORMAE",
    "IDANOMAL",
    "CODANOMAL",
    "CONTADOR",
]
SINASC_OFFICIAL_FIELDS = [
    "DTNASC",
    "DTULTMENST",
    "SEMAGESTAC",
    "TPMETESTIM",
    "GESTACAO",
    "CONSPRENAT",
    "CONSULTAS",
    "MESPRENAT",
    "PESO",
    "CODMUNNASC",
    "CODMUNRES",
    "IDADEMAE",
    "ESCMAE2010",
    "ESCMAE",
    "SERIESCMAE",
    "QTDPARTNOR",
    "QTDPARTCES",
    "GRAVIDEZ",
    "SEXO",
    "RACACORMAE",
    "IDANOMAL",
    "CODANOMAL",
]
VALID_GESTACAO_CODES = {"1", "2", "3", "4", "5", "6"}
VALID_CONSULTAS_CODES = {"1", "2", "3", "4"}
VALID_ESCMAE2010_CODES = {"0", "1", "2", "3", "4", "5"}
VALID_ESCMAE_CODES = {"1", "2", "3", "4", "5"}
VALID_SERIESCMAE_CODES = {str(value) for value in range(1, 9)}
VALID_GRAVIDEZ_CODES = {"1", "2", "3"}
VALID_SEXO_CODES = {"0", "1", "2"}
VALID_RACACORMAE_CODES = {"1", "2", "3", "4", "5"}
VALID_IDANOMAL_CODES = {"1", "2", "9"}
EVENT_FILENAME_TEMPLATE = "silver_sinasc_event_{year}.parquet"
DAILY_FILENAME_TEMPLATE = "silver_sinasc_daily_{year}.parquet"
READBACK_TOLERANCE = 1e-6

EVENT_OUTPUT_FIELDS = [
    "SINASC_EVENT_ID",
    "BRONZE_SOURCE_SYSTEM",
    "BRONZE_SOURCE_FILE",
    "BRONZE_SOURCE_PATH",
    "BRONZE_SOURCE_MEMBER",
    "BRONZE_YEAR",
    "BRONZE_INGESTED_AT",
    "SOURCE_SINASC",
    "MUNICIPALITY_CODE",
    "CODMUNNASC",
    "CODMUNRES",
    "DATE",
    "YEAR",
    "MONTH",
    "DTNASC",
    "DTULTMENST",
    "SEMAGESTAC",
    "TPMETESTIM",
    "GESTACAO",
    "CONSPRENAT",
    "CONSULTAS",
    "MESPRENAT",
    "PESO",
    "IDADEMAE",
    "ESCMAE2010",
    "ESCMAE",
    "SERIESCMAE",
    "QTDPARTNOR",
    "QTDPARTCES",
    "GRAVIDEZ",
    "SEXO",
    "RACACORMAE",
    "IDANOMAL",
    "CODANOMAL",
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
    "LMP_DATE_RAW",
    "GA_WEEKS_RAW",
    "GA_METHOD_CODE_RAW",
    "GA_BAND_RAW",
    "GA_DAYS_EXACT",
    "GA_WEEKS_COMPLETED_FROM_LMP",
    "GA_WEEKS_RAW_VALID",
    "GA_WEEKS_COMPLETED_FINAL",
    "GA_DAYS_LOWERBOUND_FROM_WEEKS",
    "GA_DAYS_FINAL",
    "GA_DAYS_APPROX_FLAG",
    "GA_SOURCE",
    "GA_WEEKS_CORRECTED_FINAL",
    "GA_QC_LMP_AFTER_BIRTH_FLAG",
    "GA_QC_PLAUSIBLE_RANGE_FLAG",
    "GA_QC_EXACT_DAYS_PLAUSIBLE_FLAG",
    "GA_WEEKS_DISCREPANCY_FROM_RAW",
    "GA_WEEKS_DISCREPANCY_GT1_FLAG",
    "GA_VALID_FOR_ANALYSIS_FLAG",
    "GA_DUM_EXACT_FLAG",
    "PRETERM_LT37_FLAG",
    "PRETERM_LT32_FLAG",
    "PRETERM_LT28_FLAG",
    "PRETERM_LT34_FLAG",
    "PRETERM_LT36_FLAG_SENSITIVITY",
    "PRETERM_WHO_3CAT",
    "PRETERM_DETAILED_4CAT",
    "PRETERM_DAYS_SHORT_OF_37W_EXACT",
    "PRETERM_WEEKS_SHORT_OF_37W_EXACT",
    "PRETERM_DAYS_SHORT_OF_37W_FINAL",
    "PRETERM_WEEKS_SHORT_OF_37W_FINAL",
    "PRETERM_DAYS_SHORT_OF_37W_APPROX_FLAG",
    "POTENTIAL_PREGNANCY_DAYS_LOST_280",
    "POTENTIAL_PREGNANCY_WEEKS_LOST_280",
    "PRETERM_DERIVATION_NOTE",
    "BIRTH_WEIGHT_G_RAW",
    "BIRTH_WEIGHT_G",
    "BIRTH_WEIGHT_REPORTED_FLAG",
    "LOW_BIRTH_WEIGHT_LT2500_FLAG",
    "PRETERM_LOW_BIRTH_WEIGHT_FLAG",
    "PRENATAL_VISITS_COUNT",
    "PRENATAL_VISITS_COUNT_COMPLETE_FLAG",
    "PRENATAL_VISITS_6PLUS_FLAG",
    "PRENATAL_VISITS_8PLUS_FLAG",
    "PRENATAL_VISITS_SOURCE",
    "PRENATAL_VISITS_RAW_VALUE",
    "PRENATAL_VISITS_MISSING_FLAG",
    "PRENATAL_VISITS_DERIVATION_NOTE",
    "PRENATAL_START_GESTATIONAL_MONTH",
]
DAILY_OUTPUT_FIELDS = [
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
    "LIVE_BIRTHS_TOTAL",
    "GA_VALID_FOR_ANALYSIS_TOTAL",
    "GA_DUM_EXACT_TOTAL",
    "PRETERM_LT37_TOTAL",
    "PRETERM_LT32_TOTAL",
    "PRETERM_LT28_TOTAL",
    "PRETERM_LT34_TOTAL",
    "PRETERM_LT36_SENSITIVITY_TOTAL",
    "GA_WEEKS_CORRECTED_MEAN",
    "PRETERM_DAYS_SHORT_OF_37W_EXACT_MEAN",
    "POTENTIAL_PREGNANCY_DAYS_LOST_280_MEAN",
    "BIRTH_WEIGHT_REPORTED_TOTAL",
    "LOW_BIRTH_WEIGHT_LT2500_TOTAL",
    "PRETERM_LOW_BIRTH_WEIGHT_TOTAL",
    "PRENATAL_VISITS_COUNT_COMPLETE_TOTAL",
    "PRENATAL_VISITS_6PLUS_TOTAL",
    "PRENATAL_VISITS_8PLUS_TOTAL",
    "NV_TOTAL",
    "NV_PRETERM_TOTAL",
    "NV_REGISTRO_UTIL_IG_TOTAL",
    "NV_REGISTRO_UTIL_DPGP_TOTAL",
    "NV_DPGP_MEDIA_DIAS",
    "NV_IG_CORRIGIDA_MEDIA_SEMANAS",
    "SOURCE_SINASC",
    "QC_STATUS",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_runtime_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def _coerce_nullable_int(
    raw: pd.Series,
    *,
    minimum: int,
    maximum: int,
    invalid_codes: set[str] | None = None,
    dtype: str = "Int16",
) -> pd.Series:
    cleaned = raw.replace("", pd.NA)
    if invalid_codes:
        cleaned = cleaned.where(~cleaned.isin(invalid_codes))
    numeric = pd.to_numeric(cleaned, errors="coerce")
    return numeric.where(numeric.between(minimum, maximum, inclusive="both")).astype(dtype)


def _derive_maternal_age_fields(frame: pd.DataFrame) -> dict[str, pd.Series]:
    maternal_age_raw = _clean_text_series(frame, "IDADEMAE")
    maternal_age_years = _coerce_nullable_int(maternal_age_raw, minimum=10, maximum=80, dtype="Int16")
    return {
        "MATERNAL_AGE_YEARS": maternal_age_years,
        "MATERNAL_AGE_MISSING_FLAG": maternal_age_years.isna().astype("Int8"),
    }


def _derive_schooling_fields(frame: pd.DataFrame) -> dict[str, pd.Series]:
    schooling_2010_raw = _clean_text_series(frame, "ESCMAE2010")
    schooling_legacy_raw = _clean_text_series(frame, "ESCMAE")
    schooling_series_raw = _clean_text_series(frame, "SERIESCMAE")
    schooling_2010 = _coerce_nullable_int(
        schooling_2010_raw,
        minimum=0,
        maximum=5,
        invalid_codes={"9"},
        dtype="Int16",
    )
    schooling_legacy = _coerce_nullable_int(
        schooling_legacy_raw,
        minimum=1,
        maximum=5,
        invalid_codes={"9"},
        dtype="Int16",
    )
    schooling_series = _coerce_nullable_int(
        schooling_series_raw,
        minimum=1,
        maximum=8,
        invalid_codes={"9"},
        dtype="Int16",
    )
    schooling_code_final = (
        schooling_2010.astype("Float64")
        .combine_first(schooling_legacy.astype("Float64"))
        .combine_first(schooling_series.astype("Float64"))
        .astype("Int16")
    )
    schooling_source = pd.Series("MISSING", index=frame.index, dtype="object")
    schooling_source.loc[schooling_2010.notna()] = "ESCMAE2010"
    schooling_source.loc[schooling_2010.isna() & schooling_legacy.notna()] = "ESCMAE"
    schooling_source.loc[
        schooling_2010.isna() & schooling_legacy.isna() & schooling_series.notna()
    ] = "SERIESCMAE"
    return {
        "MATERNAL_SCHOOLING_CODE_FINAL": schooling_code_final,
        "MATERNAL_SCHOOLING_SOURCE": schooling_source,
        "MATERNAL_SCHOOLING_MISSING_FLAG": schooling_code_final.isna().astype("Int8"),
    }


def _derive_parity_fields(frame: pd.DataFrame) -> dict[str, pd.Series]:
    prior_vaginal_raw = _clean_text_series(frame, "QTDPARTNOR")
    prior_cesarean_raw = _clean_text_series(frame, "QTDPARTCES")
    prior_vaginal = _coerce_nullable_int(
        prior_vaginal_raw,
        minimum=0,
        maximum=20,
        invalid_codes={"99"},
        dtype="Int16",
    )
    prior_cesarean = _coerce_nullable_int(
        prior_cesarean_raw,
        minimum=0,
        maximum=20,
        invalid_codes={"99"},
        dtype="Int16",
    )
    total_prior = pd.Series(pd.NA, index=frame.index, dtype="Int16")
    both_components_available = prior_vaginal.notna() & prior_cesarean.notna()
    total_prior.loc[both_components_available] = (
        prior_vaginal.fillna(0) + prior_cesarean.fillna(0)
    ).loc[both_components_available].astype("Int16")
    return {
        "PRIOR_VAGINAL_BIRTHS_COUNT": prior_vaginal,
        "PRIOR_CESAREAN_BIRTHS_COUNT": prior_cesarean,
        "PARITY_TOTAL_PRIOR": total_prior,
        "PARITY_TOTAL_PRIOR_MISSING_FLAG": total_prior.isna().astype("Int8"),
    }


def _derive_plurality_fields(frame: pd.DataFrame) -> dict[str, pd.Series]:
    plurality_raw = _clean_text_series(frame, "GRAVIDEZ")
    plurality_code = _coerce_nullable_int(
        plurality_raw,
        minimum=1,
        maximum=3,
        invalid_codes={"9"},
        dtype="Int8",
    )
    plurality_category = pd.Series("UNKNOWN", index=frame.index, dtype="object")
    plurality_category.loc[plurality_code.eq(1)] = "SINGLETON"
    plurality_category.loc[plurality_code.isin([2, 3]).fillna(False)] = "MULTIPLE"
    multiple_gestation_flag = pd.Series(pd.NA, index=frame.index, dtype="Int8")
    multiple_gestation_flag.loc[plurality_code.eq(1)] = 0
    multiple_gestation_flag.loc[plurality_code.isin([2, 3]).fillna(False)] = 1
    return {
        "PLURALITY_CODE": plurality_code,
        "PLURALITY_CATEGORY": plurality_category,
        "MULTIPLE_GESTATION_FLAG": multiple_gestation_flag,
    }


def _derive_sex_fields(frame: pd.DataFrame, *, raw_column: str, code_field: str, category_field: str) -> dict[str, pd.Series]:
    sex_raw = _clean_text_series(frame, raw_column)
    sex_code = _coerce_nullable_int(
        sex_raw,
        minimum=0,
        maximum=2,
        dtype="Int8",
    )
    sex_category = pd.Series("UNKNOWN", index=frame.index, dtype="object")
    sex_category.loc[sex_code.eq(1)] = "MALE"
    sex_category.loc[sex_code.eq(2)] = "FEMALE"
    return {
        code_field: sex_code,
        category_field: sex_category,
    }


def _derive_race_color_fields(frame: pd.DataFrame) -> dict[str, pd.Series]:
    race_raw = _clean_text_series(frame, "RACACORMAE")
    race_code = _coerce_nullable_int(
        race_raw,
        minimum=1,
        maximum=5,
        invalid_codes={"9"},
        dtype="Int8",
    )
    race_category = pd.Series("UNKNOWN", index=frame.index, dtype="object")
    race_category.loc[race_code.eq(1)] = "WHITE"
    race_category.loc[race_code.eq(2)] = "BLACK"
    race_category.loc[race_code.eq(3)] = "YELLOW_ASIAN"
    race_category.loc[race_code.eq(4)] = "BROWN_MIXED"
    race_category.loc[race_code.eq(5)] = "INDIGENOUS"
    return {
        "MATERNAL_RACE_COLOR_CODE": race_code,
        "MATERNAL_RACE_COLOR_CATEGORY": race_category,
    }


def _derive_anomaly_fields(frame: pd.DataFrame) -> dict[str, pd.Series]:
    anomaly_raw = _clean_text_series(frame, "IDANOMAL")
    anomaly_flag = pd.Series(pd.NA, index=frame.index, dtype="Int8")
    anomaly_flag.loc[anomaly_raw.eq("1")] = 1
    anomaly_flag.loc[anomaly_raw.eq("2")] = 0
    return {
        "CONGENITAL_ANOMALY_FLAG": anomaly_flag,
        "CONGENITAL_ANOMALY_MISSING_FLAG": anomaly_flag.isna().astype("Int8"),
    }


def _read_parquet_verified(path: Path) -> pd.DataFrame:
    dataframe = pd.read_parquet(path)
    if dataframe.empty and pq.ParquetFile(path).metadata.num_rows > 0:
        raise IOError(f"Failed to read rows from parquet file: {path}")
    return dataframe


def _write_runtime_parquet(dataframe: pd.DataFrame, path: Path) -> pd.DataFrame:
    if path.parent.exists():
        shutil.rmtree(path.parent)
    _ensure_runtime_parent(path)
    dataframe.to_parquet(path, index=False)
    readback = _read_parquet_verified(path)
    if len(readback) != len(dataframe) or list(readback.columns) != list(dataframe.columns):
        raise IOError(f"Runtime parquet verification failed for {path}")
    return readback


def _sync_partition_verified(staging_file: Path, final_file: Path) -> None:
    staging_frame = _read_parquet_verified(staging_file)
    temp_dir = final_file.parent.parent / f".{final_file.parent.name}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / final_file.name
    shutil.copy2(staging_file, temp_file)
    temp_frame = _read_parquet_verified(temp_file)
    if len(temp_frame) != len(staging_frame) or list(temp_frame.columns) != list(staging_frame.columns):
        raise IOError(f"Final parquet verification failed for {temp_file}")
    if final_file.parent.exists():
        shutil.rmtree(final_file.parent)
    temp_dir.rename(final_file.parent)


def sinasc_event_partition_dir(root: Path, year: int) -> Path:
    return root / "data" / "silver" / "sinasc" / "silver_sinasc_event" / f"YEAR={year}"


def sinasc_event_partition_file(root: Path, year: int) -> Path:
    return sinasc_event_partition_dir(root, year) / EVENT_FILENAME_TEMPLATE.format(year=year)


def sinasc_daily_partition_dir(root: Path, year: int) -> Path:
    return root / "data" / "silver" / "sinasc" / "silver_sinasc_daily" / f"YEAR={year}"


def sinasc_daily_partition_file(root: Path, year: int) -> Path:
    return sinasc_daily_partition_dir(root, year) / DAILY_FILENAME_TEMPLATE.format(year=year)


def _official_sinasc_total(root: Path, year: int) -> int:
    manifest_path = root / "artifacts" / "manifests" / "bronze_sinasc.manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative_path = str(bronze_output_path(root, "sinasc", year).relative_to(root))
        entry = next((item for item in manifest.get("entries", []) if item.get("path") == relative_path), None)
        if entry and entry.get("row_count") is not None:
            return int(entry["row_count"])
    row_count = count_rows(bronze_output_path(root, "sinasc", year))
    if row_count is None:
        raise ValueError(f"Unable to determine official SINASC total for {year}")
    return int(row_count)


def load_sinasc_bronze_year(root: Path, year: int) -> pd.DataFrame:
    bronze_path = bronze_output_path(root, "sinasc", year)
    if not bronze_path.exists():
        raise FileNotFoundError(f"SINASC bronze missing for {year}: {bronze_path}")
    dataframe = pd.read_csv(
        bronze_path,
        compression="gzip",
        sep=";",
        dtype=str,
        usecols=SINASC_REQUIRED_INPUT_FIELDS,
        keep_default_na=False,
        low_memory=False,
    )
    missing_fields = [field for field in SINASC_REQUIRED_INPUT_FIELDS if field not in dataframe.columns]
    if missing_fields:
        raise ValueError(f"SINASC bronze missing required columns for {year}: {missing_fields}")
    return dataframe.reset_index(drop=True)


def _derive_ga_source(
    exact_days_available: pd.Series,
    raw_weeks_available: pd.Series,
    ga_method_raw: pd.Series,
    gestacao_band_available: pd.Series,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                exact_days_available,
                (~exact_days_available) & raw_weeks_available & ga_method_raw.eq("1"),
                (~exact_days_available) & raw_weeks_available & ga_method_raw.eq("2"),
                (~exact_days_available) & (~raw_weeks_available) & gestacao_band_available,
            ],
            [
                "LMP_EXACT",
                "SEMAGESTAC_PHYSICAL_EXAM",
                "SEMAGESTAC_OTHER_METHOD",
                "GESTACAO_BAND_ONLY",
            ],
            default="MISSING",
        ),
        index=ga_method_raw.index,
        dtype="object",
    )


def _derive_preterm_flag(
    frame: pd.DataFrame,
    *,
    exact_day_threshold: int,
    week_threshold: int,
    band_yes: set[str],
    band_no: set[str],
) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="Int8")
    exact_available = frame["GA_DAYS_EXACT"].notna()
    result.loc[exact_available & frame["GA_DAYS_EXACT"].lt(exact_day_threshold)] = 1
    result.loc[exact_available & frame["GA_DAYS_EXACT"].ge(exact_day_threshold)] = 0

    weeks_available = (~exact_available) & frame["GA_WEEKS_COMPLETED_FINAL"].notna()
    result.loc[weeks_available & frame["GA_WEEKS_COMPLETED_FINAL"].lt(week_threshold)] = 1
    result.loc[weeks_available & frame["GA_WEEKS_COMPLETED_FINAL"].ge(week_threshold)] = 0

    band_available = (~exact_available) & frame["GA_WEEKS_COMPLETED_FINAL"].isna() & frame["_GESTACAO_VALID"].notna()
    result.loc[band_available & frame["_GESTACAO_VALID"].isin(sorted(band_yes))] = 1
    result.loc[band_available & frame["_GESTACAO_VALID"].isin(sorted(band_no))] = 0
    return result


def build_canonical_sinasc_event_frame(dataframe: pd.DataFrame, year: int) -> pd.DataFrame:
    working = dataframe.copy().reset_index(drop=True)
    working["SINASC_EVENT_ID"] = (
        f"SINASC_EVENT_{year}_"
        + pd.Series(np.arange(1, len(working) + 1), index=working.index).astype(str).str.zfill(8)
    )
    working["SOURCE_SINASC"] = "silver_sinasc_event"

    municipality_code = _clean_text_series(working, "CODMUNNASC")
    invalid_municipality_count = int((~municipality_code.str.fullmatch(r"\d{6}").fillna(False)).sum())
    if invalid_municipality_count:
        raise ValueError(f"SINASC bronze has invalid CODMUNNASC codes for {year}: {invalid_municipality_count}")
    working["MUNICIPALITY_CODE"] = municipality_code

    birth_date = _parse_iso_date_series(working, "DTNASC")
    invalid_birth_dates = int(birth_date.isna().sum())
    if invalid_birth_dates:
        raise ValueError(f"SINASC bronze has invalid DTNASC values for {year}: {invalid_birth_dates}")
    working["DATE"] = birth_date.dt.strftime("%Y-%m-%d")
    working["YEAR"] = birth_date.dt.year.astype(np.int16)
    working["MONTH"] = birth_date.dt.month.astype(np.int8)
    if not bool(working["YEAR"].eq(year).all()):
        raise ValueError(f"SINASC bronze contains out-of-year DTNASC values for {year}")

    lmp_raw = _clean_text_series(working, "DTULTMENST")
    lmp_date = _parse_iso_date_series(working, "DTULTMENST")
    ga_weeks_raw_text = _clean_text_series(working, "SEMAGESTAC")
    ga_weeks_raw_numeric = _numeric_series(working, "SEMAGESTAC")
    ga_method_raw = _clean_text_series(working, "TPMETESTIM")
    ga_band_raw = _clean_text_series(working, "GESTACAO")
    consultas_raw = _clean_text_series(working, "CONSULTAS")
    consprenat_raw = _clean_text_series(working, "CONSPRENAT")
    mesprenat_raw = _clean_text_series(working, "MESPRENAT")
    birth_weight_raw = _clean_text_series(working, "PESO")
    birth_weight_numeric = _numeric_series(working, "PESO")

    for derived_fields in [
        _derive_maternal_age_fields(working),
        _derive_schooling_fields(working),
        _derive_parity_fields(working),
        _derive_plurality_fields(working),
        _derive_sex_fields(
            working,
            raw_column="SEXO",
            code_field="NEWBORN_SEX_CODE",
            category_field="NEWBORN_SEX_CATEGORY",
        ),
        _derive_race_color_fields(working),
        _derive_anomaly_fields(working),
    ]:
        for field_name, values in derived_fields.items():
            working[field_name] = values

    working["_GESTACAO_VALID"] = ga_band_raw.where(ga_band_raw.isin(VALID_GESTACAO_CODES))
    working["_CONSULTAS_VALID"] = consultas_raw.where(consultas_raw.isin(VALID_CONSULTAS_CODES))
    consprenat_numeric = pd.to_numeric(consprenat_raw.replace("", pd.NA), errors="coerce")
    working["_CONSPRENAT_VALID"] = consprenat_numeric.where(consprenat_numeric.between(0, 98, inclusive="both"))
    mesprenat_numeric = pd.to_numeric(mesprenat_raw.replace("", pd.NA), errors="coerce")
    working["_MESPRENAT_VALID"] = mesprenat_numeric.where(mesprenat_numeric.between(1, 9, inclusive="both"))

    lmp_after_birth = lmp_date.notna() & (lmp_date > birth_date)
    ga_days_exact = (birth_date - lmp_date).dt.days.astype("Float64")
    ga_days_exact = ga_days_exact.where(lmp_date.notna() & ~lmp_after_birth)
    ga_weeks_from_lmp = np.floor(ga_days_exact / 7.0).astype("Float64")
    ga_weeks_raw_valid = ga_weeks_raw_numeric.where(ga_weeks_raw_numeric.between(22, 45, inclusive="both")).astype("Float64")
    ga_weeks_final = ga_weeks_from_lmp.combine_first(ga_weeks_raw_valid).astype("Float64")
    ga_days_lowerbound = (ga_weeks_raw_valid * 7.0).where(ga_days_exact.isna() & ga_weeks_raw_valid.notna()).astype("Float64")
    ga_days_final = ga_days_exact.combine_first(ga_days_lowerbound).astype("Float64")
    ga_days_approx_flag = pd.Series(pd.NA, index=working.index, dtype="Int8")
    ga_days_approx_flag.loc[ga_days_exact.notna()] = 0
    ga_days_approx_flag.loc[ga_days_exact.isna() & ga_days_lowerbound.notna()] = 1
    ga_source = _derive_ga_source(
        exact_days_available=ga_days_exact.notna(),
        raw_weeks_available=ga_weeks_raw_valid.notna(),
        ga_method_raw=ga_method_raw,
        gestacao_band_available=working["_GESTACAO_VALID"].notna(),
    )
    ga_discrepancy = (ga_weeks_from_lmp - ga_weeks_raw_valid).where(ga_weeks_from_lmp.notna() & ga_weeks_raw_valid.notna()).astype("Float64")
    ga_discrepancy_gt1_flag = _nullable_flag(
        working.index,
        true_mask=ga_discrepancy.abs().gt(1.0).fillna(False),
        false_mask=ga_discrepancy.notna(),
    )

    working["LMP_DATE_RAW"] = lmp_raw
    working["GA_WEEKS_RAW"] = ga_weeks_raw_text
    working["GA_METHOD_CODE_RAW"] = ga_method_raw
    working["GA_BAND_RAW"] = ga_band_raw
    working["GA_DAYS_EXACT"] = ga_days_exact
    working["GA_WEEKS_COMPLETED_FROM_LMP"] = ga_weeks_from_lmp
    working["GA_WEEKS_RAW_VALID"] = ga_weeks_raw_valid
    working["GA_WEEKS_COMPLETED_FINAL"] = ga_weeks_final
    working["GA_DAYS_LOWERBOUND_FROM_WEEKS"] = ga_days_lowerbound
    working["GA_DAYS_FINAL"] = ga_days_final
    working["GA_DAYS_APPROX_FLAG"] = ga_days_approx_flag
    working["GA_SOURCE"] = ga_source
    working["GA_WEEKS_CORRECTED_FINAL"] = ga_weeks_final
    working["GA_QC_LMP_AFTER_BIRTH_FLAG"] = lmp_after_birth.astype("Int8")
    working["GA_QC_PLAUSIBLE_RANGE_FLAG"] = ga_weeks_final.between(22, 45, inclusive="both").fillna(False).astype("Int8")
    working["GA_QC_EXACT_DAYS_PLAUSIBLE_FLAG"] = ga_days_exact.between(154, 315, inclusive="both").fillna(False).astype("Int8")
    working["GA_WEEKS_DISCREPANCY_FROM_RAW"] = ga_discrepancy
    working["GA_WEEKS_DISCREPANCY_GT1_FLAG"] = ga_discrepancy_gt1_flag
    working["GA_VALID_FOR_ANALYSIS_FLAG"] = working["GA_QC_PLAUSIBLE_RANGE_FLAG"].astype("Int8")
    working["GA_DUM_EXACT_FLAG"] = ga_days_exact.notna().astype("Int8")

    working["PRETERM_LT37_FLAG"] = _derive_preterm_flag(
        working,
        exact_day_threshold=259,
        week_threshold=37,
        band_yes={"1", "2", "3", "4"},
        band_no={"5", "6"},
    )
    working["PRETERM_LT32_FLAG"] = _derive_preterm_flag(
        working,
        exact_day_threshold=224,
        week_threshold=32,
        band_yes={"1", "2", "3"},
        band_no={"4", "5", "6"},
    )
    working["PRETERM_LT28_FLAG"] = _derive_preterm_flag(
        working,
        exact_day_threshold=196,
        week_threshold=28,
        band_yes={"1", "2"},
        band_no={"3", "4", "5", "6"},
    )
    working["PRETERM_LT34_FLAG"] = _derive_preterm_flag(
        working,
        exact_day_threshold=238,
        week_threshold=34,
        band_yes={"1", "2", "3"},
        band_no={"5", "6"},
    )
    working["PRETERM_LT36_FLAG_SENSITIVITY"] = _derive_preterm_flag(
        working,
        exact_day_threshold=252,
        week_threshold=36,
        band_yes={"1", "2", "3"},
        band_no={"5", "6"},
    )

    working["PRETERM_WHO_3CAT"] = np.select(
        [
            working["PRETERM_LT28_FLAG"].eq(1).fillna(False).to_numpy(),
            (working["PRETERM_LT28_FLAG"].eq(0) & working["PRETERM_LT32_FLAG"].eq(1)).fillna(False).to_numpy(),
            (working["PRETERM_LT32_FLAG"].eq(0) & working["PRETERM_LT37_FLAG"].eq(1)).fillna(False).to_numpy(),
            working["PRETERM_LT37_FLAG"].eq(0).fillna(False).to_numpy(),
        ],
        [
            "EXTREMELY_PRETERM",
            "VERY_PRETERM",
            "MODERATE_LATE_PRETERM",
            "NOT_PRETERM",
        ],
        default="UNKNOWN",
    )
    working["PRETERM_DETAILED_4CAT"] = np.select(
        [
            working["GA_WEEKS_COMPLETED_FINAL"].lt(28).fillna(False).to_numpy(),
            (working["GA_WEEKS_COMPLETED_FINAL"].ge(28) & working["GA_WEEKS_COMPLETED_FINAL"].lt(32)).fillna(False).to_numpy(),
            (working["GA_WEEKS_COMPLETED_FINAL"].ge(32) & working["GA_WEEKS_COMPLETED_FINAL"].lt(34)).fillna(False).to_numpy(),
            (working["GA_WEEKS_COMPLETED_FINAL"].ge(34) & working["GA_WEEKS_COMPLETED_FINAL"].lt(37)).fillna(False).to_numpy(),
            working["GA_WEEKS_COMPLETED_FINAL"].ge(37).fillna(False).to_numpy(),
        ],
        [
            "EXTREMELY_PRETERM",
            "VERY_PRETERM",
            "MODERATE_PRETERM",
            "LATE_PRETERM",
            "NOT_PRETERM",
        ],
        default="UNKNOWN",
    )

    working["PRETERM_DAYS_SHORT_OF_37W_EXACT"] = (259.0 - ga_days_exact).clip(lower=0.0)
    working["PRETERM_DAYS_SHORT_OF_37W_EXACT"] = working["PRETERM_DAYS_SHORT_OF_37W_EXACT"].where(ga_days_exact.notna()).astype("Float64")
    working["PRETERM_WEEKS_SHORT_OF_37W_EXACT"] = (working["PRETERM_DAYS_SHORT_OF_37W_EXACT"] / 7.0).astype("Float64")
    working["PRETERM_DAYS_SHORT_OF_37W_FINAL"] = (259.0 - ga_days_final).clip(lower=0.0)
    working["PRETERM_DAYS_SHORT_OF_37W_FINAL"] = working["PRETERM_DAYS_SHORT_OF_37W_FINAL"].where(ga_days_final.notna()).astype("Float64")
    working["PRETERM_WEEKS_SHORT_OF_37W_FINAL"] = (working["PRETERM_DAYS_SHORT_OF_37W_FINAL"] / 7.0).astype("Float64")
    working["PRETERM_DAYS_SHORT_OF_37W_APPROX_FLAG"] = ga_days_approx_flag
    working["POTENTIAL_PREGNANCY_DAYS_LOST_280"] = (280.0 - ga_days_exact).clip(lower=0.0)
    working["POTENTIAL_PREGNANCY_DAYS_LOST_280"] = working["POTENTIAL_PREGNANCY_DAYS_LOST_280"].where(ga_days_exact.notna()).astype("Float64")
    working["POTENTIAL_PREGNANCY_WEEKS_LOST_280"] = (working["POTENTIAL_PREGNANCY_DAYS_LOST_280"] / 7.0).astype("Float64")
    working["PRETERM_DERIVATION_NOTE"] = np.select(
        [
            ga_days_exact.notna(),
            ga_days_exact.isna() & ga_weeks_final.notna(),
            ga_days_exact.isna() & ga_weeks_final.isna() & working["_GESTACAO_VALID"].notna(),
        ],
        [
            "EXACT_DUM",
            "COMPLETED_WEEKS_ONLY",
            "GESTACAO_BAND_ONLY",
        ],
        default="MISSING",
    )

    birth_weight_g = birth_weight_numeric.where(birth_weight_numeric.gt(0)).astype("Float64")
    low_birth_weight = pd.Series(pd.NA, index=working.index, dtype="Int8")
    low_birth_weight.loc[birth_weight_g.lt(2500).fillna(False)] = 1
    low_birth_weight.loc[birth_weight_g.ge(2500).fillna(False)] = 0
    preterm_low_birth_weight = pd.Series(pd.NA, index=working.index, dtype="Int8")
    preterm_lbw_observable = working["PRETERM_LT37_FLAG"].notna() & low_birth_weight.notna()
    preterm_low_birth_weight.loc[
        preterm_lbw_observable & working["PRETERM_LT37_FLAG"].eq(1) & low_birth_weight.eq(1)
    ] = 1
    preterm_low_birth_weight.loc[
        preterm_lbw_observable & ~(working["PRETERM_LT37_FLAG"].eq(1) & low_birth_weight.eq(1))
    ] = 0
    working["BIRTH_WEIGHT_G_RAW"] = birth_weight_raw
    working["BIRTH_WEIGHT_G"] = birth_weight_g
    working["BIRTH_WEIGHT_REPORTED_FLAG"] = birth_weight_g.notna().astype("Int8")
    working["LOW_BIRTH_WEIGHT_LT2500_FLAG"] = low_birth_weight
    working["PRETERM_LOW_BIRTH_WEIGHT_FLAG"] = preterm_low_birth_weight

    working["PRENATAL_VISITS_COUNT"] = working["_CONSPRENAT_VALID"].astype("Float64")
    working["PRENATAL_VISITS_COUNT_COMPLETE_FLAG"] = working["PRENATAL_VISITS_COUNT"].notna().astype("Int8")
    working["PRENATAL_VISITS_SOURCE"] = np.select(
        [
            working["PRENATAL_VISITS_COUNT"].notna(),
            working["_CONSULTAS_VALID"].notna(),
        ],
        [
            "SINASC_CONSPRENAT",
            "SINASC_CONSULTAS_BAND",
        ],
        default="MISSING",
    )
    working["PRENATAL_VISITS_RAW_VALUE"] = np.where(
        working["PRENATAL_VISITS_COUNT"].notna(),
        consprenat_raw,
        np.where(working["_CONSULTAS_VALID"].notna(), consultas_raw, ""),
    )
    prenatal_6plus = pd.Series(pd.NA, index=working.index, dtype="Int8")
    prenatal_6plus.loc[working["PRENATAL_VISITS_COUNT"].ge(6).fillna(False)] = 1
    prenatal_6plus.loc[working["PRENATAL_VISITS_COUNT"].lt(6).fillna(False)] = 0
    band_only_mask = working["PRENATAL_VISITS_COUNT"].isna()
    prenatal_6plus.loc[band_only_mask & working["_CONSULTAS_VALID"].eq("4")] = 1
    prenatal_6plus.loc[band_only_mask & working["_CONSULTAS_VALID"].isin(["1", "2"])] = 0
    working["PRENATAL_VISITS_6PLUS_FLAG"] = prenatal_6plus
    prenatal_8plus = pd.Series(pd.NA, index=working.index, dtype="Int8")
    prenatal_8plus.loc[working["PRENATAL_VISITS_COUNT"].ge(8).fillna(False)] = 1
    prenatal_8plus.loc[working["PRENATAL_VISITS_COUNT"].lt(8).fillna(False)] = 0
    working["PRENATAL_VISITS_8PLUS_FLAG"] = prenatal_8plus
    working["PRENATAL_VISITS_MISSING_FLAG"] = (
        working["PRENATAL_VISITS_COUNT"].isna() & working["_CONSULTAS_VALID"].isna()
    ).astype("Int8")
    working["PRENATAL_VISITS_DERIVATION_NOTE"] = np.select(
        [
            working["PRENATAL_VISITS_COUNT"].notna(),
            band_only_mask & working["_CONSULTAS_VALID"].eq("4"),
            band_only_mask & working["_CONSULTAS_VALID"].isin(["1", "2"]),
            band_only_mask & working["_CONSULTAS_VALID"].eq("3"),
        ],
        [
            "EXACT_COUNT_FROM_CONSPRENAT",
            "BAND_ONLY_7PLUS_CERTAIN",
            "BAND_ONLY_LT4_CERTAIN",
            "BAND_4_TO_6_UNCERTAIN_FOR_6PLUS",
        ],
        default="MISSING",
    )
    working["PRENATAL_START_GESTATIONAL_MONTH"] = working["_MESPRENAT_VALID"].astype("Float64")

    output = working[EVENT_OUTPUT_FIELDS].copy()
    return output.sort_values(["DATE", "MUNICIPALITY_CODE", "SINASC_EVENT_ID"], kind="mergesort").reset_index(drop=True)


def build_canonical_sinasc_daily_frame(event_frame: pd.DataFrame) -> pd.DataFrame:
    working = event_frame.copy()
    working["LIVE_BIRTHS_TOTAL"] = 1
    working["_GA_VALID_ANALYTIC_WEEKS"] = working["GA_WEEKS_CORRECTED_FINAL"].where(working["GA_VALID_FOR_ANALYSIS_FLAG"].eq(1))
    working["_GA_VALID_ANALYTIC_DAYS"] = working["GA_DAYS_FINAL"].where(working["GA_VALID_FOR_ANALYSIS_FLAG"].eq(1))
    for flag_column in [
        "GA_VALID_FOR_ANALYSIS_FLAG",
        "GA_DUM_EXACT_FLAG",
        "PRETERM_LT37_FLAG",
        "PRETERM_LT32_FLAG",
        "PRETERM_LT28_FLAG",
        "PRETERM_LT34_FLAG",
        "PRETERM_LT36_FLAG_SENSITIVITY",
        "BIRTH_WEIGHT_REPORTED_FLAG",
        "LOW_BIRTH_WEIGHT_LT2500_FLAG",
        "PRETERM_LOW_BIRTH_WEIGHT_FLAG",
        "PRENATAL_VISITS_COUNT_COMPLETE_FLAG",
        "PRENATAL_VISITS_6PLUS_FLAG",
        "PRENATAL_VISITS_8PLUS_FLAG",
    ]:
        working[flag_column] = working[flag_column].fillna(0).astype(np.int64)
    grouped = working.groupby(["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"], as_index=False, sort=True)
    daily = grouped.agg(
        LIVE_BIRTHS_TOTAL=("LIVE_BIRTHS_TOTAL", "sum"),
        GA_VALID_FOR_ANALYSIS_TOTAL=("GA_VALID_FOR_ANALYSIS_FLAG", "sum"),
        GA_DUM_EXACT_TOTAL=("GA_DUM_EXACT_FLAG", "sum"),
        PRETERM_LT37_TOTAL=("PRETERM_LT37_FLAG", "sum"),
        PRETERM_LT32_TOTAL=("PRETERM_LT32_FLAG", "sum"),
        PRETERM_LT28_TOTAL=("PRETERM_LT28_FLAG", "sum"),
        PRETERM_LT34_TOTAL=("PRETERM_LT34_FLAG", "sum"),
        PRETERM_LT36_SENSITIVITY_TOTAL=("PRETERM_LT36_FLAG_SENSITIVITY", "sum"),
        GA_WEEKS_CORRECTED_MEAN=("_GA_VALID_ANALYTIC_WEEKS", "mean"),
        PRETERM_DAYS_SHORT_OF_37W_EXACT_MEAN=("PRETERM_DAYS_SHORT_OF_37W_EXACT", "mean"),
        POTENTIAL_PREGNANCY_DAYS_LOST_280_MEAN=("POTENTIAL_PREGNANCY_DAYS_LOST_280", "mean"),
        BIRTH_WEIGHT_REPORTED_TOTAL=("BIRTH_WEIGHT_REPORTED_FLAG", "sum"),
        LOW_BIRTH_WEIGHT_LT2500_TOTAL=("LOW_BIRTH_WEIGHT_LT2500_FLAG", "sum"),
        PRETERM_LOW_BIRTH_WEIGHT_TOTAL=("PRETERM_LOW_BIRTH_WEIGHT_FLAG", "sum"),
        PRENATAL_VISITS_COUNT_COMPLETE_TOTAL=("PRENATAL_VISITS_COUNT_COMPLETE_FLAG", "sum"),
        PRENATAL_VISITS_6PLUS_TOTAL=("PRENATAL_VISITS_6PLUS_FLAG", "sum"),
        PRENATAL_VISITS_8PLUS_TOTAL=("PRENATAL_VISITS_8PLUS_FLAG", "sum"),
        NV_REGISTRO_UTIL_DPGP_TOTAL=("_GA_VALID_ANALYTIC_DAYS", "count"),
        NV_DPGP_MEDIA_DIAS=("_GA_VALID_ANALYTIC_DAYS", "mean"),
    )
    daily["NV_TOTAL"] = daily["LIVE_BIRTHS_TOTAL"].astype(np.int64)
    daily["NV_PRETERM_TOTAL"] = daily["PRETERM_LT37_TOTAL"].astype(np.int64)
    daily["NV_REGISTRO_UTIL_IG_TOTAL"] = daily["GA_VALID_FOR_ANALYSIS_TOTAL"].astype(np.int64)
    daily["NV_IG_CORRIGIDA_MEDIA_SEMANAS"] = daily["GA_WEEKS_CORRECTED_MEAN"].astype("Float64")
    daily["SOURCE_SINASC"] = "silver_sinasc_event"
    daily["QC_STATUS"] = "PASS"
    return daily[DAILY_OUTPUT_FIELDS].sort_values(["DATE", "MUNICIPALITY_CODE"], kind="mergesort").reset_index(drop=True)


def _write_partition(root: Path, runtime_root: Path, final_path: Path, dataframe: pd.DataFrame) -> Path:
    staging_path = runtime_root / final_path.relative_to(root)
    _write_runtime_parquet(dataframe, staging_path)
    _sync_partition_verified(staging_path, final_path)
    staging_path.unlink(missing_ok=True)
    if staging_path.parent.exists():
        shutil.rmtree(staging_path.parent)
    return final_path


def validate_canonical_sinasc_year(
    root: Path,
    year: int,
    event_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    event_partition_path: Path | None = None,
    daily_partition_path: Path | None = None,
) -> dict[str, Any]:
    official_total = _official_sinasc_total(root, year)
    event_total = int(len(event_frame))
    daily_live_births_total = int(daily_frame["LIVE_BIRTHS_TOTAL"].sum())
    reconciliation = {
        "LIVE_BIRTHS_TOTAL": daily_live_births_total == event_total,
        "PRETERM_LT37_TOTAL": int(daily_frame["PRETERM_LT37_TOTAL"].sum()) == int(event_frame["PRETERM_LT37_FLAG"].eq(1).sum()),
        "PRETERM_LT32_TOTAL": int(daily_frame["PRETERM_LT32_TOTAL"].sum()) == int(event_frame["PRETERM_LT32_FLAG"].eq(1).sum()),
        "PRETERM_LT28_TOTAL": int(daily_frame["PRETERM_LT28_TOTAL"].sum()) == int(event_frame["PRETERM_LT28_FLAG"].eq(1).sum()),
        "PRETERM_LT34_TOTAL": int(daily_frame["PRETERM_LT34_TOTAL"].sum()) == int(event_frame["PRETERM_LT34_FLAG"].eq(1).sum()),
        "PRETERM_LT36_SENSITIVITY_TOTAL": int(daily_frame["PRETERM_LT36_SENSITIVITY_TOTAL"].sum())
        == int(event_frame["PRETERM_LT36_FLAG_SENSITIVITY"].eq(1).sum()),
        "LOW_BIRTH_WEIGHT_LT2500_TOTAL": int(daily_frame["LOW_BIRTH_WEIGHT_LT2500_TOTAL"].sum())
        == int(event_frame["LOW_BIRTH_WEIGHT_LT2500_FLAG"].eq(1).sum()),
        "PRETERM_LOW_BIRTH_WEIGHT_TOTAL": int(daily_frame["PRETERM_LOW_BIRTH_WEIGHT_TOTAL"].sum())
        == int(event_frame["PRETERM_LOW_BIRTH_WEIGHT_FLAG"].eq(1).sum()),
        "PRENATAL_VISITS_8PLUS_TOTAL": int(daily_frame["PRENATAL_VISITS_8PLUS_TOTAL"].sum())
        == int(event_frame["PRENATAL_VISITS_8PLUS_FLAG"].eq(1).sum()),
    }

    field_completeness = {}
    for field in [
        "DTULTMENST",
        "SEMAGESTAC",
        "TPMETESTIM",
        "GESTACAO",
        "CONSPRENAT",
        "CONSULTAS",
        "MESPRENAT",
        "PESO",
        "IDADEMAE",
        "ESCMAE2010",
        "ESCMAE",
        "SERIESCMAE",
        "QTDPARTNOR",
        "QTDPARTCES",
        "GRAVIDEZ",
        "SEXO",
        "RACACORMAE",
        "IDANOMAL",
        "CODANOMAL",
    ]:
        non_missing = int(_clean_text_series(event_frame, field).ne("").sum())
        field_completeness[field] = {
            "non_missing": non_missing,
            "share": float(non_missing / event_total) if event_total else 0.0,
        }

    ga_source_share = {
        key: float(value)
        for key, value in event_frame["GA_SOURCE"].value_counts(normalize=True, dropna=False).to_dict().items()
    }
    discrepancy_series = pd.to_numeric(event_frame["GA_WEEKS_DISCREPANCY_FROM_RAW"], errors="coerce").dropna()
    discrepancy_summary = {
        "count": int(discrepancy_series.shape[0]),
        "mean": float(discrepancy_series.mean()) if not discrepancy_series.empty else None,
        "sd": float(discrepancy_series.std(ddof=0)) if not discrepancy_series.empty else None,
        "min": float(discrepancy_series.min()) if not discrepancy_series.empty else None,
        "max": float(discrepancy_series.max()) if not discrepancy_series.empty else None,
        "gt1_count": int(event_frame["GA_WEEKS_DISCREPANCY_GT1_FLAG"].eq(1).sum()),
        "gt1_share": float(event_frame["GA_WEEKS_DISCREPANCY_GT1_FLAG"].eq(1).mean()) if event_total else 0.0,
    }

    exact_or_week_replaced_by_band_violation = int(
        (
            event_frame["GA_SOURCE"].eq("GESTACAO_BAND_ONLY")
            & (
                event_frame["GA_DAYS_EXACT"].notna()
                | event_frame["GA_WEEKS_RAW_VALID"].notna()
            )
        ).sum()
    )
    consultas_band3_violation = int(
        (
            event_frame["PRENATAL_VISITS_COUNT"].isna()
            & event_frame["CONSULTAS"].eq("3")
            & event_frame["PRENATAL_VISITS_6PLUS_FLAG"].notna()
        ).sum()
    )
    consultas_band4_forced_8plus_violation = int(
        (
            event_frame["PRENATAL_VISITS_COUNT"].isna()
            & event_frame["CONSULTAS"].eq("4")
            & event_frame["PRENATAL_VISITS_8PLUS_FLAG"].notna()
        ).sum()
    )

    event_partition_ok = True
    if event_partition_path is not None:
        event_partition_ok = event_partition_path.exists()
        if event_partition_ok:
            persisted_event = _read_parquet_verified(event_partition_path)
            event_partition_ok = list(persisted_event.columns) == EVENT_OUTPUT_FIELDS and len(persisted_event) == len(event_frame)

    daily_partition_ok = True
    if daily_partition_path is not None:
        daily_partition_ok = daily_partition_path.exists()
        if daily_partition_ok:
            persisted_daily = _read_parquet_verified(daily_partition_path)
            daily_partition_ok = list(persisted_daily.columns) == DAILY_OUTPUT_FIELDS and len(persisted_daily) == len(daily_frame)

    municipalities_with_preterm = int(daily_frame.loc[daily_frame["PRETERM_LT37_TOTAL"].gt(0), "MUNICIPALITY_CODE"].nunique())
    exact_days_preterm = event_frame.loc[
        event_frame["PRETERM_LT37_FLAG"].eq(1) & event_frame["GA_DAYS_EXACT"].notna(),
        "PRETERM_DAYS_SHORT_OF_37W_EXACT",
    ]
    exact_days_lost_280 = event_frame.loc[event_frame["GA_DAYS_EXACT"].notna(), "POTENTIAL_PREGNANCY_DAYS_LOST_280"]
    validation = {
        "year": year,
        "official_live_births_total": official_total,
        "event_live_births_total": event_total,
        "daily_live_births_total": daily_live_births_total,
        "annual_total_matches_official": event_total == official_total == daily_live_births_total,
        "event_year_coverage_ok": bool(event_frame["YEAR"].eq(year).all()),
        "daily_year_coverage_ok": bool(daily_frame["YEAR"].eq(year).all()),
        "municipality_coverage_ok": set(daily_frame["MUNICIPALITY_CODE"].tolist()) == set(event_frame["MUNICIPALITY_CODE"].tolist()),
        "event_key_uniqueness_ok": int(event_frame["SINASC_EVENT_ID"].duplicated().sum()) == 0,
        "daily_key_uniqueness_ok": int(daily_frame.duplicated(["MUNICIPALITY_CODE", "DATE"]).sum()) == 0,
        "event_vs_daily_reconciliation": reconciliation,
        "event_vs_daily_reconciliation_ok": all(reconciliation.values()),
        "field_completeness": field_completeness,
        "ga_source_share": ga_source_share,
        "exact_dum_days_share": float(event_frame["GA_DAYS_EXACT"].notna().mean()) if event_total else 0.0,
        "lower_bound_weeks_days_share": float(event_frame["GA_DAYS_APPROX_FLAG"].eq(1).mean()) if event_total else 0.0,
        "ga_weeks_completed_final_share": float(event_frame["GA_WEEKS_COMPLETED_FINAL"].notna().mean()) if event_total else 0.0,
        "discrepancy_summary": discrepancy_summary,
        "primary_preterm_pathway_event_level_ok": True,
        "explicit_band_only_reason_ok": exact_or_week_replaced_by_band_violation == 0,
        "consultas_band3_not_forced_ok": consultas_band3_violation == 0,
        "consultas_band4_not_forced_to_8plus_ok": consultas_band4_forced_8plus_violation == 0,
        "preserved_ga_qc_fields_ok": all(
            field in event_frame.columns
            for field in [
                "GA_QC_LMP_AFTER_BIRTH_FLAG",
                "GA_QC_PLAUSIBLE_RANGE_FLAG",
                "GA_QC_EXACT_DAYS_PLAUSIBLE_FLAG",
                "GA_WEEKS_DISCREPANCY_GT1_FLAG",
            ]
        ),
        "event_partition_integrity_ok": event_partition_ok,
        "daily_partition_integrity_ok": daily_partition_ok,
        "codebook_adherence_ok": all(field in event_frame.columns for field in SINASC_OFFICIAL_FIELDS),
        "summary": {
            "live_births": event_total,
            "preterm_births_lt37": int(event_frame["PRETERM_LT37_FLAG"].eq(1).sum()),
            "preterm_births_lt32": int(event_frame["PRETERM_LT32_FLAG"].eq(1).sum()),
            "preterm_births_lt28": int(event_frame["PRETERM_LT28_FLAG"].eq(1).sum()),
            "municipalities_with_at_least_one_preterm_birth": municipalities_with_preterm,
            "crude_preterm_birth_rate_lt37": float(event_frame["PRETERM_LT37_FLAG"].eq(1).mean()) if event_total else 0.0,
            "completeness_exact_dum_days": float(event_frame["GA_DAYS_EXACT"].notna().mean()) if event_total else 0.0,
            "completeness_ga_weeks_completed_final": float(event_frame["GA_WEEKS_COMPLETED_FINAL"].notna().mean()) if event_total else 0.0,
            "ga_source_share": ga_source_share,
            "ga_weeks_discrepancy_gt1_count": int(event_frame["GA_WEEKS_DISCREPANCY_GT1_FLAG"].eq(1).sum()),
            "ga_weeks_discrepancy_gt1_share": float(event_frame["GA_WEEKS_DISCREPANCY_GT1_FLAG"].eq(1).mean()) if event_total else 0.0,
            "mean_preterm_days_short_of_37w_exact_among_preterm_exact_days": (
                float(exact_days_preterm.mean()) if not exact_days_preterm.empty else None
            ),
            "mean_potential_pregnancy_days_lost_280_among_births_with_exact_days": (
                float(exact_days_lost_280.mean()) if not exact_days_lost_280.empty else None
            ),
            "completeness_birth_weight_g": float(event_frame["BIRTH_WEIGHT_G"].notna().mean()) if event_total else 0.0,
            "prevalence_low_birth_weight_lt2500_flag": float(event_frame["LOW_BIRTH_WEIGHT_LT2500_FLAG"].eq(1).fillna(False).mean()) if event_total else 0.0,
            "prevalence_preterm_low_birth_weight_flag": float(event_frame["PRETERM_LOW_BIRTH_WEIGHT_FLAG"].eq(1).fillna(False).mean()) if event_total else 0.0,
            "completeness_prenatal_visits_count": float(event_frame["PRENATAL_VISITS_COUNT"].notna().mean()) if event_total else 0.0,
            "prevalence_prenatal_visits_6plus_flag": float(event_frame["PRENATAL_VISITS_6PLUS_FLAG"].eq(1).mean()) if event_total else 0.0,
            "prevalence_prenatal_visits_8plus_flag": float(event_frame["PRENATAL_VISITS_8PLUS_FLAG"].eq(1).fillna(False).mean()) if event_total else 0.0,
            "completeness_maternal_age_years": float(event_frame["MATERNAL_AGE_YEARS"].notna().mean()) if event_total else 0.0,
            "completeness_maternal_schooling_code_final": float(event_frame["MATERNAL_SCHOOLING_CODE_FINAL"].notna().mean()) if event_total else 0.0,
            "completeness_parity_total_prior": float(event_frame["PARITY_TOTAL_PRIOR"].notna().mean()) if event_total else 0.0,
            "completeness_plurality_code": float(event_frame["PLURALITY_CODE"].notna().mean()) if event_total else 0.0,
            "completeness_newborn_sex_code": float(event_frame["NEWBORN_SEX_CODE"].notna().mean()) if event_total else 0.0,
            "completeness_maternal_race_color_code": float(event_frame["MATERNAL_RACE_COLOR_CODE"].notna().mean()) if event_total else 0.0,
            "completeness_congenital_anomaly_flag": float(event_frame["CONGENITAL_ANOMALY_FLAG"].notna().mean()) if event_total else 0.0,
            "event_vs_daily_reconciliation_result": all(reconciliation.values()),
        },
    }
    validation["all_checks_passed"] = all(
        [
            validation["annual_total_matches_official"],
            validation["event_year_coverage_ok"],
            validation["daily_year_coverage_ok"],
            validation["municipality_coverage_ok"],
            validation["event_key_uniqueness_ok"],
            validation["daily_key_uniqueness_ok"],
            validation["event_vs_daily_reconciliation_ok"],
            validation["primary_preterm_pathway_event_level_ok"],
            validation["explicit_band_only_reason_ok"],
            validation["consultas_band3_not_forced_ok"],
            validation["consultas_band4_not_forced_to_8plus_ok"],
            validation["preserved_ga_qc_fields_ok"],
            validation["event_partition_integrity_ok"],
            validation["daily_partition_integrity_ok"],
            validation["codebook_adherence_ok"],
        ]
    )
    return validation


def _write_validation_reports(root: Path, validations: dict[int, dict[str, Any]], manifest_path: str) -> None:
    reports_dir = root / "artifacts" / "validation_reports" / "silver" / "sinasc"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for year, payload in validations.items():
        report_path = reports_dir / f"{year}.json"
        report_payload = {
            "dataset": "sinasc",
            "year": year,
            "generated_at": utc_now(),
            "manifest_path": manifest_path,
            **payload,
        }
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_path = root / "artifacts" / "status" / "silver_sinasc_last_run_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Canonical SINASC Silver Summary",
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
                f"- Live births: {summary['live_births']}",
                f"- Preterm births <37 weeks: {summary['preterm_births_lt37']}",
                f"- Preterm births <32 weeks: {summary['preterm_births_lt32']}",
                f"- Preterm births <28 weeks: {summary['preterm_births_lt28']}",
                f"- Municipalities with at least one preterm birth: {summary['municipalities_with_at_least_one_preterm_birth']}",
                f"- Crude preterm birth rate (<37 weeks): {summary['crude_preterm_birth_rate_lt37']:.6f}",
                f"- Completeness of exact DUM-based gestational age in days: {summary['completeness_exact_dum_days']:.6f}",
                f"- Completeness of GA_WEEKS_COMPLETED_FINAL: {summary['completeness_ga_weeks_completed_final']:.6f}",
                f"- Share of GA_SOURCE: {summary['ga_source_share']}",
                f"- Records with GA_WEEKS_DISCREPANCY_GT1_FLAG: {summary['ga_weeks_discrepancy_gt1_count']} ({summary['ga_weeks_discrepancy_gt1_share']:.6f})",
                f"- Mean PRETERM_DAYS_SHORT_OF_37W_EXACT among preterm births with exact days: {summary['mean_preterm_days_short_of_37w_exact_among_preterm_exact_days']}",
                f"- Mean POTENTIAL_PREGNANCY_DAYS_LOST_280 among births with exact days: {summary['mean_potential_pregnancy_days_lost_280_among_births_with_exact_days']}",
                f"- Completeness of BIRTH_WEIGHT_G: {summary['completeness_birth_weight_g']:.6f}",
                f"- Prevalence of LOW_BIRTH_WEIGHT_LT2500_FLAG: {summary['prevalence_low_birth_weight_lt2500_flag']:.6f}",
                f"- Prevalence of PRETERM_LOW_BIRTH_WEIGHT_FLAG: {summary['prevalence_preterm_low_birth_weight_flag']:.6f}",
                f"- Completeness of PRENATAL_VISITS_COUNT: {summary['completeness_prenatal_visits_count']:.6f}",
                f"- Prevalence of PRENATAL_VISITS_6PLUS_FLAG: {summary['prevalence_prenatal_visits_6plus_flag']:.6f}",
                f"- Prevalence of PRENATAL_VISITS_8PLUS_FLAG: {summary['prevalence_prenatal_visits_8plus_flag']:.6f}",
                f"- Completeness of MATERNAL_AGE_YEARS: {summary['completeness_maternal_age_years']:.6f}",
                f"- Completeness of MATERNAL_SCHOOLING_CODE_FINAL: {summary['completeness_maternal_schooling_code_final']:.6f}",
                f"- Completeness of PARITY_TOTAL_PRIOR: {summary['completeness_parity_total_prior']:.6f}",
                f"- Completeness of PLURALITY_CODE: {summary['completeness_plurality_code']:.6f}",
                f"- Completeness of NEWBORN_SEX_CODE: {summary['completeness_newborn_sex_code']:.6f}",
                f"- Completeness of MATERNAL_RACE_COLOR_CODE: {summary['completeness_maternal_race_color_code']:.6f}",
                f"- Completeness of CONGENITAL_ANOMALY_FLAG: {summary['completeness_congenital_anomaly_flag']:.6f}",
                f"- Event vs daily reconciliation result: `{summary['event_vs_daily_reconciliation_result']}`",
                "",
            ]
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def promote_sinasc_silver_range(
    root: Path,
    runtime_root: Path,
    start_year: int,
    end_year: int,
    log_path: Path,
    command: str,
    force: bool = False,
) -> dict[str, Any]:
    del force  # current canonical promotion always rewrites the affected year partitions
    years = list(range(start_year, end_year + 1))
    validations: dict[int, dict[str, Any]] = {}
    event_partition_paths: dict[int, str] = {}
    daily_partition_paths: dict[int, str] = {}

    for year in years:
        event_frame = build_canonical_sinasc_event_frame(load_sinasc_bronze_year(root, year), year)
        daily_frame = build_canonical_sinasc_daily_frame(event_frame)
        prewrite_validation = validate_canonical_sinasc_year(root, year, event_frame, daily_frame)
        if not prewrite_validation["all_checks_passed"]:
            raise ValueError(f"SINASC silver validation failed before writing partitions for {year}")

        event_output_path = _write_partition(root, runtime_root, sinasc_event_partition_file(root, year), event_frame[EVENT_OUTPUT_FIELDS].copy())
        daily_output_path = _write_partition(root, runtime_root, sinasc_daily_partition_file(root, year), daily_frame[DAILY_OUTPUT_FIELDS].copy())
        validation = validate_canonical_sinasc_year(
            root,
            year,
            event_frame,
            daily_frame,
            event_partition_path=event_output_path,
            daily_partition_path=daily_output_path,
        )
        if not validation["all_checks_passed"]:
            raise ValueError(f"SINASC silver validation failed after writing partitions for {year}")

        validations[year] = validation
        event_partition_paths[year] = str(event_output_path.relative_to(root))
        daily_partition_paths[year] = str(daily_output_path.relative_to(root))
        append_run_event(
            log_path,
            phase="phase_2_silver_promotion",
            dataset="sinasc",
            year=year,
            command=command,
            decision=(
                "canonical SINASC event and daily silver outputs written from the event-level pathway; "
                f"event path={event_output_path.relative_to(root)} daily path={daily_output_path.relative_to(root)}"
            ),
            status="PASS",
        )

    manifest = build_manifest(root, "silver", "sinasc")
    manifest_path = f"artifacts/manifests/silver_{manifest['dataset']}.manifest.json"
    _write_validation_reports(root, validations, manifest_path)
    return {
        "dataset": "sinasc",
        "years": years,
        "manifest_path": manifest_path,
        "event_partition_paths": event_partition_paths,
        "daily_partition_paths": daily_partition_paths,
        "validation": validations,
    }
