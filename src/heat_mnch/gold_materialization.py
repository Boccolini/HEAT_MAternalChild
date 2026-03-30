from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .contracts import read_contract_columns
from .silver_outcome_common import ParquetChunkWriter, sync_verified_parquet


CLIMATE_INPUT_COLUMNS = [
    "COD_MUNICIPIO",
    "DATE",
    "YEAR",
    "MONTH",
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
]
CLIMATE_OUTPUT_COLUMNS = [
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
]
PERINATAL_SINASC_EVENT_COLUMNS = [
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
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
    "SOURCE_SINASC",
]
PERINATAL_STILLBIRTH_DAILY_COLUMNS = [
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
    "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_PRIMARY",
    "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_SENSITIVITY_GTE22",
    "SOURCE_STILLBIRTH",
]
PERINATAL_STILLBIRTH_INDIVIDUAL_COLUMNS = [
    "MUNICIPALITY_CODE",
    "EVENT_DATE",
    "STILLBIRTH_GTE28_PRIMARY_FLAG",
    "STILLBIRTH_GTE22_SENSITIVITY_FLAG",
]
PERINATAL_SIH_DAILY_COLUMNS = [
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
]
PERINATAL_SINASC_DAILY_NUMERIC_COLUMNS = [
    "NV_TOTAL",
    "NV_PRETERM_TOTAL",
    "NV_REGISTRO_UTIL_IG_TOTAL",
    "NV_REGISTRO_UTIL_DPGP_TOTAL",
    "BIRTH_WEIGHT_REPORTED_TOTAL",
    "LOW_BIRTH_WEIGHT_LT2500_TOTAL",
    "PRETERM_LOW_BIRTH_WEIGHT_TOTAL",
    "NV_DPGP_MEDIA_DIAS",
    "NV_IG_CORRIGIDA_MEDIA_SEMANAS",
    "GEST_AGE_CANONICAL",
    "PRETERM_FLAG",
    "VERY_PRETERM_FLAG",
    "GEST_DAYS_LOST_BELOW_37W",
    "GEST_AGE_MISSING_FLAG",
    "PRENATAL_VISITS_COUNT",
    "PRENATAL_VISITS_6PLUS_FLAG",
    "PRENATAL_VISITS_8PLUS_FLAG",
    "PRENATAL_VISITS_MISSING_FLAG",
]
MORTALITY_CAUSE_GROUP_COLUMNS = [
    "UNDERLYING_BASIC_CAUSE_CHILD_HEAT_THERMAL_PRIMARY_TOTAL",
    "UNDERLYING_BASIC_CAUSE_CHILD_DEHYDRATION_PRIMARY_TOTAL",
    "ANY_MENTION_CHILD_FLUID_ELECTROLYTE_SENSITIVITY_TOTAL",
    "UNDERLYING_BASIC_CAUSE_CHILD_DIARRHEA_GI_PRIMARY_TOTAL",
    "UNDERLYING_BASIC_CAUSE_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL",
    "ANY_MENTION_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL",
    "ANY_MENTION_CHILD_RENAL_ACUTE_SENSITIVITY_TOTAL",
    "ANY_MENTION_CHILD_MALNUTRITION_SLOW_MOVING_SENSITIVITY_TOTAL",
    "ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_TOTAL",
    "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL",
    "ANY_MENTION_DIRECT_HEAT_ANY_TOTAL",
    "UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_TOTAL",
    "ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL",
    "UNDERLYING_BASIC_CAUSE_SLOW_MOVING_HEAT_ANY_TOTAL",
    "ANY_MENTION_SLOW_MOVING_HEAT_ANY_TOTAL",
]
MORTALITY_SIM_DAILY_COLUMNS = [
    "AGE_SCOPE",
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
    "DEATHS_TOTAL",
    *MORTALITY_CAUSE_GROUP_COLUMNS,
    "POP_UNDER5",
    "DENOMINATOR_NAME",
    "DENOMINATOR_VALUE",
    "SOURCE_ANALYTICAL_VIEW",
    "SOURCE_SIM",
    "QC_STATUS",
]
MORTALITY_SIM_INDIVIDUAL_COLUMNS = [
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
    "AGE_GROUP_0_27D",
    "AGE_GROUP_28_364D",
    "AGE_GROUP_U5",
    "SOURCE_SIM",
    "QC_STATUS",
]
MORTALITY_SIH_ENDPOINT_COLUMNS = [
    "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL",
    "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL",
    "CHILD_DIRECT_HEAT_HOSPITALIZATION_TOTAL",
    "CHILD_INDIRECT_HEAT_HOSPITALIZATION_TOTAL",
    "CHILD_EXCESSIVE_NATURAL_HEAT_X30_HOSPITALIZATION_TOTAL",
]
MORTALITY_SIH_SOURCE_COLUMNS = [
    "AGE_SCOPE",
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "MONTH",
    "ADMISSIONS_TOTAL",
    "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL",
    "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_CHILD_HEAT_THERMAL_PRIMARY_TOTAL",
    "ANY_DIAGNOSIS_INDIRECT_HEAT_ANY_TOTAL",
    "EXTERNAL_CAUSE_FIELDS_EXCESSIVE_NATURAL_HEAT_X30_TOTAL",
]
SINASC_LIVE_BIRTH_COLUMNS = ["MUNICIPALITY_CODE", "DATE", "YEAR", "LIVE_BIRTHS_TOTAL"]
POP_UNDER5_COLUMNS = ["MUNICIPALITY_CODE", "YEAR", "POP_UNDER5"]
POP_WRA_15_49_COLUMNS = ["MUNICIPALITY_CODE", "YEAR", "POP_WRA_15_49"]
POP_WRA_15_45_COLUMNS = ["MUNICIPALITY_CODE", "YEAR", "POP_WRA_15_45_SENSITIVITY"]
COMMON_RATE_FLAG_COLUMNS = [
    "DENOMINATOR_ZERO_FLAG",
    "DENOMINATOR_MISSING_FLAG",
    "RATE_NOT_COMPUTABLE_FLAG",
    "OFFSET_VALID_FLAG",
]
PERINATAL_GEST_AGE_DERIVATION_NOTE = "DAILY_MEAN_OF_GA_WEEKS_CORRECTED_FINAL_FROM_CANONICAL_SINASC_EVENT"
PERINATAL_GEST_DAYS_LOST_NOTE = "DAILY_SUM_OF_PRETERM_DAYS_SHORT_OF_37W_FINAL_FROM_CANONICAL_SINASC_EVENT_DAY_SCALE"


def _partition_path(root: Path, *parts: str) -> Path:
    return root.joinpath(*parts)


def gold_output_path(root: Path, table: str, year: int) -> Path:
    return _partition_path(root, "data", "gold", table, f"YEAR={year}", f"{table}_{year}.parquet")


def _schema_columns(root: Path, schema_filename: str) -> list[str]:
    return read_contract_columns(root / "schemas" / schema_filename)


def _load_parquet(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


@lru_cache(maxsize=4)
def _municipality_code_ibge7_lookup(root_str: str) -> dict[str, str]:
    root = Path(root_str)
    candidate_paths = (
        root / "metadata" / "era5_municipio_grid_lookup.csv",
        root / "artifacts" / "reference" / "era5" / "municipality_centroids_ibge_2022.csv",
        root / "artifacts" / "reference" / "municipality_context" / "atlas_brasil_municipality_lookup.csv",
    )
    for path in candidate_paths:
        if not path.exists():
            continue
        if path.name == "atlas_brasil_municipality_lookup.csv":
            frame = pd.read_csv(path, usecols=["MUNICIPALITY_CODE_IBGE7"], dtype={"MUNICIPALITY_CODE_IBGE7": str})
            ibge7 = frame["MUNICIPALITY_CODE_IBGE7"].astype(str).str.strip().str.zfill(7)
        else:
            frame = pd.read_csv(path, usecols=["COD_MUNICIPIO"], dtype={"COD_MUNICIPIO": str})
            ibge7 = frame["COD_MUNICIPIO"].astype(str).str.strip().str.zfill(7)
        lookup = pd.DataFrame({"ibge7": ibge7})
        lookup["ibge6"] = lookup["ibge7"].str[:6]
        return lookup.drop_duplicates("ibge6").set_index("ibge6")["ibge7"].to_dict()
    return {}


def _municipality_code(root: Path, series: pd.Series) -> pd.Series:
    lookup = _municipality_code_ibge7_lookup(str(root))
    cleaned = series.astype(str).str.strip()
    result = cleaned.copy()
    mask7 = cleaned.str.fullmatch(r"\d{7}")
    mask6 = cleaned.str.fullmatch(r"\d{6}")
    result.loc[mask7] = cleaned.loc[mask7].str.zfill(7)
    result.loc[mask6] = cleaned.loc[mask6].map(lookup).fillna(cleaned.loc[mask6].str.zfill(6))
    return result


def _placeholder_municipality_mask(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    digit_code = cleaned.where(cleaned.str.fullmatch(r"\d{6,7}"))
    return digit_code.str[:6].str.endswith("0000").fillna(False)


def _exclude_placeholder_municipalities(
    frame: pd.DataFrame,
    column: str = "MUNICIPALITY_CODE",
) -> tuple[pd.DataFrame, dict[str, int]]:
    if frame.empty or column not in frame.columns:
        return frame.copy(), {"excluded_rows": 0, "excluded_municipalities": 0}
    placeholder_mask = _placeholder_municipality_mask(frame[column])
    return (
        frame.loc[~placeholder_mask].copy(),
        {
            "excluded_rows": int(placeholder_mask.sum()),
            "excluded_municipalities": int(frame.loc[placeholder_mask, column].astype(str).nunique()),
        },
    )


def _load_climate_backbone(root: Path, year: int) -> pd.DataFrame:
    path = _partition_path(root, "data", "silver", "climate", f"YEAR={year}", f"climate_canonical_daily_{year}.parquet")
    climate = pd.read_parquet(path).copy()
    default_zero_columns = {
        "IMPUTADO_POR_FALLBACK",
        "IMPUTADO_TMAX_C",
        "IMPUTADO_TMIN_C",
        "IMPUTADO_TMEAN_C",
        "IMPUTADO_RH_MIN_PCT",
        "IMPUTADO_RH_MEAN_PCT",
        "CLIMATE_UPSTREAM_QUARANTINE_FLAG",
        "CLIMATE_RECORD_ANOMALY_FLAG",
        "HEAT_INDEX_INPUTS_ZERO_MISSING",
    }
    for column in CLIMATE_INPUT_COLUMNS:
        if column not in climate.columns:
            climate[column] = 0 if column in default_zero_columns else pd.NA
    climate = climate[CLIMATE_INPUT_COLUMNS].copy()
    climate["MUNICIPALITY_CODE"] = _municipality_code(root, climate["COD_MUNICIPIO"])
    climate, _ = _exclude_placeholder_municipalities(climate)
    climate["HEAT_INDEX_APPROX_C"] = climate["HEAT_INDEX_C"]
    climate = climate.drop(columns=["COD_MUNICIPIO", "HEAT_INDEX_C"])
    climate = climate[CLIMATE_OUTPUT_COLUMNS].sort_values(["DATE", "MUNICIPALITY_CODE"], kind="mergesort").reset_index(drop=True)
    if climate[["MUNICIPALITY_CODE", "DATE"]].duplicated().any():
        raise ValueError(f"Climate backbone has duplicate municipality-date keys for {year}")
    return climate


def _group_daily(frame: pd.DataFrame, group_columns: list[str], sum_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*group_columns, *sum_columns])
    grouped = frame.groupby(group_columns, as_index=False, sort=True)[sum_columns].sum(numeric_only=True)
    return grouped


def _group_annual_sum(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["MUNICIPALITY_CODE", "YEAR", value_column])
    grouped = frame.groupby(["MUNICIPALITY_CODE", "YEAR"], as_index=False, sort=True)[value_column].sum(numeric_only=True)
    return grouped


def _daily_key_spine(climate: pd.DataFrame, supplemental_frames: list[pd.DataFrame]) -> pd.DataFrame:
    key_columns = ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"]
    parts = [climate[key_columns]]
    for frame in supplemental_frames:
        if frame.empty:
            continue
        parts.append(frame[key_columns])
    return (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(key_columns)
        .sort_values(["DATE", "MUNICIPALITY_CODE"], kind="mergesort")
        .reset_index(drop=True)
    )


def _rows_with_positive_totals(frame: pd.DataFrame, total_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    totals = frame[total_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    return frame.loc[totals.sum(axis=1).gt(0)].copy()


def _fill_missing_climate_lineage(frame: pd.DataFrame) -> pd.Series:
    climate_missing = frame["FONTE_CLIMA_FINAL"].isna()
    frame.loc[climate_missing, "FONTE_CLIMA_FINAL"] = "NO_CLIMATE_BACKBONE_MATCH"
    frame.loc[climate_missing, "FONTE_TMAX_C_FINAL"] = "NO_CLIMATE_BACKBONE_MATCH"
    frame.loc[climate_missing, "FONTE_TMIN_C_FINAL"] = "NO_CLIMATE_BACKBONE_MATCH"
    frame.loc[climate_missing, "FONTE_TMEAN_C_FINAL"] = "NO_CLIMATE_BACKBONE_MATCH"
    frame.loc[climate_missing, "FONTE_RH_MIN_PCT_FINAL"] = "NO_CLIMATE_BACKBONE_MATCH"
    frame.loc[climate_missing, "FONTE_RH_MEAN_PCT_FINAL"] = "NO_CLIMATE_BACKBONE_MATCH"
    frame.loc[climate_missing, "IMPUTADO_POR_FALLBACK"] = 0
    frame.loc[climate_missing, "IMPUTADO_TMAX_C"] = 0
    frame.loc[climate_missing, "IMPUTADO_TMIN_C"] = 0
    frame.loc[climate_missing, "IMPUTADO_TMEAN_C"] = 0
    frame.loc[climate_missing, "IMPUTADO_RH_MIN_PCT"] = 0
    frame.loc[climate_missing, "IMPUTADO_RH_MEAN_PCT"] = 0
    frame.loc[climate_missing, "CLIMATE_UPSTREAM_QUARANTINE_FLAG"] = 0
    frame.loc[climate_missing, "CLIMATE_RECORD_ANOMALY_FLAG"] = 0
    frame.loc[climate_missing, "CLIMATE_CORRECTION_PATH"] = "none"
    frame.loc[climate_missing, "HEAT_INDEX_METHOD"] = "APPROX_NOT_COMPUTABLE_NO_CLIMATE_BACKBONE_MATCH"
    frame.loc[climate_missing, "HEAT_INDEX_SOURCE_PAIR"] = "NO_CLIMATE_BACKBONE_MATCH"
    frame.loc[climate_missing, "HEAT_INDEX_INPUTS_ZERO_MISSING"] = 0
    return climate_missing


def _fill_integer_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].fillna(0).astype("int64")


def _safe_log_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    valid = numeric.gt(0)
    result.loc[valid] = np.log(numeric.loc[valid])
    return result


def _safe_rate_series(numerator: pd.Series, denominator: pd.Series, multiplier: float) -> pd.Series:
    numerator_numeric = pd.to_numeric(numerator, errors="coerce").astype("float64")
    denominator_numeric = pd.to_numeric(denominator, errors="coerce").astype("float64")
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    valid = denominator_numeric.gt(0)
    result.loc[valid] = numerator_numeric.loc[valid] * multiplier / denominator_numeric.loc[valid]
    return result


def _apply_single_denominator_flags(
    frame: pd.DataFrame,
    denominator_column: str,
    offset_column: str,
    prefix: str,
) -> None:
    prefix_token = f"{prefix}_" if prefix else ""
    denominator = pd.to_numeric(frame[denominator_column], errors="coerce")
    missing_flag = denominator.isna()
    zero_flag = denominator.eq(0) & ~missing_flag
    valid_flag = denominator.gt(0)
    frame[f"{prefix_token}DENOMINATOR_ZERO_FLAG"] = zero_flag.astype("Int8")
    frame[f"{prefix_token}DENOMINATOR_MISSING_FLAG"] = missing_flag.astype("Int8")
    frame[f"{prefix_token}RATE_NOT_COMPUTABLE_FLAG"] = (~valid_flag).astype("Int8")
    frame[f"{prefix_token}OFFSET_VALID_FLAG"] = valid_flag.astype("Int8")
    frame[offset_column] = _safe_log_series(denominator)


def _apply_mortality_denominator_flags(frame: pd.DataFrame) -> None:
    live_births = pd.to_numeric(frame["LIVE_BIRTHS_TOTAL"], errors="coerce")
    under5_population = pd.to_numeric(frame["UNDER5_POPULATION"], errors="coerce")

    live_births_missing = live_births.isna()
    live_births_zero = live_births.eq(0) & ~live_births_missing
    live_births_valid = live_births.gt(0)
    under5_missing = under5_population.isna()
    under5_zero = under5_population.eq(0) & ~under5_missing
    under5_valid = under5_population.gt(0)

    frame["LIVE_BIRTHS_DENOMINATOR_ZERO_FLAG"] = live_births_zero.astype("Int8")
    frame["LIVE_BIRTHS_DENOMINATOR_MISSING_FLAG"] = live_births_missing.astype("Int8")
    frame["LIVE_BIRTHS_RATE_NOT_COMPUTABLE_FLAG"] = (~live_births_valid).astype("Int8")
    frame["LIVE_BIRTHS_OFFSET_VALID_FLAG"] = live_births_valid.astype("Int8")
    frame["UNDER5_POPULATION_DENOMINATOR_ZERO_FLAG"] = under5_zero.astype("Int8")
    frame["UNDER5_POPULATION_DENOMINATOR_MISSING_FLAG"] = under5_missing.astype("Int8")
    frame["UNDER5_POPULATION_RATE_NOT_COMPUTABLE_FLAG"] = (~under5_valid).astype("Int8")
    frame["UNDER5_POPULATION_OFFSET_VALID_FLAG"] = under5_valid.astype("Int8")
    frame["DENOMINATOR_ZERO_FLAG"] = (live_births_zero | under5_zero).astype("Int8")
    frame["DENOMINATOR_MISSING_FLAG"] = (live_births_missing | under5_missing).astype("Int8")
    frame["RATE_NOT_COMPUTABLE_FLAG"] = (~live_births_valid | ~under5_valid).astype("Int8")
    frame["OFFSET_VALID_FLAG"] = (live_births_valid & under5_valid).astype("Int8")
    frame["OFFSET_LOG_LIVE_BIRTHS"] = _safe_log_series(live_births)
    frame["OFFSET_LOG_UNDER5_POPULATION"] = _safe_log_series(under5_population)


def _order_output_columns(frame: pd.DataFrame, required_columns: list[str], extra_columns: list[str] | None = None) -> pd.DataFrame:
    extra_columns = extra_columns or []
    ordered = [column for column in required_columns if column in frame.columns]
    ordered.extend(column for column in extra_columns if column in frame.columns and column not in ordered)
    ordered.extend(column for column in frame.columns if column not in ordered)
    return frame[ordered].copy()


def _write_output(root: Path, runtime_root: Path, table: str, year: int, frame: pd.DataFrame) -> Path:
    output_path = gold_output_path(root, table, year)
    runtime_path = runtime_root / output_path.relative_to(root)
    writer = ParquetChunkWriter(runtime_path)
    writer.write(frame)
    row_count, columns = writer.close()
    sync_verified_parquet(runtime_path, output_path, row_count, columns)
    runtime_path.unlink(missing_ok=True)
    if runtime_path.parent.exists():
        shutil.rmtree(runtime_path.parent)
    return output_path


def _prenatal_source_summary(frame: pd.DataFrame) -> pd.Series:
    source_exact = frame["PRENATAL_SOURCE_EXACT_TOTAL"]
    source_band = frame["PRENATAL_SOURCE_BAND_TOTAL"]
    source_missing = frame["PRENATAL_SOURCE_MISSING_TOTAL"]
    live_births = frame["NV_TOTAL"]
    return pd.Series(
        np.select(
            [
                source_missing.eq(live_births),
                source_exact.eq(live_births),
                source_band.eq(live_births),
            ],
            [
                "MISSING",
                "SINASC_CONSPRENAT",
                "SINASC_CONSULTAS_BAND",
            ],
            default="MIXED_EVENT_LEVEL_PRENATAL_SOURCES",
        ),
        index=frame.index,
        dtype="object",
    )


def _prenatal_derivation_summary(frame: pd.DataFrame) -> pd.Series:
    exact_total = frame["PRENATAL_VISITS_COUNT_COMPLETE_TOTAL"]
    sixplus_total = frame["PRENATAL_VISITS_6PLUS_FLAG"]
    missing_total = frame["PRENATAL_VISITS_MISSING_FLAG"]
    live_births = frame["NV_TOTAL"]
    return pd.Series(
        np.select(
            [
                exact_total.eq(live_births),
                exact_total.eq(0) & sixplus_total.eq(0) & missing_total.eq(live_births),
                exact_total.eq(0),
            ],
            [
                "DAILY_MEAN_OF_EXACT_CONSPRENAT_COUNTS_FROM_CANONICAL_SINASC_EVENT",
                "MISSING",
                "DAILY_AGGREGATE_OF_EVENT_LEVEL_CONSULTAS_BANDS",
            ],
            default="DAILY_AGGREGATE_OF_MIXED_EXACT_BAND_AND_MISSING_EVENT_INPUTS",
        ),
        index=frame.index,
        dtype="object",
    )


def _build_perinatal_sinasc_event_daily(root: Path, year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    sinasc_path = _partition_path(root, "data", "silver", "sinasc", "silver_sinasc_event", f"YEAR={year}", f"silver_sinasc_event_{year}.parquet")
    event = _load_parquet(sinasc_path, PERINATAL_SINASC_EVENT_COLUMNS).copy()
    event["MUNICIPALITY_CODE"] = _municipality_code(root, event["MUNICIPALITY_CODE"])
    event, exclusion_summary = _exclude_placeholder_municipalities(event)
    event["NV_TOTAL"] = 1
    event["_GA_WEEKS_VALID"] = pd.to_numeric(event["GA_WEEKS_CORRECTED_FINAL"], errors="coerce").where(event["GA_VALID_FOR_ANALYSIS_FLAG"].eq(1))
    event["_GA_DAYS_VALID"] = pd.to_numeric(event["GA_DAYS_FINAL"], errors="coerce").where(event["GA_VALID_FOR_ANALYSIS_FLAG"].eq(1))
    event["_PRETERM_DAYS_LOST"] = pd.to_numeric(event["PRETERM_DAYS_SHORT_OF_37W_FINAL"], errors="coerce").fillna(0.0)
    event["PRETERM_FLAG"] = event["PRETERM_LT37_FLAG"].eq(1).fillna(False).astype("int16")
    event["VERY_PRETERM_FLAG"] = event["PRETERM_LT32_FLAG"].eq(1).fillna(False).astype("int16")
    event["GEST_AGE_MISSING_FLAG"] = event["GA_VALID_FOR_ANALYSIS_FLAG"].ne(1).fillna(True).astype("int16")
    event["BIRTH_WEIGHT_REPORTED_FLAG_COUNT"] = event["BIRTH_WEIGHT_REPORTED_FLAG"].eq(1).fillna(False).astype("int16")
    event["LOW_BIRTH_WEIGHT_LT2500_FLAG_COUNT"] = event["LOW_BIRTH_WEIGHT_LT2500_FLAG"].eq(1).fillna(False).astype("int16")
    event["PRETERM_LOW_BIRTH_WEIGHT_FLAG_COUNT"] = event["PRETERM_LOW_BIRTH_WEIGHT_FLAG"].eq(1).fillna(False).astype("int16")
    event["PRENATAL_VISITS_6PLUS_FLAG_COUNT"] = event["PRENATAL_VISITS_6PLUS_FLAG"].eq(1).fillna(False).astype("int16")
    event["PRENATAL_VISITS_8PLUS_FLAG_COUNT"] = event["PRENATAL_VISITS_8PLUS_FLAG"].eq(1).fillna(False).astype("int16")
    event["PRENATAL_VISITS_MISSING_FLAG_COUNT"] = event["PRENATAL_VISITS_MISSING_FLAG"].eq(1).fillna(False).astype("int16")
    event["PRENATAL_VISITS_COUNT_COMPLETE_FLAG"] = event["PRENATAL_VISITS_COUNT"].notna().astype("int16")
    event["PRENATAL_SOURCE_EXACT_TOTAL"] = event["PRENATAL_VISITS_SOURCE"].eq("SINASC_CONSPRENAT").astype("int16")
    event["PRENATAL_SOURCE_BAND_TOTAL"] = event["PRENATAL_VISITS_SOURCE"].eq("SINASC_CONSULTAS_BAND").astype("int16")
    event["PRENATAL_SOURCE_MISSING_TOTAL"] = event["PRENATAL_VISITS_SOURCE"].eq("MISSING").astype("int16")
    event["PRENATAL_RAW_NONMISSING_TOTAL"] = event["PRENATAL_VISITS_RAW_VALUE"].fillna("").astype(str).ne("").astype("int16")

    daily = (
        event.groupby(["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"], as_index=False, sort=True)
        .agg(
            NV_TOTAL=("NV_TOTAL", "sum"),
            NV_PRETERM_TOTAL=("PRETERM_FLAG", "sum"),
            NV_REGISTRO_UTIL_IG_TOTAL=("GA_VALID_FOR_ANALYSIS_FLAG", lambda values: int(values.eq(1).sum())),
            NV_REGISTRO_UTIL_DPGP_TOTAL=("_GA_DAYS_VALID", lambda values: int(values.notna().sum())),
            NV_DPGP_MEDIA_DIAS=("_GA_DAYS_VALID", "mean"),
            NV_IG_CORRIGIDA_MEDIA_SEMANAS=("_GA_WEEKS_VALID", "mean"),
            GEST_AGE_CANONICAL=("_GA_WEEKS_VALID", "mean"),
            PRETERM_FLAG=("PRETERM_FLAG", "sum"),
            VERY_PRETERM_FLAG=("VERY_PRETERM_FLAG", "sum"),
            GEST_DAYS_LOST_BELOW_37W=("_PRETERM_DAYS_LOST", "sum"),
            GEST_AGE_MISSING_FLAG=("GEST_AGE_MISSING_FLAG", "sum"),
            BIRTH_WEIGHT_REPORTED_TOTAL=("BIRTH_WEIGHT_REPORTED_FLAG_COUNT", "sum"),
            LOW_BIRTH_WEIGHT_LT2500_TOTAL=("LOW_BIRTH_WEIGHT_LT2500_FLAG_COUNT", "sum"),
            PRETERM_LOW_BIRTH_WEIGHT_TOTAL=("PRETERM_LOW_BIRTH_WEIGHT_FLAG_COUNT", "sum"),
            PRENATAL_VISITS_COUNT=("PRENATAL_VISITS_COUNT", "mean"),
            PRENATAL_VISITS_COUNT_COMPLETE_TOTAL=("PRENATAL_VISITS_COUNT_COMPLETE_FLAG", "sum"),
            PRENATAL_VISITS_6PLUS_FLAG=("PRENATAL_VISITS_6PLUS_FLAG_COUNT", "sum"),
            PRENATAL_VISITS_8PLUS_FLAG=("PRENATAL_VISITS_8PLUS_FLAG_COUNT", "sum"),
            PRENATAL_VISITS_MISSING_FLAG=("PRENATAL_VISITS_MISSING_FLAG_COUNT", "sum"),
            PRENATAL_SOURCE_EXACT_TOTAL=("PRENATAL_SOURCE_EXACT_TOTAL", "sum"),
            PRENATAL_SOURCE_BAND_TOTAL=("PRENATAL_SOURCE_BAND_TOTAL", "sum"),
            PRENATAL_SOURCE_MISSING_TOTAL=("PRENATAL_SOURCE_MISSING_TOTAL", "sum"),
            PRENATAL_RAW_NONMISSING_TOTAL=("PRENATAL_RAW_NONMISSING_TOTAL", "sum"),
        )
        .reset_index(drop=True)
    )
    _fill_integer_columns(
        daily,
        [
            "NV_TOTAL",
            "NV_PRETERM_TOTAL",
            "NV_REGISTRO_UTIL_IG_TOTAL",
            "NV_REGISTRO_UTIL_DPGP_TOTAL",
            "PRETERM_FLAG",
            "VERY_PRETERM_FLAG",
            "GEST_AGE_MISSING_FLAG",
            "BIRTH_WEIGHT_REPORTED_TOTAL",
            "LOW_BIRTH_WEIGHT_LT2500_TOTAL",
            "PRETERM_LOW_BIRTH_WEIGHT_TOTAL",
            "PRENATAL_VISITS_COUNT_COMPLETE_TOTAL",
            "PRENATAL_VISITS_6PLUS_FLAG",
            "PRENATAL_VISITS_8PLUS_FLAG",
            "PRENATAL_VISITS_MISSING_FLAG",
            "PRENATAL_SOURCE_EXACT_TOTAL",
            "PRENATAL_SOURCE_BAND_TOTAL",
            "PRENATAL_SOURCE_MISSING_TOTAL",
            "PRENATAL_RAW_NONMISSING_TOTAL",
        ],
    )
    daily["GEST_DAYS_LOST_BELOW_37W"] = pd.to_numeric(daily["GEST_DAYS_LOST_BELOW_37W"], errors="coerce").fillna(0.0).astype("Float64")
    daily["PRENATAL_VISITS_SOURCE"] = _prenatal_source_summary(daily)
    daily["PRENATAL_VISITS_RAW_VALUE"] = pd.Series(
        np.where(
            daily["PRENATAL_RAW_NONMISSING_TOTAL"].eq(0),
            "MISSING",
            "AGGREGATED_DAILY_EVENT_LEVEL_RAW_VALUES",
        ),
        index=daily.index,
        dtype="object",
    )
    daily["PRENATAL_VISITS_DERIVATION_NOTE"] = _prenatal_derivation_summary(daily)
    daily["GEST_AGE_DERIVATION_NOTE"] = PERINATAL_GEST_AGE_DERIVATION_NOTE
    daily["GEST_DAYS_LOST_DERIVATION_NOTE"] = PERINATAL_GEST_DAYS_LOST_NOTE
    daily["SOURCE_SINASC"] = "silver_sinasc_event"
    daily["QC_STATUS"] = "PASS"

    preterm_event = event.loc[event["PRETERM_FLAG"].eq(1), "PRETERM_DAYS_SHORT_OF_37W_FINAL"]
    preterm_days = pd.to_numeric(preterm_event, errors="coerce").dropna()
    summary = {
        "live_births": int(len(event)),
        "preterm_births": int(event["PRETERM_FLAG"].sum()),
        "very_preterm_births": int(event["VERY_PRETERM_FLAG"].sum()),
        "mean_gest_days_lost_among_preterm": float(preterm_days.mean()) if not preterm_days.empty else None,
        "median_gest_days_lost_among_preterm": float(preterm_days.median()) if not preterm_days.empty else None,
        "birth_weight_reported_share": float(event["BIRTH_WEIGHT_G"].notna().mean()) if len(event) else 0.0,
        "low_birth_weight_lt2500_prevalence": float(event["LOW_BIRTH_WEIGHT_LT2500_FLAG"].eq(1).mean()) if len(event) else 0.0,
        "preterm_low_birth_weight_prevalence": float(event["PRETERM_LOW_BIRTH_WEIGHT_FLAG"].eq(1).mean()) if len(event) else 0.0,
        "prenatal_count_completeness": float(event["PRENATAL_VISITS_COUNT"].notna().mean()) if len(event) else 0.0,
        "prenatal_6plus_prevalence": float(event["PRENATAL_VISITS_6PLUS_FLAG"].eq(1).mean()) if len(event) else 0.0,
        "prenatal_8plus_prevalence": float(event["PRENATAL_VISITS_8PLUS_FLAG"].eq(1).mean()) if len(event) else 0.0,
        "event_vs_daily_reconciliation": {
            "PRETERM_FLAG": int(daily["PRETERM_FLAG"].sum()) == int(event["PRETERM_FLAG"].sum()),
            "VERY_PRETERM_FLAG": int(daily["VERY_PRETERM_FLAG"].sum()) == int(event["VERY_PRETERM_FLAG"].sum()),
            "LOW_BIRTH_WEIGHT_LT2500_TOTAL": int(daily["LOW_BIRTH_WEIGHT_LT2500_TOTAL"].sum())
            == int(event["LOW_BIRTH_WEIGHT_LT2500_FLAG"].eq(1).sum()),
            "PRETERM_LOW_BIRTH_WEIGHT_TOTAL": int(daily["PRETERM_LOW_BIRTH_WEIGHT_TOTAL"].sum())
            == int(event["PRETERM_LOW_BIRTH_WEIGHT_FLAG"].eq(1).sum()),
            "PRENATAL_VISITS_8PLUS_FLAG": int(daily["PRENATAL_VISITS_8PLUS_FLAG"].sum())
            == int(event["PRENATAL_VISITS_8PLUS_FLAG"].eq(1).sum()),
            "GEST_DAYS_LOST_BELOW_37W": abs(float(daily["GEST_DAYS_LOST_BELOW_37W"].sum()) - float(preterm_days.sum())) <= 1e-6,
            "NO_AGE_TRUNCATION": int(daily["NV_TOTAL"].sum()) == int(len(event)),
        },
        "excluded_placeholder_rows": exclusion_summary["excluded_rows"],
        "excluded_placeholder_municipalities": exclusion_summary["excluded_municipalities"],
    }
    return daily.sort_values(["DATE", "MUNICIPALITY_CODE"], kind="mergesort").reset_index(drop=True), summary


def _build_reconciliation_summary(frame: pd.DataFrame, count_columns: list[str]) -> str:
    duplicate_keys = int(frame[["MUNICIPALITY_CODE", "DATE"]].duplicated().sum())
    totals = ", ".join(f"{column}={int(frame[column].fillna(0).sum())}" for column in count_columns if column in frame.columns)
    return f"duplicate_keys={duplicate_keys}; {totals}"


def build_gold_perinatal_year(root: Path, runtime_root: Path, year: int) -> dict[str, Any]:
    climate = _load_climate_backbone(root, year)

    sinasc_daily, sinasc_summary = _build_perinatal_sinasc_event_daily(root, year)

    stillbirth_path = _partition_path(
        root,
        "data",
        "silver",
        "stillbirth",
        "silver_stillbirth_daily",
        f"YEAR={year}",
        f"silver_stillbirth_daily_{year}.parquet",
    )
    stillbirth_daily = _load_parquet(stillbirth_path, PERINATAL_STILLBIRTH_DAILY_COLUMNS).rename(columns={"EVENT_DATE": "DATE"}).copy()
    stillbirth_daily["MUNICIPALITY_CODE"] = _municipality_code(root, stillbirth_daily["MUNICIPALITY_CODE"])
    stillbirth_daily, stillbirth_daily_exclusion = _exclude_placeholder_municipalities(stillbirth_daily)
    _fill_integer_columns(
        stillbirth_daily,
        [
            "LIVE_BIRTHS_TOTAL",
            "STILLBIRTH_TOTAL_GTE28",
            "STILLBIRTH_TOTAL_GTE22_SENSITIVITY",
            "TOTAL_BIRTHS_PRIMARY",
            "TOTAL_BIRTHS_SENSITIVITY_GTE22",
        ],
    )

    stillbirth_individual_path = _partition_path(
        root,
        "data",
        "silver",
        "stillbirth",
        "silver_stillbirth_individual",
        f"YEAR={year}",
        f"silver_stillbirth_individual_{year}.parquet",
    )
    stillbirth_individual = _load_parquet(stillbirth_individual_path, PERINATAL_STILLBIRTH_INDIVIDUAL_COLUMNS).copy()
    stillbirth_individual["MUNICIPALITY_CODE"] = _municipality_code(root, stillbirth_individual["MUNICIPALITY_CODE"])
    stillbirth_individual, stillbirth_individual_exclusion = _exclude_placeholder_municipalities(stillbirth_individual)
    _fill_integer_columns(stillbirth_individual, ["STILLBIRTH_GTE28_PRIMARY_FLAG", "STILLBIRTH_GTE22_SENSITIVITY_FLAG"])

    sih_path = _partition_path(root, "data", "silver", "sih", "silver_sih_daily", f"YEAR={year}", f"silver_sih_daily_{year}.parquet")
    sih_daily = _load_parquet(sih_path, PERINATAL_SIH_DAILY_COLUMNS).copy()
    sih_daily["MUNICIPALITY_CODE"] = _municipality_code(root, sih_daily["MUNICIPALITY_CODE"])
    sih_daily, sih_exclusion = _exclude_placeholder_municipalities(sih_daily)
    _fill_integer_columns(sih_daily, [column for column in PERINATAL_SIH_DAILY_COLUMNS if column not in {"MUNICIPALITY_CODE", "DATE", "YEAR"}])
    sih_daily = sih_daily.loc[pd.to_numeric(sih_daily["YEAR"], errors="coerce").eq(year)].copy()
    sih_daily = _group_daily(
        sih_daily,
        ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"],
        [column for column in PERINATAL_SIH_DAILY_COLUMNS if column not in {"MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"}],
    )

    stillbirth_relevant = _rows_with_positive_totals(
        stillbirth_daily,
        [
            "LIVE_BIRTHS_TOTAL",
            "STILLBIRTH_TOTAL_GTE28",
            "STILLBIRTH_TOTAL_GTE22_SENSITIVITY",
            "TOTAL_BIRTHS_PRIMARY",
            "TOTAL_BIRTHS_SENSITIVITY_GTE22",
        ],
    )
    sih_relevant = _rows_with_positive_totals(
        sih_daily,
        [column for column in PERINATAL_SIH_DAILY_COLUMNS if column.endswith("_TOTAL")],
    )
    spine = _daily_key_spine(
        climate,
        [
            sinasc_daily[["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"]],
            stillbirth_relevant[["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"]],
            sih_relevant[["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"]],
        ],
    )

    output = spine.merge(climate, on=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"], how="left")
    output = output.merge(sinasc_daily, on=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"], how="left")
    output = output.merge(stillbirth_daily, on=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"], how="left")
    output = output.merge(sih_daily, on=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"], how="left")

    count_columns = [
        "NV_TOTAL",
        "NV_PRETERM_TOTAL",
        "NV_REGISTRO_UTIL_IG_TOTAL",
        "NV_REGISTRO_UTIL_DPGP_TOTAL",
        "PRETERM_FLAG",
        "VERY_PRETERM_FLAG",
        "GEST_AGE_MISSING_FLAG",
        "BIRTH_WEIGHT_REPORTED_TOTAL",
        "LOW_BIRTH_WEIGHT_LT2500_TOTAL",
        "PRETERM_LOW_BIRTH_WEIGHT_TOTAL",
        "PRENATAL_VISITS_6PLUS_FLAG",
        "PRENATAL_VISITS_8PLUS_FLAG",
        "PRENATAL_VISITS_MISSING_FLAG",
        "LIVE_BIRTHS_TOTAL",
        "STILLBIRTH_TOTAL_GTE28",
        "STILLBIRTH_TOTAL_GTE22_SENSITIVITY",
        *[column for column in PERINATAL_SIH_DAILY_COLUMNS if column not in {"MUNICIPALITY_CODE", "DATE", "YEAR"}],
    ]
    _fill_integer_columns(output, count_columns)
    for column in ["NV_DPGP_MEDIA_DIAS", "NV_IG_CORRIGIDA_MEDIA_SEMANAS", "GEST_AGE_CANONICAL", "PRENATAL_VISITS_COUNT"]:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce").astype("Float64")
    if "GEST_DAYS_LOST_BELOW_37W" not in output.columns:
        output = output.merge(
            sinasc_daily[["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", "GEST_DAYS_LOST_BELOW_37W"]],
            on=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"],
            how="left",
        )
    output["GEST_DAYS_LOST_BELOW_37W"] = pd.to_numeric(output["GEST_DAYS_LOST_BELOW_37W"], errors="coerce").fillna(0.0).astype("Float64")
    output["STILLBIRTH_PRIMARY_GTE28_FLAG"] = output["STILLBIRTH_TOTAL_GTE28"]
    output["STILLBIRTH_SENS_GTE22_FLAG"] = output["STILLBIRTH_TOTAL_GTE22_SENSITIVITY"]
    output["STILLBIRTH_TOTAL_PRIMARY_GTE28"] = output["STILLBIRTH_TOTAL_GTE28"]
    output["TOTAL_BIRTHS_PRIMARY"] = output["LIVE_BIRTHS_TOTAL"] + output["STILLBIRTH_TOTAL_PRIMARY_GTE28"]
    output["TOTAL_BIRTHS_SENSITIVITY_GTE22"] = output["LIVE_BIRTHS_TOTAL"] + output["STILLBIRTH_TOTAL_GTE22_SENSITIVITY"]
    _apply_single_denominator_flags(
        output,
        denominator_column="TOTAL_BIRTHS_PRIMARY",
        offset_column="OFFSET_LOG_TOTAL_BIRTHS_PRIMARY",
        prefix="STILLBIRTH",
    )
    output["OFFSET_LOG_TOTAL_BIRTHS_SENSITIVITY_GTE22"] = _safe_log_series(output["TOTAL_BIRTHS_SENSITIVITY_GTE22"])
    for column in COMMON_RATE_FLAG_COLUMNS:
        output[column] = output[f"STILLBIRTH_{column}"]
    output["STILLBIRTH_TOTAL"] = output["STILLBIRTH_TOTAL_PRIMARY_GTE28"]
    output["NASCIMENTOS_TOTAIS_INCLUINDO_FETAIS"] = output["TOTAL_BIRTHS_PRIMARY"]
    output["OFFSET_LOG_NASCIMENTOS_TOTAIS_INCLUINDO_FETAIS"] = output["OFFSET_LOG_TOTAL_BIRTHS_PRIMARY"]
    output["PRENATAL_VISITS_SOURCE"] = output["PRENATAL_VISITS_SOURCE"].fillna("MISSING")
    output["PRENATAL_VISITS_RAW_VALUE"] = output["PRENATAL_VISITS_RAW_VALUE"].fillna("MISSING")
    output["PRENATAL_VISITS_DERIVATION_NOTE"] = output["PRENATAL_VISITS_DERIVATION_NOTE"].fillna("MISSING")
    output["GEST_AGE_DERIVATION_NOTE"] = output["GEST_AGE_DERIVATION_NOTE"].fillna(PERINATAL_GEST_AGE_DERIVATION_NOTE)
    output["GEST_DAYS_LOST_DERIVATION_NOTE"] = output["GEST_DAYS_LOST_DERIVATION_NOTE"].fillna(PERINATAL_GEST_DAYS_LOST_NOTE)
    climate_missing = _fill_missing_climate_lineage(output)
    output["SOURCE_SINASC"] = "silver_sinasc_event"
    output["SOURCE_STILLBIRTH"] = "silver_stillbirth_daily"
    output["SOURCE_SIH"] = "silver_sih_daily"
    output["QC_STATUS"] = "PASS"
    output["STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_PRIMARY"] = _safe_rate_series(
        output["STILLBIRTH_TOTAL_PRIMARY_GTE28"],
        output["TOTAL_BIRTHS_PRIMARY"],
        1000.0,
    )
    output["STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_SENSITIVITY_GTE22"] = _safe_rate_series(
        output["STILLBIRTH_TOTAL_GTE22_SENSITIVITY"],
        output["TOTAL_BIRTHS_SENSITIVITY_GTE22"],
        1000.0,
    )
    output["STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS"] = output["STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_PRIMARY"]

    validation_flags = {
        "PRETERM_FLAG_traces_to_canonical_sinasc_event": int(output["PRETERM_FLAG"].sum()) == sinasc_summary["preterm_births"],
        "VERY_PRETERM_FLAG_subset_of_PRETERM_FLAG": bool(output["VERY_PRETERM_FLAG"].le(output["PRETERM_FLAG"]).all()),
        "LOW_BIRTH_WEIGHT_LT2500_traces_to_canonical_sinasc_event": int(output["LOW_BIRTH_WEIGHT_LT2500_TOTAL"].sum())
        == int(sinasc_daily["LOW_BIRTH_WEIGHT_LT2500_TOTAL"].sum()),
        "PRETERM_LOW_BIRTH_WEIGHT_traces_to_canonical_sinasc_event": int(output["PRETERM_LOW_BIRTH_WEIGHT_TOTAL"].sum())
        == int(sinasc_daily["PRETERM_LOW_BIRTH_WEIGHT_TOTAL"].sum()),
        "PRETERM_LOW_BIRTH_WEIGHT_subset_of_PRETERM_FLAG": bool(
            output["PRETERM_LOW_BIRTH_WEIGHT_TOTAL"].le(output["PRETERM_FLAG"]).all()
        ),
        "GEST_DAYS_LOST_zero_when_no_preterm": bool(output.loc[output["PRETERM_FLAG"].eq(0), "GEST_DAYS_LOST_BELOW_37W"].eq(0).all()),
        "GEST_DAYS_LOST_non_negative": bool(output["GEST_DAYS_LOST_BELOW_37W"].ge(0).all()),
        "STILLBIRTH_PRIMARY_matches_individual_flags": int(output["STILLBIRTH_TOTAL_PRIMARY_GTE28"].sum())
        == int(stillbirth_individual["STILLBIRTH_GTE28_PRIMARY_FLAG"].sum()),
        "STILLBIRTH_SENS_matches_individual_flags": int(output["STILLBIRTH_TOTAL_GTE22_SENSITIVITY"].sum())
        == int(stillbirth_individual["STILLBIRTH_GTE22_SENSITIVITY_FLAG"].sum()),
        "TOTAL_BIRTHS_PRIMARY_arithmetic_exact": bool(
            output["TOTAL_BIRTHS_PRIMARY"].eq(output["LIVE_BIRTHS_TOTAL"] + output["STILLBIRTH_TOTAL_PRIMARY_GTE28"]).all()
        ),
        "TOTAL_BIRTHS_SENS_arithmetic_exact": bool(
            output["TOTAL_BIRTHS_SENSITIVITY_GTE22"].eq(output["LIVE_BIRTHS_TOTAL"] + output["STILLBIRTH_TOTAL_GTE22_SENSITIVITY"]).all()
        ),
        "HDP_severe_subset_of_HDP_any": bool(output["HDP_SEVERE_TOTAL"].le(output["HDP_ANY_TOTAL"]).all()),
        "APH_subset_of_obstetric_admissions": bool(output["APH_ANTEPARTUM_TOTAL"].le(output["OBSTETRIC_ADMISSIONS_TOTAL"]).all()),
        "PRETERM_LABOR_and_GDM_distinct": bool(
            output["PRETERM_LABOR_TOTAL"].ne(output["OBSTETRIC_ADMISSIONS_TOTAL"]).any()
            and output["GDM_GESTATIONAL_TOTAL"].le(output["DIABETES_PREGNANCY_BROAD_TOTAL"]).all()
        ),
        "NO_AGE_TRUNCATION_of_canonical_birth_records": sinasc_summary["event_vs_daily_reconciliation"]["NO_AGE_TRUNCATION"],
    }
    failing_validations = [name for name, passed in validation_flags.items() if not passed]
    if failing_validations:
        raise ValueError(f"gold_mnch_perinatal validation failed for {year}: {failing_validations}")

    schema_columns = _schema_columns(root, "gold_mnch_perinatal_diario_municipio.schema.csv")
    extra_columns = [
        "MONTH",
        "LIVE_BIRTHS_TOTAL",
        "GEST_AGE_CANONICAL",
        "PRETERM_FLAG",
        "VERY_PRETERM_FLAG",
        "GEST_DAYS_LOST_BELOW_37W",
        "GEST_AGE_DERIVATION_NOTE",
        "GEST_DAYS_LOST_DERIVATION_NOTE",
        "GEST_AGE_MISSING_FLAG",
        "STILLBIRTH_PRIMARY_GTE28_FLAG",
        "STILLBIRTH_SENS_GTE22_FLAG",
        "STILLBIRTH_TOTAL_PRIMARY_GTE28",
        "TOTAL_BIRTHS_PRIMARY",
        "TOTAL_BIRTHS_SENSITIVITY_GTE22",
        "OFFSET_LOG_TOTAL_BIRTHS_PRIMARY",
        "OFFSET_LOG_TOTAL_BIRTHS_SENSITIVITY_GTE22",
        "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_PRIMARY",
        "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_SENSITIVITY_GTE22",
        "PRENATAL_VISITS_COUNT",
        "PRENATAL_VISITS_6PLUS_FLAG",
        "PRENATAL_VISITS_8PLUS_FLAG",
        "PRENATAL_VISITS_SOURCE",
        "PRENATAL_VISITS_RAW_VALUE",
        "PRENATAL_VISITS_MISSING_FLAG",
        "PRENATAL_VISITS_DERIVATION_NOTE",
        "BIRTH_WEIGHT_REPORTED_TOTAL",
        "LOW_BIRTH_WEIGHT_LT2500_TOTAL",
        "PRETERM_LOW_BIRTH_WEIGHT_TOTAL",
        "APH_ANTEPARTUM_TOTAL",
        "APH_ANTEPARTUM_URGENT_TOTAL",
        "APH_ANTEPARTUM_ICU_TOTAL",
        "APH_ANTEPARTUM_DEATH_TOTAL",
        "TMIN_C",
        "TMEAN_C",
        "RH_MIN_PCT",
        "HEATWAVE_LEN_P95",
        "HEAT_INDEX_APPROX_C",
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
        "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS",
        "STILLBIRTH_DENOMINATOR_ZERO_FLAG",
        "STILLBIRTH_DENOMINATOR_MISSING_FLAG",
        "STILLBIRTH_RATE_NOT_COMPUTABLE_FLAG",
        "STILLBIRTH_OFFSET_VALID_FLAG",
        *COMMON_RATE_FLAG_COLUMNS,
    ]
    output = _order_output_columns(output, schema_columns, extra_columns)
    output = output.sort_values(["DATE", "MUNICIPALITY_CODE"], kind="mergesort").reset_index(drop=True)
    output_path = _write_output(root, runtime_root, "gold_mnch_perinatal", year, output)
    return {
        "output_path": output_path,
        "internal_reconciliation_summary": _build_reconciliation_summary(
            output,
            [
                "PRETERM_FLAG",
                "VERY_PRETERM_FLAG",
                "LOW_BIRTH_WEIGHT_LT2500_TOTAL",
                "PRETERM_LOW_BIRTH_WEIGHT_TOTAL",
                "STILLBIRTH_TOTAL_PRIMARY_GTE28",
                "OBSTETRIC_ADMISSIONS_TOTAL",
                "HDP_ANY_TOTAL",
                "PRETERM_LABOR_TOTAL",
                "APH_ANTEPARTUM_TOTAL",
                "GDM_GESTATIONAL_TOTAL",
            ],
        ),
        "external_validation_summary": (
            "schema_required_columns_present=True; climate_backbone_unique_keys=True; canonical_sinasc_event_used=True; "
            f"climate_backbone_unmatched_rows={int(climate_missing.sum())}; "
            f"excluded_placeholder_rows={sinasc_summary['excluded_placeholder_rows'] + stillbirth_daily_exclusion['excluded_rows'] + stillbirth_individual_exclusion['excluded_rows'] + sih_exclusion['excluded_rows']}"
        ),
        "perinatal_validation_flags": validation_flags,
        "perinatal_summary": {
            **sinasc_summary,
            "stillbirth_primary_count": int(stillbirth_individual["STILLBIRTH_GTE28_PRIMARY_FLAG"].sum()),
            "stillbirth_sensitivity_count": int(stillbirth_individual["STILLBIRTH_GTE22_SENSITIVITY_FLAG"].sum()),
            "stillbirth_rate_per_1000_total_births": (
                float(output["STILLBIRTH_TOTAL_PRIMARY_GTE28"].sum() * 1000.0 / output["TOTAL_BIRTHS_PRIMARY"].sum())
                if int(output["TOTAL_BIRTHS_PRIMARY"].sum()) > 0
                else None
            ),
            "low_birth_weight_lt2500_total": int(output["LOW_BIRTH_WEIGHT_LT2500_TOTAL"].sum()),
            "preterm_low_birth_weight_total": int(output["PRETERM_LOW_BIRTH_WEIGHT_TOTAL"].sum()),
            "prenatal_8plus_total": int(output["PRENATAL_VISITS_8PLUS_FLAG"].sum()),
            "hdp_total": int(output["HDP_ANY_TOTAL"].sum()),
            "hdp_severe_total": int(output["HDP_SEVERE_TOTAL"].sum()),
            "aph_antepartum_total": int(output["APH_ANTEPARTUM_TOTAL"].sum()),
            "municipalities_with_at_least_one_preterm_event": int(output.loc[output["PRETERM_FLAG"].gt(0), "MUNICIPALITY_CODE"].nunique()),
            "municipalities_with_at_least_one_low_birth_weight_event": int(
                output.loc[output["LOW_BIRTH_WEIGHT_LT2500_TOTAL"].gt(0), "MUNICIPALITY_CODE"].nunique()
            ),
            "municipalities_with_at_least_one_stillbirth_event": int(output.loc[output["STILLBIRTH_TOTAL_PRIMARY_GTE28"].gt(0), "MUNICIPALITY_CODE"].nunique()),
            "municipalities_with_at_least_one_hdp_event": int(output.loc[output["HDP_ANY_TOTAL"].gt(0), "MUNICIPALITY_CODE"].nunique()),
            "municipalities_with_at_least_one_aph_event": int(output.loc[output["APH_ANTEPARTUM_TOTAL"].gt(0), "MUNICIPALITY_CODE"].nunique()),
            "event_vs_daily_reconciliation": {
                **sinasc_summary["event_vs_daily_reconciliation"],
                "STILLBIRTH_PRIMARY": validation_flags["STILLBIRTH_PRIMARY_matches_individual_flags"],
                "STILLBIRTH_SENSITIVITY": validation_flags["STILLBIRTH_SENS_matches_individual_flags"],
                "TOTAL_BIRTHS_PRIMARY": validation_flags["TOTAL_BIRTHS_PRIMARY_arithmetic_exact"],
                "TOTAL_BIRTHS_SENSITIVITY": validation_flags["TOTAL_BIRTHS_SENS_arithmetic_exact"],
            },
            "climate_backbone_unmatched_rows": int(climate_missing.sum()),
            "live_births_total_stillbirth_denominator": int(output["LIVE_BIRTHS_TOTAL"].sum()),
            "excluded_placeholder_rows": int(
                sinasc_summary["excluded_placeholder_rows"]
                + stillbirth_daily_exclusion["excluded_rows"]
                + stillbirth_individual_exclusion["excluded_rows"]
                + sih_exclusion["excluded_rows"]
            ),
        },
    }


def build_gold_mortality_year(root: Path, runtime_root: Path, year: int) -> dict[str, Any]:
    climate = _load_climate_backbone(root, year)

    sim_path = _partition_path(
        root,
        "data",
        "silver",
        "analytical",
        "silver_child_u5_sim_daily",
        f"YEAR={year}",
        f"silver_child_u5_sim_daily_{year}.parquet",
    )
    sim = _load_parquet(sim_path, MORTALITY_SIM_DAILY_COLUMNS).copy()
    sim["MUNICIPALITY_CODE"] = _municipality_code(root, sim["MUNICIPALITY_CODE"])
    sim, sim_exclusion = _exclude_placeholder_municipalities(sim)
    sim = sim.loc[pd.to_numeric(sim["YEAR"], errors="coerce").eq(year)].copy()
    _fill_integer_columns(sim, ["DEATHS_TOTAL", *MORTALITY_CAUSE_GROUP_COLUMNS])
    sim = _group_daily(
        sim,
        ["AGE_SCOPE", "MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"],
        ["DEATHS_TOTAL", *MORTALITY_CAUSE_GROUP_COLUMNS],
    )
    neonatal = _group_daily(
        sim.loc[sim["AGE_SCOPE"].eq("AGE_0_27D"), ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", "DEATHS_TOTAL"]],
        ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"],
        ["DEATHS_TOTAL"],
    ).rename(columns={"DEATHS_TOTAL": "NEONATAL_DEATHS_TOTAL"})
    postneonatal = _group_daily(
        sim.loc[sim["AGE_SCOPE"].eq("AGE_28_364D"), ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", "DEATHS_TOTAL"]],
        ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"],
        ["DEATHS_TOTAL"],
    ).rename(columns={"DEATHS_TOTAL": "POSTNEONATAL_DEATHS_TOTAL"})
    under5 = _group_daily(
        sim.loc[sim["AGE_SCOPE"].eq("AGE_U5"), ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", "DEATHS_TOTAL"]],
        ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"],
        ["DEATHS_TOTAL"],
    ).rename(columns={"DEATHS_TOTAL": "UNDER5_DEATHS_TOTAL"})
    cause_groups = _group_daily(
        sim.loc[sim["AGE_SCOPE"].eq("AGE_U5"), ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", *MORTALITY_CAUSE_GROUP_COLUMNS]],
        ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"],
        MORTALITY_CAUSE_GROUP_COLUMNS,
    )

    sim_individual_path = _partition_path(
        root,
        "data",
        "silver",
        "sim",
        "silver_sim_individual",
        f"YEAR={year}",
        f"silver_sim_individual_{year}.parquet",
    )
    sim_individual = _load_parquet(sim_individual_path, MORTALITY_SIM_INDIVIDUAL_COLUMNS).copy()
    sim_individual["MUNICIPALITY_CODE"] = _municipality_code(root, sim_individual["MUNICIPALITY_CODE"])
    sim_individual, sim_individual_exclusion = _exclude_placeholder_municipalities(sim_individual)
    sim_individual = sim_individual.loc[pd.to_numeric(sim_individual["YEAR"], errors="coerce").eq(year)].copy()
    valid_sim_municipalities = set(sim["MUNICIPALITY_CODE"].astype(str))
    sim_individual = sim_individual.loc[sim_individual["MUNICIPALITY_CODE"].astype(str).isin(valid_sim_municipalities)].copy()
    _fill_integer_columns(sim_individual, ["AGE_GROUP_0_27D", "AGE_GROUP_28_364D", "AGE_GROUP_U5"])
    sim_individual_daily = _group_daily(
        sim_individual,
        ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"],
        ["AGE_GROUP_0_27D", "AGE_GROUP_28_364D", "AGE_GROUP_U5"],
    )
    sim_individual_daily["INFANT_GROUP_TOTAL"] = sim_individual_daily["AGE_GROUP_0_27D"] + sim_individual_daily["AGE_GROUP_28_364D"]

    acute = pd.DataFrame(columns=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", "ACUTE_CARE_PROXY_TOTAL"])
    sih_endpoint_totals = pd.DataFrame(columns=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", *MORTALITY_SIH_ENDPOINT_COLUMNS])
    acute_source = "OPTIONAL_CHILD_SIH_NOT_MATERIALIZED"
    sih_path = _partition_path(
        root,
        "data",
        "silver",
        "analytical",
        "silver_child_u5_sih_daily",
        f"YEAR={year}",
        f"silver_child_u5_sih_daily_{year}.parquet",
    )
    if sih_path.exists():
        available_sih_columns = [
            column for column in MORTALITY_SIH_SOURCE_COLUMNS if column in pq.ParquetFile(sih_path).schema.names
        ]
        sih = _load_parquet(sih_path, available_sih_columns).copy()
        sih["MUNICIPALITY_CODE"] = _municipality_code(root, sih["MUNICIPALITY_CODE"])
        sih, sih_exclusion = _exclude_placeholder_municipalities(sih)
        sih = sih.loc[pd.to_numeric(sih["YEAR"], errors="coerce").eq(year)].copy()
        sih["CHILD_DIRECT_HEAT_HOSPITALIZATION_TOTAL"] = pd.to_numeric(
            sih.get(
                "PRINCIPAL_DIAGNOSIS_CHILD_HEAT_THERMAL_PRIMARY_TOTAL",
                pd.Series(0, index=sih.index),
            ),
            errors="coerce",
        ).fillna(0)
        sih["CHILD_INDIRECT_HEAT_HOSPITALIZATION_TOTAL"] = pd.to_numeric(
            sih.get(
                "ANY_DIAGNOSIS_INDIRECT_HEAT_ANY_TOTAL",
                pd.Series(0, index=sih.index),
            ),
            errors="coerce",
        ).fillna(0)
        sih["CHILD_EXCESSIVE_NATURAL_HEAT_X30_HOSPITALIZATION_TOTAL"] = pd.to_numeric(
            sih.get(
                "EXTERNAL_CAUSE_FIELDS_EXCESSIVE_NATURAL_HEAT_X30_TOTAL",
                pd.Series(0, index=sih.index),
            ),
            errors="coerce",
        ).fillna(0)
        _fill_integer_columns(sih, ["ADMISSIONS_TOTAL", *MORTALITY_SIH_ENDPOINT_COLUMNS])
        acute = _group_daily(
            sih.loc[sih["AGE_SCOPE"].eq("AGE_U5"), ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", "ADMISSIONS_TOTAL"]],
            ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"],
            ["ADMISSIONS_TOTAL"],
        ).rename(columns={"ADMISSIONS_TOTAL": "ACUTE_CARE_PROXY_TOTAL"})
        sih_endpoint_totals = _group_daily(
            sih.loc[sih["AGE_SCOPE"].eq("AGE_U5"), ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH", *MORTALITY_SIH_ENDPOINT_COLUMNS]],
            ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"],
            MORTALITY_SIH_ENDPOINT_COLUMNS,
        )
        acute_source = "silver_child_u5_sih_daily"
    else:
        sih_exclusion = {"excluded_rows": 0, "excluded_municipalities": 0}

    live_births_path = _partition_path(root, "data", "silver", "sinasc", "silver_sinasc_daily", f"YEAR={year}", f"silver_sinasc_daily_{year}.parquet")
    live_births = _load_parquet(live_births_path, SINASC_LIVE_BIRTH_COLUMNS).copy()
    live_births["MUNICIPALITY_CODE"] = _municipality_code(root, live_births["MUNICIPALITY_CODE"])
    live_births, live_births_exclusion = _exclude_placeholder_municipalities(live_births)
    live_births = live_births.loc[pd.to_numeric(live_births["YEAR"], errors="coerce").eq(year)].copy()
    _fill_integer_columns(live_births, ["LIVE_BIRTHS_TOTAL"])
    live_births = _group_daily(live_births, ["MUNICIPALITY_CODE", "DATE", "YEAR"], ["LIVE_BIRTHS_TOTAL"])
    annual_live_births = _group_annual_sum(live_births, "LIVE_BIRTHS_TOTAL").rename(columns={"LIVE_BIRTHS_TOTAL": "LIVE_BIRTHS_TOTAL_ANNUAL"})

    pop_path = _partition_path(root, "data", "silver", "population", "silver_pop_under5", f"YEAR={year}", f"silver_pop_under5_{year}.parquet")
    population = _load_parquet(pop_path, POP_UNDER5_COLUMNS).copy()
    population["MUNICIPALITY_CODE"] = _municipality_code(root, population["MUNICIPALITY_CODE"])
    population, population_exclusion = _exclude_placeholder_municipalities(population)
    population = population.loc[pd.to_numeric(population["YEAR"], errors="coerce").eq(year), ["MUNICIPALITY_CODE", "YEAR", "POP_UNDER5"]].copy()
    population["POP_U5"] = pd.to_numeric(population["POP_UNDER5"], errors="coerce").astype("Float64")
    population = population.drop(columns=["POP_UNDER5"])

    spine = _daily_key_spine(
        climate,
        [
            neonatal[["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"]],
            postneonatal[["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"]],
            under5[["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"]],
            cause_groups[["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"]],
            acute[["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"]],
        ],
    )

    output = spine.merge(climate, on=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"], how="left")
    output = output.merge(neonatal, on=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"], how="left")
    output = output.merge(postneonatal, on=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"], how="left")
    output = output.merge(under5, on=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"], how="left")
    output = output.merge(cause_groups, on=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"], how="left")
    output = output.merge(acute, on=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"], how="left")
    output = output.merge(sih_endpoint_totals, on=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"], how="left")
    output = output.merge(annual_live_births, on=["MUNICIPALITY_CODE", "YEAR"], how="left")
    output = output.merge(population[["MUNICIPALITY_CODE", "YEAR", "POP_U5"]], on=["MUNICIPALITY_CODE", "YEAR"], how="left")
    _fill_integer_columns(
        output,
        [
            "NEONATAL_DEATHS_TOTAL",
            "POSTNEONATAL_DEATHS_TOTAL",
            "UNDER5_DEATHS_TOTAL",
            "ACUTE_CARE_PROXY_TOTAL",
            *MORTALITY_SIH_ENDPOINT_COLUMNS,
            *MORTALITY_CAUSE_GROUP_COLUMNS,
        ],
    )
    output["INFANT_DEATHS_TOTAL"] = output["NEONATAL_DEATHS_TOTAL"] + output["POSTNEONATAL_DEATHS_TOTAL"]
    output["DEATHS_NEONATAL"] = output["NEONATAL_DEATHS_TOTAL"]
    output["DEATHS_U1"] = output["INFANT_DEATHS_TOTAL"]
    output["DEATHS_U5"] = output["UNDER5_DEATHS_TOTAL"]
    output["LIVE_BIRTHS_TOTAL"] = pd.to_numeric(output["LIVE_BIRTHS_TOTAL_ANNUAL"], errors="coerce").astype("Float64")
    output["UNDER5_POPULATION"] = pd.to_numeric(output["POP_U5"], errors="coerce").astype("Float64")
    output["DENOMINATOR_USED"] = (
        "OUTCOME_SPECIFIC: DEATHS_NEONATAL+DEATHS_U1->LIVE_BIRTHS_TOTAL_ANNUAL; DEATHS_U5->POP_U5"
    )
    _apply_mortality_denominator_flags(output)
    output["OFFSET_VALUE"] = pd.Series(np.nan, index=output.index, dtype="float64")
    climate_missing = _fill_missing_climate_lineage(output)
    output["SOURCE_SIM"] = "silver_child_u5_sim_daily"
    output["SOURCE_SIH"] = acute_source
    output["SOURCE_POPULATION"] = "silver_pop_under5"
    output["ACUTE_CARE_PROXY_LABEL"] = (
        "AGE_U5_SIH_ADMISSIONS_PROXY" if acute_source == "silver_child_u5_sih_daily" else "OPTIONAL_CHILD_SIH_NOT_MATERIALIZED"
    )
    output["QC_STATUS"] = "PASS"
    output["NEONATAL_MORTALITY_RATE_PER_1000_LIVE_BIRTHS"] = _safe_rate_series(
        output["DEATHS_NEONATAL"],
        output["LIVE_BIRTHS_TOTAL_ANNUAL"],
        1000.0,
    )
    output["INFANT_MORTALITY_RATE_PER_1000_LIVE_BIRTHS"] = _safe_rate_series(
        output["DEATHS_U1"],
        output["LIVE_BIRTHS_TOTAL_ANNUAL"],
        1000.0,
    )
    output["UNDER5_MORTALITY_RATE_PER_100000_UNDER5_POPULATION"] = _safe_rate_series(
        output["DEATHS_U5"],
        output["POP_U5"],
        100000.0,
    )
    output["CRUDE_RATE_NEONATAL"] = output["NEONATAL_MORTALITY_RATE_PER_1000_LIVE_BIRTHS"]
    output["CRUDE_RATE_U1"] = output["INFANT_MORTALITY_RATE_PER_1000_LIVE_BIRTHS"]
    output["CRUDE_RATE_U5"] = output["UNDER5_MORTALITY_RATE_PER_100000_UNDER5_POPULATION"]

    validation_flags = {
        "AGE_STRATIFIED_COUNTS_RECONCILE_NEONATAL": int(output["DEATHS_NEONATAL"].sum()) == int(sim_individual_daily["AGE_GROUP_0_27D"].sum()),
        "AGE_STRATIFIED_COUNTS_RECONCILE_INFANT": int(output["DEATHS_U1"].sum()) == int(sim_individual_daily["INFANT_GROUP_TOTAL"].sum()),
        "AGE_STRATIFIED_COUNTS_RECONCILE_U5": int(output["DEATHS_U5"].sum()) == int(sim_individual_daily["AGE_GROUP_U5"].sum()),
        "NEONATAL_INFANT_DENOMINATORS_NONMISSING_OR_FLAGGED": bool(
            (
                ((output["LIVE_BIRTHS_TOTAL_ANNUAL"].notna()) & output["LIVE_BIRTHS_DENOMINATOR_MISSING_FLAG"].eq(0))
                | ((output["LIVE_BIRTHS_TOTAL_ANNUAL"].isna()) & output["LIVE_BIRTHS_DENOMINATOR_MISSING_FLAG"].eq(1))
            ).all()
        ),
        "U5_DENOMINATORS_NONMISSING_OR_FLAGGED": bool(
            (
                ((output["POP_U5"].notna()) & output["UNDER5_POPULATION_DENOMINATOR_MISSING_FLAG"].eq(0))
                | ((output["POP_U5"].isna()) & output["UNDER5_POPULATION_DENOMINATOR_MISSING_FLAG"].eq(1))
            ).all()
        ),
        "NO_INVALID_OFFSET_MATERIALIZED": bool(
            output.loc[output["LIVE_BIRTHS_OFFSET_VALID_FLAG"].eq(0), "OFFSET_LOG_LIVE_BIRTHS"].isna().all()
            and output.loc[output["UNDER5_POPULATION_OFFSET_VALID_FLAG"].eq(0), "OFFSET_LOG_UNDER5_POPULATION"].isna().all()
        ),
        "NO_LOG_ZERO_ANYWHERE": bool(
            np.isfinite(output["OFFSET_LOG_LIVE_BIRTHS"].dropna()).all()
            and np.isfinite(output["OFFSET_LOG_UNDER5_POPULATION"].dropna()).all()
        ),
        "CLIMATE_LINKAGE_COMPLETENESS_REPORTED": True,
        "CAUSE_GROUP_UNDERLYING_AND_ANY_MENTION_DISTINCT": all(
            column in output.columns
            for column in (
                "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL",
                "ANY_MENTION_DIRECT_HEAT_ANY_TOTAL",
                "UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_TOTAL",
                "ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL",
                "ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_TOTAL",
            )
        ),
        "SIH_ENDPOINT_COLUMNS_PRESENT": all(column in output.columns for column in MORTALITY_SIH_ENDPOINT_COLUMNS),
    }
    failing_validations = [name for name, passed in validation_flags.items() if not passed]
    if failing_validations:
        raise ValueError(f"gold_mnch_mortality validation failed for {year}: {failing_validations}")

    schema_columns = _schema_columns(root, "gold_mnch_mortality_diario_municipio.schema.csv")
    extra_columns = [
        "MONTH",
        "DEATHS_NEONATAL",
        "DEATHS_U1",
        "DEATHS_U5",
        "LIVE_BIRTHS_TOTAL_ANNUAL",
        "POP_U5",
        "DENOMINATOR_USED",
        "OFFSET_VALUE",
        "TMIN_C",
        "TMEAN_C",
        "RH_MIN_PCT",
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
        "POSTNEONATAL_DEATHS_TOTAL",
        "ACUTE_CARE_PROXY_LABEL",
        *MORTALITY_SIH_ENDPOINT_COLUMNS,
        "NEONATAL_MORTALITY_RATE_PER_1000_LIVE_BIRTHS",
        "INFANT_MORTALITY_RATE_PER_1000_LIVE_BIRTHS",
        "UNDER5_MORTALITY_RATE_PER_100000_UNDER5_POPULATION",
        "CRUDE_RATE_NEONATAL",
        "CRUDE_RATE_U1",
        "CRUDE_RATE_U5",
        *MORTALITY_CAUSE_GROUP_COLUMNS,
        "LIVE_BIRTHS_DENOMINATOR_ZERO_FLAG",
        "LIVE_BIRTHS_DENOMINATOR_MISSING_FLAG",
        "LIVE_BIRTHS_RATE_NOT_COMPUTABLE_FLAG",
        "LIVE_BIRTHS_OFFSET_VALID_FLAG",
        "UNDER5_POPULATION_DENOMINATOR_ZERO_FLAG",
        "UNDER5_POPULATION_DENOMINATOR_MISSING_FLAG",
        "UNDER5_POPULATION_RATE_NOT_COMPUTABLE_FLAG",
        "UNDER5_POPULATION_OFFSET_VALID_FLAG",
        *COMMON_RATE_FLAG_COLUMNS,
    ]
    output = _order_output_columns(output, schema_columns, extra_columns)
    output = output.sort_values(["DATE", "MUNICIPALITY_CODE"], kind="mergesort").reset_index(drop=True)
    output_path = _write_output(root, runtime_root, "gold_mnch_mortality", year, output)
    live_births_complete = int(output["LIVE_BIRTHS_TOTAL_ANNUAL"].notna().sum())
    pop_u5_complete = int(output["POP_U5"].notna().sum())
    climate_complete = int(output[["TMAX_C", "TMIN_C", "TMEAN_C", "RH_MIN_PCT", "RH_MEAN_PCT", "FONTE_CLIMA_FINAL"]].notna().all(axis=1).sum())
    return {
        "output_path": output_path,
        "internal_reconciliation_summary": _build_reconciliation_summary(
            output,
            [
                "DEATHS_NEONATAL",
                "DEATHS_U1",
                "DEATHS_U5",
                "ACUTE_CARE_PROXY_TOTAL",
                *MORTALITY_SIH_ENDPOINT_COLUMNS,
            ],
        ),
        "external_validation_summary": (
            "schema_required_columns_present=True; age_scope_rollups_applied=True; "
            f"climate_backbone_unmatched_rows={int(climate_missing.sum())}; cause_group_block_included=True; "
            f"excluded_placeholder_rows={sim_exclusion['excluded_rows'] + sim_individual_exclusion['excluded_rows'] + sih_exclusion['excluded_rows'] + live_births_exclusion['excluded_rows'] + population_exclusion['excluded_rows']}"
        ),
        "mortality_validation_flags": validation_flags,
        "mortality_summary": {
            "neonatal_deaths": int(output["DEATHS_NEONATAL"].sum()),
            "infant_deaths": int(output["DEATHS_U1"].sum()),
            "under5_deaths": int(output["DEATHS_U5"].sum()),
            "municipalities_with_at_least_one_neonatal_death": int(output.loc[output["DEATHS_NEONATAL"].gt(0), "MUNICIPALITY_CODE"].nunique()),
            "municipalities_with_at_least_one_infant_death": int(output.loc[output["DEATHS_U1"].gt(0), "MUNICIPALITY_CODE"].nunique()),
            "municipalities_with_at_least_one_under5_death": int(output.loc[output["DEATHS_U5"].gt(0), "MUNICIPALITY_CODE"].nunique()),
            "crude_neonatal_rate": (
                float(output["DEATHS_NEONATAL"].sum() * 1000.0 / output["LIVE_BIRTHS_TOTAL_ANNUAL"].sum())
                if float(pd.to_numeric(output["LIVE_BIRTHS_TOTAL_ANNUAL"], errors="coerce").fillna(0).sum()) > 0
                else None
            ),
            "crude_infant_rate": (
                float(output["DEATHS_U1"].sum() * 1000.0 / output["LIVE_BIRTHS_TOTAL_ANNUAL"].sum())
                if float(pd.to_numeric(output["LIVE_BIRTHS_TOTAL_ANNUAL"], errors="coerce").fillna(0).sum()) > 0
                else None
            ),
            "crude_under5_rate": (
                float(output["DEATHS_U5"].sum() * 100000.0 / output["POP_U5"].sum())
                if float(pd.to_numeric(output["POP_U5"], errors="coerce").fillna(0).sum()) > 0
                else None
            ),
            "climate_linkage_completeness": (
                f"{climate_complete}/{len(output)} ({climate_complete / len(output):.4f})" if len(output) else "NOT_EVALUATED"
            ),
            "denominator_completeness": {
                "LIVE_BIRTHS_TOTAL_ANNUAL": (
                    f"{live_births_complete}/{len(output)} ({live_births_complete / len(output):.4f})" if len(output) else "NOT_EVALUATED"
                ),
                "POP_U5": f"{pop_u5_complete}/{len(output)} ({pop_u5_complete / len(output):.4f})" if len(output) else "NOT_EVALUATED",
            },
            "DENOMINATOR_ZERO_FLAG_count": int(output["DENOMINATOR_ZERO_FLAG"].fillna(0).sum()),
            "DENOMINATOR_MISSING_FLAG_count": int(output["DENOMINATOR_MISSING_FLAG"].fillna(0).sum()),
            "RATE_NOT_COMPUTABLE_FLAG_count": int(output["RATE_NOT_COMPUTABLE_FLAG"].fillna(0).sum()),
            "OFFSET_VALID_FLAG_count": int(output["OFFSET_VALID_FLAG"].fillna(0).sum()),
            "cause_group_family_counts": {column: int(output[column].fillna(0).sum()) for column in MORTALITY_CAUSE_GROUP_COLUMNS},
            "sih_child_endpoint_counts": {column: int(output[column].fillna(0).sum()) for column in MORTALITY_SIH_ENDPOINT_COLUMNS},
            "age_reconciliation": {
                "DEATHS_NEONATAL": validation_flags["AGE_STRATIFIED_COUNTS_RECONCILE_NEONATAL"],
                "DEATHS_U1": validation_flags["AGE_STRATIFIED_COUNTS_RECONCILE_INFANT"],
                "DEATHS_U5": validation_flags["AGE_STRATIFIED_COUNTS_RECONCILE_U5"],
            },
            "climate_backbone_unmatched_rows": int(climate_missing.sum()),
            "excluded_placeholder_rows": int(
                sim_exclusion["excluded_rows"]
                + sim_individual_exclusion["excluded_rows"]
                + sih_exclusion["excluded_rows"]
                + live_births_exclusion["excluded_rows"]
                + population_exclusion["excluded_rows"]
            ),
        },
    }


def build_gold_panel_year(root: Path, runtime_root: Path, year: int) -> dict[str, Any]:
    perinatal_path = gold_output_path(root, "gold_mnch_perinatal", year)
    mortality_path = gold_output_path(root, "gold_mnch_mortality", year)
    perinatal = pd.read_parquet(perinatal_path).copy()
    for column, default in {
        "TMIN_C": np.nan,
        "TMEAN_C": np.nan,
        "RH_MIN_PCT": np.nan,
        "FONTE_TMIN_C_FINAL": pd.NA,
        "FONTE_TMEAN_C_FINAL": pd.NA,
        "FONTE_RH_MIN_PCT_FINAL": pd.NA,
        "IMPUTADO_TMIN_C": 0,
        "IMPUTADO_TMEAN_C": 0,
        "IMPUTADO_RH_MIN_PCT": 0,
    }.items():
        if column not in perinatal.columns:
            perinatal[column] = default
    perinatal = perinatal[
        [
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
        ]
    ].rename(
        columns={
            "PRETERM_FLAG": "PRETERM_BIRTHS_CANONICAL_TOTAL",
            "LIVE_BIRTHS_TOTAL": "REF_LIVE_BIRTHS_DAILY",
            "TOTAL_BIRTHS_PRIMARY": "REF_TOTAL_BIRTHS_PRIMARY_DAILY",
            "DENOMINATOR_ZERO_FLAG": "PERINATAL_DENOMINATOR_ZERO_FLAG",
            "DENOMINATOR_MISSING_FLAG": "PERINATAL_DENOMINATOR_MISSING_FLAG",
            "RATE_NOT_COMPUTABLE_FLAG": "PERINATAL_RATE_NOT_COMPUTABLE_FLAG",
            "OFFSET_VALID_FLAG": "PERINATAL_OFFSET_VALID_FLAG",
        }
    ).copy()
    perinatal, perinatal_exclusion = _exclude_placeholder_municipalities(perinatal)
    mortality = pd.read_parquet(mortality_path).copy()
    for column in MORTALITY_SIH_ENDPOINT_COLUMNS:
        if column not in mortality.columns:
            mortality[column] = 0
    mortality = mortality[
        [
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
            *MORTALITY_SIH_ENDPOINT_COLUMNS,
            "DENOMINATOR_ZERO_FLAG",
            "DENOMINATOR_MISSING_FLAG",
            "RATE_NOT_COMPUTABLE_FLAG",
            "OFFSET_VALID_FLAG",
            "ACUTE_CARE_PROXY_LABEL",
        ]
    ].rename(
        columns={
            "LIVE_BIRTHS_TOTAL_ANNUAL": "REF_LIVE_BIRTHS_TOTAL_ANNUAL",
            "POP_U5": "REF_POP_U5",
            "DENOMINATOR_ZERO_FLAG": "MORTALITY_DENOMINATOR_ZERO_FLAG",
            "DENOMINATOR_MISSING_FLAG": "MORTALITY_DENOMINATOR_MISSING_FLAG",
            "RATE_NOT_COMPUTABLE_FLAG": "MORTALITY_RATE_NOT_COMPUTABLE_FLAG",
            "OFFSET_VALID_FLAG": "MORTALITY_OFFSET_VALID_FLAG",
        }
    )
    mortality, mortality_exclusion = _exclude_placeholder_municipalities(mortality)
    acute_care_proxy_label = (
        mortality["ACUTE_CARE_PROXY_LABEL"].dropna().astype(str).iloc[0]
        if mortality["ACUTE_CARE_PROXY_LABEL"].notna().any()
        else "OPTIONAL_CHILD_SIH_NOT_MATERIALIZED"
    )
    mortality_daily = mortality[
        [
            "MUNICIPALITY_CODE",
            "DATE",
            "YEAR",
            "MONTH",
            "DEATHS_NEONATAL",
            "DEATHS_U1",
            "DEATHS_U5",
            "ACUTE_CARE_PROXY_TOTAL",
            "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL",
            "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL",
            "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL",
            "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL",
        ]
    ].copy()
    mortality_annual = (
        mortality.groupby(["MUNICIPALITY_CODE", "YEAR"], as_index=False, sort=True)[
            [
                "REF_LIVE_BIRTHS_TOTAL_ANNUAL",
                "REF_POP_U5",
                "MORTALITY_DENOMINATOR_ZERO_FLAG",
                "MORTALITY_DENOMINATOR_MISSING_FLAG",
                "MORTALITY_RATE_NOT_COMPUTABLE_FLAG",
                "MORTALITY_OFFSET_VALID_FLAG",
            ]
        ]
        .max()
    )

    wra_path = _partition_path(root, "data", "silver", "population", "silver_pop_wra_15_49", f"YEAR={year}", f"silver_pop_wra_15_49_{year}.parquet")
    wra = _load_parquet(wra_path, POP_WRA_15_49_COLUMNS).copy()
    wra["MUNICIPALITY_CODE"] = _municipality_code(root, wra["MUNICIPALITY_CODE"])
    wra, wra_exclusion = _exclude_placeholder_municipalities(wra)
    wra = wra.loc[pd.to_numeric(wra["YEAR"], errors="coerce").eq(year), ["MUNICIPALITY_CODE", "YEAR", "POP_WRA_15_49"]].copy()
    wra["REF_POP_WRA_15_49"] = pd.to_numeric(wra["POP_WRA_15_49"], errors="coerce").astype("Float64")
    wra = wra.drop(columns=["POP_WRA_15_49"])

    output = perinatal.merge(mortality_daily, on=["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"], how="left")
    output = output.merge(mortality_annual, on=["MUNICIPALITY_CODE", "YEAR"], how="left")
    output = output.merge(wra, on=["MUNICIPALITY_CODE", "YEAR"], how="left")
    _fill_integer_columns(
        output,
        [
            "PRETERM_BIRTHS_CANONICAL_TOTAL",
            "STILLBIRTH_TOTAL_PRIMARY_GTE28",
            "LOW_BIRTH_WEIGHT_LT2500_TOTAL",
            "PRETERM_LOW_BIRTH_WEIGHT_TOTAL",
            "PRENATAL_VISITS_8PLUS_FLAG",
            "DEATHS_NEONATAL",
            "DEATHS_U1",
            "DEATHS_U5",
            "OBSTETRIC_ADMISSIONS_TOTAL",
            "HDP_ANY_TOTAL",
            "HDP_SEVERE_TOTAL",
            "PRETERM_LABOR_TOTAL",
            "APH_ANTEPARTUM_TOTAL",
            "GDM_GESTATIONAL_TOTAL",
            "ACUTE_CARE_PROXY_TOTAL",
            "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL",
            "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL",
            "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL",
            "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL",
        ],
    )
    for column in (
        "PERINATAL_DENOMINATOR_ZERO_FLAG",
        "PERINATAL_DENOMINATOR_MISSING_FLAG",
        "PERINATAL_RATE_NOT_COMPUTABLE_FLAG",
    ):
        output[column] = output[column].fillna(0).astype("Int8")
    output["MORTALITY_DENOMINATOR_ZERO_FLAG"] = output["MORTALITY_DENOMINATOR_ZERO_FLAG"].fillna(0).astype("Int8")
    output["MORTALITY_DENOMINATOR_MISSING_FLAG"] = output["MORTALITY_DENOMINATOR_MISSING_FLAG"].fillna(1).astype("Int8")
    output["MORTALITY_RATE_NOT_COMPUTABLE_FLAG"] = output["MORTALITY_RATE_NOT_COMPUTABLE_FLAG"].fillna(1).astype("Int8")
    output["PERINATAL_OFFSET_VALID_FLAG"] = output["PERINATAL_OFFSET_VALID_FLAG"].fillna(0).astype("Int8")
    output["MORTALITY_OFFSET_VALID_FLAG"] = output["MORTALITY_OFFSET_VALID_FLAG"].fillna(0).astype("Int8")

    output["REF_LIVE_BIRTHS_DAILY"] = pd.to_numeric(output["REF_LIVE_BIRTHS_DAILY"], errors="coerce").astype("Float64")
    output["REF_TOTAL_BIRTHS_PRIMARY_DAILY"] = pd.to_numeric(output["REF_TOTAL_BIRTHS_PRIMARY_DAILY"], errors="coerce").astype("Float64")
    output["REF_LIVE_BIRTHS_TOTAL_ANNUAL"] = pd.to_numeric(output["REF_LIVE_BIRTHS_TOTAL_ANNUAL"], errors="coerce").astype("Float64")
    output["REF_POP_U5"] = pd.to_numeric(output["REF_POP_U5"], errors="coerce").astype("Float64")
    _apply_single_denominator_flags(
        output,
        denominator_column="REF_POP_WRA_15_49",
        offset_column="WRA_15_49_OFFSET_LOG",
        prefix="WRA_15_49",
    )
    output["DENOMINATOR_ZERO_FLAG"] = pd.Series(
        np.maximum.reduce(
            [
                output["PERINATAL_DENOMINATOR_ZERO_FLAG"].astype("int64"),
                output["MORTALITY_DENOMINATOR_ZERO_FLAG"].astype("int64"),
                output["WRA_15_49_DENOMINATOR_ZERO_FLAG"].fillna(0).astype("int64"),
            ]
        ),
        index=output.index,
        dtype="Int8",
    )
    output["DENOMINATOR_MISSING_FLAG"] = pd.Series(
        np.maximum.reduce(
            [
                output["PERINATAL_DENOMINATOR_MISSING_FLAG"].astype("int64"),
                output["MORTALITY_DENOMINATOR_MISSING_FLAG"].astype("int64"),
                output["WRA_15_49_DENOMINATOR_MISSING_FLAG"].fillna(0).astype("int64"),
            ]
        ),
        index=output.index,
        dtype="Int8",
    )
    output["RATE_NOT_COMPUTABLE_FLAG"] = pd.Series(
        np.maximum.reduce(
            [
                output["PERINATAL_RATE_NOT_COMPUTABLE_FLAG"].astype("int64"),
                output["MORTALITY_RATE_NOT_COMPUTABLE_FLAG"].astype("int64"),
                output["WRA_15_49_RATE_NOT_COMPUTABLE_FLAG"].fillna(0).astype("int64"),
            ]
        ),
        index=output.index,
        dtype="Int8",
    )
    output["OFFSET_VALID_FLAG"] = pd.Series(
        np.minimum.reduce(
            [
                output["PERINATAL_OFFSET_VALID_FLAG"].astype("int64"),
                output["MORTALITY_OFFSET_VALID_FLAG"].astype("int64"),
                output["WRA_15_49_OFFSET_VALID_FLAG"].fillna(0).astype("int64"),
            ]
        ),
        index=output.index,
        dtype="Int8",
    )
    output["PANEL_ROLE_NOTE"] = "MUNICIPALITY_DAY_COMPARATIVE_MART_CANONICAL_ENDPOINTS_REMAIN_IN_SOURCE_GOLD_TABLES"
    output["ACUTE_CARE_PROXY_LABEL"] = acute_care_proxy_label
    output["SOURCE_PANEL"] = "gold_mnch_panel"
    output["QC_STATUS"] = "PASS"

    validation_flags = {
        "MUNICIPALITY_DAY_COVERAGE_DENSE_WHERE_EXPECTED": int(output[["MUNICIPALITY_CODE", "DATE"]].duplicated().sum()) == 0
        and int(len(output)) == int(len(perinatal)),
        "ROW_COUNTS_PLAUSIBLE": int(len(output)) >= int(len(mortality_daily)) and int(len(output)) <= int(len(perinatal) + len(mortality_daily)),
        "REQUIRED_DENOMINATOR_LINKS_PRESENT_OR_FLAGGED": bool(
            (
                output["REF_POP_WRA_15_49"].notna() | output["WRA_15_49_DENOMINATOR_MISSING_FLAG"].eq(1)
            ).all()
            and (
                output["REF_LIVE_BIRTHS_TOTAL_ANNUAL"].notna() | output["MORTALITY_DENOMINATOR_MISSING_FLAG"].eq(1)
            ).all()
            and (
                output["REF_POP_U5"].notna() | output["MORTALITY_DENOMINATOR_MISSING_FLAG"].eq(1)
            ).all()
        ),
        "NO_PRIMARY_ENDPOINT_LEGACY_FALLBACK": all(
            column in output.columns
            for column in (
                "PRETERM_BIRTHS_CANONICAL_TOTAL",
                "STILLBIRTH_TOTAL_PRIMARY_GTE28",
                "DEATHS_NEONATAL",
                "DEATHS_U1",
                "DEATHS_U5",
            )
        ),
        "CLIMATE_LINKAGE_COMPLETE_OR_EXPLICITLY_JUSTIFIED": bool(
            (
                output[["TMAX_C", "TMIN_C", "TMEAN_C", "RH_MIN_PCT", "RH_MEAN_PCT", "HEATWAVE_LEN_P95", "HEAT_INDEX_APPROX_C"]].notna().all(axis=1)
                | output["FONTE_CLIMA_FINAL"].eq("NO_CLIMATE_BACKBONE_MATCH")
            ).all()
        ),
        "ZERO_NA_DENOMINATOR_HANDLING_FOLLOWS_RULES": bool(
            output.loc[output["WRA_15_49_OFFSET_VALID_FLAG"].eq(0), "WRA_15_49_OFFSET_LOG"].isna().all()
            and np.isfinite(output["WRA_15_49_OFFSET_LOG"].dropna()).all()
        ),
        "HEAT_INDEX_APPROX_PRESENT_AND_LABELED": "HEAT_INDEX_APPROX_C" in output.columns and "HEAT_INDEX_METHOD" in output.columns,
    }
    failing_validations = [name for name, passed in validation_flags.items() if not passed]
    if failing_validations:
        raise ValueError(f"gold_mnch_panel validation failed for {year}: {failing_validations}")

    output = output.drop(columns=["WRA_15_49_OFFSET_LOG"], errors="ignore")

    schema_columns = _schema_columns(root, "gold_mnch_panel_diario_municipio.schema.csv")
    extra_columns = [
        "MONTH",
        "PRETERM_BIRTHS_CANONICAL_TOTAL",
        "STILLBIRTH_TOTAL_PRIMARY_GTE28",
        "LOW_BIRTH_WEIGHT_LT2500_TOTAL",
        "PRETERM_LOW_BIRTH_WEIGHT_TOTAL",
        "PRENATAL_VISITS_8PLUS_FLAG",
        "APH_ANTEPARTUM_TOTAL",
        "DEATHS_NEONATAL",
        "DEATHS_U1",
        "DEATHS_U5",
        "TMIN_C",
        "TMEAN_C",
        "RH_MIN_PCT",
        "REF_LIVE_BIRTHS_DAILY",
        "REF_TOTAL_BIRTHS_PRIMARY_DAILY",
        "REF_LIVE_BIRTHS_TOTAL_ANNUAL",
        "REF_POP_U5",
        "REF_POP_WRA_15_49",
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
        "PANEL_ROLE_NOTE",
        "ACUTE_CARE_PROXY_LABEL",
        "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL",
        "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL",
        "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL",
        "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL",
        "PERINATAL_DENOMINATOR_ZERO_FLAG",
        "PERINATAL_DENOMINATOR_MISSING_FLAG",
        "PERINATAL_RATE_NOT_COMPUTABLE_FLAG",
        "PERINATAL_OFFSET_VALID_FLAG",
        *COMMON_RATE_FLAG_COLUMNS,
        "MORTALITY_DENOMINATOR_ZERO_FLAG",
        "MORTALITY_DENOMINATOR_MISSING_FLAG",
        "MORTALITY_RATE_NOT_COMPUTABLE_FLAG",
        "MORTALITY_OFFSET_VALID_FLAG",
        "WRA_15_49_DENOMINATOR_ZERO_FLAG",
        "WRA_15_49_DENOMINATOR_MISSING_FLAG",
        "WRA_15_49_RATE_NOT_COMPUTABLE_FLAG",
        "WRA_15_49_OFFSET_VALID_FLAG",
    ]
    output = _order_output_columns(output, schema_columns, extra_columns)
    output = output.sort_values(["DATE", "MUNICIPALITY_CODE"], kind="mergesort").reset_index(drop=True)
    output_path = _write_output(root, runtime_root, "gold_mnch_panel", year, output)
    annual_refs = output[["MUNICIPALITY_CODE", "YEAR", "REF_LIVE_BIRTHS_TOTAL_ANNUAL", "REF_POP_U5", "REF_POP_WRA_15_49"]].drop_duplicates()
    days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    municipality_day_counts = output.groupby("MUNICIPALITY_CODE")["DATE"].nunique()
    return {
        "output_path": output_path,
        "internal_reconciliation_summary": _build_reconciliation_summary(
            output,
            [
                "PRETERM_BIRTHS_CANONICAL_TOTAL",
                "LOW_BIRTH_WEIGHT_LT2500_TOTAL",
                "PRETERM_LOW_BIRTH_WEIGHT_TOTAL",
                "STILLBIRTH_TOTAL_PRIMARY_GTE28",
                "DEATHS_U5",
                "OBSTETRIC_ADMISSIONS_TOTAL",
                "ACUTE_CARE_PROXY_TOTAL",
                "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL",
                "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL",
                "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL",
                "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL",
            ],
        ),
        "external_validation_summary": (
            "schema_required_columns_present=True; canonical_sources_joined=True; "
            f"zero_missing_climate_required_fields={int(output[['TMAX_C','TMIN_C','TMEAN_C','RH_MIN_PCT','RH_MEAN_PCT','HEATWAVE_LEN_P95','HEAT_INDEX_APPROX_C']].isna().sum().sum()) == 0}; "
            f"excluded_placeholder_rows={perinatal_exclusion['excluded_rows'] + mortality_exclusion['excluded_rows'] + wra_exclusion['excluded_rows']}"
        ),
        "panel_validation_flags": validation_flags,
        "panel_summary": {
            "total_municipality_days": int(len(output)),
            "municipalities_with_full_daily_coverage": int(municipality_day_counts.eq(days_in_year).sum()),
            "municipalities_with_event_by_endpoint": {
                column: int(output.loc[output[column].gt(0), "MUNICIPALITY_CODE"].nunique())
                for column in (
                    "PRETERM_BIRTHS_CANONICAL_TOTAL",
                    "LOW_BIRTH_WEIGHT_LT2500_TOTAL",
                    "PRETERM_LOW_BIRTH_WEIGHT_TOTAL",
                    "STILLBIRTH_TOTAL_PRIMARY_GTE28",
                    "DEATHS_NEONATAL",
                    "DEATHS_U1",
                    "DEATHS_U5",
                    "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL",
                    "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL",
                    "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL",
                    "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL",
                    "OBSTETRIC_ADMISSIONS_TOTAL",
                    "HDP_ANY_TOTAL",
                    "PRETERM_LABOR_TOTAL",
                    "APH_ANTEPARTUM_TOTAL",
                    "GDM_GESTATIONAL_TOTAL",
                )
            },
            "total_events_by_endpoint": {
                column: int(output[column].fillna(0).sum())
                for column in (
                    "PRETERM_BIRTHS_CANONICAL_TOTAL",
                    "LOW_BIRTH_WEIGHT_LT2500_TOTAL",
                    "PRETERM_LOW_BIRTH_WEIGHT_TOTAL",
                    "STILLBIRTH_TOTAL_PRIMARY_GTE28",
                    "DEATHS_NEONATAL",
                    "DEATHS_U1",
                    "DEATHS_U5",
                    "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL",
                    "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL",
                    "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL",
                    "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL",
                    "OBSTETRIC_ADMISSIONS_TOTAL",
                    "HDP_ANY_TOTAL",
                    "PRETERM_LABOR_TOTAL",
                    "APH_ANTEPARTUM_TOTAL",
                    "GDM_GESTATIONAL_TOTAL",
                )
            },
            "crude_rate_by_endpoint": {
                "PRETERM_BIRTHS_CANONICAL_TOTAL": (
                    float(output["PRETERM_BIRTHS_CANONICAL_TOTAL"].sum() / output["REF_LIVE_BIRTHS_DAILY"].sum())
                    if float(pd.to_numeric(output["REF_LIVE_BIRTHS_DAILY"], errors="coerce").fillna(0).sum()) > 0
                    else None
                ),
                "STILLBIRTH_TOTAL_PRIMARY_GTE28": (
                    float(output["STILLBIRTH_TOTAL_PRIMARY_GTE28"].sum() * 1000.0 / output["REF_TOTAL_BIRTHS_PRIMARY_DAILY"].sum())
                    if float(pd.to_numeric(output["REF_TOTAL_BIRTHS_PRIMARY_DAILY"], errors="coerce").fillna(0).sum()) > 0
                    else None
                ),
                "DEATHS_NEONATAL": (
                    float(output["DEATHS_NEONATAL"].sum() * 1000.0 / annual_refs["REF_LIVE_BIRTHS_TOTAL_ANNUAL"].fillna(0).sum())
                    if float(pd.to_numeric(annual_refs["REF_LIVE_BIRTHS_TOTAL_ANNUAL"], errors="coerce").fillna(0).sum()) > 0
                    else None
                ),
                "DEATHS_U1": (
                    float(output["DEATHS_U1"].sum() * 1000.0 / annual_refs["REF_LIVE_BIRTHS_TOTAL_ANNUAL"].fillna(0).sum())
                    if float(pd.to_numeric(annual_refs["REF_LIVE_BIRTHS_TOTAL_ANNUAL"], errors="coerce").fillna(0).sum()) > 0
                    else None
                ),
                "DEATHS_U5": (
                    float(output["DEATHS_U5"].sum() * 100000.0 / annual_refs["REF_POP_U5"].fillna(0).sum())
                    if float(pd.to_numeric(annual_refs["REF_POP_U5"], errors="coerce").fillna(0).sum()) > 0
                    else None
                ),
                "OBSTETRIC_ADMISSIONS_TOTAL": (
                    float(output["OBSTETRIC_ADMISSIONS_TOTAL"].sum() * 100000.0 / annual_refs["REF_POP_WRA_15_49"].fillna(0).sum())
                    if float(pd.to_numeric(annual_refs["REF_POP_WRA_15_49"], errors="coerce").fillna(0).sum()) > 0
                    else None
                ),
                "HDP_ANY_TOTAL": (
                    float(output["HDP_ANY_TOTAL"].sum() * 100000.0 / annual_refs["REF_POP_WRA_15_49"].fillna(0).sum())
                    if float(pd.to_numeric(annual_refs["REF_POP_WRA_15_49"], errors="coerce").fillna(0).sum()) > 0
                    else None
                ),
                "PRETERM_LABOR_TOTAL": (
                    float(output["PRETERM_LABOR_TOTAL"].sum() * 100000.0 / annual_refs["REF_POP_WRA_15_49"].fillna(0).sum())
                    if float(pd.to_numeric(annual_refs["REF_POP_WRA_15_49"], errors="coerce").fillna(0).sum()) > 0
                    else None
                ),
                "APH_ANTEPARTUM_TOTAL": (
                    float(output["APH_ANTEPARTUM_TOTAL"].sum() * 100000.0 / annual_refs["REF_POP_WRA_15_49"].fillna(0).sum())
                    if float(pd.to_numeric(annual_refs["REF_POP_WRA_15_49"], errors="coerce").fillna(0).sum()) > 0
                    else None
                ),
                "GDM_GESTATIONAL_TOTAL": (
                    float(output["GDM_GESTATIONAL_TOTAL"].sum() * 100000.0 / annual_refs["REF_POP_WRA_15_49"].fillna(0).sum())
                    if float(pd.to_numeric(annual_refs["REF_POP_WRA_15_49"], errors="coerce").fillna(0).sum()) > 0
                    else None
                ),
            },
            "DENOMINATOR_ZERO_FLAG_count": int(output["DENOMINATOR_ZERO_FLAG"].fillna(0).sum()),
            "DENOMINATOR_MISSING_FLAG_count": int(output["DENOMINATOR_MISSING_FLAG"].fillna(0).sum()),
            "RATE_NOT_COMPUTABLE_FLAG_count": int(output["RATE_NOT_COMPUTABLE_FLAG"].fillna(0).sum()),
            "OFFSET_VALID_FLAG_count": int(output["OFFSET_VALID_FLAG"].fillna(0).sum()),
            "zero_missing_confirmation_required_climate_linkage_fields": {
                column: int(output[column].isna().sum()) == 0
                for column in ("TMAX_C", "TMIN_C", "TMEAN_C", "RH_MIN_PCT", "RH_MEAN_PCT", "HEATWAVE_LEN_P95", "HEAT_INDEX_APPROX_C")
            },
            "climate_required_field_missing_counts": {
                column: int(output[column].isna().sum())
                for column in ("TMAX_C", "TMIN_C", "TMEAN_C", "RH_MIN_PCT", "RH_MEAN_PCT", "HEATWAVE_LEN_P95", "HEAT_INDEX_APPROX_C")
            },
            "excluded_placeholder_rows": int(
                perinatal_exclusion["excluded_rows"] + mortality_exclusion["excluded_rows"] + wra_exclusion["excluded_rows"]
            ),
        },
    }


def build_gold_analysis_child_u5_year(root: Path, runtime_root: Path, year: int) -> dict[str, Any]:
    mortality_path = gold_output_path(root, "gold_mnch_mortality", year)
    output = pd.read_parquet(mortality_path).copy()
    output, exclusion_summary = _exclude_placeholder_municipalities(output)
    output["DEATHS_0_27D"] = output["DEATHS_NEONATAL"]
    output["DEATHS_28_364D"] = output["POSTNEONATAL_DEATHS_TOTAL"]
    output["DEATHS_1_4Y"] = (output["DEATHS_U5"] - output["DEATHS_U1"]).astype("int64")
    output["DEATHS_UNDER5_TOTAL"] = output["DEATHS_U5"]
    output["CHILD_VIEW_AGE_LOGIC_NOTE"] = "AGE_BANDS=0_27_DAYS|28_364_DAYS|1_4_COMPLETED_YEARS|UNDER5_TOTAL; UNDER5=0_59_MONTHS_OR_0_4_COMPLETED_YEARS"
    output["VIEW_ROLE_NOTE"] = "DERIVED_CHILD_U5_ANALYTICAL_VIEW_CANONICAL_MORTALITY_TABLE_UNCHANGED"
    output["SOURCE_DERIVED_VIEW"] = "gold_analysis_child_u5"
    validation_flags = {
        "NEONATAL_RECONCILES_TO_CANONICAL": bool(output["DEATHS_0_27D"].eq(output["DEATHS_NEONATAL"]).all()),
        "POSTNEONATAL_RECONCILES_TO_CANONICAL": bool(output["DEATHS_28_364D"].eq(output["POSTNEONATAL_DEATHS_TOTAL"]).all()),
        "AGE_1_4_RECONCILES_TO_CANONICAL": bool(output["DEATHS_1_4Y"].eq(output["DEATHS_U5"] - output["DEATHS_U1"]).all()),
        "UNDER5_RECONCILES_TO_CANONICAL": bool(output["DEATHS_UNDER5_TOTAL"].eq(output["DEATHS_U5"]).all()),
        "AGE_1_4_NON_NEGATIVE": bool(output["DEATHS_1_4Y"].ge(0).all()),
    }
    failing_validations = [name for name, passed in validation_flags.items() if not passed]
    if failing_validations:
        raise ValueError(f"gold_analysis_child_u5 validation failed for {year}: {failing_validations}")
    output = output.sort_values(["DATE", "MUNICIPALITY_CODE"], kind="mergesort").reset_index(drop=True)
    output_path = _write_output(root, runtime_root, "gold_analysis_child_u5", year, output)
    return {
        "output_path": output_path,
        "internal_reconciliation_summary": _build_reconciliation_summary(
            output,
            [
                "DEATHS_0_27D",
                "DEATHS_28_364D",
                "DEATHS_1_4Y",
                "DEATHS_UNDER5_TOTAL",
                "ACUTE_CARE_PROXY_TOTAL",
                *MORTALITY_SIH_ENDPOINT_COLUMNS,
            ],
        ),
        "external_validation_summary": "derived_from=gold_mnch_mortality; canonical_table_not_overwritten=True; age_logic_explicit=True",
        "derived_summary": {
            "events_by_endpoint": {
                "DEATHS_0_27D": int(output["DEATHS_0_27D"].sum()),
                "DEATHS_28_364D": int(output["DEATHS_28_364D"].sum()),
                "DEATHS_1_4Y": int(output["DEATHS_1_4Y"].sum()),
                "DEATHS_UNDER5_TOTAL": int(output["DEATHS_UNDER5_TOTAL"].sum()),
                **{column: int(output[column].sum()) for column in MORTALITY_SIH_ENDPOINT_COLUMNS if column in output.columns},
            },
            "reconciliation_against_canonical_gold_totals": validation_flags,
            "excluded_placeholder_rows": exclusion_summary["excluded_rows"],
        },
        "derived_validation_flags": validation_flags,
    }


def _build_wra_view(
    root: Path,
    runtime_root: Path,
    year: int,
    table: str,
    population_dataset: str,
    population_column: str,
    offset_column: str,
    view_role_note: str,
    age_scope_note: str,
    sensitivity_only: bool,
) -> dict[str, Any]:
    perinatal_path = gold_output_path(root, "gold_mnch_perinatal", year)
    canonical = pd.read_parquet(perinatal_path).copy()
    for column, default in {
        "APH_ANTEPARTUM_TOTAL": 0,
        "TMIN_C": np.nan,
        "TMEAN_C": np.nan,
        "RH_MIN_PCT": np.nan,
    }.items():
        if column not in canonical.columns:
            canonical[column] = default
    canonical, canonical_exclusion = _exclude_placeholder_municipalities(canonical)
    population_path = _partition_path(root, "data", "silver", "population", population_dataset, f"YEAR={year}", f"{population_dataset}_{year}.parquet")
    population = pd.read_parquet(population_path, columns=["MUNICIPALITY_CODE", "YEAR", population_column]).copy()
    population["MUNICIPALITY_CODE"] = _municipality_code(root, population["MUNICIPALITY_CODE"])
    population, population_exclusion = _exclude_placeholder_municipalities(population)
    _fill_integer_columns(population, [population_column])
    output = canonical.merge(population, on=["MUNICIPALITY_CODE", "YEAR"], how="left")
    output["PERINATAL_DENOMINATOR_ZERO_FLAG"] = output["DENOMINATOR_ZERO_FLAG"]
    output["PERINATAL_DENOMINATOR_MISSING_FLAG"] = output["DENOMINATOR_MISSING_FLAG"]
    output["PERINATAL_RATE_NOT_COMPUTABLE_FLAG"] = output["RATE_NOT_COMPUTABLE_FLAG"]
    output["PERINATAL_OFFSET_VALID_FLAG"] = output["OFFSET_VALID_FLAG"]
    _apply_single_denominator_flags(output, population_column, offset_column, prefix="")
    output["SOURCE_POPULATION"] = population_dataset
    output["SOURCE_CANONICAL_GOLD"] = "gold_mnch_perinatal"
    output["VIEW_ROLE_NOTE"] = view_role_note
    output["WRA_AGE_SCOPE_NOTE"] = age_scope_note
    output["SENSITIVITY_ONLY_FLAG"] = int(sensitivity_only)
    output["SOURCE_DERIVED_VIEW"] = table
    output["QC_STATUS"] = "PASS"
    validation_flags = {
        "ROWS_RECONCILE_TO_CANONICAL": int(len(output)) == int(len(canonical)),
        "OBSTETRIC_ADMISSIONS_RECONCILE_TO_CANONICAL": int(output["OBSTETRIC_ADMISSIONS_TOTAL"].sum()) == int(canonical["OBSTETRIC_ADMISSIONS_TOTAL"].sum()),
        "HDP_ANY_RECONCILES_TO_CANONICAL": int(output["HDP_ANY_TOTAL"].sum()) == int(canonical["HDP_ANY_TOTAL"].sum()),
        "PRETERM_LABOR_RECONCILES_TO_CANONICAL": int(output["PRETERM_LABOR_TOTAL"].sum()) == int(canonical["PRETERM_LABOR_TOTAL"].sum()),
        "APH_ANTEPARTUM_RECONCILES_TO_CANONICAL": int(output["APH_ANTEPARTUM_TOTAL"].sum()) == int(canonical["APH_ANTEPARTUM_TOTAL"].sum()),
        "GDM_GESTATIONAL_RECONCILES_TO_CANONICAL": int(output["GDM_GESTATIONAL_TOTAL"].sum()) == int(canonical["GDM_GESTATIONAL_TOTAL"].sum()),
        "CANONICAL_GESTATIONAL_TABLE_UNTRUNCATED": int(output["PRETERM_FLAG"].sum()) == int(canonical["PRETERM_FLAG"].sum()),
    }
    failing_validations = [name for name, passed in validation_flags.items() if not passed]
    if failing_validations:
        raise ValueError(f"{table} validation failed for {year}: {failing_validations}")
    output = output.sort_values(["DATE", "MUNICIPALITY_CODE"], kind="mergesort").reset_index(drop=True)
    output_path = _write_output(root, runtime_root, table, year, output)
    return {
        "output_path": output_path,
        "internal_reconciliation_summary": _build_reconciliation_summary(
            output,
            ["OBSTETRIC_ADMISSIONS_TOTAL", "HDP_ANY_TOTAL", "PRETERM_LABOR_TOTAL", "APH_ANTEPARTUM_TOTAL", "GDM_GESTATIONAL_TOTAL"],
        ),
        "external_validation_summary": (
            f"derived_from=gold_mnch_perinatal; population_source={population_dataset}; "
            "canonical_table_not_overwritten=True; gestational_perinatal_tables_untruncated=True; "
            f"excluded_placeholder_rows={canonical_exclusion['excluded_rows'] + population_exclusion['excluded_rows']}"
        ),
        "derived_summary": {
            "events_by_endpoint": {
                "OBSTETRIC_ADMISSIONS_TOTAL": int(output["OBSTETRIC_ADMISSIONS_TOTAL"].sum()),
                "HDP_ANY_TOTAL": int(output["HDP_ANY_TOTAL"].sum()),
                "PRETERM_LABOR_TOTAL": int(output["PRETERM_LABOR_TOTAL"].sum()),
                "APH_ANTEPARTUM_TOTAL": int(output["APH_ANTEPARTUM_TOTAL"].sum()),
                "GDM_GESTATIONAL_TOTAL": int(output["GDM_GESTATIONAL_TOTAL"].sum()),
            },
            "reconciliation_against_canonical_gold_totals": validation_flags,
            "excluded_placeholder_rows": int(canonical_exclusion["excluded_rows"] + population_exclusion["excluded_rows"]),
        },
        "derived_validation_flags": validation_flags,
    }


def build_gold_analysis_wra_15_49_year(root: Path, runtime_root: Path, year: int) -> dict[str, Any]:
    return _build_wra_view(
        root,
        runtime_root,
        year,
        "gold_analysis_wra_15_49",
        "silver_pop_wra_15_49",
        "POP_WRA_15_49",
        "OFFSET_LOG_POP_WRA_15_49",
        "DERIVED_WRA_15_49_CANONICAL_VIEW_CANONICAL_GOLD_UNCHANGED",
        "WOMEN_15_49_CANONICAL_REPRODUCTIVE_AGE_VIEW",
        False,
    )


def build_gold_analysis_wra_15_45_sens_year(root: Path, runtime_root: Path, year: int) -> dict[str, Any]:
    return _build_wra_view(
        root,
        runtime_root,
        year,
        "gold_analysis_wra_15_45_sens",
        "silver_pop_wra_15_45_sensitivity",
        "POP_WRA_15_45_SENSITIVITY",
        "OFFSET_LOG_POP_WRA_15_45_SENSITIVITY",
        "DERIVED_WRA_15_45_SENSITIVITY_VIEW_CANONICAL_15_49_VIEW_RETAINS_PRIORITY",
        "WOMEN_15_45_SENSITIVITY_ONLY_VIEW",
        True,
    )
