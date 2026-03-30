from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .aliases import validate_public_output_columns
from .config import load_analysis_variable_aliases, load_round_config
from .registry import analysis_shared_marts_root


DATA_GOLD_ROOT = Path("data/gold")
CONTEXT_STATIC_PATH = DATA_GOLD_ROOT / "gold_municipality_context_static" / "gold_municipality_context_static.parquet"

SHARED_ID_COLUMNS = ["MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"]
SHARED_EXPOSURE_COLUMNS = [
    "TMAX_C",
    "RH_MEAN_PCT",
    "HEATWAVE_LEN_P95",
    "HEAT_INDEX_APPROX_C",
    "FONTE_CLIMA_FINAL",
    "FONTE_TMAX_C_FINAL",
    "FONTE_RH_MEAN_PCT_FINAL",
    "IMPUTADO_POR_FALLBACK",
    "IMPUTADO_TMAX_C",
    "IMPUTADO_RH_MEAN_PCT",
    "HEAT_INDEX_METHOD",
    "HEAT_INDEX_SOURCE_PAIR",
    "HEAT_INDEX_INPUTS_ZERO_MISSING",
    "QC_STATUS",
    "DENOMINATOR_ZERO_FLAG",
    "DENOMINATOR_MISSING_FLAG",
    "RATE_NOT_COMPUTABLE_FLAG",
    "OFFSET_VALID_FLAG",
]
EXPOSURE_CANONICAL_COLUMNS = ["TMAX_C", "RH_MEAN_PCT", "HEATWAVE_LEN_P95", "HEAT_INDEX_APPROX_C"]

ACUTE_BASE_DATASET = "gold_mnch_mortality"
ACUTE_JOIN_DATASET = "gold_mnch_panel"
ACUTE_DIRECT_COLUMNS = SHARED_ID_COLUMNS + SHARED_EXPOSURE_COLUMNS + [
    "DEATHS_NEONATAL",
    "DEATHS_U1",
    "DEATHS_U5",
    "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL",
    "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL",
    "CHILD_DIRECT_HEAT_HOSPITALIZATION_TOTAL",
    "CHILD_INDIRECT_HEAT_HOSPITALIZATION_TOTAL",
    "CHILD_EXCESSIVE_NATURAL_HEAT_X30_HOSPITALIZATION_TOTAL",
    "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL",
    "UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_TOTAL",
    "ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_TOTAL",
    "ANY_MENTION_DIRECT_HEAT_ANY_TOTAL",
    "ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL",
    "LIVE_BIRTHS_TOTAL_ANNUAL",
    "POP_U5",
    "OFFSET_LOG_LIVE_BIRTHS",
    "OFFSET_LOG_UNDER5_POPULATION",
]
ACUTE_PANEL_JOIN_COLUMNS = ["ACUTE_CARE_PROXY_TOTAL", "ACUTE_CARE_PROXY_LABEL"]
ACUTE_CHILD_U5_JOIN_DATASET = "gold_analysis_child_u5"
ACUTE_CHILD_U5_JOIN_COLUMNS = [
    "DEATHS_28_364D",
    "DEATHS_1_4Y",
    "DEATHS_UNDER5_TOTAL",
]
ACUTE_LIVE_BIRTH_FALLBACK_COLUMNS = ["MUNICIPALITY_CODE", "LIVE_BIRTHS_TOTAL"]

PERINATAL_BASE_DATASET = "gold_mnch_perinatal"
PERINATAL_DIRECT_COLUMNS = SHARED_ID_COLUMNS + SHARED_EXPOSURE_COLUMNS + [
    "PRETERM_FLAG",
    "VERY_PRETERM_FLAG",
    "GEST_DAYS_LOST_BELOW_37W",
    "LOW_BIRTH_WEIGHT_LT2500_TOTAL",
    "PRETERM_LOW_BIRTH_WEIGHT_TOTAL",
    "PRENATAL_VISITS_8PLUS_FLAG",
    "STILLBIRTH_PRIMARY_GTE28_FLAG",
    "STILLBIRTH_SENS_GTE22_FLAG",
    "LIVE_BIRTHS_TOTAL",
    "TOTAL_BIRTHS_PRIMARY",
    "TOTAL_BIRTHS_SENSITIVITY_GTE22",
    "OFFSET_LOG_TOTAL_BIRTHS_PRIMARY",
    "OFFSET_LOG_TOTAL_BIRTHS_SENSITIVITY_GTE22",
]

MATERNAL_BASE_DATASET = "gold_analysis_wra_15_49"
MATERNAL_DIRECT_COLUMNS = SHARED_ID_COLUMNS + SHARED_EXPOSURE_COLUMNS + [
    "OBSTETRIC_ADMISSIONS_TOTAL",
    "HDP_ANY_TOTAL",
    "HDP_SEVERE_TOTAL",
    "PRETERM_LABOR_TOTAL",
    "GDM_GESTATIONAL_TOTAL",
    "APH_ANTEPARTUM_TOTAL",
    "POP_WRA_15_49",
    "OFFSET_LOG_POP_WRA_15_49",
    "WRA_AGE_SCOPE_NOTE",
    "SENSITIVITY_ONLY_FLAG",
]

CONTEXT_SOURCE_DATASET = "gold_municipality_context_static"
CONTEXT_SOURCE_COLUMNS = [
    "municipality_code",
    "PROP_URBAN",
    "GINI_EXPANDED",
    "IDH_EXPANDED",
    "ALTITUDE_MEAN_M",
    "COASTAL_FLAG",
    "JOIN_STATUS_FLAG",
]

ACUTE_LAG_SPECS = [
    {
        "window": 1,
        "public_suffix": "lag0",
        "label_suffix": "lag 0",
        "description_template": "Same-day {label} aligned to the acute outcome date.",
    },
    {
        "window": 2,
        "public_suffix": "lag0_1_mean",
        "label_suffix": "lag 0 to 1 mean",
        "description_template": "Mean {label} across lag days 0 through 1 aligned to the acute outcome date.",
    },
    {
        "window": 4,
        "public_suffix": "lag0_3_mean",
        "label_suffix": "lag 0 to 3 mean",
        "description_template": "Mean {label} across lag days 0 through 3 aligned to the acute outcome date.",
    },
    {
        "window": 8,
        "public_suffix": "lag0_7_mean",
        "label_suffix": "lag 0 to 7 mean",
        "description_template": "Mean {label} across lag days 0 through 7 aligned to the acute outcome date.",
    },
]

EVENT_WINDOW_SPECS = [
    {
        "window": 7,
        "public_suffix": "days_0_6_before_event_mean",
        "label_suffix": "days 0 to 6 before event mean",
        "description_template": "Mean {label} across days 0 through 6 before the delivery or event date; this is an observable event-aligned window, not a gestational causal window.",
    },
    {
        "window": 14,
        "public_suffix": "days_0_13_before_event_mean",
        "label_suffix": "days 0 to 13 before event mean",
        "description_template": "Mean {label} across days 0 through 13 before the delivery or event date; this is an observable event-aligned window, not a gestational causal window.",
    },
    {
        "window": 28,
        "public_suffix": "days_0_27_before_event_mean",
        "label_suffix": "days 0 to 27 before event mean",
        "description_template": "Mean {label} across days 0 through 27 before the delivery or event date; this is an observable event-aligned window, not a gestational causal window.",
    },
]

ANNUALIZED_DENOMINATOR_SPECS = {
    "perinatal_delivery_window_mart": [
        {
            "source_public_column": "daily_live_birth_count",
            "derived_column": "annual_live_birth_count",
            "academic_label": "Annual live birth count",
            "description": "Annual live-birth denominator for the municipality-year, used as the under-one population proxy for preterm and very preterm outcomes.",
            "units": "births per municipality-year",
            "public_role": "denominator",
            "offset_column": "log_annual_live_birth_offset",
            "offset_label": "Log annual live-birth offset",
            "offset_description": "Log offset for the annual live-birth denominator used for preterm and very preterm outcomes.",
            "offset_units": "log births",
        },
        {
            "source_public_column": "total_birth_count_primary_definition",
            "derived_column": "annual_total_birth_count_primary_definition",
            "academic_label": "Annual total birth count under the primary definition",
            "description": "Annual total births denominator for the municipality-year under the primary stillbirth definition, equal to live births plus stillbirths.",
            "units": "births per municipality-year",
            "public_role": "denominator",
            "offset_column": "log_annual_total_birth_count_primary_offset",
            "offset_label": "Log annual total birth count offset under the primary definition",
            "offset_description": "Log offset for the annual total births denominator under the primary stillbirth definition.",
            "offset_units": "log births",
        },
        {
            "source_public_column": "total_birth_count_sensitivity_gte22_definition",
            "derived_column": "annual_total_birth_count_sensitivity_gte22_definition",
            "academic_label": "Annual total birth count under the sensitivity definition",
            "description": "Annual total births denominator for the municipality-year under the sensitivity stillbirth definition at 22 weeks or more, equal to live births plus stillbirths.",
            "units": "births per municipality-year",
            "public_role": "denominator",
            "offset_column": "log_annual_total_birth_count_sensitivity_gte22_offset",
            "offset_label": "Log annual total birth count offset under the sensitivity definition",
            "offset_description": "Log offset for the annual total births denominator under the sensitivity stillbirth definition at 22 weeks or more.",
            "offset_units": "log births",
        },
    ],
}


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _dataset_year_path(root: Path, dataset: str, year: int) -> Path:
    return root / DATA_GOLD_ROOT / dataset / f"YEAR={year}" / f"{dataset}_{year}.parquet"


def _load_alias_lookup(root: Path) -> dict[str, dict[str, Any]]:
    alias_payload = load_analysis_variable_aliases(root)
    lookup: dict[str, dict[str, Any]] = {}
    for row in alias_payload["aliases"]:
        canonical_variable = str(row["canonical_variable"])
        existing = lookup.get(canonical_variable)
        if existing is not None and str(existing["academic_variable_name"]) != str(row["academic_variable_name"]):
            raise ValueError(
                f"Canonical variable {canonical_variable!r} maps to multiple academic variable names in analysis aliases."
            )
        lookup[canonical_variable] = row
    return lookup


def _rename_map_for_columns(
    alias_lookup: dict[str, dict[str, Any]],
    canonical_columns: list[str],
) -> dict[str, str]:
    rename_map: dict[str, str] = {}
    for column in canonical_columns:
        if column not in alias_lookup:
            raise KeyError(f"Missing analysis alias entry for canonical variable {column!r}")
        rename_map[column] = str(alias_lookup[column]["academic_variable_name"])
    return rename_map


def _read_parquet(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen gold input: {path}")
    parquet_file = pq.ParquetFile(path)
    available_columns = set(parquet_file.schema.names)
    selected_columns = [column for column in columns if column in available_columns]
    frame = pd.read_parquet(path, columns=selected_columns)
    missing_columns = [column for column in columns if column not in available_columns]
    for column in missing_columns:
        frame[column] = pd.NA
    return frame.loc[:, columns]


def _load_year_with_lookback(
    root: Path,
    dataset: str,
    year: int,
    columns: list[str],
    lookback_days: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    previous_path = _dataset_year_path(root, dataset, year - 1)
    if lookback_days > 0 and previous_path.exists():
        previous = _read_parquet(previous_path, columns)
        previous["DATE"] = pd.to_datetime(previous["DATE"], errors="coerce")
        cutoff = pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=lookback_days)
        previous = previous.loc[previous["DATE"] >= cutoff].copy()
        frames.append(previous)

    current = _read_parquet(_dataset_year_path(root, dataset, year), columns)
    current["DATE"] = pd.to_datetime(current["DATE"], errors="coerce")
    frames.append(current)

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(["MUNICIPALITY_CODE", "DATE"], ignore_index=True)
    return frame


def _apply_rolling_features(
    frame: pd.DataFrame,
    exposure_public_columns: list[str],
    rolling_specs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    derived_columns: list[str] = []
    derived_metadata: list[dict[str, Any]] = []
    for exposure_column in exposure_public_columns:
        series = frame.groupby("municipality_code", sort=False)[exposure_column]
        for spec in rolling_specs:
            derived_column = f"{exposure_column}_{spec['public_suffix']}"
            if spec["window"] == 1:
                frame[derived_column] = frame[exposure_column]
            else:
                frame[derived_column] = (
                    series.rolling(spec["window"], min_periods=1).mean().reset_index(level=0, drop=True)
                )
            derived_columns.append(derived_column)
            derived_metadata.append(
                {
                    "source_exposure_public_column": exposure_column,
                    "academic_variable_name": derived_column,
                    "label_suffix": str(spec["label_suffix"]),
                    "description_template": str(spec["description_template"]),
                }
            )
    return frame, derived_columns, derived_metadata


def _safe_log_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    output = pd.Series(np.nan, index=series.index, dtype="float64")
    positive = numeric.gt(0)
    output.loc[positive] = np.log(numeric.loc[positive].astype(float))
    return output


def _apply_annualized_denominator_features(
    frame: pd.DataFrame,
    mart_name: str,
) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    specs = ANNUALIZED_DENOMINATOR_SPECS.get(mart_name, [])
    derived_columns: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    for spec in specs:
        source_public_column = str(spec["source_public_column"])
        if source_public_column not in frame.columns:
            continue
        annual_denominator = pd.to_numeric(frame[source_public_column], errors="coerce").groupby(
            frame["municipality_code"],
            sort=False,
        ).transform("sum")
        derived_column = str(spec["derived_column"])
        frame[derived_column] = annual_denominator
        derived_columns.append(derived_column)
        metadata_rows.append(
            {
                "mart_name": mart_name,
                "academic_variable_name": derived_column,
                "academic_label": str(spec["academic_label"]),
                "description": str(spec["description"]),
                "units": str(spec["units"]),
                "public_role": str(spec["public_role"]),
                "outcome_family": "perinatal_delivery_window_aggregate",
                "exposure_family": "",
                "provenance_only": False,
                "source_dataset": PERINATAL_BASE_DATASET,
                "source_columns": source_public_column,
                "canonical_field_reference": f"derived.{derived_column}",
                "derivation_rule": f"Annualized municipality-year sum derived from {source_public_column}.",
            }
        )
        offset_column = str(spec["offset_column"])
        frame[offset_column] = _safe_log_series(annual_denominator)
        derived_columns.append(offset_column)
        metadata_rows.append(
            {
                "mart_name": mart_name,
                "academic_variable_name": offset_column,
                "academic_label": str(spec["offset_label"]),
                "description": str(spec["offset_description"]),
                "units": str(spec["offset_units"]),
                "public_role": "offset",
                "outcome_family": "perinatal_delivery_window_aggregate",
                "exposure_family": "",
                "provenance_only": False,
                "source_dataset": PERINATAL_BASE_DATASET,
                "source_columns": source_public_column,
                "canonical_field_reference": f"derived.{offset_column}",
                "derivation_rule": f"Safe log of the annualized municipality-year denominator derived from {source_public_column}.",
            }
        )
    return frame, derived_columns, metadata_rows


def _load_acute_live_birth_fallback(root: Path, year: int) -> pd.DataFrame:
    fallback = _read_parquet(
        _dataset_year_path(root, PERINATAL_BASE_DATASET, year),
        ACUTE_LIVE_BIRTH_FALLBACK_COLUMNS,
    ).rename(columns={"MUNICIPALITY_CODE": "municipality_code"})
    fallback["municipality_code"] = fallback["municipality_code"].astype(str)
    annual_live_births = (
        pd.to_numeric(fallback["LIVE_BIRTHS_TOTAL"], errors="coerce")
        .groupby(fallback["municipality_code"], sort=False)
        .sum(min_count=1)
        .reset_index(name="annual_live_birth_count_fallback")
    )
    annual_live_births["log_annual_live_birth_offset_fallback"] = _safe_log_series(
        annual_live_births["annual_live_birth_count_fallback"]
    )
    return annual_live_births


def _apply_acute_live_birth_fallback(
    root: Path,
    year: int,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required_columns = {
        "municipality_code",
        "annual_live_birth_count",
        "log_annual_live_birth_offset",
    }
    if not required_columns.issubset(frame.columns):
        return frame, {"live_birth_rows_backfilled": 0, "offset_rows_backfilled": 0}

    fallback = _load_acute_live_birth_fallback(root, year)
    if fallback.empty:
        return frame, {"live_birth_rows_backfilled": 0, "offset_rows_backfilled": 0}

    working = frame.merge(fallback, on="municipality_code", how="left")
    annual_live_birth_count = pd.to_numeric(working["annual_live_birth_count"], errors="coerce")
    annual_live_birth_fallback = pd.to_numeric(working["annual_live_birth_count_fallback"], errors="coerce")
    live_birth_missing = annual_live_birth_count.isna() | annual_live_birth_count.le(0)
    live_birth_backfill_mask = live_birth_missing & annual_live_birth_fallback.gt(0)
    annual_live_birth_count = annual_live_birth_count.where(~live_birth_backfill_mask, annual_live_birth_fallback)

    log_annual_live_birth_offset = pd.to_numeric(working["log_annual_live_birth_offset"], errors="coerce")
    derived_log_offset = _safe_log_series(annual_live_birth_count)
    offset_backfill_mask = (
        (log_annual_live_birth_offset.isna() | live_birth_backfill_mask)
        & annual_live_birth_count.gt(0)
        & derived_log_offset.notna()
    )
    log_annual_live_birth_offset = log_annual_live_birth_offset.where(~offset_backfill_mask, derived_log_offset)

    working["annual_live_birth_count"] = annual_live_birth_count
    working["log_annual_live_birth_offset"] = log_annual_live_birth_offset

    repaired_mask = live_birth_backfill_mask | offset_backfill_mask
    if repaired_mask.any():
        if "denominator_missing_flag" in working.columns:
            working.loc[repaired_mask, "denominator_missing_flag"] = 0
        if "denominator_zero_flag" in working.columns:
            working.loc[repaired_mask, "denominator_zero_flag"] = 0
        if "rate_not_computable_flag" in working.columns:
            working.loc[repaired_mask, "rate_not_computable_flag"] = 0
        if "offset_valid_flag" in working.columns:
            working.loc[repaired_mask, "offset_valid_flag"] = 1

    working = working.drop(
        columns=["annual_live_birth_count_fallback", "log_annual_live_birth_offset_fallback"],
        errors="ignore",
    )
    return working, {
        "live_birth_rows_backfilled": int(live_birth_backfill_mask.sum()),
        "offset_rows_backfilled": int(offset_backfill_mask.sum()),
    }


def _validate_public_columns(frame: pd.DataFrame) -> None:
    invalid_columns = validate_public_output_columns(frame.columns.tolist(), set())
    if invalid_columns:
        raise ValueError(
            "Shared mart output contains non-public-facing column names: " + ", ".join(sorted(invalid_columns))
        )


def _metadata_row_from_alias(
    mart_name: str,
    source_dataset: str,
    source_column: str,
    alias_row: dict[str, Any],
    derivation_rule: str,
) -> dict[str, Any]:
    return {
        "mart_name": mart_name,
        "academic_variable_name": str(alias_row["academic_variable_name"]),
        "academic_label": str(alias_row["academic_label"]),
        "description": str(alias_row["description"]),
        "units": str(alias_row["units"]),
        "public_role": str(alias_row["public_role"]),
        "outcome_family": str(alias_row["outcome_family"]),
        "exposure_family": str(alias_row.get("exposure_family") or ""),
        "provenance_only": bool(alias_row["provenance_only"]),
        "source_dataset": source_dataset,
        "source_columns": source_column,
        "canonical_field_reference": f"{alias_row['canonical_dataset']}.{alias_row['canonical_variable']}",
        "derivation_rule": derivation_rule,
    }


def _direct_derivation_rule(mart_name: str, source_column: str) -> str:
    if mart_name == "acute_daily_mart" and source_column == "LIVE_BIRTHS_TOTAL_ANNUAL":
        return (
            "Direct carry-forward from frozen gold input after academic-English renaming, "
            "with municipality-year fallback from gold_mnch_perinatal.LIVE_BIRTHS_TOTAL aggregation "
            "when the acute-source annual live-birth denominator is missing."
        )
    if mart_name == "acute_daily_mart" and source_column == "OFFSET_LOG_LIVE_BIRTHS":
        return (
            "Direct carry-forward from frozen gold input after academic-English renaming, "
            "with municipality-year fallback recomputed as the safe log of the perinatal-derived annual "
            "live-birth denominator when the acute-source offset is missing."
        )
    return "Direct carry-forward from frozen gold input after academic-English renaming."


def _derived_metadata_row(
    mart_name: str,
    source_dataset: str,
    source_alias_row: dict[str, Any],
    derived_column: str,
    label_suffix: str,
    description_template: str,
) -> dict[str, Any]:
    base_label = str(source_alias_row["academic_label"])
    description = description_template.format(label=base_label.lower())
    return {
        "mart_name": mart_name,
        "academic_variable_name": derived_column,
        "academic_label": f"{base_label} {label_suffix.title()}",
        "description": description,
        "units": str(source_alias_row["units"]),
        "public_role": "derived_exposure",
        "outcome_family": str(source_alias_row["outcome_family"]),
        "exposure_family": str(source_alias_row.get("exposure_family") or ""),
        "provenance_only": False,
        "source_dataset": source_dataset,
        "source_columns": str(source_alias_row["canonical_variable"]),
        "canonical_field_reference": f"{source_alias_row['canonical_dataset']}.{source_alias_row['canonical_variable']}",
        "derivation_rule": f"Rolling mean with window defined by {label_suffix}.",
    }


def _build_partitioned_mart(
    root: Path,
    years: list[int],
    mart_name: str,
    source_dataset: str,
    source_columns: list[str],
    exposure_canonical_columns: list[str],
    rolling_specs: list[dict[str, Any]],
    alias_lookup: dict[str, dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    join_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    marts_root = analysis_shared_marts_root(root) / mart_name
    marts_root.mkdir(parents=True, exist_ok=True)

    normalized_join_specs = [
        {"dataset": str(spec["dataset"]), "columns": list(spec["columns"])}
        for spec in (join_specs or [])
    ]
    rename_map = _rename_map_for_columns(alias_lookup, source_columns)
    direct_public_columns = [rename_map[column] for column in source_columns]
    exposure_public_columns = [rename_map[column] for column in exposure_canonical_columns]
    lookback_days = max(spec["window"] for spec in rolling_specs) - 1 if rolling_specs else 0
    written_paths: list[str] = []
    row_count_total = 0

    for year in years:
        working = _load_year_with_lookback(root, source_dataset, year, source_columns, lookback_days)
        working = working.rename(columns=rename_map)
        working = working.sort_values(["municipality_code", "observation_date"], ignore_index=True)
        working, derived_columns, derived_metadata = _apply_rolling_features(
            frame=working,
            exposure_public_columns=exposure_public_columns,
            rolling_specs=rolling_specs,
        )
        current = working.loc[working["calendar_year"].eq(year)].copy()
        current, annualized_denominator_columns, annualized_denominator_metadata = _apply_annualized_denominator_features(
            current,
            mart_name=mart_name,
        )

        for join_spec in normalized_join_specs:
            join_dataset = str(join_spec["dataset"])
            join_columns = list(join_spec["columns"])
            join_frame = _read_parquet(_dataset_year_path(root, join_dataset, year), SHARED_ID_COLUMNS + join_columns)
            join_frame["DATE"] = pd.to_datetime(join_frame["DATE"], errors="coerce")
            join_rename_map = _rename_map_for_columns(alias_lookup, join_columns)
            join_frame = join_frame.rename(
                columns={
                    "MUNICIPALITY_CODE": "municipality_code",
                    "DATE": "observation_date",
                    **join_rename_map,
                }
            )
            current = current.merge(
                join_frame,
                on=["municipality_code", "observation_date"],
                how="left",
                validate="one_to_one",
            )

        if mart_name == "acute_daily_mart":
            current, acute_fallback_summary = _apply_acute_live_birth_fallback(root, year, current)
        else:
            acute_fallback_summary = {"live_birth_rows_backfilled": 0, "offset_rows_backfilled": 0}

        selected_columns = direct_public_columns + derived_columns + annualized_denominator_columns
        for join_spec in normalized_join_specs:
            selected_columns.extend(
                _rename_map_for_columns(alias_lookup, list(join_spec["columns"])).values()
            )
        selected_columns = list(dict.fromkeys(selected_columns))
        current = current.loc[:, selected_columns]
        _validate_public_columns(current)

        output_path = marts_root / f"YEAR={year}" / f"{mart_name}_{year}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        current.to_parquet(output_path, index=False)

        written_paths.append(_relative(root, output_path))
        row_count_total += int(current.shape[0])
        inventory_rows.append(
            {
                "mart_name": mart_name,
                "partition_label": str(year),
                "row_count": int(current.shape[0]),
                "column_count": int(current.shape[1]),
                "output_path": _relative(root, output_path),
                "live_birth_rows_backfilled": int(acute_fallback_summary["live_birth_rows_backfilled"]),
                "offset_rows_backfilled": int(acute_fallback_summary["offset_rows_backfilled"]),
            }
        )

    for column in source_columns:
        metadata_rows.append(
            _metadata_row_from_alias(
                mart_name=mart_name,
                source_dataset=source_dataset,
                source_column=column,
                alias_row=alias_lookup[column],
                derivation_rule=_direct_derivation_rule(mart_name, column),
            )
        )
    for join_spec in normalized_join_specs:
        join_dataset = str(join_spec["dataset"])
        for column in list(join_spec["columns"]):
            metadata_rows.append(
                _metadata_row_from_alias(
                    mart_name=mart_name,
                    source_dataset=join_dataset,
                    source_column=column,
                    alias_row=alias_lookup[column],
                    derivation_rule="Direct left join from a frozen gold supporting table after academic-English renaming.",
                )
            )
    for derived in derived_metadata:
        source_alias_row = alias_lookup[
            next(
                canonical
                for canonical, public_name in _rename_map_for_columns(alias_lookup, exposure_canonical_columns).items()
                if public_name == derived["source_exposure_public_column"]
            )
        ]
        metadata_rows.append(
            _derived_metadata_row(
                mart_name=mart_name,
                source_dataset=source_dataset,
                source_alias_row=source_alias_row,
                derived_column=str(derived["academic_variable_name"]),
                label_suffix=str(derived["label_suffix"]),
                description_template=str(derived["description_template"]),
            )
        )
    metadata_rows.extend(annualized_denominator_metadata)

    return {
        "mart_name": mart_name,
        "source_dataset": source_dataset,
        "partitions_written": len(written_paths),
        "row_count_total": row_count_total,
        "output_root": _relative(root, marts_root),
        "written_partitions": written_paths,
    }


def _build_context_lookup(
    root: Path,
    alias_lookup: dict[str, dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    context_root = analysis_shared_marts_root(root)
    context_root.mkdir(parents=True, exist_ok=True)
    context = pd.read_parquet(root / CONTEXT_STATIC_PATH, columns=CONTEXT_SOURCE_COLUMNS)
    context = context.rename(
        columns={
            "municipality_code": "municipality_code",
            "PROP_URBAN": alias_lookup["PROP_URBAN"]["academic_variable_name"],
            "GINI_EXPANDED": alias_lookup["GINI_EXPANDED"]["academic_variable_name"],
            "IDH_EXPANDED": alias_lookup["IDH_EXPANDED"]["academic_variable_name"],
            "ALTITUDE_MEAN_M": alias_lookup["ALTITUDE_MEAN_M"]["academic_variable_name"],
            "COASTAL_FLAG": alias_lookup["COASTAL_FLAG"]["academic_variable_name"],
            "JOIN_STATUS_FLAG": alias_lookup["JOIN_STATUS_FLAG"]["academic_variable_name"],
        }
    )
    context = context.loc[
        :,
        [
            "municipality_code",
            alias_lookup["PROP_URBAN"]["academic_variable_name"],
            alias_lookup["GINI_EXPANDED"]["academic_variable_name"],
            alias_lookup["IDH_EXPANDED"]["academic_variable_name"],
            alias_lookup["ALTITUDE_MEAN_M"]["academic_variable_name"],
            alias_lookup["COASTAL_FLAG"]["academic_variable_name"],
            alias_lookup["JOIN_STATUS_FLAG"]["academic_variable_name"],
        ],
    ].sort_values("municipality_code", ignore_index=True)
    _validate_public_columns(context)

    output_path = context_root / "context_lookup.parquet"
    context.to_parquet(output_path, index=False)
    inventory_rows.append(
        {
            "mart_name": "context_lookup",
            "partition_label": "static",
            "row_count": int(context.shape[0]),
            "column_count": int(context.shape[1]),
            "output_path": _relative(root, output_path),
        }
    )
    metadata_rows.append(
        {
            "mart_name": "context_lookup",
            "academic_variable_name": "municipality_code",
            "academic_label": "Municipality code",
            "description": "Resolved six-digit municipality code used for analytical linkage.",
            "units": "code",
            "public_role": "identifier",
            "outcome_family": "shared",
            "exposure_family": "",
            "provenance_only": False,
            "source_dataset": CONTEXT_SOURCE_DATASET,
            "source_columns": "municipality_code",
            "canonical_field_reference": "shared.MUNICIPALITY_CODE",
            "derivation_rule": "Direct carry-forward from the frozen static municipality-context lookup with academic-English normalization.",
        }
    )
    for column in ["PROP_URBAN", "GINI_EXPANDED", "IDH_EXPANDED", "ALTITUDE_MEAN_M", "COASTAL_FLAG", "JOIN_STATUS_FLAG"]:
        metadata_rows.append(
            _metadata_row_from_alias(
                mart_name="context_lookup",
                source_dataset=CONTEXT_SOURCE_DATASET,
                source_column=column,
                alias_row=alias_lookup[column],
                derivation_rule="Direct carry-forward from the frozen static municipality-context lookup after academic-English renaming.",
            )
        )

    return {
        "mart_name": "context_lookup",
        "source_dataset": CONTEXT_SOURCE_DATASET,
        "partitions_written": 1,
        "row_count_total": int(context.shape[0]),
        "output_root": _relative(root, output_path),
        "written_partitions": [_relative(root, output_path)],
    }


def _render_methods_note(config: dict[str, Any], outputs: dict[str, str]) -> str:
    return (
        "# Shared Mart Methods Note\n\n"
        "The official steps pipeline materializes shared analytical marts on top of frozen gold inputs without modifying canonical gold tables or canonical manifest packages.\n\n"
        "## Acute Daily Mart\n\n"
        "- Source dataset: `gold_mnch_mortality`, with the acute-care proxy joined from `gold_mnch_panel` and child age-split mortality extensions joined from `gold_analysis_child_u5`.\n"
        "- Exposure derivation: same-day exposure plus rolling means for lag windows 0 to 1, 0 to 3, and 0 to 7 days.\n"
        "- Denominator handling: annual live births are carried from mortality when present and backfilled from municipality-year aggregates of `gold_mnch_perinatal.LIVE_BIRTHS_TOTAL` when the acute source is missing; under-five population remains the acute-source denominator.\n"
        "- Outcome scope: neonatal, infant, under-five, postneonatal, and ages 1 to 4 mortality, child hospitalization, SIH heat-coded child hospitalization, heat-coded mortality, and the descriptive-only acute-care proxy.\n\n"
        "## Perinatal Delivery-Window Mart\n\n"
        "- Source dataset: `gold_mnch_perinatal`.\n"
        "- Exposure derivation: rolling means across days 0 to 6, 0 to 13, and 0 to 27 before the delivery date.\n"
        "- Outcome scope: preterm, very preterm, gestational severity, low birth weight, ANC 8+, and stillbirth endpoints.\n"
        "- Interpretation rule: these windows are observable event-aligned summaries and must not be interpreted as trimester-specific or whole-pregnancy causal exposures.\n\n"
        "## Maternal Operational Mart\n\n"
        "- Source dataset: `gold_analysis_wra_15_49`.\n"
        "- Exposure derivation: rolling means across days 0 to 6, 0 to 13, and 0 to 27 before the admission or event date.\n"
        "- Denominator handling: women aged 15 to 49 population counts and log offsets are carried forward from the frozen analytical view.\n\n"
        "- Outcome scope: obstetric admissions, HDP, severe HDP, preterm labor, gestational diabetes, and antepartum haemorrhage.\n\n"
        "## Context Lookup\n\n"
        "- Source dataset: `gold_municipality_context_static`.\n"
        "- Scope: static municipality-level contextual covariates for later stratification and heterogeneity analyses.\n\n"
        "## Metadata Outputs\n\n"
        f"- Mart variable dictionary: `{outputs['mart_variable_dictionary']}`\n"
        f"- Mart source crosswalk: `{outputs['mart_source_crosswalk']}`\n"
    )


def _render_mart_build_report(
    config: dict[str, Any],
    mart_summaries: list[dict[str, Any]],
    inventory_frame: pd.DataFrame,
    outputs: dict[str, str],
) -> str:
    lines = [
        "# Shared Mart Build Report",
        "",
        f"- Analysis title: `{config['analysis_title']}`",
        f"- Output inventory: `{outputs['analysis_mart_inventory']}`",
        f"- Methods note: `{outputs['mart_methods_note']}`",
        "",
        "## Mart Summaries",
        "",
    ]
    for summary in mart_summaries:
        lines.append(
            "- "
            f"{summary['mart_name']}: source_dataset={summary['source_dataset']}, "
            f"partitions_written={summary['partitions_written']}, "
            f"row_count_total={summary['row_count_total']}, "
            f"output_root={summary['output_root']}"
        )
    lines.extend(["", "## Inventory", ""])
    for row in inventory_frame.to_dict(orient="records"):
        lines.append(
            "- "
            f"{row['mart_name']} [{row['partition_label']}]: "
            f"rows={row['row_count']}, columns={row['column_count']}, output={row['output_path']}"
        )
    return "\n".join(lines) + "\n"


def build_analysis_shared_marts(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    config = load_round_config(root, config_path=config_path)
    alias_lookup = _load_alias_lookup(root)
    marts_root = analysis_shared_marts_root(root)
    metadata_root = analysis_shared_marts_root(root)
    marts_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    inventory_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    mart_summaries = [
        _build_partitioned_mart(
            root=root,
            years=list(config["years"]),
            mart_name="acute_daily_mart",
            source_dataset=ACUTE_BASE_DATASET,
            source_columns=ACUTE_DIRECT_COLUMNS,
            exposure_canonical_columns=EXPOSURE_CANONICAL_COLUMNS,
            rolling_specs=ACUTE_LAG_SPECS,
            alias_lookup=alias_lookup,
            inventory_rows=inventory_rows,
            metadata_rows=metadata_rows,
            join_specs=[
                {
                    "dataset": ACUTE_JOIN_DATASET,
                    "columns": ACUTE_PANEL_JOIN_COLUMNS,
                },
                {
                    "dataset": ACUTE_CHILD_U5_JOIN_DATASET,
                    "columns": ACUTE_CHILD_U5_JOIN_COLUMNS,
                },
            ],
        ),
        _build_partitioned_mart(
            root=root,
            years=list(config["years"]),
            mart_name="perinatal_delivery_window_mart",
            source_dataset=PERINATAL_BASE_DATASET,
            source_columns=PERINATAL_DIRECT_COLUMNS,
            exposure_canonical_columns=EXPOSURE_CANONICAL_COLUMNS,
            rolling_specs=EVENT_WINDOW_SPECS,
            alias_lookup=alias_lookup,
            inventory_rows=inventory_rows,
            metadata_rows=metadata_rows,
        ),
        _build_partitioned_mart(
            root=root,
            years=list(config["years"]),
            mart_name="maternal_operational_mart",
            source_dataset=MATERNAL_BASE_DATASET,
            source_columns=MATERNAL_DIRECT_COLUMNS,
            exposure_canonical_columns=EXPOSURE_CANONICAL_COLUMNS,
            rolling_specs=EVENT_WINDOW_SPECS,
            alias_lookup=alias_lookup,
            inventory_rows=inventory_rows,
            metadata_rows=metadata_rows,
        ),
        _build_context_lookup(
            root=root,
            alias_lookup=alias_lookup,
            inventory_rows=inventory_rows,
            metadata_rows=metadata_rows,
        ),
    ]

    inventory_frame = pd.DataFrame(inventory_rows).sort_values(
        ["mart_name", "partition_label"], ignore_index=True
    )
    mart_crosswalk_frame = pd.DataFrame(metadata_rows).drop_duplicates().sort_values(
        ["mart_name", "academic_variable_name", "source_dataset"], ignore_index=True
    )
    mart_dictionary_frame = mart_crosswalk_frame.loc[
        :,
        [
            "mart_name",
            "academic_variable_name",
            "academic_label",
            "description",
            "units",
            "public_role",
            "outcome_family",
            "exposure_family",
            "provenance_only",
        ],
    ].drop_duplicates().sort_values(["mart_name", "academic_variable_name"], ignore_index=True)

    inventory_path = marts_root / "analysis_mart_inventory.csv"
    report_path = marts_root / "mart_build_report.md"
    mart_dictionary_path = metadata_root / "shared_mart_variable_dictionary.csv"
    mart_crosswalk_path = metadata_root / "shared_mart_source_crosswalk.csv"
    methods_note_path = metadata_root / "mart_methods_note.md"

    inventory_frame.to_csv(inventory_path, index=False)
    mart_dictionary_frame.to_csv(mart_dictionary_path, index=False)
    mart_crosswalk_frame.to_csv(mart_crosswalk_path, index=False)

    outputs = {
        "analysis_mart_inventory": _relative(root, inventory_path),
        "mart_build_report": _relative(root, report_path),
        "mart_variable_dictionary": _relative(root, mart_dictionary_path),
        "mart_source_crosswalk": _relative(root, mart_crosswalk_path),
        "mart_methods_note": _relative(root, methods_note_path),
    }

    methods_note_path.write_text(_render_methods_note(config, outputs), encoding="utf-8")
    report_path.write_text(
        _render_mart_build_report(config, mart_summaries, inventory_frame, outputs),
        encoding="utf-8",
    )

    return {
        "status": "PASS",
        "stage": "shared_analysis_marts",
        "analysis_id": config["analysis_id"],
        "analysis_title": config["analysis_title"],
        "mart_summaries": mart_summaries,
        "outputs": outputs,
    }
