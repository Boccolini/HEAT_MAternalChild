from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .config import load_round_config
from .registry import (
    analysis_shared_marts_root,
    analysis_step0_root,
    analysis_step1_root,
    ensure_step1_directory_structure,
    incremental_publication_root,
)
from .shared_marts import build_analysis_shared_marts
from .step0 import build_step0_freeze


ACTIVE_ANALYSIS_ROLES = {"primary", "sensitivity", "descriptive_only"}
PRIMARY_ANALYSIS_ROLE = "primary"
PROMOTED_ANALYSIS_ROLES = {"primary", "sensitivity"}
EXPOSURE_COLUMNS = [
    "maximum_daily_temperature_c",
    "mean_daily_relative_humidity_pct",
    "heatwave_duration_days_p95_threshold",
    "approximate_heat_index_c",
]
FALLBACK_COLUMN = "any_climate_fallback_imputation_flag"
MART_BY_OUTCOME_FAMILY = {
    "acute_daily": "acute_daily_mart",
    "panel_proxy": "acute_daily_mart",
    "perinatal_delivery_window_aggregate": "perinatal_delivery_window_mart",
    "maternal_operational": "maternal_operational_mart",
}
MONTH_TO_SEASON = {
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
    12: "DJF",
}
SEASON_ORDER = {"DJF": 1, "MAM": 2, "JJA": 3, "SON": 4}
FALLBACK_SHARE_THRESHOLD = 0.10
SUPPORT_THRESHOLDS = {
    "min_total_count_all_years": 100.0,
    "min_nonzero_day_count_all_years": 50,
    "min_years_with_events": 3,
    "min_municipalities_with_events_in_year": 20,
}
STEP1_PUBLICATION_FILES = {
    "02_step1_descriptive_readiness.md": "# Step 1 Descriptive Readiness\n\nUse this note to carry manuscript-facing descriptive readiness decisions.\n",
    "03_step1_decision_log.md": "# Step 1 Decision Log\n\nRecord downgrade, cut, and proceed decisions after the descriptive atlas review.\n",
}
DENOMINATOR_COLUMN_SPECS = {
    "annual_live_births": {
        "column": "annual_live_birth_count",
        "offset_column": "log_annual_live_birth_offset",
        "rate_multiplier": 1000.0,
        "rate_label": "per_1000_live_births",
    },
    "under_five_population": {
        "column": "under_five_population_count",
        "offset_column": "log_under_five_population_offset",
        "rate_multiplier": 100000.0,
        "rate_label": "per_100000_under_five_population",
    },
    "daily_live_births": {
        "column": "daily_live_birth_count",
        "offset_column": None,
        "rate_multiplier": 1000.0,
        "rate_label": "per_1000_live_births",
    },
    "annual_total_births_primary": {
        "column": "annual_total_birth_count_primary_definition",
        "offset_column": "log_annual_total_birth_count_primary_offset",
        "rate_multiplier": 1000.0,
        "rate_label": "per_1000_total_births_primary",
    },
    "annual_total_births_sensitivity_gte22": {
        "column": "annual_total_birth_count_sensitivity_gte22_definition",
        "offset_column": "log_annual_total_birth_count_sensitivity_gte22_offset",
        "rate_multiplier": 1000.0,
        "rate_label": "per_1000_total_births_gte22_sensitivity",
    },
    "total_births_primary": {
        "column": "total_birth_count_primary_definition",
        "offset_column": "log_total_birth_count_primary_offset",
        "rate_multiplier": 1000.0,
        "rate_label": "per_1000_total_births_primary",
    },
    "total_births_sensitivity_gte22": {
        "column": "total_birth_count_sensitivity_gte22_definition",
        "offset_column": "log_total_birth_count_sensitivity_gte22_offset",
        "rate_multiplier": 1000.0,
        "rate_label": "per_1000_total_births_gte22_sensitivity",
    },
    "women_15_49_population": {
        "column": "women_age_15_49_population_count",
        "offset_column": "log_women_age_15_49_population_offset",
        "rate_multiplier": 100000.0,
        "rate_label": "per_100000_women_15_49",
    },
    "women_15_45_population": {
        "column": "women_age_15_45_population_count",
        "offset_column": "log_women_age_15_45_sensitivity_population_offset",
        "rate_multiplier": 100000.0,
        "rate_label": "per_100000_women_15_45",
    },
    "not_applicable_count_or_descriptor": {
        "column": None,
        "offset_column": None,
        "rate_multiplier": None,
        "rate_label": "not_applicable",
    },
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _season_from_month(month: int | float | None) -> str | None:
    if month is None or pd.isna(month):
        return None
    return MONTH_TO_SEASON.get(int(month))


def _mart_partition_path(root: Path, mart_name: str, year: int) -> Path:
    return analysis_shared_marts_root(root) / mart_name / f"YEAR={year}" / f"{mart_name}_{year}.parquet"


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _flag_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int).gt(0)


def _mean_or_none(series: pd.Series) -> float | None:
    non_missing = series.dropna()
    if non_missing.empty:
        return None
    return float(non_missing.mean())


def _quantile_or_none(series: pd.Series, quantile: float) -> float | None:
    non_missing = series.dropna()
    if non_missing.empty:
        return None
    return float(non_missing.quantile(quantile))


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _string_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return text if text and text.lower() != "nan" else None


def _denominator_validity_masks(
    frame: pd.DataFrame,
    denominator_column: str | None,
    offset_column: str | None,
) -> dict[str, pd.Series]:
    if denominator_column and denominator_column in frame.columns:
        denominator_series = _numeric_series(frame, denominator_column)
        denominator_missing = denominator_series.isna()
        denominator_zero = denominator_series.fillna(0).le(0)
        if offset_column and offset_column in frame.columns:
            offset_series = _numeric_series(frame, offset_column)
            offset_invalid = offset_series.isna()
        else:
            offset_invalid = _flag_series(frame, "offset_valid_flag").eq(False)
        rate_not_computable = denominator_missing | denominator_zero | offset_invalid
        return {
            "denominator_missing": denominator_missing,
            "denominator_zero": denominator_zero,
            "offset_invalid": offset_invalid,
            "rate_not_computable": rate_not_computable,
        }
    return {
        "denominator_missing": _flag_series(frame, "denominator_missing_flag"),
        "denominator_zero": _flag_series(frame, "denominator_zero_flag"),
        "offset_invalid": ~_flag_series(frame, "offset_valid_flag"),
        "rate_not_computable": _flag_series(frame, "rate_not_computable_flag"),
    }


def _scaffold_step1_publication_workspace(root: Path) -> dict[str, object]:
    workspace_root = incremental_publication_root(root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    created_files: list[str] = []
    existing_files: list[str] = []
    for relative_path, contents in STEP1_PUBLICATION_FILES.items():
        target = workspace_root / relative_path
        if target.exists():
            existing_files.append(_relative(root, target))
            continue
        target.write_text(contents, encoding="utf-8")
        created_files.append(_relative(root, target))
    return {
        "workspace_root": _relative(root, workspace_root),
        "created_files": created_files,
        "existing_files": existing_files,
    }


def _build_analysis_units(
    inventory: pd.DataFrame,
    denominator_contract: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    active_inventory = inventory.loc[inventory["analysis_role"].astype(str).isin(ACTIVE_ANALYSIS_ROLES)].copy()
    merged = active_inventory.merge(
        denominator_contract.loc[:, ["outcome_id", "denominator_id", "scope"]],
        on="outcome_id",
        how="left",
        suffixes=("", "_contract"),
    )
    contract_consistent = True
    rows: list[dict[str, Any]] = []
    consistency_columns = [
        "canonical_dataset",
        "canonical_variable",
        "outcome_family",
        "analysis_role",
        "denominator_id",
        "seasonality_profile_id",
        "round_path",
        "temporal_anchor",
    ]
    for academic_variable_name, group in merged.groupby("academic_variable_name", sort=True):
        normalized = group.fillna("")
        for column in consistency_columns:
            if normalized[column].astype(str).nunique(dropna=False) > 1:
                contract_consistent = False
        first = group.iloc[0]
        mart_name = MART_BY_OUTCOME_FAMILY.get(str(first["outcome_family"]))
        if mart_name is None:
            contract_consistent = False
        denominator_id = str(first["denominator_id"])
        denominator_spec = DENOMINATOR_COLUMN_SPECS.get(denominator_id)
        if denominator_spec is None:
            contract_consistent = False
            denominator_spec = {"column": None, "rate_multiplier": None, "rate_label": "unknown"}
        rows.append(
            {
                "analysis_unit_id": str(academic_variable_name),
                "linked_outcome_ids": ";".join(sorted(group["outcome_id"].astype(str).tolist())),
                "linked_public_labels": ";".join(sorted(group["public_label"].astype(str).tolist())),
                "source_outcome_id_count": int(len(group)),
                "canonical_dataset": str(first["canonical_dataset"]),
                "canonical_variable": str(first["canonical_variable"]),
                "outcome_family": str(first["outcome_family"]),
                "analysis_role": str(first["analysis_role"]),
                "readiness_endpoint_name": str(first.get("readiness_endpoint_name") or ""),
                "denominator_id": denominator_id,
                "denominator_column": denominator_spec["column"],
                "offset_column": denominator_spec["offset_column"],
                "rate_multiplier": denominator_spec["rate_multiplier"],
                "rate_label": denominator_spec["rate_label"],
                "seasonality_profile_id": str(first["seasonality_profile_id"]),
                "round_path": str(first["round_path"]),
                "temporal_anchor": str(first["temporal_anchor"]),
                "mart_name": mart_name,
                "duplicate_outcome_mapping_flag": int(len(group) > 1),
                "promoted_outcome_flag": int(str(first["analysis_role"]) in PROMOTED_ANALYSIS_ROLES),
                "primary_outcome_flag": int(str(first["analysis_role"]) == PRIMARY_ANALYSIS_ROLE),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["analysis_role", "outcome_family", "analysis_unit_id"], ignore_index=True)
    return frame, contract_consistent


def _missing_mart_partitions(root: Path, mart_names: list[str], years: list[int]) -> list[str]:
    missing: list[str] = []
    for mart_name in mart_names:
        for year in years:
            path = _mart_partition_path(root, mart_name, year)
            if not path.exists():
                missing.append(_relative(root, path))
    return missing


def _missing_required_mart_columns(root: Path, required_columns_by_mart: dict[str, set[str]], years: list[int]) -> list[str]:
    missing: list[str] = []
    for mart_name, required_columns in required_columns_by_mart.items():
        for year in years:
            path = _mart_partition_path(root, mart_name, year)
            if not path.exists():
                continue
            available_columns = set(pq.ParquetFile(path).schema.names)
            for column in sorted(required_columns):
                if column not in available_columns:
                    missing.append(f"{_relative(root, path)}:{column}")
    return missing


def _ensure_internal_marts(
    root: Path,
    mart_names: list[str],
    years: list[int],
    required_columns_by_mart: dict[str, set[str]],
) -> dict[str, Any]:
    missing_before = _missing_mart_partitions(root, mart_names, years)
    missing_columns_before = _missing_required_mart_columns(root, required_columns_by_mart, years)
    if not missing_before and not missing_columns_before:
        return {
            "materialization_mode": "reused_existing_internal_marts",
            "missing_before_build": [],
            "missing_columns_before_build": [],
            "missing_after_build": [],
            "missing_columns_after_build": [],
            "partitions_checked": len(mart_names) * len(years),
        }

    shared_marts_report = build_analysis_shared_marts(root)
    missing_after = _missing_mart_partitions(root, mart_names, years)
    missing_columns_after = _missing_required_mart_columns(root, required_columns_by_mart, years)
    if missing_after or missing_columns_after:
        raise FileNotFoundError(
            "Step 1 could not materialize the required marts. Missing artifacts after internal build: "
            + ", ".join(missing_after + missing_columns_after)
        )
    return {
        "materialization_mode": "built_internal_marts",
        "missing_before_build": missing_before,
        "missing_columns_before_build": missing_columns_before,
        "missing_after_build": [],
        "missing_columns_after_build": [],
        "partitions_checked": len(mart_names) * len(years),
        "internal_materialization_report": shared_marts_report,
    }


def _load_mart_partition(root: Path, mart_name: str, year: int, columns: list[str]) -> pd.DataFrame:
    path = _mart_partition_path(root, mart_name, year)
    available_columns = set(pq.ParquetFile(path).schema.names)
    selected_columns = [column for column in columns if column in available_columns]
    frame = pd.read_parquet(path, columns=selected_columns)
    frame["calendar_month"] = pd.to_numeric(frame["calendar_month"], errors="coerce").astype("Int64")
    frame["season"] = frame["calendar_month"].map(_season_from_month)
    return frame


def _build_exposure_annual_rows(frame: pd.DataFrame, mart_name: str, year: int) -> list[dict[str, Any]]:
    fallback_share = _mean_or_none(_numeric_series(frame, FALLBACK_COLUMN)) if FALLBACK_COLUMN in frame.columns else None
    rows: list[dict[str, Any]] = []
    for exposure in EXPOSURE_COLUMNS:
        if exposure not in frame.columns:
            continue
        series = _numeric_series(frame, exposure)
        rows.append(
            {
                "mart_name": mart_name,
                "calendar_year": year,
                "exposure_variable": exposure,
                "row_count": int(len(series)),
                "missing_row_count": int(series.isna().sum()),
                "mean_value": _mean_or_none(series),
                "median_value": _quantile_or_none(series, 0.5),
                "p95_value": _quantile_or_none(series, 0.95),
                "minimum_value": _float_or_none(series.min()),
                "maximum_value": _float_or_none(series.max()),
                "fallback_share": fallback_share,
            }
        )
    return rows


def _build_exposure_time_rows(frame: pd.DataFrame, mart_name: str, year: int, bucket: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fallback_group = None
    if FALLBACK_COLUMN in frame.columns:
        fallback_group = _numeric_series(frame, FALLBACK_COLUMN).groupby(frame[bucket], dropna=False).mean()
    for exposure in EXPOSURE_COLUMNS:
        if exposure not in frame.columns:
            continue
        grouped = _numeric_series(frame, exposure).groupby(frame[bucket], dropna=False)
        summary = pd.DataFrame(
            {
                "row_count": grouped.size(),
                "missing_row_count": grouped.apply(lambda series: int(series.isna().sum())),
                "mean_value": grouped.mean(),
                "median_value": grouped.median(),
                "p95_value": grouped.quantile(0.95),
                "minimum_value": grouped.min(),
                "maximum_value": grouped.max(),
            }
        ).reset_index()
        for row in summary.to_dict(orient="records"):
            rows.append(
                {
                    "mart_name": mart_name,
                    "calendar_year": year,
                    "exposure_variable": exposure,
                    "calendar_month": int(row["calendar_month"]) if bucket == "calendar_month" and not pd.isna(row["calendar_month"]) else None,
                    "season": str(row["season"]) if bucket == "season" and not pd.isna(row["season"]) else None,
                    "row_count": int(row["row_count"]),
                    "missing_row_count": int(row["missing_row_count"]),
                    "mean_value": _float_or_none(row["mean_value"]),
                    "median_value": _float_or_none(row["median_value"]),
                    "p95_value": _float_or_none(row["p95_value"]),
                    "minimum_value": _float_or_none(row["minimum_value"]),
                    "maximum_value": _float_or_none(row["maximum_value"]),
                    "fallback_share": _float_or_none(fallback_group.get(row[bucket])) if fallback_group is not None else None,
                }
            )
    return rows


def _build_denominator_time_rows(
    frame: pd.DataFrame,
    mart_name: str,
    year: int,
    bucket: str,
    analysis_units: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for denominator_id, linked_units in analysis_units.groupby("denominator_id", sort=True):
        spec = DENOMINATOR_COLUMN_SPECS[str(denominator_id)]
        denominator_column = spec["column"]
        offset_column = spec["offset_column"]
        linked_analysis_units = ";".join(sorted(linked_units["analysis_unit_id"].astype(str).tolist()))
        validity = _denominator_validity_masks(frame, denominator_column, offset_column)
        denominator_zero = validity["denominator_zero"]
        denominator_missing = validity["denominator_missing"]
        rate_not_computable = validity["rate_not_computable"]
        offset_invalid = validity["offset_invalid"]
        if denominator_column is None or denominator_column not in frame.columns:
            group_values = sorted(value for value in frame[bucket].dropna().unique().tolist())
            for group_value in group_values:
                rows.append(
                    {
                        "mart_name": mart_name,
                        "denominator_id": str(denominator_id),
                        "denominator_column": denominator_column or "",
                        "linked_analysis_units": linked_analysis_units,
                        "calendar_year": year,
                        "calendar_month": int(group_value) if bucket == "calendar_month" else None,
                        "season": str(group_value) if bucket == "season" else None,
                        "rows_total": int((frame[bucket] == group_value).sum()),
                        "denominator_total": None,
                        "mean_denominator_value": None,
                        "p95_denominator_value": None,
                        "denominator_missing_row_count": 0,
                        "denominator_zero_row_count": 0,
                        "offset_invalid_row_count": int(offset_invalid.groupby(frame[bucket], dropna=False).sum().get(group_value, 0)),
                        "rate_not_computable_row_count": int(rate_not_computable.groupby(frame[bucket], dropna=False).sum().get(group_value, 0)),
                    }
                )
            continue

        denominator_grouped = _numeric_series(frame, denominator_column).groupby(frame[bucket], dropna=False)
        summary = pd.DataFrame(
            {
                "rows_total": denominator_grouped.size(),
                "denominator_total": denominator_grouped.sum(min_count=1),
                "mean_denominator_value": denominator_grouped.mean(),
                "p95_denominator_value": denominator_grouped.quantile(0.95),
                "denominator_missing_row_count": denominator_missing.groupby(frame[bucket], dropna=False).sum(),
                "denominator_zero_row_count": denominator_zero.groupby(frame[bucket], dropna=False).sum(),
                "offset_invalid_row_count": offset_invalid.groupby(frame[bucket], dropna=False).sum(),
                "rate_not_computable_row_count": rate_not_computable.groupby(frame[bucket], dropna=False).sum(),
            }
        ).reset_index()
        for row in summary.to_dict(orient="records"):
            rows.append(
                {
                    "mart_name": mart_name,
                    "denominator_id": str(denominator_id),
                    "denominator_column": denominator_column,
                    "linked_analysis_units": linked_analysis_units,
                    "calendar_year": year,
                    "calendar_month": int(row["calendar_month"]) if bucket == "calendar_month" and not pd.isna(row["calendar_month"]) else None,
                    "season": str(row["season"]) if bucket == "season" and not pd.isna(row["season"]) else None,
                    "rows_total": int(row["rows_total"]),
                    "denominator_total": _float_or_none(row["denominator_total"]),
                    "mean_denominator_value": _float_or_none(row["mean_denominator_value"]),
                    "p95_denominator_value": _float_or_none(row["p95_denominator_value"]),
                    "denominator_missing_row_count": int(row["denominator_missing_row_count"]),
                    "denominator_zero_row_count": int(row["denominator_zero_row_count"]),
                    "offset_invalid_row_count": int(row["offset_invalid_row_count"]),
                    "rate_not_computable_row_count": int(row["rate_not_computable_row_count"]),
                }
            )
    return rows


def _build_outcome_year_rows(
    frame: pd.DataFrame,
    mart_name: str,
    year: int,
    analysis_units: pd.DataFrame,
    annual_exposure_lookup: dict[tuple[str, int, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    denominator_rows: list[dict[str, Any]] = []
    missing_outcome_columns: list[str] = []
    missing_denominator_columns: list[str] = []
    municipality_count = int(frame["municipality_code"].astype(str).nunique())
    fallback_series = _numeric_series(frame, FALLBACK_COLUMN) if FALLBACK_COLUMN in frame.columns else pd.Series(dtype="float64")
    for unit in analysis_units.to_dict(orient="records"):
        outcome_column = str(unit["analysis_unit_id"])
        if outcome_column not in frame.columns:
            missing_outcome_columns.append(f"{mart_name}:{outcome_column}")
            continue
        outcome_series = _numeric_series(frame, outcome_column)
        positive_rows = outcome_series.fillna(0).gt(0)
        municipality_totals = frame.groupby("municipality_code", sort=False)[outcome_column].sum(min_count=1)
        municipalities_with_events = int(municipality_totals.fillna(0).gt(0).sum())
        total_count = _float_or_none(outcome_series.sum())
        denominator_total = None
        crude_rate = None
        positive_outcome_invalid_denominator_row_count = 0
        positive_outcome_missing_denominator_row_count = 0
        positive_outcome_zero_denominator_row_count = 0
        rate_not_computable_positive_row_count = 0

        denominator_column = _string_or_none(unit["denominator_column"])
        offset_column = _string_or_none(unit.get("offset_column"))
        validity = _denominator_validity_masks(frame, denominator_column, offset_column)
        denominator_zero = validity["denominator_zero"]
        denominator_missing = validity["denominator_missing"]
        rate_not_computable = validity["rate_not_computable"]
        offset_invalid = validity["offset_invalid"]
        if denominator_column:
            if denominator_column not in frame.columns:
                missing_denominator_columns.append(f"{mart_name}:{denominator_column}")
            else:
                denominator_series = _numeric_series(frame, str(denominator_column))
                denominator_total = _float_or_none(denominator_series.sum())
                if denominator_total not in {None, 0.0} and unit["rate_multiplier"] is not None and total_count is not None:
                    crude_rate = float(total_count / denominator_total * float(unit["rate_multiplier"]))
                positive_outcome_invalid_denominator_row_count = int((positive_rows & offset_invalid).sum())
                positive_outcome_missing_denominator_row_count = int((positive_rows & denominator_missing).sum())
                positive_outcome_zero_denominator_row_count = int((positive_rows & denominator_zero).sum())
                rate_not_computable_positive_row_count = int((positive_rows & rate_not_computable).sum())
                denominator_rows.append(
                    {
                        "analysis_unit_id": outcome_column,
                        "linked_outcome_ids": unit["linked_outcome_ids"],
                        "analysis_role": unit["analysis_role"],
                        "calendar_year": year,
                        "mart_name": mart_name,
                        "denominator_id": unit["denominator_id"],
                        "denominator_column": denominator_column,
                        "denominator_total": denominator_total,
                        "denominator_missing_row_count": int(denominator_missing.sum()),
                        "denominator_zero_row_count": int(denominator_zero.sum()),
                        "offset_invalid_row_count": int(offset_invalid.sum()),
                        "positive_outcome_invalid_denominator_row_count": positive_outcome_invalid_denominator_row_count,
                        "positive_outcome_missing_denominator_row_count": positive_outcome_missing_denominator_row_count,
                        "positive_outcome_zero_denominator_row_count": positive_outcome_zero_denominator_row_count,
                        "rate_not_computable_positive_row_count": rate_not_computable_positive_row_count,
                        "denominator_consistency_status": "PASS"
                        if positive_outcome_invalid_denominator_row_count == 0
                        and positive_outcome_missing_denominator_row_count == 0
                        and positive_outcome_zero_denominator_row_count == 0
                        and rate_not_computable_positive_row_count == 0
                        else "FAIL",
                    }
                )
        elif unit["denominator_id"] == "not_applicable_count_or_descriptor":
            denominator_rows.append(
                {
                    "analysis_unit_id": outcome_column,
                    "linked_outcome_ids": unit["linked_outcome_ids"],
                    "analysis_role": unit["analysis_role"],
                    "calendar_year": year,
                    "mart_name": mart_name,
                    "denominator_id": unit["denominator_id"],
                    "denominator_column": "",
                    "denominator_total": None,
                    "denominator_missing_row_count": 0,
                    "denominator_zero_row_count": 0,
                    "offset_invalid_row_count": int(offset_invalid.sum()),
                    "positive_outcome_invalid_denominator_row_count": 0,
                    "positive_outcome_missing_denominator_row_count": 0,
                    "positive_outcome_zero_denominator_row_count": 0,
                    "rate_not_computable_positive_row_count": 0,
                    "denominator_consistency_status": "PASS",
                }
            )

        top10_share = None
        if total_count not in {None, 0.0}:
            top10_share = float(municipality_totals.fillna(0).sort_values(ascending=False).head(10).sum() / float(total_count))

        fallback_share = _mean_or_none(fallback_series)
        event_row_fallback_share = _mean_or_none(fallback_series.loc[positive_rows]) if not fallback_series.empty and positive_rows.any() else None
        exposure_mean_tmax = annual_exposure_lookup.get((mart_name, year, "maximum_daily_temperature_c"), {}).get("mean_value")
        exposure_p95_tmax = annual_exposure_lookup.get((mart_name, year, "maximum_daily_temperature_c"), {}).get("p95_value")
        exposure_mean_hi = annual_exposure_lookup.get((mart_name, year, "approximate_heat_index_c"), {}).get("mean_value")
        exposure_p95_hi = annual_exposure_lookup.get((mart_name, year, "approximate_heat_index_c"), {}).get("p95_value")
        rows.append(
            {
                "analysis_unit_id": outcome_column,
                "linked_outcome_ids": unit["linked_outcome_ids"],
                "linked_public_labels": unit["linked_public_labels"],
                "analysis_role": unit["analysis_role"],
                "outcome_family": unit["outcome_family"],
                "mart_name": mart_name,
                "seasonality_profile_id": unit["seasonality_profile_id"],
                "denominator_id": unit["denominator_id"],
                "denominator_column": denominator_column or "",
                "rate_label": unit["rate_label"],
                "calendar_year": year,
                "total_count": total_count,
                "missing_row_count": int(outcome_series.isna().sum()),
                "mean_row_count": _mean_or_none(outcome_series),
                "nonzero_day_count": int(positive_rows.sum()),
                "nonzero_day_share": float(positive_rows.mean()),
                "municipality_day_count": int(len(outcome_series)),
                "municipalities_with_events": municipalities_with_events,
                "municipalities_with_events_share": float(municipalities_with_events / municipality_count) if municipality_count else None,
                "top_10_municipality_share": top10_share,
                "denominator_total": denominator_total,
                "crude_rate": crude_rate,
                "positive_outcome_invalid_denominator_row_count": positive_outcome_invalid_denominator_row_count,
                "positive_outcome_missing_denominator_row_count": positive_outcome_missing_denominator_row_count,
                "positive_outcome_zero_denominator_row_count": positive_outcome_zero_denominator_row_count,
                "rate_not_computable_positive_row_count": rate_not_computable_positive_row_count,
                "climate_fallback_share": fallback_share,
                "event_row_climate_fallback_share": event_row_fallback_share,
                "maximum_daily_temperature_mean": exposure_mean_tmax,
                "maximum_daily_temperature_p95": exposure_p95_tmax,
                "approximate_heat_index_mean": exposure_mean_hi,
                "approximate_heat_index_p95": exposure_p95_hi,
            }
        )
    return rows, denominator_rows, missing_outcome_columns, missing_denominator_columns


def _build_outcome_time_rows(
    frame: pd.DataFrame,
    mart_name: str,
    year: int,
    analysis_units: pd.DataFrame,
    bucket: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fallback_series = _numeric_series(frame, FALLBACK_COLUMN) if FALLBACK_COLUMN in frame.columns else pd.Series(dtype="float64")
    fallback_group = fallback_series.groupby(frame[bucket], dropna=False).mean() if not fallback_series.empty else None
    for unit in analysis_units.to_dict(orient="records"):
        outcome_column = str(unit["analysis_unit_id"])
        if outcome_column not in frame.columns:
            continue
        outcome_series = _numeric_series(frame, outcome_column)
        denominator_column = _string_or_none(unit["denominator_column"])
        offset_column = _string_or_none(unit.get("offset_column"))
        validity = _denominator_validity_masks(frame, denominator_column, offset_column)
        denominator_zero = validity["denominator_zero"]
        denominator_missing = validity["denominator_missing"]
        rate_not_computable = validity["rate_not_computable"]
        offset_invalid = validity["offset_invalid"]
        grouped = outcome_series.groupby(frame[bucket], dropna=False)
        summary = pd.DataFrame(
            {
                "row_count": grouped.size(),
                "total_count": grouped.sum(min_count=1),
                "missing_row_count": grouped.apply(lambda series: int(series.isna().sum())),
                "nonzero_day_count": outcome_series.fillna(0).gt(0).groupby(frame[bucket], dropna=False).sum(),
                "positive_outcome_invalid_denominator_row_count": (outcome_series.fillna(0).gt(0) & offset_invalid).groupby(frame[bucket], dropna=False).sum(),
                "positive_outcome_missing_denominator_row_count": (outcome_series.fillna(0).gt(0) & denominator_missing).groupby(frame[bucket], dropna=False).sum(),
                "positive_outcome_zero_denominator_row_count": (outcome_series.fillna(0).gt(0) & denominator_zero).groupby(frame[bucket], dropna=False).sum(),
                "rate_not_computable_positive_row_count": (outcome_series.fillna(0).gt(0) & rate_not_computable).groupby(frame[bucket], dropna=False).sum(),
            }
        ).reset_index()
        if denominator_column and denominator_column in frame.columns:
            denominator_group = _numeric_series(frame, str(denominator_column)).groupby(frame[bucket], dropna=False)
            summary["denominator_total"] = denominator_group.sum(min_count=1).values
        else:
            summary["denominator_total"] = None

        for row in summary.to_dict(orient="records"):
            denominator_total = _float_or_none(row["denominator_total"])
            crude_rate = None
            if denominator_total not in {None, 0.0} and unit["rate_multiplier"] is not None and not pd.isna(row["total_count"]):
                crude_rate = float(row["total_count"] / denominator_total * float(unit["rate_multiplier"]))
            bucket_value = row[bucket]
            rows.append(
                {
                    "analysis_unit_id": outcome_column,
                    "linked_outcome_ids": unit["linked_outcome_ids"],
                    "analysis_role": unit["analysis_role"],
                    "outcome_family": unit["outcome_family"],
                    "mart_name": mart_name,
                    "calendar_year": year,
                    "calendar_month": int(bucket_value) if bucket == "calendar_month" and not pd.isna(bucket_value) else None,
                    "season": str(bucket_value) if bucket == "season" and not pd.isna(bucket_value) else None,
                    "denominator_id": unit["denominator_id"],
                    "rate_label": unit["rate_label"],
                    "row_count": int(row["row_count"]),
                    "total_count": _float_or_none(row["total_count"]),
                    "missing_row_count": int(row["missing_row_count"]),
                    "nonzero_day_count": int(row["nonzero_day_count"]),
                    "denominator_total": denominator_total,
                    "crude_rate": crude_rate,
                    "positive_outcome_invalid_denominator_row_count": int(row["positive_outcome_invalid_denominator_row_count"]),
                    "positive_outcome_missing_denominator_row_count": int(row["positive_outcome_missing_denominator_row_count"]),
                    "positive_outcome_zero_denominator_row_count": int(row["positive_outcome_zero_denominator_row_count"]),
                    "rate_not_computable_positive_row_count": int(row["rate_not_computable_positive_row_count"]),
                    "climate_fallback_share": _float_or_none(fallback_group.get(bucket_value)) if fallback_group is not None else None,
                }
            )
    return rows


def _build_sparse_flags(outcome_dossier: pd.DataFrame) -> pd.DataFrame:
    if outcome_dossier.empty:
        return pd.DataFrame(
            columns=[
                "analysis_unit_id",
                "linked_outcome_ids",
                "analysis_role",
                "denominator_id",
                "total_count_all_years",
                "nonzero_day_count_all_years",
                "years_with_any_events",
                "max_municipalities_with_events_in_year",
                "max_event_row_climate_fallback_share",
                "sparse_total_count_flag",
                "sparse_nonzero_day_flag",
                "sparse_year_coverage_flag",
                "sparse_geography_flag",
                "any_sparse_flag",
                "baseline_model_support_status",
                "recommendation",
            ]
        )
    rows: list[dict[str, Any]] = []
    for analysis_unit_id, group in outcome_dossier.groupby("analysis_unit_id", sort=True):
        analysis_role = str(group["analysis_role"].iloc[0])
        required_years = min(
            SUPPORT_THRESHOLDS["min_years_with_events"],
            max(int(group["calendar_year"].nunique()), 1),
        )
        total_count_all_years = float(group["total_count"].fillna(0).sum())
        nonzero_day_count_all_years = int(group["nonzero_day_count"].fillna(0).sum())
        years_with_any_events = int(group["total_count"].fillna(0).gt(0).sum())
        max_municipalities_with_events = int(group["municipalities_with_events"].fillna(0).max())
        max_event_row_fallback_share = _float_or_none(group["event_row_climate_fallback_share"].dropna().max())
        sparse_total_count_flag = total_count_all_years < SUPPORT_THRESHOLDS["min_total_count_all_years"]
        sparse_nonzero_day_flag = nonzero_day_count_all_years < SUPPORT_THRESHOLDS["min_nonzero_day_count_all_years"]
        sparse_year_coverage_flag = years_with_any_events < required_years
        sparse_geography_flag = max_municipalities_with_events < SUPPORT_THRESHOLDS["min_municipalities_with_events_in_year"]
        any_sparse_flag = any(
            [
                sparse_total_count_flag,
                sparse_nonzero_day_flag,
                sparse_year_coverage_flag,
                sparse_geography_flag,
            ]
        )
        if analysis_role == "descriptive_only":
            recommendation = "retain_descriptive_only"
        elif any_sparse_flag and analysis_role == "primary":
            recommendation = "downgrade_before_step2"
        elif any_sparse_flag:
            recommendation = "keep_as_sensitivity_or_defer"
        else:
            recommendation = "proceed_to_step2"
        rows.append(
            {
                "analysis_unit_id": analysis_unit_id,
                "linked_outcome_ids": str(group["linked_outcome_ids"].iloc[0]),
                "analysis_role": analysis_role,
                "denominator_id": str(group["denominator_id"].iloc[0]),
                "total_count_all_years": total_count_all_years,
                "nonzero_day_count_all_years": nonzero_day_count_all_years,
                "years_with_any_events": years_with_any_events,
                "max_municipalities_with_events_in_year": max_municipalities_with_events,
                "max_event_row_climate_fallback_share": max_event_row_fallback_share,
                "sparse_total_count_flag": bool(sparse_total_count_flag),
                "sparse_nonzero_day_flag": bool(sparse_nonzero_day_flag),
                "sparse_year_coverage_flag": bool(sparse_year_coverage_flag),
                "sparse_geography_flag": bool(sparse_geography_flag),
                "any_sparse_flag": bool(any_sparse_flag),
                "baseline_model_support_status": "PASS" if (not any_sparse_flag or analysis_role == "descriptive_only") else "FAIL",
                "recommendation": recommendation,
            }
        )
    return pd.DataFrame(rows).sort_values(["analysis_role", "analysis_unit_id"], ignore_index=True)


def _build_decision_checklist(
    analysis_units: pd.DataFrame,
    sparse_flags: pd.DataFrame,
    monthly_outcome_atlas: pd.DataFrame,
    seasonal_outcome_atlas: pd.DataFrame,
    denominator_audit: pd.DataFrame,
    exposure_annual_summary: pd.DataFrame,
    years: list[int],
    gates: dict[str, bool],
) -> pd.DataFrame:
    primary_units = analysis_units.loc[analysis_units["analysis_role"].astype(str).eq("primary"), "analysis_unit_id"].astype(str)
    promoted_units = analysis_units.loc[analysis_units["promoted_outcome_flag"].eq(1), "analysis_unit_id"].astype(str)
    primary_pass_count = int(
        sparse_flags.loc[
            sparse_flags["analysis_unit_id"].astype(str).isin(primary_units)
            & sparse_flags["baseline_model_support_status"].astype(str).eq("PASS")
        ].shape[0]
    )
    promoted_denominator_fail_count = int(
        denominator_audit.loc[
            denominator_audit["analysis_unit_id"].astype(str).isin(promoted_units)
            & denominator_audit["denominator_consistency_status"].astype(str).eq("FAIL")
        ].shape[0]
    )
    expected_month_rows = len(years) * 12
    expected_season_rows = len(years) * 4
    monthly_complete_count = int(
        monthly_outcome_atlas.loc[monthly_outcome_atlas["analysis_unit_id"].astype(str).isin(promoted_units)]
        .groupby("analysis_unit_id")["calendar_month"]
        .size()
        .eq(expected_month_rows)
        .sum()
    )
    seasonal_complete_count = int(
        seasonal_outcome_atlas.loc[seasonal_outcome_atlas["analysis_unit_id"].astype(str).isin(promoted_units)]
        .groupby("analysis_unit_id")["season"]
        .size()
        .eq(expected_season_rows)
        .sum()
    )
    promoted_count = int(len(promoted_units))
    max_fallback_share = _float_or_none(exposure_annual_summary["fallback_share"].dropna().max())
    rows = [
        {
            "checklist_item": "Primary endpoints have adequate event support for at least a baseline model.",
            "pass_flag": bool(gates["primary_outcomes_have_baseline_support"]),
            "evidence": f"{primary_pass_count}/{int(len(primary_units))} primary analysis units passed the support thresholds.",
        },
        {
            "checklist_item": "Monthly and seasonal profiles have been inspected and documented before outcome promotion.",
            "pass_flag": bool(gates["promoted_outcomes_have_complete_monthly_and_seasonal_profiles"]),
            "evidence": f"{monthly_complete_count}/{promoted_count} promoted units have complete month coverage and {seasonal_complete_count}/{promoted_count} have complete season coverage.",
        },
        {
            "checklist_item": "Denominator arithmetic is internally consistent for every promoted outcome.",
            "pass_flag": bool(gates["promoted_outcomes_have_consistent_denominator_arithmetic"]),
            "evidence": f"{promoted_denominator_fail_count} promoted analysis units showed positive-outcome rows with invalid denominator arithmetic.",
        },
        {
            "checklist_item": "Climate fallback reliance is quantified and acceptable for the intended main analysis.",
            "pass_flag": bool(gates["climate_fallback_quantified_and_acceptable"]),
            "evidence": f"Maximum annual fallback share across marts was {max_fallback_share if max_fallback_share is not None else 'NA'}.",
        },
    ]
    return pd.DataFrame(rows)


def _write_workbook(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in tables.items():
            frame.to_excel(writer, index=False, sheet_name=sheet_name[:31])


def _render_step1_report(
    round_config: dict[str, Any],
    step0_report: dict[str, Any],
    materialization_report: dict[str, Any],
    gates: dict[str, bool],
    summary: dict[str, Any],
    outputs: dict[str, str],
    workspace_report: dict[str, object],
    sparse_flags: pd.DataFrame,
    missing_columns: dict[str, list[str]],
) -> str:
    lines = [
        "# Step 1 Descriptive Readiness Atlas Report",
        "",
        f"- Generated at: `{_now_utc()}`",
        f"- Analysis ID: `{round_config['analysis_id']}`",
        f"- Analysis title: `{round_config['analysis_title']}`",
        f"- Years: `{', '.join(str(year) for year in round_config['years'])}`",
        f"- Upstream Step 0 status: `{step0_report['status']}`",
        f"- Internal mart materialization mode: `{materialization_report['materialization_mode']}`",
        "",
        "## Gates",
        "",
    ]
    for gate_name, passed in gates.items():
        lines.append(f"- {gate_name}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Active analysis units: `{summary['analysis_unit_count']}`",
            f"- Primary analysis units: `{summary['primary_analysis_unit_count']}`",
            f"- Sensitivity analysis units: `{summary['sensitivity_analysis_unit_count']}`",
            f"- Descriptive-only analysis units: `{summary['descriptive_only_analysis_unit_count']}`",
            f"- Sparse or downgraded units: `{summary['sparse_or_downgrade_count']}`",
            f"- Maximum annual climate fallback share: `{summary['max_fallback_share']}`",
        ]
    )
    if missing_columns["outcome_columns"] or missing_columns["denominator_columns"]:
        lines.extend(["", "## Missing Columns", ""])
        for value in missing_columns["outcome_columns"]:
            lines.append(f"- Missing outcome column: `{value}`")
        for value in missing_columns["denominator_columns"]:
            lines.append(f"- Missing denominator column: `{value}`")
    flagged = (
        sparse_flags.loc[sparse_flags["recommendation"].astype(str).ne("proceed_to_step2")]
        if not sparse_flags.empty and "recommendation" in sparse_flags.columns
        else pd.DataFrame()
    )
    if not flagged.empty:
        lines.extend(["", "## Flagged Analysis Units", ""])
        for row in flagged.to_dict(orient="records"):
            lines.append(
                "- "
                f"{row['analysis_unit_id']} ({row['analysis_role']}): "
                f"recommendation=`{row['recommendation']}`, any_sparse_flag=`{row['any_sparse_flag']}`"
            )
    lines.extend(["", "## Outputs", ""])
    for output_name, relative_path in outputs.items():
        lines.append(f"- {output_name}: `{relative_path}`")
    lines.extend(
        [
            "",
            "## Publication Workspace",
            "",
            f"- Workspace root: `{workspace_report['workspace_root']}`",
            f"- Files created: `{len(workspace_report['created_files'])}`",
            f"- Files already present: `{len(workspace_report['existing_files'])}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_step1_descriptive_atlas(
    root: Path,
    config_path: Path | None = None,
    step0_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directories = ensure_step1_directory_structure(root)
    workspace_report = _scaffold_step1_publication_workspace(root)
    round_config = load_round_config(root, config_path=config_path)
    step0_payload = step0_report or build_step0_freeze(root, config_path=config_path)

    output_root = analysis_step1_root(root)
    report_path = output_root / "step1_report.md"
    publication_note_path = incremental_publication_root(root) / "02_step1_descriptive_readiness.md"
    decision_log_path = incremental_publication_root(root) / "03_step1_decision_log.md"

    if step0_payload["status"] != "PASS":
        gates = {"upstream_step0_pass": False}
        outputs = {"step1_report": _relative(root, report_path)}
        report_path.write_text(
            _render_step1_report(
                round_config=round_config,
                step0_report=step0_payload,
                materialization_report={"materialization_mode": "not_started"},
                gates=gates,
                summary={
                    "analysis_unit_count": 0,
                    "primary_analysis_unit_count": 0,
                    "sensitivity_analysis_unit_count": 0,
                    "descriptive_only_analysis_unit_count": 0,
                    "sparse_or_downgrade_count": 0,
                    "max_fallback_share": None,
                },
                outputs=outputs,
                workspace_report=workspace_report,
                sparse_flags=pd.DataFrame(),
                missing_columns={"outcome_columns": [], "denominator_columns": []},
            ),
            encoding="utf-8",
        )
        publication_note_path.write_text(
            (
                "# Step 1 Descriptive Readiness\n\n"
                "- Status: `FAIL`\n"
                f"- Report: `{outputs['step1_report']}`\n"
            ),
            encoding="utf-8",
        )
        decision_log_path.write_text(
            "# Step 1 Decision Log\n\n- Status: `FAIL`\n- Upstream Step 0 did not pass, so no Step 1 decisions were issued.\n",
            encoding="utf-8",
        )
        return {
            "status": "FAIL",
            "step": "step1_descriptive_readiness_atlas",
            "analysis_id": round_config["analysis_id"],
            "years": list(round_config["years"]),
            "directories": directories,
            "upstream_step0_status": step0_payload["status"],
            "gates": gates,
            "outputs": outputs,
            "workspace_report": workspace_report,
        }

    inventory = pd.read_csv(analysis_step0_root(root) / "frozen_outcome_inventory.csv")
    denominator_contract = pd.read_csv(analysis_step0_root(root) / "denominator_contract.csv")
    analysis_units, analysis_unit_contract_consistent = _build_analysis_units(inventory, denominator_contract)
    mart_names = sorted(analysis_units["mart_name"].dropna().astype(str).unique().tolist())
    required_columns_by_mart: dict[str, set[str]] = {}
    promoted_units_for_materialization = analysis_units.loc[analysis_units["promoted_outcome_flag"].eq(1)].copy()
    for mart_name, mart_units in promoted_units_for_materialization.groupby("mart_name", sort=True):
        required_columns_by_mart[str(mart_name)] = {
            *mart_units["analysis_unit_id"].astype(str).tolist(),
            *[
                value
                for value in (_string_or_none(item) for item in mart_units["denominator_column"].tolist())
                if value
            ],
            *[
                value
                for value in (_string_or_none(item) for item in mart_units["offset_column"].tolist())
                if value
            ],
            "municipality_code",
            "calendar_month",
            FALLBACK_COLUMN,
            "denominator_zero_flag",
            "denominator_missing_flag",
            "rate_not_computable_flag",
            "offset_valid_flag",
            *EXPOSURE_COLUMNS,
        }
    materialization_report = _ensure_internal_marts(
        root,
        mart_names,
        list(round_config["years"]),
        required_columns_by_mart,
    )

    annual_exposure_rows: list[dict[str, Any]] = []
    monthly_exposure_rows: list[dict[str, Any]] = []
    seasonal_exposure_rows: list[dict[str, Any]] = []
    outcome_year_rows: list[dict[str, Any]] = []
    denominator_audit_rows: list[dict[str, Any]] = []
    monthly_outcome_rows: list[dict[str, Any]] = []
    seasonal_outcome_rows: list[dict[str, Any]] = []
    monthly_denominator_rows: list[dict[str, Any]] = []
    seasonal_denominator_rows: list[dict[str, Any]] = []
    missing_outcome_columns: list[str] = []
    missing_denominator_columns: list[str] = []

    for mart_name in mart_names:
        mart_units = analysis_units.loc[analysis_units["mart_name"].astype(str).eq(mart_name)].copy()
        denominator_columns = [
            str(value)
            for value in mart_units["denominator_column"].dropna().astype(str).tolist()
            if value and value != "None"
        ]
        offset_columns = [
            str(value)
            for value in mart_units["offset_column"].dropna().astype(str).tolist()
            if value and value != "None"
        ]
        selected_columns = sorted(
            {
                "municipality_code",
                "calendar_month",
                FALLBACK_COLUMN,
                "denominator_zero_flag",
                "denominator_missing_flag",
                "rate_not_computable_flag",
                "offset_valid_flag",
                *EXPOSURE_COLUMNS,
                *mart_units["analysis_unit_id"].astype(str).tolist(),
                *denominator_columns,
                *offset_columns,
            }
        )
        annual_lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
        for year in round_config["years"]:
            frame = _load_mart_partition(root, mart_name, int(year), selected_columns)
            annual_exposures = _build_exposure_annual_rows(frame, mart_name, int(year))
            annual_exposure_rows.extend(annual_exposures)
            for row in annual_exposures:
                annual_lookup[(mart_name, int(year), str(row["exposure_variable"]))] = row
            monthly_exposure_rows.extend(_build_exposure_time_rows(frame, mart_name, int(year), "calendar_month"))
            seasonal_exposure_rows.extend(_build_exposure_time_rows(frame, mart_name, int(year), "season"))
            month_denominator_rows = _build_denominator_time_rows(frame, mart_name, int(year), "calendar_month", mart_units)
            season_denominator_rows = _build_denominator_time_rows(frame, mart_name, int(year), "season", mart_units)
            monthly_denominator_rows.extend(month_denominator_rows)
            seasonal_denominator_rows.extend(season_denominator_rows)
            year_rows, year_denominator_rows, missing_outcomes, missing_denominators = _build_outcome_year_rows(
                frame,
                mart_name,
                int(year),
                mart_units,
                annual_lookup,
            )
            outcome_year_rows.extend(year_rows)
            denominator_audit_rows.extend(year_denominator_rows)
            missing_outcome_columns.extend(missing_outcomes)
            missing_denominator_columns.extend(missing_denominators)
            monthly_outcome_rows.extend(_build_outcome_time_rows(frame, mart_name, int(year), mart_units, "calendar_month"))
            seasonal_outcome_rows.extend(_build_outcome_time_rows(frame, mart_name, int(year), mart_units, "season"))

    analysis_units_path = output_root / "analysis_unit_roster.csv"
    outcome_dossier_path = output_root / "outcome_descriptive_dossier.csv"
    monthly_outcome_path = output_root / "monthly_outcome_atlas.csv"
    seasonal_outcome_path = output_root / "seasonal_outcome_atlas.csv"
    exposure_annual_path = output_root / "exposure_annual_summary.csv"
    monthly_exposure_path = output_root / "monthly_exposure_atlas.csv"
    seasonal_exposure_path = output_root / "seasonal_exposure_atlas.csv"
    denominator_audit_path = output_root / "denominator_integrity_audit.csv"
    monthly_denominator_path = output_root / "monthly_denominator_atlas.csv"
    seasonal_denominator_path = output_root / "seasonal_denominator_atlas.csv"
    sparse_flags_path = output_root / "sparse_outcome_flags.csv"
    checklist_path = output_root / "step1_decision_checklist.csv"
    workbook_path = output_root / "step1_descriptive_summary_tables.xlsx"

    exposure_annual_summary = pd.DataFrame(annual_exposure_rows).sort_values(
        ["mart_name", "calendar_year", "exposure_variable"], ignore_index=True
    )
    monthly_exposure_atlas = pd.DataFrame(monthly_exposure_rows).sort_values(
        ["mart_name", "calendar_year", "calendar_month", "exposure_variable"], ignore_index=True
    )
    seasonal_exposure_atlas = pd.DataFrame(seasonal_exposure_rows)
    if not seasonal_exposure_atlas.empty:
        seasonal_exposure_atlas["season_order"] = seasonal_exposure_atlas["season"].map(SEASON_ORDER)
        seasonal_exposure_atlas = seasonal_exposure_atlas.sort_values(
            ["mart_name", "calendar_year", "season_order", "exposure_variable"], ignore_index=True
        ).drop(columns=["season_order"])

    outcome_dossier = pd.DataFrame(outcome_year_rows).sort_values(
        ["analysis_role", "analysis_unit_id", "calendar_year"], ignore_index=True
    )
    monthly_outcome_atlas = pd.DataFrame(monthly_outcome_rows).sort_values(
        ["analysis_role", "analysis_unit_id", "calendar_year", "calendar_month"], ignore_index=True
    )
    seasonal_outcome_atlas = pd.DataFrame(seasonal_outcome_rows)
    if not seasonal_outcome_atlas.empty:
        seasonal_outcome_atlas["season_order"] = seasonal_outcome_atlas["season"].map(SEASON_ORDER)
        seasonal_outcome_atlas = seasonal_outcome_atlas.sort_values(
            ["analysis_role", "analysis_unit_id", "calendar_year", "season_order"], ignore_index=True
        ).drop(columns=["season_order"])

    denominator_audit = pd.DataFrame(denominator_audit_rows).sort_values(
        ["analysis_role", "analysis_unit_id", "calendar_year"], ignore_index=True
    )
    monthly_denominator_atlas = pd.DataFrame(monthly_denominator_rows).sort_values(
        ["mart_name", "denominator_id", "calendar_year", "calendar_month"], ignore_index=True
    )
    seasonal_denominator_atlas = pd.DataFrame(seasonal_denominator_rows)
    if not seasonal_denominator_atlas.empty:
        seasonal_denominator_atlas["season_order"] = seasonal_denominator_atlas["season"].map(SEASON_ORDER)
        seasonal_denominator_atlas = seasonal_denominator_atlas.sort_values(
            ["mart_name", "denominator_id", "calendar_year", "season_order"], ignore_index=True
        ).drop(columns=["season_order"])

    sparse_flags = _build_sparse_flags(outcome_dossier)
    promoted_units = analysis_units.loc[analysis_units["promoted_outcome_flag"].eq(1), "analysis_unit_id"].astype(str)
    primary_units = analysis_units.loc[analysis_units["primary_outcome_flag"].eq(1), "analysis_unit_id"].astype(str)
    promoted_monthly_counts = (
        monthly_outcome_atlas.loc[monthly_outcome_atlas["analysis_unit_id"].astype(str).isin(promoted_units)]
        .groupby("analysis_unit_id")["calendar_month"]
        .size()
        if not monthly_outcome_atlas.empty
        else pd.Series(dtype="int64")
    )
    promoted_seasonal_counts = (
        seasonal_outcome_atlas.loc[seasonal_outcome_atlas["analysis_unit_id"].astype(str).isin(promoted_units)]
        .groupby("analysis_unit_id")["season"]
        .size()
        if not seasonal_outcome_atlas.empty
        else pd.Series(dtype="int64")
    )
    expected_monthly_rows = len(round_config["years"]) * 12
    expected_seasonal_rows = len(round_config["years"]) * 4
    promoted_outcome_column_names = set(promoted_units.tolist())
    promoted_denominator_column_names = {
        value
        for value in (_string_or_none(value) for value in analysis_units.loc[analysis_units["promoted_outcome_flag"].eq(1), "denominator_column"])
        if value
    }
    missing_promoted_outcome_columns = sorted(
        {
            item
            for item in missing_outcome_columns
            if item.split(":", 1)[1] in promoted_outcome_column_names
        }
    )
    missing_promoted_denominator_columns = sorted(
        {
            item
            for item in missing_denominator_columns
            if item.split(":", 1)[1] in promoted_denominator_column_names
        }
    )
    denominator_fail_count = int(
        denominator_audit.loc[
            denominator_audit["analysis_unit_id"].astype(str).isin(promoted_units)
            & denominator_audit["denominator_consistency_status"].astype(str).eq("FAIL")
        ].shape[0]
    )
    max_fallback_share = _float_or_none(exposure_annual_summary["fallback_share"].dropna().max())
    gates = {
        "upstream_step0_pass": True,
        "analysis_unit_contract_consistent": analysis_unit_contract_consistent,
        "required_outcome_columns_available": len(missing_promoted_outcome_columns) == 0,
        "required_denominator_columns_available": len(missing_promoted_denominator_columns) == 0,
        "primary_outcomes_have_baseline_support": bool(
            sparse_flags.loc[
                sparse_flags["analysis_unit_id"].astype(str).isin(primary_units)
                & sparse_flags["baseline_model_support_status"].astype(str).eq("FAIL")
            ].empty
        ),
        "promoted_outcomes_have_complete_monthly_and_seasonal_profiles": bool(
            promoted_monthly_counts.eq(expected_monthly_rows).all()
            and promoted_seasonal_counts.eq(expected_seasonal_rows).all()
            and len(promoted_monthly_counts) == len(promoted_units)
            and len(promoted_seasonal_counts) == len(promoted_units)
        ),
        "promoted_outcomes_have_consistent_denominator_arithmetic": denominator_fail_count == 0,
        "climate_fallback_quantified_and_acceptable": max_fallback_share is not None and max_fallback_share <= FALLBACK_SHARE_THRESHOLD,
    }
    status = "PASS" if all(gates.values()) else "FAIL"

    decision_checklist = _build_decision_checklist(
        analysis_units=analysis_units,
        sparse_flags=sparse_flags,
        monthly_outcome_atlas=monthly_outcome_atlas,
        seasonal_outcome_atlas=seasonal_outcome_atlas,
        denominator_audit=denominator_audit,
        exposure_annual_summary=exposure_annual_summary,
        years=list(round_config["years"]),
        gates=gates,
    )

    output_tables = {
        "analysis_units": analysis_units,
        "outcome_dossier": outcome_dossier,
        "monthly_outcomes": monthly_outcome_atlas,
        "seasonal_outcomes": seasonal_outcome_atlas,
        "exposure_annual": exposure_annual_summary,
        "exposure_monthly": monthly_exposure_atlas,
        "exposure_seasonal": seasonal_exposure_atlas,
        "denominator_audit": denominator_audit,
        "denominator_monthly": monthly_denominator_atlas,
        "denominator_seasonal": seasonal_denominator_atlas,
        "sparse_flags": sparse_flags,
        "decision_checklist": decision_checklist,
    }

    analysis_units.to_csv(analysis_units_path, index=False)
    outcome_dossier.to_csv(outcome_dossier_path, index=False)
    monthly_outcome_atlas.to_csv(monthly_outcome_path, index=False)
    seasonal_outcome_atlas.to_csv(seasonal_outcome_path, index=False)
    exposure_annual_summary.to_csv(exposure_annual_path, index=False)
    monthly_exposure_atlas.to_csv(monthly_exposure_path, index=False)
    seasonal_exposure_atlas.to_csv(seasonal_exposure_path, index=False)
    denominator_audit.to_csv(denominator_audit_path, index=False)
    monthly_denominator_atlas.to_csv(monthly_denominator_path, index=False)
    seasonal_denominator_atlas.to_csv(seasonal_denominator_path, index=False)
    sparse_flags.to_csv(sparse_flags_path, index=False)
    decision_checklist.to_csv(checklist_path, index=False)
    _write_workbook(workbook_path, output_tables)

    outputs = {
        "analysis_unit_roster": _relative(root, analysis_units_path),
        "outcome_descriptive_dossier": _relative(root, outcome_dossier_path),
        "monthly_outcome_atlas": _relative(root, monthly_outcome_path),
        "seasonal_outcome_atlas": _relative(root, seasonal_outcome_path),
        "exposure_annual_summary": _relative(root, exposure_annual_path),
        "monthly_exposure_atlas": _relative(root, monthly_exposure_path),
        "seasonal_exposure_atlas": _relative(root, seasonal_exposure_path),
        "denominator_integrity_audit": _relative(root, denominator_audit_path),
        "monthly_denominator_atlas": _relative(root, monthly_denominator_path),
        "seasonal_denominator_atlas": _relative(root, seasonal_denominator_path),
        "sparse_outcome_flags": _relative(root, sparse_flags_path),
        "decision_checklist": _relative(root, checklist_path),
        "summary_workbook": _relative(root, workbook_path),
        "step1_report": _relative(root, report_path),
    }
    summary = {
        "analysis_unit_count": int(len(analysis_units)),
        "primary_analysis_unit_count": int(analysis_units["analysis_role"].astype(str).eq("primary").sum()),
        "sensitivity_analysis_unit_count": int(analysis_units["analysis_role"].astype(str).eq("sensitivity").sum()),
        "descriptive_only_analysis_unit_count": int(analysis_units["analysis_role"].astype(str).eq("descriptive_only").sum()),
        "sparse_or_downgrade_count": int(
            sparse_flags["recommendation"].astype(str).isin({"downgrade_before_step2", "keep_as_sensitivity_or_defer"}).sum()
        ),
        "max_fallback_share": max_fallback_share,
    }

    report_contents = _render_step1_report(
        round_config=round_config,
        step0_report=step0_payload,
        materialization_report=materialization_report,
        gates=gates,
        summary=summary,
        outputs=outputs,
        workspace_report=workspace_report,
        sparse_flags=sparse_flags,
        missing_columns={
            "outcome_columns": sorted(set(missing_outcome_columns)),
            "denominator_columns": sorted(set(missing_denominator_columns)),
        },
    )
    report_path.write_text(report_contents, encoding="utf-8")
    publication_note_path.write_text(
        (
            "# Step 1 Descriptive Readiness\n\n"
            f"- Status: `{status}`\n"
            f"- Primary support gate: `{'PASS' if gates['primary_outcomes_have_baseline_support'] else 'FAIL'}`\n"
            f"- Denominator arithmetic gate: `{'PASS' if gates['promoted_outcomes_have_consistent_denominator_arithmetic'] else 'FAIL'}`\n"
            f"- Climate fallback gate: `{'PASS' if gates['climate_fallback_quantified_and_acceptable'] else 'FAIL'}`\n"
            f"- Report: `{outputs['step1_report']}`\n"
            f"- Workbook: `{outputs['summary_workbook']}`\n"
        ),
        encoding="utf-8",
    )
    decision_log_path.write_text(
        (
            "# Step 1 Decision Log\n\n"
            f"- Status: `{status}`\n"
            f"- Sparse or downgraded units: `{summary['sparse_or_downgrade_count']}`\n"
            f"- Primary support gate: `{'PASS' if gates['primary_outcomes_have_baseline_support'] else 'FAIL'}`\n"
            f"- Sparse outcome flags: `{outputs['sparse_outcome_flags']}`\n"
            f"- Decision checklist: `{outputs['decision_checklist']}`\n"
        ),
        encoding="utf-8",
    )

    payload = {
        "status": status,
        "step": "step1_descriptive_readiness_atlas",
        "analysis_id": round_config["analysis_id"],
        "years": list(round_config["years"]),
        "directories": directories,
        "upstream_step0_status": step0_payload["status"],
        "materialization": {key: value for key, value in materialization_report.items() if key != "internal_materialization_report"},
        "gates": gates,
        "summary": summary,
        "outputs": outputs,
        "workspace_report": workspace_report,
    }
    return payload
