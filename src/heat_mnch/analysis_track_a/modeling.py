from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from .config import load_track_a_config
from .registry import track_a_root
from .screening_aggregations import (
    PROBABILITY_FLOOR,
    aggregate_binary_outcomes,
    aggregate_quintile_outcomes,
    apply_probability_floor,
    grouped_wide_to_long,
    pool_binary_rate_ratios,
    pool_quintile_trends,
)
from .screening_cache import SCREENING_STAGE_VERSION, ScreeningCacheManager, screening_scope_key


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


BASE_EXPOSURES: dict[str, dict[str, str]] = {
    "maximum_daily_temperature_c": {
        "label": "Maximum daily temperature",
        "family_id": "family_1_tmax_primary",
        "family_name": "Maximum daily temperature",
    },
    "mean_daily_relative_humidity_pct": {
        "label": "Mean daily relative humidity",
        "family_id": "family_2_tmax_plus_humidity",
        "family_name": "Maximum daily temperature with relative humidity",
    },
    "heatwave_duration_days_p95_threshold": {
        "label": "Heatwave duration above the p95 threshold",
        "family_id": "family_3_heatwave_duration",
        "family_name": "Heatwave duration sensitivity",
    },
    "approximate_heat_index_c": {
        "label": "Approximate heat index",
        "family_id": "family_4_heat_index_sensitivity",
        "family_name": "Approximate heat index sensitivity",
    },
}

ACUTE_OUTCOMES: list[dict[str, str]] = [
    {
        "outcome_variable": "neonatal_death_count",
        "outcome_label": "Neonatal death count",
        "denominator_column": "annual_live_birth_count",
        "outcome_family": "acute_daily",
    },
    {
        "outcome_variable": "infant_death_count",
        "outcome_label": "Infant death count",
        "denominator_column": "annual_live_birth_count",
        "outcome_family": "acute_daily",
    },
    {
        "outcome_variable": "under_five_death_count",
        "outcome_label": "Under-five death count",
        "denominator_column": "under_five_population_count",
        "outcome_family": "acute_daily",
    },
    {
        "outcome_variable": "acute_care_proxy_count",
        "outcome_label": "Acute-care proxy count",
        "denominator_column": "under_five_population_count",
        "outcome_family": "panel_proxy",
    },
]

PERINATAL_OUTCOMES: list[dict[str, str]] = [
    {
        "outcome_variable": "preterm_birth_count",
        "outcome_label": "Preterm birth count",
        "denominator_column": "annual_live_birth_count",
        "outcome_family": "perinatal_delivery_window_aggregate",
    },
    {
        "outcome_variable": "very_preterm_birth_count",
        "outcome_label": "Very preterm birth count",
        "denominator_column": "annual_live_birth_count",
        "outcome_family": "perinatal_delivery_window_aggregate",
    },
    {
        "outcome_variable": "gestational_days_lost_below_37_weeks",
        "outcome_label": "Gestational days lost below 37 weeks",
        "denominator_column": "daily_live_birth_count",
        "outcome_family": "perinatal_delivery_window_aggregate",
    },
    {
        "outcome_variable": "stillbirth_primary_gte28_count",
        "outcome_label": "Primary stillbirth count at 28 weeks or more",
        "denominator_column": "annual_total_birth_count_primary_definition",
        "outcome_family": "perinatal_delivery_window_aggregate",
    },
    {
        "outcome_variable": "stillbirth_sensitivity_gte22_count",
        "outcome_label": "Sensitivity stillbirth count at 22 weeks or more",
        "denominator_column": "annual_total_birth_count_sensitivity_gte22_definition",
        "outcome_family": "perinatal_delivery_window_aggregate",
    },
]

MATERNAL_OUTCOMES: list[dict[str, str]] = [
    {
        "outcome_variable": "obstetric_admission_count",
        "outcome_label": "Obstetric admission count",
        "denominator_column": "women_age_15_49_population_count",
        "outcome_family": "maternal_operational",
    },
    {
        "outcome_variable": "hypertensive_disorder_of_pregnancy_count",
        "outcome_label": "Hypertensive disorder of pregnancy count",
        "denominator_column": "women_age_15_49_population_count",
        "outcome_family": "maternal_operational",
    },
    {
        "outcome_variable": "severe_hypertensive_disorder_of_pregnancy_count",
        "outcome_label": "Severe hypertensive disorder of pregnancy count",
        "denominator_column": "women_age_15_49_population_count",
        "outcome_family": "maternal_operational",
    },
    {
        "outcome_variable": "preterm_labor_admission_count",
        "outcome_label": "Preterm labor admission count",
        "denominator_column": "women_age_15_49_population_count",
        "outcome_family": "maternal_operational",
    },
    {
        "outcome_variable": "gestational_diabetes_admission_count",
        "outcome_label": "Gestational diabetes admission count",
        "denominator_column": "women_age_15_49_population_count",
        "outcome_family": "maternal_operational",
    },
]

WINDOW_SUFFIX_LABELS = {
    "lag0": "lag 0",
    "lag0_1_mean": "lag 0 to 1 mean",
    "lag0_3_mean": "lag 0 to 3 mean",
    "lag0_7_mean": "lag 0 to 7 mean",
    "days_0_6_before_event_mean": "days 0 to 6 before event mean",
    "days_0_13_before_event_mean": "days 0 to 13 before event mean",
    "days_0_27_before_event_mean": "days 0 to 27 before event mean",
}

ACUTE_TEMPERATURE_LAG_SUFFIXES = ["lag0", "lag0_1_mean", "lag0_3_mean", "lag0_7_mean"]
DELIVERY_WINDOW_SUFFIXES = [
    "days_0_6_before_event_mean",
    "days_0_13_before_event_mean",
    "days_0_27_before_event_mean",
]
ACUTE_CORRELATION_EXPOSURES = [
    "maximum_daily_temperature_c",
    "mean_daily_relative_humidity_pct",
    "heatwave_duration_days_p95_threshold",
    "approximate_heat_index_c",
]

ACUTE_REQUIRED_COLUMNS = [
    "municipality_code",
    "observation_date",
    "calendar_year",
    "denominator_zero_flag",
    "denominator_missing_flag",
    "rate_not_computable_flag",
    "offset_valid_flag",
    "quality_control_status",
    "annual_live_birth_count",
    "under_five_population_count",
    "log_annual_live_birth_offset",
    "log_under_five_population_offset",
    "maximum_daily_temperature_c",
    "mean_daily_relative_humidity_pct",
    "heatwave_duration_days_p95_threshold",
    "approximate_heat_index_c",
    "maximum_daily_temperature_c_lag0",
    "maximum_daily_temperature_c_lag0_1_mean",
    "maximum_daily_temperature_c_lag0_3_mean",
    "maximum_daily_temperature_c_lag0_7_mean",
    "heatwave_duration_days_p95_threshold_lag0",
    "neonatal_death_count",
    "infant_death_count",
    "under_five_death_count",
    "acute_care_proxy_count",
]

PERINATAL_REQUIRED_COLUMNS = [
    "municipality_code",
    "observation_date",
    "calendar_year",
    "denominator_zero_flag",
    "denominator_missing_flag",
    "rate_not_computable_flag",
    "offset_valid_flag",
    "quality_control_status",
    "annual_live_birth_count",
    "annual_total_birth_count_primary_definition",
    "annual_total_birth_count_sensitivity_gte22_definition",
    "daily_live_birth_count",
    "total_birth_count_primary_definition",
    "total_birth_count_sensitivity_gte22_definition",
    "log_annual_live_birth_offset",
    "log_annual_total_birth_count_primary_offset",
    "log_annual_total_birth_count_sensitivity_gte22_offset",
    "log_total_birth_count_primary_offset",
    "log_total_birth_count_sensitivity_gte22_offset",
    "preterm_birth_count",
    "very_preterm_birth_count",
    "gestational_days_lost_below_37_weeks",
    "stillbirth_primary_gte28_count",
    "stillbirth_sensitivity_gte22_count",
]

DENOMINATOR_OFFSET_COLUMN_LOOKUP = {
    "annual_live_birth_count": "log_annual_live_birth_offset",
    "under_five_population_count": "log_under_five_population_offset",
    "total_birth_count_primary_definition": "log_total_birth_count_primary_offset",
    "total_birth_count_sensitivity_gte22_definition": "log_total_birth_count_sensitivity_gte22_offset",
    "annual_total_birth_count_primary_definition": "log_annual_total_birth_count_primary_offset",
    "annual_total_birth_count_sensitivity_gte22_definition": "log_annual_total_birth_count_sensitivity_gte22_offset",
    "women_age_15_49_population_count": "log_women_age_15_49_population_offset",
    "women_age_15_45_sensitivity_population_count": "log_women_age_15_45_sensitivity_population_offset",
}

MATERNAL_REQUIRED_COLUMNS = [
    "municipality_code",
    "observation_date",
    "calendar_year",
    "denominator_zero_flag",
    "denominator_missing_flag",
    "rate_not_computable_flag",
    "offset_valid_flag",
    "quality_control_status",
    "women_age_15_49_population_count",
    "log_women_age_15_49_population_offset",
    "obstetric_admission_count",
    "hypertensive_disorder_of_pregnancy_count",
    "severe_hypertensive_disorder_of_pregnancy_count",
    "preterm_labor_admission_count",
    "gestational_diabetes_admission_count",
]

ACUTE_BINARY_COMBINATION_COLUMNS = [
    "screening_domain",
    "exposure_variable",
    "exposure_family_id",
    "exposure_family_name",
    "lag_or_window",
    "denominator_column",
    "outcome_variable",
]
ACUTE_BINARY_STRATUM_COLUMNS = ["municipality_code", "calendar_year"]
WINDOW_COMBINATION_COLUMNS = [
    "mart_name",
    "screening_domain",
    "exposure_variable",
    "exposure_family_id",
    "exposure_family_name",
    "lag_or_window",
    "denominator_column",
    "outcome_variable",
]
WINDOW_STRATUM_COLUMNS = ["municipality_code", "calendar_year"]
ACUTE_BINARY_GROUPED_COLUMNS = (
    ACUTE_BINARY_COMBINATION_COLUMNS
    + ACUTE_BINARY_STRATUM_COLUMNS
    + ["screen_group", "outcome_sum", "denominator_sum", "row_count"]
)
WINDOW_GROUPED_COLUMNS = [
    "mart_name",
    "screening_domain_prefix",
    "exposure_variable",
    "exposure_family_id",
    "exposure_family_name",
    "lag_or_window",
    "denominator_column",
    "outcome_variable",
    "municipality_code",
    "calendar_year",
    "exposure_quintile",
    "outcome_sum",
    "denominator_sum",
    "row_count",
]
ACUTE_WEEKLY_COLUMNS = [
    "municipality_code",
    "calendar_year",
    "epidemiologic_week",
    *ACUTE_CORRELATION_EXPOSURES,
    *(outcome["outcome_variable"] for outcome in ACUTE_OUTCOMES),
]


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _mart_partition_path(root: Path, mart_name: str, year: int) -> Path:
    return track_a_root(root) / "analysis_marts" / mart_name / f"YEAR={year}" / f"{mart_name}_{year}.parquet"


def _screening_root(root: Path) -> Path:
    return track_a_root(root) / "screening"


def _load_partition(root: Path, mart_name: str, year: int, columns: list[str]) -> pd.DataFrame:
    path = _mart_partition_path(root, mart_name, year)
    if not path.exists():
        raise FileNotFoundError(f"Stage 4 requires the Stage 2 mart partition at {path}. Run track-a-build-marts first.")
    return pd.read_parquet(path, columns=columns)


def _load_readiness(root: Path) -> pd.DataFrame:
    path = track_a_root(root) / "freeze" / "analysis_ready_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stage 4 requires the Stage 1 readiness matrix at {path}. Run track-a-freeze first.")
    readiness = pd.read_csv(path)
    if "public_concept_name" not in readiness.columns:
        readiness["public_concept_name"] = readiness["academic_variable_name"]
    if "public_concept_label" not in readiness.columns:
        readiness["public_concept_label"] = readiness["academic_label"]
    if "headline_eligible_flag" not in readiness.columns:
        readiness["headline_eligible_flag"] = readiness["stage1_classification"].eq("benchmark_ready").map(
            {True: "yes", False: "no"}
        )
    if "benchmark_model_eligible_flag" not in readiness.columns:
        readiness["benchmark_model_eligible_flag"] = readiness["stage1_classification"].eq("benchmark_ready").map(
            {True: "yes", False: "no"}
        )
    if "duplicate_prompt_concept_flag" not in readiness.columns:
        readiness["duplicate_prompt_concept_flag"] = "no"
    return readiness


def _readiness_lookup(readiness: pd.DataFrame) -> dict[str, dict[str, str]]:
    frame = readiness.sort_values(
        ["academic_variable_name", "stage1_classification", "prompt_label"]
    ).drop_duplicates("academic_variable_name")
    return {
        str(row["academic_variable_name"]): {
            "public_concept_name": str(row["public_concept_name"]),
            "public_concept_label": str(row["public_concept_label"]),
            "stage1_classification": str(row["stage1_classification"]),
            "headline_eligible_flag": str(row["headline_eligible_flag"]),
            "benchmark_model_eligible_flag": str(row["benchmark_model_eligible_flag"]),
            "duplicate_prompt_concept_flag": str(row["duplicate_prompt_concept_flag"]),
        }
        for row in frame.to_dict(orient="records")
    }


def _normal_two_sided_p(z_value: float) -> float:
    if not math.isfinite(z_value):
        return float("nan")
    return float(math.erfc(abs(z_value) / math.sqrt(2.0)))


def _filter_valid_rows(frame: pd.DataFrame, denominator_column: str) -> pd.DataFrame:
    offset_column = DENOMINATOR_OFFSET_COLUMN_LOOKUP.get(denominator_column)
    denominator_valid = frame[denominator_column].fillna(0).gt(0)
    if offset_column and offset_column in frame.columns:
        denominator_valid = denominator_valid & pd.to_numeric(frame[offset_column], errors="coerce").notna()
    else:
        denominator_valid = (
            denominator_valid
            & frame["offset_valid_flag"].eq(1)
            & frame["denominator_zero_flag"].eq(0)
            & frame["denominator_missing_flag"].eq(0)
            & frame["rate_not_computable_flag"].eq(0)
        )
    valid = frame.loc[
        frame["quality_control_status"].astype(str).eq("PASS")
        & denominator_valid
    ].copy()
    return valid


def _filter_base_screening_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        frame["quality_control_status"].astype(str).eq("PASS")
        & frame["offset_valid_flag"].eq(1)
        & frame["denominator_zero_flag"].eq(0)
        & frame["denominator_missing_flag"].eq(0)
        & frame["rate_not_computable_flag"].eq(0)
    ].copy()


def _binary_stratum_summary(
    frame: pd.DataFrame,
    outcome_column: str,
    denominator_column: str,
    group_column: str,
) -> pd.DataFrame:
    subset = frame.loc[frame[group_column].isin(["exposed", "reference"])].copy()
    if subset.empty:
        return pd.DataFrame()
    grouped = (
        subset.groupby(["municipality_code", "calendar_year", group_column], as_index=False)
        .agg(
            outcome_sum=(outcome_column, "sum"),
            denominator_sum=(denominator_column, "sum"),
            row_count=(group_column, "size"),
        )
    )
    pivot = grouped.pivot_table(
        index=["municipality_code", "calendar_year"],
        columns=group_column,
        values=["outcome_sum", "denominator_sum", "row_count"],
        aggfunc="first",
        fill_value=0,
    )
    pivot.columns = [f"{metric}_{group}" for metric, group in pivot.columns.to_flat_index()]
    summary = pivot.reset_index()
    required_columns = [
        "outcome_sum_exposed",
        "outcome_sum_reference",
        "denominator_sum_exposed",
        "denominator_sum_reference",
        "row_count_exposed",
        "row_count_reference",
    ]
    for column in required_columns:
        if column not in summary.columns:
            summary[column] = 0.0
    return summary


def _binary_stratum_summary_multi(
    frame: pd.DataFrame,
    outcome_columns: list[str],
    denominator_column: str,
    group_column: str,
) -> pd.DataFrame:
    subset = frame.loc[frame[group_column].isin(["exposed", "reference"])].copy()
    if subset.empty:
        return pd.DataFrame()
    aggregations: dict[str, tuple[str, str]] = {
        "denominator_sum": (denominator_column, "sum"),
        "row_count": (group_column, "size"),
    }
    for outcome_column in outcome_columns:
        aggregations[f"{outcome_column}_sum"] = (outcome_column, "sum")
    return subset.groupby(["municipality_code", "calendar_year", group_column], as_index=False).agg(**aggregations)


def _binary_outcome_summary_from_grouped(grouped: pd.DataFrame, outcome_variable: str) -> pd.DataFrame:
    if grouped.empty:
        return pd.DataFrame()
    subset = grouped.loc[
        :,
        [
            "municipality_code",
            "calendar_year",
            "screen_group",
            f"{outcome_variable}_sum",
            "denominator_sum",
            "row_count",
        ],
    ].rename(columns={f"{outcome_variable}_sum": "outcome_sum"})
    pivot = subset.pivot_table(
        index=["municipality_code", "calendar_year"],
        columns="screen_group",
        values=["outcome_sum", "denominator_sum", "row_count"],
        aggfunc="first",
        fill_value=0,
    )
    pivot.columns = [f"{metric}_{group}" for metric, group in pivot.columns.to_flat_index()]
    summary = pivot.reset_index()
    for column in [
        "outcome_sum_exposed",
        "outcome_sum_reference",
        "denominator_sum_exposed",
        "denominator_sum_reference",
        "row_count_exposed",
        "row_count_reference",
    ]:
        if column not in summary.columns:
            summary[column] = 0.0
    return summary


def _pool_log_rate_ratio(summary: pd.DataFrame) -> dict[str, float | int]:
    if summary.empty:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "valid_strata_count": 0,
            "exposed_row_count": 0,
            "reference_row_count": 0,
            "exposed_outcome_sum": 0.0,
            "reference_outcome_sum": 0.0,
            "exposed_denominator_sum": 0.0,
            "reference_denominator_sum": 0.0,
        }
    valid = summary.loc[
        summary["row_count_exposed"].gt(0)
        & summary["row_count_reference"].gt(0)
        & summary["denominator_sum_exposed"].gt(0)
        & summary["denominator_sum_reference"].gt(0)
    ].copy()
    if valid.empty:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "valid_strata_count": 0,
            "exposed_row_count": int(summary["row_count_exposed"].sum()),
            "reference_row_count": int(summary["row_count_reference"].sum()),
            "exposed_outcome_sum": float(summary["outcome_sum_exposed"].sum()),
            "reference_outcome_sum": float(summary["outcome_sum_reference"].sum()),
            "exposed_denominator_sum": float(summary["denominator_sum_exposed"].sum()),
            "reference_denominator_sum": float(summary["denominator_sum_reference"].sum()),
        }

    exposed_cases = valid["outcome_sum_exposed"].astype(float) + 0.5
    reference_cases = valid["outcome_sum_reference"].astype(float) + 0.5
    exposed_denominator = valid["denominator_sum_exposed"].astype(float)
    reference_denominator = valid["denominator_sum_reference"].astype(float)

    log_effect = np.log(exposed_cases / exposed_denominator) - np.log(reference_cases / reference_denominator)
    variance = (1.0 / exposed_cases) + (1.0 / reference_cases)
    weights = 1.0 / variance
    pooled_log_effect = float(np.average(log_effect, weights=weights))
    pooled_se = float(np.sqrt(1.0 / weights.sum()))
    z_value = pooled_log_effect / pooled_se if pooled_se > 0 else float("nan")
    lower_ci = pooled_log_effect - 1.96 * pooled_se
    upper_ci = pooled_log_effect + 1.96 * pooled_se

    return {
        "estimate": float(math.exp(pooled_log_effect)),
        "lower_ci": float(math.exp(lower_ci)),
        "upper_ci": float(math.exp(upper_ci)),
        "p_value": _normal_two_sided_p(z_value),
        "valid_strata_count": int(valid.shape[0]),
        "exposed_row_count": int(valid["row_count_exposed"].sum()),
        "reference_row_count": int(valid["row_count_reference"].sum()),
        "exposed_outcome_sum": float(valid["outcome_sum_exposed"].sum()),
        "reference_outcome_sum": float(valid["outcome_sum_reference"].sum()),
        "exposed_denominator_sum": float(valid["denominator_sum_exposed"].sum()),
        "reference_denominator_sum": float(valid["denominator_sum_reference"].sum()),
    }


def _assign_quintiles(series: pd.Series) -> pd.Series:
    values = series.astype(float).to_numpy(copy=False)
    valid_mask = np.isfinite(values)
    quintiles = np.full(values.shape, np.nan, dtype=float)
    if valid_mask.sum() < 5:
        return pd.Series(quintiles, index=series.index, dtype="Float64")
    thresholds = np.quantile(values[valid_mask], [0.2, 0.4, 0.6, 0.8])
    quintiles[valid_mask] = np.searchsorted(thresholds, values[valid_mask], side="right") + 1
    return pd.Series(quintiles, index=series.index, dtype="Float64")


def _quintile_stratum_summary(
    frame: pd.DataFrame,
    outcome_column: str,
    denominator_column: str,
    quintile_column: str,
) -> pd.DataFrame:
    subset = frame.loc[frame[quintile_column].notna()].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["exposure_quintile"] = subset[quintile_column].astype(int)
    return (
        subset.groupby(["municipality_code", "calendar_year", "exposure_quintile"], as_index=False)
        .agg(
            outcome_sum=(outcome_column, "sum"),
            denominator_sum=(denominator_column, "sum"),
            row_count=(quintile_column, "size"),
            exposure_mean=(quintile_column, "mean"),
        )
    )


def _quintile_stratum_summary_multi(
    frame: pd.DataFrame,
    outcome_columns: list[str],
    denominator_column: str,
    quintile_column: str,
) -> pd.DataFrame:
    subset = frame.loc[frame[quintile_column].notna()].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["exposure_quintile"] = subset[quintile_column].astype(int)
    aggregations: dict[str, tuple[str, str]] = {
        "denominator_sum": (denominator_column, "sum"),
        "row_count": (quintile_column, "size"),
    }
    for outcome_column in outcome_columns:
        aggregations[f"{outcome_column}_sum"] = (outcome_column, "sum")
    return subset.groupby(["municipality_code", "calendar_year", "exposure_quintile"], as_index=False).agg(**aggregations)


def _quintile_outcome_summary_from_grouped(grouped: pd.DataFrame, outcome_variable: str) -> pd.DataFrame:
    if grouped.empty:
        return pd.DataFrame()
    return grouped.loc[
        :,
        [
            "municipality_code",
            "calendar_year",
            "exposure_quintile",
            f"{outcome_variable}_sum",
            "denominator_sum",
            "row_count",
        ],
    ].rename(columns={f"{outcome_variable}_sum": "outcome_sum"})


def _q1_q5_from_quintile_summary(summary: pd.DataFrame) -> pd.DataFrame:
    subset = summary.loc[summary["exposure_quintile"].isin([1, 5])].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["group_label"] = subset["exposure_quintile"].map({1: "reference", 5: "exposed"})
    pivot = subset.pivot_table(
        index=["municipality_code", "calendar_year"],
        columns="group_label",
        values=["outcome_sum", "denominator_sum", "row_count"],
        aggfunc="first",
        fill_value=0,
    )
    pivot.columns = [f"{metric}_{group}" for metric, group in pivot.columns.to_flat_index()]
    result = pivot.reset_index()
    for column in [
        "outcome_sum_exposed",
        "outcome_sum_reference",
        "denominator_sum_exposed",
        "denominator_sum_reference",
        "row_count_exposed",
        "row_count_reference",
    ]:
        if column not in result.columns:
            result[column] = 0.0
    return result


def _estimate_stratum_quintile_slope(group: pd.DataFrame) -> tuple[float, float] | None:
    valid = group.loc[group["denominator_sum"].gt(0) & group["row_count"].gt(0)].copy()
    if valid.shape[0] < 3:
        return None
    x = valid["exposure_quintile"].astype(float).to_numpy()
    y = np.log((valid["outcome_sum"].astype(float).to_numpy() + 0.5) / valid["denominator_sum"].astype(float).to_numpy())
    weights = valid["denominator_sum"].astype(float).to_numpy()
    if np.unique(x).shape[0] < 3:
        return None
    design = np.column_stack([np.ones_like(x), x])
    xtwx = design.T @ (weights[:, None] * design)
    try:
        xtwx_inv = np.linalg.inv(xtwx)
    except np.linalg.LinAlgError:
        return None
    beta = xtwx_inv @ (design.T @ (weights * y))
    residual = y - (design @ beta)
    degrees_of_freedom = valid.shape[0] - 2
    if degrees_of_freedom <= 0:
        return None
    sigma_squared = float((weights * residual * residual).sum() / degrees_of_freedom)
    covariance = sigma_squared * xtwx_inv
    slope = float(beta[1])
    slope_se = float(math.sqrt(covariance[1, 1])) if covariance[1, 1] > 0 else float("nan")
    if not math.isfinite(slope_se) or slope_se <= 0:
        return None
    return slope, slope_se


def _pool_quintile_trend(summary: pd.DataFrame) -> dict[str, float | int]:
    if summary.empty:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "valid_strata_count": 0,
            "exposed_row_count": 0,
            "reference_row_count": 0,
            "exposed_outcome_sum": 0.0,
            "reference_outcome_sum": 0.0,
            "exposed_denominator_sum": 0.0,
            "reference_denominator_sum": 0.0,
        }

    effect_rows: list[dict[str, float]] = []
    for _, group in summary.groupby(["municipality_code", "calendar_year"], sort=False):
        effect = _estimate_stratum_quintile_slope(group)
        if effect is None:
            continue
        slope, slope_se = effect
        effect_rows.append({"slope": slope, "slope_se": slope_se})
    if not effect_rows:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "valid_strata_count": 0,
            "exposed_row_count": 0,
            "reference_row_count": 0,
            "exposed_outcome_sum": 0.0,
            "reference_outcome_sum": 0.0,
            "exposed_denominator_sum": 0.0,
            "reference_denominator_sum": 0.0,
        }

    effects = pd.DataFrame(effect_rows)
    weights = 1.0 / (effects["slope_se"].astype(float) ** 2)
    pooled_slope = float(np.average(effects["slope"].astype(float), weights=weights))
    pooled_se = float(np.sqrt(1.0 / weights.sum()))
    z_value = pooled_slope / pooled_se if pooled_se > 0 else float("nan")
    lower_ci = pooled_slope - 1.96 * pooled_se
    upper_ci = pooled_slope + 1.96 * pooled_se

    q1 = summary.loc[summary["exposure_quintile"].eq(1)]
    q5 = summary.loc[summary["exposure_quintile"].eq(5)]

    return {
        "estimate": float(math.exp(pooled_slope)),
        "lower_ci": float(math.exp(lower_ci)),
        "upper_ci": float(math.exp(upper_ci)),
        "p_value": _normal_two_sided_p(z_value),
        "valid_strata_count": int(effects.shape[0]),
        "exposed_row_count": int(q5["row_count"].sum()),
        "reference_row_count": int(q1["row_count"].sum()),
        "exposed_outcome_sum": float(q5["outcome_sum"].sum()),
        "reference_outcome_sum": float(q1["outcome_sum"].sum()),
        "exposed_denominator_sum": float(q5["denominator_sum"].sum()),
        "reference_denominator_sum": float(q1["denominator_sum"].sum()),
    }


def _pearson_correlation(frame: pd.DataFrame, x_column: str, y_column: str) -> dict[str, float | int]:
    valid = frame.loc[frame[x_column].notna() & frame[y_column].notna()].copy()
    if valid.shape[0] < 4:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "valid_strata_count": 0,
            "exposed_row_count": 0,
            "reference_row_count": 0,
            "exposed_outcome_sum": 0.0,
            "reference_outcome_sum": 0.0,
            "exposed_denominator_sum": 0.0,
            "reference_denominator_sum": 0.0,
        }
    x = valid[x_column].astype(float).to_numpy()
    y = valid[y_column].astype(float).to_numpy()
    if np.std(x) == 0 or np.std(y) == 0:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "valid_strata_count": int(valid.shape[0]),
            "exposed_row_count": 0,
            "reference_row_count": 0,
            "exposed_outcome_sum": 0.0,
            "reference_outcome_sum": 0.0,
            "exposed_denominator_sum": 0.0,
            "reference_denominator_sum": 0.0,
        }
    estimate = float(np.corrcoef(x, y)[0, 1])
    fisher_z = np.arctanh(np.clip(estimate, -0.999999, 0.999999))
    fisher_se = 1.0 / math.sqrt(valid.shape[0] - 3)
    z_value = fisher_z / fisher_se
    lower_ci = math.tanh(fisher_z - 1.96 * fisher_se)
    upper_ci = math.tanh(fisher_z + 1.96 * fisher_se)
    return {
        "estimate": estimate,
        "lower_ci": float(lower_ci),
        "upper_ci": float(upper_ci),
        "p_value": _normal_two_sided_p(float(z_value)),
        "valid_strata_count": int(valid.shape[0]),
        "exposed_row_count": 0,
        "reference_row_count": 0,
        "exposed_outcome_sum": 0.0,
        "reference_outcome_sum": 0.0,
        "exposed_denominator_sum": 0.0,
        "reference_denominator_sum": 0.0,
    }


def _base_exposure_label(base_exposure: str, suffix: str) -> str:
    return f"{BASE_EXPOSURES[base_exposure]['label']} ({WINDOW_SUFFIX_LABELS[suffix]})"


def _split_window_exposure_column(exposure_column: str) -> tuple[str, str]:
    for base_exposure in BASE_EXPOSURES:
        for suffix in DELIVERY_WINDOW_SUFFIXES:
            candidate = f"{base_exposure}_{suffix}"
            if candidate == exposure_column:
                return base_exposure, suffix
    raise KeyError(f"Unable to resolve delivery-window exposure column {exposure_column!r}")


def _make_result_row(
    readiness_lookup: dict[str, dict[str, str]],
    outcome_variable: str,
    outcome_label: str,
    exposure_variable: str,
    exposure_label: str,
    exposure_family_id: str,
    exposure_family_name: str,
    lag_or_window: str,
    screening_domain: str,
    screening_method: str,
    estimate_scale: str,
    comparison_label: str,
    contrast_definition: str,
    statistics: dict[str, float | int],
) -> dict[str, Any]:
    readiness = readiness_lookup.get(
        outcome_variable,
        {
            "public_concept_name": outcome_variable,
            "public_concept_label": outcome_label,
            "stage1_classification": "unknown",
            "headline_eligible_flag": "no",
            "benchmark_model_eligible_flag": "no",
            "duplicate_prompt_concept_flag": "no",
        },
    )
    estimate = statistics["estimate"]
    if estimate_scale.startswith("rate_ratio"):
        direction = "positive" if math.isfinite(float(estimate)) and float(estimate) > 1 else "negative"
    else:
        direction = "positive" if math.isfinite(float(estimate)) and float(estimate) > 0 else "negative"
    if not math.isfinite(float(estimate)):
        direction = "indeterminate"
    return {
        "public_concept_name": readiness["public_concept_name"],
        "public_concept_label": readiness["public_concept_label"],
        "outcome_variable": outcome_variable,
        "outcome_label": outcome_label,
        "stage1_classification": readiness["stage1_classification"],
        "headline_eligible_flag": readiness["headline_eligible_flag"],
        "benchmark_model_eligible_flag": readiness["benchmark_model_eligible_flag"],
        "duplicate_prompt_concept_flag": readiness["duplicate_prompt_concept_flag"],
        "exposure_variable": exposure_variable,
        "exposure_label": exposure_label,
        "exposure_family_id": exposure_family_id,
        "exposure_family_name": exposure_family_name,
        "lag_or_window": lag_or_window,
        "screening_domain": screening_domain,
        "screening_method": screening_method,
        "comparison_label": comparison_label,
        "contrast_definition": contrast_definition,
        "estimate_scale": estimate_scale,
        "estimate": statistics["estimate"],
        "lower_ci": statistics["lower_ci"],
        "upper_ci": statistics["upper_ci"],
        "p_value": statistics["p_value"],
        "p_value_floor_applied_flag": str(statistics.get("p_value_floor_applied_flag", "no")),
        "valid_strata_count": statistics["valid_strata_count"],
        "exposed_row_count": statistics["exposed_row_count"],
        "reference_row_count": statistics["reference_row_count"],
        "exposed_outcome_sum": statistics["exposed_outcome_sum"],
        "reference_outcome_sum": statistics["reference_outcome_sum"],
        "exposed_denominator_sum": statistics["exposed_denominator_sum"],
        "reference_denominator_sum": statistics["reference_denominator_sum"],
        "signal_direction": direction,
    }


def _readiness_matrix_path(root: Path) -> Path:
    return track_a_root(root) / "freeze" / "analysis_ready_matrix.csv"


def _screening_cache_root(root: Path) -> Path:
    return _screening_root(root) / "cache"


def _acute_denominator_groups() -> dict[str, list[str]]:
    return {
        "annual_live_birth_count": ["neonatal_death_count", "infant_death_count"],
        "under_five_population_count": ["under_five_death_count", "acute_care_proxy_count"],
    }


def _window_denominator_groups(outcomes: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for outcome in outcomes:
        groups.setdefault(outcome["denominator_column"], []).append(outcome)
    return groups


def _empty_acute_binary_grouped() -> pd.DataFrame:
    return pd.DataFrame(columns=ACUTE_BINARY_GROUPED_COLUMNS)


def _empty_window_grouped() -> pd.DataFrame:
    return pd.DataFrame(columns=WINDOW_GROUPED_COLUMNS)


def _empty_weekly_acute() -> pd.DataFrame:
    columns = ACUTE_WEEKLY_COLUMNS + [f"{column}_anomaly" for column in ACUTE_CORRELATION_EXPOSURES + [item["outcome_variable"] for item in ACUTE_OUTCOMES]]
    return pd.DataFrame(columns=columns)


def _statistics_from_frame_row(row: pd.Series) -> dict[str, float | int | str]:
    return {
        "estimate": float(row["estimate"]) if pd.notna(row["estimate"]) else float("nan"),
        "lower_ci": float(row["lower_ci"]) if pd.notna(row["lower_ci"]) else float("nan"),
        "upper_ci": float(row["upper_ci"]) if pd.notna(row["upper_ci"]) else float("nan"),
        "p_value": float(row["p_value"]) if pd.notna(row["p_value"]) else float("nan"),
        "p_value_floor_applied_flag": str(row.get("p_value_floor_applied_flag", "no")),
        "valid_strata_count": int(row.get("valid_strata_count", 0) or 0),
        "exposed_row_count": int(row.get("exposed_row_count", 0) or 0),
        "reference_row_count": int(row.get("reference_row_count", 0) or 0),
        "exposed_outcome_sum": float(row.get("exposed_outcome_sum", 0.0) or 0.0),
        "reference_outcome_sum": float(row.get("reference_outcome_sum", 0.0) or 0.0),
        "exposed_denominator_sum": float(row.get("exposed_denominator_sum", 0.0) or 0.0),
        "reference_denominator_sum": float(row.get("reference_denominator_sum", 0.0) or 0.0),
    }


def _build_acute_binary_grouped_for_year(root: Path, year: int) -> pd.DataFrame:
    frame = _load_partition(root, "acute_daily_mart", year, ACUTE_REQUIRED_COLUMNS)
    base_valid = _filter_base_screening_rows(frame)
    if base_valid.empty:
        return _empty_acute_binary_grouped()

    result_frames: list[pd.DataFrame] = []
    denominator_groups = _acute_denominator_groups()
    id_columns = ["municipality_code", "calendar_year", "screen_group"]

    for lag_suffix in ACUTE_TEMPERATURE_LAG_SUFFIXES:
        exposure_column = f"maximum_daily_temperature_c_{lag_suffix}"
        quantiles = (
            base_valid.groupby("municipality_code")[exposure_column]
            .quantile([0.40, 0.60, 0.95])
            .unstack()
            .rename(columns={0.40: "quantile_40", 0.60: "quantile_60", 0.95: "quantile_95"})
        )
        working = base_valid.loc[
            :,
            [
                "municipality_code",
                "calendar_year",
                "annual_live_birth_count",
                "under_five_population_count",
                exposure_column,
                "neonatal_death_count",
                "infant_death_count",
                "under_five_death_count",
                "acute_care_proxy_count",
            ],
        ].merge(
            quantiles,
            left_on="municipality_code",
            right_index=True,
            how="left",
        )
        working["screen_group"] = np.where(
            working[exposure_column] >= working["quantile_95"],
            "exposed",
            np.where(
                (working[exposure_column] >= working["quantile_40"])
                & (working[exposure_column] <= working["quantile_60"]),
                "reference",
                "exclude",
            ),
        )
        for denominator_column, outcome_variables in denominator_groups.items():
            denominator_valid = working.loc[working[denominator_column].fillna(0).gt(0)].copy()
            if denominator_valid.empty:
                continue
            grouped = aggregate_binary_outcomes(
                denominator_valid,
                index_columns=["municipality_code", "calendar_year"],
                outcome_columns=outcome_variables,
                denominator_column=denominator_column,
                group_column="screen_group",
            )
            long_frame = grouped_wide_to_long(grouped, id_columns=id_columns, outcome_columns=outcome_variables)
            if long_frame.empty:
                continue
            long_frame["screening_domain"] = "acute_temperature_reference_contrast"
            long_frame["exposure_variable"] = exposure_column
            long_frame["exposure_family_id"] = "family_1_tmax_primary"
            long_frame["exposure_family_name"] = "Maximum daily temperature"
            long_frame["lag_or_window"] = lag_suffix
            long_frame["denominator_column"] = denominator_column
            result_frames.append(long_frame.loc[:, ACUTE_BINARY_GROUPED_COLUMNS])

    heatwave_column = "heatwave_duration_days_p95_threshold_lag0"
    working = base_valid.loc[
        :,
        [
            "municipality_code",
            "calendar_year",
            "annual_live_birth_count",
            "under_five_population_count",
            heatwave_column,
            "neonatal_death_count",
            "infant_death_count",
            "under_five_death_count",
            "acute_care_proxy_count",
        ],
    ].copy()
    working["screen_group"] = np.where(working[heatwave_column] > 0, "exposed", "reference")
    for denominator_column, outcome_variables in denominator_groups.items():
        denominator_valid = working.loc[working[denominator_column].fillna(0).gt(0)].copy()
        if denominator_valid.empty:
            continue
        grouped = aggregate_binary_outcomes(
            denominator_valid,
            index_columns=["municipality_code", "calendar_year"],
            outcome_columns=outcome_variables,
            denominator_column=denominator_column,
            group_column="screen_group",
        )
        long_frame = grouped_wide_to_long(grouped, id_columns=id_columns, outcome_columns=outcome_variables)
        if long_frame.empty:
            continue
        long_frame["screening_domain"] = "acute_heatwave_binary_contrast"
        long_frame["exposure_variable"] = heatwave_column
        long_frame["exposure_family_id"] = "family_3_heatwave_duration"
        long_frame["exposure_family_name"] = "Heatwave duration sensitivity"
        long_frame["lag_or_window"] = "lag0"
        long_frame["denominator_column"] = denominator_column
        result_frames.append(long_frame.loc[:, ACUTE_BINARY_GROUPED_COLUMNS])

    if not result_frames:
        return _empty_acute_binary_grouped()
    return pd.concat(result_frames, ignore_index=True)


def _build_acute_weekly_for_year(root: Path, year: int) -> pd.DataFrame:
    frame = _load_partition(root, "acute_daily_mart", year, ACUTE_REQUIRED_COLUMNS)
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
    frame = frame.loc[frame["observation_date"].notna()].copy()
    if frame.empty:
        return _empty_weekly_acute()

    weekly = (
        frame.assign(epidemiologic_week=frame["observation_date"].dt.isocalendar().week.astype(int))
        .groupby(["municipality_code", "calendar_year", "epidemiologic_week"], as_index=False)
        .agg(
            maximum_daily_temperature_c=("maximum_daily_temperature_c", "mean"),
            mean_daily_relative_humidity_pct=("mean_daily_relative_humidity_pct", "mean"),
            heatwave_duration_days_p95_threshold=("heatwave_duration_days_p95_threshold", "mean"),
            approximate_heat_index_c=("approximate_heat_index_c", "mean"),
            neonatal_death_count=("neonatal_death_count", "sum"),
            infant_death_count=("infant_death_count", "sum"),
            under_five_death_count=("under_five_death_count", "sum"),
            acute_care_proxy_count=("acute_care_proxy_count", "sum"),
        )
    )
    for column in ACUTE_CORRELATION_EXPOSURES + [item["outcome_variable"] for item in ACUTE_OUTCOMES]:
        weekly[f"{column}_anomaly"] = weekly[column] - weekly.groupby(["municipality_code", "calendar_year"])[column].transform("mean")
    return weekly


def _build_window_grouped_for_year(
    root: Path,
    year: int,
    mart_name: str,
    required_columns: list[str],
    outcomes: list[dict[str, str]],
    screening_domain_prefix: str,
) -> pd.DataFrame:
    exposure_columns = [
        f"{base_exposure}_{suffix}"
        for base_exposure in BASE_EXPOSURES
        for suffix in DELIVERY_WINDOW_SUFFIXES
    ]
    frame = _load_partition(root, mart_name, year, required_columns + exposure_columns)
    base_valid = _filter_base_screening_rows(frame)
    if base_valid.empty:
        return _empty_window_grouped()

    result_frames: list[pd.DataFrame] = []
    denominator_groups = _window_denominator_groups(outcomes)
    id_columns = ["municipality_code", "calendar_year", "exposure_quintile"]

    for base_exposure, exposure_meta in BASE_EXPOSURES.items():
        for suffix in DELIVERY_WINDOW_SUFFIXES:
            exposure_column = f"{base_exposure}_{suffix}"
            quintiles = _assign_quintiles(base_valid[exposure_column])
            if quintiles.notna().sum() == 0:
                continue
            for denominator_column, outcome_specs in denominator_groups.items():
                outcome_variables = [item["outcome_variable"] for item in outcome_specs]
                valid = base_valid.loc[
                    base_valid[denominator_column].fillna(0).gt(0),
                    [
                        "municipality_code",
                        "calendar_year",
                        denominator_column,
                        *outcome_variables,
                    ],
                ].copy()
                if valid.empty:
                    continue
                valid["exposure_quintile_value"] = quintiles.reindex(valid.index)
                grouped = aggregate_quintile_outcomes(
                    valid,
                    index_columns=["municipality_code", "calendar_year"],
                    outcome_columns=outcome_variables,
                    denominator_column=denominator_column,
                    quintile_column="exposure_quintile_value",
                )
                long_frame = grouped_wide_to_long(grouped, id_columns=id_columns, outcome_columns=outcome_variables)
                if long_frame.empty:
                    continue
                long_frame["mart_name"] = mart_name
                long_frame["screening_domain_prefix"] = screening_domain_prefix
                long_frame["exposure_variable"] = exposure_column
                long_frame["exposure_family_id"] = exposure_meta["family_id"]
                long_frame["exposure_family_name"] = exposure_meta["family_name"]
                long_frame["lag_or_window"] = suffix
                long_frame["denominator_column"] = denominator_column
                result_frames.append(long_frame.loc[:, WINDOW_GROUPED_COLUMNS])

    if not result_frames:
        return _empty_window_grouped()
    return pd.concat(result_frames, ignore_index=True)


def _render_runtime_report(metrics: dict[str, Any]) -> str:
    phase_lines = []
    for phase_name, seconds in metrics["phase_seconds"].items():
        phase_lines.append(f"- `{phase_name}`: {seconds:.3f} seconds")
    return (
        "# Track A Screening Runtime Report\n\n"
        f"- Run scope: {metrics['scope_key']}\n"
        f"- Cache state: {metrics['cache_state']}\n"
        f"- Cache hits: {metrics['cache_hit_count']}\n"
        f"- Cache misses: {metrics['cache_miss_count']}\n"
        f"- Result row count: {metrics['result_count']}\n"
        f"- Triage public concept count: {metrics['triage_public_concept_count']}\n"
        f"- Total runtime seconds: {metrics['runtime_seconds']:.3f}\n\n"
        "## Phase Timings\n\n"
        + "\n".join(phase_lines)
        + "\n"
    )


def _render_stage4_review_note(
    years: list[int],
    metrics: dict[str, Any],
    triage_frame: pd.DataFrame,
    prior_summary: dict[str, Any] | None,
) -> str:
    triage_counts = metrics["triage_counts"]
    benchmark_ready = triage_frame.loc[triage_frame["benchmark_stage_recommendation"].eq("yes")].copy()
    top_concepts = benchmark_ready.sort_values(
        ["triage_priority_classification", "minimum_q_value", "minimum_p_value"],
        ascending=[True, True, True],
        na_position="last",
    )["public_concept_label"].head(5).tolist()
    guardrail_held = triage_frame.loc[
        triage_frame["benchmark_stage_recommendation"].eq("no")
        & triage_frame["minimum_p_value"].fillna(1.0).le(0.10)
    ]["public_concept_label"].tolist()

    comparison_lines = ["- No prior validation summary was available for comparison."]
    if prior_summary is not None:
        prior_counts = prior_summary.get("triage_counts", {})
        comparison_lines = [
            f"- Prior scope: {prior_summary.get('scope_key', 'unknown')}",
            f"- Prior runtime seconds: {float(prior_summary.get('runtime_seconds', float('nan'))):.3f}",
            f"- Prior triage counts: advance={int(prior_counts.get('advance', 0))}, advance_with_caveats={int(prior_counts.get('advance_with_caveats', 0))}, descriptive_only={int(prior_counts.get('descriptive_only', 0))}",
        ]
        prior_triage_map = prior_summary.get("triage_map", {})
        changed = []
        current_triage_map = {
            str(row["public_concept_name"]): str(row["triage_priority_classification"])
            for row in triage_frame.to_dict(orient="records")
        }
        for concept_name, classification in current_triage_map.items():
            previous = prior_triage_map.get(concept_name)
            if previous is not None and previous != classification:
                changed.append(f"{concept_name}: {previous} -> {classification}")
        if changed:
            comparison_lines.append("- Triage shifts versus the earlier validation run: " + "; ".join(changed[:5]))
        else:
            comparison_lines.append("- No triage classification shifts were detected versus the earlier validation run.")

    can_proceed = (
        metrics["runtime_seconds"] <= 600.0
        and metrics["result_count"] == 276
        and metrics["triage_public_concept_count"] == 14
        and triage_frame.loc[
            triage_frame["public_concept_name"].isin(
                [
                    "acute_care_proxy_count",
                    "severe_hypertensive_disorder_of_pregnancy_count",
                ]
            ),
            "benchmark_stage_recommendation",
        ].eq("no").all()
    )
    recommendation = "Stage 5 may proceed." if can_proceed else "Stage 5 should not proceed yet."

    return (
        "# Track A Stage 4 Review Note\n\n"
        f"- Run scope: {years[0]}-{years[-1]}\n"
        f"- Cache state: {metrics['cache_state']}\n"
        f"- Runtime seconds: {metrics['runtime_seconds']:.3f}\n"
        f"- Total screening rows: {metrics['result_count']}\n"
        f"- Triage counts: advance={triage_counts['advance']}, advance_with_caveats={triage_counts['advance_with_caveats']}, descriptive_only={triage_counts['descriptive_only']}\n"
        f"- Top benchmark-ready concepts: {', '.join(top_concepts) if top_concepts else 'None'}\n"
        f"- Concepts held back by guardrails: {', '.join(guardrail_held) if guardrail_held else 'None'}\n\n"
        "## Comparison With Earlier Validation Run\n\n"
        + "\n".join(comparison_lines)
        + "\n\n## Recommendation\n\n"
        + recommendation
        + "\n"
    )


def _acute_screening_results(root: Path, years: list[int], readiness_lookup: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    threshold_accumulator: dict[tuple[str, str, str], list[pd.DataFrame]] = {}
    weekly_frames: list[pd.DataFrame] = []
    acute_denominator_groups = {
        "annual_live_birth_count": ["neonatal_death_count", "infant_death_count"],
        "under_five_population_count": ["under_five_death_count", "acute_care_proxy_count"],
    }
    acute_outcome_lookup = {item["outcome_variable"]: item for item in ACUTE_OUTCOMES}

    for year in years:
        frame = _load_partition(root, "acute_daily_mart", year, ACUTE_REQUIRED_COLUMNS)
        frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
        base_valid = _filter_base_screening_rows(frame)
        if base_valid.empty:
            continue

        for lag_suffix in ACUTE_TEMPERATURE_LAG_SUFFIXES:
            exposure_column = f"maximum_daily_temperature_c_{lag_suffix}"
            quantiles = (
                base_valid.groupby("municipality_code")[exposure_column]
                .quantile([0.40, 0.60, 0.95])
                .unstack()
                .rename(columns={0.40: "quantile_40", 0.60: "quantile_60", 0.95: "quantile_95"})
            )
            working = base_valid.loc[
                :,
                [
                    "municipality_code",
                    "calendar_year",
                    "annual_live_birth_count",
                    "under_five_population_count",
                    exposure_column,
                    "neonatal_death_count",
                    "infant_death_count",
                    "under_five_death_count",
                    "acute_care_proxy_count",
                ],
            ].merge(
                quantiles,
                left_on="municipality_code",
                right_index=True,
                how="left",
            )
            working["screen_group"] = np.where(
                working[exposure_column] >= working["quantile_95"],
                "exposed",
                np.where(
                    (working[exposure_column] >= working["quantile_40"])
                    & (working[exposure_column] <= working["quantile_60"]),
                    "reference",
                    "exclude",
                ),
            )
            for denominator_column, outcome_variables in acute_denominator_groups.items():
                denominator_valid = working.loc[working[denominator_column].fillna(0).gt(0)].copy()
                if denominator_valid.empty:
                    continue
                grouped = _binary_stratum_summary_multi(
                    denominator_valid,
                    outcome_columns=outcome_variables,
                    denominator_column=denominator_column,
                    group_column="screen_group",
                )
                for outcome_variable in outcome_variables:
                    summary = _binary_outcome_summary_from_grouped(grouped, outcome_variable)
                    key = ("acute_temperature_reference_contrast", exposure_column, outcome_variable)
                    threshold_accumulator.setdefault(key, []).append(summary)

        heatwave_column = "heatwave_duration_days_p95_threshold_lag0"
        working = base_valid.loc[
            :,
            [
                "municipality_code",
                "calendar_year",
                "annual_live_birth_count",
                "under_five_population_count",
                heatwave_column,
                "neonatal_death_count",
                "infant_death_count",
                "under_five_death_count",
                "acute_care_proxy_count",
            ],
        ].copy()
        working["screen_group"] = np.where(working[heatwave_column] > 0, "exposed", "reference")
        for denominator_column, outcome_variables in acute_denominator_groups.items():
            denominator_valid = working.loc[working[denominator_column].fillna(0).gt(0)].copy()
            if denominator_valid.empty:
                continue
            grouped = _binary_stratum_summary_multi(
                denominator_valid,
                outcome_columns=outcome_variables,
                denominator_column=denominator_column,
                group_column="screen_group",
            )
            for outcome_variable in outcome_variables:
                summary = _binary_outcome_summary_from_grouped(grouped, outcome_variable)
                key = ("acute_heatwave_binary_contrast", heatwave_column, outcome_variable)
                threshold_accumulator.setdefault(key, []).append(summary)

        weekly = frame.loc[
            :,
            [
                "municipality_code",
                "calendar_year",
                "observation_date",
                *ACUTE_CORRELATION_EXPOSURES,
                *(outcome["outcome_variable"] for outcome in ACUTE_OUTCOMES),
            ],
        ].copy()
        weekly["epidemiologic_week"] = weekly["observation_date"].dt.isocalendar().week.astype(int)
        weekly = (
            weekly.groupby(["municipality_code", "calendar_year", "epidemiologic_week"], as_index=False)
            .agg(
                maximum_daily_temperature_c=("maximum_daily_temperature_c", "mean"),
                mean_daily_relative_humidity_pct=("mean_daily_relative_humidity_pct", "mean"),
                heatwave_duration_days_p95_threshold=("heatwave_duration_days_p95_threshold", "mean"),
                approximate_heat_index_c=("approximate_heat_index_c", "mean"),
                neonatal_death_count=("neonatal_death_count", "sum"),
                infant_death_count=("infant_death_count", "sum"),
                under_five_death_count=("under_five_death_count", "sum"),
                acute_care_proxy_count=("acute_care_proxy_count", "sum"),
            )
        )
        for column in ACUTE_CORRELATION_EXPOSURES + [outcome["outcome_variable"] for outcome in ACUTE_OUTCOMES]:
            weekly[f"{column}_anomaly"] = weekly[column] - weekly.groupby(["municipality_code", "calendar_year"])[column].transform("mean")
        weekly_frames.append(weekly)

    result_rows: list[dict[str, Any]] = []
    for (screening_domain, exposure_column, outcome_variable), frames in threshold_accumulator.items():
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        statistics = _pool_log_rate_ratio(combined)
        outcome_spec = acute_outcome_lookup[outcome_variable]
        if screening_domain == "acute_temperature_reference_contrast":
            suffix = exposure_column.split("maximum_daily_temperature_c_", 1)[1]
            result_rows.append(
                _make_result_row(
                    readiness_lookup=readiness_lookup,
                    outcome_variable=outcome_variable,
                    outcome_label=outcome_spec["outcome_label"],
                    exposure_variable=exposure_column,
                    exposure_label=_base_exposure_label("maximum_daily_temperature_c", suffix),
                    exposure_family_id="family_1_tmax_primary",
                    exposure_family_name="Maximum daily temperature",
                    lag_or_window=suffix,
                    screening_domain=screening_domain,
                    screening_method="within_municipality_year_p95_vs_reference_rate_ratio",
                    estimate_scale="rate_ratio",
                    comparison_label="Municipality-year p95 versus municipality-year p40 to p60 reference band",
                    contrast_definition="Rows with exposure at or above the municipality-year p95 are compared with rows in the municipality-year p40 to p60 reference band.",
                    statistics=statistics,
                )
            )
        else:
            result_rows.append(
                _make_result_row(
                    readiness_lookup=readiness_lookup,
                    outcome_variable=outcome_variable,
                    outcome_label=outcome_spec["outcome_label"],
                    exposure_variable=exposure_column,
                    exposure_label=_base_exposure_label("heatwave_duration_days_p95_threshold", "lag0"),
                    exposure_family_id="family_3_heatwave_duration",
                    exposure_family_name="Heatwave duration sensitivity",
                    lag_or_window="lag0",
                    screening_domain=screening_domain,
                    screening_method="within_municipality_year_heatwave_vs_non_heatwave_rate_ratio",
                    estimate_scale="rate_ratio",
                    comparison_label="Heatwave day versus non-heatwave day",
                    contrast_definition="Rows with heatwave duration above zero are compared with rows without a heatwave signal in the same municipality-year.",
                    statistics=statistics,
                )
            )

    weekly_frame = pd.concat(weekly_frames, ignore_index=True) if weekly_frames else pd.DataFrame()
    if not weekly_frame.empty:
        for exposure_variable in ACUTE_CORRELATION_EXPOSURES:
            exposure_meta = BASE_EXPOSURES[exposure_variable]
            for outcome in ACUTE_OUTCOMES:
                statistics = _pearson_correlation(
                    weekly_frame,
                    x_column=f"{exposure_variable}_anomaly",
                    y_column=f"{outcome['outcome_variable']}_anomaly",
                )
                result_rows.append(
                    _make_result_row(
                        readiness_lookup=readiness_lookup,
                        outcome_variable=outcome["outcome_variable"],
                        outcome_label=outcome["outcome_label"],
                        exposure_variable=exposure_variable,
                        exposure_label=f"{exposure_meta['label']} (weekly anomaly)",
                        exposure_family_id=exposure_meta["family_id"],
                        exposure_family_name=exposure_meta["family_name"],
                        lag_or_window="weekly_anomaly",
                        screening_domain="acute_deseasonalized_weekly_anomaly_correlation",
                        screening_method="pearson_correlation_on_within_municipality_year_weekly_anomalies",
                        estimate_scale="correlation_coefficient",
                        comparison_label="Deseasonalized weekly anomaly correlation",
                        contrast_definition="Weekly municipality-year anomalies are correlated after centering each municipality-year series around its own mean.",
                        statistics=statistics,
                    )
                )
    return result_rows


def _window_screening_results(
    root: Path,
    years: list[int],
    mart_name: str,
    required_columns: list[str],
    outcomes: list[dict[str, str]],
    readiness_lookup: dict[str, dict[str, str]],
    screening_domain_prefix: str,
) -> list[dict[str, Any]]:
    q1_q5_accumulator: dict[tuple[str, str], list[pd.DataFrame]] = {}
    trend_accumulator: dict[tuple[str, str], list[pd.DataFrame]] = {}
    denominator_groups: dict[str, list[dict[str, str]]] = {}
    for outcome in outcomes:
        denominator_groups.setdefault(outcome["denominator_column"], []).append(outcome)
    outcome_lookup = {item["outcome_variable"]: item for item in outcomes}

    exposure_columns = [
        f"{base_exposure}_{suffix}"
        for base_exposure in BASE_EXPOSURES
        for suffix in DELIVERY_WINDOW_SUFFIXES
    ]

    for year in years:
        frame = _load_partition(root, mart_name, year, required_columns + exposure_columns)
        base_valid = _filter_base_screening_rows(frame)
        if base_valid.empty:
            continue
        for exposure_column in exposure_columns:
            base_valid[f"{exposure_column}_quintile"] = _assign_quintiles(base_valid[exposure_column])
        for base_exposure in BASE_EXPOSURES:
            for suffix in DELIVERY_WINDOW_SUFFIXES:
                exposure_column = f"{base_exposure}_{suffix}"
                quintile_column = f"{exposure_column}_quintile"
                for denominator_column, outcome_specs in denominator_groups.items():
                    outcome_variables = [item["outcome_variable"] for item in outcome_specs]
                    valid = base_valid.loc[base_valid[denominator_column].fillna(0).gt(0)].copy()
                    if valid.empty:
                        continue
                    working = valid.loc[
                        :,
                        [
                            "municipality_code",
                            "calendar_year",
                            denominator_column,
                            quintile_column,
                            *outcome_variables,
                        ],
                    ].copy()
                    grouped = _quintile_stratum_summary_multi(
                        working,
                        outcome_columns=outcome_variables,
                        denominator_column=denominator_column,
                        quintile_column=quintile_column,
                    )
                    for outcome_variable in outcome_variables:
                        summary = _quintile_outcome_summary_from_grouped(grouped, outcome_variable)
                        q1_q5_accumulator.setdefault((exposure_column, outcome_variable), []).append(
                            _q1_q5_from_quintile_summary(summary)
                        )
                        trend_accumulator.setdefault((exposure_column, outcome_variable), []).append(summary)

    result_rows: list[dict[str, Any]] = []
    for (exposure_column, outcome_variable), frames in q1_q5_accumulator.items():
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        statistics = _pool_log_rate_ratio(combined)
        outcome_spec = outcome_lookup[outcome_variable]
        base_exposure, suffix = _split_window_exposure_column(exposure_column)
        exposure_meta = BASE_EXPOSURES[base_exposure]
        result_rows.append(
            _make_result_row(
                readiness_lookup=readiness_lookup,
                outcome_variable=outcome_variable,
                outcome_label=outcome_spec["outcome_label"],
                exposure_variable=exposure_column,
                exposure_label=_base_exposure_label(base_exposure, suffix),
                exposure_family_id=exposure_meta["family_id"],
                exposure_family_name=exposure_meta["family_name"],
                lag_or_window=suffix,
                screening_domain=f"{screening_domain_prefix}_q5_vs_q1",
                screening_method="within_municipality_year_quintile_extremes_rate_ratio",
                estimate_scale="rate_ratio",
                comparison_label="Exposure quintile 5 versus exposure quintile 1",
                contrast_definition="Rows in the highest year-specific exposure quintile are compared with rows in the lowest year-specific exposure quintile.",
                statistics=statistics,
            )
        )

    for (exposure_column, outcome_variable), frames in trend_accumulator.items():
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        statistics = _pool_quintile_trend(combined)
        outcome_spec = outcome_lookup[outcome_variable]
        base_exposure, suffix = _split_window_exposure_column(exposure_column)
        exposure_meta = BASE_EXPOSURES[base_exposure]
        result_rows.append(
            _make_result_row(
                readiness_lookup=readiness_lookup,
                outcome_variable=outcome_variable,
                outcome_label=outcome_spec["outcome_label"],
                exposure_variable=exposure_column,
                exposure_label=_base_exposure_label(base_exposure, suffix),
                exposure_family_id=exposure_meta["family_id"],
                exposure_family_name=exposure_meta["family_name"],
                lag_or_window=suffix,
                screening_domain=f"{screening_domain_prefix}_quintile_trend",
                screening_method="within_municipality_year_log_rate_quintile_trend",
                estimate_scale="rate_ratio_per_quintile",
                comparison_label="Per-quintile monotonic trend",
                contrast_definition="A weighted within-stratum log-rate trend is estimated across year-specific exposure quintiles one through five and then pooled across municipality-years.",
                statistics=statistics,
            )
        )
    return result_rows


def _apply_benjamini_hochberg(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    frame["q_value"] = np.nan
    for _, index in frame.groupby(["outcome_family", "exposure_family_id"]).groups.items():
        subset = frame.loc[index, "p_value"].astype(float)
        valid_subset = subset.loc[subset.notna()].sort_values()
        if valid_subset.empty:
            continue
        rank = np.arange(1, valid_subset.shape[0] + 1, dtype=float)
        adjusted = (valid_subset.to_numpy(dtype=float) * valid_subset.shape[0]) / rank
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)
        frame.loc[valid_subset.index, "q_value"] = adjusted
    return frame


def _row_priority_classification(row: pd.Series) -> str:
    q_value = float(row["q_value"]) if pd.notna(row["q_value"]) else float("nan")
    p_value = float(row["p_value"]) if pd.notna(row["p_value"]) else float("nan")
    benchmark_eligible = str(row["benchmark_model_eligible_flag"]) == "yes"
    if benchmark_eligible and math.isfinite(q_value) and q_value <= 0.10:
        return "advance"
    if (benchmark_eligible and math.isfinite(q_value) and q_value <= 0.20) or (math.isfinite(p_value) and p_value <= 0.10):
        return "advance_with_caveats"
    return "descriptive_only"


def _signal_sign(row: pd.Series) -> int:
    if not math.isfinite(float(row["estimate"])):
        return 0
    if str(row["estimate_scale"]).startswith("rate_ratio"):
        if float(row["estimate"]) > 1:
            return 1
        if float(row["estimate"]) < 1:
            return -1
        return 0
    if float(row["estimate"]) > 0:
        return 1
    if float(row["estimate"]) < 0:
        return -1
    return 0


def _build_triage_signal_matrix(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = results.groupby(["public_concept_name", "public_concept_label"], sort=True)
    for (_, _), group in grouped:
        ordered = group.sort_values(["q_value", "p_value"], na_position="last")
        best = ordered.iloc[0]
        signal_subset = group.loc[group["p_value"].fillna(1.0) <= 0.10].copy()
        sign_values = signal_subset.apply(_signal_sign, axis=1).tolist()
        non_zero_signs = [value for value in sign_values if value != 0]
        sign_consistency = (
            abs(sum(non_zero_signs)) / len(non_zero_signs)
            if non_zero_signs
            else 0.0
        )
        significant_domain_count = int(signal_subset["screening_domain"].nunique())
        q_value_min = float(group["q_value"].min()) if group["q_value"].notna().any() else float("nan")
        p_value_min = float(group["p_value"].min()) if group["p_value"].notna().any() else float("nan")
        benchmark_eligible = str(best["benchmark_model_eligible_flag"]) == "yes"

        if benchmark_eligible and math.isfinite(q_value_min) and q_value_min <= 0.10 and significant_domain_count >= 2 and sign_consistency >= 0.5:
            triage = "advance"
        elif (benchmark_eligible and ((math.isfinite(q_value_min) and q_value_min <= 0.20) or significant_domain_count >= 1)) or (
            not benchmark_eligible and significant_domain_count >= 1
        ):
            triage = "advance_with_caveats"
        else:
            triage = "descriptive_only"

        rows.append(
            {
                "public_concept_name": best["public_concept_name"],
                "public_concept_label": best["public_concept_label"],
                "stage1_classification": best["stage1_classification"],
                "headline_eligible_flag": best["headline_eligible_flag"],
                "benchmark_model_eligible_flag": best["benchmark_model_eligible_flag"],
                "best_exposure_variable": best["exposure_variable"],
                "best_exposure_label": best["exposure_label"],
                "best_screening_domain": best["screening_domain"],
                "best_screening_method": best["screening_method"],
                "best_lag_or_window": best["lag_or_window"],
                "best_estimate_scale": best["estimate_scale"],
                "best_estimate": best["estimate"],
                "best_lower_ci": best["lower_ci"],
                "best_upper_ci": best["upper_ci"],
                "minimum_p_value": p_value_min,
                "minimum_q_value": q_value_min,
                "screening_result_count": int(group.shape[0]),
                "signal_domain_count_p_le_0_10": significant_domain_count,
                "signal_direction_consistency": round(sign_consistency, 4),
                "triage_priority_classification": triage,
                "benchmark_stage_recommendation": "yes" if benchmark_eligible and triage in {"advance", "advance_with_caveats"} else "no",
            }
        )
    return pd.DataFrame(rows).sort_values(["triage_priority_classification", "minimum_q_value", "minimum_p_value"], ascending=[True, True, True], na_position="last")


def _render_methods_note() -> str:
    return (
        "# Track A Screening Methods Note\n\n"
        "Stage 4 implements a disciplined screening layer for Track A. The purpose is triage, not definitive inference.\n\n"
        "## Acute Daily Screening\n\n"
        "- Temperature contrasts compare municipality-year exposure values at or above the municipality-year p95 with a municipality-year reference band spanning the p40 to p60 range.\n"
        "- Heatwave contrasts compare municipality-days with a positive heatwave duration against municipality-days without a heatwave signal within the same municipality-year.\n"
        "- Binary contrasts are summarized as pooled within-municipality-year rate ratios using continuity-corrected log-rate contrasts. This provides a fixed-effect-aligned screening signal but does not replace the benchmark ladder planned for Stage 6.\n"
        "- Weekly anomaly correlations are computed after aggregating to municipality-year epidemiologic weeks and centering each municipality-year series around its own mean.\n\n"
        "## Delivery-Aligned Window Screening\n\n"
        "- Perinatal and maternal operational outcomes are screened against year-specific exposure quintiles defined on delivery-aligned or event-aligned windows spanning days 0 to 6, 0 to 13, and 0 to 27 before the event date.\n"
        "- Quintile-extreme contrasts compare quintile 5 with quintile 1 within municipality-year strata.\n"
        "- Monotonic trends are estimated as pooled within-stratum log-rate slopes across exposure quintiles one through five.\n"
        "- These windows remain observable event-aligned summaries and must not be interpreted as trimester-specific or whole-pregnancy causal exposures.\n\n"
        "## Multiplicity and Triage\n\n"
        "- False discovery rate control uses the Benjamini-Hochberg procedure within each outcome family and exposure family.\n"
        f"- Extremely small probability values are floored at {PROBABILITY_FLOOR:.0e} in the exported screening table, with explicit floor flags retained for auditability.\n"
        "- Stage 1 guardrails remain active: descriptive-only proxy concepts are not promoted to headline status, and outcomes without benchmark eligibility remain in supplementary triage only.\n"
        "- Runtime telemetry and cache metadata are recorded alongside the screening outputs to support reproducibility of the full-range review run.\n"
        "- Final triage classes are `advance`, `advance_with_caveats`, and `descriptive_only`.\n"
    )


def _plot_acute_temperature_lags(results: pd.DataFrame, output_path: Path) -> None:
    figure = plt.figure(figsize=(10, 6))
    axis = figure.add_subplot(1, 1, 1)
    subset = results.loc[
        results["screening_domain"].eq("acute_temperature_reference_contrast")
        & results["headline_eligible_flag"].eq("yes")
    ].copy()
    lag_order = {lag: index for index, lag in enumerate(ACUTE_TEMPERATURE_LAG_SUFFIXES)}
    if subset.empty:
        axis.text(0.5, 0.5, "No eligible acute temperature screening results.", ha="center", va="center")
    else:
        for outcome_label, group in subset.groupby("public_concept_label", sort=True):
            group = group.sort_values("lag_or_window", key=lambda series: series.map(lag_order))
            axis.plot(
                [lag_order[str(value)] for value in group["lag_or_window"]],
                group["estimate"],
                marker="o",
                label=outcome_label,
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(list(lag_order.values()), [WINDOW_SUFFIX_LABELS[item] for item in ACUTE_TEMPERATURE_LAG_SUFFIXES], rotation=20)
        axis.set_ylabel("Rate ratio")
        axis.legend(fontsize=8)
    axis.set_title("Acute screening: temperature contrasts across short lags")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_acute_heatwave(results: pd.DataFrame, output_path: Path) -> None:
    figure = plt.figure(figsize=(10, 6))
    axis = figure.add_subplot(1, 1, 1)
    subset = results.loc[
        results["screening_domain"].eq("acute_heatwave_binary_contrast")
    ].copy()
    if subset.empty:
        axis.text(0.5, 0.5, "No acute heatwave screening results.", ha="center", va="center")
    else:
        subset = subset.sort_values("estimate", ascending=False)
        positions = np.arange(subset.shape[0])
        axis.bar(positions, subset["estimate"], color="#C65D00")
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(positions, subset["public_concept_label"], rotation=25, ha="right")
        axis.set_ylabel("Rate ratio")
    axis.set_title("Acute screening: heatwave versus non-heatwave days")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_delivery_window_screening(results: pd.DataFrame, output_path: Path) -> None:
    figure = plt.figure(figsize=(10, 6))
    axis = figure.add_subplot(1, 1, 1)
    subset = results.loc[
        results["screening_domain"].isin(
            [
                "perinatal_delivery_window_q5_vs_q1",
                "maternal_operational_window_q5_vs_q1",
            ]
        )
        & results["exposure_family_id"].eq("family_1_tmax_primary")
        & results["headline_eligible_flag"].eq("yes")
    ].copy()
    window_order = {lag: index for index, lag in enumerate(DELIVERY_WINDOW_SUFFIXES)}
    if subset.empty:
        axis.text(0.5, 0.5, "No eligible delivery-window screening results.", ha="center", va="center")
    else:
        top_outcomes = (
            subset.sort_values(["q_value", "p_value"], na_position="last")
            .drop_duplicates("public_concept_label")
            .head(6)["public_concept_label"]
            .tolist()
        )
        subset = subset.loc[subset["public_concept_label"].isin(top_outcomes)]
        for outcome_label, group in subset.groupby("public_concept_label", sort=True):
            group = group.sort_values("lag_or_window", key=lambda series: series.map(window_order))
            axis.plot(
                [window_order[str(value)] for value in group["lag_or_window"]],
                group["estimate"],
                marker="o",
                label=outcome_label,
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(list(window_order.values()), [WINDOW_SUFFIX_LABELS[item] for item in DELIVERY_WINDOW_SUFFIXES], rotation=20)
        axis.set_ylabel("Rate ratio")
        axis.legend(fontsize=8)
    axis.set_title("Delivery-window screening: quintile 5 versus quintile 1")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_triage_summary(triage: pd.DataFrame, output_path: Path) -> None:
    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(1, 1, 1)
    if triage.empty:
        axis.text(0.5, 0.5, "No triage results available.", ha="center", va="center")
    else:
        counts = (
            triage["triage_priority_classification"]
            .value_counts()
            .reindex(["advance", "advance_with_caveats", "descriptive_only"], fill_value=0)
        )
        axis.bar(counts.index, counts.values, color=["#2C7F4F", "#D98E04", "#7A7A7A"])
        axis.set_ylabel("Public concept count")
    axis.set_title("Track A screening triage summary")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _acute_screening_rows_from_grouped(
    grouped: pd.DataFrame,
    weekly_frame: pd.DataFrame,
    readiness_lookup: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    result_rows: list[dict[str, Any]] = []
    outcome_lookup = {item["outcome_variable"]: item for item in ACUTE_OUTCOMES}
    pooled_binary = pool_binary_rate_ratios(
        grouped,
        combination_columns=ACUTE_BINARY_COMBINATION_COLUMNS,
        stratum_columns=ACUTE_BINARY_STRATUM_COLUMNS,
        group_column="screen_group",
    )
    for record in pooled_binary.to_dict(orient="records"):
        row = pd.Series(record)
        outcome_spec = outcome_lookup[str(row["outcome_variable"])]
        screening_domain = str(row["screening_domain"])
        if screening_domain == "acute_temperature_reference_contrast":
            result_rows.append(
                _make_result_row(
                    readiness_lookup=readiness_lookup,
                    outcome_variable=outcome_spec["outcome_variable"],
                    outcome_label=outcome_spec["outcome_label"],
                    exposure_variable=str(row["exposure_variable"]),
                    exposure_label=_base_exposure_label("maximum_daily_temperature_c", str(row["lag_or_window"])),
                    exposure_family_id=str(row["exposure_family_id"]),
                    exposure_family_name=str(row["exposure_family_name"]),
                    lag_or_window=str(row["lag_or_window"]),
                    screening_domain=screening_domain,
                    screening_method="within_municipality_year_p95_vs_reference_rate_ratio",
                    estimate_scale="rate_ratio",
                    comparison_label="Municipality-year p95 versus municipality-year p40 to p60 reference band",
                    contrast_definition="Rows with exposure at or above the municipality-year p95 are compared with rows in the municipality-year p40 to p60 reference band.",
                    statistics=_statistics_from_frame_row(row),
                )
            )
            continue
        result_rows.append(
            _make_result_row(
                readiness_lookup=readiness_lookup,
                outcome_variable=outcome_spec["outcome_variable"],
                outcome_label=outcome_spec["outcome_label"],
                exposure_variable=str(row["exposure_variable"]),
                exposure_label=_base_exposure_label("heatwave_duration_days_p95_threshold", "lag0"),
                exposure_family_id=str(row["exposure_family_id"]),
                exposure_family_name=str(row["exposure_family_name"]),
                lag_or_window=str(row["lag_or_window"]),
                screening_domain=screening_domain,
                screening_method="within_municipality_year_heatwave_vs_non_heatwave_rate_ratio",
                estimate_scale="rate_ratio",
                comparison_label="Heatwave day versus non-heatwave day",
                contrast_definition="Rows with heatwave duration above zero are compared with rows without a heatwave signal in the same municipality-year.",
                statistics=_statistics_from_frame_row(row),
            )
        )

    if weekly_frame.empty:
        return result_rows

    for exposure_variable in ACUTE_CORRELATION_EXPOSURES:
        exposure_meta = BASE_EXPOSURES[exposure_variable]
        for outcome in ACUTE_OUTCOMES:
            statistics = _pearson_correlation(
                weekly_frame,
                x_column=f"{exposure_variable}_anomaly",
                y_column=f"{outcome['outcome_variable']}_anomaly",
            )
            result_rows.append(
                _make_result_row(
                    readiness_lookup=readiness_lookup,
                    outcome_variable=outcome["outcome_variable"],
                    outcome_label=outcome["outcome_label"],
                    exposure_variable=exposure_variable,
                    exposure_label=f"{exposure_meta['label']} (weekly anomaly)",
                    exposure_family_id=exposure_meta["family_id"],
                    exposure_family_name=exposure_meta["family_name"],
                    lag_or_window="weekly_anomaly",
                    screening_domain="acute_deseasonalized_weekly_anomaly_correlation",
                    screening_method="pearson_correlation_on_within_municipality_year_weekly_anomalies",
                    estimate_scale="correlation_coefficient",
                    comparison_label="Deseasonalized weekly anomaly correlation",
                    contrast_definition="Weekly municipality-year anomalies are correlated after centering each municipality-year series around its own mean.",
                    statistics=statistics,
                )
            )
    return result_rows


def _window_screening_rows_from_grouped(
    grouped: pd.DataFrame,
    outcomes: list[dict[str, str]],
    readiness_lookup: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    if grouped.empty:
        return []

    outcome_lookup = {item["outcome_variable"]: item for item in outcomes}
    result_rows: list[dict[str, Any]] = []

    q1_q5_input = grouped.loc[grouped["exposure_quintile"].isin([1, 5])].copy()
    q1_q5_input["screen_group"] = q1_q5_input["exposure_quintile"].map({1: "reference", 5: "exposed"})
    q1_q5_input["screening_domain"] = q1_q5_input["screening_domain_prefix"].astype(str) + "_q5_vs_q1"
    pooled_binary = pool_binary_rate_ratios(
        q1_q5_input,
        combination_columns=WINDOW_COMBINATION_COLUMNS,
        stratum_columns=WINDOW_STRATUM_COLUMNS,
        group_column="screen_group",
    )
    for record in pooled_binary.to_dict(orient="records"):
        row = pd.Series(record)
        outcome_spec = outcome_lookup[str(row["outcome_variable"])]
        base_exposure, suffix = _split_window_exposure_column(str(row["exposure_variable"]))
        result_rows.append(
            _make_result_row(
                readiness_lookup=readiness_lookup,
                outcome_variable=outcome_spec["outcome_variable"],
                outcome_label=outcome_spec["outcome_label"],
                exposure_variable=str(row["exposure_variable"]),
                exposure_label=_base_exposure_label(base_exposure, suffix),
                exposure_family_id=str(row["exposure_family_id"]),
                exposure_family_name=str(row["exposure_family_name"]),
                lag_or_window=str(row["lag_or_window"]),
                screening_domain=str(row["screening_domain"]),
                screening_method="within_municipality_year_quintile_extremes_rate_ratio",
                estimate_scale="rate_ratio",
                comparison_label="Exposure quintile 5 versus exposure quintile 1",
                contrast_definition="Rows in the highest year-specific exposure quintile are compared with rows in the lowest year-specific exposure quintile.",
                statistics=_statistics_from_frame_row(row),
            )
        )

    trend_input = grouped.copy()
    trend_input["screening_domain"] = trend_input["screening_domain_prefix"].astype(str) + "_quintile_trend"
    pooled_trends = pool_quintile_trends(
        trend_input,
        combination_columns=WINDOW_COMBINATION_COLUMNS,
        stratum_columns=WINDOW_STRATUM_COLUMNS,
    )
    for record in pooled_trends.to_dict(orient="records"):
        row = pd.Series(record)
        outcome_spec = outcome_lookup[str(row["outcome_variable"])]
        base_exposure, suffix = _split_window_exposure_column(str(row["exposure_variable"]))
        result_rows.append(
            _make_result_row(
                readiness_lookup=readiness_lookup,
                outcome_variable=outcome_spec["outcome_variable"],
                outcome_label=outcome_spec["outcome_label"],
                exposure_variable=str(row["exposure_variable"]),
                exposure_label=_base_exposure_label(base_exposure, suffix),
                exposure_family_id=str(row["exposure_family_id"]),
                exposure_family_name=str(row["exposure_family_name"]),
                lag_or_window=str(row["lag_or_window"]),
                screening_domain=str(row["screening_domain"]),
                screening_method="within_municipality_year_log_rate_quintile_trend",
                estimate_scale="rate_ratio_per_quintile",
                comparison_label="Per-quintile monotonic trend",
                contrast_definition="A weighted within-stratum log-rate trend is estimated across year-specific exposure quintiles one through five and then pooled across municipality-years.",
                statistics=_statistics_from_frame_row(row),
            )
        )

    return result_rows


def build_track_a_screening(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    total_start = time.perf_counter()
    phase_seconds: dict[str, float] = {}

    config = load_track_a_config(root, config_path=config_path)
    years = [int(year) for year in config["years"]]
    scope_key = screening_scope_key(years)
    cache_manager = ScreeningCacheManager(_screening_cache_root(root), scope_key=scope_key, stage_version=SCREENING_STAGE_VERSION)
    readiness_path = _readiness_matrix_path(root)

    readiness_start = time.perf_counter()
    readiness = _load_readiness(root)
    readiness_lookup = _readiness_lookup(readiness)
    phase_seconds["readiness_loading"] = time.perf_counter() - readiness_start

    acute_build_start = time.perf_counter()
    acute_binary_frames: list[pd.DataFrame] = []
    acute_weekly_frames: list[pd.DataFrame] = []
    for year in years:
        acute_partition_path = _mart_partition_path(root, "acute_daily_mart", year)
        acute_binary_grouped, _, _ = cache_manager.load_or_build_parquet(
            entry_name=f"acute_binary_grouped_{year}",
            input_paths=[acute_partition_path, readiness_path],
            builder=lambda year=year: _build_acute_binary_grouped_for_year(root, year),
        )
        if not acute_binary_grouped.empty:
            acute_binary_frames.append(acute_binary_grouped)
        acute_weekly, _, _ = cache_manager.load_or_build_parquet(
            entry_name=f"acute_weekly_grouped_{year}",
            input_paths=[acute_partition_path, readiness_path],
            builder=lambda year=year: _build_acute_weekly_for_year(root, year),
        )
        if not acute_weekly.empty:
            acute_weekly_frames.append(acute_weekly)
    phase_seconds["acute_grouped_summary_build"] = time.perf_counter() - acute_build_start

    window_build_start = time.perf_counter()
    perinatal_frames: list[pd.DataFrame] = []
    maternal_frames: list[pd.DataFrame] = []
    for year in years:
        perinatal_partition_path = _mart_partition_path(root, "perinatal_delivery_window_mart", year)
        perinatal_grouped, _, _ = cache_manager.load_or_build_parquet(
            entry_name=f"perinatal_window_grouped_{year}",
            input_paths=[perinatal_partition_path, readiness_path],
            builder=lambda year=year: _build_window_grouped_for_year(
                root=root,
                year=year,
                mart_name="perinatal_delivery_window_mart",
                required_columns=PERINATAL_REQUIRED_COLUMNS,
                outcomes=PERINATAL_OUTCOMES,
                screening_domain_prefix="perinatal_delivery_window",
            ),
        )
        if not perinatal_grouped.empty:
            perinatal_frames.append(perinatal_grouped)

        maternal_partition_path = _mart_partition_path(root, "maternal_operational_mart", year)
        maternal_grouped, _, _ = cache_manager.load_or_build_parquet(
            entry_name=f"maternal_window_grouped_{year}",
            input_paths=[maternal_partition_path, readiness_path],
            builder=lambda year=year: _build_window_grouped_for_year(
                root=root,
                year=year,
                mart_name="maternal_operational_mart",
                required_columns=MATERNAL_REQUIRED_COLUMNS,
                outcomes=MATERNAL_OUTCOMES,
                screening_domain_prefix="maternal_operational_window",
            ),
        )
        if not maternal_grouped.empty:
            maternal_frames.append(maternal_grouped)
    phase_seconds["window_grouped_summary_build"] = time.perf_counter() - window_build_start

    acute_pool_start = time.perf_counter()
    acute_binary_grouped = pd.concat(acute_binary_frames, ignore_index=True) if acute_binary_frames else _empty_acute_binary_grouped()
    acute_weekly_grouped = pd.concat(acute_weekly_frames, ignore_index=True) if acute_weekly_frames else _empty_weekly_acute()
    results = _acute_screening_rows_from_grouped(acute_binary_grouped, acute_weekly_grouped, readiness_lookup)
    phase_seconds["acute_pooling"] = time.perf_counter() - acute_pool_start

    window_pool_start = time.perf_counter()
    perinatal_grouped = pd.concat(perinatal_frames, ignore_index=True) if perinatal_frames else _empty_window_grouped()
    maternal_grouped = pd.concat(maternal_frames, ignore_index=True) if maternal_frames else _empty_window_grouped()
    results.extend(_window_screening_rows_from_grouped(perinatal_grouped, PERINATAL_OUTCOMES, readiness_lookup))
    results.extend(_window_screening_rows_from_grouped(maternal_grouped, MATERNAL_OUTCOMES, readiness_lookup))
    phase_seconds["window_pooling"] = time.perf_counter() - window_pool_start

    results_frame = pd.DataFrame(results)
    if results_frame.empty:
        raise ValueError("Stage 4 produced no screening results. Verify Stage 1 readiness assets and Stage 2 marts.")

    fdr_start = time.perf_counter()
    outcome_family_lookup = {
        item["outcome_variable"]: item["outcome_family"]
        for item in ACUTE_OUTCOMES + PERINATAL_OUTCOMES + MATERNAL_OUTCOMES
    }
    results_frame["outcome_family"] = results_frame["outcome_variable"].map(outcome_family_lookup).fillna("shared")
    results_frame = _apply_benjamini_hochberg(results_frame)
    results_frame = apply_probability_floor(results_frame)
    results_frame["row_priority_classification"] = results_frame.apply(_row_priority_classification, axis=1)
    triage_frame = _build_triage_signal_matrix(results_frame)
    phase_seconds["fdr_and_triage"] = time.perf_counter() - fdr_start

    screening_root = _screening_root(root)
    plots_root = screening_root / "screening_plots"
    screening_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    results_frame = results_frame.sort_values(
        [
            "row_priority_classification",
            "headline_eligible_flag",
            "benchmark_model_eligible_flag",
            "q_value",
            "p_value",
            "public_concept_label",
            "screening_domain",
        ],
        ascending=[True, False, False, True, True, True, True],
        na_position="last",
    )
    triage_frame = triage_frame.sort_values(
        ["triage_priority_classification", "minimum_q_value", "minimum_p_value", "public_concept_label"],
        ascending=[True, True, True, True],
        na_position="last",
    )

    plot_start = time.perf_counter()
    figure_paths = [
        plots_root / "figure_01_acute_temperature_lag_screening.png",
        plots_root / "figure_02_acute_heatwave_screening.png",
        plots_root / "figure_03_delivery_window_screening.png",
        plots_root / "figure_04_triage_summary.png",
    ]
    _plot_acute_temperature_lags(results_frame, figure_paths[0])
    _plot_acute_heatwave(results_frame, figure_paths[1])
    _plot_delivery_window_screening(results_frame, figure_paths[2])
    _plot_triage_summary(triage_frame, figure_paths[3])
    phase_seconds["plotting"] = time.perf_counter() - plot_start

    triage_counts = (
        triage_frame["triage_priority_classification"]
        .value_counts()
        .reindex(["advance", "advance_with_caveats", "descriptive_only"], fill_value=0)
        .to_dict()
    )
    triage_counts = {key: int(value) for key, value in triage_counts.items()}

    screening_results_path = screening_root / "screening_results.csv"
    triage_path = screening_root / "triage_signal_matrix.csv"
    methods_note_path = screening_root / "screening_methods_note.md"
    runtime_metrics_path = screening_root / "screening_runtime_metrics.json"
    runtime_report_path = screening_root / "screening_runtime_report.md"
    review_note_path = screening_root / "stage4_review_note.md"

    report_start = time.perf_counter()
    results_frame.to_csv(screening_results_path, index=False)
    triage_frame.to_csv(triage_path, index=False)
    methods_note_path.write_text(_render_methods_note(), encoding="utf-8")
    phase_seconds["report_writing"] = time.perf_counter() - report_start

    metrics_payload = {
        "scope_key": scope_key,
        "analysis_id": config["analysis_id"],
        "years": years,
        "cache_state": cache_manager.classify_run_cache_state(),
        "cache_hit_count": int(cache_manager.cache_hit_count),
        "cache_miss_count": int(cache_manager.cache_miss_count),
        "result_count": int(results_frame.shape[0]),
        "triage_public_concept_count": int(triage_frame.shape[0]),
        "triage_counts": triage_counts,
        "figure_count": len(figure_paths),
        "phase_seconds": phase_seconds,
        "runtime_seconds": 0.0,
    }

    comparison_summary = None
    if years != [2024]:
        comparison_summary = cache_manager.read_scope_json("years_2024", "run_summary.json")

    review_note_path.write_text(
        _render_stage4_review_note(years, metrics_payload, triage_frame, comparison_summary),
        encoding="utf-8",
    )
    runtime_report_path.write_text(_render_runtime_report(metrics_payload), encoding="utf-8")
    runtime_metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

    metrics_payload["runtime_seconds"] = time.perf_counter() - total_start
    metrics_payload["phase_seconds"] = {
        key: round(value, 6)
        for key, value in phase_seconds.items()
    }

    runtime_metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")
    runtime_report_path.write_text(_render_runtime_report(metrics_payload), encoding="utf-8")
    review_note_path.write_text(
        _render_stage4_review_note(years, metrics_payload, triage_frame, comparison_summary),
        encoding="utf-8",
    )

    cache_manager.write_scope_json(
        "run_summary.json",
        {
            "scope_key": scope_key,
            "runtime_seconds": metrics_payload["runtime_seconds"],
            "result_count": metrics_payload["result_count"],
            "triage_public_concept_count": metrics_payload["triage_public_concept_count"],
            "triage_counts": triage_counts,
            "triage_map": {
                str(row["public_concept_name"]): str(row["triage_priority_classification"])
                for row in triage_frame.to_dict(orient="records")
            },
        },
    )

    return {
        "status": "PASS",
        "stage": "stage4_screening_and_statistical_triage",
        "analysis_id": config["analysis_id"],
        "result_count": int(results_frame.shape[0]),
        "triage_public_concept_count": int(triage_frame.shape[0]),
        "figure_count": len(figure_paths),
        "triage_counts": triage_counts,
        "runtime_seconds": float(metrics_payload["runtime_seconds"]),
        "cache_hit_count": int(cache_manager.cache_hit_count),
        "cache_miss_count": int(cache_manager.cache_miss_count),
        "runtime_metrics_path": _relative(root, runtime_metrics_path),
        "review_note_path": _relative(root, review_note_path),
        "outputs": {
            "screening_results": _relative(root, screening_results_path),
            "triage_signal_matrix": _relative(root, triage_path),
            "screening_methods_note": _relative(root, methods_note_path),
            "screening_runtime_report": _relative(root, runtime_report_path),
            "screening_plots_directory": _relative(root, plots_root),
            "screening_cache_directory": _relative(root, _screening_cache_root(root)),
        },
    }


def dags_stage_not_implemented(root: Path) -> dict[str, object]:
    return {
        "status": "PENDING_IMPLEMENTATION",
        "stage": "stage5_dag_package_and_adjustment_architecture",
        "root": str(root),
        "message": "The Track A DAG stage is scaffolded but not implemented in Stage 0.",
    }


def screening_stage_not_implemented(root: Path) -> dict[str, object]:
    return {
        "status": "PENDING_IMPLEMENTATION",
        "stage": "stage4_screening_and_statistical_triage",
        "root": str(root),
        "message": "The Track A screening command is scaffolded but not implemented in this CLI entrypoint.",
    }


def benchmark_stage_not_implemented(root: Path) -> dict[str, object]:
    return {
        "status": "PENDING_IMPLEMENTATION",
        "stage": "stage6_progressive_benchmark_modeling",
        "root": str(root),
        "message": "The Track A benchmark stage is scaffolded but not implemented in Stage 0.",
    }


def sensitivity_stage_not_implemented(root: Path) -> dict[str, object]:
    return {
        "status": "PENDING_IMPLEMENTATION",
        "stage": "stage7_sensitivity_heterogeneity_and_vulnerability",
        "root": str(root),
        "message": "The Track A sensitivity stage is scaffolded but not implemented in Stage 0.",
    }


def synthesis_stage_not_implemented(root: Path) -> dict[str, object]:
    return {
        "status": "PENDING_IMPLEMENTATION",
        "stage": "stage8_framework_synthesis_and_indicator_scorecard",
        "root": str(root),
        "message": "The Track A synthesis stage is scaffolded but not implemented in Stage 0.",
    }
