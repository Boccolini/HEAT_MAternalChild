from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    import duckdb
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    duckdb = None

from .config import load_round_config
from .registry import (
    analysis_shared_marts_root,
    analysis_step2_root,
    analysis_step3_root,
    ensure_step3_directory_structure,
    incremental_publication_root,
)
from .step2 import build_step2_baseline_models


STEP3_PUBLICATION_FILES = {
    "06_step3_threshold_screening.md": "# Step 3 Threshold Screening\n\nRecord threshold-screening interpretations here.\n",
    "07_step3_exposure_family_decision.md": "# Step 3 Exposure Family Decision\n\nDocument tmax versus heat index headline decisions here.\n",
}

PROBABILITY_FLOOR = 1e-300
STEP3_NAME = "step3_threshold_discovery"
SUPPORT_GRADE_RANK = {"D": 0, "C": 1, "B": 2, "A": 3}
DUCKDB_FAST_PATH_AVAILABLE = duckdb is not None
P_VALUE_TIE_TOLERANCE = 1e-12

SCREENING_PROFILE_SPECS = {
    "acute_daily_standard": {
        "kind": "acute",
        "mart_name": "acute_daily_mart",
        "family_exposures": {
            "family_1_tmax_primary": [
                ("maximum_daily_temperature_c_lag0", "Maximum daily temperature (lag 0)", "lag0"),
                ("maximum_daily_temperature_c_lag0_1_mean", "Maximum daily temperature (lag 0 to 1 mean)", "lag0_1"),
                ("maximum_daily_temperature_c_lag0_3_mean", "Maximum daily temperature (lag 0 to 3 mean)", "lag0_3"),
                ("maximum_daily_temperature_c_lag0_7_mean", "Maximum daily temperature (lag 0 to 7 mean)", "lag0_7"),
            ],
            "family_4_heat_index_parallel": [
                ("approximate_heat_index_c_lag0", "Approximate heat index (lag 0)", "lag0"),
                ("approximate_heat_index_c_lag0_1_mean", "Approximate heat index (lag 0 to 1 mean)", "lag0_1"),
                ("approximate_heat_index_c_lag0_3_mean", "Approximate heat index (lag 0 to 3 mean)", "lag0_3"),
                ("approximate_heat_index_c_lag0_7_mean", "Approximate heat index (lag 0 to 7 mean)", "lag0_7"),
            ],
        },
    },
    "neonatal_acute_with_epiweek": {
        "kind": "acute",
        "mart_name": "acute_daily_mart",
        "family_exposures": {
            "family_1_tmax_primary": [
                ("maximum_daily_temperature_c_lag0", "Maximum daily temperature (lag 0)", "lag0"),
                ("maximum_daily_temperature_c_lag0_1_mean", "Maximum daily temperature (lag 0 to 1 mean)", "lag0_1"),
                ("maximum_daily_temperature_c_lag0_3_mean", "Maximum daily temperature (lag 0 to 3 mean)", "lag0_3"),
                ("maximum_daily_temperature_c_lag0_7_mean", "Maximum daily temperature (lag 0 to 7 mean)", "lag0_7"),
            ],
            "family_4_heat_index_parallel": [
                ("approximate_heat_index_c_lag0", "Approximate heat index (lag 0)", "lag0"),
                ("approximate_heat_index_c_lag0_1_mean", "Approximate heat index (lag 0 to 1 mean)", "lag0_1"),
                ("approximate_heat_index_c_lag0_3_mean", "Approximate heat index (lag 0 to 3 mean)", "lag0_3"),
                ("approximate_heat_index_c_lag0_7_mean", "Approximate heat index (lag 0 to 7 mean)", "lag0_7"),
            ],
        },
    },
    "maternal_daily_standard": {
        "kind": "window",
        "mart_name": "maternal_operational_mart",
        "screening_domain_prefix": "maternal_operational_window",
        "family_exposures": {
            "family_1_tmax_primary": [
                ("maximum_daily_temperature_c_days_0_6_before_event_mean", "Maximum daily temperature (days 0 to 6 before event mean)", "0_6"),
                ("maximum_daily_temperature_c_days_0_13_before_event_mean", "Maximum daily temperature (days 0 to 13 before event mean)", "0_13"),
                ("maximum_daily_temperature_c_days_0_27_before_event_mean", "Maximum daily temperature (days 0 to 27 before event mean)", "0_27"),
            ],
            "family_4_heat_index_parallel": [
                ("approximate_heat_index_c_days_0_6_before_event_mean", "Approximate heat index (days 0 to 6 before event mean)", "0_6"),
                ("approximate_heat_index_c_days_0_13_before_event_mean", "Approximate heat index (days 0 to 13 before event mean)", "0_13"),
                ("approximate_heat_index_c_days_0_27_before_event_mean", "Approximate heat index (days 0 to 27 before event mean)", "0_27"),
            ],
        },
    },
    "perinatal_delivery_aligned": {
        "kind": "window",
        "mart_name": "perinatal_delivery_window_mart",
        "screening_domain_prefix": "perinatal_delivery_window",
        "family_exposures": {
            "family_1_tmax_primary": [
                ("maximum_daily_temperature_c_days_0_6_before_event_mean", "Maximum daily temperature (days 0 to 6 before event mean)", "0_6"),
                ("maximum_daily_temperature_c_days_0_13_before_event_mean", "Maximum daily temperature (days 0 to 13 before event mean)", "0_13"),
                ("maximum_daily_temperature_c_days_0_27_before_event_mean", "Maximum daily temperature (days 0 to 27 before event mean)", "0_27"),
            ],
            "family_4_heat_index_parallel": [
                ("approximate_heat_index_c_days_0_6_before_event_mean", "Approximate heat index (days 0 to 6 before event mean)", "0_6"),
                ("approximate_heat_index_c_days_0_13_before_event_mean", "Approximate heat index (days 0 to 13 before event mean)", "0_13"),
                ("approximate_heat_index_c_days_0_27_before_event_mean", "Approximate heat index (days 0 to 27 before event mean)", "0_27"),
            ],
        },
    },
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _scaffold_step3_publication_workspace(root: Path) -> dict[str, object]:
    workspace_root = incremental_publication_root(root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    created_files: list[str] = []
    existing_files: list[str] = []
    for relative_path, contents in STEP3_PUBLICATION_FILES.items():
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


def _normal_two_sided_p_from_z(z_value: float) -> tuple[float, str]:
    if not math.isfinite(z_value):
        return float("nan"), "no"
    probability = float(math.erfc(abs(z_value) / math.sqrt(2.0)))
    if probability < PROBABILITY_FLOOR:
        return PROBABILITY_FLOOR, "yes"
    return probability, "no"


def _baseline_valid_rows(frame: pd.DataFrame, denominator_column: str, exposure_column: str) -> pd.DataFrame:
    return frame.loc[
        frame["quality_control_status"].astype(str).eq("PASS")
        & frame["offset_valid_flag"].eq(1)
        & frame["denominator_zero_flag"].eq(0)
        & frame["denominator_missing_flag"].eq(0)
        & frame["rate_not_computable_flag"].eq(0)
        & frame[denominator_column].fillna(0).gt(0)
        & frame[exposure_column].notna()
    ].copy()


def _estimate_direction(value: Any, estimate_scale: str) -> str:
    numeric = float(value) if value is not None and not pd.isna(value) else float("nan")
    if not math.isfinite(numeric):
        return ""
    if estimate_scale == "mean_difference_per_interquartile_range_increase":
        if numeric > 0:
            return "positive"
        if numeric < 0:
            return "negative"
        return "null"
    if numeric > 1:
        return "positive"
    if numeric < 1:
        return "negative"
    return "null"


def _pool_binary_summary(summary: pd.DataFrame, model_kind: str) -> dict[str, Any]:
    if summary.empty:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "p_value_floor_applied_flag": "no",
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
            "p_value_floor_applied_flag": "no",
            "valid_strata_count": 0,
            "exposed_row_count": int(summary["row_count_exposed"].sum()),
            "reference_row_count": int(summary["row_count_reference"].sum()),
            "exposed_outcome_sum": float(summary["outcome_sum_exposed"].sum()),
            "reference_outcome_sum": float(summary["outcome_sum_reference"].sum()),
            "exposed_denominator_sum": float(summary["denominator_sum_exposed"].sum()),
            "reference_denominator_sum": float(summary["denominator_sum_reference"].sum()),
        }

    if model_kind == "count":
        exposed_cases = valid["outcome_sum_exposed"].astype(float) + 0.5
        reference_cases = valid["outcome_sum_reference"].astype(float) + 0.5
        exposed_denominator = valid["denominator_sum_exposed"].astype(float)
        reference_denominator = valid["denominator_sum_reference"].astype(float)
        effect = np.log(exposed_cases / exposed_denominator) - np.log(reference_cases / reference_denominator)
        variance = (1.0 / exposed_cases) + (1.0 / reference_cases)
        weights = 1.0 / variance
        pooled_effect = float(np.average(effect, weights=weights))
        pooled_se = float(np.sqrt(1.0 / weights.sum()))
        z_value = pooled_effect / pooled_se if pooled_se > 0 else float("nan")
        p_value, p_floor_flag = _normal_two_sided_p_from_z(z_value)
        lower_ci = pooled_effect - 1.96 * pooled_se
        upper_ci = pooled_effect + 1.96 * pooled_se
        estimate = math.exp(pooled_effect)
        lower = math.exp(lower_ci)
        upper = math.exp(upper_ci)
    else:
        exposed_mean = valid["outcome_sum_exposed"].astype(float) / valid["denominator_sum_exposed"].astype(float)
        reference_mean = valid["outcome_sum_reference"].astype(float) / valid["denominator_sum_reference"].astype(float)
        effect = exposed_mean - reference_mean
        variance = (1.0 / valid["denominator_sum_exposed"].astype(float)) + (1.0 / valid["denominator_sum_reference"].astype(float))
        weights = 1.0 / variance
        pooled_effect = float(np.average(effect, weights=weights))
        pooled_se = float(np.sqrt(1.0 / weights.sum()))
        z_value = pooled_effect / pooled_se if pooled_se > 0 else float("nan")
        p_value, p_floor_flag = _normal_two_sided_p_from_z(z_value)
        estimate = pooled_effect
        lower = pooled_effect - 1.96 * pooled_se
        upper = pooled_effect + 1.96 * pooled_se

    return {
        "estimate": float(estimate),
        "lower_ci": float(lower),
        "upper_ci": float(upper),
        "p_value": float(p_value),
        "p_value_floor_applied_flag": p_floor_flag,
        "valid_strata_count": int(valid.shape[0]),
        "exposed_row_count": int(valid["row_count_exposed"].sum()),
        "reference_row_count": int(valid["row_count_reference"].sum()),
        "exposed_outcome_sum": float(valid["outcome_sum_exposed"].sum()),
        "reference_outcome_sum": float(valid["outcome_sum_reference"].sum()),
        "exposed_denominator_sum": float(valid["denominator_sum_exposed"].sum()),
        "reference_denominator_sum": float(valid["denominator_sum_reference"].sum()),
    }


def _assign_quintiles(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(copy=False)
    valid_mask = np.isfinite(values)
    quintiles = np.full(values.shape, np.nan, dtype=float)
    if valid_mask.sum() < 5:
        return pd.Series(quintiles, index=series.index, dtype="Float64")
    thresholds = np.quantile(values[valid_mask], [0.2, 0.4, 0.6, 0.8])
    quintiles[valid_mask] = np.searchsorted(thresholds, values[valid_mask], side="right") + 1
    return pd.Series(quintiles, index=series.index, dtype="Float64")


def _pool_quintile_trend(summary: pd.DataFrame, model_kind: str) -> dict[str, Any]:
    if summary.empty:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "p_value_floor_applied_flag": "no",
            "valid_strata_count": 0,
            "exposed_row_count": 0,
            "reference_row_count": 0,
            "exposed_outcome_sum": 0.0,
            "reference_outcome_sum": 0.0,
            "exposed_denominator_sum": 0.0,
            "reference_denominator_sum": 0.0,
        }

    effect_rows: list[dict[str, float]] = []
    for _, group in summary.groupby(["stratum_id"], sort=False):
        valid = group.loc[group["denominator_sum"].gt(0) & group["row_count"].gt(0)].copy()
        if valid.shape[0] < 3 or valid["exposure_quintile"].nunique() < 3:
            continue
        x = valid["exposure_quintile"].astype(float).to_numpy()
        if model_kind == "count":
            y = np.log((valid["outcome_sum"].astype(float).to_numpy() + 0.5) / valid["denominator_sum"].astype(float).to_numpy())
        else:
            y = valid["outcome_sum"].astype(float).to_numpy() / valid["denominator_sum"].astype(float).to_numpy()
        weights = valid["denominator_sum"].astype(float).to_numpy()
        design = np.column_stack([np.ones_like(x), x])
        xtwx = design.T @ (weights[:, None] * design)
        try:
            xtwx_inv = np.linalg.inv(xtwx)
        except np.linalg.LinAlgError:
            continue
        beta = xtwx_inv @ (design.T @ (weights * y))
        residual = y - (design @ beta)
        degrees_of_freedom = valid.shape[0] - 2
        if degrees_of_freedom <= 0:
            continue
        sigma_squared = float((weights * residual * residual).sum() / degrees_of_freedom)
        covariance = sigma_squared * xtwx_inv
        slope = float(beta[1])
        slope_se = float(math.sqrt(covariance[1, 1])) if covariance[1, 1] > 0 else float("nan")
        if not math.isfinite(slope_se) or slope_se <= 0:
            continue
        effect_rows.append({"slope": slope, "slope_se": slope_se})

    if not effect_rows:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "p_value_floor_applied_flag": "no",
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
    p_value, p_floor_flag = _normal_two_sided_p_from_z(z_value)
    lower_ci = pooled_slope - 1.96 * pooled_se
    upper_ci = pooled_slope + 1.96 * pooled_se

    q1 = summary.loc[summary["exposure_quintile"].eq(1)]
    q5 = summary.loc[summary["exposure_quintile"].eq(5)]
    if model_kind == "count":
        estimate = math.exp(pooled_slope)
        lower = math.exp(lower_ci)
        upper = math.exp(upper_ci)
    else:
        estimate = pooled_slope
        lower = lower_ci
        upper = upper_ci
    return {
        "estimate": float(estimate),
        "lower_ci": float(lower),
        "upper_ci": float(upper),
        "p_value": float(p_value),
        "p_value_floor_applied_flag": p_floor_flag,
        "valid_strata_count": int(effects.shape[0]),
        "exposed_row_count": int(q5["row_count"].sum()),
        "reference_row_count": int(q1["row_count"].sum()),
        "exposed_outcome_sum": float(q5["outcome_sum"].sum()),
        "reference_outcome_sum": float(q1["outcome_sum"].sum()),
        "exposed_denominator_sum": float(q5["denominator_sum"].sum()),
        "reference_denominator_sum": float(q1["denominator_sum"].sum()),
    }


def _pearson_correlation(valid: pd.DataFrame, x_column: str, y_column: str) -> dict[str, Any]:
    subset = valid.loc[valid[x_column].notna() & valid[y_column].notna()].copy()
    if subset.shape[0] < 4:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "p_value_floor_applied_flag": "no",
            "valid_strata_count": 0,
            "exposed_row_count": 0,
            "reference_row_count": 0,
            "exposed_outcome_sum": 0.0,
            "reference_outcome_sum": 0.0,
            "exposed_denominator_sum": 0.0,
            "reference_denominator_sum": 0.0,
        }
    x = subset[x_column].astype(float).to_numpy()
    y = subset[y_column].astype(float).to_numpy()
    correlation = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(correlation):
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "p_value_floor_applied_flag": "no",
            "valid_strata_count": 0,
            "exposed_row_count": 0,
            "reference_row_count": 0,
            "exposed_outcome_sum": 0.0,
            "reference_outcome_sum": 0.0,
            "exposed_denominator_sum": 0.0,
            "reference_denominator_sum": 0.0,
        }
    z = np.arctanh(np.clip(correlation, -0.999999, 0.999999))
    se = 1.0 / math.sqrt(subset.shape[0] - 3) if subset.shape[0] > 3 else float("nan")
    z_value = z / se if se and math.isfinite(se) and se > 0 else float("nan")
    p_value, p_floor_flag = _normal_two_sided_p_from_z(z_value)
    lower_z = z - 1.96 * se
    upper_z = z + 1.96 * se
    return {
        "estimate": correlation,
        "lower_ci": float(np.tanh(lower_z)),
        "upper_ci": float(np.tanh(upper_z)),
        "p_value": float(p_value),
        "p_value_floor_applied_flag": p_floor_flag,
        "valid_strata_count": int(subset["stratum_id"].nunique()) if "stratum_id" in subset.columns else int(subset[["municipality_code", "calendar_year"]].drop_duplicates().shape[0]),
        "exposed_row_count": int(subset.shape[0]),
        "reference_row_count": 0,
        "exposed_outcome_sum": float(subset[y_column].sum()),
        "reference_outcome_sum": 0.0,
        "exposed_denominator_sum": 0.0,
        "reference_denominator_sum": 0.0,
    }


def _load_step2_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    step2_root = analysis_step2_root(root)
    candidate_roster = pd.read_csv(step2_root / "step2_candidate_roster.csv")
    promotion_decisions = pd.read_csv(step2_root / "step2_promotion_decisions.csv")
    preferred_models = pd.read_csv(step2_root / "preferred_baseline_models.csv")
    return candidate_roster, promotion_decisions, preferred_models


def _eligible_step3_pairs(
    candidate_roster: pd.DataFrame,
    promotion_decisions: pd.DataFrame,
    preferred_models: pd.DataFrame,
) -> pd.DataFrame:
    if candidate_roster.empty or promotion_decisions.empty or preferred_models.empty:
        return pd.DataFrame()
    approved_candidates = promotion_decisions.loc[
        promotion_decisions["approved_for_next_steps_flag"].eq(True),
        ["analysis_unit_id", "candidate_track"],
    ].drop_duplicates()
    preferred_pairs = preferred_models.loc[
        preferred_models["estimate"].apply(lambda value: math.isfinite(float(value)))
        & preferred_models["valid_strata_count"].fillna(0).gt(0),
        ["analysis_unit_id", "candidate_track", "exposure_family_id"],
    ].drop_duplicates()
    eligible = candidate_roster.merge(
        approved_candidates,
        on=["analysis_unit_id", "candidate_track"],
        how="inner",
    ).merge(
        preferred_pairs,
        on=["analysis_unit_id", "candidate_track", "exposure_family_id"],
        how="inner",
    )
    return eligible.sort_values(
        ["candidate_track", "analysis_role", "analysis_unit_id", "exposure_family_id"],
        ignore_index=True,
    )


def _required_columns_by_mart(candidate_roster: pd.DataFrame) -> dict[str, set[str]]:
    required: dict[str, set[str]] = {}
    for mart_name, group in candidate_roster.groupby("mart_name", sort=True):
        required[str(mart_name)] = {
            "municipality_code",
            "observation_date",
            "calendar_year",
            "calendar_month",
            "quality_control_status",
            "offset_valid_flag",
            "denominator_zero_flag",
            "denominator_missing_flag",
            "rate_not_computable_flag",
        }
        for record in group.to_dict(orient="records"):
            required[str(mart_name)].add(str(record["analysis_unit_id"]))
            denominator_column = str(record.get("denominator_column_for_step2") or "")
            if denominator_column:
                required[str(mart_name)].add(denominator_column)
            profile_spec = SCREENING_PROFILE_SPECS.get(str(record["seasonality_profile_id"]), {})
            family_exposures = profile_spec.get("family_exposures", {}).get(str(record["exposure_family_id"]), [])
            for exposure_column, _, _ in family_exposures:
                required[str(mart_name)].add(str(exposure_column))

        mart_profile = candidate_roster.loc[candidate_roster["mart_name"].astype(str).eq(mart_name), "seasonality_profile_id"].astype(str)
        if mart_profile.isin(["acute_daily_standard", "neonatal_acute_with_epiweek"]).any():
            required[str(mart_name)].add("calendar_month")
    return required


def _load_mart_frame(root: Path, years: list[int], mart_name: str, required_columns: set[str]) -> pd.DataFrame:
    partitions: list[pd.DataFrame] = []
    selected_columns = sorted(required_columns)
    for year in years:
        path = analysis_shared_marts_root(root) / mart_name / f"YEAR={year}" / f"{mart_name}_{year}.parquet"
        partition = pd.read_parquet(path, columns=selected_columns)
        partition["observation_date"] = pd.to_datetime(partition["observation_date"], errors="coerce")
        partition["calendar_month"] = pd.to_numeric(partition["calendar_month"], errors="coerce").astype("Int64")
        partition["day_of_week"] = partition["observation_date"].dt.dayofweek.astype("Int64")
        partition["epidemiologic_week"] = partition["observation_date"].dt.isocalendar().week.astype("Int64")
        partitions.append(partition)
    return pd.concat(partitions, ignore_index=True)


def _mart_parquet_paths(root: Path, years: list[int], mart_name: str) -> list[str]:
    return [
        str(analysis_shared_marts_root(root) / mart_name / f"YEAR={year}" / f"{mart_name}_{year}.parquet")
        for year in years
    ]


def _sql_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _parquet_read_sql(paths: list[str]) -> str:
    path_list = "[" + ",".join(repr(path) for path in paths) + "]"
    return f"read_parquet({path_list}, union_by_name=true)"


def _duckdb_connection() -> Any:
    if duckdb is None:  # pragma: no cover - protected by caller
        raise RuntimeError("duckdb fast path requested but duckdb is unavailable")
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=8")
    return connection


def _duckdb_acute_screening_bundle(
    parquet_paths: list[str],
    exposure_column: str,
    denominator_column: str,
    outcome_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcome_select = ", ".join(_sql_identifier(column) for column in outcome_columns)
    tagged_outcome_select = ", ".join(f"b.{_sql_identifier(column)}" for column in outcome_columns)
    outcome_sum_sql = ", ".join(f"sum({_sql_identifier(column)}) as {_sql_identifier(column)}" for column in outcome_columns)
    outcome_passthrough_sql = ", ".join(_sql_identifier(column) for column in outcome_columns)
    connection = _duckdb_connection()
    query = f"""
with base as (
    select
        municipality_code,
        calendar_year,
        week(observation_date) as epidemiologic_week,
        {_sql_identifier(denominator_column)} as denominator_value,
        {_sql_identifier(exposure_column)} as exposure_value,
        {outcome_select}
    from {_parquet_read_sql(parquet_paths)}
    where quality_control_status = 'PASS'
      and offset_valid_flag = 1
      and denominator_zero_flag = 0
      and denominator_missing_flag = 0
      and rate_not_computable_flag = 0
      and {_sql_identifier(denominator_column)} > 0
      and {_sql_identifier(exposure_column)} is not null
),
quantiles as (
    select
        municipality_code,
        calendar_year,
        quantile_cont(exposure_value, 0.40) as p40,
        quantile_cont(exposure_value, 0.60) as p60,
        quantile_cont(exposure_value, 0.95) as p95
    from base
    group by 1, 2
),
tagged as (
    select
        dense_rank() over(order by b.municipality_code, b.calendar_year) as stratum_id,
        b.epidemiologic_week,
        b.denominator_value,
        b.exposure_value,
        case
            when b.exposure_value >= q.p95 then 'exposed'
            when b.exposure_value between q.p40 and q.p60 then 'reference'
            else 'other'
        end as screen_group,
        {tagged_outcome_select}
    from base b
    join quantiles q
      on b.municipality_code = q.municipality_code
     and b.calendar_year = q.calendar_year
),
binary_summary as (
    select
        stratum_id,
        screen_group,
        sum(denominator_value) as denominator_sum,
        count(*) as row_count,
        {outcome_sum_sql}
    from tagged
    where screen_group in ('exposed', 'reference')
    group by 1, 2
),
weekly_exposure as (
    select
        stratum_id,
        epidemiologic_week,
        avg(exposure_value) as exposure_value
    from tagged
    group by 1, 2
),
weekly_exposure_anomaly as (
    select
        stratum_id,
        epidemiologic_week,
        exposure_value - avg(exposure_value) over(partition by stratum_id) as exposure_anomaly
    from weekly_exposure
),
weekly_outcomes as (
    select
        stratum_id,
        epidemiologic_week,
        sum(denominator_value) as denominator_sum,
        {outcome_sum_sql}
    from tagged
    group by 1, 2
),
weekly_joined as (
    select
        w.stratum_id,
        w.epidemiologic_week,
        w.denominator_sum,
        e.exposure_anomaly,
        {", ".join(f"w.{_sql_identifier(column)}" for column in outcome_columns)}
    from weekly_outcomes w
    join weekly_exposure_anomaly e
      on w.stratum_id = e.stratum_id
     and w.epidemiologic_week = e.epidemiologic_week
    where w.denominator_sum > 0
)
select
    'binary' as table_section,
    stratum_id,
    screen_group,
    cast(null as bigint) as epidemiologic_week,
    denominator_sum,
    row_count,
    cast(null as double) as exposure_anomaly,
    {outcome_passthrough_sql}
from binary_summary
union all
select
    'weekly' as table_section,
    stratum_id,
    cast(null as varchar) as screen_group,
    epidemiologic_week,
    denominator_sum,
    cast(null as bigint) as row_count,
    exposure_anomaly,
    {outcome_passthrough_sql}
from weekly_joined
"""
    bundle = connection.execute(query).fetchdf()
    connection.close()
    binary = bundle.loc[bundle["table_section"].astype(str).eq("binary")].drop(columns=["table_section"]).reset_index(drop=True)
    weekly = bundle.loc[bundle["table_section"].astype(str).eq("weekly")].drop(columns=["table_section"]).reset_index(drop=True)
    return binary, weekly


def _duckdb_window_quintile_table(
    parquet_paths: list[str],
    exposure_column: str,
    denominator_column: str,
    outcome_columns: list[str],
) -> pd.DataFrame:
    outcome_select = ", ".join(_sql_identifier(column) for column in outcome_columns)
    tagged_outcome_select = ", ".join(f"b.{_sql_identifier(column)}" for column in outcome_columns)
    outcome_sum_sql = ", ".join(f"sum({_sql_identifier(column)}) as {_sql_identifier(column)}" for column in outcome_columns)
    connection = _duckdb_connection()
    query = f"""
with base as (
    select
        municipality_code,
        calendar_year,
        {_sql_identifier(denominator_column)} as denominator_value,
        {_sql_identifier(exposure_column)} as exposure_value,
        {outcome_select}
    from {_parquet_read_sql(parquet_paths)}
    where quality_control_status = 'PASS'
      and offset_valid_flag = 1
      and denominator_zero_flag = 0
      and denominator_missing_flag = 0
      and rate_not_computable_flag = 0
      and {_sql_identifier(denominator_column)} > 0
      and {_sql_identifier(exposure_column)} is not null
),
quantiles as (
    select
        municipality_code,
        calendar_year,
        count(*) as row_count_in_stratum,
        quantile_cont(exposure_value, 0.20) as q20,
        quantile_cont(exposure_value, 0.40) as q40,
        quantile_cont(exposure_value, 0.60) as q60,
        quantile_cont(exposure_value, 0.80) as q80
    from base
    group by 1, 2
),
tagged as (
    select
        dense_rank() over(order by b.municipality_code, b.calendar_year) as stratum_id,
        case
            when q.row_count_in_stratum < 5 then null
            else 1
                + case when b.exposure_value >= q.q20 then 1 else 0 end
                + case when b.exposure_value >= q.q40 then 1 else 0 end
                + case when b.exposure_value >= q.q60 then 1 else 0 end
                + case when b.exposure_value >= q.q80 then 1 else 0 end
        end as exposure_quintile,
        b.denominator_value,
        {tagged_outcome_select}
    from base b
    join quantiles q
      on b.municipality_code = q.municipality_code
     and b.calendar_year = q.calendar_year
)
select
    stratum_id,
    exposure_quintile,
    sum(denominator_value) as denominator_sum,
    count(*) as row_count,
    {outcome_sum_sql}
from tagged
where exposure_quintile is not null
group by 1, 2
"""
    table = connection.execute(query).fetchdf()
    connection.close()
    return table


def _prepare_acute_screening_inputs(
    frame: pd.DataFrame,
    exposure_column: str,
    denominator_column: str,
) -> dict[str, pd.DataFrame]:
    valid = _baseline_valid_rows(frame, denominator_column, exposure_column)
    if valid.empty:
        return {
            "valid_rows": pd.DataFrame(),
            "screen_subset": pd.DataFrame(),
            "weekly_exposure": pd.DataFrame(),
        }
    valid["stratum_id"] = pd.factorize(
        pd.MultiIndex.from_frame(valid.loc[:, ["municipality_code", "calendar_year"]]),
        sort=False,
    )[0]
    quantiles = (
        valid.groupby("stratum_id", sort=False)[exposure_column]
        .quantile([0.40, 0.60, 0.95])
        .unstack()
        .rename(columns={0.40: "p40", 0.60: "p60", 0.95: "p95"})
    )
    valid = valid.join(quantiles, on="stratum_id")
    valid["screen_group"] = np.where(
        valid[exposure_column].astype(float) >= valid["p95"].astype(float),
        "exposed",
        np.where(
            valid[exposure_column].astype(float).between(valid["p40"].astype(float), valid["p60"].astype(float), inclusive="both"),
            "reference",
            "other",
        ),
    )
    screen_subset = valid.loc[
        valid["screen_group"].isin(["exposed", "reference"]),
        ["stratum_id", denominator_column, "screen_group"],
    ].copy()
    weekly_exposure = (
        valid.groupby(["stratum_id", "epidemiologic_week"], as_index=False)
        .agg(exposure_value=(exposure_column, "mean"))
    )
    weekly_exposure["exposure_anomaly"] = (
        weekly_exposure["exposure_value"]
        - weekly_exposure.groupby("stratum_id", sort=False)["exposure_value"].transform("mean")
    )
    return {
        "valid_rows": valid.loc[:, ["stratum_id", "epidemiologic_week", denominator_column]].copy(),
        "screen_subset": screen_subset,
        "weekly_exposure": weekly_exposure,
    }


def _acute_binary_summary_from_prepared(
    prepared: pd.DataFrame,
    outcome_array: np.ndarray,
    denominator_column: str,
) -> pd.DataFrame:
    if prepared.empty:
        return pd.DataFrame()
    subset = prepared.copy()
    subset["outcome_value"] = outcome_array[subset.index.to_numpy()]
    grouped_summary = (
        subset.groupby(["stratum_id", "screen_group"], as_index=False)
        .agg(
            outcome_sum=("outcome_value", "sum"),
            denominator_sum=(denominator_column, "sum"),
            row_count=("screen_group", "size"),
        )
    )
    pivot = grouped_summary.pivot_table(
        index=["stratum_id"],
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


def _acute_weekly_anomalies_from_prepared(
    prepared: pd.DataFrame,
    weekly_exposure: pd.DataFrame,
    outcome_array: np.ndarray,
    denominator_column: str,
    model_kind: str,
) -> pd.DataFrame:
    if prepared.empty or weekly_exposure.empty:
        return pd.DataFrame()
    weekly = (
        prepared.assign(
            outcome_value=outcome_array[prepared.index.to_numpy()]
        )
        .groupby(["stratum_id", "epidemiologic_week"], as_index=False)
        .agg(
            outcome_sum=("outcome_value", "sum"),
            denominator_sum=(denominator_column, "sum"),
        )
    )
    weekly = weekly.loc[weekly["denominator_sum"].gt(0)].copy()
    if weekly.empty:
        return pd.DataFrame()
    weekly = weekly.merge(
        weekly_exposure,
        on=["stratum_id", "epidemiologic_week"],
        how="inner",
        validate="one_to_one",
    )
    if weekly.empty:
        return pd.DataFrame()
    if model_kind == "count":
        weekly["outcome_metric"] = np.log((weekly["outcome_sum"].astype(float) + 0.5) / weekly["denominator_sum"].astype(float))
    else:
        weekly["outcome_metric"] = weekly["outcome_sum"].astype(float) / weekly["denominator_sum"].astype(float)
    weekly["outcome_anomaly"] = weekly["outcome_metric"] - weekly.groupby("stratum_id")["outcome_metric"].transform("mean")
    return weekly


def _prepare_window_quintiles(
    frame: pd.DataFrame,
    exposure_column: str,
    denominator_column: str,
) -> pd.DataFrame:
    valid = _baseline_valid_rows(frame, denominator_column, exposure_column)
    if valid.empty:
        return pd.DataFrame()
    valid["stratum_id"] = pd.factorize(
        pd.MultiIndex.from_frame(valid.loc[:, ["municipality_code", "calendar_year"]]),
        sort=False,
    )[0]
    quantiles = (
        valid.groupby("stratum_id", sort=False)[exposure_column]
        .quantile([0.20, 0.40, 0.60, 0.80])
        .unstack()
        .rename(columns={0.20: "q20", 0.40: "q40", 0.60: "q60", 0.80: "q80"})
    )
    valid = valid.join(quantiles, on="stratum_id")
    group_size = valid.groupby("stratum_id", sort=False)[exposure_column].transform("size")
    exposure_values = pd.to_numeric(valid[exposure_column], errors="coerce")
    valid["exposure_quintile"] = np.where(
        group_size.to_numpy() >= 5,
        1
        + (exposure_values.to_numpy() >= valid["q20"].to_numpy()).astype(int)
        + (exposure_values.to_numpy() >= valid["q40"].to_numpy()).astype(int)
        + (exposure_values.to_numpy() >= valid["q60"].to_numpy()).astype(int)
        + (exposure_values.to_numpy() >= valid["q80"].to_numpy()).astype(int),
        np.nan,
    )
    valid = valid.loc[valid["exposure_quintile"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["exposure_quintile"] = valid["exposure_quintile"].astype(int)
    return valid.loc[:, ["stratum_id", denominator_column, "exposure_quintile"]].copy()


def _window_quintile_summary_from_prepared(
    prepared: pd.DataFrame,
    outcome_array: np.ndarray,
    denominator_column: str,
) -> pd.DataFrame:
    if prepared.empty:
        return pd.DataFrame()
    return (
        prepared.assign(
            outcome_value=outcome_array[prepared.index.to_numpy()]
        )
        .groupby(["stratum_id", "exposure_quintile"], as_index=False)
        .agg(
            outcome_sum=("outcome_value", "sum"),
            denominator_sum=(denominator_column, "sum"),
            row_count=("exposure_quintile", "size"),
        )
    )


def _q1_q5_summary(summary: pd.DataFrame) -> pd.DataFrame:
    subset = summary.loc[summary["exposure_quintile"].isin([1, 5])].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["screen_group"] = subset["exposure_quintile"].map({1: "reference", 5: "exposed"})
    pivot = subset.pivot_table(
        index=["stratum_id"],
        columns="screen_group",
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


def _numeric_outcome_block(frame: pd.DataFrame, row_index: np.ndarray, outcome_columns: list[str]) -> pd.DataFrame:
    return frame.iloc[row_index][outcome_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _acute_binary_summaries_for_outcomes(
    prepared: pd.DataFrame,
    frame: pd.DataFrame,
    outcome_columns: list[str],
    denominator_column: str,
) -> dict[str, pd.DataFrame]:
    if prepared.empty or not outcome_columns:
        return {}
    outcome_block = _numeric_outcome_block(frame, prepared.index.to_numpy(), outcome_columns)
    subset = pd.concat([prepared.reset_index(drop=True), outcome_block.reset_index(drop=True)], axis=1)
    aggregate_map: dict[str, tuple[str, str]] = {column: (column, "sum") for column in outcome_columns}
    grouped = subset.groupby(["stratum_id", "screen_group"], as_index=False).agg(
        denominator_sum=(denominator_column, "sum"),
        row_count=("screen_group", "size"),
        **aggregate_map,
    )
    base = grouped.pivot(index="stratum_id", columns="screen_group", values=["denominator_sum", "row_count"])
    base.columns = [f"{metric}_{group}" for metric, group in base.columns.to_flat_index()]
    base = base.reset_index()
    for column in [
        "denominator_sum_exposed",
        "denominator_sum_reference",
        "row_count_exposed",
        "row_count_reference",
    ]:
        if column not in base.columns:
            base[column] = 0.0
    results: dict[str, pd.DataFrame] = {}
    for outcome_column in outcome_columns:
        outcome_pivot = grouped.pivot(index="stratum_id", columns="screen_group", values=outcome_column)
        outcome_pivot.columns = [f"outcome_sum_{group}" for group in outcome_pivot.columns.to_flat_index()]
        outcome_pivot = outcome_pivot.reset_index()
        summary = base.merge(outcome_pivot, on="stratum_id", how="left")
        for column in ["outcome_sum_exposed", "outcome_sum_reference"]:
            if column not in summary.columns:
                summary[column] = 0.0
        results[outcome_column] = summary
    return results


def _acute_weekly_table_for_outcomes(
    prepared: pd.DataFrame,
    weekly_exposure: pd.DataFrame,
    frame: pd.DataFrame,
    outcome_columns: list[str],
    denominator_column: str,
) -> pd.DataFrame:
    if prepared.empty or weekly_exposure.empty or not outcome_columns:
        return pd.DataFrame()
    outcome_block = _numeric_outcome_block(frame, prepared.index.to_numpy(), outcome_columns)
    subset = pd.concat([prepared.reset_index(drop=True), outcome_block.reset_index(drop=True)], axis=1)
    aggregate_map: dict[str, tuple[str, str]] = {column: (column, "sum") for column in outcome_columns}
    weekly = subset.groupby(["stratum_id", "epidemiologic_week"], as_index=False).agg(
        denominator_sum=(denominator_column, "sum"),
        **aggregate_map,
    )
    weekly = weekly.loc[weekly["denominator_sum"].gt(0)].copy()
    if weekly.empty:
        return pd.DataFrame()
    return weekly.merge(
        weekly_exposure,
        on=["stratum_id", "epidemiologic_week"],
        how="inner",
        validate="one_to_one",
    )


def _window_quintile_table_for_outcomes(
    prepared: pd.DataFrame,
    frame: pd.DataFrame,
    outcome_columns: list[str],
    denominator_column: str,
) -> pd.DataFrame:
    if prepared.empty or not outcome_columns:
        return pd.DataFrame()
    outcome_block = _numeric_outcome_block(frame, prepared.index.to_numpy(), outcome_columns)
    subset = pd.concat([prepared.reset_index(drop=True), outcome_block.reset_index(drop=True)], axis=1)
    aggregate_map: dict[str, tuple[str, str]] = {column: (column, "sum") for column in outcome_columns}
    return subset.groupby(["stratum_id", "exposure_quintile"], as_index=False).agg(
        denominator_sum=(denominator_column, "sum"),
        row_count=("exposure_quintile", "size"),
        **aggregate_map,
    )


def _screen_acute_record_batch(
    records: list[dict[str, Any]],
    prepared: dict[str, pd.DataFrame],
    frame: pd.DataFrame,
    exposure_column: str,
    exposure_label: str,
    lag_or_window: str,
    denominator_column: str,
    model_kind: str,
) -> list[dict[str, Any]]:
    if not records:
        return []
    outcome_columns = [str(record["analysis_unit_id"]) for record in records]
    binary_summaries = _acute_binary_summaries_for_outcomes(
        prepared["screen_subset"],
        frame,
        outcome_columns,
        denominator_column,
    )
    weekly = _acute_weekly_table_for_outcomes(
        prepared["valid_rows"],
        prepared["weekly_exposure"],
        frame,
        outcome_columns,
        denominator_column,
    )
    rows: list[dict[str, Any]] = []
    for record in records:
        outcome_column = str(record["analysis_unit_id"])
        binary_result = _pool_binary_summary(binary_summaries.get(outcome_column, pd.DataFrame()), model_kind=model_kind)
        rows.append(
            {
                "analysis_unit_id": outcome_column,
                "linked_outcome_ids": str(record["linked_outcome_ids"]),
                "candidate_track": str(record["candidate_track"]),
                "analysis_role": str(record["analysis_role"]),
                "outcome_family": str(record["outcome_family"]),
                "mart_name": str(record["mart_name"]),
                "exposure_family_id": str(record["exposure_family_id"]),
                "exposure_family_name": str(record["exposure_family_name"]),
                "exposure_variable": exposure_column,
                "exposure_label": exposure_label,
                "headline_eligible_flag": str(record["headline_eligible_flag"]),
                "denominator_column": denominator_column,
                "model_kind": model_kind,
                "estimate_scale": str(record["estimate_scale"]),
                "lag_or_window": lag_or_window,
                "screening_domain": "acute_p95_vs_reference",
                "screening_method": "within_municipality_year_p95_vs_reference",
                "comparison_label": "Municipality-year p95 versus municipality-year p40 to p60 reference band",
                "seasonality_controls_inherited": str(record["seasonality_controls_applied"]),
                **binary_result,
            }
        )
        if weekly.empty:
            correlation_result = {
                "estimate": float("nan"),
                "lower_ci": float("nan"),
                "upper_ci": float("nan"),
                "p_value": float("nan"),
                "p_value_floor_applied_flag": "no",
                "valid_strata_count": 0,
                "exposed_row_count": 0,
                "reference_row_count": 0,
                "exposed_outcome_sum": 0.0,
                "reference_outcome_sum": 0.0,
                "exposed_denominator_sum": 0.0,
                "reference_denominator_sum": 0.0,
            }
        else:
            outcome_metric = (
                np.log((weekly[outcome_column].astype(float) + 0.5) / weekly["denominator_sum"].astype(float))
                if model_kind == "count"
                else weekly[outcome_column].astype(float) / weekly["denominator_sum"].astype(float)
            )
            working = weekly.loc[:, ["stratum_id", "exposure_anomaly"]].copy()
            working["outcome_metric"] = outcome_metric
            working["outcome_anomaly"] = working["outcome_metric"] - working.groupby("stratum_id")["outcome_metric"].transform("mean")
            correlation_result = _pearson_correlation(working, "exposure_anomaly", "outcome_anomaly")
        rows.append(
            {
                "analysis_unit_id": outcome_column,
                "linked_outcome_ids": str(record["linked_outcome_ids"]),
                "candidate_track": str(record["candidate_track"]),
                "analysis_role": str(record["analysis_role"]),
                "outcome_family": str(record["outcome_family"]),
                "mart_name": str(record["mart_name"]),
                "exposure_family_id": str(record["exposure_family_id"]),
                "exposure_family_name": str(record["exposure_family_name"]),
                "exposure_variable": exposure_column,
                "exposure_label": f"{exposure_label} (weekly anomaly)",
                "headline_eligible_flag": str(record["headline_eligible_flag"]),
                "denominator_column": denominator_column,
                "model_kind": model_kind,
                "estimate_scale": str(record["estimate_scale"]),
                "lag_or_window": lag_or_window,
                "screening_domain": "acute_weekly_anomaly_correlation",
                "screening_method": "pearson_correlation_on_weekly_anomalies",
                "comparison_label": "Deseasonalized weekly anomaly correlation",
                "seasonality_controls_inherited": str(record["seasonality_controls_applied"]),
                **correlation_result,
            }
        )
    return rows


def _screen_window_record_batch(
    records: list[dict[str, Any]],
    prepared: pd.DataFrame,
    frame: pd.DataFrame,
    exposure_column: str,
    exposure_label: str,
    lag_or_window: str,
    denominator_column: str,
    model_kind: str,
    domain_prefix: str,
) -> list[dict[str, Any]]:
    if not records:
        return []
    outcome_columns = [str(record["analysis_unit_id"]) for record in records]
    quintile_table = _window_quintile_table_for_outcomes(
        prepared,
        frame,
        outcome_columns,
        denominator_column,
    )
    rows: list[dict[str, Any]] = []
    for record in records:
        outcome_column = str(record["analysis_unit_id"])
        if quintile_table.empty:
            outcome_summary = pd.DataFrame()
        else:
            outcome_summary = quintile_table.loc[
                :,
                ["stratum_id", "exposure_quintile", "denominator_sum", "row_count", outcome_column],
            ].rename(columns={outcome_column: "outcome_sum"})
        q5_q1 = _pool_binary_summary(_q1_q5_summary(outcome_summary), model_kind=model_kind)
        rows.append(
            {
                "analysis_unit_id": outcome_column,
                "linked_outcome_ids": str(record["linked_outcome_ids"]),
                "candidate_track": str(record["candidate_track"]),
                "analysis_role": str(record["analysis_role"]),
                "outcome_family": str(record["outcome_family"]),
                "mart_name": str(record["mart_name"]),
                "exposure_family_id": str(record["exposure_family_id"]),
                "exposure_family_name": str(record["exposure_family_name"]),
                "exposure_variable": exposure_column,
                "exposure_label": exposure_label,
                "headline_eligible_flag": str(record["headline_eligible_flag"]),
                "denominator_column": denominator_column,
                "model_kind": model_kind,
                "estimate_scale": str(record["estimate_scale"]),
                "lag_or_window": lag_or_window,
                "screening_domain": f"{domain_prefix}_q5_vs_q1",
                "screening_method": "within_municipality_year_quintile_extremes",
                "comparison_label": "Municipality-year quintile 5 versus quintile 1",
                "seasonality_controls_inherited": str(record["seasonality_controls_applied"]),
                **q5_q1,
            }
        )
        quintile_trend = _pool_quintile_trend(outcome_summary, model_kind=model_kind)
        rows.append(
            {
                "analysis_unit_id": outcome_column,
                "linked_outcome_ids": str(record["linked_outcome_ids"]),
                "candidate_track": str(record["candidate_track"]),
                "analysis_role": str(record["analysis_role"]),
                "outcome_family": str(record["outcome_family"]),
                "mart_name": str(record["mart_name"]),
                "exposure_family_id": str(record["exposure_family_id"]),
                "exposure_family_name": str(record["exposure_family_name"]),
                "exposure_variable": exposure_column,
                "exposure_label": exposure_label,
                "headline_eligible_flag": str(record["headline_eligible_flag"]),
                "denominator_column": denominator_column,
                "model_kind": model_kind,
                "estimate_scale": str(record["estimate_scale"]),
                "lag_or_window": lag_or_window,
                "screening_domain": f"{domain_prefix}_quintile_trend",
                "screening_method": "within_municipality_year_quintile_trend",
                "comparison_label": "Municipality-year monotonic quintile trend",
                "seasonality_controls_inherited": str(record["seasonality_controls_applied"]),
                **quintile_trend,
            }
        )
    return rows


def _binary_summary_from_aggregated_table(binary_table: pd.DataFrame, outcome_column: str) -> pd.DataFrame:
    if binary_table.empty:
        return pd.DataFrame()
    summary = binary_table.loc[:, ["stratum_id", "screen_group", "denominator_sum", "row_count", outcome_column]].rename(
        columns={outcome_column: "outcome_sum"}
    )
    pivot = summary.pivot_table(
        index=["stratum_id"],
        columns="screen_group",
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


def _screen_acute_record_batch_from_aggregates(
    records: list[dict[str, Any]],
    binary_table: pd.DataFrame,
    weekly_table: pd.DataFrame,
    exposure_column: str,
    exposure_label: str,
    lag_or_window: str,
    denominator_column: str,
    model_kind: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        outcome_column = str(record["analysis_unit_id"])
        binary_result = _pool_binary_summary(
            _binary_summary_from_aggregated_table(binary_table, outcome_column),
            model_kind=model_kind,
        )
        rows.append(
            {
                "analysis_unit_id": outcome_column,
                "linked_outcome_ids": str(record["linked_outcome_ids"]),
                "candidate_track": str(record["candidate_track"]),
                "analysis_role": str(record["analysis_role"]),
                "outcome_family": str(record["outcome_family"]),
                "mart_name": str(record["mart_name"]),
                "exposure_family_id": str(record["exposure_family_id"]),
                "exposure_family_name": str(record["exposure_family_name"]),
                "exposure_variable": exposure_column,
                "exposure_label": exposure_label,
                "headline_eligible_flag": str(record["headline_eligible_flag"]),
                "denominator_column": denominator_column,
                "model_kind": model_kind,
                "estimate_scale": str(record["estimate_scale"]),
                "lag_or_window": lag_or_window,
                "screening_domain": "acute_p95_vs_reference",
                "screening_method": "within_municipality_year_p95_vs_reference",
                "comparison_label": "Municipality-year p95 versus municipality-year p40 to p60 reference band",
                "seasonality_controls_inherited": str(record["seasonality_controls_applied"]),
                **binary_result,
            }
        )
        if weekly_table.empty:
            correlation_result = {
                "estimate": float("nan"),
                "lower_ci": float("nan"),
                "upper_ci": float("nan"),
                "p_value": float("nan"),
                "p_value_floor_applied_flag": "no",
                "valid_strata_count": 0,
                "exposed_row_count": 0,
                "reference_row_count": 0,
                "exposed_outcome_sum": 0.0,
                "reference_outcome_sum": 0.0,
                "exposed_denominator_sum": 0.0,
                "reference_denominator_sum": 0.0,
            }
        else:
            outcome_metric = (
                np.log((weekly_table[outcome_column].astype(float) + 0.5) / weekly_table["denominator_sum"].astype(float))
                if model_kind == "count"
                else weekly_table[outcome_column].astype(float) / weekly_table["denominator_sum"].astype(float)
            )
            working = weekly_table.loc[:, ["stratum_id", "exposure_anomaly"]].copy()
            working["outcome_metric"] = outcome_metric
            working["outcome_anomaly"] = working["outcome_metric"] - working.groupby("stratum_id")["outcome_metric"].transform("mean")
            correlation_result = _pearson_correlation(working, "exposure_anomaly", "outcome_anomaly")
        rows.append(
            {
                "analysis_unit_id": outcome_column,
                "linked_outcome_ids": str(record["linked_outcome_ids"]),
                "candidate_track": str(record["candidate_track"]),
                "analysis_role": str(record["analysis_role"]),
                "outcome_family": str(record["outcome_family"]),
                "mart_name": str(record["mart_name"]),
                "exposure_family_id": str(record["exposure_family_id"]),
                "exposure_family_name": str(record["exposure_family_name"]),
                "exposure_variable": exposure_column,
                "exposure_label": f"{exposure_label} (weekly anomaly)",
                "headline_eligible_flag": str(record["headline_eligible_flag"]),
                "denominator_column": denominator_column,
                "model_kind": model_kind,
                "estimate_scale": str(record["estimate_scale"]),
                "lag_or_window": lag_or_window,
                "screening_domain": "acute_weekly_anomaly_correlation",
                "screening_method": "pearson_correlation_on_weekly_anomalies",
                "comparison_label": "Deseasonalized weekly anomaly correlation",
                "seasonality_controls_inherited": str(record["seasonality_controls_applied"]),
                **correlation_result,
            }
        )
    return rows


def _screen_window_record_batch_from_aggregates(
    records: list[dict[str, Any]],
    quintile_table: pd.DataFrame,
    exposure_column: str,
    exposure_label: str,
    lag_or_window: str,
    denominator_column: str,
    model_kind: str,
    domain_prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        outcome_column = str(record["analysis_unit_id"])
        if quintile_table.empty:
            outcome_summary = pd.DataFrame()
        else:
            outcome_summary = quintile_table.loc[
                :,
                ["stratum_id", "exposure_quintile", "denominator_sum", "row_count", outcome_column],
            ].rename(columns={outcome_column: "outcome_sum"})
        q5_q1 = _pool_binary_summary(_q1_q5_summary(outcome_summary), model_kind=model_kind)
        rows.append(
            {
                "analysis_unit_id": outcome_column,
                "linked_outcome_ids": str(record["linked_outcome_ids"]),
                "candidate_track": str(record["candidate_track"]),
                "analysis_role": str(record["analysis_role"]),
                "outcome_family": str(record["outcome_family"]),
                "mart_name": str(record["mart_name"]),
                "exposure_family_id": str(record["exposure_family_id"]),
                "exposure_family_name": str(record["exposure_family_name"]),
                "exposure_variable": exposure_column,
                "exposure_label": exposure_label,
                "headline_eligible_flag": str(record["headline_eligible_flag"]),
                "denominator_column": denominator_column,
                "model_kind": model_kind,
                "estimate_scale": str(record["estimate_scale"]),
                "lag_or_window": lag_or_window,
                "screening_domain": f"{domain_prefix}_q5_vs_q1",
                "screening_method": "within_municipality_year_quintile_extremes",
                "comparison_label": "Municipality-year quintile 5 versus quintile 1",
                "seasonality_controls_inherited": str(record["seasonality_controls_applied"]),
                **q5_q1,
            }
        )
        quintile_trend = _pool_quintile_trend(outcome_summary, model_kind=model_kind)
        rows.append(
            {
                "analysis_unit_id": outcome_column,
                "linked_outcome_ids": str(record["linked_outcome_ids"]),
                "candidate_track": str(record["candidate_track"]),
                "analysis_role": str(record["analysis_role"]),
                "outcome_family": str(record["outcome_family"]),
                "mart_name": str(record["mart_name"]),
                "exposure_family_id": str(record["exposure_family_id"]),
                "exposure_family_name": str(record["exposure_family_name"]),
                "exposure_variable": exposure_column,
                "exposure_label": exposure_label,
                "headline_eligible_flag": str(record["headline_eligible_flag"]),
                "denominator_column": denominator_column,
                "model_kind": model_kind,
                "estimate_scale": str(record["estimate_scale"]),
                "lag_or_window": lag_or_window,
                "screening_domain": f"{domain_prefix}_quintile_trend",
                "screening_method": "within_municipality_year_quintile_trend",
                "comparison_label": "Municipality-year monotonic quintile trend",
                "seasonality_controls_inherited": str(record["seasonality_controls_applied"]),
                **quintile_trend,
            }
        )
    return rows


def _screen_candidate_pair(
    record: dict[str, Any],
    frame: pd.DataFrame,
    acute_cache: dict[tuple[str, str, str], dict[str, pd.DataFrame]],
    window_cache: dict[tuple[str, str, str], pd.DataFrame],
    outcome_cache: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    seasonality_profile_id = str(record["seasonality_profile_id"])
    profile_spec = SCREENING_PROFILE_SPECS[seasonality_profile_id]
    model_kind = str(record["model_kind"])
    denominator_column = str(record["denominator_column_for_step2"])
    outcome_column = str(record["analysis_unit_id"])
    outcome_array = outcome_cache.get(outcome_column)
    if outcome_array is None:
        outcome_array = pd.to_numeric(frame[outcome_column], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
        outcome_cache[outcome_column] = outcome_array
    family_exposures = profile_spec["family_exposures"][str(record["exposure_family_id"])]
    rows: list[dict[str, Any]] = []

    if profile_spec["kind"] == "acute":
        for exposure_column, exposure_label, lag_or_window in family_exposures:
            cache_key = (str(record["mart_name"]), exposure_column, denominator_column)
            prepared = acute_cache.get(cache_key)
            if prepared is None:
                prepared = _prepare_acute_screening_inputs(frame, exposure_column, denominator_column)
                acute_cache[cache_key] = prepared
            binary_summary = _acute_binary_summary_from_prepared(
                prepared["screen_subset"],
                outcome_array,
                denominator_column,
            )
            binary_result = _pool_binary_summary(binary_summary, model_kind=model_kind)
            rows.append(
                {
                    "analysis_unit_id": outcome_column,
                    "linked_outcome_ids": str(record["linked_outcome_ids"]),
                    "candidate_track": str(record["candidate_track"]),
                    "analysis_role": str(record["analysis_role"]),
                    "outcome_family": str(record["outcome_family"]),
                    "mart_name": str(record["mart_name"]),
                    "exposure_family_id": str(record["exposure_family_id"]),
                    "exposure_family_name": str(record["exposure_family_name"]),
                    "exposure_variable": exposure_column,
                    "exposure_label": exposure_label,
                    "headline_eligible_flag": str(record["headline_eligible_flag"]),
                    "denominator_column": denominator_column,
                    "model_kind": model_kind,
                    "estimate_scale": str(record["estimate_scale"]),
                    "lag_or_window": lag_or_window,
                    "screening_domain": "acute_p95_vs_reference",
                    "screening_method": "within_municipality_year_p95_vs_reference",
                    "comparison_label": "Municipality-year p95 versus municipality-year p40 to p60 reference band",
                    "seasonality_controls_inherited": str(record["seasonality_controls_applied"]),
                    **binary_result,
                }
            )

            weekly = _acute_weekly_anomalies_from_prepared(
                prepared["valid_rows"],
                prepared["weekly_exposure"],
                outcome_array,
                denominator_column,
                model_kind=model_kind,
            )
            correlation_result = _pearson_correlation(weekly, "exposure_anomaly", "outcome_anomaly") if not weekly.empty else {
                "estimate": float("nan"),
                "lower_ci": float("nan"),
                "upper_ci": float("nan"),
                "p_value": float("nan"),
                "p_value_floor_applied_flag": "no",
                "valid_strata_count": 0,
                "exposed_row_count": 0,
                "reference_row_count": 0,
                "exposed_outcome_sum": 0.0,
                "reference_outcome_sum": 0.0,
                "exposed_denominator_sum": 0.0,
                "reference_denominator_sum": 0.0,
            }
            rows.append(
                {
                    "analysis_unit_id": outcome_column,
                    "linked_outcome_ids": str(record["linked_outcome_ids"]),
                    "candidate_track": str(record["candidate_track"]),
                    "analysis_role": str(record["analysis_role"]),
                    "outcome_family": str(record["outcome_family"]),
                    "mart_name": str(record["mart_name"]),
                    "exposure_family_id": str(record["exposure_family_id"]),
                    "exposure_family_name": str(record["exposure_family_name"]),
                    "exposure_variable": exposure_column,
                    "exposure_label": f"{exposure_label} (weekly anomaly)",
                    "headline_eligible_flag": str(record["headline_eligible_flag"]),
                    "denominator_column": denominator_column,
                    "model_kind": model_kind,
                    "estimate_scale": str(record["estimate_scale"]),
                    "lag_or_window": lag_or_window,
                    "screening_domain": "acute_weekly_anomaly_correlation",
                    "screening_method": "pearson_correlation_on_weekly_anomalies",
                    "comparison_label": "Deseasonalized weekly anomaly correlation",
                    "seasonality_controls_inherited": str(record["seasonality_controls_applied"]),
                    **correlation_result,
                }
            )
    else:
        domain_prefix = str(profile_spec["screening_domain_prefix"])
        for exposure_column, exposure_label, lag_or_window in family_exposures:
            cache_key = (str(record["mart_name"]), exposure_column, denominator_column)
            prepared = window_cache.get(cache_key)
            if prepared is None:
                prepared = _prepare_window_quintiles(frame, exposure_column, denominator_column)
                window_cache[cache_key] = prepared
            quintile_summary = _window_quintile_summary_from_prepared(
                prepared,
                outcome_array,
                denominator_column,
            )
            q5_q1 = _pool_binary_summary(_q1_q5_summary(quintile_summary), model_kind=model_kind)
            rows.append(
                {
                    "analysis_unit_id": outcome_column,
                    "linked_outcome_ids": str(record["linked_outcome_ids"]),
                    "candidate_track": str(record["candidate_track"]),
                    "analysis_role": str(record["analysis_role"]),
                    "outcome_family": str(record["outcome_family"]),
                    "mart_name": str(record["mart_name"]),
                    "exposure_family_id": str(record["exposure_family_id"]),
                    "exposure_family_name": str(record["exposure_family_name"]),
                    "exposure_variable": exposure_column,
                    "exposure_label": exposure_label,
                    "headline_eligible_flag": str(record["headline_eligible_flag"]),
                    "denominator_column": denominator_column,
                    "model_kind": model_kind,
                    "estimate_scale": str(record["estimate_scale"]),
                    "lag_or_window": lag_or_window,
                    "screening_domain": f"{domain_prefix}_q5_vs_q1",
                    "screening_method": "within_municipality_year_quintile_extremes",
                    "comparison_label": "Municipality-year quintile 5 versus quintile 1",
                    "seasonality_controls_inherited": str(record["seasonality_controls_applied"]),
                    **q5_q1,
                }
            )
            quintile_trend = _pool_quintile_trend(quintile_summary, model_kind=model_kind)
            rows.append(
                {
                    "analysis_unit_id": outcome_column,
                    "linked_outcome_ids": str(record["linked_outcome_ids"]),
                    "candidate_track": str(record["candidate_track"]),
                    "analysis_role": str(record["analysis_role"]),
                    "outcome_family": str(record["outcome_family"]),
                    "mart_name": str(record["mart_name"]),
                    "exposure_family_id": str(record["exposure_family_id"]),
                    "exposure_family_name": str(record["exposure_family_name"]),
                    "exposure_variable": exposure_column,
                    "exposure_label": exposure_label,
                    "headline_eligible_flag": str(record["headline_eligible_flag"]),
                    "denominator_column": denominator_column,
                    "model_kind": model_kind,
                    "estimate_scale": str(record["estimate_scale"]),
                    "lag_or_window": lag_or_window,
                    "screening_domain": f"{domain_prefix}_quintile_trend",
                    "screening_method": "within_municipality_year_quintile_trend",
                    "comparison_label": "Municipality-year monotonic quintile trend",
                    "seasonality_controls_inherited": str(record["seasonality_controls_applied"]),
                    **quintile_trend,
                }
            )
    return rows


def _build_signal_matrix(screening_results: pd.DataFrame) -> pd.DataFrame:
    if screening_results.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in screening_results.groupby(
        ["analysis_unit_id", "candidate_track", "analysis_role", "outcome_family", "exposure_family_id", "exposure_family_name"],
        sort=True,
    ):
        finite = group.loc[
            group["estimate"].apply(lambda value: math.isfinite(float(value)))
            & group["valid_strata_count"].fillna(0).gt(0)
        ].copy()
        finite["effect_direction"] = finite.apply(
            lambda row: _estimate_direction(row["estimate"], str(row["estimate_scale"])),
            axis=1,
        ) if not finite.empty else pd.Series(dtype="object")
        if not finite.empty:
            finite = finite.sort_values(["p_value", "valid_strata_count"], ascending=[True, False], ignore_index=True)
            best = finite.iloc[0]
            directions = {value for value in finite["effect_direction"].astype(str).tolist() if value}
            sign_consistency_flag = len(directions) <= 1
            screening_domain_count = int(finite["screening_domain"].astype(str).nunique())
            finite_result_count = int(len(finite))
            best_p_value = float(best["p_value"])
        else:
            best = None
            sign_consistency_flag = False
            screening_domain_count = 0
            finite_result_count = 0
            best_p_value = float("nan")

        if finite_result_count >= 3 and screening_domain_count >= 2 and sign_consistency_flag and math.isfinite(best_p_value) and best_p_value < 0.05:
            support_grade = "A"
        elif finite_result_count >= 2 and screening_domain_count >= 2 and sign_consistency_flag:
            support_grade = "B"
        elif finite_result_count >= 1:
            support_grade = "C"
        else:
            support_grade = "D"

        rows.append(
            {
                "analysis_unit_id": str(keys[0]),
                "candidate_track": str(keys[1]),
                "analysis_role": str(keys[2]),
                "outcome_family": str(keys[3]),
                "exposure_family_id": str(keys[4]),
                "exposure_family_name": str(keys[5]),
                "screening_result_count": int(group.shape[0]),
                "finite_result_count": finite_result_count,
                "screening_domain_count": screening_domain_count,
                "sign_consistency_flag": bool(sign_consistency_flag),
                "support_grade": support_grade,
                "support_grade_rank": SUPPORT_GRADE_RANK[support_grade],
                "best_lag_or_window": str(best["lag_or_window"]) if best is not None else "",
                "best_screening_domain": str(best["screening_domain"]) if best is not None else "",
                "best_screening_method": str(best["screening_method"]) if best is not None else "",
                "best_exposure_variable": str(best["exposure_variable"]) if best is not None else "",
                "best_exposure_label": str(best["exposure_label"]) if best is not None else "",
                "best_estimate": float(best["estimate"]) if best is not None else float("nan"),
                "best_lower_ci": float(best["lower_ci"]) if best is not None else float("nan"),
                "best_upper_ci": float(best["upper_ci"]) if best is not None else float("nan"),
                "best_p_value": best_p_value,
                "best_valid_strata_count": int(best["valid_strata_count"]) if best is not None else 0,
                "headline_eligible_flag": str(group["headline_eligible_flag"].iloc[0]),
                "estimate_scale": str(group["estimate_scale"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["candidate_track", "analysis_role", "analysis_unit_id", "exposure_family_id"],
        ignore_index=True,
    )


def _coherence_not_weaker(candidate: dict[str, Any] | None, reference: dict[str, Any] | None) -> bool:
    if candidate is None:
        return False
    if reference is None:
        return True
    if int(candidate["support_grade_rank"]) < int(reference["support_grade_rank"]):
        return False
    if bool(candidate["sign_consistency_flag"]) < bool(reference["sign_consistency_flag"]):
        return False
    if int(candidate["screening_domain_count"]) < int(reference["screening_domain_count"]):
        return False
    return True


def _effective_p_value(row: dict[str, Any] | None) -> float:
    if row is None:
        return 1.0
    value = float(row["best_p_value"]) if math.isfinite(float(row["best_p_value"])) else 1.0
    return value


def _build_family_threshold_comparison(
    signal_matrix: pd.DataFrame,
    promotion_decisions: pd.DataFrame,
) -> pd.DataFrame:
    if promotion_decisions.empty:
        return pd.DataFrame()
    approved = promotion_decisions.loc[
        promotion_decisions["approved_for_next_steps_flag"].eq(True),
        ["analysis_unit_id", "candidate_track", "analysis_role", "estimable_exposure_family_ids"],
    ].drop_duplicates()
    rows: list[dict[str, Any]] = []
    for record in approved.to_dict(orient="records"):
        subset = signal_matrix.loc[
            signal_matrix["analysis_unit_id"].astype(str).eq(str(record["analysis_unit_id"]))
            & signal_matrix["candidate_track"].astype(str).eq(str(record["candidate_track"]))
        ].copy()
        tmax = subset.loc[subset["exposure_family_id"].astype(str).eq("family_1_tmax_primary")]
        heat_index = subset.loc[subset["exposure_family_id"].astype(str).eq("family_4_heat_index_parallel")]
        tmax_row = tmax.iloc[0].to_dict() if not tmax.empty else None
        heat_index_row = heat_index.iloc[0].to_dict() if not heat_index.empty else None

        recommendation = "no_preference_yet"
        reason = "families_tied_or_inconclusive"
        if heat_index_row and not tmax_row:
            recommendation = "heat_index"
            reason = "only_heat_index_screening_available"
        elif tmax_row and not heat_index_row:
            recommendation = "tmax"
            reason = "only_tmax_screening_available"
        elif heat_index_row and tmax_row:
            heat_index_p_value = _effective_p_value(heat_index_row)
            tmax_p_value = _effective_p_value(tmax_row)
            heat_index_better = (
                _coherence_not_weaker(heat_index_row, tmax_row)
                and (
                    int(heat_index_row["support_grade_rank"]) > int(tmax_row["support_grade_rank"])
                    or (
                        int(heat_index_row["support_grade_rank"]) == int(tmax_row["support_grade_rank"])
                        and heat_index_p_value <= (tmax_p_value + P_VALUE_TIE_TOLERANCE)
                    )
                )
            )
            tmax_better = (
                _coherence_not_weaker(tmax_row, heat_index_row)
                and (
                    int(tmax_row["support_grade_rank"]) > int(heat_index_row["support_grade_rank"])
                    or (
                        int(tmax_row["support_grade_rank"]) == int(heat_index_row["support_grade_rank"])
                        and tmax_p_value < (heat_index_p_value - P_VALUE_TIE_TOLERANCE)
                    )
                )
            )
            if heat_index_better and not tmax_better:
                recommendation = "heat_index"
                reason = "heat_index_equal_or_better_support_without_weaker_coherence"
            elif tmax_better and not heat_index_better:
                recommendation = "tmax"
                reason = "tmax_stronger_or_more_coherent_screening_support"

        rows.append(
            {
                "analysis_unit_id": str(record["analysis_unit_id"]),
                "candidate_track": str(record["candidate_track"]),
                "analysis_role": str(record["analysis_role"]),
                "estimable_exposure_family_ids": str(record["estimable_exposure_family_ids"]),
                "tmax_support_grade": str(tmax_row["support_grade"]) if tmax_row else "",
                "tmax_best_p_value": float(tmax_row["best_p_value"]) if tmax_row else float("nan"),
                "tmax_screening_domain_count": int(tmax_row["screening_domain_count"]) if tmax_row else 0,
                "tmax_sign_consistency_flag": bool(tmax_row["sign_consistency_flag"]) if tmax_row else False,
                "heat_index_support_grade": str(heat_index_row["support_grade"]) if heat_index_row else "",
                "heat_index_best_p_value": float(heat_index_row["best_p_value"]) if heat_index_row else float("nan"),
                "heat_index_screening_domain_count": int(heat_index_row["screening_domain_count"]) if heat_index_row else 0,
                "heat_index_sign_consistency_flag": bool(heat_index_row["sign_consistency_flag"]) if heat_index_row else False,
                "headline_family_recommendation": recommendation,
                "family_comparison_reason": reason,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["candidate_track", "analysis_role", "analysis_unit_id"],
        ignore_index=True,
    )


def _build_decision_checklist(
    eligible_pairs: pd.DataFrame,
    approved_candidates: pd.DataFrame,
    signal_matrix: pd.DataFrame,
    family_threshold_comparison: pd.DataFrame,
    gates: dict[str, bool],
) -> pd.DataFrame:
    approved_candidate_count = int(len(approved_candidates))
    recommendation_count = int(len(family_threshold_comparison))
    rows = [
        {
            "checklist_item": "Every Step 3-approved candidate was screened in at least one exposure family.",
            "pass_flag": bool(gates["approved_candidates_screened"]),
            "evidence": f"{int(signal_matrix['analysis_unit_id'].nunique()) if not signal_matrix.empty else 0}/{approved_candidate_count} approved candidates produced at least one family-level screening summary.",
        },
        {
            "checklist_item": "Headline-family recommendation was written for every approved candidate.",
            "pass_flag": bool(gates["headline_recommendations_written"]),
            "evidence": f"{recommendation_count}/{approved_candidate_count} approved candidates received a Step 3 family recommendation.",
        },
        {
            "checklist_item": "Seasonality controls were inherited from Step 2 in every screening row.",
            "pass_flag": bool(gates["seasonality_controls_inherited"]),
            "evidence": f"{int(len(eligible_pairs))} approved candidate-family pairs were screened under inherited temporal controls.",
        },
        {
            "checklist_item": "Threshold-screening outputs were written for the approved Step 2 roster.",
            "pass_flag": bool(gates["screening_results_written"]),
            "evidence": f"{int(len(signal_matrix))} candidate-family summaries were written across {int(len(approved_candidates))} approved candidates.",
        },
    ]
    return pd.DataFrame(rows)


def _render_step3_report(
    round_config: dict[str, Any],
    step2_report: dict[str, Any],
    gates: dict[str, bool],
    summary: dict[str, Any],
    outputs: dict[str, str],
    workspace_report: dict[str, object],
    family_threshold_comparison: pd.DataFrame,
) -> str:
    lines = [
        "# Step 3 Threshold Discovery Report",
        "",
        f"- Generated at: `{_now_utc()}`",
        f"- Analysis ID: `{round_config['analysis_id']}`",
        f"- Analysis title: `{round_config['analysis_title']}`",
        f"- Years: `{', '.join(str(year) for year in round_config['years'])}`",
        f"- Upstream Step 2 status: `{step2_report['status']}`",
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
            f"- Approved Step 2 candidates: `{summary['approved_candidate_count']}`",
            f"- Step 3 candidate-family pairs: `{summary['eligible_pair_count']}`",
            f"- Screening result rows: `{summary['screening_result_count']}`",
            f"- Signal matrix rows: `{summary['signal_matrix_count']}`",
            f"- Tmax headline recommendations: `{summary['tmax_recommendation_count']}`",
            f"- Heat-index headline recommendations: `{summary['heat_index_recommendation_count']}`",
            f"- No-preference-yet recommendations: `{summary['no_preference_count']}`",
        ]
    )
    if not family_threshold_comparison.empty:
        lines.extend(["", "## Headline Recommendations", ""])
        for record in family_threshold_comparison.to_dict(orient="records"):
            lines.append(
                "- "
                f"{record['analysis_unit_id']}: headline=`{record['headline_family_recommendation']}`, "
                f"reason=`{record['family_comparison_reason']}`"
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


def _write_step3_publication_notes(
    root: Path,
    *,
    status: str,
    summary: dict[str, Any],
    outputs: dict[str, str],
) -> None:
    workspace_root = incremental_publication_root(root)
    (workspace_root / "06_step3_threshold_screening.md").write_text(
        (
            "# Step 3 Threshold Screening\n\n"
            f"- Status: `{status}`\n"
            f"- Approved candidates screened: `{summary['approved_candidate_count']}`\n"
            f"- Screening result rows: `{summary['screening_result_count']}`\n"
            f"- Screening results: `{outputs['screening_results']}`\n"
            f"- Technical report: `{outputs['step3_report']}`\n"
        ),
        encoding="utf-8",
    )
    (workspace_root / "07_step3_exposure_family_decision.md").write_text(
        (
            "# Step 3 Exposure Family Decision\n\n"
            f"- Status: `{status}`\n"
            f"- Heat-index headline recommendations: `{summary['heat_index_recommendation_count']}`\n"
            f"- Tmax headline recommendations: `{summary['tmax_recommendation_count']}`\n"
            f"- Family comparison table: `{outputs['family_threshold_comparison']}`\n"
        ),
        encoding="utf-8",
    )


def build_step3_threshold_discovery(
    root: Path,
    config_path: Path | None = None,
    step2_report: dict[str, Any] | None = None,
) -> dict[str, object]:
    directories = ensure_step3_directory_structure(root)
    round_config = load_round_config(root, config_path=config_path)
    workspace_report = _scaffold_step3_publication_workspace(root)
    upstream_step2_report = step2_report or build_step2_baseline_models(root, config_path=config_path)

    screening_results_path = analysis_step3_root(root) / "step3_screening_results.csv"
    signal_matrix_path = analysis_step3_root(root) / "step3_signal_matrix.csv"
    family_comparison_path = analysis_step3_root(root) / "step3_family_threshold_comparison.csv"
    checklist_path = analysis_step3_root(root) / "step3_decision_checklist.csv"
    report_path = analysis_step3_root(root) / "step3_report.md"

    outputs = {
        "screening_results": _relative(root, screening_results_path),
        "signal_matrix": _relative(root, signal_matrix_path),
        "family_threshold_comparison": _relative(root, family_comparison_path),
        "decision_checklist": _relative(root, checklist_path),
        "step3_report": _relative(root, report_path),
    }

    empty = pd.DataFrame()
    if upstream_step2_report["status"] != "PASS":
        gates = {
            "upstream_step2_pass": False,
            "approved_candidates_screened": False,
            "headline_recommendations_written": False,
            "seasonality_controls_inherited": False,
            "screening_results_written": False,
        }
        for path in (screening_results_path, signal_matrix_path, family_comparison_path, checklist_path):
            empty.to_csv(path, index=False)
        report_path.write_text(
            _render_step3_report(
                round_config=round_config,
                step2_report=upstream_step2_report,
                gates=gates,
                summary={
                    "approved_candidate_count": 0,
                    "eligible_pair_count": 0,
                    "screening_result_count": 0,
                    "signal_matrix_count": 0,
                    "tmax_recommendation_count": 0,
                    "heat_index_recommendation_count": 0,
                    "no_preference_count": 0,
                },
                outputs=outputs,
                workspace_report=workspace_report,
                family_threshold_comparison=pd.DataFrame(),
            ),
            encoding="utf-8",
        )
        _write_step3_publication_notes(
            root,
            status="FAIL",
            summary={
                "approved_candidate_count": 0,
                "screening_result_count": 0,
                "heat_index_recommendation_count": 0,
                "tmax_recommendation_count": 0,
            },
            outputs=outputs,
        )
        return {
            "status": "FAIL",
            "step": STEP3_NAME,
            "analysis_id": round_config["analysis_id"],
            "years": list(round_config["years"]),
            "directories": directories,
            "upstream_step2_status": upstream_step2_report["status"],
            "gates": gates,
            "summary": {
                "approved_candidate_count": 0,
                "eligible_pair_count": 0,
                "screening_result_count": 0,
                "signal_matrix_count": 0,
                "tmax_recommendation_count": 0,
                "heat_index_recommendation_count": 0,
                "no_preference_count": 0,
            },
            "outputs": outputs,
            "workspace_report": workspace_report,
        }

    candidate_roster, promotion_decisions, preferred_models = _load_step2_inputs(root)
    eligible_pairs = _eligible_step3_pairs(candidate_roster, promotion_decisions, preferred_models)
    required_columns = _required_columns_by_mart(eligible_pairs) if not eligible_pairs.empty else {}

    screening_rows: list[dict[str, Any]] = []
    for mart_name, mart_group in eligible_pairs.groupby("mart_name", sort=True):
        parquet_paths = _mart_parquet_paths(root, list(round_config["years"]), str(mart_name))
        frame = None if DUCKDB_FAST_PATH_AVAILABLE else _load_mart_frame(
            root,
            list(round_config["years"]),
            str(mart_name),
            required_columns[str(mart_name)],
        )
        acute_cache: dict[tuple[str, str, str], dict[str, pd.DataFrame]] = {}
        window_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
        group_keys = [
            "seasonality_profile_id",
            "exposure_family_id",
            "denominator_column_for_step2",
            "model_kind",
            "estimate_scale",
        ]
        for _, batch in mart_group.groupby(group_keys, sort=False):
            records = batch.to_dict(orient="records")
            first_record = records[0]
            seasonality_profile_id = str(first_record["seasonality_profile_id"])
            exposure_family_id = str(first_record["exposure_family_id"])
            denominator_column = str(first_record["denominator_column_for_step2"])
            model_kind = str(first_record["model_kind"])
            profile_spec = SCREENING_PROFILE_SPECS[seasonality_profile_id]
            family_exposures = profile_spec["family_exposures"][exposure_family_id]
            outcome_columns = [str(record["analysis_unit_id"]) for record in records]
            if profile_spec["kind"] == "acute":
                for exposure_column, exposure_label, lag_or_window in family_exposures:
                    if DUCKDB_FAST_PATH_AVAILABLE:
                        cache_key = (str(mart_name), exposure_column, denominator_column, tuple(sorted(outcome_columns)))
                        aggregated = acute_cache.get(cache_key)
                        if aggregated is None:
                            aggregated = _duckdb_acute_screening_bundle(
                                parquet_paths,
                                exposure_column,
                                denominator_column,
                                outcome_columns,
                            )
                            acute_cache[cache_key] = aggregated
                        screening_rows.extend(
                            _screen_acute_record_batch_from_aggregates(
                                records,
                                aggregated[0],
                                aggregated[1],
                                exposure_column,
                                exposure_label,
                                lag_or_window,
                                denominator_column,
                                model_kind,
                            )
                        )
                    else:
                        cache_key = (str(mart_name), exposure_column, denominator_column)
                        prepared = acute_cache.get(cache_key)
                        if prepared is None:
                            prepared = _prepare_acute_screening_inputs(frame, exposure_column, denominator_column)
                            acute_cache[cache_key] = prepared
                        screening_rows.extend(
                            _screen_acute_record_batch(
                                records,
                                prepared,
                                frame,
                                exposure_column,
                                exposure_label,
                                lag_or_window,
                                denominator_column,
                                model_kind,
                            )
                        )
            else:
                domain_prefix = str(profile_spec["screening_domain_prefix"])
                for exposure_column, exposure_label, lag_or_window in family_exposures:
                    if DUCKDB_FAST_PATH_AVAILABLE:
                        cache_key = (str(mart_name), exposure_column, denominator_column, tuple(sorted(outcome_columns)))
                        quintile_table = window_cache.get(cache_key)
                        if quintile_table is None:
                            quintile_table = _duckdb_window_quintile_table(
                                parquet_paths,
                                exposure_column,
                                denominator_column,
                                outcome_columns,
                            )
                            window_cache[cache_key] = quintile_table
                        screening_rows.extend(
                            _screen_window_record_batch_from_aggregates(
                                records,
                                quintile_table,
                                exposure_column,
                                exposure_label,
                                lag_or_window,
                                denominator_column,
                                model_kind,
                                domain_prefix,
                            )
                        )
                    else:
                        cache_key = (str(mart_name), exposure_column, denominator_column)
                        prepared = window_cache.get(cache_key)
                        if prepared is None:
                            prepared = _prepare_window_quintiles(frame, exposure_column, denominator_column)
                            window_cache[cache_key] = prepared
                        screening_rows.extend(
                            _screen_window_record_batch(
                                records,
                                prepared,
                                frame,
                                exposure_column,
                                exposure_label,
                                lag_or_window,
                                denominator_column,
                                model_kind,
                                domain_prefix,
                            )
                        )

    screening_results = pd.DataFrame(screening_rows).sort_values(
        ["candidate_track", "analysis_role", "analysis_unit_id", "exposure_family_id", "lag_or_window", "screening_domain"],
        ignore_index=True,
    ) if screening_rows else pd.DataFrame()
    signal_matrix = _build_signal_matrix(screening_results)
    family_threshold_comparison = _build_family_threshold_comparison(signal_matrix, promotion_decisions)

    approved_candidates = promotion_decisions.loc[
        promotion_decisions["approved_for_next_steps_flag"].eq(True),
        ["analysis_unit_id", "candidate_track"],
    ].drop_duplicates()
    gates = {
        "upstream_step2_pass": True,
        "approved_candidates_screened": (
            signal_matrix[["analysis_unit_id", "candidate_track"]].drop_duplicates().shape[0] == approved_candidates.shape[0]
            if not approved_candidates.empty
            else False
        ),
        "headline_recommendations_written": (
            family_threshold_comparison[["analysis_unit_id", "candidate_track"]].drop_duplicates().shape[0] == approved_candidates.shape[0]
            if not approved_candidates.empty
            else False
        ),
        "seasonality_controls_inherited": bool(
            not screening_results.empty
            and screening_results["seasonality_controls_inherited"].astype(str).str.len().gt(0).all()
        ),
        "screening_results_written": not screening_results.empty,
    }
    status = "PASS" if all(gates.values()) else "FAIL"

    screening_results.to_csv(screening_results_path, index=False)
    signal_matrix.to_csv(signal_matrix_path, index=False)
    family_threshold_comparison.to_csv(family_comparison_path, index=False)
    checklist = _build_decision_checklist(
        eligible_pairs=eligible_pairs,
        approved_candidates=approved_candidates,
        signal_matrix=signal_matrix,
        family_threshold_comparison=family_threshold_comparison,
        gates=gates,
    )
    checklist.to_csv(checklist_path, index=False)

    summary = {
        "approved_candidate_count": int(approved_candidates.shape[0]),
        "eligible_pair_count": int(eligible_pairs.shape[0]),
        "screening_result_count": int(screening_results.shape[0]),
        "signal_matrix_count": int(signal_matrix.shape[0]),
        "tmax_recommendation_count": int(
            family_threshold_comparison["headline_family_recommendation"].astype(str).eq("tmax").sum()
        ) if not family_threshold_comparison.empty else 0,
        "heat_index_recommendation_count": int(
            family_threshold_comparison["headline_family_recommendation"].astype(str).eq("heat_index").sum()
        ) if not family_threshold_comparison.empty else 0,
        "no_preference_count": int(
            family_threshold_comparison["headline_family_recommendation"].astype(str).eq("no_preference_yet").sum()
        ) if not family_threshold_comparison.empty else 0,
    }
    report_path.write_text(
        _render_step3_report(
            round_config=round_config,
            step2_report=upstream_step2_report,
            gates=gates,
            summary=summary,
            outputs=outputs,
            workspace_report=workspace_report,
            family_threshold_comparison=family_threshold_comparison,
        ),
        encoding="utf-8",
    )
    _write_step3_publication_notes(root, status=status, summary=summary, outputs=outputs)

    return {
        "status": status,
        "step": STEP3_NAME,
        "analysis_id": round_config["analysis_id"],
        "years": list(round_config["years"]),
        "directories": directories,
        "upstream_step2_status": upstream_step2_report["status"],
        "gates": gates,
        "summary": summary,
        "outputs": outputs,
        "workspace_report": workspace_report,
    }
