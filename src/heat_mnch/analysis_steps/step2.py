from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import load_round_config, load_seasonality_contract
from .registry import (
    analysis_shared_marts_root,
    analysis_step1_root,
    analysis_step2_root,
    ensure_step2_directory_structure,
    incremental_publication_root,
)
from .step1 import build_step1_descriptive_atlas


STEP2_PUBLICATION_FILES = {
    "04_step2_baseline_models.md": "# Step 2 Baseline Models\n\nRecord the preferred baseline specifications and caveats here.\n",
    "05_step2_decision_log.md": "# Step 2 Decision Log\n\nDocument promotion, downgrade, and exposure-family decisions here.\n",
}
PROMOTED_ANALYSIS_ROLES = {"primary", "sensitivity"}
COUNT_ESTIMATE_SCALE = "rate_ratio_per_interquartile_range_increase"
SEVERITY_ESTIMATE_SCALE = "mean_difference_per_interquartile_range_increase"
OVERDISPERSION_THRESHOLD = 1.5
PROBABILITY_FLOOR = 1e-300
STEP2_NEXT_STEP_ROUTE = "step3_threshold_discovery"

EXPOSURE_FAMILY_SPECS = {
    "family_1_tmax_primary": {
        "exposure_family_name": "Maximum daily temperature",
        "headline_eligible_flag": "yes",
    },
    "family_4_heat_index_parallel": {
        "exposure_family_name": "Approximate heat index",
        "headline_eligible_flag": "yes",
    },
}

BASELINE_PROFILE_SPECS = {
    "acute_daily_standard": {
        "mart_name": "acute_daily_mart",
        "main_strata": ["municipality_code", "calendar_year", "calendar_month", "day_of_week"],
        "main_controls": ["calendar_year", "calendar_month", "day_of_week"],
        "robustness": [],
        "model_kind": "count",
        "family_exposures": {
            "family_1_tmax_primary": {
                "exposure_variable": "maximum_daily_temperature_c_lag0_1_mean",
                "exposure_label": "Maximum daily temperature (lag 0 to 1 mean)",
            },
            "family_4_heat_index_parallel": {
                "exposure_variable": "approximate_heat_index_c_lag0_1_mean",
                "exposure_label": "Approximate heat index (lag 0 to 1 mean)",
            },
        },
    },
    "neonatal_acute_with_epiweek": {
        "mart_name": "acute_daily_mart",
        "main_strata": ["municipality_code", "calendar_year", "calendar_month", "day_of_week"],
        "main_controls": ["calendar_year", "calendar_month", "day_of_week"],
        "robustness": [
            {
                "model_label": "epiweek_robustness_proxy",
                "model_variant": "robustness",
                "strata": ["municipality_code", "calendar_year", "epidemiologic_week"],
                "seasonality_controls": ["calendar_year", "epiweek"],
            }
        ],
        "model_kind": "count",
        "family_exposures": {
            "family_1_tmax_primary": {
                "exposure_variable": "maximum_daily_temperature_c_lag0_1_mean",
                "exposure_label": "Maximum daily temperature (lag 0 to 1 mean)",
            },
            "family_4_heat_index_parallel": {
                "exposure_variable": "approximate_heat_index_c_lag0_1_mean",
                "exposure_label": "Approximate heat index (lag 0 to 1 mean)",
            },
        },
    },
    "maternal_daily_standard": {
        "mart_name": "maternal_operational_mart",
        "main_strata": ["municipality_code", "calendar_year", "calendar_month", "day_of_week"],
        "main_controls": ["calendar_year", "calendar_month", "day_of_week"],
        "robustness": [],
        "model_kind": "count",
        "family_exposures": {
            "family_1_tmax_primary": {
                "exposure_variable": "maximum_daily_temperature_c_days_0_6_before_event_mean",
                "exposure_label": "Maximum daily temperature (days 0 to 6 before event mean)",
            },
            "family_4_heat_index_parallel": {
                "exposure_variable": "approximate_heat_index_c_days_0_6_before_event_mean",
                "exposure_label": "Approximate heat index (days 0 to 6 before event mean)",
            },
        },
    },
    "perinatal_delivery_aligned": {
        "mart_name": "perinatal_delivery_window_mart",
        "main_strata": ["municipality_code", "calendar_year", "calendar_month"],
        "main_controls": ["calendar_year", "event_or_delivery_month"],
        "robustness": [],
        "model_kind": "count",
        "family_exposures": {
            "family_1_tmax_primary": {
                "exposure_variable": "maximum_daily_temperature_c_days_0_6_before_event_mean",
                "exposure_label": "Maximum daily temperature (days 0 to 6 before event mean)",
            },
            "family_4_heat_index_parallel": {
                "exposure_variable": "approximate_heat_index_c_days_0_6_before_event_mean",
                "exposure_label": "Approximate heat index (days 0 to 6 before event mean)",
            },
        },
    },
}

SPECIAL_ANALYSIS_UNIT_SPECS = {
    "gestational_days_lost_below_37_weeks": {
        "model_kind": "severity",
        "denominator_column_override": "daily_live_birth_count",
        "estimate_scale": SEVERITY_ESTIMATE_SCALE,
        "denominator_contract_note": "supplementary severity proxy weighted by daily live births rather than a count offset",
    },
}

FALLBACK_BASELINE_PROFILE_BY_OUTCOME_FAMILY = {
    "acute_daily": "acute_daily_standard",
    "maternal_operational": "maternal_daily_standard",
    "perinatal_delivery_window_aggregate": "perinatal_delivery_aligned",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _optional_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def _bool_like(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _scaffold_step2_publication_workspace(root: Path) -> dict[str, object]:
    workspace_root = incremental_publication_root(root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    created_files: list[str] = []
    existing_files: list[str] = []
    for relative_path, contents in STEP2_PUBLICATION_FILES.items():
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


def _exposure_scale_value(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 1.0
    q25 = float(numeric.quantile(0.25))
    q75 = float(numeric.quantile(0.75))
    iqr = q75 - q25
    if math.isfinite(iqr) and iqr > 0:
        return iqr
    standard_deviation = float(numeric.std(ddof=0))
    if math.isfinite(standard_deviation) and standard_deviation > 0:
        return standard_deviation
    return 1.0


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


def _pooled_continuous_fe_proxy(
    frame: pd.DataFrame,
    exposure_column: str,
    outcome_column: str,
    denominator_column: str,
    stratum_columns: list[str],
    model_kind: str,
) -> dict[str, Any]:
    valid = _baseline_valid_rows(frame, denominator_column, exposure_column)
    if valid.empty:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "p_value_floor_applied_flag": "no",
            "valid_strata_count": 0,
            "observation_count": 0,
            "outcome_sum": 0.0,
            "denominator_sum": 0.0,
            "overdispersion_ratio": float("nan"),
            "internal_slope": float("nan"),
            "internal_slope_se": float("nan"),
            "exposure_scale_value": 1.0,
        }

    valid["x"] = valid[exposure_column].astype(float)
    valid["w"] = valid[denominator_column].astype(float)
    exposure_scale_value = _exposure_scale_value(valid["x"])
    if model_kind == "count":
        valid["y"] = np.log((valid[outcome_column].astype(float) + 0.5) / valid[denominator_column].astype(float))
    else:
        valid["y"] = valid[outcome_column].astype(float) / valid[denominator_column].astype(float)

    valid["wx"] = valid["w"] * valid["x"]
    valid["wy"] = valid["w"] * valid["y"]
    valid["wxx"] = valid["w"] * valid["x"] * valid["x"]
    valid["wxy"] = valid["w"] * valid["x"] * valid["y"]
    valid["wyy"] = valid["w"] * valid["y"] * valid["y"]

    grouped = valid.groupby(stratum_columns, as_index=False).agg(
        observation_count=("x", "size"),
        sw=("w", "sum"),
        swx=("wx", "sum"),
        swy=("wy", "sum"),
        swxx=("wxx", "sum"),
        swxy=("wxy", "sum"),
        swyy=("wyy", "sum"),
        outcome_sum=(outcome_column, "sum"),
        denominator_sum=(denominator_column, "sum"),
    )
    grouped["det"] = (grouped["sw"] * grouped["swxx"]) - (grouped["swx"] ** 2)
    grouped["degrees_of_freedom"] = grouped["observation_count"] - 2
    grouped = grouped.loc[
        grouped["observation_count"].ge(3)
        & grouped["det"].gt(0)
        & grouped["degrees_of_freedom"].gt(0)
    ].copy()
    if grouped.empty:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "p_value_floor_applied_flag": "no",
            "valid_strata_count": 0,
            "observation_count": int(valid.shape[0]),
            "outcome_sum": float(valid[outcome_column].sum()),
            "denominator_sum": float(valid[denominator_column].sum()),
            "overdispersion_ratio": float("nan"),
            "internal_slope": float("nan"),
            "internal_slope_se": float("nan"),
            "exposure_scale_value": exposure_scale_value,
        }

    grouped["slope"] = ((grouped["sw"] * grouped["swxy"]) - (grouped["swx"] * grouped["swy"])) / grouped["det"]
    grouped["intercept"] = (grouped["swy"] - (grouped["slope"] * grouped["swx"])) / grouped["sw"]
    grouped["weighted_sse"] = grouped["swyy"] - (
        (grouped["intercept"] * grouped["swy"]) + (grouped["slope"] * grouped["swxy"])
    )
    grouped["weighted_sse"] = grouped["weighted_sse"].clip(lower=0.0)
    grouped["sigma_squared"] = grouped["weighted_sse"] / grouped["degrees_of_freedom"]
    grouped["slope_variance"] = (grouped["sigma_squared"] * grouped["sw"]) / grouped["det"]
    grouped = grouped.loc[grouped["slope_variance"].gt(0)].copy()
    if grouped.empty:
        return {
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "p_value_floor_applied_flag": "no",
            "valid_strata_count": 0,
            "observation_count": int(valid.shape[0]),
            "outcome_sum": float(valid[outcome_column].sum()),
            "denominator_sum": float(valid[denominator_column].sum()),
            "overdispersion_ratio": float("nan"),
            "internal_slope": float("nan"),
            "internal_slope_se": float("nan"),
            "exposure_scale_value": exposure_scale_value,
        }

    grouped["weight"] = 1.0 / grouped["slope_variance"]
    pooled_slope = float(np.average(grouped["slope"], weights=grouped["weight"]))
    pooled_se = float(np.sqrt(1.0 / grouped["weight"].sum()))
    z_value = pooled_slope / pooled_se if pooled_se > 0 else float("nan")
    p_value, p_floor_flag = _normal_two_sided_p_from_z(z_value)
    overdispersion_ratio = float(grouped["weighted_sse"].sum() / grouped["degrees_of_freedom"].sum())

    scaled_slope = pooled_slope * exposure_scale_value
    scaled_se = pooled_se * exposure_scale_value
    if model_kind == "count":
        estimate = math.exp(scaled_slope)
        lower_ci = math.exp(scaled_slope - (1.96 * scaled_se))
        upper_ci = math.exp(scaled_slope + (1.96 * scaled_se))
    else:
        estimate = scaled_slope
        lower_ci = scaled_slope - (1.96 * scaled_se)
        upper_ci = scaled_slope + (1.96 * scaled_se)

    return {
        "estimate": float(estimate),
        "lower_ci": float(lower_ci),
        "upper_ci": float(upper_ci),
        "p_value": float(p_value),
        "p_value_floor_applied_flag": p_floor_flag,
        "valid_strata_count": int(grouped.shape[0]),
        "observation_count": int(grouped["observation_count"].sum()),
        "outcome_sum": float(grouped["outcome_sum"].sum()),
        "denominator_sum": float(grouped["denominator_sum"].sum()),
        "overdispersion_ratio": overdispersion_ratio,
        "internal_slope": pooled_slope,
        "internal_slope_se": pooled_se,
        "exposure_scale_value": exposure_scale_value,
    }


def _dispersion_adjusted_result(base_result: dict[str, Any], model_kind: str) -> dict[str, Any]:
    phi = max(float(base_result["overdispersion_ratio"]), 1.0)
    scale_value = float(base_result.get("exposure_scale_value", 1.0))
    adjusted_se = float(base_result["internal_slope_se"]) * math.sqrt(phi)
    scaled_slope = float(base_result["internal_slope"]) * scale_value
    scaled_adjusted_se = adjusted_se * scale_value
    z_value = scaled_slope / scaled_adjusted_se if scaled_adjusted_se > 0 else float("nan")
    p_value, p_floor_flag = _normal_two_sided_p_from_z(z_value)
    if model_kind == "count":
        estimate = math.exp(scaled_slope)
        lower_ci = math.exp(scaled_slope - (1.96 * scaled_adjusted_se))
        upper_ci = math.exp(scaled_slope + (1.96 * scaled_adjusted_se))
    else:
        estimate = scaled_slope
        lower_ci = scaled_slope - (1.96 * scaled_adjusted_se)
        upper_ci = scaled_slope + (1.96 * scaled_adjusted_se)
    return {
        **base_result,
        "estimate": float(estimate),
        "lower_ci": float(lower_ci),
        "upper_ci": float(upper_ci),
        "p_value": float(p_value),
        "p_value_floor_applied_flag": p_floor_flag,
        "internal_slope_se": adjusted_se,
        "exposure_scale_value": scale_value,
    }


def _load_step1_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    step1_root = analysis_step1_root(root)
    analysis_units = pd.read_csv(step1_root / "analysis_unit_roster.csv")
    sparse_flags = pd.read_csv(step1_root / "sparse_outcome_flags.csv")
    return analysis_units, sparse_flags


def _build_seasonality_lookup(seasonality_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(profile["seasonality_profile_id"]): dict(profile)
        for profile in seasonality_payload["profiles"]
    }


def _build_base_candidate_roster(
    analysis_units: pd.DataFrame,
    sparse_flags: pd.DataFrame,
    backlog_review_analysis_units: set[str],
) -> pd.DataFrame:
    sparse_subset = sparse_flags.loc[
        :,
        ["analysis_unit_id", "recommendation", "baseline_model_support_status", "any_sparse_flag"],
    ].drop_duplicates(subset=["analysis_unit_id"])

    promoted = analysis_units.loc[analysis_units["promoted_outcome_flag"].eq(1)].copy()
    main_candidates = promoted.merge(sparse_subset, on="analysis_unit_id", how="left")
    main_candidates = main_candidates.loc[
        main_candidates["recommendation"].astype(str).eq("proceed_to_step2")
    ].copy()
    main_candidates["candidate_track"] = "main_round"
    main_candidates["candidate_origin"] = "promoted_round_candidate"

    backlog_candidates = analysis_units.loc[
        analysis_units["analysis_unit_id"].astype(str).isin(backlog_review_analysis_units)
        & analysis_units["analysis_role"].astype(str).eq("descriptive_only")
    ].copy()
    if not backlog_candidates.empty:
        backlog_candidates = backlog_candidates.merge(sparse_subset, on="analysis_unit_id", how="left")
        backlog_candidates["candidate_track"] = "backlog_review"
        backlog_candidates["candidate_origin"] = "explicit_backlog_review"

    merged = pd.concat([main_candidates, backlog_candidates], ignore_index=True, sort=False)
    if merged.empty:
        return pd.DataFrame()

    merged["candidate_track_priority"] = merged["candidate_track"].map({"main_round": 0, "backlog_review": 1}).fillna(9)
    merged = merged.sort_values(["candidate_track_priority", "analysis_unit_id"], ignore_index=True)
    merged = merged.drop_duplicates(subset=["analysis_unit_id"], keep="first").drop(columns=["candidate_track_priority"])
    return merged.sort_values(
        ["candidate_track", "analysis_role", "outcome_family", "analysis_unit_id"],
        ignore_index=True,
    )


def _expand_candidate_roster(
    base_candidate_roster: pd.DataFrame,
    seasonality_lookup: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    if base_candidate_roster.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for record in base_candidate_roster.to_dict(orient="records"):
        analysis_unit_id = str(record["analysis_unit_id"])
        seasonality_profile_id = str(record["seasonality_profile_id"])
        profile_spec = BASELINE_PROFILE_SPECS.get(seasonality_profile_id)
        if profile_spec is None:
            fallback_profile_id = FALLBACK_BASELINE_PROFILE_BY_OUTCOME_FAMILY.get(str(record["outcome_family"]))
            if fallback_profile_id is not None:
                seasonality_profile_id = fallback_profile_id
                profile_spec = BASELINE_PROFILE_SPECS.get(seasonality_profile_id)
        if profile_spec is None:
            continue
        special_spec = SPECIAL_ANALYSIS_UNIT_SPECS.get(analysis_unit_id, {})
        model_kind = str(special_spec.get("model_kind", profile_spec["model_kind"]))
        denominator_column = str(
            special_spec.get("denominator_column_override")
            or _optional_string(record.get("denominator_column"))
        )
        estimate_scale = str(
            special_spec.get(
                "estimate_scale",
                SEVERITY_ESTIMATE_SCALE if model_kind == "severity" else COUNT_ESTIMATE_SCALE,
            )
        )
        denominator_contract_note = str(
            special_spec.get("denominator_contract_note")
            or ("contract_offset_preserved" if model_kind == "count" else "non_count_model")
        )
        seasonality_profile = seasonality_lookup.get(seasonality_profile_id, {})
        for exposure_family_id, exposure_spec in profile_spec["family_exposures"].items():
            family_spec = EXPOSURE_FAMILY_SPECS[exposure_family_id]
            rows.append(
                {
                    **record,
                    "seasonality_profile_id": seasonality_profile_id,
                    "mart_name": str(profile_spec["mart_name"]),
                    "model_kind": model_kind,
                    "exposure_family_id": exposure_family_id,
                    "exposure_family_name": str(family_spec["exposure_family_name"]),
                    "exposure_variable": str(exposure_spec["exposure_variable"]),
                    "exposure_label": str(exposure_spec["exposure_label"]),
                    "headline_eligible_flag": str(family_spec["headline_eligible_flag"]),
                    "baseline_strata": ";".join(str(value) for value in profile_spec["main_strata"]),
                    "seasonality_controls_applied": ";".join(str(value) for value in profile_spec["main_controls"]),
                    "seasonality_notes": str(seasonality_profile.get("notes", "")),
                    "denominator_column_for_step2": denominator_column,
                    "estimate_scale": estimate_scale,
                    "denominator_contract_note": denominator_contract_note,
                    "candidate_track": str(record.get("candidate_track") or "main_round"),
                    "candidate_origin": str(record.get("candidate_origin") or "promoted_round_candidate"),
                    "step1_recommendation": str(record.get("recommendation") or ""),
                    "step1_baseline_model_support_status": str(record.get("baseline_model_support_status") or ""),
                    "step1_any_sparse_flag": bool(_bool_like(record.get("any_sparse_flag"))),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["candidate_track", "analysis_role", "outcome_family", "analysis_unit_id", "exposure_family_id"],
        ignore_index=True,
    )


def _required_columns_by_mart(candidate_roster: pd.DataFrame) -> dict[str, set[str]]:
    required: dict[str, set[str]] = {}
    base_columns = {
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
    for mart_name, group in candidate_roster.groupby("mart_name", sort=True):
        required[str(mart_name)] = set(base_columns)
        for record in group.to_dict(orient="records"):
            required[str(mart_name)].add(str(record["analysis_unit_id"]))
            required[str(mart_name)].add(str(record["exposure_variable"]))
            denominator_column = str(record.get("denominator_column_for_step2") or "")
            if denominator_column:
                required[str(mart_name)].add(denominator_column)
    return required


def _missing_mart_columns(root: Path, years: list[int], required_columns_by_mart: dict[str, set[str]]) -> list[str]:
    missing: list[str] = []
    for mart_name, required_columns in required_columns_by_mart.items():
        for year in years:
            path = analysis_shared_marts_root(root) / mart_name / f"YEAR={year}" / f"{mart_name}_{year}.parquet"
            if not path.exists():
                missing.append(_relative(root, path))
                continue
            available_columns = set(pq.ParquetFile(path).schema.names)
            for column in sorted(required_columns):
                if column not in available_columns:
                    missing.append(f"{_relative(root, path)}:{column}")
    return missing


def _missing_active_backlog_columns(root: Path, years: list[int], analysis_units: pd.DataFrame) -> list[str]:
    required_by_mart: dict[str, set[str]] = {}
    active_units = analysis_units.loc[
        analysis_units["analysis_role"].astype(str).isin(["primary", "sensitivity", "descriptive_only"])
    ].copy()
    for mart_name, group in active_units.groupby("mart_name", sort=True):
        required_by_mart[str(mart_name)] = {
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
            required_by_mart[str(mart_name)].add(str(record["analysis_unit_id"]))
            denominator_column = _optional_string(record.get("denominator_column"))
            if denominator_column:
                required_by_mart[str(mart_name)].add(denominator_column)
    return _missing_mart_columns(root, years, required_by_mart)


def _load_model_frames(root: Path, years: list[int], required_columns_by_mart: dict[str, set[str]]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for mart_name, required_columns in required_columns_by_mart.items():
        partitions: list[pd.DataFrame] = []
        selected_columns = sorted(required_columns)
        for year in years:
            path = analysis_shared_marts_root(root) / mart_name / f"YEAR={year}" / f"{mart_name}_{year}.parquet"
            partition = pd.read_parquet(path, columns=selected_columns)
            partition["observation_date"] = pd.to_datetime(partition["observation_date"], errors="coerce")
            if "calendar_month" not in partition.columns:
                partition["calendar_month"] = partition["observation_date"].dt.month.astype("Int64")
            partition["day_of_week"] = partition["observation_date"].dt.dayofweek.astype("Int64")
            partition["epidemiologic_week"] = partition["observation_date"].dt.isocalendar().week.astype("Int64")
            partitions.append(partition)
        frames[mart_name] = pd.concat(partitions, ignore_index=True)
    return frames


def _finite_estimate(record: dict[str, Any]) -> bool:
    return all(math.isfinite(float(record[key])) for key in ("estimate", "lower_ci", "upper_ci"))


def _fit_candidate_models(
    candidate_roster: pd.DataFrame,
    frames_by_mart: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    specification_rows: list[dict[str, Any]] = []

    for record in candidate_roster.to_dict(orient="records"):
        analysis_unit_id = str(record["analysis_unit_id"])
        mart_name = str(record["mart_name"])
        profile_spec = BASELINE_PROFILE_SPECS[str(record["seasonality_profile_id"])]
        frame = frames_by_mart[mart_name]
        model_kind = str(record["model_kind"])
        denominator_column = str(record["denominator_column_for_step2"])
        exposure_column = str(record["exposure_variable"])

        main_result = _pooled_continuous_fe_proxy(
            frame=frame,
            exposure_column=exposure_column,
            outcome_column=analysis_unit_id,
            denominator_column=denominator_column,
            stratum_columns=list(profile_spec["main_strata"]),
            model_kind=model_kind,
        )
        preferred_distribution = "gaussian" if model_kind == "severity" else (
            "negative_binomial_proxy"
            if float(main_result.get("overdispersion_ratio", float("nan"))) > OVERDISPERSION_THRESHOLD
            else "poisson_proxy"
        )
        main_model_label = "gaussian_fe_proxy" if model_kind == "severity" else "poisson_fe_proxy"

        specification_rows.append(
            {
                "analysis_unit_id": analysis_unit_id,
                "linked_outcome_ids": record["linked_outcome_ids"],
                "candidate_track": record["candidate_track"],
                "analysis_role": record["analysis_role"],
                "outcome_family": record["outcome_family"],
                "mart_name": mart_name,
                "model_variant": "main",
                "model_label": main_model_label,
                "model_kind": model_kind,
                "exposure_family_id": record["exposure_family_id"],
                "exposure_family_name": record["exposure_family_name"],
                "exposure_variable": exposure_column,
                "exposure_label": record["exposure_label"],
                "headline_eligible_flag": record["headline_eligible_flag"],
                "denominator_column": denominator_column,
                "denominator_id": record["denominator_id"],
                "denominator_contract_note": record["denominator_contract_note"],
                "seasonality_profile_id": record["seasonality_profile_id"],
                "seasonality_controls_applied": record["seasonality_controls_applied"],
                "seasonality_notes": record["seasonality_notes"],
                "stratum_columns": ";".join(str(value) for value in profile_spec["main_strata"]),
                "preferred_distribution": preferred_distribution,
            }
        )
        result_rows.append(
            {
                "analysis_unit_id": analysis_unit_id,
                "linked_outcome_ids": record["linked_outcome_ids"],
                "candidate_track": record["candidate_track"],
                "analysis_role": record["analysis_role"],
                "outcome_family": record["outcome_family"],
                "mart_name": mart_name,
                "model_variant": "main",
                "model_label": main_model_label,
                "preferred_model_flag": bool(preferred_distribution == ("gaussian" if model_kind == "severity" else "poisson_proxy")),
                "exposure_family_id": record["exposure_family_id"],
                "exposure_family_name": record["exposure_family_name"],
                "exposure_variable": exposure_column,
                "exposure_label": record["exposure_label"],
                "headline_eligible_flag": record["headline_eligible_flag"],
                "denominator_column": denominator_column,
                "denominator_id": record["denominator_id"],
                "denominator_contract_note": record["denominator_contract_note"],
                "estimate_scale": record["estimate_scale"],
                "seasonality_profile_id": record["seasonality_profile_id"],
                "seasonality_controls_applied": record["seasonality_controls_applied"],
                **main_result,
            }
        )
        diagnostic_rows.append(
            {
                "analysis_unit_id": analysis_unit_id,
                "candidate_track": record["candidate_track"],
                "analysis_role": record["analysis_role"],
                "outcome_family": record["outcome_family"],
                "exposure_family_id": record["exposure_family_id"],
                "exposure_family_name": record["exposure_family_name"],
                "exposure_variable": exposure_column,
                "model_variant": "main",
                "model_label": main_model_label,
                "preferred_distribution": preferred_distribution,
                "seasonality_controls_applied": record["seasonality_controls_applied"],
                "valid_strata_count": main_result.get("valid_strata_count", 0),
                "observation_count": main_result.get("observation_count", 0),
                "outcome_sum": main_result.get("outcome_sum", 0.0),
                "denominator_sum": main_result.get("denominator_sum", 0.0),
                "internal_slope": main_result.get("internal_slope", float("nan")),
                "internal_slope_se": main_result.get("internal_slope_se", float("nan")),
                "overdispersion_ratio": main_result.get("overdispersion_ratio", float("nan")),
            }
        )

        if model_kind == "count" and float(main_result.get("overdispersion_ratio", float("nan"))) > OVERDISPERSION_THRESHOLD:
            nb_result = _dispersion_adjusted_result(main_result, model_kind=model_kind)
            specification_rows.append(
                {
                    "analysis_unit_id": analysis_unit_id,
                    "linked_outcome_ids": record["linked_outcome_ids"],
                    "candidate_track": record["candidate_track"],
                    "analysis_role": record["analysis_role"],
                    "outcome_family": record["outcome_family"],
                    "mart_name": mart_name,
                    "model_variant": "overdispersion_adjusted",
                    "model_label": "negative_binomial_fe_proxy",
                    "model_kind": model_kind,
                    "exposure_family_id": record["exposure_family_id"],
                    "exposure_family_name": record["exposure_family_name"],
                    "exposure_variable": exposure_column,
                    "exposure_label": record["exposure_label"],
                    "headline_eligible_flag": record["headline_eligible_flag"],
                    "denominator_column": denominator_column,
                    "denominator_id": record["denominator_id"],
                    "denominator_contract_note": record["denominator_contract_note"],
                    "seasonality_profile_id": record["seasonality_profile_id"],
                    "seasonality_controls_applied": record["seasonality_controls_applied"],
                    "seasonality_notes": record["seasonality_notes"],
                    "stratum_columns": ";".join(str(value) for value in profile_spec["main_strata"]),
                    "preferred_distribution": "negative_binomial_proxy",
                }
            )
            result_rows.append(
                {
                    "analysis_unit_id": analysis_unit_id,
                    "linked_outcome_ids": record["linked_outcome_ids"],
                    "candidate_track": record["candidate_track"],
                    "analysis_role": record["analysis_role"],
                    "outcome_family": record["outcome_family"],
                    "mart_name": mart_name,
                    "model_variant": "overdispersion_adjusted",
                    "model_label": "negative_binomial_fe_proxy",
                    "preferred_model_flag": True,
                    "exposure_family_id": record["exposure_family_id"],
                    "exposure_family_name": record["exposure_family_name"],
                    "exposure_variable": exposure_column,
                    "exposure_label": record["exposure_label"],
                    "headline_eligible_flag": record["headline_eligible_flag"],
                    "denominator_column": denominator_column,
                    "denominator_id": record["denominator_id"],
                    "denominator_contract_note": record["denominator_contract_note"],
                    "estimate_scale": record["estimate_scale"],
                    "seasonality_profile_id": record["seasonality_profile_id"],
                    "seasonality_controls_applied": record["seasonality_controls_applied"],
                    **nb_result,
                }
            )
            diagnostic_rows.append(
                {
                    "analysis_unit_id": analysis_unit_id,
                    "candidate_track": record["candidate_track"],
                    "analysis_role": record["analysis_role"],
                    "outcome_family": record["outcome_family"],
                    "exposure_family_id": record["exposure_family_id"],
                    "exposure_family_name": record["exposure_family_name"],
                    "exposure_variable": exposure_column,
                    "model_variant": "overdispersion_adjusted",
                    "model_label": "negative_binomial_fe_proxy",
                    "preferred_distribution": "negative_binomial_proxy",
                    "seasonality_controls_applied": record["seasonality_controls_applied"],
                    "valid_strata_count": nb_result.get("valid_strata_count", 0),
                    "observation_count": nb_result.get("observation_count", 0),
                    "outcome_sum": nb_result.get("outcome_sum", 0.0),
                    "denominator_sum": nb_result.get("denominator_sum", 0.0),
                    "internal_slope": nb_result.get("internal_slope", float("nan")),
                    "internal_slope_se": nb_result.get("internal_slope_se", float("nan")),
                    "overdispersion_ratio": nb_result.get("overdispersion_ratio", float("nan")),
                }
            )

        for robustness_spec in profile_spec["robustness"]:
            robustness_result = _pooled_continuous_fe_proxy(
                frame=frame,
                exposure_column=exposure_column,
                outcome_column=analysis_unit_id,
                denominator_column=denominator_column,
                stratum_columns=list(robustness_spec["strata"]),
                model_kind=model_kind,
            )
            specification_rows.append(
                {
                    "analysis_unit_id": analysis_unit_id,
                    "linked_outcome_ids": record["linked_outcome_ids"],
                    "candidate_track": record["candidate_track"],
                    "analysis_role": record["analysis_role"],
                    "outcome_family": record["outcome_family"],
                    "mart_name": mart_name,
                    "model_variant": str(robustness_spec["model_variant"]),
                    "model_label": str(robustness_spec["model_label"]),
                    "model_kind": model_kind,
                    "exposure_family_id": record["exposure_family_id"],
                    "exposure_family_name": record["exposure_family_name"],
                    "exposure_variable": exposure_column,
                    "exposure_label": record["exposure_label"],
                    "headline_eligible_flag": record["headline_eligible_flag"],
                    "denominator_column": denominator_column,
                    "denominator_id": record["denominator_id"],
                    "denominator_contract_note": record["denominator_contract_note"],
                    "seasonality_profile_id": record["seasonality_profile_id"],
                    "seasonality_controls_applied": ";".join(str(value) for value in robustness_spec["seasonality_controls"]),
                    "seasonality_notes": record["seasonality_notes"],
                    "stratum_columns": ";".join(str(value) for value in robustness_spec["strata"]),
                    "preferred_distribution": "robustness_proxy",
                }
            )
            result_rows.append(
                {
                    "analysis_unit_id": analysis_unit_id,
                    "linked_outcome_ids": record["linked_outcome_ids"],
                    "candidate_track": record["candidate_track"],
                    "analysis_role": record["analysis_role"],
                    "outcome_family": record["outcome_family"],
                    "mart_name": mart_name,
                    "model_variant": str(robustness_spec["model_variant"]),
                    "model_label": str(robustness_spec["model_label"]),
                    "preferred_model_flag": False,
                    "exposure_family_id": record["exposure_family_id"],
                    "exposure_family_name": record["exposure_family_name"],
                    "exposure_variable": exposure_column,
                    "exposure_label": record["exposure_label"],
                    "headline_eligible_flag": record["headline_eligible_flag"],
                    "denominator_column": denominator_column,
                    "denominator_id": record["denominator_id"],
                    "denominator_contract_note": record["denominator_contract_note"],
                    "estimate_scale": record["estimate_scale"],
                    "seasonality_profile_id": record["seasonality_profile_id"],
                    "seasonality_controls_applied": ";".join(str(value) for value in robustness_spec["seasonality_controls"]),
                    **robustness_result,
                }
            )
            diagnostic_rows.append(
                {
                    "analysis_unit_id": analysis_unit_id,
                    "candidate_track": record["candidate_track"],
                    "analysis_role": record["analysis_role"],
                    "outcome_family": record["outcome_family"],
                    "exposure_family_id": record["exposure_family_id"],
                    "exposure_family_name": record["exposure_family_name"],
                    "exposure_variable": exposure_column,
                    "model_variant": str(robustness_spec["model_variant"]),
                    "model_label": str(robustness_spec["model_label"]),
                    "preferred_distribution": "robustness_proxy",
                    "seasonality_controls_applied": ";".join(str(value) for value in robustness_spec["seasonality_controls"]),
                    "valid_strata_count": robustness_result.get("valid_strata_count", 0),
                    "observation_count": robustness_result.get("observation_count", 0),
                    "outcome_sum": robustness_result.get("outcome_sum", 0.0),
                    "denominator_sum": robustness_result.get("denominator_sum", 0.0),
                    "internal_slope": robustness_result.get("internal_slope", float("nan")),
                    "internal_slope_se": robustness_result.get("internal_slope_se", float("nan")),
                    "overdispersion_ratio": robustness_result.get("overdispersion_ratio", float("nan")),
                }
            )

    results = pd.DataFrame(result_rows).sort_values(
        [
            "candidate_track",
            "analysis_role",
            "outcome_family",
            "analysis_unit_id",
            "exposure_family_id",
            "model_variant",
            "model_label",
        ],
        ignore_index=True,
    )
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(
        ["candidate_track", "analysis_unit_id", "exposure_family_id", "model_variant", "model_label"],
        ignore_index=True,
    )
    specifications = pd.DataFrame(specification_rows).sort_values(
        ["candidate_track", "analysis_unit_id", "exposure_family_id", "model_variant", "model_label"],
        ignore_index=True,
    )
    return results, diagnostics, specifications


def _preferred_model_frame(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    preferred = results.loc[results["preferred_model_flag"].eq(True)].copy()
    if preferred.empty:
        preferred = results.loc[results["model_variant"].astype(str).eq("main")].copy()
    return preferred.sort_values(
        ["candidate_track", "analysis_role", "outcome_family", "analysis_unit_id", "exposure_family_id"],
        ignore_index=True,
    ).drop_duplicates(subset=["analysis_unit_id", "candidate_track", "exposure_family_id"], keep="first")


def _estimate_direction(estimate: Any, estimate_scale: str) -> str:
    value = float(estimate) if estimate is not None and not pd.isna(estimate) else float("nan")
    if not math.isfinite(value):
        return ""
    if estimate_scale == SEVERITY_ESTIMATE_SCALE:
        if value > 0:
            return "positive"
        if value < 0:
            return "negative"
        return "null"
    if value > 1:
        return "positive"
    if value < 1:
        return "negative"
    return "null"


def _family_rank_key(row: dict[str, Any]) -> tuple[int, int, float, float]:
    estimable = 1 if bool(row.get("estimable_flag")) else 0
    valid_strata = int(row.get("valid_strata_count") or 0)
    p_value = float(row.get("p_value") if row.get("p_value") is not None else float("nan"))
    if not math.isfinite(p_value):
        p_value = 1.0
    return (estimable, valid_strata, -p_value, -abs(float(row.get("estimate") or 0.0)))


def _build_exposure_family_comparison(
    base_candidate_roster: pd.DataFrame,
    preferred_models: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    if base_candidate_roster.empty:
        return pd.DataFrame()

    preferred_lookup = {
        (str(row["analysis_unit_id"]), str(row["candidate_track"]), str(row["exposure_family_id"])): row
        for row in preferred_models.to_dict(orient="records")
    } if not preferred_models.empty else {}
    diagnostic_lookup = {
        (str(row["analysis_unit_id"]), str(row["candidate_track"]), str(row["exposure_family_id"])): row
        for row in diagnostics.loc[diagnostics["model_variant"].astype(str).eq("main")].to_dict(orient="records")
    } if (not diagnostics.empty and "model_variant" in diagnostics.columns) else {}

    rows: list[dict[str, Any]] = []
    for record in base_candidate_roster.to_dict(orient="records"):
        family_rows: dict[str, dict[str, Any]] = {}
        for exposure_family_id in EXPOSURE_FAMILY_SPECS:
            preferred = preferred_lookup.get((str(record["analysis_unit_id"]), str(record["candidate_track"]), exposure_family_id), {})
            diagnostic = diagnostic_lookup.get((str(record["analysis_unit_id"]), str(record["candidate_track"]), exposure_family_id), {})
            estimate = preferred.get("estimate", float("nan"))
            valid_strata_count = int(preferred.get("valid_strata_count", 0) or 0)
            estimable_flag = bool(math.isfinite(float(estimate)) and valid_strata_count > 0)
            family_rows[exposure_family_id] = {
                "estimable_flag": estimable_flag,
                "estimate": float(estimate) if estimate is not None and not pd.isna(estimate) else float("nan"),
                "p_value": float(preferred.get("p_value", float("nan"))),
                "valid_strata_count": valid_strata_count,
                "preferred_model_label": str(preferred.get("model_label") or ""),
                "preferred_model_variant": str(preferred.get("model_variant") or ""),
                "overdispersion_ratio": float(diagnostic.get("overdispersion_ratio", float("nan"))),
                "estimate_direction": _estimate_direction(preferred.get("estimate"), str(preferred.get("estimate_scale") or COUNT_ESTIMATE_SCALE)),
            }

        estimable_families = [family_id for family_id, value in family_rows.items() if value["estimable_flag"]]
        direction_consistency = True
        finite_directions = [value["estimate_direction"] for value in family_rows.values() if value["estimate_direction"]]
        if len(set(finite_directions)) > 1:
            direction_consistency = False

        strongest_family = ""
        if estimable_families:
            strongest_family = max(
                estimable_families,
                key=lambda family_id: _family_rank_key(family_rows[family_id]),
            )

        row = {
            "analysis_unit_id": str(record["analysis_unit_id"]),
            "linked_outcome_ids": str(record["linked_outcome_ids"]),
            "candidate_track": str(record["candidate_track"]),
            "analysis_role": str(record["analysis_role"]),
            "outcome_family": str(record["outcome_family"]),
            "estimable_exposure_family_count": int(len(estimable_families)),
            "estimable_exposure_family_ids": ";".join(estimable_families),
            "effect_direction_consistency_flag": bool(direction_consistency),
            "strongest_exposure_family_id": strongest_family,
            "strongest_exposure_family_name": str(EXPOSURE_FAMILY_SPECS.get(strongest_family, {}).get("exposure_family_name", "")),
        }
        for exposure_family_id in EXPOSURE_FAMILY_SPECS:
            prefix = "tmax" if exposure_family_id == "family_1_tmax_primary" else "heat_index"
            row[f"{prefix}_estimable_flag"] = bool(family_rows[exposure_family_id]["estimable_flag"])
            row[f"{prefix}_estimate"] = family_rows[exposure_family_id]["estimate"]
            row[f"{prefix}_p_value"] = family_rows[exposure_family_id]["p_value"]
            row[f"{prefix}_valid_strata_count"] = family_rows[exposure_family_id]["valid_strata_count"]
            row[f"{prefix}_preferred_model_label"] = family_rows[exposure_family_id]["preferred_model_label"]
            row[f"{prefix}_preferred_model_variant"] = family_rows[exposure_family_id]["preferred_model_variant"]
            row[f"{prefix}_overdispersion_ratio"] = family_rows[exposure_family_id]["overdispersion_ratio"]
            row[f"{prefix}_estimate_direction"] = family_rows[exposure_family_id]["estimate_direction"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["candidate_track", "analysis_role", "analysis_unit_id"],
        ignore_index=True,
    )


def _build_promotion_decisions(
    base_candidate_roster: pd.DataFrame,
    sparse_flags: pd.DataFrame,
    exposure_family_comparison: pd.DataFrame,
) -> pd.DataFrame:
    if base_candidate_roster.empty:
        return pd.DataFrame(
            columns=[
                "analysis_unit_id",
                "linked_outcome_ids",
                "candidate_track",
                "analysis_role",
                "step1_recommendation",
                "baseline_model_support_status",
                "any_sparse_flag",
                "estimable_exposure_family_count",
                "estimable_exposure_family_ids",
                "dual_family_step3_flag",
                "single_family_step3_flag",
                "approved_for_next_steps_flag",
                "next_step_route",
                "decision_reason",
            ]
        )

    sparse_subset = sparse_flags.loc[
        :, ["analysis_unit_id", "recommendation", "baseline_model_support_status", "any_sparse_flag"]
    ].drop_duplicates(subset=["analysis_unit_id"])
    merged = base_candidate_roster.merge(sparse_subset, on="analysis_unit_id", how="left", suffixes=("", "_step1"))
    merged = merged.merge(
        exposure_family_comparison.loc[
            :,
            [
                "analysis_unit_id",
                "candidate_track",
                "estimable_exposure_family_count",
                "estimable_exposure_family_ids",
                "effect_direction_consistency_flag",
            ],
        ],
        on=["analysis_unit_id", "candidate_track"],
        how="left",
    )

    rows: list[dict[str, Any]] = []
    for record in merged.to_dict(orient="records"):
        estimable_exposure_family_count = int(record.get("estimable_exposure_family_count") or 0)
        estimable_flag = estimable_exposure_family_count > 0
        any_sparse_flag = _bool_like(record.get("any_sparse_flag"))
        baseline_model_support_status = str(
            record.get("baseline_model_support_status")
            or record.get("step1_baseline_model_support_status")
            or ""
        )
        baseline_support_pass = baseline_model_support_status.upper() == "PASS"
        approved_for_next_steps_flag = bool(estimable_flag and baseline_support_pass and not any_sparse_flag)
        if approved_for_next_steps_flag and str(record["candidate_track"]) == "backlog_review" and estimable_exposure_family_count >= 2:
            decision_reason = "approved_dual_family_from_backlog_review"
        elif approved_for_next_steps_flag and str(record["candidate_track"]) == "backlog_review":
            decision_reason = "approved_single_family_from_backlog_review"
        elif approved_for_next_steps_flag and estimable_exposure_family_count >= 2:
            decision_reason = "retained_main_round_candidate_dual_family"
        elif approved_for_next_steps_flag:
            decision_reason = "retained_main_round_candidate_single_family"
        elif not estimable_flag:
            decision_reason = "not_estimable_in_any_exposure_family"
        elif any_sparse_flag:
            decision_reason = "fails_step1_signal_thresholds"
        else:
            decision_reason = "fails_step2_review_contract"
        rows.append(
            {
                "analysis_unit_id": str(record["analysis_unit_id"]),
                "linked_outcome_ids": str(record["linked_outcome_ids"]),
                "candidate_track": str(record["candidate_track"]),
                "analysis_role": str(record["analysis_role"]),
                "step1_recommendation": str(record.get("recommendation") or record.get("step1_recommendation") or ""),
                "baseline_model_support_status": baseline_model_support_status,
                "any_sparse_flag": any_sparse_flag,
                "estimable_exposure_family_count": estimable_exposure_family_count,
                "estimable_exposure_family_ids": str(record.get("estimable_exposure_family_ids") or ""),
                "effect_direction_consistency_flag": bool(_bool_like(record.get("effect_direction_consistency_flag"))),
                "dual_family_step3_flag": bool(estimable_exposure_family_count >= 2 and approved_for_next_steps_flag),
                "single_family_step3_flag": bool(estimable_exposure_family_count == 1 and approved_for_next_steps_flag),
                "approved_for_next_steps_flag": approved_for_next_steps_flag,
                "next_step_route": STEP2_NEXT_STEP_ROUTE if approved_for_next_steps_flag else "",
                "decision_reason": decision_reason,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["candidate_track", "analysis_role", "analysis_unit_id"],
        ignore_index=True,
    )


def _build_decision_checklist(
    base_candidate_roster: pd.DataFrame,
    expanded_candidate_roster: pd.DataFrame,
    preferred_models: pd.DataFrame,
    diagnostics: pd.DataFrame,
    promotion_decisions: pd.DataFrame,
    exposure_family_comparison: pd.DataFrame,
    gates: dict[str, bool],
) -> pd.DataFrame:
    main_candidates = base_candidate_roster.loc[
        base_candidate_roster["candidate_track"].astype(str).eq("main_round")
    ].copy() if not base_candidate_roster.empty else pd.DataFrame()
    backlog_candidates = base_candidate_roster.loc[
        base_candidate_roster["candidate_track"].astype(str).eq("backlog_review")
    ].copy() if not base_candidate_roster.empty else pd.DataFrame()
    main_estimable_count = int(
        promotion_decisions.loc[
            promotion_decisions["candidate_track"].astype(str).eq("main_round")
            & promotion_decisions["estimable_exposure_family_count"].fillna(0).gt(0)
        ].shape[0]
    ) if not promotion_decisions.empty else 0
    overdispersed_count = int(
        diagnostics.loc[
            diagnostics["model_variant"].astype(str).eq("main")
            & diagnostics["overdispersion_ratio"].fillna(0).gt(OVERDISPERSION_THRESHOLD)
        ].shape[0]
    ) if not diagnostics.empty else 0
    dual_family_approved_count = int(
        promotion_decisions["dual_family_step3_flag"].fillna(False).astype(bool).sum()
    ) if not promotion_decisions.empty else 0
    rows = [
        {
            "checklist_item": "Every Step 2 main-round candidate was estimable in at least one exposure family.",
            "pass_flag": bool(gates["all_main_candidates_estimable_in_at_least_one_family"]),
            "evidence": f"{main_estimable_count}/{int(len(main_candidates))} main-round candidates produced a finite preferred estimate in at least one family.",
        },
        {
            "checklist_item": "Both exposure families were run for every Step 2 candidate.",
            "pass_flag": bool(gates["all_candidates_expanded_to_dual_family"]),
            "evidence": f"{int(len(expanded_candidate_roster))} candidate-family rows were written for {int(len(base_candidate_roster))} candidates.",
        },
        {
            "checklist_item": "Explicit seasonality controls were declared and applied for every baseline specification.",
            "pass_flag": bool(gates["seasonality_controls_declared"]),
            "evidence": f"{int(len(expanded_candidate_roster))} candidate-family rows were routed through a seasonality profile with an explicit control set.",
        },
        {
            "checklist_item": "The descriptive backlog is materialized in the marts.",
            "pass_flag": bool(gates["analysis_backlog_materialized"]),
            "evidence": "All active Step 1 analysis-unit columns resolve in the delivered marts." if gates["analysis_backlog_materialized"] else "Some active Step 1 analysis-unit columns are still missing from the marts.",
        },
        {
            "checklist_item": "Exposure-family comparison output is complete for all Step 2 candidates.",
            "pass_flag": bool(gates["family_comparison_written"]),
            "evidence": f"{int(len(exposure_family_comparison))}/{int(len(base_candidate_roster))} candidate-level family comparisons were written; {dual_family_approved_count} approved candidates advance with both families.",
        },
        {
            "checklist_item": "Overdispersion review was completed for count outcomes.",
            "pass_flag": bool(gates["overdispersion_review_complete"]),
            "evidence": f"{overdispersed_count} main model rows crossed the overdispersion review threshold of {OVERDISPERSION_THRESHOLD}.",
        },
    ]
    if not backlog_candidates.empty:
        backlog_approved_count = int(
            promotion_decisions.loc[
                promotion_decisions["candidate_track"].astype(str).eq("backlog_review")
                & promotion_decisions["approved_for_next_steps_flag"].eq(True)
            ].shape[0]
        ) if not promotion_decisions.empty else 0
        rows.append(
            {
                "checklist_item": "Configured backlog-review outcomes were screened and decisioned.",
                "pass_flag": bool(gates["configured_backlog_review_screened"]),
                "evidence": (
                    f"{int(len(backlog_candidates))}/{int(len(backlog_candidates))} configured backlog-review candidates were tested; "
                    f"{backlog_approved_count} were approved for {STEP2_NEXT_STEP_ROUTE}."
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_step2_report(
    round_config: dict[str, Any],
    step1_report: dict[str, Any],
    gates: dict[str, bool],
    summary: dict[str, Any],
    outputs: dict[str, str],
    workspace_report: dict[str, object],
    base_candidate_roster: pd.DataFrame,
    expanded_candidate_roster: pd.DataFrame,
    promotion_decisions: pd.DataFrame,
    exposure_family_comparison: pd.DataFrame,
    configured_backlog_review_analysis_units: list[str],
    missing_backlog_review_units: list[str],
    untested_backlog_review_units: list[str],
    missing_backlog_columns: list[str],
    required_model_columns_missing: list[str],
) -> str:
    lines = [
        "# Step 2 Baseline Model Report",
        "",
        f"- Generated at: `{_now_utc()}`",
        f"- Analysis ID: `{round_config['analysis_id']}`",
        f"- Analysis title: `{round_config['analysis_title']}`",
        f"- Years: `{', '.join(str(year) for year in round_config['years'])}`",
        f"- Upstream Step 1 status: `{step1_report['status']}`",
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
            f"- Step 2 candidates: `{summary['candidate_count']}`",
            f"- Step 2 candidate-family rows: `{summary['candidate_family_count']}`",
            f"- Primary candidates: `{summary['primary_candidate_count']}`",
            f"- Sensitivity candidates: `{summary['sensitivity_candidate_count']}`",
            f"- Backlog-review candidates: `{summary['backlog_review_candidate_count']}`",
            f"- Backlog-review approvals for next steps: `{summary['backlog_review_approved_count']}`",
            f"- Dual-family approvals for Step 3: `{summary['dual_family_approved_count']}`",
            f"- Single-family approvals for Step 3: `{summary['single_family_approved_count']}`",
            f"- Preferred baseline models written: `{summary['preferred_model_count']}`",
            f"- Robustness models written: `{summary['robustness_model_count']}`",
            f"- Overdispersion-adjusted models written: `{summary['overdispersion_adjusted_count']}`",
            f"- Main-round candidates estimable in at least one family: `{summary['main_estimable_candidate_count']}`",
        ]
    )
    if configured_backlog_review_analysis_units:
        lines.extend(["", "## Backlog Review Scope", ""])
        lines.append(
            f"- Configured backlog-review analysis units: `{', '.join(configured_backlog_review_analysis_units)}`"
        )
        if missing_backlog_review_units:
            for item in missing_backlog_review_units:
                lines.append(f"- Missing from Step 1 roster: `{item}`")
        if untested_backlog_review_units:
            for item in untested_backlog_review_units:
                lines.append(f"- Not tested in Step 2: `{item}`")
    if missing_backlog_columns:
        lines.extend(["", "## Backlog Gaps", ""])
        for item in missing_backlog_columns:
            lines.append(f"- Missing active mart column: `{item}`")
    if required_model_columns_missing:
        lines.extend(["", "## Missing Step 2 Columns", ""])
        for item in required_model_columns_missing:
            lines.append(f"- Missing modeling column: `{item}`")
    if not expanded_candidate_roster.empty:
        lines.extend(["", "## Candidate-Family Roster", ""])
        for record in expanded_candidate_roster.to_dict(orient="records"):
            lines.append(
                "- "
                f"{record['analysis_unit_id']} ({record['analysis_role']}, {record['candidate_track']}, {record['exposure_family_id']}): "
                f"mart=`{record['mart_name']}`, exposure=`{record['exposure_variable']}`, "
                f"seasonality=`{record['seasonality_controls_applied']}`, denominator=`{record['denominator_column_for_step2']}`"
            )
    if not promotion_decisions.empty:
        lines.extend(["", "## Next-Step Promotion Decisions", ""])
        for record in promotion_decisions.to_dict(orient="records"):
            decision = "approve" if bool(record["approved_for_next_steps_flag"]) else "defer"
            route = str(record["next_step_route"] or "stay_in_backlog")
            lines.append(
                "- "
                f"{record['analysis_unit_id']}: `{decision}` "
                f"(families=`{record['estimable_exposure_family_ids'] or 'none'}`, reason=`{record['decision_reason']}`, route=`{route}`)"
            )
    if not exposure_family_comparison.empty:
        lines.extend(["", "## Exposure-Family Comparison", ""])
        for record in exposure_family_comparison.to_dict(orient="records"):
            strongest = str(record["strongest_exposure_family_id"] or "none")
            lines.append(
                "- "
                f"{record['analysis_unit_id']}: estimable_families=`{record['estimable_exposure_family_ids'] or 'none'}`, "
                f"strongest=`{strongest}`, direction_consistency=`{record['effect_direction_consistency_flag']}`"
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


def _write_step2_publication_notes(
    root: Path,
    *,
    status: str,
    summary: dict[str, Any],
    outputs: dict[str, str],
) -> None:
    workspace_root = incremental_publication_root(root)
    (workspace_root / "04_step2_baseline_models.md").write_text(
        (
            "# Step 2 Baseline Models\n\n"
            f"- Status: `{status}`\n"
            f"- Candidates screened: `{summary['candidate_count']}`\n"
            f"- Candidate-family rows: `{summary['candidate_family_count']}`\n"
            f"- Preferred models: `{outputs['preferred_baseline_models']}`\n"
            f"- Technical report: `{outputs['step2_report']}`\n"
        ),
        encoding="utf-8",
    )
    (workspace_root / "05_step2_decision_log.md").write_text(
        (
            "# Step 2 Decision Log\n\n"
            f"- Status: `{status}`\n"
            f"- Dual-family approvals for Step 3: `{summary['dual_family_approved_count']}`\n"
            f"- Single-family approvals for Step 3: `{summary['single_family_approved_count']}`\n"
            f"- Promotion decisions: `{outputs['promotion_decisions']}`\n"
            f"- Exposure-family comparison: `{outputs['exposure_family_comparison']}`\n"
        ),
        encoding="utf-8",
    )


def build_step2_baseline_models(
    root: Path,
    config_path: Path | None = None,
    step1_report: dict[str, Any] | None = None,
) -> dict[str, object]:
    directories = ensure_step2_directory_structure(root)
    round_config = load_round_config(root, config_path=config_path)
    workspace_report = _scaffold_step2_publication_workspace(root)
    upstream_step1_report = step1_report or build_step1_descriptive_atlas(root, config_path=config_path)

    results_path = analysis_step2_root(root) / "baseline_model_results.csv"
    preferred_path = analysis_step2_root(root) / "preferred_baseline_models.csv"
    diagnostics_path = analysis_step2_root(root) / "baseline_model_diagnostics.csv"
    specifications_path = analysis_step2_root(root) / "baseline_model_specification_matrix.csv"
    roster_path = analysis_step2_root(root) / "step2_candidate_roster.csv"
    checklist_path = analysis_step2_root(root) / "step2_decision_checklist.csv"
    promotion_path = analysis_step2_root(root) / "step2_promotion_decisions.csv"
    family_comparison_path = analysis_step2_root(root) / "step2_exposure_family_comparison.csv"
    report_path = analysis_step2_root(root) / "step2_report.md"

    outputs = {
        "candidate_roster": _relative(root, roster_path),
        "baseline_model_results": _relative(root, results_path),
        "preferred_baseline_models": _relative(root, preferred_path),
        "baseline_model_diagnostics": _relative(root, diagnostics_path),
        "baseline_model_specification_matrix": _relative(root, specifications_path),
        "promotion_decisions": _relative(root, promotion_path),
        "exposure_family_comparison": _relative(root, family_comparison_path),
        "decision_checklist": _relative(root, checklist_path),
        "step2_report": _relative(root, report_path),
    }

    empty = pd.DataFrame()
    if upstream_step1_report["status"] != "PASS":
        gates = {
            "upstream_step1_pass": False,
            "candidate_roster_nonempty": False,
            "all_candidates_expanded_to_dual_family": False,
            "analysis_backlog_materialized": False,
            "required_model_columns_available": False,
            "all_main_candidates_estimable_in_at_least_one_family": False,
            "seasonality_controls_declared": False,
            "denominator_contract_preserved": False,
            "overdispersion_review_complete": False,
            "configured_backlog_review_screened": False,
            "family_comparison_written": False,
        }
        for path in (
            results_path,
            preferred_path,
            diagnostics_path,
            specifications_path,
            roster_path,
            promotion_path,
            family_comparison_path,
            checklist_path,
        ):
            empty.to_csv(path, index=False)
        report_path.write_text(
            _render_step2_report(
                round_config=round_config,
                step1_report=upstream_step1_report,
                gates=gates,
                summary={
                    "candidate_count": 0,
                    "candidate_family_count": 0,
                    "primary_candidate_count": 0,
                    "sensitivity_candidate_count": 0,
                    "backlog_review_candidate_count": 0,
                    "backlog_review_approved_count": 0,
                    "dual_family_approved_count": 0,
                    "single_family_approved_count": 0,
                    "preferred_model_count": 0,
                    "robustness_model_count": 0,
                    "overdispersion_adjusted_count": 0,
                    "main_estimable_candidate_count": 0,
                },
                outputs=outputs,
                workspace_report=workspace_report,
                base_candidate_roster=pd.DataFrame(),
                expanded_candidate_roster=pd.DataFrame(),
                promotion_decisions=pd.DataFrame(),
                exposure_family_comparison=pd.DataFrame(),
                configured_backlog_review_analysis_units=[],
                missing_backlog_review_units=[],
                untested_backlog_review_units=[],
                missing_backlog_columns=[],
                required_model_columns_missing=[],
            ),
            encoding="utf-8",
        )
        _write_step2_publication_notes(
            root,
            status="FAIL",
            summary={
                "candidate_count": 0,
                "candidate_family_count": 0,
                "dual_family_approved_count": 0,
                "single_family_approved_count": 0,
            },
            outputs=outputs,
        )
        return {
            "status": "FAIL",
            "step": "step2_baseline_aggregated_models",
            "analysis_id": round_config["analysis_id"],
            "years": list(round_config["years"]),
            "directories": directories,
            "upstream_step1_status": upstream_step1_report["status"],
            "gates": gates,
            "summary": {
                "candidate_count": 0,
                "candidate_family_count": 0,
                "primary_candidate_count": 0,
                "sensitivity_candidate_count": 0,
                "backlog_review_candidate_count": 0,
                "backlog_review_approved_count": 0,
                "dual_family_approved_count": 0,
                "single_family_approved_count": 0,
                "preferred_model_count": 0,
                "robustness_model_count": 0,
                "overdispersion_adjusted_count": 0,
                "main_estimable_candidate_count": 0,
            },
            "outputs": outputs,
            "workspace_report": workspace_report,
        }

    analysis_units, sparse_flags = _load_step1_inputs(root)
    seasonality_lookup = _build_seasonality_lookup(
        load_seasonality_contract(root, Path(round_config["seasonality_contract_path"]))
    )
    configured_backlog_review_analysis_units = sorted(
        {str(value) for value in round_config.get("step2_backlog_review_analysis_units", [])}
    )
    step1_analysis_unit_ids = set(analysis_units["analysis_unit_id"].astype(str).tolist())
    missing_backlog_review_units = sorted(
        value for value in configured_backlog_review_analysis_units if value not in step1_analysis_unit_ids
    )

    base_candidate_roster = _build_base_candidate_roster(
        analysis_units=analysis_units,
        sparse_flags=sparse_flags,
        backlog_review_analysis_units=set(configured_backlog_review_analysis_units),
    )
    candidate_roster = _expand_candidate_roster(base_candidate_roster, seasonality_lookup)
    candidate_roster.to_csv(roster_path, index=False)

    tested_backlog_review_units = set(
        base_candidate_roster.loc[
            base_candidate_roster["candidate_track"].astype(str).eq("backlog_review"),
            "analysis_unit_id",
        ].astype(str).tolist()
    ) if not base_candidate_roster.empty else set()
    untested_backlog_review_units = sorted(
        value for value in configured_backlog_review_analysis_units if value not in tested_backlog_review_units
    )

    missing_backlog_columns = _missing_active_backlog_columns(root, list(round_config["years"]), analysis_units)
    required_columns = _required_columns_by_mart(candidate_roster) if not candidate_roster.empty else {}
    required_model_columns_missing = (
        _missing_mart_columns(root, list(round_config["years"]), required_columns) if required_columns else []
    )
    frames_by_mart = _load_model_frames(root, list(round_config["years"]), required_columns) if required_columns and not required_model_columns_missing else {}

    if frames_by_mart:
        results, diagnostics, specifications = _fit_candidate_models(candidate_roster, frames_by_mart)
        preferred_models = _preferred_model_frame(results)
    else:
        results, diagnostics, specifications, preferred_models = empty, empty, empty, empty

    exposure_family_comparison = _build_exposure_family_comparison(base_candidate_roster, preferred_models, diagnostics)
    promotion_decisions = _build_promotion_decisions(base_candidate_roster, sparse_flags, exposure_family_comparison)

    count_candidates = candidate_roster.loc[candidate_roster["model_kind"].astype(str).eq("count")]
    count_denominator_ok = bool(
        count_candidates["denominator_column_for_step2"].astype(str).eq(count_candidates["denominator_column"].astype(str)).all()
    ) if not count_candidates.empty else True
    severity_denominator_ok = bool(
        candidate_roster.loc[candidate_roster["model_kind"].astype(str).eq("severity"), "denominator_contract_note"]
        .astype(str)
        .str.contains("weighted by daily live births", case=False, regex=False)
        .all()
    ) if not candidate_roster.loc[candidate_roster["model_kind"].astype(str).eq("severity")].empty else True

    main_candidate_count = int(
        base_candidate_roster["candidate_track"].astype(str).eq("main_round").sum()
    ) if not base_candidate_roster.empty else 0
    main_estimable_candidate_count = int(
        promotion_decisions.loc[
            promotion_decisions["candidate_track"].astype(str).eq("main_round")
            & promotion_decisions["estimable_exposure_family_count"].fillna(0).gt(0)
        ].shape[0]
    ) if not promotion_decisions.empty else 0
    backlog_review_candidate_count = int(
        base_candidate_roster["candidate_track"].astype(str).eq("backlog_review").sum()
    ) if not base_candidate_roster.empty else 0
    backlog_review_approved_count = int(
        promotion_decisions.loc[
            promotion_decisions["candidate_track"].astype(str).eq("backlog_review")
            & promotion_decisions["approved_for_next_steps_flag"].eq(True)
        ].shape[0]
    ) if not promotion_decisions.empty else 0
    dual_family_approved_count = int(
        promotion_decisions["dual_family_step3_flag"].fillna(False).astype(bool).sum()
    ) if not promotion_decisions.empty else 0
    single_family_approved_count = int(
        promotion_decisions["single_family_step3_flag"].fillna(False).astype(bool).sum()
    ) if not promotion_decisions.empty else 0

    expected_candidate_family_count = int(len(base_candidate_roster) * len(EXPOSURE_FAMILY_SPECS))
    gates = {
        "upstream_step1_pass": True,
        "candidate_roster_nonempty": main_candidate_count > 0,
        "all_candidates_expanded_to_dual_family": len(candidate_roster) == expected_candidate_family_count if not base_candidate_roster.empty else False,
        "analysis_backlog_materialized": len(missing_backlog_columns) == 0,
        "required_model_columns_available": len(required_model_columns_missing) == 0,
        "all_main_candidates_estimable_in_at_least_one_family": (
            main_estimable_candidate_count == main_candidate_count if main_candidate_count > 0 else False
        ),
        "seasonality_controls_declared": bool(
            (not specifications.empty)
            and specifications["seasonality_controls_applied"].astype(str).str.len().gt(0).all()
        ),
        "denominator_contract_preserved": count_denominator_ok and severity_denominator_ok,
        "overdispersion_review_complete": bool(
            not diagnostics.empty
            and diagnostics.loc[diagnostics["model_variant"].astype(str).eq("main"), "preferred_distribution"].astype(str).str.len().gt(0).all()
        ),
        "configured_backlog_review_screened": (
            len(missing_backlog_review_units) == 0
            and len(untested_backlog_review_units) == 0
        ),
        "family_comparison_written": len(exposure_family_comparison) == len(base_candidate_roster) if not base_candidate_roster.empty else False,
    }
    status = "PASS" if all(gates.values()) else "FAIL"

    results.to_csv(results_path, index=False)
    preferred_models.to_csv(preferred_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    specifications.to_csv(specifications_path, index=False)
    promotion_decisions.to_csv(promotion_path, index=False)
    exposure_family_comparison.to_csv(family_comparison_path, index=False)

    checklist = _build_decision_checklist(
        base_candidate_roster=base_candidate_roster,
        expanded_candidate_roster=candidate_roster,
        preferred_models=preferred_models,
        diagnostics=diagnostics,
        promotion_decisions=promotion_decisions,
        exposure_family_comparison=exposure_family_comparison,
        gates=gates,
    )
    checklist.to_csv(checklist_path, index=False)

    summary = {
        "candidate_count": int(len(base_candidate_roster)),
        "candidate_family_count": int(len(candidate_roster)),
        "primary_candidate_count": int(
            base_candidate_roster.loc[
                base_candidate_roster["candidate_track"].astype(str).eq("main_round"),
                "analysis_role",
            ].astype(str).eq("primary").sum()
        ) if not base_candidate_roster.empty else 0,
        "sensitivity_candidate_count": int(
            base_candidate_roster.loc[
                base_candidate_roster["candidate_track"].astype(str).eq("main_round"),
                "analysis_role",
            ].astype(str).eq("sensitivity").sum()
        ) if not base_candidate_roster.empty else 0,
        "backlog_review_candidate_count": backlog_review_candidate_count,
        "backlog_review_approved_count": backlog_review_approved_count,
        "dual_family_approved_count": dual_family_approved_count,
        "single_family_approved_count": single_family_approved_count,
        "preferred_model_count": int(len(preferred_models)),
        "robustness_model_count": int(results["model_variant"].astype(str).eq("robustness").sum()) if not results.empty else 0,
        "overdispersion_adjusted_count": int(results["model_variant"].astype(str).eq("overdispersion_adjusted").sum()) if not results.empty else 0,
        "main_estimable_candidate_count": main_estimable_candidate_count,
    }
    report_path.write_text(
        _render_step2_report(
            round_config=round_config,
            step1_report=upstream_step1_report,
            gates=gates,
            summary=summary,
            outputs=outputs,
            workspace_report=workspace_report,
            base_candidate_roster=base_candidate_roster,
            expanded_candidate_roster=candidate_roster,
            promotion_decisions=promotion_decisions,
            exposure_family_comparison=exposure_family_comparison,
            configured_backlog_review_analysis_units=configured_backlog_review_analysis_units,
            missing_backlog_review_units=missing_backlog_review_units,
            untested_backlog_review_units=untested_backlog_review_units,
            missing_backlog_columns=missing_backlog_columns,
            required_model_columns_missing=required_model_columns_missing,
        ),
        encoding="utf-8",
    )
    _write_step2_publication_notes(root, status=status, summary=summary, outputs=outputs)

    return {
        "status": status,
        "step": "step2_baseline_aggregated_models",
        "analysis_id": round_config["analysis_id"],
        "years": list(round_config["years"]),
        "directories": directories,
        "upstream_step1_status": upstream_step1_report["status"],
        "gates": gates,
        "summary": summary,
        "outputs": outputs,
        "workspace_report": workspace_report,
    }
