from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from pandas.errors import EmptyDataError

from heat_mnch.analysis_steps.step2 import (
    OVERDISPERSION_THRESHOLD,
    _dispersion_adjusted_result,
    _pooled_continuous_fe_proxy,
)
from heat_mnch.analysis_steps.step7 import (
    _load_municipality_lookup,
    _run_live_birth_escalations,
    _run_neonatal_escalation,
    _step7_spec,
)
from heat_mnch.config import repository_root

from .config import load_lrh_americas_round2_validation_config
from .registry import ensure_lrh_americas_round2_directory_structure, lrh_americas_round2_root


STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_SKIPPED = "SKIPPED"
FINAL_PHASES = {"freeze", "phase1", "phase2", "phase3", "phase4"}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _phase_paths(root: Path) -> dict[str, Path]:
    dirs = ensure_lrh_americas_round2_directory_structure(root)
    runtime_root = dirs["runtime_root"]
    return {
        **dirs,
        "manifest_path": runtime_root / "round_manifest.json",
        "job_status_path": runtime_root / "job_status.csv",
    }


def _load_phase_status(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "phase",
                "status",
                "started_at",
                "completed_at",
                "details_relpath",
                "error_message",
            ]
        )
    return pd.read_csv(path)


def _record_phase_status(root: Path, phase: str, status: str, details_path: Path | None = None, error_message: str = "") -> None:
    paths = _phase_paths(root)
    frame = _load_phase_status(paths["job_status_path"])
    frame = frame.loc[frame["phase"].astype(str) != phase].copy()
    row = {
        "phase": phase,
        "status": status,
        "started_at": "",
        "completed_at": _now_utc(),
        "details_relpath": _relative(root, details_path) if details_path is not None and details_path.exists() else "",
        "error_message": error_message,
    }
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame.to_csv(paths["job_status_path"], index=False)


def _phase_completed(root: Path, phase: str) -> bool:
    frame = _load_phase_status(_phase_paths(root)["job_status_path"])
    subset = frame.loc[frame["phase"].astype(str).eq(phase) & frame["status"].astype(str).eq(STATUS_PASS)]
    return not subset.empty


def _load_inputs(root: Path, spec: dict[str, Any]) -> dict[str, pd.DataFrame]:
    inputs: dict[str, pd.DataFrame] = {}
    for key, value in dict(spec.get("freeze_inputs", {})).items():
        inputs[key] = _read_csv(_resolve_path(root, str(value)))
    return inputs


def _selected_outcomes(spec: dict[str, Any]) -> list[str]:
    tier_defaults = dict(spec.get("tier_defaults", {}))
    return [*list(tier_defaults.get("tier1", [])), *list(tier_defaults.get("tier2", []))]


def _tier_lookup(spec: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    tier_defaults = dict(spec.get("tier_defaults", {}))
    for analysis_unit_id in tier_defaults.get("tier1", []):
        lookup[str(analysis_unit_id)] = "tier1"
    for analysis_unit_id in tier_defaults.get("tier2", []):
        lookup[str(analysis_unit_id)] = "tier2"
    return lookup


def _current_role_for_tier(spec: dict[str, Any], priority_tier: str) -> str:
    roles = dict(spec.get("manuscript_roles", {}))
    return str(roles.get(priority_tier, "supportive"))


def _input_inventory(root: Path, spec: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, value in dict(spec.get("freeze_inputs", {})).items():
        path = _resolve_path(root, str(value))
        record_count = 0
        if path.exists() and path.suffix.lower() == ".csv":
            try:
                record_count = int(pd.read_csv(path).shape[0])
            except Exception:
                record_count = 0
        rows.append(
            {
                "input_key": key,
                "relative_path": _relative(root, path),
                "exists_flag": path.exists(),
                "record_count": record_count,
            }
        )
    return pd.DataFrame(rows)


def _estimate_scale_to_model_kind(estimate_scale: str) -> str:
    return "severity" if "mean_difference" in str(estimate_scale) else "count"


def _acute_exposure_column(exposure_family_id: str, lag_or_window: str) -> str:
    prefix = "approximate_heat_index_c" if exposure_family_id == "family_4_heat_index_parallel" else "maximum_daily_temperature_c"
    mapping = {
        "lag0": f"{prefix}_lag0",
        "lag0_1": f"{prefix}_lag0_1_mean",
        "lag0_3": f"{prefix}_lag0_3_mean",
        "lag0_7": f"{prefix}_lag0_7_mean",
    }
    return mapping[str(lag_or_window)]


def _perinatal_exposure_column(exposure_family_id: str, lag_or_window: str) -> str:
    prefix = "approximate_heat_index_c" if exposure_family_id == "family_4_heat_index_parallel" else "maximum_daily_temperature_c"
    mapping = {
        "0_6": f"{prefix}_days_0_6_before_event_mean",
        "0_13": f"{prefix}_days_0_13_before_event_mean",
        "0_27": f"{prefix}_days_0_27_before_event_mean",
    }
    return mapping[str(lag_or_window)]


def _selected_exposure_variable(mart_name: str, exposure_family_id: str, lag_or_window: str) -> str:
    if mart_name == "acute_daily_mart":
        return _acute_exposure_column(exposure_family_id, lag_or_window)
    if mart_name == "perinatal_delivery_window_mart":
        return _perinatal_exposure_column(exposure_family_id, lag_or_window)
    if mart_name == "maternal_operational_mart":
        return _perinatal_exposure_column(exposure_family_id, lag_or_window)
    raise KeyError(f"Unsupported mart/exposure mapping for {mart_name} {exposure_family_id} {lag_or_window}")


def _effect_transform(estimate_scale: str, value: float) -> float:
    if not math.isfinite(float(value)):
        return float("nan")
    if _estimate_scale_to_model_kind(estimate_scale) == "severity":
        return float(value)
    if float(value) <= 0:
        return float("nan")
    return math.log(float(value))


def _harmful_direction_flag(estimate_scale: str, estimate: float) -> bool:
    effect = _effect_transform(estimate_scale, estimate)
    return math.isfinite(effect) and effect > 0


def _ci_overlap(estimate_scale: str, lower_a: float, upper_a: float, lower_b: float, upper_b: float) -> bool:
    a_low = _effect_transform(estimate_scale, lower_a)
    a_high = _effect_transform(estimate_scale, upper_a)
    b_low = _effect_transform(estimate_scale, lower_b)
    b_high = _effect_transform(estimate_scale, upper_b)
    if not all(math.isfinite(value) for value in (a_low, a_high, b_low, b_high)):
        return False
    return max(a_low, b_low) <= min(a_high, b_high)


def _magnitude_ratio(estimate_scale: str, derivation_estimate: float, validation_estimate: float) -> float:
    derivation_effect = abs(_effect_transform(estimate_scale, derivation_estimate))
    validation_effect = abs(_effect_transform(estimate_scale, validation_estimate))
    if derivation_effect <= 0 or not math.isfinite(derivation_effect) or not math.isfinite(validation_effect):
        return float("nan")
    return validation_effect / derivation_effect


def _magnitude_compatible(spec: dict[str, Any], estimate_scale: str, derivation_estimate: float, validation_estimate: float) -> bool:
    ratio = _magnitude_ratio(estimate_scale, derivation_estimate, validation_estimate)
    phase1_spec = dict(spec.get("phase1", {}))
    lower = float(phase1_spec.get("magnitude_ratio_min", 0.5))
    upper = float(phase1_spec.get("magnitude_ratio_max", 1.5))
    return math.isfinite(ratio) and lower <= ratio <= upper


def _derive_controls(row: pd.Series) -> list[str]:
    raw_value = str(row.get("primary_temporal_controls", "") or row.get("seasonality_controls_applied", ""))
    parts = [segment.strip() for token in raw_value.split(";") for segment in token.split(",")]
    return [part for part in parts if part]


def _event_level_denominator(analysis_unit_id: str) -> str:
    if analysis_unit_id == "neonatal_death_count":
        return "annual_live_birth_count"
    return "birth_count"


def _build_freeze_registry(root: Path, spec: dict[str, Any], inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    tier_lookup = _tier_lookup(spec)
    selected = set(_selected_outcomes(spec))
    step7_candidate_catalog = inputs["step7_candidate_catalog"].copy()
    step6_candidate_catalog = inputs["step6_candidate_catalog"].copy()
    step7_event_results = inputs["step7_event_level_results"].copy()
    temporal_appendix = inputs["step7_temporal_controls_appendix"].copy()

    step7_candidate_catalog = step7_candidate_catalog.loc[
        step7_candidate_catalog["analysis_unit_id"].astype(str).isin(selected)
    ].copy()
    step6_lookup = step6_candidate_catalog.set_index("analysis_unit_id", drop=False).to_dict(orient="index")
    temporal_lookup = temporal_appendix.set_index("analysis_unit_id", drop=False).to_dict(orient="index")
    if {"analysis_unit_id", "result_origin"}.issubset(step7_event_results.columns):
        event_lookup = (
            step7_event_results.loc[step7_event_results["result_origin"].astype(str).eq("event_level_primary")]
            .drop_duplicates(subset=["analysis_unit_id"])
            .set_index("analysis_unit_id", drop=False)
            .to_dict(orient="index")
        )
    else:
        event_lookup = {}

    rows: list[dict[str, Any]] = []
    for candidate in step7_candidate_catalog.to_dict(orient="records"):
        analysis_unit_id = str(candidate["analysis_unit_id"])
        step6_row = dict(step6_lookup.get(analysis_unit_id, {}))
        temporal_row = dict(temporal_lookup.get(analysis_unit_id, {}))
        event_row = dict(event_lookup.get(analysis_unit_id, {}))
        priority_tier = tier_lookup[analysis_unit_id]
        route_class = str(candidate.get("step7_route_class", ""))
        mart_name = str(step6_row.get("mart_name", ""))
        if analysis_unit_id in set(spec.get("phase2", {}).get("acute_outcomes", [])) | set(
            spec.get("phase3", {}).get("acute_outcomes", [])
        ):
            mart_name = "acute_daily_mart"
        selected_exposure_family_id = str(candidate.get("selected_exposure_family_id", step6_row.get("selected_exposure_family_id", "")))
        preferred_lag_or_window = str(candidate.get("preferred_lag_or_window", step6_row.get("preferred_lag_or_window", "")))
        selected_exposure_variable = ""
        try:
            if mart_name:
                selected_exposure_variable = _selected_exposure_variable(mart_name, selected_exposure_family_id, preferred_lag_or_window)
        except Exception:
            selected_exposure_variable = ""
        estimate_scale = str(step6_row.get("estimate_scale", "rate_ratio_per_interquartile_range_increase"))
        if route_class == "event_level_escalated":
            current_estimate = float(event_row.get("estimate", float("nan")))
            current_lower_ci = float(event_row.get("lower_ci", float("nan")))
            current_upper_ci = float(event_row.get("upper_ci", float("nan")))
            current_p_value = float(event_row.get("p_value", float("nan")))
        else:
            current_estimate = float(step6_row.get("estimate", float("nan")))
            current_lower_ci = float(step6_row.get("lower_ci", float("nan")))
            current_upper_ci = float(step6_row.get("upper_ci", float("nan")))
            current_p_value = float(step6_row.get("p_value", float("nan")))
        rows.append(
            {
                "analysis_unit_id": analysis_unit_id,
                "priority_tier_next_round": priority_tier,
                "current_manuscript_role": _current_role_for_tier(spec, priority_tier),
                "route_class": route_class,
                "estimator_family": "event_level_grouped_proxy" if route_class == "event_level_escalated" else "aggregated_fe_proxy",
                "mart_name": mart_name,
                "data_source": str(event_row.get("data_source", step6_row.get("mart_name", ""))),
                "selected_exposure_family_id": selected_exposure_family_id,
                "selected_exposure_family_name": str(candidate.get("selected_exposure_family_name", step6_row.get("selected_exposure_family_name", ""))),
                "selected_exposure_variable": selected_exposure_variable,
                "preferred_lag_or_window": preferred_lag_or_window,
                "effect_scale": estimate_scale,
                "contrast_operator": str(step6_row.get("contrast_operator", "")),
                "denominator_column": str(
                    step6_row.get("denominator_column", _event_level_denominator(analysis_unit_id) if route_class == "event_level_escalated" else "")
                ),
                "primary_temporal_controls": str(
                    temporal_row.get("primary_temporal_controls", step6_row.get("seasonality_controls_applied", ""))
                ),
                "robustness_temporal_controls": str(temporal_row.get("robustness_temporal_controls", "")),
                "derivation_validation_required_flag": True,
                "dlnm_required_flag": analysis_unit_id in set(spec.get("phase2", {}).get("acute_outcomes", [])),
                "case_crossover_required_flag": analysis_unit_id in set(spec.get("phase3", {}).get("acute_outcomes", [])),
                "outcome_family": str(step6_row.get("outcome_family", "")),
                "current_estimate": current_estimate,
                "current_lower_ci": current_lower_ci,
                "current_upper_ci": current_upper_ci,
                "current_p_value": current_p_value,
            }
        )
    frame = pd.DataFrame(rows).sort_values(["priority_tier_next_round", "analysis_unit_id"], ignore_index=True)
    return frame


def _load_mart_years(root: Path, mart_name: str, years: list[int], columns: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required_columns = list(dict.fromkeys(columns))
    for year in years:
        path = root / "artifacts" / "analysis" / "shared_marts" / mart_name / f"YEAR={year}" / f"{mart_name}_{year}.parquet"
        if not path.exists():
            continue
        schema = pq.read_schema(path)
        available_columns = [column for column in required_columns if column in schema.names]
        if "observation_date" not in available_columns and (
            "day_of_week" in required_columns or "epidemiologic_week" in required_columns
        ) and "observation_date" in schema.names:
            available_columns.append("observation_date")
        frame = pd.read_parquet(path, columns=available_columns)
        if "day_of_week" in required_columns and "day_of_week" not in frame.columns and "observation_date" in frame.columns:
            frame["day_of_week"] = pd.to_datetime(frame["observation_date"]).dt.dayofweek.astype(int)
        if "epidemiologic_week" in required_columns and "epidemiologic_week" not in frame.columns and "observation_date" in frame.columns:
            frame["epidemiologic_week"] = pd.to_datetime(frame["observation_date"]).dt.isocalendar().week.astype(int)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=required_columns)
    return pd.concat(frames, ignore_index=True)


def _mart_columns_for_selection(row: pd.Series, exposure_column: str) -> list[str]:
    controls = _derive_controls(row)
    base = [
        "municipality_code",
        exposure_column,
        str(row["analysis_unit_id"]),
        str(row["denominator_column"]),
        "quality_control_status",
        "offset_valid_flag",
        "denominator_zero_flag",
        "denominator_missing_flag",
        "rate_not_computable_flag",
    ]
    return [*base, *controls]


def _fit_aggregated_spec(frame: pd.DataFrame, row: pd.Series, exposure_column: str) -> dict[str, Any]:
    controls = _derive_controls(row)
    stratum_columns = ["municipality_code", *controls]
    model_kind = _estimate_scale_to_model_kind(str(row["effect_scale"]))
    result = _pooled_continuous_fe_proxy(
        frame,
        exposure_column=exposure_column,
        outcome_column=str(row["analysis_unit_id"]),
        denominator_column=str(row["denominator_column"]),
        stratum_columns=stratum_columns,
        model_kind=model_kind,
    )
    if (
        model_kind == "count"
        and math.isfinite(float(result.get("estimate", float("nan"))))
        and float(result.get("overdispersion_ratio", float("nan"))) > OVERDISPERSION_THRESHOLD
    ):
        result = _dispersion_adjusted_result(result, "count")
    return result


def _aggregated_phase1_rows(
    *,
    root: Path,
    spec: dict[str, Any],
    registry: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    derivation_years = list(spec.get("phase1", {}).get("derivation_years", []))
    validation_years = list(spec.get("phase1", {}).get("validation_years", []))
    lag_comparison = inputs["step4_lag_comparison"].copy()
    derivation_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for _, registry_row in registry.loc[registry["route_class"].astype(str) != "event_level_escalated"].iterrows():
        outcome_id = str(registry_row["analysis_unit_id"])
        grid = lag_comparison.loc[lag_comparison["analysis_unit_id"].astype(str).eq(outcome_id)].copy()
        if grid.empty:
            grid = pd.DataFrame(
                [
                    {
                        "analysis_unit_id": outcome_id,
                        "selected_exposure_family_id": registry_row["selected_exposure_family_id"],
                        "selected_exposure_family_name": registry_row["selected_exposure_family_name"],
                        "lag_or_window": registry_row["preferred_lag_or_window"],
                        "lag_window_order": 0,
                        "support_grade": "NA",
                        "support_grade_rank": 99,
                    }
                ]
            )
        candidate_rows: list[dict[str, Any]] = []
        for _, grid_row in grid.iterrows():
            family_id = str(grid_row["selected_exposure_family_id"])
            lag_or_window = str(grid_row["lag_or_window"])
            try:
                exposure_column = _selected_exposure_variable(str(registry_row["mart_name"]), family_id, lag_or_window)
            except Exception:
                continue
            frame = _load_mart_years(
                root,
                str(registry_row["mart_name"]),
                derivation_years,
                _mart_columns_for_selection(registry_row, exposure_column),
            )
            if frame.empty or exposure_column not in frame.columns:
                continue
            result = _fit_aggregated_spec(frame, registry_row, exposure_column)
            if not math.isfinite(float(result.get("estimate", float("nan")))):
                continue
            candidate_rows.append(
                {
                    "analysis_unit_id": outcome_id,
                    "selected_exposure_family_id": family_id,
                    "selected_exposure_family_name": str(grid_row.get("selected_exposure_family_name", registry_row["selected_exposure_family_name"])),
                    "selected_exposure_variable": exposure_column,
                    "preferred_lag_or_window": lag_or_window,
                    "support_grade": str(grid_row.get("support_grade", "NA")),
                    "support_grade_rank": int(pd.to_numeric(grid_row.get("support_grade_rank", 99), errors="coerce")),
                    "lag_window_order": int(pd.to_numeric(grid_row.get("lag_window_order", 999), errors="coerce")),
                    "estimate": float(result["estimate"]),
                    "lower_ci": float(result["lower_ci"]),
                    "upper_ci": float(result["upper_ci"]),
                    "p_value": float(result["p_value"]),
                    "valid_strata_count": int(result["valid_strata_count"]),
                    "harmful_direction_flag": _harmful_direction_flag(str(registry_row["effect_scale"]), float(result["estimate"])),
                    "current_match_flag": family_id == str(registry_row["selected_exposure_family_id"])
                    and lag_or_window == str(registry_row["preferred_lag_or_window"]),
                }
            )
        candidate_frame = pd.DataFrame(candidate_rows)
        if candidate_frame.empty:
            derivation_rows.append(
                {
                    "analysis_unit_id": outcome_id,
                    "route_class": registry_row["route_class"],
                    "selection_mode": "aggregated_grid_replay",
                    "selection_status": "not_estimable",
                }
            )
            validation_rows.append(
                {
                    "analysis_unit_id": outcome_id,
                    "validation_status": "skipped_no_derivation_selection",
                }
            )
            continue
        candidate_frame = candidate_frame.sort_values(
            by=["harmful_direction_flag", "support_grade_rank", "p_value", "current_match_flag", "lag_window_order"],
            ascending=[False, True, True, False, True],
            ignore_index=True,
        )
        selected_row = candidate_frame.iloc[0].to_dict()
        derivation_rows.append(
            {
                **selected_row,
                "route_class": registry_row["route_class"],
                "selection_mode": "aggregated_grid_replay",
                "selection_status": "selected",
                "candidate_count_evaluated": int(candidate_frame.shape[0]),
            }
        )
        validation_exposure = str(selected_row["selected_exposure_variable"])
        validation_frame = _load_mart_years(
            root,
            str(registry_row["mart_name"]),
            validation_years,
            _mart_columns_for_selection(registry_row, validation_exposure),
        )
        validation_result = (
            _fit_aggregated_spec(validation_frame, registry_row, validation_exposure) if not validation_frame.empty else {"estimate": float("nan"), "lower_ci": float("nan"), "upper_ci": float("nan"), "p_value": float("nan"), "valid_strata_count": 0}
        )
        validation_rows.append(
            {
                "analysis_unit_id": outcome_id,
                "selected_exposure_family_id": selected_row["selected_exposure_family_id"],
                "preferred_lag_or_window": selected_row["preferred_lag_or_window"],
                "selected_exposure_variable": validation_exposure,
                "validation_estimate": float(validation_result["estimate"]),
                "validation_lower_ci": float(validation_result["lower_ci"]),
                "validation_upper_ci": float(validation_result["upper_ci"]),
                "validation_p_value": float(validation_result["p_value"]),
                "validation_valid_strata_count": int(validation_result["valid_strata_count"]),
                "validation_status": "completed" if math.isfinite(float(validation_result["estimate"])) else "not_estimable",
            }
        )
    return pd.DataFrame(derivation_rows), pd.DataFrame(validation_rows)


def _event_level_phase1_rows(root: Path, spec: dict[str, Any], registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    derivation_years = list(spec.get("phase1", {}).get("derivation_years", []))
    validation_years = list(spec.get("phase1", {}).get("validation_years", []))
    event_registry = registry.loc[registry["route_class"].astype(str).eq("event_level_escalated")].copy()
    if event_registry.empty:
        return pd.DataFrame(), pd.DataFrame()
    lookup = _load_municipality_lookup(root)
    step7_spec = _step7_spec(root)
    live_birth_outcomes = {"preterm_birth_count", "very_preterm_birth_count", "low_birth_weight_lt2500_count", "gestational_days_lost_below_37_weeks"}
    neonatal_outcomes = {"neonatal_death_count"}

    def _run(years: list[int]) -> pd.DataFrame:
        result_frames: list[pd.DataFrame] = []
        live_birth_candidates = event_registry.loc[event_registry["analysis_unit_id"].astype(str).isin(live_birth_outcomes)].copy()
        neonatal_candidates = event_registry.loc[event_registry["analysis_unit_id"].astype(str).isin(neonatal_outcomes)].copy()
        if not live_birth_candidates.empty:
            event_rows, _, _ = _run_live_birth_escalations(
                root=root,
                years=years,
                candidates=live_birth_candidates,
                lookup=lookup,
                spec=step7_spec,
            )
            result_frames.append(event_rows)
        if not neonatal_candidates.empty:
            event_rows, _, _ = _run_neonatal_escalation(
                root=root,
                years=years,
                candidates=neonatal_candidates,
                lookup=lookup,
                spec=step7_spec,
            )
            result_frames.append(event_rows)
        if not result_frames:
            return pd.DataFrame()
        combined = pd.concat(result_frames, ignore_index=True)
        return combined.loc[combined["result_origin"].astype(str).eq("event_level_primary")].copy()

    derivation = _run(derivation_years)
    validation = _run(validation_years)
    derivation_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    derivation_lookup = derivation.set_index("analysis_unit_id", drop=False).to_dict(orient="index") if not derivation.empty else {}
    validation_lookup = validation.set_index("analysis_unit_id", drop=False).to_dict(orient="index") if not validation.empty else {}
    for _, registry_row in event_registry.iterrows():
        outcome_id = str(registry_row["analysis_unit_id"])
        derivation_row = dict(derivation_lookup.get(outcome_id, {}))
        validation_row = dict(validation_lookup.get(outcome_id, {}))
        derivation_rows.append(
            {
                "analysis_unit_id": outcome_id,
                "route_class": registry_row["route_class"],
                "selection_mode": "retained_event_level_replay",
                "selection_status": "selected" if derivation_row else "not_estimable",
                "selected_exposure_family_id": registry_row["selected_exposure_family_id"],
                "selected_exposure_family_name": registry_row["selected_exposure_family_name"],
                "selected_exposure_variable": registry_row["selected_exposure_variable"],
                "preferred_lag_or_window": registry_row["preferred_lag_or_window"],
                "estimate": float(derivation_row.get("estimate", float("nan"))),
                "lower_ci": float(derivation_row.get("lower_ci", float("nan"))),
                "upper_ci": float(derivation_row.get("upper_ci", float("nan"))),
                "p_value": float(derivation_row.get("p_value", float("nan"))),
                "valid_strata_count": int(derivation_row.get("valid_strata_count", 0)),
                "candidate_count_evaluated": 1,
                "harmful_direction_flag": _harmful_direction_flag(str(registry_row["effect_scale"]), float(derivation_row.get("estimate", float("nan")))),
                "current_match_flag": True,
            }
        )
        validation_rows.append(
            {
                "analysis_unit_id": outcome_id,
                "selected_exposure_family_id": registry_row["selected_exposure_family_id"],
                "preferred_lag_or_window": registry_row["preferred_lag_or_window"],
                "selected_exposure_variable": registry_row["selected_exposure_variable"],
                "validation_estimate": float(validation_row.get("estimate", float("nan"))),
                "validation_lower_ci": float(validation_row.get("lower_ci", float("nan"))),
                "validation_upper_ci": float(validation_row.get("upper_ci", float("nan"))),
                "validation_p_value": float(validation_row.get("p_value", float("nan"))),
                "validation_valid_strata_count": int(validation_row.get("valid_strata_count", 0)),
                "validation_status": "completed" if validation_row else "not_estimable",
            }
        )
    return pd.DataFrame(derivation_rows), pd.DataFrame(validation_rows)


def build_freeze(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    spec = load_lrh_americas_round2_validation_config(root, config_path)
    paths = _phase_paths(root)
    inputs = _load_inputs(root, spec)
    registry = _build_freeze_registry(root, spec, inputs)
    inventory = _input_inventory(root, spec)
    registry_path = paths["freeze_root"] / "retained_endpoint_registry.csv"
    inventory_path = paths["freeze_root"] / "frozen_input_inventory.csv"
    manifest_path = paths["freeze_root"] / "frozen_input_manifest.json"
    registry.to_csv(registry_path, index=False)
    inventory.to_csv(inventory_path, index=False)
    _write_json(
        manifest_path,
        {
            "analysis_id": spec.get("analysis_id", ""),
            "generated_at": _now_utc(),
            "selected_outcomes": registry["analysis_unit_id"].tolist(),
            "phase_order": spec.get("phase_order", []),
            "roadmap_doc": spec.get("roadmap_doc", ""),
        },
    )
    _write_json(
        paths["manifest_path"],
        {
            "analysis_id": spec.get("analysis_id", ""),
            "generated_at": _now_utc(),
            "root": _relative(root, lrh_americas_round2_root(root)),
            "selected_outcomes": registry["analysis_unit_id"].tolist(),
        },
    )
    return {
        "status": STATUS_PASS,
        "phase": "freeze",
        "outputs": {
            "retained_endpoint_registry": _relative(root, registry_path),
            "frozen_input_inventory": _relative(root, inventory_path),
            "frozen_input_manifest": _relative(root, manifest_path),
        },
        "selected_outcome_count": int(registry.shape[0]),
    }


def run_phase1(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    spec = load_lrh_americas_round2_validation_config(root, config_path)
    paths = _phase_paths(root)
    registry_path = paths["freeze_root"] / "retained_endpoint_registry.csv"
    if not registry_path.exists():
        build_freeze(root, config_path=config_path)
    registry = pd.read_csv(registry_path)
    inputs = _load_inputs(root, spec)
    aggregated_derivation, aggregated_validation = _aggregated_phase1_rows(root=root, spec=spec, registry=registry, inputs=inputs)
    event_derivation, event_validation = _event_level_phase1_rows(root, spec, registry)
    derivation = pd.concat([aggregated_derivation, event_derivation], ignore_index=True) if not aggregated_derivation.empty or not event_derivation.empty else pd.DataFrame()
    validation = pd.concat([aggregated_validation, event_validation], ignore_index=True) if not aggregated_validation.empty or not event_validation.empty else pd.DataFrame()
    derivation_path = paths["phase1_root"] / "temporal_derivation_selection_table.csv"
    validation_path = paths["phase1_root"] / "temporal_validation_results_table.csv"
    decision_path = paths["phase1_root"] / "temporal_stability_editorial_decision_table.csv"
    summary_path = paths["phase1_root"] / "phase1_summary.md"
    derivation.to_csv(derivation_path, index=False)
    validation.to_csv(validation_path, index=False)
    decision_rows: list[dict[str, Any]] = []
    deriv_lookup = derivation.set_index("analysis_unit_id", drop=False).to_dict(orient="index") if not derivation.empty else {}
    valid_lookup = validation.set_index("analysis_unit_id", drop=False).to_dict(orient="index") if not validation.empty else {}
    for _, registry_row in registry.iterrows():
        outcome_id = str(registry_row["analysis_unit_id"])
        derivation_row = dict(deriv_lookup.get(outcome_id, {}))
        validation_row = dict(valid_lookup.get(outcome_id, {}))
        derivation_estimate = float(derivation_row.get("estimate", float("nan")))
        validation_estimate = float(validation_row.get("validation_estimate", float("nan")))
        direction_preserved_flag = (
            math.isfinite(_effect_transform(str(registry_row["effect_scale"]), derivation_estimate))
            and math.isfinite(_effect_transform(str(registry_row["effect_scale"]), validation_estimate))
            and (_effect_transform(str(registry_row["effect_scale"]), derivation_estimate) > 0)
            == (_effect_transform(str(registry_row["effect_scale"]), validation_estimate) > 0)
        )
        ci_overlap_flag = _ci_overlap(
            str(registry_row["effect_scale"]),
            float(derivation_row.get("lower_ci", float("nan"))),
            float(derivation_row.get("upper_ci", float("nan"))),
            float(validation_row.get("validation_lower_ci", float("nan"))),
            float(validation_row.get("validation_upper_ci", float("nan"))),
        )
        magnitude_compatibility_flag = _magnitude_compatible(
            spec,
            str(registry_row["effect_scale"]),
            derivation_estimate,
            validation_estimate,
        )
        if not math.isfinite(validation_estimate) or not math.isfinite(derivation_estimate):
            editorial_classification = "provisional"
            decision_reason = "derivation_or_validation_not_estimable"
        elif direction_preserved_flag and (ci_overlap_flag or magnitude_compatibility_flag):
            editorial_classification = "principal"
            decision_reason = "direction_preserved_with_compatibility"
        elif direction_preserved_flag:
            editorial_classification = "supportive"
            decision_reason = "direction_preserved_without_broad_compatibility"
        else:
            editorial_classification = "provisional"
            decision_reason = "direction_reversal_or_material_conflict"
        decision_rows.append(
            {
                "analysis_unit_id": outcome_id,
                "priority_tier_next_round": registry_row["priority_tier_next_round"],
                "route_class": registry_row["route_class"],
                "derivation_estimate": derivation_estimate,
                "validation_estimate": validation_estimate,
                "direction_preserved_flag": direction_preserved_flag,
                "ci_overlap_flag": ci_overlap_flag,
                "magnitude_compatibility_flag": magnitude_compatibility_flag,
                "editorial_classification": editorial_classification,
                "decision_reason": decision_reason,
            }
        )
    decision_frame = pd.DataFrame(decision_rows)
    decision_frame.to_csv(decision_path, index=False)
    _write_markdown(
        summary_path,
        [
            "# Phase 1 Temporal Validation Summary",
            "",
            f"- Generated at: `{_now_utc()}`",
            f"- Outcomes reviewed: `{int(registry.shape[0])}`",
            f"- Principal retained: `{int(decision_frame['editorial_classification'].astype(str).eq('principal').sum())}`",
            f"- Supportive retained: `{int(decision_frame['editorial_classification'].astype(str).eq('supportive').sum())}`",
            f"- Provisional: `{int(decision_frame['editorial_classification'].astype(str).eq('provisional').sum())}`",
        ],
    )
    return {
        "status": STATUS_PASS,
        "phase": "phase1",
        "outputs": {
            "temporal_derivation_selection_table": _relative(root, derivation_path),
            "temporal_validation_results_table": _relative(root, validation_path),
            "temporal_stability_editorial_decision_table": _relative(root, decision_path),
            "phase1_summary": _relative(root, summary_path),
        },
    }


def _check_r_packages(root: Path, packages: list[str]) -> dict[str, Any]:
    script_path = root / "scripts" / "r" / "check_lrh_round2_packages.R"
    command = ["Rscript", str(script_path), "--packages", ",".join(packages)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    stdout = completed.stdout.strip() or "{}"
    payload = json.loads(stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"R package preflight failed: {payload.get('missing_packages', [])}")
    return payload


def _build_acute_input(root: Path, registry: pd.DataFrame, target_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in registry.iterrows():
        outcome_id = str(row["analysis_unit_id"])
        mart_name = "acute_daily_mart"
        exposure_family_id = str(row["selected_exposure_family_id"])
        raw_exposure_column = _acute_exposure_column(exposure_family_id, "lag0")
        columns = [
            "municipality_code",
            "observation_date",
            "calendar_year",
            "calendar_month",
            outcome_id,
            str(row["denominator_column"]),
            raw_exposure_column,
            "quality_control_status",
            "offset_valid_flag",
            "denominator_zero_flag",
            "denominator_missing_flag",
            "rate_not_computable_flag",
        ]
        frame = _load_mart_years(root, mart_name, [2020, 2021, 2022, 2023, 2024], columns)
        if frame.empty:
            continue
        if "day_of_week" not in frame.columns:
            frame["day_of_week"] = pd.to_datetime(frame["observation_date"]).dt.dayofweek.astype(int)
        frame = frame.loc[
            frame["quality_control_status"].astype(str).eq("PASS")
            & pd.to_numeric(frame["offset_valid_flag"], errors="coerce").fillna(0).eq(1)
            & pd.to_numeric(frame["denominator_zero_flag"], errors="coerce").fillna(0).eq(0)
            & pd.to_numeric(frame["denominator_missing_flag"], errors="coerce").fillna(0).eq(0)
            & pd.to_numeric(frame["rate_not_computable_flag"], errors="coerce").fillna(0).eq(0)
        ].copy()
        frame = frame.rename(
            columns={
                outcome_id: "outcome_count",
                str(row["denominator_column"]): "denominator_count",
                raw_exposure_column: "exposure_value",
            }
        )
        frame = frame.loc[
            frame["denominator_count"].fillna(0).gt(0) & frame["exposure_value"].notna(),
            ["municipality_code", "observation_date", "calendar_year", "calendar_month", "day_of_week", "outcome_count", "denominator_count", "exposure_value"],
        ].copy()
        if frame.empty:
            continue
        output_path = target_root / f"{outcome_id}.csv"
        frame.to_csv(output_path, index=False)
        rows.append(
            {
                "analysis_unit_id": outcome_id,
                "input_csv_relpath": _relative(root, output_path),
                "row_count": int(frame.shape[0]),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(target_root / "input_manifest.csv", index=False)
    return manifest


def run_phase2(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    spec = load_lrh_americas_round2_validation_config(root, config_path)
    paths = _phase_paths(root)
    if not (paths["freeze_root"] / "retained_endpoint_registry.csv").exists():
        build_freeze(root, config_path=config_path)
    registry = pd.read_csv(paths["freeze_root"] / "retained_endpoint_registry.csv")
    acute_registry = registry.loc[registry["dlnm_required_flag"].fillna(False)].copy()
    if acute_registry.empty:
        empty_paths = {
            "summary": paths["phase2_root"] / "dlnm_lag_response_summary_table.csv",
            "grid": paths["phase2_root"] / "dlnm_exposure_response_grid.csv",
            "lag_grid": paths["phase2_root"] / "dlnm_lag_response_grid.csv",
            "compare": paths["phase2_root"] / "dlnm_vs_current_simplified_estimand_table.csv",
            "note": paths["phase2_root"] / "dlnm_editorial_note.md",
        }
        pd.DataFrame().to_csv(empty_paths["summary"], index=False)
        pd.DataFrame().to_csv(empty_paths["grid"], index=False)
        pd.DataFrame().to_csv(empty_paths["lag_grid"], index=False)
        pd.DataFrame().to_csv(empty_paths["compare"], index=False)
        _write_markdown(empty_paths["note"], ["# DLNM Editorial Note", "", "No acute outcomes were in scope for this round."])
        return {"status": STATUS_SKIPPED, "phase": "phase2"}
    _check_r_packages(root, list(spec.get("phase2", {}).get("required_r_packages", [])))
    _build_acute_input(root, acute_registry, paths["phase2_input_root"])
    script_path = root / "scripts" / "r" / "run_lrh_phase2_dlnm.R"
    command = [
        "Rscript",
        str(script_path),
        "--registry",
        str(paths["freeze_root"] / "retained_endpoint_registry.csv"),
        "--input-dir",
        str(paths["phase2_input_root"]),
        "--output-dir",
        str(paths["phase2_root"]),
        "--lag-max-default",
        str(spec.get("phase2", {}).get("lag_max_default", 7)),
        "--lag-max-extended",
        str(spec.get("phase2", {}).get("lag_max_extended", 21)),
        "--extended-lag-outcomes",
        ",".join(spec.get("phase2", {}).get("extended_lag_allowed_outcomes", [])),
        "--reference-quantile",
        str(spec.get("phase2", {}).get("reference_quantile", 0.5)),
        "--hot-quantile",
        str(spec.get("phase2", {}).get("hot_quantile", 0.95)),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    stdout_path = paths["phase2_root"] / "phase2_r_stdout.log"
    stderr_path = paths["phase2_root"] / "phase2_r_stderr.log"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "Phase 2 R execution failed")
    return {
        "status": STATUS_PASS,
        "phase": "phase2",
        "outputs": {
            "dlnm_lag_response_summary_table": _relative(root, paths["phase2_root"] / "dlnm_lag_response_summary_table.csv"),
            "dlnm_vs_current_simplified_estimand_table": _relative(root, paths["phase2_root"] / "dlnm_vs_current_simplified_estimand_table.csv"),
            "dlnm_editorial_note": _relative(root, paths["phase2_root"] / "dlnm_editorial_note.md"),
            "dlnm_outcome_diagnostics": _relative(root, paths["phase2_root"] / "dlnm_outcome_diagnostics.csv"),
            "phase2_r_stdout_log": _relative(root, stdout_path),
            "phase2_r_stderr_log": _relative(root, stderr_path),
        },
    }


def run_phase3(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    spec = load_lrh_americas_round2_validation_config(root, config_path)
    paths = _phase_paths(root)
    if not (paths["freeze_root"] / "retained_endpoint_registry.csv").exists():
        build_freeze(root, config_path=config_path)
    registry = pd.read_csv(paths["freeze_root"] / "retained_endpoint_registry.csv")
    acute_registry = registry.loc[registry["case_crossover_required_flag"].fillna(False)].copy()
    if acute_registry.empty:
        result_path = paths["phase3_root"] / "case_crossover_results_table.csv"
        compare_path = paths["phase3_root"] / "fe_proxy_vs_case_crossover_comparison_table.csv"
        note_path = paths["phase3_root"] / "acute_design_concordance_note.md"
        pd.DataFrame().to_csv(result_path, index=False)
        pd.DataFrame().to_csv(compare_path, index=False)
        _write_markdown(note_path, ["# Acute Design Concordance Note", "", "No acute outcomes were in scope for this round."])
        return {"status": STATUS_SKIPPED, "phase": "phase3"}
    _check_r_packages(root, list(spec.get("phase3", {}).get("required_r_packages", [])))
    _build_acute_input(root, acute_registry, paths["phase3_input_root"])
    script_path = root / "scripts" / "r" / "run_lrh_phase3_case_crossover.R"
    command = [
        "Rscript",
        str(script_path),
        "--registry",
        str(paths["freeze_root"] / "retained_endpoint_registry.csv"),
        "--input-dir",
        str(paths["phase3_input_root"]),
        "--output-dir",
        str(paths["phase3_root"]),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    stdout_path = paths["phase3_root"] / "phase3_r_stdout.log"
    stderr_path = paths["phase3_root"] / "phase3_r_stderr.log"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "Phase 3 R execution failed")
    diagnostics_path = paths["phase3_root"] / "case_crossover_outcome_diagnostics.csv"
    diagnostics = _read_csv(diagnostics_path)
    if diagnostics.empty:
        raise RuntimeError("Phase 3 completed without diagnostics output")
    successful_outcomes = diagnostics.loc[diagnostics["status"].astype(str).eq("success"), "analysis_unit_id"].astype(str).tolist()
    if not successful_outcomes:
        raise RuntimeError("Phase 3 completed without any successful acute outcomes")
    return {
        "status": STATUS_PASS,
        "phase": "phase3",
        "outputs": {
            "case_crossover_results_table": _relative(root, paths["phase3_root"] / "case_crossover_results_table.csv"),
            "comparison_table": _relative(root, paths["phase3_root"] / "fe_proxy_vs_case_crossover_comparison_table.csv"),
            "acute_design_concordance_note": _relative(root, paths["phase3_root"] / "acute_design_concordance_note.md"),
            "case_crossover_outcome_diagnostics": _relative(root, diagnostics_path),
            "phase3_r_stdout_log": _relative(root, stdout_path),
            "phase3_r_stderr_log": _relative(root, stderr_path),
        },
    }


def run_phase4(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    spec = load_lrh_americas_round2_validation_config(root, config_path)
    paths = _phase_paths(root)
    registry = pd.read_csv(paths["freeze_root"] / "retained_endpoint_registry.csv")
    phase1 = pd.read_csv(paths["phase1_root"] / "temporal_stability_editorial_decision_table.csv")
    phase1_lookup = phase1.set_index("analysis_unit_id", drop=False).to_dict(orient="index")
    phase2_path = paths["phase2_root"] / "dlnm_lag_response_summary_table.csv"
    phase3_path = paths["phase3_root"] / "fe_proxy_vs_case_crossover_comparison_table.csv"
    phase2 = _read_csv(phase2_path)
    phase3 = _read_csv(phase3_path)
    phase2_lookup = phase2.set_index("analysis_unit_id", drop=False).to_dict(orient="index") if not phase2.empty else {}
    phase3_lookup = phase3.set_index("analysis_unit_id", drop=False).to_dict(orient="index") if not phase3.empty else {}
    missing_acute = [
        analysis_unit_id
        for _, registry_row in registry.loc[
            registry["dlnm_required_flag"].fillna(False) | registry["case_crossover_required_flag"].fillna(False)
        ].iterrows()
        for analysis_unit_id in [str(registry_row["analysis_unit_id"])]
        if analysis_unit_id not in phase1_lookup
        or (bool(registry_row.get("dlnm_required_flag", False)) and analysis_unit_id not in phase2_lookup)
        or (bool(registry_row.get("case_crossover_required_flag", False)) and analysis_unit_id not in phase3_lookup)
    ]
    if missing_acute:
        raise ValueError(f"Missing acute outputs required for phase4: {missing_acute}")
    hierarchy_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for _, registry_row in registry.iterrows():
        outcome_id = str(registry_row["analysis_unit_id"])
        phase1_row = dict(phase1_lookup.get(outcome_id, {}))
        phase2_row = dict(phase2_lookup.get(outcome_id, {}))
        phase3_row = dict(phase3_lookup.get(outcome_id, {}))
        final_status = str(phase1_row.get("editorial_classification", "provisional"))
        acute_outcome = bool(registry_row.get("dlnm_required_flag", False) or registry_row.get("case_crossover_required_flag", False))
        if acute_outcome:
            if str(phase2_row.get("dlnm_classification", "")).startswith("contradict") or str(phase3_row.get("concordance_classification", "")) == "discordant":
                final_status = "provisional"
            elif final_status == "principal" and (
                str(phase2_row.get("dlnm_classification", "")) == "refines_retained_signal"
                or str(phase3_row.get("concordance_classification", "")) == "partial_concordance"
            ):
                final_status = "supportive"
        hierarchy_rows.append(
            {
                "analysis_unit_id": outcome_id,
                "priority_tier_next_round": registry_row["priority_tier_next_round"],
                "current_manuscript_role": registry_row["current_manuscript_role"],
                "route_class": registry_row["route_class"],
                "phase1_classification": phase1_row.get("editorial_classification", ""),
                "phase2_classification": phase2_row.get("dlnm_classification", ""),
                "phase3_classification": phase3_row.get("concordance_classification", ""),
                "final_editorial_status": final_status,
            }
        )
        matrix_rows.append(
            {
                "analysis_unit_id": outcome_id,
                "principal_flag": final_status == "principal",
                "supportive_flag": final_status == "supportive",
                "provisional_flag": final_status == "provisional",
            }
        )
    hierarchy = pd.DataFrame(hierarchy_rows)
    matrix = pd.DataFrame(matrix_rows)
    hierarchy_path = paths["phase4_root"] / "updated_endpoint_hierarchy_table.csv"
    matrix_path = paths["phase4_root"] / "updated_principal_supportive_provisional_matrix.csv"
    memo_path = paths["phase4_root"] / "editorial_round_decision_memo.md"
    hierarchy.to_csv(hierarchy_path, index=False)
    matrix.to_csv(matrix_path, index=False)
    _write_markdown(
        memo_path,
        [
            "# Editorial Round Decision Memo",
            "",
            f"- Generated at: `{_now_utc()}`",
            f"- Principal outcomes after re-synthesis: `{int(hierarchy['final_editorial_status'].astype(str).eq('principal').sum())}`",
            f"- Supportive outcomes after re-synthesis: `{int(hierarchy['final_editorial_status'].astype(str).eq('supportive').sum())}`",
            f"- Provisional outcomes after re-synthesis: `{int(hierarchy['final_editorial_status'].astype(str).eq('provisional').sum())}`",
            "",
            "This memo integrates temporal validation, DLNM, and time-stratified conditional Poisson case-crossover equivalent outputs.",
        ],
    )
    return {
        "status": STATUS_PASS,
        "phase": "phase4",
        "outputs": {
            "updated_endpoint_hierarchy_table": _relative(root, hierarchy_path),
            "updated_principal_supportive_provisional_matrix": _relative(root, matrix_path),
            "editorial_round_decision_memo": _relative(root, memo_path),
        },
    }


def run_lrh_americas_round2_validation(
    *,
    root: Path | None = None,
    phase: str = "all",
    config_path: Path | None = None,
    resume: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    project_root = root or repository_root()
    phase = str(phase)
    if phase not in {*FINAL_PHASES, "all"}:
        raise ValueError(f"Unsupported phase {phase!r}")
    paths = _phase_paths(project_root)
    if force and phase in FINAL_PHASES:
        phase_dir_map = {
            "freeze": paths["freeze_root"],
            "phase1": paths["phase1_root"],
            "phase2": paths["phase2_root"],
            "phase3": paths["phase3_root"],
            "phase4": paths["phase4_root"],
        }
        shutil.rmtree(phase_dir_map[phase], ignore_errors=True)
        ensure_lrh_americas_round2_directory_structure(project_root)
    if phase in FINAL_PHASES and resume and _phase_completed(project_root, phase):
        return {"status": STATUS_SKIPPED, "phase": phase, "reason": "resume_skip_completed"}

    def _run_single(target_phase: str) -> dict[str, Any]:
        if target_phase == "freeze":
            return build_freeze(project_root, config_path=config_path)
        if target_phase == "phase1":
            return run_phase1(project_root, config_path=config_path)
        if target_phase == "phase2":
            return run_phase2(project_root, config_path=config_path)
        if target_phase == "phase3":
            return run_phase3(project_root, config_path=config_path)
        if target_phase == "phase4":
            return run_phase4(project_root, config_path=config_path)
        raise ValueError(target_phase)

    if phase != "all":
        try:
            payload = _run_single(phase)
            _record_phase_status(project_root, phase, payload.get("status", STATUS_PASS))
            return payload
        except Exception as exc:
            _record_phase_status(project_root, phase, STATUS_FAIL, error_message=str(exc))
            raise

    results: dict[str, Any] = {}
    for target_phase in ("freeze", "phase1"):
        try:
            payload = _run_single(target_phase)
            results[target_phase] = payload
            _record_phase_status(project_root, target_phase, payload.get("status", STATUS_PASS))
        except Exception as exc:
            _record_phase_status(project_root, target_phase, STATUS_FAIL, error_message=str(exc))
            raise
    spec = load_lrh_americas_round2_validation_config(project_root, config_path)
    if bool(spec.get("parallel_acute_phases", False)):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                "phase2": executor.submit(_run_single, "phase2"),
                "phase3": executor.submit(_run_single, "phase3"),
            }
            for target_phase, future in futures.items():
                try:
                    payload = future.result()
                    results[target_phase] = payload
                    _record_phase_status(project_root, target_phase, payload.get("status", STATUS_PASS))
                except Exception as exc:
                    _record_phase_status(project_root, target_phase, STATUS_FAIL, error_message=str(exc))
                    raise
    else:
        for target_phase in ("phase2", "phase3"):
            try:
                payload = _run_single(target_phase)
                results[target_phase] = payload
                _record_phase_status(project_root, target_phase, payload.get("status", STATUS_PASS))
            except Exception as exc:
                _record_phase_status(project_root, target_phase, STATUS_FAIL, error_message=str(exc))
                raise
    try:
        payload = _run_single("phase4")
        results["phase4"] = payload
        _record_phase_status(project_root, "phase4", payload.get("status", STATUS_PASS))
    except Exception as exc:
        _record_phase_status(project_root, "phase4", STATUS_FAIL, error_message=str(exc))
        raise
    return {"status": STATUS_PASS, "phase": "all", "results": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_lrh_americas_round2_validation.py")
    parser.add_argument("--phase", choices=["freeze", "phase1", "phase2", "phase3", "phase4", "all"], default="all")
    parser.add_argument("--config-path")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_lrh_americas_round2_validation(
        root=repository_root(),
        phase=args.phase,
        config_path=Path(args.config_path) if args.config_path else None,
        resume=bool(args.resume),
        force=bool(args.force),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
