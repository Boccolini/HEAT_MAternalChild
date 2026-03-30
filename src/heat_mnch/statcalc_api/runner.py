from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

from ..analysis_track_a.benchmark import COUNT_MODEL_SCALE, SEVERITY_MODEL_SCALE
from ..analysis_track_a.config import load_track_a_config
from ..analysis_track_a.sensitivity import (
    MODIFIER_SPECS,
    _candidate_frame as track_a_candidate_frame,
    _combine_candidate_frames as track_a_combine_candidate_frames,
    _context_with_strata as track_a_context_with_strata,
    _estimate_candidate_effect as track_a_estimate_candidate_effect,
    _exposure_label as track_a_exposure_label,
    _load_triage as track_a_load_triage,
)
from ..analysis_track_a.benchmark import _candidate_rows as track_a_candidate_rows
from ..analysis_track_b.config import load_track_b_config
from ..analysis_track_b.neonatal import _add_lags as track_b_add_lags
from ..analysis_track_b.neonatal import _load_mortality as track_b_load_mortality
from ..analysis_track_b.preterm import _fit_with_fallback as track_b_preterm_fit_with_fallback
from ..analysis_track_b.shared import pooled_continuous_proxy
from ..analysis_track_b.stillbirth import _fit_with_fallback as track_b_stillbirth_fit_with_fallback
from ..analysis_track_b.shared import load_track_b_matrix_partitions
from .registry import StatCalcRegistry
from .schemas import AnalysisResultRow, AnalysisRunResult, AnalysisRunSpec
from .storage import StatCalcStorage


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TRACK_B_PRETERM_COLUMNS = [
    "record_type",
    "municipality_code",
    "calendar_year",
    "month_of_conception",
    "gestational_age_source",
    "ga_exact_measurement_flag",
    "gestational_age_valid_for_analysis_flag",
    "preterm_birth_flag",
    "very_preterm_birth_flag",
    "gestational_days_lost_below_37_weeks",
    "third_trimester_mean_tmax_c",
    "third_trimester_mean_rh_mean_pct",
    "third_trimester_mean_heat_index_c",
    "third_trimester_heatwave_days",
    "third_trimester_full_window_observed_flag",
    "third_trimester_fallback_day_share",
    "days_0_27_before_event_mean_tmax_c",
    "days_0_27_before_event_full_window_observed_flag",
    "days_0_27_before_event_fallback_day_share",
    "whole_pregnancy_mean_tmax_c",
    "whole_pregnancy_full_window_observed_flag",
    "whole_pregnancy_fallback_day_share",
]

TRACK_B_STILLBIRTH_COLUMNS = [
    "record_type",
    "municipality_code",
    "calendar_year",
    "event_date",
    "gestational_age_source",
    "gestational_age_weeks_final",
    "stillbirth_primary_gte28_flag",
    "stillbirth_sensitivity_gte22_flag",
    "days_0_6_before_event_mean_tmax_c",
    "days_0_6_before_event_mean_heat_index_c",
    "days_0_6_before_event_full_window_observed_flag",
    "days_0_6_before_event_fallback_day_share",
    "days_0_13_before_event_mean_tmax_c",
    "days_0_13_before_event_full_window_observed_flag",
    "days_0_27_before_event_mean_tmax_c",
    "days_0_27_before_event_full_window_observed_flag",
    "third_trimester_mean_tmax_c",
    "third_trimester_full_window_observed_flag",
]

TRACK_B_PRETERM_OUTCOME_LABELS = {
    "preterm_birth_flag": "Preterm birth",
    "very_preterm_birth_flag": "Very preterm birth",
    "gestational_days_lost_below_37_weeks": "Gestational days lost below 37 weeks",
}

TRACK_B_STILLBIRTH_OUTCOME_LABELS = {
    "stillbirth_primary_gte28_flag": "Stillbirth at or beyond 28 weeks",
    "stillbirth_sensitivity_gte22_flag": "Stillbirth at or beyond 22 weeks",
}

TRACK_B_PRETERM_EXPOSURE_FLAGS = {
    "third_trimester_mean_tmax_c": ("third_trimester_full_window_observed_flag", "third_trimester_fallback_day_share"),
    "third_trimester_mean_heat_index_c": ("third_trimester_full_window_observed_flag", "third_trimester_fallback_day_share"),
    "third_trimester_heatwave_days": ("third_trimester_full_window_observed_flag", "third_trimester_fallback_day_share"),
    "third_trimester_mean_tmax_x_rh_mean_pct": ("third_trimester_full_window_observed_flag", "third_trimester_fallback_day_share"),
    "days_0_27_before_event_mean_tmax_c": ("days_0_27_before_event_full_window_observed_flag", "days_0_27_before_event_fallback_day_share"),
    "whole_pregnancy_mean_tmax_c": ("whole_pregnancy_full_window_observed_flag", "whole_pregnancy_fallback_day_share"),
}

TRACK_B_STILLBIRTH_EXPOSURE_FLAGS = {
    "days_0_6_before_event_mean_tmax_c": ("days_0_6_before_event_full_window_observed_flag", "days_0_6_before_event_fallback_day_share"),
    "days_0_6_before_event_mean_heat_index_c": ("days_0_6_before_event_full_window_observed_flag", "days_0_6_before_event_fallback_day_share"),
    "days_0_13_before_event_mean_tmax_c": ("days_0_13_before_event_full_window_observed_flag", None),
    "days_0_27_before_event_mean_tmax_c": ("days_0_27_before_event_full_window_observed_flag", None),
    "third_trimester_mean_tmax_c": ("third_trimester_full_window_observed_flag", None),
}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        normalized.append(
            {
                "row_id": str(row.get("row_id") or index),
                "label": str(row.get("label") or f"Row {index}"),
                "group_label": row.get("group_label"),
                "estimate": _safe_float(row.get("estimate")),
                "lower_ci": _safe_float(row.get("lower_ci")),
                "upper_ci": _safe_float(row.get("upper_ci")),
                "p_value": _safe_float(row.get("p_value")),
                "estimate_scale": str(row.get("estimate_scale") or ""),
                "model_label": str(row.get("model_label") or ""),
                "valid_strata_count": int(row.get("valid_strata_count") or 0),
                "observation_count": int(row["observation_count"]) if row.get("observation_count") is not None else None,
                "outcome_sum": _safe_float(row.get("outcome_sum")),
                "denominator_sum": _safe_float(row.get("denominator_sum")),
                "primary_row_flag": bool(row.get("primary_row_flag", index == 1)),
                "metadata": row.get("metadata") or {},
            }
        )
    return normalized


def _result_status(rows: list[dict[str, Any]], warnings: list[str]) -> str:
    if warnings:
        return "completed_with_warnings"
    if rows and max(int(row.get("valid_strata_count") or 0) for row in rows) == 0:
        return "completed_with_warnings"
    return "completed"


def _forest_null_value(estimate_scale: str) -> float:
    return 0.0 if "mean_difference" in estimate_scale else 1.0


def _write_forest_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    figure = plt.figure(figsize=(10.5, max(4.0, 1.2 + (0.6 * max(len(rows), 1)))))
    axis = figure.add_subplot(1, 1, 1)
    valid = [row for row in rows if row.get("estimate") is not None]
    if not valid:
        axis.text(0.5, 0.5, "No estimable rows were produced for this run.", ha="center", va="center")
        axis.axis("off")
    else:
        valid = list(reversed(valid))
        y_positions = list(range(len(valid)))
        estimates = [row["estimate"] for row in valid]
        lowers = [row["estimate"] - row["lower_ci"] if row["lower_ci"] is not None else 0.0 for row in valid]
        uppers = [row["upper_ci"] - row["estimate"] if row["upper_ci"] is not None else 0.0 for row in valid]
        axis.errorbar(
            estimates,
            y_positions,
            xerr=[lowers, uppers],
            fmt="o",
            color="#7f1d1d",
            ecolor="#334155",
            capsize=3,
        )
        null_value = _forest_null_value(str(valid[0]["estimate_scale"]))
        axis.axvline(null_value, linestyle="--", color="black", linewidth=1)
        axis.set_yticks(y_positions, [row["label"] for row in valid])
        axis.set_xlabel(valid[0]["estimate_scale"].replace("_", " "))
    axis.set_title("Statcalc forest plot")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _persist_result(
    storage: StatCalcStorage,
    run_id: str,
    *,
    track_id: str,
    recipe_id: str,
    execution_mode: str,
    spec: AnalysisRunSpec,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    warnings: list[str],
) -> AnalysisRunResult:
    run = storage.get_run(run_id)
    run_dir = Path(run.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    normalized_rows = _normalize_rows(rows)
    table_frame = pd.DataFrame(normalized_rows)
    table_path = run_dir / "table.csv"
    table_frame.to_csv(table_path, index=False)
    forest_path = run_dir / "forest_plot.png"
    _write_forest_plot(normalized_rows, forest_path)

    forest_url = f"/statcalc-assets/runs/{run_id}/forest_plot.png"
    table_url = f"/statcalc-assets/runs/{run_id}/table.csv"
    result = AnalysisRunResult(
        run_id=run_id,
        track_id=track_id,
        recipe_id=recipe_id,
        execution_mode=execution_mode,
        summary=summary,
        rows=[AnalysisResultRow.model_validate(row) for row in normalized_rows],
        warnings=warnings,
        forest_plot_url=forest_url,
        table_csv_url=table_url,
        spec=spec,
    )
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "spec.json").write_text(json.dumps(spec.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "run_log.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "track_id": track_id,
                "recipe_id": recipe_id,
                "execution_mode": execution_mode,
                "warnings": warnings,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    storage.set_run_result(
        run_id,
        status=_result_status(normalized_rows, warnings),
        warnings=warnings,
        result_json_path=str(result_path),
    )
    return result


def _load_reference_result(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _resolve_track_a_reference(root: Path, spec: AnalysisRunSpec) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    exposure_column = f"{spec.exposure_variable}_{spec.lag_or_window}"
    benchmark = _load_reference_result(root / "artifacts/track_a/models/benchmark_results.csv")
    sensitivity = _load_reference_result(root / "artifacts/track_a/sensitivity/sensitivity_matrix.csv")
    heterogeneity = _load_reference_result(root / "artifacts/track_a/sensitivity/heterogeneity_results.csv")

    if spec.stratifier_id:
        rows = heterogeneity.loc[
            heterogeneity["modifier_name"].astype(str).eq(spec.stratifier_id)
            & heterogeneity["outcome_variable"].astype(str).eq(spec.outcome_variable)
            & heterogeneity["exposure_variable"].astype(str).eq(exposure_column)
        ].copy()
        if rows.empty:
            return None
        return (
            [
                {
                    "label": f"{row['outcome_label']} - {row['modifier_stratum']}",
                    "group_label": row["modifier_stratum"],
                    "estimate": row["estimate"],
                    "lower_ci": row["lower_ci"],
                    "upper_ci": row["upper_ci"],
                    "p_value": row["p_value"],
                    "estimate_scale": row["estimate_scale"],
                    "model_label": row["model_label"],
                    "valid_strata_count": row["valid_strata_count"],
                    "observation_count": row["observation_count"],
                    "outcome_sum": row["outcome_sum"],
                    "denominator_sum": row["denominator_sum"],
                    "metadata": {
                        "direction_vs_full_sample": row.get("direction_vs_full_sample"),
                        "heterogeneity_signal_classification": row.get("heterogeneity_signal_classification"),
                    },
                }
                for _, row in rows.iterrows()
            ],
            {
                "outcome_label": str(rows.iloc[0]["public_concept_label"]),
                "exposure_label": str(rows.iloc[0]["exposure_label"]),
                "offset_label": "Automatic denominator selected by recipe",
                "reference_artifact": "artifacts/track_a/sensitivity/heterogeneity_results.csv",
            },
        )

    if spec.source_filter == "no_fallback":
        rows = sensitivity.loc[
            sensitivity["sensitivity_domain"].astype(str).eq("source_restriction_no_fallback")
            & sensitivity["outcome_variable"].astype(str).eq(spec.outcome_variable)
            & sensitivity["exposure_variable"].astype(str).eq(exposure_column)
        ].copy()
        if rows.empty:
            return None
        row = rows.iloc[0]
        return (
            [
                {
                    "label": str(row["outcome_label"]),
                    "estimate": row["estimate"],
                    "lower_ci": row["lower_ci"],
                    "upper_ci": row["upper_ci"],
                    "p_value": row["p_value"],
                    "estimate_scale": row["estimate_scale"],
                    "model_label": row["model_label"],
                    "valid_strata_count": row["valid_strata_count"],
                    "observation_count": row["observation_count"],
                    "outcome_sum": row["outcome_sum"],
                    "denominator_sum": row["denominator_sum"],
                }
            ],
            {
                "outcome_label": str(row["outcome_label"]),
                "exposure_label": str(row["exposure_label"]),
                "offset_label": "Automatic denominator selected by recipe",
                "reference_artifact": "artifacts/track_a/sensitivity/sensitivity_matrix.csv",
            },
        )

    rows = benchmark.loc[
        benchmark["outcome_variable"].astype(str).eq(spec.outcome_variable)
        & benchmark["exposure_variable"].astype(str).eq(exposure_column)
    ].copy()
    if not rows.empty:
        row = rows.iloc[0]
        return (
            [
                {
                    "label": str(row["outcome_label"]),
                    "estimate": row["estimate"],
                    "lower_ci": row["lower_ci"],
                    "upper_ci": row["upper_ci"],
                    "p_value": row["p_value"],
                    "estimate_scale": row["estimate_scale"],
                    "model_label": row["model_label"],
                    "valid_strata_count": row["valid_strata_count"],
                    "observation_count": row["observation_count"],
                    "outcome_sum": row["outcome_sum"],
                    "denominator_sum": row["denominator_sum"],
                }
            ],
            {
                "outcome_label": str(row["outcome_label"]),
                "exposure_label": str(row["exposure_label"]),
                "offset_label": "Automatic denominator selected by recipe",
                "reference_artifact": "artifacts/track_a/models/benchmark_results.csv",
            },
        )

    rows = sensitivity.loc[
        sensitivity["outcome_variable"].astype(str).eq(spec.outcome_variable)
        & sensitivity["exposure_variable"].astype(str).eq(exposure_column)
        & sensitivity["sample_filter"].astype(str).eq("quality_control_pass_and_valid_offset")
    ].copy()
    if rows.empty:
        return None
    row = rows.iloc[0]
    return (
        [
            {
                "label": str(row["outcome_label"]),
                "estimate": row["estimate"],
                "lower_ci": row["lower_ci"],
                "upper_ci": row["upper_ci"],
                "p_value": row["p_value"],
                "estimate_scale": row["estimate_scale"],
                "model_label": row["model_label"],
                "valid_strata_count": row["valid_strata_count"],
                "observation_count": row["observation_count"],
                "outcome_sum": row["outcome_sum"],
                "denominator_sum": row["denominator_sum"],
            }
        ],
        {
            "outcome_label": str(row["outcome_label"]),
            "exposure_label": str(row["exposure_label"]),
            "offset_label": "Automatic denominator selected by recipe",
            "reference_artifact": "artifacts/track_a/sensitivity/sensitivity_matrix.csv",
        },
    )


def _resolve_track_b_reference(root: Path, spec: AnalysisRunSpec) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    if spec.stratifier_id or spec.source_filter != "all_sources":
        return None
    if spec.gestational_source_mode not in {None, "all_sources"}:
        return None

    path_map = {
        "track_b_preterm_fe_proxy": [
            root / "artifacts/track_b/models/preterm_main_models.csv",
            root / "artifacts/track_b/models/gest_days_lost_models.csv",
        ],
        "track_b_stillbirth_fe_proxy": [
            root / "artifacts/track_b/models/stillbirth_main_models.csv",
            root / "artifacts/track_b/models/stillbirth_sensitivity_models.csv",
        ],
        "track_b_neonatal_fe_proxy": [
            root / "artifacts/track_b/models/neonatal_main_models.csv",
        ],
    }
    for path in path_map.get(spec.recipe_id, []):
        frame = _load_reference_result(path)
        if frame.empty:
            continue
        rows = frame.loc[
            frame["outcome_variable"].astype(str).eq(spec.outcome_variable)
            & frame["exposure_variable"].astype(str).eq(spec.exposure_variable)
        ].copy()
        if rows.empty:
            continue
        row = rows.iloc[0]
        return (
            [
                {
                    "label": str(row["outcome_label"]),
                    "estimate": row["estimate"],
                    "lower_ci": row["lower_ci"],
                    "upper_ci": row["upper_ci"],
                    "p_value": row["p_value"],
                    "estimate_scale": row["estimate_scale"],
                    "model_label": row["model_label"],
                    "valid_strata_count": row["valid_strata_count"],
                    "observation_count": row.get("observation_count", row.get("aggregated_row_count")),
                    "outcome_sum": row.get("outcome_count", row.get("outcome_total", row.get("death_count"))),
                    "denominator_sum": row.get("birth_count", row.get("annual_live_birth_denominator_sum")),
                    "metadata": {
                        "stratum_specification": row.get("stratum_specification"),
                        "exposure_bin_count": row.get("exposure_bin_count"),
                    },
                }
            ],
            {
                "outcome_label": str(row["outcome_label"]),
                "exposure_label": str(row["exposure_label"]),
                "offset_label": "Automatic denominator selected by recipe",
                "reference_artifact": str(path.relative_to(root)),
            },
        )
    return None


def _track_a_candidate(root: Path, outcome_variable: str) -> dict[str, Any]:
    triage = track_a_load_triage(root)
    candidates = track_a_candidate_rows(triage)
    for candidate in candidates:
        if str(candidate["outcome_variable"]) == outcome_variable:
            return candidate
    raise ValueError(
        f"The selected aggregated-stream outcome {outcome_variable!r} is not benchmark-eligible. Run the screening and benchmark stages first."
    )


def _context_subset(root: Path, stratifier_id: str) -> tuple[pd.DataFrame, str]:
    context = track_a_context_with_strata(root)
    stratifier_map = {item["modifier_name"]: item["stratum_column"] for item in MODIFIER_SPECS}
    if stratifier_id not in stratifier_map:
        raise ValueError(f"Unknown stratifier {stratifier_id!r}.")
    stratum_column = str(stratifier_map[stratifier_id])
    return context.loc[:, ["municipality_code", "municipality_merge_key", stratum_column]].copy(), stratum_column


def _execute_track_a_live(root: Path, registry: StatCalcRegistry, spec: AnalysisRunSpec) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    config = load_track_a_config(root)
    years = [int(year) for year in config["years"]]
    candidate = _track_a_candidate(root, spec.outcome_variable)
    frame_lookup = track_a_combine_candidate_frames(root, years, [candidate])
    frame = track_a_candidate_frame(candidate, frame_lookup)
    exposure_column = f"{spec.exposure_variable}_{spec.lag_or_window}"
    if exposure_column not in frame.columns:
        raise ValueError(
            f"Exposure {exposure_column!r} is not available in the aggregated analytical marts for {spec.outcome_variable!r}."
        )
    if spec.source_filter == "no_fallback":
        frame = frame.loc[frame["any_climate_fallback_imputation_flag"].fillna(0).eq(0)].copy()

    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    outcome_label = str(candidate["outcome_label"])
    exposure_label = track_a_exposure_label(candidate, exposure_column)

    if spec.stratifier_id:
        context_frame, stratum_column = _context_subset(root, spec.stratifier_id)
        merged = frame.merge(context_frame, on=["municipality_merge_key"], how="left", suffixes=("", "_context"))
        for stratum_value in sorted(merged[stratum_column].dropna().astype(str).unique().tolist()):
            subset = merged.loc[merged[stratum_column].astype(str).eq(stratum_value)].copy()
            model_result, model_label, estimate_scale = track_a_estimate_candidate_effect(subset, candidate, exposure_column)
            rows.append(
                {
                    "label": f"{outcome_label} - {stratum_value}",
                    "group_label": stratum_value,
                    "estimate": model_result.get("estimate"),
                    "lower_ci": model_result.get("lower_ci"),
                    "upper_ci": model_result.get("upper_ci"),
                    "p_value": model_result.get("p_value"),
                    "estimate_scale": estimate_scale,
                    "model_label": model_label,
                    "valid_strata_count": model_result.get("valid_strata_count", 0),
                    "observation_count": model_result.get("observation_count"),
                    "outcome_sum": model_result.get("outcome_sum"),
                    "denominator_sum": model_result.get("denominator_sum"),
                    "metadata": {"stratum_column": stratum_column},
                }
            )
        if not rows:
            warnings.append("No contextual strata were available for the selected stratifier.")
    else:
        model_result, model_label, estimate_scale = track_a_estimate_candidate_effect(frame, candidate, exposure_column)
        rows.append(
            {
                "label": outcome_label,
                "estimate": model_result.get("estimate"),
                "lower_ci": model_result.get("lower_ci"),
                "upper_ci": model_result.get("upper_ci"),
                "p_value": model_result.get("p_value"),
                "estimate_scale": estimate_scale,
                "model_label": model_label,
                "valid_strata_count": model_result.get("valid_strata_count", 0),
                "observation_count": model_result.get("observation_count"),
                "outcome_sum": model_result.get("outcome_sum"),
                "denominator_sum": model_result.get("denominator_sum"),
            }
        )

    return rows, {"outcome_label": outcome_label, "exposure_label": exposure_label, "offset_label": spec.offset_id}, warnings


def _merge_track_b_context(root: Path, frame: pd.DataFrame, stratifier_id: str) -> tuple[pd.DataFrame, str]:
    context_frame, stratum_column = _context_subset(root, stratifier_id)
    merged = frame.merge(context_frame.loc[:, ["municipality_code", stratum_column]], on="municipality_code", how="left")
    return merged, stratum_column


def _execute_track_b_preterm_live(root: Path, spec: AnalysisRunSpec) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    config = load_track_b_config(root)
    years = [int(year) for year in config["years"]]
    frame = load_track_b_matrix_partitions(root, years, columns=TRACK_B_PRETERM_COLUMNS)
    live_births = frame.loc[
        frame["record_type"].eq("live_birth")
        & pd.to_numeric(frame["gestational_age_valid_for_analysis_flag"], errors="coerce").fillna(0).eq(1)
    ].copy()
    live_births["birth_count"] = 1.0
    live_births["month_of_conception"] = pd.to_numeric(live_births["month_of_conception"], errors="coerce").fillna(-1).astype(int)
    live_births["source_sensitive_flag"] = (
        pd.to_numeric(live_births["third_trimester_fallback_day_share"], errors="coerce").fillna(0).gt(0)
    ).astype(int)
    live_births["third_trimester_mean_tmax_x_rh_mean_pct"] = (
        pd.to_numeric(live_births["third_trimester_mean_tmax_c"], errors="coerce")
        * pd.to_numeric(live_births["third_trimester_mean_rh_mean_pct"], errors="coerce")
    )
    if spec.gestational_source_mode == "exact_only":
        live_births = live_births.loc[pd.to_numeric(live_births["ga_exact_measurement_flag"], errors="coerce").fillna(0).eq(1)].copy()

    observed_flag, fallback_column = TRACK_B_PRETERM_EXPOSURE_FLAGS[spec.exposure_variable]
    subset = live_births.loc[pd.to_numeric(live_births[observed_flag], errors="coerce").fillna(0).eq(1)].copy()
    warnings: list[str] = []
    if spec.source_filter == "no_fallback" and fallback_column:
        subset = subset.loc[pd.to_numeric(subset[fallback_column], errors="coerce").fillna(0).eq(0)].copy()
    elif spec.source_filter == "no_fallback":
        warnings.append("The selected event-level preterm exposure does not expose fallback-day filtering; observed rows were used without additional source restriction.")

    outcome_label = TRACK_B_PRETERM_OUTCOME_LABELS[spec.outcome_variable]
    rows: list[dict[str, Any]] = []
    if spec.stratifier_id:
        merged, stratum_column = _merge_track_b_context(root, subset, spec.stratifier_id)
        for stratum_value in sorted(merged[stratum_column].dropna().astype(str).unique().tolist()):
            stratum_subset = merged.loc[merged[stratum_column].astype(str).eq(stratum_value)].copy()
            aggregated, result, stratum_specification, bin_count = track_b_preterm_fit_with_fallback(
                stratum_subset,
                exposure_column=spec.exposure_variable,
                outcome_column=spec.outcome_variable,
            )
            rows.append(
                {
                    "label": f"{outcome_label} - {stratum_value}",
                    "group_label": stratum_value,
                    "estimate": result.get("estimate"),
                    "lower_ci": result.get("lower_ci"),
                    "upper_ci": result.get("upper_ci"),
                    "p_value": result.get("p_value"),
                    "estimate_scale": "mean_difference_per_one_unit_increase"
                    if spec.outcome_variable == "gestational_days_lost_below_37_weeks"
                    else "rate_ratio_per_one_unit_increase",
                    "model_label": "gaussian_fe_proxy" if spec.outcome_variable == "gestational_days_lost_below_37_weeks" else "modified_poisson_fe_proxy",
                    "valid_strata_count": result.get("valid_strata_count", 0),
                    "observation_count": int(aggregated.shape[0]),
                    "outcome_sum": _safe_float(aggregated[spec.outcome_variable].sum()) if not aggregated.empty else 0.0,
                    "denominator_sum": _safe_float(aggregated["denominator_births"].sum()) if not aggregated.empty else 0.0,
                    "metadata": {"stratum_specification": stratum_specification, "exposure_bin_count": bin_count},
                }
            )
    else:
        aggregated, result, stratum_specification, bin_count = track_b_preterm_fit_with_fallback(
            subset,
            exposure_column=spec.exposure_variable,
            outcome_column=spec.outcome_variable,
        )
        rows.append(
            {
                "label": outcome_label,
                "estimate": result.get("estimate"),
                "lower_ci": result.get("lower_ci"),
                "upper_ci": result.get("upper_ci"),
                "p_value": result.get("p_value"),
                "estimate_scale": "mean_difference_per_one_unit_increase"
                if spec.outcome_variable == "gestational_days_lost_below_37_weeks"
                else "rate_ratio_per_one_unit_increase",
                "model_label": "gaussian_fe_proxy" if spec.outcome_variable == "gestational_days_lost_below_37_weeks" else "modified_poisson_fe_proxy",
                "valid_strata_count": result.get("valid_strata_count", 0),
                "observation_count": int(aggregated.shape[0]),
                "outcome_sum": _safe_float(aggregated[spec.outcome_variable].sum()) if not aggregated.empty else 0.0,
                "denominator_sum": _safe_float(aggregated["denominator_births"].sum()) if not aggregated.empty else 0.0,
                "metadata": {"stratum_specification": stratum_specification, "exposure_bin_count": bin_count},
            }
        )
    return rows, {"outcome_label": outcome_label, "exposure_label": spec.exposure_variable, "offset_label": spec.offset_id}, warnings


def _execute_track_b_stillbirth_live(root: Path, spec: AnalysisRunSpec) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    config = load_track_b_config(root)
    years = [int(year) for year in config["years"]]
    frame = load_track_b_matrix_partitions(root, years, columns=TRACK_B_STILLBIRTH_COLUMNS)
    frame["birth_count"] = 1.0
    frame["event_month"] = pd.to_datetime(frame["event_date"], errors="coerce").dt.month.fillna(-1).astype(int)
    frame["source_sensitive_flag"] = (
        pd.to_numeric(frame["days_0_6_before_event_fallback_day_share"], errors="coerce").fillna(0).gt(0)
    ).astype(int)
    observed_flag, fallback_column = TRACK_B_STILLBIRTH_EXPOSURE_FLAGS[spec.exposure_variable]
    subset = frame.loc[pd.to_numeric(frame[observed_flag], errors="coerce").fillna(0).eq(1)].copy()
    warnings: list[str] = []
    if spec.source_filter == "no_fallback" and fallback_column:
        subset = subset.loc[pd.to_numeric(subset[fallback_column], errors="coerce").fillna(0).eq(0)].copy()
    elif spec.source_filter == "no_fallback":
        warnings.append("Fallback-day source restriction is not available for this stillbirth exposure; observed rows were used.")

    outcome_label = TRACK_B_STILLBIRTH_OUTCOME_LABELS[spec.outcome_variable]
    rows: list[dict[str, Any]] = []
    if spec.stratifier_id:
        merged, stratum_column = _merge_track_b_context(root, subset, spec.stratifier_id)
        for stratum_value in sorted(merged[stratum_column].dropna().astype(str).unique().tolist()):
            stratum_subset = merged.loc[merged[stratum_column].astype(str).eq(stratum_value)].copy()
            aggregated, result, stratum_specification, bin_count = track_b_stillbirth_fit_with_fallback(
                stratum_subset,
                exposure_column=spec.exposure_variable,
                outcome_column=spec.outcome_variable,
            )
            rows.append(
                {
                    "label": f"{outcome_label} - {stratum_value}",
                    "group_label": stratum_value,
                    "estimate": result.get("estimate"),
                    "lower_ci": result.get("lower_ci"),
                    "upper_ci": result.get("upper_ci"),
                    "p_value": result.get("p_value"),
                    "estimate_scale": "rate_ratio_per_one_unit_increase",
                    "model_label": "poisson_fe_proxy_total_birth_offset",
                    "valid_strata_count": result.get("valid_strata_count", 0),
                    "observation_count": int(aggregated.shape[0]),
                    "outcome_sum": _safe_float(aggregated[spec.outcome_variable].sum()) if not aggregated.empty else 0.0,
                    "denominator_sum": _safe_float(aggregated["denominator_total_births"].sum()) if not aggregated.empty else 0.0,
                    "metadata": {"stratum_specification": stratum_specification, "exposure_bin_count": bin_count},
                }
            )
    else:
        aggregated, result, stratum_specification, bin_count = track_b_stillbirth_fit_with_fallback(
            subset,
            exposure_column=spec.exposure_variable,
            outcome_column=spec.outcome_variable,
        )
        rows.append(
            {
                "label": outcome_label,
                "estimate": result.get("estimate"),
                "lower_ci": result.get("lower_ci"),
                "upper_ci": result.get("upper_ci"),
                "p_value": result.get("p_value"),
                "estimate_scale": "rate_ratio_per_one_unit_increase",
                "model_label": "poisson_fe_proxy_total_birth_offset",
                "valid_strata_count": result.get("valid_strata_count", 0),
                "observation_count": int(aggregated.shape[0]),
                "outcome_sum": _safe_float(aggregated[spec.outcome_variable].sum()) if not aggregated.empty else 0.0,
                "denominator_sum": _safe_float(aggregated["denominator_total_births"].sum()) if not aggregated.empty else 0.0,
                "metadata": {"stratum_specification": stratum_specification, "exposure_bin_count": bin_count},
            }
        )
    return rows, {"outcome_label": outcome_label, "exposure_label": spec.exposure_variable, "offset_label": spec.offset_id}, warnings


def _execute_track_b_neonatal_live(root: Path, spec: AnalysisRunSpec) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    config = load_track_b_config(root)
    years = [int(year) for year in config["years"]]
    frame = track_b_load_mortality(root, years)
    frame = track_b_add_lags(frame)
    frame = frame.loc[
        frame["QC_STATUS"].astype(str).eq("PASS")
        & pd.to_numeric(frame["OFFSET_VALID_FLAG"], errors="coerce").fillna(0).eq(1)
        & pd.to_numeric(frame["DENOMINATOR_ZERO_FLAG"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(frame["DENOMINATOR_MISSING_FLAG"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(frame["RATE_NOT_COMPUTABLE_FLAG"], errors="coerce").fillna(0).eq(0)
    ].copy()
    frame["epidemiologic_week"] = frame["observation_date"].dt.isocalendar().week.astype(int)
    if spec.source_filter == "no_fallback":
        frame = frame.loc[pd.to_numeric(frame["IMPUTADO_POR_FALLBACK"], errors="coerce").fillna(0).eq(0)].copy()

    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    def _single_result(subset: pd.DataFrame, label: str, group_label: str | None = None) -> dict[str, Any]:
        result = pooled_continuous_proxy(
            subset,
            exposure_column=spec.exposure_variable,
            outcome_column="DEATHS_NEONATAL",
            denominator_column="LIVE_BIRTHS_TOTAL_ANNUAL",
            stratum_columns=["municipality_code", "calendar_year", "calendar_month", "day_of_week"],
            model_kind="count",
            exposure_scale_value=1.0,
        )
        return {
            "label": label,
            "group_label": group_label,
            "estimate": result.get("estimate"),
            "lower_ci": result.get("lower_ci"),
            "upper_ci": result.get("upper_ci"),
            "p_value": result.get("p_value"),
            "estimate_scale": "rate_ratio_per_one_degree_celsius_increase",
            "model_label": "poisson_fe_proxy",
            "valid_strata_count": result.get("valid_strata_count", 0),
            "observation_count": result.get("observation_count"),
            "outcome_sum": result.get("outcome_sum"),
            "denominator_sum": result.get("denominator_sum"),
        }

    if spec.stratifier_id:
        merged, stratum_column = _merge_track_b_context(root, frame, spec.stratifier_id)
        for stratum_value in sorted(merged[stratum_column].dropna().astype(str).unique().tolist()):
            subset = merged.loc[merged[stratum_column].astype(str).eq(stratum_value)].copy()
            rows.append(_single_result(subset, f"Neonatal mortality - {stratum_value}", stratum_value))
    else:
        rows.append(_single_result(frame, "Neonatal mortality"))
    return rows, {"outcome_label": "Neonatal mortality", "exposure_label": spec.exposure_variable, "offset_label": spec.offset_id}, warnings


def resolve_reference(root: Path, registry: StatCalcRegistry, spec: AnalysisRunSpec) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    recipe = registry.recipe_map[spec.recipe_id]
    if recipe.track_id == "track_a":
        return _resolve_track_a_reference(root, spec)
    return _resolve_track_b_reference(root, spec)


def execute_live(root: Path, registry: StatCalcRegistry, spec: AnalysisRunSpec) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    recipe = registry.recipe_map[spec.recipe_id]
    if recipe.recipe_id == "track_a_acute_fe_proxy" or recipe.recipe_id == "track_a_perinatal_fe_proxy" or recipe.recipe_id == "track_a_maternal_fe_proxy":
        return _execute_track_a_live(root, registry, spec)
    if recipe.recipe_id == "track_b_preterm_fe_proxy":
        return _execute_track_b_preterm_live(root, spec)
    if recipe.recipe_id == "track_b_stillbirth_fe_proxy":
        return _execute_track_b_stillbirth_live(root, spec)
    if recipe.recipe_id == "track_b_neonatal_fe_proxy":
        return _execute_track_b_neonatal_live(root, spec)
    raise ValueError(f"No live executor is configured for recipe {recipe.recipe_id!r}.")


def submit_run(root: Path, storage: StatCalcStorage, registry: StatCalcRegistry, spec: AnalysisRunSpec) -> tuple[Any, AnalysisRunResult | None]:
    recipe, normalized_spec = registry.validate_spec(spec)
    reference = resolve_reference(root, registry, normalized_spec)
    if reference is not None:
        run = storage.create_run(
            track_id=recipe.track_id,
            recipe_id=recipe.recipe_id,
            execution_mode="reference",
            is_reference=True,
            status="running",
            spec=normalized_spec,
        )
        storage.mark_run_running(run.run_id)
        try:
            rows, summary = reference
            summary = {
                **summary,
                "execution_behavior": recipe.execution_behavior,
                "track_id": recipe.track_id,
                "recipe_label": recipe.label,
            }
            result = _persist_result(
                storage,
                run.run_id,
                track_id=recipe.track_id,
                recipe_id=recipe.recipe_id,
                execution_mode="reference",
                spec=normalized_spec,
                summary=summary,
                rows=rows,
                warnings=[],
            )
            return storage.get_run(run.run_id), result
        except Exception as exc:
            storage.set_run_result(run.run_id, status="failed", warnings=[], error_message=str(exc))
            raise

    if recipe.track_id == "track_a":
        run = storage.create_run(
            track_id=recipe.track_id,
            recipe_id=recipe.recipe_id,
            execution_mode="live",
            is_reference=False,
            status="running",
            spec=normalized_spec,
        )
        storage.mark_run_running(run.run_id)
        try:
            rows, summary, warnings = execute_live(root, registry, normalized_spec)
            summary = {
                **summary,
                "execution_behavior": recipe.execution_behavior,
                "track_id": recipe.track_id,
                "recipe_label": recipe.label,
            }
            result = _persist_result(
                storage,
                run.run_id,
                track_id=recipe.track_id,
                recipe_id=recipe.recipe_id,
                execution_mode="live",
                spec=normalized_spec,
                summary=summary,
                rows=rows,
                warnings=warnings,
            )
            return storage.get_run(run.run_id), result
        except Exception as exc:
            storage.set_run_result(run.run_id, status="failed", warnings=[], error_message=str(exc))
            raise

    run = storage.create_run(
        track_id=recipe.track_id,
        recipe_id=recipe.recipe_id,
        execution_mode="live",
        is_reference=False,
        status="queued",
        spec=normalized_spec,
    )
    storage.enqueue_run(run.run_id)
    return storage.get_run(run.run_id), None


def run_queued_job(root: Path, storage_dir: Path, run_id: str) -> int:
    storage = StatCalcStorage(storage_dir)
    registry = StatCalcRegistry(root)
    run = storage.get_run(run_id)
    try:
        rows, summary, warnings = execute_live(root, registry, run.spec)
        recipe = registry.recipe_map[run.recipe_id]
        summary = {
            **summary,
            "execution_behavior": recipe.execution_behavior,
            "track_id": recipe.track_id,
            "recipe_label": recipe.label,
        }
        _persist_result(
            storage,
            run_id,
            track_id=run.track_id,
            recipe_id=run.recipe_id,
            execution_mode=run.execution_mode,
            spec=run.spec,
            summary=summary,
            rows=rows,
            warnings=warnings,
        )
        storage.mark_job_finished(run_id, "completed")
        return 0
    except Exception as exc:
        storage.set_run_result(run_id, status="failed", warnings=[], error_message=str(exc))
        storage.mark_job_finished(run_id, "failed")
        raise


class TrackBWorker:
    def __init__(self, root: Path, storage_dir: Path):
        self.root = root
        self.storage_dir = storage_dir
        self.storage = StatCalcStorage(storage_dir)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.storage.mark_interrupted_running_jobs()
        self._thread = threading.Thread(target=self._loop, name="statcalc-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            run_id = self.storage.claim_next_queued_run()
            if run_id is None:
                self._stop_event.wait(1.0)
                continue
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "heat_mnch.cli",
                        "statcalc-run-job",
                        "--run-id",
                        run_id,
                        "--storage-dir",
                        str(self.storage_dir),
                        "--root",
                        str(self.root),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    error_message = completed.stderr.strip() or completed.stdout.strip() or "Event-level job failed."
                    self.storage.set_run_result(run_id, status="failed", warnings=[], error_message=error_message)
                    self.storage.mark_job_finished(run_id, "failed")
            except Exception as exc:
                self.storage.set_run_result(run_id, status="failed", warnings=[], error_message=str(exc))
                self.storage.mark_job_finished(run_id, "failed")
            time.sleep(0.05)
