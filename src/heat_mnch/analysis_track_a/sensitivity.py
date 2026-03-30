from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from .benchmark import (
    COUNT_MODEL_SCALE,
    SEVERITY_MODEL_SCALE,
    _acute_effect_label,
    _candidate_rows,
    _dispersion_adjusted_result,
    _pooled_continuous_fe_proxy,
    _split_acute_exposure_column,
    _split_window_exposure_column,
)
from .config import load_track_a_config
from .modeling import (
    ACUTE_TEMPERATURE_LAG_SUFFIXES,
    BASE_EXPOSURES,
    DELIVERY_WINDOW_SUFFIXES,
)
from .registry import track_a_root


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COUNT_NULL_VALUE = 1.0
SEVERITY_NULL_VALUE = 0.0
COASTAL_STRATUM_MAP = {0: "inland", 1: "coastal"}
CENTROID_LOOKUP_RELATIVE_PATH = Path("metadata/era5_municipio_grid_lookup.csv")

MODIFIER_SPECS: list[dict[str, str]] = [
    {
        "modifier_name": "urban_population_proportion",
        "modifier_label": "Urban population proportion",
        "context_column": "urban_population_proportion",
        "stratum_column": "urbanicity_tertile",
        "strata_kind": "tertile",
        "low_label": "low urbanicity tertile",
        "middle_label": "middle urbanicity tertile",
        "high_label": "high urbanicity tertile",
        "composite_high_risk_label": "high urbanicity tertile",
    },
    {
        "modifier_name": "expanded_gini_index",
        "modifier_label": "Expanded Gini index",
        "context_column": "expanded_gini_index",
        "stratum_column": "inequality_tertile",
        "strata_kind": "tertile",
        "low_label": "low inequality tertile",
        "middle_label": "middle inequality tertile",
        "high_label": "high inequality tertile",
        "composite_high_risk_label": "high inequality tertile",
    },
    {
        "modifier_name": "expanded_human_development_index",
        "modifier_label": "Expanded human development index",
        "context_column": "expanded_human_development_index",
        "stratum_column": "human_development_tertile",
        "strata_kind": "tertile",
        "low_label": "low human development tertile",
        "middle_label": "middle human development tertile",
        "high_label": "high human development tertile",
        "composite_high_risk_label": "low human development tertile",
    },
    {
        "modifier_name": "mean_altitude_m",
        "modifier_label": "Mean altitude",
        "context_column": "mean_altitude_m",
        "stratum_column": "altitude_tertile",
        "strata_kind": "tertile",
        "low_label": "low altitude tertile",
        "middle_label": "middle altitude tertile",
        "high_label": "high altitude tertile",
        "composite_high_risk_label": "",
    },
    {
        "modifier_name": "coastal_municipality_flag",
        "modifier_label": "Coastal municipality status",
        "context_column": "coastal_municipality_flag",
        "stratum_column": "coastal_stratum",
        "strata_kind": "binary",
        "low_label": "inland",
        "middle_label": "",
        "high_label": "coastal",
        "composite_high_risk_label": "coastal",
    },
]


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _sensitivity_root(root: Path) -> Path:
    return track_a_root(root) / "sensitivity"


def _screening_root(root: Path) -> Path:
    return track_a_root(root) / "screening"


def _benchmark_root(root: Path) -> Path:
    return track_a_root(root) / "models"


def _mart_partition_path(root: Path, mart_name: str, year: int) -> Path:
    return track_a_root(root) / "analysis_marts" / mart_name / f"YEAR={year}" / f"{mart_name}_{year}.parquet"


def _load_triage(root: Path) -> pd.DataFrame:
    path = _screening_root(root) / "triage_signal_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stage 7 requires the Stage 4 triage matrix at {path}. Run track-a-screening first.")
    return pd.read_csv(path)


def _load_benchmark_results(root: Path) -> pd.DataFrame:
    path = _benchmark_root(root) / "benchmark_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stage 7 requires the Stage 6 benchmark results at {path}. Run track-a-benchmark first.")
    return pd.read_csv(path)


def _load_context(root: Path) -> pd.DataFrame:
    path = track_a_root(root) / "analysis_marts" / "context_lookup.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Stage 7 requires the Stage 2 context asset at {path}. Run track-a-build-marts first.")
    context = pd.read_parquet(path).copy()
    context["municipality_code"] = context["municipality_code"].astype(str)
    context["municipality_merge_key"] = _municipality_merge_key(context["municipality_code"])
    return context


def _load_centroids(root: Path) -> pd.DataFrame:
    path = root / CENTROID_LOOKUP_RELATIVE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Stage 7 requires municipality centroids at {path}.")
    centroids = pd.read_csv(path, dtype={"COD_MUNICIPIO": str})
    centroids = centroids.rename(
        columns={
            "COD_MUNICIPIO": "municipality_code",
            "NM_MUNICIPIO": "municipality_name",
            "UF": "uf",
            "MUNICIPIO_CENTROID_LAT": "municipality_centroid_lat",
            "MUNICIPIO_CENTROID_LON": "municipality_centroid_lon",
        }
    )
    centroids["municipality_code"] = centroids["municipality_code"].astype(str)
    centroids["municipality_merge_key"] = _municipality_merge_key(centroids["municipality_code"])
    return centroids[
        [
            "municipality_code",
            "municipality_merge_key",
            "municipality_name",
            "uf",
            "municipality_centroid_lat",
            "municipality_centroid_lon",
        ]
    ]


def _municipality_merge_key(series: pd.Series) -> pd.Series:
    numeric = series.astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    return numeric.map(lambda value: value[:6] if len(value) >= 6 else value)


def _ranked_tertile(
    series: pd.Series,
    *,
    low_label: str,
    middle_label: str,
    high_label: str,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    output = pd.Series(index=series.index, dtype="object")
    valid = numeric.dropna()
    if valid.empty:
        return output
    percentile_rank = valid.rank(method="first", pct=True)
    bins = pd.cut(
        percentile_rank,
        bins=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        labels=[low_label, middle_label, high_label],
        include_lowest=True,
    )
    output.loc[valid.index] = bins.astype("object")
    return output


def _context_with_strata(root: Path) -> pd.DataFrame:
    context = _load_context(root)
    context["urbanicity_tertile"] = _ranked_tertile(
        context["urban_population_proportion"],
        low_label="low urbanicity tertile",
        middle_label="middle urbanicity tertile",
        high_label="high urbanicity tertile",
    )
    context["inequality_tertile"] = _ranked_tertile(
        context["expanded_gini_index"],
        low_label="low inequality tertile",
        middle_label="middle inequality tertile",
        high_label="high inequality tertile",
    )
    context["human_development_tertile"] = _ranked_tertile(
        context["expanded_human_development_index"],
        low_label="low human development tertile",
        middle_label="middle human development tertile",
        high_label="high human development tertile",
    )
    context["altitude_tertile"] = _ranked_tertile(
        context["mean_altitude_m"],
        low_label="low altitude tertile",
        middle_label="middle altitude tertile",
        high_label="high altitude tertile",
    )
    context["coastal_stratum"] = (
        pd.to_numeric(context["coastal_municipality_flag"], errors="coerce")
        .fillna(0)
        .astype(int)
        .map(COASTAL_STRATUM_MAP)
    )
    context["contextual_vulnerability_marker_count"] = (
        context["urbanicity_tertile"].eq("high urbanicity tertile").astype(int)
        + context["inequality_tertile"].eq("high inequality tertile").astype(int)
        + context["human_development_tertile"].eq("low human development tertile").astype(int)
        + context["coastal_stratum"].eq("coastal").astype(int)
    )
    centroids = _load_centroids(root)
    merged = context.merge(
        centroids,
        on="municipality_merge_key",
        how="left",
        suffixes=("_context", ""),
    )
    merged["municipality_code"] = merged["municipality_code"].fillna(merged["municipality_code_context"])
    return merged.drop(columns=["municipality_code_context"])


def _acute_stage7_columns(candidates: list[dict[str, Any]]) -> list[str]:
    exposure_columns: set[str] = set()
    outcome_columns: set[str] = set()
    denominator_columns: set[str] = set()
    for candidate in candidates:
        if candidate["outcome_family"] != "acute_daily":
            continue
        outcome_columns.add(str(candidate["outcome_variable"]))
        denominator_columns.add(str(candidate["denominator_column"]))
        for base_exposure in BASE_EXPOSURES:
            for suffix in ACUTE_TEMPERATURE_LAG_SUFFIXES:
                exposure_columns.add(f"{base_exposure}_{suffix}")
    return sorted(
        {
            "municipality_code",
            "calendar_year",
            "observation_date",
            "quality_control_status",
            "offset_valid_flag",
            "denominator_zero_flag",
            "denominator_missing_flag",
            "rate_not_computable_flag",
            "final_climate_source",
            "any_climate_fallback_imputation_flag",
            *outcome_columns,
            *denominator_columns,
            *exposure_columns,
        }
    )


def _window_stage7_columns(candidates: list[dict[str, Any]], mart_name: str) -> list[str]:
    exposure_columns: set[str] = set()
    outcome_columns: set[str] = set()
    denominator_columns: set[str] = set()
    for candidate in candidates:
        if candidate["mart_name"] != mart_name:
            continue
        outcome_columns.add(str(candidate["outcome_variable"]))
        denominator_columns.add(str(candidate["denominator_column"]))
        for base_exposure in BASE_EXPOSURES:
            for suffix in DELIVERY_WINDOW_SUFFIXES:
                exposure_columns.add(f"{base_exposure}_{suffix}")
    return sorted(
        {
            "municipality_code",
            "calendar_year",
            "observation_date",
            "quality_control_status",
            "offset_valid_flag",
            "denominator_zero_flag",
            "denominator_missing_flag",
            "rate_not_computable_flag",
            "final_climate_source",
            "any_climate_fallback_imputation_flag",
            *outcome_columns,
            *denominator_columns,
            *exposure_columns,
        }
    )


def _load_partition(root: Path, mart_name: str, year: int, columns: list[str]) -> pd.DataFrame:
    path = _mart_partition_path(root, mart_name, year)
    if not path.exists():
        raise FileNotFoundError(f"Stage 7 requires the Stage 2 mart partition at {path}. Run track-a-build-marts first.")
    frame = pd.read_parquet(path, columns=columns)
    frame["municipality_code"] = frame["municipality_code"].astype(str)
    frame["municipality_merge_key"] = _municipality_merge_key(frame["municipality_code"])
    return frame


def _combine_candidate_frames(root: Path, years: list[int], candidates: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    acute_candidates = [candidate for candidate in candidates if candidate["mart_name"] == "acute_daily_mart"]
    perinatal_candidates = [
        candidate for candidate in candidates if candidate["mart_name"] == "perinatal_delivery_window_mart"
    ]
    maternal_candidates = [candidate for candidate in candidates if candidate["mart_name"] == "maternal_operational_mart"]

    frames: dict[str, pd.DataFrame] = {}
    if acute_candidates:
        frames["acute_daily_mart"] = pd.concat(
            [
                _load_partition(root, "acute_daily_mart", year, _acute_stage7_columns(acute_candidates))
                for year in years
            ],
            ignore_index=True,
        )
    if perinatal_candidates:
        frames["perinatal_delivery_window_mart"] = pd.concat(
            [
                _load_partition(
                    root,
                    "perinatal_delivery_window_mart",
                    year,
                    _window_stage7_columns(perinatal_candidates, "perinatal_delivery_window_mart"),
                )
                for year in years
            ],
            ignore_index=True,
        )
    if maternal_candidates:
        frames["maternal_operational_mart"] = pd.concat(
            [
                _load_partition(
                    root,
                    "maternal_operational_mart",
                    year,
                    _window_stage7_columns(maternal_candidates, "maternal_operational_mart"),
                )
                for year in years
            ],
            ignore_index=True,
        )
    return frames


def _candidate_frame(candidate: dict[str, Any], frame_lookup: dict[str, pd.DataFrame]) -> pd.DataFrame:
    source = frame_lookup[candidate["mart_name"]]
    selected_exposure = _selected_exposure_column(candidate)
    columns = [
        "municipality_code",
        "municipality_merge_key",
        "calendar_year",
        "observation_date",
        "quality_control_status",
        "offset_valid_flag",
        "denominator_zero_flag",
        "denominator_missing_flag",
        "rate_not_computable_flag",
        "final_climate_source",
        "any_climate_fallback_imputation_flag",
        candidate["denominator_column"],
        candidate["outcome_variable"],
    ]
    if candidate["outcome_family"] == "acute_daily":
        base_exposure, _ = _split_acute_exposure_column(selected_exposure)
        columns.extend(f"{base_exposure}_{suffix}" for suffix in ACUTE_TEMPERATURE_LAG_SUFFIXES)
        for exposure_name in BASE_EXPOSURES:
            columns.extend(f"{exposure_name}_{suffix}" for suffix in ACUTE_TEMPERATURE_LAG_SUFFIXES)
    else:
        base_exposure, _ = _split_window_exposure_column(selected_exposure)
        columns.extend(f"{base_exposure}_{suffix}" for suffix in DELIVERY_WINDOW_SUFFIXES)
        for exposure_name in BASE_EXPOSURES:
            columns.extend(f"{exposure_name}_{suffix}" for suffix in DELIVERY_WINDOW_SUFFIXES)
    unique_columns = list(dict.fromkeys(columns))
    return source.loc[:, unique_columns].copy()


def _selected_exposure_column(candidate: dict[str, Any]) -> str:
    best_exposure = str(candidate["best_exposure_variable"])
    if best_exposure.endswith("_weekly_anomaly"):
        base_exposure = best_exposure.removesuffix("_weekly_anomaly")
        if base_exposure in BASE_EXPOSURES:
            return f"{base_exposure}_lag0"
    if best_exposure not in BASE_EXPOSURES:
        return best_exposure
    best_lag_or_window = str(candidate["best_lag_or_window"])
    if candidate["outcome_family"] == "acute_daily":
        if best_lag_or_window not in ACUTE_TEMPERATURE_LAG_SUFFIXES:
            best_lag_or_window = "lag0"
    else:
        if best_lag_or_window not in DELIVERY_WINDOW_SUFFIXES:
            best_lag_or_window = "days_0_6_before_event_mean"
    return f"{best_exposure}_{best_lag_or_window}"


def _effect_direction(estimate: float, estimate_scale: str) -> str:
    if not math.isfinite(float(estimate)):
        return "not_estimated"
    if estimate_scale == SEVERITY_MODEL_SCALE:
        if abs(float(estimate)) < 1e-12:
            return "null"
        return "positive" if float(estimate) > 0 else "negative"
    if abs(float(estimate) - COUNT_NULL_VALUE) < 1e-12:
        return "null"
    return "positive" if float(estimate) > COUNT_NULL_VALUE else "negative"


def _direction_vs_reference(estimate: float, reference_estimate: float, estimate_scale: str) -> str:
    current_direction = _effect_direction(estimate, estimate_scale)
    reference_direction = _effect_direction(reference_estimate, estimate_scale)
    if current_direction in {"not_estimated", "null"} or reference_direction in {"not_estimated", "null"}:
        return "indeterminate"
    if current_direction == reference_direction:
        return "same_direction"
    return "opposite_direction"


def _relative_change_pct(estimate: float, reference_estimate: float, estimate_scale: str) -> float:
    if not math.isfinite(float(estimate)) or not math.isfinite(float(reference_estimate)):
        return float("nan")
    if estimate_scale == SEVERITY_MODEL_SCALE:
        if abs(float(reference_estimate)) < 1e-12:
            return float("nan")
        return ((float(estimate) - float(reference_estimate)) / abs(float(reference_estimate))) * 100.0
    if abs(float(reference_estimate)) < 1e-12:
        return float("nan")
    return ((float(estimate) / float(reference_estimate)) - 1.0) * 100.0


def _estimate_candidate_effect(
    frame: pd.DataFrame,
    candidate: dict[str, Any],
    exposure_column: str,
) -> tuple[dict[str, Any], str, str]:
    model_result = _pooled_continuous_fe_proxy(
        frame,
        exposure_column=exposure_column,
        outcome_column=str(candidate["outcome_variable"]),
        denominator_column=str(candidate["denominator_column"]),
        stratum_columns=["municipality_code", "calendar_year"],
        model_kind=str(candidate["model_kind"]),
    )
    if candidate["model_kind"] == "severity":
        return model_result, "gaussian_fe_proxy", SEVERITY_MODEL_SCALE
    if math.isfinite(float(model_result.get("overdispersion_ratio", float("nan")))) and float(
        model_result["overdispersion_ratio"]
    ) > 1.5:
        return _dispersion_adjusted_result(model_result, model_kind="count"), "negative_binomial_fe_proxy", COUNT_MODEL_SCALE
    return model_result, "poisson_fe_proxy", COUNT_MODEL_SCALE


def _exposure_label(candidate: dict[str, Any], exposure_column: str) -> str:
    if candidate["outcome_family"] == "acute_daily":
        base_exposure, suffix = _split_acute_exposure_column(exposure_column)
        return _acute_effect_label(base_exposure, suffix)
    base_exposure, suffix = _split_window_exposure_column(exposure_column)
    return f"{BASE_EXPOSURES[base_exposure]['label']} ({suffix.replace('_', ' ')})"


def _build_sensitivity_row(
    *,
    candidate: dict[str, Any],
    sensitivity_domain: str,
    sensitivity_label: str,
    exposure_column: str,
    lag_or_window: str,
    sample_filter: str,
    final_climate_source: str,
    model_label: str,
    estimate_scale: str,
    model_result: dict[str, Any],
    reference_estimate: float,
    reference_lower_ci: float,
    reference_upper_ci: float,
    reference_p_value: float,
) -> dict[str, Any]:
    exposure_base = exposure_column
    if candidate["outcome_family"] == "acute_daily":
        exposure_base, _ = _split_acute_exposure_column(exposure_column)
    elif "_days_" in exposure_column:
        exposure_base, _ = _split_window_exposure_column(exposure_column)
    exposure_meta = BASE_EXPOSURES.get(exposure_base, {"family_id": "", "family_name": "", "label": exposure_column})
    estimate = float(model_result.get("estimate", float("nan")))
    relative_change_pct = _relative_change_pct(estimate, reference_estimate, estimate_scale)
    direction_vs_reference = _direction_vs_reference(estimate, reference_estimate, estimate_scale)
    return {
        "public_concept_name": candidate["public_concept_name"],
        "public_concept_label": candidate["public_concept_label"],
        "outcome_variable": candidate["outcome_variable"],
        "outcome_label": candidate["outcome_label"],
        "outcome_family": candidate["outcome_family"],
        "stage1_classification": candidate["stage1_classification"],
        "triage_priority_classification": candidate["triage_priority_classification"],
        "benchmark_stage_recommendation": "yes",
        "sensitivity_domain": sensitivity_domain,
        "sensitivity_label": sensitivity_label,
        "sample_filter": sample_filter,
        "final_climate_source": final_climate_source,
        "exposure_variable": exposure_column,
        "exposure_label": _exposure_label(candidate, exposure_column),
        "exposure_family_id": exposure_meta["family_id"],
        "exposure_family_name": exposure_meta["family_name"],
        "lag_or_window": lag_or_window,
        "model_label": model_label,
        "estimate_scale": estimate_scale,
        "estimate": estimate,
        "lower_ci": model_result.get("lower_ci", float("nan")),
        "upper_ci": model_result.get("upper_ci", float("nan")),
        "p_value": model_result.get("p_value", float("nan")),
        "p_value_floor_applied_flag": model_result.get("p_value_floor_applied_flag", "no"),
        "valid_strata_count": model_result.get("valid_strata_count", 0),
        "observation_count": model_result.get("observation_count", 0),
        "outcome_sum": model_result.get("outcome_sum", 0.0),
        "denominator_sum": model_result.get("denominator_sum", 0.0),
        "overdispersion_ratio": model_result.get("overdispersion_ratio", float("nan")),
        "reference_estimate": reference_estimate,
        "reference_lower_ci": reference_lower_ci,
        "reference_upper_ci": reference_upper_ci,
        "reference_p_value": reference_p_value,
        "direction_vs_reference": direction_vs_reference,
        "relative_change_pct": relative_change_pct,
        "source_dependence_flag": (
            "yes"
            if sensitivity_domain in {"source_restriction_no_fallback", "climate_source_stratum"}
            and (direction_vs_reference == "opposite_direction" or (math.isfinite(relative_change_pct) and abs(relative_change_pct) >= 10.0))
            else "no"
        ),
    }


def _build_heterogeneity_row(
    *,
    candidate: dict[str, Any],
    modifier_spec: dict[str, str],
    modifier_stratum: str,
    exposure_column: str,
    model_label: str,
    estimate_scale: str,
    model_result: dict[str, Any],
    municipality_count: int,
    reference_estimate: float,
) -> dict[str, Any]:
    direction_vs_full_sample = _direction_vs_reference(
        float(model_result.get("estimate", float("nan"))),
        reference_estimate,
        estimate_scale,
    )
    if int(model_result.get("valid_strata_count", 0)) <= 0:
        heterogeneity_signal_classification = "insufficient_data"
    elif direction_vs_full_sample == "same_direction":
        heterogeneity_signal_classification = "directionally_consistent"
    elif direction_vs_full_sample == "opposite_direction":
        heterogeneity_signal_classification = "directionally_inconsistent"
    else:
        heterogeneity_signal_classification = "null_or_imprecise"
    return {
        "public_concept_name": candidate["public_concept_name"],
        "public_concept_label": candidate["public_concept_label"],
        "outcome_variable": candidate["outcome_variable"],
        "outcome_family": candidate["outcome_family"],
        "stage1_classification": candidate["stage1_classification"],
        "triage_priority_classification": candidate["triage_priority_classification"],
        "benchmark_stage_recommendation": "yes",
        "modifier_name": modifier_spec["modifier_name"],
        "modifier_label": modifier_spec["modifier_label"],
        "modifier_stratum": modifier_stratum,
        "exposure_variable": exposure_column,
        "exposure_label": _exposure_label(candidate, exposure_column),
        "lag_or_window": (
            _split_acute_exposure_column(exposure_column)[1]
            if candidate["outcome_family"] == "acute_daily"
            else _split_window_exposure_column(exposure_column)[1]
        ),
        "model_label": model_label,
        "estimate_scale": estimate_scale,
        "estimate": model_result.get("estimate", float("nan")),
        "lower_ci": model_result.get("lower_ci", float("nan")),
        "upper_ci": model_result.get("upper_ci", float("nan")),
        "p_value": model_result.get("p_value", float("nan")),
        "p_value_floor_applied_flag": model_result.get("p_value_floor_applied_flag", "no"),
        "valid_strata_count": model_result.get("valid_strata_count", 0),
        "observation_count": model_result.get("observation_count", 0),
        "outcome_sum": model_result.get("outcome_sum", 0.0),
        "denominator_sum": model_result.get("denominator_sum", 0.0),
        "overdispersion_ratio": model_result.get("overdispersion_ratio", float("nan")),
        "municipality_count": municipality_count,
        "full_sample_estimate": reference_estimate,
        "direction_vs_full_sample": direction_vs_full_sample,
        "heterogeneity_signal_classification": heterogeneity_signal_classification,
    }


def _render_methods_note() -> str:
    return (
        "# Track A Sensitivity, Heterogeneity, and Vulnerability Methods Note\n\n"
        "Stage 7 extends the Stage 6 benchmark package for benchmark-eligible concepts only.\n\n"
        "## Sensitivity Design\n\n"
        "- Source-aware restriction compares the full analytical sample with a subset restricted to rows without climate fallback imputation.\n"
        "- Climate-source sensitivity stratifies the analytical sample by the final municipal climate source label.\n"
        "- Alternative lag and window sensitivity re-estimates the selected exposure family across the full acute lag ladder or the full delivery-aligned 7-day, 14-day, and 28-day window ladder.\n"
        "- Exposure-family comparison re-estimates each concept at the selected lag or window across all Track A exposure families without combining them into a single model.\n\n"
        "## Heterogeneity Design\n\n"
        "- Heterogeneity is evaluated one modifier at a time using municipal context strata from the Stage 2 context lookup.\n"
        "- Urbanicity, inequality, human development, and altitude are operationalized as ranked tertiles. Coastal status remains binary.\n"
        "- Context variables are treated as effect modifiers and vulnerability descriptors, not as default confounder adjustments in the main fixed-effects proxy models.\n\n"
        "## Vulnerability Maps\n\n"
        "- Vulnerability outputs are centroid-based municipal point maps derived from the existing municipality centroid registry. They are not burden maps and they do not claim local causal effects.\n"
        "- The contextual vulnerability marker count summarizes high urbanicity, high inequality, low human development, and coastal location. Altitude is retained as a separate modifier and is not folded into the composite marker count.\n"
        "- The signal stability score summarizes the share of modifier-specific heterogeneity rows that remain directionally aligned with the full-sample benchmark estimate for the municipality's observed context strata.\n"
    )


def _render_stage_report(
    sensitivity_rows: pd.DataFrame,
    heterogeneity_rows: pd.DataFrame,
    figure_paths: list[Path],
) -> str:
    source_sensitive_concepts = sorted(
        sensitivity_rows.loc[
            sensitivity_rows["source_dependence_flag"].astype(str).eq("yes"),
            "public_concept_name",
        ].dropna().astype(str).unique().tolist()
    )
    inconsistent_concepts = sorted(
        heterogeneity_rows.loc[
            heterogeneity_rows["heterogeneity_signal_classification"].astype(str).eq("directionally_inconsistent"),
            "public_concept_name",
        ].dropna().astype(str).unique().tolist()
    )
    return (
        "# Track A Sensitivity Stage Report\n\n"
        f"- Sensitivity rows: `{len(sensitivity_rows)}`\n"
        f"- Heterogeneity rows: `{len(heterogeneity_rows)}`\n"
        f"- Vulnerability figures: `{len(figure_paths)}`\n"
        f"- Concepts with strong source dependence flags: `{len(source_sensitive_concepts)}`\n"
        f"- Concepts with at least one directionally inconsistent heterogeneity row: `{len(inconsistent_concepts)}`\n\n"
        "## Source Dependence Signals\n\n"
        + ("\n".join(f"- {concept}" for concept in source_sensitive_concepts) if source_sensitive_concepts else "- None")
        + "\n\n## Heterogeneity Signals\n\n"
        + ("\n".join(f"- {concept}" for concept in inconsistent_concepts) if inconsistent_concepts else "- None")
        + "\n"
    )


def _scatter_map(
    frame: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    output_path: Path,
    colorbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    valid = frame.loc[
        frame["municipality_centroid_lon"].notna()
        & frame["municipality_centroid_lat"].notna()
        & frame[value_column].notna()
    ].copy()
    figure = plt.figure(figsize=(8.5, 8.5))
    axis = figure.add_subplot(1, 1, 1)
    if valid.empty:
        axis.text(0.5, 0.5, "No mapped municipalities were available.", ha="center", va="center")
    else:
        scatter = axis.scatter(
            valid["municipality_centroid_lon"],
            valid["municipality_centroid_lat"],
            c=valid[value_column],
            cmap=cmap,
            s=10,
            alpha=0.85,
            linewidths=0,
            vmin=vmin,
            vmax=vmax,
        )
        colorbar = figure.colorbar(scatter, ax=axis, shrink=0.8)
        colorbar.set_label(colorbar_label)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title(title)
    axis.set_xlim(-74.5, -33.0)
    axis.set_ylim(-34.5, 6.5)
    axis.grid(alpha=0.15, linewidth=0.5)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def build_track_a_sensitivity(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    config = load_track_a_config(root, config_path=config_path)
    years = [int(year) for year in config["years"]]

    triage = _load_triage(root)
    benchmark_results = _load_benchmark_results(root)
    benchmark_concepts = set(benchmark_results["public_concept_name"].dropna().astype(str).unique().tolist())
    candidates = [candidate for candidate in _candidate_rows(triage) if candidate["public_concept_name"] in benchmark_concepts]
    if not candidates:
        raise ValueError("Stage 7 produced no benchmark-eligible candidates. Verify Stage 4 triage and Stage 6 benchmark outputs.")

    frame_lookup = _combine_candidate_frames(root, years, candidates)
    context = _context_with_strata(root)

    sensitivity_rows: list[dict[str, Any]] = []
    heterogeneity_rows: list[dict[str, Any]] = []
    baseline_lookup: dict[str, dict[str, Any]] = {}

    for candidate in candidates:
        frame = _candidate_frame(candidate, frame_lookup)
        selected_exposure = _selected_exposure_column(candidate)
        baseline_result, baseline_model_label, baseline_scale = _estimate_candidate_effect(
            frame,
            candidate,
            selected_exposure,
        )
        baseline_lookup[candidate["public_concept_name"]] = {
            "estimate": float(baseline_result.get("estimate", float("nan"))),
            "lower_ci": float(baseline_result.get("lower_ci", float("nan"))),
            "upper_ci": float(baseline_result.get("upper_ci", float("nan"))),
            "p_value": float(baseline_result.get("p_value", float("nan"))),
            "estimate_scale": baseline_scale,
            "model_label": baseline_model_label,
            "exposure_column": selected_exposure,
        }
        selected_lag_or_window = (
            _split_acute_exposure_column(selected_exposure)[1]
            if candidate["outcome_family"] == "acute_daily"
            else _split_window_exposure_column(selected_exposure)[1]
        )
        sensitivity_rows.append(
            _build_sensitivity_row(
                candidate=candidate,
                sensitivity_domain="benchmark_reference",
                sensitivity_label="Full analytical benchmark sample",
                exposure_column=selected_exposure,
                lag_or_window=selected_lag_or_window,
                sample_filter="quality_control_pass_and_valid_offset",
                final_climate_source="all_sources",
                model_label=baseline_model_label,
                estimate_scale=baseline_scale,
                model_result=baseline_result,
                reference_estimate=baseline_lookup[candidate["public_concept_name"]]["estimate"],
                reference_lower_ci=baseline_lookup[candidate["public_concept_name"]]["lower_ci"],
                reference_upper_ci=baseline_lookup[candidate["public_concept_name"]]["upper_ci"],
                reference_p_value=baseline_lookup[candidate["public_concept_name"]]["p_value"],
            )
        )

        restricted_frame = frame.loc[frame["any_climate_fallback_imputation_flag"].fillna(0).eq(0)].copy()
        restricted_result, restricted_model_label, restricted_scale = _estimate_candidate_effect(
            restricted_frame,
            candidate,
            selected_exposure,
        )
        sensitivity_rows.append(
            _build_sensitivity_row(
                candidate=candidate,
                sensitivity_domain="source_restriction_no_fallback",
                sensitivity_label="Restriction to rows without climate fallback imputation",
                exposure_column=selected_exposure,
                lag_or_window=selected_lag_or_window,
                sample_filter="quality_control_pass_and_valid_offset;any_climate_fallback_imputation_flag=0",
                final_climate_source="all_sources",
                model_label=restricted_model_label,
                estimate_scale=restricted_scale,
                model_result=restricted_result,
                reference_estimate=baseline_lookup[candidate["public_concept_name"]]["estimate"],
                reference_lower_ci=baseline_lookup[candidate["public_concept_name"]]["lower_ci"],
                reference_upper_ci=baseline_lookup[candidate["public_concept_name"]]["upper_ci"],
                reference_p_value=baseline_lookup[candidate["public_concept_name"]]["p_value"],
            )
        )

        for source_label in sorted(frame["final_climate_source"].dropna().astype(str).unique().tolist()):
            source_frame = frame.loc[frame["final_climate_source"].astype(str).eq(source_label)].copy()
            source_result, source_model_label, source_scale = _estimate_candidate_effect(
                source_frame,
                candidate,
                selected_exposure,
            )
            sensitivity_rows.append(
                _build_sensitivity_row(
                    candidate=candidate,
                    sensitivity_domain="climate_source_stratum",
                    sensitivity_label=f"Climate-source stratum: {source_label}",
                    exposure_column=selected_exposure,
                    lag_or_window=selected_lag_or_window,
                    sample_filter=f"quality_control_pass_and_valid_offset;final_climate_source={source_label}",
                    final_climate_source=source_label,
                    model_label=source_model_label,
                    estimate_scale=source_scale,
                    model_result=source_result,
                    reference_estimate=baseline_lookup[candidate["public_concept_name"]]["estimate"],
                    reference_lower_ci=baseline_lookup[candidate["public_concept_name"]]["lower_ci"],
                    reference_upper_ci=baseline_lookup[candidate["public_concept_name"]]["upper_ci"],
                    reference_p_value=baseline_lookup[candidate["public_concept_name"]]["p_value"],
                )
            )

        if candidate["outcome_family"] == "acute_daily":
            base_exposure, _ = _split_acute_exposure_column(selected_exposure)
            suffixes = ACUTE_TEMPERATURE_LAG_SUFFIXES
        else:
            base_exposure, _ = _split_window_exposure_column(selected_exposure)
            suffixes = DELIVERY_WINDOW_SUFFIXES
        for suffix in suffixes:
            alternative_exposure = f"{base_exposure}_{suffix}"
            alternative_result, alternative_model_label, alternative_scale = _estimate_candidate_effect(
                frame,
                candidate,
                alternative_exposure,
            )
            sensitivity_rows.append(
                _build_sensitivity_row(
                    candidate=candidate,
                    sensitivity_domain="alternative_lag_or_window",
                    sensitivity_label=f"Alternative lag or window: {suffix}",
                    exposure_column=alternative_exposure,
                    lag_or_window=suffix,
                    sample_filter="quality_control_pass_and_valid_offset",
                    final_climate_source="all_sources",
                    model_label=alternative_model_label,
                    estimate_scale=alternative_scale,
                    model_result=alternative_result,
                    reference_estimate=baseline_lookup[candidate["public_concept_name"]]["estimate"],
                    reference_lower_ci=baseline_lookup[candidate["public_concept_name"]]["lower_ci"],
                    reference_upper_ci=baseline_lookup[candidate["public_concept_name"]]["upper_ci"],
                    reference_p_value=baseline_lookup[candidate["public_concept_name"]]["p_value"],
                )
            )

        for base_name in BASE_EXPOSURES:
            comparison_exposure = f"{base_name}_{selected_lag_or_window}"
            comparison_result, comparison_model_label, comparison_scale = _estimate_candidate_effect(
                frame,
                candidate,
                comparison_exposure,
            )
            sensitivity_rows.append(
                _build_sensitivity_row(
                    candidate=candidate,
                    sensitivity_domain="exposure_family_comparison",
                    sensitivity_label=f"Exposure-family comparison: {BASE_EXPOSURES[base_name]['family_name']}",
                    exposure_column=comparison_exposure,
                    lag_or_window=selected_lag_or_window,
                    sample_filter="quality_control_pass_and_valid_offset",
                    final_climate_source="all_sources",
                    model_label=comparison_model_label,
                    estimate_scale=comparison_scale,
                    model_result=comparison_result,
                    reference_estimate=baseline_lookup[candidate["public_concept_name"]]["estimate"],
                    reference_lower_ci=baseline_lookup[candidate["public_concept_name"]]["lower_ci"],
                    reference_upper_ci=baseline_lookup[candidate["public_concept_name"]]["upper_ci"],
                    reference_p_value=baseline_lookup[candidate["public_concept_name"]]["p_value"],
                )
            )

        frame_with_context = frame.merge(
            context.loc[
                :,
                [
                    "municipality_merge_key",
                    "urbanicity_tertile",
                    "inequality_tertile",
                    "human_development_tertile",
                    "altitude_tertile",
                    "coastal_stratum",
                ],
            ],
            on="municipality_merge_key",
            how="left",
        )
        for modifier_spec in MODIFIER_SPECS:
            modifier_column = modifier_spec["stratum_column"]
            for modifier_stratum in (
                frame_with_context[modifier_column].dropna().astype(str).sort_values().unique().tolist()
            ):
                stratum_frame = frame_with_context.loc[
                    frame_with_context[modifier_column].astype(str).eq(modifier_stratum)
                ].copy()
                stratum_result, stratum_model_label, stratum_scale = _estimate_candidate_effect(
                    stratum_frame,
                    candidate,
                    selected_exposure,
                )
                heterogeneity_rows.append(
                    _build_heterogeneity_row(
                        candidate=candidate,
                        modifier_spec=modifier_spec,
                        modifier_stratum=modifier_stratum,
                        exposure_column=selected_exposure,
                        model_label=stratum_model_label,
                        estimate_scale=stratum_scale,
                        model_result=stratum_result,
                        municipality_count=int(stratum_frame["municipality_code"].nunique()),
                        reference_estimate=baseline_lookup[candidate["public_concept_name"]]["estimate"],
                    )
                )

    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    heterogeneity_frame = pd.DataFrame(heterogeneity_rows)
    sensitivity_frame = sensitivity_frame.loc[sensitivity_frame["valid_strata_count"].fillna(0).astype(int).gt(0)].copy()
    heterogeneity_frame = heterogeneity_frame.loc[
        heterogeneity_frame["valid_strata_count"].fillna(0).astype(int).gt(0)
    ].copy()
    if sensitivity_frame.empty or heterogeneity_frame.empty:
        raise ValueError("Stage 7 produced empty sensitivity or heterogeneity outputs. Verify Stage 2 marts and Stage 6 benchmark inputs.")

    municipality_modifier_long_rows: list[dict[str, Any]] = []
    for modifier_spec in MODIFIER_SPECS:
        modifier_column = modifier_spec["stratum_column"]
        subset = context.loc[
            :,
            [
                "municipality_code",
                "municipality_merge_key",
                "municipality_name",
                "uf",
                "municipality_centroid_lat",
                "municipality_centroid_lon",
                "contextual_vulnerability_marker_count",
                modifier_column,
            ],
        ].copy()
        subset["modifier_name"] = modifier_spec["modifier_name"]
        subset["modifier_label"] = modifier_spec["modifier_label"]
        subset = subset.rename(columns={modifier_column: "modifier_stratum"})
        municipality_modifier_long_rows.extend(subset.to_dict(orient="records"))
    municipality_modifier_long = pd.DataFrame(municipality_modifier_long_rows)
    municipality_signal = municipality_modifier_long.merge(
        heterogeneity_frame.loc[
            :,
            [
                "public_concept_name",
                "modifier_name",
                "modifier_stratum",
                "direction_vs_full_sample",
                "heterogeneity_signal_classification",
            ],
        ],
        on=["modifier_name", "modifier_stratum"],
        how="left",
    )
    municipality_signal["same_direction_flag"] = municipality_signal["direction_vs_full_sample"].map(
        {"same_direction": 1.0, "opposite_direction": 0.0}
    )
    municipality_vulnerability_profile = (
        municipality_signal.groupby(
            [
                "municipality_code",
                "municipality_merge_key",
                "municipality_name",
                "uf",
                "municipality_centroid_lat",
                "municipality_centroid_lon",
                "contextual_vulnerability_marker_count",
            ],
            as_index=False,
        )
        .agg(
            signal_stability_score=("same_direction_flag", "mean"),
            heterogeneity_rows_linked=("public_concept_name", "count"),
        )
    )
    municipality_vulnerability_profile["signal_stability_score"] = municipality_vulnerability_profile[
        "signal_stability_score"
    ].fillna(0.0)

    sensitivity_root = _sensitivity_root(root)
    vulnerability_root = sensitivity_root / "vulnerability_maps"
    sensitivity_root.mkdir(parents=True, exist_ok=True)
    vulnerability_root.mkdir(parents=True, exist_ok=True)

    sensitivity_matrix_path = sensitivity_root / "sensitivity_matrix.csv"
    heterogeneity_results_path = sensitivity_root / "heterogeneity_results.csv"
    methods_note_path = sensitivity_root / "sensitivity_methods_note.md"
    stage_report_path = sensitivity_root / "sensitivity_stage_report.md"
    vulnerability_profile_path = vulnerability_root / "municipality_vulnerability_profile.csv"
    vulnerability_bridge_path = vulnerability_root / "municipality_modifier_stability_bridge.csv"

    sensitivity_frame = sensitivity_frame.sort_values(
        ["public_concept_label", "sensitivity_domain", "exposure_family_name", "lag_or_window"],
        ascending=[True, True, True, True],
        na_position="last",
    )
    heterogeneity_frame = heterogeneity_frame.sort_values(
        ["public_concept_label", "modifier_label", "modifier_stratum"],
        ascending=[True, True, True],
        na_position="last",
    )

    sensitivity_frame.to_csv(sensitivity_matrix_path, index=False)
    heterogeneity_frame.to_csv(heterogeneity_results_path, index=False)
    methods_note_path.write_text(_render_methods_note(), encoding="utf-8")
    municipality_vulnerability_profile.to_csv(vulnerability_profile_path, index=False)
    municipality_signal.to_csv(vulnerability_bridge_path, index=False)

    figure_paths = [
        vulnerability_root / "figure_01_contextual_vulnerability_marker_count_map.png",
        vulnerability_root / "figure_02_signal_stability_score_map.png",
        vulnerability_root / "figure_03_urbanicity_tertile_map.png",
        vulnerability_root / "figure_04_human_development_tertile_map.png",
    ]
    _scatter_map(
        municipality_vulnerability_profile,
        value_column="contextual_vulnerability_marker_count",
        title="Contextual vulnerability marker count",
        output_path=figure_paths[0],
        colorbar_label="Marker count",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=4.0,
    )
    _scatter_map(
        municipality_vulnerability_profile,
        value_column="signal_stability_score",
        title="Directional signal stability score",
        output_path=figure_paths[1],
        colorbar_label="Stability score",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    _scatter_map(
        context,
        value_column="urban_population_proportion",
        title="Urban population proportion context map",
        output_path=figure_paths[2],
        colorbar_label="Urban population proportion",
        cmap="Blues",
    )
    _scatter_map(
        context,
        value_column="expanded_human_development_index",
        title="Expanded human development index context map",
        output_path=figure_paths[3],
        colorbar_label="Expanded human development index",
        cmap="PuBuGn",
    )

    stage_report_path.write_text(
        _render_stage_report(sensitivity_frame, heterogeneity_frame, figure_paths),
        encoding="utf-8",
    )

    source_sensitive_concepts = int(
        sensitivity_frame.loc[sensitivity_frame["source_dependence_flag"].astype(str).eq("yes"), "public_concept_name"]
        .dropna()
        .astype(str)
        .nunique()
    )
    heterogeneity_inconsistent_concepts = int(
        heterogeneity_frame.loc[
            heterogeneity_frame["heterogeneity_signal_classification"].astype(str).eq("directionally_inconsistent"),
            "public_concept_name",
        ]
        .dropna()
        .astype(str)
        .nunique()
    )

    return {
        "status": "PASS",
        "stage": "stage7_sensitivity_heterogeneity_and_vulnerability",
        "analysis_id": str(config["analysis_id"]),
        "analysis_title": str(config["analysis_title"]),
        "sensitivity_row_count": int(len(sensitivity_frame)),
        "heterogeneity_row_count": int(len(heterogeneity_frame)),
        "figure_count": len(figure_paths),
        "source_sensitive_concept_count": source_sensitive_concepts,
        "heterogeneity_inconsistent_concept_count": heterogeneity_inconsistent_concepts,
        "outputs": {
            "sensitivity_matrix": _relative(root, sensitivity_matrix_path),
            "heterogeneity_results": _relative(root, heterogeneity_results_path),
            "sensitivity_methods_note": _relative(root, methods_note_path),
            "sensitivity_stage_report": _relative(root, stage_report_path),
            "vulnerability_maps_directory": _relative(root, vulnerability_root),
            "municipality_vulnerability_profile": _relative(root, vulnerability_profile_path),
            "municipality_modifier_stability_bridge": _relative(root, vulnerability_bridge_path),
        },
    }
