from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from .config import load_round_config
from .registry import (
    analysis_shared_marts_root,
    analysis_step2_root,
    analysis_step4_root,
    analysis_step5_root,
    ensure_step5_directory_structure,
    incremental_publication_root,
)
from .step2 import (
    BASELINE_PROFILE_SPECS,
    OVERDISPERSION_THRESHOLD,
    _baseline_valid_rows,
    _dispersion_adjusted_result,
    _estimate_direction,
    _pooled_continuous_fe_proxy,
)
from .step3 import SCREENING_PROFILE_SPECS
from .step4 import build_step4_lag_refinement


STEP5_PUBLICATION_FILES = {
    "10_step5_heterogeneity_synthesis.md": "# Step 5 Heterogeneity Synthesis\n\nDocument modifier-specific heterogeneity interpretation here.\n",
    "11_step5_step6_gate.md": "# Step 5 to Step 6 Gate\n\nDocument which candidates can advance to absolute burden translation.\n",
}
STEP5_NAME = "step5_heterogeneity_synthesis"
STEP5_NEXT_STEP_ROUTE = "step6_absolute_burden_translation"
STEP5_HOLD_ROUTE = "step6_relative_only_hold"
STEP5_MATRIX_RELATIVE_ROOT = Path("output/spreadsheet")
GINI_FORMAL_EXCLUSION_STATUSES = {"MISSING_GINI|IDH"}
NON_BLOCKING_PROGRESS_MODIFIERS = {"mean_altitude_m", "coastal_municipality_flag"}
COUNT_ESTIMATE_SCALE = "rate_ratio_per_interquartile_range_increase"
SEVERITY_ESTIMATE_SCALE = "mean_difference_per_interquartile_range_increase"

MODIFIER_SPECS = {
    "urban_population_proportion": {
        "modifier_label": "Urban population proportion",
        "kind": "tertile",
        "stratum_column": "urban_population_proportion_stratum",
        "strata": [("low", 1), ("middle", 2), ("high", 3)],
    },
    "expanded_gini_index": {
        "modifier_label": "Expanded Gini index",
        "kind": "tertile",
        "stratum_column": "expanded_gini_index_stratum",
        "strata": [("low", 1), ("middle", 2), ("high", 3)],
    },
    "expanded_human_development_index": {
        "modifier_label": "Expanded human development index",
        "kind": "tertile",
        "stratum_column": "expanded_human_development_index_stratum",
        "strata": [("low", 1), ("middle", 2), ("high", 3)],
    },
    "mean_altitude_m": {
        "modifier_label": "Mean altitude",
        "kind": "tertile",
        "stratum_column": "mean_altitude_m_stratum",
        "strata": [("low", 1), ("middle", 2), ("high", 3)],
    },
    "coastal_municipality_flag": {
        "modifier_label": "Coastal municipality flag",
        "kind": "binary",
        "stratum_column": "coastal_municipality_flag_stratum",
        "strata": [("inland", 0), ("coastal", 1)],
    },
}
BLOCK_ORDER = {
    "Maternal": 0,
    "Perinatal and Birth": 1,
    "Child Mortality": 2,
    "Child Hospitalization": 3,
}
MATRIX_BLOCK_FILL = PatternFill(fill_type="solid", fgColor="1F4E78")
MATRIX_HEADER_FILL = PatternFill(fill_type="solid", fgColor="D9EAF7")
MATRIX_STABLE_FILL = PatternFill(fill_type="solid", fgColor="C6EFCE")
MATRIX_REVERSAL_FILL = PatternFill(fill_type="solid", fgColor="FCE5CD")
MATRIX_SIGNIFICANT_REVERSAL_FILL = PatternFill(fill_type="solid", fgColor="FFC7CE")
MATRIX_READY_FILL = PatternFill(fill_type="solid", fgColor="D9EAD3")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _effect_is_significant(
    estimate_scale: str,
    lower_ci: Any,
    upper_ci: Any,
    p_value: Any,
) -> bool:
    try:
        lower_value = float(lower_ci)
        upper_value = float(upper_ci)
        p_value_numeric = float(p_value)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(lower_value) and math.isfinite(upper_value) and math.isfinite(p_value_numeric)):
        return False
    if estimate_scale == SEVERITY_ESTIMATE_SCALE:
        ci_excludes_null = lower_value > 0 or upper_value < 0
    else:
        ci_excludes_null = lower_value > 1 or upper_value < 1
    return bool(ci_excludes_null and p_value_numeric < 0.05)


def _scaffold_step5_publication_workspace(root: Path) -> dict[str, object]:
    workspace_root = incremental_publication_root(root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    created_files: list[str] = []
    existing_files: list[str] = []
    for relative_path, contents in STEP5_PUBLICATION_FILES.items():
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


def _load_step4_inputs(root: Path) -> pd.DataFrame:
    return pd.read_csv(analysis_step4_root(root) / "step4_promotion_decisions.csv")


def _load_step2_preferred_models(root: Path) -> pd.DataFrame:
    return pd.read_csv(analysis_step2_root(root) / "preferred_baseline_models.csv")


def _expected_heterogeneity_rows(candidate_count: int) -> int:
    return candidate_count * sum(len(spec["strata"]) for spec in MODIFIER_SPECS.values())


def _tertile_labels(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    output = pd.Series(pd.NA, index=series.index, dtype="string")
    valid = numeric.dropna()
    if valid.nunique() < 3:
        return output
    ranked = numeric.rank(method="first")
    bins = pd.qcut(ranked, 3, labels=["low", "middle", "high"])
    output.loc[bins.index] = bins.astype("string")
    return output


def _context_join_key(value: Any) -> str:
    digits = "".join(character for character in str(value) if character.isdigit())
    if len(digits) >= 7:
        return digits[:6]
    return digits


def _build_context_lookup_frame(root: Path) -> pd.DataFrame:
    path = analysis_shared_marts_root(root) / "context_lookup.parquet"
    context = pd.read_parquet(path).copy()
    context["municipality_code"] = context["municipality_code"].astype(str)
    context["municipality_context_key"] = context["municipality_code"].map(_context_join_key)
    context["context_join_status"] = context["context_join_status"].astype(str)
    derived = context.loc[:, ["municipality_context_key", "context_join_status"]].copy()
    derived["gini_analytic_universe_included_flag"] = ~context["context_join_status"].isin(GINI_FORMAL_EXCLUSION_STATUSES)
    for modifier_name, spec in MODIFIER_SPECS.items():
        column_name = str(spec["stratum_column"])
        if spec["kind"] == "binary":
            numeric = pd.to_numeric(context[modifier_name], errors="coerce")
            derived[column_name] = pd.Series(pd.NA, index=context.index, dtype="string")
            derived.loc[numeric.eq(1), column_name] = "coastal"
            derived.loc[numeric.eq(0), column_name] = "inland"
        else:
            derived[column_name] = _tertile_labels(context[modifier_name])
    return derived.drop_duplicates(subset=["municipality_context_key"]).reset_index(drop=True)


def _build_gini_exclusion_frame(root: Path) -> pd.DataFrame:
    columns = [
        "municipality_code",
        "context_join_status",
        "exclusion_scope",
        "exclusion_reason",
    ]
    path = analysis_shared_marts_root(root) / "context_lookup.parquet"
    context = pd.read_parquet(path).copy()
    context["municipality_code"] = context["municipality_code"].astype(str)
    context["context_join_status"] = context["context_join_status"].astype(str)
    excluded = context.loc[
        context["context_join_status"].isin(GINI_FORMAL_EXCLUSION_STATUSES),
        ["municipality_code", "context_join_status"],
    ].drop_duplicates(subset=["municipality_code"], keep="first")
    if excluded.empty:
        return pd.DataFrame(columns=columns)
    excluded["exclusion_scope"] = "expanded_gini_index_only"
    excluded["exclusion_reason"] = "formal_exclusion_from_gini_analytic_universe"
    return excluded.loc[:, columns].sort_values("municipality_code", ignore_index=True)


def _selected_exposure_spec(
    seasonality_profile_id: str,
    exposure_family_id: str,
    lag_or_window: str,
) -> tuple[str, str]:
    profile = SCREENING_PROFILE_SPECS.get(seasonality_profile_id, {})
    exposures = profile.get("family_exposures", {}).get(exposure_family_id, [])
    for exposure_variable, exposure_label, code in exposures:
        if str(code) == str(lag_or_window):
            return str(exposure_variable), str(exposure_label)
    return "", ""


def _build_candidate_selection(step4_decisions: pd.DataFrame, preferred_models: pd.DataFrame) -> pd.DataFrame:
    ready = step4_decisions.loc[
        step4_decisions["candidate_ready_for_step5_flag"].eq(True)
    ].copy()
    if ready.empty or preferred_models.empty:
        return pd.DataFrame()

    preferred_subset = preferred_models.loc[
        :,
        [
            "analysis_unit_id",
            "linked_outcome_ids",
            "candidate_track",
            "analysis_role",
            "outcome_family",
            "mart_name",
            "exposure_family_id",
            "exposure_family_name",
            "denominator_column",
            "denominator_id",
            "estimate_scale",
            "seasonality_profile_id",
            "seasonality_controls_applied",
        ],
    ].drop_duplicates(
        subset=["analysis_unit_id", "candidate_track", "exposure_family_id"],
        keep="first",
    )

    selected = ready.merge(
        preferred_subset,
        left_on=["analysis_unit_id", "candidate_track", "selected_exposure_family_id"],
        right_on=["analysis_unit_id", "candidate_track", "exposure_family_id"],
        how="left",
        suffixes=("", "_step2"),
    )
    selected["selected_exposure_variable"] = ""
    selected["selected_exposure_label"] = ""
    for index, row in selected.iterrows():
        exposure_variable, exposure_label = _selected_exposure_spec(
            seasonality_profile_id=str(row.get("seasonality_profile_id") or ""),
            exposure_family_id=str(row.get("selected_exposure_family_id") or ""),
            lag_or_window=str(row.get("preferred_lag_or_window") or ""),
        )
        selected.at[index, "selected_exposure_variable"] = exposure_variable
        selected.at[index, "selected_exposure_label"] = exposure_label
    return selected.sort_values(
        ["candidate_track", "analysis_role", "analysis_unit_id"],
        ignore_index=True,
    )


def _required_columns_by_mart(candidate_selection: pd.DataFrame) -> dict[str, set[str]]:
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
    for mart_name, group in candidate_selection.groupby("mart_name", sort=True):
        required[str(mart_name)] = set(base_columns)
        for record in group.to_dict(orient="records"):
            required[str(mart_name)].add(str(record["analysis_unit_id"]))
            required[str(mart_name)].add(str(record["selected_exposure_variable"]))
            required[str(mart_name)].add(str(record["denominator_column"]))
    return required


def _load_model_frames(
    root: Path,
    years: list[int],
    required_columns_by_mart: dict[str, set[str]],
    context_lookup: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for mart_name, required_columns in required_columns_by_mart.items():
        partitions: list[pd.DataFrame] = []
        selected_columns = sorted(required_columns)
        for year in years:
            path = analysis_shared_marts_root(root) / mart_name / f"YEAR={year}" / f"{mart_name}_{year}.parquet"
            partition = pd.read_parquet(path, columns=selected_columns)
            partition["observation_date"] = pd.to_datetime(partition["observation_date"], errors="coerce")
            partition["municipality_code"] = partition["municipality_code"].astype(str)
            partition["municipality_context_key"] = partition["municipality_code"].map(_context_join_key)
            partition["calendar_month"] = pd.to_numeric(partition["calendar_month"], errors="coerce").astype("Int64")
            partition["day_of_week"] = partition["observation_date"].dt.dayofweek.astype("Int64")
            partitions.append(partition)
        frame = pd.concat(partitions, ignore_index=True)
        frames[mart_name] = frame.merge(context_lookup, on="municipality_context_key", how="left")
    return frames


def _finalize_effect_result(result: dict[str, Any], model_kind: str) -> tuple[dict[str, Any], str]:
    overdispersion_ratio = float(result.get("overdispersion_ratio", float("nan")))
    if model_kind == "count" and math.isfinite(overdispersion_ratio) and overdispersion_ratio > OVERDISPERSION_THRESHOLD:
        return _dispersion_adjusted_result(result, model_kind), "negative_binomial_fe_proxy"
    return result, ("gaussian_fe_proxy" if model_kind == "severity" else "poisson_fe_proxy")


def _compute_effect(
    valid: pd.DataFrame,
    exposure_column: str,
    outcome_column: str,
    denominator_column: str,
    stratum_columns: list[str],
    model_kind: str,
) -> tuple[dict[str, Any], str]:
    base_result = _pooled_continuous_fe_proxy(
        frame=valid,
        exposure_column=exposure_column,
        outcome_column=outcome_column,
        denominator_column=denominator_column,
        stratum_columns=stratum_columns,
        model_kind=model_kind,
    )
    return _finalize_effect_result(base_result, model_kind)


def _relative_change_pct(estimate: Any, reference_estimate: Any) -> float:
    estimate_numeric = float(estimate) if estimate is not None and not pd.isna(estimate) else float("nan")
    reference_numeric = float(reference_estimate) if reference_estimate is not None and not pd.isna(reference_estimate) else float("nan")
    if not math.isfinite(estimate_numeric) or not math.isfinite(reference_numeric) or abs(reference_numeric) < 1e-12:
        return float("nan")
    return float(((estimate_numeric - reference_numeric) / abs(reference_numeric)) * 100.0)


def _direction_vs_reference(direction: str, reference_direction: str) -> str:
    if not direction or not reference_direction:
        return "not_estimated"
    if direction == reference_direction:
        return "same_direction"
    if direction == "null" or reference_direction == "null":
        return "includes_null_direction"
    return "opposite_direction"


def _heterogeneity_signal_classification(direction_vs_reference: str) -> str:
    if direction_vs_reference == "opposite_direction":
        return "directionally_inconsistent"
    if direction_vs_reference == "not_estimated":
        return "not_estimated"
    return "directionally_consistent"


def _build_heterogeneity_results(
    candidate_selection: pd.DataFrame,
    frames_by_mart: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    missing_context_rows = 0
    gini_suppressed_rows = 0

    for record in candidate_selection.to_dict(orient="records"):
        mart_name = str(record["mart_name"])
        frame = frames_by_mart[mart_name]
        exposure_column = str(record["selected_exposure_variable"])
        outcome_column = str(record["analysis_unit_id"])
        denominator_column = str(record["denominator_column"])
        model_kind = "severity" if str(record["estimate_scale"]) == "mean_difference_per_interquartile_range_increase" else "count"
        stratum_columns = list(BASELINE_PROFILE_SPECS[str(record["seasonality_profile_id"])]["main_strata"])

        candidate_valid = _baseline_valid_rows(frame, denominator_column, exposure_column)
        missing_context_rows += int(candidate_valid["context_join_status"].isna().sum())
        full_sample_result, full_sample_model_label = _compute_effect(
            valid=candidate_valid,
            exposure_column=exposure_column,
            outcome_column=outcome_column,
            denominator_column=denominator_column,
            stratum_columns=stratum_columns,
            model_kind=model_kind,
        )
        full_sample_estimate = float(full_sample_result["estimate"])
        full_sample_direction = _estimate_direction(full_sample_estimate, str(record["estimate_scale"]))
        full_sample_significant_flag = _effect_is_significant(
            estimate_scale=str(record["estimate_scale"]),
            lower_ci=full_sample_result["lower_ci"],
            upper_ci=full_sample_result["upper_ci"],
            p_value=full_sample_result["p_value"],
        )

        for modifier_name, spec in MODIFIER_SPECS.items():
            modifier_label = str(spec["modifier_label"])
            modifier_column = str(spec["stratum_column"])
            modifier_source = candidate_valid.copy()
            if modifier_name == "expanded_gini_index":
                excluded_for_gini = ~modifier_source["gini_analytic_universe_included_flag"].fillna(True).astype(bool)
                gini_not_stratifiable = modifier_source[modifier_column].isna()
                gini_suppressed_rows += int((excluded_for_gini | gini_not_stratifiable).sum())
                modifier_source = modifier_source.loc[(~excluded_for_gini) & modifier_source[modifier_column].notna()].copy()
            for modifier_stratum, modifier_stratum_order in spec["strata"]:
                modifier_valid = modifier_source.loc[
                    modifier_source[modifier_column].astype("string").eq(modifier_stratum)
                ].copy()
                modifier_result, modifier_model_label = _compute_effect(
                    valid=modifier_valid,
                    exposure_column=exposure_column,
                    outcome_column=outcome_column,
                    denominator_column=denominator_column,
                    stratum_columns=stratum_columns,
                    model_kind=model_kind,
                )
                estimate = float(modifier_result["estimate"])
                direction = _estimate_direction(estimate, str(record["estimate_scale"]))
                direction_vs_full_sample = _direction_vs_reference(direction, full_sample_direction)
                stratum_significant_flag = _effect_is_significant(
                    estimate_scale=str(record["estimate_scale"]),
                    lower_ci=modifier_result["lower_ci"],
                    upper_ci=modifier_result["upper_ci"],
                    p_value=modifier_result["p_value"],
                )
                rows.append(
                    {
                        "analysis_unit_id": str(record["analysis_unit_id"]),
                        "linked_outcome_ids": str(record.get("linked_outcome_ids") or ""),
                        "candidate_track": str(record["candidate_track"]),
                        "analysis_role": str(record["analysis_role"]),
                        "outcome_family": str(record["outcome_family"]),
                        "mart_name": mart_name,
                        "selected_exposure_family_id": str(record["selected_exposure_family_id"]),
                        "selected_exposure_family_name": str(record["selected_exposure_family_name"]),
                        "selected_exposure_variable": exposure_column,
                        "selected_exposure_label": str(record["selected_exposure_label"]),
                        "preferred_lag_or_window": str(record["preferred_lag_or_window"]),
                        "seasonality_profile_id": str(record["seasonality_profile_id"]),
                        "seasonality_controls_applied": str(record["seasonality_controls_applied"]),
                        "modifier_name": modifier_name,
                        "modifier_label": modifier_label,
                        "modifier_kind": str(spec["kind"]),
                        "modifier_stratum": modifier_stratum,
                        "modifier_stratum_order": modifier_stratum_order,
                        "model_label": modifier_model_label,
                        "estimate_scale": str(record["estimate_scale"]),
                        "estimate": estimate,
                        "lower_ci": float(modifier_result["lower_ci"]),
                        "upper_ci": float(modifier_result["upper_ci"]),
                        "p_value": float(modifier_result["p_value"]),
                        "p_value_floor_applied_flag": str(modifier_result["p_value_floor_applied_flag"]),
                        "valid_strata_count": int(modifier_result["valid_strata_count"]),
                        "observation_count": int(modifier_result["observation_count"]),
                        "outcome_sum": float(modifier_result["outcome_sum"]),
                        "denominator_sum": float(modifier_result["denominator_sum"]),
                        "overdispersion_ratio": float(modifier_result["overdispersion_ratio"]),
                        "municipality_count": int(modifier_valid["municipality_code"].nunique()),
                        "full_sample_model_label": full_sample_model_label,
                        "full_sample_estimate": full_sample_estimate,
                        "full_sample_lower_ci": float(full_sample_result["lower_ci"]),
                        "full_sample_upper_ci": float(full_sample_result["upper_ci"]),
                        "full_sample_p_value": float(full_sample_result["p_value"]),
                        "full_sample_valid_strata_count": int(full_sample_result["valid_strata_count"]),
                        "full_sample_direction": full_sample_direction,
                        "full_sample_significant_flag": full_sample_significant_flag,
                        "direction_vs_full_sample": direction_vs_full_sample,
                        "stratum_significant_flag": stratum_significant_flag,
                        "opposite_direction_significant_flag": bool(
                            direction_vs_full_sample == "opposite_direction" and stratum_significant_flag
                        ),
                        "relative_change_pct": _relative_change_pct(estimate, full_sample_estimate),
                        "heterogeneity_signal_classification": _heterogeneity_signal_classification(direction_vs_full_sample),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        [
            "candidate_track",
            "analysis_role",
            "analysis_unit_id",
            "modifier_name",
            "modifier_stratum_order",
        ],
        ignore_index=True,
    ), {
        "missing_context_rows": missing_context_rows,
        "gini_suppressed_rows": gini_suppressed_rows,
    }


def _ensure_heterogeneity_annotation_columns(heterogeneity_results: pd.DataFrame) -> pd.DataFrame:
    if heterogeneity_results.empty:
        return heterogeneity_results.copy()
    enriched = heterogeneity_results.copy()
    if "full_sample_significant_flag" not in enriched.columns:
        enriched["full_sample_significant_flag"] = enriched.apply(
            lambda row: _effect_is_significant(
                estimate_scale=str(row.get("estimate_scale") or ""),
                lower_ci=row.get("full_sample_lower_ci"),
                upper_ci=row.get("full_sample_upper_ci"),
                p_value=row.get("full_sample_p_value"),
            ),
            axis=1,
        )
    if "stratum_significant_flag" not in enriched.columns:
        enriched["stratum_significant_flag"] = enriched.apply(
            lambda row: _effect_is_significant(
                estimate_scale=str(row.get("estimate_scale") or ""),
                lower_ci=row.get("lower_ci"),
                upper_ci=row.get("upper_ci"),
                p_value=row.get("p_value"),
            ),
            axis=1,
        )
    if "opposite_direction_significant_flag" not in enriched.columns:
        enriched["opposite_direction_significant_flag"] = (
            enriched["direction_vs_full_sample"].astype(str).eq("opposite_direction")
            & enriched["stratum_significant_flag"].fillna(False).astype(bool)
        )
    return enriched


def _display_label(analysis_unit_id: str) -> str:
    text = str(analysis_unit_id)
    for suffix in ("_count", "_flag", "_total"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text.replace("_", " ").strip().title()


def _clean_exposure_family_label(value: str) -> str:
    text = str(value).strip().lower()
    if "heat index" in text:
        return "Heat index"
    if "tmax" in text or "maximum daily temperature" in text or "temperature" in text:
        return "Tmax"
    return str(value).strip()


def _clean_window_label(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("lag"):
        suffix = text[3:].replace("_", "-")
        return f"lag {suffix}"
    if "_" in text and all(part.isdigit() for part in text.split("_")):
        return f"{text.replace('_', '-')}d"
    return text.replace("_", "-")


def _clean_decision_reason(value: str) -> str:
    mapping = {
        "promoted_to_step6_consistent": "Promoted to Step 6 without directional reversal",
        "promoted_to_step6_with_reversal": "Promoted to Step 6 with directional reversal",
        "promoted_to_step6_with_significant_reversal": "Promoted to Step 6 with significant directional reversal",
        "selected_window_not_estimable_in_step5_full_sample": "Selected window not estimable in the full sample",
        "insufficient_context_support": "Insufficient contextual support",
    }
    return mapping.get(str(value), str(value).replace("_", " ").strip().title())


def _matrix_block_name(outcome_family: str, analysis_unit_id: str) -> str:
    analysis_unit_id = str(analysis_unit_id)
    if str(outcome_family) == "maternal_operational":
        return "Maternal"
    if "admission" in analysis_unit_id or "hospitalization" in analysis_unit_id:
        return "Child Hospitalization"
    if "death" in analysis_unit_id or "stillbirth" in analysis_unit_id:
        return "Child Mortality"
    return "Perinatal and Birth"


def _build_matrix_frame(
    scorecard: pd.DataFrame,
    effect_modification_matrix: pd.DataFrame,
) -> pd.DataFrame:
    if scorecard.empty:
        return pd.DataFrame()

    pivot = effect_modification_matrix.pivot_table(
        index=["analysis_unit_id", "candidate_track"],
        columns="modifier_name",
        values="matrix_flag",
        aggfunc="first",
    ).reset_index()
    matrix = scorecard.merge(pivot, on=["analysis_unit_id", "candidate_track"], how="left")
    matrix["block_name"] = matrix.apply(
        lambda row: _matrix_block_name(str(row["outcome_family"]), str(row["analysis_unit_id"])),
        axis=1,
    )
    matrix["block_order"] = matrix["block_name"].map(BLOCK_ORDER).fillna(99).astype(int)
    matrix["display_label"] = matrix["analysis_unit_id"].map(_display_label)
    matrix["exposure_display"] = matrix["selected_exposure_family_name"].map(_clean_exposure_family_label)
    matrix["window_display"] = matrix["preferred_lag_or_window"].map(_clean_window_label)
    matrix["urbanization_status"] = matrix["urban_population_proportion"].fillna("consistent")
    matrix["gini_status"] = matrix["expanded_gini_index"].fillna("consistent")
    matrix["hdi_status"] = matrix["expanded_human_development_index"].fillna("consistent")
    matrix["altitude_status"] = matrix["mean_altitude_m"].fillna("consistent")
    matrix["coastal_status"] = matrix["coastal_municipality_flag"].fillna("consistent")
    matrix["step6_status"] = "promoted"
    matrix["reversal_count"] = matrix["inconsistent_modifier_count"].fillna(0).astype(int)
    matrix["significant_reversal_count"] = matrix["significant_reversal_modifier_count"].fillna(0).astype(int)
    matrix["decision_note"] = matrix["decision_reason"].map(_clean_decision_reason)
    return matrix.loc[
        :,
        [
            "block_name",
            "block_order",
            "display_label",
            "analysis_unit_id",
            "candidate_track",
            "exposure_display",
            "window_display",
            "urbanization_status",
            "gini_status",
            "hdi_status",
            "altitude_status",
            "coastal_status",
            "reversal_count",
            "significant_reversal_count",
            "step6_status",
            "decision_note",
        ],
    ].sort_values(
        ["block_order", "display_label"],
        ascending=[True, True],
        ignore_index=True,
    )


def _style_matrix_status(cell: Any, value: str) -> None:
    if value == "consistent":
        cell.fill = MATRIX_STABLE_FILL
    elif value == "reversal":
        cell.fill = MATRIX_REVERSAL_FILL
    else:
        cell.fill = MATRIX_SIGNIFICANT_REVERSAL_FILL


def _empty_reversal_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "analysis_unit_id",
            "selected_exposure_family",
            "preferred_window",
            "full_sample_significant",
            "reversal_modifiers",
            "has_significant_reversal",
            "significant_reversal_classifiers",
            "all_reversal_classifiers",
            "reversal_row_count",
        ]
    )


def _empty_reversal_detail_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "analysis_unit_id",
            "selected_exposure_family_name",
            "preferred_window",
            "modifier_label",
            "classifier_label",
            "modifier_classifier",
            "estimate_scale",
            "estimate",
            "lower_ci",
            "upper_ci",
            "p_value",
            "reversal_significance",
            "full_sample_estimate",
            "full_sample_lower_ci",
            "full_sample_upper_ci",
            "full_sample_p_value",
            "full_sample_significance",
        ]
    )


def _build_directional_reversal_diagnostics(
    heterogeneity_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    heterogeneity_results = _ensure_heterogeneity_annotation_columns(heterogeneity_results)
    if heterogeneity_results.empty:
        return _empty_reversal_summary_frame(), _empty_reversal_detail_frame()

    stratum_label_map = {
        "low": "low",
        "middle": "middle",
        "high": "high",
        "coastal": "coastal",
        "inland": "inland",
    }
    modifier_label_map = {
        "urban_population_proportion": "Urbanization",
        "expanded_gini_index": "Gini",
        "expanded_human_development_index": "HDI",
        "mean_altitude_m": "Altitude",
        "coastal_municipality_flag": "Coastal status",
    }

    subset = heterogeneity_results.copy()
    subset["modifier_label"] = subset["modifier_name"].map(modifier_label_map).fillna(subset["modifier_label"])
    subset["classifier_label"] = subset["modifier_stratum"].map(stratum_label_map).fillna(subset["modifier_stratum"])
    subset["selected_exposure_family_name"] = subset["selected_exposure_family_name"].replace(
        {
            "Approximate heat index": "Heat index",
            "Maximum daily temperature": "Tmax",
        }
    )
    subset["preferred_window"] = (
        subset["preferred_lag_or_window"]
        .astype(str)
        .str.replace("_", "-", regex=False)
        .str.replace("lag", "lag ", regex=False)
    )
    subset["modifier_classifier"] = subset["modifier_label"] + ": " + subset["classifier_label"].astype(str)

    reverse_rows = subset.loc[
        subset["direction_vs_full_sample"].astype(str).eq("opposite_direction")
    ].copy()
    if reverse_rows.empty:
        return _empty_reversal_summary_frame(), _empty_reversal_detail_frame()

    reverse_rows["reversal_significance"] = reverse_rows["opposite_direction_significant_flag"].map(
        {True: "yes", False: "no"}
    )
    reverse_rows["full_sample_significance"] = reverse_rows["full_sample_significant_flag"].map(
        {True: "yes", False: "no"}
    )
    detail = reverse_rows.loc[
        :,
        [
            "analysis_unit_id",
            "selected_exposure_family_name",
            "preferred_window",
            "modifier_label",
            "classifier_label",
            "modifier_classifier",
            "estimate_scale",
            "estimate",
            "lower_ci",
            "upper_ci",
            "p_value",
            "reversal_significance",
            "full_sample_estimate",
            "full_sample_lower_ci",
            "full_sample_upper_ci",
            "full_sample_p_value",
            "full_sample_significance",
        ],
    ].sort_values(["analysis_unit_id", "modifier_label", "classifier_label"], ignore_index=True)

    summary_rows: list[dict[str, Any]] = []
    for analysis_unit_id, group in subset.groupby("analysis_unit_id", sort=True):
        reverse_group = group.loc[group["direction_vs_full_sample"].astype(str).eq("opposite_direction")].copy()
        if reverse_group.empty:
            continue
        significant_group = reverse_group.loc[reverse_group["opposite_direction_significant_flag"].fillna(False).astype(bool)]
        summary_rows.append(
            {
                "analysis_unit_id": analysis_unit_id,
                "selected_exposure_family": str(group["selected_exposure_family_name"].iloc[0]),
                "preferred_window": str(group["preferred_window"].iloc[0]),
                "full_sample_significant": "yes" if bool(group["full_sample_significant_flag"].iloc[0]) else "no",
                "reversal_modifiers": ", ".join(reverse_group["modifier_label"].drop_duplicates().tolist()),
                "has_significant_reversal": "yes" if not significant_group.empty else "no",
                "significant_reversal_classifiers": "; ".join(significant_group["modifier_classifier"].tolist()),
                "all_reversal_classifiers": "; ".join(reverse_group["modifier_classifier"].tolist()),
                "reversal_row_count": int(len(reverse_group)),
            }
        )
    return (
        pd.DataFrame(summary_rows).sort_values("analysis_unit_id", ignore_index=True),
        detail,
    )


def _write_matrix_artifacts(root: Path, matrix_frame: pd.DataFrame) -> tuple[str, str]:
    output_root = root / STEP5_MATRIX_RELATIVE_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "step5_block_matrix.csv"
    xlsx_path = output_root / "step5_block_matrix.xlsx"
    if matrix_frame.empty:
        matrix_frame = pd.DataFrame(
            columns=[
                "block_name",
                "display_label",
                "exposure_display",
                "window_display",
                "urbanization_status",
                "gini_status",
                "hdi_status",
                "altitude_status",
                "coastal_status",
                "reversal_count",
                "significant_reversal_count",
                "step6_status",
                "decision_note",
                "analysis_unit_id",
                "candidate_track",
            ]
        )
    view_frame = matrix_frame.loc[
        :,
        [
            "block_name",
            "display_label",
            "exposure_display",
            "window_display",
            "urbanization_status",
            "gini_status",
            "hdi_status",
            "altitude_status",
            "coastal_status",
            "step6_status",
        ],
    ].rename(
        columns={
            "block_name": "Block",
            "display_label": "Outcome",
            "exposure_display": "Exposure",
            "window_display": "Window",
            "urbanization_status": "Urbanization",
            "gini_status": "Gini",
            "hdi_status": "HDI",
            "altitude_status": "Altitude",
            "coastal_status": "Coastal status",
            "step6_status": "Step 6",
        }
    )
    view_frame.to_csv(csv_path, index=False)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Matrix"
    sheet.sheet_view.showGridLines = False
    headers = view_frame.columns.tolist()
    for column_index, header in enumerate(headers, start=1):
        cell = sheet.cell(row=1, column=column_index, value=header)
        cell.font = Font(bold=True)
        cell.fill = MATRIX_HEADER_FILL
        cell.alignment = Alignment(horizontal="center")
    for row_index, record in enumerate(view_frame.to_dict(orient="records"), start=2):
        for column_index, header in enumerate(headers, start=1):
            value = record[header]
            cell = sheet.cell(row=row_index, column=column_index, value=value)
            cell.alignment = Alignment(horizontal="center" if column_index > 2 else "left", vertical="center")
            if header in {"Urbanization", "Gini", "HDI", "Altitude", "Coastal status"}:
                _style_matrix_status(cell, str(value))
            if header == "Step 6":
                cell.fill = MATRIX_READY_FILL

    widths = {
        "A": 18,
        "B": 14,
        "C": 12,
        "D": 34,
        "E": 12,
        "F": 12,
        "G": 12,
        "H": 12,
        "I": 12,
        "J": 16,
    }
    for column_label, width in widths.items():
        sheet.column_dimensions[column_label].width = width
    sheet.freeze_panes = "A2"

    workbook.create_sheet("Support")
    support_sheet = workbook["Support"]
    support_sheet.sheet_view.showGridLines = False
    support_headers = [
        "Block",
        "Outcome",
        "Exposure",
        "Window",
        "Reversal count",
        "Significant reversal count",
        "Step 6",
        "Decision note",
        "Analysis unit",
        "Track",
    ]
    for column_index, header in enumerate(support_headers, start=1):
        support_sheet.cell(row=1, column=column_index, value=header).font = Font(bold=True)
        support_sheet.cell(row=1, column=column_index).fill = MATRIX_HEADER_FILL
    for row_index, record in enumerate(matrix_frame.to_dict(orient="records"), start=2):
        row_values = [
            record["block_name"],
            record["display_label"],
            record["exposure_display"],
            record["window_display"],
            int(record["reversal_count"]),
            int(record["significant_reversal_count"]),
            record["step6_status"],
            record["decision_note"],
            record["analysis_unit_id"],
            record["candidate_track"],
        ]
        for column_index, value in enumerate(row_values, start=1):
            support_sheet.cell(row=row_index, column=column_index, value=value)
    support_sheet.freeze_panes = "A2"

    workbook.create_sheet("RawMatrix")
    raw_sheet = workbook["RawMatrix"]
    raw_sheet.sheet_state = "hidden"
    raw_headers = matrix_frame.columns.tolist()
    for column_index, header in enumerate(raw_headers, start=1):
        raw_sheet.cell(row=1, column=column_index, value=header).font = Font(bold=True)
    for row_index, row in enumerate(matrix_frame.to_dict(orient="records"), start=2):
        for column_index, header in enumerate(raw_headers, start=1):
            raw_sheet.cell(row=row_index, column=column_index, value=row[header])
    workbook.save(xlsx_path)
    return _relative(root, csv_path), _relative(root, xlsx_path)


def _write_directional_reversal_artifacts(
    root: Path,
    summary_frame: pd.DataFrame,
    detail_frame: pd.DataFrame,
) -> tuple[str, str, str]:
    output_root = analysis_step5_root(root)
    summary_path = output_root / "step5_directional_reversal_summary.csv"
    detail_csv_path = output_root / "step5_directional_reversal_diagnostic.csv"
    detail_xlsx_path = output_root / "step5_directional_reversal_diagnostic.xlsx"
    summary_frame.to_csv(summary_path, index=False)
    detail_frame.to_csv(detail_csv_path, index=False)

    workbook = Workbook()
    summary_sheet = workbook.active
    summary_sheet.title = "Summary"
    summary_sheet.sheet_view.showGridLines = False
    for column_index, header in enumerate(summary_frame.columns.tolist(), start=1):
        cell = summary_sheet.cell(row=1, column=column_index, value=header)
        cell.font = Font(bold=True)
        cell.fill = MATRIX_HEADER_FILL
    for row_index, record in enumerate(summary_frame.to_dict(orient="records"), start=2):
        for column_index, header in enumerate(summary_frame.columns.tolist(), start=1):
            summary_sheet.cell(row=row_index, column=column_index, value=record.get(header))
    summary_sheet.freeze_panes = "A2"

    detail_sheet = workbook.create_sheet("Detail")
    detail_sheet.sheet_view.showGridLines = False
    for column_index, header in enumerate(detail_frame.columns.tolist(), start=1):
        cell = detail_sheet.cell(row=1, column=column_index, value=header)
        cell.font = Font(bold=True)
        cell.fill = MATRIX_HEADER_FILL
    for row_index, record in enumerate(detail_frame.to_dict(orient="records"), start=2):
        for column_index, header in enumerate(detail_frame.columns.tolist(), start=1):
            detail_sheet.cell(row=row_index, column=column_index, value=record.get(header))
    detail_sheet.freeze_panes = "A2"
    workbook.save(detail_xlsx_path)
    return (
        _relative(root, summary_path),
        _relative(root, detail_csv_path),
        _relative(root, detail_xlsx_path),
    )


def _build_effect_modification_matrix(heterogeneity_results: pd.DataFrame) -> pd.DataFrame:
    heterogeneity_results = _ensure_heterogeneity_annotation_columns(heterogeneity_results)
    if heterogeneity_results.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_columns = [
        "analysis_unit_id",
        "candidate_track",
        "analysis_role",
        "outcome_family",
        "selected_exposure_family_id",
        "selected_exposure_family_name",
        "preferred_lag_or_window",
        "modifier_name",
        "modifier_label",
        "modifier_kind",
    ]
    for keys, group in heterogeneity_results.groupby(group_columns, sort=True):
        estimated_rows = int(group["estimate"].apply(lambda value: math.isfinite(float(value))).sum())
        opposite_direction_rows = int(group["direction_vs_full_sample"].astype(str).eq("opposite_direction").sum())
        opposite_direction_significant_rows = int(
            group["opposite_direction_significant_flag"].fillna(False).astype(bool).sum()
        )
        consistent_rows = int(group["heterogeneity_signal_classification"].astype(str).eq("directionally_consistent").sum())
        if estimated_rows <= 0:
            summary_classification = "not_estimable"
        elif opposite_direction_rows > 0:
            summary_classification = "directional_reversal_present"
        else:
            summary_classification = "stable_or_gradient"
        if opposite_direction_significant_rows > 0:
            matrix_flag = "significant_reversal"
        elif opposite_direction_rows > 0:
            matrix_flag = "reversal"
        else:
            matrix_flag = "consistent"
        rows.append(
            {
                "analysis_unit_id": str(keys[0]),
                "candidate_track": str(keys[1]),
                "analysis_role": str(keys[2]),
                "outcome_family": str(keys[3]),
                "selected_exposure_family_id": str(keys[4]),
                "selected_exposure_family_name": str(keys[5]),
                "preferred_lag_or_window": str(keys[6]),
                "modifier_name": str(keys[7]),
                "modifier_label": str(keys[8]),
                "modifier_kind": str(keys[9]),
                "modifier_row_count": int(len(group)),
                "estimated_stratum_count": estimated_rows,
                "consistent_stratum_count": consistent_rows,
                "inconsistent_stratum_count": opposite_direction_rows,
                "opposite_direction_stratum_count": opposite_direction_rows,
                "opposite_direction_significant_stratum_count": opposite_direction_significant_rows,
                "summary_classification": summary_classification,
                "matrix_flag": matrix_flag,
                "progression_blocker_flag": bool(
                    summary_classification == "directional_reversal_present"
                    and str(keys[7]) not in NON_BLOCKING_PROGRESS_MODIFIERS
                ),
                "max_abs_relative_change_pct": float(
                    pd.to_numeric(group["relative_change_pct"], errors="coerce").abs().max()
                ) if estimated_rows > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["candidate_track", "analysis_role", "analysis_unit_id", "modifier_name"],
        ignore_index=True,
    )


def _build_scorecard(
    candidate_selection: pd.DataFrame,
    heterogeneity_results: pd.DataFrame,
    effect_modification_matrix: pd.DataFrame,
) -> pd.DataFrame:
    if candidate_selection.empty:
        return pd.DataFrame()

    full_sample_summary = heterogeneity_results.loc[
        :,
        [
            "analysis_unit_id",
            "candidate_track",
            "full_sample_estimate",
            "full_sample_lower_ci",
            "full_sample_upper_ci",
            "full_sample_p_value",
            "full_sample_valid_strata_count",
        ],
    ].drop_duplicates(subset=["analysis_unit_id", "candidate_track"])
    modifier_summary = effect_modification_matrix.groupby(
        ["analysis_unit_id", "candidate_track"],
        as_index=False,
    ).agg(
        modifier_count_reviewed=("modifier_name", "nunique"),
        modifier_count_with_estimates=("estimated_stratum_count", lambda series: int((pd.to_numeric(series, errors="coerce") > 0).sum())),
        inconsistent_modifier_count=("matrix_flag", lambda series: int(pd.Series(series).astype(str).ne("consistent").sum())),
        significant_reversal_modifier_count=("matrix_flag", lambda series: int(pd.Series(series).astype(str).eq("significant_reversal").sum())),
        blocking_inconsistent_modifier_count=("progression_blocker_flag", lambda series: int(pd.Series(series).fillna(False).astype(bool).sum())),
        stable_modifier_count=("summary_classification", lambda series: int(series.astype(str).eq("stable_or_gradient").sum())),
    )

    scorecard = candidate_selection.merge(
        full_sample_summary,
        on=["analysis_unit_id", "candidate_track"],
        how="left",
    ).merge(
        modifier_summary,
        on=["analysis_unit_id", "candidate_track"],
        how="left",
    )
    scorecard["modifier_count_reviewed"] = scorecard["modifier_count_reviewed"].fillna(0).astype(int)
    scorecard["modifier_count_with_estimates"] = scorecard["modifier_count_with_estimates"].fillna(0).astype(int)
    scorecard["inconsistent_modifier_count"] = scorecard["inconsistent_modifier_count"].fillna(0).astype(int)
    scorecard["significant_reversal_modifier_count"] = scorecard["significant_reversal_modifier_count"].fillna(0).astype(int)
    scorecard["blocking_inconsistent_modifier_count"] = scorecard["blocking_inconsistent_modifier_count"].fillna(0).astype(int)
    scorecard["non_blocking_inconsistent_modifier_count"] = (
        scorecard["inconsistent_modifier_count"] - scorecard["blocking_inconsistent_modifier_count"]
    ).clip(lower=0).astype(int)
    scorecard["has_any_significant_reversal_flag"] = scorecard["significant_reversal_modifier_count"].gt(0)
    scorecard["stable_modifier_count"] = scorecard["stable_modifier_count"].fillna(0).astype(int)
    scorecard["full_sample_estimable_flag"] = scorecard["full_sample_estimate"].apply(lambda value: math.isfinite(float(value)))
    scorecard["candidate_ready_for_step6_flag"] = True
    scorecard["next_step_route"] = STEP5_NEXT_STEP_ROUTE
    scorecard["decision_reason"] = np.select(
        [
            ~scorecard["full_sample_estimable_flag"],
            scorecard["significant_reversal_modifier_count"].gt(0),
            scorecard["inconsistent_modifier_count"].gt(0),
        ],
        [
            "selected_window_not_estimable_in_step5_full_sample",
            "promoted_to_step6_with_significant_reversal",
            "promoted_to_step6_with_reversal",
        ],
        default="promoted_to_step6_consistent",
    )
    return scorecard.loc[
        :,
        [
            "analysis_unit_id",
            "linked_outcome_ids",
            "candidate_track",
            "analysis_role",
            "outcome_family",
            "selected_exposure_family_id",
            "selected_exposure_family_name",
            "selected_exposure_variable",
            "selected_exposure_label",
            "preferred_lag_or_window",
            "seasonality_profile_id",
            "seasonality_controls_applied",
            "mart_name",
            "denominator_id",
            "denominator_column",
            "estimate_scale",
            "full_sample_estimable_flag",
            "full_sample_estimate",
            "full_sample_lower_ci",
            "full_sample_upper_ci",
            "full_sample_p_value",
            "full_sample_valid_strata_count",
            "modifier_count_reviewed",
            "modifier_count_with_estimates",
            "stable_modifier_count",
            "inconsistent_modifier_count",
            "significant_reversal_modifier_count",
            "has_any_significant_reversal_flag",
            "blocking_inconsistent_modifier_count",
            "non_blocking_inconsistent_modifier_count",
            "candidate_ready_for_step6_flag",
            "next_step_route",
            "decision_reason",
        ],
    ].sort_values(
        ["candidate_track", "analysis_role", "analysis_unit_id"],
        ignore_index=True,
    )


def _build_decision_checklist(
    candidate_count: int,
    heterogeneity_results: pd.DataFrame,
    effect_modification_matrix: pd.DataFrame,
    scorecard: pd.DataFrame,
    gates: dict[str, bool],
    summary: dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "checklist_item": "Every Step 4-ready candidate entered Step 5 selection.",
                "pass_flag": bool(gates["candidate_selection_nonempty"]),
                "evidence": f"{candidate_count} candidates were selected from Step 4.",
            },
            {
                "checklist_item": "Selected exposure family and lag/window resolved to a concrete mart exposure column for every candidate.",
                "pass_flag": bool(gates["selected_specs_resolved_for_candidates"]),
                "evidence": f"{candidate_count}/{candidate_count} selected candidates resolved to an exposure column." if candidate_count > 0 else "0 candidates selected.",
            },
            {
                "checklist_item": "Context lookup was available and joined without missing rows in candidate-valid data.",
                "pass_flag": bool(gates["context_join_complete_for_candidates"]),
                "evidence": (
                    f"Missing joined-context rows: {int(gates.get('missing_context_rows', 0))}; "
                    f"Gini-only suppressed rows: {int(gates.get('gini_suppressed_rows', 0))}; "
                    f"Gini-only excluded municipalities: {int(summary.get('gini_excluded_municipality_count', 0))}."
                ),
            },
            {
                "checklist_item": "Heterogeneity results were written for every candidate-modifier-stratum combination.",
                "pass_flag": bool(gates["heterogeneity_results_written"]),
                "evidence": f"{len(heterogeneity_results)}/{_expected_heterogeneity_rows(candidate_count)} expected heterogeneity rows were written.",
            },
            {
                "checklist_item": "A modifier summary and Step 6 decision was written for every Step 5 candidate.",
                "pass_flag": bool(gates["step6_decisions_written"]),
                "evidence": f"{len(scorecard)}/{candidate_count} candidates received a Step 6 route; {len(effect_modification_matrix)} modifier summaries were written.",
            },
        ]
    )


def _render_step5_report(
    round_config: dict[str, Any],
    step4_report: dict[str, Any],
    gates: dict[str, bool],
    summary: dict[str, Any],
    outputs: dict[str, str],
    workspace_report: dict[str, object],
    scorecard: pd.DataFrame,
    gini_exclusion_frame: pd.DataFrame,
) -> str:
    lines = [
        "# Step 5 Heterogeneity Synthesis Report",
        "",
        f"- Generated at: `{_now_utc()}`",
        f"- Analysis ID: `{round_config['analysis_id']}`",
        f"- Analysis title: `{round_config['analysis_title']}`",
        f"- Years: `{', '.join(str(year) for year in round_config['years'])}`",
        f"- Upstream Step 4 status: `{step4_report['status']}`",
        "",
        "## Gates",
        "",
    ]
    for gate_name, passed in gates.items():
        if gate_name.endswith("_rows"):
            continue
        lines.append(f"- {gate_name}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Step 5 candidates reviewed: `{summary['candidate_count']}`",
            f"- Heterogeneity result rows: `{summary['heterogeneity_row_count']}`",
            f"- Modifier summaries written: `{summary['modifier_summary_count']}`",
            f"- Candidates with directional reversals: `{summary['inconsistent_candidate_count']}`",
            f"- Candidates with significant directional reversals: `{summary['significant_reversal_candidate_count']}`",
            f"- Candidates ready for Step 6: `{summary['step6_ready_count']}`",
            f"- Rows suppressed only for Gini stratification: `{summary['gini_suppressed_rows']}`",
            f"- Municipalities formally excluded only for Gini stratification: `{summary['gini_excluded_municipality_count']}`",
        ]
    )
    if not gini_exclusion_frame.empty:
        lines.append(
            f"- Gini-only excluded municipality codes: `{', '.join(gini_exclusion_frame['municipality_code'].astype(str).tolist())}`"
        )
    if not scorecard.empty:
        lines.extend(["", "## Step 5 Decisions", ""])
        for record in scorecard.to_dict(orient="records"):
            lines.append(
                "- "
                f"{record['analysis_unit_id']}: family=`{record['selected_exposure_family_id']}`, "
                f"lag_window=`{record['preferred_lag_or_window']}`, "
                f"inconsistent_modifiers=`{record['inconsistent_modifier_count']}`, "
                f"significant_reversal_modifiers=`{record['significant_reversal_modifier_count']}`, "
                f"step6_route=`{record['next_step_route']}`, "
                f"reason=`{record['decision_reason']}`"
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


def _write_step5_publication_notes(
    root: Path,
    *,
    status: str,
    summary: dict[str, Any],
    outputs: dict[str, str],
) -> None:
    workspace_root = incremental_publication_root(root)
    (workspace_root / "10_step5_heterogeneity_synthesis.md").write_text(
        (
            "# Step 5 Heterogeneity Synthesis\n\n"
            f"- Status: `{status}`\n"
            f"- Candidates reviewed: `{summary['candidate_count']}`\n"
            f"- Candidates with directional reversals: `{summary['inconsistent_candidate_count']}`\n"
            f"- Candidates with significant directional reversals: `{summary['significant_reversal_candidate_count']}`\n"
            f"- Canonical matrix: `{outputs['step5_block_matrix_csv']}`\n"
            f"- Technical report: `{outputs['step5_report']}`\n"
        ),
        encoding="utf-8",
    )
    (workspace_root / "11_step5_step6_gate.md").write_text(
        (
            "# Step 5 to Step 6 Gate\n\n"
            f"- Status: `{status}`\n"
            f"- Candidates promoted to Step 6: `{summary['step6_ready_count']}`\n"
            f"- Step 6 scorecard: `{outputs['step5_scorecard']}`\n"
            f"- Reversal summary: `{outputs['step5_directional_reversal_summary']}`\n"
        ),
        encoding="utf-8",
    )


def build_step5_heterogeneity_synthesis(
    root: Path,
    config_path: Path | None = None,
    step4_report: dict[str, Any] | None = None,
) -> dict[str, object]:
    directories = ensure_step5_directory_structure(root)
    round_config = load_round_config(root, config_path=config_path)
    workspace_report = _scaffold_step5_publication_workspace(root)
    upstream_step4_report = step4_report or build_step4_lag_refinement(root, config_path=config_path)

    heterogeneity_path = analysis_step5_root(root) / "heterogeneity_results.csv"
    modifier_matrix_path = analysis_step5_root(root) / "step5_effect_modification_matrix.csv"
    scorecard_path = analysis_step5_root(root) / "step5_scorecard.csv"
    gini_exclusion_path = analysis_step5_root(root) / "gini_analytic_universe_exclusions.csv"
    checklist_path = analysis_step5_root(root) / "step5_decision_checklist.csv"
    report_path = analysis_step5_root(root) / "step5_report.md"
    reversal_summary_path = analysis_step5_root(root) / "step5_directional_reversal_summary.csv"
    reversal_detail_csv_path = analysis_step5_root(root) / "step5_directional_reversal_diagnostic.csv"
    reversal_detail_xlsx_path = analysis_step5_root(root) / "step5_directional_reversal_diagnostic.xlsx"
    matrix_csv_relative = _relative(root, root / STEP5_MATRIX_RELATIVE_ROOT / "step5_block_matrix.csv")
    matrix_xlsx_relative = _relative(root, root / STEP5_MATRIX_RELATIVE_ROOT / "step5_block_matrix.xlsx")

    outputs = {
        "heterogeneity_results": _relative(root, heterogeneity_path),
        "effect_modification_matrix": _relative(root, modifier_matrix_path),
        "step5_scorecard": _relative(root, scorecard_path),
        "gini_analytic_universe_exclusions": _relative(root, gini_exclusion_path),
        "step5_directional_reversal_summary": _relative(root, reversal_summary_path),
        "step5_directional_reversal_diagnostic": _relative(root, reversal_detail_csv_path),
        "step5_directional_reversal_diagnostic_xlsx": _relative(root, reversal_detail_xlsx_path),
        "step5_block_matrix_csv": matrix_csv_relative,
        "step5_block_matrix_xlsx": matrix_xlsx_relative,
        "decision_checklist": _relative(root, checklist_path),
        "step5_report": _relative(root, report_path),
    }

    empty = pd.DataFrame()
    if upstream_step4_report["status"] != "PASS":
        gates = {
            "upstream_step4_pass": False,
            "candidate_selection_nonempty": False,
            "selected_specs_resolved_for_candidates": False,
            "context_lookup_available": False,
            "context_join_complete_for_candidates": False,
            "heterogeneity_results_written": False,
            "modifier_review_complete": False,
            "step6_decisions_written": False,
            "missing_context_rows": 0,
            "gini_suppressed_rows": 0,
        }
        for path in (
            heterogeneity_path,
            modifier_matrix_path,
            scorecard_path,
            gini_exclusion_path,
            reversal_summary_path,
            reversal_detail_csv_path,
            checklist_path,
        ):
            empty.to_csv(path, index=False)
        Workbook().save(reversal_detail_xlsx_path)
        matrix_csv_relative, matrix_xlsx_relative = _write_matrix_artifacts(root, pd.DataFrame())
        outputs["step5_block_matrix_csv"] = matrix_csv_relative
        outputs["step5_block_matrix_xlsx"] = matrix_xlsx_relative
        report_path.write_text(
            _render_step5_report(
                round_config=round_config,
                step4_report=upstream_step4_report,
                gates=gates,
                summary={
                    "candidate_count": 0,
                    "heterogeneity_row_count": 0,
                    "modifier_summary_count": 0,
                    "inconsistent_candidate_count": 0,
                    "significant_reversal_candidate_count": 0,
                    "step6_ready_count": 0,
                    "gini_suppressed_rows": 0,
                    "gini_excluded_municipality_count": 0,
                },
                outputs=outputs,
                workspace_report=workspace_report,
                scorecard=pd.DataFrame(),
                gini_exclusion_frame=pd.DataFrame(),
            ),
            encoding="utf-8",
        )
        _write_step5_publication_notes(
            root,
            status="FAIL",
            summary={
                "candidate_count": 0,
                "inconsistent_candidate_count": 0,
                "significant_reversal_candidate_count": 0,
                "step6_ready_count": 0,
            },
            outputs=outputs,
        )
        return {
            "status": "FAIL",
            "step": STEP5_NAME,
            "analysis_id": round_config["analysis_id"],
            "years": list(round_config["years"]),
            "directories": directories,
            "upstream_step4_status": upstream_step4_report["status"],
            "gates": gates,
            "summary": {
                "candidate_count": 0,
                "heterogeneity_row_count": 0,
                "modifier_summary_count": 0,
                "inconsistent_candidate_count": 0,
                "significant_reversal_candidate_count": 0,
                "step6_ready_count": 0,
                "gini_suppressed_rows": 0,
                "gini_excluded_municipality_count": 0,
            },
            "outputs": outputs,
            "workspace_report": workspace_report,
        }

    step4_decisions = _load_step4_inputs(root)
    preferred_models = _load_step2_preferred_models(root)
    candidate_selection = _build_candidate_selection(step4_decisions, preferred_models)
    context_lookup_path = analysis_shared_marts_root(root) / "context_lookup.parquet"
    context_lookup = _build_context_lookup_frame(root) if context_lookup_path.exists() else pd.DataFrame()
    gini_exclusion_frame = _build_gini_exclusion_frame(root) if context_lookup_path.exists() else pd.DataFrame(
        columns=["municipality_code", "context_join_status", "exclusion_scope", "exclusion_reason"]
    )

    selected_specs_resolved = bool(
        not candidate_selection.empty
        and candidate_selection["mart_name"].notna().all()
        and candidate_selection["seasonality_profile_id"].notna().all()
        and candidate_selection["selected_exposure_variable"].astype(str).str.len().gt(0).all()
    )

    required_columns = _required_columns_by_mart(candidate_selection) if selected_specs_resolved else {}
    frames_by_mart = (
        _load_model_frames(root, list(round_config["years"]), required_columns, context_lookup)
        if required_columns and not context_lookup.empty
        else {}
    )

    if frames_by_mart:
        heterogeneity_results, diagnostics = _build_heterogeneity_results(candidate_selection, frames_by_mart)
        effect_modification_matrix = _build_effect_modification_matrix(heterogeneity_results)
        scorecard = _build_scorecard(candidate_selection, heterogeneity_results, effect_modification_matrix)
        reversal_summary_frame, reversal_detail_frame = _build_directional_reversal_diagnostics(heterogeneity_results)
    else:
        heterogeneity_results, diagnostics = empty, {"missing_context_rows": 0, "gini_suppressed_rows": 0}
        effect_modification_matrix, scorecard = empty, empty
        reversal_summary_frame, reversal_detail_frame = _empty_reversal_summary_frame(), _empty_reversal_detail_frame()

    candidate_count = int(len(candidate_selection))
    expected_rows = _expected_heterogeneity_rows(candidate_count)
    expected_modifier_rows = candidate_count * len(MODIFIER_SPECS)
    inconsistent_candidate_count = int(
        scorecard["inconsistent_modifier_count"].fillna(0).gt(0).sum()
    ) if not scorecard.empty else 0
    significant_reversal_candidate_count = int(
        scorecard["has_any_significant_reversal_flag"].fillna(False).astype(bool).sum()
    ) if not scorecard.empty else 0
    step6_ready_count = int(len(scorecard)) if not scorecard.empty else 0

    gates = {
        "upstream_step4_pass": True,
        "candidate_selection_nonempty": candidate_count > 0,
        "selected_specs_resolved_for_candidates": selected_specs_resolved,
        "context_lookup_available": not context_lookup.empty,
        "context_join_complete_for_candidates": int(diagnostics["missing_context_rows"]) == 0,
        "heterogeneity_results_written": len(heterogeneity_results) == expected_rows if candidate_count > 0 else False,
        "modifier_review_complete": len(effect_modification_matrix) == expected_modifier_rows if candidate_count > 0 else False,
        "step6_decisions_written": len(scorecard) == candidate_count if candidate_count > 0 else False,
        "missing_context_rows": int(diagnostics["missing_context_rows"]),
        "gini_suppressed_rows": int(diagnostics["gini_suppressed_rows"]),
    }

    heterogeneity_results.to_csv(heterogeneity_path, index=False)
    effect_modification_matrix.to_csv(modifier_matrix_path, index=False)
    scorecard.to_csv(scorecard_path, index=False)
    gini_exclusion_frame.to_csv(gini_exclusion_path, index=False)
    reversal_summary_relative, reversal_detail_csv_relative, reversal_detail_xlsx_relative = _write_directional_reversal_artifacts(
        root,
        reversal_summary_frame,
        reversal_detail_frame,
    )
    outputs["step5_directional_reversal_summary"] = reversal_summary_relative
    outputs["step5_directional_reversal_diagnostic"] = reversal_detail_csv_relative
    outputs["step5_directional_reversal_diagnostic_xlsx"] = reversal_detail_xlsx_relative
    matrix_frame = _build_matrix_frame(scorecard, effect_modification_matrix)
    matrix_csv_relative, matrix_xlsx_relative = _write_matrix_artifacts(root, matrix_frame)
    outputs["step5_block_matrix_csv"] = matrix_csv_relative
    outputs["step5_block_matrix_xlsx"] = matrix_xlsx_relative
    checklist = _build_decision_checklist(
        candidate_count=candidate_count,
        heterogeneity_results=heterogeneity_results,
        effect_modification_matrix=effect_modification_matrix,
        scorecard=scorecard,
        gates=gates,
        summary={
            "gini_excluded_municipality_count": int(len(gini_exclusion_frame)),
        },
    )
    checklist.to_csv(checklist_path, index=False)

    summary = {
        "candidate_count": candidate_count,
        "heterogeneity_row_count": int(len(heterogeneity_results)),
        "modifier_summary_count": int(len(effect_modification_matrix)),
        "inconsistent_candidate_count": inconsistent_candidate_count,
        "significant_reversal_candidate_count": significant_reversal_candidate_count,
        "step6_ready_count": step6_ready_count,
        "gini_suppressed_rows": int(diagnostics["gini_suppressed_rows"]),
        "gini_excluded_municipality_count": int(len(gini_exclusion_frame)),
    }
    report_path.write_text(
        _render_step5_report(
            round_config=round_config,
            step4_report=upstream_step4_report,
            gates=gates,
            summary=summary,
            outputs=outputs,
            workspace_report=workspace_report,
            scorecard=scorecard,
            gini_exclusion_frame=gini_exclusion_frame,
        ),
        encoding="utf-8",
    )
    status = "PASS" if all(bool(value) for key, value in gates.items() if not key.endswith("_rows")) else "FAIL"
    _write_step5_publication_notes(root, status=status, summary=summary, outputs=outputs)
    return {
        "status": status,
        "step": STEP5_NAME,
        "analysis_id": round_config["analysis_id"],
        "years": list(round_config["years"]),
        "directories": directories,
        "upstream_step4_status": upstream_step4_report["status"],
        "gates": gates,
        "summary": summary,
        "outputs": outputs,
        "workspace_report": workspace_report,
    }
