from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from .config import load_track_b_config
from .registry import ensure_track_b_directory_structure, track_b_models_root
from .shared import (
    assign_grouped_quantile_bins,
    floor_probability,
    load_track_b_matrix_partitions,
    pooled_continuous_proxy,
    relative_to_root,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _plot_preterm_models(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    figure = plt.figure(figsize=(11, 5.5))
    axis = figure.add_subplot(1, 1, 1)
    if frame.empty:
        axis.text(0.5, 0.5, "No preterm model rows available.", ha="center", va="center")
    else:
        subset = frame.sort_values(["outcome_label", "model_sequence"])
        y_positions = np.arange(subset.shape[0])
        axis.errorbar(
            subset["estimate"],
            y_positions,
            xerr=[subset["estimate"] - subset["lower_ci"], subset["upper_ci"] - subset["estimate"]],
            fmt="o",
            color="#8b1e3f",
            ecolor="#4d4d4d",
            capsize=3,
        )
        axis.axvline(1.0, linestyle="--", color="black", linewidth=1)
        axis.set_yticks(y_positions, subset["display_label"].tolist())
        axis.set_xlabel("Rate ratio per one-unit increase")
    axis.set_title(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_severity_models(frame: pd.DataFrame, output_path: Path) -> None:
    figure = plt.figure(figsize=(9, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    if frame.empty:
        axis.text(0.5, 0.5, "No severity rows available.", ha="center", va="center")
    else:
        y_positions = np.arange(frame.shape[0])
        axis.errorbar(
            frame["estimate"],
            y_positions,
            xerr=[frame["estimate"] - frame["lower_ci"], frame["upper_ci"] - frame["estimate"]],
            fmt="o",
            color="#005f73",
            ecolor="#4d4d4d",
            capsize=3,
        )
        axis.axvline(0.0, linestyle="--", color="black", linewidth=1)
        axis.set_yticks(y_positions, frame["display_label"].tolist())
        axis.set_xlabel("Mean difference in gestational days lost per one-unit increase")
    axis.set_title("Track B preterm severity models")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _aggregate_quantile_model_data(
    frame: pd.DataFrame,
    *,
    exposure_column: str,
    outcome_column: str,
    stratum_columns: list[str],
    bin_count: int,
) -> pd.DataFrame:
    working = frame.loc[
        frame[exposure_column].notna()
        & frame[outcome_column].notna()
        & frame["birth_count"].gt(0)
    ].copy()
    working["exposure_bin"] = assign_grouped_quantile_bins(
        working,
        exposure_column=exposure_column,
        group_columns=stratum_columns,
        bin_count=bin_count,
    )
    working = working.loc[working["exposure_bin"].notna()].copy()
    grouped = (
        working.groupby(stratum_columns + ["exposure_bin"], as_index=False)
        .agg(
            births_total=("birth_count", "sum"),
            outcome_total=(outcome_column, "sum"),
            exposure_mean=(exposure_column, "mean"),
            source_sensitive_births=("source_sensitive_flag", "sum"),
        )
    )
    return grouped.rename(
        columns={
            "births_total": "denominator_births",
            "outcome_total": outcome_column,
            "exposure_mean": exposure_column,
        }
    )


def _build_methods_note() -> str:
    return (
        "# Track B Preterm Methods Note\n\n"
        "Stage B2 implements the flagship preterm package from the event-level gestational exposure matrix.\n\n"
        "## Design Adaptation\n\n"
        "- The current repository environment does not yet fit a tractable full event-level modified Poisson model with thousands of municipality fixed effects over the full national cohort.\n"
        "- Accordingly, Stage B2 uses a grouped fixed-effects proxy design: live births are grouped within municipality-year-conception-month-gestational-age-source-exactness strata and exposure deciles, then modeled through weighted within-stratum log-rate regression and inverse-variance pooling.\n"
        "- Model labels explicitly state `modified_poisson_fe_proxy` or `gaussian_fe_proxy`; the package does not present them as exact maximum-likelihood estimators.\n\n"
        "## Covariate Contract\n\n"
        "- The primary cohort requires `gestational_age_valid_for_analysis_flag = 1` and full observability for the selected exposure window.\n"
        "- The flagship third-trimester exposure is operationalized as the fully observed pregnancy-specific segment from 28 gestational weeks to delivery, rather than as a fixed 84-day window that would mechanically exclude preterm births.\n"
        "- The adjustment architecture is operationalized through the grouping strata: municipality, calendar year, conception month, gestational-age source, and exact gestational-age measurement flag.\n"
        "- When the fully specified grouping strata are too sparse to estimate stable within-stratum exposure contrasts, the implementation relaxes the grouping hierarchy in a predeclared order and records the specification used in the output table.\n"
        "- Maternal age, schooling, and parity are not included because they are not available in `silver_sinasc_event`; they remain conceptual-only for Track B.\n"
    )


def _fit_with_fallback(
    subset: pd.DataFrame,
    *,
    exposure_column: str,
    outcome_column: str,
) -> tuple[pd.DataFrame, dict[str, Any], str, int]:
    candidate_strata = [
        ["municipality_code", "calendar_year", "month_of_conception", "gestational_age_source", "ga_exact_measurement_flag"],
        ["municipality_code", "calendar_year", "month_of_conception", "gestational_age_source"],
        ["municipality_code", "calendar_year", "month_of_conception"],
        ["municipality_code", "calendar_year"],
        ["calendar_year"],
    ]
    for stratum_columns in candidate_strata:
        for bin_count in [10, 5, 4]:
            aggregated = _aggregate_quantile_model_data(
                subset,
                exposure_column=exposure_column,
                outcome_column=outcome_column,
                stratum_columns=stratum_columns,
                bin_count=bin_count,
            )
            result = pooled_continuous_proxy(
                aggregated,
                exposure_column=exposure_column,
                outcome_column=outcome_column,
                denominator_column="denominator_births",
                stratum_columns=stratum_columns,
                model_kind="count" if outcome_column != "gestational_days_lost_below_37_weeks" else "severity",
                exposure_scale_value=1.0,
            )
            if result["valid_strata_count"] > 0:
                return aggregated, result, "|".join(stratum_columns), bin_count
    return aggregated, result, "|".join(candidate_strata[-1]), 4


def build_track_b_preterm_models(root: Path, config_path: Path | None = None) -> dict[str, object]:
    ensure_track_b_directory_structure(root)
    config = load_track_b_config(root, config_path=config_path)
    years = [int(year) for year in config["years"]]
    columns = [
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
    frame = load_track_b_matrix_partitions(root, years, columns=columns)
    live_births = frame.loc[
        frame["record_type"].eq("live_birth")
        & pd.to_numeric(frame["gestational_age_valid_for_analysis_flag"], errors="coerce").fillna(0).eq(1)
    ].copy()
    live_births["birth_count"] = 1.0
    live_births["month_of_conception"] = pd.to_numeric(live_births["month_of_conception"], errors="coerce").fillna(-1).astype(int)
    live_births["source_sensitive_flag"] = (
        pd.to_numeric(live_births["third_trimester_fallback_day_share"], errors="coerce").fillna(0).gt(0)
    ).astype(np.int8)
    live_births["third_trimester_mean_tmax_x_rh_mean_pct"] = (
        pd.to_numeric(live_births["third_trimester_mean_tmax_c"], errors="coerce")
        * pd.to_numeric(live_births["third_trimester_mean_rh_mean_pct"], errors="coerce")
    )
    model_specs = [
        {
            "outcome_column": "preterm_birth_flag",
            "outcome_label": "Preterm birth",
            "exposure_column": "third_trimester_mean_tmax_c",
            "exposure_label": "Third-trimester mean maximum temperature",
            "observed_flag": "third_trimester_full_window_observed_flag",
            "primary_model_flag": "yes",
            "model_sequence": 1,
        },
        {
            "outcome_column": "preterm_birth_flag",
            "outcome_label": "Preterm birth",
            "exposure_column": "days_0_27_before_event_mean_tmax_c",
            "exposure_label": "Mean maximum temperature during the 28 days before delivery",
            "observed_flag": "days_0_27_before_event_full_window_observed_flag",
            "primary_model_flag": "no",
            "model_sequence": 2,
        },
        {
            "outcome_column": "preterm_birth_flag",
            "outcome_label": "Preterm birth",
            "exposure_column": "whole_pregnancy_mean_tmax_c",
            "exposure_label": "Whole-pregnancy mean maximum temperature",
            "observed_flag": "whole_pregnancy_full_window_observed_flag",
            "primary_model_flag": "no",
            "model_sequence": 3,
        },
        {
            "outcome_column": "preterm_birth_flag",
            "outcome_label": "Preterm birth",
            "exposure_column": "third_trimester_mean_tmax_x_rh_mean_pct",
            "exposure_label": "Third-trimester maximum temperature by humidity interaction proxy",
            "observed_flag": "third_trimester_full_window_observed_flag",
            "primary_model_flag": "no",
            "model_sequence": 4,
        },
        {
            "outcome_column": "preterm_birth_flag",
            "outcome_label": "Preterm birth",
            "exposure_column": "third_trimester_heatwave_days",
            "exposure_label": "Third-trimester heatwave days",
            "observed_flag": "third_trimester_full_window_observed_flag",
            "primary_model_flag": "no",
            "model_sequence": 5,
        },
        {
            "outcome_column": "preterm_birth_flag",
            "outcome_label": "Preterm birth",
            "exposure_column": "third_trimester_mean_heat_index_c",
            "exposure_label": "Third-trimester mean heat index",
            "observed_flag": "third_trimester_full_window_observed_flag",
            "primary_model_flag": "no",
            "model_sequence": 6,
        },
        {
            "outcome_column": "very_preterm_birth_flag",
            "outcome_label": "Very preterm birth",
            "exposure_column": "third_trimester_mean_tmax_c",
            "exposure_label": "Third-trimester mean maximum temperature",
            "observed_flag": "third_trimester_full_window_observed_flag",
            "primary_model_flag": "no",
            "model_sequence": 7,
        },
    ]
    result_rows: list[dict[str, Any]] = []
    severity_rows: list[dict[str, Any]] = []
    for spec in model_specs:
        subset = live_births.loc[pd.to_numeric(live_births[spec["observed_flag"]], errors="coerce").fillna(0).eq(1)].copy()
        aggregated, result, stratum_specification, bin_count = _fit_with_fallback(
            subset,
            exposure_column=spec["exposure_column"],
            outcome_column=spec["outcome_column"],
        )
        result_rows.append(
            {
                "outcome_variable": spec["outcome_column"],
                "outcome_label": spec["outcome_label"],
                "exposure_variable": spec["exposure_column"],
                "exposure_label": spec["exposure_label"],
                "model_label": "modified_poisson_fe_proxy",
                "estimate_scale": "rate_ratio_per_one_unit_increase",
                "primary_model_flag": spec["primary_model_flag"],
                "cohort_definition": "live_births_with_valid_gestational_age",
                "model_sequence": spec["model_sequence"],
                "estimate": result["estimate"],
                "lower_ci": result["lower_ci"],
                "upper_ci": result["upper_ci"],
                "p_value": result["p_value"],
                "p_value_floor_applied_flag": result["p_value_floor_applied_flag"],
                "valid_strata_count": result["valid_strata_count"],
                "aggregated_row_count": int(aggregated.shape[0]),
                "birth_count": float(aggregated["denominator_births"].sum()) if not aggregated.empty else 0.0,
                "outcome_count": float(aggregated[spec["outcome_column"]].sum()) if not aggregated.empty else 0.0,
                "overdispersion_ratio": result["overdispersion_ratio"],
                "internal_slope": result["internal_slope"],
                "internal_slope_se": result["internal_slope_se"],
                "stratum_specification": stratum_specification,
                "exposure_bin_count": bin_count,
                "source_sensitive_births": float(aggregated["source_sensitive_births"].sum()) if not aggregated.empty else 0.0,
            }
        )

    severity_subset = live_births.loc[
        pd.to_numeric(live_births["third_trimester_full_window_observed_flag"], errors="coerce").fillna(0).eq(1)
    ].copy()
    severity_aggregated, severity_result, severity_stratum_specification, severity_bin_count = _fit_with_fallback(
        severity_subset,
        exposure_column="third_trimester_mean_tmax_c",
        outcome_column="gestational_days_lost_below_37_weeks",
    )
    severity_rows.append(
        {
            "outcome_variable": "gestational_days_lost_below_37_weeks",
            "outcome_label": "Gestational days lost below 37 weeks",
            "exposure_variable": "third_trimester_mean_tmax_c",
            "exposure_label": "Third-trimester mean maximum temperature",
            "model_label": "gaussian_fe_proxy",
            "estimate_scale": "mean_difference_per_one_unit_increase",
            "primary_model_flag": "yes",
            "display_label": "Gestational days lost\nThird-trimester mean maximum temperature",
            "estimate": severity_result["estimate"],
            "lower_ci": severity_result["lower_ci"],
            "upper_ci": severity_result["upper_ci"],
            "p_value": severity_result["p_value"],
            "p_value_floor_applied_flag": severity_result["p_value_floor_applied_flag"],
            "valid_strata_count": severity_result["valid_strata_count"],
            "aggregated_row_count": int(severity_aggregated.shape[0]),
            "birth_count": float(severity_aggregated["denominator_births"].sum()) if not severity_aggregated.empty else 0.0,
            "outcome_total": float(severity_aggregated["gestational_days_lost_below_37_weeks"].sum()) if not severity_aggregated.empty else 0.0,
            "internal_slope": severity_result["internal_slope"],
            "internal_slope_se": severity_result["internal_slope_se"],
            "stratum_specification": severity_stratum_specification,
            "exposure_bin_count": severity_bin_count,
        }
    )

    main_results = pd.DataFrame(result_rows)
    severity_results = pd.DataFrame(severity_rows)
    main_results["p_value"], main_results["p_value_floor_applied_flag_numeric"] = floor_probability(main_results["p_value"])
    severity_results["p_value"], severity_results["p_value_floor_applied_flag_numeric"] = floor_probability(severity_results["p_value"])
    main_results["display_label"] = main_results["outcome_label"] + "\n" + main_results["exposure_label"]

    models_root = track_b_models_root(root)
    plots_root = models_root / "preterm_effect_plots"
    models_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)
    main_path = models_root / "preterm_main_models.csv"
    severity_path = models_root / "gest_days_lost_models.csv"
    methods_path = models_root / "preterm_methods_note.md"
    figure_main_path = plots_root / "figure_01_preterm_main_models.png"
    figure_severity_path = plots_root / "figure_02_preterm_severity_models.png"

    main_results.to_csv(main_path, index=False)
    severity_results.to_csv(severity_path, index=False)
    methods_path.write_text(_build_methods_note(), encoding="utf-8")
    _plot_preterm_models(main_results, figure_main_path, "Track B preterm flagship models")
    _plot_severity_models(severity_results, figure_severity_path)

    payload = {
        "status": "PASS",
        "stage": "track_b_stage_b2_preterm_models",
        "analysis_id": config["analysis_id"],
        "main_model_row_count": int(main_results.shape[0]),
        "severity_model_row_count": int(severity_results.shape[0]),
        "outputs": {
            "preterm_main_models": relative_to_root(root, main_path),
            "gest_days_lost_models": relative_to_root(root, severity_path),
            "preterm_methods_note": relative_to_root(root, methods_path),
            "preterm_effect_plots_directory": relative_to_root(root, plots_root),
        },
    }
    (models_root / "preterm_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
