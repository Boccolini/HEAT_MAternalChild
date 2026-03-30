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


def _aggregate_stillbirth_model_data(
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
            total_births=("birth_count", "sum"),
            stillbirth_total=(outcome_column, "sum"),
            exposure_mean=(exposure_column, "mean"),
            source_sensitive_births=("source_sensitive_flag", "sum"),
        )
    )
    return grouped.rename(
        columns={
            "total_births": "denominator_total_births",
            "stillbirth_total": outcome_column,
            "exposure_mean": exposure_column,
        }
    )


def _plot_stillbirth_models(frame: pd.DataFrame, output_path: Path) -> None:
    figure = plt.figure(figsize=(10.5, 5.0))
    axis = figure.add_subplot(1, 1, 1)
    if frame.empty:
        axis.text(0.5, 0.5, "No stillbirth models available.", ha="center", va="center")
    else:
        subset = frame.sort_values(["outcome_label", "model_sequence"])
        y_positions = np.arange(subset.shape[0])
        axis.errorbar(
            subset["estimate"],
            y_positions,
            xerr=[subset["estimate"] - subset["lower_ci"], subset["upper_ci"] - subset["estimate"]],
            fmt="o",
            color="#9a031e",
            ecolor="#555555",
            capsize=3,
        )
        axis.axvline(1.0, linestyle="--", color="black", linewidth=1)
        axis.set_yticks(y_positions, subset["display_label"].tolist())
        axis.set_xlabel("Rate ratio per one-unit increase")
    axis.set_title("Track B stillbirth flagship models")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _denominator_report(frame: pd.DataFrame) -> str:
    return (
        "# Stillbirth Denominator Benchmark Report\n\n"
        f"- Total analytic births: `{int(frame['birth_count'].sum())}`\n"
        f"- Stillbirth events at >=28 weeks: `{int(frame['stillbirth_primary_gte28_flag'].sum())}`\n"
        f"- Stillbirth events at >=22 weeks: `{int(frame['stillbirth_sensitivity_gte22_flag'].sum())}`\n"
        "- Denominator logic uses the harmonized total-birth event file built from live births plus stillbirth events.\n"
        "- No hidden removal of denominator-valid rows is performed at the reporting stage; analytic exclusions are limited to missing exposure observability or missing gestational-age fields where the selected specification requires them.\n"
    )


def _build_fetuses_at_risk_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.loc[
        frame["gestational_age_weeks_final"].notna()
        & frame["days_0_6_before_event_mean_tmax_c"].notna()
        & frame["gestational_age_weeks_final"].ge(28)
    ].copy()
    working["gestational_week_floor"] = np.floor(pd.to_numeric(working["gestational_age_weeks_final"], errors="coerce")).astype(int)
    working["exposure_quartile"] = assign_grouped_quantile_bins(
        working,
        exposure_column="days_0_6_before_event_mean_tmax_c",
        group_columns=["calendar_year"],
        bin_count=4,
    )
    rows: list[dict[str, Any]] = []
    for week in range(28, 43):
        risk_set = working.loc[working["gestational_week_floor"].ge(week)].copy()
        if risk_set.empty:
            continue
        grouped = risk_set.groupby(["calendar_year", "exposure_quartile"], as_index=False).agg(
            pregnancies_at_risk=("birth_count", "sum"),
            stillbirths_at_risk=("stillbirth_primary_gte28_flag", "sum"),
            mean_temperature=("days_0_6_before_event_mean_tmax_c", "mean"),
        )
        grouped["gestational_week_threshold"] = week
        rows.extend(grouped.to_dict(orient="records"))
    return pd.DataFrame(rows)


def _methods_note() -> str:
    return (
        "# Track B Stillbirth Methods Note\n\n"
        "Stage B3 uses the harmonized total-birth event file built from live births and stillbirth records.\n\n"
        "## Design Adaptation\n\n"
        "- The flagship stillbirth package uses grouped fixed-effects proxy models for tractability over the national event file.\n"
        "- Events are grouped within municipality-year-calendar-month-gestational-age-source strata and exposure deciles.\n"
        "- When the most granular grouping is too sparse for stable within-stratum estimation, the implementation relaxes the stratum hierarchy and records the specification used in the exported model table.\n"
        "- The primary estimate targets `stillbirth_primary_gte28_flag` with `days_0_6_before_event_mean_tmax_c` as the headline exposure.\n"
        "- A supplement-only fetuses-at-risk sensitivity proxy is exported separately and explicitly labeled as a proxy risk-set benchmark rather than as a full survival model.\n"
    )


def _fit_with_fallback(
    subset: pd.DataFrame,
    *,
    exposure_column: str,
    outcome_column: str,
) -> tuple[pd.DataFrame, dict[str, Any], str, int]:
    candidate_strata = [
        ["municipality_code", "calendar_year", "event_month", "gestational_age_source"],
        ["municipality_code", "calendar_year", "event_month"],
        ["municipality_code", "calendar_year"],
    ]
    for stratum_columns in candidate_strata:
        for bin_count in [10, 5, 4]:
            aggregated = _aggregate_stillbirth_model_data(
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
                denominator_column="denominator_total_births",
                stratum_columns=stratum_columns,
                model_kind="count",
                exposure_scale_value=1.0,
            )
            if result["valid_strata_count"] > 0:
                return aggregated, result, "|".join(stratum_columns), bin_count
    return aggregated, result, "|".join(candidate_strata[-1]), 4


def build_track_b_stillbirth_models(root: Path, config_path: Path | None = None) -> dict[str, object]:
    ensure_track_b_directory_structure(root)
    config = load_track_b_config(root, config_path=config_path)
    years = [int(year) for year in config["years"]]
    columns = [
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
    frame = load_track_b_matrix_partitions(root, years, columns=columns)
    frame["birth_count"] = 1.0
    frame["event_month"] = pd.to_datetime(frame["event_date"], errors="coerce").dt.month.fillna(-1).astype(int)
    frame["source_sensitive_flag"] = (
        pd.to_numeric(frame["days_0_6_before_event_fallback_day_share"], errors="coerce").fillna(0).gt(0)
    ).astype(np.int8)
    model_specs = [
        {
            "outcome_column": "stillbirth_primary_gte28_flag",
            "outcome_label": "Stillbirth at or beyond 28 weeks",
            "exposure_column": "days_0_6_before_event_mean_tmax_c",
            "exposure_label": "Mean maximum temperature during the seven days before the event",
            "observed_flag": "days_0_6_before_event_full_window_observed_flag",
            "primary_model_flag": "yes",
            "model_sequence": 1,
        },
        {
            "outcome_column": "stillbirth_primary_gte28_flag",
            "outcome_label": "Stillbirth at or beyond 28 weeks",
            "exposure_column": "days_0_13_before_event_mean_tmax_c",
            "exposure_label": "Mean maximum temperature during the 14 days before the event",
            "observed_flag": "days_0_13_before_event_full_window_observed_flag",
            "primary_model_flag": "no",
            "model_sequence": 2,
        },
        {
            "outcome_column": "stillbirth_primary_gte28_flag",
            "outcome_label": "Stillbirth at or beyond 28 weeks",
            "exposure_column": "days_0_27_before_event_mean_tmax_c",
            "exposure_label": "Mean maximum temperature during the 28 days before the event",
            "observed_flag": "days_0_27_before_event_full_window_observed_flag",
            "primary_model_flag": "no",
            "model_sequence": 3,
        },
        {
            "outcome_column": "stillbirth_primary_gte28_flag",
            "outcome_label": "Stillbirth at or beyond 28 weeks",
            "exposure_column": "third_trimester_mean_tmax_c",
            "exposure_label": "Third-trimester mean maximum temperature",
            "observed_flag": "third_trimester_full_window_observed_flag",
            "primary_model_flag": "no",
            "model_sequence": 4,
        },
        {
            "outcome_column": "stillbirth_sensitivity_gte22_flag",
            "outcome_label": "Stillbirth at or beyond 22 weeks",
            "exposure_column": "days_0_6_before_event_mean_tmax_c",
            "exposure_label": "Mean maximum temperature during the seven days before the event",
            "observed_flag": "days_0_6_before_event_full_window_observed_flag",
            "primary_model_flag": "no",
            "model_sequence": 5,
        },
        {
            "outcome_column": "stillbirth_sensitivity_gte22_flag",
            "outcome_label": "Stillbirth at or beyond 22 weeks",
            "exposure_column": "days_0_6_before_event_mean_heat_index_c",
            "exposure_label": "Mean heat index during the seven days before the event",
            "observed_flag": "days_0_6_before_event_full_window_observed_flag",
            "primary_model_flag": "no",
            "model_sequence": 6,
        },
    ]
    rows: list[dict[str, Any]] = []
    for spec in model_specs:
        subset = frame.loc[pd.to_numeric(frame[spec["observed_flag"]], errors="coerce").fillna(0).eq(1)].copy()
        aggregated, result, stratum_specification, bin_count = _fit_with_fallback(
            subset,
            exposure_column=spec["exposure_column"],
            outcome_column=spec["outcome_column"],
        )
        rows.append(
            {
                "outcome_variable": spec["outcome_column"],
                "outcome_label": spec["outcome_label"],
                "exposure_variable": spec["exposure_column"],
                "exposure_label": spec["exposure_label"],
                "model_label": "poisson_fe_proxy_total_birth_offset",
                "estimate_scale": "rate_ratio_per_one_unit_increase",
                "primary_model_flag": spec["primary_model_flag"],
                "model_sequence": spec["model_sequence"],
                "estimate": result["estimate"],
                "lower_ci": result["lower_ci"],
                "upper_ci": result["upper_ci"],
                "p_value": result["p_value"],
                "p_value_floor_applied_flag": result["p_value_floor_applied_flag"],
                "valid_strata_count": result["valid_strata_count"],
                "aggregated_row_count": int(aggregated.shape[0]),
                "birth_count": float(aggregated["denominator_total_births"].sum()) if not aggregated.empty else 0.0,
                "outcome_count": float(aggregated[spec["outcome_column"]].sum()) if not aggregated.empty else 0.0,
                "overdispersion_ratio": result["overdispersion_ratio"],
                "internal_slope": result["internal_slope"],
                "internal_slope_se": result["internal_slope_se"],
                "stratum_specification": stratum_specification,
                "exposure_bin_count": bin_count,
                "source_sensitive_births": float(aggregated["source_sensitive_births"].sum()) if not aggregated.empty else 0.0,
            }
        )

    model_results = pd.DataFrame(rows)
    model_results["p_value"], model_results["p_value_floor_applied_flag_numeric"] = floor_probability(model_results["p_value"])
    model_results["display_label"] = model_results["outcome_label"] + "\n" + model_results["exposure_label"]
    main_results = model_results.loc[model_results["outcome_variable"].eq("stillbirth_primary_gte28_flag")].copy()
    sensitivity_results = model_results.loc[~model_results["outcome_variable"].eq("stillbirth_primary_gte28_flag")].copy()

    far_proxy = _build_fetuses_at_risk_proxy(frame)

    models_root = track_b_models_root(root)
    plots_root = models_root / "stillbirth_effect_plots"
    models_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)
    main_path = models_root / "stillbirth_main_models.csv"
    sensitivity_path = models_root / "stillbirth_sensitivity_models.csv"
    denominator_path = models_root / "denominator_benchmark_report.md"
    far_path = models_root / "fetuses_at_risk_sensitivity.csv"
    methods_path = models_root / "stillbirth_methods_note.md"
    figure_path = plots_root / "figure_01_stillbirth_models.png"

    main_results.to_csv(main_path, index=False)
    sensitivity_results.to_csv(sensitivity_path, index=False)
    far_proxy.to_csv(far_path, index=False)
    denominator_path.write_text(_denominator_report(frame), encoding="utf-8")
    methods_path.write_text(_methods_note(), encoding="utf-8")
    _plot_stillbirth_models(model_results, figure_path)

    payload = {
        "status": "PASS",
        "stage": "track_b_stage_b3_stillbirth_models",
        "analysis_id": config["analysis_id"],
        "main_model_row_count": int(main_results.shape[0]),
        "sensitivity_model_row_count": int(sensitivity_results.shape[0]),
        "fetuses_at_risk_row_count": int(far_proxy.shape[0]),
        "outputs": {
            "stillbirth_main_models": relative_to_root(root, main_path),
            "stillbirth_sensitivity_models": relative_to_root(root, sensitivity_path),
            "denominator_benchmark_report": relative_to_root(root, denominator_path),
            "fetuses_at_risk_sensitivity": relative_to_root(root, far_path),
            "stillbirth_methods_note": relative_to_root(root, methods_path),
        },
    }
    (models_root / "stillbirth_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
