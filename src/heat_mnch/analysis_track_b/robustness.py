from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_track_b_config
from .registry import ensure_track_b_directory_structure, track_b_models_root, track_b_robustness_root
from .shared import assign_grouped_quantile_bins, load_track_b_matrix_partitions, pooled_continuous_proxy, relative_to_root
from .neonatal import _add_lags, _load_mortality


def _aggregate(frame: pd.DataFrame, *, exposure_column: str, outcome_column: str, denominator_column: str, stratum_columns: list[str]) -> pd.DataFrame:
    working = frame.loc[
        frame[exposure_column].notna()
        & frame[outcome_column].notna()
        & frame[denominator_column].fillna(0).gt(0)
    ].copy()
    working["exposure_bin"] = assign_grouped_quantile_bins(
        working,
        exposure_column=exposure_column,
        group_columns=stratum_columns,
        bin_count=10,
    )
    working = working.loc[working["exposure_bin"].notna()].copy()
    grouped = (
        working.groupby(stratum_columns + ["exposure_bin"], as_index=False)
        .agg(
            denominator_total=(denominator_column, "sum"),
            outcome_total=(outcome_column, "sum"),
            exposure_mean=(exposure_column, "mean"),
        )
    )
    return grouped.rename(columns={"denominator_total": denominator_column, "outcome_total": outcome_column, "exposure_mean": exposure_column})


def _bias_note() -> str:
    return (
        "# Track B Bias and Robustness Note\n\n"
        "Stage B7 summarizes the main robustness domains required for the flagship package.\n\n"
        "- Source-aware restrictions exclude rows with fallback-imputed exposure days in the primary window where those summaries are available.\n"
        "- Fixed-cohort bias is approximated by restricting gestational models to births or events occurring on or after 1 April 2020.\n"
        "- The week-of-gestation sensitivity file documents the current scaffold status of the long-format weekly dataset.\n"
    )


def build_track_b_robustness(root: Path, config_path: Path | None = None) -> dict[str, object]:
    ensure_track_b_directory_structure(root)
    config = load_track_b_config(root, config_path=config_path)
    years = [int(year) for year in config["years"]]
    robustness_root = track_b_robustness_root(root)
    robustness_root.mkdir(parents=True, exist_ok=True)

    preterm_models = pd.read_csv(track_b_models_root(root) / "preterm_main_models.csv")
    stillbirth_models = pd.read_csv(track_b_models_root(root) / "stillbirth_main_models.csv")
    neonatal_models = pd.read_csv(track_b_models_root(root) / "neonatal_main_models.csv")

    matrix_columns = [
        "record_type",
        "municipality_code",
        "calendar_year",
        "event_date",
        "month_of_conception",
        "gestational_age_source",
        "ga_exact_measurement_flag",
        "preterm_birth_flag",
        "stillbirth_primary_gte28_flag",
        "third_trimester_mean_tmax_c",
        "third_trimester_full_window_observed_flag",
        "third_trimester_fallback_day_share",
        "days_0_6_before_event_mean_tmax_c",
        "days_0_6_before_event_full_window_observed_flag",
        "days_0_6_before_event_fallback_day_share",
    ]
    event_matrix = load_track_b_matrix_partitions(root, years, columns=matrix_columns)
    event_matrix["birth_count"] = 1.0
    event_matrix["event_date"] = pd.to_datetime(event_matrix["event_date"], errors="coerce")
    rows: list[dict[str, Any]] = []

    preterm_source = event_matrix.loc[
        event_matrix["record_type"].eq("live_birth")
        & pd.to_numeric(event_matrix["third_trimester_full_window_observed_flag"], errors="coerce").fillna(0).eq(1)
        & pd.to_numeric(event_matrix["third_trimester_fallback_day_share"], errors="coerce").fillna(0).eq(0)
    ].copy()
    preterm_source["month_of_conception"] = pd.to_numeric(preterm_source["month_of_conception"], errors="coerce").fillna(-1).astype(int)
    preterm_agg = _aggregate(
        preterm_source,
        exposure_column="third_trimester_mean_tmax_c",
        outcome_column="preterm_birth_flag",
        denominator_column="birth_count",
        stratum_columns=["municipality_code", "calendar_year", "month_of_conception", "gestational_age_source", "ga_exact_measurement_flag"],
    )
    preterm_source_result = pooled_continuous_proxy(
        preterm_agg,
        exposure_column="third_trimester_mean_tmax_c",
        outcome_column="preterm_birth_flag",
        denominator_column="birth_count",
        stratum_columns=["municipality_code", "calendar_year", "month_of_conception", "gestational_age_source", "ga_exact_measurement_flag"],
        model_kind="count",
    )
    rows.append(
        {
            "outcome_label": "Preterm birth",
            "robustness_domain": "source_aware_primary_window_restriction",
            "estimate": preterm_source_result["estimate"],
            "lower_ci": preterm_source_result["lower_ci"],
            "upper_ci": preterm_source_result["upper_ci"],
            "p_value": preterm_source_result["p_value"],
            "analytic_row_count": int(preterm_agg.shape[0]),
        }
    )

    fixed_cohort = event_matrix.loc[
        event_matrix["record_type"].eq("live_birth")
        & pd.to_numeric(event_matrix["third_trimester_full_window_observed_flag"], errors="coerce").fillna(0).eq(1)
        & event_matrix["event_date"].ge(pd.Timestamp("2020-04-01"))
    ].copy()
    fixed_cohort["month_of_conception"] = pd.to_numeric(fixed_cohort["month_of_conception"], errors="coerce").fillna(-1).astype(int)
    fixed_agg = _aggregate(
        fixed_cohort,
        exposure_column="third_trimester_mean_tmax_c",
        outcome_column="preterm_birth_flag",
        denominator_column="birth_count",
        stratum_columns=["municipality_code", "calendar_year", "month_of_conception", "gestational_age_source", "ga_exact_measurement_flag"],
    )
    fixed_result = pooled_continuous_proxy(
        fixed_agg,
        exposure_column="third_trimester_mean_tmax_c",
        outcome_column="preterm_birth_flag",
        denominator_column="birth_count",
        stratum_columns=["municipality_code", "calendar_year", "month_of_conception", "gestational_age_source", "ga_exact_measurement_flag"],
        model_kind="count",
    )
    rows.append(
        {
            "outcome_label": "Preterm birth",
            "robustness_domain": "fixed_cohort_bias_restriction",
            "estimate": fixed_result["estimate"],
            "lower_ci": fixed_result["lower_ci"],
            "upper_ci": fixed_result["upper_ci"],
            "p_value": fixed_result["p_value"],
            "analytic_row_count": int(fixed_agg.shape[0]),
        }
    )

    stillbirth_source = event_matrix.loc[
        pd.to_numeric(event_matrix["days_0_6_before_event_full_window_observed_flag"], errors="coerce").fillna(0).eq(1)
        & pd.to_numeric(event_matrix["days_0_6_before_event_fallback_day_share"], errors="coerce").fillna(0).eq(0)
    ].copy()
    stillbirth_source["event_month"] = stillbirth_source["event_date"].dt.month.fillna(-1).astype(int)
    stillbirth_agg = _aggregate(
        stillbirth_source,
        exposure_column="days_0_6_before_event_mean_tmax_c",
        outcome_column="stillbirth_primary_gte28_flag",
        denominator_column="birth_count",
        stratum_columns=["municipality_code", "calendar_year", "event_month", "gestational_age_source"],
    )
    stillbirth_source_result = pooled_continuous_proxy(
        stillbirth_agg,
        exposure_column="days_0_6_before_event_mean_tmax_c",
        outcome_column="stillbirth_primary_gte28_flag",
        denominator_column="birth_count",
        stratum_columns=["municipality_code", "calendar_year", "event_month", "gestational_age_source"],
        model_kind="count",
    )
    rows.append(
        {
            "outcome_label": "Stillbirth",
            "robustness_domain": "source_aware_primary_window_restriction",
            "estimate": stillbirth_source_result["estimate"],
            "lower_ci": stillbirth_source_result["lower_ci"],
            "upper_ci": stillbirth_source_result["upper_ci"],
            "p_value": stillbirth_source_result["p_value"],
            "analytic_row_count": int(stillbirth_agg.shape[0]),
        }
    )

    neonatal = _add_lags(_load_mortality(root, years))
    neonatal = neonatal.loc[
        neonatal["QC_STATUS"].astype(str).eq("PASS")
        & pd.to_numeric(neonatal["OFFSET_VALID_FLAG"], errors="coerce").fillna(0).eq(1)
        & pd.to_numeric(neonatal["DENOMINATOR_ZERO_FLAG"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(neonatal["DENOMINATOR_MISSING_FLAG"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(neonatal["IMPUTADO_POR_FALLBACK"], errors="coerce").fillna(0).eq(0)
    ].copy()
    neonatal["day_of_week"] = neonatal["observation_date"].dt.dayofweek.fillna(-1).astype(int)
    neonatal["calendar_month"] = neonatal["observation_date"].dt.month.fillna(-1).astype(int)
    neonatal_result = pooled_continuous_proxy(
        neonatal,
        exposure_column="lag0_1_mean_tmax_c",
        outcome_column="DEATHS_NEONATAL",
        denominator_column="LIVE_BIRTHS_TOTAL_ANNUAL",
        stratum_columns=["municipality_code", "calendar_year", "calendar_month", "day_of_week"],
        model_kind="count",
    )
    rows.append(
        {
            "outcome_label": "Neonatal mortality",
            "robustness_domain": "source_aware_daily_restriction",
            "estimate": neonatal_result["estimate"],
            "lower_ci": neonatal_result["lower_ci"],
            "upper_ci": neonatal_result["upper_ci"],
            "p_value": neonatal_result["p_value"],
            "analytic_row_count": int(neonatal.shape[0]),
        }
    )

    for frame, label in [(preterm_models, "preterm_supporting_models"), (stillbirth_models, "stillbirth_supporting_models"), (neonatal_models, "neonatal_supporting_models")]:
        export = frame.copy()
        export["robustness_domain"] = label
        rows.extend(
            export.loc[:, ["outcome_label", "robustness_domain", "estimate", "lower_ci", "upper_ci", "p_value"]]
            .assign(analytic_row_count=pd.NA)
            .to_dict(orient="records")
        )

    robustness_frame = pd.DataFrame(rows)
    robustness_frame.to_csv(robustness_root / "flagship_robustness_matrix.csv", index=False)

    week_of_gestation = pd.DataFrame(
        [
            {
                "sensitivity_domain": "week_of_gestation_distributed_lag",
                "status": "deferred_with_scaffold_present",
                "note": "The long-format week-of-gestation dataset is scaffolded in Stage B1, but a fully estimated weekly distributed-lag model is not yet exported in this implementation pass.",
            }
        ]
    )
    week_of_gestation.to_csv(robustness_root / "week_of_gestation_sensitivity_results.csv", index=False)
    (robustness_root / "bias_sensitivity_note.md").write_text(_bias_note(), encoding="utf-8")

    payload = {
        "status": "PASS",
        "stage": "track_b_stage_b7_robustness",
        "analysis_id": config["analysis_id"],
        "robustness_row_count": int(robustness_frame.shape[0]),
        "outputs": {
            "flagship_robustness_matrix": relative_to_root(root, robustness_root / "flagship_robustness_matrix.csv"),
            "bias_sensitivity_note": relative_to_root(root, robustness_root / "bias_sensitivity_note.md"),
            "week_of_gestation_sensitivity_results": relative_to_root(root, robustness_root / "week_of_gestation_sensitivity_results.csv"),
        },
    }
    (robustness_root / "robustness_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
