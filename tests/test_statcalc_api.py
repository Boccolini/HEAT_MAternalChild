from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from heat_mnch.statcalc_api.app import create_app


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_statcalc_fixture(root: Path) -> None:
    _write_json(
        root / "config" / "analysis" / "track_a.yaml",
        {
            "analysis_id": "track_a_framework_validation",
            "analysis_title": "Track A Framework Validation",
            "years": [2024],
            "output_roots": {
                "artifacts_root": "artifacts/track_a",
                "metadata_root": "artifacts/track_a/metadata",
                "publication_root": "reports/publications/track_a_framework_validation",
            },
        },
    )
    _write_json(
        root / "config" / "analysis" / "track_b.yaml",
        {
            "analysis_id": "track_b_lanplh_flagship_perinatal",
            "analysis_title": "Track B Lancet Planetary Health Flagship Perinatal Paper",
            "years": [2024],
            "output_roots": {
                "artifacts_root": "artifacts/track_b",
                "event_level_root": "artifacts/track_b/event_level",
                "publication_root": "reports/publications/track_b_lanplh_flagship_perinatal",
            },
        },
    )

    metadata = pd.DataFrame(
        [
            {
                "COD_MUNICIPIO": 110001,
                "NM_MUNICIPIO": "Alta Floresta D'Oeste",
                "UF": "RO",
                "MUNICIPIO_CENTROID_LAT": -11.9,
                "MUNICIPIO_CENTROID_LON": -61.99,
                "ERA5_LAT": -11.9,
                "ERA5_LON": -61.99,
                "TIMEZONE_NAME": "America/Porto_Velho",
            },
            {
                "COD_MUNICIPIO": 110002,
                "NM_MUNICIPIO": "Ariquemes",
                "UF": "RO",
                "MUNICIPIO_CENTROID_LAT": -9.9,
                "MUNICIPIO_CENTROID_LON": -63.0,
                "ERA5_LAT": -9.9,
                "ERA5_LON": -63.0,
                "TIMEZONE_NAME": "America/Porto_Velho",
            },
        ]
    )
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_dir / "era5_municipio_grid_lookup.csv", index=False)

    context_lookup = pd.DataFrame(
        [
            {
                "municipality_code": "110001",
                "urban_population_proportion": 0.60,
                "expanded_gini_index": 0.55,
                "expanded_human_development_index": 0.64,
                "mean_altitude_m": 150.0,
                "coastal_municipality_flag": 0,
                "context_join_status": "OK",
            },
            {
                "municipality_code": "110002",
                "urban_population_proportion": 0.85,
                "expanded_gini_index": 0.48,
                "expanded_human_development_index": 0.71,
                "mean_altitude_m": 220.0,
                "coastal_municipality_flag": 1,
                "context_join_status": "OK",
            },
        ]
    )
    (root / "artifacts" / "track_a" / "analysis_marts").mkdir(parents=True, exist_ok=True)
    context_lookup.to_parquet(root / "artifacts" / "track_a" / "analysis_marts" / "context_lookup.parquet", index=False)

    triage = pd.DataFrame(
        [
            {
                "public_concept_name": "neonatal_death_count",
                "public_concept_label": "Neonatal death count",
                "stage1_classification": "benchmark_ready",
                "headline_eligible_flag": "yes",
                "benchmark_model_eligible_flag": "yes",
                "best_exposure_variable": "maximum_daily_temperature_c",
                "best_exposure_label": "Maximum daily temperature (lag 0)",
                "best_screening_domain": "acute_temperature_reference_contrast",
                "best_screening_method": "synthetic",
                "best_lag_or_window": "lag0",
                "best_estimate_scale": "rate_ratio",
                "best_estimate": 1.1,
                "best_lower_ci": 1.0,
                "best_upper_ci": 1.2,
                "minimum_p_value": 0.01,
                "minimum_q_value": 0.01,
                "screening_result_count": 1,
                "signal_domain_count_p_le_0_10": 1,
                "signal_direction_consistency": 1.0,
                "triage_priority_classification": "advance",
                "benchmark_stage_recommendation": "yes",
            }
        ]
    )
    screening_root = root / "artifacts" / "track_a" / "screening"
    screening_root.mkdir(parents=True, exist_ok=True)
    triage.to_csv(screening_root / "triage_signal_matrix.csv", index=False)

    benchmark = pd.DataFrame(
        [
            {
                "public_concept_name": "neonatal_death_count",
                "public_concept_label": "Neonatal death count",
                "outcome_variable": "neonatal_death_count",
                "outcome_label": "Neonatal death count",
                "outcome_family": "acute_daily",
                "stage1_classification": "benchmark_ready",
                "triage_priority_classification": "advance",
                "benchmark_stage_recommendation": "yes",
                "exposure_variable": "maximum_daily_temperature_c_lag0",
                "exposure_label": "Maximum daily temperature (lag 0)",
                "exposure_family_id": "family_1_tmax_primary",
                "exposure_family_name": "Maximum daily temperature",
                "lag_or_window": "lag0",
                "model_label": "poisson_fe_proxy",
                "model_family": "acute_benchmark_ladder",
                "estimate_scale": "rate_ratio_per_interquartile_range_increase",
                "exposure_scale_value": 1.0,
                "estimate": 1.05,
                "lower_ci": 1.01,
                "upper_ci": 1.09,
                "p_value": 0.01,
                "p_value_floor_applied_flag": "no",
                "valid_strata_count": 2,
                "observation_count": 8,
                "outcome_sum": 5.0,
                "denominator_sum": 8000.0,
                "overdispersion_ratio": 1.2,
            }
        ]
    )
    models_root = root / "artifacts" / "track_a" / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(models_root / "benchmark_results.csv", index=False)

    sensitivity = pd.DataFrame(
        [
            {
                "public_concept_name": "neonatal_death_count",
                "public_concept_label": "Neonatal death count",
                "outcome_variable": "neonatal_death_count",
                "outcome_label": "Neonatal death count",
                "outcome_family": "acute_daily",
                "stage1_classification": "benchmark_ready",
                "triage_priority_classification": "advance",
                "benchmark_stage_recommendation": "yes",
                "sensitivity_domain": "source_restriction_no_fallback",
                "sensitivity_label": "Restriction to rows without climate fallback imputation",
                "sample_filter": "quality_control_pass_and_valid_offset;any_climate_fallback_imputation_flag=0",
                "final_climate_source": "all_sources",
                "exposure_variable": "maximum_daily_temperature_c_lag0",
                "exposure_label": "Maximum daily temperature (lag 0)",
                "exposure_family_id": "family_1_tmax_primary",
                "exposure_family_name": "Maximum daily temperature",
                "lag_or_window": "lag0",
                "model_label": "poisson_fe_proxy",
                "estimate_scale": "rate_ratio_per_interquartile_range_increase",
                "estimate": 1.03,
                "lower_ci": 1.00,
                "upper_ci": 1.07,
                "p_value": 0.02,
                "p_value_floor_applied_flag": "no",
                "valid_strata_count": 1,
                "observation_count": 4,
                "outcome_sum": 2.0,
                "denominator_sum": 4000.0,
                "overdispersion_ratio": 1.0,
                "reference_estimate": 1.05,
                "reference_lower_ci": 1.01,
                "reference_upper_ci": 1.09,
                "reference_p_value": 0.01,
                "direction_vs_reference": "same_direction",
                "relative_change_pct": -1.9,
                "source_dependence_flag": "no",
            }
        ]
    )
    sensitivity_root = root / "artifacts" / "track_a" / "sensitivity"
    sensitivity_root.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(sensitivity_root / "sensitivity_matrix.csv", index=False)

    heterogeneity = pd.DataFrame(
        [
            {
                "public_concept_name": "neonatal_death_count",
                "public_concept_label": "Neonatal death count",
                "outcome_variable": "neonatal_death_count",
                "outcome_family": "acute_daily",
                "stage1_classification": "benchmark_ready",
                "triage_priority_classification": "advance",
                "benchmark_stage_recommendation": "yes",
                "modifier_name": "coastal_municipality_flag",
                "modifier_label": "Coastal municipality status",
                "modifier_stratum": "inland",
                "exposure_variable": "maximum_daily_temperature_c_lag0",
                "exposure_label": "Maximum daily temperature (lag 0)",
                "lag_or_window": "lag0",
                "model_label": "poisson_fe_proxy",
                "estimate_scale": "rate_ratio_per_interquartile_range_increase",
                "estimate": 1.02,
                "lower_ci": 1.0,
                "upper_ci": 1.05,
                "p_value": 0.05,
                "p_value_floor_applied_flag": "no",
                "valid_strata_count": 1,
                "observation_count": 4,
                "outcome_sum": 2.0,
                "denominator_sum": 4000.0,
                "overdispersion_ratio": 1.0,
                "municipality_count": 1,
                "full_sample_estimate": 1.05,
                "direction_vs_full_sample": "same_direction",
                "heterogeneity_signal_classification": "directionally_consistent",
            },
            {
                "public_concept_name": "neonatal_death_count",
                "public_concept_label": "Neonatal death count",
                "outcome_variable": "neonatal_death_count",
                "outcome_family": "acute_daily",
                "stage1_classification": "benchmark_ready",
                "triage_priority_classification": "advance",
                "benchmark_stage_recommendation": "yes",
                "modifier_name": "coastal_municipality_flag",
                "modifier_label": "Coastal municipality status",
                "modifier_stratum": "coastal",
                "exposure_variable": "maximum_daily_temperature_c_lag0",
                "exposure_label": "Maximum daily temperature (lag 0)",
                "lag_or_window": "lag0",
                "model_label": "poisson_fe_proxy",
                "estimate_scale": "rate_ratio_per_interquartile_range_increase",
                "estimate": 1.08,
                "lower_ci": 1.01,
                "upper_ci": 1.15,
                "p_value": 0.03,
                "p_value_floor_applied_flag": "no",
                "valid_strata_count": 1,
                "observation_count": 4,
                "outcome_sum": 3.0,
                "denominator_sum": 4000.0,
                "overdispersion_ratio": 1.0,
                "municipality_count": 1,
                "full_sample_estimate": 1.05,
                "direction_vs_full_sample": "same_direction",
                "heterogeneity_signal_classification": "directionally_consistent",
            },
        ]
    )
    heterogeneity.to_csv(sensitivity_root / "heterogeneity_results.csv", index=False)

    acute_rows = []
    for municipality_code, fallback, coastal_temp in [("110001", 0, 30.0), ("110002", 1, 33.0)]:
        for day_offset in range(4):
            acute_rows.append(
                {
                    "municipality_code": municipality_code,
                    "observation_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day_offset),
                    "calendar_year": 2024,
                    "quality_control_status": "PASS",
                    "offset_valid_flag": 1,
                    "denominator_zero_flag": 0,
                    "denominator_missing_flag": 0,
                    "rate_not_computable_flag": 0,
                    "final_climate_source": "ERA5_FULL",
                    "any_climate_fallback_imputation_flag": fallback,
                    "annual_live_birth_count": 1000.0,
                    "under_five_population_count": 5000.0,
                    "neonatal_death_count": 1 if day_offset % 2 == 0 else 0,
                    "infant_death_count": 1 if day_offset % 2 == 0 else 0,
                    "under_five_death_count": 1 if day_offset % 3 == 0 else 0,
                    "acute_care_proxy_count": 2 if day_offset % 2 == 0 else 1,
                    **{
                        f"{exposure}_{lag}": coastal_temp + day_offset + index
                        for index, exposure in enumerate(
                            [
                                "maximum_daily_temperature_c",
                                "mean_daily_relative_humidity_pct",
                                "heatwave_duration_days_p95_threshold",
                                "approximate_heat_index_c",
                            ]
                        )
                        for lag in ["lag0", "lag0_1_mean", "lag0_3_mean", "lag0_7_mean"]
                    },
                }
            )
    acute_frame = pd.DataFrame(acute_rows)
    acute_path = root / "artifacts" / "track_a" / "analysis_marts" / "acute_daily_mart" / "YEAR=2024"
    acute_path.mkdir(parents=True, exist_ok=True)
    acute_frame.to_parquet(acute_path / "acute_daily_mart_2024.parquet", index=False)

    track_b_models_root = root / "artifacts" / "track_b" / "models"
    track_b_models_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "outcome_variable": "preterm_birth_flag",
                "outcome_label": "Preterm birth",
                "exposure_variable": "third_trimester_mean_tmax_c",
                "exposure_label": "Third-trimester mean maximum temperature",
                "model_label": "modified_poisson_fe_proxy",
                "estimate_scale": "rate_ratio_per_one_unit_increase",
                "primary_model_flag": "yes",
                "cohort_definition": "live_births_with_valid_gestational_age",
                "model_sequence": 1,
                "estimate": 1.02,
                "lower_ci": 1.01,
                "upper_ci": 1.04,
                "p_value": 0.01,
                "p_value_floor_applied_flag": "no",
                "valid_strata_count": 1,
                "aggregated_row_count": 4,
                "birth_count": 4.0,
                "outcome_count": 2.0,
                "overdispersion_ratio": 1.0,
                "internal_slope": 0.02,
                "internal_slope_se": 0.01,
                "stratum_specification": "municipality_code|calendar_year|month_of_conception|gestational_age_source|ga_exact_measurement_flag",
                "exposure_bin_count": 4,
                "source_sensitive_births": 0.0,
                "p_value_floor_applied_flag_numeric": 0,
                "display_label": "Preterm birth\nThird-trimester mean maximum temperature",
            }
        ]
    ).to_csv(track_b_models_root / "preterm_main_models.csv", index=False)
    pd.DataFrame(
        [
            {
                "outcome_variable": "gestational_days_lost_below_37_weeks",
                "outcome_label": "Gestational days lost below 37 weeks",
                "exposure_variable": "third_trimester_mean_tmax_c",
                "exposure_label": "Third-trimester mean maximum temperature",
                "model_label": "gaussian_fe_proxy",
                "estimate_scale": "mean_difference_per_one_unit_increase",
                "primary_model_flag": "yes",
                "display_label": "Gestational days lost\nThird-trimester mean maximum temperature",
                "estimate": 0.3,
                "lower_ci": 0.1,
                "upper_ci": 0.5,
                "p_value": 0.02,
                "p_value_floor_applied_flag": "no",
                "valid_strata_count": 1,
                "aggregated_row_count": 4,
                "birth_count": 4.0,
                "outcome_total": 20.0,
                "internal_slope": 0.3,
                "internal_slope_se": 0.1,
                "stratum_specification": "municipality_code|calendar_year",
                "exposure_bin_count": 4,
                "p_value_floor_applied_flag_numeric": 0,
            }
        ]
    ).to_csv(track_b_models_root / "gest_days_lost_models.csv", index=False)
    pd.DataFrame(
        [
            {
                "outcome_variable": "stillbirth_primary_gte28_flag",
                "outcome_label": "Stillbirth at or beyond 28 weeks",
                "exposure_variable": "days_0_6_before_event_mean_tmax_c",
                "exposure_label": "Mean maximum temperature during the seven days before the event",
                "model_label": "poisson_fe_proxy_total_birth_offset",
                "estimate_scale": "rate_ratio_per_one_unit_increase",
                "primary_model_flag": "yes",
                "model_sequence": 1,
                "estimate": 1.01,
                "lower_ci": 1.0,
                "upper_ci": 1.03,
                "p_value": 0.04,
                "p_value_floor_applied_flag": "no",
                "valid_strata_count": 1,
                "aggregated_row_count": 4,
                "birth_count": 4.0,
                "outcome_count": 1.0,
                "overdispersion_ratio": 1.0,
                "internal_slope": 0.01,
                "internal_slope_se": 0.01,
                "stratum_specification": "municipality_code|calendar_year|event_month|gestational_age_source",
                "exposure_bin_count": 4,
                "source_sensitive_births": 0.0,
                "p_value_floor_applied_flag_numeric": 0,
                "display_label": "Stillbirth\n7-day tmax",
            }
        ]
    ).to_csv(track_b_models_root / "stillbirth_main_models.csv", index=False)
    pd.DataFrame(columns=["outcome_variable", "outcome_label", "exposure_variable", "exposure_label"]).to_csv(
        track_b_models_root / "stillbirth_sensitivity_models.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "outcome_variable": "neonatal_death_count",
                "outcome_label": "Neonatal mortality",
                "exposure_variable": "lag0_1_mean_tmax_c",
                "exposure_label": "lag0 1 mean tmax c",
                "model_label": "poisson_fe_proxy",
                "estimate_scale": "rate_ratio_per_one_degree_celsius_increase",
                "primary_model_flag": "yes",
                "lag_or_window": "lag0_1_mean_tmax_c",
                "estimate": 1.01,
                "lower_ci": 1.0,
                "upper_ci": 1.03,
                "p_value": 0.03,
                "p_value_floor_applied_flag": "no",
                "valid_strata_count": 4,
                "observation_count": 28,
                "death_count": 2.0,
                "annual_live_birth_denominator_sum": 28000.0,
                "overdispersion_ratio": 1.0,
                "internal_slope": 0.01,
                "internal_slope_se": 0.01,
                "source_sensitive_days": 0.0,
                "p_value_floor_applied_flag_numeric": 0,
            }
        ]
    ).to_csv(track_b_models_root / "neonatal_main_models.csv", index=False)

    event_rows = []
    for municipality_code in ["110001", "110002"]:
        for idx in range(4):
            event_rows.append(
                {
                    "record_type": "live_birth",
                    "municipality_code": municipality_code,
                    "calendar_year": 2024,
                    "month_of_conception": 1,
                    "gestational_age_source": "LMP_EXACT",
                    "ga_exact_measurement_flag": 1,
                    "gestational_age_valid_for_analysis_flag": 1,
                    "preterm_birth_flag": 1 if idx % 2 == 0 else 0,
                    "very_preterm_birth_flag": 1 if idx == 0 else 0,
                    "gestational_days_lost_below_37_weeks": float(idx + 1),
                    "third_trimester_mean_tmax_c": 30.0 + idx,
                    "third_trimester_mean_rh_mean_pct": 70.0 + idx,
                    "third_trimester_mean_heat_index_c": 33.0 + idx,
                    "third_trimester_heatwave_days": float(idx),
                    "third_trimester_full_window_observed_flag": 1,
                    "third_trimester_fallback_day_share": 0.0 if municipality_code == "110001" else 0.2,
                    "days_0_27_before_event_mean_tmax_c": 29.0 + idx,
                    "days_0_27_before_event_full_window_observed_flag": 1,
                    "days_0_27_before_event_fallback_day_share": 0.0,
                    "whole_pregnancy_mean_tmax_c": 28.0 + idx,
                    "whole_pregnancy_full_window_observed_flag": 1,
                    "whole_pregnancy_fallback_day_share": 0.0,
                    "event_date": str(pd.Timestamp("2024-04-01") + pd.Timedelta(days=idx)),
                    "gestational_age_weeks_final": 30.0 + idx,
                    "stillbirth_primary_gte28_flag": 1 if idx == 0 else 0,
                    "stillbirth_sensitivity_gte22_flag": 1 if idx == 0 else 0,
                    "days_0_6_before_event_mean_tmax_c": 31.0 + idx,
                    "days_0_6_before_event_mean_heat_index_c": 34.0 + idx,
                    "days_0_6_before_event_full_window_observed_flag": 1,
                    "days_0_6_before_event_fallback_day_share": 0.0 if municipality_code == "110001" else 0.2,
                    "days_0_13_before_event_mean_tmax_c": 30.5 + idx,
                    "days_0_13_before_event_full_window_observed_flag": 1,
                    "days_0_27_before_event_mean_tmax_c": 29.5 + idx,
                    "days_0_27_before_event_full_window_observed_flag": 1,
                }
            )
    event_root = root / "artifacts" / "track_b" / "event_level" / "perinatal_window_exposure_matrix.parquet" / "YEAR=2024"
    event_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(event_rows).to_parquet(event_root / "perinatal_window_exposure_matrix_2024.parquet", index=False)

    mortality_rows = []
    start = pd.Timestamp("2024-01-01")
    for municipality_code, offset in [("110001", 0.0), ("110002", 1.0)]:
        for day_offset in range(28):
            current_date = start + pd.Timedelta(days=day_offset)
            mortality_rows.append(
                {
                    "MUNICIPALITY_CODE": municipality_code,
                    "DATE": str(current_date.date()),
                    "YEAR": 2024,
                    "MONTH": current_date.month,
                    "DEATHS_NEONATAL": 1 if day_offset % 9 == 0 else 0,
                    "LIVE_BIRTHS_TOTAL_ANNUAL": 1000,
                    "TMAX_C": 30.0 + offset + (day_offset % 4),
                    "RH_MEAN_PCT": 70.0,
                    "FONTE_CLIMA_FINAL": "ERA5_FULL",
                    "IMPUTADO_POR_FALLBACK": 0 if municipality_code == "110001" else 1,
                    "OFFSET_VALID_FLAG": 1,
                    "DENOMINATOR_ZERO_FLAG": 0,
                    "DENOMINATOR_MISSING_FLAG": 0,
                    "RATE_NOT_COMPUTABLE_FLAG": 0,
                    "QC_STATUS": "PASS",
                }
            )
    mortality_root = root / "data" / "gold" / "gold_mnch_mortality" / "YEAR=2024"
    mortality_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mortality_rows).to_parquet(mortality_root / "gold_mnch_mortality_2024.parquet", index=False)


@pytest.fixture
def statcalc_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    storage_dir = tmp_path / "storage"
    _build_statcalc_fixture(root)
    return root, storage_dir


def test_statcalc_metadata_and_reference_run(statcalc_fixture: tuple[Path, Path]) -> None:
    root, storage_dir = statcalc_fixture
    client = TestClient(create_app(root=root, storage_dir=storage_dir, start_worker=False))

    recipes_response = client.get("/api/v1/statcalc/meta/recipes")
    assert recipes_response.status_code == 200
    recipes = recipes_response.json()
    assert any(recipe["recipe_id"] == "track_a_acute_fe_proxy" for recipe in recipes)
    assert any(recipe["recipe_id"] == "track_b_preterm_fe_proxy" for recipe in recipes)

    run_response = client.post(
        "/api/v1/statcalc/runs",
        json={
            "recipe_id": "track_a_acute_fe_proxy",
            "outcome_variable": "neonatal_death_count",
            "exposure_variable": "maximum_daily_temperature_c",
            "lag_or_window": "lag0",
            "template_id": "track_a_benchmark_proxy",
            "source_filter": "all_sources",
            "grouping_template": "track_a_municipality_year",
            "adjustment_context_variables": [],
        },
    )
    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["execution_mode"] == "reference"
    assert run_payload["status"] == "completed"

    result_response = client.get(f"/api/v1/statcalc/runs/{run_payload['run_id']}/result")
    assert result_response.status_code == 200
    result_payload = result_response.json()
    assert result_payload["summary"]["reference_artifact"] == "artifacts/track_a/models/benchmark_results.csv"
    assert result_payload["rows"][0]["estimate"] == pytest.approx(1.05)


def test_statcalc_track_a_live_and_presets(statcalc_fixture: tuple[Path, Path]) -> None:
    root, storage_dir = statcalc_fixture
    client = TestClient(create_app(root=root, storage_dir=storage_dir, start_worker=False))

    run_response = client.post(
        "/api/v1/statcalc/runs",
        json={
            "recipe_id": "track_a_acute_fe_proxy",
            "outcome_variable": "neonatal_death_count",
            "exposure_variable": "mean_daily_relative_humidity_pct",
            "lag_or_window": "lag0",
            "template_id": "track_a_benchmark_proxy",
            "stratifier_id": "coastal_municipality_flag",
            "source_filter": "all_sources",
            "grouping_template": "track_a_municipality_year",
            "adjustment_context_variables": [],
        },
    )
    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["execution_mode"] == "live"
    assert run_payload["status"] in {"completed", "completed_with_warnings"}

    result_response = client.get(f"/api/v1/statcalc/runs/{run_payload['run_id']}/result")
    assert result_response.status_code == 200
    result_payload = result_response.json()
    assert len(result_payload["rows"]) >= 1

    preset_response = client.post(
        "/api/v1/statcalc/presets",
        json={
            "name": "Humidity stratified",
            "description": "Track A live preset",
            "run_spec": {
                "recipe_id": "track_a_acute_fe_proxy",
                "outcome_variable": "neonatal_death_count",
                "exposure_variable": "mean_daily_relative_humidity_pct",
                "lag_or_window": "lag0",
                "template_id": "track_a_benchmark_proxy",
                "stratifier_id": "coastal_municipality_flag",
                "source_filter": "all_sources",
                "grouping_template": "track_a_municipality_year",
                "adjustment_context_variables": [],
            },
        },
    )
    assert preset_response.status_code == 200
    preset_payload = preset_response.json()

    delete_response = client.delete(f"/api/v1/statcalc/presets/{preset_payload['preset_id']}")
    assert delete_response.status_code == 200


def test_statcalc_track_b_async_job(statcalc_fixture: tuple[Path, Path]) -> None:
    root, storage_dir = statcalc_fixture
    with TestClient(create_app(root=root, storage_dir=storage_dir, start_worker=True)) as client:
        run_response = client.post(
            "/api/v1/statcalc/runs",
            json={
                "recipe_id": "track_b_preterm_fe_proxy",
                "outcome_variable": "preterm_birth_flag",
                "exposure_variable": "third_trimester_mean_tmax_c",
                "template_id": "track_b_flagship_grouped_fe",
                "stratifier_id": "coastal_municipality_flag",
                "source_filter": "all_sources",
                "grouping_template": "track_b_flagship_default",
                "gestational_source_mode": "exact_only",
                "adjustment_context_variables": [],
            },
        )
        assert run_response.status_code == 200
        run_payload = run_response.json()
        assert run_payload["status"] == "queued"

        deadline = time.time() + 10
        final_run = run_payload
        while time.time() < deadline:
            current = client.get(f"/api/v1/statcalc/runs/{run_payload['run_id']}").json()
            if current["status"] not in {"queued", "running"}:
                final_run = current
                break
            time.sleep(0.2)

        assert final_run["status"] in {"completed", "completed_with_warnings"}
        result_response = client.get(f"/api/v1/statcalc/runs/{run_payload['run_id']}/result")
        assert result_response.status_code == 200
        result_payload = result_response.json()
        assert result_payload["execution_mode"] == "live"
        assert result_payload["rows"]


def test_statcalc_rejects_context_adjustment(statcalc_fixture: tuple[Path, Path]) -> None:
    root, storage_dir = statcalc_fixture
    client = TestClient(create_app(root=root, storage_dir=storage_dir, start_worker=False))
    response = client.post(
        "/api/v1/statcalc/runs",
        json={
            "recipe_id": "track_a_acute_fe_proxy",
            "outcome_variable": "neonatal_death_count",
            "exposure_variable": "maximum_daily_temperature_c",
            "lag_or_window": "lag0",
            "template_id": "track_a_benchmark_proxy",
            "source_filter": "all_sources",
            "grouping_template": "track_a_municipality_year",
            "adjustment_context_variables": ["coastal_municipality_flag"],
        },
    )
    assert response.status_code == 400
    assert "restricted to stratification" in response.json()["detail"]
