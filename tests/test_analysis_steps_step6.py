from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from heat_mnch.analysis_steps.step6 import build_step6_absolute_burden
from heat_mnch.cli import main


def _write_json_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_round_config(root: Path) -> None:
    _write_json_yaml(
        root / "config" / "analysis" / "round_config.yaml",
        {
            "analysis_id": "heat_mnch_incremental_round",
            "analysis_title": "HEAT-MNCH Incremental Analysis Round",
            "years": [2024],
        },
    )
    _write_json_yaml(
        root / "config" / "analysis" / "step6_burden.yaml",
        {
            "step6_spec": {
                "step_name": "step6_absolute_burden_translation",
                "contrast_operator_by_outcome_family": {
                    "acute_daily": "acute_p95_vs_reference",
                    "maternal_operational": "q5_vs_q1",
                    "perinatal_delivery_window_aggregate": "q5_vs_q1",
                },
                "publication_thresholds": {
                    "national_min_exposed_outcome_sum": 1,
                    "national_min_valid_strata": 1,
                    "macroregion_min_exposed_outcome_sum": 1,
                    "state_min_exposed_outcome_sum": 10,
                },
                "headline_group_definitions": [
                    {"group_id": "maternal", "group_label": "Maternal"},
                    {"group_id": "perinatal_nascimento", "group_label": "Perinatal and Birth"},
                    {"group_id": "mortalidade_infantil", "group_label": "Child Mortality"},
                    {"group_id": "hospitalizacao_infantil", "group_label": "Child Hospitalization"},
                ],
                "secondary_reason_labels": {
                    "service_or_non_harmful_proxy": "Service use or proxy.",
                    "continuous_severity_non_event": "Continuous severity.",
                    "duplicate_or_alias": "Duplicate.",
                    "overlap_intersection": "Overlap.",
                    "coding_specific_composite": "Coding.",
                    "failed_public_burden_gate": "Failed gate.",
                },
                "json_contract": [],
                "output_workbook_tabs": [
                    "Headlines",
                    "Supplementary",
                    "Secondary",
                    "Exclusions",
                    "States",
                    "Macroregions",
                    "MunicipalitySupport",
                ],
            }
        },
    )
    _write_json_yaml(
        root / "config" / "analysis" / "step6_outcome_translation_catalog.yaml",
        {
            "analysis_units": [
                {
                    "analysis_unit_id": "hypertensive_disorder_of_pregnancy_count",
                    "group_id": "maternal",
                    "group_label": "Maternal",
                    "translation_track": "headline_order_candidate",
                    "headline_priority": 1,
                    "secondary_reason": None,
                    "overlap_class": "canonical",
                    "headline_allowed_flag": True,
                    "supplementary_absolute_allowed_flag": True,
                },
                {
                    "analysis_unit_id": "preterm_birth_count",
                    "group_id": "perinatal_nascimento",
                    "group_label": "Perinatal and Birth",
                    "translation_track": "headline_order_candidate",
                    "headline_priority": 1,
                    "secondary_reason": None,
                    "overlap_class": "canonical",
                    "headline_allowed_flag": True,
                    "supplementary_absolute_allowed_flag": True,
                },
                {
                    "analysis_unit_id": "gestational_days_lost_below_37_weeks",
                    "group_id": "perinatal_nascimento",
                    "group_label": "Perinatal and Birth",
                    "translation_track": "secondary_parallel_only",
                    "headline_priority": None,
                    "secondary_reason": "continuous_severity_non_event",
                    "overlap_class": "continuous",
                    "headline_allowed_flag": False,
                    "supplementary_absolute_allowed_flag": False,
                },
                {
                    "analysis_unit_id": "prenatal_eight_plus_visit_count",
                    "group_id": "perinatal_nascimento",
                    "group_label": "Perinatal and Birth",
                    "translation_track": "secondary_parallel_only",
                    "headline_priority": None,
                    "secondary_reason": "service_or_non_harmful_proxy",
                    "overlap_class": "service_use",
                    "headline_allowed_flag": False,
                    "supplementary_absolute_allowed_flag": False,
                },
                {
                    "analysis_unit_id": "under_five_death_count",
                    "group_id": "mortalidade_infantil",
                    "group_label": "Child Mortality",
                    "translation_track": "headline_order_candidate",
                    "headline_priority": 1,
                    "secondary_reason": None,
                    "overlap_class": "canonical",
                    "headline_allowed_flag": True,
                    "supplementary_absolute_allowed_flag": True,
                },
                {
                    "analysis_unit_id": "child_dehydration_admission_count",
                    "group_id": "hospitalizacao_infantil",
                    "group_label": "Child Hospitalization",
                    "translation_track": "headline_order_candidate",
                    "headline_priority": 1,
                    "secondary_reason": None,
                    "overlap_class": "canonical",
                    "headline_allowed_flag": True,
                    "supplementary_absolute_allowed_flag": True,
                },
            ]
        },
    )


def _write_step_artifacts(root: Path) -> None:
    step0_root = root / "artifacts" / "analysis" / "step0"
    step1_root = root / "artifacts" / "analysis" / "step1"
    step3_root = root / "artifacts" / "analysis" / "step3"
    step4_root = root / "artifacts" / "analysis" / "step4"
    step5_root = root / "artifacts" / "analysis" / "step5"
    step0_root.mkdir(parents=True, exist_ok=True)
    step1_root.mkdir(parents=True, exist_ok=True)
    step3_root.mkdir(parents=True, exist_ok=True)
    step4_root.mkdir(parents=True, exist_ok=True)
    step5_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"outcome_id": "HDP_ANY", "public_label": "Hypertensive disorder of pregnancy"},
            {"outcome_id": "PRETERM_FLAG", "public_label": "Preterm birth"},
            {"outcome_id": "GEST_DAYS_LOST_BELOW_37W", "public_label": "Gestational days lost below 37 weeks"},
            {"outcome_id": "PRENATAL_VISITS_8PLUS_FLAG", "public_label": "Prenatal visits 8+"},
            {"outcome_id": "DEATHS_U5", "public_label": "Under-five deaths"},
            {"outcome_id": "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL", "public_label": "Child dehydration admission"},
        ]
    ).to_csv(step0_root / "frozen_outcome_inventory.csv", index=False)

    pd.DataFrame(
        [
            {
                "analysis_unit_id": "hypertensive_disorder_of_pregnancy_count",
                "linked_outcome_ids": "HDP_ANY",
                "linked_public_labels": "Hypertensive disorder of pregnancy",
                "denominator_id": "women_15_49_population",
                "denominator_column": "women_age_15_49_population_count",
                "seasonality_profile_id": "maternal_daily_standard",
                "mart_name": "maternal_operational_mart",
            },
            {
                "analysis_unit_id": "preterm_birth_count",
                "linked_outcome_ids": "PRETERM_FLAG",
                "linked_public_labels": "Preterm birth",
                "denominator_id": "annual_live_births",
                "denominator_column": "annual_live_birth_count",
                "seasonality_profile_id": "perinatal_delivery_aligned",
                "mart_name": "perinatal_delivery_window_mart",
            },
            {
                "analysis_unit_id": "gestational_days_lost_below_37_weeks",
                "linked_outcome_ids": "GEST_DAYS_LOST_BELOW_37W",
                "linked_public_labels": "Gestational days lost below 37 weeks",
                "denominator_id": "annual_live_births",
                "denominator_column": "daily_live_birth_count",
                "seasonality_profile_id": "perinatal_delivery_aligned",
                "mart_name": "perinatal_delivery_window_mart",
            },
            {
                "analysis_unit_id": "prenatal_eight_plus_visit_count",
                "linked_outcome_ids": "PRENATAL_VISITS_8PLUS_FLAG",
                "linked_public_labels": "Prenatal visits 8+",
                "denominator_id": "annual_live_births",
                "denominator_column": "annual_live_birth_count",
                "seasonality_profile_id": "perinatal_delivery_aligned",
                "mart_name": "perinatal_delivery_window_mart",
            },
            {
                "analysis_unit_id": "under_five_death_count",
                "linked_outcome_ids": "DEATHS_U5",
                "linked_public_labels": "Under-five deaths",
                "denominator_id": "under_five_population",
                "denominator_column": "under_five_population_count",
                "seasonality_profile_id": "acute_daily_standard",
                "mart_name": "acute_daily_mart",
            },
            {
                "analysis_unit_id": "child_dehydration_admission_count",
                "linked_outcome_ids": "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL",
                "linked_public_labels": "Child dehydration admission",
                "denominator_id": "under_five_population",
                "denominator_column": "under_five_population_count",
                "seasonality_profile_id": "acute_daily_standard",
                "mart_name": "acute_daily_mart",
            },
        ]
    ).to_csv(step1_root / "analysis_unit_roster.csv", index=False)

    pd.DataFrame(
        [
            {
                "analysis_unit_id": "hypertensive_disorder_of_pregnancy_count",
                "candidate_track": "main_round",
                "analysis_role": "primary",
                "outcome_family": "maternal_operational",
                "exposure_family_id": "family_4_heat_index_parallel",
                "exposure_family_name": "Approximate heat index",
                "lag_or_window": "0_6",
                "screening_domain": "maternal_operational_window_q5_vs_q1",
                "screening_method": "within_municipality_year_quintile_extremes",
                "estimate": 1.8,
                "lower_ci": 1.2,
                "upper_ci": 2.4,
                "p_value": 0.01,
                "valid_strata_count": 3,
                "exposed_outcome_sum": 9.0,
                "reference_outcome_sum": 3.0,
                "exposed_denominator_sum": 300.0,
                "reference_denominator_sum": 300.0,
            },
            {
                "analysis_unit_id": "preterm_birth_count",
                "candidate_track": "main_round",
                "analysis_role": "primary",
                "outcome_family": "perinatal_delivery_window_aggregate",
                "exposure_family_id": "family_4_heat_index_parallel",
                "exposure_family_name": "Approximate heat index",
                "lag_or_window": "0_6",
                "screening_domain": "perinatal_delivery_window_q5_vs_q1",
                "screening_method": "within_municipality_year_quintile_extremes",
                "estimate": 1.6,
                "lower_ci": 1.1,
                "upper_ci": 2.2,
                "p_value": 0.02,
                "valid_strata_count": 3,
                "exposed_outcome_sum": 8.0,
                "reference_outcome_sum": 4.0,
                "exposed_denominator_sum": 300.0,
                "reference_denominator_sum": 300.0,
            },
            {
                "analysis_unit_id": "gestational_days_lost_below_37_weeks",
                "candidate_track": "main_round",
                "analysis_role": "sensitivity",
                "outcome_family": "perinatal_delivery_window_aggregate",
                "exposure_family_id": "family_4_heat_index_parallel",
                "exposure_family_name": "Approximate heat index",
                "lag_or_window": "0_6",
                "screening_domain": "perinatal_delivery_window_q5_vs_q1",
                "screening_method": "within_municipality_year_quintile_extremes",
                "estimate": 0.4,
                "lower_ci": 0.2,
                "upper_ci": 0.6,
                "p_value": 0.01,
                "valid_strata_count": 3,
                "exposed_outcome_sum": 15.0,
                "reference_outcome_sum": 6.0,
                "exposed_denominator_sum": 300.0,
                "reference_denominator_sum": 300.0,
            },
            {
                "analysis_unit_id": "prenatal_eight_plus_visit_count",
                "candidate_track": "main_round",
                "analysis_role": "descriptive_only",
                "outcome_family": "perinatal_delivery_window_aggregate",
                "exposure_family_id": "family_4_heat_index_parallel",
                "exposure_family_name": "Approximate heat index",
                "lag_or_window": "0_6",
                "screening_domain": "perinatal_delivery_window_q5_vs_q1",
                "screening_method": "within_municipality_year_quintile_extremes",
                "estimate": 0.8,
                "lower_ci": 0.6,
                "upper_ci": 1.0,
                "p_value": 0.06,
                "valid_strata_count": 3,
                "exposed_outcome_sum": 12.0,
                "reference_outcome_sum": 15.0,
                "exposed_denominator_sum": 300.0,
                "reference_denominator_sum": 300.0,
            },
            {
                "analysis_unit_id": "under_five_death_count",
                "candidate_track": "main_round",
                "analysis_role": "primary",
                "outcome_family": "acute_daily",
                "exposure_family_id": "family_4_heat_index_parallel",
                "exposure_family_name": "Approximate heat index",
                "lag_or_window": "lag0",
                "screening_domain": "acute_p95_vs_reference",
                "screening_method": "within_municipality_year_p95_vs_reference",
                "estimate": 1.9,
                "lower_ci": 1.3,
                "upper_ci": 2.5,
                "p_value": 0.01,
                "valid_strata_count": 3,
                "exposed_outcome_sum": 6.0,
                "reference_outcome_sum": 2.0,
                "exposed_denominator_sum": 300.0,
                "reference_denominator_sum": 300.0,
            },
            {
                "analysis_unit_id": "child_dehydration_admission_count",
                "candidate_track": "main_round",
                "analysis_role": "primary",
                "outcome_family": "acute_daily",
                "exposure_family_id": "family_4_heat_index_parallel",
                "exposure_family_name": "Approximate heat index",
                "lag_or_window": "lag0",
                "screening_domain": "acute_p95_vs_reference",
                "screening_method": "within_municipality_year_p95_vs_reference",
                "estimate": 2.1,
                "lower_ci": 1.4,
                "upper_ci": 2.8,
                "p_value": 0.01,
                "valid_strata_count": 3,
                "exposed_outcome_sum": 7.0,
                "reference_outcome_sum": 2.0,
                "exposed_denominator_sum": 300.0,
                "reference_denominator_sum": 300.0,
            },
        ]
    ).to_csv(step3_root / "step3_screening_results.csv", index=False)

    pd.DataFrame(
        [
            {
                "analysis_unit_id": "hypertensive_disorder_of_pregnancy_count",
                "candidate_track": "main_round",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "preferred_lag_or_window": "0_6",
            },
            {
                "analysis_unit_id": "preterm_birth_count",
                "candidate_track": "main_round",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "preferred_lag_or_window": "0_6",
            },
            {
                "analysis_unit_id": "gestational_days_lost_below_37_weeks",
                "candidate_track": "main_round",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "preferred_lag_or_window": "0_6",
            },
            {
                "analysis_unit_id": "prenatal_eight_plus_visit_count",
                "candidate_track": "main_round",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "preferred_lag_or_window": "0_6",
            },
            {
                "analysis_unit_id": "under_five_death_count",
                "candidate_track": "main_round",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "preferred_lag_or_window": "lag0",
            },
            {
                "analysis_unit_id": "child_dehydration_admission_count",
                "candidate_track": "main_round",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "preferred_lag_or_window": "lag0",
            },
        ]
    ).to_csv(step4_root / "step4_promotion_decisions.csv", index=False)

    pd.DataFrame(
        [
            {
                "analysis_unit_id": "hypertensive_disorder_of_pregnancy_count",
                "linked_outcome_ids": "HDP_ANY",
                "candidate_track": "main_round",
                "analysis_role": "primary",
                "outcome_family": "maternal_operational",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "selected_exposure_family_name": "Approximate heat index",
                "selected_exposure_variable": "approximate_heat_index_c_days_0_6_before_event_mean",
                "selected_exposure_label": "Heat index 0_6",
                "preferred_lag_or_window": "0_6",
                "seasonality_profile_id": "maternal_daily_standard",
                "seasonality_controls_applied": "calendar_year + calendar_month + day_of_week",
                "mart_name": "maternal_operational_mart",
                "denominator_id": "women_15_49_population",
                "denominator_column": "women_age_15_49_population_count",
                "estimate_scale": "rate_ratio_per_interquartile_range_increase",
                "has_any_significant_reversal_flag": False,
                "candidate_ready_for_step6_flag": True,
                "next_step_route": "step6_absolute_burden_translation",
                "decision_reason": "promoted_to_step6_consistent",
            },
            {
                "analysis_unit_id": "preterm_birth_count",
                "linked_outcome_ids": "PRETERM_FLAG",
                "candidate_track": "main_round",
                "analysis_role": "primary",
                "outcome_family": "perinatal_delivery_window_aggregate",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "selected_exposure_family_name": "Approximate heat index",
                "selected_exposure_variable": "approximate_heat_index_c_days_0_6_before_event_mean",
                "selected_exposure_label": "Heat index 0_6",
                "preferred_lag_or_window": "0_6",
                "seasonality_profile_id": "perinatal_delivery_aligned",
                "seasonality_controls_applied": "calendar_year + event_or_delivery_month",
                "mart_name": "perinatal_delivery_window_mart",
                "denominator_id": "annual_live_births",
                "denominator_column": "annual_live_birth_count",
                "estimate_scale": "rate_ratio_per_interquartile_range_increase",
                "has_any_significant_reversal_flag": False,
                "candidate_ready_for_step6_flag": True,
                "next_step_route": "step6_absolute_burden_translation",
                "decision_reason": "promoted_to_step6_consistent",
            },
            {
                "analysis_unit_id": "gestational_days_lost_below_37_weeks",
                "linked_outcome_ids": "GEST_DAYS_LOST_BELOW_37W",
                "candidate_track": "main_round",
                "analysis_role": "sensitivity",
                "outcome_family": "perinatal_delivery_window_aggregate",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "selected_exposure_family_name": "Approximate heat index",
                "selected_exposure_variable": "approximate_heat_index_c_days_0_6_before_event_mean",
                "selected_exposure_label": "Heat index 0_6",
                "preferred_lag_or_window": "0_6",
                "seasonality_profile_id": "perinatal_delivery_aligned",
                "seasonality_controls_applied": "calendar_year + event_or_delivery_month",
                "mart_name": "perinatal_delivery_window_mart",
                "denominator_id": "annual_live_births",
                "denominator_column": "daily_live_birth_count",
                "estimate_scale": "mean_difference_per_interquartile_range_increase",
                "has_any_significant_reversal_flag": False,
                "candidate_ready_for_step6_flag": True,
                "next_step_route": "step6_absolute_burden_translation",
                "decision_reason": "promoted_to_step6_consistent",
            },
            {
                "analysis_unit_id": "prenatal_eight_plus_visit_count",
                "linked_outcome_ids": "PRENATAL_VISITS_8PLUS_FLAG",
                "candidate_track": "main_round",
                "analysis_role": "descriptive_only",
                "outcome_family": "perinatal_delivery_window_aggregate",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "selected_exposure_family_name": "Approximate heat index",
                "selected_exposure_variable": "approximate_heat_index_c_days_0_6_before_event_mean",
                "selected_exposure_label": "Heat index 0_6",
                "preferred_lag_or_window": "0_6",
                "seasonality_profile_id": "perinatal_delivery_aligned",
                "seasonality_controls_applied": "calendar_year + event_or_delivery_month",
                "mart_name": "perinatal_delivery_window_mart",
                "denominator_id": "annual_live_births",
                "denominator_column": "annual_live_birth_count",
                "estimate_scale": "rate_ratio_per_interquartile_range_increase",
                "has_any_significant_reversal_flag": False,
                "candidate_ready_for_step6_flag": True,
                "next_step_route": "step6_absolute_burden_translation",
                "decision_reason": "promoted_to_step6_consistent",
            },
            {
                "analysis_unit_id": "under_five_death_count",
                "linked_outcome_ids": "DEATHS_U5",
                "candidate_track": "main_round",
                "analysis_role": "primary",
                "outcome_family": "acute_daily",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "selected_exposure_family_name": "Approximate heat index",
                "selected_exposure_variable": "approximate_heat_index_c_lag0",
                "selected_exposure_label": "Heat index lag0",
                "preferred_lag_or_window": "lag0",
                "seasonality_profile_id": "acute_daily_standard",
                "seasonality_controls_applied": "calendar_year + calendar_month + day_of_week",
                "mart_name": "acute_daily_mart",
                "denominator_id": "under_five_population",
                "denominator_column": "under_five_population_count",
                "estimate_scale": "rate_ratio_per_interquartile_range_increase",
                "has_any_significant_reversal_flag": False,
                "candidate_ready_for_step6_flag": True,
                "next_step_route": "step6_absolute_burden_translation",
                "decision_reason": "promoted_to_step6_consistent",
            },
            {
                "analysis_unit_id": "child_dehydration_admission_count",
                "linked_outcome_ids": "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL",
                "candidate_track": "main_round",
                "analysis_role": "primary",
                "outcome_family": "acute_daily",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "selected_exposure_family_name": "Approximate heat index",
                "selected_exposure_variable": "approximate_heat_index_c_lag0",
                "selected_exposure_label": "Heat index lag0",
                "preferred_lag_or_window": "lag0",
                "seasonality_profile_id": "acute_daily_standard",
                "seasonality_controls_applied": "calendar_year + calendar_month + day_of_week",
                "mart_name": "acute_daily_mart",
                "denominator_id": "under_five_population",
                "denominator_column": "under_five_population_count",
                "estimate_scale": "rate_ratio_per_interquartile_range_increase",
                "has_any_significant_reversal_flag": True,
                "candidate_ready_for_step6_flag": True,
                "next_step_route": "step6_absolute_burden_translation",
                "decision_reason": "promoted_to_step6_with_significant_reversal",
            },
        ]
    ).to_csv(step5_root / "step5_scorecard.csv", index=False)


def _write_reference_lookup(root: Path) -> None:
    path = root / "artifacts" / "reference" / "municipality_context" / "atlas_brasil_municipality_lookup.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"atlas_id": 1, "MUNICIPALITY_CODE_IBGE7": "1100015", "UF": "RO", "NM_MUNICIPIO": "Alta Floresta"},
            {"atlas_id": 2, "MUNICIPALITY_CODE_IBGE7": "1200013", "UF": "AC", "NM_MUNICIPIO": "Acrelandia"},
            {"atlas_id": 3, "MUNICIPALITY_CODE_IBGE7": "1200054", "UF": "AC", "NM_MUNICIPIO": "Assis Brasil"},
        ]
    ).to_csv(path, index=False)


def _write_marts(root: Path) -> None:
    mart_root = root / "artifacts" / "analysis" / "shared_marts"
    mart_root.mkdir(parents=True, exist_ok=True)

    acute_rows = []
    for municipality_code, offset in [("110001", 0.0), ("120001", 1.0), ("120005", 2.0)]:
        for day in range(1, 11):
            acute_rows.append(
                {
                    "municipality_code": municipality_code,
                    "observation_date": f"2024-01-{day:02d}",
                    "calendar_year": 2024,
                    "quality_control_status": "PASS",
                    "offset_valid_flag": 1,
                    "denominator_zero_flag": 0,
                    "denominator_missing_flag": 0,
                    "rate_not_computable_flag": 0,
                    "under_five_population_count": 100.0,
                    "approximate_heat_index_c_lag0": float(day) + offset,
                    "under_five_death_count": 3.0 if day >= 10 else (1.0 if 4 <= day <= 6 else 0.0),
                    "child_dehydration_admission_count": 4.0 if day >= 10 else (1.0 if 4 <= day <= 6 else 0.0),
                }
            )
    acute_frame = pd.DataFrame(acute_rows)
    acute_path = mart_root / "acute_daily_mart" / "YEAR=2024" / "acute_daily_mart_2024.parquet"
    acute_path.parent.mkdir(parents=True, exist_ok=True)
    acute_frame.to_parquet(acute_path, index=False)

    window_rows = []
    for municipality_code, offset in [("110001", 0.0), ("120001", 1.0), ("120005", 2.0)]:
        for rank in range(1, 6):
            window_rows.append(
                {
                    "municipality_code": municipality_code,
                    "observation_date": f"2024-02-{rank:02d}",
                    "calendar_year": 2024,
                    "quality_control_status": "PASS",
                    "offset_valid_flag": 1,
                    "denominator_zero_flag": 0,
                    "denominator_missing_flag": 0,
                    "rate_not_computable_flag": 0,
                    "annual_live_birth_count": 100.0,
                    "daily_live_birth_count": 100.0,
                    "women_age_15_49_population_count": 100.0,
                    "approximate_heat_index_c_days_0_6_before_event_mean": float(rank) + offset,
                    "preterm_birth_count": 3.0 if rank == 5 else (1.0 if rank == 1 else 0.0),
                    "gestational_days_lost_below_37_weeks": 2.0 if rank == 5 else (0.5 if rank == 1 else 0.0),
                    "prenatal_eight_plus_visit_count": 1.0 if rank == 5 else (2.0 if rank == 1 else 0.0),
                    "hypertensive_disorder_of_pregnancy_count": 4.0 if rank == 5 else (1.0 if rank == 1 else 0.0),
                }
            )
    window_frame = pd.DataFrame(window_rows)
    perinatal_path = mart_root / "perinatal_delivery_window_mart" / "YEAR=2024" / "perinatal_delivery_window_mart_2024.parquet"
    perinatal_path.parent.mkdir(parents=True, exist_ok=True)
    window_frame.to_parquet(perinatal_path, index=False)
    maternal_path = mart_root / "maternal_operational_mart" / "YEAR=2024" / "maternal_operational_mart_2024.parquet"
    maternal_path.parent.mkdir(parents=True, exist_ok=True)
    window_frame.to_parquet(maternal_path, index=False)


def _prepare_step6_fixture(root: Path) -> None:
    _write_round_config(root)
    _write_step_artifacts(root)
    _write_reference_lookup(root)
    _write_marts(root)


def test_build_step6_absolute_burden_writes_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _prepare_step6_fixture(root)

        report = build_step6_absolute_burden(root)

        assert report["status"] == "PASS"
        assert report["step"] == "step6_absolute_burden_translation"
        assert report["summary"]["candidate_count"] == 6
        assert report["summary"]["headline_group_count"] == 4
        assert report["summary"]["headline_group_filled_count"] == 4
        assert report["summary"]["headline_absolute_count"] == 4
        assert report["summary"]["secondary_parallel_count"] == 2
        assert report["gates"]["upstream_step5_pass"] is True
        assert report["gates"]["candidate_catalog_complete"] is True
        assert report["gates"]["step3_anchor_resolved_for_all_candidates"] is True
        assert report["gates"]["step4_alignment_verified"] is True
        assert report["gates"]["headline_selected_for_all_groups"] is True
        assert (root / report["outputs"]["headline_burden_estimates"]).exists()
        assert (root / report["outputs"]["supplementary_burden_estimates"]).exists()
        assert (root / report["outputs"]["secondary_parallel_translation"]).exists()
        assert (root / report["outputs"]["step6_burden_translation_xlsx"]).exists()

        group_headlines = pd.read_csv(root / report["outputs"]["step6_group_headlines"])
        headline_rows = pd.read_csv(root / report["outputs"]["headline_burden_estimates"])
        secondary_rows = pd.read_csv(root / report["outputs"]["secondary_parallel_translation"])
        state_rows = pd.read_csv(root / report["outputs"]["state_burden_publication_gate"])
        municipality_rows = pd.read_csv(root / report["outputs"]["municipality_contribution_detail"])

        assert group_headlines["headline_selected_flag"].fillna(False).astype(bool).all()
        assert set(group_headlines["headline_analysis_unit_id"]) == {
            "hypertensive_disorder_of_pregnancy_count",
            "preterm_birth_count",
            "under_five_death_count",
            "child_dehydration_admission_count",
        }
        assert set(headline_rows["geography_level"]) == {"national", "macroregion", "state"}
        assert secondary_rows["analysis_unit_id"].astype(str).isin(
            {"gestational_days_lost_below_37_weeks", "prenatal_eight_plus_visit_count"}
        ).all()
        assert secondary_rows.loc[
            secondary_rows["analysis_unit_id"].astype(str).eq("gestational_days_lost_below_37_weeks"),
            "recommended_public_use",
        ].iloc[0] == "continuous_internal_translation_only"
        assert secondary_rows.loc[
            secondary_rows["analysis_unit_id"].astype(str).eq("prenatal_eight_plus_visit_count"),
            "recommended_public_use",
        ].iloc[0] == "relative_only"
        assert state_rows["publishable_flag"].fillna(False).astype(bool).eq(False).all()
        assert municipality_rows["headline_analysis_unit_id"].nunique() == 4


def test_build_step6_absolute_burden_fails_when_group_has_no_headline() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _prepare_step6_fixture(root)
        step3_path = root / "artifacts" / "analysis" / "step3" / "step3_screening_results.csv"
        step3 = pd.read_csv(step3_path)
        mask = step3["analysis_unit_id"].astype(str).eq("child_dehydration_admission_count")
        step3.loc[mask, "p_value"] = 0.2
        step3.to_csv(step3_path, index=False)

        report = build_step6_absolute_burden(root)

        assert report["status"] == "FAIL"
        assert report["gates"]["headline_selected_for_all_groups"] is False
        assert report["gates"]["headline_group_missing_count"] == 1


def test_step6_cli_command() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _prepare_step6_fixture(root)
        stdout = io.StringIO()
        with patch("sys.argv", ["heat-mnch", "step-6-absolute-burden"]), patch("heat_mnch.cli.root_path", return_value=root), redirect_stdout(stdout):
            exit_code = main()
        payload = json.loads(stdout.getvalue())
        assert exit_code == 0
        assert payload["step"] == "step6_absolute_burden_translation"
        assert payload["status"] == "PASS"
