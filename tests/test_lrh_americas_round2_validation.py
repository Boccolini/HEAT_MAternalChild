from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from heat_mnch.future_rounds.lrh_americas_round2_validation import (
    _check_r_packages,
    build_freeze,
    run_phase2,
    run_lrh_americas_round2_validation,
    run_phase1,
    run_phase3,
    run_phase4,
)


def _real_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_json_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_r_scripts(root: Path) -> None:
    repo_root = _real_repo_root()
    scripts_root = root / "scripts" / "r"
    scripts_root.mkdir(parents=True, exist_ok=True)
    for filename in (
        "check_lrh_round2_packages.R",
        "run_lrh_phase2_dlnm.R",
        "run_lrh_phase3_case_crossover.R",
    ):
        shutil.copy2(repo_root / "scripts" / "r" / filename, scripts_root / filename)


def _write_round2_config(
    root: Path,
    *,
    tier1: list[str],
    tier2: list[str],
    phase2_outcomes: list[str],
    phase3_outcomes: list[str],
) -> None:
    _write_json_yaml(
        root / "config" / "analysis" / "lrh_americas_round2_validation.yaml",
        {
            "round2_validation_spec": {
                "analysis_id": "lrh_americas_round2_validation_test",
                "analysis_title": "Test round",
                "phase_order": ["freeze", "phase1", "phase2", "phase3", "phase4"],
                "parallel_acute_phases": False,
                "freeze_inputs": {
                    "step3_signal_matrix": "artifacts/analysis/step3/step3_signal_matrix.csv",
                    "step4_lag_comparison": "artifacts/analysis/step4/step4_lag_comparison.csv",
                    "step4_promotion_decisions": "artifacts/analysis/step4/step4_promotion_decisions.csv",
                    "step5_scorecard": "artifacts/analysis/step5/step5_scorecard.csv",
                    "step6_candidate_catalog": "artifacts/analysis/step6/step6_candidate_catalog.csv",
                    "step7_candidate_catalog": "artifacts/analysis/step7/step7_candidate_catalog.csv",
                    "step7_event_level_results": "artifacts/analysis/step7/step7_event_level_results.csv",
                    "step7_temporal_controls_appendix": "artifacts/analysis/step7/step7_temporal_controls_appendix.csv",
                    "mart_inventory": "artifacts/analysis/shared_marts/analysis_mart_inventory.csv",
                },
                "roadmap_doc": "reports/publications/incremental_analysis_round/28_lrh_americas_future_analytic_round_roadmap.md",
                "tier_defaults": {
                    "tier1": tier1,
                    "tier2": tier2,
                    "exclude_all_other_outcomes_by_default": True,
                },
                "manuscript_roles": {
                    "tier1": "principal",
                    "tier2": "supportive",
                },
                "phase1": {
                    "derivation_years": [2020, 2021, 2022, 2023],
                    "validation_years": [2024],
                    "magnitude_ratio_min": 0.5,
                    "magnitude_ratio_max": 1.5,
                },
                "phase2": {
                    "acute_outcomes": phase2_outcomes,
                    "required_r_packages": ["dlnm", "fixest"],
                    "lag_max_default": 7,
                    "lag_max_extended": 21,
                    "extended_lag_allowed_outcomes": [],
                    "reference_quantile": 0.5,
                    "hot_quantile": 0.95,
                },
                "phase3": {
                    "acute_outcomes": phase3_outcomes,
                    "required_r_packages": ["fixest"],
                },
            }
        },
    )


def _acute_rows_for_year(year: int) -> pd.DataFrame:
    dates = pd.date_range(f"{year}-01-01", periods=28, freq="D")
    rows: list[dict[str, object]] = []
    for municipality_code in ("110001", "110002"):
        for index, date in enumerate(dates):
            exposure = 28.0 if index % 7 in {4, 5, 6} else 20.0
            if year == 2024:
                exposure = 26.0 if index % 7 in {4, 5, 6} else 20.0
            exposure = exposure + (index * 0.05)
            rows.append(
                {
                    "municipality_code": municipality_code,
                    "observation_date": date,
                    "calendar_year": year,
                    "calendar_month": int(date.month),
                    "day_of_week": int(date.dayofweek),
                    "quality_control_status": "PASS",
                    "offset_valid_flag": 1,
                    "denominator_zero_flag": 0,
                    "denominator_missing_flag": 0,
                    "rate_not_computable_flag": 0,
                    "under_five_death_count": 4 if exposure > 25 else 1,
                    "child_diarrhea_gi_admission_count": 7 if exposure > 25 else 2,
                    "under_five_population_count": 1000.0,
                    "annual_live_birth_count": 500.0,
                    "approximate_heat_index_c_lag0": exposure,
                    "approximate_heat_index_c_lag0_1_mean": exposure - 0.25,
                    "approximate_heat_index_c_lag0_3_mean": exposure - 0.5,
                    "approximate_heat_index_c_lag0_7_mean": exposure - 0.75,
                    "maximum_daily_temperature_c_lag0": exposure - 3,
                    "maximum_daily_temperature_c_lag0_1_mean": exposure - 3.25,
                    "maximum_daily_temperature_c_lag0_3_mean": exposure - 3.5,
                    "maximum_daily_temperature_c_lag0_7_mean": exposure - 3.75,
                }
            )
    return pd.DataFrame(rows)


def _write_acute_mart(root: Path) -> None:
    mart_root = root / "artifacts" / "analysis" / "shared_marts" / "acute_daily_mart"
    inventory_rows: list[dict[str, object]] = []
    for year in (2020, 2021, 2022, 2023, 2024):
        year_root = mart_root / f"YEAR={year}"
        year_root.mkdir(parents=True, exist_ok=True)
        frame = _acute_rows_for_year(year)
        output_path = year_root / f"acute_daily_mart_{year}.parquet"
        frame.to_parquet(output_path, index=False)
        inventory_rows.append(
            {
                "mart_name": "acute_daily_mart",
                "partition_label": year,
                "row_count": int(frame.shape[0]),
                "column_count": int(frame.shape[1]),
                "output_path": f"artifacts/analysis/shared_marts/acute_daily_mart/YEAR={year}/acute_daily_mart_{year}.parquet",
                "live_birth_rows_backfilled": 0,
                "offset_rows_backfilled": 0,
            }
        )
    pd.DataFrame(inventory_rows).to_csv(
        root / "artifacts" / "analysis" / "shared_marts" / "analysis_mart_inventory.csv",
        index=False,
    )


def _write_minimal_step_inputs(root: Path, outcomes: list[str]) -> None:
    (root / "artifacts" / "analysis" / "step3").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "analysis" / "step4").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "analysis" / "step5").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "analysis" / "step6").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "analysis" / "step7").mkdir(parents=True, exist_ok=True)

    step3_rows = []
    step4_rows = []
    step4_decision_rows = []
    step5_rows = []
    step6_rows = []
    step7_rows = []
    appendix_rows = []
    for outcome in outcomes:
        step3_rows.append(
            {
                "analysis_unit_id": outcome,
                "candidate_track": "main_round",
                "analysis_role": "primary",
                "outcome_family": "acute_daily",
                "exposure_family_id": "family_4_heat_index_parallel",
                "exposure_family_name": "Approximate heat index",
                "screening_result_count": 4,
                "finite_result_count": 4,
                "screening_domain_count": 1,
                "sign_consistency_flag": True,
                "support_grade": "A",
                "support_grade_rank": 1,
                "best_lag_or_window": "lag0",
                "best_screening_domain": "acute_p95_vs_reference",
                "best_screening_method": "within_municipality_year_quintile_extremes",
                "best_exposure_variable": "approximate_heat_index_c_lag0",
                "best_exposure_label": "Approximate heat index (lag 0)",
                "best_estimate": 1.2,
                "best_lower_ci": 1.1,
                "best_upper_ci": 1.3,
                "best_p_value": 0.001,
                "best_valid_strata_count": 40,
                "headline_eligible_flag": "yes",
                "estimate_scale": "rate_ratio_per_interquartile_range_increase",
            }
        )
        for lag, order, estimate in (("lag0", 0, 1.3), ("lag0_1", 1, 1.15)):
            step4_rows.append(
                {
                    "analysis_unit_id": outcome,
                    "candidate_track": "main_round",
                    "analysis_role": "primary",
                    "outcome_family": "acute_daily",
                    "headline_family_recommendation": "heat_index",
                    "selected_exposure_family_id": "family_4_heat_index_parallel",
                    "selected_exposure_family_name": "Approximate heat index",
                    "lag_or_window": lag,
                    "lag_window_order": order,
                    "finite_result_count": 4,
                    "screening_domain_count": 1,
                    "sign_consistency_flag": True,
                    "support_grade": "A",
                    "support_grade_rank": 1,
                    "best_screening_domain": "acute_p95_vs_reference",
                    "best_screening_method": "within_municipality_year_quintile_extremes",
                    "best_estimate": estimate,
                    "best_lower_ci": estimate - 0.1,
                    "best_upper_ci": estimate + 0.1,
                    "best_p_value": 0.001 if lag == "lag0" else 0.01,
                    "best_valid_strata_count": 40,
                    "seasonality_controls_inherited": "calendar_year;calendar_month;day_of_week",
                    "seasonality_controls_fixed_flag": True,
                }
            )
        step4_decision_rows.append(
            {
                "analysis_unit_id": outcome,
                "candidate_track": "main_round",
                "analysis_role": "primary",
                "outcome_family": "acute_daily",
                "headline_family_recommendation": "heat_index",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "selected_exposure_family_name": "Approximate heat index",
                "preferred_lag_or_window": "lag0",
                "preferred_support_grade": "A",
                "preferred_support_grade_rank": 1,
                "preferred_best_screening_domain": "acute_p95_vs_reference",
                "preferred_best_p_value": 0.001,
                "seasonality_controls_fixed_flag": True,
                "decision_status": "promote_selected_window",
                "selected_reason": "fixture",
                "candidate_ready_for_step5_flag": True,
                "next_step_route": "step5_heterogeneity_synthesis",
            }
        )
        step5_rows.append(
            {
                "analysis_unit_id": outcome,
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "selected_exposure_family_name": "Approximate heat index",
                "selected_exposure_variable": "approximate_heat_index_c_lag0",
                "preferred_lag_or_window": "lag0",
                "estimate_scale": "rate_ratio_per_interquartile_range_increase",
                "full_sample_estimable_flag": True,
                "full_sample_estimate": 1.3,
                "full_sample_lower_ci": 1.2,
                "full_sample_upper_ci": 1.4,
                "full_sample_p_value": 0.001,
                "full_sample_valid_strata_count": 40,
                "modifier_count_reviewed": 1,
                "modifier_count_with_estimates": 1,
                "stable_modifier_count": 1,
                "inconsistent_modifier_count": 0,
                "significant_reversal_modifier_count": 0,
                "has_any_significant_reversal_flag": False,
                "blocking_inconsistent_modifier_count": 0,
                "non_blocking_inconsistent_modifier_count": 0,
                "candidate_ready_for_step6_flag": True,
                "next_step_route": "step6_absolute_burden_translation",
                "decision_reason": "fixture",
            }
        )
        step6_rows.append(
            {
                "analysis_unit_id": outcome,
                "mart_name": "acute_daily_mart",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "selected_exposure_family_name": "Approximate heat index",
                "selected_exposure_variable": "approximate_heat_index_c_lag0",
                "preferred_lag_or_window": "lag0",
                "estimate_scale": "rate_ratio_per_interquartile_range_increase",
                "denominator_column": "under_five_population_count",
                "seasonality_controls_applied": "calendar_year;calendar_month;day_of_week",
                "contrast_operator": "acute_p95_vs_reference",
                "outcome_family": "acute_daily",
                "estimate": 1.3 if outcome == "under_five_death_count" else 1.25,
                "lower_ci": 1.15,
                "upper_ci": 1.4,
                "p_value": 0.001,
            }
        )
        step7_rows.append(
            {
                "analysis_unit_id": outcome,
                "group_id": "child_mortality",
                "group_label": "Child Mortality",
                "public_label": outcome,
                "step6_final_translation_stage": "headline_absolute",
                "step7_route_class": "aggregated_inherited",
                "step7_route_reason": "fixture",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "selected_exposure_family_name": "Approximate heat index",
                "preferred_lag_or_window": "lag0",
                "headline_group_surface_flag": True,
                "all_candidate_surface_flag": True,
                "public_use": "relative_plus_observed_hot_volume",
                "absolute_burden_allowed": True,
                "headline_group_allowed": True,
                "has_any_significant_reversal_flag": False,
                "significant_reversal_modifier_count": 0,
            }
        )
        appendix_rows.append(
            {
                "analysis_unit_id": outcome,
                "data_source": "acute_daily_mart",
                "selected_exposure_family_name": "Approximate heat index",
                "preferred_lag_or_window": "lag0",
                "primary_temporal_controls": "calendar_year,calendar_month,day_of_week",
                "robustness_temporal_controls": "",
            }
        )

    pd.DataFrame(step3_rows).to_csv(root / "artifacts" / "analysis" / "step3" / "step3_signal_matrix.csv", index=False)
    pd.DataFrame(step4_rows).to_csv(root / "artifacts" / "analysis" / "step4" / "step4_lag_comparison.csv", index=False)
    pd.DataFrame(step4_decision_rows).to_csv(root / "artifacts" / "analysis" / "step4" / "step4_promotion_decisions.csv", index=False)
    pd.DataFrame(step5_rows).to_csv(root / "artifacts" / "analysis" / "step5" / "step5_scorecard.csv", index=False)
    pd.DataFrame(step6_rows).to_csv(root / "artifacts" / "analysis" / "step6" / "step6_candidate_catalog.csv", index=False)
    pd.DataFrame(step7_rows).to_csv(root / "artifacts" / "analysis" / "step7" / "step7_candidate_catalog.csv", index=False)
    pd.DataFrame().to_csv(root / "artifacts" / "analysis" / "step7" / "step7_event_level_results.csv", index=False)
    pd.DataFrame(appendix_rows).to_csv(root / "artifacts" / "analysis" / "step7" / "step7_temporal_controls_appendix.csv", index=False)


def _build_fixture(
    *,
    tier1: list[str],
    tier2: list[str],
    phase2_outcomes: list[str],
    phase3_outcomes: list[str],
) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    tmpdir = tempfile.TemporaryDirectory()
    root = Path(tmpdir.name)
    _copy_r_scripts(root)
    all_outcomes = list(dict.fromkeys([*tier1, *tier2]))
    _write_round2_config(root, tier1=tier1, tier2=tier2, phase2_outcomes=phase2_outcomes, phase3_outcomes=phase3_outcomes)
    _write_acute_mart(root)
    _write_minimal_step_inputs(root, all_outcomes)
    return root, tmpdir


def test_build_freeze_registry_uses_tier_defaults_and_excludes_others() -> None:
    root, tmpdir = _build_fixture(
        tier1=["under_five_death_count"],
        tier2=["child_diarrhea_gi_admission_count"],
        phase2_outcomes=[],
        phase3_outcomes=[],
    )
    try:
        report = build_freeze(root)
        assert report["status"] == "PASS"
        registry = pd.read_csv(root / report["outputs"]["retained_endpoint_registry"])
        assert registry["analysis_unit_id"].tolist() == ["under_five_death_count", "child_diarrhea_gi_admission_count"]
        assert registry["priority_tier_next_round"].tolist() == ["tier1", "tier2"]
        assert registry["route_class"].eq("aggregated_inherited").all()
    finally:
        tmpdir.cleanup()


def test_phase1_temporal_validation_writes_decision_table() -> None:
    root, tmpdir = _build_fixture(
        tier1=["under_five_death_count"],
        tier2=[],
        phase2_outcomes=[],
        phase3_outcomes=[],
    )
    try:
        build_freeze(root)
        report = run_phase1(root)
        assert report["status"] == "PASS"
        decision = pd.read_csv(root / report["outputs"]["temporal_stability_editorial_decision_table"])
        assert decision.loc[0, "analysis_unit_id"] == "under_five_death_count"
        assert decision.loc[0, "editorial_classification"] in {"principal", "supportive", "provisional"}
    finally:
        tmpdir.cleanup()


def test_r_preflight_detects_missing_package() -> None:
    root, tmpdir = _build_fixture(
        tier1=["under_five_death_count"],
        tier2=[],
        phase2_outcomes=["under_five_death_count"],
        phase3_outcomes=[],
    )
    try:
        try:
            _check_r_packages(root, ["package_that_should_not_exist_round2"])
        except RuntimeError as exc:
            assert "package_that_should_not_exist_round2" in str(exc)
        else:
            raise AssertionError("Expected missing package to fail preflight")
    finally:
        tmpdir.cleanup()


def test_phase2_runs_on_minimal_fixture_and_writes_diagnostics() -> None:
    root, tmpdir = _build_fixture(
        tier1=["under_five_death_count"],
        tier2=[],
        phase2_outcomes=["under_five_death_count"],
        phase3_outcomes=[],
    )
    try:
        try:
            _check_r_packages(root, ["dlnm", "fixest"])
        except RuntimeError:
            pytest.skip("Required R packages for phase2 are not available")
        build_freeze(root)
        report = run_phase2(root)
        assert report["status"] == "PASS"
        diagnostics = pd.read_csv(root / report["outputs"]["dlnm_outcome_diagnostics"])
        assert diagnostics.loc[0, "analysis_unit_id"] == "under_five_death_count"
        assert diagnostics.loc[0, "status"] == "success"
        summary = pd.read_csv(root / report["outputs"]["dlnm_lag_response_summary_table"])
        assert summary.loc[0, "analysis_unit_id"] == "under_five_death_count"
        assert (root / report["outputs"]["phase2_r_stderr_log"]).exists()
    finally:
        tmpdir.cleanup()


def test_phase3_runs_on_minimal_fixture_and_writes_outputs() -> None:
    root, tmpdir = _build_fixture(
        tier1=["under_five_death_count"],
        tier2=[],
        phase2_outcomes=[],
        phase3_outcomes=["under_five_death_count"],
    )
    try:
        try:
            _check_r_packages(root, ["fixest"])
        except RuntimeError:
            pytest.skip("Required R packages for phase3 are not available")
        build_freeze(root)
        report = run_phase3(root)
        assert report["status"] == "PASS"
        comparison = pd.read_csv(root / report["outputs"]["comparison_table"])
        assert comparison.loc[0, "analysis_unit_id"] == "under_five_death_count"
        assert comparison.loc[0, "concordance_classification"] in {"concordant", "partial_concordance", "discordant"}
        diagnostics = pd.read_csv(root / report["outputs"]["case_crossover_outcome_diagnostics"])
        assert diagnostics.loc[0, "analysis_unit_id"] == "under_five_death_count"
        assert diagnostics.loc[0, "status"] == "success"
        assert (root / report["outputs"]["phase3_r_stderr_log"]).exists()
    finally:
        tmpdir.cleanup()


def test_phase3_fails_when_all_acute_outcomes_fail() -> None:
    root, tmpdir = _build_fixture(
        tier1=["under_five_death_count"],
        tier2=[],
        phase2_outcomes=[],
        phase3_outcomes=["under_five_death_count"],
    )
    try:
        try:
            _check_r_packages(root, ["fixest"])
        except RuntimeError:
            pytest.skip("Required R packages for phase3 are not available")
        mart_root = root / "artifacts" / "analysis" / "shared_marts" / "acute_daily_mart"
        for parquet_path in mart_root.glob("YEAR=*/acute_daily_mart_*.parquet"):
            frame = pd.read_parquet(parquet_path)
            frame["offset_valid_flag"] = 0
            frame.to_parquet(parquet_path, index=False)
        build_freeze(root)
        with pytest.raises(RuntimeError, match="without any successful acute outcomes"):
            run_phase3(root)
    finally:
        tmpdir.cleanup()


def test_phase4_fails_when_acute_outputs_are_missing() -> None:
    root, tmpdir = _build_fixture(
        tier1=["under_five_death_count"],
        tier2=[],
        phase2_outcomes=["under_five_death_count"],
        phase3_outcomes=["under_five_death_count"],
    )
    try:
        build_freeze(root)
        run_phase1(root)
        try:
            run_phase4(root)
        except ValueError as exc:
            assert "Missing acute outputs" in str(exc)
        else:
            raise AssertionError("Expected phase4 to fail without acute outputs")
    finally:
        tmpdir.cleanup()


def test_phase4_runs_after_phase2_and_phase3_are_available() -> None:
    root, tmpdir = _build_fixture(
        tier1=["under_five_death_count"],
        tier2=[],
        phase2_outcomes=["under_five_death_count"],
        phase3_outcomes=["under_five_death_count"],
    )
    try:
        try:
            _check_r_packages(root, ["dlnm", "fixest"])
        except RuntimeError:
            pytest.skip("Required R packages for phase2/phase3 are not available")
        build_freeze(root)
        run_phase1(root)
        run_phase2(root)
        run_phase3(root)
        report = run_phase4(root)
        assert report["status"] == "PASS"
        hierarchy = pd.read_csv(root / report["outputs"]["updated_endpoint_hierarchy_table"])
        assert hierarchy.loc[0, "analysis_unit_id"] == "under_five_death_count"
        assert hierarchy.loc[0, "phase3_classification"] in {"concordant", "partial_concordance", "discordant"}
    finally:
        tmpdir.cleanup()


def test_end_to_end_all_runs_when_acute_r_phases_are_empty() -> None:
    root, tmpdir = _build_fixture(
        tier1=["under_five_death_count"],
        tier2=[],
        phase2_outcomes=[],
        phase3_outcomes=[],
    )
    try:
        payload = run_lrh_americas_round2_validation(root=root, phase="all", resume=False)
        assert payload["status"] == "PASS"
        hierarchy = pd.read_csv(root / "artifacts" / "analysis" / "lrh_americas_round2_validation" / "phase4_resynthesis" / "updated_endpoint_hierarchy_table.csv")
        assert hierarchy.loc[0, "analysis_unit_id"] == "under_five_death_count"
    finally:
        tmpdir.cleanup()
