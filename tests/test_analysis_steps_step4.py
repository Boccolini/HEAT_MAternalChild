from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from heat_mnch.analysis_steps.step2 import build_step2_baseline_models
from heat_mnch.analysis_steps.step3 import build_step3_threshold_discovery
from heat_mnch.analysis_steps.step4 import build_step4_lag_refinement
from heat_mnch.cli import main
from test_analysis_steps_step2 import (
    _write_step0_inputs_for_cli,
    _write_step1_artifacts,
    _write_step2_config_bundle,
    _write_shared_marts,
)


def _prepare_step4_fixture(root: Path) -> dict[str, object]:
    _write_step2_config_bundle(root)
    _write_step1_artifacts(root)
    _write_shared_marts(root)
    _write_step0_inputs_for_cli(root)
    step2_report = build_step2_baseline_models(root, step1_report={"status": "PASS"})
    return build_step3_threshold_discovery(root, step2_report=step2_report)


def test_build_step4_lag_refinement_writes_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        step3_report = _prepare_step4_fixture(root)

        report = build_step4_lag_refinement(root, step3_report=step3_report)

        assert report["status"] == "PASS"
        assert report["step"] == "step4_lag_refinement"
        assert report["summary"]["candidate_count"] == 5
        assert report["summary"]["selected_count"] == 5
        assert report["summary"]["step5_ready_count"] == 5
        assert report["gates"]["upstream_step3_pass"] is True
        assert report["gates"]["headline_family_resolved_for_candidates"] is True
        assert report["gates"]["lag_comparison_written"] is True
        assert report["gates"]["selected_lag_written_for_candidates"] is True
        assert report["gates"]["seasonal_controls_fixed_across_comparisons"] is True
        assert report["gates"]["unstable_complex_lags_not_promoted"] is True
        assert (root / report["outputs"]["lag_comparison"]).exists()
        assert (root / report["outputs"]["timing_narratives"]).exists()
        assert (root / report["outputs"]["promotion_decisions"]).exists()
        assert (root / report["outputs"]["decision_checklist"]).exists()
        assert (root / report["outputs"]["step4_report"]).exists()

        lag_comparison = pd.read_csv(root / report["outputs"]["lag_comparison"])
        timing_narratives = pd.read_csv(root / report["outputs"]["timing_narratives"])
        promotion_decisions = pd.read_csv(root / report["outputs"]["promotion_decisions"])
        assert lag_comparison["analysis_unit_id"].nunique() == 5
        assert lag_comparison["selected_exposure_family_id"].nunique() == 1
        assert set(lag_comparison["selected_exposure_family_id"].tolist()) == {"family_4_heat_index_parallel"}
        assert lag_comparison["seasonality_controls_fixed_flag"].astype(bool).all()
        assert "outcome_family" in lag_comparison.columns
        assert promotion_decisions["candidate_ready_for_step5_flag"].astype(bool).all()
        assert set(timing_narratives["preferred_lag_or_window"].tolist()) <= {"lag0", "lag0_1", "lag0_3", "lag0_7", "0_6", "0_13", "0_27"}


def test_build_step4_lag_refinement_fails_when_step3_fails() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_step2_config_bundle(root)

        report = build_step4_lag_refinement(root, step3_report={"status": "FAIL"})

        assert report["status"] == "FAIL"
        assert report["gates"]["upstream_step3_pass"] is False
        assert report["summary"]["candidate_count"] == 0
        assert (root / report["outputs"]["lag_comparison"]).exists()
        assert (root / report["outputs"]["promotion_decisions"]).exists()


def test_build_step4_treats_fallback_to_simpler_window_as_safe() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_step2_config_bundle(root)

        screening_results = pd.DataFrame(
            [
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
                    "estimate_scale": "rate_ratio_per_interquartile_range_increase",
                    "estimate": 1.1,
                    "lower_ci": 1.0,
                    "upper_ci": 1.2,
                    "p_value": 0.04,
                    "valid_strata_count": 10,
                    "seasonality_controls_inherited": "calendar_year,event_or_delivery_month",
                }
            ]
        )
        family_threshold_comparison = pd.DataFrame(
            [
                {
                    "analysis_unit_id": "preterm_birth_count",
                    "candidate_track": "main_round",
                    "analysis_role": "primary",
                    "headline_family_recommendation": "heat_index",
                }
            ]
        )
        lag_comparison = pd.DataFrame(
            [
                {
                    "analysis_unit_id": "preterm_birth_count",
                    "candidate_track": "main_round",
                    "analysis_role": "primary",
                    "outcome_family": "perinatal_delivery_window_aggregate",
                    "headline_family_recommendation": "heat_index",
                    "selected_exposure_family_id": "family_4_heat_index_parallel",
                    "selected_exposure_family_name": "Approximate heat index",
                    "lag_or_window": "0_6",
                    "lag_window_order": 0,
                    "finite_result_count": 2,
                    "screening_domain_count": 2,
                    "sign_consistency_flag": True,
                    "support_grade": "B",
                    "support_grade_rank": 2,
                    "best_screening_domain": "perinatal_delivery_window_q5_vs_q1",
                    "best_screening_method": "within_municipality_year_quintile_extremes",
                    "best_estimate": 1.1,
                    "best_lower_ci": 1.0,
                    "best_upper_ci": 1.2,
                    "best_p_value": 0.04,
                    "best_valid_strata_count": 10,
                    "seasonality_controls_inherited": "calendar_year,event_or_delivery_month",
                    "seasonality_controls_fixed_flag": True,
                }
            ]
        )
        decisions = pd.DataFrame(
            [
                {
                    "analysis_unit_id": "preterm_birth_count",
                    "candidate_track": "main_round",
                    "analysis_role": "primary",
                    "outcome_family": "perinatal_delivery_window_aggregate",
                    "headline_family_recommendation": "heat_index",
                    "selected_exposure_family_id": "family_4_heat_index_parallel",
                    "selected_exposure_family_name": "Approximate heat index",
                    "preferred_lag_or_window": "0_6",
                    "preferred_support_grade": "B",
                    "preferred_support_grade_rank": 2,
                    "preferred_best_screening_domain": "perinatal_delivery_window_q5_vs_q1",
                    "preferred_best_p_value": 0.04,
                    "seasonality_controls_fixed_flag": True,
                    "decision_status": "promote_selected_window",
                    "selected_reason": "fallback_to_simpler_window_due_to_no_clear_gain",
                    "candidate_ready_for_step5_flag": True,
                    "next_step_route": "step5_heterogeneity_synthesis",
                }
            ]
        )
        timing = pd.DataFrame(
            [
                {
                    "analysis_unit_id": "preterm_birth_count",
                    "candidate_track": "main_round",
                    "analysis_role": "primary",
                    "outcome_family": "perinatal_delivery_window_aggregate",
                    "headline_family_recommendation": "heat_index",
                    "selected_exposure_family_id": "family_4_heat_index_parallel",
                    "selected_exposure_family_name": "Approximate heat index",
                    "preferred_lag_or_window": "0_6",
                    "timing_narrative": "Selected the shortest event-aligned 0 to 6 day window.",
                    "timing_scope_note": "Selected the shortest event-aligned 0 to 6 day window.",
                }
            ]
        )

        with (
            patch("heat_mnch.analysis_steps.step4._load_step3_inputs", return_value=(screening_results, family_threshold_comparison)),
            patch("heat_mnch.analysis_steps.step4._build_lag_comparison", return_value=lag_comparison),
            patch("heat_mnch.analysis_steps.step4._build_step4_decisions", return_value=(decisions, timing)),
        ):
            report = build_step4_lag_refinement(root, step3_report={"status": "PASS"})

        assert report["status"] == "PASS"
        assert report["gates"]["unstable_complex_lags_not_promoted"] is True


def test_cli_smoke_runs_step4_and_steps_run_through_step4() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        step3_report = _prepare_step4_fixture(root)

        stdout = io.StringIO()
        with (
            patch("heat_mnch.cli.root_path", return_value=root),
            patch("heat_mnch.analysis_steps.step4.build_step3_threshold_discovery", return_value=step3_report),
            redirect_stdout(stdout),
        ):
            exit_code = main(["step-4-lag-refinement"])

        assert exit_code == 0
        payload = json.loads(stdout.getvalue())
        assert payload["status"] == "PASS"

        stdout = io.StringIO()
        with (
            patch("heat_mnch.cli.root_path", return_value=root),
            patch("heat_mnch.analysis_steps.step0.build_step0_freeze", return_value={"status": "PASS"}),
            patch("heat_mnch.analysis_steps.step1.build_step1_descriptive_atlas", return_value={"status": "PASS"}),
            redirect_stdout(stdout),
        ):
            exit_code = main(["steps-run", "--through", "step4"])

        assert exit_code == 0
        payload = json.loads(stdout.getvalue())
        assert payload["run_mode"] == "through_step4"
        assert payload["step3"]["status"] == "PASS"
        assert payload["step4"]["status"] == "PASS"
