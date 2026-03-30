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
from heat_mnch.analysis_steps.step5 import (
    _build_scorecard,
    build_step5_heterogeneity_synthesis,
)
from heat_mnch.cli import main
from test_analysis_steps_step2 import (
    _write_step0_inputs_for_cli,
    _write_step1_artifacts,
    _write_step2_config_bundle,
    _write_shared_marts,
)


def _expand_marts_for_step5(root: Path) -> None:
    context_rows = [
        {
            "municipality_code": "110001",
            "urban_population_proportion": 0.25,
            "expanded_gini_index": 0.42,
            "expanded_human_development_index": 0.56,
            "mean_altitude_m": 120.0,
            "coastal_municipality_flag": 0,
            "context_join_status": "OK",
        },
        {
            "municipality_code": "110002",
            "urban_population_proportion": 0.55,
            "expanded_gini_index": 0.52,
            "expanded_human_development_index": 0.66,
            "mean_altitude_m": 420.0,
            "coastal_municipality_flag": 1,
            "context_join_status": "OK",
        },
        {
            "municipality_code": "110003",
            "urban_population_proportion": 0.85,
            "expanded_gini_index": 0.62,
            "expanded_human_development_index": 0.76,
            "mean_altitude_m": 720.0,
            "coastal_municipality_flag": 0,
            "context_join_status": "OK",
        },
    ]
    context = pd.DataFrame(context_rows)
    context_path = root / "artifacts" / "analysis" / "shared_marts" / "context_lookup.parquet"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context.to_parquet(context_path, index=False)

    municipality_offsets = {
        "110001": 0.0,
        "110002": 1.0,
        "110003": 2.0,
    }
    outcome_columns = {
        "acute_daily_mart": [
            "neonatal_death_count",
            "postneonatal_death_count",
            "child_excessive_natural_heat_x30_any_mention_death_count",
        ],
        "perinatal_delivery_window_mart": [
            "preterm_birth_count",
            "gestational_days_lost_below_37_weeks",
        ],
        "maternal_operational_mart": [
            "hypertensive_disorder_of_pregnancy_count",
        ],
    }
    exposure_prefixes = {
        "maximum_daily_temperature_c",
        "approximate_heat_index_c",
    }
    denominator_columns = {
        "annual_live_birth_count",
        "under_five_population_count",
        "women_age_15_49_population_count",
        "daily_live_birth_count",
    }

    for mart_name in ("acute_daily_mart", "perinatal_delivery_window_mart", "maternal_operational_mart"):
        path = root / "artifacts" / "analysis" / "shared_marts" / mart_name / "YEAR=2024" / f"{mart_name}_2024.parquet"
        base = pd.read_parquet(path)
        expanded_frames = []
        for municipality_code, offset in municipality_offsets.items():
            frame = base.copy()
            frame["municipality_code"] = municipality_code
            for column in frame.columns:
                if any(column.startswith(prefix) for prefix in exposure_prefixes):
                    frame[column] = pd.to_numeric(frame[column], errors="coerce") + offset
                elif column in outcome_columns[mart_name]:
                    if column == "gestational_days_lost_below_37_weeks":
                        frame[column] = pd.to_numeric(frame[column], errors="coerce") + (offset * 0.2)
                    else:
                        frame[column] = pd.to_numeric(frame[column], errors="coerce") + int(offset)
                elif column in denominator_columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce") + (offset * 5.0)
            expanded_frames.append(frame)
        pd.concat(expanded_frames, ignore_index=True).to_parquet(path, index=False)


def _prepare_step5_fixture(root: Path) -> dict[str, object]:
    _write_step2_config_bundle(root)
    _write_step1_artifacts(root)
    _write_shared_marts(root)
    _expand_marts_for_step5(root)
    _write_step0_inputs_for_cli(root)
    step2_report = build_step2_baseline_models(root, step1_report={"status": "PASS"})
    step3_report = build_step3_threshold_discovery(root, step2_report=step2_report)
    return build_step4_lag_refinement(root, step3_report=step3_report)


def test_build_step5_heterogeneity_synthesis_writes_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        step4_report = _prepare_step5_fixture(root)

        report = build_step5_heterogeneity_synthesis(root, step4_report=step4_report)

        assert report["status"] == "PASS"
        assert report["step"] == "step5_heterogeneity_synthesis"
        assert report["summary"]["candidate_count"] == 5
        assert report["gates"]["upstream_step4_pass"] is True
        assert report["gates"]["selected_specs_resolved_for_candidates"] is True
        assert report["gates"]["context_lookup_available"] is True
        assert report["gates"]["context_join_complete_for_candidates"] is True
        assert report["gates"]["heterogeneity_results_written"] is True
        assert report["gates"]["modifier_review_complete"] is True
        assert report["gates"]["step6_decisions_written"] is True
        assert report["summary"]["gini_excluded_municipality_count"] == 0
        assert report["summary"]["step6_ready_count"] == report["summary"]["candidate_count"]
        assert (root / report["outputs"]["heterogeneity_results"]).exists()
        assert (root / report["outputs"]["effect_modification_matrix"]).exists()
        assert (root / report["outputs"]["step5_scorecard"]).exists()
        assert (root / report["outputs"]["gini_analytic_universe_exclusions"]).exists()
        assert (root / report["outputs"]["step5_directional_reversal_summary"]).exists()
        assert (root / report["outputs"]["step5_directional_reversal_diagnostic"]).exists()
        assert (root / report["outputs"]["step5_directional_reversal_diagnostic_xlsx"]).exists()
        assert (root / report["outputs"]["step5_block_matrix_csv"]).exists()
        assert (root / report["outputs"]["step5_block_matrix_xlsx"]).exists()
        assert (root / report["outputs"]["decision_checklist"]).exists()
        assert (root / report["outputs"]["step5_report"]).exists()

        heterogeneity = pd.read_csv(root / report["outputs"]["heterogeneity_results"])
        modifier_matrix = pd.read_csv(root / report["outputs"]["effect_modification_matrix"])
        scorecard = pd.read_csv(root / report["outputs"]["step5_scorecard"])
        matrix_csv = pd.read_csv(root / report["outputs"]["step5_block_matrix_csv"])
        assert heterogeneity["analysis_unit_id"].nunique() == 5
        assert set(heterogeneity["modifier_name"]) == {
            "urban_population_proportion",
            "expanded_gini_index",
            "expanded_human_development_index",
            "mean_altitude_m",
            "coastal_municipality_flag",
        }
        assert "direction_vs_full_sample" in heterogeneity.columns
        assert len(modifier_matrix) == 25
        assert len(scorecard) == 5
        assert modifier_matrix["matrix_flag"].isin({"consistent", "reversal", "significant_reversal"}).all()
        assert scorecard["candidate_ready_for_step6_flag"].fillna(False).astype(bool).all()
        assert scorecard["next_step_route"].astype(str).eq("step6_absolute_burden_translation").all()
        assert "significant_reversal_modifier_count" in scorecard.columns
        assert "has_any_significant_reversal_flag" in scorecard.columns
        assert not matrix_csv.empty
        for column in ["Urbanization", "Gini", "HDI", "Altitude", "Coastal status"]:
            assert matrix_csv[column].astype(str).isin({"consistent", "reversal", "significant_reversal"}).all()
        assert matrix_csv["Step 6"].astype(str).eq("promoted").all()
        assert scorecard["next_step_route"].astype(str).str.len().gt(0).all()


def test_build_step5_heterogeneity_synthesis_fails_when_step4_fails() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_step2_config_bundle(root)

        report = build_step5_heterogeneity_synthesis(root, step4_report={"status": "FAIL"})

        assert report["status"] == "FAIL"
        assert report["gates"]["upstream_step4_pass"] is False
        assert report["summary"]["candidate_count"] == 0
        assert (root / report["outputs"]["heterogeneity_results"]).exists()
        assert (root / report["outputs"]["step5_scorecard"]).exists()


def test_build_step5_excludes_missing_gini_municipality_only_for_gini() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        step4_report = _prepare_step5_fixture(root)
        context_path = root / "artifacts" / "analysis" / "shared_marts" / "context_lookup.parquet"
        context = pd.read_parquet(context_path)
        mask = context["municipality_code"].astype(str).eq("110003")
        context.loc[mask, "expanded_gini_index"] = pd.NA
        context.loc[mask, "expanded_human_development_index"] = pd.NA
        context.loc[mask, "context_join_status"] = "MISSING_GINI|IDH"
        context.to_parquet(context_path, index=False)

        report = build_step5_heterogeneity_synthesis(root, step4_report=step4_report)

        assert report["status"] == "PASS"
        assert report["gates"]["context_join_complete_for_candidates"] is True
        assert report["summary"]["gini_excluded_municipality_count"] == 1
        assert report["summary"]["gini_suppressed_rows"] > 0

        exclusions = pd.read_csv(root / report["outputs"]["gini_analytic_universe_exclusions"])
        assert exclusions["municipality_code"].astype(str).tolist() == ["110003"]
        assert exclusions["exclusion_scope"].astype(str).tolist() == ["expanded_gini_index_only"]

        heterogeneity = pd.read_csv(root / report["outputs"]["heterogeneity_results"])
        urban_rows = heterogeneity.loc[
            heterogeneity["modifier_name"].astype(str).eq("urban_population_proportion")
        ]
        gini_rows = heterogeneity.loc[
            heterogeneity["modifier_name"].astype(str).eq("expanded_gini_index")
        ]
        assert urban_rows["municipality_count"].max() >= gini_rows["municipality_count"].max()
        assert (root / report["outputs"]["step5_block_matrix_xlsx"]).exists()


def test_build_scorecard_promotes_all_candidates_to_step6() -> None:
    candidate_selection = pd.DataFrame(
        [
            {
                "analysis_unit_id": "candidate_altitude_only",
                "linked_outcome_ids": "CANDIDATE_ALTITUDE_ONLY",
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
            },
            {
                "analysis_unit_id": "candidate_gini_blocked",
                "linked_outcome_ids": "CANDIDATE_GINI_BLOCKED",
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
            },
        ]
    )
    heterogeneity_results = pd.DataFrame(
        [
            {
                "analysis_unit_id": "candidate_altitude_only",
                "candidate_track": "main_round",
                "full_sample_estimate": 1.10,
                "full_sample_lower_ci": 1.02,
                "full_sample_upper_ci": 1.18,
                "full_sample_p_value": 0.01,
                "full_sample_valid_strata_count": 10,
            },
            {
                "analysis_unit_id": "candidate_gini_blocked",
                "candidate_track": "main_round",
                "full_sample_estimate": 1.12,
                "full_sample_lower_ci": 1.03,
                "full_sample_upper_ci": 1.20,
                "full_sample_p_value": 0.01,
                "full_sample_valid_strata_count": 10,
            },
        ]
    )
    effect_modification_matrix = pd.DataFrame(
        [
            {
                "analysis_unit_id": "candidate_altitude_only",
                "candidate_track": "main_round",
                "modifier_name": "mean_altitude_m",
                "estimated_stratum_count": 3,
                "summary_classification": "directional_reversal_present",
                "matrix_flag": "reversal",
                "progression_blocker_flag": False,
            },
            {
                "analysis_unit_id": "candidate_altitude_only",
                "candidate_track": "main_round",
                "modifier_name": "urban_population_proportion",
                "estimated_stratum_count": 3,
                "summary_classification": "stable_or_gradient",
                "matrix_flag": "consistent",
                "progression_blocker_flag": False,
            },
            {
                "analysis_unit_id": "candidate_gini_blocked",
                "candidate_track": "main_round",
                "modifier_name": "expanded_gini_index",
                "estimated_stratum_count": 3,
                "summary_classification": "directional_reversal_present",
                "matrix_flag": "significant_reversal",
                "progression_blocker_flag": True,
            },
            {
                "analysis_unit_id": "candidate_gini_blocked",
                "candidate_track": "main_round",
                "modifier_name": "urban_population_proportion",
                "estimated_stratum_count": 3,
                "summary_classification": "stable_or_gradient",
                "matrix_flag": "consistent",
                "progression_blocker_flag": False,
            },
        ]
    )

    scorecard = _build_scorecard(candidate_selection, heterogeneity_results, effect_modification_matrix)

    altitude_row = scorecard.loc[scorecard["analysis_unit_id"].eq("candidate_altitude_only")].iloc[0]
    gini_row = scorecard.loc[scorecard["analysis_unit_id"].eq("candidate_gini_blocked")].iloc[0]

    assert bool(altitude_row["candidate_ready_for_step6_flag"]) is True
    assert int(altitude_row["inconsistent_modifier_count"]) == 1
    assert int(altitude_row["blocking_inconsistent_modifier_count"]) == 0
    assert str(altitude_row["decision_reason"]) == "promoted_to_step6_with_reversal"

    assert bool(gini_row["candidate_ready_for_step6_flag"]) is True
    assert int(gini_row["blocking_inconsistent_modifier_count"]) == 1
    assert int(gini_row["significant_reversal_modifier_count"]) == 1
    assert bool(gini_row["has_any_significant_reversal_flag"]) is True
    assert str(gini_row["decision_reason"]) == "promoted_to_step6_with_significant_reversal"


def test_cli_smoke_runs_step5_and_steps_run_through_step5() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        step4_report = _prepare_step5_fixture(root)

        stdout = io.StringIO()
        with (
            patch("heat_mnch.cli.root_path", return_value=root),
            patch("heat_mnch.analysis_steps.step5.build_step4_lag_refinement", return_value=step4_report),
            redirect_stdout(stdout),
        ):
            exit_code = main(["step-5-heterogeneity-synthesis"])

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
            exit_code = main(["steps-run", "--through", "step5"])

        assert exit_code == 0
        payload = json.loads(stdout.getvalue())
        assert payload["run_mode"] == "through_step5"
        assert payload["step5"]["status"] == "PASS"
