from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from heat_mnch.analysis_steps.step2 import build_step2_baseline_models
from heat_mnch.cli import main


def _write_json_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_step2_config_bundle(root: Path) -> None:
    _write_json_yaml(
        root / "config" / "analysis" / "round_config.yaml",
        {
            "analysis_id": "heat_mnch_incremental_round",
            "analysis_title": "HEAT-MNCH Incremental Analysis Round",
            "strategy": "neutral_plus_alias",
            "years": [2024],
            "frozen_inputs": {
                "gold_release_report_path": "gold_release_candidate_report.md",
                "gold_release_report_fallback_path": "artifacts/status/gold_release_candidate_report.md",
                "endpoint_readiness_ledger_path": "endpoint_readiness_ledger.csv",
                "stage5_dictionary_matrix_path": "gold_stage5_dictionary_matrix.csv",
                "gold_manifest_package_dir": "gold_release_candidate_manifests",
            },
            "outcome_inventory_path": "config/analysis/outcome_inventory.yaml",
            "analysis_variable_aliases_path": "config/analysis/analysis_variable_aliases.yaml",
            "seasonality_contract_path": "config/analysis/seasonality_contract.yaml",
            "escalation_protocol_path": "config/analysis/escalation_protocol.yaml",
            "step2_backlog_review_analysis_units": [
                "postneonatal_death_count",
                "child_excessive_natural_heat_x30_any_mention_death_count",
            ],
            "output_roots": {
                "artifacts_root": "artifacts/analysis/step0",
                "publication_root": "reports/publications/incremental_analysis_round",
            },
            "protected_contracts": {
                "dashboard_manifest_path": "artifacts/dashboard/v1/manifest.json",
                "statcalc_registry_module_path": "src/heat_mnch/statcalc_api/registry.py",
                "statcalc_runner_module_path": "src/heat_mnch/statcalc_api/runner.py",
            },
            "supersession_entries": [],
        },
    )
    _write_json_yaml(
        root / "config" / "analysis" / "seasonality_contract.yaml",
        {
            "profiles": [
                {
                    "seasonality_profile_id": "acute_daily_standard",
                    "outcome_family": "acute_daily",
                    "outcome_id_optional": None,
                    "time_anchor": "observation_date",
                    "main_controls": ["calendar_year", "calendar_month", "day_of_week"],
                    "robustness_controls": [],
                    "required_from_step": "step0",
                    "notes": "Synthetic acute profile.",
                },
                {
                    "seasonality_profile_id": "neonatal_acute_with_epiweek",
                    "outcome_family": "acute_daily",
                    "outcome_id_optional": "DEATHS_NEONATAL",
                    "time_anchor": "observation_date",
                    "main_controls": ["calendar_year", "calendar_month", "day_of_week"],
                    "robustness_controls": ["epiweek"],
                    "required_from_step": "step0",
                    "notes": "Synthetic neonatal acute profile.",
                },
                {
                    "seasonality_profile_id": "perinatal_delivery_aligned",
                    "outcome_family": "perinatal_delivery_window_aggregate",
                    "outcome_id_optional": None,
                    "time_anchor": "delivery_event_anchor",
                    "main_controls": ["calendar_year", "event_or_delivery_month"],
                    "robustness_controls": [],
                    "required_from_step": "step0",
                    "notes": "Synthetic perinatal profile.",
                },
                {
                    "seasonality_profile_id": "maternal_daily_standard",
                    "outcome_family": "maternal_operational",
                    "outcome_id_optional": None,
                    "time_anchor": "observation_date",
                    "main_controls": ["calendar_year", "calendar_month", "day_of_week"],
                    "robustness_controls": [],
                    "required_from_step": "step0",
                    "notes": "Synthetic maternal profile.",
                },
            ]
        },
    )


def _write_step1_artifacts(root: Path) -> None:
    step1_root = root / "artifacts" / "analysis" / "step1"
    step1_root.mkdir(parents=True, exist_ok=True)
    analysis_units = pd.DataFrame(
        [
            {
                "analysis_unit_id": "neonatal_death_count",
                "linked_outcome_ids": "DEATHS_NEONATAL",
                "analysis_role": "primary",
                "outcome_family": "acute_daily",
                "denominator_id": "annual_live_births",
                "denominator_column": "annual_live_birth_count",
                "seasonality_profile_id": "neonatal_acute_with_epiweek",
                "mart_name": "acute_daily_mart",
                "promoted_outcome_flag": 1,
            },
            {
                "analysis_unit_id": "preterm_birth_count",
                "linked_outcome_ids": "PRETERM_FLAG",
                "analysis_role": "primary",
                "outcome_family": "perinatal_delivery_window_aggregate",
                "denominator_id": "annual_live_births",
                "denominator_column": "annual_live_birth_count",
                "seasonality_profile_id": "perinatal_delivery_aligned",
                "mart_name": "perinatal_delivery_window_mart",
                "promoted_outcome_flag": 1,
            },
            {
                "analysis_unit_id": "gestational_days_lost_below_37_weeks",
                "linked_outcome_ids": "GEST_DAYS_LOST_BELOW_37W",
                "analysis_role": "sensitivity",
                "outcome_family": "perinatal_delivery_window_aggregate",
                "denominator_id": "not_applicable_count_or_descriptor",
                "denominator_column": "",
                "seasonality_profile_id": "perinatal_delivery_aligned",
                "mart_name": "perinatal_delivery_window_mart",
                "promoted_outcome_flag": 1,
            },
            {
                "analysis_unit_id": "hypertensive_disorder_of_pregnancy_count",
                "linked_outcome_ids": "HDP_ANY",
                "analysis_role": "primary",
                "outcome_family": "maternal_operational",
                "denominator_id": "women_15_49_population",
                "denominator_column": "women_age_15_49_population_count",
                "seasonality_profile_id": "maternal_daily_standard",
                "mart_name": "maternal_operational_mart",
                "promoted_outcome_flag": 1,
            },
            {
                "analysis_unit_id": "postneonatal_death_count",
                "linked_outcome_ids": "DEATHS_28_364D",
                "analysis_role": "descriptive_only",
                "outcome_family": "acute_daily",
                "denominator_id": "annual_live_births",
                "denominator_column": "annual_live_birth_count",
                "seasonality_profile_id": "acute_daily_standard",
                "mart_name": "acute_daily_mart",
                "promoted_outcome_flag": 0,
            },
            {
                "analysis_unit_id": "child_excessive_natural_heat_x30_any_mention_death_count",
                "linked_outcome_ids": "ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_TOTAL",
                "analysis_role": "descriptive_only",
                "outcome_family": "acute_daily",
                "denominator_id": "under_five_population",
                "denominator_column": "under_five_population_count",
                "seasonality_profile_id": "acute_daily_standard",
                "mart_name": "acute_daily_mart",
                "promoted_outcome_flag": 0,
            },
        ]
    )
    sparse_flags = pd.DataFrame(
        [
            {
                "analysis_unit_id": "neonatal_death_count",
                "recommendation": "proceed_to_step2",
                "baseline_model_support_status": "PASS",
                "any_sparse_flag": False,
            },
            {
                "analysis_unit_id": "preterm_birth_count",
                "recommendation": "proceed_to_step2",
                "baseline_model_support_status": "PASS",
                "any_sparse_flag": False,
            },
            {
                "analysis_unit_id": "gestational_days_lost_below_37_weeks",
                "recommendation": "proceed_to_step2",
                "baseline_model_support_status": "PASS",
                "any_sparse_flag": False,
            },
            {
                "analysis_unit_id": "hypertensive_disorder_of_pregnancy_count",
                "recommendation": "proceed_to_step2",
                "baseline_model_support_status": "PASS",
                "any_sparse_flag": False,
            },
            {
                "analysis_unit_id": "postneonatal_death_count",
                "recommendation": "retain_descriptive_only",
                "baseline_model_support_status": "PASS",
                "any_sparse_flag": False,
            },
            {
                "analysis_unit_id": "child_excessive_natural_heat_x30_any_mention_death_count",
                "recommendation": "retain_descriptive_only",
                "baseline_model_support_status": "PASS",
                "any_sparse_flag": True,
            },
        ]
    )
    analysis_units.to_csv(step1_root / "analysis_unit_roster.csv", index=False)
    sparse_flags.to_csv(step1_root / "sparse_outcome_flags.csv", index=False)


def _synthetic_daily_frame(
    *,
    outcome_column: str,
    denominator_column: str,
    exposure_column: str,
    observation_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    rows = []
    for offset, observation_date in enumerate(observation_dates):
        rows.append(
            {
                "municipality_code": "110001",
                "observation_date": observation_date,
                "calendar_year": observation_date.year,
                "calendar_month": observation_date.month,
                "quality_control_status": "PASS",
                "offset_valid_flag": 1,
                "denominator_zero_flag": 0,
                "denominator_missing_flag": 0,
                "rate_not_computable_flag": 0,
                denominator_column: 100.0,
                exposure_column: 30.0 + (offset * 0.3),
                outcome_column: 1 + (offset % 3),
            }
        )
    return pd.DataFrame(rows)


def _write_shared_marts(root: Path) -> None:
    dates = pd.date_range("2024-01-01", periods=35, freq="D")
    acute = _synthetic_daily_frame(
        outcome_column="neonatal_death_count",
        denominator_column="annual_live_birth_count",
        exposure_column="maximum_daily_temperature_c_lag0_1_mean",
        observation_dates=dates,
    )
    acute["maximum_daily_temperature_c_lag0"] = acute["maximum_daily_temperature_c_lag0_1_mean"] - 0.4
    acute["maximum_daily_temperature_c_lag0_3_mean"] = acute["maximum_daily_temperature_c_lag0_1_mean"] + 0.6
    acute["maximum_daily_temperature_c_lag0_7_mean"] = acute["maximum_daily_temperature_c_lag0_1_mean"] + 1.2
    acute["approximate_heat_index_c_lag0"] = acute["maximum_daily_temperature_c_lag0"] + 2.5
    acute["approximate_heat_index_c_lag0_1_mean"] = acute["maximum_daily_temperature_c_lag0_1_mean"] + 2.5
    acute["approximate_heat_index_c_lag0_3_mean"] = acute["maximum_daily_temperature_c_lag0_3_mean"] + 2.5
    acute["approximate_heat_index_c_lag0_7_mean"] = acute["maximum_daily_temperature_c_lag0_7_mean"] + 2.5
    acute["postneonatal_death_count"] = 1 + (pd.Series(range(len(dates))) % 2)
    acute["child_excessive_natural_heat_x30_any_mention_death_count"] = (
        pd.Series(range(len(dates))).isin([0, 10, 20, 30]).astype(int)
    )
    acute["under_five_population_count"] = 500.0
    acute["log_annual_live_birth_offset"] = 4.605170
    acute["log_under_five_population_offset"] = 6.214608
    perinatal = _synthetic_daily_frame(
        outcome_column="preterm_birth_count",
        denominator_column="annual_live_birth_count",
        exposure_column="maximum_daily_temperature_c_days_0_6_before_event_mean",
        observation_dates=dates,
    )
    perinatal["maximum_daily_temperature_c_days_0_13_before_event_mean"] = (
        perinatal["maximum_daily_temperature_c_days_0_6_before_event_mean"] + 0.8
    )
    perinatal["maximum_daily_temperature_c_days_0_27_before_event_mean"] = (
        perinatal["maximum_daily_temperature_c_days_0_6_before_event_mean"] + 1.4
    )
    perinatal["approximate_heat_index_c_days_0_6_before_event_mean"] = (
        perinatal["maximum_daily_temperature_c_days_0_6_before_event_mean"] + 1.75
    )
    perinatal["approximate_heat_index_c_days_0_13_before_event_mean"] = (
        perinatal["maximum_daily_temperature_c_days_0_13_before_event_mean"] + 1.75
    )
    perinatal["approximate_heat_index_c_days_0_27_before_event_mean"] = (
        perinatal["maximum_daily_temperature_c_days_0_27_before_event_mean"] + 1.75
    )
    perinatal["gestational_days_lost_below_37_weeks"] = 2.0 + (pd.Series(range(len(dates))) * 0.1)
    perinatal["daily_live_birth_count"] = 3.0
    perinatal["log_annual_live_birth_offset"] = 4.605170
    maternal = _synthetic_daily_frame(
        outcome_column="hypertensive_disorder_of_pregnancy_count",
        denominator_column="women_age_15_49_population_count",
        exposure_column="maximum_daily_temperature_c_days_0_6_before_event_mean",
        observation_dates=dates,
    )
    maternal["maximum_daily_temperature_c_days_0_13_before_event_mean"] = (
        maternal["maximum_daily_temperature_c_days_0_6_before_event_mean"] + 0.7
    )
    maternal["maximum_daily_temperature_c_days_0_27_before_event_mean"] = (
        maternal["maximum_daily_temperature_c_days_0_6_before_event_mean"] + 1.3
    )
    maternal["approximate_heat_index_c_days_0_6_before_event_mean"] = (
        maternal["maximum_daily_temperature_c_days_0_6_before_event_mean"] + 1.5
    )
    maternal["approximate_heat_index_c_days_0_13_before_event_mean"] = (
        maternal["maximum_daily_temperature_c_days_0_13_before_event_mean"] + 1.5
    )
    maternal["approximate_heat_index_c_days_0_27_before_event_mean"] = (
        maternal["maximum_daily_temperature_c_days_0_27_before_event_mean"] + 1.5
    )
    maternal["log_women_age_15_49_population_offset"] = 6.907755

    for mart_name, frame in (
        ("acute_daily_mart", acute),
        ("perinatal_delivery_window_mart", perinatal),
        ("maternal_operational_mart", maternal),
    ):
        path = root / "artifacts" / "analysis" / "shared_marts" / mart_name / "YEAR=2024" / f"{mart_name}_2024.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


def _write_step0_inputs_for_cli(root: Path) -> None:
    (root / "artifacts" / "dashboard" / "v1").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "dashboard" / "v1" / "manifest.json").write_text(
        json.dumps({"artifacts": {"dashboard_metrics": "artifacts/dashboard/v1/dashboard_metrics.parquet"}}, indent=2),
        encoding="utf-8",
    )
    (root / "src" / "heat_mnch" / "statcalc_api").mkdir(parents=True, exist_ok=True)
    (root / "src" / "heat_mnch" / "statcalc_api" / "registry.py").write_text("class StatCalcRegistry:\n    def __init__(self, root):\n        self.recipes = []\n", encoding="utf-8")
    (root / "src" / "heat_mnch" / "statcalc_api" / "runner.py").write_text("# synthetic\n", encoding="utf-8")


def test_build_step2_baseline_models_writes_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_step2_config_bundle(root)
        _write_step1_artifacts(root)
        _write_shared_marts(root)
        _write_step0_inputs_for_cli(root)

        report = build_step2_baseline_models(root, step1_report={"status": "PASS"})

        assert report["status"] == "PASS"
        assert report["step"] == "step2_baseline_aggregated_models"
        assert report["summary"]["candidate_count"] == 6
        assert report["summary"]["candidate_family_count"] == 12
        assert report["summary"]["preferred_model_count"] == 12
        assert report["summary"]["backlog_review_candidate_count"] == 2
        assert report["summary"]["backlog_review_approved_count"] == 1
        assert report["summary"]["dual_family_approved_count"] == 5
        assert report["gates"]["analysis_backlog_materialized"] is True
        assert report["gates"]["all_main_candidates_estimable_in_at_least_one_family"] is True
        assert report["gates"]["configured_backlog_review_screened"] is True
        assert (root / report["outputs"]["baseline_model_results"]).exists()
        assert (root / report["outputs"]["preferred_baseline_models"]).exists()
        assert (root / report["outputs"]["baseline_model_diagnostics"]).exists()
        assert (root / report["outputs"]["baseline_model_specification_matrix"]).exists()
        assert (root / report["outputs"]["exposure_family_comparison"]).exists()
        assert (root / report["outputs"]["promotion_decisions"]).exists()
        assert (root / report["outputs"]["step2_report"]).exists()

        results = pd.read_csv(root / report["outputs"]["baseline_model_results"])
        preferred = pd.read_csv(root / report["outputs"]["preferred_baseline_models"])
        specs = pd.read_csv(root / report["outputs"]["baseline_model_specification_matrix"])
        family_comparison = pd.read_csv(root / report["outputs"]["exposure_family_comparison"])
        promotions = pd.read_csv(root / report["outputs"]["promotion_decisions"])
        assert "epiweek_robustness_proxy" in results["model_label"].tolist()
        assert "gaussian_fe_proxy" in results["model_label"].tolist()
        assert preferred["analysis_unit_id"].nunique() == 6
        assert preferred["exposure_family_id"].nunique() == 2
        assert specs.loc[specs["analysis_unit_id"].eq("neonatal_death_count"), "seasonality_controls_applied"].iloc[0] == "calendar_year;calendar_month;day_of_week"
        assert family_comparison.loc[
            family_comparison["analysis_unit_id"].eq("postneonatal_death_count"),
            "estimable_exposure_family_count",
        ].iloc[0] == 2
        assert promotions.loc[promotions["analysis_unit_id"].eq("postneonatal_death_count"), "approved_for_next_steps_flag"].iloc[0]
        assert promotions.loc[promotions["analysis_unit_id"].eq("postneonatal_death_count"), "dual_family_step3_flag"].iloc[0]
        assert not promotions.loc[
            promotions["analysis_unit_id"].eq("child_excessive_natural_heat_x30_any_mention_death_count"),
            "approved_for_next_steps_flag",
        ].iloc[0]


def test_cli_smoke_runs_step2_and_steps_run_through_step2() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_step2_config_bundle(root)
        _write_step1_artifacts(root)
        _write_shared_marts(root)
        _write_step0_inputs_for_cli(root)

        stdout = io.StringIO()
        with (
            patch("heat_mnch.cli.root_path", return_value=root),
            patch("heat_mnch.analysis_steps.step2.build_step1_descriptive_atlas", return_value={"status": "PASS"}),
            redirect_stdout(stdout),
        ):
            exit_code = main(["step-2-baseline-models"])

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
            exit_code = main(["steps-run", "--through", "step2"])

        assert exit_code == 0
        payload = json.loads(stdout.getvalue())
        assert payload["run_mode"] == "through_step2"
        assert payload["step2"]["status"] == "PASS"


def test_step2_backlog_review_falls_back_from_descriptive_only_profile() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_step2_config_bundle(root)
        _write_step1_artifacts(root)
        _write_shared_marts(root)
        _write_step0_inputs_for_cli(root)

        roster_path = root / "artifacts" / "analysis" / "step1" / "analysis_unit_roster.csv"
        roster = pd.read_csv(roster_path)
        roster = pd.concat(
            [
                roster,
                pd.DataFrame(
                    [
                        {
                            "analysis_unit_id": "antepartum_haemorrhage_admission_count",
                            "linked_outcome_ids": "APH_ANTEPARTUM_TOTAL",
                            "analysis_role": "descriptive_only",
                            "outcome_family": "maternal_operational",
                            "denominator_id": "women_15_49_population",
                            "denominator_column": "women_age_15_49_population_count",
                            "seasonality_profile_id": "descriptive_only_profile",
                            "mart_name": "maternal_operational_mart",
                            "promoted_outcome_flag": 0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        roster.to_csv(roster_path, index=False)

        sparse_path = root / "artifacts" / "analysis" / "step1" / "sparse_outcome_flags.csv"
        sparse = pd.read_csv(sparse_path)
        sparse = pd.concat(
            [
                sparse,
                pd.DataFrame(
                    [
                        {
                            "analysis_unit_id": "antepartum_haemorrhage_admission_count",
                            "recommendation": "retain_descriptive_only",
                            "baseline_model_support_status": "PASS",
                            "any_sparse_flag": False,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        sparse.to_csv(sparse_path, index=False)

        round_config_path = root / "config" / "analysis" / "round_config.yaml"
        round_config = json.loads(round_config_path.read_text(encoding="utf-8"))
        round_config["step2_backlog_review_analysis_units"].append("antepartum_haemorrhage_admission_count")
        round_config_path.write_text(json.dumps(round_config, indent=2), encoding="utf-8")

        maternal_path = (
            root
            / "artifacts"
            / "analysis"
            / "shared_marts"
            / "maternal_operational_mart"
            / "YEAR=2024"
            / "maternal_operational_mart_2024.parquet"
        )
        maternal = pd.read_parquet(maternal_path)
        maternal["antepartum_haemorrhage_admission_count"] = 1 + (pd.Series(range(len(maternal))) % 4)
        maternal.to_parquet(maternal_path, index=False)

        report = build_step2_baseline_models(root, step1_report={"status": "PASS"})

        assert report["status"] == "PASS"
        roster = pd.read_csv(root / report["outputs"]["candidate_roster"])
        unit_rows = roster.loc[roster["analysis_unit_id"].eq("antepartum_haemorrhage_admission_count")]
        assert len(unit_rows) == 2
        assert set(unit_rows["exposure_family_id"]) == {
            "family_1_tmax_primary",
            "family_4_heat_index_parallel",
        }
        assert set(unit_rows["seasonality_profile_id"]) == {"maternal_daily_standard"}
