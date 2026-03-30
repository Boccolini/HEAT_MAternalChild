from __future__ import annotations

import io
import json
import shutil
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from heat_mnch.analysis_steps.step0 import build_step0_freeze
from heat_mnch.cli import main


def _copy_step0_bundle(source_root: Path, target_root: Path) -> None:
    source_analysis_dir = source_root / "config" / "analysis"
    target_analysis_dir = target_root / "config" / "analysis"
    target_analysis_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "round_config.yaml",
        "outcome_inventory.yaml",
        "analysis_variable_aliases.yaml",
        "seasonality_contract.yaml",
        "escalation_protocol.yaml",
    ):
        shutil.copy2(source_analysis_dir / filename, target_analysis_dir / filename)
    for filename in (
        "gold_release_candidate_report.md",
        "endpoint_readiness_ledger.csv",
        "gold_stage5_dictionary_matrix.csv",
    ):
        shutil.copy2(source_root / filename, target_root / filename)


def _write_dashboard_manifest(root: Path) -> Path:
    manifest_path = root / "artifacts" / "dashboard" / "v1" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"artifacts": {"dashboard_metrics": "artifacts/dashboard/v1/dashboard_metrics.parquet"}}, indent=2), encoding="utf-8")
    return manifest_path


def test_build_step0_freeze_writes_outputs() -> None:
    source_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _copy_step0_bundle(source_root, root)
        _write_dashboard_manifest(root)

        report = build_step0_freeze(root)

        assert report["status"] == "PASS"
        assert report["step"] == "step0_freeze_governance_reset"
        assert report["inventory_summary"]["total_outcomes"] == 42
        assert report["inventory_summary"]["descriptive_only_count"] == 18
        assert report["inventory_summary"]["deferred_event_level_count"] == 8
        assert (root / report["outputs"]["frozen_outcome_inventory"]).exists()
        assert (root / report["outputs"]["denominator_contract"]).exists()
        assert (root / report["outputs"]["seasonality_contract"]).exists()
        assert (root / report["outputs"]["supersession_map"]).exists()
        assert (root / report["outputs"]["round_charter"]).exists()
        assert (root / report["outputs"]["step0_report"]).exists()

        inventory = pd.read_csv(root / report["outputs"]["frozen_outcome_inventory"])
        assert "PRETERM_FLAG" in inventory["outcome_id"].tolist()
        assert "PRETERM_FLAG__event_level" in inventory["outcome_id"].tolist()
        assert "PRENATAL_VISITS_8PLUS_FLAG" in inventory["outcome_id"].tolist()
        assert "DEATHS_28_364D" in inventory["outcome_id"].tolist()
        assert "DEATHS_1_4Y" in inventory["outcome_id"].tolist()
        assert "DEATHS_UNDER5_TOTAL" in inventory["outcome_id"].tolist()
        assert "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL" in inventory["outcome_id"].tolist()
        assert "ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL" in inventory["outcome_id"].tolist()
        assert "CHILD_DIRECT_HEAT_HOSPITALIZATION_TOTAL" in inventory["outcome_id"].tolist()
        assert "CHILD_INDIRECT_HEAT_HOSPITALIZATION_TOTAL" in inventory["outcome_id"].tolist()
        assert "CHILD_EXCESSIVE_NATURAL_HEAT_X30_HOSPITALIZATION_TOTAL" in inventory["outcome_id"].tolist()
        assert "DEATHS_0_27D" not in inventory["outcome_id"].tolist()
        assert inventory.loc[inventory["outcome_id"].eq("ACUTE_UTIL"), "denominator_id"].iloc[0] == "under_five_population"
        assert inventory.loc[inventory["outcome_id"].eq("DEATHS_NEONATAL"), "denominator_id"].iloc[0] == "annual_live_births"
        assert inventory.loc[inventory["outcome_id"].eq("DEATHS_28_364D"), "denominator_id"].iloc[0] == "annual_live_births"
        assert inventory.loc[inventory["outcome_id"].eq("DEATHS_1_4Y"), "denominator_id"].iloc[0] == "under_five_population"
        assert inventory.loc[inventory["outcome_id"].eq("DEATHS_UNDER5_TOTAL"), "denominator_id"].iloc[0] == "under_five_population"
        assert inventory.loc[inventory["outcome_id"].eq("PRENATAL_VISITS_8PLUS_FLAG"), "denominator_id"].iloc[0] == "annual_live_births"
        assert inventory.loc[inventory["outcome_id"].eq("UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL"), "denominator_id"].iloc[0] == "under_five_population"
        assert inventory.loc[inventory["outcome_id"].eq("ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL"), "denominator_id"].iloc[0] == "under_five_population"
        assert inventory.loc[inventory["outcome_id"].eq("CHILD_DIRECT_HEAT_HOSPITALIZATION_TOTAL"), "denominator_id"].iloc[0] == "under_five_population"
        assert inventory.loc[inventory["outcome_id"].eq("CHILD_INDIRECT_HEAT_HOSPITALIZATION_TOTAL"), "denominator_id"].iloc[0] == "under_five_population"
        assert inventory.loc[inventory["outcome_id"].eq("CHILD_EXCESSIVE_NATURAL_HEAT_X30_HOSPITALIZATION_TOTAL"), "denominator_id"].iloc[0] == "under_five_population"
        assert inventory.loc[inventory["outcome_id"].eq("PRETERM_FLAG"), "denominator_id"].iloc[0] == "annual_live_births"
        assert inventory.loc[inventory["outcome_id"].eq("VERY_PRETERM_FLAG"), "denominator_id"].iloc[0] == "annual_live_births"
        assert inventory.loc[inventory["outcome_id"].eq("HOSP_U5"), "denominator_id"].iloc[0] == "under_five_population"
        assert inventory.loc[inventory["outcome_id"].eq("PRENATAL_VISITS_8PLUS_FLAG"), "seasonality_profile_id"].iloc[0] == "perinatal_delivery_aligned"
        assert inventory.loc[inventory["outcome_id"].eq("DEATHS_28_364D"), "seasonality_profile_id"].iloc[0] == "acute_daily_standard"
        assert inventory.loc[inventory["outcome_id"].eq("ANY_MENTION_DIRECT_HEAT_ANY_TOTAL"), "seasonality_profile_id"].iloc[0] == "acute_daily_standard"
        assert inventory.loc[inventory["outcome_id"].eq("CHILD_DIRECT_HEAT_HOSPITALIZATION_TOTAL"), "seasonality_profile_id"].iloc[0] == "acute_daily_standard"

        denominator_contract = pd.read_csv(root / report["outputs"]["denominator_contract"])
        assert (
            denominator_contract.loc[
                denominator_contract["outcome_id"].eq("STILLBIRTH_PRIMARY_GTE28_FLAG"),
                "denominator_id",
            ].iloc[0]
            == "annual_total_births_primary"
        )
        assert (
            denominator_contract.loc[
                denominator_contract["outcome_id"].eq("STILLBIRTH_SENS_GTE22_FLAG"),
                "denominator_id",
            ].iloc[0]
            == "annual_total_births_sensitivity_gte22"
        )


def test_build_step0_freeze_fails_when_release_not_pass() -> None:
    source_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _copy_step0_bundle(source_root, root)
        _write_dashboard_manifest(root)
        report_path = root / "gold_release_candidate_report.md"
        report_path.write_text(
            report_path.read_text(encoding="utf-8").replace("- Overall status: `PASS`", "- Overall status: `FAIL`"),
            encoding="utf-8",
        )

        report = build_step0_freeze(root)

        assert report["status"] == "FAIL"
        assert report["gates"]["gold_release_pass"] is False


def test_build_step0_freeze_fails_for_missing_seasonality_profile() -> None:
    source_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _copy_step0_bundle(source_root, root)
        _write_dashboard_manifest(root)
        inventory_path = root / "config" / "analysis" / "outcome_inventory.yaml"
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
        payload["outcomes"][0]["seasonality_profile_id"] = "missing_profile"
        inventory_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        report = build_step0_freeze(root)

        assert report["status"] == "FAIL"
        assert report["gates"]["seasonality_profiles_complete"] is False


def test_build_step0_freeze_fails_for_locked_denominator_rule_violation() -> None:
    source_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _copy_step0_bundle(source_root, root)
        _write_dashboard_manifest(root)
        inventory_path = root / "config" / "analysis" / "outcome_inventory.yaml"
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
        for outcome in payload["outcomes"]:
            if outcome["outcome_id"] == "ACUTE_UTIL":
                outcome["denominator_id"] = "not_applicable_count_or_descriptor"
                break
        inventory_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        report = build_step0_freeze(root)

        assert report["status"] == "FAIL"
        assert report["gates"]["locked_denominator_rules_respected"] is False
        assert report["locked_denominator_rule_mismatches"] == [
            {
                "outcome_id": "ACUTE_UTIL",
                "expected_denominator_id": "under_five_population",
                "observed_denominator_id": "not_applicable_count_or_descriptor",
            }
        ]


def test_cli_smoke_runs_step_0_freeze_and_steps_run() -> None:
    source_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _copy_step0_bundle(source_root, root)
        _write_dashboard_manifest(root)

        stdout = io.StringIO()
        with patch("heat_mnch.cli.root_path", return_value=root), redirect_stdout(stdout):
            exit_code = main(["step-0-freeze"])

        assert exit_code == 0
        payload = json.loads(stdout.getvalue())
        assert payload["status"] == "PASS"

        stdout = io.StringIO()
        with patch("heat_mnch.cli.root_path", return_value=root), redirect_stdout(stdout):
            exit_code = main(["steps-run", "--through", "step0"])

        assert exit_code == 0
        payload = json.loads(stdout.getvalue())
        assert payload["run_mode"] == "through_step0"
        assert payload["step0"]["status"] == "PASS"


def test_step0_preserves_dashboard_manifest_and_statcalc_ids() -> None:
    source_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _copy_step0_bundle(source_root, root)
        manifest_path = _write_dashboard_manifest(root)
        dashboard_manifest_before = manifest_path.read_text(encoding="utf-8")

        report = build_step0_freeze(root)

        assert report["status"] == "PASS"
        assert report["gates"]["dashboard_contract_unchanged"] is True
        assert report["gates"]["statcalc_recipe_ids_unchanged"] is True
        assert manifest_path.read_text(encoding="utf-8") == dashboard_manifest_before


def test_cli_blocks_legacy_track_commands() -> None:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["track-a-screening"])

    assert exit_code == 2
    payload = json.loads(stdout.getvalue())
    assert payload["status"] == "BLOCKED"
    assert payload["reason"] == "legacy_pipeline_disabled"

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["track-b-run", "--through-stage", "b8"])

    assert exit_code == 2
    payload = json.loads(stdout.getvalue())
    assert payload["status"] == "BLOCKED"
    assert payload["command"] == "track-b-run"
