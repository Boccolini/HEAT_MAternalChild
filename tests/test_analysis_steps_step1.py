from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from heat_mnch.analysis_steps.step1 import build_step1_descriptive_atlas
from heat_mnch.cli import main


def _write_json_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_step1_config_bundle(root: Path) -> None:
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
        root / "config" / "analysis" / "outcome_inventory.yaml",
        {
            "outcomes": [
                {
                    "outcome_id": "PRETERM_FLAG",
                    "public_label": "Preterm birth count",
                    "academic_variable_name": "preterm_birth_count",
                    "canonical_dataset": "gold_mnch_perinatal",
                    "canonical_variable": "PRETERM_FLAG",
                    "outcome_family": "perinatal_delivery_window_aggregate",
                    "analysis_role": "primary",
                    "readiness_endpoint_name": "PRETERM_FLAG",
                    "denominator_id": "annual_live_births",
                    "temporal_anchor": "delivery_event_anchor",
                    "seasonality_profile_id": "perinatal_delivery_aligned",
                    "round_path": "immediate_main_round",
                    "source_origin": "framework_outcome_map",
                },
                {
                    "outcome_id": "VERY_PRETERM_FLAG",
                    "public_label": "Very preterm birth count",
                    "academic_variable_name": "very_preterm_birth_count",
                    "canonical_dataset": "gold_mnch_perinatal",
                    "canonical_variable": "VERY_PRETERM_FLAG",
                    "outcome_family": "perinatal_delivery_window_aggregate",
                    "analysis_role": "sensitivity",
                    "readiness_endpoint_name": "VERY_PRETERM_FLAG",
                    "denominator_id": "annual_live_births",
                    "temporal_anchor": "delivery_event_anchor",
                    "seasonality_profile_id": "perinatal_delivery_aligned",
                    "round_path": "supplementary_sensitivity_round",
                    "source_origin": "framework_outcome_map",
                },
                {
                    "outcome_id": "HDP_SEVERE",
                    "public_label": "Severe hypertensive disorder of pregnancy count",
                    "academic_variable_name": "severe_hypertensive_disorder_of_pregnancy_count",
                    "canonical_dataset": "gold_mnch_perinatal",
                    "canonical_variable": "HDP_SEVERE_TOTAL",
                    "outcome_family": "maternal_operational",
                    "analysis_role": "descriptive_only",
                    "readiness_endpoint_name": None,
                    "denominator_id": "women_15_49_population",
                    "temporal_anchor": "observation_date",
                    "seasonality_profile_id": "descriptive_only_profile",
                    "round_path": "descriptive_only_round",
                    "source_origin": "framework_outcome_map",
                },
                {
                    "outcome_id": "ACUTE_UTIL",
                    "public_label": "Acute-care proxy count",
                    "academic_variable_name": "acute_care_proxy_count",
                    "canonical_dataset": "gold_mnch_panel",
                    "canonical_variable": "ACUTE_CARE_PROXY_TOTAL",
                    "outcome_family": "panel_proxy",
                    "analysis_role": "descriptive_only",
                    "readiness_endpoint_name": "ACUTE_CARE_PROXY_TOTAL",
                    "denominator_id": "under_five_population",
                    "temporal_anchor": "observation_date",
                    "seasonality_profile_id": "descriptive_only_profile",
                    "round_path": "descriptive_only_round",
                    "source_origin": "framework_outcome_map",
                },
                {
                    "outcome_id": "HOSP_U5",
                    "public_label": "Acute-care proxy count",
                    "academic_variable_name": "acute_care_proxy_count",
                    "canonical_dataset": "gold_mnch_panel",
                    "canonical_variable": "ACUTE_CARE_PROXY_TOTAL",
                    "outcome_family": "panel_proxy",
                    "analysis_role": "descriptive_only",
                    "readiness_endpoint_name": "ACUTE_CARE_PROXY_TOTAL",
                    "denominator_id": "under_five_population",
                    "temporal_anchor": "observation_date",
                    "seasonality_profile_id": "descriptive_only_profile",
                    "round_path": "descriptive_only_round",
                    "source_origin": "framework_outcome_map",
                },
            ]
        },
    )
    _write_json_yaml(
        root / "config" / "analysis" / "analysis_variable_aliases.yaml",
        {
            "aliases": [
                {
                    "canonical_dataset": "gold_mnch_perinatal",
                    "canonical_variable": "PRETERM_FLAG",
                    "academic_variable_name": "preterm_birth_count",
                    "academic_label": "Preterm birth count",
                    "description": "Synthetic preterm daily count.",
                    "units": "births per municipality-day",
                    "public_role": "outcome",
                    "outcome_family": "perinatal_delivery_window_aggregate",
                    "exposure_family": None,
                    "provenance_only": False,
                },
                {
                    "canonical_dataset": "gold_mnch_perinatal",
                    "canonical_variable": "VERY_PRETERM_FLAG",
                    "academic_variable_name": "very_preterm_birth_count",
                    "academic_label": "Very preterm birth count",
                    "description": "Synthetic very preterm daily count.",
                    "units": "births per municipality-day",
                    "public_role": "outcome",
                    "outcome_family": "perinatal_delivery_window_aggregate",
                    "exposure_family": None,
                    "provenance_only": False,
                },
                {
                    "canonical_dataset": "gold_mnch_perinatal",
                    "canonical_variable": "HDP_SEVERE_TOTAL",
                    "academic_variable_name": "severe_hypertensive_disorder_of_pregnancy_count",
                    "academic_label": "Severe hypertensive disorder of pregnancy count",
                    "description": "Synthetic severe HDP daily count.",
                    "units": "admissions per municipality-day",
                    "public_role": "outcome",
                    "outcome_family": "maternal_operational",
                    "exposure_family": None,
                    "provenance_only": False,
                },
                {
                    "canonical_dataset": "gold_mnch_panel",
                    "canonical_variable": "ACUTE_CARE_PROXY_TOTAL",
                    "academic_variable_name": "acute_care_proxy_count",
                    "academic_label": "Acute-care proxy count",
                    "description": "Synthetic acute-care proxy count.",
                    "units": "events per municipality-day",
                    "public_role": "outcome",
                    "outcome_family": "panel_proxy",
                    "exposure_family": None,
                    "provenance_only": False,
                },
            ]
        },
    )
    _write_json_yaml(
        root / "config" / "analysis" / "seasonality_contract.yaml",
        {
            "profiles": [
                {
                    "seasonality_profile_id": "perinatal_delivery_aligned",
                    "outcome_family": "perinatal_delivery_window_aggregate",
                    "outcome_id_optional": None,
                    "time_anchor": "delivery_event_anchor",
                    "main_controls": ["calendar_year", "event_or_delivery_month"],
                    "robustness_controls": [],
                    "required_from_step": "step0",
                    "notes": "Synthetic delivery-aligned profile.",
                },
                {
                    "seasonality_profile_id": "descriptive_only_profile",
                    "outcome_family": "descriptive_only",
                    "outcome_id_optional": None,
                    "time_anchor": "context_dependent",
                    "main_controls": ["monthly_and_seasonal_descriptive_screen_only"],
                    "robustness_controls": [],
                    "required_from_step": "step0",
                    "notes": "Synthetic descriptive-only profile.",
                },
            ]
        },
    )
    _write_json_yaml(
        root / "config" / "analysis" / "escalation_protocol.yaml",
        {
            "flagship_protocol": {
                "analysis_id": "synthetic_event_level",
                "journal_target": "Synthetic",
                "flagship_question": "Synthetic",
                "co_primary_outcomes": ["PRETERM_FLAG"],
                "supporting_secondary_outcomes": [],
                "primary_exposure_family": "TMAX_C",
                "secondary_exposure_families": [],
                "primary_main_text_estimates": {"PRETERM_FLAG": "days_0_6_before_event_mean_tmax_c"},
            }
        },
    )


def _write_step0_inputs(root: Path) -> None:
    (root / "artifacts" / "status").mkdir(parents=True, exist_ok=True)
    (root / "gold_release_candidate_manifests").mkdir(parents=True, exist_ok=True)
    (root / "gold_release_candidate_report.md").write_text(
        "\n".join(
            [
                "# Synthetic release report",
                "- Overall status: `PASS`",
                "- Ready now endpoints: `3`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "endpoint_name": "PRETERM_FLAG",
                "status": "Ready now",
                "gold_table": "gold_mnch_perinatal",
                "canonical_source": "synthetic",
                "required_fields": "PRETERM_FLAG",
                "required_fields_present_yes_no": "yes",
                "denominator_rule": "daily live births",
                "denominator_valid_yes_no": "yes",
                "climate_enrichment_complete_yes_no": "yes",
                "internal_validation_pass_fail": "pass",
                "external_validation_pass_fail": "pass",
                "primary_or_sensitivity": "PRIMARY",
                "age_scope": "all births",
                "blocker": "",
                "next_corrective_action": "",
            },
            {
                "endpoint_name": "VERY_PRETERM_FLAG",
                "status": "Ready now",
                "gold_table": "gold_mnch_perinatal",
                "canonical_source": "synthetic",
                "required_fields": "VERY_PRETERM_FLAG",
                "required_fields_present_yes_no": "yes",
                "denominator_rule": "daily live births",
                "denominator_valid_yes_no": "yes",
                "climate_enrichment_complete_yes_no": "yes",
                "internal_validation_pass_fail": "pass",
                "external_validation_pass_fail": "pass",
                "primary_or_sensitivity": "SENSITIVITY",
                "age_scope": "all births",
                "blocker": "",
                "next_corrective_action": "",
            },
            {
                "endpoint_name": "ACUTE_CARE_PROXY_TOTAL",
                "status": "Ready now",
                "gold_table": "gold_mnch_panel",
                "canonical_source": "synthetic",
                "required_fields": "ACUTE_CARE_PROXY_TOTAL",
                "required_fields_present_yes_no": "yes",
                "denominator_rule": "not_applicable_count_or_descriptor",
                "denominator_valid_yes_no": "yes",
                "climate_enrichment_complete_yes_no": "yes",
                "internal_validation_pass_fail": "pass",
                "external_validation_pass_fail": "pass",
                "primary_or_sensitivity": "SENSITIVITY",
                "age_scope": "comparative",
                "blocker": "",
                "next_corrective_action": "",
            },
        ]
    ).to_csv(root / "endpoint_readiness_ledger.csv", index=False)
    pd.DataFrame(
        [
            {
                "table": "gold_mnch_perinatal",
                "year": 2024,
                "layer": "gold",
                "status": "PASS",
                "output_gold_path": "data/gold/gold_mnch_perinatal/YEAR=2024/gold_mnch_perinatal_2024.parquet",
                "rows": 240,
                "variable_count": 10,
                "documented_variable_count": 10,
                "schema_consistent_with_dictionary_yes_no": "yes",
                "required_documented_yes_no": "yes",
                "climate_enrichment_complete_yes_no": "yes",
                "internal_validation_pass_fail": "pass",
                "external_validation_pass_fail": "pass",
                "blocking_issue": "",
            }
        ]
    ).to_csv(root / "gold_stage5_dictionary_matrix.csv", index=False)
    manifest_path = root / "artifacts" / "dashboard" / "v1" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"artifacts": {"dashboard_metrics": "synthetic.parquet"}}, indent=2), encoding="utf-8")


def _base_rows(municipality_count: int, fallback_value: int = 0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for municipality_index in range(municipality_count):
        municipality_code = f"{1100000 + municipality_index:07d}"
        for month in range(1, 13):
            rows.append(
                {
                    "municipality_code": municipality_code,
                    "calendar_month": month,
                    "maximum_daily_temperature_c": 30.0 + month / 10.0,
                    "mean_daily_relative_humidity_pct": 60.0 + month,
                    "heatwave_duration_days_p95_threshold": float(month % 3),
                    "approximate_heat_index_c": 34.0 + month / 5.0,
                    "any_climate_fallback_imputation_flag": fallback_value,
                    "denominator_zero_flag": 0,
                    "denominator_missing_flag": 0,
                    "rate_not_computable_flag": 0,
                    "offset_valid_flag": 1,
                }
            )
    return rows


def _write_marts(root: Path, municipality_count: int = 20) -> None:
    marts_root = root / "artifacts" / "analysis" / "shared_marts"
    marts_root.mkdir(parents=True, exist_ok=True)

    perinatal_rows = _base_rows(municipality_count)
    for row in perinatal_rows:
        row["preterm_birth_count"] = 1
        row["very_preterm_birth_count"] = 1
        row["daily_live_birth_count"] = 10
        row["annual_live_birth_count"] = 120
        row["log_annual_live_birth_offset"] = 4.787491742782046
    perinatal_path = marts_root / "perinatal_delivery_window_mart" / "YEAR=2024" / "perinatal_delivery_window_mart_2024.parquet"
    perinatal_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(perinatal_rows).to_parquet(perinatal_path, index=False)

    maternal_rows = _base_rows(municipality_count)
    for row in maternal_rows:
        row["severe_hypertensive_disorder_of_pregnancy_count"] = 1
        row["women_age_15_49_population_count"] = 1000
    maternal_path = marts_root / "maternal_operational_mart" / "YEAR=2024" / "maternal_operational_mart_2024.parquet"
    maternal_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(maternal_rows).to_parquet(maternal_path, index=False)

    acute_rows = _base_rows(municipality_count)
    for row in acute_rows:
        row["acute_care_proxy_count"] = 2
        row["under_five_population_count"] = 500
        row["log_under_five_population_offset"] = 6.214608098422191
    acute_path = marts_root / "acute_daily_mart" / "YEAR=2024" / "acute_daily_mart_2024.parquet"
    acute_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(acute_rows).to_parquet(acute_path, index=False)


def _write_sparse_perinatal_mart(root: Path) -> None:
    marts_root = root / "artifacts" / "analysis" / "shared_marts"
    rows = _base_rows(2)
    for row in rows:
        row["preterm_birth_count"] = 1
        row["very_preterm_birth_count"] = 1
        row["daily_live_birth_count"] = 10
        row["annual_live_birth_count"] = 12
        row["log_annual_live_birth_offset"] = 2.4849066497880004
    sparse_path = marts_root / "perinatal_delivery_window_mart" / "YEAR=2024" / "perinatal_delivery_window_mart_2024.parquet"
    sparse_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(sparse_path, index=False)


def _build_fixture(root: Path, municipality_count: int = 20) -> None:
    _write_step1_config_bundle(root)
    _write_step0_inputs(root)
    _write_marts(root, municipality_count=municipality_count)


def test_build_step1_descriptive_atlas_writes_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _build_fixture(root, municipality_count=20)

        report = build_step1_descriptive_atlas(root)

        assert report["status"] == "PASS"
        assert report["gates"]["analysis_unit_contract_consistent"] is True
        assert report["gates"]["primary_outcomes_have_baseline_support"] is True
        assert (root / report["outputs"]["outcome_descriptive_dossier"]).exists()
        assert (root / report["outputs"]["monthly_outcome_atlas"]).exists()
        assert (root / report["outputs"]["seasonal_outcome_atlas"]).exists()
        assert (root / report["outputs"]["denominator_integrity_audit"]).exists()
        assert (root / report["outputs"]["summary_workbook"]).exists()

        dossier = pd.read_csv(root / report["outputs"]["outcome_descriptive_dossier"])
        assert "preterm_birth_count" in dossier["analysis_unit_id"].tolist()
        assert dossier.loc[dossier["analysis_unit_id"].eq("preterm_birth_count"), "total_count"].iloc[0] == 240.0


def test_build_step1_descriptive_atlas_flags_sparse_primary_support() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _build_fixture(root, municipality_count=20)
        _write_sparse_perinatal_mart(root)

        report = build_step1_descriptive_atlas(root)

        assert report["status"] == "FAIL"
        assert report["gates"]["primary_outcomes_have_baseline_support"] is False
        sparse_flags = pd.read_csv(root / report["outputs"]["sparse_outcome_flags"])
        preterm_row = sparse_flags.loc[sparse_flags["analysis_unit_id"].eq("preterm_birth_count")].iloc[0]
        assert bool(preterm_row["any_sparse_flag"]) is True
        assert preterm_row["recommendation"] == "downgrade_before_step2"


def test_cli_smoke_runs_step1_and_steps_run() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _build_fixture(root, municipality_count=20)

        stdout = io.StringIO()
        with patch("heat_mnch.cli.root_path", return_value=root), redirect_stdout(stdout):
            exit_code = main(["step-1-descriptive-atlas"])

        assert exit_code == 0
        payload = json.loads(stdout.getvalue())
        assert payload["status"] == "PASS"

        stdout = io.StringIO()
        with patch("heat_mnch.cli.root_path", return_value=root), redirect_stdout(stdout):
            exit_code = main(["steps-run", "--through", "step1"])

        assert exit_code == 0
        payload = json.loads(stdout.getvalue())
        assert payload["run_mode"] == "through_step1"
        assert payload["step0"]["status"] == "PASS"
        assert payload["step1"]["status"] == "PASS"
