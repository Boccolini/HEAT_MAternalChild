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
from heat_mnch.cli import main
from test_analysis_steps_step2 import (
    _write_step0_inputs_for_cli,
    _write_step1_artifacts,
    _write_step2_config_bundle,
    _write_shared_marts,
)


def _prepare_step3_fixture(root: Path) -> dict[str, object]:
    _write_step2_config_bundle(root)
    _write_step1_artifacts(root)
    _write_shared_marts(root)
    _write_step0_inputs_for_cli(root)
    return build_step2_baseline_models(root, step1_report={"status": "PASS"})


def test_build_step3_threshold_discovery_writes_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        step2_report = _prepare_step3_fixture(root)

        report = build_step3_threshold_discovery(root, step2_report=step2_report)

        assert report["status"] == "PASS"
        assert report["step"] == "step3_threshold_discovery"
        assert report["summary"]["approved_candidate_count"] == 5
        assert report["summary"]["eligible_pair_count"] == 10
        assert report["summary"]["screening_result_count"] == 68
        assert report["summary"]["signal_matrix_count"] == 10
        assert report["summary"]["heat_index_recommendation_count"] == 5
        assert report["summary"]["tmax_recommendation_count"] == 0
        assert report["gates"]["approved_candidates_screened"] is True
        assert report["gates"]["headline_recommendations_written"] is True
        assert report["gates"]["seasonality_controls_inherited"] is True
        assert report["gates"]["screening_results_written"] is True
        assert (root / report["outputs"]["screening_results"]).exists()
        assert (root / report["outputs"]["signal_matrix"]).exists()
        assert (root / report["outputs"]["family_threshold_comparison"]).exists()
        assert (root / report["outputs"]["decision_checklist"]).exists()
        assert (root / report["outputs"]["step3_report"]).exists()

        screening_results = pd.read_csv(root / report["outputs"]["screening_results"])
        signal_matrix = pd.read_csv(root / report["outputs"]["signal_matrix"])
        family_comparison = pd.read_csv(root / report["outputs"]["family_threshold_comparison"])
        assert screening_results["exposure_family_id"].nunique() == 2
        assert "acute_weekly_anomaly_correlation" in screening_results["screening_domain"].tolist()
        assert "perinatal_delivery_window_quintile_trend" in screening_results["screening_domain"].tolist()
        assert signal_matrix["analysis_unit_id"].nunique() == 5
        assert set(family_comparison["headline_family_recommendation"].tolist()) == {"heat_index"}


def test_cli_smoke_runs_step3_and_steps_run_through_step3() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        step2_report = _prepare_step3_fixture(root)

        stdout = io.StringIO()
        with (
            patch("heat_mnch.cli.root_path", return_value=root),
            patch("heat_mnch.analysis_steps.step3.build_step2_baseline_models", return_value=step2_report),
            redirect_stdout(stdout),
        ):
            exit_code = main(["step-3-threshold-discovery"])

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
            exit_code = main(["steps-run", "--through", "step3"])

        assert exit_code == 0
        payload = json.loads(stdout.getvalue())
        assert payload["run_mode"] == "through_step3"
        assert payload["step2"]["status"] == "PASS"
        assert payload["step3"]["status"] == "PASS"
