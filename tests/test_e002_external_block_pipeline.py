from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/Users/digicode.agent/Documents/HEAT_MAternalChild")
REGISTRY_DIR = ROOT / "artifacts/analysis/step9_rebuild/registries"
RESULTS_DIR = ROOT / "artifacts/analysis/step9_rebuild/results/remote_benchmarks/e002"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_block_surfaces_exist_and_cover_135_year_uf_blocks() -> None:
    manifest = _read_csv(REGISTRY_DIR / "e002_external_block_manifest.csv")
    qc = _read_csv(REGISTRY_DIR / "e002_external_block_qc.csv")
    stage = _read_csv(REGISTRY_DIR / "e002_external_block_stage_log.csv")

    assert len(manifest) == 135
    assert len(qc) == 135
    assert len(stage) == 135

    assert len({row["block_id"] for row in manifest}) == 135
    assert len({row["block_id"] for row in qc}) == 135
    assert len({row["block_id"] for row in stage}) == 135


def test_manifest_and_stage_outputs_are_complete() -> None:
    manifest = _read_csv(REGISTRY_DIR / "e002_external_block_manifest.csv")
    stage = _read_csv(REGISTRY_DIR / "e002_external_block_stage_log.csv")

    assert all(row["expected_month_count"] == "12" for row in manifest)
    assert all(row["observed_month_count"] == "12" for row in manifest)
    assert all(row["all_source_months_present"] == "yes" for row in manifest)

    for row in stage:
        assert row["extraction_status"] == "completed"
        assert row["qc_status"] == "completed"
        assert row["artifact_status"] == "written"
        assert (ROOT / row["stage_artifact_path"]).exists()


def test_qc_marks_plurality_invariant_and_stage_rows_match() -> None:
    qc = _read_csv(REGISTRY_DIR / "e002_external_block_qc.csv")
    stage = {
        row["block_id"]: row
        for row in _read_csv(REGISTRY_DIR / "e002_external_block_stage_log.csv")
    }

    for row in qc:
        assert row["plurality_code_invariant"] == "yes"
        assert "plurality_code" in row["invariant_terms"]
        assert row["two_level_exact_fit_eligible"] == "no"
        assert stage[row["block_id"]]["stage_rows_written"] == row["row_count_after_policy"]


def test_comparison_surface_tracks_simple_baseline_and_complete_stepup() -> None:
    comparison = _read_csv(RESULTS_DIR / "e002_remote_benchmark_comparison.csv")
    assert len(comparison) == 1
    row = comparison[0]

    assert row["slice_name"] == "year2020_states_12_14_16_remote"
    assert row["simple_row_count"] == "393683"
    assert row["simple_event_count"] == "3804"
    assert row["simple_singular_flag"] == "FALSE"
    assert row["comparison_status"] == "complete"
    assert row["gate_status"] == "ready_for_external_ladder"
    assert row["stepup_row_count"] == "393683"
    assert row["stepup_event_count"] == "3804"
    assert row["stepup_singular_flag"] == "FALSE"
