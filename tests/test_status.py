from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from heat_mnch.status import init_status_note


class StatusNoteTest(unittest.TestCase):
    def _write_outcomes(self, root: Path, outcome_name: str = "preterm_birth") -> None:
        (root / "config" / "rules").mkdir(parents=True)
        (root / "artifacts" / "status").mkdir(parents=True)
        (root / "artifacts" / "manifests").mkdir(parents=True)
        (root / "config" / "rules" / "outcomes.yaml").write_text(
            '{\n'
            '  "outcomes": [\n'
            '    {\n'
            f'      "outcome": "{outcome_name}",\n'
            '      "gold_table": "gold_mnch_perinatal",\n'
            '      "required_gold_fields": ["NV_PRETERM_TOTAL"],\n'
            '      "gold_ready_criterion": "test"\n'
            '    }\n'
            '  ]\n'
            '}\n',
            encoding="utf-8",
        )

    def test_init_status_note_blocks_without_silver_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_outcomes(root)
            output_path = init_status_note(root)
            text = output_path.read_text(encoding="utf-8")
            self.assertIn("- Silver release gate: `MISSING`", text)
            self.assertIn("- Status: Blocked", text)
            self.assertIn("preterm_birth", text)

    def test_init_status_note_marks_outcome_ready_when_release_and_manifests_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_outcomes(root)
            (root / "artifacts" / "status" / "silver_release_candidate_report.md").write_text("# report\n", encoding="utf-8")
            (root / "artifacts" / "manifests" / "silver_sinasc.manifest.json").write_text("{}\n", encoding="utf-8")
            (root / "artifacts" / "manifests" / "silver_climate.manifest.json").write_text("{}\n", encoding="utf-8")
            (root / "artifacts" / "manifests" / "silver_release_manifest.json").write_text(
                '{\n'
                '  "status": "PASS",\n'
                '  "report_path": "artifacts/status/silver_release_candidate_report.md",\n'
                '  "included_manifests": {\n'
                '    "silver_sinasc": "artifacts/manifests/silver_sinasc.manifest.json",\n'
                '    "silver_climate": "artifacts/manifests/silver_climate.manifest.json"\n'
                '  }\n'
                '}\n',
                encoding="utf-8",
            )
            output_path = init_status_note(root)
            text = output_path.read_text(encoding="utf-8")
            self.assertIn("- Silver release gate: `PASS`", text)
            self.assertIn("- Status: Ready now", text)
            self.assertIn("artifacts/manifests/silver_sinasc.manifest.json", text)

    def test_init_status_note_marks_outcome_partially_ready_when_prereq_manifest_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_outcomes(root, outcome_name="obstetric_admissions")
            (root / "artifacts" / "status" / "silver_release_candidate_report.md").write_text("# report\n", encoding="utf-8")
            (root / "artifacts" / "manifests" / "silver_sih.manifest.json").write_text("{}\n", encoding="utf-8")
            (root / "artifacts" / "manifests" / "silver_release_manifest.json").write_text(
                '{\n'
                '  "status": "PASS",\n'
                '  "report_path": "artifacts/status/silver_release_candidate_report.md",\n'
                '  "included_manifests": {\n'
                '    "silver_sih": "artifacts/manifests/silver_sih.manifest.json"\n'
                '  }\n'
                '}\n',
                encoding="utf-8",
            )
            output_path = init_status_note(root)
            text = output_path.read_text(encoding="utf-8")
            self.assertIn("- Status: Partially ready", text)
            self.assertIn("silver_climate", text)
            self.assertIn("mnch_outcome_dictionary_icd10", text)


if __name__ == "__main__":
    unittest.main()
