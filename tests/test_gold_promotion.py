from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from heat_mnch.gold_promotion import (
    _approved_stage2_mortality,
    _approved_stage3_panel,
    _approved_stage4_derived,
    _approved_stage5_dictionary,
    _perinatal_schema_blockers,
)


class GoldPromotionBlockerTest(unittest.TestCase):
    def test_perinatal_blockers_flag_missing_upstream_gold_prerequisites(self) -> None:
        sih_columns = {
            "MUNICIPALITY_CODE",
            "DATE",
            "YEAR",
            "SOURCE_SIH",
            "QC_STATUS",
        }
        dictionary_groups = {
            "HDP_ANY_PRIMARY",
            "HDP_SEVERE_PRIMARY",
            "PRETERM_LABOR_PRIMARY",
            "THREATENED_PRETERM_LABOR_SENSITIVITY",
            "APH_ANTEPARTUM_PRIMARY",
            "GDM_GESTATIONAL_PRIMARY",
            "DIABETES_PREGNANCY_BROAD_SENSITIVITY",
        }

        blockers = _perinatal_schema_blockers(sih_columns, dictionary_groups)

        joined = " | ".join(blockers)
        self.assertIn("missing required maternal core columns", joined)
        self.assertIn("gestational-HTN, preeclampsia, and eclampsia subtype groups", joined)

    def test_perinatal_blockers_clear_when_upstream_fields_are_present(self) -> None:
        sih_columns = {
            "MUNICIPALITY_CODE",
            "DATE",
            "YEAR",
            "MONTH",
            "OBSTETRIC_ADMISSIONS_TOTAL",
            "OBSTETRIC_URGENT_TOTAL",
            "OBSTETRIC_ICU_TOTAL",
            "HDP_ANY_TOTAL",
            "HDP_SEVERE_TOTAL",
            "HDP_GESTATIONAL_HTN_TOTAL",
            "HDP_PREECLAMPSIA_TOTAL",
            "HDP_ECLAMPSIA_TOTAL",
            "PRETERM_LABOR_TOTAL",
            "PRETERM_LABOR_URGENT_TOTAL",
            "PRETERM_LABOR_ICU_TOTAL",
            "PRETERM_LABOR_DEATH_TOTAL",
            "APH_ANTEPARTUM_TOTAL",
            "APH_ANTEPARTUM_URGENT_TOTAL",
            "APH_ANTEPARTUM_ICU_TOTAL",
            "APH_ANTEPARTUM_DEATH_TOTAL",
            "GDM_GESTATIONAL_TOTAL",
            "GDM_GESTATIONAL_URGENT_TOTAL",
            "GDM_GESTATIONAL_ICU_TOTAL",
            "GDM_GESTATIONAL_DEATH_TOTAL",
            "DIABETES_PREGNANCY_BROAD_TOTAL",
            "SOURCE_SIH",
            "QC_STATUS",
        }
        dictionary_groups = {
            "HDP_ANY_PRIMARY",
            "HDP_SEVERE_PRIMARY",
            "PRETERM_LABOR_PRIMARY",
            "THREATENED_PRETERM_LABOR_SENSITIVITY",
            "APH_ANTEPARTUM_PRIMARY",
            "GDM_GESTATIONAL_PRIMARY",
            "DIABETES_PREGNANCY_BROAD_SENSITIVITY",
            "HDP_GESTATIONAL_HTN_PRIMARY",
            "HDP_PREECLAMPSIA_PRIMARY",
            "HDP_ECLAMPSIA_PRIMARY",
        }

        blockers = _perinatal_schema_blockers(sih_columns, dictionary_groups)

        self.assertEqual(blockers, [])


class GoldPromotionStageApprovalTest(unittest.TestCase):
    def test_approved_stage2_mortality_passes_with_complete_package(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "artifacts" / "manifests").mkdir(parents=True, exist_ok=True)
            (root / "gold_stage2_mortality_manifests").mkdir(parents=True, exist_ok=True)
            (root / "gold_stage2_mortality_report.md").write_text("# report\n", encoding="utf-8")
            (root / "gold_stage2_mortality_ledger.csv").write_text(
                "stage,table,year,status,decision,blocking_issue\n"
                "canonical_gold,gold_mnch_mortality,2024,PASS,materialized,\n",
                encoding="utf-8",
            )
            (root / "artifacts" / "manifests" / "gold_stage2_mortality_manifest.json").write_text(
                json.dumps({"status": "PASS", "years": [2024]}),
                encoding="utf-8",
            )
            (root / "gold_stage2_mortality_manifests" / "manifest_index.json").write_text(
                json.dumps({"entry_count": 1}),
                encoding="utf-8",
            )

            approval = _approved_stage2_mortality(root)

            self.assertTrue(approval["ok"])
            self.assertEqual(approval["status"], "PASS")

    def test_approved_stage3_panel_passes_with_complete_package(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "artifacts" / "manifests").mkdir(parents=True, exist_ok=True)
            (root / "gold_stage3_panel_manifests").mkdir(parents=True, exist_ok=True)
            (root / "gold_stage3_panel_report.md").write_text("# report\n", encoding="utf-8")
            (root / "gold_stage3_panel_ledger.csv").write_text(
                "stage,table,year,status,decision,blocking_issue\n"
                "canonical_gold,gold_mnch_panel,2024,PASS,materialized,\n",
                encoding="utf-8",
            )
            (root / "artifacts" / "manifests" / "gold_stage3_panel_manifest.json").write_text(
                json.dumps({"status": "PASS", "years": [2024]}),
                encoding="utf-8",
            )
            (root / "gold_stage3_panel_manifests" / "manifest_index.json").write_text(
                json.dumps({"entry_count": 1}),
                encoding="utf-8",
            )

            approval = _approved_stage3_panel(root)

            self.assertTrue(approval["ok"])
            self.assertEqual(approval["status"], "PASS")

    def test_approved_stage4_derived_passes_with_complete_package(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "artifacts" / "manifests").mkdir(parents=True, exist_ok=True)
            (root / "gold_stage4_derived_manifests").mkdir(parents=True, exist_ok=True)
            (root / "gold_stage4_derived_report.md").write_text("# report\n", encoding="utf-8")
            (root / "gold_stage4_derived_ledger.csv").write_text(
                "stage,table,year,status,decision,blocking_issue\n"
                "derived_gold,gold_analysis_child_u5,2024,PASS,materialized,\n",
                encoding="utf-8",
            )
            (root / "artifacts" / "manifests" / "gold_stage4_derived_manifest.json").write_text(
                json.dumps({"status": "PASS", "years": [2024]}),
                encoding="utf-8",
            )
            (root / "gold_stage4_derived_manifests" / "manifest_index.json").write_text(
                json.dumps({"entry_count": 1}),
                encoding="utf-8",
            )

            approval = _approved_stage4_derived(root)

            self.assertTrue(approval["ok"])
            self.assertEqual(approval["status"], "PASS")

    def test_approved_stage5_dictionary_passes_with_complete_package(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "artifacts" / "manifests").mkdir(parents=True, exist_ok=True)
            (root / "gold_stage5_dictionary_manifests").mkdir(parents=True, exist_ok=True)
            (root / "gold_stage5_dictionary_report.md").write_text("# report\n", encoding="utf-8")
            (root / "gold_stage5_readiness_ledger.csv").write_text(
                "endpoint_name,status,gold_table,canonical_source,required_fields,required_fields_present_yes_no,denominator_rule,denominator_valid_yes_no,climate_enrichment_complete_yes_no,internal_validation_pass_fail,external_validation_pass_fail,primary_or_sensitivity,age_scope,blocker,next_corrective_action\n"
                "PRETERM_FLAG,Ready now,gold_mnch_perinatal,source,PRETERM_FLAG,yes,rule,yes,yes,pass,pass,PRIMARY,all births,,none\n",
                encoding="utf-8",
            )
            (root / "artifacts" / "manifests" / "gold_stage5_dictionary_manifest.json").write_text(
                json.dumps({"status": "PASS", "years": [2024]}),
                encoding="utf-8",
            )
            (root / "gold_stage5_dictionary_manifests" / "manifest_index.json").write_text(
                json.dumps({"entry_count": 1}),
                encoding="utf-8",
            )

            approval = _approved_stage5_dictionary(root)

            self.assertTrue(approval["ok"])
            self.assertEqual(approval["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
