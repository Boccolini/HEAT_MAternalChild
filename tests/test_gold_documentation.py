from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from heat_mnch.gold_documentation import (
    build_gold_readiness_ledger,
    build_gold_table_summaries,
    build_gold_variable_dictionary,
)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


class GoldDocumentationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "artifacts" / "status").mkdir(parents=True, exist_ok=True)
        (self.root / "schemas").mkdir(parents=True, exist_ok=True)

        repo_root = Path(__file__).resolve().parents[1]
        for filename in (
            "gold_mnch_perinatal_diario_municipio.schema.csv",
            "gold_mnch_mortality_diario_municipio.schema.csv",
            "gold_mnch_panel_diario_municipio.schema.csv",
        ):
            shutil.copy2(repo_root / "schemas" / filename, self.root / "schemas" / filename)

        self._write_gold_outputs()
        self._write_stage_mini_reports()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_gold_outputs(self) -> None:
        perinatal = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "GEST_AGE_CANONICAL": 36.0,
                    "PRETERM_FLAG": 1,
                    "VERY_PRETERM_FLAG": 0,
                    "GEST_DAYS_LOST_BELOW_37W": 7.0,
                    "GEST_AGE_DERIVATION_NOTE": "completed weeks",
                    "GEST_DAYS_LOST_DERIVATION_NOTE": "37*7 minus canonical days",
                    "GEST_AGE_MISSING_FLAG": 0,
                    "LIVE_BIRTHS_TOTAL": 3,
                    "STILLBIRTH_TOTAL_PRIMARY_GTE28": 1,
                    "STILLBIRTH_TOTAL_GTE22_SENSITIVITY": 1,
                    "TOTAL_BIRTHS_PRIMARY": 4,
                    "TOTAL_BIRTHS_SENSITIVITY_GTE22": 4,
                    "OFFSET_LOG_TOTAL_BIRTHS_PRIMARY": math.log(4),
                    "OFFSET_LOG_TOTAL_BIRTHS_SENSITIVITY_GTE22": math.log(4),
                    "TMAX_C": None,
                    "TMIN_C": None,
                    "TMEAN_C": None,
                    "RH_MIN_PCT": None,
                    "RH_MEAN_PCT": 55.0,
                    "HEATWAVE_LEN_P95": 2.0,
                    "HEAT_INDEX_APPROX_C": 36.5,
                    "FONTE_CLIMA_FINAL": "NO_CLIMATE_BACKBONE_MATCH",
                    "CLIMATE_UPSTREAM_QUARANTINE_FLAG": 0,
                    "CLIMATE_RECORD_ANOMALY_FLAG": 0,
                    "CLIMATE_CORRECTION_PATH": "none",
                    "DENOMINATOR_ZERO_FLAG": 0,
                    "DENOMINATOR_MISSING_FLAG": 0,
                    "RATE_NOT_COMPUTABLE_FLAG": 0,
                    "OFFSET_VALID_FLAG": 1,
                    "QC_STATUS": "PASS",
                }
            ]
        )
        _write_parquet(
            self.root / "data" / "gold" / "gold_mnch_perinatal" / "YEAR=2024" / "gold_mnch_perinatal_2024.parquet",
            perinatal,
        )

        mortality = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "DEATHS_NEONATAL": 1,
                    "DEATHS_U1": 1,
                    "DEATHS_U5": 2,
                    "LIVE_BIRTHS_TOTAL_ANNUAL": 100.0,
                    "POP_U5": 200.0,
                    "OFFSET_LOG_LIVE_BIRTHS": math.log(100.0),
                    "OFFSET_LOG_UNDER5_POPULATION": math.log(200.0),
                    "TMAX_C": 33.0,
                    "TMIN_C": 24.0,
                    "TMEAN_C": 28.5,
                    "RH_MIN_PCT": 44.0,
                    "RH_MEAN_PCT": 58.0,
                    "HEATWAVE_LEN_P95": 1.0,
                    "HEAT_INDEX_APPROX_C": 35.0,
                    "FONTE_CLIMA_FINAL": "mixed",
                    "CLIMATE_UPSTREAM_QUARANTINE_FLAG": 1,
                    "CLIMATE_RECORD_ANOMALY_FLAG": 0,
                    "CLIMATE_CORRECTION_PATH": "upstream_station_quarantine_to_era5",
                    "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL": 1,
                    "ANY_MENTION_DIRECT_HEAT_ANY_TOTAL": 1,
                    "UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_TOTAL": 1,
                    "ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL": 1,
                    "ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_TOTAL": 1,
                    "DENOMINATOR_ZERO_FLAG": 0,
                    "DENOMINATOR_MISSING_FLAG": 0,
                    "RATE_NOT_COMPUTABLE_FLAG": 0,
                    "OFFSET_VALID_FLAG": 1,
                    "QC_STATUS": "PASS",
                }
            ]
        )
        _write_parquet(
            self.root / "data" / "gold" / "gold_mnch_mortality" / "YEAR=2024" / "gold_mnch_mortality_2024.parquet",
            mortality,
        )

        panel = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "ACUTE_CARE_PROXY_TOTAL": 2,
                    "ACUTE_CARE_PROXY_LABEL": "proxy",
                    "TMAX_C": None,
                    "TMIN_C": None,
                    "TMEAN_C": None,
                    "RH_MIN_PCT": None,
                    "RH_MEAN_PCT": 55.0,
                    "HEATWAVE_LEN_P95": 2.0,
                    "HEAT_INDEX_APPROX_C": 36.5,
                    "FONTE_CLIMA_FINAL": "NO_CLIMATE_BACKBONE_MATCH",
                    "CLIMATE_UPSTREAM_QUARANTINE_FLAG": 0,
                    "CLIMATE_RECORD_ANOMALY_FLAG": 0,
                    "CLIMATE_CORRECTION_PATH": "none",
                    "DENOMINATOR_ZERO_FLAG": 0,
                    "DENOMINATOR_MISSING_FLAG": 0,
                    "RATE_NOT_COMPUTABLE_FLAG": 0,
                    "OFFSET_VALID_FLAG": 1,
                    "QC_STATUS": "PASS",
                }
            ]
        )
        _write_parquet(
            self.root / "data" / "gold" / "gold_mnch_panel" / "YEAR=2024" / "gold_mnch_panel_2024.parquet",
            panel,
        )

        child_view = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "DEATHS_0_27D": 1,
                    "DEATHS_28_364D": 0,
                    "DEATHS_1_4Y": 1,
                    "DEATHS_UNDER5_TOTAL": 2,
                    "LIVE_BIRTHS_TOTAL_ANNUAL": 100.0,
                    "POP_U5": 200.0,
                    "TMAX_C": 33.0,
                    "TMIN_C": 24.0,
                    "TMEAN_C": 28.5,
                    "RH_MIN_PCT": 44.0,
                    "RH_MEAN_PCT": 58.0,
                    "HEATWAVE_LEN_P95": 1.0,
                    "HEAT_INDEX_APPROX_C": 35.0,
                    "FONTE_CLIMA_FINAL": "mixed",
                    "CLIMATE_UPSTREAM_QUARANTINE_FLAG": 1,
                    "CLIMATE_RECORD_ANOMALY_FLAG": 0,
                    "CLIMATE_CORRECTION_PATH": "upstream_station_quarantine_to_era5",
                    "DENOMINATOR_ZERO_FLAG": 0,
                    "DENOMINATOR_MISSING_FLAG": 0,
                    "RATE_NOT_COMPUTABLE_FLAG": 0,
                    "OFFSET_VALID_FLAG": 1,
                    "CHILD_VIEW_AGE_LOGIC_NOTE": "0_59_MONTHS_EXPLICIT",
                    "VIEW_ROLE_NOTE": "analytical only",
                    "SOURCE_DERIVED_VIEW": "gold_analysis_child_u5",
                    "QC_STATUS": "PASS",
                }
            ]
        )
        _write_parquet(
            self.root / "data" / "gold" / "gold_analysis_child_u5" / "YEAR=2024" / "gold_analysis_child_u5_2024.parquet",
            child_view,
        )

        wra_15_49 = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "OBSTETRIC_ADMISSIONS_TOTAL": 3,
                    "HDP_ANY_TOTAL": 1,
                    "PRETERM_LABOR_TOTAL": 1,
                    "GDM_GESTATIONAL_TOTAL": 1,
                    "POP_WRA_15_49": 500.0,
                    "OFFSET_LOG_POP_WRA_15_49": math.log(500.0),
                    "TMAX_C": None,
                    "TMIN_C": None,
                    "TMEAN_C": None,
                    "RH_MIN_PCT": None,
                    "RH_MEAN_PCT": 55.0,
                    "HEATWAVE_LEN_P95": 2.0,
                    "HEAT_INDEX_APPROX_C": 36.5,
                    "FONTE_CLIMA_FINAL": "NO_CLIMATE_BACKBONE_MATCH",
                    "CLIMATE_UPSTREAM_QUARANTINE_FLAG": 0,
                    "CLIMATE_RECORD_ANOMALY_FLAG": 0,
                    "CLIMATE_CORRECTION_PATH": "none",
                    "DENOMINATOR_ZERO_FLAG": 0,
                    "DENOMINATOR_MISSING_FLAG": 0,
                    "RATE_NOT_COMPUTABLE_FLAG": 0,
                    "OFFSET_VALID_FLAG": 1,
                    "SOURCE_CANONICAL_GOLD": "gold_mnch_perinatal",
                    "VIEW_ROLE_NOTE": "analytical only",
                    "WRA_AGE_SCOPE_NOTE": "WOMEN_15_49_CANONICAL_VIEW",
                    "SENSITIVITY_ONLY_FLAG": 0,
                    "SOURCE_DERIVED_VIEW": "gold_analysis_wra_15_49",
                    "QC_STATUS": "PASS",
                }
            ]
        )
        _write_parquet(
            self.root / "data" / "gold" / "gold_analysis_wra_15_49" / "YEAR=2024" / "gold_analysis_wra_15_49_2024.parquet",
            wra_15_49,
        )

        wra_15_45 = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "OBSTETRIC_ADMISSIONS_TOTAL": 3,
                    "HDP_ANY_TOTAL": 1,
                    "PRETERM_LABOR_TOTAL": 1,
                    "GDM_GESTATIONAL_TOTAL": 1,
                    "POP_WRA_15_45_SENSITIVITY": 450.0,
                    "OFFSET_LOG_POP_WRA_15_45_SENSITIVITY": math.log(450.0),
                    "TMAX_C": None,
                    "TMIN_C": None,
                    "TMEAN_C": None,
                    "RH_MIN_PCT": None,
                    "RH_MEAN_PCT": 55.0,
                    "HEATWAVE_LEN_P95": 2.0,
                    "HEAT_INDEX_APPROX_C": 36.5,
                    "FONTE_CLIMA_FINAL": "NO_CLIMATE_BACKBONE_MATCH",
                    "CLIMATE_UPSTREAM_QUARANTINE_FLAG": 0,
                    "CLIMATE_RECORD_ANOMALY_FLAG": 0,
                    "CLIMATE_CORRECTION_PATH": "none",
                    "DENOMINATOR_ZERO_FLAG": 0,
                    "DENOMINATOR_MISSING_FLAG": 0,
                    "RATE_NOT_COMPUTABLE_FLAG": 0,
                    "OFFSET_VALID_FLAG": 1,
                    "SOURCE_CANONICAL_GOLD": "gold_mnch_perinatal",
                    "VIEW_ROLE_NOTE": "analytical sensitivity only",
                    "WRA_AGE_SCOPE_NOTE": "WOMEN_15_45_SENSITIVITY_ONLY_VIEW",
                    "SENSITIVITY_ONLY_FLAG": 1,
                    "SOURCE_DERIVED_VIEW": "gold_analysis_wra_15_45_sens",
                    "QC_STATUS": "PASS",
                }
            ]
        )
        _write_parquet(
            self.root / "data" / "gold" / "gold_analysis_wra_15_45_sens" / "YEAR=2024" / "gold_analysis_wra_15_45_sens_2024.parquet",
            wra_15_45,
        )

    def _write_stage_mini_reports(self) -> None:
        _write_csv(
            self.root / "artifacts" / "status" / "gold_stage1_perinatal_mini_report.csv",
            pd.DataFrame(
                [
                    {
                        "table": "gold_mnch_perinatal",
                        "year": 2024,
                        "STATUS": "PASS",
                        "internal_reconciliation_summary": "canonical_sinasc_event_used=True",
                        "external_validation_summary": "schema_required_columns_present=True",
                    }
                ]
            ),
        )
        _write_csv(
            self.root / "artifacts" / "status" / "gold_stage2_mortality_mini_report.csv",
            pd.DataFrame(
                [
                    {
                        "table": "gold_mnch_mortality",
                        "year": 2024,
                        "STATUS": "PASS",
                        "internal_reconciliation_summary": "age_scope_rollups_applied=True",
                        "external_validation_summary": "schema_required_columns_present=True",
                    }
                ]
            ),
        )
        _write_csv(
            self.root / "artifacts" / "status" / "gold_stage3_panel_mini_report.csv",
            pd.DataFrame(
                [
                    {
                        "table": "gold_mnch_panel",
                        "year": 2024,
                        "STATUS": "PASS",
                        "internal_reconciliation_summary": "panel_ready=True",
                        "external_validation_summary": "schema_required_columns_present=True",
                    }
                ]
            ),
        )
        _write_csv(
            self.root / "artifacts" / "status" / "gold_stage4_derived_mini_report.csv",
            pd.DataFrame(
                [
                    {
                        "view": "gold_analysis_child_u5",
                        "year": 2024,
                        "STATUS": "PASS",
                        "internal_reconciliation_summary": "age_logic_explicit=True",
                        "external_validation_summary": "canonical_table_not_overwritten=True",
                    },
                    {
                        "view": "gold_analysis_wra_15_49",
                        "year": 2024,
                        "STATUS": "PASS",
                        "internal_reconciliation_summary": "derived_from=gold_mnch_perinatal",
                        "external_validation_summary": "canonical_table_not_overwritten=True",
                    },
                    {
                        "view": "gold_analysis_wra_15_45_sens",
                        "year": 2024,
                        "STATUS": "PASS",
                        "internal_reconciliation_summary": "derived_from=gold_mnch_perinatal",
                        "external_validation_summary": "canonical_table_not_overwritten=True",
                    },
                ]
            ),
        )

    def test_gold_variable_dictionary_documents_required_variables_and_identifiers(self) -> None:
        result = build_gold_variable_dictionary(self.root, 2024, 2024)
        dictionary = result["dictionary"]

        self.assertTrue(result["required_documented"])
        self.assertEqual(result["missing_required_variables"], [])
        self.assertTrue(
            {
                "variable_name",
                "layer",
                "table_name",
                "variable_class",
                "plain_english_label",
                "definition",
                "unit",
                "key_role",
                "source_provenance",
                "primary_or_sensitivity",
                "expected_missingness",
                "outcome_family",
                "age_scope",
                "denominator_rule",
                "canonical_or_legacy",
                "notes_on_use",
            }.issubset(set(dictionary.columns))
        )
        required_variables = {
            "GEST_AGE_CANONICAL",
            "PRETERM_FLAG",
            "VERY_PRETERM_FLAG",
            "GEST_DAYS_LOST_BELOW_37W",
            "HEAT_INDEX_APPROX_C",
            "CLIMATE_UPSTREAM_QUARANTINE_FLAG",
            "CLIMATE_RECORD_ANOMALY_FLAG",
            "CLIMATE_CORRECTION_PATH",
            "DENOMINATOR_ZERO_FLAG",
            "DENOMINATOR_MISSING_FLAG",
            "RATE_NOT_COMPUTABLE_FLAG",
            "OFFSET_VALID_FLAG",
            "SOURCE_DERIVED_VIEW",
            "VIEW_ROLE_NOTE",
            "WRA_AGE_SCOPE_NOTE",
            "CHILD_VIEW_AGE_LOGIC_NOTE",
            "SENSITIVITY_ONLY_FLAG",
        }
        self.assertTrue(required_variables.issubset(set(dictionary["variable_name"])))

        gest_days_row = dictionary.loc[dictionary["variable_name"] == "GEST_DAYS_LOST_BELOW_37W"].iloc[0]
        self.assertEqual(str(gest_days_row["primary_or_sensitivity"]), "SENSITIVITY")
        correction_path_row = dictionary.loc[dictionary["variable_name"] == "CLIMATE_CORRECTION_PATH"].iloc[0]
        self.assertEqual(str(correction_path_row["primary_or_sensitivity"]), "DESCRIPTOR")

        derived_source_rows = dictionary.loc[dictionary["variable_name"] == "SOURCE_DERIVED_VIEW"]
        self.assertEqual(set(derived_source_rows["table_name"]), {"gold_analysis_child_u5", "gold_analysis_wra_15_49", "gold_analysis_wra_15_45_sens"})

    def test_gold_variable_dictionary_uses_requested_year_range_not_oldest_partition(self) -> None:
        old_perinatal = pd.DataFrame(
            [
                {
                    "MUNICIPALITY_CODE": "123456",
                    "DATE": "2023-01-01",
                    "YEAR": 2023,
                    "MONTH": 1,
                    "GEST_AGE_CANONICAL": 36.0,
                    "PRETERM_FLAG": 1,
                    "LIVE_BIRTHS_TOTAL": 3,
                    "RH_MEAN_PCT": 55.0,
                    "FONTE_CLIMA_FINAL": "NO_CLIMATE_BACKBONE_MATCH",
                    "QC_STATUS": "PASS",
                }
            ]
        )
        _write_parquet(
            self.root / "data" / "gold" / "gold_mnch_perinatal" / "YEAR=2023" / "gold_mnch_perinatal_2023.parquet",
            old_perinatal,
        )

        result = build_gold_variable_dictionary(self.root, 2024, 2024)
        dictionary = result["dictionary"]
        perinatal_2024_variables = dictionary.loc[
            dictionary["table_name"] == "gold_mnch_perinatal",
            "variable_name",
        ]

        self.assertIn("TMIN_C", set(perinatal_2024_variables))
        self.assertIn("TMEAN_C", set(perinatal_2024_variables))
        self.assertIn("RH_MIN_PCT", set(perinatal_2024_variables))

    def test_gold_readiness_ledger_marks_partial_when_climate_is_not_zero_missing(self) -> None:
        summaries = build_gold_table_summaries(self.root, 2024, 2024)
        ledger = build_gold_readiness_ledger(self.root, 2024, 2024, summaries)

        preterm = ledger.loc[ledger["endpoint_name"] == "PRETERM_FLAG"].iloc[0]
        neonatal = ledger.loc[ledger["endpoint_name"] == "DEATHS_NEONATAL"].iloc[0]
        wra = ledger.loc[ledger["endpoint_name"] == "OBSTETRIC_ADMISSIONS_TOTAL_WRA_15_49"].iloc[0]

        self.assertEqual(str(preterm["status"]), "Partially ready")
        self.assertEqual(str(preterm["climate_enrichment_complete_yes_no"]), "no")
        self.assertIn("climate enrichment is not zero-missing", str(preterm["blocker"]))

        self.assertEqual(str(neonatal["status"]), "Ready now")
        self.assertEqual(str(neonatal["climate_enrichment_complete_yes_no"]), "yes")
        self.assertEqual(str(neonatal["denominator_valid_yes_no"]), "yes")

        self.assertEqual(str(wra["status"]), "Partially ready")
        self.assertEqual(str(wra["denominator_valid_yes_no"]), "yes")
        self.assertEqual(str(wra["internal_validation_pass_fail"]), "pass")


if __name__ == "__main__":
    unittest.main()
