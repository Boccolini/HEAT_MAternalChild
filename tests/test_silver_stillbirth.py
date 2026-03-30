from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from heat_mnch.silver_stillbirth import (
    STILLBIRTH_REQUIRED_INPUT_FIELDS,
    build_canonical_stillbirth_individual_frame,
    promote_stillbirth_silver_range,
    stillbirth_daily_partition_file,
    stillbirth_individual_partition_file,
)


def _base_stillbirth_row(**overrides: str) -> dict[str, str]:
    row = {
        "BRONZE_SOURCE_SYSTEM": "datasus",
        "BRONZE_SOURCE_FILE": "DOFET20.dbc",
        "BRONZE_SOURCE_PATH": "data/raw/stillbirth/DOFET20.dbc",
        "BRONZE_SOURCE_MEMBER": "",
        "BRONZE_YEAR": "2020",
        "BRONZE_INGESTED_AT": "2026-03-19T00:00:00+00:00",
        "TIPOBITO": "1",
        "DTOBITO": "2020-04-03",
        "DTNASC": "",
        "CODMUNRES": "110020",
        "CODMUNOCOR": "110020",
        "IDADEMAE": "25",
        "ESCMAE2010": "",
        "ESCMAE": "",
        "SERIESCMAE": "",
        "SEMAGESTAC": "",
        "GESTACAO": "",
        "GRAVIDEZ": "1",
        "PARTO": "1",
        "OBITOPARTO": "1",
        "PESO": "2500",
        "SEXO": "1",
        "CONTADOR": "1",
    }
    row.update(overrides)
    return row


class SilverStillbirthDerivationTest(unittest.TestCase):
    def _write_bronze(self, root: Path, year: int, rows: list[dict[str, str]]) -> None:
        path = root / "data" / "bronze" / "stillbirth" / str(year) / f"stillbirth_{year}_source_preserving.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=STILLBIRTH_REQUIRED_INPUT_FIELDS, delimiter=";")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in STILLBIRTH_REQUIRED_INPUT_FIELDS})

    def test_exact_weeks_are_primary_analytic_stillbirth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_bronze(root, 2020, [_base_stillbirth_row(SEMAGESTAC="30", GESTACAO="4", DTNASC="2020-04-02")])
            frame = build_canonical_stillbirth_individual_frame(root, 2020)
            row = frame.iloc[0]
            self.assertEqual(float(row["GA_WEEKS_RAW_VALID"]), 30.0)
            self.assertEqual(float(row["GA_WEEKS_COMPLETED_FINAL"]), 30.0)
            self.assertEqual(row["GA_SOURCE"], "SEMAGESTAC_EXACT_WEEKS")
            self.assertEqual(int(row["STILLBIRTH_GTE28_PRIMARY_FLAG"]), 1)
            self.assertEqual(int(row["STILLBIRTH_GTE22_SENSITIVITY_FLAG"]), 1)
            self.assertEqual(row["STILLBIRTH_ANALYTIC_STATUS"], "PRIMARY_GTE28")
            self.assertEqual(int(row["GA_QC_DTOBITO_EQUALS_DTNASC_FLAG"]), 0)

    def test_gestacao_band_is_used_only_for_threshold_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_bronze(root, 2020, [_base_stillbirth_row(GESTACAO="2", CONTADOR="2")])
            frame = build_canonical_stillbirth_individual_frame(root, 2020)
            row = frame.iloc[0]
            self.assertTrue(pd.isna(row["GA_WEEKS_COMPLETED_FINAL"]))
            self.assertEqual(row["GA_SOURCE"], "GESTACAO_BAND_ONLY")
            self.assertEqual(int(row["STILLBIRTH_GTE28_PRIMARY_FLAG"]), 0)
            self.assertEqual(int(row["STILLBIRTH_GTE22_SENSITIVITY_FLAG"]), 1)
            self.assertEqual(row["STILLBIRTH_ANALYTIC_STATUS"], "SENSITIVITY_GTE22_ONLY")
            self.assertEqual(int(row["GA_QC_BAND_ONLY_FLAG"]), 1)

    def test_missing_gestational_age_keeps_row_and_null_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_bronze(root, 2020, [_base_stillbirth_row(CONTADOR="3", GESTACAO="", SEMAGESTAC="", PESO="")])
            frame = build_canonical_stillbirth_individual_frame(root, 2020)
            row = frame.iloc[0]
            self.assertEqual(row["GA_SOURCE"], "MISSING")
            self.assertEqual(int(row["GA_QC_MISSING_FLAG"]), 1)
            self.assertTrue(pd.isna(row["STILLBIRTH_GTE28_PRIMARY_FLAG"]))
            self.assertTrue(pd.isna(row["STILLBIRTH_GTE22_SENSITIVITY_FLAG"]))
            self.assertEqual(row["STILLBIRTH_ANALYTIC_STATUS"], "UNKNOWN_GESTATIONAL_AGE")
            self.assertEqual(int(row["BIRTH_WEIGHT_REPORTED_FLAG"]), 0)

    def test_native_round3_covariates_are_exposed_without_invented_parity_or_race(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_bronze(
                root,
                2020,
                [
                    _base_stillbirth_row(
                        SEMAGESTAC="30",
                        ESCMAE="4",
                        GRAVIDEZ="3",
                        SEXO="2",
                        IDADEMAE="37",
                    )
                ],
            )
            frame = build_canonical_stillbirth_individual_frame(root, 2020)
            row = frame.iloc[0]
            self.assertEqual(int(row["MATERNAL_AGE_YEARS"]), 37)
            self.assertEqual(int(row["MATERNAL_SCHOOLING_CODE_FINAL"]), 4)
            self.assertEqual(row["MATERNAL_SCHOOLING_SOURCE"], "ESCMAE")
            self.assertEqual(int(row["PLURALITY_CODE"]), 3)
            self.assertEqual(row["PLURALITY_CATEGORY"], "MULTIPLE")
            self.assertEqual(int(row["MULTIPLE_GESTATION_FLAG"]), 1)
            self.assertEqual(int(row["FETAL_SEX_CODE"]), 2)
            self.assertEqual(row["FETAL_SEX_CATEGORY"], "FEMALE")
            self.assertNotIn("PARITY_TOTAL_PRIOR", frame.columns)
            self.assertNotIn("MATERNAL_RACE_COLOR_CODE", frame.columns)


class SilverStillbirthIntegrationTest(unittest.TestCase):
    def _write_bronze(self, root: Path, year: int, rows: list[dict[str, str]]) -> None:
        path = root / "data" / "bronze" / "stillbirth" / str(year) / f"stillbirth_{year}_source_preserving.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=STILLBIRTH_REQUIRED_INPUT_FIELDS, delimiter=";")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in STILLBIRTH_REQUIRED_INPUT_FIELDS})

    def _write_sinasc_canonical_inputs(self, root: Path, year: int) -> None:
        event_rows = pd.DataFrame(
            [
                {"DATE": "2020-04-03", "CODMUNRES": "110020"},
                {"DATE": "2020-04-03", "CODMUNRES": "110020"},
                {"DATE": "2020-04-04", "CODMUNRES": "110020"},
                {"DATE": "2020-04-05", "CODMUNRES": "110021"},
            ]
        )
        daily_rows = pd.DataFrame(
            [
                {"MUNICIPALITY_CODE": "110020", "DATE": "2020-04-03", "LIVE_BIRTHS_TOTAL": 2},
                {"MUNICIPALITY_CODE": "110020", "DATE": "2020-04-04", "LIVE_BIRTHS_TOTAL": 1},
                {"MUNICIPALITY_CODE": "110021", "DATE": "2020-04-05", "LIVE_BIRTHS_TOTAL": 1},
            ]
        )
        event_path = root / "data" / "silver" / "sinasc" / "silver_sinasc_event" / f"YEAR={year}" / f"silver_sinasc_event_{year}.parquet"
        daily_path = root / "data" / "silver" / "sinasc" / "silver_sinasc_daily" / f"YEAR={year}" / f"silver_sinasc_daily_{year}.parquet"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        daily_path.parent.mkdir(parents=True, exist_ok=True)
        event_rows.to_parquet(event_path, index=False)
        daily_rows.to_parquet(daily_path, index=False)

    def test_promote_single_year_writes_individual_daily_and_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / "runtime"
            rows = [
                _base_stillbirth_row(CONTADOR="1", DTOBITO="2020-04-03", CODMUNRES="110020", CODMUNOCOR="110020", SEMAGESTAC="30", GESTACAO="4"),
                _base_stillbirth_row(CONTADOR="2", DTOBITO="2020-04-03", CODMUNRES="110020", CODMUNOCOR="110021", SEMAGESTAC="", GESTACAO="2"),
                _base_stillbirth_row(CONTADOR="3", DTOBITO="2020-04-04", CODMUNRES="110020", CODMUNOCOR="110020", SEMAGESTAC="", GESTACAO="", PESO=""),
            ]
            self._write_bronze(root, 2020, rows)
            self._write_sinasc_canonical_inputs(root, 2020)
            log_path = root / "artifacts" / "run_logs" / "stillbirth_test.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("", encoding="utf-8")

            report = promote_stillbirth_silver_range(
                root=root,
                runtime_root=runtime_root,
                start_year=2020,
                end_year=2020,
                log_path=log_path,
                command="test-promote-silver-stillbirth",
                force=False,
            )

            individual_path = stillbirth_individual_partition_file(root, 2020)
            daily_path = stillbirth_daily_partition_file(root, 2020)
            self.assertTrue(individual_path.exists())
            self.assertTrue(daily_path.exists())

            individual = pd.read_parquet(individual_path)
            daily = pd.read_parquet(daily_path)
            self.assertEqual(len(individual), 3)
            self.assertIn("MATERNAL_AGE_YEARS", individual.columns)
            self.assertIn("MATERNAL_SCHOOLING_CODE_FINAL", individual.columns)
            self.assertIn("PLURALITY_CODE", individual.columns)
            self.assertIn("FETAL_SEX_CODE", individual.columns)
            self.assertEqual(int(daily["STILLBIRTH_TOTAL_RAW"].sum()), 3)
            self.assertEqual(int(daily["STILLBIRTH_TOTAL_GTE28"].sum()), 1)
            self.assertEqual(int(daily["STILLBIRTH_TOTAL_GTE22_SENSITIVITY"].sum()), 2)
            self.assertEqual(int(daily["LIVE_BIRTHS_TOTAL"].sum()), 4)
            self.assertEqual(int(daily["TOTAL_BIRTHS_PRIMARY"].sum()), 5)
            self.assertEqual(int(daily["TOTAL_BIRTHS_SENSITIVITY_GTE22"].sum()), 6)
            self.assertTrue((daily["STILLBIRTH_TOTAL"] == daily["STILLBIRTH_TOTAL_GTE28"]).all())
            self.assertTrue((daily["NASCIMENTOS_TOTAIS_INCLUINDO_FETAIS"] == daily["TOTAL_BIRTHS_PRIMARY"]).all())
            self.assertEqual(
                daily.loc[daily["EVENT_DATE"].eq("2020-04-05"), "LIVE_BIRTHS_TOTAL"].iloc[0],
                1,
            )
            self.assertEqual(
                daily.loc[daily["EVENT_DATE"].eq("2020-04-05"), "STILLBIRTH_TOTAL_RAW"].iloc[0],
                0,
            )
            self.assertEqual(
                daily["LIVE_BIRTHS_SOURCE"].iloc[0],
                "silver_sinasc_event_residence_benchmarked_to_silver_sinasc_daily",
            )
            self.assertTrue(report["validation"][2020]["all_checks_passed"])

            manifest_path = root / "artifacts" / "manifests" / "silver_stillbirth.manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["entry_count"], 2)


if __name__ == "__main__":
    unittest.main()
