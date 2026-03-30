from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from heat_mnch.silver_sinasc import (
    SINASC_REQUIRED_INPUT_FIELDS,
    build_canonical_sinasc_daily_frame,
    build_canonical_sinasc_event_frame,
    promote_sinasc_silver_range,
    sinasc_daily_partition_file,
    sinasc_event_partition_file,
)


def _base_row(**overrides: str) -> dict[str, str]:
    row = {
        "BRONZE_SOURCE_SYSTEM": "dados_abertos_sus",
        "BRONZE_SOURCE_FILE": "SINASC_2020_csv.zip",
        "BRONZE_SOURCE_PATH": "data/raw/sinasc/2020/SINASC_2020_csv.zip",
        "BRONZE_SOURCE_MEMBER": "SINASC_2020_csv.csv",
        "BRONZE_YEAR": "2020",
        "BRONZE_INGESTED_AT": "2026-03-19T00:00:00+00:00",
        "CODMUNNASC": "110020",
        "CODMUNRES": "110020",
        "DTNASC": "2020-04-03",
        "DTULTMENST": "",
        "SEMAGESTAC": "",
        "TPMETESTIM": "",
        "GESTACAO": "",
        "CONSPRENAT": "",
        "CONSULTAS": "",
        "MESPRENAT": "",
        "PESO": "",
        "IDADEMAE": "",
        "ESCMAE2010": "",
        "ESCMAE": "",
        "SERIESCMAE": "",
        "QTDPARTNOR": "",
        "QTDPARTCES": "",
        "GRAVIDEZ": "",
        "SEXO": "",
        "RACACORMAE": "",
        "IDANOMAL": "",
        "CODANOMAL": "",
        "CONTADOR": "1",
    }
    row.update(overrides)
    return row


class SilverSinascDerivationTest(unittest.TestCase):
    def test_exact_lmp_dating_is_preferred(self) -> None:
        frame = pd.DataFrame(
            [
                _base_row(
                    DTULTMENST="2019-07-05",
                    SEMAGESTAC="38",
                    TPMETESTIM="2",
                    GESTACAO="5",
                    CONSPRENAT="06",
                    CONSULTAS="3",
                    MESPRENAT="02",
                )
            ]
        )
        event = build_canonical_sinasc_event_frame(frame, 2020)
        row = event.iloc[0]
        self.assertEqual(float(row["GA_DAYS_EXACT"]), 273.0)
        self.assertEqual(float(row["GA_WEEKS_COMPLETED_FROM_LMP"]), 39.0)
        self.assertEqual(float(row["GA_WEEKS_COMPLETED_FINAL"]), 39.0)
        self.assertEqual(row["GA_SOURCE"], "LMP_EXACT")
        self.assertEqual(int(row["GA_DAYS_APPROX_FLAG"]), 0)
        self.assertEqual(int(row["PRETERM_LT37_FLAG"]), 0)
        self.assertEqual(int(row["GA_VALID_FOR_ANALYSIS_FLAG"]), 1)

    def test_raw_weeks_fallback_uses_physical_exam_source(self) -> None:
        frame = pd.DataFrame(
            [
                _base_row(
                    SEMAGESTAC="35",
                    TPMETESTIM="1",
                    GESTACAO="4",
                )
            ]
        )
        event = build_canonical_sinasc_event_frame(frame, 2020)
        row = event.iloc[0]
        self.assertTrue(pd.isna(row["GA_DAYS_EXACT"]))
        self.assertEqual(float(row["GA_WEEKS_COMPLETED_FINAL"]), 35.0)
        self.assertEqual(float(row["GA_DAYS_FINAL"]), 245.0)
        self.assertEqual(int(row["GA_DAYS_APPROX_FLAG"]), 1)
        self.assertEqual(row["GA_SOURCE"], "SEMAGESTAC_PHYSICAL_EXAM")
        self.assertEqual(int(row["PRETERM_LT37_FLAG"]), 1)
        self.assertEqual(row["PRETERM_DERIVATION_NOTE"], "COMPLETED_WEEKS_ONLY")

    def test_band_only_path_keeps_lt34_and_lt36_null_for_band_four(self) -> None:
        frame = pd.DataFrame([_base_row(GESTACAO="4")])
        event = build_canonical_sinasc_event_frame(frame, 2020)
        row = event.iloc[0]
        self.assertEqual(row["GA_SOURCE"], "GESTACAO_BAND_ONLY")
        self.assertEqual(int(row["PRETERM_LT37_FLAG"]), 1)
        self.assertEqual(int(row["PRETERM_LT32_FLAG"]), 0)
        self.assertTrue(pd.isna(row["PRETERM_LT34_FLAG"]))
        self.assertTrue(pd.isna(row["PRETERM_LT36_FLAG_SENSITIVITY"]))
        self.assertEqual(row["PRETERM_DETAILED_4CAT"], "UNKNOWN")

    def test_consultas_band_three_does_not_force_six_plus(self) -> None:
        frame = pd.DataFrame([_base_row(CONSULTAS="3")])
        event = build_canonical_sinasc_event_frame(frame, 2020)
        row = event.iloc[0]
        self.assertTrue(pd.isna(row["PRENATAL_VISITS_COUNT"]))
        self.assertEqual(row["PRENATAL_VISITS_SOURCE"], "SINASC_CONSULTAS_BAND")
        self.assertTrue(pd.isna(row["PRENATAL_VISITS_6PLUS_FLAG"]))
        self.assertTrue(pd.isna(row["PRENATAL_VISITS_8PLUS_FLAG"]))
        self.assertEqual(row["PRENATAL_VISITS_DERIVATION_NOTE"], "BAND_4_TO_6_UNCERTAIN_FOR_6PLUS")

    def test_consprenat_exact_count_overrides_band(self) -> None:
        frame = pd.DataFrame([_base_row(CONSPRENAT="06", CONSULTAS="2", MESPRENAT="03")])
        event = build_canonical_sinasc_event_frame(frame, 2020)
        row = event.iloc[0]
        self.assertEqual(float(row["PRENATAL_VISITS_COUNT"]), 6.0)
        self.assertEqual(int(row["PRENATAL_VISITS_6PLUS_FLAG"]), 1)
        self.assertEqual(int(row["PRENATAL_VISITS_8PLUS_FLAG"]), 0)
        self.assertEqual(row["PRENATAL_VISITS_SOURCE"], "SINASC_CONSPRENAT")
        self.assertEqual(row["PRENATAL_VISITS_RAW_VALUE"], "06")
        self.assertEqual(float(row["PRENATAL_START_GESTATIONAL_MONTH"]), 3.0)

    def test_birth_weight_and_strict_anc8_are_derived_from_exact_values(self) -> None:
        frame = pd.DataFrame(
            [
                _base_row(
                    SEMAGESTAC="35",
                    TPMETESTIM="1",
                    GESTACAO="4",
                    CONSPRENAT="08",
                    CONSULTAS="4",
                    PESO="2499",
                )
            ]
        )
        event = build_canonical_sinasc_event_frame(frame, 2020)
        row = event.iloc[0]
        self.assertEqual(float(row["BIRTH_WEIGHT_G"]), 2499.0)
        self.assertEqual(int(row["BIRTH_WEIGHT_REPORTED_FLAG"]), 1)
        self.assertEqual(int(row["LOW_BIRTH_WEIGHT_LT2500_FLAG"]), 1)
        self.assertEqual(int(row["PRETERM_LOW_BIRTH_WEIGHT_FLAG"]), 1)
        self.assertEqual(int(row["PRENATAL_VISITS_6PLUS_FLAG"]), 1)
        self.assertEqual(int(row["PRENATAL_VISITS_8PLUS_FLAG"]), 1)

    def test_lmp_after_birth_sets_qc_and_falls_back_to_weeks(self) -> None:
        frame = pd.DataFrame(
            [
                _base_row(
                    DTULTMENST="2020-05-01",
                    SEMAGESTAC="38",
                    TPMETESTIM="2",
                    GESTACAO="5",
                )
            ]
        )
        event = build_canonical_sinasc_event_frame(frame, 2020)
        row = event.iloc[0]
        self.assertEqual(int(row["GA_QC_LMP_AFTER_BIRTH_FLAG"]), 1)
        self.assertTrue(pd.isna(row["GA_DAYS_EXACT"]))
        self.assertEqual(float(row["GA_WEEKS_COMPLETED_FINAL"]), 38.0)
        self.assertEqual(row["GA_SOURCE"], "SEMAGESTAC_OTHER_METHOD")

    def test_round3_covariates_are_derived_from_native_fields(self) -> None:
        frame = pd.DataFrame(
            [
                _base_row(
                    IDADEMAE="31",
                    ESCMAE2010="3",
                    QTDPARTNOR="01",
                    QTDPARTCES="02",
                    GRAVIDEZ="2",
                    SEXO="2",
                    RACACORMAE="4",
                    IDANOMAL="2",
                    CODANOMAL="",
                )
            ]
        )
        event = build_canonical_sinasc_event_frame(frame, 2020)
        row = event.iloc[0]
        self.assertEqual(int(row["MATERNAL_AGE_YEARS"]), 31)
        self.assertEqual(int(row["MATERNAL_AGE_MISSING_FLAG"]), 0)
        self.assertEqual(int(row["MATERNAL_SCHOOLING_CODE_FINAL"]), 3)
        self.assertEqual(row["MATERNAL_SCHOOLING_SOURCE"], "ESCMAE2010")
        self.assertEqual(int(row["PRIOR_VAGINAL_BIRTHS_COUNT"]), 1)
        self.assertEqual(int(row["PRIOR_CESAREAN_BIRTHS_COUNT"]), 2)
        self.assertEqual(int(row["PARITY_TOTAL_PRIOR"]), 3)
        self.assertEqual(int(row["PARITY_TOTAL_PRIOR_MISSING_FLAG"]), 0)
        self.assertEqual(int(row["PLURALITY_CODE"]), 2)
        self.assertEqual(row["PLURALITY_CATEGORY"], "MULTIPLE")
        self.assertEqual(int(row["MULTIPLE_GESTATION_FLAG"]), 1)
        self.assertEqual(int(row["NEWBORN_SEX_CODE"]), 2)
        self.assertEqual(row["NEWBORN_SEX_CATEGORY"], "FEMALE")
        self.assertEqual(int(row["MATERNAL_RACE_COLOR_CODE"]), 4)
        self.assertEqual(row["MATERNAL_RACE_COLOR_CATEGORY"], "BROWN_MIXED")
        self.assertEqual(int(row["CONGENITAL_ANOMALY_FLAG"]), 0)
        self.assertEqual(int(row["CONGENITAL_ANOMALY_MISSING_FLAG"]), 0)

    def test_schooling_fallbacks_and_incomplete_parity_stay_explicit(self) -> None:
        frame = pd.DataFrame(
            [
                _base_row(CONTADOR="1", ESCMAE="4", QTDPARTNOR="01", QTDPARTCES="", GRAVIDEZ="1", SEXO="1"),
                _base_row(CONTADOR="2", ESCMAE="", SERIESCMAE="6", GRAVIDEZ="3", SEXO="0", IDANOMAL="9"),
            ]
        )
        event = build_canonical_sinasc_event_frame(frame, 2020)
        legacy_row = event.iloc[0]
        series_row = event.iloc[1]
        self.assertEqual(int(legacy_row["MATERNAL_SCHOOLING_CODE_FINAL"]), 4)
        self.assertEqual(legacy_row["MATERNAL_SCHOOLING_SOURCE"], "ESCMAE")
        self.assertTrue(pd.isna(legacy_row["PARITY_TOTAL_PRIOR"]))
        self.assertEqual(int(legacy_row["PARITY_TOTAL_PRIOR_MISSING_FLAG"]), 1)
        self.assertEqual(legacy_row["PLURALITY_CATEGORY"], "SINGLETON")
        self.assertEqual(series_row["MATERNAL_SCHOOLING_SOURCE"], "SERIESCMAE")
        self.assertEqual(int(series_row["MATERNAL_SCHOOLING_CODE_FINAL"]), 6)
        self.assertEqual(series_row["PLURALITY_CATEGORY"], "MULTIPLE")
        self.assertEqual(series_row["NEWBORN_SEX_CATEGORY"], "UNKNOWN")
        self.assertTrue(pd.isna(series_row["CONGENITAL_ANOMALY_FLAG"]))
        self.assertEqual(int(series_row["CONGENITAL_ANOMALY_MISSING_FLAG"]), 1)

    def test_daily_aggregation_uses_event_level_flags(self) -> None:
        frame = pd.DataFrame(
            [
                _base_row(CONTADOR="1", DTULTMENST="2019-07-05", SEMAGESTAC="39", TPMETESTIM="2", GESTACAO="5", CONSPRENAT="06", MESPRENAT="02", PESO="2600"),
                _base_row(CONTADOR="2", DTULTMENST="", SEMAGESTAC="35", TPMETESTIM="1", GESTACAO="4", CONSPRENAT="", CONSULTAS="4", MESPRENAT="03", PESO="2400"),
                _base_row(CONTADOR="3", DTNASC="2020-04-04", DTULTMENST="", GESTACAO="4", CONSPRENAT="", CONSULTAS="3"),
            ]
        )
        event = build_canonical_sinasc_event_frame(frame, 2020)
        daily = build_canonical_sinasc_daily_frame(event)
        april_3 = daily.loc[daily["DATE"].eq("2020-04-03")].iloc[0]
        april_4 = daily.loc[daily["DATE"].eq("2020-04-04")].iloc[0]
        self.assertEqual(int(april_3["LIVE_BIRTHS_TOTAL"]), 2)
        self.assertEqual(int(april_3["PRETERM_LT37_TOTAL"]), 1)
        self.assertEqual(int(april_3["NV_TOTAL"]), 2)
        self.assertEqual(int(april_3["PRENATAL_VISITS_6PLUS_TOTAL"]), 2)
        self.assertEqual(int(april_3["PRENATAL_VISITS_8PLUS_TOTAL"]), 0)
        self.assertEqual(int(april_3["LOW_BIRTH_WEIGHT_LT2500_TOTAL"]), 1)
        self.assertEqual(int(april_3["PRETERM_LOW_BIRTH_WEIGHT_TOTAL"]), 1)
        self.assertEqual(int(april_4["PRETERM_LT37_TOTAL"]), 1)
        self.assertEqual(int(april_4["PRETERM_LT34_TOTAL"]), 0)


class SilverSinascIntegrationTest(unittest.TestCase):
    def _write_bronze(self, root: Path, year: int, rows: list[dict[str, str]]) -> None:
        path = root / "data" / "bronze" / "sinasc" / str(year) / f"sinasc_{year}_source_preserving.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SINASC_REQUIRED_INPUT_FIELDS, delimiter=";")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in SINASC_REQUIRED_INPUT_FIELDS})

    def test_promote_single_year_writes_event_daily_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / "runtime"
            rows = [
                _base_row(CONTADOR="1", DTULTMENST="2019-07-05", SEMAGESTAC="39", TPMETESTIM="2", GESTACAO="5", CONSPRENAT="06", MESPRENAT="02"),
                _base_row(CONTADOR="2", DTULTMENST="", SEMAGESTAC="35", TPMETESTIM="1", GESTACAO="4", CONSULTAS="4", MESPRENAT="03"),
                _base_row(CONTADOR="3", DTNASC="2020-04-04", DTULTMENST="", GESTACAO="4", CONSULTAS="3"),
            ]
            self._write_bronze(root, 2020, rows)
            log_path = root / "artifacts" / "run_logs" / "sinasc_test.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("", encoding="utf-8")

            report = promote_sinasc_silver_range(
                root=root,
                runtime_root=runtime_root,
                start_year=2020,
                end_year=2020,
                log_path=log_path,
                command="test-promote-silver-sinasc",
                force=False,
            )

            event_path = sinasc_event_partition_file(root, 2020)
            daily_path = sinasc_daily_partition_file(root, 2020)
            self.assertTrue(event_path.exists())
            self.assertTrue(daily_path.exists())

            event = pd.read_parquet(event_path)
            daily = pd.read_parquet(daily_path)
            self.assertEqual(len(event), 3)
            self.assertEqual(int(daily["LIVE_BIRTHS_TOTAL"].sum()), 3)
            self.assertEqual(int(daily["PRETERM_LT37_TOTAL"].sum()), int(event["PRETERM_LT37_FLAG"].eq(1).sum()))
            self.assertIn("MATERNAL_AGE_YEARS", event.columns)
            self.assertIn("MATERNAL_SCHOOLING_CODE_FINAL", event.columns)
            self.assertIn("PARITY_TOTAL_PRIOR", event.columns)
            self.assertIn("PLURALITY_CODE", event.columns)
            self.assertIn("NEWBORN_SEX_CODE", event.columns)
            self.assertTrue(report["validation"][2020]["all_checks_passed"])

            manifest_path = root / "artifacts" / "manifests" / "silver_sinasc.manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["entry_count"], 2)


if __name__ == "__main__":
    unittest.main()
