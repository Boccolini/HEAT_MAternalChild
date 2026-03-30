from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from heat_mnch.outcome_dictionary_icd10 import (
    build_unified_outcome_dictionary,
    derive_sih_age_flags,
    derive_sim_age_flags,
    split_sim_cause_tokens,
)
from heat_mnch.silver_sih import SIH_REQUIRED_INPUT_FIELDS, promote_sih_silver_range, sih_daily_partition_file, sih_individual_partition_file
from heat_mnch.silver_sim import SIM_REQUIRED_INPUT_FIELDS, promote_sim_silver_range, sim_daily_partition_file, sim_individual_partition_file


def _base_sim_row(**overrides: str) -> dict[str, str]:
    row = {
        "BRONZE_DATASET": "sim",
        "BRONZE_SOURCE_SYSTEM": "dados_abertos_sus",
        "BRONZE_SOURCE_FILE": "SIM_2020.csv",
        "BRONZE_SOURCE_PATH": "data/raw/sim/2020/SIM_2020.csv",
        "BRONZE_SOURCE_MEMBER": "SIM_2020.csv",
        "BRONZE_YEAR": "2020",
        "BRONZE_INGESTED_AT": "2026-03-19T00:00:00+00:00",
        "ORIGEM": "1",
        "TIPOBITO": "2",
        "DTOBITO": "2020-01-10",
        "DTNASC": "2019-12-25",
        "IDADE": "201",
        "SEXO": "1",
        "CODMUNRES": "110020",
        "CODMUNOCOR": "110020",
        "CAUSABAS": "A001",
        "LINHAA": "",
        "LINHAB": "",
        "LINHAC": "",
        "LINHAD": "",
        "LINHAII": "",
        "CONTADOR": "1",
    }
    row.update(overrides)
    return row


def _base_sih_row(**overrides: str) -> dict[str, str]:
    row = {
        "BRONZE_DATASET": "sih",
        "BRONZE_SOURCE_SYSTEM": "dados_abertos_sus",
        "BRONZE_SOURCE_FILE": "SIH_2020.csv",
        "BRONZE_SOURCE_PATH": "data/raw/sih/2020/SIH_2020.csv",
        "BRONZE_SOURCE_MEMBER": "SIH_2020.csv",
        "BRONZE_YEAR": "2020",
        "BRONZE_INGESTED_AT": "2026-03-19T00:00:00+00:00",
        "ANO_CMPT": "2020",
        "MES_CMPT": "01",
        "N_AIH": "1220100000001",
        "IDENT": "1",
        "MUNIC_RES": "110020",
        "MUNIC_MOV": "110020",
        "DT_INTER": "2020-01-10",
        "DT_SAIDA": "2020-01-12",
        "COD_IDADE": "4",
        "IDADE": "25",
        "SEXO": "3",
        "DIAG_PRINC": "O140",
        "DIAG_SECUN": "0000",
        "DIAGSEC1": "",
        "DIAGSEC2": "",
        "DIAGSEC3": "",
        "DIAGSEC4": "",
        "DIAGSEC5": "",
        "DIAGSEC6": "",
        "DIAGSEC7": "",
        "DIAGSEC8": "",
        "DIAGSEC9": "",
        "CID_NOTIF": "",
        "CID_ASSO": "",
        "CID_MORTE": "",
        "MORTE": "0",
        "CAR_INT": "02",
        "UTI_MES_TO": "0",
        "UTI_INT_TO": "0",
        "MARCA_UTI": "00",
        "VAL_UTI": "0.0",
        "VAL_UCI": "0.0",
        "CNES": "2000121",
        "CGC_HOSP": "11111111111111",
        "SEQUENCIA": "1",
    }
    row.update(overrides)
    return row


class SilverOutcomeDictionaryTest(unittest.TestCase):
    def test_sim_age_flags_follow_encoded_units(self) -> None:
        flags = derive_sim_age_flags(pd.Series(["202", "305", "415", "449"]))
        self.assertEqual(int(flags.loc[0, "AGE_GROUP_0_27D"]), 1)
        self.assertEqual(int(flags.loc[1, "AGE_GROUP_28_364D"]), 1)
        self.assertEqual(int(flags.loc[2, "AGE_GROUP_WRA_15_49"]), 1)
        self.assertEqual(int(flags.loc[3, "AGE_GROUP_U5"]), 0)

    def test_sih_age_flags_follow_cod_idade(self) -> None:
        flags = derive_sih_age_flags(pd.Series(["2", "3", "4", "4"]), pd.Series(["10", "5", "3", "20"]))
        self.assertEqual(int(flags.loc[0, "AGE_GROUP_0_27D"]), 1)
        self.assertEqual(int(flags.loc[1, "AGE_GROUP_28_364D"]), 1)
        self.assertEqual(int(flags.loc[2, "AGE_GROUP_1_4Y"]), 1)
        self.assertEqual(int(flags.loc[3, "AGE_GROUP_WRA_15_49"]), 1)

    def test_dictionary_is_expanded_without_wildcards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_bronze_csv(root / "data" / "bronze" / "sim" / "2020" / "sim_2020_source_preserving.csv.gz", SIM_REQUIRED_INPUT_FIELDS, [
                _base_sim_row(CAUSABAS="O141", LINHAA="T671", LINHAB="E870"),
                _base_sim_row(CONTADOR="2", CAUSABAS="A009", LINHAA="J450"),
            ])
            _write_bronze_csv(root / "data" / "bronze" / "sih" / "2020" / "sih_2020_source_preserving.csv.gz", SIH_REQUIRED_INPUT_FIELDS, [
                _base_sih_row(DIAG_PRINC="O244", DIAGSEC1="X300"),
                _base_sih_row(N_AIH="1220100000002", DIAG_PRINC="N179", DIAGSEC1="P743"),
                _base_sih_row(N_AIH="1220100000003", DIAG_PRINC="O13"),
                _base_sih_row(N_AIH="1220100000004", DIAG_PRINC="O150"),
                _base_sih_row(N_AIH="1220100000005", DIAG_PRINC="O460"),
            ])
            result = build_unified_outcome_dictionary(root, 2020, 2020)
            dictionary = result["dictionary"]
            self.assertTrue(dictionary["ICD10_CODE"].str.contains(r"[\*\-]", regex=True).eq(False).all())
            self.assertTrue({"O141", "T671", "A009", "N179", "X300"}.issubset(set(dictionary["ICD10_CODE"])))
            self.assertTrue(
                {
                    "HDP_GESTATIONAL_HTN_PRIMARY",
                    "HDP_PREECLAMPSIA_PRIMARY",
                    "HDP_ECLAMPSIA_PRIMARY",
                    "APH_ANTEPARTUM_PRIMARY",
                }.issubset(set(dictionary["OUTCOME_GROUP"]))
            )

    def test_split_sim_cause_tokens_preserves_atomic_codes(self) -> None:
        self.assertEqual(split_sim_cause_tokens("*J159*B342*U071"), ("J159", "B342", "U071"))
        self.assertEqual(split_sim_cause_tokens("O14.1"), ("O141",))


class SilverSimSihIntegrationTest(unittest.TestCase):
    def test_promote_single_year_sim_and_sih(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / "runtime"
            _write_bronze_csv(
                root / "data" / "bronze" / "sim" / "2020" / "sim_2020_source_preserving.csv.gz",
                SIM_REQUIRED_INPUT_FIELDS,
                [
                    _base_sim_row(CAUSABAS="A001", LINHAA="T671", CONTADOR="1"),
                    _base_sim_row(CONTADOR="2", CAUSABAS="J189", LINHAA="X300", IDADE="415"),
                ],
            )
            _write_bronze_csv(
                root / "data" / "bronze" / "sih" / "2020" / "sih_2020_source_preserving.csv.gz",
                SIH_REQUIRED_INPUT_FIELDS,
                [
                    _base_sih_row(DIAG_PRINC="O140", DIAGSEC1="X300", UTI_MES_TO="2", MARCA_UTI="75", VAL_UTI="10.0", SEQUENCIA="1"),
                    _base_sih_row(N_AIH="1220100000002", DIAG_PRINC="J189", DIAGSEC1="J450", COD_IDADE="3", IDADE="5", SEQUENCIA="2"),
                    _base_sih_row(N_AIH="1220100000003", DIAG_PRINC="O460", MORTE="1", CAR_INT="02", UTI_MES_TO="0", SEQUENCIA="3"),
                ],
            )

            sim_log = root / "artifacts" / "run_logs" / "sim_test.jsonl"
            sim_log.parent.mkdir(parents=True, exist_ok=True)
            sim_log.write_text("", encoding="utf-8")
            sim_report = promote_sim_silver_range(
                root=root,
                runtime_root=runtime_root,
                start_year=2020,
                end_year=2020,
                log_path=sim_log,
                command="test-promote-silver-sim",
                force=False,
            )
            sim_individual = pd.read_parquet(sim_individual_partition_file(root, 2020))
            sim_daily = pd.read_parquet(sim_daily_partition_file(root, 2020))
            self.assertEqual(len(sim_individual), 2)
            self.assertEqual(int(sim_daily["DEATHS_TOTAL"].sum()), 2)
            self.assertEqual(int(sim_individual.loc[1, "UNDERLYING_BASIC_CAUSE_CHILD_HEAT_THERMAL_PRIMARY_FLAG"]), 0)
            self.assertEqual(int(sim_individual.loc[1, "ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_FLAG"]), 1)
            self.assertTrue(sim_report["validation"][2020]["all_checks_passed"])

            sih_log = root / "artifacts" / "run_logs" / "sih_test.jsonl"
            sih_log.write_text("", encoding="utf-8")
            sih_report = promote_sih_silver_range(
                root=root,
                runtime_root=runtime_root,
                start_year=2020,
                end_year=2020,
                log_path=sih_log,
                command="test-promote-silver-sih",
                force=False,
            )
            sih_individual = pd.read_parquet(sih_individual_partition_file(root, 2020))
            sih_daily = pd.read_parquet(sih_daily_partition_file(root, 2020))
            self.assertEqual(len(sih_individual), 3)
            self.assertEqual(int(sih_daily["ADMISSIONS_TOTAL"].sum()), 3)
            self.assertEqual(int(sih_individual.loc[0, "PRINCIPAL_DIAGNOSIS_HDP_ANY_PRIMARY_FLAG"]), 1)
            self.assertEqual(int(sih_individual.loc[0, "PRINCIPAL_DIAGNOSIS_HDP_PREECLAMPSIA_PRIMARY_FLAG"]), 1)
            self.assertEqual(int(sih_individual.loc[0, "EXTERNAL_CAUSE_FIELDS_EXCESSIVE_NATURAL_HEAT_X30_FLAG"]), 1)
            self.assertEqual(int(sih_individual.loc[0, "URGENT_ADMISSION_OBSERVED_FLAG"]), 1)
            self.assertEqual(int(sih_individual.loc[0, "ICU_ANY_STAY_FLAG"]), 1)
            self.assertEqual(int(sih_individual.loc[0, "OBSTETRIC_ADMISSION_PRIMARY_FLAG"]), 1)
            self.assertEqual(int(sih_individual.loc[1, "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_FLAG"]), 1)
            self.assertEqual(int(sih_individual.loc[2, "PRINCIPAL_DIAGNOSIS_APH_ANTEPARTUM_PRIMARY_FLAG"]), 1)
            self.assertEqual(int(sih_individual.loc[2, "APH_ANTEPARTUM_URGENT_FLAG"]), 1)
            self.assertEqual(int(sih_individual.loc[2, "APH_ANTEPARTUM_DEATH_FLAG"]), 1)
            self.assertEqual(int(sih_daily["OBSTETRIC_ADMISSIONS_TOTAL"].sum()), 2)
            self.assertEqual(int(sih_daily["OBSTETRIC_URGENT_TOTAL"].sum()), 2)
            self.assertEqual(int(sih_daily["OBSTETRIC_ICU_TOTAL"].sum()), 1)
            self.assertEqual(int(sih_daily["HDP_PREECLAMPSIA_TOTAL"].sum()), 1)
            self.assertEqual(int(sih_daily["APH_ANTEPARTUM_TOTAL"].sum()), 1)
            self.assertEqual(int(sih_daily["APH_ANTEPARTUM_URGENT_TOTAL"].sum()), 1)
            self.assertEqual(int(sih_daily["APH_ANTEPARTUM_DEATH_TOTAL"].sum()), 1)
            self.assertTrue(sih_report["validation"][2020]["all_checks_passed"])

            sim_manifest = json.loads((root / "artifacts" / "manifests" / "silver_sim.manifest.json").read_text(encoding="utf-8"))
            sih_manifest = json.loads((root / "artifacts" / "manifests" / "silver_sih.manifest.json").read_text(encoding="utf-8"))
            dictionary_manifest = json.loads((root / "artifacts" / "manifests" / "mnch_outcome_dictionary_icd10.manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(sim_manifest["entry_count"], 2)
            self.assertEqual(sih_manifest["entry_count"], 2)
            self.assertGreater(dictionary_manifest["row_count"], 0)

    def test_sim_any_mention_parses_multicode_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / "runtime"
            _write_bronze_csv(
                root / "data" / "bronze" / "sim" / "2020" / "sim_2020_source_preserving.csv.gz",
                SIM_REQUIRED_INPUT_FIELDS,
                [
                    _base_sim_row(CAUSABAS="R999", LINHAA="*X30*J189", CONTADOR="1"),
                ],
            )
            _write_bronze_csv(
                root / "data" / "bronze" / "sih" / "2020" / "sih_2020_source_preserving.csv.gz",
                SIH_REQUIRED_INPUT_FIELDS,
                [
                    _base_sih_row(DIAG_PRINC="J189", DIAGSEC1="X300"),
                ],
            )
            sim_log = root / "artifacts" / "run_logs" / "sim_test.jsonl"
            sim_log.parent.mkdir(parents=True, exist_ok=True)
            sim_log.write_text("", encoding="utf-8")
            report = promote_sim_silver_range(
                root=root,
                runtime_root=runtime_root,
                start_year=2020,
                end_year=2020,
                log_path=sim_log,
                command="test-promote-silver-sim",
                force=False,
            )
            sim_individual = pd.read_parquet(sim_individual_partition_file(root, 2020))
            dictionary = pd.read_csv(root / "artifacts" / "dictionaries" / "mnch_outcome_dictionary_icd10.csv")
            self.assertEqual(sim_individual.loc[0, "LINHAA_ICD10"], "X30|J189")
            self.assertEqual(int(sim_individual.loc[0, "ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_FLAG"]), 1)
            self.assertEqual(int(sim_individual.loc[0, "ANY_MENTION_CHILD_RESPIRATORY_ACUTE_PRIMARY_FLAG"]), 1)
            self.assertFalse(dictionary["ICD10_CODE"].eq("X30J189").any())
            self.assertTrue(report["validation"][2020]["all_checks_passed"])


def _write_bronze_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    unittest.main()
