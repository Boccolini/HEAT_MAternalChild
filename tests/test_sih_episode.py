from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from heat_mnch.bronze_audit import AUDIT_SPECS, analyze_bronze_csv
from heat_mnch.sih_episode import build_canonical_silver, collapse_same_n_aih


def make_sih_row(**overrides: str) -> dict[str, str]:
    row = {
        "BRONZE_DATASET": "sih",
        "BRONZE_SOURCE_SYSTEM": "datasus",
        "BRONZE_SOURCE_FILE": "RDAC2401.dbc",
        "BRONZE_SOURCE_PATH": "data/raw/sih/2024/RDAC2401.dbc",
        "BRONZE_SOURCE_MEMBER": "",
        "BRONZE_YEAR": "2024",
        "BRONZE_INGESTED_AT": "2026-03-19T00:00:00+00:00",
        "ANO_CMPT": "2024",
        "MES_CMPT": "01",
        "MUNIC_RES": "120060",
        "N_AIH": "1000000000001",
        "IDENT": "1",
        "DIAG_PRINC": "O800",
        "DT_INTER": "2024-01-10",
        "DT_SAIDA": "2024-01-12",
        "SEQUENCIA": "10",
        "VAL_TOT": "100.00",
        "CNES": "2000121",
        "CGC_HOSP": "04034526003240",
        "NASC": "1990-01-01",
        "SEXO": "3",
        "CEP": "69970000",
        "PROC_SOLIC": "0301060010",
        "PROC_REA": "0301060010",
        "UTI_MES_TO": "0",
        "UTI_INT_TO": "0",
        "MARCA_UTI": "00",
        "VAL_UTI": "0.0",
        "VAL_UCI": "0.0",
        "VAL_TRANSP": "0.0",
    }
    row.update(overrides)
    return row


class SihBronzeAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = next(spec for spec in AUDIT_SPECS if spec.internal_dataset == "sih")
        self.official_registry = {
            2024: {
                "code6": {"120060"},
                "code7": set(),
            }
        }

    def _analyze_rows(self, rows: list[dict[str, str]]) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sih.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter=";")
                writer.writeheader()
                writer.writerows(rows)
            return analyze_bronze_csv(Path(tmpdir), self.spec, 2024, path, self.official_registry)

    def test_repeated_n_aih_is_profiled_but_not_counted_as_fatal_duplicate(self) -> None:
        analysis = self._analyze_rows(
            [
                make_sih_row(N_AIH="1224100000001", IDENT="1", DT_SAIDA="2024-01-12", SEQUENCIA="10"),
                make_sih_row(N_AIH="1224100000001", IDENT="5", DT_SAIDA="2024-01-15", SEQUENCIA="11"),
            ]
        )
        self.assertEqual(analysis["duplicate_count"], 0)
        self.assertEqual(analysis["repeated_n_aih_rows"], 1)
        self.assertEqual(analysis["repeated_n_aih_groups"], 1)
        self.assertEqual(analysis["literal_duplicate_rows"], 0)

    def test_literal_duplicate_rows_are_profiled_separately(self) -> None:
        row = make_sih_row(N_AIH="1224100000002", SEQUENCIA="20")
        analysis = self._analyze_rows([row, dict(row)])
        self.assertEqual(analysis["duplicate_count"], 0)
        self.assertEqual(analysis["repeated_n_aih_rows"], 1)
        self.assertEqual(analysis["repeated_n_aih_groups"], 1)
        self.assertEqual(analysis["literal_duplicate_rows"], 1)


class SihEpisodeHelpersTest(unittest.TestCase):
    def test_collapse_same_n_aih_marks_continuation_ident(self) -> None:
        dataframe = pd.DataFrame(
            [
                make_sih_row(N_AIH="1224100000100", IDENT="1", DT_INTER="2024-01-10", DT_SAIDA="2024-01-12", SEQUENCIA="10", VAL_TOT="100.00"),
                make_sih_row(N_AIH="1224100000100", IDENT="5", DT_INTER="2024-01-10", DT_SAIDA="2024-01-15", SEQUENCIA="99", VAL_TOT="140.00"),
            ]
        )
        collapsed = collapse_same_n_aih(dataframe)
        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed.loc[0, "N_SOURCE_ROWS"], 2)
        self.assertEqual(collapsed.loc[0, "HAS_CONTINUATION_IDENT"], 1)
        self.assertEqual(collapsed.loc[0, "DT_INT_MIN"], "2024-01-10")
        self.assertEqual(collapsed.loc[0, "DT_SAIDA_MAX"], "2024-01-15")
        self.assertEqual(collapsed.loc[0, "CANONICAL_ROW_SELECTION_USES_SEQUENCIA_ONLY"], 0)

    def test_two_separate_n_aih_dates_stay_as_two_episodes_and_two_canonicals(self) -> None:
        dataframe = pd.DataFrame(
            [
                make_sih_row(N_AIH="1224100000200", DT_INTER="2024-01-10", DT_SAIDA="2024-01-12"),
                make_sih_row(N_AIH="1224100000201", DT_INTER="2024-01-20", DT_SAIDA="2024-01-22"),
            ]
        )
        labeled = build_canonical_silver(dataframe)
        self.assertEqual(labeled["EPISODE_ID"].nunique(), 2)
        self.assertEqual(int(labeled["PROMOTE_TO_CANONICAL"].sum()), 2)
        self.assertEqual(int(labeled["NO_GO_TO_SILVER"].sum()), 0)

    def test_contiguous_same_patient_same_facility_uses_earliest_aih_as_canonical(self) -> None:
        dataframe = pd.DataFrame(
            [
                make_sih_row(N_AIH="1224100000300", DT_INTER="2024-01-10", DT_SAIDA="2024-01-12", SEQUENCIA="10"),
                make_sih_row(N_AIH="1224100000301", DT_INTER="2024-01-13", DT_SAIDA="2024-01-16", SEQUENCIA="999"),
            ]
        )
        labeled = build_canonical_silver(dataframe)
        self.assertEqual(labeled["EPISODE_ID"].nunique(), 1)
        canonical = labeled.loc[labeled["PROMOTE_TO_CANONICAL"] == 1].iloc[0]
        follow_on = labeled.loc[labeled["PROMOTE_TO_CANONICAL"] == 0].iloc[0]
        self.assertEqual(canonical["N_AIH"], "1224100000300")
        self.assertEqual(follow_on["NO_GO_TO_SILVER"], 1)
        self.assertEqual(follow_on["CANONICAL_SELECTION_USES_SEQUENCIA_ONLY"], 0)

    def test_higher_sequencia_on_later_aih_does_not_override_canonical_choice(self) -> None:
        dataframe = pd.DataFrame(
            [
                make_sih_row(N_AIH="1224100000400", DT_INTER="2024-01-10", DT_SAIDA="2024-01-11", SEQUENCIA="1"),
                make_sih_row(N_AIH="1224100000401", DT_INTER="2024-01-11", DT_SAIDA="2024-01-14", SEQUENCIA="9999"),
            ]
        )
        labeled = build_canonical_silver(dataframe)
        canonical = labeled.loc[labeled["PROMOTE_TO_CANONICAL"] == 1].iloc[0]
        self.assertEqual(canonical["N_AIH"], "1224100000400")
        self.assertEqual(canonical["CANONICAL_SELECTION_USES_SEQUENCIA_ONLY"], 0)

    def test_singleton_episode_is_promoted(self) -> None:
        dataframe = pd.DataFrame([make_sih_row(N_AIH="1224100000500")])
        labeled = build_canonical_silver(dataframe)
        row = labeled.iloc[0]
        self.assertEqual(row["EPISODE_SINGLE_AIH"], 1)
        self.assertEqual(row["PROMOTE_SINGLE_AIH_TO_CANONICAL"], 1)
        self.assertEqual(row["PROMOTE_TO_CANONICAL"], 1)
        self.assertEqual(row["NO_GO_TO_SILVER"], 0)
        self.assertEqual(row["NO_GO_REASON"], "single_aih_episode_promoted")

    def test_multi_aih_episode_with_uti_follow_on_is_excluded(self) -> None:
        dataframe = pd.DataFrame(
            [
                make_sih_row(N_AIH="1224100000600", DT_INTER="2024-01-10", DT_SAIDA="2024-01-12", UTI_MES_TO="0", MARCA_UTI="00"),
                make_sih_row(N_AIH="1224100000601", DT_INTER="2024-01-12", DT_SAIDA="2024-01-15", UTI_MES_TO="3", MARCA_UTI="02"),
            ]
        )
        labeled = build_canonical_silver(dataframe)
        canonical = labeled.loc[labeled["PROMOTE_TO_CANONICAL"] == 1].iloc[0]
        follow_on = labeled.loc[labeled["PROMOTE_TO_CANONICAL"] == 0].iloc[0]
        self.assertEqual(canonical["N_AIH"], "1224100000600")
        self.assertEqual(follow_on["LIKELY_COMPLICATION_OR_TRANSFER"], 1)
        self.assertEqual(follow_on["NO_GO_TO_SILVER"], 1)
        self.assertEqual(follow_on["NO_GO_REASON"], "uti_or_bed_movement_followon")


if __name__ == "__main__":
    unittest.main()
