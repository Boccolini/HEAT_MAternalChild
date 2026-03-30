from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from heat_mnch.round3_gold_report import GoldTableSpec, _scan_table_columns


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


class Round3GoldReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_scan_table_columns_uses_positive_live_birth_rows_for_prenatal_missingness(self) -> None:
        path = self.root / "data" / "gold" / "gold_analysis_wra_15_49" / "YEAR=2024" / "gold_analysis_wra_15_49_2024.parquet"
        _write_parquet(
            path,
            pd.DataFrame(
                [
                    {"PRENATAL_VISITS_COUNT": pd.NA, "LIVE_BIRTHS_TOTAL": 0},
                    {"PRENATAL_VISITS_COUNT": 6.0, "LIVE_BIRTHS_TOTAL": 1},
                    {"PRENATAL_VISITS_COUNT": pd.NA, "LIVE_BIRTHS_TOTAL": 2},
                    {"PRENATAL_VISITS_COUNT": pd.NA, "LIVE_BIRTHS_TOTAL": 0},
                    {"PRENATAL_VISITS_COUNT": 4.0, "LIVE_BIRTHS_TOTAL": 1},
                ]
            ),
        )
        spec = GoldTableSpec("gold_analysis_wra_15_49", str(path), "derived_gold", 0)
        con = duckdb.connect()
        try:
            describe, total_rows = _scan_table_columns(con, spec)
        finally:
            con.close()

        prenatal = describe.loc[describe["column_name"] == "PRENATAL_VISITS_COUNT"].iloc[0]
        births = describe.loc[describe["column_name"] == "LIVE_BIRTHS_TOTAL"].iloc[0]

        self.assertEqual(total_rows, 5)
        self.assertEqual(int(prenatal["total_rows"]), 5)
        self.assertEqual(int(prenatal["missingness_denominator_rows"]), 3)
        self.assertEqual(str(prenatal["missingness_scope"]), "rows with LIVE_BIRTHS_TOTAL > 0")
        self.assertEqual(int(prenatal["missing_rows"]), 1)
        self.assertEqual(int(prenatal["non_missing_rows"]), 2)
        self.assertEqual(int(prenatal["approx_distinct_non_missing"]), 2)
        self.assertAlmostEqual(float(prenatal["missing_pct"]), 1.0 / 3.0)

        self.assertEqual(int(births["missingness_denominator_rows"]), 5)
        self.assertEqual(str(births["missingness_scope"]), "all rows")
        self.assertEqual(int(births["missing_rows"]), 0)
        self.assertAlmostEqual(float(births["missing_pct"]), 0.0)

    def test_scan_table_columns_keeps_default_scope_without_live_births_column(self) -> None:
        path = self.root / "data" / "gold" / "gold_mnch_perinatal" / "YEAR=2024" / "gold_mnch_perinatal_2024.parquet"
        _write_parquet(
            path,
            pd.DataFrame(
                [
                    {"PRENATAL_VISITS_COUNT": pd.NA},
                    {"PRENATAL_VISITS_COUNT": 5.0},
                    {"PRENATAL_VISITS_COUNT": pd.NA},
                ]
            ),
        )
        spec = GoldTableSpec("gold_mnch_perinatal", str(path), "gold", 0)
        con = duckdb.connect()
        try:
            describe, _ = _scan_table_columns(con, spec)
        finally:
            con.close()

        prenatal = describe.loc[describe["column_name"] == "PRENATAL_VISITS_COUNT"].iloc[0]
        self.assertEqual(int(prenatal["missingness_denominator_rows"]), 3)
        self.assertEqual(str(prenatal["missingness_scope"]), "all rows")
        self.assertEqual(int(prenatal["missing_rows"]), 2)
        self.assertEqual(int(prenatal["non_missing_rows"]), 1)
        self.assertAlmostEqual(float(prenatal["missing_pct"]), 2.0 / 3.0)
