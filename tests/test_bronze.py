from __future__ import annotations

import gzip
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import pandas as pd

from heat_mnch.bronze import (
    INMET_ERA5_FALLBACK_FLAG_FIELD,
    INMET_HUMIDITY_FIELD,
    INMET_MISSING_HUMIDITY_OFFICIAL_FIELD,
    INMET_MISSING_TEMP_OFFICIAL_FIELD,
    INMET_TEMPERATURE_FIELD,
    aggregate_population_denominator,
    apply_inmet_missing_flags,
    normalize_column_name,
    normalize_value,
    population_external_benchmarks_ready,
    population_derivative_expected_codes,
    population_derivative_excluded_codes,
    write_csv_gz,
)
from heat_mnch.manifest import count_rows


class BronzeHelpersTest(unittest.TestCase):
    def test_normalize_column_name_removes_accents_and_symbols(self) -> None:
        self.assertEqual(normalize_column_name("CODIGO (WMO)"), "CODIGO_WMO")
        self.assertEqual(normalize_column_name("NOME DO MUNICÍPIO"), "NOME_DO_MUNICIPIO")

    def test_normalize_value_formats_dates_and_hours(self) -> None:
        self.assertEqual(normalize_value("DTOBITO", "18052020"), "2020-05-18")
        self.assertEqual(normalize_value("DT_INTER", "20191203"), "2019-12-03")
        self.assertEqual(normalize_value("DT_SAIDA", "20200119"), "2020-01-19")
        self.assertEqual(normalize_value("DT_INTER", "1203-19-20"), "2019-12-03")
        self.assertEqual(normalize_value("HORAOBITO", "1452"), "14:52")
        self.assertEqual(normalize_value("DATA", "2024/01/01"), "2024-01-01")

    def test_count_rows_supports_csv_gz_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv.gz"
            row_count, _ = write_csv_gz(path, ["A", "B"], [{"A": "1", "B": "x"}, {"A": "2", "B": "y"}])
            self.assertEqual(row_count, 2)
            self.assertEqual(count_rows(path), 2)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                self.assertIn("A;B", handle.readline())

    def test_apply_inmet_missing_flags_marks_official_missing_and_fallback(self) -> None:
        flagged = apply_inmet_missing_flags(
            {
                INMET_TEMPERATURE_FIELD: "",
                INMET_HUMIDITY_FIELD: "91",
            }
        )
        self.assertEqual(flagged[INMET_MISSING_TEMP_OFFICIAL_FIELD], "1")
        self.assertEqual(flagged[INMET_MISSING_HUMIDITY_OFFICIAL_FIELD], "0")
        self.assertEqual(flagged[INMET_ERA5_FALLBACK_FLAG_FIELD], "1")

    def test_aggregate_population_denominator_uses_datasus_popsvs_rules(self) -> None:
        frame = pd.DataFrame(
            [
                {"COD_MUN": "1100015", "ANO_NUM": 2024, "SEXO_NUM": 1, "IDADE_NUM": 0, "POP_NUM": 10},
                {"COD_MUN": "1100015", "ANO_NUM": 2024, "SEXO_NUM": 2, "IDADE_NUM": 4, "POP_NUM": 20},
                {"COD_MUN": "1100015", "ANO_NUM": 2024, "SEXO_NUM": 2, "IDADE_NUM": 10, "POP_NUM": 20},
                {"COD_MUN": "1100015", "ANO_NUM": 2024, "SEXO_NUM": 2, "IDADE_NUM": 15, "POP_NUM": 30},
                {"COD_MUN": "1100015", "ANO_NUM": 2024, "SEXO_NUM": 2, "IDADE_NUM": 49, "POP_NUM": 40},
                {"COD_MUN": "1100015", "ANO_NUM": 2024, "SEXO_NUM": 1, "IDADE_NUM": 20, "POP_NUM": 50},
                {"COD_MUN": "1100015", "ANO_NUM": 2024, "SEXO_NUM": 2, "IDADE_NUM": 50, "POP_NUM": 60},
            ]
        )

        under5 = aggregate_population_denominator(frame, "pop_under5", 2024)
        self.assertEqual(under5.to_dict(orient="records"), [{"COD_MUNICIPIO": "1100015", "ANO": "2024", "POP_UNDER5": "30"}])

        wra = aggregate_population_denominator(frame, "pop_wra", 2024)
        self.assertEqual(wra.to_dict(orient="records"), [{"COD_MUNICIPIO": "1100015", "ANO": "2024", "POP_WRA": "90"}])

    def test_population_external_benchmarks_ready_requires_presence_not_row_level_match(self) -> None:
        self.assertTrue(
            population_external_benchmarks_ready(
                {
                    "exists": True,
                    "joinable_municipality_count": 5570,
                    "missing_population_rows": 17,
                },
                sidra_exists=True,
            )
        )
        self.assertFalse(
            population_external_benchmarks_ready(
                {
                    "exists": True,
                    "joinable_municipality_count": 0,
                },
                sidra_exists=True,
            )
        )

    def test_population_derivative_expected_codes_suppresses_known_2024_gap(self) -> None:
        with mock.patch(
            "heat_mnch.bronze.official_population_codes",
            return_value={"1100015", "5101837"},
        ):
            self.assertEqual(population_derivative_excluded_codes(2024), {"5101837"})
            self.assertEqual(
                population_derivative_expected_codes(Path("/unused"), 2024),
                {"1100015"},
            )
            self.assertEqual(
                population_derivative_expected_codes(Path("/unused"), 2023),
                {"1100015", "5101837"},
            )


if __name__ == "__main__":
    unittest.main()
