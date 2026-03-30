from __future__ import annotations

import unittest

from heat_mnch.raw_sources import (
    IBGE_POPULATION_YEAR_URLS,
    datasus_popsvs_spec,
    ibge_sidra_population_spec,
    ibge_spec,
    inmet_spec,
    sinasc_open_data_spec,
)


class RawSourceSpecTest(unittest.TestCase):
    def test_inmet_spec_points_to_official_year_zip(self) -> None:
        spec = inmet_spec(2024)
        self.assertEqual(spec.dataset, "climate")
        self.assertIn("2024.zip", spec.url)
        self.assertEqual(spec.target_relative_path, "data/raw/climate/inmet/2024.zip")

    def test_ibge_mapping_covers_2020_to_2024(self) -> None:
        self.assertEqual(sorted(IBGE_POPULATION_YEAR_URLS.keys()), [2020, 2021, 2022, 2023, 2024])
        spec = ibge_spec(2023)
        self.assertEqual(spec.dataset, "population")
        self.assertIn("POP_DOU_2023_Municipios_POP2022_Malha2023.xls", spec.url)

    def test_ibge_sidra_population_spec_points_to_official_api(self) -> None:
        spec = ibge_sidra_population_spec(2024)
        self.assertEqual(spec.dataset, "population")
        self.assertIn("/values/t/6407/", spec.url)
        self.assertEqual(
            spec.target_relative_path,
            "data/raw/population/ibge_sidra_6407/2024/sidra_6407_population_municipality_age_sex_2024.json",
        )

    def test_datasus_popsvs_spec_points_to_official_fallback_zip(self) -> None:
        spec = datasus_popsvs_spec(2024)
        self.assertEqual(spec.dataset, "population")
        self.assertIn("/dissemin/publicos/IBGE/POPSVS/POPSBR24.zip", spec.url)
        self.assertEqual(spec.target_relative_path, "data/raw/population/datasus_popsvs/2024/POPSBR24.zip")

    def test_sinasc_spec_points_to_official_open_data_zip(self) -> None:
        spec = sinasc_open_data_spec(2024)
        self.assertEqual(spec.dataset, "sinasc")
        self.assertIn("SINASC_2024_csv.zip", spec.url)
        self.assertEqual(spec.target_relative_path, "data/raw/sinasc/2024/SINASC_2024_csv.zip")


if __name__ == "__main__":
    unittest.main()
