from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from heat_mnch.era5_land import (
    ERA5_AREA_BRAZIL,
    ERA5_NO_USABLE_DATA_FIELD,
    ERA5_TIMES,
    _format_float,
    _era5_no_usable_data_flag,
    _filter_invalid_lookup_codes,
    _relative_humidity_array,
    bronze_rows_from_dataframe,
    build_era5_request,
    era5_month_days,
    relative_humidity_percent,
)


class Era5LandHelpersTest(unittest.TestCase):
    def test_era5_request_uses_brazil_area_and_full_hourly_cycle(self) -> None:
        request = build_era5_request(2024, 2)
        self.assertEqual(request["area"], ERA5_AREA_BRAZIL)
        self.assertEqual(request["time"], ERA5_TIMES)
        self.assertEqual(request["day"], [f"{day:02d}" for day in range(1, 30)])

    def test_era5_month_days_handles_leap_year(self) -> None:
        self.assertEqual(era5_month_days(2024, 2)[-1], "29")
        self.assertEqual(era5_month_days(2023, 2)[-1], "28")

    def test_relative_humidity_is_clipped_and_equals_hundred_at_saturation(self) -> None:
        self.assertAlmostEqual(relative_humidity_percent(25.0, 25.0), 100.0, places=6)
        self.assertLessEqual(relative_humidity_percent(35.0, 34.0), 100.0)
        self.assertGreaterEqual(relative_humidity_percent(35.0, 10.0), 0.0)

    def test_relative_humidity_array_matches_scalar_formula(self) -> None:
        temps = np.array([25.0, 35.0, 35.0], dtype=float)
        dewpoints = np.array([25.0, 34.0, 10.0], dtype=float)
        expected = np.array([relative_humidity_percent(temp, dewpoint) for temp, dewpoint in zip(temps, dewpoints)])
        observed = _relative_humidity_array(temps, dewpoints)
        self.assertTrue(np.allclose(observed, expected))

    def test_format_float_returns_blank_for_nan(self) -> None:
        self.assertEqual(_format_float(float("nan"), 4), "")
        self.assertEqual(_format_float(12.34567, 2), "12.35")

    def test_filter_invalid_lookup_codes_removes_excluded_municipalities(self) -> None:
        dataframe = pd.DataFrame(
            {
                "COD_MUNICIPIO": ["4300001", "1100015", "4300002", "5101704"],
                "UF": ["RS", "RO", "RS", "MT"],
            }
        )
        filtered = _filter_invalid_lookup_codes(dataframe)
        self.assertEqual(filtered["COD_MUNICIPIO"].tolist(), ["1100015", "5101704"])

    def test_no_usable_data_flag_marks_zero_observations(self) -> None:
        self.assertEqual(_era5_no_usable_data_flag(0), "1")
        self.assertEqual(_era5_no_usable_data_flag(24), "0")

    def test_bronze_rows_emit_flag_for_zero_observations(self) -> None:
        dataframe = pd.DataFrame(
            [
                {
                    "COD_MUNICIPIO": "1100015",
                    "NM_MUNICIPIO": "Alta Floresta D'Oeste",
                    "UF": "RO",
                    "TIMEZONE_NAME": "America/Porto_Velho",
                    "DATE": "2024-01-01",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "MUNICIPIO_CENTROID_LAT": -12.470132,
                    "MUNICIPIO_CENTROID_LON": -62.274661,
                    "ERA5_LAT": -12.5,
                    "ERA5_LON": -62.3,
                    "TMAX_C": np.nan,
                    "RH_MIN_PCT": np.nan,
                    "RH_MEAN_PCT": np.nan,
                    "RH_MAX_PCT": np.nan,
                    "N_OBS": 0,
                    "BRONZE_SOURCE_FILE": "era5_land_br_hourly_t2m_d2m_202401.nc",
                    "BRONZE_SOURCE_PATH": "data/raw/climate/era5_land/2024/era5_land_br_hourly_t2m_d2m_202401.nc",
                }
            ]
        )
        rows = list(bronze_rows_from_dataframe(dataframe, 2024))
        self.assertEqual(rows[0][ERA5_NO_USABLE_DATA_FIELD], "1")
        self.assertEqual(rows[0]["TMAX_C"], "")
        self.assertEqual(rows[0]["RH_MIN_PCT"], "")
        self.assertEqual(rows[0]["RH_MEAN_PCT"], "")
        self.assertEqual(rows[0]["N_OBS"], "0")

    def test_bronze_rows_emit_optional_temperature_columns_when_present(self) -> None:
        dataframe = pd.DataFrame(
            [
                {
                    "COD_MUNICIPIO": "1100015",
                    "NM_MUNICIPIO": "Alta Floresta D'Oeste",
                    "UF": "RO",
                    "TIMEZONE_NAME": "America/Porto_Velho",
                    "DATE": "2024-01-02",
                    "YEAR": 2024,
                    "MONTH": 1,
                    "MUNICIPIO_CENTROID_LAT": -12.470132,
                    "MUNICIPIO_CENTROID_LON": -62.274661,
                    "ERA5_LAT": -12.5,
                    "ERA5_LON": -62.3,
                    "TMAX_C": 34.125,
                    "TMIN_C": 22.5,
                    "TMEAN_C": 27.875,
                    "RH_MIN_PCT": 45.0,
                    "RH_MEAN_PCT": 60.0,
                    "RH_MAX_PCT": 80.0,
                    "N_OBS": 24,
                    "BRONZE_SOURCE_FILE": "era5_land_br_hourly_t2m_d2m_202401.nc",
                    "BRONZE_SOURCE_PATH": "data/raw/climate/era5_land/2024/era5_land_br_hourly_t2m_d2m_202401.nc",
                }
            ]
        )
        rows = list(bronze_rows_from_dataframe(dataframe, 2024))
        self.assertEqual(rows[0]["TMAX_C"], "34.1250")
        self.assertEqual(rows[0]["TMIN_C"], "22.5000")
        self.assertEqual(rows[0]["TMEAN_C"], "27.8750")
        self.assertEqual(rows[0][ERA5_NO_USABLE_DATA_FIELD], "0")


if __name__ == "__main__":
    unittest.main()
