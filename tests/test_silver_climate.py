from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from heat_mnch.manifest import build_manifest
from heat_mnch.silver_climate import (
    PRIMARY_CLIMATE_VARS,
    _build_neighbor_graph,
    _build_station_qc_table,
    _prepare_output_frame,
    derive_heat_index,
    impute_missing_climate_variables,
    load_or_build_preselected_climate_year,
    promote_climate_silver_range,
    recompute_heatwave_len_p95,
    silver_partition_file,
)


def _preselected_row(
    *,
    date_value: str,
    cod_municipio: str = "1100015",
    nm_municipio: str = "Alta Floresta D'Oeste",
    uf: str = "RO",
    timezone_name: str = "America/Porto_Velho",
    municipio_centroid_lat: float = -12.470132,
    municipio_centroid_lon: float = -62.274661,
    era5_lat: float = -12.5,
    era5_lon: float = -62.3,
    inmet: dict[str, float | None] | None = None,
    era5: dict[str, float | None] | None = None,
) -> dict[str, object]:
    inmet = dict(inmet or {})
    era5 = dict(era5 or {})
    for source in (inmet, era5):
        if "TMAX_C" in source and "TMIN_C" not in source:
            source["TMIN_C"] = float(source["TMAX_C"]) - 8.0
        if "TMAX_C" in source and "TMEAN_C" not in source:
            tmin_value = float(source["TMIN_C"]) if "TMIN_C" in source else float(source["TMAX_C"]) - 8.0
            source["TMEAN_C"] = (float(source["TMAX_C"]) + tmin_value) / 2.0
    timestamp = pd.Timestamp(date_value)
    row: dict[str, object] = {
        "COD_MUNICIPIO": cod_municipio,
        "NM_MUNICIPIO": nm_municipio,
        "UF": uf,
        "TIMEZONE_NAME": timezone_name,
        "DATE": timestamp.strftime("%Y-%m-%d"),
        "YEAR": int(timestamp.year),
        "MONTH": int(timestamp.month),
        "DAY_OF_YEAR": int(timestamp.dayofyear),
        "MUNICIPIO_CENTROID_LAT": municipio_centroid_lat,
        "MUNICIPIO_CENTROID_LON": municipio_centroid_lon,
        "ERA5_LAT": era5_lat,
        "ERA5_LON": era5_lon,
        "CLIMATE_UPSTREAM_QUARANTINE_FLAG": 0,
        "CLIMATE_UPSTREAM_QUARANTINE_REASON": "",
        "CLIMATE_RECORD_ANOMALY_FLAG": 0,
        "CLIMATE_RECORD_ANOMALY_REASON": "",
        "CLIMATE_CORRECTION_PATH": "none",
    }
    for variable in PRIMARY_CLIMATE_VARS:
        inmet_value = inmet.get(variable, np.nan)
        era5_value = era5.get(variable, np.nan)
        row[f"INMET_{variable}"] = inmet_value
        row[f"ERA5_{variable}"] = era5_value
        row[f"TEM_INMET_{variable}"] = int(not pd.isna(inmet_value))
        row[f"TEM_ERA5_{variable}"] = int(not pd.isna(era5_value))
        if not pd.isna(inmet_value):
            row[f"PRELIM_{variable}"] = inmet_value
            row[f"PRELIM_FONTE_{variable}"] = "INMET"
        elif not pd.isna(era5_value):
            row[f"PRELIM_{variable}"] = era5_value
            row[f"PRELIM_FONTE_{variable}"] = "ERA5"
        else:
            row[f"PRELIM_{variable}"] = np.nan
            row[f"PRELIM_FONTE_{variable}"] = ""
    row["TEM_INMET"] = max(int(row[f"TEM_INMET_{variable}"]) for variable in PRIMARY_CLIMATE_VARS)
    row["TEM_ERA5"] = max(int(row[f"TEM_ERA5_{variable}"]) for variable in PRIMARY_CLIMATE_VARS)
    return row

def _finalize_frames(preselected_frames: dict[int, pd.DataFrame], root: Path | None = None) -> dict[int, pd.DataFrame]:
    completed_frames, _ = impute_missing_climate_variables(preselected_frames, root=root)
    prepared: dict[int, pd.DataFrame] = {}
    ordered_years = sorted(completed_frames)
    heatwave_input = pd.concat(
        [
            completed_frames[year][["COD_MUNICIPIO", "DATE", "TMAX_C", "IMPUTADO_TMAX_C"]].copy()
            for year in ordered_years
        ],
        ignore_index=True,
    )
    heatwave_input["HEATWAVE_LEN_P95"] = recompute_heatwave_len_p95(heatwave_input)
    cursor = 0
    for year in ordered_years:
        frame = _prepare_output_frame(completed_frames[year].copy())
        frame["HEATWAVE_LEN_P95"] = heatwave_input.iloc[cursor : cursor + len(frame)]["HEATWAVE_LEN_P95"].to_numpy()
        prepared[year] = frame
        cursor += len(frame)
    return prepared


def _lookup_row(
    *,
    cod_municipio: str,
    nm_municipio: str,
    uf: str,
    municipio_centroid_lat: float,
    municipio_centroid_lon: float,
    era5_lat: float,
    era5_lon: float,
    timezone_name: str,
) -> dict[str, str]:
    return {
        "COD_MUNICIPIO": cod_municipio,
        "NM_MUNICIPIO": nm_municipio,
        "UF": uf,
        "MUNICIPIO_CENTROID_LAT": f"{municipio_centroid_lat}",
        "MUNICIPIO_CENTROID_LON": f"{municipio_centroid_lon}",
        "ERA5_LAT": f"{era5_lat}",
        "ERA5_LON": f"{era5_lon}",
        "TIMEZONE_NAME": timezone_name,
    }


def _write_lookup(root: Path, rows: list[dict[str, str]] | None = None) -> None:
    path = root / "metadata" / "era5_municipio_grid_lookup.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = rows or [
        _lookup_row(
            cod_municipio="1100015",
            nm_municipio="Alta Floresta D'Oeste",
            uf="RO",
            municipio_centroid_lat=-12.470132,
            municipio_centroid_lon=-62.274661,
            era5_lat=-12.5,
            era5_lon=-62.3,
            timezone_name="America/Porto_Velho",
        )
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "COD_MUNICIPIO",
                "NM_MUNICIPIO",
                "UF",
                "MUNICIPIO_CENTROID_LAT",
                "MUNICIPIO_CENTROID_LON",
                "ERA5_LAT",
                "ERA5_LON",
                "TIMEZONE_NAME",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_inmet_bronze(root: Path, year: int, rows: list[dict[str, str]]) -> None:
    path = root / "data" / "bronze" / "climate" / "inmet" / str(year) / f"inmet_{year}_source_preserving.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "CODIGO_WMO",
                "UF",
                "LATITUDE",
                "LONGITUDE",
                "DATA",
                "TEMPERATURA_DO_AR_BULBO_SECO_HORARIA_C",
                "TEMPERATURA_MAXIMA_NA_HORA_ANT_AUT_C",
                "UMIDADE_REL_MIN_NA_HORA_ANT_AUT",
                "UMIDADE_RELATIVA_DO_AR_HORARIA",
                "UMIDADE_REL_MAX_NA_HORA_ANT_AUT",
            ],
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_empty_inmet_bronze(root: Path, year: int) -> None:
    _write_inmet_bronze(root, year, [])


def _write_era5_bronze(
    root: Path,
    year: int,
    missing_dates: set[str] | None = None,
    tmax_base: float = 31.0,
    tmin_base: float = 23.0,
    tmean_base: float = 27.0,
    rh_min_base: float = 42.0,
    rh_mean_base: float = 58.0,
    rh_max_base: float = 79.0,
) -> None:
    path = root / "data" / "bronze" / "climate" / "era5_land" / str(year) / f"era5_land_bronze_daily_municipio_{year}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    missing_dates = missing_dates or set()
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "COD_MUNICIPIO",
                "DATE",
                "TMAX_C",
                "TMIN_C",
                "TMEAN_C",
                "RH_MIN_PCT",
                "RH_MEAN_PCT",
                "RH_MAX_PCT",
                "ERA5_NO_USABLE_DATA",
            ],
            delimiter=";",
        )
        writer.writeheader()
        for day in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"):
            if day.strftime("%Y-%m-%d") in missing_dates:
                continue
            offset = float(day.dayofyear % 7)
            writer.writerow(
                {
                    "COD_MUNICIPIO": "1100015",
                    "DATE": day.strftime("%Y-%m-%d"),
                    "TMAX_C": f"{tmax_base + offset:.4f}",
                    "TMIN_C": f"{tmin_base + offset:.4f}",
                    "TMEAN_C": f"{tmean_base + offset:.4f}",
                    "RH_MIN_PCT": f"{rh_min_base + (offset / 2.0):.4f}",
                    "RH_MEAN_PCT": f"{rh_mean_base + (offset / 3.0):.4f}",
                    "RH_MAX_PCT": f"{rh_max_base + (offset / 4.0):.4f}",
                    "ERA5_NO_USABLE_DATA": "0",
                }
            )


def _write_era5_bronze_rows(root: Path, year: int, rows: list[dict[str, str]]) -> None:
    path = root / "data" / "bronze" / "climate" / "era5_land" / str(year) / f"era5_land_bronze_daily_municipio_{year}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        normalized = dict(row)
        tmax_raw = normalized.get("TMAX_C", "")
        try:
            tmax_value = float(tmax_raw) if str(tmax_raw) != "" else None
        except ValueError:
            tmax_value = None
        if "TMIN_C" not in normalized and tmax_value is not None:
            normalized["TMIN_C"] = f"{tmax_value - 8.0:.4f}"
        if "TMEAN_C" not in normalized and tmax_value is not None:
            tmin_value = float(normalized["TMIN_C"]) if str(normalized.get("TMIN_C", "")) != "" else tmax_value - 8.0
            normalized["TMEAN_C"] = f"{(tmax_value + tmin_value) / 2.0:.4f}"
        normalized_rows.append(normalized)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "COD_MUNICIPIO",
                "DATE",
                "TMAX_C",
                "TMIN_C",
                "TMEAN_C",
                "RH_MIN_PCT",
                "RH_MEAN_PCT",
                "RH_MAX_PCT",
                "ERA5_NO_USABLE_DATA",
            ],
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(normalized_rows)


class SilverClimateHelpersTest(unittest.TestCase):
    def test_station_qc_excludes_c891_for_bbox_and_denylist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_lookup(
                root,
                rows=[
                    _lookup_row(
                        cod_municipio="3522653",
                        nm_municipio="Itapirapua Paulista",
                        uf="SP",
                        municipio_centroid_lat=-24.5723,
                        municipio_centroid_lon=-48.1742,
                        era5_lat=-24.5,
                        era5_lon=-48.2,
                        timezone_name="America/Sao_Paulo",
                    )
                ],
            )
            _write_inmet_bronze(
                root,
                2020,
                [
                    {
                        "CODIGO_WMO": "C891",
                        "UF": "SP",
                        "LATITUDE": "-84",
                        "LONGITUDE": "-79.49416666",
                        "DATA": "2020-01-01",
                        "TEMPERATURA_DO_AR_BULBO_SECO_HORARIA_C": "-40",
                        "TEMPERATURA_MAXIMA_NA_HORA_ANT_AUT_C": "-39",
                        "UMIDADE_REL_MIN_NA_HORA_ANT_AUT": "70",
                        "UMIDADE_RELATIVA_DO_AR_HORARIA": "75",
                        "UMIDADE_REL_MAX_NA_HORA_ANT_AUT": "80",
                    }
                ],
            )
            qc = _build_station_qc_table(root, 2020)
            row = qc.loc[qc["CODIGO_WMO"].eq("C891")].iloc[0]
            self.assertEqual(int(row["station_quarantined_flag"]), 1)
            self.assertIn("outside_brazil_bbox", str(row["station_quarantine_reason"]))
            self.assertIn("manual_denylist", str(row["station_quarantine_reason"]))

    def test_station_qc_excludes_station_over_distance_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_lookup(root)
            _write_inmet_bronze(
                root,
                2020,
                [
                    {
                        "CODIGO_WMO": "A999",
                        "UF": "RO",
                        "LATITUDE": "-1.0000",
                        "LONGITUDE": "-35.0000",
                        "DATA": "2020-01-01",
                        "TEMPERATURA_DO_AR_BULBO_SECO_HORARIA_C": "30",
                        "TEMPERATURA_MAXIMA_NA_HORA_ANT_AUT_C": "31",
                        "UMIDADE_REL_MIN_NA_HORA_ANT_AUT": "45",
                        "UMIDADE_RELATIVA_DO_AR_HORARIA": "60",
                        "UMIDADE_REL_MAX_NA_HORA_ANT_AUT": "75",
                    }
                ],
            )
            qc = _build_station_qc_table(root, 2020)
            row = qc.loc[qc["CODIGO_WMO"].eq("A999")].iloc[0]
            self.assertEqual(int(row["station_quarantined_flag"]), 1)
            self.assertIn("distance_threshold_exceeded", str(row["station_quarantine_reason"]))

    def test_neighbor_graph_is_deterministic_and_excludes_self(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_lookup(
                root,
                rows=[
                    _lookup_row(
                        cod_municipio="3500001",
                        nm_municipio="Town 1",
                        uf="SP",
                        municipio_centroid_lat=-23.0,
                        municipio_centroid_lon=-46.0,
                        era5_lat=-23.0,
                        era5_lon=-46.0,
                        timezone_name="America/Sao_Paulo",
                    ),
                    _lookup_row(
                        cod_municipio="3500002",
                        nm_municipio="Town 2",
                        uf="SP",
                        municipio_centroid_lat=-23.1,
                        municipio_centroid_lon=-46.1,
                        era5_lat=-23.1,
                        era5_lon=-46.1,
                        timezone_name="America/Sao_Paulo",
                    ),
                    _lookup_row(
                        cod_municipio="3500003",
                        nm_municipio="Town 3",
                        uf="SP",
                        municipio_centroid_lat=-23.2,
                        municipio_centroid_lon=-46.2,
                        era5_lat=-23.2,
                        era5_lon=-46.2,
                        timezone_name="America/Sao_Paulo",
                    ),
                    _lookup_row(
                        cod_municipio="3500004",
                        nm_municipio="Town 4",
                        uf="SP",
                        municipio_centroid_lat=-23.3,
                        municipio_centroid_lon=-46.3,
                        era5_lat=-23.3,
                        era5_lon=-46.3,
                        timezone_name="America/Sao_Paulo",
                    ),
                ],
            )
            graph_one = _build_neighbor_graph(root, k=2)
            graph_two = _build_neighbor_graph(root, k=2)
            pd.testing.assert_frame_equal(graph_one, graph_two)
            self.assertTrue(graph_one["COD_MUNICIPIO"].ne(graph_one["NEIGHBOR_COD_MUNICIPIO"]).all())
            self.assertTrue(graph_one.groupby("COD_MUNICIPIO")["NEIGHBOR_RANK"].nunique().eq(2).all())

    def test_full_inmet_row_keeps_inmet_provenance_and_derives_heat_index(self) -> None:
        frame = pd.DataFrame(
            [
                _preselected_row(
                    date_value="2020-01-01",
                    inmet={
                        "TMAX_C": 34.0,
                        "RH_MIN_PCT": 45.0,
                        "RH_MEAN_PCT": 60.0,
                        "RH_MAX_PCT": 81.0,
                    },
                )
            ]
        )
        finalized = _finalize_frames({2020: frame})[2020]
        row = finalized.iloc[0]
        self.assertEqual(row["FONTE_CLIMA_FINAL"], "INMET_FULL")
        self.assertEqual(row["FONTE_TMAX_C_FINAL"], "INMET")
        self.assertEqual(row["FONTE_RH_MEAN_PCT_FINAL"], "INMET")
        self.assertEqual(row["TEM_INMET"], 1)
        self.assertEqual(row["TEM_ERA5"], 0)
        self.assertEqual(row["IMPUTADO_POR_FALLBACK"], 0)
        self.assertFalse(pd.isna(row["HEAT_INDEX_C"]))
        self.assertEqual(row["HEAT_INDEX_INPUTS_ZERO_MISSING"], 1)

    def test_partial_variable_level_fallback_stays_honest(self) -> None:
        frame = pd.DataFrame(
            [
                _preselected_row(
                    date_value="2020-01-01",
                    inmet={
                        "TMAX_C": 35.0,
                        "RH_MIN_PCT": 40.0,
                        "RH_MAX_PCT": 88.0,
                    },
                    era5={"RH_MEAN_PCT": 62.0},
                )
            ]
        )
        finalized = _finalize_frames({2020: frame})[2020]
        row = finalized.iloc[0]
        self.assertEqual(row["TMAX_C"], 35.0)
        self.assertEqual(row["RH_MEAN_PCT"], 62.0)
        self.assertEqual(row["FONTE_TMAX_C_FINAL"], "INMET")
        self.assertEqual(row["FONTE_RH_MEAN_PCT_FINAL"], "ERA5")
        self.assertEqual(row["FONTE_CLIMA_FINAL"], "MIXED_INMET_ERA5")
        self.assertEqual(row["HEAT_INDEX_SOURCE_PAIR"], "DERIVADO_INMET_ERA5")

    def test_imputation_last_resort_fills_missing_and_records_method(self) -> None:
        frames = {
            2020: pd.DataFrame(
                [
                    _preselected_row(
                        date_value="2020-01-05",
                        inmet={"RH_MIN_PCT": 44.0, "RH_MEAN_PCT": 60.0, "RH_MAX_PCT": 82.0},
                    )
                ]
            ),
            2021: pd.DataFrame(
                [
                    _preselected_row(
                        date_value="2021-01-05",
                        inmet={
                            "TMAX_C": 33.0,
                            "RH_MIN_PCT": 43.0,
                            "RH_MEAN_PCT": 59.0,
                            "RH_MAX_PCT": 81.0,
                        },
                    )
                ]
            ),
        }
        finalized = _finalize_frames(frames)[2020]
        row = finalized.iloc[0]
        self.assertEqual(row["TMAX_C"], 33.0)
        self.assertEqual(row["FONTE_TMAX_C_FINAL"], "IMPUTACAO")
        self.assertEqual(row["IMPUTADO_TMAX_C"], 1)
        self.assertEqual(row["IMPUTACAO_METODO_TMAX_C"], "municipality_day_of_year_median")
        self.assertEqual(int(finalized[["TMAX_C", "RH_MIN_PCT", "RH_MEAN_PCT", "RH_MAX_PCT", "HEAT_INDEX_C"]].isna().sum().sum()), 0)

    def test_honest_row_level_provenance_reports_all_three_sources(self) -> None:
        frames = {
            2020: pd.DataFrame(
                [
                    _preselected_row(
                        date_value="2020-01-06",
                        inmet={"TMAX_C": 35.0, "RH_MAX_PCT": 90.0},
                        era5={"RH_MIN_PCT": 42.0},
                    )
                ]
            ),
            2021: pd.DataFrame(
                [
                    _preselected_row(
                        date_value="2021-01-06",
                        era5={"RH_MEAN_PCT": 61.0},
                    )
                ]
            ),
        }
        finalized = _finalize_frames(frames)[2020]
        row = finalized.iloc[0]
        self.assertEqual(row["FONTE_TMAX_C_FINAL"], "INMET")
        self.assertEqual(row["FONTE_RH_MIN_PCT_FINAL"], "ERA5")
        self.assertEqual(row["FONTE_RH_MEAN_PCT_FINAL"], "IMPUTACAO")
        self.assertEqual(row["FONTE_RH_MAX_PCT_FINAL"], "INMET")
        self.assertEqual(row["FONTE_CLIMA_FINAL"], "MIXED_INMET_ERA5_IMPUTED")

    def test_heat_index_formula_regression_96f_65pct(self) -> None:
        dataframe = pd.DataFrame(
            {
                "TMAX_C": [35.5555556],
                "RH_MEAN_PCT": [65.0],
                "FONTE_TMAX_C_FINAL": ["INMET"],
                "FONTE_RH_MEAN_PCT_FINAL": ["INMET"],
            }
        )
        derived = derive_heat_index(dataframe)
        self.assertAlmostEqual(float(derived.iloc[0]["HEAT_INDEX_C"]), 49.4444444, delta=0.25)

    def test_heat_index_formula_regression_100f_55pct(self) -> None:
        dataframe = pd.DataFrame(
            {
                "TMAX_C": [37.7777778],
                "RH_MEAN_PCT": [55.0],
                "FONTE_TMAX_C_FINAL": ["ERA5"],
                "FONTE_RH_MEAN_PCT_FINAL": ["ERA5"],
            }
        )
        derived = derive_heat_index(dataframe)
        self.assertAlmostEqual(float(derived.iloc[0]["HEAT_INDEX_C"]), 51.1111111, delta=0.25)

    def test_heat_index_branch_selection_uses_expected_methods(self) -> None:
        dataframe = pd.DataFrame(
            {
                "TMAX_C": [24.0, 35.0],
                "RH_MEAN_PCT": [40.0, 60.0],
                "FONTE_TMAX_C_FINAL": ["INMET", "ERA5"],
                "FONTE_RH_MEAN_PCT_FINAL": ["INMET", "ERA5"],
            }
        )
        derived = derive_heat_index(dataframe)
        self.assertEqual(derived.iloc[0]["HEAT_INDEX_METHOD"], "nws_simple_pre80f")
        self.assertIn(derived.iloc[1]["HEAT_INDEX_METHOD"], {"nws_rothfusz", "nws_rothfusz_low_rh_adjustment", "nws_rothfusz_high_rh_adjustment"})

    def test_heatwave_reproducibility_matches_recomputed_series(self) -> None:
        frame = pd.DataFrame(
            {
                "COD_MUNICIPIO": ["1100015"] * 5,
                "DATE": [f"2020-01-0{day}" for day in range(1, 6)],
                "TMAX_C": [29.0, 31.0, 33.0, 40.0, 41.0],
                "IMPUTADO_TMAX_C": [0, 0, 0, 0, 0],
            }
        )
        stored = recompute_heatwave_len_p95(frame)
        frame["HEATWAVE_LEN_P95"] = stored
        rerun = recompute_heatwave_len_p95(frame)
        self.assertTrue(np.array_equal(stored.to_numpy(), rerun.to_numpy()))

    def test_upstream_quarantine_prefers_era5_in_preselected_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / "runtime"
            _write_lookup(
                root,
                rows=[
                    _lookup_row(
                        cod_municipio="3522653",
                        nm_municipio="Itapirapua Paulista",
                        uf="SP",
                        municipio_centroid_lat=-24.5723,
                        municipio_centroid_lon=-48.1742,
                        era5_lat=-24.5,
                        era5_lon=-48.2,
                        timezone_name="America/Sao_Paulo",
                    )
                ],
            )
            _write_inmet_bronze(
                root,
                2020,
                [
                    {
                        "CODIGO_WMO": "C891",
                        "UF": "SP",
                        "LATITUDE": "-84",
                        "LONGITUDE": "-79.49416666",
                        "DATA": "2020-01-01",
                        "TEMPERATURA_DO_AR_BULBO_SECO_HORARIA_C": "-49.6",
                        "TEMPERATURA_MAXIMA_NA_HORA_ANT_AUT_C": "-49.6",
                        "UMIDADE_REL_MIN_NA_HORA_ANT_AUT": "70",
                        "UMIDADE_RELATIVA_DO_AR_HORARIA": "75",
                        "UMIDADE_REL_MAX_NA_HORA_ANT_AUT": "80",
                    }
                ],
            )
            _write_era5_bronze_rows(
                root,
                2020,
                [
                    {
                        "COD_MUNICIPIO": "3522653",
                        "DATE": "2020-01-01",
                        "TMAX_C": "31.5",
                        "RH_MIN_PCT": "45",
                        "RH_MEAN_PCT": "60",
                        "RH_MAX_PCT": "78",
                        "ERA5_NO_USABLE_DATA": "0",
                    }
                ],
            )
            preselected, _ = load_or_build_preselected_climate_year(root, runtime_root, 2020, force=True)
            row = preselected.loc[preselected["DATE"].eq("2020-01-01")].iloc[0]
            self.assertEqual(row["PRELIM_FONTE_TMAX_C"], "ERA5")
            self.assertEqual(int(row["CLIMATE_UPSTREAM_QUARANTINE_FLAG"]), 1)
            self.assertIn("outside_brazil_bbox", str(row["CLIMATE_UPSTREAM_QUARANTINE_REASON"]))

    def test_record_anomaly_hard_bound_uses_neighbor_imputation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_lookup(
                root,
                rows=[
                    _lookup_row(cod_municipio="3500001", nm_municipio="Town 1", uf="SP", municipio_centroid_lat=-23.0, municipio_centroid_lon=-46.0, era5_lat=-23.0, era5_lon=-46.0, timezone_name="America/Sao_Paulo"),
                    _lookup_row(cod_municipio="3500002", nm_municipio="Town 2", uf="SP", municipio_centroid_lat=-23.1, municipio_centroid_lon=-46.1, era5_lat=-23.1, era5_lon=-46.1, timezone_name="America/Sao_Paulo"),
                    _lookup_row(cod_municipio="3500003", nm_municipio="Town 3", uf="SP", municipio_centroid_lat=-23.2, municipio_centroid_lon=-46.2, era5_lat=-23.2, era5_lon=-46.2, timezone_name="America/Sao_Paulo"),
                    _lookup_row(cod_municipio="3500004", nm_municipio="Town 4", uf="SP", municipio_centroid_lat=-23.3, municipio_centroid_lon=-46.3, era5_lat=-23.3, era5_lon=-46.3, timezone_name="America/Sao_Paulo"),
                    _lookup_row(cod_municipio="3500005", nm_municipio="Town 5", uf="SP", municipio_centroid_lat=-23.4, municipio_centroid_lon=-46.4, era5_lat=-23.4, era5_lon=-46.4, timezone_name="America/Sao_Paulo"),
                ],
            )
            rows = [
                _preselected_row(
                    date_value="2020-01-01",
                    cod_municipio="3500001",
                    nm_municipio="Town 1",
                    uf="SP",
                    timezone_name="America/Sao_Paulo",
                    municipio_centroid_lat=-23.0,
                    municipio_centroid_lon=-46.0,
                    era5_lat=-23.0,
                    era5_lon=-46.0,
                    inmet={"TMAX_C": -20.0, "RH_MIN_PCT": 40.0, "RH_MEAN_PCT": 55.0, "RH_MAX_PCT": 70.0},
                    era5={"TMAX_C": 30.0, "RH_MIN_PCT": 42.0, "RH_MEAN_PCT": 58.0, "RH_MAX_PCT": 74.0},
                )
            ]
            for index, code in enumerate(("3500002", "3500003", "3500004", "3500005"), start=1):
                rows.append(
                    _preselected_row(
                        date_value="2020-01-01",
                        cod_municipio=code,
                        nm_municipio=f"Town {index + 1}",
                        uf="SP",
                        timezone_name="America/Sao_Paulo",
                        municipio_centroid_lat=-23.0 - (index / 10.0),
                        municipio_centroid_lon=-46.0 - (index / 10.0),
                        era5_lat=-23.0 - (index / 10.0),
                        era5_lon=-46.0 - (index / 10.0),
                        era5={
                            "TMAX_C": 30.0 + index,
                            "RH_MIN_PCT": 40.0 + index,
                            "RH_MEAN_PCT": 55.0 + index,
                            "RH_MAX_PCT": 70.0 + index,
                        },
                    )
                )
            finalized = _finalize_frames({2020: pd.DataFrame(rows)}, root=root)[2020]
            row = finalized.loc[finalized["COD_MUNICIPIO"].eq("3500001")].iloc[0]
            self.assertEqual(int(row["CLIMATE_RECORD_ANOMALY_FLAG"]), 1)
            self.assertEqual(row["CLIMATE_RECORD_ANOMALY_REASON"], "hard_physical_bound_tmax")
            self.assertEqual(row["CLIMATE_CORRECTION_PATH"], "record_anomaly_neighbor_imputation")
            self.assertEqual(row["IMPUTACAO_METODO_TMAX_C"], "neighbor_same_day_k5_median")
            self.assertGreater(float(row["TMAX_C"]), 20.0)
            self.assertGreater(float(row["HEAT_INDEX_C"]), 0.0)

    def test_record_anomaly_divergence_uses_neighbor_imputation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_lookup(
                root,
                rows=[
                    _lookup_row(cod_municipio="3500011", nm_municipio="Town 11", uf="SP", municipio_centroid_lat=-22.0, municipio_centroid_lon=-47.0, era5_lat=-22.0, era5_lon=-47.0, timezone_name="America/Sao_Paulo"),
                    _lookup_row(cod_municipio="3500012", nm_municipio="Town 12", uf="SP", municipio_centroid_lat=-22.1, municipio_centroid_lon=-47.1, era5_lat=-22.1, era5_lon=-47.1, timezone_name="America/Sao_Paulo"),
                    _lookup_row(cod_municipio="3500013", nm_municipio="Town 13", uf="SP", municipio_centroid_lat=-22.2, municipio_centroid_lon=-47.2, era5_lat=-22.2, era5_lon=-47.2, timezone_name="America/Sao_Paulo"),
                    _lookup_row(cod_municipio="3500014", nm_municipio="Town 14", uf="SP", municipio_centroid_lat=-22.3, municipio_centroid_lon=-47.3, era5_lat=-22.3, era5_lon=-47.3, timezone_name="America/Sao_Paulo"),
                    _lookup_row(cod_municipio="3500015", nm_municipio="Town 15", uf="SP", municipio_centroid_lat=-22.4, municipio_centroid_lon=-47.4, era5_lat=-22.4, era5_lon=-47.4, timezone_name="America/Sao_Paulo"),
                ],
            )
            rows = [
                _preselected_row(
                    date_value="2020-02-01",
                    cod_municipio="3500011",
                    nm_municipio="Town 11",
                    uf="SP",
                    timezone_name="America/Sao_Paulo",
                    municipio_centroid_lat=-22.0,
                    municipio_centroid_lon=-47.0,
                    era5_lat=-22.0,
                    era5_lon=-47.0,
                    inmet={"TMAX_C": 44.0, "RH_MIN_PCT": 42.0, "RH_MEAN_PCT": 58.0, "RH_MAX_PCT": 72.0},
                    era5={"TMAX_C": 20.0, "RH_MIN_PCT": 43.0, "RH_MEAN_PCT": 59.0, "RH_MAX_PCT": 73.0},
                )
            ]
            for index, code in enumerate(("3500012", "3500013", "3500014", "3500015"), start=1):
                rows.append(
                    _preselected_row(
                        date_value="2020-02-01",
                        cod_municipio=code,
                        nm_municipio=f"Town {index + 11}",
                        uf="SP",
                        timezone_name="America/Sao_Paulo",
                        municipio_centroid_lat=-22.0 - (index / 10.0),
                        municipio_centroid_lon=-47.0 - (index / 10.0),
                        era5_lat=-22.0 - (index / 10.0),
                        era5_lon=-47.0 - (index / 10.0),
                        era5={
                            "TMAX_C": 20.0 + (index / 2.0),
                            "RH_MIN_PCT": 40.0 + index,
                            "RH_MEAN_PCT": 55.0 + index,
                            "RH_MAX_PCT": 70.0 + index,
                        },
                    )
                )
            finalized = _finalize_frames({2020: pd.DataFrame(rows)}, root=root)[2020]
            row = finalized.loc[finalized["COD_MUNICIPIO"].eq("3500011")].iloc[0]
            self.assertEqual(int(row["CLIMATE_RECORD_ANOMALY_FLAG"]), 1)
            self.assertEqual(row["CLIMATE_RECORD_ANOMALY_REASON"], "cross_source_and_neighbor_divergence_tmax")
            self.assertEqual(row["CLIMATE_CORRECTION_PATH"], "record_anomaly_neighbor_imputation")
            self.assertEqual(row["IMPUTACAO_METODO_TMAX_C"], "neighbor_same_day_k5_median")

    def test_anomaly_falls_through_to_historical_fallback_when_neighbors_are_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_lookup(
                root,
                rows=[
                    _lookup_row(cod_municipio="3500101", nm_municipio="Town 101", uf="SP", municipio_centroid_lat=-21.0, municipio_centroid_lon=-46.0, era5_lat=-21.0, era5_lon=-46.0, timezone_name="America/Sao_Paulo"),
                    _lookup_row(cod_municipio="3500102", nm_municipio="Town 102", uf="SP", municipio_centroid_lat=-21.1, municipio_centroid_lon=-46.1, era5_lat=-21.1, era5_lon=-46.1, timezone_name="America/Sao_Paulo"),
                    _lookup_row(cod_municipio="3500103", nm_municipio="Town 103", uf="SP", municipio_centroid_lat=-21.2, municipio_centroid_lon=-46.2, era5_lat=-21.2, era5_lon=-46.2, timezone_name="America/Sao_Paulo"),
                ],
            )
            frames = {
                2020: pd.DataFrame(
                    [
                        _preselected_row(
                            date_value="2020-01-05",
                            cod_municipio="3500101",
                            nm_municipio="Town 101",
                            uf="SP",
                            timezone_name="America/Sao_Paulo",
                            municipio_centroid_lat=-21.0,
                            municipio_centroid_lon=-46.0,
                            era5_lat=-21.0,
                            era5_lon=-46.0,
                            inmet={"TMAX_C": -10.0, "RH_MIN_PCT": 40.0, "RH_MEAN_PCT": 55.0, "RH_MAX_PCT": 70.0},
                            era5={"TMAX_C": 29.0, "RH_MIN_PCT": 41.0, "RH_MEAN_PCT": 56.0, "RH_MAX_PCT": 71.0},
                        ),
                        _preselected_row(
                            date_value="2020-01-05",
                            cod_municipio="3500102",
                            nm_municipio="Town 102",
                            uf="SP",
                            timezone_name="America/Sao_Paulo",
                            municipio_centroid_lat=-21.1,
                            municipio_centroid_lon=-46.1,
                            era5_lat=-21.1,
                            era5_lon=-46.1,
                            era5={"TMAX_C": 30.0, "RH_MIN_PCT": 42.0, "RH_MEAN_PCT": 57.0, "RH_MAX_PCT": 72.0},
                        ),
                        _preselected_row(
                            date_value="2020-01-05",
                            cod_municipio="3500103",
                            nm_municipio="Town 103",
                            uf="SP",
                            timezone_name="America/Sao_Paulo",
                            municipio_centroid_lat=-21.2,
                            municipio_centroid_lon=-46.2,
                            era5_lat=-21.2,
                            era5_lon=-46.2,
                            era5={"TMAX_C": 31.0, "RH_MIN_PCT": 43.0, "RH_MEAN_PCT": 58.0, "RH_MAX_PCT": 73.0},
                        ),
                    ]
                ),
                2021: pd.DataFrame(
                    [
                        _preselected_row(
                            date_value="2021-01-05",
                            cod_municipio="3500101",
                            nm_municipio="Town 101",
                            uf="SP",
                            timezone_name="America/Sao_Paulo",
                            municipio_centroid_lat=-21.0,
                            municipio_centroid_lon=-46.0,
                            era5_lat=-21.0,
                            era5_lon=-46.0,
                            era5={"TMAX_C": 28.5, "RH_MIN_PCT": 41.5, "RH_MEAN_PCT": 56.5, "RH_MAX_PCT": 71.5},
                        )
                    ]
                ),
            }
            finalized = _finalize_frames(frames, root=root)[2020]
            row = finalized.loc[finalized["COD_MUNICIPIO"].eq("3500101")].iloc[0]
            self.assertEqual(int(row["CLIMATE_RECORD_ANOMALY_FLAG"]), 1)
            self.assertEqual(row["CLIMATE_CORRECTION_PATH"], "record_anomaly_fallback_imputation")
            self.assertEqual(row["IMPUTACAO_METODO_TMAX_C"], "municipality_day_of_year_median")


class SilverClimateIntegrationTest(unittest.TestCase):
    def test_dense_panel_completion_fills_holes_to_zero_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / "runtime"
            _write_lookup(root)
            _write_empty_inmet_bronze(root, 2020)
            _write_era5_bronze(root, 2020, missing_dates={"2020-01-10", "2020-02-20"})
            preselected, _ = load_or_build_preselected_climate_year(root, runtime_root, 2020, force=False)
            self.assertEqual(len(preselected), 366)
            finalized = _finalize_frames({2020: preselected})[2020]
            self.assertEqual(len(finalized), 366)
            self.assertEqual(int(finalized[["TMAX_C", "RH_MIN_PCT", "RH_MEAN_PCT", "RH_MAX_PCT", "HEAT_INDEX_C"]].isna().sum().sum()), 0)

    def test_unreadable_premerged_year_recovery_rebuilds_and_refreshes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / "runtime"
            _write_lookup(root)
            _write_empty_inmet_bronze(root, 2020)
            _write_era5_bronze(root, 2020)
            broken_runtime = runtime_root / "silver_climate_preselected" / "YEAR=2020" / "climate_preselected_daily.parquet"
            broken_runtime.parent.mkdir(parents=True, exist_ok=True)
            broken_runtime.write_bytes(b"not-a-parquet-file")
            log_path = root / "artifacts" / "run_logs" / "test.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("", encoding="utf-8")

            report = promote_climate_silver_range(
                root=root,
                runtime_root=runtime_root,
                start_year=2020,
                end_year=2020,
                log_path=log_path,
                command="test-promote-silver",
                force=False,
            )

            self.assertTrue(report["preselected_metadata"][2020]["rebuilt_from_components"])
            partition_path = silver_partition_file(root, 2020)
            self.assertTrue(partition_path.exists())
            persisted = pd.read_parquet(partition_path)
            self.assertEqual(len(persisted), 366)
            self.assertEqual(int(persisted[["TMAX_C", "RH_MIN_PCT", "RH_MEAN_PCT", "RH_MAX_PCT", "HEAT_INDEX_C"]].isna().sum().sum()), 0)

            manifest_path = root / "artifacts" / "manifests" / "silver_climate.manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["entry_count"], 1)
            self.assertEqual(manifest["entries"][0]["row_count"], 366)
            self.assertEqual(build_manifest(root, "silver", "climate")["entries"][0]["row_count"], 366)

    def test_invalid_upstream_station_falls_back_to_era5_and_writes_qc_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / "runtime"
            _write_lookup(
                root,
                rows=[
                    _lookup_row(
                        cod_municipio="3522653",
                        nm_municipio="Itapirapua Paulista",
                        uf="SP",
                        municipio_centroid_lat=-24.5723,
                        municipio_centroid_lon=-48.1742,
                        era5_lat=-24.5,
                        era5_lon=-48.2,
                        timezone_name="America/Sao_Paulo",
                    )
                ],
            )
            _write_inmet_bronze(
                root,
                2020,
                [
                    {
                        "CODIGO_WMO": "C891",
                        "UF": "SP",
                        "LATITUDE": "-84",
                        "LONGITUDE": "-79.49416666",
                        "DATA": day.strftime("%Y-%m-%d"),
                        "TEMPERATURA_DO_AR_BULBO_SECO_HORARIA_C": "-49.6",
                        "TEMPERATURA_MAXIMA_NA_HORA_ANT_AUT_C": "-49.6",
                        "UMIDADE_REL_MIN_NA_HORA_ANT_AUT": "70",
                        "UMIDADE_RELATIVA_DO_AR_HORARIA": "75",
                        "UMIDADE_REL_MAX_NA_HORA_ANT_AUT": "80",
                    }
                    for day in pd.date_range("2020-01-01", "2020-12-31", freq="D")
                ],
            )
            _write_era5_bronze_rows(
                root,
                2020,
                [
                    {
                        "COD_MUNICIPIO": "3522653",
                        "DATE": day.strftime("%Y-%m-%d"),
                        "TMAX_C": "31.5",
                        "RH_MIN_PCT": "45",
                        "RH_MEAN_PCT": "60",
                        "RH_MAX_PCT": "78",
                        "ERA5_NO_USABLE_DATA": "0",
                    }
                    for day in pd.date_range("2020-01-01", "2020-12-31", freq="D")
                ],
            )
            log_path = root / "artifacts" / "run_logs" / "test.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("", encoding="utf-8")

            report = promote_climate_silver_range(
                root=root,
                runtime_root=runtime_root,
                start_year=2020,
                end_year=2020,
                log_path=log_path,
                command="test-promote-silver",
                force=True,
            )

            persisted = pd.read_parquet(silver_partition_file(root, 2020))
            municipality_rows = persisted.loc[persisted["COD_MUNICIPIO"].eq("3522653")]
            self.assertFalse(municipality_rows["FONTE_CLIMA_FINAL"].eq("INMET_FULL").any())
            self.assertGreater(float(municipality_rows["TMAX_C"].min()), 0.0)
            self.assertGreater(float(municipality_rows["HEAT_INDEX_C"].min()), 0.0)
            self.assertTrue(municipality_rows["CLIMATE_UPSTREAM_QUARANTINE_FLAG"].eq(1).any())

            station_qc_path = root / "artifacts" / "validation_reports" / "silver" / "climate" / "2020_station_qc.csv"
            anomaly_rows_path = root / "artifacts" / "validation_reports" / "silver" / "climate" / "2020_anomaly_rows.csv"
            self.assertTrue(station_qc_path.exists())
            self.assertTrue(anomaly_rows_path.exists())
            station_qc = pd.read_csv(station_qc_path, dtype=str)
            c891_row = station_qc.loc[station_qc["CODIGO_WMO"].eq("C891")].iloc[0]
            self.assertIn("outside_brazil_bbox", str(c891_row["station_quarantine_reason"]))
            self.assertEqual(report["validation"]["years"][2020]["upstream_station_quarantine_count"], 1)


if __name__ == "__main__":
    unittest.main()
