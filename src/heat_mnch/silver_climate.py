from __future__ import annotations

import ast
import json
import math
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .bronze import bronze_output_path, raw_paths_for_dataset
from .era5_land import ERA5_NO_USABLE_DATA_FIELD, build_bronze_annual_rows, era5_lookup_path
from .manifest import build_manifest
from .run_log import append_run_event


PRIMARY_CLIMATE_VARS = ("TMAX_C", "TMIN_C", "TMEAN_C", "RH_MIN_PCT", "RH_MEAN_PCT", "RH_MAX_PCT")
CLIMATE_KEY_FIELDS = ("COD_MUNICIPIO", "DATE")
READBACK_TOLERANCE = 1e-6
HEAT_INDEX_TOLERANCE = 1e-5
RUNTIME_PRESELECTED_DIRNAME = "silver_climate_preselected"
PRESELECTED_FILENAME = "climate_preselected_daily.parquet"
FINAL_FILENAME_TEMPLATE = "climate_canonical_daily_{year}.parquet"
INMET_STATION_QC_CONFIG = Path("config") / "climate" / "inmet_station_qc.yaml"
DEFAULT_INMET_STATION_QC = {
    "brazil_latitude_min": -35.0,
    "brazil_latitude_max": 6.0,
    "brazil_longitude_min": -75.0,
    "brazil_longitude_max": -32.0,
    "max_station_to_municipality_distance_km": 150.0,
    "manual_station_denylist": ["C891"],
}
NEIGHBOR_IMPUTATION_METHOD = "neighbor_same_day_k5_median"
CLIMATE_CORRECTION_PATHS = {
    "none",
    "upstream_station_quarantine_to_era5",
    "record_anomaly_neighbor_imputation",
    "record_anomaly_fallback_imputation",
}
ANOMALY_TMAX_MIN_C = -5.0
ANOMALY_TMAX_MAX_C = 50.0
ANOMALY_ERA5_DIVERGENCE_C = 20.0
ANOMALY_NEIGHBOR_DIVERGENCE_C = 15.0
NEIGHBOR_GRAPH_K = 5
NEIGHBOR_GRAPH_MIN_REQUIRED = 3

SILVER_BASE_FIELDS = [
    "COD_MUNICIPIO",
    "NM_MUNICIPIO",
    "UF",
    "TIMEZONE_NAME",
    "DATE",
    "YEAR",
    "MONTH",
    "MUNICIPIO_CENTROID_LAT",
    "MUNICIPIO_CENTROID_LON",
    "ERA5_LAT",
    "ERA5_LON",
]
SILVER_REQUIRED_FIELDS = [
    "TMAX_C",
    "TMIN_C",
    "TMEAN_C",
    "RH_MIN_PCT",
    "RH_MEAN_PCT",
    "RH_MAX_PCT",
    "FONTE_CLIMA_FINAL",
    "TEM_INMET",
    "TEM_ERA5",
    "IMPUTADO_POR_FALLBACK",
    "CLIMATE_UPSTREAM_QUARANTINE_FLAG",
    "CLIMATE_UPSTREAM_QUARANTINE_REASON",
    "CLIMATE_RECORD_ANOMALY_FLAG",
    "CLIMATE_RECORD_ANOMALY_REASON",
    "CLIMATE_CORRECTION_PATH",
    "HEATWAVE_LEN_P95",
    "TEM_INMET_TMAX_C",
    "TEM_INMET_TMIN_C",
    "TEM_INMET_TMEAN_C",
    "TEM_INMET_RH_MIN_PCT",
    "TEM_INMET_RH_MEAN_PCT",
    "TEM_INMET_RH_MAX_PCT",
    "TEM_ERA5_TMAX_C",
    "TEM_ERA5_TMIN_C",
    "TEM_ERA5_TMEAN_C",
    "TEM_ERA5_RH_MIN_PCT",
    "TEM_ERA5_RH_MEAN_PCT",
    "TEM_ERA5_RH_MAX_PCT",
    "FONTE_TMAX_C_FINAL",
    "FONTE_TMIN_C_FINAL",
    "FONTE_TMEAN_C_FINAL",
    "FONTE_RH_MIN_PCT_FINAL",
    "FONTE_RH_MEAN_PCT_FINAL",
    "FONTE_RH_MAX_PCT_FINAL",
    "IMPUTADO_TMAX_C",
    "IMPUTADO_TMIN_C",
    "IMPUTADO_TMEAN_C",
    "IMPUTADO_RH_MIN_PCT",
    "IMPUTADO_RH_MEAN_PCT",
    "IMPUTADO_RH_MAX_PCT",
    "IMPUTACAO_METODO_TMAX_C",
    "IMPUTACAO_METODO_TMIN_C",
    "IMPUTACAO_METODO_TMEAN_C",
    "IMPUTACAO_METODO_RH_MIN_PCT",
    "IMPUTACAO_METODO_RH_MEAN_PCT",
    "IMPUTACAO_METODO_RH_MAX_PCT",
    "HEAT_INDEX_C",
    "HEAT_INDEX_METHOD",
    "HEAT_INDEX_SOURCE_PAIR",
    "HEAT_INDEX_INPUTS_ZERO_MISSING",
]
SILVER_OUTPUT_FIELDS = SILVER_BASE_FIELDS + SILVER_REQUIRED_FIELDS
PRESELECTED_REQUIRED_FIELDS = set(SILVER_BASE_FIELDS + ["DAY_OF_YEAR"])
for _variable in PRIMARY_CLIMATE_VARS:
    PRESELECTED_REQUIRED_FIELDS.update(
        {
            f"INMET_{_variable}",
            f"ERA5_{_variable}",
            f"TEM_INMET_{_variable}",
            f"TEM_ERA5_{_variable}",
            f"PRELIM_{_variable}",
            f"PRELIM_FONTE_{_variable}",
        }
    )
PRESELECTED_REQUIRED_FIELDS.update({"TEM_INMET", "TEM_ERA5"})
PRESELECTED_REQUIRED_FIELDS.update(
    {
        "CLIMATE_UPSTREAM_QUARANTINE_FLAG",
        "CLIMATE_UPSTREAM_QUARANTINE_REASON",
        "CLIMATE_RECORD_ANOMALY_FLAG",
        "CLIMATE_RECORD_ANOMALY_REASON",
        "CLIMATE_CORRECTION_PATH",
    }
)
SOURCE_SET_TO_SUMMARY = {
    frozenset({"INMET"}): "INMET_FULL",
    frozenset({"ERA5"}): "ERA5_FULL",
    frozenset({"IMPUTACAO"}): "IMPUTED_FULL",
    frozenset({"INMET", "ERA5"}): "MIXED_INMET_ERA5",
    frozenset({"INMET", "IMPUTACAO"}): "MIXED_INMET_IMPUTED",
    frozenset({"ERA5", "IMPUTACAO"}): "MIXED_ERA5_IMPUTED",
    frozenset({"INMET", "ERA5", "IMPUTACAO"}): "MIXED_INMET_ERA5_IMPUTED",
}
IMPUTATION_HIERARCHY = (
    ("municipality_day_of_year_median", ["COD_MUNICIPIO", "DAY_OF_YEAR"]),
    ("municipality_month_within_year_median", ["COD_MUNICIPIO", "YEAR", "MONTH"]),
    ("municipality_month_pooled_2020_2024_median", ["COD_MUNICIPIO", "MONTH"]),
    ("state_month_median", ["UF", "MONTH"]),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_runtime_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _clean_relative_humidity(series: pd.Series) -> pd.Series:
    numeric = _numeric_series(series)
    return numeric.where(numeric.between(0.0, 100.0, inclusive="both"))


def _load_inmet_station_qc_config(root: Path) -> dict[str, Any]:
    config = dict(DEFAULT_INMET_STATION_QC)
    path = root / INMET_STATION_QC_CONFIG
    if not path.exists():
        return config
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if key not in config:
            continue
        if key == "manual_station_denylist":
            parsed = ast.literal_eval(value) if value else []
            if not isinstance(parsed, list):
                raise ValueError(f"manual_station_denylist must be a YAML-style list in {path}")
            config[key] = [str(item).strip() for item in parsed if str(item).strip()]
            continue
        config[key] = float(value)
    return config


def _haversine_km(
    latitude_1: float | np.ndarray,
    longitude_1: float | np.ndarray,
    latitude_2: float | np.ndarray,
    longitude_2: float | np.ndarray,
) -> np.ndarray:
    lat1 = np.radians(np.asarray(latitude_1, dtype=float))
    lon1 = np.radians(np.asarray(longitude_1, dtype=float))
    lat2 = np.radians(np.asarray(latitude_2, dtype=float))
    lon2 = np.radians(np.asarray(longitude_2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    haversine = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
    earth_radius_km = 6371.0088
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def _climate_lookup_frame(root: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(
        era5_lookup_path(root),
        dtype={"COD_MUNICIPIO": str, "UF": str, "TIMEZONE_NAME": str},
    ).copy()
    for column in ("MUNICIPIO_CENTROID_LAT", "MUNICIPIO_CENTROID_LON", "ERA5_LAT", "ERA5_LON"):
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return dataframe.sort_values("COD_MUNICIPIO").reset_index(drop=True)


def _dense_calendar_frame(year: int) -> pd.DataFrame:
    dates = pd.date_range(date(year, 1, 1), date(year, 12, 31), freq="D")
    return pd.DataFrame(
        {
            "DATE": dates.strftime("%Y-%m-%d"),
            "YEAR": np.full(len(dates), year, dtype=np.int16),
            "MONTH": dates.month.astype(np.int16),
            "DAY_OF_YEAR": dates.dayofyear.astype(np.int16),
        }
    )


def silver_partition_dir(root: Path, year: int) -> Path:
    return root / "data" / "silver" / "climate" / f"YEAR={year}"


def silver_partition_file(root: Path, year: int) -> Path:
    return silver_partition_dir(root, year) / FINAL_FILENAME_TEMPLATE.format(year=year)


def runtime_preselected_file(runtime_root: Path, year: int) -> Path:
    return runtime_root / RUNTIME_PRESELECTED_DIRNAME / f"YEAR={year}" / PRESELECTED_FILENAME


def _read_parquet_verified(path: Path) -> pd.DataFrame:
    dataframe = pd.read_parquet(path)
    if dataframe.empty and pq.ParquetFile(path).metadata.num_rows > 0:
        raise IOError(f"Failed to read rows from parquet file: {path}")
    return dataframe


def _write_runtime_parquet(dataframe: pd.DataFrame, path: Path) -> pd.DataFrame:
    if path.parent.exists():
        shutil.rmtree(path.parent)
    _ensure_runtime_parent(path)
    dataframe.to_parquet(path, index=False)
    readback = _read_parquet_verified(path)
    if len(readback) != len(dataframe) or list(readback.columns) != list(dataframe.columns):
        raise IOError(f"Runtime parquet verification failed for {path}")
    return readback


def _validate_preselected_runtime_frame(root: Path, year: int, dataframe: pd.DataFrame) -> None:
    missing_columns = sorted(PRESELECTED_REQUIRED_FIELDS - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Preselected climate runtime parquet is missing columns for {year}: {missing_columns}")
    _validate_unique_keys("preselected climate runtime parquet", dataframe)
    lookup = _climate_lookup_frame(root)
    expected_rows = int(lookup["COD_MUNICIPIO"].nunique()) * len(pd.date_range(date(year, 1, 1), date(year, 12, 31), freq="D"))
    if len(dataframe) != expected_rows:
        raise ValueError(
            f"Preselected climate runtime parquet has stale row count for {year}: "
            f"{len(dataframe)} != {expected_rows}"
        )


def _sync_partition_verified(staging_file: Path, final_file: Path) -> None:
    staging_frame = _read_parquet_verified(staging_file)
    temp_dir = final_file.parent.parent / f".{final_file.parent.name}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / final_file.name
    shutil.copy2(staging_file, temp_file)
    temp_frame = _read_parquet_verified(temp_file)
    if len(temp_frame) != len(staging_frame) or list(temp_frame.columns) != list(staging_frame.columns):
        raise IOError(f"Final parquet verification failed for {temp_file}")
    if final_file.parent.exists():
        shutil.rmtree(final_file.parent)
    temp_dir.rename(final_file.parent)


def _build_station_qc_table(root: Path, year: int) -> pd.DataFrame:
    path = bronze_output_path(root, "inmet", year)
    config = _load_inmet_station_qc_config(root)
    station_meta = pd.read_csv(
        path,
        compression="gzip",
        sep=";",
        dtype=str,
        usecols=["CODIGO_WMO", "UF", "LATITUDE", "LONGITUDE"],
        keep_default_na=False,
    ).drop_duplicates(subset=["CODIGO_WMO"])
    station_meta["LATITUDE_NUM"] = _numeric_series(station_meta["LATITUDE"])
    station_meta["LONGITUDE_NUM"] = _numeric_series(station_meta["LONGITUDE"])
    lookup = _climate_lookup_frame(root)
    mapping_rows: list[dict[str, object]] = []
    for uf, station_group in station_meta.groupby("UF", sort=False):
        municipality_group = lookup[lookup["UF"] == uf].reset_index(drop=True)
        municipality_codes = municipality_group["COD_MUNICIPIO"].astype(str).to_numpy(dtype=object)
        municipality_names = municipality_group["NM_MUNICIPIO"].astype(str).to_numpy(dtype=object)
        municipality_lats = municipality_group["MUNICIPIO_CENTROID_LAT"].to_numpy(dtype=float) if not municipality_group.empty else np.array([], dtype=float)
        municipality_lons = municipality_group["MUNICIPIO_CENTROID_LON"].to_numpy(dtype=float) if not municipality_group.empty else np.array([], dtype=float)
        coordinates = municipality_group[["MUNICIPIO_CENTROID_LAT", "MUNICIPIO_CENTROID_LON"]].to_numpy(dtype=float)
        for row in station_group.itertuples(index=False):
            nearest_index = -1
            nearest_distance = np.nan
            nearest_code = ""
            nearest_name = ""
            if municipality_group.empty or pd.isna(row.LATITUDE_NUM) or pd.isna(row.LONGITUDE_NUM):
                distances = np.array([], dtype=float)
            else:
                distances = _haversine_km(float(row.LATITUDE_NUM), float(row.LONGITUDE_NUM), municipality_lats, municipality_lons)
            if distances.size:
                nearest_index = int(distances.argmin())
                nearest_distance = float(distances[nearest_index])
                nearest_code = str(municipality_codes[nearest_index])
                nearest_name = str(municipality_names[nearest_index])
            inside_bbox = bool(
                pd.notna(row.LATITUDE_NUM)
                and pd.notna(row.LONGITUDE_NUM)
                and float(config["brazil_latitude_min"]) <= float(row.LATITUDE_NUM) <= float(config["brazil_latitude_max"])
                and float(config["brazil_longitude_min"]) <= float(row.LONGITUDE_NUM) <= float(config["brazil_longitude_max"])
            )
            distance_threshold_pass = bool(
                nearest_code
                and pd.notna(nearest_distance)
                and nearest_distance <= float(config["max_station_to_municipality_distance_km"])
            )
            manual_denylist = str(row.CODIGO_WMO) in set(config["manual_station_denylist"])
            quarantine_reasons: list[str] = []
            if pd.isna(row.LATITUDE_NUM) or pd.isna(row.LONGITUDE_NUM):
                quarantine_reasons.append("missing_station_coordinates")
            if not inside_bbox:
                quarantine_reasons.append("outside_brazil_bbox")
            if nearest_code == "":
                quarantine_reasons.append("no_same_uf_municipality_match")
            elif not distance_threshold_pass:
                quarantine_reasons.append("distance_threshold_exceeded")
            if manual_denylist:
                quarantine_reasons.append("manual_denylist")
            mapping_rows.append(
                {
                    "CODIGO_WMO": str(row.CODIGO_WMO),
                    "UF": str(row.UF),
                    "station_latitude": float(row.LATITUDE_NUM) if pd.notna(row.LATITUDE_NUM) else np.nan,
                    "station_longitude": float(row.LONGITUDE_NUM) if pd.notna(row.LONGITUDE_NUM) else np.nan,
                    "nearest_municipality_code": nearest_code,
                    "nearest_municipality_name": nearest_name,
                    "station_to_municipality_distance_km": nearest_distance,
                    "inside_brazil_bbox_flag": np.int8(1 if inside_bbox else 0),
                    "distance_threshold_pass_flag": np.int8(1 if distance_threshold_pass else 0),
                    "manual_denylist_flag": np.int8(1 if manual_denylist else 0),
                    "station_quarantined_flag": np.int8(1 if quarantine_reasons else 0),
                    "station_quarantine_reason": "|".join(quarantine_reasons),
                }
            )
    columns = [
        "CODIGO_WMO",
        "UF",
        "station_latitude",
        "station_longitude",
        "nearest_municipality_code",
        "nearest_municipality_name",
        "station_to_municipality_distance_km",
        "inside_brazil_bbox_flag",
        "distance_threshold_pass_flag",
        "manual_denylist_flag",
        "station_quarantined_flag",
        "station_quarantine_reason",
    ]
    if not mapping_rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(mapping_rows, columns=columns).drop_duplicates(subset=["CODIGO_WMO"]).reset_index(drop=True)


def _legacy_station_to_municipality_map(root: Path, year: int) -> pd.DataFrame:
    path = bronze_output_path(root, "inmet", year)
    station_meta = pd.read_csv(
        path,
        compression="gzip",
        sep=";",
        dtype=str,
        usecols=["CODIGO_WMO", "UF", "LATITUDE", "LONGITUDE"],
        keep_default_na=False,
    ).drop_duplicates(subset=["CODIGO_WMO"])
    station_meta["LATITUDE_NUM"] = _numeric_series(station_meta["LATITUDE"])
    station_meta["LONGITUDE_NUM"] = _numeric_series(station_meta["LONGITUDE"])
    station_meta = station_meta.dropna(subset=["LATITUDE_NUM", "LONGITUDE_NUM"])
    lookup = _climate_lookup_frame(root)
    mapping_rows: list[dict[str, object]] = []
    for uf, station_group in station_meta.groupby("UF", sort=False):
        municipality_group = lookup[lookup["UF"] == uf].reset_index(drop=True)
        if municipality_group.empty:
            continue
        coordinates = municipality_group[["MUNICIPIO_CENTROID_LAT", "MUNICIPIO_CENTROID_LON"]].to_numpy(dtype=float)
        for row in station_group.itertuples(index=False):
            diff = ((coordinates[:, 0] - float(row.LATITUDE_NUM)) ** 2) + (
                (coordinates[:, 1] - float(row.LONGITUDE_NUM)) ** 2
            )
            nearest_index = int(diff.argmin())
            mapping_rows.append(
                {
                    "CODIGO_WMO": str(row.CODIGO_WMO),
                    "COD_MUNICIPIO": str(municipality_group.iloc[nearest_index]["COD_MUNICIPIO"]),
                }
            )
    if not mapping_rows:
        return pd.DataFrame(columns=["CODIGO_WMO", "COD_MUNICIPIO"])
    return pd.DataFrame(mapping_rows).drop_duplicates(subset=["CODIGO_WMO"]).reset_index(drop=True)


def _valid_station_to_municipality_map(root: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    station_qc = _build_station_qc_table(root, year)
    valid_map = (
        station_qc.loc[
            station_qc["station_quarantined_flag"].eq(0) & station_qc["nearest_municipality_code"].astype(str).ne(""),
            ["CODIGO_WMO", "nearest_municipality_code"],
        ]
        .rename(columns={"nearest_municipality_code": "COD_MUNICIPIO"})
        .drop_duplicates(subset=["CODIGO_WMO"])
        .reset_index(drop=True)
    )
    return valid_map, station_qc


def _build_neighbor_graph(root: Path, k: int = NEIGHBOR_GRAPH_K) -> pd.DataFrame:
    lookup = _climate_lookup_frame(root)[["COD_MUNICIPIO", "UF", "MUNICIPIO_CENTROID_LAT", "MUNICIPIO_CENTROID_LON"]].copy()
    rows: list[dict[str, object]] = []
    for _, municipality_group in lookup.groupby("UF", sort=True):
        if len(municipality_group) <= 1:
            continue
        codes = municipality_group["COD_MUNICIPIO"].astype(str).to_numpy(dtype=object)
        latitudes = municipality_group["MUNICIPIO_CENTROID_LAT"].to_numpy(dtype=float)
        longitudes = municipality_group["MUNICIPIO_CENTROID_LON"].to_numpy(dtype=float)
        neighbor_count = min(k, len(municipality_group) - 1)
        for index, municipality_code in enumerate(codes):
            distances = _haversine_km(latitudes[index], longitudes[index], latitudes, longitudes)
            distances[index] = np.inf
            nearest_indexes = np.argsort(distances)[:neighbor_count]
            for rank, neighbor_index in enumerate(nearest_indexes, start=1):
                rows.append(
                    {
                        "COD_MUNICIPIO": str(municipality_code),
                        "NEIGHBOR_RANK": int(rank),
                        "NEIGHBOR_COD_MUNICIPIO": str(codes[neighbor_index]),
                        "NEIGHBOR_DISTANCE_KM": float(distances[neighbor_index]),
                    }
                )
    return pd.DataFrame(rows, columns=["COD_MUNICIPIO", "NEIGHBOR_RANK", "NEIGHBOR_COD_MUNICIPIO", "NEIGHBOR_DISTANCE_KM"])


def _empty_inmet_candidates() -> pd.DataFrame:
    columns = [
        "COD_MUNICIPIO",
        "DATE",
        "INMET_TMAX_C",
        "INMET_TMIN_C",
        "INMET_TMEAN_C",
        "INMET_RH_MIN_PCT",
        "INMET_RH_MEAN_PCT",
        "INMET_RH_MAX_PCT",
        "TEM_INMET_TMAX_C",
        "TEM_INMET_TMIN_C",
        "TEM_INMET_TMEAN_C",
        "TEM_INMET_RH_MIN_PCT",
        "TEM_INMET_RH_MEAN_PCT",
        "TEM_INMET_RH_MAX_PCT",
    ]
    return pd.DataFrame(columns=columns)


def load_inmet_daily_candidates(
    root: Path,
    year: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    inmet_path = bronze_output_path(root, "inmet", year)
    if not inmet_path.exists():
        raise FileNotFoundError(f"INMET bronze missing for {year}: {inmet_path}")
    station_map, station_qc = _valid_station_to_municipality_map(root, year)
    legacy_station_map = _legacy_station_to_municipality_map(root, year)
    quarantined_station_map = (
        station_qc.loc[
            station_qc["station_quarantined_flag"].eq(1),
            ["CODIGO_WMO", "station_quarantine_reason"],
        ]
        .merge(legacy_station_map, on="CODIGO_WMO", how="left")
        .reset_index(drop=True)
    )
    metadata = {
        "station_count": int(station_qc["CODIGO_WMO"].nunique()),
        "upstream_station_quarantine_count": int(station_qc["station_quarantined_flag"].sum()),
        "upstream_station_quarantine_wmo_codes": sorted(
            station_qc.loc[station_qc["station_quarantined_flag"].eq(1), "CODIGO_WMO"].astype(str).unique().tolist()
        ),
    }
    partials: list[pd.DataFrame] = []
    quarantine_partials: list[pd.DataFrame] = []
    usecols = [
        "CODIGO_WMO",
        "DATA",
        "TEMPERATURA_DO_AR_BULBO_SECO_HORARIA_C",
        "TEMPERATURA_MAXIMA_NA_HORA_ANT_AUT_C",
        "UMIDADE_REL_MIN_NA_HORA_ANT_AUT",
        "UMIDADE_RELATIVA_DO_AR_HORARIA",
        "UMIDADE_REL_MAX_NA_HORA_ANT_AUT",
    ]
    for chunk in pd.read_csv(
        inmet_path,
        compression="gzip",
        sep=";",
        dtype=str,
        usecols=usecols,
        keep_default_na=False,
        chunksize=250_000,
        low_memory=False,
    ):
        valid_chunk = chunk.merge(station_map, on="CODIGO_WMO", how="inner")
        if not valid_chunk.empty:
            dry_bulb = _numeric_series(valid_chunk["TEMPERATURA_DO_AR_BULBO_SECO_HORARIA_C"])
            prior_hour_max = _numeric_series(valid_chunk["TEMPERATURA_MAXIMA_NA_HORA_ANT_AUT_C"])
            hourly_rh = _clean_relative_humidity(valid_chunk["UMIDADE_RELATIVA_DO_AR_HORARIA"])
            prior_hour_rh_min = _clean_relative_humidity(valid_chunk["UMIDADE_REL_MIN_NA_HORA_ANT_AUT"])
            prior_hour_rh_max = _clean_relative_humidity(valid_chunk["UMIDADE_REL_MAX_NA_HORA_ANT_AUT"])

            valid_chunk["INMET_TMAX_C"] = pd.concat([dry_bulb, prior_hour_max], axis=1).max(axis=1, skipna=True)
            valid_chunk["INMET_TMIN_C"] = dry_bulb
            valid_chunk["INMET_TMEAN_C"] = dry_bulb
            valid_chunk["INMET_TMEAN_SUM"] = dry_bulb.fillna(0.0)
            valid_chunk["INMET_TMEAN_COUNT"] = dry_bulb.notna().astype(np.int16)
            valid_chunk["INMET_RH_MIN_PCT"] = pd.concat([prior_hour_rh_min, hourly_rh], axis=1).min(axis=1, skipna=True)
            valid_chunk["INMET_RH_MEAN_PCT"] = hourly_rh
            valid_chunk["INMET_RH_MAX_PCT"] = pd.concat([prior_hour_rh_max, hourly_rh], axis=1).max(axis=1, skipna=True)
            for variable in PRIMARY_CLIMATE_VARS:
                valid_chunk[f"TEM_INMET_{variable}"] = valid_chunk[f"INMET_{variable}"].notna().astype(np.int8)
            partials.append(
                valid_chunk.groupby(["COD_MUNICIPIO", "DATA"], as_index=False).agg(
                    INMET_TMAX_C=("INMET_TMAX_C", "max"),
                    INMET_TMIN_C=("INMET_TMIN_C", "min"),
                    INMET_TMEAN_SUM=("INMET_TMEAN_SUM", "sum"),
                    INMET_TMEAN_COUNT=("INMET_TMEAN_COUNT", "sum"),
                    INMET_RH_MIN_PCT=("INMET_RH_MIN_PCT", "min"),
                    INMET_RH_MEAN_SUM=("INMET_RH_MEAN_PCT", "sum"),
                    INMET_RH_MEAN_COUNT=("INMET_RH_MEAN_PCT", "count"),
                    INMET_RH_MAX_PCT=("INMET_RH_MAX_PCT", "max"),
                    TEM_INMET_TMAX_C=("TEM_INMET_TMAX_C", "max"),
                    TEM_INMET_TMIN_C=("TEM_INMET_TMIN_C", "max"),
                    TEM_INMET_TMEAN_C=("TEM_INMET_TMEAN_C", "max"),
                    TEM_INMET_RH_MIN_PCT=("TEM_INMET_RH_MIN_PCT", "max"),
                    TEM_INMET_RH_MEAN_PCT=("TEM_INMET_RH_MEAN_PCT", "max"),
                    TEM_INMET_RH_MAX_PCT=("TEM_INMET_RH_MAX_PCT", "max"),
                )
            )
        quarantine_chunk = chunk.merge(quarantined_station_map, on="CODIGO_WMO", how="inner")
        if not quarantine_chunk.empty:
            quarantine_partials.append(
                quarantine_chunk.groupby(["COD_MUNICIPIO", "DATA"], as_index=False).agg(
                    CLIMATE_UPSTREAM_QUARANTINE_REASON=("station_quarantine_reason", lambda values: "|".join(sorted({str(value) for value in values if str(value)}))),
                )
            )
    if not partials:
        aggregated = _empty_inmet_candidates()
    else:
        combined = pd.concat(partials, ignore_index=True)
        aggregated = (
            combined.groupby(["COD_MUNICIPIO", "DATA"], as_index=False)
            .agg(
                INMET_TMAX_C=("INMET_TMAX_C", "max"),
                INMET_TMIN_C=("INMET_TMIN_C", "min"),
                INMET_TMEAN_SUM=("INMET_TMEAN_SUM", "sum"),
                INMET_TMEAN_COUNT=("INMET_TMEAN_COUNT", "sum"),
                INMET_RH_MIN_PCT=("INMET_RH_MIN_PCT", "min"),
                INMET_RH_MEAN_SUM=("INMET_RH_MEAN_SUM", "sum"),
                INMET_RH_MEAN_COUNT=("INMET_RH_MEAN_COUNT", "sum"),
                INMET_RH_MAX_PCT=("INMET_RH_MAX_PCT", "max"),
                TEM_INMET_TMAX_C=("TEM_INMET_TMAX_C", "max"),
                TEM_INMET_TMIN_C=("TEM_INMET_TMIN_C", "max"),
                TEM_INMET_TMEAN_C=("TEM_INMET_TMEAN_C", "max"),
                TEM_INMET_RH_MIN_PCT=("TEM_INMET_RH_MIN_PCT", "max"),
                TEM_INMET_RH_MEAN_PCT=("TEM_INMET_RH_MEAN_PCT", "max"),
                TEM_INMET_RH_MAX_PCT=("TEM_INMET_RH_MAX_PCT", "max"),
            )
            .rename(columns={"DATA": "DATE"})
            .sort_values(["DATE", "COD_MUNICIPIO"])
            .reset_index(drop=True)
        )
        aggregated["INMET_RH_MEAN_PCT"] = np.where(
            aggregated["INMET_RH_MEAN_COUNT"] > 0,
            aggregated["INMET_RH_MEAN_SUM"] / aggregated["INMET_RH_MEAN_COUNT"],
            np.nan,
        )
        aggregated["INMET_TMEAN_C"] = np.where(
            aggregated["INMET_TMEAN_COUNT"] > 0,
            aggregated["INMET_TMEAN_SUM"] / aggregated["INMET_TMEAN_COUNT"],
            np.nan,
        )
        aggregated = aggregated.drop(columns=["INMET_TMEAN_SUM", "INMET_TMEAN_COUNT", "INMET_RH_MEAN_SUM", "INMET_RH_MEAN_COUNT"])
    if quarantine_partials:
        quarantine_rows = (
            pd.concat(quarantine_partials, ignore_index=True)
            .groupby(["COD_MUNICIPIO", "DATA"], as_index=False)
            .agg(
                CLIMATE_UPSTREAM_QUARANTINE_REASON=(
                    "CLIMATE_UPSTREAM_QUARANTINE_REASON",
                    lambda values: "|".join(sorted({str(value) for value in values if str(value)})),
                )
            )
            .rename(columns={"DATA": "DATE"})
            .sort_values(["DATE", "COD_MUNICIPIO"])
            .reset_index(drop=True)
        )
        quarantine_rows["CLIMATE_UPSTREAM_QUARANTINE_FLAG"] = np.int8(1)
    else:
        quarantine_rows = pd.DataFrame(
            columns=["COD_MUNICIPIO", "DATE", "CLIMATE_UPSTREAM_QUARANTINE_REASON", "CLIMATE_UPSTREAM_QUARANTINE_FLAG"]
        )
    metadata["mapped_municipality_count"] = int(aggregated["COD_MUNICIPIO"].nunique()) if not aggregated.empty else 0
    metadata["quarantined_municipality_day_count"] = int(len(quarantine_rows))
    return aggregated, metadata, quarantine_rows, station_qc


def _empty_era5_candidates() -> pd.DataFrame:
    columns = [
        "COD_MUNICIPIO",
        "DATE",
        "ERA5_TMAX_C",
        "ERA5_TMIN_C",
        "ERA5_TMEAN_C",
        "ERA5_RH_MIN_PCT",
        "ERA5_RH_MEAN_PCT",
        "ERA5_RH_MAX_PCT",
        "TEM_ERA5_TMAX_C",
        "TEM_ERA5_TMIN_C",
        "TEM_ERA5_TMEAN_C",
        "TEM_ERA5_RH_MIN_PCT",
        "TEM_ERA5_RH_MEAN_PCT",
        "TEM_ERA5_RH_MAX_PCT",
    ]
    return pd.DataFrame(columns=columns)


def load_era5_daily_candidates(root: Path, year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    bronze_path = bronze_output_path(root, "era5_land", year)
    metadata = {"rebuilt_from_raw": False, "source_path": str(bronze_path.relative_to(root))}
    try:
        dataframe = pd.read_csv(
            bronze_path,
            compression="gzip",
            sep=";",
            dtype=str,
            usecols=[
                "COD_MUNICIPIO",
                "DATE",
                "TMAX_C",
                "TMIN_C",
                "TMEAN_C",
                "RH_MIN_PCT",
                "RH_MEAN_PCT",
                "RH_MAX_PCT",
                ERA5_NO_USABLE_DATA_FIELD,
            ],
            keep_default_na=False,
            low_memory=False,
        ).fillna("")
    except (FileNotFoundError, ValueError, OSError, pd.errors.EmptyDataError) as exc:
        raw_paths = [path for path in raw_paths_for_dataset(root, "era5_land", year) if path.is_file()]
        if not raw_paths:
            raise FileNotFoundError(f"ERA5 raw components missing for {year}") from exc
        dataframe = build_bronze_annual_rows(root, year, raw_paths)
        metadata = {
            "rebuilt_from_raw": True,
            "source_path": "|".join(str(path.relative_to(root)) for path in raw_paths),
            "rebuild_reason": str(exc),
        }
    if dataframe.empty:
        return _empty_era5_candidates(), metadata
    result = pd.DataFrame(
        {
            "COD_MUNICIPIO": dataframe["COD_MUNICIPIO"].astype(str).str.zfill(7),
            "DATE": dataframe["DATE"].astype(str),
            "ERA5_TMAX_C": pd.to_numeric(dataframe["TMAX_C"], errors="coerce"),
            "ERA5_TMIN_C": pd.to_numeric(dataframe["TMIN_C"], errors="coerce"),
            "ERA5_TMEAN_C": pd.to_numeric(dataframe["TMEAN_C"], errors="coerce"),
            "ERA5_RH_MIN_PCT": pd.to_numeric(dataframe["RH_MIN_PCT"], errors="coerce"),
            "ERA5_RH_MEAN_PCT": pd.to_numeric(dataframe["RH_MEAN_PCT"], errors="coerce"),
            "ERA5_RH_MAX_PCT": pd.to_numeric(dataframe["RH_MAX_PCT"], errors="coerce"),
        }
    )
    for variable in ("RH_MIN_PCT", "RH_MEAN_PCT", "RH_MAX_PCT"):
        result[f"ERA5_{variable}"] = result[f"ERA5_{variable}"].where(
            result[f"ERA5_{variable}"].between(0.0, 100.0, inclusive="both")
        )
    for variable in PRIMARY_CLIMATE_VARS:
        result[f"TEM_ERA5_{variable}"] = result[f"ERA5_{variable}"].notna().astype(np.int8)
    return result.sort_values(["DATE", "COD_MUNICIPIO"]).reset_index(drop=True), metadata


def _validate_unique_keys(label: str, dataframe: pd.DataFrame) -> None:
    duplicate_count = int(dataframe.duplicated(list(CLIMATE_KEY_FIELDS)).sum())
    if duplicate_count:
        raise ValueError(f"{label} has duplicated municipality-day keys: {duplicate_count}")


def summarize_row_level_climate_source(dataframe: pd.DataFrame) -> pd.Series:
    source_frame = dataframe[[f"FONTE_{variable}_FINAL" for variable in PRIMARY_CLIMATE_VARS]]
    has_inmet = source_frame.eq("INMET").any(axis=1)
    has_era5 = source_frame.eq("ERA5").any(axis=1)
    has_imputation = source_frame.eq("IMPUTACAO").any(axis=1)
    categories = np.full(len(dataframe), "", dtype=object)
    for source_set, summary in SOURCE_SET_TO_SUMMARY.items():
        expected_inmet = "INMET" in source_set
        expected_era5 = "ERA5" in source_set
        expected_imputation = "IMPUTACAO" in source_set
        mask = (
            (has_inmet == expected_inmet)
            & (has_era5 == expected_era5)
            & (has_imputation == expected_imputation)
        )
        categories[mask.to_numpy()] = summary
    if (categories == "").any():
        raise ValueError("Unable to summarize row-level climate provenance")
    return pd.Series(categories, index=dataframe.index, dtype="object")


def _row_level_source_summary_matches(dataframe: pd.DataFrame) -> bool:
    source_frame = dataframe[[f"FONTE_{variable}_FINAL" for variable in PRIMARY_CLIMATE_VARS]]
    expected_code = (
        source_frame.eq("INMET").any(axis=1).to_numpy(dtype=np.uint8)
        + (2 * source_frame.eq("ERA5").any(axis=1).to_numpy(dtype=np.uint8))
        + (4 * source_frame.eq("IMPUTACAO").any(axis=1).to_numpy(dtype=np.uint8))
    )
    observed_code = (
        dataframe["FONTE_CLIMA_FINAL"]
        .map(
            {
                "INMET_FULL": 1,
                "ERA5_FULL": 2,
                "IMPUTED_FULL": 4,
                "MIXED_INMET_ERA5": 3,
                "MIXED_INMET_IMPUTED": 5,
                "MIXED_ERA5_IMPUTED": 6,
                "MIXED_INMET_ERA5_IMPUTED": 7,
            }
        )
        .fillna(0)
        .to_numpy(dtype=np.uint8)
    )
    return bool(np.array_equal(expected_code, observed_code))


def _ensure_correction_provenance_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    defaults: dict[str, Any] = {
        "CLIMATE_UPSTREAM_QUARANTINE_FLAG": np.int8(0),
        "CLIMATE_UPSTREAM_QUARANTINE_REASON": "",
        "CLIMATE_RECORD_ANOMALY_FLAG": np.int8(0),
        "CLIMATE_RECORD_ANOMALY_REASON": "",
        "CLIMATE_CORRECTION_PATH": "none",
    }
    for column, default_value in defaults.items():
        if column not in dataframe.columns:
            dataframe[column] = default_value
    dataframe["CLIMATE_UPSTREAM_QUARANTINE_FLAG"] = (
        pd.to_numeric(dataframe["CLIMATE_UPSTREAM_QUARANTINE_FLAG"], errors="coerce").fillna(0).astype(np.int8)
    )
    dataframe["CLIMATE_RECORD_ANOMALY_FLAG"] = (
        pd.to_numeric(dataframe["CLIMATE_RECORD_ANOMALY_FLAG"], errors="coerce").fillna(0).astype(np.int8)
    )
    dataframe["CLIMATE_UPSTREAM_QUARANTINE_REASON"] = dataframe["CLIMATE_UPSTREAM_QUARANTINE_REASON"].fillna("").astype("object")
    dataframe["CLIMATE_RECORD_ANOMALY_REASON"] = dataframe["CLIMATE_RECORD_ANOMALY_REASON"].fillna("").astype("object")
    dataframe["CLIMATE_CORRECTION_PATH"] = dataframe["CLIMATE_CORRECTION_PATH"].fillna("none").astype("object")
    return dataframe


def _compute_same_day_neighbor_median(
    target_frame: pd.DataFrame,
    source_frame: pd.DataFrame,
    neighbor_graph: pd.DataFrame,
    value_column: str,
    min_neighbors: int = NEIGHBOR_GRAPH_MIN_REQUIRED,
) -> pd.DataFrame:
    if target_frame.empty:
        return pd.DataFrame(columns=["COD_MUNICIPIO", "DATE", "neighbor_nonmissing_count", "neighbor_same_day_median"])
    targets = target_frame[["COD_MUNICIPIO", "DATE"]].drop_duplicates().copy()
    sources = (
        source_frame[["COD_MUNICIPIO", "DATE", value_column]]
        .dropna(subset=[value_column])
        .rename(columns={"COD_MUNICIPIO": "NEIGHBOR_COD_MUNICIPIO", value_column: "NEIGHBOR_VALUE"})
    )
    joined = (
        targets.merge(neighbor_graph, on="COD_MUNICIPIO", how="left")
        .merge(sources, on=["NEIGHBOR_COD_MUNICIPIO", "DATE"], how="left")
    )
    summary = (
        joined.groupby(["COD_MUNICIPIO", "DATE"], as_index=False)
        .agg(
            neighbor_nonmissing_count=("NEIGHBOR_VALUE", lambda values: int(pd.Series(values).notna().sum())),
            neighbor_same_day_median=("NEIGHBOR_VALUE", "median"),
        )
        .sort_values(["DATE", "COD_MUNICIPIO"], kind="mergesort")
        .reset_index(drop=True)
    )
    summary.loc[summary["neighbor_nonmissing_count"] < int(min_neighbors), "neighbor_same_day_median"] = np.nan
    return summary


def _flag_record_level_climate_anomalies(
    dataframe: pd.DataFrame,
    neighbor_graph: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = _ensure_correction_provenance_columns(dataframe.copy())
    inmet_selected_mask = dataframe["PRELIM_FONTE_TMAX_C"].eq("INMET")
    hard_bound_mask = inmet_selected_mask & (
        dataframe["PRELIM_TMAX_C"].lt(ANOMALY_TMAX_MIN_C) | dataframe["PRELIM_TMAX_C"].gt(ANOMALY_TMAX_MAX_C)
    )
    source_candidates = dataframe.loc[
        dataframe["PRELIM_TMAX_C"].notna()
        & dataframe["CLIMATE_UPSTREAM_QUARANTINE_FLAG"].eq(0)
        & ~hard_bound_mask,
        ["COD_MUNICIPIO", "DATE", "PRELIM_TMAX_C"],
    ].copy()
    target_candidates = dataframe.loc[inmet_selected_mask, ["COD_MUNICIPIO", "DATE"]].copy()
    neighbor_summary = _compute_same_day_neighbor_median(
        target_candidates,
        source_candidates,
        neighbor_graph,
        "PRELIM_TMAX_C",
    )
    dataframe = dataframe.merge(neighbor_summary, on=["COD_MUNICIPIO", "DATE"], how="left")
    era5_divergence_mask = (
        inmet_selected_mask
        & ~hard_bound_mask
        & dataframe["PRELIM_TMAX_C"].notna()
        & dataframe["ERA5_TMAX_C"].notna()
        & dataframe["neighbor_same_day_median"].notna()
        & dataframe["PRELIM_TMAX_C"].sub(dataframe["ERA5_TMAX_C"]).abs().gt(ANOMALY_ERA5_DIVERGENCE_C)
        & dataframe["PRELIM_TMAX_C"].sub(dataframe["neighbor_same_day_median"]).abs().gt(ANOMALY_NEIGHBOR_DIVERGENCE_C)
    )
    anomaly_mask = hard_bound_mask | era5_divergence_mask
    dataframe.loc[hard_bound_mask, "CLIMATE_RECORD_ANOMALY_REASON"] = "hard_physical_bound_tmax"
    dataframe.loc[era5_divergence_mask, "CLIMATE_RECORD_ANOMALY_REASON"] = "cross_source_and_neighbor_divergence_tmax"
    dataframe.loc[anomaly_mask, "CLIMATE_RECORD_ANOMALY_FLAG"] = np.int8(1)
    if bool(anomaly_mask.any()):
        for variable in PRIMARY_CLIMATE_VARS:
            dataframe.loc[anomaly_mask, f"PRELIM_{variable}"] = np.nan
            dataframe.loc[anomaly_mask, f"PRELIM_FONTE_{variable}"] = ""
    dataframe.loc[
        dataframe["CLIMATE_UPSTREAM_QUARANTINE_FLAG"].eq(1) & dataframe["CLIMATE_RECORD_ANOMALY_FLAG"].eq(0),
        "CLIMATE_CORRECTION_PATH",
    ] = "upstream_station_quarantine_to_era5"
    return dataframe


def select_final_climate_sources(root: Path, year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    lookup = _climate_lookup_frame(root)
    calendar_frame = _dense_calendar_frame(year)
    dense_panel = (
        lookup.assign(_join=1)
        .merge(calendar_frame.assign(_join=1), on="_join", how="inner")
        .drop(columns="_join")
        .sort_values(["DATE", "COD_MUNICIPIO"])
        .reset_index(drop=True)
    )
    inmet_candidates, inmet_metadata, quarantine_rows, station_qc = load_inmet_daily_candidates(root, year)
    era5_candidates, era5_metadata = load_era5_daily_candidates(root, year)
    if not inmet_candidates.empty:
        _validate_unique_keys("INMET daily candidates", inmet_candidates)
    if not era5_candidates.empty:
        _validate_unique_keys("ERA5 daily candidates", era5_candidates)
    panel = dense_panel.merge(inmet_candidates, on=list(CLIMATE_KEY_FIELDS), how="left")
    panel = panel.merge(era5_candidates, on=list(CLIMATE_KEY_FIELDS), how="left")
    panel = panel.merge(quarantine_rows, on=list(CLIMATE_KEY_FIELDS), how="left")
    if len(panel) != len(dense_panel):
        raise ValueError(f"Climate merge introduced row multiplication for {year}")
    for variable in PRIMARY_CLIMATE_VARS:
        panel[f"TEM_INMET_{variable}"] = panel[f"TEM_INMET_{variable}"].fillna(0).astype(np.int8)
        panel[f"TEM_ERA5_{variable}"] = panel[f"TEM_ERA5_{variable}"].fillna(0).astype(np.int8)
        panel[f"PRELIM_{variable}"] = panel[f"INMET_{variable}"].combine_first(panel[f"ERA5_{variable}"])
        panel[f"PRELIM_FONTE_{variable}"] = np.select(
            [
                panel[f"TEM_INMET_{variable}"].eq(1),
                panel[f"TEM_ERA5_{variable}"].eq(1),
            ],
            ["INMET", "ERA5"],
            default="",
        )
    panel["TEM_INMET"] = panel[[f"TEM_INMET_{variable}" for variable in PRIMARY_CLIMATE_VARS]].max(axis=1).astype(np.int8)
    panel["TEM_ERA5"] = panel[[f"TEM_ERA5_{variable}" for variable in PRIMARY_CLIMATE_VARS]].max(axis=1).astype(np.int8)
    panel = _ensure_correction_provenance_columns(panel)
    panel.loc[
        panel["CLIMATE_UPSTREAM_QUARANTINE_FLAG"].eq(1),
        "CLIMATE_CORRECTION_PATH",
    ] = "upstream_station_quarantine_to_era5"
    return panel, {
        "expected_rows": int(len(dense_panel)),
        "observed_rows": int(len(panel)),
        "municipality_count": int(lookup["COD_MUNICIPIO"].nunique()),
        "inmet_metadata": inmet_metadata,
        "era5_metadata": era5_metadata,
        "station_qc_summary": {
            "row_count": int(len(station_qc)),
            "quarantine_count": int(station_qc["station_quarantined_flag"].sum()),
        },
    }


def load_or_build_preselected_climate_year(
    root: Path,
    runtime_root: Path,
    year: int,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    runtime_path = runtime_preselected_file(runtime_root, year)
    if runtime_path.exists() and not force:
        try:
            dataframe = _read_parquet_verified(runtime_path)
            _validate_preselected_runtime_frame(root, year, dataframe)
            return dataframe, {
                "preselected_runtime_path": str(runtime_path),
                "reused_preselected_runtime": True,
                "rebuilt_from_components": False,
            }
        except Exception as exc:
            rebuild_reason = str(exc)
        else:  # pragma: no cover - kept for clarity
            rebuild_reason = ""
    else:
        rebuild_reason = ""
    dataframe, build_metadata = select_final_climate_sources(root, year)
    readback = _write_runtime_parquet(dataframe, runtime_path)
    return readback, {
        **build_metadata,
        "preselected_runtime_path": str(runtime_path),
        "reused_preselected_runtime": False,
        "rebuilt_from_components": True,
        "rebuild_reason": rebuild_reason,
    }


def _build_imputation_tables(preselected_frames: dict[int, pd.DataFrame]) -> dict[str, dict[str, pd.DataFrame]]:
    compact = pd.concat(
        [
            frame[["COD_MUNICIPIO", "UF", "YEAR", "MONTH", "DAY_OF_YEAR"] + [f"PRELIM_{variable}" for variable in PRIMARY_CLIMATE_VARS]]
            for frame in preselected_frames.values()
        ],
        ignore_index=True,
    )
    tables: dict[str, dict[str, pd.DataFrame]] = {}
    for variable in PRIMARY_CLIMATE_VARS:
        value_column = f"PRELIM_{variable}"
        variable_frame = compact[["COD_MUNICIPIO", "UF", "YEAR", "MONTH", "DAY_OF_YEAR", value_column]].dropna(
            subset=[value_column]
        )
        variable_tables: dict[str, pd.DataFrame] = {}
        for method_name, keys in IMPUTATION_HIERARCHY:
            variable_tables[method_name] = (
                variable_frame.groupby(keys, as_index=False)[value_column]
                .median()
                .rename(columns={value_column: f"IMPUTE_{variable}"})
            )
        tables[variable] = variable_tables
    return tables


def _same_day_neighbor_fill_candidates(
    dataframe: pd.DataFrame,
    variable: str,
    neighbor_graph: pd.DataFrame,
) -> pd.DataFrame:
    target_rows = dataframe.loc[
        dataframe["CLIMATE_RECORD_ANOMALY_FLAG"].eq(1) & dataframe[f"PRELIM_{variable}"].isna(),
        ["COD_MUNICIPIO", "DATE"],
    ].copy()
    if target_rows.empty:
        return pd.DataFrame(columns=["COD_MUNICIPIO", "DATE", "neighbor_nonmissing_count", "neighbor_same_day_median"])
    source_rows = dataframe.loc[
        dataframe["CLIMATE_RECORD_ANOMALY_FLAG"].eq(0),
        ["COD_MUNICIPIO", "DATE", f"PRELIM_{variable}"],
    ].copy()
    return _compute_same_day_neighbor_median(target_rows, source_rows, neighbor_graph, f"PRELIM_{variable}")


def _apply_single_variable_imputation(
    dataframe: pd.DataFrame,
    variable: str,
    tables: dict[str, pd.DataFrame],
    neighbor_graph: pd.DataFrame | None = None,
) -> pd.DataFrame:
    final_values = dataframe[f"PRELIM_{variable}"].astype(float).copy()
    final_sources = dataframe[f"PRELIM_FONTE_{variable}"].astype("object").copy()
    imputed_mask = pd.Series(np.zeros(len(dataframe), dtype=np.int8), index=dataframe.index)
    methods = pd.Series("", index=dataframe.index, dtype="object")
    imputed_values = pd.Series(np.nan, index=dataframe.index, dtype=float)
    if neighbor_graph is not None:
        neighbor_candidates = _same_day_neighbor_fill_candidates(dataframe, variable, neighbor_graph)
        if not neighbor_candidates.empty:
            neighbor_candidates = neighbor_candidates.set_index(["COD_MUNICIPIO", "DATE"])
            row_index = pd.MultiIndex.from_frame(dataframe[["COD_MUNICIPIO", "DATE"]])
            neighbor_reindexed = neighbor_candidates.reindex(row_index)
            neighbor_values = pd.Series(
                neighbor_reindexed["neighbor_same_day_median"].to_numpy(),
                index=dataframe.index,
                dtype=float,
            )
            neighbor_counts = pd.Series(
                neighbor_reindexed["neighbor_nonmissing_count"].to_numpy(),
                index=dataframe.index,
                dtype=float,
            )
            fill_mask = (
                final_values.isna()
                & dataframe["CLIMATE_RECORD_ANOMALY_FLAG"].eq(1)
                & neighbor_values.notna()
                & neighbor_counts.fillna(0).ge(NEIGHBOR_GRAPH_MIN_REQUIRED)
            )
            if bool(fill_mask.any()):
                target_index = dataframe.index[fill_mask]
                final_values.loc[target_index] = neighbor_values.loc[target_index].to_numpy(dtype=float)
                final_sources.loc[target_index] = "IMPUTACAO"
                imputed_mask.loc[target_index] = np.int8(1)
                methods.loc[target_index] = NEIGHBOR_IMPUTATION_METHOD
                imputed_values.loc[target_index] = neighbor_values.loc[target_index].to_numpy(dtype=float)
    for method_name, keys in IMPUTATION_HIERARCHY:
        missing_mask = final_values.isna()
        if not bool(missing_mask.any()):
            break
        table = tables[method_name]
        joined = dataframe.loc[missing_mask, keys].merge(table, on=keys, how="left")
        candidate = pd.Series(joined[f"IMPUTE_{variable}"].to_numpy(), index=dataframe.index[missing_mask])
        fill_mask = candidate.notna()
        if not bool(fill_mask.any()):
            continue
        target_index = candidate.index[fill_mask]
        final_values.loc[target_index] = candidate.loc[target_index]
        final_sources.loc[target_index] = "IMPUTACAO"
        imputed_mask.loc[target_index] = np.int8(1)
        methods.loc[target_index] = method_name
        imputed_values.loc[target_index] = candidate.loc[target_index]
    if bool(final_values.isna().any()):
        raise ValueError(f"Deterministic imputation left residual missingness in {variable}")
    dataframe[variable] = final_values.astype(float)
    dataframe[f"FONTE_{variable}_FINAL"] = final_sources
    dataframe[f"IMPUTADO_{variable}"] = imputed_mask.astype(np.int8)
    dataframe[f"IMPUTACAO_METODO_{variable}"] = methods
    dataframe[f"IMPUTED_VALUE_{variable}"] = imputed_values.astype(float)
    return dataframe


def impute_missing_climate_variables(
    preselected_frames: dict[int, pd.DataFrame],
    root: Path | None = None,
) -> tuple[dict[int, pd.DataFrame], dict[str, dict[str, pd.DataFrame]]]:
    working_frames: dict[int, pd.DataFrame] = {}
    neighbor_graph = _build_neighbor_graph(root, k=NEIGHBOR_GRAPH_K) if root is not None else None
    for year, frame in preselected_frames.items():
        year_frame = _ensure_correction_provenance_columns(frame.copy())
        if neighbor_graph is not None:
            year_frame = _flag_record_level_climate_anomalies(year_frame, neighbor_graph)
        working_frames[year] = year_frame
    tables = _build_imputation_tables(working_frames)
    completed: dict[int, pd.DataFrame] = {}
    for year, frame in working_frames.items():
        year_frame = frame.copy()
        for variable in PRIMARY_CLIMATE_VARS:
            year_frame = _apply_single_variable_imputation(year_frame, variable, tables[variable], neighbor_graph=neighbor_graph)
        year_frame["IMPUTADO_POR_FALLBACK"] = year_frame[
            [f"IMPUTADO_{variable}" for variable in PRIMARY_CLIMATE_VARS]
        ].max(axis=1).astype(np.int8)
        neighbor_imputation_mask = year_frame[
            [f"IMPUTACAO_METODO_{variable}" for variable in PRIMARY_CLIMATE_VARS]
        ].eq(NEIGHBOR_IMPUTATION_METHOD).any(axis=1)
        year_frame["CLIMATE_CORRECTION_PATH"] = np.where(
            year_frame["CLIMATE_RECORD_ANOMALY_FLAG"].eq(1) & neighbor_imputation_mask,
            "record_anomaly_neighbor_imputation",
            year_frame["CLIMATE_CORRECTION_PATH"],
        )
        year_frame["CLIMATE_CORRECTION_PATH"] = np.where(
            year_frame["CLIMATE_RECORD_ANOMALY_FLAG"].eq(1) & ~neighbor_imputation_mask,
            "record_anomaly_fallback_imputation",
            year_frame["CLIMATE_CORRECTION_PATH"],
        )
        year_frame["CLIMATE_CORRECTION_PATH"] = np.where(
            year_frame["CLIMATE_UPSTREAM_QUARANTINE_FLAG"].eq(1) & year_frame["CLIMATE_RECORD_ANOMALY_FLAG"].eq(0),
            "upstream_station_quarantine_to_era5",
            year_frame["CLIMATE_CORRECTION_PATH"],
        )
        completed[year] = year_frame
    return completed, tables


def derive_heat_index(dataframe: pd.DataFrame) -> pd.DataFrame:
    t_c = dataframe["TMAX_C"].astype(float).to_numpy()
    rh = dataframe["RH_MEAN_PCT"].astype(float).to_numpy()
    t_f = (t_c * 9.0 / 5.0) + 32.0
    hi_simple = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094))
    hi_precheck = (hi_simple + t_f) / 2.0
    heat_index_f = hi_precheck.copy()
    methods = np.full(len(dataframe), "nws_simple_pre80f", dtype=object)

    rothfusz_mask = hi_precheck >= 80.0
    if np.any(rothfusz_mask):
        tf = t_f[rothfusz_mask]
        rh_masked = rh[rothfusz_mask]
        hi_regression = (
            -42.379
            + (2.04901523 * tf)
            + (10.14333127 * rh_masked)
            - (0.22475541 * tf * rh_masked)
            - (0.00683783 * tf * tf)
            - (0.05481717 * rh_masked * rh_masked)
            + (0.00122874 * tf * tf * rh_masked)
            + (0.00085282 * tf * rh_masked * rh_masked)
            - (0.00000199 * tf * tf * rh_masked * rh_masked)
        )
        low_rh_mask = (rh_masked < 13.0) & (tf >= 80.0) & (tf <= 112.0)
        high_rh_mask = (rh_masked > 85.0) & (tf >= 80.0) & (tf <= 87.0) & ~low_rh_mask
        if np.any(low_rh_mask):
            adjustment = ((13.0 - rh_masked[low_rh_mask]) / 4.0) * np.sqrt(
                (17.0 - np.abs(tf[low_rh_mask] - 95.0)) / 17.0
            )
            hi_regression[low_rh_mask] = hi_regression[low_rh_mask] - adjustment
        if np.any(high_rh_mask):
            adjustment = ((rh_masked[high_rh_mask] - 85.0) / 10.0) * ((87.0 - tf[high_rh_mask]) / 5.0)
            hi_regression[high_rh_mask] = hi_regression[high_rh_mask] + adjustment
        heat_index_f[rothfusz_mask] = hi_regression
        methods[rothfusz_mask] = "nws_rothfusz"
        methods[np.where(rothfusz_mask)[0][low_rh_mask]] = "nws_rothfusz_low_rh_adjustment"
        methods[np.where(rothfusz_mask)[0][high_rh_mask]] = "nws_rothfusz_high_rh_adjustment"

    dataframe["HEAT_INDEX_C"] = (heat_index_f - 32.0) * 5.0 / 9.0
    dataframe["HEAT_INDEX_METHOD"] = pd.Series(methods, index=dataframe.index, dtype="object")
    dataframe["HEAT_INDEX_SOURCE_PAIR"] = (
        "DERIVADO_"
        + dataframe["FONTE_TMAX_C_FINAL"].astype(str)
        + "_"
        + dataframe["FONTE_RH_MEAN_PCT_FINAL"].astype(str)
    )
    dataframe["HEAT_INDEX_INPUTS_ZERO_MISSING"] = (
        dataframe["TMAX_C"].notna() & dataframe["RH_MEAN_PCT"].notna()
    ).astype(np.int8)
    if int(dataframe["HEAT_INDEX_INPUTS_ZERO_MISSING"].min()) != 1:
        raise ValueError("Heat index inputs remain missing after climate completion")
    return dataframe


def recompute_heatwave_len_p95(dataframe: pd.DataFrame) -> pd.Series:
    working = dataframe[["COD_MUNICIPIO", "DATE", "TMAX_C", "IMPUTADO_TMAX_C"]].copy()
    working["DATE_TS"] = pd.to_datetime(working["DATE"], errors="raise")
    working = working.sort_values(["COD_MUNICIPIO", "DATE_TS"]).reset_index()

    base_frame = working[working["IMPUTADO_TMAX_C"].eq(0)].copy()
    thresholds = base_frame.groupby("COD_MUNICIPIO")["TMAX_C"].quantile(0.95)
    fallback_thresholds = working.groupby("COD_MUNICIPIO")["TMAX_C"].quantile(0.95)
    thresholds = thresholds.combine_first(fallback_thresholds)

    working["P95_TMAX_C"] = working["COD_MUNICIPIO"].map(thresholds)
    hot_mask = working["TMAX_C"] >= working["P95_TMAX_C"]
    run_id = (
        working["COD_MUNICIPIO"].ne(working["COD_MUNICIPIO"].shift())
        | ~hot_mask
        | ~hot_mask.shift(fill_value=False)
    ).cumsum()
    streak_length = working.groupby(run_id).cumcount() + 1
    working["HEATWAVE_LEN_P95"] = np.where(hot_mask, streak_length, 0).astype(np.int16)
    return working.set_index("index")["HEATWAVE_LEN_P95"].reindex(dataframe.index).astype(np.int16)


def _prepare_output_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = _ensure_correction_provenance_columns(dataframe)
    dataframe["DATE"] = pd.to_datetime(dataframe["DATE"], errors="raise").dt.strftime("%Y-%m-%d")
    dataframe["FONTE_CLIMA_FINAL"] = summarize_row_level_climate_source(dataframe)
    dataframe = derive_heat_index(dataframe)
    for column in (
        "TEM_INMET",
        "TEM_ERA5",
        "IMPUTADO_POR_FALLBACK",
        "CLIMATE_UPSTREAM_QUARANTINE_FLAG",
        "CLIMATE_RECORD_ANOMALY_FLAG",
    ):
        dataframe[column] = dataframe[column].astype(np.int8)
    for variable in PRIMARY_CLIMATE_VARS:
        dataframe[f"IMPUTADO_{variable}"] = dataframe[f"IMPUTADO_{variable}"].astype(np.int8)
        dataframe[f"TEM_INMET_{variable}"] = dataframe[f"TEM_INMET_{variable}"].astype(np.int8)
        dataframe[f"TEM_ERA5_{variable}"] = dataframe[f"TEM_ERA5_{variable}"].astype(np.int8)
    invalid_paths = sorted(set(dataframe["CLIMATE_CORRECTION_PATH"].astype(str)) - CLIMATE_CORRECTION_PATHS)
    if invalid_paths:
        raise ValueError(f"Unexpected climate correction path labels: {invalid_paths}")
    if "HEATWAVE_LEN_P95" in dataframe.columns:
        dataframe["HEATWAVE_LEN_P95"] = dataframe["HEATWAVE_LEN_P95"].astype(np.int16)
    return dataframe


def _partition_summary(frame: pd.DataFrame, year: int) -> dict[str, Any]:
    total_rows = int(len(frame))
    share_variable_level: dict[str, dict[str, float]] = {}
    for variable in PRIMARY_CLIMATE_VARS:
        shares = frame[f"FONTE_{variable}_FINAL"].value_counts(normalize=True).to_dict()
        share_variable_level[variable] = {source: float(value) for source, value in shares.items()}
    row_source_share = {
        source: float(value)
        for source, value in frame["FONTE_CLIMA_FINAL"].value_counts(normalize=True).to_dict().items()
    }
    coverage = frame.groupby("COD_MUNICIPIO", as_index=False).size()
    expected_days = len(pd.date_range(date(year, 1, 1), date(year, 12, 31), freq="D"))
    return {
        "total_municipality_days": total_rows,
        "municipalities_with_full_daily_coverage": int(coverage["size"].eq(expected_days).sum()),
        "missing_zero": {variable: int(frame[variable].isna().sum()) == 0 for variable in PRIMARY_CLIMATE_VARS}
        | {"HEAT_INDEX_C": int(frame["HEAT_INDEX_C"].isna().sum()) == 0},
        "variable_level_source_share": share_variable_level,
        "row_level_source_share": row_source_share,
        "hot_spell_municipality_days": int(frame["HEATWAVE_LEN_P95"].gt(0).sum()),
        "max_heatwave_len_p95": int(frame["HEATWAVE_LEN_P95"].max()),
        "mean_tmax_c": float(frame["TMAX_C"].mean()),
        "sd_tmax_c": float(frame["TMAX_C"].std(ddof=0)),
        "mean_tmin_c": float(frame["TMIN_C"].mean()),
        "sd_tmin_c": float(frame["TMIN_C"].std(ddof=0)),
        "mean_tmean_c": float(frame["TMEAN_C"].mean()),
        "sd_tmean_c": float(frame["TMEAN_C"].std(ddof=0)),
        "mean_rh_mean_pct": float(frame["RH_MEAN_PCT"].mean()),
        "sd_rh_mean_pct": float(frame["RH_MEAN_PCT"].std(ddof=0)),
        "mean_heat_index_c": float(frame["HEAT_INDEX_C"].mean()),
        "sd_heat_index_c": float(frame["HEAT_INDEX_C"].std(ddof=0)),
    }


def validate_canonical_climate_silver(
    root: Path,
    frames_by_year: dict[int, pd.DataFrame],
    partition_files: dict[int, Path] | None = None,
    station_qc_by_year: dict[int, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    lookup = _climate_lookup_frame(root)
    expected_codes = set(lookup["COD_MUNICIPIO"].tolist())
    validation: dict[str, Any] = {"years": {}, "schema_consistent": True, "partition_write_integrity": True}
    combined = pd.concat(
        [
            frame[
                [
                    "COD_MUNICIPIO",
                    "DATE",
                    "TMAX_C",
                    "RH_MEAN_PCT",
                    "IMPUTADO_TMAX_C",
                    "HEATWAVE_LEN_P95",
                    "HEAT_INDEX_C",
                    "HEAT_INDEX_METHOD",
                    "FONTE_TMAX_C_FINAL",
                    "FONTE_RH_MEAN_PCT_FINAL",
                ]
            ].copy()
            for frame in frames_by_year.values()
        ],
        ignore_index=True,
    )
    recomputed_heatwave = recompute_heatwave_len_p95(combined)
    validation["heatwave_reproducible"] = bool((recomputed_heatwave.to_numpy() == combined["HEATWAVE_LEN_P95"].to_numpy()).all())
    recomputed_heat_index_frame = derive_heat_index(
        combined[["TMAX_C", "RH_MEAN_PCT", "FONTE_TMAX_C_FINAL", "FONTE_RH_MEAN_PCT_FINAL"]].copy()
    )
    validation["heat_index_reproducible"] = bool(
        np.allclose(
            recomputed_heat_index_frame["HEAT_INDEX_C"].to_numpy(),
            combined["HEAT_INDEX_C"].to_numpy(),
            atol=HEAT_INDEX_TOLERANCE,
            rtol=0.0,
        )
    ) and bool(
        recomputed_heat_index_frame["HEAT_INDEX_METHOD"].equals(combined["HEAT_INDEX_METHOD"])
    )

    for year, frame in frames_by_year.items():
        expected_dates = set(pd.date_range(date(year, 1, 1), date(year, 12, 31), freq="D").strftime("%Y-%m-%d").tolist())
        observed_dates = set(pd.to_datetime(frame["DATE"], errors="raise").dt.strftime("%Y-%m-%d").tolist())
        missing_counts = {variable: int(frame[variable].isna().sum()) for variable in PRIMARY_CLIMATE_VARS}
        missing_counts["HEAT_INDEX_C"] = int(frame["HEAT_INDEX_C"].isna().sum())
        key_duplicates = int(frame.duplicated(list(CLIMATE_KEY_FIELDS)).sum())
        municipality_coverage = set(frame["COD_MUNICIPIO"].tolist())
        row_source_ok = _row_level_source_summary_matches(frame)
        provenance_ok = True
        correction_paths_ok = set(frame["CLIMATE_CORRECTION_PATH"].astype(str)).issubset(CLIMATE_CORRECTION_PATHS)
        upstream_reason_ok = bool(
            frame.loc[frame["CLIMATE_UPSTREAM_QUARANTINE_FLAG"].eq(1), "CLIMATE_UPSTREAM_QUARANTINE_REASON"]
            .astype(str)
            .str.len()
            .gt(0)
            .all()
        )
        anomaly_reason_ok = bool(
            frame.loc[frame["CLIMATE_RECORD_ANOMALY_FLAG"].eq(1), "CLIMATE_RECORD_ANOMALY_REASON"]
            .astype(str)
            .str.len()
            .gt(0)
            .all()
        )
        for variable in PRIMARY_CLIMATE_VARS:
            inmet_mask = frame[f"FONTE_{variable}_FINAL"].eq("INMET")
            era5_mask = frame[f"FONTE_{variable}_FINAL"].eq("ERA5")
            imputed_mask = frame[f"FONTE_{variable}_FINAL"].eq("IMPUTACAO")
            if bool(inmet_mask.any()):
                provenance_ok = provenance_ok and bool(
                    np.allclose(
                        frame.loc[inmet_mask, variable].to_numpy(),
                        frame.loc[inmet_mask, f"INMET_{variable}"].to_numpy(),
                        atol=READBACK_TOLERANCE,
                        rtol=0.0,
                    )
                )
            if bool(era5_mask.any()):
                provenance_ok = provenance_ok and bool(
                    np.allclose(
                        frame.loc[era5_mask, variable].to_numpy(),
                        frame.loc[era5_mask, f"ERA5_{variable}"].to_numpy(),
                        atol=READBACK_TOLERANCE,
                        rtol=0.0,
                    )
                )
            if bool(imputed_mask.any()):
                provenance_ok = provenance_ok and bool(
                    np.allclose(
                        frame.loc[imputed_mask, variable].to_numpy(),
                        frame.loc[imputed_mask, f"IMPUTED_VALUE_{variable}"].to_numpy(),
                        atol=READBACK_TOLERANCE,
                        rtol=0.0,
                    )
                )
        partition_ok = True
        if partition_files is not None:
            partition_path = partition_files[year]
            partition_ok = partition_path.exists()
            if partition_ok:
                persisted = _read_parquet_verified(partition_path)
                partition_ok = list(persisted.columns) == SILVER_OUTPUT_FIELDS and len(persisted) == len(frame)
        station_qc = station_qc_by_year.get(year) if station_qc_by_year is not None else None
        neighbor_imputation_row_count = int(
            frame[[f"IMPUTACAO_METODO_{variable}" for variable in PRIMARY_CLIMATE_VARS]]
            .eq(NEIGHBOR_IMPUTATION_METHOD)
            .any(axis=1)
            .sum()
        )
        fallback_imputation_after_anomaly_row_count = int(
            (
                frame["CLIMATE_RECORD_ANOMALY_FLAG"].eq(1)
                & ~frame[[f"IMPUTACAO_METODO_{variable}" for variable in PRIMARY_CLIMATE_VARS]]
                .eq(NEIGHBOR_IMPUTATION_METHOD)
                .any(axis=1)
            ).sum()
        )
        validation["years"][year] = {
            "expected_rows": len(expected_codes) * len(expected_dates),
            "observed_rows": int(len(frame)),
            "dense_panel_ok": int(len(frame)) == len(expected_codes) * len(expected_dates),
            "calendar_coverage_ok": observed_dates == expected_dates,
            "municipality_coverage_ok": municipality_coverage == expected_codes,
            "key_uniqueness_ok": key_duplicates == 0,
            "provenance_ok": provenance_ok,
            "row_level_provenance_ok": row_source_ok,
            "correction_path_labels_ok": correction_paths_ok,
            "upstream_quarantine_reason_ok": upstream_reason_ok,
            "record_anomaly_reason_ok": anomaly_reason_ok,
            "missing_counts": missing_counts,
            "zero_missing_ok": all(value == 0 for value in missing_counts.values()),
            "partition_integrity_ok": partition_ok,
            "upstream_station_quarantine_count": int(station_qc["station_quarantined_flag"].sum()) if station_qc is not None else 0,
            "upstream_station_quarantine_wmo_codes": (
                sorted(station_qc.loc[station_qc["station_quarantined_flag"].eq(1), "CODIGO_WMO"].astype(str).unique().tolist())
                if station_qc is not None
                else []
            ),
            "record_anomaly_row_count": int(frame["CLIMATE_RECORD_ANOMALY_FLAG"].sum()),
            "neighbor_imputation_row_count": neighbor_imputation_row_count,
            "fallback_imputation_after_anomaly_row_count": fallback_imputation_after_anomaly_row_count,
            "summary": _partition_summary(frame, year),
        }
        validation["partition_write_integrity"] = validation["partition_write_integrity"] and partition_ok

    if partition_files is not None:
        schemas = [pq.read_schema(path).to_string() for path in partition_files.values()]
        validation["schema_consistent"] = len(set(schemas)) == 1 if schemas else True
    validation["all_checks_passed"] = (
        validation["heatwave_reproducible"]
        and validation["heat_index_reproducible"]
        and validation["schema_consistent"]
        and validation["partition_write_integrity"]
        and all(
            item["dense_panel_ok"]
            and item["calendar_coverage_ok"]
            and item["municipality_coverage_ok"]
            and item["key_uniqueness_ok"]
            and item["provenance_ok"]
            and item["row_level_provenance_ok"]
            and item["correction_path_labels_ok"]
            and item["upstream_quarantine_reason_ok"]
            and item["record_anomaly_reason_ok"]
            and item["zero_missing_ok"]
            for item in validation["years"].values()
        )
    )
    return validation


def write_canonical_climate_silver(
    root: Path,
    runtime_root: Path,
    year: int,
    dataframe: pd.DataFrame,
) -> Path:
    output_path = silver_partition_file(root, year)
    staging_path = runtime_root / output_path.relative_to(root)
    write_frame = dataframe[SILVER_OUTPUT_FIELDS].copy()
    _write_runtime_parquet(write_frame, staging_path)
    _sync_partition_verified(staging_path, output_path)
    staging_path.unlink(missing_ok=True)
    if staging_path.parent.exists():
        shutil.rmtree(staging_path.parent)
    return output_path


def _write_validation_reports(
    root: Path,
    validation: dict[str, Any],
    manifest_path: str,
    frames_by_year: dict[int, pd.DataFrame],
    station_qc_by_year: dict[int, pd.DataFrame],
) -> None:
    reports_dir = root / "artifacts" / "validation_reports" / "silver" / "climate"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for year, payload in validation["years"].items():
        station_qc_path = reports_dir / f"{year}_station_qc.csv"
        station_qc_by_year[year].to_csv(station_qc_path, index=False)
        anomaly_rows_path = reports_dir / f"{year}_anomaly_rows.csv"
        frames_by_year[year].loc[
            frames_by_year[year]["CLIMATE_RECORD_ANOMALY_FLAG"].eq(1),
            [
                "COD_MUNICIPIO",
                "NM_MUNICIPIO",
                "UF",
                "DATE",
                "CLIMATE_RECORD_ANOMALY_REASON",
                "CLIMATE_CORRECTION_PATH",
                "FONTE_CLIMA_FINAL",
                "FONTE_TMAX_C_FINAL",
                "FONTE_RH_MEAN_PCT_FINAL",
                "TMAX_C",
                "RH_MEAN_PCT",
                "HEAT_INDEX_C",
            ],
        ].to_csv(anomaly_rows_path, index=False)
        report_path = reports_dir / f"{year}.json"
        report_payload = {
            "dataset": "climate",
            "year": year,
            "generated_at": utc_now(),
            "manifest_path": manifest_path,
            "station_qc_path": str(station_qc_path.relative_to(root)),
            "anomaly_rows_path": str(anomaly_rows_path.relative_to(root)),
            **payload,
        }
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_path = root / "artifacts" / "status" / "silver_climate_last_run_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Silver Climate Canonical Summary",
        "",
        f"- Generated at: {utc_now()}",
        f"- Manifest: `{manifest_path}`",
        f"- All checks passed: `{validation['all_checks_passed']}`",
        f"- Heat index reproducible: `{validation['heat_index_reproducible']}`",
        f"- Heatwave reproducible: `{validation['heatwave_reproducible']}`",
        f"- Partition schema consistent: `{validation['schema_consistent']}`",
        "",
    ]
    for year in sorted(validation["years"]):
        payload = validation["years"][year]
        summary = payload["summary"]
        lines.extend(
            [
                f"## {year}",
                "",
                f"- total municipality-days: {summary['total_municipality_days']}",
                f"- municipalities with full daily coverage: {summary['municipalities_with_full_daily_coverage']}",
                f"- missingness zero for primary variables: {summary['missing_zero']}",
                f"- share INMET / ERA5 / imputed at variable level: {summary['variable_level_source_share']}",
                f"- share of FONTE_CLIMA_FINAL categories: {summary['row_level_source_share']}",
                f"- hot-spell municipality-days: {summary['hot_spell_municipality_days']}",
                f"- max HEATWAVE_LEN_P95: {summary['max_heatwave_len_p95']}",
                f"- mean/SD TMAX_C: {summary['mean_tmax_c']:.4f} / {summary['sd_tmax_c']:.4f}",
                f"- mean/SD TMIN_C: {summary['mean_tmin_c']:.4f} / {summary['sd_tmin_c']:.4f}",
                f"- mean/SD TMEAN_C: {summary['mean_tmean_c']:.4f} / {summary['sd_tmean_c']:.4f}",
                f"- mean/SD RH_MEAN_PCT: {summary['mean_rh_mean_pct']:.4f} / {summary['sd_rh_mean_pct']:.4f}",
                f"- mean/SD HEAT_INDEX_C: {summary['mean_heat_index_c']:.4f} / {summary['sd_heat_index_c']:.4f}",
                f"- upstream station quarantine count: {payload['upstream_station_quarantine_count']}",
                f"- record anomaly row count: {payload['record_anomaly_row_count']}",
                f"- neighbor imputation row count: {payload['neighbor_imputation_row_count']}",
                f"- fallback-after-anomaly row count: {payload['fallback_imputation_after_anomaly_row_count']}",
                "",
            ]
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def promote_climate_silver_range(
    root: Path,
    runtime_root: Path,
    start_year: int,
    end_year: int,
    log_path: Path,
    command: str,
    force: bool = False,
) -> dict[str, Any]:
    years = list(range(start_year, end_year + 1))
    preselected_frames: dict[int, pd.DataFrame] = {}
    preselected_metadata: dict[int, dict[str, Any]] = {}
    station_qc_by_year = {year: _build_station_qc_table(root, year) for year in years}
    for year in years:
        frame, metadata = load_or_build_preselected_climate_year(root, runtime_root, year, force=force)
        preselected_frames[year] = frame
        preselected_metadata[year] = metadata

    completed_frames, imputation_tables = impute_missing_climate_variables(preselected_frames, root=root)
    preselected_frames.clear()
    for year in years:
        completed_frames[year] = _prepare_output_frame(completed_frames[year])

    heatwave_frame = pd.concat(
        [
            completed_frames[year][["COD_MUNICIPIO", "DATE", "TMAX_C", "IMPUTADO_TMAX_C"]].copy()
            for year in years
        ],
        ignore_index=True,
    )
    heatwave_frame["HEATWAVE_LEN_P95"] = recompute_heatwave_len_p95(heatwave_frame)
    split_frames: dict[int, pd.DataFrame] = {}
    cursor = 0
    for year in years:
        frame = completed_frames[year]
        split_frames[year] = frame.copy()
        split_frames[year]["HEATWAVE_LEN_P95"] = heatwave_frame.iloc[cursor : cursor + len(frame)]["HEATWAVE_LEN_P95"].to_numpy()
        cursor += len(frame)
    completed_frames.clear()

    prewrite_validation = validate_canonical_climate_silver(root, split_frames, station_qc_by_year=station_qc_by_year)
    if not prewrite_validation["all_checks_passed"]:
        raise ValueError("Climate silver validation failed before writing partitions")

    partition_files: dict[int, Path] = {}
    for year in years:
        partition_files[year] = write_canonical_climate_silver(root, runtime_root, year, split_frames[year])

    manifest = build_manifest(root, "silver", "climate")
    validation = validate_canonical_climate_silver(
        root,
        split_frames,
        partition_files=partition_files,
        station_qc_by_year=station_qc_by_year,
    )
    if not validation["all_checks_passed"]:
        raise ValueError("Climate silver validation failed after writing partitions")

    manifest_path = f"artifacts/manifests/silver_{manifest['dataset']}.manifest.json"
    _write_validation_reports(root, validation, manifest_path, split_frames, station_qc_by_year)
    for year in years:
        append_run_event(
            log_path,
            phase="phase_2_silver_promotion",
            dataset="climate",
            year=year,
            command=command,
            decision=(
                f"canonical climate silver written with zero missing primary variables and heat index; "
                f"partition path={partition_files[year].relative_to(root)}"
            ),
            status="PASS",
        )
    return {
        "dataset": "climate",
        "years": years,
        "manifest_path": manifest_path,
        "partition_paths": {year: str(path.relative_to(root)) for year, path in partition_files.items()},
        "preselected_metadata": preselected_metadata,
        "imputation_methods": {
            variable: {name: len(table) for name, table in tables.items()}
            for variable, tables in imputation_tables.items()
        },
        "validation": validation,
    }
