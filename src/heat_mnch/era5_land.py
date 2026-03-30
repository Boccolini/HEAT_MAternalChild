from __future__ import annotations

import calendar
import csv
import json
import math
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import cdsapi
import numpy as np
import pandas as pd
import shapefile
import xarray as xr
from zoneinfo import ZoneInfo

from .manifest import build_manifest
from .run_log import append_run_event


ERA5_AREA_BRAZIL = [6, -75, -35, -32]
ERA5_VARIABLES = ["2m_temperature", "2m_dewpoint_temperature"]
ERA5_TIMES = [f"{hour:02d}:00" for hour in range(24)]
ERA5_IBGE_MALHA_2022_URL = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/"
    "malhas_municipais/municipio_2022/Brasil/BR/BR_Municipios_2022.zip"
)
ERA5_IBGE_MALHA_2022_FILE = "BR_Municipios_2022.zip"
ERA5_LOOKUP_HEADER = [
    "COD_MUNICIPIO",
    "NM_MUNICIPIO",
    "UF",
    "MUNICIPIO_CENTROID_LAT",
    "MUNICIPIO_CENTROID_LON",
    "ERA5_LAT",
    "ERA5_LON",
    "TIMEZONE_NAME",
]
ERA5_EXCLUDED_MUNICIPALITY_CODES = {"4300001", "4300002"}
ERA5_NO_USABLE_DATA_FIELD = "ERA5_NO_USABLE_DATA"
ERA5_BRONZE_FIELDS = [
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
    "TMAX_C",
    "TMIN_C",
    "TMEAN_C",
    "RH_MIN_PCT",
    "RH_MEAN_PCT",
    "RH_MAX_PCT",
    "N_OBS",
    ERA5_NO_USABLE_DATA_FIELD,
]
UF_TIMEZONE_MAP = {
    "AC": "America/Rio_Branco",
    "AL": "America/Sao_Paulo",
    "AM": "America/Manaus",
    "AP": "America/Belem",
    "BA": "America/Sao_Paulo",
    "CE": "America/Sao_Paulo",
    "DF": "America/Sao_Paulo",
    "ES": "America/Sao_Paulo",
    "GO": "America/Sao_Paulo",
    "MA": "America/Sao_Paulo",
    "MG": "America/Sao_Paulo",
    "MS": "America/Campo_Grande",
    "MT": "America/Cuiaba",
    "PA": "America/Belem",
    "PB": "America/Sao_Paulo",
    "PE": "America/Sao_Paulo",
    "PI": "America/Sao_Paulo",
    "PR": "America/Sao_Paulo",
    "RJ": "America/Sao_Paulo",
    "RN": "America/Sao_Paulo",
    "RO": "America/Porto_Velho",
    "RR": "America/Boa_Vista",
    "RS": "America/Sao_Paulo",
    "SC": "America/Sao_Paulo",
    "SE": "America/Sao_Paulo",
    "SP": "America/Sao_Paulo",
    "TO": "America/Araguaina",
}


@dataclass(frozen=True)
class Era5MonthPaths:
    year: int
    month: int
    source_path: Path
    request_path: Path


@dataclass(frozen=True)
class Era5ReferenceGridPaths:
    source_path: Path
    request_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def era5_month_filename(year: int, month: int) -> str:
    return f"era5_land_br_hourly_t2m_d2m_{year}{month:02d}.nc"


def era5_request_filename(year: int, month: int) -> str:
    return f"era5_land_br_hourly_t2m_d2m_{year}{month:02d}.request.json"


def era5_month_days(year: int, month: int) -> list[str]:
    return [f"{day:02d}" for day in range(1, calendar.monthrange(year, month)[1] + 1)]


def sync_verified(staging_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(staging_path, final_path)
    if final_path.stat().st_size != staging_path.stat().st_size:
        raise IOError(f"Size mismatch after sync for {final_path}")


def era5_month_paths(root: Path, year: int, month: int) -> Era5MonthPaths:
    return Era5MonthPaths(
        year=year,
        month=month,
        source_path=root / "data" / "raw" / "climate" / "era5_land" / str(year) / era5_month_filename(year, month),
        request_path=(
            root
            / "data"
            / "raw"
            / "climate"
            / "era5_land_requests"
            / str(year)
            / era5_request_filename(year, month)
        ),
    )


def era5_lookup_path(root: Path) -> Path:
    return root / "metadata" / "era5_municipio_grid_lookup.csv"


def era5_support_zip_path(root: Path) -> Path:
    return root / "artifacts" / "reference" / "era5" / ERA5_IBGE_MALHA_2022_FILE


def era5_centroid_cache_path(root: Path) -> Path:
    return root / "artifacts" / "reference" / "era5" / "municipality_centroids_ibge_2022.csv"


def era5_reference_grid_paths(root: Path) -> Era5ReferenceGridPaths:
    return Era5ReferenceGridPaths(
        source_path=root / "artifacts" / "reference" / "era5" / "era5_land_brazil_reference_grid.nc",
        request_path=root / "artifacts" / "reference" / "era5" / "era5_land_brazil_reference_grid.request.json",
    )


def era5_client() -> cdsapi.Client:
    url = __import__("os").environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
    key = __import__("os").environ.get("CDSAPI_KEY")
    if not key:
        raise EnvironmentError("CDSAPI_KEY must be set to extract ERA5-Land raw files")
    return cdsapi.Client(url=url, key=key, quiet=True)


def relative_humidity_percent(temp_c: float, dewpoint_c: float) -> float:
    if math.isnan(temp_c) or math.isnan(dewpoint_c):
        return math.nan
    numerator = math.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
    denominator = math.exp((17.625 * temp_c) / (243.04 + temp_c))
    value = 100.0 * (numerator / denominator)
    return max(0.0, min(100.0, value))


def _relative_humidity_vectorized(temp_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
    numerator = np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
    denominator = np.exp((17.625 * temp_c) / (243.04 + temp_c))
    return pd.Series(np.clip(100.0 * (numerator / denominator), 0.0, 100.0), index=temp_c.index)


def _relative_humidity_array(temp_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    numerator = np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
    denominator = np.exp((17.625 * temp_c) / (243.04 + temp_c))
    return np.clip(100.0 * (numerator / denominator), 0.0, 100.0)


def _format_float(value: float, precision: int) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.{precision}f}"


def _filter_invalid_lookup_codes(lookup: pd.DataFrame) -> pd.DataFrame:
    return lookup[~lookup["COD_MUNICIPIO"].isin(ERA5_EXCLUDED_MUNICIPALITY_CODES)].reset_index(drop=True)


def _era5_no_usable_data_flag(n_obs: int) -> str:
    return "1" if int(n_obs) == 0 else "0"


def _lookup_with_grid_indices(grid_path: Path, lookup: pd.DataFrame) -> pd.DataFrame:
    with xr.open_dataset(grid_path) as dataset:
        latitudes = np.round(dataset["latitude"].astype(float).values, 4)
        longitudes = np.round(dataset["longitude"].astype(float).values, 4)
    latitude_index = {float(value): int(index) for index, value in enumerate(latitudes)}
    longitude_index = {float(value): int(index) for index, value in enumerate(longitudes)}
    indexed_lookup = lookup.copy()
    indexed_lookup["ERA5_LAT_INDEX"] = indexed_lookup["ERA5_LAT"].round(4).map(latitude_index)
    indexed_lookup["ERA5_LON_INDEX"] = indexed_lookup["ERA5_LON"].round(4).map(longitude_index)
    if indexed_lookup["ERA5_LAT_INDEX"].isna().any() or indexed_lookup["ERA5_LON_INDEX"].isna().any():
        raise ValueError("ERA5 municipality lookup contains coordinates outside the reference grid")
    indexed_lookup["ERA5_LAT_INDEX"] = indexed_lookup["ERA5_LAT_INDEX"].astype(int)
    indexed_lookup["ERA5_LON_INDEX"] = indexed_lookup["ERA5_LON_INDEX"].astype(int)
    return indexed_lookup


def _ring_area_centroid(ring: list[tuple[float, float]]) -> tuple[float, float, float]:
    area_twice = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        cross = (x0 * y1) - (x1 * y0)
        area_twice += cross
        centroid_x += (x0 + x1) * cross
        centroid_y += (y0 + y1) * cross
    if area_twice == 0.0:
        xs = [point[0] for point in ring[:-1]]
        ys = [point[1] for point in ring[:-1]]
        return 0.0, sum(xs) / len(xs), sum(ys) / len(ys)
    area = area_twice / 2.0
    return area, centroid_x / (3.0 * area_twice), centroid_y / (3.0 * area_twice)


def _shape_centroid(shape: shapefile.Shape) -> tuple[float, float]:
    points = list(shape.points)
    parts = list(shape.parts) + [len(points)]
    total_area = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    for start, end in zip(parts, parts[1:]):
        ring_points = points[start:end]
        if len(ring_points) < 3:
            continue
        ring: list[tuple[float, float]] = [(float(x), float(y)) for x, y in ring_points]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        area, centroid_x, centroid_y = _ring_area_centroid(ring)
        if area == 0.0:
            continue
        total_area += area
        weighted_x += centroid_x * area
        weighted_y += centroid_y * area
    if total_area == 0.0:
        min_x, min_y, max_x, max_y = shape.bbox
        return (min_y + max_y) / 2.0, (min_x + max_x) / 2.0
    return weighted_y / total_area, weighted_x / total_area


def _download_support_file(url: str, runtime_path: Path, final_path: Path) -> None:
    from urllib.request import Request, urlopen

    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "heat-mnch/0.1 (+era5 support acquisition)"})
    with urlopen(request, timeout=300) as response, runtime_path.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    sync_verified(runtime_path, final_path)
    runtime_path.unlink(missing_ok=True)


def ensure_ibge_malha_zip(root: Path, runtime_root: Path, force: bool = False) -> Path:
    final_path = era5_support_zip_path(root)
    if final_path.exists() and not force:
        return final_path
    runtime_path = runtime_root / "artifacts" / "reference" / "era5" / f"{ERA5_IBGE_MALHA_2022_FILE}.part"
    _download_support_file(ERA5_IBGE_MALHA_2022_URL, runtime_path, final_path)
    return final_path


def build_centroid_cache(root: Path, runtime_root: Path, force: bool = False) -> Path:
    final_path = era5_centroid_cache_path(root)
    if final_path.exists() and not force:
        return final_path
    malha_zip = ensure_ibge_malha_zip(root, runtime_root, force=force)
    runtime_path = runtime_root / "artifacts" / "reference" / "era5" / final_path.name
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile(malha_zip) as archive:
            archive.extractall(tmpdir)
        reader = shapefile.Reader(str(Path(tmpdir) / "BR_Municipios_2022.shp"), encoding="latin1")
        for shape_record in reader.iterShapeRecords():
            code, name, uf = shape_record.record[0], shape_record.record[1], shape_record.record[2]
            centroid_lat, centroid_lon = _shape_centroid(shape_record.shape)
            rows.append(
                {
                    "COD_MUNICIPIO": str(code),
                    "NM_MUNICIPIO": str(name),
                    "UF": str(uf),
                    "MUNICIPIO_CENTROID_LAT": round(float(centroid_lat), 6),
                    "MUNICIPIO_CENTROID_LON": round(float(centroid_lon), 6),
                    "TIMEZONE_NAME": UF_TIMEZONE_MAP[str(uf)],
                }
            )
    dataframe = pd.DataFrame(rows).sort_values(["UF", "COD_MUNICIPIO"]).reset_index(drop=True)
    dataframe.to_csv(runtime_path, index=False)
    sync_verified(runtime_path, final_path)
    runtime_path.unlink(missing_ok=True)
    return final_path


def build_lookup_from_reference_grid(root: Path, runtime_root: Path, reference_grid_path: Path, force: bool = False) -> Path:
    final_path = era5_lookup_path(root)
    if final_path.exists() and not force:
        return final_path
    centroid_cache = build_centroid_cache(root, runtime_root, force=force)
    runtime_path = runtime_root / "metadata" / final_path.name
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    centroids = pd.read_csv(centroid_cache, dtype={"COD_MUNICIPIO": str, "UF": str})
    with xr.open_dataset(reference_grid_path) as dataset:
        latitudes = dataset["latitude"].astype(float).values
        longitudes = dataset["longitude"].astype(float).values
    nearest_lats = latitudes[np.abs(latitudes[:, None] - centroids["MUNICIPIO_CENTROID_LAT"].to_numpy()[None, :]).argmin(axis=0)]
    nearest_lons = longitudes[np.abs(longitudes[:, None] - centroids["MUNICIPIO_CENTROID_LON"].to_numpy()[None, :]).argmin(axis=0)]
    lookup = centroids.assign(
        ERA5_LAT=np.round(nearest_lats.astype(float), 4),
        ERA5_LON=np.round(nearest_lons.astype(float), 4),
    )[ERA5_LOOKUP_HEADER]
    lookup = _filter_invalid_lookup_codes(lookup)
    lookup.to_csv(runtime_path, index=False)
    sync_verified(runtime_path, final_path)
    runtime_path.unlink(missing_ok=True)
    return final_path


def build_era5_request(year: int, month: int) -> dict[str, object]:
    return {
        "variable": ERA5_VARIABLES,
        "year": [f"{year:04d}"],
        "month": [f"{month:02d}"],
        "day": era5_month_days(year, month),
        "time": ERA5_TIMES,
        "area": ERA5_AREA_BRAZIL,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def build_era5_reference_grid_request() -> dict[str, object]:
    return {
        "variable": ERA5_VARIABLES,
        "year": ["2024"],
        "month": ["01"],
        "day": ["01", "02"],
        "time": ["00:00", "01:00"],
        "area": ERA5_AREA_BRAZIL,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def ensure_reference_grid(root: Path, runtime_root: Path, force: bool = False) -> Era5ReferenceGridPaths:
    paths = era5_reference_grid_paths(root)
    if paths.source_path.exists() and paths.request_path.exists() and not force:
        return paths
    client = era5_client()
    request_payload = build_era5_reference_grid_request()
    runtime_source = runtime_root / paths.source_path.relative_to(root)
    runtime_request = runtime_root / paths.request_path.relative_to(root)
    runtime_source.parent.mkdir(parents=True, exist_ok=True)
    runtime_request.parent.mkdir(parents=True, exist_ok=True)
    runtime_part = runtime_source.with_suffix(".nc.part")
    if runtime_part.exists():
        runtime_part.unlink()
    client.retrieve("reanalysis-era5-land", request_payload, str(runtime_part))
    runtime_request.write_text(json.dumps(request_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sync_verified(runtime_part, paths.source_path)
    sync_verified(runtime_request, paths.request_path)
    runtime_part.unlink(missing_ok=True)
    runtime_request.unlink(missing_ok=True)
    return paths


def ensure_raw_month(
    root: Path,
    runtime_root: Path,
    client: cdsapi.Client,
    year: int,
    month: int,
    force: bool = False,
) -> Era5MonthPaths:
    paths = era5_month_paths(root, year, month)
    if paths.source_path.exists() and paths.request_path.exists() and not force:
        return paths
    request_payload = build_era5_request(year, month)
    runtime_source = runtime_root / paths.source_path.relative_to(root)
    runtime_request = runtime_root / paths.request_path.relative_to(root)
    runtime_source.parent.mkdir(parents=True, exist_ok=True)
    runtime_request.parent.mkdir(parents=True, exist_ok=True)
    runtime_part = runtime_source.with_suffix(".nc.part")
    if runtime_part.exists():
        runtime_part.unlink()
    client.retrieve("reanalysis-era5-land", request_payload, str(runtime_part))
    runtime_request.write_text(json.dumps(request_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sync_verified(runtime_part, paths.source_path)
    sync_verified(runtime_request, paths.request_path)
    runtime_part.unlink(missing_ok=True)
    runtime_request.unlink(missing_ok=True)
    return paths


def _monthly_local_daily_frame(root: Path, source_path: Path, lookup: pd.DataFrame, target_year: int) -> pd.DataFrame:
    latitude_indices = lookup["ERA5_LAT_INDEX"].to_numpy(dtype=int)
    longitude_indices = lookup["ERA5_LON_INDEX"].to_numpy(dtype=int)
    with xr.open_dataset(source_path) as dataset:
        utc_timestamps = pd.to_datetime(dataset["valid_time"].values, utc=True)
        t2m_cube = dataset["t2m"].values
        temp_c = t2m_cube[:, latitude_indices, longitude_indices].astype(np.float64) - 273.15
        del t2m_cube
        d2m_cube = dataset["d2m"].values
        dewpoint_c = d2m_cube[:, latitude_indices, longitude_indices].astype(np.float64) - 273.15
        del d2m_cube
    rh_pct = _relative_humidity_array(temp_c, dewpoint_c)
    rows: list[pd.DataFrame] = []
    lookup = lookup.reset_index(drop=True)
    for timezone_name, timezone_lookup in lookup.groupby("TIMEZONE_NAME", sort=False):
        municipality_indices = timezone_lookup.index.to_numpy(dtype=int)
        local_timestamps = utc_timestamps.tz_convert(ZoneInfo(timezone_name))
        local_years = local_timestamps.year
        local_months = local_timestamps.month
        local_dates = local_timestamps.strftime("%Y-%m-%d")
        target_mask = local_years == target_year
        if not np.any(target_mask):
            continue
        timezone_temp = temp_c[:, municipality_indices]
        timezone_rh = rh_pct[:, municipality_indices]
        unique_dates = pd.Index(local_dates[target_mask]).unique()
        for date in unique_dates:
            day_mask = target_mask & (local_dates == date)
            day_temp = timezone_temp[day_mask, :]
            day_rh = timezone_rh[day_mask, :]
            valid_counts = np.sum(~np.isnan(day_rh), axis=0).astype(int)
            temp_valid_counts = np.sum(~np.isnan(day_temp), axis=0).astype(int)
            tmax = np.full(day_temp.shape[1], np.nan)
            tmin = np.full(day_temp.shape[1], np.nan)
            rh_min = np.full(day_rh.shape[1], np.nan)
            rh_max = np.full(day_rh.shape[1], np.nan)
            temp_sum = np.nansum(day_temp, axis=0)
            has_temp = np.any(~np.isnan(day_temp), axis=0)
            has_rh = valid_counts > 0
            if np.any(has_temp):
                tmax[has_temp] = np.nanmax(day_temp[:, has_temp], axis=0)
                tmin[has_temp] = np.nanmin(day_temp[:, has_temp], axis=0)
            if np.any(has_rh):
                rh_min[has_rh] = np.nanmin(day_rh[:, has_rh], axis=0)
                rh_max[has_rh] = np.nanmax(day_rh[:, has_rh], axis=0)
            rh_sum = np.nansum(day_rh, axis=0)
            row_frame = timezone_lookup.copy()
            row_frame["DATE"] = date
            row_frame["YEAR"] = target_year
            row_frame["MONTH"] = int(local_months[day_mask][0])
            row_frame["TMAX_C"] = tmax
            row_frame["TMIN_C"] = tmin
            row_frame["TEMP_SUM"] = temp_sum
            row_frame["TEMP_N_OBS"] = temp_valid_counts
            row_frame["RH_MIN_PCT"] = rh_min
            row_frame["RH_SUM"] = rh_sum
            row_frame["RH_MAX_PCT"] = rh_max
            row_frame["N_OBS"] = valid_counts
            rows.append(row_frame)
    if not rows:
        return pd.DataFrame(
            columns=[
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
                "TMAX_C",
                "TMIN_C",
                "TEMP_SUM",
                "TEMP_N_OBS",
                "RH_MIN_PCT",
                "RH_SUM",
                "RH_MAX_PCT",
                "N_OBS",
                "BRONZE_SOURCE_FILE",
                "BRONZE_SOURCE_PATH",
            ]
        )
    grouped = pd.concat(rows, ignore_index=True)
    grouped["BRONZE_SOURCE_FILE"] = source_path.name
    grouped["BRONZE_SOURCE_PATH"] = str(source_path.relative_to(root))
    return grouped


def build_bronze_annual_rows(root: Path, year: int, raw_paths: list[Path]) -> pd.DataFrame:
    lookup = pd.read_csv(
        era5_lookup_path(root),
        dtype={"COD_MUNICIPIO": str, "UF": str, "TIMEZONE_NAME": str},
    )
    lookup = _filter_invalid_lookup_codes(lookup)
    reference_grid_path = era5_reference_grid_paths(root).source_path
    lookup = _lookup_with_grid_indices(
        reference_grid_path if reference_grid_path.exists() else raw_paths[0],
        lookup,
    )
    monthly_frames = [_monthly_local_daily_frame(root, path, lookup, year) for path in raw_paths]
    combined = pd.concat(monthly_frames, ignore_index=True)
    aggregated = (
        combined.groupby(
            [
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
            ],
            as_index=False,
        )
        .agg(
            TMAX_C=("TMAX_C", "max"),
            TMIN_C=("TMIN_C", "min"),
            TEMP_SUM=("TEMP_SUM", "sum"),
            TEMP_N_OBS=("TEMP_N_OBS", "sum"),
            RH_MIN_PCT=("RH_MIN_PCT", "min"),
            RH_SUM=("RH_SUM", "sum"),
            RH_MAX_PCT=("RH_MAX_PCT", "max"),
            N_OBS=("N_OBS", "sum"),
            BRONZE_SOURCE_FILE=("BRONZE_SOURCE_FILE", lambda values: "|".join(sorted(set(values)))),
            BRONZE_SOURCE_PATH=("BRONZE_SOURCE_PATH", lambda values: "|".join(sorted(set(values)))),
        )
    )
    aggregated["TMEAN_C"] = np.where(
        aggregated["TEMP_N_OBS"] > 0,
        aggregated["TEMP_SUM"] / aggregated["TEMP_N_OBS"],
        np.nan,
    )
    aggregated["RH_MEAN_PCT"] = np.where(
        aggregated["N_OBS"] > 0,
        aggregated["RH_SUM"] / aggregated["N_OBS"],
        np.nan,
    )
    return aggregated.drop(columns=["TEMP_SUM", "TEMP_N_OBS", "RH_SUM"]).sort_values(["DATE", "COD_MUNICIPIO"]).reset_index(drop=True)


def bronze_rows_from_dataframe(dataframe: pd.DataFrame, year: int) -> Iterable[dict[str, str]]:
    timestamp = utc_now()
    for row in dataframe.itertuples(index=False):
        tmax_c = float(getattr(row, "TMAX_C", math.nan))
        tmin_c = float(getattr(row, "TMIN_C", math.nan))
        tmean_c = float(getattr(row, "TMEAN_C", math.nan))
        rh_min_pct = float(getattr(row, "RH_MIN_PCT", math.nan))
        rh_mean_pct = float(getattr(row, "RH_MEAN_PCT", math.nan))
        rh_max_pct = float(getattr(row, "RH_MAX_PCT", math.nan))
        yield {
            "BRONZE_DATASET": "era5_land",
            "BRONZE_SOURCE_SYSTEM": "era5_land",
            "BRONZE_SOURCE_FILE": str(row.BRONZE_SOURCE_FILE),
            "BRONZE_SOURCE_PATH": str(row.BRONZE_SOURCE_PATH),
            "BRONZE_SOURCE_MEMBER": "",
            "BRONZE_YEAR": str(year),
            "BRONZE_INGESTED_AT": timestamp,
            "COD_MUNICIPIO": str(row.COD_MUNICIPIO),
            "NM_MUNICIPIO": str(row.NM_MUNICIPIO),
            "UF": str(row.UF),
            "TIMEZONE_NAME": str(row.TIMEZONE_NAME),
            "DATE": str(row.DATE),
            "YEAR": str(int(row.YEAR)),
            "MONTH": str(int(row.MONTH)),
            "MUNICIPIO_CENTROID_LAT": f"{float(row.MUNICIPIO_CENTROID_LAT):.6f}",
            "MUNICIPIO_CENTROID_LON": f"{float(row.MUNICIPIO_CENTROID_LON):.6f}",
            "ERA5_LAT": f"{float(row.ERA5_LAT):.4f}",
            "ERA5_LON": f"{float(row.ERA5_LON):.4f}",
            "TMAX_C": _format_float(tmax_c, 4),
            "TMIN_C": _format_float(tmin_c, 4),
            "TMEAN_C": _format_float(tmean_c, 4),
            "RH_MIN_PCT": _format_float(rh_min_pct, 4),
            "RH_MEAN_PCT": _format_float(rh_mean_pct, 4),
            "RH_MAX_PCT": _format_float(rh_max_pct, 4),
            "N_OBS": str(int(row.N_OBS)),
            ERA5_NO_USABLE_DATA_FIELD: _era5_no_usable_data_flag(int(row.N_OBS)),
        }


def extract_era5_land_dataset(
    root: Path,
    runtime_root: Path,
    years: list[int],
    log_path: Path,
    command: str,
    force: bool = False,
    max_workers: int = 1,
) -> dict[str, object]:
    report: dict[str, object] = {
        "dataset": "era5_land",
        "requested_years": years,
        "generated_at": utc_now(),
        "force": force,
        "runtime_root": str(runtime_root),
        "max_workers": max_workers,
        "downloads": [],
        "skipped_existing": [],
        "support_artifacts": [],
    }
    support_zip = ensure_ibge_malha_zip(root, runtime_root, force=force)
    centroid_cache = build_centroid_cache(root, runtime_root, force=force)
    report["support_artifacts"].extend(
        [
            str(support_zip.relative_to(root)),
            str(centroid_cache.relative_to(root)),
        ]
    )

    reference_grid = ensure_reference_grid(root, runtime_root, force=force)
    report["reference_grid_source"] = str(reference_grid.source_path.relative_to(root))
    report["support_artifacts"].append(str(reference_grid.request_path.relative_to(root)))
    lookup_path = build_lookup_from_reference_grid(root, runtime_root, reference_grid.source_path, force=force)
    report["support_artifacts"].append(str(lookup_path.relative_to(root)))

    per_year_counts = {year: 0 for year in years}
    per_year_bytes = {year: 0 for year in years}
    for year in years:
        months = list(range(1, 13))

        def materialize(month: int) -> tuple[int, Era5MonthPaths, bool]:
            prior = era5_month_paths(root, year, month).source_path.exists() and era5_month_paths(root, year, month).request_path.exists()
            paths = ensure_raw_month(root, runtime_root, era5_client(), year, month, force=force)
            return month, paths, prior

        year_records: list[tuple[int, Era5MonthPaths, bool]] = []
        if months:
            if max_workers <= 1:
                year_records.extend(materialize(month) for month in months)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_map = {executor.submit(materialize, month): month for month in months}
                    for future in as_completed(future_map):
                        year_records.append(future.result())
        for month, paths, prior_exists in sorted(year_records, key=lambda item: item[0]):
            kind = "skipped_existing" if prior_exists and not force else "downloads"
            report[kind].append(
                {
                    "source_name": "era5_land",
                    "dataset": "climate",
                    "year": year,
                    "month": month,
                    "source_path": str(paths.source_path.relative_to(root)),
                    "request_path": str(paths.request_path.relative_to(root)),
                    "size_bytes": paths.source_path.stat().st_size,
                }
            )
            per_year_counts[year] += 1
            per_year_bytes[year] += paths.source_path.stat().st_size

    boundary_year = max(years) + 1
    boundary_paths = ensure_raw_month(root, runtime_root, era5_client(), boundary_year, 1, force=force)
    report["boundary_support_month"] = {
        "year": boundary_year,
        "month": 1,
        "source_path": str(boundary_paths.source_path.relative_to(root)),
        "request_path": str(boundary_paths.request_path.relative_to(root)),
        "reason": "required to complete local-time aggregation for December 31 of the last requested year",
    }

    manifest = build_manifest(root, "raw", "climate")
    report["manifest_path"] = f"artifacts/manifests/raw_{manifest['dataset']}.manifest.json"
    report_path = root / "artifacts" / "status" / "raw_era5_land_extraction_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for year in years:
        append_run_event(
            log_path,
            phase="phase_1_raw_extraction",
            dataset="era5_land",
            year=year,
            command=command,
            decision=f"materialized 12 monthly ERA5-Land files totaling {per_year_bytes[year]} bytes",
            status="completed" if per_year_counts[year] == 12 else "blocked",
        )
    return report


def write_era5_bronze_summary(
    root: Path,
    year: int,
    source_glob: str,
    target_path: Path,
    raw_paths: list[Path],
    row_count: int,
    target_entry: dict[str, object],
) -> None:
    summary_path = root / "artifacts" / "status" / "bronze_era5_land_last_run_summary.md"
    manifest_csv = root / "artifacts" / "status" / "bronze_era5_land_manifest.csv"
    manifest_jsonl = root / "artifacts" / "status" / "bronze_era5_land_manifest.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now()
    summary_path.write_text(
        "\n".join(
            [
                "bronze granularity: daily_municipio_localtime",
                f"year: {year}",
                "variables: TMAX_C, TMIN_C, TMEAN_C, RH_MIN_PCT, RH_MEAN_PCT, RH_MAX_PCT, N_OBS",
                f"source_glob: {source_glob}",
                f"target_path: {target_path.relative_to(root)}",
                f"row_count: {row_count}",
                f"months_found: {', '.join(sorted(path.stem[-6:] for path in raw_paths))}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not manifest_csv.exists():
        manifest_csv.write_text(
            "timestamp,year,source_glob,target_path,row_count,file_size_bytes,sha256,months_found\n",
            encoding="utf-8",
        )
    months_found = "|".join(sorted(path.stem[-6:] for path in raw_paths))
    with manifest_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                timestamp,
                year,
                source_glob,
                str(target_path.relative_to(root)),
                row_count,
                target_entry["size_bytes"],
                target_entry["sha256"],
                months_found,
            ]
        )
    with manifest_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "year": year,
                    "source_glob": source_glob,
                    "target_path": str(target_path.relative_to(root)),
                    "row_count": row_count,
                    "file_size_bytes": target_entry["size_bytes"],
                    "sha256": target_entry["sha256"],
                    "months_found": months_found.split("|"),
                }
            )
        )
        handle.write("\n")
