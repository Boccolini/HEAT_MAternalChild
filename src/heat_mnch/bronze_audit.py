from __future__ import annotations

import calendar
import argparse
import csv
import gzip
import json
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import datasus_dbc
import pandas as pd
import xarray as xr

from .bronze import (
    INMET_ERA5_FALLBACK_FLAG_FIELD,
    INMET_FLAG_FIELDS,
    INMET_HUMIDITY_FIELD,
    INMET_MISSING_HUMIDITY_OFFICIAL_FIELD,
    INMET_MISSING_TEMP_OFFICIAL_FIELD,
    INMET_TEMPERATURE_FIELD,
    bronze_output_path,
    load_ibge_total_benchmark_frame,
    load_bronze_rules,
    population_external_benchmarks_ready,
    population_derivative_expected_codes,
    raw_paths_for_dataset,
)
from .config import load_project_config
from .era5_land import era5_month_filename
from .era5_land import ERA5_EXCLUDED_MUNICIPALITY_CODES, ERA5_NO_USABLE_DATA_FIELD, era5_lookup_path
from .manifest import build_manifest, count_rows
from .preflight import probe_path
from .raw_sources import UF_CODES


YEARS = tuple(range(2020, 2025))
MUNICIPALITY_INVALID_BLOCK_RATE = 0.0001
DATE_COLUMNS_BY_DATASET = {
    "sim": ("DTOBITO",),
    "sinasc": ("DTNASC",),
    "stillbirth": ("DTOBITO",),
    "inmet": ("DATA",),
    "era5_land": ("DATE",),
    "sih": ("DT_INTER", "DT_SAIDA"),
}
VALID_UFS = set(UF_CODES)


@dataclass(frozen=True)
class DatasetAuditSpec:
    dataset: str
    label: str
    internal_dataset: str | None
    critical_fields: tuple[str, ...]
    municipality_fields: tuple[str, ...]
    municipality_code_length: int | None
    coverage_field: str | None
    daily_date_field: str | None
    month_field: str | None
    year_field: str | None
    duplicate_key_fields: tuple[str, ...]
    duplicate_group_field: str | None
    conceptual_only: bool = False
    expected_bronze_path_hint: str | None = None


AUDIT_SPECS: tuple[DatasetAuditSpec, ...] = (
    DatasetAuditSpec(
        dataset="bronze_inmet",
        label="bronze_inmet",
        internal_dataset="inmet",
        critical_fields=(
            "DATA",
            "HORA_UTC",
            "CODIGO_WMO",
            "UF",
            "TEMPERATURA_DO_AR_BULBO_SECO_HORARIA_C",
            "UMIDADE_RELATIVA_DO_AR_HORARIA",
        ),
        municipality_fields=(),
        municipality_code_length=None,
        coverage_field=None,
        daily_date_field="DATA",
        month_field=None,
        year_field=None,
        duplicate_key_fields=("DATA", "HORA_UTC"),
        duplicate_group_field="BRONZE_SOURCE_MEMBER",
    ),
    DatasetAuditSpec(
        dataset="bronze_era5_land",
        label="bronze_era5_land",
        internal_dataset="era5_land",
        critical_fields=("DATE", "COD_MUNICIPIO", "TMAX_C", "RH_MEAN_PCT", "N_OBS"),
        municipality_fields=("COD_MUNICIPIO",),
        municipality_code_length=7,
        coverage_field="COD_MUNICIPIO",
        daily_date_field="DATE",
        month_field="MONTH",
        year_field="YEAR",
        duplicate_key_fields=("DATE", "COD_MUNICIPIO"),
        duplicate_group_field="DATE",
    ),
    DatasetAuditSpec(
        dataset="bronze_sinasc",
        label="bronze_sinasc",
        internal_dataset="sinasc",
        critical_fields=("DTNASC", "CODMUNNASC", "CODMUNRES", "SEXO", "PESO"),
        municipality_fields=("CODMUNNASC", "CODMUNRES"),
        municipality_code_length=6,
        coverage_field="CODMUNNASC",
        daily_date_field="DTNASC",
        month_field=None,
        year_field=None,
        duplicate_key_fields=("CONTADOR",),
        duplicate_group_field=None,
    ),
    DatasetAuditSpec(
        dataset="bronze_sim",
        label="bronze_sim",
        internal_dataset="sim",
        critical_fields=("DTOBITO", "CODMUNOCOR", "CODMUNRES", "SEXO", "CAUSABAS"),
        municipality_fields=("CODMUNOCOR", "CODMUNRES"),
        municipality_code_length=6,
        coverage_field="CODMUNOCOR",
        daily_date_field="DTOBITO",
        month_field=None,
        year_field=None,
        duplicate_key_fields=("CONTADOR",),
        duplicate_group_field="BRONZE_SOURCE_FILE",
    ),
    DatasetAuditSpec(
        dataset="bronze_sih",
        label="bronze_sih",
        internal_dataset="sih",
        critical_fields=("ANO_CMPT", "MES_CMPT", "MUNIC_RES", "N_AIH", "DIAG_PRINC"),
        municipality_fields=("MUNIC_RES",),
        municipality_code_length=6,
        coverage_field="MUNIC_RES",
        daily_date_field=None,
        month_field="MES_CMPT",
        year_field="ANO_CMPT",
        # SIH bronze is source-preserving: repeated N_AIH values are profiled as
        # quality signals here, while analytical deduplication happens in silver.
        duplicate_key_fields=(),
        duplicate_group_field="BRONZE_SOURCE_FILE",
    ),
    DatasetAuditSpec(
        dataset="bronze_stillbirth",
        label="bronze_stillbirth_dofet",
        internal_dataset="stillbirth",
        critical_fields=("DTOBITO", "CODMUNOCOR", "CODMUNRES"),
        municipality_fields=("CODMUNOCOR", "CODMUNRES"),
        municipality_code_length=6,
        coverage_field="CODMUNOCOR",
        daily_date_field="DTOBITO",
        month_field=None,
        year_field=None,
        duplicate_key_fields=("CONTADOR",),
        duplicate_group_field="BRONZE_SOURCE_FILE",
    ),
    DatasetAuditSpec(
        dataset="bronze_pop_under5",
        label="bronze_pop_under5",
        internal_dataset=None,
        critical_fields=("COD_MUNICIPIO", "ANO", "POP_UNDER5"),
        municipality_fields=("COD_MUNICIPIO",),
        municipality_code_length=7,
        coverage_field="COD_MUNICIPIO",
        daily_date_field=None,
        month_field=None,
        year_field="ANO",
        duplicate_key_fields=("COD_MUNICIPIO", "ANO"),
        duplicate_group_field=None,
        conceptual_only=True,
        expected_bronze_path_hint="data/bronze/population/under5/{year}/pop_under5_{year}_source_preserving.csv.gz",
    ),
    DatasetAuditSpec(
        dataset="bronze_pop_wra",
        label="bronze_pop_wra",
        internal_dataset=None,
        critical_fields=("COD_MUNICIPIO", "ANO", "POP_WRA"),
        municipality_fields=("COD_MUNICIPIO",),
        municipality_code_length=7,
        coverage_field="COD_MUNICIPIO",
        daily_date_field=None,
        month_field=None,
        year_field="ANO",
        duplicate_key_fields=("COD_MUNICIPIO", "ANO"),
        duplicate_group_field=None,
        conceptual_only=True,
        expected_bronze_path_hint="data/bronze/population/wra/{year}/pop_wra_{year}_source_preserving.csv.gz",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def official_calendar_dates(year: int) -> list[str]:
    return [date(year, month, day).isoformat() for month in range(1, 13) for day in range(1, calendar.monthrange(year, month)[1] + 1)]


def official_year_months() -> set[int]:
    return set(range(1, 13))


def parse_iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def parse_int(value: str) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def normalize_municipality_code(value: str | None, expected_length: int) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return None
    if len(digits) > expected_length:
        return None
    digits = digits.zfill(expected_length)
    if digits in {"000000", "0000000"}:
        return None
    return digits


def build_official_municipality_registry(root: Path) -> dict[int, dict[str, Any]]:
    registry: dict[int, dict[str, Any]] = {}
    for year in YEARS:
        year_dir = root / "data" / "raw" / "population" / "ibge" / str(year)
        files = sorted(year_dir.glob("*.xls"))
        if not files:
            registry[year] = {"code6": set(), "code7": set(), "source_path": None}
            continue
        path = files[0]
        workbook = pd.ExcelFile(path)
        target_sheet = next((name for name in workbook.sheet_names if "MUNIC" in str(name).upper()), workbook.sheet_names[0])
        dataframe = pd.read_excel(path, sheet_name=target_sheet, header=1)
        dataframe = dataframe.dropna(axis=1, how="all")
        columns = {str(column).upper(): column for column in dataframe.columns}
        uf_col = next(column for name, column in columns.items() if name == "UF")
        coduf_col = next(column for name, column in columns.items() if "COD. UF" in name or "COD UF" in name)
        codmun_col = next(column for name, column in columns.items() if "COD. MUNIC" in name or "COD MUNIC" in name)
        dataframe = dataframe[[uf_col, coduf_col, codmun_col]].copy()
        dataframe = dataframe[dataframe[uf_col].astype(str).str.fullmatch(r"[A-Z]{2}")]
        dataframe = dataframe.assign(
            CODUF_NUM=pd.to_numeric(dataframe[coduf_col], errors="coerce"),
            CODMUN_NUM=pd.to_numeric(dataframe[codmun_col], errors="coerce"),
        )
        dataframe = dataframe[dataframe["CODUF_NUM"].notna() & dataframe["CODMUN_NUM"].notna()].copy()
        code7 = (
            dataframe["CODUF_NUM"].astype(int).astype(str).str.zfill(2)
            + dataframe["CODMUN_NUM"].astype(int).astype(str).str.zfill(5)
        )
        code7_set = set(code7.tolist())
        code6_set = {code[:-1] for code in code7_set}
        registry[year] = {
            "code6": code6_set,
            "code7": code7_set,
            "expected_municipality_count_6": len(code6_set),
            "expected_municipality_count_7": len(code7_set),
            "source_path": str(path.relative_to(root)),
        }
    return registry


def build_era5_lookup_registry(root: Path) -> dict[str, Any]:
    path = era5_lookup_path(root)
    if not path.exists():
        return {
            "code7": set(),
            "expected_municipality_count_7": 0,
            "source_path": None,
            "excluded_codes": sorted(ERA5_EXCLUDED_MUNICIPALITY_CODES),
        }
    dataframe = pd.read_csv(path, dtype={"COD_MUNICIPIO": str})
    if "COD_MUNICIPIO" not in dataframe.columns:
        return {
            "code7": set(),
            "expected_municipality_count_7": 0,
            "source_path": str(path.relative_to(root)),
            "excluded_codes": sorted(ERA5_EXCLUDED_MUNICIPALITY_CODES),
        }
    code7_set = {
        normalized
        for value in dataframe["COD_MUNICIPIO"].tolist()
        if (normalized := normalize_municipality_code(value, 7)) and normalized not in ERA5_EXCLUDED_MUNICIPALITY_CODES
    }
    return {
        "code7": code7_set,
        "expected_municipality_count_7": len(code7_set),
        "source_path": str(path.relative_to(root)),
        "excluded_codes": sorted(ERA5_EXCLUDED_MUNICIPALITY_CODES),
    }


def load_bronze_promotion_benchmarks(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    benchmarks: dict[tuple[str, int], dict[str, Any]] = {}
    logs_dir = root / "artifacts" / "run_logs"
    if not logs_dir.exists():
        return benchmarks
    for path in sorted(logs_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("event_type") != "bronze_promotion":
                    continue
                dataset = payload.get("dataset")
                year = payload.get("year")
                row_count = payload.get("row_count")
                status = str(payload.get("status", "")).lower()
                if not isinstance(dataset, str) or not isinstance(year, int):
                    continue
                if status != "pass" or not isinstance(row_count, int):
                    continue
                existing = benchmarks.get((dataset, year))
                if existing and existing["row_count"] > 0 and row_count == 0:
                    continue
                benchmarks[(dataset, year)] = {
                    "row_count": row_count,
                    "log_path": str(path.relative_to(root)),
                }
    return benchmarks


def manifest_targets_for_specs(specs: list[DatasetAuditSpec]) -> list[tuple[str, str]]:
    targets: set[tuple[str, str]] = set()
    for spec in specs:
        if spec.conceptual_only:
            targets.add(("raw", "population"))
            targets.add(("bronze", "population"))
            continue
        assert spec.internal_dataset is not None
        if spec.internal_dataset in {"inmet", "era5_land"}:
            targets.add(("raw", "climate"))
            targets.add(("bronze", "climate"))
            continue
        targets.add(("raw", spec.internal_dataset))
        targets.add(("bronze", spec.internal_dataset))
    return sorted(targets)


def manifest_is_stale(root: Path, layer: str, dataset: str) -> bool:
    manifest_path = root / "artifacts" / "manifests" / f"{layer}_{dataset}.manifest.json"
    if not manifest_path.exists():
        return True
    dataset_dir = root / "data" / layer / dataset
    if not dataset_dir.exists():
        return False
    manifest_mtime = manifest_path.stat().st_mtime
    return any(path.stat().st_mtime > manifest_mtime for path in dataset_dir.rglob("*") if path.is_file())


def build_manifests_for_audit(root: Path, specs: list[DatasetAuditSpec]) -> dict[str, str]:
    manifests: dict[str, str] = {}
    for layer, dataset in manifest_targets_for_specs(specs):
        manifest_path = root / "artifacts" / "manifests" / f"{layer}_{dataset}.manifest.json"
        if manifest_is_stale(root, layer, dataset):
            manifest = build_manifest(root, layer, dataset)
            manifests[f"{layer}_{dataset}"] = f"artifacts/manifests/{layer}_{manifest['dataset']}.manifest.json"
        else:
            manifests[f"{layer}_{dataset}"] = str(manifest_path.relative_to(root))
    return manifests


def manifest_entry_for_path(root: Path, manifest_path: Path, target_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists() or not target_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_relative = str(target_path.relative_to(root))
    return next((entry for entry in payload.get("entries", []) if entry.get("path") == target_relative), None)


def dbc_record_count(path: Path) -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        dbf_path = Path(tmpdir) / f"{path.stem}.dbf"
        datasus_dbc.decompress(str(path), str(dbf_path))
        with dbf_path.open("rb") as handle:
            handle.seek(4)
            return int.from_bytes(handle.read(4), "little")


def zip_csv_row_count(path: Path) -> int:
    with ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one CSV member in {path}, found {len(members)}")
        with archive.open(members[0]) as handle:
            row_count = -1
            for row_count, _ in enumerate(handle, start=0):
                pass
        return max(row_count, 0)


def inmet_zip_row_count(path: Path) -> tuple[int, int]:
    total_rows = 0
    member_count = 0
    with ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".csv"):
                continue
            member_count += 1
            with archive.open(member) as handle:
                line_count = sum(1 for _ in handle)
            total_rows += max(line_count - 9, 0)
    return total_rows, member_count


def zip_dbf_record_count(path: Path) -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".dbf")]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one DBF member in {path}, found {len(members)}")
            member = members[0]
            dbf_path = Path(tmpdir) / Path(member).name
            with archive.open(member) as source, dbf_path.open("wb") as target:
                target.write(source.read())
        with dbf_path.open("rb") as handle:
            handle.seek(4)
            return int.from_bytes(handle.read(4), "little")


def era5_raw_completeness(root: Path, year: int) -> dict[str, Any]:
    year_dir = root / "data" / "raw" / "climate" / "era5_land" / str(year)
    observed_months = sorted(
        int(path.stem[-2:])
        for path in year_dir.glob("*.nc")
        if path.name == era5_month_filename(year, int(path.stem[-2:]))
    )
    boundary_path = root / "data" / "raw" / "climate" / "era5_land" / str(year + 1) / era5_month_filename(year + 1, 1)
    return {
        "raw_paths": [str(path.relative_to(root)) for path in sorted(year_dir.glob("*.nc"))] + ([str(boundary_path.relative_to(root))] if boundary_path.exists() else []),
        "observed_months": observed_months,
        "missing_months": sorted(month for month in range(1, 13) if month not in observed_months),
        "boundary_support_present": boundary_path.exists(),
    }


def raw_status_for_existing_dataset(
    root: Path,
    spec: DatasetAuditSpec,
    year: int,
    benchmarks: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    assert spec.internal_dataset is not None
    raw_paths = [path for path in raw_paths_for_dataset(root, spec.internal_dataset, year) if path.is_file()]
    probes = [probe_path(path) for path in raw_paths]
    all_readable = bool(raw_paths) and all(probe["readable"] for probe in probes)
    benchmark = benchmarks.get((spec.internal_dataset, year))

    if spec.internal_dataset in {"sim", "sih", "stillbirth"}:
        raw_row_count = benchmark["row_count"] if benchmark else sum(dbc_record_count(path) for path in raw_paths)
        return {
            "raw_paths": [str(path.relative_to(root)) for path in raw_paths],
            "raw_file_count": len(raw_paths),
            "raw_readable": all_readable,
            "raw_row_count": raw_row_count,
            "raw_row_count_source": "bronze_promotion_log" if benchmark else "direct_raw_count",
            "benchmark_log_path": benchmark["log_path"] if benchmark else None,
            "raw_probes": probes,
        }

    if spec.internal_dataset == "sinasc":
        raw_row_count = benchmark["row_count"] if benchmark else sum(zip_csv_row_count(path) for path in raw_paths)
        return {
            "raw_paths": [str(path.relative_to(root)) for path in raw_paths],
            "raw_file_count": len(raw_paths),
            "raw_readable": all_readable,
            "raw_row_count": raw_row_count,
            "raw_row_count_source": "bronze_promotion_log" if benchmark else "direct_raw_count",
            "benchmark_log_path": benchmark["log_path"] if benchmark else None,
            "raw_probes": probes,
        }

    if spec.internal_dataset == "inmet":
        raw_row_count = benchmark["row_count"] if benchmark else 0
        raw_member_count = 0
        for path in raw_paths:
            rows, members = inmet_zip_row_count(path)
            if not benchmark:
                raw_row_count += rows
            raw_member_count += members
        return {
            "raw_paths": [str(path.relative_to(root)) for path in raw_paths],
            "raw_file_count": len(raw_paths),
            "raw_readable": all_readable,
            "raw_row_count": raw_row_count,
            "raw_member_count": raw_member_count,
            "raw_row_count_source": "bronze_promotion_log" if benchmark else "direct_raw_count",
            "benchmark_log_path": benchmark["log_path"] if benchmark else None,
            "raw_probes": probes,
        }

    if spec.internal_dataset == "era5_land":
        completeness = era5_raw_completeness(root, year)
        raw_readable = all_readable
        for path in raw_paths:
            with xr.open_dataset(path):
                pass
        boundary_readable = True
        boundary_path = root / "data" / "raw" / "climate" / "era5_land" / str(year + 1) / era5_month_filename(year + 1, 1)
        if boundary_path.exists():
            with xr.open_dataset(boundary_path):
                pass
        else:
            boundary_readable = False
        return {
            **completeness,
            "raw_file_count": len(raw_paths) + (1 if boundary_path.exists() else 0),
            "raw_readable": raw_readable and boundary_readable,
            "raw_probes": probes + ([probe_path(boundary_path)] if boundary_path.exists() else []),
        }

    raise ValueError(f"Unsupported existing dataset for raw status: {spec.internal_dataset}")


def raw_status_for_conceptual_population(root: Path, year: int) -> dict[str, Any]:
    ibge_paths = sorted((root / "data" / "raw" / "population" / "ibge" / str(year)).glob("*.xls"))
    sidra_paths = sorted((root / "data" / "raw" / "population" / "ibge_sidra_6407" / str(year)).glob("*.json"))
    popsvs_paths = sorted((root / "data" / "raw" / "population" / "datasus_popsvs" / str(year)).glob("*.zip"))
    raw_paths = ibge_paths + sidra_paths + popsvs_paths
    probes = [probe_path(path) for path in raw_paths]
    official_counts = {
        "ibge_total_rows": None,
        "sidra_rows": None,
        "popsvs_rows": None,
    }
    if ibge_paths:
        workbook = pd.ExcelFile(ibge_paths[0])
        target_sheet = next((name for name in workbook.sheet_names if "MUNIC" in str(name).upper()), workbook.sheet_names[0])
        dataframe = pd.read_excel(ibge_paths[0], sheet_name=target_sheet, header=1)
        dataframe = dataframe.dropna(axis=1, how="all")
        official_counts["ibge_total_rows"] = int(dataframe.shape[0])
    if sidra_paths:
        payload = json.loads(sidra_paths[0].read_text(encoding="utf-8"))
        official_counts["sidra_rows"] = len(payload) - 1 if isinstance(payload, list) else None
    if popsvs_paths:
        official_counts["popsvs_rows"] = zip_dbf_record_count(popsvs_paths[0])
    return {
        "raw_paths": [str(path.relative_to(root)) for path in raw_paths],
        "raw_file_count": len(raw_paths),
        "raw_readable": bool(raw_paths) and all(probe["readable"] for probe in probes),
        "raw_probes": probes,
        "underlying_official_counts": official_counts,
    }


def bronze_usecols(spec: DatasetAuditSpec, header: list[str]) -> list[str]:
    if spec.internal_dataset == "sih":
        return header
    usecols = set(spec.critical_fields)
    usecols.update(spec.municipality_fields)
    usecols.update(DATE_COLUMNS_BY_DATASET.get(spec.internal_dataset or "", ()))
    usecols.update(spec.duplicate_key_fields)
    if spec.duplicate_group_field:
        usecols.add(spec.duplicate_group_field)
    if spec.month_field:
        usecols.add(spec.month_field)
    if spec.year_field:
        usecols.add(spec.year_field)
    if spec.internal_dataset == "inmet":
        usecols.update({"BRONZE_SOURCE_MEMBER", "CODIGO_WMO", "UF", "HORA_UTC", *INMET_FLAG_FIELDS})
    if spec.internal_dataset == "era5_land":
        usecols.update({"UF", ERA5_NO_USABLE_DATA_FIELD})
    return [column for column in header if column in usecols]


def schema_fields_for_spec(spec: DatasetAuditSpec) -> list[str]:
    fields = list(spec.critical_fields)
    fields.extend(spec.duplicate_key_fields)
    if spec.duplicate_group_field:
        fields.append(spec.duplicate_group_field)
    if spec.daily_date_field:
        fields.append(spec.daily_date_field)
    if spec.month_field:
        fields.append(spec.month_field)
    if spec.year_field:
        fields.append(spec.year_field)
    if spec.internal_dataset == "inmet":
        fields.extend(INMET_FLAG_FIELDS)
    if spec.internal_dataset == "era5_land":
        fields.append(ERA5_NO_USABLE_DATA_FIELD)
    return list(dict.fromkeys(fields))


def analyze_bronze_csv(
    root: Path,
    spec: DatasetAuditSpec,
    year: int,
    bronze_path: Path,
    official_registry: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    missing_counts = {field: 0 for field in spec.critical_fields}
    invalid_municipality_counts = {field: 0 for field in spec.municipality_fields}
    parseable_date_counts = {field: 0 for field in DATE_COLUMNS_BY_DATASET.get(spec.internal_dataset or "", ())}
    impossible_date_counts = {field: 0 for field in DATE_COLUMNS_BY_DATASET.get(spec.internal_dataset or "", ())}
    observed_dates: set[str] = set()
    observed_months: set[int] = set()
    observed_municipalities: set[str] = set()
    duplicate_count = 0
    current_group: str | None = None
    seen_keys: set[tuple[str, ...]] = set()
    unique_source_members: set[str] = set()
    unique_wmo: set[str] = set()
    invalid_hour_count = 0
    invalid_uf_count = 0
    invalid_month_count = 0
    invalid_year_count = 0
    inmet_flag_missing_counts = {field: 0 for field in INMET_FLAG_FIELDS}
    inmet_flag_mismatch_counts = {
        INMET_MISSING_TEMP_OFFICIAL_FIELD: 0,
        INMET_MISSING_HUMIDITY_OFFICIAL_FIELD: 0,
        INMET_ERA5_FALLBACK_FLAG_FIELD: 0,
    }
    era5_flag_missing_counts = {ERA5_NO_USABLE_DATA_FIELD: 0}
    era5_flag_mismatch_counts = {ERA5_NO_USABLE_DATA_FIELD: 0}
    era5_unflagged_climate_missing_count = 0
    sih_repeated_n_aih_rows = 0
    sih_repeated_n_aih_group_keys: set[tuple[str, str]] = set()
    sih_literal_duplicate_rows = 0
    sih_current_source: str | None = None
    sih_seen_n_aih: set[str] = set()
    sih_seen_row_hashes: set[int] = set()
    row_count = 0
    with gzip.open(bronze_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        header = reader.fieldnames or []
    missing_schema_fields = [field for field in schema_fields_for_spec(spec) if field not in header]
    usecols = bronze_usecols(spec, header)
    chunk_iter = pd.read_csv(
        bronze_path,
        sep=";",
        dtype=str,
        compression="gzip",
        usecols=usecols,
        keep_default_na=False,
        chunksize=100_000,
        low_memory=False,
    )
    for chunk in chunk_iter:
        chunk = chunk.fillna("")
        row_count += len(chunk)

        for field in spec.critical_fields:
            if field in chunk.columns:
                missing_counts[field] += int(chunk[field].str.strip().eq("").sum())

        if spec.duplicate_key_fields:
            duplicate_columns = [spec.duplicate_group_field] + list(spec.duplicate_key_fields) if spec.duplicate_group_field else list(spec.duplicate_key_fields)
            duplicate_frame = chunk[duplicate_columns]
            for values in duplicate_frame.itertuples(index=False, name=None):
                if spec.duplicate_group_field:
                    group_value = str(values[0]).strip()
                    key = tuple(str(value).strip() for value in values[1:])
                else:
                    group_value = None
                    key = tuple(str(value).strip() for value in values)
                if group_value != current_group:
                    current_group = group_value
                    seen_keys.clear()
                if key in seen_keys:
                    duplicate_count += 1
                else:
                    seen_keys.add(key)

        if spec.daily_date_field and spec.daily_date_field in chunk.columns:
            daily_series = chunk[spec.daily_date_field].str.strip()
            daily_present = daily_series.ne("")
            daily_parsed = pd.to_datetime(daily_series.where(daily_present, None), format="%Y-%m-%d", errors="coerce")
            parseable_date_counts[spec.daily_date_field] += int(daily_parsed.notna().sum())
            impossible_date_counts[spec.daily_date_field] += int((daily_present & daily_parsed.isna()).sum())
            if daily_parsed.notna().any():
                parsed_dates = daily_parsed.dropna()
                observed_dates.update(parsed_dates.dt.strftime("%Y-%m-%d").unique().tolist())
                observed_months.update(int(value) for value in parsed_dates.dt.month.unique().tolist())
                impossible_date_counts[spec.daily_date_field] += int((parsed_dates.dt.year != year).sum())

        for field in DATE_COLUMNS_BY_DATASET.get(spec.internal_dataset or "", ()):
            if field == spec.daily_date_field or field not in chunk.columns:
                continue
            series = chunk[field].str.strip()
            present = series.ne("")
            parsed = pd.to_datetime(series.where(present, None), format="%Y-%m-%d", errors="coerce")
            parseable_date_counts[field] += int(parsed.notna().sum())
            impossible_date_counts[field] += int((present & parsed.isna()).sum())

        if spec.month_field and spec.month_field in chunk.columns:
            month_values = pd.to_numeric(chunk[spec.month_field].str.strip(), errors="coerce")
            valid_month_mask = month_values.isin(list(official_year_months()))
            invalid_month_count += int((chunk[spec.month_field].str.strip().ne("") & ~valid_month_mask).sum())
            observed_months.update(int(value) for value in month_values[valid_month_mask].dropna().unique().tolist())

        if spec.year_field and spec.year_field in chunk.columns:
            year_values = pd.to_numeric(chunk[spec.year_field].str.strip(), errors="coerce")
            invalid_year_count += int((chunk[spec.year_field].str.strip().ne("") & (year_values != year)).sum())

        if spec.internal_dataset == "inmet":
            for field in INMET_FLAG_FIELDS:
                if field in chunk.columns:
                    inmet_flag_missing_counts[field] += int(chunk[field].str.strip().eq("").sum())
            if "BRONZE_SOURCE_MEMBER" in chunk.columns:
                unique_source_members.update(value for value in chunk["BRONZE_SOURCE_MEMBER"].str.strip().unique().tolist() if value)
            if "CODIGO_WMO" in chunk.columns:
                unique_wmo.update(value for value in chunk["CODIGO_WMO"].str.strip().unique().tolist() if value)
            if "UF" in chunk.columns:
                uf_series = chunk["UF"].str.strip()
                invalid_uf_count += int((uf_series.ne("") & ~uf_series.isin(VALID_UFS)).sum())
            if "HORA_UTC" in chunk.columns:
                hour_series = chunk["HORA_UTC"].str.strip()
                invalid_hour_count += int((hour_series.ne("") & ~hour_series.str.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d")).sum())
            if INMET_TEMPERATURE_FIELD in chunk.columns and INMET_MISSING_TEMP_OFFICIAL_FIELD in chunk.columns:
                temp_missing = chunk[INMET_TEMPERATURE_FIELD].str.strip().eq("")
                temp_expected = temp_missing.map({True: "1", False: "0"})
                temp_flag = chunk[INMET_MISSING_TEMP_OFFICIAL_FIELD].str.strip()
                inmet_flag_mismatch_counts[INMET_MISSING_TEMP_OFFICIAL_FIELD] += int((temp_flag != temp_expected).sum())
            if INMET_HUMIDITY_FIELD in chunk.columns and INMET_MISSING_HUMIDITY_OFFICIAL_FIELD in chunk.columns:
                humidity_missing = chunk[INMET_HUMIDITY_FIELD].str.strip().eq("")
                humidity_expected = humidity_missing.map({True: "1", False: "0"})
                humidity_flag = chunk[INMET_MISSING_HUMIDITY_OFFICIAL_FIELD].str.strip()
                inmet_flag_mismatch_counts[INMET_MISSING_HUMIDITY_OFFICIAL_FIELD] += int((humidity_flag != humidity_expected).sum())
            if (
                INMET_TEMPERATURE_FIELD in chunk.columns
                and INMET_HUMIDITY_FIELD in chunk.columns
                and INMET_ERA5_FALLBACK_FLAG_FIELD in chunk.columns
            ):
                fallback_expected = (
                    chunk[INMET_TEMPERATURE_FIELD].str.strip().eq("")
                    | chunk[INMET_HUMIDITY_FIELD].str.strip().eq("")
                ).map({True: "1", False: "0"})
                fallback_flag = chunk[INMET_ERA5_FALLBACK_FLAG_FIELD].str.strip()
                inmet_flag_mismatch_counts[INMET_ERA5_FALLBACK_FLAG_FIELD] += int((fallback_flag != fallback_expected).sum())

        if spec.internal_dataset == "era5_land" and "UF" in chunk.columns:
            uf_series = chunk["UF"].str.strip()
            invalid_uf_count += int((uf_series.ne("") & ~uf_series.isin(VALID_UFS)).sum())
            if ERA5_NO_USABLE_DATA_FIELD in chunk.columns:
                era5_flag_missing_counts[ERA5_NO_USABLE_DATA_FIELD] += int(chunk[ERA5_NO_USABLE_DATA_FIELD].str.strip().eq("").sum())
            if {"N_OBS", ERA5_NO_USABLE_DATA_FIELD}.issubset(chunk.columns):
                n_obs_numeric = pd.to_numeric(chunk["N_OBS"].str.strip(), errors="coerce")
                expected_flag = n_obs_numeric.eq(0).map({True: "1", False: "0"})
                observed_flag = chunk[ERA5_NO_USABLE_DATA_FIELD].str.strip()
                era5_flag_mismatch_counts[ERA5_NO_USABLE_DATA_FIELD] += int((observed_flag != expected_flag).sum())
            if {"TMAX_C", "RH_MEAN_PCT", "N_OBS", ERA5_NO_USABLE_DATA_FIELD}.issubset(chunk.columns):
                no_usable = chunk[ERA5_NO_USABLE_DATA_FIELD].str.strip().eq("1")
                climate_missing = chunk["TMAX_C"].str.strip().eq("") | chunk["RH_MEAN_PCT"].str.strip().eq("")
                zero_obs = pd.to_numeric(chunk["N_OBS"].str.strip(), errors="coerce").eq(0)
                era5_unflagged_climate_missing_count += int((climate_missing & ~(zero_obs & no_usable)).sum())

        if spec.internal_dataset == "sih" and {"BRONZE_SOURCE_FILE", "N_AIH"}.issubset(chunk.columns):
            row_hashes = pd.util.hash_pandas_object(chunk[header], index=False).to_numpy(dtype="uint64", copy=False)
            source_files = chunk["BRONZE_SOURCE_FILE"].str.strip().tolist()
            n_aih_values = chunk["N_AIH"].str.strip().tolist()
            for source_file, n_aih_value, row_hash in zip(source_files, n_aih_values, row_hashes):
                if source_file != sih_current_source:
                    sih_current_source = source_file
                    sih_seen_n_aih.clear()
                    sih_seen_row_hashes.clear()
                if n_aih_value:
                    if n_aih_value in sih_seen_n_aih:
                        sih_repeated_n_aih_rows += 1
                        sih_repeated_n_aih_group_keys.add((source_file, n_aih_value))
                    else:
                        sih_seen_n_aih.add(n_aih_value)
                row_hash_int = int(row_hash)
                if row_hash_int in sih_seen_row_hashes:
                    sih_literal_duplicate_rows += 1
                else:
                    sih_seen_row_hashes.add(row_hash_int)

        if spec.municipality_fields and spec.municipality_code_length is not None:
            if spec.conceptual_only and spec.dataset in {"bronze_pop_under5", "bronze_pop_wra"}:
                official_set = population_derivative_expected_codes(root, year)
            else:
                official_set = official_registry[year][f"code{spec.municipality_code_length}"]
            zero_code = "0" * spec.municipality_code_length
            for field in spec.municipality_fields:
                if field not in chunk.columns:
                    continue
                series = chunk[field].str.strip()
                present = series.ne("")
                normalized = series.str.replace(r"\D", "", regex=True)
                valid_length = normalized.str.len().between(1, spec.municipality_code_length)
                normalized = normalized.where(valid_length, "")
                normalized = normalized.where(normalized.ne(""), "").str.zfill(spec.municipality_code_length)
                valid_mask = present & normalized.ne(zero_code) & normalized.isin(official_set)
                invalid_municipality_counts[field] += int((present & ~valid_mask).sum())
                if field == spec.coverage_field:
                    observed_municipalities.update(normalized[valid_mask].unique().tolist())

    return {
        "header": header,
        "missing_schema_fields": missing_schema_fields,
        "row_count": row_count,
        "missing_counts": missing_counts,
        "invalid_municipality_counts": invalid_municipality_counts,
        "parseable_date_counts": parseable_date_counts,
        "impossible_date_counts": impossible_date_counts,
        "observed_dates": sorted(observed_dates),
        "observed_months": sorted(observed_months),
        "observed_municipalities": sorted(observed_municipalities),
        "duplicate_count": duplicate_count,
        "unique_source_member_count": len(unique_source_members),
        "unique_wmo_count": len({value for value in unique_wmo if value}),
        "invalid_hour_count": invalid_hour_count,
        "invalid_uf_count": invalid_uf_count,
        "invalid_month_count": invalid_month_count,
        "invalid_year_count": invalid_year_count,
        "inmet_flag_missing_counts": inmet_flag_missing_counts,
        "inmet_flag_mismatch_counts": inmet_flag_mismatch_counts,
        "era5_flag_missing_counts": era5_flag_missing_counts,
        "era5_flag_mismatch_counts": era5_flag_mismatch_counts,
        "era5_unflagged_climate_missing_count": era5_unflagged_climate_missing_count,
        "repeated_n_aih_rows": sih_repeated_n_aih_rows,
        "repeated_n_aih_groups": len(sih_repeated_n_aih_group_keys),
        "literal_duplicate_rows": sih_literal_duplicate_rows,
    }


def expected_bronze_path(root: Path, spec: DatasetAuditSpec, year: int) -> Path | None:
    if spec.internal_dataset is None:
        return None
    return bronze_output_path(root, spec.internal_dataset, year)


def safe_rate(numerator: int, denominator: int) -> float:
    return round((numerator / denominator) if denominator else 0.0, 6)


def summarize_missing_counts(missing_counts: dict[str, int], row_count: int) -> dict[str, dict[str, Any]]:
    return {
        field: {
            "missing_rows": value,
            "missing_rate": safe_rate(value, row_count),
        }
        for field, value in missing_counts.items()
    }


def summarize_invalid_counts(invalid_counts: dict[str, int], row_count: int) -> dict[str, dict[str, Any]]:
    return {
        field: {
            "invalid_rows": value,
            "invalid_rate": safe_rate(value, row_count),
        }
        for field, value in invalid_counts.items()
    }


def build_blockers(validation_flags: dict[str, bool], details: dict[str, Any]) -> tuple[str, str]:
    blockers: list[str] = []
    corrective_actions: list[str] = []
    if not validation_flags["bronze_exists"]:
        blockers.append("bronze ausente")
        corrective_actions.append("materializar o bronze canônico antes da promoção para silver")
    if not validation_flags["bronze_readable"]:
        blockers.append("bronze ilegível")
        corrective_actions.append("regerar o arquivo bronze e revalidar a integridade física")
    if not validation_flags["raw_readable"]:
        blockers.append("raw ilegível ou ausente")
        corrective_actions.append("corrigir ou rebaixar os arquivos raw oficiais antes de qualquer promoção")
    if not validation_flags["manifest_ok"]:
        blockers.append("manifest inconsistente")
        corrective_actions.append("reconstruir manifests e confirmar entry_path/year/row_count")
    if not validation_flags["schema_ok"]:
        blockers.append("schema/chaves básicas falharam")
        corrective_actions.append("corrigir cabeçalho, colunas obrigatórias e chave técnica duplicada")
    if not validation_flags["municipality_ok"]:
        blockers.append("código municipal inválido")
        corrective_actions.append("normalizar códigos IBGE/DATASUS e remover sentinelas inválidas")
    if not validation_flags["date_ok"]:
        blockers.append("datas inválidas ou fora do ano")
        corrective_actions.append("corrigir parsing de datas e eliminar datas impossíveis")
    if not validation_flags["temporal_ok"]:
        blockers.append("completude temporal falhou")
        corrective_actions.append("fechar gaps de dia/mês/ano antes da promoção para silver")
    if not validation_flags["critical_missing_ok"]:
        blockers.append("missing em variáveis críticas")
        corrective_actions.append("sanear campos críticos vazios ou documentar/excluir o ano")
    if not validation_flags["raw_vs_bronze_ok"]:
        blockers.append("reconciliação raw vs bronze falhou")
        corrective_actions.append("recontar raw, bronze e expected rows e refazer a promoção do ano afetado")
    if not validation_flags["external_ok"]:
        blockers.append("validação externa falhou")
        corrective_actions.append("alinhar calendário oficial, cobertura municipal e codebook oficial")
    if not blockers:
        return "", ""
    return " | ".join(blockers), " | ".join(dict.fromkeys(corrective_actions))


def audit_existing_dataset_year(
    root: Path,
    spec: DatasetAuditSpec,
    year: int,
    official_registry: dict[int, dict[str, Any]],
    benchmarks: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    raw_status = raw_status_for_existing_dataset(root, spec, year, benchmarks)
    bronze_path = expected_bronze_path(root, spec, year)
    assert bronze_path is not None
    bronze_exists = bronze_path.exists()
    bronze_readable = False
    bronze_probe = probe_path(bronze_path)
    analysis: dict[str, Any] = {
        "row_count": 0,
        "missing_counts": {field: 0 for field in spec.critical_fields},
        "invalid_municipality_counts": {field: 0 for field in spec.municipality_fields},
        "parseable_date_counts": {},
        "impossible_date_counts": {},
        "observed_dates": [],
        "observed_months": [],
        "observed_municipalities": [],
        "duplicate_count": 0,
        "unique_source_member_count": 0,
        "unique_wmo_count": 0,
        "invalid_hour_count": 0,
        "invalid_uf_count": 0,
        "invalid_month_count": 0,
        "invalid_year_count": 0,
        "missing_schema_fields": list(spec.critical_fields),
        "header": [],
    }
    if bronze_exists:
        with gzip.open(bronze_path, "rt", encoding="utf-8", newline="") as handle:
            bronze_readable = bool(handle.readline())
        analysis = analyze_bronze_csv(root, spec, year, bronze_path, official_registry)

    manifest_name = "bronze_population" if spec.internal_dataset in {"ibge_total", "ibge_sidra", "datasus_popsvs"} else (
        "bronze_climate" if spec.internal_dataset in {"inmet", "era5_land"} else f"bronze_{spec.internal_dataset}"
    )
    manifest_path = root / "artifacts" / "manifests" / f"{manifest_name}.manifest.json"
    manifest_entry = manifest_entry_for_path(root, manifest_path, bronze_path) if bronze_exists else None
    manifest_ok = bool(
        bronze_exists
        and manifest_entry
        and manifest_entry.get("year") == year
        and manifest_entry.get("row_count") == analysis["row_count"]
    )

    official_dates = set(official_calendar_dates(year))
    observed_dates = set(analysis["observed_dates"])
    missing_dates = sorted(official_dates - observed_dates) if spec.daily_date_field else []
    extra_dates = sorted(observed_dates - official_dates) if spec.daily_date_field else []
    missing_months = sorted(official_year_months() - set(analysis["observed_months"]))
    era5_lookup_registry: dict[str, Any] | None = None
    expected_municipality_count = 0
    if spec.municipality_code_length is not None:
        expected_municipality_count = official_registry[year][f"expected_municipality_count_{spec.municipality_code_length}"]
    expected_rows = raw_status.get("raw_row_count")
    if spec.internal_dataset == "era5_land":
        era5_lookup_registry = build_era5_lookup_registry(root)
        if era5_lookup_registry["expected_municipality_count_7"] > 0:
            expected_municipality_count = era5_lookup_registry["expected_municipality_count_7"]
        expected_rows = len(official_dates) * expected_municipality_count

    critical_missing_summary = summarize_missing_counts(analysis["missing_counts"], analysis["row_count"])
    invalid_municipality_summary = summarize_invalid_counts(analysis["invalid_municipality_counts"], analysis["row_count"])
    coverage_invalid_count = analysis["invalid_municipality_counts"].get(spec.coverage_field, 0) if spec.coverage_field else 0
    coverage_invalid_rate = safe_rate(coverage_invalid_count, analysis["row_count"])
    coverage_observed = len(analysis["observed_municipalities"])
    secondary_invalid_municipality_counts = {
        field: count
        for field, count in analysis["invalid_municipality_counts"].items()
        if field != spec.coverage_field and count > 0
    }

    municipality_ok = (
        coverage_invalid_count == 0 or coverage_invalid_rate <= MUNICIPALITY_INVALID_BLOCK_RATE
        if spec.coverage_field
        else True
    )
    date_ok = (
        all(value == 0 for value in analysis["impossible_date_counts"].values())
        and analysis["invalid_month_count"] == 0
        and analysis["invalid_year_count"] == 0
        and analysis["invalid_hour_count"] == 0
    )
    temporal_ok = True
    if spec.daily_date_field:
        temporal_ok = not missing_dates and not extra_dates
    if spec.month_field:
        temporal_ok = temporal_ok and not missing_months
    if spec.internal_dataset == "era5_land":
        temporal_ok = temporal_ok and raw_status["boundary_support_present"] and not raw_status["missing_months"]

    critical_missing_ok = all(
        critical_missing_summary[field]["missing_rows"] == 0
        for field in spec.critical_fields
        if field in set(spec.municipality_fields) | ({spec.daily_date_field} if spec.daily_date_field else set()) | set(spec.duplicate_key_fields)
    )
    if spec.internal_dataset == "era5_land":
        critical_missing_ok = critical_missing_ok and critical_missing_summary["N_OBS"]["missing_rows"] == 0
        critical_missing_ok = critical_missing_ok and analysis["era5_flag_missing_counts"].get(ERA5_NO_USABLE_DATA_FIELD, 0) == 0
        critical_missing_ok = critical_missing_ok and analysis["era5_flag_mismatch_counts"].get(ERA5_NO_USABLE_DATA_FIELD, 0) == 0
        critical_missing_ok = critical_missing_ok and analysis["era5_unflagged_climate_missing_count"] == 0
    if spec.internal_dataset == "inmet":
        critical_missing_ok = critical_missing_ok and all(
            critical_missing_summary[field]["missing_rows"] == 0 for field in ("CODIGO_WMO", "UF")
        )
        critical_missing_ok = critical_missing_ok and all(
            analysis["inmet_flag_missing_counts"].get(field, 0) == 0 for field in INMET_FLAG_FIELDS
        )
        critical_missing_ok = critical_missing_ok and all(
            analysis["inmet_flag_mismatch_counts"].get(field, 0) == 0 for field in INMET_FLAG_FIELDS
        )

    raw_vs_bronze_ok = bronze_exists and raw_status["raw_readable"]
    if expected_rows is not None:
        raw_vs_bronze_ok = raw_vs_bronze_ok and expected_rows == analysis["row_count"]
    if spec.internal_dataset == "inmet":
        raw_vs_bronze_ok = raw_vs_bronze_ok and raw_status.get("raw_member_count") == analysis["unique_source_member_count"]
    if spec.internal_dataset == "era5_land":
        raw_vs_bronze_ok = raw_vs_bronze_ok and raw_status["boundary_support_present"] and not raw_status["missing_months"] and analysis["row_count"] == expected_rows

    schema_ok = bronze_exists and not analysis["missing_schema_fields"]
    if spec.internal_dataset != "sih":
        schema_ok = schema_ok and analysis["duplicate_count"] == 0
    external_ok = True
    if spec.internal_dataset in {"sim", "sinasc", "sih", "stillbirth"}:
        external_ok = municipality_ok
    if spec.internal_dataset == "inmet":
        external_ok = analysis["invalid_uf_count"] == 0 and analysis["invalid_hour_count"] == 0 and raw_status.get("raw_member_count") == analysis["unique_source_member_count"]
    if spec.internal_dataset == "era5_land":
        external_ok = (
            municipality_ok
            and analysis["invalid_uf_count"] == 0
            and analysis["row_count"] == expected_rows
            and coverage_observed == expected_municipality_count
        )

    validation_flags = {
        "bronze_exists": bronze_exists,
        "bronze_readable": bronze_readable,
        "raw_readable": raw_status["raw_readable"],
        "manifest_ok": manifest_ok,
        "schema_ok": schema_ok,
        "municipality_ok": municipality_ok,
        "date_ok": date_ok,
        "temporal_ok": temporal_ok,
        "critical_missing_ok": critical_missing_ok,
        "raw_vs_bronze_ok": raw_vs_bronze_ok,
        "external_ok": external_ok,
    }
    blocker, corrective_action = build_blockers(validation_flags, analysis)
    status = "PASS" if all(validation_flags.values()) else "FAIL"
    ready_for_silver = "sim" if status == "PASS" else "não"

    return {
        "dataset": spec.dataset,
        "label": spec.label,
        "year": year,
        "status": status,
        "ready_for_silver": ready_for_silver,
        "blocker": blocker,
        "corrective_action": corrective_action,
        "bronze_path": str(bronze_path.relative_to(root)) if bronze_exists else "",
        "bronze_probe": bronze_probe,
        "manifest_path": str(manifest_path.relative_to(root)),
        "manifest_entry_path": manifest_entry.get("path") if manifest_entry else None,
        "raw_paths": raw_status.get("raw_paths", []),
        "raw_file_count": raw_status.get("raw_file_count", 0),
        "raw_row_count": raw_status.get("raw_row_count"),
        "bronze_row_count": analysis["row_count"],
        "expected_rows": expected_rows,
        "expected_municipality_count": expected_municipality_count,
        "observed_municipality_count": coverage_observed,
        "missing_dates_count": len(missing_dates),
        "missing_months_count": len(missing_months),
        "missing_dates_sample": missing_dates[:10],
        "extra_dates_sample": extra_dates[:10],
        "missing_months": missing_months,
        "validation_flags": validation_flags,
        "critical_missing_summary": critical_missing_summary,
        "invalid_municipality_summary": invalid_municipality_summary,
        "secondary_invalid_municipality_counts": secondary_invalid_municipality_counts,
        "coverage_invalid_rate": coverage_invalid_rate,
        "impossible_date_counts": analysis["impossible_date_counts"],
        "duplicate_count": analysis["duplicate_count"],
        "repeated_n_aih_rows": analysis.get("repeated_n_aih_rows", 0),
        "repeated_n_aih_groups": analysis.get("repeated_n_aih_groups", 0),
        "literal_duplicate_rows": analysis.get("literal_duplicate_rows", 0),
        "inmet_flag_missing_counts": analysis.get("inmet_flag_missing_counts", {}),
        "inmet_flag_mismatch_counts": analysis.get("inmet_flag_mismatch_counts", {}),
        "era5_flag_missing_counts": analysis.get("era5_flag_missing_counts", {}),
        "era5_flag_mismatch_counts": analysis.get("era5_flag_mismatch_counts", {}),
        "era5_unflagged_climate_missing_count": analysis.get("era5_unflagged_climate_missing_count", 0),
        "external_benchmarks": {
            "official_calendar_days": len(official_dates),
            "official_registry_source": official_registry[year]["source_path"],
            "official_municipality_count_6": official_registry[year]["expected_municipality_count_6"],
            "official_municipality_count_7": official_registry[year]["expected_municipality_count_7"],
            "era5_lookup_source": era5_lookup_registry["source_path"] if era5_lookup_registry else None,
            "era5_lookup_municipality_count_7": (
                era5_lookup_registry["expected_municipality_count_7"] if era5_lookup_registry else None
            ),
            "era5_lookup_excluded_codes": era5_lookup_registry["excluded_codes"] if era5_lookup_registry else [],
            "raw_member_count": raw_status.get("raw_member_count"),
            "unique_source_member_count": analysis["unique_source_member_count"],
            "raw_row_count_source": raw_status.get("raw_row_count_source"),
            "benchmark_log_path": raw_status.get("benchmark_log_path"),
        },
        "raw_status": raw_status,
    }


def audit_conceptual_population_year(
    root: Path,
    spec: DatasetAuditSpec,
    year: int,
    official_registry: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    raw_status = raw_status_for_conceptual_population(root, year)
    expected_municipality_count = len(population_derivative_expected_codes(root, year))
    bronze_path = root / spec.expected_bronze_path_hint.format(year=year) if spec.expected_bronze_path_hint else None
    bronze_exists = bronze_path is not None and bronze_path.exists()
    bronze_readable = False
    analysis = {
        "row_count": 0,
        "missing_counts": {field: 0 for field in spec.critical_fields},
        "invalid_municipality_counts": {field: 0 for field in spec.municipality_fields},
        "parseable_date_counts": {},
        "impossible_date_counts": {},
        "observed_dates": [],
        "observed_months": [],
        "observed_municipalities": [],
        "duplicate_count": 0,
        "invalid_month_count": 0,
        "invalid_year_count": 0,
        "missing_schema_fields": list(spec.critical_fields),
        "header": [],
    }
    if bronze_exists and bronze_path is not None:
        with gzip.open(bronze_path, "rt", encoding="utf-8", newline="") as handle:
            bronze_readable = bool(handle.readline())
        analysis = analyze_bronze_csv(root, spec, year, bronze_path, official_registry)

    manifest_path = root / "artifacts" / "manifests" / "bronze_population.manifest.json"
    manifest_entry = manifest_entry_for_path(root, manifest_path, bronze_path) if bronze_exists and bronze_path is not None else None
    manifest_ok = bool(
        bronze_exists
        and manifest_entry
        and manifest_entry.get("year") == year
        and manifest_entry.get("row_count") == analysis["row_count"]
    )
    critical_missing_summary = summarize_missing_counts(analysis["missing_counts"], analysis["row_count"])
    invalid_municipality_summary = summarize_invalid_counts(analysis["invalid_municipality_counts"], analysis["row_count"])
    coverage_invalid_count = analysis["invalid_municipality_counts"].get(spec.coverage_field, 0) if spec.coverage_field else 0
    observed_municipality_count = len(analysis["observed_municipalities"])
    municipality_ok = (
        bronze_exists
        and coverage_invalid_count == 0
        and observed_municipality_count == expected_municipality_count
    )
    date_ok = all(value == 0 for value in analysis["impossible_date_counts"].values()) and analysis["invalid_year_count"] == 0
    temporal_ok = bronze_exists and analysis["row_count"] == expected_municipality_count and analysis["invalid_year_count"] == 0
    critical_missing_ok = bronze_exists and all(
        critical_missing_summary[field]["missing_rows"] == 0 for field in spec.critical_fields
    )
    schema_ok = bronze_exists and not analysis["missing_schema_fields"] and analysis["duplicate_count"] == 0
    raw_vs_bronze_ok = bronze_exists and raw_status["raw_readable"] and analysis["duplicate_count"] == 0
    if bronze_exists:
        raw_vs_bronze_ok = raw_vs_bronze_ok and analysis["row_count"] == observed_municipality_count

    rows_exceeding_ibge_total = 0
    rows_missing_ibge_total_benchmark = 0
    ibge_total_benchmark, ibge_total_summary = load_ibge_total_benchmark_frame(root, year)
    if bronze_exists and bronze_path is not None and analysis["row_count"] > 0:
        population_field = spec.critical_fields[-1]
        observed_frame = pd.read_csv(
            bronze_path,
            sep=";",
            dtype=str,
            compression="gzip",
            usecols=["COD_MUNICIPIO", population_field],
            keep_default_na=False,
            low_memory=False,
        )
        observed_frame[population_field] = pd.to_numeric(observed_frame[population_field], errors="coerce")
        benchmark_join = observed_frame.merge(ibge_total_benchmark, on="COD_MUNICIPIO", how="left")
        rows_exceeding_ibge_total = int(
            (
                benchmark_join["IBGE_TOTAL_POP"].notna()
                & benchmark_join[population_field].notna()
                & (benchmark_join[population_field] > benchmark_join["IBGE_TOTAL_POP"])
            ).sum()
        )
        rows_missing_ibge_total_benchmark = int(benchmark_join["IBGE_TOTAL_POP"].isna().sum())
    sidra_benchmark_path = bronze_output_path(root, "ibge_sidra", year)
    sidra_exists = sidra_benchmark_path.exists()
    external_ok = bronze_exists and population_external_benchmarks_ready(ibge_total_summary, sidra_exists)

    validation_flags = {
        "bronze_exists": bronze_exists,
        "bronze_readable": bronze_readable,
        "raw_readable": raw_status["raw_readable"],
        "manifest_ok": manifest_ok,
        "schema_ok": schema_ok,
        "municipality_ok": municipality_ok,
        "date_ok": date_ok,
        "temporal_ok": temporal_ok,
        "critical_missing_ok": critical_missing_ok,
        "raw_vs_bronze_ok": raw_vs_bronze_ok,
        "external_ok": external_ok,
    }
    blocker, corrective_action = build_blockers(
        validation_flags,
        {
            "analysis": analysis,
            "expected_rows": expected_municipality_count,
            "observed_municipality_count": observed_municipality_count,
        },
    )
    status = "PASS" if all(validation_flags.values()) else "FAIL"
    return {
        "dataset": spec.dataset,
        "label": spec.label,
        "year": year,
        "status": status,
        "ready_for_silver": "sim" if status == "PASS" else "não",
        "blocker": blocker,
        "corrective_action": corrective_action,
        "bronze_path": str(bronze_path.relative_to(root)) if bronze_path is not None else "",
        "bronze_probe": {
            "path": str(bronze_path.relative_to(root)) if bronze_path is not None else "",
            "exists": bronze_exists,
            "readable": bronze_readable,
        },
        "manifest_path": str(manifest_path.relative_to(root)),
        "manifest_entry_path": manifest_entry.get("path") if manifest_entry else None,
        "raw_paths": raw_status.get("raw_paths", []),
        "raw_file_count": raw_status.get("raw_file_count", 0),
        "raw_row_count": raw_status["underlying_official_counts"].get("popsvs_rows"),
        "bronze_row_count": analysis["row_count"],
        "expected_rows": expected_municipality_count,
        "expected_municipality_count": expected_municipality_count,
        "observed_municipality_count": observed_municipality_count,
        "missing_dates_count": 0,
        "missing_months_count": 0,
        "missing_dates_sample": [],
        "extra_dates_sample": [],
        "missing_months": [],
        "validation_flags": validation_flags,
        "critical_missing_summary": critical_missing_summary,
        "invalid_municipality_summary": invalid_municipality_summary,
        "impossible_date_counts": analysis["impossible_date_counts"],
        "duplicate_count": analysis["duplicate_count"],
        "external_benchmarks": {
            "official_registry_source": official_registry[year]["source_path"],
            "official_municipality_count_7": expected_municipality_count,
            "underlying_official_counts": raw_status["underlying_official_counts"],
            "ibge_total_benchmark": {
                **ibge_total_summary,
                "rows_exceeding_total_population": rows_exceeding_ibge_total,
                "rows_missing_total_population_benchmark": rows_missing_ibge_total_benchmark,
            },
            "sidra_benchmark": {
                "path": str(sidra_benchmark_path.relative_to(root)),
                "exists": sidra_exists,
                "row_count": count_rows(sidra_benchmark_path) if sidra_exists else None,
                "used_only_for_external_validation": True,
            },
        },
        "raw_status": raw_status,
    }


def audit_dataset_year(
    root: Path,
    spec: DatasetAuditSpec,
    year: int,
    official_registry: dict[int, dict[str, Any]],
    benchmarks: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    if spec.conceptual_only:
        return audit_conceptual_population_year(root, spec, year, official_registry)
    return audit_existing_dataset_year(root, spec, year, official_registry, benchmarks)


def compact_row(report: dict[str, Any]) -> dict[str, Any]:
    flags = report["validation_flags"]
    return {
        "dataset": report["dataset"],
        "ano": report["year"],
        "status": report["status"],
        "bloqueio": report["blocker"],
        "acao_corretiva": report["corrective_action"],
        "pronto_para_promover": report["ready_for_silver"],
        "bronze_path": report["bronze_path"],
        "raw_file_count": report["raw_file_count"],
        "raw_row_count": report["raw_row_count"],
        "bronze_row_count": report["bronze_row_count"],
        "expected_rows": report["expected_rows"],
        "expected_municipality_count": report["expected_municipality_count"],
        "observed_municipality_count": report["observed_municipality_count"],
        "missing_dates_count": report["missing_dates_count"],
        "missing_months_count": report["missing_months_count"],
        "duplicates": report["duplicate_count"],
        "bronze_exists": flags["bronze_exists"],
        "bronze_readable": flags["bronze_readable"],
        "raw_readable": flags["raw_readable"],
        "manifest_ok": flags["manifest_ok"],
        "schema_ok": flags["schema_ok"],
        "municipality_ok": flags["municipality_ok"],
        "date_ok": flags["date_ok"],
        "temporal_ok": flags["temporal_ok"],
        "critical_missing_ok": flags["critical_missing_ok"],
        "raw_vs_bronze_ok": flags["raw_vs_bronze_ok"],
        "external_ok": flags["external_ok"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: Path, rows: list[dict[str, Any]], manifests: dict[str, str], root: Path, specs: list[DatasetAuditSpec]) -> None:
    ready = [row for row in rows if row["ready_for_silver"] == "sim"]
    failed = [row for row in rows if row["ready_for_silver"] != "sim"]
    lines = [
        "# Bronze Zero-Base Audit",
        "",
        f"- Generated at: {utc_now()}",
        f"- Repository root: {root}",
        f"- Scope: {', '.join(spec.dataset for spec in specs)}",
        f"- PASS rows: {len(ready)}",
        f"- FAIL rows: {len(failed)}",
        "",
        "## Manifests refreshed",
        "",
    ]
    for key, manifest_path in sorted(manifests.items()):
        lines.append(f"- `{key}`: `{manifest_path}`")
    lines.extend(["", "## Dataset-years liberados para silver", ""])
    if ready:
        for row in ready:
            lines.append(f"- {row['dataset']} {row['year']}")
    else:
        lines.append("- Nenhum dataset-ano foi liberado para silver.")
    lines.extend(["", "## Principais bloqueios", ""])
    if failed:
        for row in failed:
            lines.append(f"- {row['dataset']} {row['year']}: {row['blocker']}")
    else:
        lines.append("- Sem bloqueios.")
    lines.extend(["", "## Resumo por dataset", ""])
    dataset_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        dataset_rows.setdefault(row["dataset"], []).append(row)
    for dataset, reports in dataset_rows.items():
        statuses = ", ".join(f"{report['year']}={report['status']}" for report in sorted(reports, key=lambda item: item["year"]))
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append(f"- Status por ano: {statuses}")
        sample = next(iter(sorted(reports, key=lambda item: item["year"])))
        external = sample["external_benchmarks"]
        if external.get("official_registry_source"):
            lines.append(f"- Cobertura municipal oficial usada: `{external['official_registry_source']}`")
        if any(report["missing_dates_count"] for report in reports):
            lines.append(
                "- Gaps de datas detectados em: "
                + ", ".join(f"{report['year']} ({report['missing_dates_count']})" for report in reports if report["missing_dates_count"])
            )
        if any(report["duplicate_count"] for report in reports):
            lines.append(
                "- Duplicatas detectadas em: "
                + ", ".join(f"{report['year']} ({report['duplicate_count']})" for report in reports if report["duplicate_count"])
            )
        if any(report["bronze_path"] for report in reports):
            bronze_paths = sorted({report["bronze_path"] for report in reports if report["bronze_path"]})
            lines.append("- Bronze paths auditados:")
            for bronze_path in bronze_paths:
                lines.append(f"  - `{bronze_path}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_specs(
    include_datasets: list[str] | None = None,
    exclude_datasets: list[str] | None = None,
) -> list[DatasetAuditSpec]:
    include = set(include_datasets or [])
    exclude = set(exclude_datasets or [])
    selected = []
    for spec in AUDIT_SPECS:
        if include and spec.dataset not in include:
            continue
        if spec.dataset in exclude:
            continue
        selected.append(spec)
    if not selected:
        raise ValueError("No datasets selected for bronze audit")
    return selected


def run_bronze_zero_base_audit(
    root: Path,
    include_datasets: list[str] | None = None,
    exclude_datasets: list[str] | None = None,
) -> dict[str, Any]:
    config = load_project_config(root)
    specs = resolve_specs(include_datasets=include_datasets, exclude_datasets=exclude_datasets)
    manifests = build_manifests_for_audit(root, specs)
    official_registry = build_official_municipality_registry(root)
    benchmarks = load_bronze_promotion_benchmarks(root)
    rows = [audit_dataset_year(root, spec, year, official_registry, benchmarks) for spec in specs for year in YEARS]
    matrix_rows = [compact_row(row) for row in rows]
    matrix_path = root / "artifacts" / "status" / "bronze_audit_matrix.csv"
    report_path = root / "artifacts" / "status" / "bronze_audit_report.md"
    queue_path = root / "artifacts" / "status" / "promotion_queue_ready.csv"
    write_csv(matrix_path, matrix_rows)
    write_csv(
        queue_path,
        [
            {
                "dataset": row["dataset"],
                "ano": row["year"],
                "bronze_path": row["bronze_path"],
                "bronze_row_count": row["bronze_row_count"],
            }
            for row in rows
            if row["ready_for_silver"] == "sim"
        ],
    )
    write_report(report_path, rows, manifests, root, specs)
    payload = {
        "generated_at": utc_now(),
        "repository_root": str(root),
        "series_start_year": config["project"]["series_start_year"],
        "series_end_year": config["project"]["series_end_year"],
        "scope": [spec.dataset for spec in specs],
        "matrix_path": str(matrix_path.relative_to(root)),
        "report_path": str(report_path.relative_to(root)),
        "promotion_queue_path": str(queue_path.relative_to(root)),
        "ready_dataset_years": [{"dataset": row["dataset"], "year": row["year"]} for row in rows if row["ready_for_silver"] == "sim"],
        "failed_dataset_years": [{"dataset": row["dataset"], "year": row["year"]} for row in rows if row["ready_for_silver"] != "sim"],
        "rows": rows,
    }
    details_path = root / "artifacts" / "status" / "bronze_audit_details.json"
    details_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["details_path"] = str(details_path.relative_to(root))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_bronze_zero_base_audit.py")
    parser.add_argument("--dataset", action="append", dest="datasets", help="Dataset scope to include; repeat the flag to add more.")
    parser.add_argument("--exclude-dataset", action="append", dest="exclude_datasets", help="Dataset scope to exclude; repeat the flag to add more.")
    return parser
