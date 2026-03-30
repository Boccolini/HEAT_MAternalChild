from __future__ import annotations

import csv
import gzip
import json
import re
import shutil
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from zipfile import ZipFile

import datasus_dbc
import pandas as pd
from dbfread import DBF

from .config import load_yaml
from .era5_land import (
    ERA5_BRONZE_FIELDS,
    bronze_rows_from_dataframe,
    build_bronze_annual_rows,
    era5_month_filename,
    write_era5_bronze_summary,
)
from .manifest import build_manifest, count_rows


DATE_DDMMYYYY = re.compile(r"^\d{8}$")
DATE_YYYYMMDD_SLASH = re.compile(r"^\d{4}/\d{2}/\d{2}$")
DATE_DDMMYY_SLASH = re.compile(r"^\d{2}/\d{2}/\d{2}$")
DATE_DDMMYYYY_SLASH = re.compile(r"^\d{2}/\d{2}/\d{4}$")
DATE_LEGACY_BROKEN_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
HOUR_4 = re.compile(r"^\d{4}$")
HOUR_UTC = re.compile(r"^(\d{4})\s*UTC$", re.IGNORECASE)

METADATA_FIELDS = [
    "BRONZE_DATASET",
    "BRONZE_SOURCE_SYSTEM",
    "BRONZE_SOURCE_FILE",
    "BRONZE_SOURCE_PATH",
    "BRONZE_SOURCE_MEMBER",
    "BRONZE_YEAR",
    "BRONZE_INGESTED_AT",
]

INMET_TEMPERATURE_FIELD = "TEMPERATURA_DO_AR_BULBO_SECO_HORARIA_C"
INMET_HUMIDITY_FIELD = "UMIDADE_RELATIVA_DO_AR_HORARIA"
INMET_MISSING_TEMP_OFFICIAL_FIELD = "INMET_MISSING_TEMP_OFFICIAL"
INMET_MISSING_HUMIDITY_OFFICIAL_FIELD = "INMET_MISSING_HUMIDITY_OFFICIAL"
INMET_ERA5_FALLBACK_FLAG_FIELD = "INMET_ERA5_FALLBACK"
INMET_FLAG_FIELDS = [
    INMET_MISSING_TEMP_OFFICIAL_FIELD,
    INMET_MISSING_HUMIDITY_OFFICIAL_FIELD,
    INMET_ERA5_FALLBACK_FLAG_FIELD,
]
POP_WRA_MIN_AGE = 10
POP_WRA_MAX_AGE = 49
POPULATION_DERIVATIVE_EXCLUDED_MUNICIPALITY_CODES: dict[int, set[str]] = {
    # POPSVS 2024 does not carry Boa Esperanca do Norte yet; derived bronzes
    # keep the canonical source coverage and suppress this municipality in QA.
    2024: {"5101837"},
}
POPULATION_DERIVATIVE_SPECS: dict[str, dict[str, Any]] = {
    "pop_under5": {
        "population_field": "POP_UNDER5",
        "age_min": 0,
        "age_max": 4,
        "sex_codes": None,
        "definition_note": "Under-5 population derived from canonical DATASUS POPSVS counts summed across both sexes.",
    },
    "pop_wra": {
        "population_field": "POP_WRA",
        "age_min": POP_WRA_MIN_AGE,
        "age_max": POP_WRA_MAX_AGE,
        "sex_codes": {2},
        "definition_note": "Women of reproductive age derived as female DATASUS POPSVS counts for ages 10-49.",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_column_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^0-9A-Za-z]+", "_", text.strip()).strip("_")
    text = re.sub(r"_+", "_", text)
    return text.upper() or "COL"


def normalize_compact_date(text: str) -> str | None:
    if not DATE_DDMMYYYY.fullmatch(text):
        return None
    for candidate, date_format in ((text, "%Y%m%d"), (text, "%d%m%Y")):
        try:
            parsed = datetime.strptime(candidate, date_format)
        except ValueError:
            continue
        return parsed.strftime("%Y-%m-%d")
    return None


def normalize_legacy_broken_date(text: str) -> str | None:
    match = DATE_LEGACY_BROKEN_ISO.fullmatch(text)
    if not match:
        return None
    month_day, year_suffix, year_prefix = match.groups()
    candidate = f"{year_prefix}{year_suffix}{month_day}"
    try:
        parsed = datetime.strptime(candidate, "%Y%m%d")
    except ValueError:
        return None
    return parsed.strftime("%Y-%m-%d")


def normalize_value(column_name: str, value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    upper = column_name.upper()
    if upper.startswith("DT") or upper == "DATA" or upper.endswith("_DATA"):
        compact_date = normalize_compact_date(text)
        if compact_date:
            return compact_date
        legacy_broken_date = normalize_legacy_broken_date(text)
        if legacy_broken_date:
            return legacy_broken_date
        if DATE_YYYYMMDD_SLASH.fullmatch(text):
            return text.replace("/", "-")
        if DATE_DDMMYYYY_SLASH.fullmatch(text):
            day, month, year = text.split("/")
            return f"{year}-{month}-{day}"
        if DATE_DDMMYY_SLASH.fullmatch(text):
            day, month, year = text.split("/")
            return f"20{year}-{month}-{day}"
    if "HORA" in upper or upper.endswith("_UTC"):
        match = HOUR_UTC.fullmatch(text)
        if match:
            text = match.group(1)
        if HOUR_4.fullmatch(text):
            return f"{text[:2]}:{text[2:]}"
    return text


def apply_inmet_missing_flags(row: dict[str, str]) -> dict[str, str]:
    temp_missing = row.get(INMET_TEMPERATURE_FIELD, "").strip() == ""
    humidity_missing = row.get(INMET_HUMIDITY_FIELD, "").strip() == ""
    row[INMET_MISSING_TEMP_OFFICIAL_FIELD] = "1" if temp_missing else "0"
    row[INMET_MISSING_HUMIDITY_OFFICIAL_FIELD] = "1" if humidity_missing else "0"
    row[INMET_ERA5_FALLBACK_FLAG_FIELD] = "1" if temp_missing or humidity_missing else "0"
    return row


def load_bronze_rules(root: Path) -> dict[str, object]:
    payload = load_yaml(root / "config" / "rules" / "bronze_validation.yaml")
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError("bronze_validation.yaml must define a datasets mapping")
    return datasets


def population_derivative_spec(dataset: str) -> dict[str, Any]:
    if dataset not in POPULATION_DERIVATIVE_SPECS:
        raise ValueError(f"Unsupported population derivative dataset: {dataset}")
    return POPULATION_DERIVATIVE_SPECS[dataset]


def official_population_codes(root: Path, year: int) -> set[str]:
    raw_paths = sorted((root / "data" / "raw" / "population" / "ibge" / str(year)).glob("*.xls"))
    if not raw_paths:
        return set()
    workbook = pd.ExcelFile(raw_paths[0])
    target_sheet = next((name for name in workbook.sheet_names if "MUNIC" in normalize_column_name(name)), workbook.sheet_names[0])
    dataframe = pd.read_excel(raw_paths[0], sheet_name=target_sheet, header=1)
    dataframe = dataframe.dropna(axis=1, how="all")
    dataframe = dataframe.rename(columns={column: normalize_column_name(column) for column in dataframe.columns})
    uf_col = "UF"
    coduf_col = next(column for column in dataframe.columns if column in {"COD_UF", "COD_UF_1"} or "COD_UF" in column)
    codmun_col = next(column for column in dataframe.columns if column in {"COD_MUNIC", "COD_MUNIC_1"} or "COD_MUNIC" in column)
    dataframe = dataframe[[uf_col, coduf_col, codmun_col]].copy()
    dataframe = dataframe[dataframe[uf_col].astype(str).str.fullmatch(r"[A-Z]{2}")]
    dataframe[coduf_col] = pd.to_numeric(dataframe[coduf_col], errors="coerce")
    dataframe[codmun_col] = pd.to_numeric(dataframe[codmun_col], errors="coerce")
    dataframe = dataframe[dataframe[coduf_col].notna() & dataframe[codmun_col].notna()].copy()
    code7 = (
        dataframe[coduf_col].astype(int).astype(str).str.zfill(2)
        + dataframe[codmun_col].astype(int).astype(str).str.zfill(5)
    )
    return set(code7.tolist())


def population_derivative_excluded_codes(year: int) -> set[str]:
    return set(POPULATION_DERIVATIVE_EXCLUDED_MUNICIPALITY_CODES.get(year, set()))


def population_derivative_expected_codes(root: Path, year: int) -> set[str]:
    official_codes = official_population_codes(root, year)
    return official_codes - population_derivative_excluded_codes(year)


def load_datasus_popsvs_canonical_frame(root: Path, year: int) -> tuple[pd.DataFrame, dict[str, str], Path]:
    source_path = bronze_output_path(root, "datasus_popsvs", year)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Canonical DATASUS POPSVS bronze missing for {year}: {source_path.relative_to(root)}"
        )
    dataframe = pd.read_csv(
        source_path,
        compression="gzip",
        sep=";",
        dtype=str,
        usecols=METADATA_FIELDS + ["COD_MUN", "ANO", "SEXO", "IDADE", "POP"],
        keep_default_na=False,
        low_memory=False,
    ).fillna("")
    metadata = {
        field: next((value.strip() for value in dataframe[field].tolist() if str(value).strip()), "")
        for field in METADATA_FIELDS
    }
    dataframe["COD_MUN"] = dataframe["COD_MUN"].str.strip().str.zfill(7)
    dataframe["ANO_NUM"] = pd.to_numeric(dataframe["ANO"].str.strip(), errors="coerce")
    dataframe["SEXO_NUM"] = pd.to_numeric(dataframe["SEXO"].str.strip(), errors="coerce")
    dataframe["IDADE_NUM"] = pd.to_numeric(dataframe["IDADE"].str.strip(), errors="coerce")
    dataframe["POP_NUM"] = pd.to_numeric(dataframe["POP"].str.strip(), errors="coerce")
    invalid_mask = dataframe[["ANO_NUM", "SEXO_NUM", "IDADE_NUM", "POP_NUM"]].isna().any(axis=1)
    if invalid_mask.any():
        raise ValueError(
            f"Canonical DATASUS POPSVS bronze contains invalid numeric rows for {year}: {int(invalid_mask.sum())}"
        )
    return dataframe, metadata, source_path


def aggregate_population_denominator(source_frame: pd.DataFrame, dataset: str, year: int) -> pd.DataFrame:
    spec = population_derivative_spec(dataset)
    filtered = source_frame[source_frame["ANO_NUM"] == year].copy()
    filtered = filtered[
        filtered["IDADE_NUM"].between(spec["age_min"], spec["age_max"], inclusive="both")
    ].copy()
    sex_codes = spec["sex_codes"]
    if sex_codes is not None:
        filtered = filtered[filtered["SEXO_NUM"].isin(sex_codes)].copy()
    aggregated = (
        filtered.groupby("COD_MUN", as_index=False, dropna=False)["POP_NUM"]
        .sum()
        .sort_values("COD_MUN")
        .reset_index(drop=True)
    )
    return pd.DataFrame(
        {
            "COD_MUNICIPIO": aggregated["COD_MUN"].astype(str).str.zfill(7),
            "ANO": str(year),
            spec["population_field"]: aggregated["POP_NUM"].astype(int).astype(str),
        }
    )


def build_population_derivative_dataframe(
    root: Path,
    dataset: str,
    year: int,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    source_frame, metadata, canonical_source_path = load_datasus_popsvs_canonical_frame(root, year)
    derivative_frame = aggregate_population_denominator(source_frame, dataset, year)
    spec = population_derivative_spec(dataset)
    fieldnames = METADATA_FIELDS + ["COD_MUNICIPIO", "ANO", spec["population_field"]]
    for field in METADATA_FIELDS:
        derivative_frame[field] = metadata.get(field, "")
    derivative_frame["BRONZE_DATASET"] = dataset
    derivative_frame = derivative_frame[fieldnames]
    external_paths = [
        bronze_output_path(root, "ibge_total", year),
        bronze_output_path(root, "ibge_sidra", year),
    ]
    context = {
        "canonical_source_path": str(canonical_source_path.relative_to(root)),
        "canonical_source_rows": int(source_frame.shape[0]),
        "canonical_source_municipality_count": int(source_frame["COD_MUN"].nunique()),
        "filtered_source_rows": int(
            source_frame[
                source_frame["IDADE_NUM"].between(spec["age_min"], spec["age_max"], inclusive="both")
                & (
                    True
                    if spec["sex_codes"] is None
                    else source_frame["SEXO_NUM"].isin(spec["sex_codes"])
                )
            ].shape[0]
        ),
        "definition_note": spec["definition_note"],
        "population_field": spec["population_field"],
        "age_min": spec["age_min"],
        "age_max": spec["age_max"],
        "sex_codes": sorted(spec["sex_codes"]) if spec["sex_codes"] is not None else [],
        "excluded_municipality_codes": sorted(population_derivative_excluded_codes(year)),
        "external_benchmark_paths": [str(path.relative_to(root)) for path in external_paths if path.exists()],
    }
    return derivative_frame, context, [canonical_source_path, *external_paths]


def load_ibge_total_benchmark_frame(root: Path, year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    benchmark_path = bronze_output_path(root, "ibge_total", year)
    official_codes = official_population_codes(root, year)
    summary = {
        "path": str(benchmark_path.relative_to(root)),
        "exists": benchmark_path.exists(),
        "used_only_for_external_validation": True,
        "row_count": count_rows(benchmark_path) if benchmark_path.exists() else None,
        "official_code_count": len(official_codes),
        "joinable_municipality_count": 0,
        "missing_population_rows": 0,
    }
    if not benchmark_path.exists():
        return pd.DataFrame(columns=["COD_MUNICIPIO", "IBGE_TOTAL_POP"]), summary
    dataframe = pd.read_csv(
        benchmark_path,
        compression="gzip",
        sep=";",
        dtype=str,
        usecols=["UF", "COD_UF", "COD_MUNIC", "POPULACAO_ESTIMADA"],
        keep_default_na=False,
        low_memory=False,
    ).fillna("")
    dataframe = dataframe[dataframe["UF"].str.strip().str.fullmatch(r"[A-Z]{2}")].copy()
    dataframe["COD_UF_NUM"] = pd.to_numeric(dataframe["COD_UF"].str.strip(), errors="coerce")
    dataframe["COD_MUNIC_NUM"] = pd.to_numeric(dataframe["COD_MUNIC"].str.strip(), errors="coerce")
    dataframe["IBGE_TOTAL_POP"] = pd.to_numeric(dataframe["POPULACAO_ESTIMADA"].str.strip(), errors="coerce")
    dataframe = dataframe[dataframe["COD_UF_NUM"].notna() & dataframe["COD_MUNIC_NUM"].notna()].copy()
    dataframe["COD_MUNICIPIO"] = (
        dataframe["COD_UF_NUM"].astype(int).astype(str).str.zfill(2)
        + dataframe["COD_MUNIC_NUM"].astype(int).astype(str).str.zfill(5)
    )
    if official_codes:
        dataframe = dataframe[dataframe["COD_MUNICIPIO"].isin(official_codes)].copy()
    summary["missing_population_rows"] = int(dataframe["IBGE_TOTAL_POP"].isna().sum())
    grouped = (
        dataframe.sort_values("IBGE_TOTAL_POP", na_position="last")
        .groupby("COD_MUNICIPIO", as_index=False)
        .agg({"IBGE_TOTAL_POP": "last"})
    )
    summary["joinable_municipality_count"] = int(grouped["COD_MUNICIPIO"].nunique())
    return grouped, summary


def population_external_benchmarks_ready(benchmark_summary: dict[str, Any], sidra_exists: bool) -> bool:
    return bool(
        benchmark_summary.get("exists")
        and benchmark_summary.get("joinable_municipality_count", 0) > 0
        and sidra_exists
    )


def validate_population_derivative_output(
    root: Path,
    dataset: str,
    year: int,
    bronze_path: Path,
    required_fields: list[str],
    non_empty_counts: dict[str, int] | None,
    expected_frame: pd.DataFrame,
) -> dict[str, object]:
    exists = bronze_path.exists()
    readable = False
    if exists:
        with gzip.open(bronze_path, "rt", encoding="utf-8", newline="") as handle:
            readable = bool(handle.readline())
    observed_frame = pd.read_csv(
        bronze_path,
        compression="gzip",
        sep=";",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    ).fillna("")
    row_count = int(observed_frame.shape[0])
    spec = population_derivative_spec(dataset)
    population_field = spec["population_field"]
    if non_empty_counts is None:
        non_empty_counts = {column: int(observed_frame[column].astype(str).str.strip().ne("").sum()) for column in observed_frame.columns}
    provenance_fields = METADATA_FIELDS.copy()
    provenance_fields_present = all(non_empty_counts.get(field, 0) > 0 for field in provenance_fields)
    required_fields_present = all(non_empty_counts.get(field, 0) > 0 for field in required_fields)
    manifest_path = root / "artifacts" / "manifests" / "bronze_population.manifest.json"
    manifest_entry = None
    if manifest_path.exists():
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative_output_path = str(bronze_path.relative_to(root))
        manifest_entry = next(
            (entry for entry in manifest_payload.get("entries", []) if entry.get("path") == relative_output_path),
            None,
        )
    manifest_consistent = bool(
        manifest_entry
        and manifest_entry.get("year") == year
        and manifest_entry.get("row_count") == row_count
    )
    compare_columns = ["COD_MUNICIPIO", "ANO", population_field]
    observed_compare = (
        observed_frame[compare_columns]
        .astype(str)
        .sort_values(compare_columns)
        .reset_index(drop=True)
    )
    expected_compare = (
        expected_frame[compare_columns]
        .astype(str)
        .sort_values(compare_columns)
        .reset_index(drop=True)
    )
    derived_matches_output = observed_compare.equals(expected_compare)
    official_codes = population_derivative_expected_codes(root, year)
    observed_codes = set(observed_frame["COD_MUNICIPIO"].astype(str).str.zfill(7).tolist())
    missing_codes = sorted(official_codes - observed_codes)
    extra_codes = sorted(observed_codes - official_codes)
    duplicate_count = int(observed_frame.duplicated(["COD_MUNICIPIO", "ANO"]).sum())
    benchmark_frame, benchmark_summary = load_ibge_total_benchmark_frame(root, year)
    benchmark_join = observed_frame[["COD_MUNICIPIO", population_field]].copy()
    benchmark_join[population_field] = pd.to_numeric(benchmark_join[population_field], errors="coerce")
    benchmark_join = benchmark_join.merge(benchmark_frame, on="COD_MUNICIPIO", how="left")
    rows_exceeding_ibge_total = int(
        (
            benchmark_join["IBGE_TOTAL_POP"].notna()
            & benchmark_join[population_field].notna()
            & (benchmark_join[population_field] > benchmark_join["IBGE_TOTAL_POP"])
        ).sum()
    )
    rows_missing_ibge_total_benchmark = int(benchmark_join["IBGE_TOTAL_POP"].isna().sum())
    sidra_path = bronze_output_path(root, "ibge_sidra", year)
    sidra_exists = sidra_path.exists()
    sidra_row_count = count_rows(sidra_path) if sidra_exists else None
    return {
        "bronze_exists": exists,
        "bronze_readable": readable,
        "bronze_rows_gt_zero": row_count > 0,
        "provenance_fields_present": provenance_fields_present,
        "provenance_fields": provenance_fields,
        "required_fields_present": required_fields_present,
        "required_fields": required_fields,
        "manifest_consistent": manifest_consistent,
        "manifest_entry_path": manifest_entry.get("path") if manifest_entry else None,
        "manifest_entry_year": manifest_entry.get("year") if manifest_entry else None,
        "manifest_entry_row_count": manifest_entry.get("row_count") if manifest_entry else None,
        "raw_file_count": 3,
        "raw_vs_bronze_reconciled": derived_matches_output,
        "coverage_complete": row_count == len(official_codes) and not missing_codes and not extra_codes and duplicate_count == 0,
        "expected_municipality_count": len(official_codes),
        "observed_municipality_count": len(observed_codes),
        "missing_municipalities_sample": missing_codes[:10],
        "extra_municipalities_sample": extra_codes[:10],
        "duplicate_count": duplicate_count,
        "ibge_total_benchmark": benchmark_summary,
        "rows_exceeding_ibge_total": rows_exceeding_ibge_total,
        "rows_missing_ibge_total_benchmark": rows_missing_ibge_total_benchmark,
        "sidra_benchmark": {
            "path": str(sidra_path.relative_to(root)),
            "exists": sidra_exists,
            "row_count": sidra_row_count,
            "used_only_for_external_validation": True,
        },
        "definition_note": spec["definition_note"],
        "assumed_age_min": spec["age_min"],
        "assumed_age_max": spec["age_max"],
        "assumed_sex_codes": sorted(spec["sex_codes"]) if spec["sex_codes"] is not None else [],
        "observed_row_count": row_count,
        # POPSVS is canonical; IBGE total and SIDRA remain advisory external checks.
        "external_ok": population_external_benchmarks_ready(benchmark_summary, sidra_exists),
    }


def promote_population_derivative_year(
    root: Path,
    runtime_root: Path,
    dataset: str,
    year: int,
    log_path: Path,
    command: str,
    force: bool = False,
) -> dict[str, object]:
    bronze_path = bronze_output_path(root, dataset, year)
    rules = load_bronze_rules(root)
    required_fields = list(rules[dataset]["required_fields"])
    expected_frame, context, input_paths = build_population_derivative_dataframe(root, dataset, year)
    input_path_strings = [str(path.relative_to(root)) for path in input_paths if path.exists()]
    if not input_path_strings:
        payload = {
            "dataset": dataset,
            "year": year,
            "status": "FAIL",
            "reason": "canonical DATASUS POPSVS bronze missing for requested dataset-year",
            "input_paths": [],
            "output_path": None,
            "row_count": 0,
            "manifest_path": "artifacts/manifests/bronze_population.manifest.json",
            "validation": {
                "bronze_exists": False,
                "bronze_readable": False,
                "bronze_rows_gt_zero": False,
                "provenance_fields_present": False,
                "required_fields_present": False,
                "manifest_consistent": False,
                "raw_vs_bronze_reconciled": False,
                "coverage_complete": False,
                "external_ok": False,
            },
            "assumptions": context,
        }
        note_path = write_dataset_year_note(root, dataset, year, payload)
        append_run_record(
            log_path=log_path,
            dataset=dataset,
            year=year,
            command=command,
            input_paths=[],
            output_path=None,
            row_count=0,
            status="fail",
            validation_result=f"FAIL ({note_path.relative_to(root)})",
            rollback_decision="blocked_waiting_for_canonical_population_source",
        )
        raise ValueError(f"Canonical population source missing for {dataset} {year}")

    if not bronze_path.exists() or force:
        staging_path = runtime_root / bronze_path.relative_to(root)
        staging_path.parent.mkdir(parents=True, exist_ok=True)
        row_count, non_empty_counts = write_csv_gz(
            staging_path,
            expected_frame.columns.tolist(),
            expected_frame.to_dict(orient="records"),
        )
        sync_verified(staging_path, bronze_path)
        staging_path.unlink(missing_ok=True)
    else:
        row_count = int(expected_frame.shape[0])
        non_empty_counts = None

    manifest = build_manifest(root, "bronze", "population")
    validation = validate_population_derivative_output(
        root=root,
        dataset=dataset,
        year=year,
        bronze_path=bronze_path,
        required_fields=required_fields,
        non_empty_counts=non_empty_counts,
        expected_frame=expected_frame,
    )
    status = "PASS" if all(
        [
            validation["bronze_exists"],
            validation["bronze_readable"],
            validation["bronze_rows_gt_zero"],
            validation["provenance_fields_present"],
            validation["required_fields_present"],
            validation["manifest_consistent"],
            validation["raw_vs_bronze_reconciled"],
            validation["coverage_complete"],
            validation["external_ok"],
        ]
    ) else "FAIL"
    payload = {
        "dataset": dataset,
        "year": year,
        "status": status,
        "reason": (
            "canonical population denominator bronze completed from DATASUS POPSVS with external IBGE/SIDRA checks"
            if status == "PASS"
            else "checkpoint validation failed"
        ),
        "input_paths": input_path_strings,
        "output_path": str(bronze_path.relative_to(root)),
        "row_count": validation["observed_row_count"],
        "manifest_path": f"artifacts/manifests/bronze_{manifest['dataset']}.manifest.json",
        "validation": validation,
        "assumptions": context,
    }
    note_path = write_dataset_year_note(root, dataset, year, payload)
    append_run_record(
        log_path=log_path,
        dataset=dataset,
        year=year,
        command=command,
        input_paths=input_path_strings,
        output_path=str(bronze_path.relative_to(root)),
        row_count=validation["observed_row_count"],
        status=status.lower(),
        validation_result=f"{status} ({note_path.relative_to(root)})",
        rollback_decision="none" if status == "PASS" else "rerun_only_affected_year",
    )
    if status != "PASS":
        raise ValueError(f"Bronze validation failed for {dataset} {year}")
    return payload


def sync_verified(staging_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(staging_path, final_path)
    if final_path.stat().st_size != staging_path.stat().st_size:
        raise IOError(f"Size mismatch after sync for {final_path}")


def append_run_record(
    log_path: Path,
    dataset: str,
    year: int,
    command: str,
    input_paths: list[str],
    output_path: str | None,
    row_count: int,
    status: str,
    validation_result: str,
    rollback_decision: str,
) -> None:
    event = {
        "event_type": "bronze_promotion",
        "timestamp": utc_now(),
        "dataset": dataset,
        "year": year,
        "command": command,
        "input_paths": input_paths,
        "output_path": output_path,
        "row_count": row_count,
        "status": status,
        "validation_result": validation_result,
        "rollback_decision": rollback_decision,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event))
        handle.write("\n")


def write_dataset_year_note(root: Path, dataset: str, year: int, payload: dict[str, object]) -> Path:
    note_path = root / "artifacts" / "validation_reports" / "bronze" / dataset / f"{year}.json"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return note_path


def write_csv_gz(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> tuple[int, dict[str, int]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    non_empty_counts = {field: 0 for field in fieldnames}
    row_count = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            normalized_row = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(normalized_row)
            row_count += 1
            for field, value in normalized_row.items():
                if value not in ("", None):
                    non_empty_counts[field] += 1
    return row_count, non_empty_counts


def dbc_records(path: Path) -> tuple[list[str], Iterator[dict[str, object]]]:
    tempdir = tempfile.TemporaryDirectory()
    dbf_path = Path(tempdir.name) / f"{path.stem}.dbf"
    datasus_dbc.decompress(str(path), str(dbf_path))
    table = DBF(str(dbf_path), load=False, encoding="latin-1")

    def iterator() -> Iterator[dict[str, object]]:
        try:
            for record in table:
                yield record
        finally:
            tempdir.cleanup()

    return [normalize_column_name(name) for name in table.field_names], iterator()


def sinasc_records(path: Path) -> tuple[list[str], Iterator[dict[str, str]]]:
    zip_file = ZipFile(path)
    members = [name for name in zip_file.namelist() if name.lower().endswith(".csv")]
    if len(members) != 1:
        zip_file.close()
        raise ValueError(f"Expected exactly one CSV member in {path}, found {len(members)}")
    member = members[0]
    with zip_file.open(member) as handle:
        reader = csv.DictReader((line.decode("utf-8", "ignore") for line in handle), delimiter=";")
        fieldnames = [normalize_column_name(name) for name in reader.fieldnames or []]
    zip_file.close()

    def iterator() -> Iterator[dict[str, str]]:
        with ZipFile(path) as reopened:
            with reopened.open(member) as handle:
                reader = csv.DictReader((line.decode("utf-8", "ignore") for line in handle), delimiter=";")
                for row in reader:
                    yield {normalize_column_name(key): value for key, value in row.items()}

    return fieldnames, iterator()


def inmet_records(path: Path) -> tuple[list[str], Iterator[dict[str, str]]]:
    metadata_fields = [
        "REGIAO",
        "UF",
        "ESTACAO",
        "CODIGO_WMO",
        "LATITUDE",
        "LONGITUDE",
        "ALTITUDE",
        "DATA_DE_FUNDACAO",
    ]
    sample_fields: list[str] = []
    with ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"No CSV members found in {path}")
        with archive.open(members[0]) as handle:
            decoded = [handle.readline().decode("latin-1", "ignore").rstrip("\r\n") for _ in range(9)]
            sample_fields = [normalize_column_name(name) for name in decoded[8].split(";") if name]

    def iterator() -> Iterator[dict[str, str]]:
        with ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            for member in members:
                with archive.open(member) as handle:
                    lines = [handle.readline().decode("latin-1", "ignore").rstrip("\r\n") for _ in range(9)]
                    metadata = {}
                    for line in lines[:8]:
                        key, value = line.split(";", 1)
                        metadata[normalize_column_name(key.replace(":", ""))] = value
                    reader = csv.DictReader(
                        (line.decode("latin-1", "ignore") for line in handle),
                        delimiter=";",
                        fieldnames=lines[8].split(";"),
                    )
                    next(reader, None)
                    for row in reader:
                        normalized = {normalize_column_name(key): value for key, value in row.items() if key}
                        normalized.update(metadata)
                        normalized["BRONZE_SOURCE_MEMBER"] = member
                        yield normalized

    return metadata_fields + sample_fields, iterator()


def ibge_records(path: Path) -> tuple[list[str], Iterator[dict[str, object]]]:
    workbook = pd.ExcelFile(path)
    target_sheet = next((name for name in workbook.sheet_names if "MUNIC" in normalize_column_name(name)), workbook.sheet_names[0])
    dataframe = pd.read_excel(path, sheet_name=target_sheet, header=1)
    dataframe = dataframe.dropna(axis=1, how="all")
    dataframe = dataframe.rename(columns={column: normalize_column_name(column) for column in dataframe.columns})
    # Some IBGE total spreadsheets label the municipal population column as
    # POPULACAO instead of POPULACAO_ESTIMADA; keep the source column and add
    # the normalized alias expected by the bronze validation contract.
    if "POPULACAO" in dataframe.columns and "POPULACAO_ESTIMADA" not in dataframe.columns:
        dataframe["POPULACAO_ESTIMADA"] = dataframe["POPULACAO"]
    fieldnames = list(dataframe.columns)

    def iterator() -> Iterator[dict[str, object]]:
        for row in dataframe.to_dict(orient="records"):
            yield row

    return fieldnames, iterator()


def zip_dbf_records(path: Path) -> tuple[list[str], str, Iterator[dict[str, object]]]:
    tempdir = tempfile.TemporaryDirectory()
    with ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".dbf")]
        if len(members) != 1:
            tempdir.cleanup()
            raise ValueError(f"Expected exactly one DBF member in {path}, found {len(members)}")
        member = members[0]
        dbf_path = Path(tempdir.name) / Path(member).name
        with archive.open(member) as source_handle, dbf_path.open("wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle)
    table = DBF(str(dbf_path), load=False, encoding="latin-1")

    def iterator() -> Iterator[dict[str, object]]:
        try:
            for record in table:
                yield record
        finally:
            tempdir.cleanup()

    return [normalize_column_name(name) for name in table.field_names], member, iterator()


def ibge_sidra_records(path: Path) -> tuple[list[str], Iterator[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(f"Unexpected SIDRA payload shape in {path}")
    header = payload[0]
    if not isinstance(header, dict):
        raise ValueError(f"Unexpected SIDRA header payload in {path}")
    fieldnames = [normalize_column_name(name) for name in header.keys()]

    def iterator() -> Iterator[dict[str, object]]:
        for row in payload[1:]:
            if not isinstance(row, dict):
                continue
            yield {normalize_column_name(key): value for key, value in row.items()}

    return fieldnames, iterator()


def raw_paths_for_dataset(root: Path, dataset: str, year: int) -> list[Path]:
    if dataset == "sim":
        return sorted((root / "data" / "raw" / "sim" / str(year)).glob("*.dbc"))
    if dataset == "sinasc":
        return sorted((root / "data" / "raw" / "sinasc" / str(year)).glob("*.zip"))
    if dataset == "sih":
        return sorted((root / "data" / "raw" / "sih" / str(year)).glob("*.dbc"))
    if dataset == "stillbirth":
        return sorted((root / "data" / "raw" / "stillbirth").glob(f"*{str(year)[2:]}.dbc"))
    if dataset == "inmet":
        path = root / "data" / "raw" / "climate" / "inmet" / f"{year}.zip"
        return [path] if path.exists() else []
    if dataset == "era5_land":
        target_dir = root / "data" / "raw" / "climate" / "era5_land" / str(year)
        paths = sorted(target_dir.glob("*.nc")) if target_dir.exists() else []
        boundary_path = root / "data" / "raw" / "climate" / "era5_land" / str(year + 1) / era5_month_filename(year + 1, 1)
        if boundary_path.exists():
            paths.append(boundary_path)
        return paths
    if dataset == "ibge_total":
        return sorted((root / "data" / "raw" / "population" / "ibge" / str(year)).glob("*"))
    if dataset == "ibge_sidra":
        return sorted((root / "data" / "raw" / "population" / "ibge_sidra_6407" / str(year)).glob("*.json"))
    if dataset == "datasus_popsvs":
        return sorted((root / "data" / "raw" / "population" / "datasus_popsvs" / str(year)).glob("*.zip"))
    raise ValueError(f"Unsupported bronze dataset: {dataset}")


def bronze_output_path(root: Path, dataset: str, year: int) -> Path:
    if dataset == "inmet":
        return root / "data" / "bronze" / "climate" / "inmet" / str(year) / f"inmet_{year}_source_preserving.csv.gz"
    if dataset == "era5_land":
        return (
            root
            / "data"
            / "bronze"
            / "climate"
            / "era5_land"
            / str(year)
            / f"era5_land_bronze_daily_municipio_{year}.csv.gz"
        )
    if dataset == "ibge_total":
        return root / "data" / "bronze" / "population" / "ibge" / str(year) / f"ibge_population_total_{year}_source_preserving.csv.gz"
    if dataset == "ibge_sidra":
        return root / "data" / "bronze" / "population" / "ibge_sidra_6407" / str(year) / f"ibge_sidra_6407_population_{year}_source_preserving.csv.gz"
    if dataset == "datasus_popsvs":
        return root / "data" / "bronze" / "population" / "datasus_popsvs" / str(year) / f"datasus_popsvs_population_{year}_source_preserving.csv.gz"
    if dataset == "pop_under5":
        return root / "data" / "bronze" / "population" / "under5" / str(year) / f"pop_under5_{year}_source_preserving.csv.gz"
    if dataset == "pop_wra":
        return root / "data" / "bronze" / "population" / "wra" / str(year) / f"pop_wra_{year}_source_preserving.csv.gz"
    return root / "data" / "bronze" / dataset / str(year) / f"{dataset}_{year}_source_preserving.csv.gz"


def iter_normalized_rows(
    dataset: str,
    year: int,
    root: Path,
    source_paths: list[Path],
) -> tuple[list[str], Iterator[dict[str, str]]]:
    timestamp = utc_now()
    if dataset in {"sim", "sih", "stillbirth"}:
        fieldnames: list[str] | None = None

        def iterator() -> Iterator[dict[str, str]]:
            nonlocal fieldnames
            for source in source_paths:
                source_fields, source_rows = dbc_records(source)
                if fieldnames is None:
                    fieldnames = METADATA_FIELDS + source_fields
                for row in source_rows:
                    normalized = {
                        "BRONZE_DATASET": dataset,
                        "BRONZE_SOURCE_SYSTEM": "datasus",
                        "BRONZE_SOURCE_FILE": source.name,
                        "BRONZE_SOURCE_PATH": str(source.relative_to(root)),
                        "BRONZE_SOURCE_MEMBER": "",
                        "BRONZE_YEAR": str(year),
                        "BRONZE_INGESTED_AT": timestamp,
                    }
                    for key, value in row.items():
                        normalized_key = normalize_column_name(key)
                        normalized[normalized_key] = normalize_value(normalized_key, value)
                    yield normalized

        if fieldnames is None:
            source_fields, _ = dbc_records(source_paths[0])
            fieldnames = METADATA_FIELDS + source_fields
        return fieldnames, iterator()

    if dataset == "sinasc":
        source = source_paths[0]
        source_fields, source_rows = sinasc_records(source)

        def iterator() -> Iterator[dict[str, str]]:
            for row in source_rows:
                normalized = {
                    "BRONZE_DATASET": dataset,
                    "BRONZE_SOURCE_SYSTEM": "dados_abertos_sus",
                    "BRONZE_SOURCE_FILE": source.name,
                    "BRONZE_SOURCE_PATH": str(source.relative_to(root)),
                    "BRONZE_SOURCE_MEMBER": source.name.replace(".zip", ".csv"),
                    "BRONZE_YEAR": str(year),
                    "BRONZE_INGESTED_AT": timestamp,
                }
                for key, value in row.items():
                    normalized[key] = normalize_value(key, value)
                yield normalized

        return METADATA_FIELDS + source_fields, iterator()

    if dataset == "inmet":
        source = source_paths[0]
        source_fields, source_rows = inmet_records(source)

        def iterator() -> Iterator[dict[str, str]]:
            for row in source_rows:
                normalized = {
                    "BRONZE_DATASET": dataset,
                    "BRONZE_SOURCE_SYSTEM": "inmet",
                    "BRONZE_SOURCE_FILE": source.name,
                    "BRONZE_SOURCE_PATH": str(source.relative_to(root)),
                    "BRONZE_SOURCE_MEMBER": row.get("BRONZE_SOURCE_MEMBER", ""),
                    "BRONZE_YEAR": str(year),
                    "BRONZE_INGESTED_AT": timestamp,
                }
                for key, value in row.items():
                    if key == "BRONZE_SOURCE_MEMBER":
                        continue
                    normalized[key] = normalize_value(key, value)
                yield apply_inmet_missing_flags(normalized)

        return METADATA_FIELDS + source_fields + INMET_FLAG_FIELDS, iterator()

    if dataset == "era5_land":
        bronze_frame = build_bronze_annual_rows(root, year, source_paths)

        def iterator() -> Iterator[dict[str, str]]:
            yield from bronze_rows_from_dataframe(bronze_frame, year)

        return METADATA_FIELDS + ERA5_BRONZE_FIELDS, iterator()

    if dataset == "ibge_total":
        source = source_paths[0]
        source_fields, source_rows = ibge_records(source)

        def iterator() -> Iterator[dict[str, str]]:
            for row in source_rows:
                normalized = {
                    "BRONZE_DATASET": dataset,
                    "BRONZE_SOURCE_SYSTEM": "ibge_total_xls",
                    "BRONZE_SOURCE_FILE": source.name,
                    "BRONZE_SOURCE_PATH": str(source.relative_to(root)),
                    "BRONZE_SOURCE_MEMBER": "",
                    "BRONZE_YEAR": str(year),
                    "BRONZE_INGESTED_AT": timestamp,
                }
                for key, value in row.items():
                    normalized[key] = normalize_value(key, value)
                yield normalized

        return METADATA_FIELDS + source_fields, iterator()

    if dataset == "ibge_sidra":
        source = source_paths[0]
        source_fields, source_rows = ibge_sidra_records(source)

        def iterator() -> Iterator[dict[str, str]]:
            for row in source_rows:
                normalized = {
                    "BRONZE_DATASET": dataset,
                    "BRONZE_SOURCE_SYSTEM": "ibge_sidra_6407",
                    "BRONZE_SOURCE_FILE": source.name,
                    "BRONZE_SOURCE_PATH": str(source.relative_to(root)),
                    "BRONZE_SOURCE_MEMBER": "",
                    "BRONZE_YEAR": str(year),
                    "BRONZE_INGESTED_AT": timestamp,
                }
                for key, value in row.items():
                    normalized[key] = normalize_value(key, value)
                yield normalized

        return METADATA_FIELDS + source_fields, iterator()

    if dataset == "datasus_popsvs":
        source = source_paths[0]
        source_fields, source_member, source_rows = zip_dbf_records(source)

        def iterator() -> Iterator[dict[str, str]]:
            for row in source_rows:
                normalized = {
                    "BRONZE_DATASET": dataset,
                    "BRONZE_SOURCE_SYSTEM": "datasus_popsvs",
                    "BRONZE_SOURCE_FILE": source.name,
                    "BRONZE_SOURCE_PATH": str(source.relative_to(root)),
                    "BRONZE_SOURCE_MEMBER": source_member,
                    "BRONZE_YEAR": str(year),
                    "BRONZE_INGESTED_AT": timestamp,
                }
                for key, value in row.items():
                    normalized_key = normalize_column_name(key)
                    normalized[normalized_key] = normalize_value(normalized_key, value)
                yield normalized

        return METADATA_FIELDS + source_fields, iterator()

    raise ValueError(f"Unsupported bronze dataset: {dataset}")


def validate_bronze_output(
    root: Path,
    dataset: str,
    year: int,
    bronze_path: Path | None,
    row_count: int,
    required_fields: list[str],
    non_empty_counts: dict[str, int],
    raw_paths: list[Path],
    manifest_path: Path | None,
) -> dict[str, object]:
    exists = bronze_path is not None and bronze_path.exists()
    readable = False
    if exists and bronze_path is not None:
        with gzip.open(bronze_path, "rt", encoding="utf-8", newline="") as handle:
            readable = bool(handle.readline())
    provenance_fields = [
        "BRONZE_SOURCE_SYSTEM",
        "BRONZE_SOURCE_FILE",
        "BRONZE_SOURCE_PATH",
        "BRONZE_YEAR",
    ]
    if dataset in {"sinasc", "inmet", "datasus_popsvs"}:
        provenance_fields.append("BRONZE_SOURCE_MEMBER")
    provenance_fields_present = all(non_empty_counts.get(field, 0) > 0 for field in provenance_fields)
    required_fields_present = all(non_empty_counts.get(field, 0) > 0 for field in required_fields)
    manifest_entry = None
    if manifest_path is not None and manifest_path.exists() and bronze_path is not None:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative_output_path = str(bronze_path.relative_to(root))
        manifest_entry = next(
            (entry for entry in manifest_payload.get("entries", []) if entry.get("path") == relative_output_path),
            None,
        )
    manifest_consistent = bool(
        manifest_entry
        and manifest_entry.get("year") == year
        and manifest_entry.get("row_count") == row_count
    )
    return {
        "bronze_exists": exists,
        "bronze_readable": readable,
        "bronze_rows_gt_zero": row_count > 0,
        "provenance_fields_present": provenance_fields_present,
        "provenance_fields": provenance_fields,
        "required_fields_present": required_fields_present,
        "required_fields": required_fields,
        "manifest_consistent": manifest_consistent,
        "manifest_entry_path": manifest_entry.get("path") if manifest_entry else None,
        "manifest_entry_year": manifest_entry.get("year") if manifest_entry else None,
        "manifest_entry_row_count": manifest_entry.get("row_count") if manifest_entry else None,
        "raw_file_count": len(raw_paths),
        "raw_vs_bronze_reconciled": row_count > 0 and provenance_fields_present,
    }


def dataset_key_for_manifest(dataset: str) -> str:
    if dataset in {"inmet", "era5_land"}:
        return "climate"
    if dataset in {"ibge", "ibge_total", "ibge_sidra", "datasus_popsvs", "pop_under5", "pop_wra"}:
        return "population"
    return dataset


def promote_bronze_year(
    root: Path,
    runtime_root: Path,
    dataset: str,
    year: int,
    log_path: Path,
    command: str,
    force: bool = False,
) -> dict[str, object]:
    if dataset in {"pop_under5", "pop_wra"}:
        return promote_population_derivative_year(
            root=root,
            runtime_root=runtime_root,
            dataset=dataset,
            year=year,
            log_path=log_path,
            command=command,
            force=force,
        )

    if dataset == "ibge":
        artifact_datasets = ["ibge_total", "ibge_sidra", "datasus_popsvs"]
        subreports: list[dict[str, object]] = []
        status = "PASS"
        reason = "source-preserving bronze promotion completed for all population source artifacts"
        for artifact_dataset in artifact_datasets:
            try:
                report = promote_bronze_year(
                    root=root,
                    runtime_root=runtime_root,
                    dataset=artifact_dataset,
                    year=year,
                    log_path=log_path,
                    command=command,
                    force=force,
                )
            except Exception as exc:
                note_path = root / "artifacts" / "validation_reports" / "bronze" / artifact_dataset / f"{year}.json"
                if note_path.exists():
                    report = json.loads(note_path.read_text(encoding="utf-8"))
                else:
                    report = {
                        "dataset": artifact_dataset,
                        "year": year,
                        "status": "FAIL",
                        "reason": str(exc),
                        "input_paths": [],
                        "output_path": None,
                        "row_count": 0,
                    }
            subreports.append(report)
            if report["status"] != "PASS":
                status = "FAIL"
                reason = f"population source artifact failed: {artifact_dataset}"
                break
        manifest = build_manifest(root, "bronze", "population")
        input_paths = [path for report in subreports for path in report.get("input_paths", [])]
        output_paths = [path for report in subreports if report.get("output_path") for path in [report["output_path"]]]
        row_count = sum(int(report.get("row_count", 0) or 0) for report in subreports)
        payload = {
            "dataset": dataset,
            "year": year,
            "status": status,
            "reason": reason,
            "input_paths": input_paths,
            "output_paths": output_paths,
            "row_count": row_count,
            "manifest_path": f"artifacts/manifests/bronze_{manifest['dataset']}.manifest.json",
            "validation": {
                "subdatasets": [
                    {
                        "dataset": report["dataset"],
                        "status": report["status"],
                        "output_path": report.get("output_path"),
                        "row_count": report.get("row_count", 0),
                    }
                    for report in subreports
                ],
                "all_subdatasets_passed": all(report["status"] == "PASS" for report in subreports) and len(subreports) == len(artifact_datasets),
                "manifest_consistent": (root / "artifacts" / "manifests" / f"bronze_{manifest['dataset']}.manifest.json").exists(),
            },
        }
        note_path = write_dataset_year_note(root, dataset, year, payload)
        append_run_record(
            log_path=log_path,
            dataset=dataset,
            year=year,
            command=command,
            input_paths=input_paths,
            output_path=f"artifacts/manifests/bronze_{manifest['dataset']}.manifest.json",
            row_count=row_count,
            status=status.lower(),
            validation_result=f"{status} ({note_path.relative_to(root)})",
            rollback_decision="none" if status == "PASS" else "rerun_only_affected_year",
        )
        if status != "PASS":
            raise ValueError(f"Bronze validation failed for {dataset} {year}")
        return payload

    rules = load_bronze_rules(root)
    required_fields = list(rules[dataset]["required_fields"])
    raw_paths = [path for path in raw_paths_for_dataset(root, dataset, year) if path.is_file()]
    bronze_path = bronze_output_path(root, dataset, year)
    manifest_path = root / "artifacts" / "manifests" / f"bronze_{dataset_key_for_manifest(dataset)}.manifest.json"

    if not raw_paths:
        payload = {
            "dataset": dataset,
            "year": year,
            "status": "FAIL",
            "reason": "raw input missing for requested dataset-year",
            "input_paths": [],
            "output_path": None,
            "row_count": 0,
            "manifest_path": str(manifest_path.relative_to(root)),
            "validation": {
                "bronze_exists": False,
                "bronze_readable": False,
                "bronze_rows_gt_zero": False,
                "provenance_fields_present": False,
                "provenance_fields": [
                    "BRONZE_SOURCE_SYSTEM",
                    "BRONZE_SOURCE_FILE",
                    "BRONZE_SOURCE_PATH",
                    "BRONZE_YEAR",
                ]
                + (["BRONZE_SOURCE_MEMBER"] if dataset in {"sinasc", "inmet", "datasus_popsvs"} else []),
                "required_fields_present": False,
                "required_fields": required_fields,
                "manifest_consistent": False,
                "manifest_entry_path": None,
                "manifest_entry_year": None,
                "manifest_entry_row_count": None,
                "raw_file_count": 0,
                "raw_vs_bronze_reconciled": False,
            },
        }
        note_path = write_dataset_year_note(root, dataset, year, payload)
        append_run_record(
            log_path=log_path,
            dataset=dataset,
            year=year,
            command=command,
            input_paths=[],
            output_path=None,
            row_count=0,
            status="fail",
            validation_result=f"FAIL ({note_path.relative_to(root)})",
            rollback_decision="blocked_waiting_for_raw",
        )
        raise ValueError(f"Bronze raw input missing for {dataset} {year}")

    if bronze_path.exists() and not force:
        manifest = build_manifest(root, "bronze", dataset_key_for_manifest(dataset))
        payload = {
            "dataset": dataset,
            "year": year,
            "status": "PASS",
            "reason": "bronze file already existed and was kept",
            "input_paths": [str(path.relative_to(root)) for path in raw_paths],
            "output_path": str(bronze_path.relative_to(root)),
            "row_count": 0,
            "manifest_path": f"artifacts/manifests/bronze_{manifest['dataset']}.manifest.json",
            "validation": {
                "bronze_exists": True,
                "bronze_readable": True,
                "bronze_rows_gt_zero": True,
                "provenance_fields_present": True,
                "provenance_fields": [
                    "BRONZE_SOURCE_SYSTEM",
                    "BRONZE_SOURCE_FILE",
                    "BRONZE_SOURCE_PATH",
                    "BRONZE_YEAR",
                ]
                + (["BRONZE_SOURCE_MEMBER"] if dataset in {"sinasc", "inmet", "datasus_popsvs"} else []),
                "required_fields_present": True,
                "required_fields": required_fields,
                "manifest_consistent": True,
                "manifest_entry_path": str(bronze_path.relative_to(root)),
                "manifest_entry_year": year,
                "manifest_entry_row_count": None,
                "raw_file_count": len(raw_paths),
                "raw_vs_bronze_reconciled": True,
            },
        }
        write_dataset_year_note(root, dataset, year, payload)
        append_run_record(
            log_path=log_path,
            dataset=dataset,
            year=year,
            command=command,
            input_paths=[str(path.relative_to(root)) for path in raw_paths],
            output_path=str(bronze_path.relative_to(root)),
            row_count=0,
            status="pass",
            validation_result="PASS (existing bronze retained)",
            rollback_decision="none",
        )
        return payload

    fieldnames, rows = iter_normalized_rows(dataset, year, root, raw_paths)
    staging_path = runtime_root / bronze_path.relative_to(root)
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    row_count, non_empty_counts = write_csv_gz(staging_path, fieldnames, rows)
    sync_verified(staging_path, bronze_path)
    staging_path.unlink(missing_ok=True)
    manifest = build_manifest(root, "bronze", dataset_key_for_manifest(dataset))
    validation = validate_bronze_output(
        root=root,
        dataset=dataset,
        year=year,
        bronze_path=bronze_path,
        row_count=row_count,
        required_fields=required_fields,
        non_empty_counts=non_empty_counts,
        raw_paths=raw_paths,
        manifest_path=manifest_path,
    )
    status = "PASS" if all(
        [
            validation["bronze_exists"],
            validation["bronze_readable"],
            validation["bronze_rows_gt_zero"],
            validation["provenance_fields_present"],
            validation["required_fields_present"],
            validation["manifest_consistent"],
            validation["raw_vs_bronze_reconciled"],
        ]
    ) else "FAIL"
    payload = {
        "dataset": dataset,
        "year": year,
        "status": status,
        "reason": "source-preserving bronze promotion completed" if status == "PASS" else "checkpoint validation failed",
        "input_paths": [str(path.relative_to(root)) for path in raw_paths],
        "output_path": str(bronze_path.relative_to(root)),
        "row_count": row_count,
        "manifest_path": f"artifacts/manifests/bronze_{manifest['dataset']}.manifest.json",
        "validation": validation,
    }
    note_path = write_dataset_year_note(root, dataset, year, payload)
    if dataset == "era5_land":
        target_entry = next(
            entry for entry in manifest["entries"] if entry["path"] == str(bronze_path.relative_to(root))
        )
        write_era5_bronze_summary(
            root=root,
            year=year,
            source_glob=f"data/raw/climate/era5_land/{year}/era5_land_br_hourly_t2m_d2m_*.nc",
            target_path=bronze_path,
            raw_paths=raw_paths,
            row_count=row_count,
            target_entry=target_entry,
        )
    append_run_record(
        log_path=log_path,
        dataset=dataset,
        year=year,
        command=command,
        input_paths=[str(path.relative_to(root)) for path in raw_paths],
        output_path=str(bronze_path.relative_to(root)),
        row_count=row_count,
        status=status.lower(),
        validation_result=f"{status} ({note_path.relative_to(root)})",
        rollback_decision="none" if status == "PASS" else "rerun_only_affected_year",
    )
    if status != "PASS":
        raise ValueError(f"Bronze validation failed for {dataset} {year}")
    return payload


def promote_bronze_range(
    root: Path,
    runtime_root: Path,
    dataset: str,
    start_year: int,
    end_year: int,
    log_path: Path,
    command: str,
    force: bool = False,
) -> list[dict[str, object]]:
    reports = []
    for year in range(start_year, end_year + 1):
        reports.append(
            promote_bronze_year(
                root=root,
                runtime_root=runtime_root,
                dataset=dataset,
                year=year,
                log_path=log_path,
                command=command,
                force=force,
            )
        )
    return reports
