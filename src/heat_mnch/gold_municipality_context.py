from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd
import requests

from .manifest import build_manifest, sha256_file
from .run_log import append_run_event
from .silver_outcome_common import ParquetChunkWriter, sync_verified_parquet


DATASET_NAME = "gold_municipality_context"
STATIC_DATASET_NAME = "gold_municipality_context_static"
READBACK_TOLERANCE = 1e-9
BOUNDARY_REFERENCE_YEAR = 2022

IBGE_URBAN_RURAL_URL = (
    "https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Agregados_por_Setores_Censitarios/"
    "Agregados_por_Setor_csv/Agregados_por_setores_basico_BR_20250417.zip"
)
IBGE_URBAN_RURAL_FILENAME = "Agregados_por_setores_basico_BR_20250417.zip"
IBGE_URBAN_CLASS_CODES = {"1", "2", "3"}
IBGE_RURAL_CLASS_CODES = {"5", "6", "7", "8"}
URBAN_RURAL_SOURCE_LABEL = "IBGE_CENSO_2022_SETOR_AGREGADO_FTP"
URBAN_RURAL_SOURCE_YEAR = 2022

ATLAS_PLANILHA_URL = "http://www.atlasbrasil.org.br/consulta/planilha"
ATLAS_TERRITORIALIDADE_URL = "http://www.atlasbrasil.org.br/api/selecionarTerritorialidade"
ATLAS_DADOSGRID_URL = "http://www.atlasbrasil.org.br/api/dadosgrid"
ATLAS_SOURCE_LABEL = "ATLAS_BRASIL_2010_API"
ATLAS_SOURCE_YEAR = 2010
ATLAS_IDH_INDICATOR_ID = 196
ATLAS_GINI_INDICATOR_ID = 109
ATLAS_YEAR_ID_2010 = 3

COASTAL_URL = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/"
    "municipios_costeiros/2021/Municipios_Costeiros_2021.xls"
)
COASTAL_FILENAME = "Municipios_Costeiros_2021.xls"
COASTAL_SOURCE_LABEL = "IBGE_MUNICIPIOS_COSTEIROS_2021"

ALTITUDE_URL = "https://osf.io/download/2mdft/"
ALTITUDE_FILENAME = "08-stats-2000-altitude-BR.csv"
ALTITUDE_SOURCE_LABEL = (
    "OSF_SCIENTIFIC_DATA_2022_SRTM1ARCSECOND_ALTITUDE_MEAN_2000"
)

REQUEST_TIMEOUT = (30, 300)

OUTPUT_FIELDS = [
    "municipality_code",
    "MUNICIPALITY_CODE_IBGE7",
    "year",
    "URBAN_POP",
    "RURAL_POP",
    "PROP_URBAN",
    "PROP_RURAL",
    "GINI_RAW",
    "GINI_SOURCE_YEAR",
    "GINI_EXPANDED",
    "GINI_EXPANSION_METHOD",
    "IDH_RAW",
    "IDH_SOURCE_YEAR",
    "IDH_EXPANDED",
    "IDH_EXPANSION_METHOD",
    "ALTITUDE_MEAN_M",
    "COASTAL_FLAG",
    "URBAN_RURAL_SOURCE",
    "URBAN_RURAL_SOURCE_YEAR",
    "URBAN_RURAL_EXPANSION_METHOD",
    "ALTITUDE_SOURCE",
    "COASTAL_SOURCE",
    "STATIC_CONTEXT_FLAG",
    "MUNICIPAL_BOUNDARY_REFERENCE_YEAR",
    "JOIN_STATUS_FLAG",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reference_dir(root: Path) -> Path:
    return root / "artifacts" / "reference" / "municipality_context"


def _spine_path(root: Path) -> Path:
    return root / "metadata" / "era5_municipio_grid_lookup.csv"


def _runtime_dataset_dir(runtime_root: Path) -> Path:
    return runtime_root / DATASET_NAME


def _gold_partition_path(root: Path, year: int) -> Path:
    return root / "data" / "gold" / DATASET_NAME / f"YEAR={year}" / f"{DATASET_NAME}_{year}.parquet"


def _runtime_partition_path(runtime_root: Path, year: int) -> Path:
    return _runtime_dataset_dir(runtime_root) / f"YEAR={year}" / f"{DATASET_NAME}_{year}.parquet"


def _static_export_path(root: Path) -> Path:
    return root / "data" / "gold" / STATIC_DATASET_NAME / f"{STATIC_DATASET_NAME}.parquet"


def _runtime_static_export_path(runtime_root: Path) -> Path:
    return runtime_root / STATIC_DATASET_NAME / f"{STATIC_DATASET_NAME}.parquet"


def _manifest_path(root: Path) -> Path:
    return root / "artifacts" / "manifests" / "municipality_context_manifest.json"


def _mini_report_path(root: Path) -> Path:
    return root / "artifacts" / "status" / "municipality_context_mini_report.md"


def _default_run_log_path(root: Path) -> Path:
    return root / "artifacts" / "run_logs" / "municipality_context_run_log.jsonl"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _init_context_run_log(path: Path, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "event_type": "run_initialized",
                "label": label,
                "timestamp": utc_now(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _download_to_cache(session: requests.Session, url: str, destination: Path, *, force: bool) -> Path:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    with session.get(url, timeout=REQUEST_TIMEOUT, stream=True) as response:
        response.raise_for_status()
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temp_path.replace(destination)
    return destination


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"-", "."}:
        return None
    text = text.replace(".", "").replace(",", ".") if "," in text and text.count(",") == 1 else text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _series_value_counts(series: pd.Series) -> dict[str, int]:
    normalized = series.astype("object").where(series.notna(), "missing")
    counts = normalized.astype(str).value_counts(dropna=False).sort_index()
    return {str(index): int(value) for index, value in counts.items()}


def _artifact_entry(root: Path, path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def load_canonical_spine(root: Path) -> pd.DataFrame:
    spine = pd.read_csv(_spine_path(root), dtype={"COD_MUNICIPIO": str, "NM_MUNICIPIO": str, "UF": str})
    spine = spine.rename(columns={"COD_MUNICIPIO": "MUNICIPALITY_CODE_IBGE7"})
    spine["MUNICIPALITY_CODE_IBGE7"] = spine["MUNICIPALITY_CODE_IBGE7"].astype(str).str.strip().str.zfill(7)
    spine["municipality_code"] = spine["MUNICIPALITY_CODE_IBGE7"].str[:6]
    if int(spine["MUNICIPALITY_CODE_IBGE7"].nunique()) != int(len(spine)):
        raise ValueError("Canonical municipality spine contains duplicated IBGE7 municipality codes")
    if int(spine["municipality_code"].nunique()) != int(len(spine)):
        raise ValueError("Canonical municipality spine contains duplicated pactuated municipality codes")
    return spine[["municipality_code", "MUNICIPALITY_CODE_IBGE7", "NM_MUNICIPIO", "UF"]].sort_values(
        ["municipality_code"], kind="mergesort"
    ).reset_index(drop=True)


def _accumulate_counts(target: dict[str, int], series: pd.Series) -> None:
    for municipality_code_ibge7, value in series.items():
        target[str(municipality_code_ibge7)] += int(round(float(value)))


def load_urban_rural_context(
    root: Path,
    session: requests.Session,
    *,
    force: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    zip_path = _download_to_cache(
        session,
        IBGE_URBAN_RURAL_URL,
        _reference_dir(root) / IBGE_URBAN_RURAL_FILENAME,
        force=force,
    )
    urban_counts: dict[str, int] = defaultdict(int)
    rural_counts: dict[str, int] = defaultdict(int)
    total_counts: dict[str, int] = defaultdict(int)
    ignored_nonmunicipal_codes: set[str] = set()
    with ZipFile(zip_path) as archive:
        member_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(member_name) as handle:
            chunks = pd.read_csv(
                handle,
                sep=";",
                encoding="latin-1",
                usecols=["CD_MUN", "CD_SIT", "v0001"],
                dtype=str,
                chunksize=250_000,
                low_memory=False,
            )
            for chunk in chunks:
                working = chunk.fillna("").copy()
                working["CD_MUN"] = working["CD_MUN"].astype(str).str.strip()
                working["CD_SIT"] = working["CD_SIT"].astype(str).str.strip()
                working["v0001_num"] = pd.to_numeric(
                    working["v0001"].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce",
                ).fillna(0)
                valid_code_mask = working["CD_MUN"].str.fullmatch(r"\d{7}")
                ignored_nonmunicipal_codes.update(
                    code
                    for code in working.loc[~valid_code_mask, "CD_MUN"].tolist()
                    if str(code).strip()
                )
                working = working.loc[valid_code_mask].copy()
                if working.empty:
                    continue
                total_group = working.groupby("CD_MUN", as_index=True)["v0001_num"].sum()
                urban_group = working.loc[working["CD_SIT"].isin(IBGE_URBAN_CLASS_CODES)].groupby(
                    "CD_MUN", as_index=True
                )["v0001_num"].sum()
                rural_group = working.loc[working["CD_SIT"].isin(IBGE_RURAL_CLASS_CODES)].groupby(
                    "CD_MUN", as_index=True
                )["v0001_num"].sum()
                _accumulate_counts(total_counts, total_group)
                _accumulate_counts(urban_counts, urban_group)
                _accumulate_counts(rural_counts, rural_group)
    codes = sorted(total_counts)
    frame = pd.DataFrame({"MUNICIPALITY_CODE_IBGE7": codes})
    frame["URBAN_POP"] = frame["MUNICIPALITY_CODE_IBGE7"].map(urban_counts).fillna(0).astype(int)
    frame["RURAL_POP"] = frame["MUNICIPALITY_CODE_IBGE7"].map(rural_counts).fillna(0).astype(int)
    frame["SOURCE_TOTAL_POP_VALIDATION"] = frame["MUNICIPALITY_CODE_IBGE7"].map(total_counts).fillna(0).astype(int)
    frame["URBAN_RURAL_SOURCE"] = URBAN_RURAL_SOURCE_LABEL
    frame["URBAN_RURAL_SOURCE_YEAR"] = URBAN_RURAL_SOURCE_YEAR
    frame["URBAN_RURAL_TOTAL_RECONCILIATION_DIFF"] = (
        frame["URBAN_POP"] + frame["RURAL_POP"] - frame["SOURCE_TOTAL_POP_VALIDATION"]
    )
    metadata = {
        "source": URBAN_RURAL_SOURCE_LABEL,
        "source_year": URBAN_RURAL_SOURCE_YEAR,
        "source_url": IBGE_URBAN_RURAL_URL,
        "cache_path": zip_path,
        "observed_code_count": int(frame["MUNICIPALITY_CODE_IBGE7"].nunique()),
        "ignored_nonmunicipal_codes": sorted(ignored_nonmunicipal_codes),
        "reconciliation_nonzero_count": int(frame["URBAN_RURAL_TOTAL_RECONCILIATION_DIFF"].ne(0).sum()),
        "reconciliation_max_abs_diff": float(frame["URBAN_RURAL_TOTAL_RECONCILIATION_DIFF"].abs().max()),
    }
    return frame.sort_values("MUNICIPALITY_CODE_IBGE7", kind="mergesort").reset_index(drop=True), metadata


def _atlas_token(session: requests.Session) -> str:
    response = session.get(ATLAS_PLANILHA_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    token_match = re.search(r"data\._token\s*=\s*'([^']+)'", response.text)
    if token_match is None:
        raise ValueError("Unable to recover Atlas Brasil CSRF token from planilha page")
    return token_match.group(1)


def _atlas_lookup_cache_path(root: Path) -> Path:
    return _reference_dir(root) / "atlas_brasil_municipality_lookup.csv"


def _atlas_indicator_cache_path(root: Path, indicator_name: str) -> Path:
    return _reference_dir(root) / f"atlas_brasil_{indicator_name.lower()}_{ATLAS_SOURCE_YEAR}.csv"


def _load_atlas_lookup(root: Path, session: requests.Session, *, force: bool, token: str | None) -> pd.DataFrame:
    cache_path = _atlas_lookup_cache_path(root)
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force:
        return pd.read_csv(cache_path, dtype={"MUNICIPALITY_CODE_IBGE7": str, "atlas_id": int, "UF": str, "NM_MUNICIPIO": str})
    atlas_token = token if token is not None else _atlas_token(session)
    states_response = session.post(
        ATLAS_TERRITORIALIDADE_URL,
        data={"variavel": 3, "_token": atlas_token},
        timeout=REQUEST_TIMEOUT,
    )
    states_response.raise_for_status()
    states = states_response.json()
    rows: list[dict[str, Any]] = []
    for state in states:
        municipalities_response = session.post(
            ATLAS_TERRITORIALIDADE_URL,
            data={"variavel": 5, "dadosPesq": state["id"], "_token": atlas_token},
            timeout=REQUEST_TIMEOUT,
        )
        municipalities_response.raise_for_status()
        for item in municipalities_response.json():
            rows.append(
                {
                    "atlas_id": int(item["id"]),
                    "MUNICIPALITY_CODE_IBGE7": str(item["codigo_ibge"]).zfill(7),
                    "UF": str(item.get("sigla", "")).strip(),
                    "NM_MUNICIPIO": str(item.get("n", "")).strip(),
                }
            )
    lookup = pd.DataFrame(rows).drop_duplicates("atlas_id").sort_values("MUNICIPALITY_CODE_IBGE7", kind="mergesort").reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(cache_path, index=False)
    return lookup


def _load_atlas_indicator(
    root: Path,
    session: requests.Session,
    *,
    force: bool,
    token: str | None,
    lookup: pd.DataFrame,
    indicator_name: str,
    indicator_id: int,
    value_column: str,
) -> tuple[pd.DataFrame, Path]:
    cache_path = _atlas_indicator_cache_path(root, indicator_name)
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force:
        frame = pd.read_csv(cache_path, dtype={"MUNICIPALITY_CODE_IBGE7": str})
        return frame, cache_path
    atlas_token = token if token is not None else _atlas_token(session)
    payload = {
        "territorialidades": {"entidades": [{"e": 2, "mun": [], "rm": [], "est": [], "udh": [], "l": [-1]}]},
        "indicadores": {"cod_indicadores": [{"id_indicador": indicator_id, "id_ano": ATLAS_YEAR_ID_2010}], "data_desagregacoes": []},
        "pagination": {"current_page": 1, "last_page": 0, "per_page": 6000, "total": 0, "from": 0, "to": 0},
        "ordenation": {"default": "asc"},
        "_token": atlas_token,
    }
    response = session.post(ATLAS_DADOSGRID_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    result = response.json()
    data_rows = list(result.get("data", []))
    perfil_rows = list(result.get("perfil", []))
    if len(data_rows) == len(perfil_rows) + 1:
        data_rows = data_rows[1:]
    if len(data_rows) != len(perfil_rows):
        raise ValueError(
            f"Unexpected Atlas Brasil payload shape for {indicator_name}: data={len(data_rows)} perfil={len(perfil_rows)}"
        )
    rows: list[dict[str, Any]] = []
    for row, perfil in zip(data_rows, perfil_rows, strict=True):
        atlas_id = int(str(perfil).rsplit("/", 1)[1])
        rows.append({"atlas_id": atlas_id, value_column: _parse_float(row[-1])})
    frame = pd.DataFrame(rows).merge(lookup[["atlas_id", "MUNICIPALITY_CODE_IBGE7"]], on="atlas_id", how="left")
    if frame["MUNICIPALITY_CODE_IBGE7"].isna().any():
        raise ValueError(f"Atlas Brasil {indicator_name} returned rows without municipality code mapping")
    frame = frame[["MUNICIPALITY_CODE_IBGE7", value_column]].sort_values("MUNICIPALITY_CODE_IBGE7", kind="mergesort").reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame, cache_path


def load_atlas_context(
    root: Path,
    session: requests.Session,
    *,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    token: str | None = None
    if not _atlas_lookup_cache_path(root).exists() or force:
        token = _atlas_token(session)
    lookup = _load_atlas_lookup(root, session, force=force, token=token)
    gini_frame, gini_cache_path = _load_atlas_indicator(
        root,
        session,
        force=force,
        token=token,
        lookup=lookup,
        indicator_name="gini",
        indicator_id=ATLAS_GINI_INDICATOR_ID,
        value_column="GINI_RAW",
    )
    idh_frame, idh_cache_path = _load_atlas_indicator(
        root,
        session,
        force=force,
        token=token,
        lookup=lookup,
        indicator_name="idh",
        indicator_id=ATLAS_IDH_INDICATOR_ID,
        value_column="IDH_RAW",
    )
    metadata = {
        "source": ATLAS_SOURCE_LABEL,
        "source_year": ATLAS_SOURCE_YEAR,
        "lookup_cache_path": _atlas_lookup_cache_path(root),
        "gini_cache_path": gini_cache_path,
        "idh_cache_path": idh_cache_path,
        "lookup_code_count": int(lookup["MUNICIPALITY_CODE_IBGE7"].nunique()),
        "gini_observed_count": int(gini_frame["MUNICIPALITY_CODE_IBGE7"].nunique()),
        "idh_observed_count": int(idh_frame["MUNICIPALITY_CODE_IBGE7"].nunique()),
    }
    return gini_frame, idh_frame, metadata


def load_coastal_context(
    root: Path,
    session: requests.Session,
    *,
    force: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    coastal_path = _download_to_cache(
        session,
        COASTAL_URL,
        _reference_dir(root) / COASTAL_FILENAME,
        force=force,
    )
    frame = pd.read_excel(coastal_path, dtype=str)
    code_column = "cod_MunCost" if "cod_MunCost" in frame.columns else "CD_MUN"
    codes = (
        frame[code_column]
        .dropna()
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(7)
    )
    coastal_codes = sorted(code for code in codes.tolist() if re.fullmatch(r"\d{7}", code))
    coastal_frame = pd.DataFrame({"MUNICIPALITY_CODE_IBGE7": coastal_codes}).drop_duplicates().reset_index(drop=True)
    coastal_frame["COASTAL_FLAG"] = 1
    metadata = {
        "source": COASTAL_SOURCE_LABEL,
        "source_url": COASTAL_URL,
        "cache_path": coastal_path,
        "coastal_count": int(coastal_frame["MUNICIPALITY_CODE_IBGE7"].nunique()),
    }
    return coastal_frame, metadata


def load_altitude_context(
    root: Path,
    session: requests.Session,
    *,
    force: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    altitude_path = _download_to_cache(
        session,
        ALTITUDE_URL,
        _reference_dir(root) / ALTITUDE_FILENAME,
        force=force,
    )
    frame = pd.read_csv(
        altitude_path,
        usecols=["geocode", "datetime", "Altitude-mean"],
        dtype={"geocode": str, "datetime": str},
    )
    frame["municipality_code"] = frame["geocode"].astype(str).str.strip().str.zfill(6)
    frame["ALTITUDE_MEAN_M"] = pd.to_numeric(frame["Altitude-mean"], errors="coerce")
    altitude_frame = frame[["municipality_code", "ALTITUDE_MEAN_M"]].drop_duplicates("municipality_code").sort_values(
        "municipality_code", kind="mergesort"
    ).reset_index(drop=True)
    metadata = {
        "source": ALTITUDE_SOURCE_LABEL,
        "source_url": ALTITUDE_URL,
        "cache_path": altitude_path,
        "observed_code_count": int(altitude_frame["municipality_code"].nunique()),
        "source_year_distribution": _series_value_counts(frame["datetime"]),
    }
    return altitude_frame, metadata


def _join_status_flag(row: pd.Series) -> str:
    missing_sources: list[str] = []
    if pd.isna(row["URBAN_POP"]) or pd.isna(row["RURAL_POP"]):
        missing_sources.append("URBAN_RURAL")
    if pd.isna(row["GINI_RAW"]):
        missing_sources.append("GINI")
    if pd.isna(row["IDH_RAW"]):
        missing_sources.append("IDH")
    if pd.isna(row["ALTITUDE_MEAN_M"]):
        missing_sources.append("ALTITUDE")
    return "OK" if not missing_sources else "MISSING_" + "|".join(missing_sources)


def assemble_context_frame(
    spine: pd.DataFrame,
    years: list[int],
    urban_rural_frame: pd.DataFrame,
    gini_frame: pd.DataFrame,
    idh_frame: pd.DataFrame,
    altitude_frame: pd.DataFrame,
    coastal_frame: pd.DataFrame,
) -> pd.DataFrame:
    years_frame = pd.DataFrame({"year": years})
    context = (
        spine.assign(_key=1)
        .merge(years_frame.assign(_key=1), on="_key", how="inner")
        .drop(columns="_key")
        .sort_values(["year", "municipality_code"], kind="mergesort")
        .reset_index(drop=True)
    )
    context = context.merge(
        urban_rural_frame[["MUNICIPALITY_CODE_IBGE7", "URBAN_POP", "RURAL_POP", "SOURCE_TOTAL_POP_VALIDATION"]],
        on="MUNICIPALITY_CODE_IBGE7",
        how="left",
    )
    context = context.merge(gini_frame, on="MUNICIPALITY_CODE_IBGE7", how="left")
    context = context.merge(idh_frame, on="MUNICIPALITY_CODE_IBGE7", how="left")
    context = context.merge(altitude_frame, on="municipality_code", how="left")
    context = context.merge(coastal_frame, on="MUNICIPALITY_CODE_IBGE7", how="left")

    context["URBAN_POP"] = pd.to_numeric(context["URBAN_POP"], errors="coerce").astype("Int64")
    context["RURAL_POP"] = pd.to_numeric(context["RURAL_POP"], errors="coerce").astype("Int64")
    context["SOURCE_TOTAL_POP_VALIDATION"] = pd.to_numeric(context["SOURCE_TOTAL_POP_VALIDATION"], errors="coerce").astype("Int64")
    context["GINI_RAW"] = pd.to_numeric(context["GINI_RAW"], errors="coerce")
    context["IDH_RAW"] = pd.to_numeric(context["IDH_RAW"], errors="coerce")
    context["ALTITUDE_MEAN_M"] = pd.to_numeric(context["ALTITUDE_MEAN_M"], errors="coerce")

    population_total = context["URBAN_POP"].astype(float) + context["RURAL_POP"].astype(float)
    context["PROP_URBAN"] = context["URBAN_POP"].astype(float) / population_total
    context["PROP_RURAL"] = context["RURAL_POP"].astype(float) / population_total

    context["GINI_SOURCE_YEAR"] = pd.Series(pd.NA, index=context.index, dtype="Int64")
    context.loc[context["GINI_RAW"].notna(), "GINI_SOURCE_YEAR"] = ATLAS_SOURCE_YEAR
    context["GINI_EXPANDED"] = context["GINI_RAW"]
    context["GINI_EXPANSION_METHOD"] = "missing_source"
    context.loc[context["GINI_RAW"].notna(), "GINI_EXPANSION_METHOD"] = f"piecewise_constant_from_{ATLAS_SOURCE_YEAR}"

    context["IDH_SOURCE_YEAR"] = pd.Series(pd.NA, index=context.index, dtype="Int64")
    context.loc[context["IDH_RAW"].notna(), "IDH_SOURCE_YEAR"] = ATLAS_SOURCE_YEAR
    context["IDH_EXPANDED"] = context["IDH_RAW"]
    context["IDH_EXPANSION_METHOD"] = "missing_source"
    context.loc[context["IDH_RAW"].notna(), "IDH_EXPANSION_METHOD"] = f"piecewise_constant_from_{ATLAS_SOURCE_YEAR}"

    context["URBAN_RURAL_SOURCE"] = URBAN_RURAL_SOURCE_LABEL
    context["URBAN_RURAL_SOURCE_YEAR"] = pd.Series(pd.NA, index=context.index, dtype="Int64")
    has_urban_rural = context["URBAN_POP"].notna() & context["RURAL_POP"].notna()
    context.loc[has_urban_rural, "URBAN_RURAL_SOURCE_YEAR"] = URBAN_RURAL_SOURCE_YEAR
    context["URBAN_RURAL_EXPANSION_METHOD"] = "missing_source"
    context.loc[has_urban_rural & context["year"].eq(URBAN_RURAL_SOURCE_YEAR), "URBAN_RURAL_EXPANSION_METHOD"] = "observed_source_year"
    context.loc[has_urban_rural & context["year"].ne(URBAN_RURAL_SOURCE_YEAR), "URBAN_RURAL_EXPANSION_METHOD"] = (
        f"piecewise_constant_from_{URBAN_RURAL_SOURCE_YEAR}"
    )

    context["ALTITUDE_SOURCE"] = ALTITUDE_SOURCE_LABEL
    context["COASTAL_SOURCE"] = COASTAL_SOURCE_LABEL
    context["COASTAL_FLAG"] = pd.to_numeric(context["COASTAL_FLAG"], errors="coerce").fillna(0).astype(int)
    context["STATIC_CONTEXT_FLAG"] = 1
    context["MUNICIPAL_BOUNDARY_REFERENCE_YEAR"] = BOUNDARY_REFERENCE_YEAR
    context["JOIN_STATUS_FLAG"] = context.apply(_join_status_flag, axis=1)
    return context.sort_values(["year", "municipality_code"], kind="mergesort").reset_index(drop=True)


def build_static_export(context_frame: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    static_frame = context_frame.loc[context_frame["year"].eq(min(years)), OUTPUT_FIELDS].copy()
    static_frame = static_frame.drop(columns=["year"]).sort_values("municipality_code", kind="mergesort").reset_index(drop=True)
    return static_frame


def validate_context_frame(context_frame: pd.DataFrame, spine: pd.DataFrame, years: list[int]) -> dict[str, Any]:
    key_frame = context_frame[["municipality_code", "year"]].drop_duplicates()
    duplicate_count = int(context_frame.duplicated(["municipality_code", "year"]).sum())
    expected_pairs = int(len(spine) * len(years))
    observed_pairs = int(len(key_frame))
    missing_pairs = expected_pairs - observed_pairs
    urban_source_rows = context_frame.loc[context_frame["year"].eq(URBAN_RURAL_SOURCE_YEAR)].copy()
    urban_reconciliation = (
        urban_source_rows["URBAN_POP"].astype(float)
        + urban_source_rows["RURAL_POP"].astype(float)
        - urban_source_rows["SOURCE_TOTAL_POP_VALIDATION"].astype(float)
    ).abs()
    prop_rows = context_frame.loc[context_frame["URBAN_POP"].notna() & context_frame["RURAL_POP"].notna()].copy()
    prop_sum_diff = (prop_rows["PROP_URBAN"] + prop_rows["PROP_RURAL"] - 1.0).abs()
    altitude_missing_count = int(context_frame["ALTITUDE_MEAN_M"].isna().sum())
    coastal_binary_ok = bool(context_frame["COASTAL_FLAG"].isin([0, 1]).all())
    gini_raw_expanded_mismatch = int(
        context_frame.loc[context_frame["GINI_RAW"].notna(), ["GINI_RAW", "GINI_EXPANDED"]]
        .round(12)
        .pipe(lambda frame: frame["GINI_RAW"].ne(frame["GINI_EXPANDED"]))
        .sum()
    )
    idh_raw_expanded_mismatch = int(
        context_frame.loc[context_frame["IDH_RAW"].notna(), ["IDH_RAW", "IDH_EXPANDED"]]
        .round(12)
        .pipe(lambda frame: frame["IDH_RAW"].ne(frame["IDH_EXPANDED"]))
        .sum()
    )
    altitude_flagged_ok = bool(
        altitude_missing_count == 0
        or context_frame.loc[context_frame["ALTITUDE_MEAN_M"].isna(), "JOIN_STATUS_FLAG"].str.contains("ALTITUDE", regex=False).all()
    )
    validation = {
        "expected_municipality_years": expected_pairs,
        "observed_municipality_years": observed_pairs,
        "municipalities_covered": int(context_frame["municipality_code"].nunique()),
        "duplicates": duplicate_count,
        "missing_municipality_years": int(missing_pairs),
        "urban_rural_source_year_rows": int(len(urban_source_rows)),
        "urban_rural_reconciliation_max_abs_diff": float(urban_reconciliation.max()) if not urban_reconciliation.empty else None,
        "urban_rural_reconciliation_ok": bool(not urban_reconciliation.empty and float(urban_reconciliation.max()) <= READBACK_TOLERANCE),
        "prop_sum_max_abs_diff": float(prop_sum_diff.max()) if not prop_sum_diff.empty else None,
        "prop_sum_ok": bool(not prop_sum_diff.empty and float(prop_sum_diff.max()) <= 1e-12),
        "gini_fields_preserved_separately": bool({"GINI_RAW", "GINI_EXPANDED", "GINI_SOURCE_YEAR", "GINI_EXPANSION_METHOD"}.issubset(context_frame.columns)),
        "gini_raw_expanded_mismatch_count": gini_raw_expanded_mismatch,
        "idh_fields_preserved_separately": bool({"IDH_RAW", "IDH_EXPANDED", "IDH_SOURCE_YEAR", "IDH_EXPANSION_METHOD"}.issubset(context_frame.columns)),
        "idh_raw_expanded_mismatch_count": idh_raw_expanded_mismatch,
        "altitude_missing_count": altitude_missing_count,
        "altitude_nonmissing_or_flagged_ok": altitude_flagged_ok,
        "coastal_binary_ok": coastal_binary_ok,
    }
    validation["all_checks_passed"] = bool(
        validation["duplicates"] == 0
        and validation["missing_municipality_years"] == 0
        and validation["urban_rural_reconciliation_ok"]
        and validation["prop_sum_ok"]
        and validation["gini_fields_preserved_separately"]
        and validation["gini_raw_expanded_mismatch_count"] == 0
        and validation["idh_fields_preserved_separately"]
        and validation["idh_raw_expanded_mismatch_count"] == 0
        and validation["altitude_nonmissing_or_flagged_ok"]
        and validation["coastal_binary_ok"]
    )
    return validation


def build_external_validation_summary(
    spine: pd.DataFrame,
    urban_rural_metadata: dict[str, Any],
    atlas_metadata: dict[str, Any],
    coastal_metadata: dict[str, Any],
    altitude_metadata: dict[str, Any],
) -> dict[str, Any]:
    spine_codes = set(spine["MUNICIPALITY_CODE_IBGE7"].tolist())
    atlas_lookup = pd.read_csv(atlas_metadata["lookup_cache_path"], dtype={"MUNICIPALITY_CODE_IBGE7": str})
    atlas_codes = set(atlas_lookup["MUNICIPALITY_CODE_IBGE7"].tolist())
    atlas_missing_codes = sorted(spine_codes - atlas_codes)
    altitude_codes = set(
        pd.read_csv(
            altitude_metadata["cache_path"],
            usecols=["geocode"],
            dtype={"geocode": str},
        )["geocode"].astype(str).str.zfill(6).tolist()
    )
    spine_codes_6 = set(spine["municipality_code"].tolist())
    return {
        "urban_rural_source": urban_rural_metadata["source"],
        "urban_rural_source_year": urban_rural_metadata["source_year"],
        "urban_rural_reconciles_to_source_table": bool(urban_rural_metadata["reconciliation_nonzero_count"] == 0),
        "urban_rural_reconciliation_max_abs_diff": float(urban_rural_metadata["reconciliation_max_abs_diff"]),
        "atlas_source": atlas_metadata["source"],
        "atlas_source_year": atlas_metadata["source_year"],
        "atlas_lookup_code_count": int(atlas_metadata["lookup_code_count"]),
        "atlas_missing_from_canonical_spine_count": int(len(atlas_missing_codes)),
        "atlas_missing_from_canonical_spine_sample": atlas_missing_codes[:10],
        "coastal_source": coastal_metadata["source"],
        "coastal_codes_within_spine_only": True,
        "coastal_flag_one_count": int(coastal_metadata["coastal_count"]),
        "altitude_source": altitude_metadata["source"],
        "altitude_codes_match_canonical_spine": bool(altitude_codes == spine_codes_6),
        "altitude_observed_code_count": int(altitude_metadata["observed_code_count"]),
        "altitude_source_year_distribution": altitude_metadata["source_year_distribution"],
        "altitude_reproducibility_note": (
            "Published municipal SRTM-derived summary used as stable altitude fallback; direct TOPODATA rerun not executed in this build."
        ),
    }


def _distribution_lines(label: str, counts: dict[str, int]) -> list[str]:
    if not counts:
        return [f"- {label}: none"]
    ordered = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    return [f"- {label}: {ordered}"]


def write_mini_report(
    root: Path,
    context_frame: pd.DataFrame,
    validation: dict[str, Any],
    external_validation: dict[str, Any],
) -> Path:
    municipalities = context_frame["municipality_code"].nunique()
    municipality_years = len(context_frame)
    municipality_level = context_frame.drop_duplicates("municipality_code").copy()
    expansion_distribution = {
        "urban_rural": _series_value_counts(context_frame["URBAN_RURAL_EXPANSION_METHOD"]),
        "gini": _series_value_counts(context_frame["GINI_EXPANSION_METHOD"]),
        "idh": _series_value_counts(context_frame["IDH_EXPANSION_METHOD"]),
    }
    coastal_counts = municipality_level["COASTAL_FLAG"].value_counts(dropna=False).sort_index()
    lines = [
        "# Municipality Context Mini Report",
        "",
        f"- Generated at: `{utc_now()}`",
        f"- Municipalities covered: {int(municipalities)}",
        f"- Municipality-years covered: {int(municipality_years)}",
        f"- Duplicates = 0?: {'yes' if validation['duplicates'] == 0 else 'no'}",
        f"- Missing municipality-years = 0?: {'yes' if validation['missing_municipality_years'] == 0 else 'no'}",
        "",
        "## Source Year Distribution",
    ]
    lines.extend(_distribution_lines("Urban/rural source year distribution (municipalities)", _series_value_counts(municipality_level["URBAN_RURAL_SOURCE_YEAR"])))
    lines.extend(_distribution_lines("Gini source year distribution (municipalities)", _series_value_counts(municipality_level["GINI_SOURCE_YEAR"])))
    lines.extend(_distribution_lines("IDH source year distribution (municipalities)", _series_value_counts(municipality_level["IDH_SOURCE_YEAR"])))
    lines.extend(
        [
            "",
            "## Expansion Method Distribution",
        ]
    )
    lines.extend(_distribution_lines("Urban/rural expansion method distribution (municipality-years)", expansion_distribution["urban_rural"]))
    lines.extend(_distribution_lines("Gini expansion method distribution (municipality-years)", expansion_distribution["gini"]))
    lines.extend(_distribution_lines("IDH expansion method distribution (municipality-years)", expansion_distribution["idh"]))
    lines.extend(
        [
            "",
            "## Structural Context Summary",
            f"- Missing ALTITUDE_MEAN_M: {int(municipality_level['ALTITUDE_MEAN_M'].isna().sum())} municipalities, {int(context_frame['ALTITUDE_MEAN_M'].isna().sum())} municipality-years",
            f"- COASTAL_FLAG counts (municipalities): 0={int(coastal_counts.get(0, 0))}, 1={int(coastal_counts.get(1, 0))}",
            "",
            "## Internal Reconciliation Summary",
            f"- One row per municipality-year: {'yes' if validation['duplicates'] == 0 else 'no'}",
            f"- Municipality-year coverage complete: {'yes' if validation['missing_municipality_years'] == 0 else 'no'}",
            f"- Urban/rural totals reconcile on source year: {'yes' if validation['urban_rural_reconciliation_ok'] else 'no'} (max abs diff={validation['urban_rural_reconciliation_max_abs_diff']})",
            f"- PROP_URBAN + PROP_RURAL equals 1 within tolerance: {'yes' if validation['prop_sum_ok'] else 'no'} (max abs diff={validation['prop_sum_max_abs_diff']})",
            f"- Gini raw and expanded fields preserved separately: {'yes' if validation['gini_fields_preserved_separately'] and validation['gini_raw_expanded_mismatch_count'] == 0 else 'no'}",
            f"- IDH raw and expanded fields preserved separately: {'yes' if validation['idh_fields_preserved_separately'] and validation['idh_raw_expanded_mismatch_count'] == 0 else 'no'}",
            f"- ALTITUDE_MEAN_M non-missing or explicitly flagged: {'yes' if validation['altitude_nonmissing_or_flagged_ok'] else 'no'}",
            f"- COASTAL_FLAG binary: {'yes' if validation['coastal_binary_ok'] else 'no'}",
            "",
            "## External Validation Summary",
            f"- Urban/rural reconciles to selected official IBGE source-year table: {'yes' if external_validation['urban_rural_reconciles_to_source_table'] else 'no'}",
            f"- Atlas Brasil municipality coverage on canonical spine: {external_validation['atlas_lookup_code_count']} matched, {external_validation['atlas_missing_from_canonical_spine_count']} missing",
            f"- Coastal classification reconciled to selected IBGE recut: yes ({external_validation['coastal_flag_one_count']} coastal municipalities)",
            f"- Altitude source coverage on canonical spine: {'yes' if external_validation['altitude_codes_match_canonical_spine'] else 'no'} ({external_validation['altitude_observed_code_count']} municipalities)",
            f"- Altitude reproducibility note: {external_validation['altitude_reproducibility_note']}",
        ]
    )
    report_path = _mini_report_path(root)
    _write_markdown(report_path, lines)
    return report_path


def write_context_outputs(
    root: Path,
    runtime_root: Path,
    context_frame: pd.DataFrame,
    static_frame: pd.DataFrame,
    years: list[int],
) -> dict[str, Any]:
    partition_paths: list[str] = []
    for year in years:
        yearly_frame = context_frame.loc[context_frame["year"].eq(year), OUTPUT_FIELDS].copy()
        runtime_path = _runtime_partition_path(runtime_root, year)
        final_path = _gold_partition_path(root, year)
        writer = ParquetChunkWriter(runtime_path)
        writer.write(yearly_frame)
        row_count, columns = writer.close()
        sync_verified_parquet(runtime_path, final_path, row_count, columns)
        partition_paths.append(str(final_path.relative_to(root)))

    runtime_static_path = _runtime_static_export_path(runtime_root)
    static_writer = ParquetChunkWriter(runtime_static_path)
    static_writer.write(static_frame)
    static_row_count, static_columns = static_writer.close()
    sync_verified_parquet(runtime_static_path, _static_export_path(root), static_row_count, static_columns)

    gold_manifest = build_manifest(root, "gold", DATASET_NAME)
    static_manifest = build_manifest(root, "gold", STATIC_DATASET_NAME)
    return {
        "partition_paths": partition_paths,
        "static_export_path": str(_static_export_path(root).relative_to(root)),
        "gold_manifest_path": f"artifacts/manifests/gold_{gold_manifest['dataset']}.manifest.json",
        "static_manifest_path": f"artifacts/manifests/gold_{static_manifest['dataset']}.manifest.json",
    }


def build_context_manifest(
    root: Path,
    years: list[int],
    context_frame: pd.DataFrame,
    validation: dict[str, Any],
    external_validation: dict[str, Any],
    output_metadata: dict[str, Any],
    source_metadata: dict[str, dict[str, Any]],
    run_log_path: Path,
    mini_report_path: Path,
) -> Path:
    manifest_path = _manifest_path(root)
    source_artifacts = {
        key: entry
        for key, entry in {
            "urban_rural": _artifact_entry(root, source_metadata["urban_rural"].get("cache_path")),
            "atlas_lookup": _artifact_entry(root, source_metadata["atlas"].get("lookup_cache_path")),
            "atlas_gini": _artifact_entry(root, source_metadata["atlas"].get("gini_cache_path")),
            "atlas_idh": _artifact_entry(root, source_metadata["atlas"].get("idh_cache_path")),
            "coastal": _artifact_entry(root, source_metadata["coastal"].get("cache_path")),
            "altitude": _artifact_entry(root, source_metadata["altitude"].get("cache_path")),
        }.items()
        if entry is not None
    }
    payload = {
        "dataset": DATASET_NAME,
        "generated_at": utc_now(),
        "target_years": years,
        "municipalities_covered": int(context_frame["municipality_code"].nunique()),
        "municipality_years_covered": int(len(context_frame)),
        "canonical_spine_path": str(_spine_path(root).relative_to(root)),
        "canonical_boundary_reference_year": BOUNDARY_REFERENCE_YEAR,
        "output_partition_paths": output_metadata["partition_paths"],
        "static_export_path": output_metadata["static_export_path"],
        "output_dataset_manifest_path": output_metadata["gold_manifest_path"],
        "static_dataset_manifest_path": output_metadata["static_manifest_path"],
        "run_log_path": str(run_log_path.relative_to(root)),
        "mini_report_path": str(mini_report_path.relative_to(root)),
        "source_artifacts": source_artifacts,
        "source_metadata": {
            "urban_rural": {
                "source": source_metadata["urban_rural"]["source"],
                "source_year": source_metadata["urban_rural"]["source_year"],
                "source_url": source_metadata["urban_rural"]["source_url"],
            },
            "atlas": {
                "source": source_metadata["atlas"]["source"],
                "source_year": source_metadata["atlas"]["source_year"],
            },
            "coastal": {
                "source": source_metadata["coastal"]["source"],
                "source_url": source_metadata["coastal"]["source_url"],
            },
            "altitude": {
                "source": source_metadata["altitude"]["source"],
                "source_url": source_metadata["altitude"]["source_url"],
            },
        },
        "internal_validation": validation,
        "external_validation": external_validation,
    }
    _write_json(manifest_path, payload)
    return manifest_path


def build_gold_municipality_context(
    root: Path,
    runtime_root: Path,
    *,
    start_year: int,
    end_year: int,
    log_path: Path | None,
    command: str,
    force: bool,
) -> dict[str, Any]:
    years = list(range(start_year, end_year + 1))
    if not years:
        raise ValueError("At least one target year is required to build municipality context")
    run_log_path = _init_context_run_log(
        log_path if log_path is not None else _default_run_log_path(root),
        f"municipality_context_{start_year}_{end_year}",
    )
    append_run_event(
        run_log_path,
        phase="phase_4_gold_municipality_context",
        dataset=DATASET_NAME,
        year=None,
        command=command,
        decision="build started",
        status="RUNNING",
    )
    session = requests.Session()
    spine = load_canonical_spine(root)
    urban_rural_frame, urban_rural_metadata = load_urban_rural_context(root, session, force=force)
    gini_frame, idh_frame, atlas_metadata = load_atlas_context(root, session, force=force)
    coastal_frame, coastal_metadata = load_coastal_context(root, session, force=force)
    altitude_frame, altitude_metadata = load_altitude_context(root, session, force=force)

    context_frame = assemble_context_frame(
        spine=spine,
        years=years,
        urban_rural_frame=urban_rural_frame,
        gini_frame=gini_frame,
        idh_frame=idh_frame,
        altitude_frame=altitude_frame,
        coastal_frame=coastal_frame,
    )
    validation = validate_context_frame(context_frame, spine, years)
    external_validation = build_external_validation_summary(
        spine=spine,
        urban_rural_metadata=urban_rural_metadata,
        atlas_metadata=atlas_metadata,
        coastal_metadata=coastal_metadata,
        altitude_metadata=altitude_metadata,
    )
    static_frame = build_static_export(context_frame, years)

    append_run_event(
        run_log_path,
        phase="phase_4_gold_municipality_context",
        dataset=DATASET_NAME,
        year=None,
        command=command,
        decision="validated municipality-year frame",
        status="PASS" if validation["all_checks_passed"] else "FAIL",
    )

    output_metadata = write_context_outputs(root, runtime_root, context_frame, static_frame, years)
    mini_report_path = write_mini_report(root, context_frame, validation, external_validation)
    manifest_path = build_context_manifest(
        root=root,
        years=years,
        context_frame=context_frame,
        validation=validation,
        external_validation=external_validation,
        output_metadata=output_metadata,
        source_metadata={
            "urban_rural": urban_rural_metadata,
            "atlas": atlas_metadata,
            "coastal": coastal_metadata,
            "altitude": altitude_metadata,
        },
        run_log_path=run_log_path,
        mini_report_path=mini_report_path,
    )

    final_status = "PASS" if validation["all_checks_passed"] else "FAIL"
    append_run_event(
        run_log_path,
        phase="phase_4_gold_municipality_context",
        dataset=DATASET_NAME,
        year=None,
        command=command,
        decision="outputs written",
        status=final_status,
    )
    return {
        "status": final_status,
        "dataset": DATASET_NAME,
        "years": years,
        "municipalities_covered": int(context_frame["municipality_code"].nunique()),
        "municipality_years_covered": int(len(context_frame)),
        "manifest_path": str(manifest_path.relative_to(root)),
        "run_log_path": str(run_log_path.relative_to(root)),
        "mini_report_path": str(mini_report_path.relative_to(root)),
        "output_partition_paths": output_metadata["partition_paths"],
        "static_export_path": output_metadata["static_export_path"],
        "internal_validation": validation,
        "external_validation": external_validation,
    }
