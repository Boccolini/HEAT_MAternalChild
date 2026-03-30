from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from ftplib import FTP
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .manifest import build_manifest
from .era5_land import extract_era5_land_dataset
from .raw_sources import (
    DATASUS_FTP_HOST,
    SIM_DORES_DIR,
    SIH_RD_DIR,
    UF_CODES,
    DownloadSpec,
    datasus_popsvs_spec,
    ibge_spec,
    ibge_sidra_population_spec,
    inmet_spec,
    sinasc_open_data_spec,
    IBGE_POPULATION_TOTAL_YEAR_URLS,
)
from .run_log import append_run_event


Builder = Callable[[list[int]], list[DownloadSpec]]

USER_AGENT = "heat-mnch/0.1 (+official raw extraction)"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ftp_list(host: str, directory: str) -> list[str]:
    ftp = FTP(host, timeout=120)
    ftp.login()
    try:
        ftp.cwd(directory)
        return sorted(ftp.nlst())
    finally:
        try:
            ftp.quit()
        except OSError:
            ftp.close()


def _normalize_catalog(entries: list[str]) -> dict[str, str]:
    return {entry.upper(): entry for entry in entries}


def _datasus_specs(
    directory: str,
    prefix_builder: Callable[[int, str], str],
    target_root: str,
    years: list[int],
) -> list[DownloadSpec]:
    catalog = _normalize_catalog(_ftp_list(DATASUS_FTP_HOST, directory))
    specs: list[DownloadSpec] = []
    missing: list[str] = []
    for year in years:
        for uf in UF_CODES:
            expected_prefix = prefix_builder(year, uf).upper()
            matched = next((catalog[name] for name in catalog if name.startswith(expected_prefix)), None)
            if matched is None:
                missing.append(expected_prefix)
                continue
            specs.append(
                DownloadSpec(
                    source_name="datasus",
                    dataset=target_root.split("/")[-1],
                    year=year,
                    url=f"ftp://{DATASUS_FTP_HOST}{directory}/{matched}",
                    target_relative_path=f"data/raw/{target_root}/{year}/{matched}",
                )
            )
    if missing:
        missing_preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"Missing files in official catalog for {directory}: {missing_preview}")
    return specs


def build_sim_specs(years: list[int]) -> list[DownloadSpec]:
    return _datasus_specs(
        directory=SIM_DORES_DIR,
        prefix_builder=lambda year, uf: f"DO{uf}{year}",
        target_root="sim",
        years=years,
    )


def build_sinasc_specs(years: list[int]) -> list[DownloadSpec]:
    return [sinasc_open_data_spec(year) for year in years]


def build_sih_specs(years: list[int]) -> list[DownloadSpec]:
    catalog = _normalize_catalog(_ftp_list(DATASUS_FTP_HOST, SIH_RD_DIR))
    specs: list[DownloadSpec] = []
    missing: list[str] = []
    for year in years:
        year_suffix = year % 100
        for month in range(1, 13):
            for uf in UF_CODES:
                expected_prefix = f"RD{uf}{year_suffix:02d}{month:02d}".upper()
                matched = next((catalog[name] for name in catalog if name.startswith(expected_prefix)), None)
                if matched is None:
                    missing.append(expected_prefix)
                    continue
                specs.append(
                    DownloadSpec(
                        source_name="datasus",
                        dataset="sih",
                        year=year,
                        url=f"ftp://{DATASUS_FTP_HOST}{SIH_RD_DIR}/{matched}",
                        target_relative_path=f"data/raw/sih/{year}/{matched}",
                    )
                )
    if missing:
        missing_preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"Missing SIH monthly files in official catalog: {missing_preview}")
    return specs


def build_inmet_specs(years: list[int]) -> list[DownloadSpec]:
    return [inmet_spec(year) for year in years]


def build_ibge_specs(years: list[int]) -> list[DownloadSpec]:
    missing = [year for year in years if year not in IBGE_POPULATION_TOTAL_YEAR_URLS]
    if missing:
        raise FileNotFoundError(f"Missing IBGE official mapping for years: {missing}")
    specs: list[DownloadSpec] = []
    for year in years:
        specs.extend(
            [
                ibge_spec(year),
                ibge_sidra_population_spec(year),
                datasus_popsvs_spec(year),
            ]
        )
    return specs


BUILDERS: dict[str, Builder] = {
    "sim": build_sim_specs,
    "sinasc": build_sinasc_specs,
    "sih": build_sih_specs,
    "inmet": build_inmet_specs,
    "era5_land": lambda years: [],
    "ibge": build_ibge_specs,
}


def _download_http(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=180) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)


def _download_ftp(url: str, destination: Path) -> None:
    parsed = urlparse(url)
    remote_path = Path(parsed.path)
    ftp = FTP(parsed.hostname or "", timeout=180)
    ftp.login()
    try:
        ftp.cwd(str(remote_path.parent))
        with destination.open("wb") as handle:
            ftp.retrbinary(f"RETR {remote_path.name}", handle.write, blocksize=1024 * 1024)
    finally:
        try:
            ftp.quit()
        except OSError:
            ftp.close()


def _download_with_retry(url: str, destination: Path, retries: int = 3) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if url.startswith("ftp://"):
                _download_ftp(url, destination)
            else:
                _download_http(url, destination)
            return
        except Exception as exc:  # pragma: no cover - network failure path
            last_error = exc
            if destination.exists():
                destination.unlink()
            if attempt == retries:
                break
            time.sleep(attempt * 2)
    if last_error is not None:
        raise last_error


def _validate_binary_file(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        head = handle.read(16)
    return {
        "readable": True,
        "size_bytes": path.stat().st_size,
        "head_hex": head.hex(),
    }


def _sync_from_runtime(staging_path: Path, final_path: Path) -> dict[str, object]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(staging_path, final_path)
    staging_stats = staging_path.stat()
    final_stats = final_path.stat()
    if final_stats.st_size != staging_stats.st_size:
        raise IOError(f"Size mismatch after sync for {final_path}")
    return _validate_binary_file(final_path)


def _materialize_spec(
    root: Path,
    runtime_root: Path,
    spec: DownloadSpec,
    force: bool,
) -> dict[str, object]:
    final_path = root / spec.target_relative_path
    runtime_path = runtime_root / spec.target_relative_path
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists() and not force:
        validation = _validate_binary_file(final_path)
        return {
            "kind": "skipped_existing",
            "record": {
                **asdict(spec),
                "final_path": str(final_path.relative_to(root)),
                "validation": validation,
            },
            "size_bytes": int(validation["size_bytes"]),
        }

    temp_path = runtime_path.with_suffix(runtime_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()
    _download_with_retry(spec.url, temp_path)
    validation = _sync_from_runtime(temp_path, final_path)
    temp_path.unlink(missing_ok=True)
    return {
        "kind": "downloads",
        "record": {
            **asdict(spec),
            "final_path": str(final_path.relative_to(root)),
            "validation": validation,
            "downloaded_at": _utc_now(),
        },
        "size_bytes": int(validation["size_bytes"]),
    }


def extract_dataset(
    root: Path,
    runtime_root: Path,
    dataset: str,
    years: list[int],
    log_path: Path,
    command: str,
    force: bool = False,
    max_workers: int = 1,
) -> dict[str, object]:
    if dataset == "era5_land":
        return extract_era5_land_dataset(
            root=root,
            runtime_root=runtime_root,
            years=years,
            log_path=log_path,
            command=command,
            force=force,
            max_workers=max_workers,
        )
    builder = BUILDERS[dataset]
    specs = builder(years)
    report: dict[str, object] = {
        "dataset": dataset,
        "requested_years": years,
        "generated_at": _utc_now(),
        "force": force,
        "max_workers": max_workers,
        "runtime_root": str(runtime_root),
        "downloads": [],
        "skipped_existing": [],
    }
    year_counts: dict[int, int] = {year: 0 for year in years}
    year_bytes: dict[int, int] = {year: 0 for year in years}

    if max_workers <= 1:
        for spec in specs:
            result = _materialize_spec(root, runtime_root, spec, force)
            report[result["kind"]].append(result["record"])
            year_counts[spec.year] += 1
            year_bytes[spec.year] += int(result["size_bytes"])
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_materialize_spec, root, runtime_root, spec, force): spec
                for spec in specs
            }
            for future in as_completed(future_map):
                spec = future_map[future]
                result = future.result()
                report[result["kind"]].append(result["record"])
                year_counts[spec.year] += 1
                year_bytes[spec.year] += int(result["size_bytes"])

    manifest = build_manifest(root, "raw", "climate" if dataset == "inmet" else "population" if dataset == "ibge" else dataset)
    report["manifest_path"] = f"artifacts/manifests/raw_{manifest['dataset']}.manifest.json"
    for year in years:
        append_run_event(
            log_path,
            phase="phase_1_raw_extraction",
            dataset=dataset,
            year=year,
            command=f"extract-raw {dataset} {year}",
            decision=f"materialized {year_counts[year]} files totaling {year_bytes[year]} bytes",
            status="completed" if year_counts[year] > 0 else "blocked",
        )
    report_path = root / "artifacts" / "status" / f"raw_{dataset}_extraction_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
