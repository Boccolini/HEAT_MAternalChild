from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from .config import load_round_config, load_step7_escalation_config, load_step7_route_catalog
from .registry import (
    analysis_step5_root,
    analysis_step6_root,
    analysis_step7_root,
    analysis_step7_runtime_root,
    ensure_step7_directory_structure,
    incremental_publication_root,
)
from .step2 import OVERDISPERSION_THRESHOLD, _dispersion_adjusted_result, _pooled_continuous_fe_proxy
from .step6 import build_step6_absolute_burden


STEP7_NAME = "step7_escalation_round"
STEP7_PUBLICATION_FILES = {
    "14_step7_escalation_scope.md": "# Step 7 Escalation Scope\n\nDocument the escalation scope and route classes here.\n",
    "15_step7_headline_group_surface.md": "# Step 7 Headline-Group Surface\n\nDocument the headline-group surface here.\n",
    "16_step7_all_candidate_surface.md": "# Step 7 All-Candidate Surface\n\nDocument the complete candidate surface here.\n",
}
COUNT_ESTIMATE_SCALE = "rate_ratio_per_interquartile_range_increase"
SEVERITY_ESTIMATE_SCALE = "mean_difference_per_interquartile_range_increase"
WORKBOOK_HEADER_FILL = PatternFill(fill_type="solid", fgColor="D9EAF7")
WORKBOOK_EVENT_FILL = PatternFill(fill_type="solid", fgColor="D9EAD3")
WORKBOOK_RELATIVE_ONLY_FILL = PatternFill(fill_type="solid", fgColor="FCE5CD")
WORKBOOK_FALLBACK_FILL = PatternFill(fill_type="solid", fgColor="F4CCCC")
STEP7_EXECUTION_MODE = "round_robin"
STEP7_COOLDOWN_SECONDS = 5
STEP7_RUNTIME_MANIFEST = "step7_round_robin_manifest.json"
STEP7_RUNTIME_JOB_STATUS = "step7_job_status.csv"
STEP7_JOB_STATUS_PENDING = "pending"
STEP7_JOB_STATUS_RUNNING = "running"
STEP7_JOB_STATUS_COMPLETED = "completed"
STEP7_JOB_STATUS_FAILED = "failed"
STEP7_JOB_STATUS_SKIPPED = "skipped"
STEP7_FINAL_JOB_STATUSES = {
    STEP7_JOB_STATUS_COMPLETED,
    STEP7_JOB_STATUS_FAILED,
    STEP7_JOB_STATUS_SKIPPED,
}
STEP7_BASE_JOB_ATTEMPT_LIMIT = 2
STEP7_CANDIDATE_YEAR_ATTEMPT_LIMIT = 2
STEP7_REDUCER_ATTEMPT_LIMIT = 1
STEP7_ASSEMBLY_ATTEMPT_LIMIT = 1
STEP7_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_MAX_THREADS": "1",
}
STEP7_EVENT_CANDIDATES_IN_RING = [
    "preterm_birth_count",
    "stillbirth_primary_gte28_count",
    "neonatal_death_count",
    "very_preterm_birth_count",
    "stillbirth_sensitivity_gte22_count",
    "low_birth_weight_lt2500_count",
    "gestational_days_lost_below_37_weeks",
]
UF_TO_MACROREGION = {
    "AC": "North",
    "AL": "Northeast",
    "AP": "North",
    "AM": "North",
    "BA": "Northeast",
    "CE": "Northeast",
    "DF": "Central-West",
    "ES": "Southeast",
    "GO": "Central-West",
    "MA": "Northeast",
    "MT": "Central-West",
    "MS": "Central-West",
    "MG": "Southeast",
    "PA": "North",
    "PB": "Northeast",
    "PR": "South",
    "PE": "Northeast",
    "PI": "Northeast",
    "RJ": "Southeast",
    "RN": "Northeast",
    "RS": "South",
    "RO": "North",
    "RR": "North",
    "SC": "South",
    "SP": "Southeast",
    "SE": "Northeast",
    "TO": "North",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _step7_cooldown_seconds() -> int:
    raw_value = os.environ.get("HEAT_MNCH_STEP7_COOLDOWN_SECONDS", str(STEP7_COOLDOWN_SECONDS))
    try:
        return max(int(raw_value), 0)
    except ValueError:
        return STEP7_COOLDOWN_SECONDS


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _optional_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def _normalize_municipality_code(series: pd.Series, width: int = 6) -> pd.Series:
    values: list[str] = []
    for value in series.tolist():
        text = "" if value is None or pd.isna(value) else str(value)
        digits = "".join(character for character in text if character.isdigit())
        values.append(digits.zfill(width)[:width])
    return pd.Series(values, index=series.index, dtype="string")


def _bool_like(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _step7_spec(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    del config_path
    payload = load_step7_escalation_config(root)
    return dict(payload.get("step7_spec", {}))


def _route_catalog_frame(root: Path, config_path: Path | None = None) -> pd.DataFrame:
    del config_path
    payload = load_step7_route_catalog(root)
    return pd.DataFrame(payload.get("analysis_units", []))


def _step7_runtime_paths(root: Path) -> dict[str, Path]:
    runtime_root = analysis_step7_runtime_root(root)
    return {
        "runtime_root": runtime_root,
        "manifest_path": runtime_root / STEP7_RUNTIME_MANIFEST,
        "job_status_path": runtime_root / STEP7_RUNTIME_JOB_STATUS,
        "logs_root": runtime_root / "logs",
        "errors_root": runtime_root / "errors",
        "base_live_birth_root": runtime_root / "base" / "live_birth",
        "base_stillbirth_root": runtime_root / "base" / "stillbirth",
        "base_neonatal_root": runtime_root / "base" / "neonatal",
        "candidate_year_root": runtime_root / "candidate_year",
        "reduced_root": runtime_root / "reduced",
    }


def _ensure_step7_runtime_directories(root: Path) -> dict[str, Path]:
    paths = _step7_runtime_paths(root)
    for path in paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    return paths


def _job_output_relpath(root: Path, job: dict[str, Any]) -> str:
    source_family = _optional_string(job.get("source_family"))
    analysis_unit_id = _optional_string(job.get("analysis_unit_id"))
    year = job.get("source_year")
    runtime_paths = _step7_runtime_paths(root)
    if _optional_string(job.get("job_class")) == "initialize":
        return _relative(root, analysis_step7_root(root) / "step7_candidate_catalog.csv")
    if _optional_string(job.get("job_class")) == "base_year":
        if source_family == "live_birth":
            return _relative(root, runtime_paths["base_live_birth_root"] / f"year={year}.parquet")
        if source_family == "stillbirth":
            return _relative(root, runtime_paths["base_stillbirth_root"] / f"year={year}.parquet")
        if source_family == "neonatal":
            return _relative(root, runtime_paths["base_neonatal_root"] / f"year={year}.parquet")
    if _optional_string(job.get("job_class")) == "candidate_year":
        return _relative(root, runtime_paths["candidate_year_root"] / analysis_unit_id / f"year={year}.parquet")
    if _optional_string(job.get("job_class")) == "reducer":
        return _relative(root, runtime_paths["reduced_root"] / f"{analysis_unit_id}.json")
    if _optional_string(job.get("job_id")) == "assemble_event_tables":
        return _relative(root, analysis_step7_root(root) / "step7_event_level_results.csv")
    if _optional_string(job.get("job_id")) == "assemble_surfaces":
        return _relative(root, analysis_step7_root(root) / "step7_all_candidate_surface.csv")
    if _optional_string(job.get("job_id")) == "assemble_publication_cards":
        return _relative(root, analysis_step7_root(root) / "step7_publication_cards.md")
    if _optional_string(job.get("job_id")) == "assemble_report_and_workbook":
        return _relative(root, analysis_step7_root(root) / "step7_report.md")
    return ""


def _job_error_relpath(root: Path, job_id: str) -> str:
    return _relative(root, _step7_runtime_paths(root)["errors_root"] / f"{job_id}.txt")


def _job_log_path(root: Path, job_id: str) -> Path:
    return _step7_runtime_paths(root)["logs_root"] / f"{job_id}.log"


def _job_error_path(root: Path, job_id: str) -> Path:
    return _step7_runtime_paths(root)["errors_root"] / f"{job_id}.txt"


def _base_year_output_path(root: Path, source_family: str, year: int) -> Path:
    runtime_paths = _step7_runtime_paths(root)
    if source_family == "live_birth":
        return runtime_paths["base_live_birth_root"] / f"year={year}.parquet"
    if source_family == "stillbirth":
        return runtime_paths["base_stillbirth_root"] / f"year={year}.parquet"
    if source_family == "neonatal":
        return runtime_paths["base_neonatal_root"] / f"year={year}.parquet"
    raise ValueError(f"Unsupported Step 7 source family {source_family!r}")


def _candidate_year_output_path(root: Path, analysis_unit_id: str, year: int) -> Path:
    runtime_paths = _step7_runtime_paths(root)
    return runtime_paths["candidate_year_root"] / analysis_unit_id / f"year={year}.parquet"


def _reduced_json_path(root: Path, analysis_unit_id: str) -> Path:
    return _step7_runtime_paths(root)["reduced_root"] / f"{analysis_unit_id}.json"


def _reduced_parquet_path(root: Path, analysis_unit_id: str) -> Path:
    return _step7_runtime_paths(root)["reduced_root"] / f"{analysis_unit_id}.parquet"


def _manifest_job_counts(manifest: dict[str, Any]) -> dict[str, int]:
    jobs = list(manifest.get("jobs", []))
    completed_count = sum(1 for job in jobs if _optional_string(job.get("status")) == STEP7_JOB_STATUS_COMPLETED)
    failed_count = sum(1 for job in jobs if _optional_string(job.get("status")) == STEP7_JOB_STATUS_FAILED)
    skipped_count = sum(1 for job in jobs if _optional_string(job.get("status")) == STEP7_JOB_STATUS_SKIPPED)
    required_failed_count = sum(
        1
        for job in jobs
        if _optional_string(job.get("status")) == STEP7_JOB_STATUS_FAILED and _bool_like(job.get("required_flag"))
    )
    return {
        "job_count": len(jobs),
        "completed_count": completed_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "required_failed_count": required_failed_count,
    }


def _refresh_manifest_counts(manifest: dict[str, Any]) -> dict[str, Any]:
    counts = _manifest_job_counts(manifest)
    manifest.update(counts)
    jobs = list(manifest.get("jobs", []))
    if all(_optional_string(job.get("status")) in STEP7_FINAL_JOB_STATUSES for job in jobs):
        manifest["status"] = "COMPLETE"
    elif any(_optional_string(job.get("status")) == STEP7_JOB_STATUS_RUNNING for job in jobs):
        manifest["status"] = "RUNNING"
    else:
        manifest["status"] = "IN_PROGRESS"
    manifest["updated_at"] = _now_utc()
    return manifest


def _write_step7_runtime_state(root: Path, manifest: dict[str, Any]) -> None:
    runtime_paths = _ensure_step7_runtime_directories(root)
    payload = _refresh_manifest_counts(dict(manifest))
    runtime_paths["manifest_path"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(payload.get("jobs", [])).to_csv(runtime_paths["job_status_path"], index=False)


def _load_step7_runtime_manifest(root: Path) -> dict[str, Any]:
    manifest_path = _step7_runtime_paths(root)["manifest_path"]
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _jobs_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(job["job_id"]): job for job in manifest.get("jobs", [])}


def _job_attempt_limit(job_class: str) -> int:
    if job_class == "initialize":
        return 2
    if job_class == "base_year":
        return STEP7_BASE_JOB_ATTEMPT_LIMIT
    if job_class == "candidate_year":
        return STEP7_CANDIDATE_YEAR_ATTEMPT_LIMIT
    if job_class == "reducer":
        return STEP7_REDUCER_ATTEMPT_LIMIT
    if job_class == "assembly":
        return STEP7_ASSEMBLY_ATTEMPT_LIMIT
    return 1


def _job_is_final(job: dict[str, Any]) -> bool:
    return _optional_string(job.get("status")) in STEP7_FINAL_JOB_STATUSES


def _dependencies_are_final(job: dict[str, Any], job_map: dict[str, dict[str, Any]]) -> bool:
    return all(_job_is_final(job_map[dependency]) for dependency in job.get("depends_on", []))


def _reset_jobs_for_resume(manifest: dict[str, Any]) -> dict[str, Any]:
    job_map = _jobs_by_id(manifest)
    for job in manifest.get("jobs", []):
        status = _optional_string(job.get("status"))
        job_class = _optional_string(job.get("job_class"))
        attempt_limit = _job_attempt_limit(job_class)
        if status == STEP7_JOB_STATUS_FAILED and int(job.get("attempt_count", 0)) < attempt_limit:
            job["status"] = STEP7_JOB_STATUS_PENDING
            job["started_at"] = ""
            job["finished_at"] = ""
            job["error_relpath"] = ""
            job["fallback_reason"] = ""
        elif status == STEP7_JOB_STATUS_SKIPPED and _optional_string(job.get("fallback_reason")) == "base_year_unavailable":
            dependency_statuses = {_optional_string(job_map[dependency].get("status")) for dependency in job.get("depends_on", [])}
            if dependency_statuses & {STEP7_JOB_STATUS_PENDING, STEP7_JOB_STATUS_RUNNING, STEP7_JOB_STATUS_COMPLETED}:
                job["status"] = STEP7_JOB_STATUS_PENDING
                job["started_at"] = ""
                job["finished_at"] = ""
                job["fallback_reason"] = ""
    job_map = _jobs_by_id(manifest)
    for job in manifest.get("jobs", []):
        status = _optional_string(job.get("status"))
        if status != STEP7_JOB_STATUS_FAILED:
            continue
        if _optional_string(job.get("job_class")) not in {"reducer", "assembly"}:
            continue
        if any(_optional_string(job_map[dependency].get("status")) == STEP7_JOB_STATUS_PENDING for dependency in job.get("depends_on", [])):
            job["status"] = STEP7_JOB_STATUS_PENDING
            job["started_at"] = ""
            job["finished_at"] = ""
            job["error_relpath"] = ""
            job["fallback_reason"] = ""
    return manifest


def _build_step7_jobs(years: list[int], route_catalog: pd.DataFrame, spec: dict[str, Any]) -> list[dict[str, Any]]:
    event_rows = route_catalog.loc[route_catalog["step7_route_class"].astype(str).eq("event_level_escalated")].copy()
    priority_required = set(str(item) for item in spec.get("priority_required_analysis_units", []))
    live_birth_ids = [
        analysis_unit_id
        for analysis_unit_id in STEP7_EVENT_CANDIDATES_IN_RING
        if analysis_unit_id in set(spec.get("perinatal_live_birth_specs", {}).keys())
    ]
    stillbirth_ids = [
        analysis_unit_id
        for analysis_unit_id in STEP7_EVENT_CANDIDATES_IN_RING
        if analysis_unit_id in set(spec.get("stillbirth_specs", {}).keys())
    ]
    neonatal_ids = [
        analysis_unit_id
        for analysis_unit_id in STEP7_EVENT_CANDIDATES_IN_RING
        if analysis_unit_id in set(spec.get("neonatal_specs", {}).keys())
    ]
    job_rows: list[dict[str, Any]] = [
        {
            "job_id": "initialize_step7_runtime",
            "phase": "initialization",
            "job_class": "initialize",
            "analysis_unit_id": "",
            "source_family": "",
            "source_year": "",
            "required_flag": True,
            "priority_flag": True,
            "depends_on": [],
            "status": STEP7_JOB_STATUS_PENDING,
            "attempt_count": 0,
            "started_at": "",
            "finished_at": "",
            "output_relpath": "",
            "error_relpath": "",
            "fallback_reason": "",
        }
    ]
    for year in years:
        for source_family in ("live_birth", "stillbirth", "neonatal"):
            job_rows.append(
                {
                    "job_id": f"prepare_{source_family}_base_{year}",
                    "phase": "base_year",
                    "job_class": "base_year",
                    "analysis_unit_id": "",
                    "source_family": source_family,
                    "source_year": year,
                    "required_flag": True,
                    "priority_flag": True,
                    "depends_on": ["initialize_step7_runtime"],
                    "status": STEP7_JOB_STATUS_PENDING,
                    "attempt_count": 0,
                    "started_at": "",
                    "finished_at": "",
                    "output_relpath": "",
                    "error_relpath": "",
                    "fallback_reason": "",
                }
            )
    for year in years:
        for analysis_unit_id in STEP7_EVENT_CANDIDATES_IN_RING:
            if analysis_unit_id in live_birth_ids:
                depends_on = [f"prepare_live_birth_base_{year}"]
                source_family = "live_birth"
            elif analysis_unit_id in stillbirth_ids:
                depends_on = [f"prepare_live_birth_base_{year}", f"prepare_stillbirth_base_{year}"]
                source_family = "stillbirth"
            elif analysis_unit_id in neonatal_ids:
                depends_on = [f"prepare_neonatal_base_{year}"]
                source_family = "neonatal"
            else:
                continue
            job_rows.append(
                {
                    "job_id": f"shard_{analysis_unit_id}_{year}",
                    "phase": "candidate_year",
                    "job_class": "candidate_year",
                    "analysis_unit_id": analysis_unit_id,
                    "source_family": source_family,
                    "source_year": year,
                    "required_flag": analysis_unit_id in priority_required,
                    "priority_flag": analysis_unit_id in priority_required,
                    "depends_on": depends_on,
                    "status": STEP7_JOB_STATUS_PENDING,
                    "attempt_count": 0,
                    "started_at": "",
                    "finished_at": "",
                    "output_relpath": "",
                    "error_relpath": "",
                    "fallback_reason": "",
                }
            )
    for analysis_unit_id in STEP7_EVENT_CANDIDATES_IN_RING:
        job_rows.append(
            {
                "job_id": f"reduce_{analysis_unit_id}",
                "phase": "reducer",
                "job_class": "reducer",
                "analysis_unit_id": analysis_unit_id,
                "source_family": "",
                "source_year": "",
                "required_flag": analysis_unit_id in priority_required,
                "priority_flag": analysis_unit_id in priority_required,
                "depends_on": [f"shard_{analysis_unit_id}_{year}" for year in years],
                "status": STEP7_JOB_STATUS_PENDING,
                "attempt_count": 0,
                "started_at": "",
                "finished_at": "",
                "output_relpath": "",
                "error_relpath": "",
                "fallback_reason": "",
            }
        )
    assembly_dependencies = [f"reduce_{analysis_unit_id}" for analysis_unit_id in STEP7_EVENT_CANDIDATES_IN_RING]
    for job_id in (
        "assemble_event_tables",
        "assemble_surfaces",
        "assemble_publication_cards",
        "assemble_report_and_workbook",
    ):
        depends_on = assembly_dependencies if job_id == "assemble_event_tables" else [prev for prev in [
            "assemble_event_tables" if job_id != "assemble_event_tables" else None,
            "assemble_surfaces" if job_id in {"assemble_publication_cards", "assemble_report_and_workbook"} else None,
            "assemble_publication_cards" if job_id == "assemble_report_and_workbook" else None,
        ] if prev is not None]
        job_rows.append(
            {
                "job_id": job_id,
                "phase": "assembly",
                "job_class": "assembly",
                "analysis_unit_id": "",
                "source_family": "",
                "source_year": "",
                "required_flag": True,
                "priority_flag": True,
                "depends_on": depends_on,
                "status": STEP7_JOB_STATUS_PENDING,
                "attempt_count": 0,
                "started_at": "",
                "finished_at": "",
                "output_relpath": "",
                "error_relpath": "",
                "fallback_reason": "",
            }
        )
    return job_rows


def _build_step7_runtime_manifest(
    root: Path,
    *,
    round_config: dict[str, Any],
    route_catalog: pd.DataFrame,
    spec: dict[str, Any],
    max_active_jobs: int,
    execution_mode: str,
) -> dict[str, Any]:
    years = [int(year) for year in round_config.get("years", [])]
    jobs = _build_step7_jobs(years, route_catalog, spec)
    manifest = {
        "step": STEP7_NAME,
        "execution_mode": execution_mode,
        "created_at": _now_utc(),
        "updated_at": _now_utc(),
        "max_active_jobs": max_active_jobs,
        "cooldown_seconds": _step7_cooldown_seconds(),
        "status": "IN_PROGRESS",
        "jobs": jobs,
    }
    return _refresh_manifest_counts(manifest)


def _scaffold_step7_publication_workspace(root: Path, spec: dict[str, Any]) -> dict[str, object]:
    workspace_root = incremental_publication_root(root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    created_files: list[str] = []
    existing_files: list[str] = []
    publication_files = dict(STEP7_PUBLICATION_FILES)
    publication_files.update(spec.get("publication_files", {}))
    for relative_path, contents in publication_files.items():
        target = workspace_root / relative_path
        if target.exists():
            existing_files.append(_relative(root, target))
            continue
        target.write_text(str(contents), encoding="utf-8")
        created_files.append(_relative(root, target))
    return {
        "workspace_root": _relative(root, workspace_root),
        "created_files": created_files,
        "existing_files": existing_files,
    }


def _resolve_upstream_step6_report(
    root: Path,
    *,
    config_path: Path | None,
    step6_report: dict[str, Any] | None,
) -> dict[str, Any]:
    if step6_report is not None:
        return step6_report
    step6_root = analysis_step6_root(root)
    if (step6_root / "step6_candidate_catalog.csv").exists() and (step6_root / "step6_group_headlines.csv").exists():
        return {"status": "REUSED_EXISTING_STEP6_OUTPUTS"}
    return build_step6_absolute_burden(root=root, config_path=config_path)


def _load_step5_scorecard(root: Path) -> pd.DataFrame:
    return pd.read_csv(analysis_step5_root(root) / "step5_scorecard.csv")


def _load_step6_candidate_catalog(root: Path) -> pd.DataFrame:
    return pd.read_csv(analysis_step6_root(root) / "step6_candidate_catalog.csv")


def _load_step6_group_headlines(root: Path) -> pd.DataFrame:
    return pd.read_csv(analysis_step6_root(root) / "step6_group_headlines.csv")


def _load_step6_secondary_translation(root: Path) -> pd.DataFrame:
    return pd.read_csv(analysis_step6_root(root) / "secondary_parallel_translation.csv")


def _load_step4_decisions(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "artifacts" / "analysis" / "step4" / "step4_promotion_decisions.csv")


def _load_step3_family_comparison(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "artifacts" / "analysis" / "step3" / "step3_family_threshold_comparison.csv")


def _load_municipality_lookup(root: Path) -> pd.DataFrame:
    path = root / "artifacts" / "reference" / "municipality_context" / "atlas_brasil_municipality_lookup.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    lookup = pd.read_csv(path)
    lookup["municipality_code7"] = _normalize_municipality_code(lookup["MUNICIPALITY_CODE_IBGE7"], width=7)
    lookup["municipality_code"] = lookup["municipality_code7"].str[:6]
    lookup["municipality_name"] = lookup["NM_MUNICIPIO"].astype(str)
    lookup["state_code"] = lookup["UF"].astype(str)
    lookup["macroregion"] = lookup["state_code"].map(UF_TO_MACROREGION).fillna("Unknown")
    return lookup.loc[
        :,
        ["municipality_code", "municipality_code7", "municipality_name", "state_code", "macroregion"],
    ].drop_duplicates()


def _load_climate_year_window(root: Path, year: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for candidate_year in (year - 1, year):
        path = root / "data" / "silver" / "climate" / f"YEAR={candidate_year}" / f"climate_canonical_daily_{candidate_year}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["COD_MUNICIPIO", "DATE", "TMAX_C", "HEAT_INDEX_C"])
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["municipality_code", "date", "tmax_c", "heat_index_c"]
        )
    climate = pd.concat(frames, ignore_index=True)
    climate["municipality_code"] = _normalize_municipality_code(climate["COD_MUNICIPIO"], width=6)
    climate["date"] = pd.to_datetime(climate["DATE"])
    climate = climate.rename(columns={"TMAX_C": "tmax_c", "HEAT_INDEX_C": "heat_index_c"})
    climate = climate.loc[:, ["municipality_code", "date", "tmax_c", "heat_index_c"]].sort_values(
        ["municipality_code", "date"],
        ignore_index=True,
    )
    return climate


def _build_fixed_windows(climate: pd.DataFrame, year: int) -> pd.DataFrame:
    if climate.empty:
        return pd.DataFrame(
            columns=[
                "municipality_code",
                "event_date",
                "days_0_6_before_event_mean_tmax_c",
                "days_0_6_before_event_mean_heat_index_c",
                "days_0_6_before_event_full_window_observed_flag",
                "days_0_13_before_event_mean_tmax_c",
                "days_0_13_before_event_mean_heat_index_c",
                "days_0_13_before_event_full_window_observed_flag",
                "days_0_27_before_event_mean_tmax_c",
                "days_0_27_before_event_mean_heat_index_c",
                "days_0_27_before_event_full_window_observed_flag",
            ]
        )
    working = climate.sort_values(["municipality_code", "date"], ignore_index=True).copy()
    grouped = working.groupby("municipality_code", group_keys=False)
    for prefix, window in {
        "days_0_6_before_event": 7,
        "days_0_13_before_event": 14,
        "days_0_27_before_event": 28,
    }.items():
        working[f"{prefix}_mean_tmax_c"] = (
            grouped["tmax_c"].rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
        )
        working[f"{prefix}_mean_heat_index_c"] = (
            grouped["heat_index_c"].rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
        )
        working[f"{prefix}_full_window_observed_flag"] = working[f"{prefix}_mean_tmax_c"].notna().astype(np.int8)
    working = working.loc[working["date"].dt.year.eq(year)].copy()
    working = working.rename(columns={"date": "event_date"})
    columns = ["municipality_code", "event_date"]
    for prefix in ("days_0_6_before_event", "days_0_13_before_event", "days_0_27_before_event"):
        columns.extend(
            [
                f"{prefix}_mean_tmax_c",
                f"{prefix}_mean_heat_index_c",
                f"{prefix}_full_window_observed_flag",
            ]
        )
    return working.loc[:, columns]


def _attach_fixed_windows(events: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    if windows.empty:
        working = events.copy()
        for prefix in ("days_0_6_before_event", "days_0_13_before_event", "days_0_27_before_event"):
            working[f"{prefix}_mean_tmax_c"] = float("nan")
            working[f"{prefix}_mean_heat_index_c"] = float("nan")
            working[f"{prefix}_full_window_observed_flag"] = np.int8(0)
        return working
    return events.merge(windows, on=["municipality_code", "event_date"], how="left")


def _load_live_birth_year_frame(root: Path, year: int, lookup: pd.DataFrame) -> pd.DataFrame:
    path = root / "data" / "silver" / "sinasc" / "silver_sinasc_event" / f"YEAR={year}" / f"silver_sinasc_event_{year}.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(
        path,
        columns=[
            "MUNICIPALITY_CODE",
            "DATE",
            "YEAR",
            "MONTH",
            "GA_DAYS_FINAL",
            "GA_WEEKS_COMPLETED_FINAL",
            "GA_SOURCE",
            "PRETERM_LT37_FLAG",
            "PRETERM_LT32_FLAG",
            "PRETERM_DAYS_SHORT_OF_37W_FINAL",
            "LOW_BIRTH_WEIGHT_LT2500_FLAG",
            "GA_VALID_FOR_ANALYSIS_FLAG",
        ],
    )
    frame["municipality_code"] = _normalize_municipality_code(frame["MUNICIPALITY_CODE"], width=6)
    frame["event_date"] = pd.to_datetime(frame["DATE"])
    frame["calendar_year"] = pd.to_numeric(frame["YEAR"], errors="coerce").fillna(year).astype(int)
    frame["delivery_month"] = pd.to_numeric(frame["MONTH"], errors="coerce").fillna(frame["event_date"].dt.month).astype(int)
    ga_days = pd.to_numeric(frame["GA_DAYS_FINAL"], errors="coerce")
    conception_date = frame["event_date"] - pd.to_timedelta(ga_days.fillna(0.0), unit="D")
    conception_month = conception_date.dt.month.astype("Int64")
    frame["conception_month"] = conception_month.fillna(0).astype(int)
    frame["gestational_age_source"] = frame["GA_SOURCE"].fillna("missing").astype(str)
    frame["ga_weeks_completed_final"] = pd.to_numeric(frame["GA_WEEKS_COMPLETED_FINAL"], errors="coerce")
    frame["birth_count"] = 1.0
    frame["preterm_birth_flag"] = pd.to_numeric(frame["PRETERM_LT37_FLAG"], errors="coerce").fillna(0).astype(float)
    frame["very_preterm_birth_flag"] = pd.to_numeric(frame["PRETERM_LT32_FLAG"], errors="coerce").fillna(0).astype(float)
    frame["low_birth_weight_lt2500_flag"] = pd.to_numeric(frame["LOW_BIRTH_WEIGHT_LT2500_FLAG"], errors="coerce").fillna(0).astype(float)
    frame["gestational_days_lost_below_37_weeks"] = pd.to_numeric(
        frame["PRETERM_DAYS_SHORT_OF_37W_FINAL"], errors="coerce"
    ).fillna(0.0)
    frame["ga_valid_for_analysis_flag"] = pd.to_numeric(frame["GA_VALID_FOR_ANALYSIS_FLAG"], errors="coerce").fillna(0).astype(np.int8)
    frame = frame.merge(lookup.loc[:, ["municipality_code", "state_code", "macroregion"]], on="municipality_code", how="left")
    windows = _build_fixed_windows(_load_climate_year_window(root, year), year)
    frame = _attach_fixed_windows(frame, windows)
    frame = frame.loc[
        :,
        [
            "municipality_code",
            "event_date",
            "calendar_year",
            "delivery_month",
            "conception_month",
            "state_code",
            "macroregion",
            "gestational_age_source",
            "ga_weeks_completed_final",
            "birth_count",
            "preterm_birth_flag",
            "very_preterm_birth_flag",
            "low_birth_weight_lt2500_flag",
            "gestational_days_lost_below_37_weeks",
            "ga_valid_for_analysis_flag",
            "days_0_6_before_event_mean_tmax_c",
            "days_0_6_before_event_mean_heat_index_c",
            "days_0_6_before_event_full_window_observed_flag",
            "days_0_13_before_event_mean_tmax_c",
            "days_0_13_before_event_mean_heat_index_c",
            "days_0_13_before_event_full_window_observed_flag",
            "days_0_27_before_event_mean_tmax_c",
            "days_0_27_before_event_mean_heat_index_c",
            "days_0_27_before_event_full_window_observed_flag",
        ],
    ]
    group_columns = [
        "municipality_code",
        "event_date",
        "calendar_year",
        "delivery_month",
        "conception_month",
        "state_code",
        "macroregion",
        "gestational_age_source",
        "ga_weeks_completed_final",
        "ga_valid_for_analysis_flag",
        "days_0_6_before_event_mean_tmax_c",
        "days_0_6_before_event_mean_heat_index_c",
        "days_0_6_before_event_full_window_observed_flag",
        "days_0_13_before_event_mean_tmax_c",
        "days_0_13_before_event_mean_heat_index_c",
        "days_0_13_before_event_full_window_observed_flag",
        "days_0_27_before_event_mean_tmax_c",
        "days_0_27_before_event_mean_heat_index_c",
        "days_0_27_before_event_full_window_observed_flag",
    ]
    return (
        frame.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            birth_count=("birth_count", "sum"),
            preterm_birth_flag=("preterm_birth_flag", "sum"),
            very_preterm_birth_flag=("very_preterm_birth_flag", "sum"),
            low_birth_weight_lt2500_flag=("low_birth_weight_lt2500_flag", "sum"),
            gestational_days_lost_below_37_weeks=("gestational_days_lost_below_37_weeks", "sum"),
        )
    )


def _load_stillbirth_year_frame(root: Path, year: int, lookup: pd.DataFrame) -> pd.DataFrame:
    path = root / "data" / "silver" / "stillbirth" / "silver_stillbirth_individual" / f"YEAR={year}" / f"silver_stillbirth_individual_{year}.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(
        path,
        columns=[
            "MUNICIPALITY_CODE",
            "EVENT_DATE",
            "YEAR",
            "MONTH",
            "GA_WEEKS_COMPLETED_FINAL",
            "GA_SOURCE",
            "GA_VALID_FOR_ANALYSIS_FLAG",
            "STILLBIRTH_GTE28_PRIMARY_FLAG",
            "STILLBIRTH_GTE22_SENSITIVITY_FLAG",
        ],
    )
    frame["municipality_code"] = _normalize_municipality_code(frame["MUNICIPALITY_CODE"], width=6)
    frame["event_date"] = pd.to_datetime(frame["EVENT_DATE"])
    frame["calendar_year"] = pd.to_numeric(frame["YEAR"], errors="coerce").fillna(year).astype(int)
    frame["event_month"] = pd.to_numeric(frame["MONTH"], errors="coerce").fillna(frame["event_date"].dt.month).astype(int)
    frame["birth_count"] = 1.0
    frame["gestational_age_source"] = frame["GA_SOURCE"].fillna("missing").astype(str)
    frame["ga_weeks_completed_final"] = pd.to_numeric(frame["GA_WEEKS_COMPLETED_FINAL"], errors="coerce")
    frame["ga_valid_for_analysis_flag"] = pd.to_numeric(frame["GA_VALID_FOR_ANALYSIS_FLAG"], errors="coerce").fillna(0).astype(np.int8)
    frame["stillbirth_primary_gte28_flag"] = pd.to_numeric(
        frame["STILLBIRTH_GTE28_PRIMARY_FLAG"], errors="coerce"
    ).fillna(0).astype(float)
    frame["stillbirth_sensitivity_gte22_flag"] = pd.to_numeric(
        frame["STILLBIRTH_GTE22_SENSITIVITY_FLAG"], errors="coerce"
    ).fillna(0).astype(float)
    frame = frame.merge(lookup.loc[:, ["municipality_code", "state_code", "macroregion"]], on="municipality_code", how="left")
    windows = _build_fixed_windows(_load_climate_year_window(root, year), year)
    frame = _attach_fixed_windows(frame, windows)
    frame = frame.loc[
        :,
        [
            "municipality_code",
            "event_date",
            "calendar_year",
            "event_month",
            "state_code",
            "macroregion",
            "gestational_age_source",
            "ga_weeks_completed_final",
            "ga_valid_for_analysis_flag",
            "birth_count",
            "stillbirth_primary_gte28_flag",
            "stillbirth_sensitivity_gte22_flag",
            "days_0_6_before_event_mean_tmax_c",
            "days_0_6_before_event_mean_heat_index_c",
            "days_0_6_before_event_full_window_observed_flag",
            "days_0_13_before_event_mean_tmax_c",
            "days_0_13_before_event_mean_heat_index_c",
            "days_0_13_before_event_full_window_observed_flag",
            "days_0_27_before_event_mean_tmax_c",
            "days_0_27_before_event_mean_heat_index_c",
            "days_0_27_before_event_full_window_observed_flag",
        ],
    ]
    group_columns = [
        "municipality_code",
        "event_date",
        "calendar_year",
        "event_month",
        "state_code",
        "macroregion",
        "gestational_age_source",
        "ga_weeks_completed_final",
        "ga_valid_for_analysis_flag",
        "days_0_6_before_event_mean_tmax_c",
        "days_0_6_before_event_mean_heat_index_c",
        "days_0_6_before_event_full_window_observed_flag",
        "days_0_13_before_event_mean_tmax_c",
        "days_0_13_before_event_mean_heat_index_c",
        "days_0_13_before_event_full_window_observed_flag",
        "days_0_27_before_event_mean_tmax_c",
        "days_0_27_before_event_mean_heat_index_c",
        "days_0_27_before_event_full_window_observed_flag",
    ]
    return (
        frame.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            birth_count=("birth_count", "sum"),
            stillbirth_primary_gte28_flag=("stillbirth_primary_gte28_flag", "sum"),
            stillbirth_sensitivity_gte22_flag=("stillbirth_sensitivity_gte22_flag", "sum"),
        )
    )


def _load_neonatal_daily_frame(root: Path, years: list[int], lookup: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    lookup_subset = lookup.loc[:, ["municipality_code7", "state_code", "macroregion"]].drop_duplicates()
    for year in years:
        path = root / "data" / "gold" / "gold_mnch_mortality" / f"YEAR={year}" / f"gold_mnch_mortality_{year}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(
            path,
            columns=[
                "MUNICIPALITY_CODE",
                "DATE",
                "YEAR",
                "MONTH",
                "DEATHS_NEONATAL",
                "LIVE_BIRTHS_TOTAL_ANNUAL",
                "TMAX_C",
                "HEAT_INDEX_APPROX_C",
                "QC_STATUS",
                "OFFSET_VALID_FLAG",
                "DENOMINATOR_ZERO_FLAG",
                "DENOMINATOR_MISSING_FLAG",
                "RATE_NOT_COMPUTABLE_FLAG",
            ],
        )
        frame["municipality_code7"] = _normalize_municipality_code(frame["MUNICIPALITY_CODE"], width=7)
        frame["event_date"] = pd.to_datetime(frame["DATE"])
        frame["calendar_year"] = pd.to_numeric(frame["YEAR"], errors="coerce").fillna(year).astype(int)
        frame["calendar_month"] = pd.to_numeric(frame["MONTH"], errors="coerce").fillna(frame["event_date"].dt.month).astype(int)
        frame["day_of_week"] = frame["event_date"].dt.dayofweek.astype(int)
        frame["epidemiologic_week"] = frame["event_date"].dt.isocalendar().week.astype(int)
        frame["quality_control_status"] = frame["QC_STATUS"].fillna("FAIL").astype(str)
        frame["offset_valid_flag"] = pd.to_numeric(frame["OFFSET_VALID_FLAG"], errors="coerce").fillna(0).astype(np.int8)
        frame["denominator_zero_flag"] = pd.to_numeric(frame["DENOMINATOR_ZERO_FLAG"], errors="coerce").fillna(0).astype(np.int8)
        frame["denominator_missing_flag"] = pd.to_numeric(frame["DENOMINATOR_MISSING_FLAG"], errors="coerce").fillna(0).astype(np.int8)
        frame["rate_not_computable_flag"] = pd.to_numeric(frame["RATE_NOT_COMPUTABLE_FLAG"], errors="coerce").fillna(0).astype(np.int8)
        frame = frame.rename(
            columns={
                "DEATHS_NEONATAL": "neonatal_death_count",
                "LIVE_BIRTHS_TOTAL_ANNUAL": "annual_live_birth_count",
                "TMAX_C": "tmax_c",
                "HEAT_INDEX_APPROX_C": "heat_index_c",
            }
        )
        frame = frame.sort_values(["municipality_code7", "event_date"], ignore_index=True)
        grouped = frame.groupby("municipality_code7", group_keys=False)
        for prefix, source_column, window in (
            ("tmax", "tmax_c", 1),
            ("tmax_lag0_1_mean", "tmax_c", 2),
            ("tmax_lag0_3_mean", "tmax_c", 4),
            ("tmax_lag0_7_mean", "tmax_c", 8),
            ("heat_index", "heat_index_c", 1),
            ("heat_index_lag0_1_mean", "heat_index_c", 2),
            ("heat_index_lag0_3_mean", "heat_index_c", 4),
            ("heat_index_lag0_7_mean", "heat_index_c", 8),
        ):
            if window == 1:
                frame[prefix] = frame[source_column]
            else:
                frame[prefix] = grouped[source_column].rolling(window, min_periods=window).mean().reset_index(
                    level=0, drop=True
                )
        frame = frame.merge(lookup_subset, on="municipality_code7", how="left")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _perinatal_exposure_column(selected_exposure_family_id: str, preferred_lag_or_window: str) -> tuple[str, str]:
    prefix = {
        "family_1_tmax_primary": "tmax_c",
        "family_4_heat_index_parallel": "heat_index_c",
    }.get(selected_exposure_family_id)
    if prefix is None:
        raise ValueError(f"Unsupported Step 7 exposure family {selected_exposure_family_id!r}")
    base = f"days_{preferred_lag_or_window}_before_event"
    return f"{base}_mean_{prefix}", f"{base}_full_window_observed_flag"


def _neonatal_exposure_column(selected_exposure_family_id: str, preferred_lag_or_window: str) -> str:
    if selected_exposure_family_id == "family_1_tmax_primary":
        mapping = {
            "lag0": "tmax",
            "lag0_1": "tmax_lag0_1_mean",
            "lag0_3": "tmax_lag0_3_mean",
            "lag0_7": "tmax_lag0_7_mean",
        }
    elif selected_exposure_family_id == "family_4_heat_index_parallel":
        mapping = {
            "lag0": "heat_index",
            "lag0_1": "heat_index_lag0_1_mean",
            "lag0_3": "heat_index_lag0_3_mean",
            "lag0_7": "heat_index_lag0_7_mean",
        }
    else:
        raise ValueError(f"Unsupported Step 7 exposure family {selected_exposure_family_id!r}")
    if preferred_lag_or_window not in mapping:
        raise ValueError(f"Unsupported Step 7 lag/window {preferred_lag_or_window!r}")
    return mapping[preferred_lag_or_window]


def _assign_grouped_quantile_bins(
    frame: pd.DataFrame,
    *,
    exposure_column: str,
    group_columns: list[str],
    bin_count: int = 10,
) -> pd.Series:
    working = frame.loc[:, group_columns + [exposure_column]].copy()
    working["__rank_pct"] = working.groupby(group_columns, dropna=False)[exposure_column].rank(method="first", pct=True)
    bins = np.ceil(working["__rank_pct"] * bin_count).clip(lower=1, upper=bin_count)
    insufficient = working.groupby(group_columns, dropna=False)[exposure_column].transform("count").lt(3)
    return bins.where(~insufficient)


def _aggregate_grouped_proxy_data(
    frame: pd.DataFrame,
    *,
    exposure_column: str,
    outcome_column: str,
    denominator_column: str,
    group_columns: list[str],
    bin_count: int,
    observed_flag_column: str | None = None,
) -> pd.DataFrame:
    columns = list(dict.fromkeys(group_columns + [exposure_column, outcome_column, denominator_column]))
    if observed_flag_column is not None:
        columns.append(observed_flag_column)
    working = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    if observed_flag_column is not None and observed_flag_column in working.columns:
        working = working.loc[pd.to_numeric(working[observed_flag_column], errors="coerce").fillna(0).eq(1)].copy()
    working = working.loc[
        working[exposure_column].notna()
        & pd.to_numeric(working[denominator_column], errors="coerce").fillna(0).gt(0)
        & working[outcome_column].notna()
    ].copy()
    if working.empty:
        return pd.DataFrame()
    working["__denominator_value"] = pd.to_numeric(working[denominator_column], errors="coerce")
    working["__outcome_value"] = pd.to_numeric(working[outcome_column], errors="coerce")
    working["__exposure_numerator"] = pd.to_numeric(working[exposure_column], errors="coerce") * working["__denominator_value"]
    collapsed = (
        working.groupby(group_columns + [exposure_column], dropna=False, as_index=False)
        .agg(
            denominator_value=("__denominator_value", "sum"),
            outcome_value=("__outcome_value", "sum"),
            exposure_numerator=("__exposure_numerator", "sum"),
        )
    )
    if collapsed.empty:
        return pd.DataFrame()
    collapsed["exposure_bin"] = _assign_grouped_quantile_bins(
        collapsed,
        exposure_column=exposure_column,
        group_columns=group_columns,
        bin_count=bin_count,
    )
    collapsed = collapsed.loc[collapsed["exposure_bin"].notna()].copy()
    if collapsed.empty:
        return pd.DataFrame()
    aggregated = (
        collapsed.groupby(group_columns + ["exposure_bin"], dropna=False, as_index=False)
        .agg(
            exposure_numerator=("exposure_numerator", "sum"),
            outcome_value=("outcome_value", "sum"),
            denominator_value=("denominator_value", "sum"),
        )
    )
    aggregated["exposure_value"] = aggregated["exposure_numerator"] / aggregated["denominator_value"]
    aggregated["outcome_value"] = pd.to_numeric(aggregated["outcome_value"], errors="coerce").fillna(0.0)
    aggregated = aggregated.drop(columns=["exposure_numerator"])
    aggregated["quality_control_status"] = "PASS"
    aggregated["offset_valid_flag"] = 1
    aggregated["denominator_zero_flag"] = 0
    aggregated["denominator_missing_flag"] = 0
    aggregated["rate_not_computable_flag"] = 0
    return aggregated


def _empty_proxy_result() -> dict[str, Any]:
    return {
        "estimate": float("nan"),
        "lower_ci": float("nan"),
        "upper_ci": float("nan"),
        "p_value": float("nan"),
        "p_value_floor_applied_flag": "no",
        "valid_strata_count": 0,
        "observation_count": 0,
        "outcome_sum": 0.0,
        "denominator_sum": 0.0,
        "overdispersion_ratio": float("nan"),
        "internal_slope": float("nan"),
        "internal_slope_se": float("nan"),
        "exposure_scale_value": 1.0,
    }


def _fit_grouped_proxy_with_fallback(
    *,
    years: list[int],
    load_year_frame: Any,
    exposure_column: str,
    outcome_column: str,
    denominator_column: str,
    hierarchy: list[list[str]],
    model_kind: str,
    observed_flag_column: str | None = None,
    preprocess_year_frame: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], str, int]:
    for fallback_index, group_columns in enumerate(hierarchy, start=1):
        for bin_count in (10, 5, 4):
            aggregated_frames: list[pd.DataFrame] = []
            for year in years:
                year_frame = load_year_frame(year)
                if year_frame is None or year_frame.empty:
                    continue
                working = year_frame if preprocess_year_frame is None else preprocess_year_frame(year_frame.copy())
                if working.empty:
                    continue
                aggregated = _aggregate_grouped_proxy_data(
                    working,
                    exposure_column=exposure_column,
                    outcome_column=outcome_column,
                    denominator_column=denominator_column,
                    group_columns=group_columns,
                    bin_count=bin_count,
                    observed_flag_column=observed_flag_column,
                )
                if not aggregated.empty:
                    aggregated_frames.append(aggregated)
            if not aggregated_frames:
                continue
            pooled = pd.concat(aggregated_frames, ignore_index=True)
            result = _pooled_continuous_fe_proxy(
                pooled,
                exposure_column="exposure_value",
                outcome_column="outcome_value",
                denominator_column="denominator_value",
                stratum_columns=group_columns,
                model_kind="count" if model_kind == "count" else "severity",
            )
            if (
                model_kind == "count"
                and math.isfinite(float(result.get("estimate", float("nan"))))
                and float(result.get("overdispersion_ratio", float("nan"))) > OVERDISPERSION_THRESHOLD
            ):
                result = _dispersion_adjusted_result(result, "count")
            if math.isfinite(float(result.get("estimate", float("nan")))) and int(result.get("valid_strata_count", 0)) > 0:
                return pooled, result, " > ".join(group_columns), bin_count
    return pd.DataFrame(), _empty_proxy_result(), "", 0


def _event_result_row(
    *,
    candidate: dict[str, Any],
    spec: dict[str, Any],
    result: dict[str, Any],
    fallback_level_used: str,
    data_source: str,
    result_origin: str,
) -> dict[str, Any]:
    return {
        "analysis_unit_id": candidate["analysis_unit_id"],
        "data_source": data_source,
        "model_kind": spec.get("model_kind", "count"),
        "primary_temporal_controls": _optional_string(spec.get("primary_temporal_controls")),
        "fallback_level_used": fallback_level_used,
        "estimate": float(result["estimate"]),
        "lower_ci": float(result["lower_ci"]),
        "upper_ci": float(result["upper_ci"]),
        "p_value": float(result["p_value"]),
        "valid_strata_count": int(result["valid_strata_count"]),
        "selected_exposure_family_id": candidate.get("selected_exposure_family_id", ""),
        "selected_exposure_family_name": candidate.get("selected_exposure_family_name", ""),
        "preferred_lag_or_window": candidate.get("preferred_lag_or_window", ""),
        "result_origin": result_origin,
    }


def _temporal_controls_row(
    *,
    candidate: dict[str, Any],
    data_source: str,
    primary_temporal_controls: str,
    robustness_temporal_controls: str = "",
) -> dict[str, Any]:
    return {
        "analysis_unit_id": candidate["analysis_unit_id"],
        "data_source": data_source,
        "selected_exposure_family_name": candidate.get("selected_exposure_family_name", ""),
        "preferred_lag_or_window": candidate.get("preferred_lag_or_window", ""),
        "primary_temporal_controls": primary_temporal_controls,
        "robustness_temporal_controls": robustness_temporal_controls,
    }


def _fallback_row(candidate: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "analysis_unit_id": candidate["analysis_unit_id"],
        "step7_route_class": candidate.get("step7_route_class", ""),
        "step7_route_reason": candidate.get("step7_route_reason", ""),
        "fallback_reason": reason,
        "used_inherited_evidence_flag": True,
    }


def _run_live_birth_escalations(
    *,
    root: Path,
    years: list[int],
    candidates: pd.DataFrame,
    lookup: pd.DataFrame,
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    live_birth_specs = dict(spec.get("perinatal_live_birth_specs", {}))
    per_year_frames: dict[int, pd.DataFrame] = {}
    event_rows: list[dict[str, Any]] = []
    appendix_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []

    def _load_year(year: int) -> pd.DataFrame:
        if year not in per_year_frames:
            per_year_frames[year] = _load_live_birth_year_frame(root, year, lookup)
        return per_year_frames[year]

    for candidate in candidates.to_dict(orient="records"):
        candidate_spec = dict(live_birth_specs.get(str(candidate["analysis_unit_id"]), {}))
        exposure_column, observed_flag = _perinatal_exposure_column(
            str(candidate["selected_exposure_family_id"]),
            str(candidate["preferred_lag_or_window"]),
        )
        hierarchy = [
            [part for part in str(level).split("|") if part]
            for level in candidate_spec.get("fallback_hierarchy", [])
        ]
        if not hierarchy:
            hierarchy = [
                ["municipality_code", "calendar_year", "delivery_month", "conception_month"],
                ["municipality_code", "calendar_year", "delivery_month"],
                ["state_code", "calendar_year", "delivery_month"],
            ]

        def _preprocess(frame: pd.DataFrame) -> pd.DataFrame:
            working = frame.copy()
            working["conception_month"] = pd.to_numeric(working["conception_month"], errors="coerce").fillna(0).astype(int)
            if str(candidate["analysis_unit_id"]) in {"preterm_birth_count", "very_preterm_birth_count", "gestational_days_lost_below_37_weeks"}:
                working = working.loc[working["ga_valid_for_analysis_flag"].fillna(0).astype(int).eq(1)].copy()
            return working

        aggregated, result, fallback_level_used, _ = _fit_grouped_proxy_with_fallback(
            years=years,
            load_year_frame=_load_year,
            preprocess_year_frame=_preprocess,
            exposure_column=exposure_column,
            outcome_column=str(candidate_spec.get("outcome_column")),
            denominator_column="birth_count",
            hierarchy=hierarchy,
            model_kind=str(candidate_spec.get("model_kind", "count")),
            observed_flag_column=observed_flag,
        )
        appendix_rows.append(
            _temporal_controls_row(
                candidate=candidate,
                data_source=str(candidate_spec.get("data_source", "silver_sinasc_event")),
                primary_temporal_controls=_optional_string(candidate_spec.get("primary_temporal_controls")),
            )
        )
        if aggregated.empty:
            fallback_rows.append(_fallback_row(candidate, "event_level_birth_escalation_not_estimable"))
            continue
        event_rows.append(
            _event_result_row(
                candidate=candidate,
                spec=candidate_spec,
                result=result,
                fallback_level_used=fallback_level_used,
                data_source=str(candidate_spec.get("data_source", "silver_sinasc_event")),
                result_origin="event_level_primary",
            )
        )
    return pd.DataFrame(event_rows), pd.DataFrame(appendix_rows), pd.DataFrame(fallback_rows)


def _run_stillbirth_escalations(
    *,
    root: Path,
    years: list[int],
    candidates: pd.DataFrame,
    lookup: pd.DataFrame,
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    stillbirth_specs = dict(spec.get("stillbirth_specs", {}))
    birth_support_frames: dict[int, pd.DataFrame] = {}
    stillbirth_frames: dict[int, pd.DataFrame] = {}
    event_rows: list[dict[str, Any]] = []
    appendix_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []

    def _load_combined_year(year: int) -> pd.DataFrame:
        if year not in birth_support_frames:
            birth_support_frames[year] = _load_live_birth_year_frame(root, year, lookup)
        if year not in stillbirth_frames:
            stillbirth_frames[year] = _load_stillbirth_year_frame(root, year, lookup)
        live_births = birth_support_frames[year]
        stillbirths = stillbirth_frames[year]
        live_birth_support = live_births.copy()
        live_birth_support["event_month"] = live_birth_support["delivery_month"]
        live_birth_support["stillbirth_primary_gte28_flag"] = 0.0
        live_birth_support["stillbirth_sensitivity_gte22_flag"] = 0.0
        shared_columns = [
            "municipality_code",
            "event_date",
            "calendar_year",
            "event_month",
            "state_code",
            "macroregion",
            "gestational_age_source",
            "ga_weeks_completed_final",
            "ga_valid_for_analysis_flag",
            "birth_count",
            "stillbirth_primary_gte28_flag",
            "stillbirth_sensitivity_gte22_flag",
            "days_0_6_before_event_mean_tmax_c",
            "days_0_6_before_event_mean_heat_index_c",
            "days_0_6_before_event_full_window_observed_flag",
            "days_0_13_before_event_mean_tmax_c",
            "days_0_13_before_event_mean_heat_index_c",
            "days_0_13_before_event_full_window_observed_flag",
            "days_0_27_before_event_mean_tmax_c",
            "days_0_27_before_event_mean_heat_index_c",
            "days_0_27_before_event_full_window_observed_flag",
        ]
        return pd.concat(
            [
                live_birth_support.loc[:, shared_columns],
                stillbirths.loc[:, shared_columns],
            ],
            ignore_index=True,
        )

    for candidate in candidates.to_dict(orient="records"):
        candidate_spec = dict(stillbirth_specs.get(str(candidate["analysis_unit_id"]), {}))
        exposure_column, observed_flag = _perinatal_exposure_column(
            str(candidate["selected_exposure_family_id"]),
            str(candidate["preferred_lag_or_window"]),
        )
        hierarchy = [
            [part for part in str(level).split("|") if part]
            for level in candidate_spec.get("fallback_hierarchy", [])
        ]
        if not hierarchy:
            hierarchy = [
                ["municipality_code", "calendar_year", "event_month", "gestational_age_source"],
                ["municipality_code", "calendar_year", "event_month"],
                ["municipality_code", "calendar_year"],
            ]
        minimum_weeks = float(candidate_spec.get("minimum_gestational_age_weeks", 0))

        def _preprocess(frame: pd.DataFrame) -> pd.DataFrame:
            working = frame.copy()
            working = working.loc[working["ga_valid_for_analysis_flag"].fillna(0).astype(int).eq(1)].copy()
            if minimum_weeks > 0:
                working = working.loc[pd.to_numeric(working["ga_weeks_completed_final"], errors="coerce").ge(minimum_weeks)].copy()
            return working

        aggregated, result, fallback_level_used, _ = _fit_grouped_proxy_with_fallback(
            years=years,
            load_year_frame=_load_combined_year,
            preprocess_year_frame=_preprocess,
            exposure_column=exposure_column,
            outcome_column=str(candidate_spec.get("outcome_column")),
            denominator_column="birth_count",
            hierarchy=hierarchy,
            model_kind="count",
            observed_flag_column=observed_flag,
        )
        appendix_rows.append(
            _temporal_controls_row(
                candidate=candidate,
                data_source=str(candidate_spec.get("data_source", "silver_stillbirth_individual")),
                primary_temporal_controls=_optional_string(candidate_spec.get("primary_temporal_controls")),
            )
        )
        if aggregated.empty:
            fallback_rows.append(_fallback_row(candidate, "event_level_stillbirth_escalation_not_estimable"))
            continue
        event_rows.append(
            _event_result_row(
                candidate=candidate,
                spec=candidate_spec,
                result=result,
                fallback_level_used=fallback_level_used,
                data_source=str(candidate_spec.get("data_source", "silver_stillbirth_individual")),
                result_origin="event_level_primary",
            )
        )
    return pd.DataFrame(event_rows), pd.DataFrame(appendix_rows), pd.DataFrame(fallback_rows)


def _run_neonatal_escalation(
    *,
    root: Path,
    years: list[int],
    candidates: pd.DataFrame,
    lookup: pd.DataFrame,
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    neonatal_specs = dict(spec.get("neonatal_specs", {}))
    frame = _load_neonatal_daily_frame(root, years, lookup)
    event_rows: list[dict[str, Any]] = []
    appendix_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dict(orient="records"):
        candidate_spec = dict(neonatal_specs.get(str(candidate["analysis_unit_id"]), {}))
        exposure_column = _neonatal_exposure_column(
            str(candidate["selected_exposure_family_id"]),
            str(candidate["preferred_lag_or_window"]),
        )
        denominator_column = str(candidate_spec.get("denominator_column", "annual_live_birth_count"))
        if denominator_column not in frame.columns and denominator_column == "LIVE_BIRTHS_TOTAL_ANNUAL":
            denominator_column = "annual_live_birth_count"
        primary_controls = [part for part in _optional_string(candidate_spec.get("primary_temporal_controls")).split(",") if part]
        robustness_controls = [
            part for part in _optional_string(candidate_spec.get("robustness_temporal_controls")).split(",") if part
        ]
        appendix_rows.append(
            _temporal_controls_row(
                candidate=candidate,
                data_source=str(candidate_spec.get("data_source", "gold_mnch_mortality")),
                primary_temporal_controls=", ".join(primary_controls),
                robustness_temporal_controls=", ".join(robustness_controls),
            )
        )
        if frame.empty:
            fallback_rows.append(_fallback_row(candidate, "grouped_daily_neonatal_frame_missing"))
            continue
        result = _pooled_continuous_fe_proxy(
            frame,
            exposure_column=exposure_column,
            outcome_column="neonatal_death_count",
            denominator_column=denominator_column,
            stratum_columns=primary_controls,
            model_kind="count",
        )
        if (
            math.isfinite(float(result.get("estimate", float("nan"))))
            and float(result.get("overdispersion_ratio", float("nan"))) > OVERDISPERSION_THRESHOLD
        ):
            result = _dispersion_adjusted_result(result, "count")
        if not math.isfinite(float(result.get("estimate", float("nan")))):
            fallback_rows.append(_fallback_row(candidate, "grouped_daily_neonatal_primary_not_estimable"))
            continue
        event_rows.append(
            _event_result_row(
                candidate=candidate,
                spec={**candidate_spec, "model_kind": "count", "primary_temporal_controls": ", ".join(primary_controls)},
                result=result,
                fallback_level_used=" > ".join(primary_controls),
                data_source=str(candidate_spec.get("data_source", "gold_mnch_mortality")),
                result_origin="event_level_primary",
            )
        )
        if robustness_controls:
            robustness = _pooled_continuous_fe_proxy(
                frame,
                exposure_column=exposure_column,
                outcome_column="neonatal_death_count",
                denominator_column=denominator_column,
                stratum_columns=robustness_controls,
                model_kind="count",
            )
            if (
                math.isfinite(float(robustness.get("estimate", float("nan"))))
                and float(robustness.get("overdispersion_ratio", float("nan"))) > OVERDISPERSION_THRESHOLD
            ):
                robustness = _dispersion_adjusted_result(robustness, "count")
            if math.isfinite(float(robustness.get("estimate", float("nan")))):
                event_rows.append(
                    _event_result_row(
                        candidate=candidate,
                        spec={**candidate_spec, "model_kind": "count", "primary_temporal_controls": ", ".join(robustness_controls)},
                        result=robustness,
                        fallback_level_used=" > ".join(robustness_controls),
                        data_source=str(candidate_spec.get("data_source", "gold_mnch_mortality")),
                        result_origin="robustness_epidemiologic_week",
                    )
                )
    return pd.DataFrame(event_rows), pd.DataFrame(appendix_rows), pd.DataFrame(fallback_rows)


def _candidate_catalog_output_columns() -> list[str]:
    return [
        "analysis_unit_id",
        "group_id",
        "group_label",
        "public_label",
        "step6_final_translation_stage",
        "step7_route_class",
        "step7_route_reason",
        "selected_exposure_family_id",
        "selected_exposure_family_name",
        "preferred_lag_or_window",
        "headline_group_surface_flag",
        "all_candidate_surface_flag",
        "public_use",
        "absolute_burden_allowed",
        "headline_group_allowed",
        "has_any_significant_reversal_flag",
        "significant_reversal_modifier_count",
    ]


def _candidate_catalog_output_frame(candidate_catalog: pd.DataFrame) -> pd.DataFrame:
    working = candidate_catalog.copy()
    defaults: dict[str, Any] = {
        "absolute_burden_allowed": False,
        "headline_group_allowed": False,
        "has_any_significant_reversal_flag": False,
        "significant_reversal_modifier_count": 0,
    }
    for column in _candidate_catalog_output_columns():
        if column not in working.columns:
            working[column] = defaults.get(column, "")
    return working.loc[:, _candidate_catalog_output_columns()].sort_values(
        ["group_id", "analysis_unit_id"],
        ignore_index=True,
    )


def _step7_start_condition(root: Path) -> tuple[bool, list[str], list[str]]:
    step6_group_headlines = _load_step6_group_headlines(root)
    missing_groups = sorted(
        set(
            step6_group_headlines.loc[
                step6_group_headlines["headline_missing_flag"].fillna(False).astype(bool),
                "group_id",
            ].astype(str)
        )
    )
    selected_groups = set(
        step6_group_headlines.loc[
            step6_group_headlines["headline_selected_flag"].fillna(False).astype(bool),
            "group_id",
        ].astype(str)
    )
    expected_groups = set(step6_group_headlines["group_id"].astype(str))
    allowed_missing = set(str(item) for item in _step7_spec(root).get("allowed_missing_headline_groups", []))
    non_allowed_missing = sorted(set(missing_groups) - allowed_missing)
    selected_non_missing = expected_groups - set(missing_groups)
    acceptable = not non_allowed_missing and selected_non_missing.issubset(selected_groups)
    return bool(acceptable), missing_groups, non_allowed_missing


def _load_runtime_candidate_catalog(root: Path) -> pd.DataFrame:
    path = analysis_step7_root(root) / "step7_candidate_catalog.csv"
    if not path.exists():
        return pd.DataFrame(columns=_candidate_catalog_output_columns())
    catalog = pd.read_csv(path)
    step6_candidate_path = analysis_step6_root(root) / "step6_candidate_catalog.csv"
    if not step6_candidate_path.exists():
        return catalog
    step6_catalog = pd.read_csv(step6_candidate_path)
    fill_columns = [
        "estimate",
        "lower_ci",
        "upper_ci",
        "p_value",
        "final_translation_stage",
        "public_burden_gate_pass",
        "harmful_direction_flag",
        "significant_flag",
        "gate_failure_reason",
    ]
    available_fill_columns = ["analysis_unit_id", *[column for column in fill_columns if column in step6_catalog.columns]]
    if len(available_fill_columns) == 1:
        return catalog
    catalog = catalog.merge(
        step6_catalog.loc[:, available_fill_columns].drop_duplicates(subset=["analysis_unit_id"]),
        on="analysis_unit_id",
        how="left",
        suffixes=("", "_step6"),
    )
    for column in fill_columns:
        step6_column = f"{column}_step6"
        if step6_column not in catalog.columns:
            continue
        if column in catalog.columns:
            catalog[column] = catalog[column].combine_first(catalog[step6_column])
        else:
            catalog[column] = catalog[step6_column]
        catalog = catalog.drop(columns=[step6_column])
    return catalog


def _analysis_unit_record(candidate_catalog: pd.DataFrame, analysis_unit_id: str) -> dict[str, Any]:
    subset = candidate_catalog.loc[candidate_catalog["analysis_unit_id"].astype(str).eq(analysis_unit_id)]
    if subset.empty:
        raise KeyError(f"Missing Step 7 candidate {analysis_unit_id!r}")
    return subset.iloc[0].to_dict()


def _normalize_shard_frame(
    aggregated: pd.DataFrame,
    *,
    analysis_unit_id: str,
    source_year: int,
    fallback_signature: str,
    bin_count: int,
) -> pd.DataFrame:
    if aggregated.empty:
        return pd.DataFrame()
    output = aggregated.copy()
    output["analysis_unit_id"] = analysis_unit_id
    output["source_year"] = int(source_year)
    output["fallback_signature"] = fallback_signature
    output["bin_count"] = int(bin_count)
    output["quality_control_status"] = "PASS"
    output["offset_valid_flag"] = 1
    output["denominator_zero_flag"] = 0
    output["denominator_missing_flag"] = 0
    output["rate_not_computable_flag"] = 0
    ordered = [
        "analysis_unit_id",
        "source_year",
        "fallback_signature",
        "bin_count",
        "municipality_code",
        "state_code",
        "calendar_year",
        "delivery_month",
        "conception_month",
        "event_month",
        "gestational_age_source",
        "exposure_bin",
        "exposure_value",
        "outcome_value",
        "denominator_value",
        "quality_control_status",
        "offset_valid_flag",
        "denominator_zero_flag",
        "denominator_missing_flag",
        "rate_not_computable_flag",
    ]
    for column in ordered:
        if column not in output.columns:
            output[column] = np.nan
    return output.loc[:, ordered].copy()


def _live_birth_hierarchy(candidate_spec: dict[str, Any]) -> list[list[str]]:
    hierarchy = [[part for part in str(level).split("|") if part] for level in candidate_spec.get("fallback_hierarchy", [])]
    if hierarchy:
        return hierarchy
    return [
        ["municipality_code", "calendar_year", "delivery_month", "conception_month"],
        ["municipality_code", "calendar_year", "delivery_month"],
        ["state_code", "calendar_year", "delivery_month"],
    ]


def _stillbirth_hierarchy(candidate_spec: dict[str, Any]) -> list[list[str]]:
    hierarchy = [[part for part in str(level).split("|") if part] for level in candidate_spec.get("fallback_hierarchy", [])]
    if hierarchy:
        return hierarchy
    return [
        ["municipality_code", "calendar_year", "event_month", "gestational_age_source"],
        ["municipality_code", "calendar_year", "event_month"],
        ["municipality_code", "calendar_year"],
    ]


def _preprocess_live_birth_candidate(frame: pd.DataFrame, analysis_unit_id: str) -> pd.DataFrame:
    working = frame.copy()
    working["conception_month"] = pd.to_numeric(working["conception_month"], errors="coerce").fillna(0).astype(int)
    if analysis_unit_id in {"preterm_birth_count", "very_preterm_birth_count", "gestational_days_lost_below_37_weeks"}:
        working = working.loc[working["ga_valid_for_analysis_flag"].fillna(0).astype(int).eq(1)].copy()
    return working


def _preprocess_stillbirth_candidate(frame: pd.DataFrame, minimum_weeks: float) -> pd.DataFrame:
    working = frame.copy()
    working = working.loc[working["ga_valid_for_analysis_flag"].fillna(0).astype(int).eq(1)].copy()
    if minimum_weeks > 0:
        working = working.loc[pd.to_numeric(working["ga_weeks_completed_final"], errors="coerce").ge(minimum_weeks)].copy()
    return working


def _build_live_birth_candidate_year_shard(
    frame: pd.DataFrame,
    *,
    analysis_unit_id: str,
    candidate_spec: dict[str, Any],
    candidate: dict[str, Any],
    source_year: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    exposure_column, observed_flag = _perinatal_exposure_column(
        _optional_string(candidate.get("selected_exposure_family_id")),
        _optional_string(candidate.get("preferred_lag_or_window")),
    )
    working = _preprocess_live_birth_candidate(frame, analysis_unit_id)
    if working.empty:
        return pd.DataFrame()
    shard_frames: list[pd.DataFrame] = []
    for group_columns in _live_birth_hierarchy(candidate_spec):
        fallback_signature = " > ".join(group_columns)
        for bin_count in (10, 5, 4):
            aggregated = _aggregate_grouped_proxy_data(
                working,
                exposure_column=exposure_column,
                outcome_column=str(candidate_spec.get("outcome_column")),
                denominator_column="birth_count",
                group_columns=group_columns,
                bin_count=bin_count,
                observed_flag_column=observed_flag,
            )
            normalized = _normalize_shard_frame(
                aggregated,
                analysis_unit_id=analysis_unit_id,
                source_year=source_year,
                fallback_signature=fallback_signature,
                bin_count=bin_count,
            )
            if not normalized.empty:
                shard_frames.append(normalized)
    if not shard_frames:
        return pd.DataFrame()
    return pd.concat(shard_frames, ignore_index=True)


def _build_stillbirth_support_year_frame(live_births: pd.DataFrame, stillbirths: pd.DataFrame) -> pd.DataFrame:
    live_birth_support = live_births.copy()
    if live_birth_support.empty and stillbirths.empty:
        return pd.DataFrame()
    if not live_birth_support.empty:
        live_birth_support["event_month"] = live_birth_support["delivery_month"]
        live_birth_support["stillbirth_primary_gte28_flag"] = 0.0
        live_birth_support["stillbirth_sensitivity_gte22_flag"] = 0.0
    shared_columns = [
        "municipality_code",
        "event_date",
        "calendar_year",
        "event_month",
        "state_code",
        "macroregion",
        "gestational_age_source",
        "ga_weeks_completed_final",
        "ga_valid_for_analysis_flag",
        "birth_count",
        "stillbirth_primary_gte28_flag",
        "stillbirth_sensitivity_gte22_flag",
        "days_0_6_before_event_mean_tmax_c",
        "days_0_6_before_event_mean_heat_index_c",
        "days_0_6_before_event_full_window_observed_flag",
        "days_0_13_before_event_mean_tmax_c",
        "days_0_13_before_event_mean_heat_index_c",
        "days_0_13_before_event_full_window_observed_flag",
        "days_0_27_before_event_mean_tmax_c",
        "days_0_27_before_event_mean_heat_index_c",
        "days_0_27_before_event_full_window_observed_flag",
    ]
    frames: list[pd.DataFrame] = []
    if not live_birth_support.empty:
        frames.append(live_birth_support.loc[:, shared_columns])
    if not stillbirths.empty:
        frames.append(stillbirths.loc[:, shared_columns])
    if not frames:
        return pd.DataFrame(columns=shared_columns)
    return pd.concat(frames, ignore_index=True)


def _build_stillbirth_candidate_year_shard(
    combined: pd.DataFrame,
    *,
    analysis_unit_id: str,
    candidate_spec: dict[str, Any],
    candidate: dict[str, Any],
    source_year: int,
) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()
    exposure_column, observed_flag = _perinatal_exposure_column(
        _optional_string(candidate.get("selected_exposure_family_id")),
        _optional_string(candidate.get("preferred_lag_or_window")),
    )
    working = _preprocess_stillbirth_candidate(combined, float(candidate_spec.get("minimum_gestational_age_weeks", 0)))
    if working.empty:
        return pd.DataFrame()
    shard_frames: list[pd.DataFrame] = []
    for group_columns in _stillbirth_hierarchy(candidate_spec):
        fallback_signature = " > ".join(group_columns)
        for bin_count in (10, 5, 4):
            aggregated = _aggregate_grouped_proxy_data(
                working,
                exposure_column=exposure_column,
                outcome_column=str(candidate_spec.get("outcome_column")),
                denominator_column="birth_count",
                group_columns=group_columns,
                bin_count=bin_count,
                observed_flag_column=observed_flag,
            )
            normalized = _normalize_shard_frame(
                aggregated,
                analysis_unit_id=analysis_unit_id,
                source_year=source_year,
                fallback_signature=fallback_signature,
                bin_count=bin_count,
            )
            if not normalized.empty:
                shard_frames.append(normalized)
    if not shard_frames:
        return pd.DataFrame()
    return pd.concat(shard_frames, ignore_index=True)


def _build_neonatal_candidate_year_shard(
    frame: pd.DataFrame,
    *,
    analysis_unit_id: str,
    candidate: dict[str, Any],
    source_year: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    exposure_column = _neonatal_exposure_column(
        _optional_string(candidate.get("selected_exposure_family_id")),
        _optional_string(candidate.get("preferred_lag_or_window")),
    )
    working = frame.copy()
    working = working.loc[
        working[exposure_column].notna()
        & pd.to_numeric(working["annual_live_birth_count"], errors="coerce").fillna(0).gt(0)
    ].copy()
    if working.empty:
        return pd.DataFrame()
    output = working.loc[
        :,
        [
            "municipality_code7",
            "calendar_year",
            "calendar_month",
            "day_of_week",
            "epidemiologic_week",
            "state_code",
            "macroregion",
            exposure_column,
            "neonatal_death_count",
            "annual_live_birth_count",
            "quality_control_status",
            "offset_valid_flag",
            "denominator_zero_flag",
            "denominator_missing_flag",
            "rate_not_computable_flag",
        ],
    ].copy()
    output = output.rename(columns={exposure_column: "exposure_value"})
    output["analysis_unit_id"] = analysis_unit_id
    output["source_year"] = int(source_year)
    ordered = [
        "analysis_unit_id",
        "source_year",
        "municipality_code7",
        "calendar_year",
        "calendar_month",
        "day_of_week",
        "epidemiologic_week",
        "state_code",
        "macroregion",
        "exposure_value",
        "neonatal_death_count",
        "annual_live_birth_count",
        "quality_control_status",
        "offset_valid_flag",
        "denominator_zero_flag",
        "denominator_missing_flag",
        "rate_not_computable_flag",
    ]
    return output.loc[:, ordered].copy()


def _decode_fallback_signature(signature: str) -> list[str]:
    return [part.strip() for part in str(signature).split(">") if part.strip()]


def _candidate_reducer_manifest_status(root: Path, analysis_unit_id: str) -> list[str]:
    manifest = _load_step7_runtime_manifest(root)
    relevant = [
        _optional_string(job.get("status"))
        for job in manifest.get("jobs", [])
        if _optional_string(job.get("job_class")) == "candidate_year"
        and _optional_string(job.get("analysis_unit_id")) == analysis_unit_id
    ]
    return relevant


def _reduce_grouped_candidate_shards(
    *,
    root: Path,
    analysis_unit_id: str,
    candidate_spec: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_statuses = _candidate_reducer_manifest_status(root, analysis_unit_id)
    if any(status != STEP7_JOB_STATUS_COMPLETED for status in manifest_statuses):
        return pd.DataFrame(), {
            "analysis_unit_id": analysis_unit_id,
            "status": "fallback",
            "result_origin": "",
            "fallback_reason": "candidate_year_shards_incomplete",
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "valid_strata_count": 0,
            "selected_exposure_family_id": "",
            "selected_exposure_family_name": "",
            "preferred_lag_or_window": "",
            "data_source": _optional_string(candidate_spec.get("data_source")),
            "model_kind": _optional_string(candidate_spec.get("model_kind")),
            "primary_temporal_controls": _optional_string(candidate_spec.get("primary_temporal_controls")),
            "fallback_level_used": "",
        }
    shard_frames: list[pd.DataFrame] = []
    for path in sorted((_step7_runtime_paths(root)["candidate_year_root"] / analysis_unit_id).glob("year=*.parquet")):
        shard_frames.append(pd.read_parquet(path))
    if not shard_frames:
        return pd.DataFrame(), {
            "analysis_unit_id": analysis_unit_id,
            "status": "fallback",
            "result_origin": "",
            "fallback_reason": "candidate_year_shards_missing",
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "valid_strata_count": 0,
            "selected_exposure_family_id": "",
            "selected_exposure_family_name": "",
            "preferred_lag_or_window": "",
            "data_source": _optional_string(candidate_spec.get("data_source")),
            "model_kind": _optional_string(candidate_spec.get("model_kind")),
            "primary_temporal_controls": _optional_string(candidate_spec.get("primary_temporal_controls")),
            "fallback_level_used": "",
        }
    pooled = pd.concat(shard_frames, ignore_index=True)
    hierarchy = (
        _live_birth_hierarchy(candidate_spec)
        if "delivery_month" in " ".join(candidate_spec.get("fallback_hierarchy", [])) or "conception_month" in " ".join(candidate_spec.get("fallback_hierarchy", []))
        else _stillbirth_hierarchy(candidate_spec)
    )
    model_kind = _optional_string(candidate_spec.get("model_kind")) or "count"
    for group_columns in hierarchy:
        fallback_signature = " > ".join(group_columns)
        for bin_count in (10, 5, 4):
            subset = pooled.loc[
                pooled["fallback_signature"].astype(str).eq(fallback_signature)
                & pd.to_numeric(pooled["bin_count"], errors="coerce").fillna(0).eq(bin_count)
            ].copy()
            if subset.empty:
                continue
            result = _pooled_continuous_fe_proxy(
                subset,
                exposure_column="exposure_value",
                outcome_column="outcome_value",
                denominator_column="denominator_value",
                stratum_columns=group_columns,
                model_kind="count" if model_kind == "count" else "severity",
            )
            if (
                model_kind == "count"
                and math.isfinite(float(result.get("estimate", float("nan"))))
                and float(result.get("overdispersion_ratio", float("nan"))) > OVERDISPERSION_THRESHOLD
            ):
                result = _dispersion_adjusted_result(result, "count")
            if math.isfinite(float(result.get("estimate", float("nan")))) and int(result.get("valid_strata_count", 0)) > 0:
                return subset, {
                    "analysis_unit_id": analysis_unit_id,
                    "status": "completed",
                    "result_origin": "event_level_primary",
                    "fallback_reason": "",
                    "estimate": float(result["estimate"]),
                    "lower_ci": float(result["lower_ci"]),
                    "upper_ci": float(result["upper_ci"]),
                    "p_value": float(result["p_value"]),
                    "valid_strata_count": int(result["valid_strata_count"]),
                    "selected_exposure_family_id": "",
                    "selected_exposure_family_name": "",
                    "preferred_lag_or_window": "",
                    "data_source": _optional_string(candidate_spec.get("data_source")),
                    "model_kind": model_kind,
                    "primary_temporal_controls": _optional_string(candidate_spec.get("primary_temporal_controls")),
                    "fallback_level_used": fallback_signature,
                }
    return pd.DataFrame(), {
        "analysis_unit_id": analysis_unit_id,
        "status": "fallback",
        "result_origin": "",
        "fallback_reason": "grouped_proxy_not_estimable",
        "estimate": float("nan"),
        "lower_ci": float("nan"),
        "upper_ci": float("nan"),
        "p_value": float("nan"),
        "valid_strata_count": 0,
        "selected_exposure_family_id": "",
        "selected_exposure_family_name": "",
        "preferred_lag_or_window": "",
        "data_source": _optional_string(candidate_spec.get("data_source")),
        "model_kind": model_kind,
        "primary_temporal_controls": _optional_string(candidate_spec.get("primary_temporal_controls")),
        "fallback_level_used": "",
    }


def _reduce_neonatal_candidate_shards(
    *,
    root: Path,
    analysis_unit_id: str,
    candidate_spec: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_statuses = _candidate_reducer_manifest_status(root, analysis_unit_id)
    if any(status != STEP7_JOB_STATUS_COMPLETED for status in manifest_statuses):
        return pd.DataFrame(), {
            "analysis_unit_id": analysis_unit_id,
            "status": "fallback",
            "result_origin": "",
            "fallback_reason": "candidate_year_shards_incomplete",
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "valid_strata_count": 0,
            "selected_exposure_family_id": "",
            "selected_exposure_family_name": "",
            "preferred_lag_or_window": "",
            "data_source": _optional_string(candidate_spec.get("data_source")),
            "model_kind": "count",
            "primary_temporal_controls": _optional_string(candidate_spec.get("primary_temporal_controls")),
            "fallback_level_used": "",
        }
    shard_frames: list[pd.DataFrame] = []
    for path in sorted((_step7_runtime_paths(root)["candidate_year_root"] / analysis_unit_id).glob("year=*.parquet")):
        shard_frames.append(pd.read_parquet(path))
    if not shard_frames:
        return pd.DataFrame(), {
            "analysis_unit_id": analysis_unit_id,
            "status": "fallback",
            "result_origin": "",
            "fallback_reason": "candidate_year_shards_missing",
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "valid_strata_count": 0,
            "selected_exposure_family_id": "",
            "selected_exposure_family_name": "",
            "preferred_lag_or_window": "",
            "data_source": _optional_string(candidate_spec.get("data_source")),
            "model_kind": "count",
            "primary_temporal_controls": _optional_string(candidate_spec.get("primary_temporal_controls")),
            "fallback_level_used": "",
        }
    pooled = pd.concat(shard_frames, ignore_index=True)
    primary_controls = [part for part in _optional_string(candidate_spec.get("primary_temporal_controls")).split(",") if part]
    robustness_controls = [part for part in _optional_string(candidate_spec.get("robustness_temporal_controls")).split(",") if part]
    result = _pooled_continuous_fe_proxy(
        pooled,
        exposure_column="exposure_value",
        outcome_column="neonatal_death_count",
        denominator_column="annual_live_birth_count",
        stratum_columns=primary_controls,
        model_kind="count",
    )
    if (
        math.isfinite(float(result.get("estimate", float("nan"))))
        and float(result.get("overdispersion_ratio", float("nan"))) > OVERDISPERSION_THRESHOLD
    ):
        result = _dispersion_adjusted_result(result, "count")
    if not math.isfinite(float(result.get("estimate", float("nan")))):
        return pd.DataFrame(), {
            "analysis_unit_id": analysis_unit_id,
            "status": "fallback",
            "result_origin": "",
            "fallback_reason": "grouped_daily_neonatal_primary_not_estimable",
            "estimate": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "p_value": float("nan"),
            "valid_strata_count": 0,
            "selected_exposure_family_id": "",
            "selected_exposure_family_name": "",
            "preferred_lag_or_window": "",
            "data_source": _optional_string(candidate_spec.get("data_source")),
            "model_kind": "count",
            "primary_temporal_controls": ", ".join(primary_controls),
            "fallback_level_used": "",
        }
    rows = [
        {
            "analysis_unit_id": analysis_unit_id,
            "data_source": _optional_string(candidate_spec.get("data_source")),
            "model_kind": "count",
            "primary_temporal_controls": ", ".join(primary_controls),
            "fallback_level_used": " > ".join(primary_controls),
            "estimate": float(result["estimate"]),
            "lower_ci": float(result["lower_ci"]),
            "upper_ci": float(result["upper_ci"]),
            "p_value": float(result["p_value"]),
            "valid_strata_count": int(result["valid_strata_count"]),
            "selected_exposure_family_id": "",
            "selected_exposure_family_name": "",
            "preferred_lag_or_window": "",
            "result_origin": "event_level_primary",
        }
    ]
    if robustness_controls:
        robustness = _pooled_continuous_fe_proxy(
            pooled,
            exposure_column="exposure_value",
            outcome_column="neonatal_death_count",
            denominator_column="annual_live_birth_count",
            stratum_columns=robustness_controls,
            model_kind="count",
        )
        if (
            math.isfinite(float(robustness.get("estimate", float("nan"))))
            and float(robustness.get("overdispersion_ratio", float("nan"))) > OVERDISPERSION_THRESHOLD
        ):
            robustness = _dispersion_adjusted_result(robustness, "count")
        if math.isfinite(float(robustness.get("estimate", float("nan")))):
            rows.append(
                {
                    "analysis_unit_id": analysis_unit_id,
                    "data_source": _optional_string(candidate_spec.get("data_source")),
                    "model_kind": "count",
                    "primary_temporal_controls": ", ".join(robustness_controls),
                    "fallback_level_used": " > ".join(robustness_controls),
                    "estimate": float(robustness["estimate"]),
                    "lower_ci": float(robustness["lower_ci"]),
                    "upper_ci": float(robustness["upper_ci"]),
                    "p_value": float(robustness["p_value"]),
                    "valid_strata_count": int(robustness["valid_strata_count"]),
                    "selected_exposure_family_id": "",
                    "selected_exposure_family_name": "",
                    "preferred_lag_or_window": "",
                    "result_origin": "robustness_epidemiologic_week",
                }
            )
    return pd.DataFrame(rows), {
        "analysis_unit_id": analysis_unit_id,
        "status": "completed",
        "result_origin": "event_level_primary",
        "fallback_reason": "",
        "estimate": float(result["estimate"]),
        "lower_ci": float(result["lower_ci"]),
        "upper_ci": float(result["upper_ci"]),
        "p_value": float(result["p_value"]),
        "valid_strata_count": int(result["valid_strata_count"]),
        "selected_exposure_family_id": "",
        "selected_exposure_family_name": "",
        "preferred_lag_or_window": "",
        "data_source": _optional_string(candidate_spec.get("data_source")),
        "model_kind": "count",
        "primary_temporal_controls": ", ".join(primary_controls),
        "fallback_level_used": " > ".join(primary_controls),
    }


def _build_temporal_controls_appendix(candidate_catalog: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    perinatal_specs = dict(spec.get("perinatal_live_birth_specs", {}))
    stillbirth_specs = dict(spec.get("stillbirth_specs", {}))
    neonatal_specs = dict(spec.get("neonatal_specs", {}))
    for candidate in candidate_catalog.loc[
        candidate_catalog["step7_route_class"].astype(str).eq("event_level_escalated")
    ].to_dict(orient="records"):
        analysis_unit_id = _optional_string(candidate.get("analysis_unit_id"))
        if analysis_unit_id in perinatal_specs:
            rows.append(
                _temporal_controls_row(
                    candidate=candidate,
                    data_source=_optional_string(perinatal_specs[analysis_unit_id].get("data_source")),
                    primary_temporal_controls=_optional_string(perinatal_specs[analysis_unit_id].get("primary_temporal_controls")),
                )
            )
        elif analysis_unit_id in stillbirth_specs:
            rows.append(
                _temporal_controls_row(
                    candidate=candidate,
                    data_source=_optional_string(stillbirth_specs[analysis_unit_id].get("data_source")),
                    primary_temporal_controls=_optional_string(stillbirth_specs[analysis_unit_id].get("primary_temporal_controls")),
                )
            )
        elif analysis_unit_id in neonatal_specs:
            rows.append(
                _temporal_controls_row(
                    candidate=candidate,
                    data_source=_optional_string(neonatal_specs[analysis_unit_id].get("data_source")),
                    primary_temporal_controls=_optional_string(neonatal_specs[analysis_unit_id].get("primary_temporal_controls")),
                    robustness_temporal_controls=_optional_string(neonatal_specs[analysis_unit_id].get("robustness_temporal_controls")),
                )
            )
    return pd.DataFrame(rows)


def _load_reduced_payload(root: Path, analysis_unit_id: str) -> dict[str, Any] | None:
    path = _reduced_json_path(root, analysis_unit_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_candidate_fallback_log(root: Path, candidate_catalog: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidate_catalog.loc[
        candidate_catalog["step7_route_class"].astype(str).eq("event_level_escalated")
    ].to_dict(orient="records"):
        payload = _load_reduced_payload(root, _optional_string(candidate.get("analysis_unit_id")))
        if payload is None:
            rows.append(_fallback_row(candidate, "reducer_output_missing"))
            continue
        if _optional_string(payload.get("status")) != "completed":
            rows.append(_fallback_row(candidate, _optional_string(payload.get("fallback_reason")) or "event_level_escalation_not_estimable"))
    return pd.DataFrame(
        rows,
        columns=[
            "analysis_unit_id",
            "step7_route_class",
            "step7_route_reason",
            "fallback_reason",
            "used_inherited_evidence_flag",
        ],
    )


def _public_use(record: dict[str, Any]) -> str:
    route_class = _optional_string(record.get("step7_route_class"))
    if route_class == "relative_only_inherited":
        return "relative_only"
    final_translation_stage = _optional_string(record.get("final_translation_stage"))
    if final_translation_stage == "headline_absolute":
        return "headline_burden"
    if final_translation_stage == "supplementary_absolute":
        return "supplementary_burden"
    preferred = _optional_string(record.get("recommended_public_use"))
    if preferred:
        return preferred
    return "relative_only"


def _absolute_burden_allowed(record: dict[str, Any]) -> bool:
    override = record.get("absolute_burden_allowed_override")
    if override is not None and not pd.isna(override):
        return _bool_like(override)
    return _bool_like(record.get("public_burden_gate_pass"))


def _headline_group_allowed(record: dict[str, Any]) -> bool:
    override = record.get("headline_group_allowed_override")
    if override is not None and not pd.isna(override):
        return _bool_like(override)
    return _bool_like(record.get("headline_group_surface_flag"))


def _build_candidate_catalog(
    *,
    step5_scorecard: pd.DataFrame,
    step6_candidates: pd.DataFrame,
    step6_group_headlines: pd.DataFrame,
    secondary_translation: pd.DataFrame,
    route_catalog: pd.DataFrame,
    step4_decisions: pd.DataFrame,
    step3_family_comparison: pd.DataFrame,
) -> pd.DataFrame:
    headline_groups = set(
        step6_group_headlines.loc[
            step6_group_headlines["headline_selected_flag"].fillna(False).astype(bool),
            "group_id",
        ].astype(str)
    )
    secondary_subset = secondary_translation.loc[
        :,
        ["analysis_unit_id", "recommended_public_use"],
    ].drop_duplicates(subset=["analysis_unit_id"])
    step4_subset = step4_decisions.loc[
        :,
        ["analysis_unit_id", "selected_exposure_family_id", "preferred_lag_or_window"],
    ].drop_duplicates(subset=["analysis_unit_id"]).rename(
        columns={
            "selected_exposure_family_id": "step4_selected_exposure_family_id_check",
            "preferred_lag_or_window": "step4_preferred_lag_or_window_check",
        }
    )
    step3_subset = step3_family_comparison.loc[
        :,
        ["analysis_unit_id", "headline_family_recommendation", "family_comparison_reason"],
    ].drop_duplicates(subset=["analysis_unit_id"])
    step5_subset = step5_scorecard.loc[
        :,
        [
            "analysis_unit_id",
            "has_any_significant_reversal_flag",
            "significant_reversal_modifier_count",
            "decision_reason",
        ],
    ].drop_duplicates(subset=["analysis_unit_id"]).rename(
        columns={
            "has_any_significant_reversal_flag": "step5_has_any_significant_reversal_flag",
            "significant_reversal_modifier_count": "step5_significant_reversal_modifier_count",
            "decision_reason": "step5_decision_reason",
        }
    )

    catalog = step6_candidates.merge(route_catalog, on="analysis_unit_id", how="left")
    catalog = catalog.merge(secondary_subset, on="analysis_unit_id", how="left")
    catalog = catalog.merge(step5_subset, on="analysis_unit_id", how="left")
    catalog = catalog.merge(step4_subset, on="analysis_unit_id", how="left")
    catalog = catalog.merge(step3_subset, on="analysis_unit_id", how="left")
    if "has_any_significant_reversal_flag" in catalog.columns:
        catalog["has_any_significant_reversal_flag"] = catalog["has_any_significant_reversal_flag"].combine_first(
            catalog.get("step5_has_any_significant_reversal_flag")
        )
    else:
        catalog["has_any_significant_reversal_flag"] = catalog.get("step5_has_any_significant_reversal_flag", False)
    if "significant_reversal_modifier_count" in catalog.columns:
        catalog["significant_reversal_modifier_count"] = pd.to_numeric(
            catalog["significant_reversal_modifier_count"].combine_first(
                catalog.get("step5_significant_reversal_modifier_count")
            ),
            errors="coerce",
        ).fillna(0).astype(int)
    else:
        catalog["significant_reversal_modifier_count"] = pd.to_numeric(
            catalog.get("step5_significant_reversal_modifier_count", 0),
            errors="coerce",
        ).fillna(0).astype(int)
    catalog["headline_group_surface_flag"] = catalog["group_id"].astype(str).isin(headline_groups)
    catalog["all_candidate_surface_flag"] = True
    catalog["step6_final_translation_stage"] = catalog["final_translation_stage"].astype(str)
    catalog["public_use"] = catalog.apply(lambda row: _public_use(row.to_dict()), axis=1)
    catalog["absolute_burden_allowed"] = catalog.apply(lambda row: _absolute_burden_allowed(row.to_dict()), axis=1)
    catalog["headline_group_allowed"] = catalog.apply(lambda row: _headline_group_allowed(row.to_dict()), axis=1)
    catalog["step4_exposure_lock_flag"] = (
        catalog["selected_exposure_family_id"].astype(str).eq(catalog["step4_selected_exposure_family_id_check"].astype(str))
        & catalog["preferred_lag_or_window"].astype(str).eq(catalog["step4_preferred_lag_or_window_check"].astype(str))
    )
    return catalog.sort_values(["group_id", "analysis_unit_id"], ignore_index=True)


def _caution_note(record: dict[str, Any], evidence_origin: str) -> str:
    notes: list[str] = []
    if evidence_origin == "inherited_aggregated_fallback":
        notes.append("Event-level escalation was not estimable; inherited aggregated evidence was retained.")
    if _optional_string(record.get("step7_route_class")) == "relative_only_inherited":
        notes.append("Relative-only evidence; burden translation is not allowed.")
    if _bool_like(record.get("has_any_significant_reversal_flag")):
        notes.append("A significant Step 5 reversal flag remains present.")
    public_use = _optional_string(record.get("public_use"))
    if public_use == "continuous_internal_translation_only":
        notes.append("Continuous severity outcome; internal translation remains the preferred interpretation.")
    if public_use == "relative_plus_observed_hot_volume":
        notes.append("Relative interpretation should be paired with observed hot-period event volume.")
    return " ".join(notes)


def _build_surface_row(candidate: dict[str, Any], event_primary_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    analysis_unit_id = _optional_string(candidate.get("analysis_unit_id"))
    route_class = _optional_string(candidate.get("step7_route_class"))
    if analysis_unit_id in event_primary_map:
        event_row = event_primary_map[analysis_unit_id]
        evidence_origin = "event_level_primary"
        estimate = float(event_row["estimate"])
        lower_ci = float(event_row["lower_ci"])
        upper_ci = float(event_row["upper_ci"])
        p_value = float(event_row["p_value"])
    elif route_class == "event_level_escalated":
        evidence_origin = "inherited_aggregated_fallback"
        estimate = float(candidate.get("estimate", float("nan")))
        lower_ci = float(candidate.get("lower_ci", float("nan")))
        upper_ci = float(candidate.get("upper_ci", float("nan")))
        p_value = float(candidate.get("p_value", float("nan")))
    elif route_class == "relative_only_inherited":
        evidence_origin = "inherited_aggregated_relative_only"
        estimate = float(candidate.get("estimate", float("nan")))
        lower_ci = float(candidate.get("lower_ci", float("nan")))
        upper_ci = float(candidate.get("upper_ci", float("nan")))
        p_value = float(candidate.get("p_value", float("nan")))
    else:
        evidence_origin = "inherited_aggregated"
        estimate = float(candidate.get("estimate", float("nan")))
        lower_ci = float(candidate.get("lower_ci", float("nan")))
        upper_ci = float(candidate.get("upper_ci", float("nan")))
        p_value = float(candidate.get("p_value", float("nan")))
    row = {
        "analysis_unit_id": analysis_unit_id,
        "group_id": _optional_string(candidate.get("group_id")),
        "group_label": _optional_string(candidate.get("group_label")),
        "public_label": _optional_string(candidate.get("public_label")),
        "step6_final_translation_stage": _optional_string(candidate.get("step6_final_translation_stage")),
        "step7_route_class": route_class,
        "evidence_origin": evidence_origin,
        "selected_exposure_family_name": _optional_string(candidate.get("selected_exposure_family_name")),
        "preferred_lag_or_window": _optional_string(candidate.get("preferred_lag_or_window")),
        "estimate": estimate,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "p_value": p_value,
        "public_use": _optional_string(candidate.get("public_use")),
    }
    row["caution_note"] = _caution_note({**candidate, **row}, evidence_origin)
    return row


def _render_step7_report(
    *,
    round_config: dict[str, Any],
    step6_report: dict[str, Any],
    gates: dict[str, bool],
    summary: dict[str, Any],
    outputs: dict[str, str],
    workspace_report: dict[str, object],
    step6_missing_groups: list[str],
) -> str:
    lines = [
        "# Step 7 Escalation Round",
        "",
        f"- Analysis ID: {round_config['analysis_id']}",
        f"- Years: {', '.join(str(year) for year in round_config['years'])}",
        f"- Generated at: {_now_utc()}",
        f"- Upstream Step 6 status: {_optional_string(step6_report.get('status'))}",
        f"- Missing Step 6 headline groups: {', '.join(step6_missing_groups) if step6_missing_groups else 'none'}",
        "",
        "## Summary",
        "",
        f"- Candidate count: {summary['candidate_count']}",
        f"- Headline-group surface count: {summary['headline_group_surface_count']}",
        f"- All-candidate surface count: {summary['all_candidate_surface_count']}",
        f"- Event-level escalation count: {summary['event_level_escalation_count']}",
        f"- Aggregated inherited count: {summary['aggregated_inherited_count']}",
        f"- Relative-only inherited count: {summary['relative_only_inherited_count']}",
        f"- Event-level result rows written: {summary['event_level_result_row_count']}",
        "",
        "## Gates",
        "",
    ]
    for gate_name, gate_value in gates.items():
        lines.append(f"- {gate_name}: {'PASS' if gate_value else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for output_name, relative_path in outputs.items():
        lines.append(f"- {output_name}: `{relative_path}`")
    lines.extend(
        [
            "",
            "## Publication Workspace",
            "",
            f"- Workspace root: `{workspace_report['workspace_root']}`",
            f"- Created files: {', '.join(workspace_report['created_files']) if workspace_report['created_files'] else 'none'}",
            f"- Existing files: {', '.join(workspace_report['existing_files']) if workspace_report['existing_files'] else 'none'}",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_publication_cards(
    *,
    summary: dict[str, Any],
    candidate_catalog: pd.DataFrame,
    headline_surface: pd.DataFrame,
    all_surface: pd.DataFrame,
) -> str:
    route_counts = (
        candidate_catalog.groupby("step7_route_class", dropna=False).size().rename("count").reset_index()
    )
    lines = [
        "# Step 7 Publication Cards",
        "",
        "## Round Profile",
        "",
        f"- The escalation round reviewed {summary['candidate_count']} candidates.",
        f"- The headline-group surface contains {summary['headline_group_surface_count']} candidates from groups with a filled Step 6 headline.",
        f"- The all-candidate surface contains {summary['all_candidate_surface_count']} candidates, including maternal relative-only rows.",
        "",
        "## Route Classes",
        "",
    ]
    for row in route_counts.to_dict(orient="records"):
        lines.append(f"- {row['step7_route_class']}: {int(row['count'])}")
    lines.extend(
        [
            "",
            "## Headline-Group Surface",
            "",
        ]
    )
    for row in headline_surface.sort_values(["group_label", "public_label"]).to_dict(orient="records"):
        lines.append(
            f"- {row['group_label']}: {row['public_label']} ({row['selected_exposure_family_name']}, {row['preferred_lag_or_window']}, {row['evidence_origin']})"
        )
    lines.extend(
        [
            "",
            "## All-Candidate Surface",
            "",
            f"- Relative-only maternal candidates: {int(all_surface['public_use'].astype(str).eq('relative_only').sum())}",
            f"- Event-level primary rows represented in public surfaces: {int(all_surface['evidence_origin'].astype(str).eq('event_level_primary').sum())}",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_publication_workspace(
    *,
    root: Path,
    summary: dict[str, Any],
    candidate_catalog: pd.DataFrame,
    headline_surface: pd.DataFrame,
    all_surface: pd.DataFrame,
) -> None:
    workspace_root = incremental_publication_root(root)
    scope_path = workspace_root / "14_step7_escalation_scope.md"
    headline_path = workspace_root / "15_step7_headline_group_surface.md"
    all_path = workspace_root / "16_step7_all_candidate_surface.md"

    scope_lines = [
        "# Step 7 Escalation Scope",
        "",
        f"- Candidate count: {summary['candidate_count']}",
        f"- Headline-group surface count: {summary['headline_group_surface_count']}",
        f"- All-candidate surface count: {summary['all_candidate_surface_count']}",
        "",
        "## Route Allocation",
        "",
    ]
    for route_class, count in candidate_catalog.groupby("step7_route_class").size().items():
        scope_lines.append(f"- {route_class}: {int(count)}")
    scope_path.write_text("\n".join(scope_lines) + "\n", encoding="utf-8")

    headline_lines = [
        "# Step 7 Headline-Group Surface",
        "",
        "Only candidates from groups with a filled Step 6 headline are included here.",
        "",
    ]
    for row in headline_surface.sort_values(["group_label", "public_label"]).to_dict(orient="records"):
        headline_lines.append(
            f"- {row['group_label']}: {row['public_label']} [{row['step7_route_class']}; {row['selected_exposure_family_name']}; {row['preferred_lag_or_window']}]"
        )
    headline_path.write_text("\n".join(headline_lines) + "\n", encoding="utf-8")

    all_lines = [
        "# Step 7 All-Candidate Surface",
        "",
        "All promoted candidates are retained in this surface. Maternal rows remain relative-only and non-headline.",
        "",
    ]
    for row in all_surface.sort_values(["group_label", "public_label"]).to_dict(orient="records"):
        all_lines.append(
            f"- {row['group_label']}: {row['public_label']} [{row['step7_route_class']}; public_use={row['public_use']}]"
        )
    all_path.write_text("\n".join(all_lines) + "\n", encoding="utf-8")


def _write_workbook(path: Path, tabs: list[str], frames: dict[str, pd.DataFrame]) -> None:
    workbook = Workbook()
    default_sheet = workbook.active
    workbook.remove(default_sheet)
    for tab_name in tabs:
        sheet = workbook.create_sheet(title=tab_name)
        frame = frames.get(tab_name, pd.DataFrame())
        if frame.empty:
            sheet.append(["No rows available"])
            continue
        sheet.append(list(frame.columns))
        for cell in sheet[1]:
            cell.font = Font(bold=True)
            cell.fill = WORKBOOK_HEADER_FILL
            cell.alignment = Alignment(horizontal="center")
        for row in frame.itertuples(index=False, name=None):
            sheet.append(list(row))
        for row_index in range(2, sheet.max_row + 1):
            route_class = _optional_string(sheet.cell(row=row_index, column=min(6, sheet.max_column)).value)
            public_use = _optional_string(sheet.cell(row=row_index, column=min(14, sheet.max_column)).value)
            fill = None
            if route_class == "event_level_escalated" or public_use == "headline_burden":
                fill = WORKBOOK_EVENT_FILL
            elif public_use == "relative_only":
                fill = WORKBOOK_RELATIVE_ONLY_FILL
            elif _optional_string(sheet.cell(row=row_index, column=min(7, sheet.max_column)).value) == "inherited_aggregated_fallback":
                fill = WORKBOOK_FALLBACK_FILL
            if fill is not None:
                for cell in sheet[row_index]:
                    cell.fill = fill
        for column_cells in sheet.columns:
            width = max(len(_optional_string(cell.value)) for cell in column_cells[: min(len(column_cells), 50)]) + 2
            sheet.column_dimensions[column_cells[0].column_letter].width = min(max(width, 14), 42)
    workbook.save(path)


def _step7_output_paths(root: Path) -> tuple[dict[str, Path], dict[str, str]]:
    step7_root = analysis_step7_root(root)
    path_map = {
        "step7_candidate_catalog": step7_root / "step7_candidate_catalog.csv",
        "step7_event_level_results": step7_root / "step7_event_level_results.csv",
        "step7_temporal_controls_appendix": step7_root / "step7_temporal_controls_appendix.csv",
        "step7_fallback_log": step7_root / "step7_fallback_log.csv",
        "step7_headline_group_surface": step7_root / "step7_headline_group_surface.csv",
        "step7_all_candidate_surface": step7_root / "step7_all_candidate_surface.csv",
        "step7_report": step7_root / "step7_report.md",
        "step7_publication_cards": step7_root / "step7_publication_cards.md",
        "step7_escalation_bundle_xlsx": step7_root / "step7_escalation_bundle.xlsx",
    }
    return path_map, {key: _relative(root, value) for key, value in path_map.items()}


def _build_step7_runtime_candidate_catalog(
    *,
    root: Path,
    config_path: Path | None,
) -> tuple[pd.DataFrame, bool, list[str]]:
    del config_path
    step5_scorecard = _load_step5_scorecard(root)
    step6_candidates = _load_step6_candidate_catalog(root)
    step6_group_headlines = _load_step6_group_headlines(root)
    secondary_translation = _load_step6_secondary_translation(root)
    step4_decisions = _load_step4_decisions(root)
    step3_family_comparison = _load_step3_family_comparison(root)
    route_catalog = _route_catalog_frame(root)
    candidate_catalog = _build_candidate_catalog(
        step5_scorecard=step5_scorecard,
        step6_candidates=step6_candidates,
        step6_group_headlines=step6_group_headlines,
        secondary_translation=secondary_translation,
        route_catalog=route_catalog,
        step4_decisions=step4_decisions,
        step3_family_comparison=step3_family_comparison,
    )
    acceptable, missing_groups, _ = _step7_start_condition(root)
    return candidate_catalog, acceptable, missing_groups


def _run_initialize_step7_runtime_job(root: Path, config_path: Path | None) -> dict[str, Any]:
    candidate_catalog, acceptable, missing_groups = _build_step7_runtime_candidate_catalog(root=root, config_path=config_path)
    output = _candidate_catalog_output_frame(candidate_catalog)
    output_path = analysis_step7_root(root) / "step7_candidate_catalog.csv"
    output.to_csv(output_path, index=False)
    if not acceptable:
        return {
            "status": "FAIL",
            "job_status": STEP7_JOB_STATUS_FAILED,
            "job_id": "initialize_step7_runtime",
            "output_relpath": _relative(root, output_path),
            "fallback_reason": f"step6_start_condition_invalid:{','.join(missing_groups)}",
        }
    return {
        "status": "PASS",
        "job_status": STEP7_JOB_STATUS_COMPLETED,
        "job_id": "initialize_step7_runtime",
        "output_relpath": _relative(root, output_path),
    }


def _run_base_year_job(root: Path, job: dict[str, Any]) -> dict[str, Any]:
    year = int(job["source_year"])
    source_family = _optional_string(job.get("source_family"))
    lookup = _load_municipality_lookup(root)
    if source_family == "live_birth":
        frame = _load_live_birth_year_frame(root, year, lookup)
    elif source_family == "stillbirth":
        frame = _load_stillbirth_year_frame(root, year, lookup)
    elif source_family == "neonatal":
        frame = _load_neonatal_daily_frame(root, [year], lookup)
    else:
        raise ValueError(f"Unsupported Step 7 base-year source family {source_family!r}")
    if frame.empty:
        return {
            "status": "FAIL",
            "job_status": STEP7_JOB_STATUS_FAILED,
            "job_id": _optional_string(job.get("job_id")),
            "fallback_reason": "base_year_frame_empty",
        }
    output_path = _base_year_output_path(root, source_family, year)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return {
        "status": "PASS",
        "job_status": STEP7_JOB_STATUS_COMPLETED,
        "job_id": _optional_string(job.get("job_id")),
        "output_relpath": _relative(root, output_path),
    }


def _run_candidate_year_job(root: Path, job: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    candidate_catalog = _load_runtime_candidate_catalog(root)
    analysis_unit_id = _optional_string(job.get("analysis_unit_id"))
    candidate = _analysis_unit_record(candidate_catalog, analysis_unit_id)
    source_year = int(job["source_year"])
    source_family = _optional_string(job.get("source_family"))
    if source_family == "live_birth":
        base_path = _base_year_output_path(root, "live_birth", source_year)
        frame = pd.read_parquet(base_path) if base_path.exists() else pd.DataFrame()
        candidate_spec = dict(spec.get("perinatal_live_birth_specs", {}).get(analysis_unit_id, {}))
        shard = _build_live_birth_candidate_year_shard(
            frame,
            analysis_unit_id=analysis_unit_id,
            candidate_spec=candidate_spec,
            candidate=candidate,
            source_year=source_year,
        )
    elif source_family == "stillbirth":
        live_birth_path = _base_year_output_path(root, "live_birth", source_year)
        stillbirth_path = _base_year_output_path(root, "stillbirth", source_year)
        live_births = pd.read_parquet(live_birth_path) if live_birth_path.exists() else pd.DataFrame()
        stillbirths = pd.read_parquet(stillbirth_path) if stillbirth_path.exists() else pd.DataFrame()
        combined = _build_stillbirth_support_year_frame(live_births, stillbirths)
        candidate_spec = dict(spec.get("stillbirth_specs", {}).get(analysis_unit_id, {}))
        shard = _build_stillbirth_candidate_year_shard(
            combined,
            analysis_unit_id=analysis_unit_id,
            candidate_spec=candidate_spec,
            candidate=candidate,
            source_year=source_year,
        )
    elif source_family == "neonatal":
        base_path = _base_year_output_path(root, "neonatal", source_year)
        frame = pd.read_parquet(base_path) if base_path.exists() else pd.DataFrame()
        shard = _build_neonatal_candidate_year_shard(
            frame,
            analysis_unit_id=analysis_unit_id,
            candidate=candidate,
            source_year=source_year,
        )
    else:
        raise ValueError(f"Unsupported Step 7 candidate-year source family {source_family!r}")
    if shard.empty:
        return {
            "status": "FAIL",
            "job_status": STEP7_JOB_STATUS_FAILED,
            "job_id": _optional_string(job.get("job_id")),
            "fallback_reason": "candidate_year_shard_empty",
        }
    output_path = _candidate_year_output_path(root, analysis_unit_id, source_year)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard.to_parquet(output_path, index=False)
    return {
        "status": "PASS",
        "job_status": STEP7_JOB_STATUS_COMPLETED,
        "job_id": _optional_string(job.get("job_id")),
        "output_relpath": _relative(root, output_path),
    }


def _reducer_payload_with_candidate(candidate: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(payload)
    enriched["selected_exposure_family_id"] = _optional_string(candidate.get("selected_exposure_family_id"))
    enriched["selected_exposure_family_name"] = _optional_string(candidate.get("selected_exposure_family_name"))
    enriched["preferred_lag_or_window"] = _optional_string(candidate.get("preferred_lag_or_window"))
    return enriched


def _run_reducer_job(root: Path, job: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    candidate_catalog = _load_runtime_candidate_catalog(root)
    analysis_unit_id = _optional_string(job.get("analysis_unit_id"))
    candidate = _analysis_unit_record(candidate_catalog, analysis_unit_id)
    if analysis_unit_id in spec.get("perinatal_live_birth_specs", {}):
        reduced_frame, payload = _reduce_grouped_candidate_shards(
            root=root,
            analysis_unit_id=analysis_unit_id,
            candidate_spec=dict(spec["perinatal_live_birth_specs"][analysis_unit_id]),
        )
        payload = _reducer_payload_with_candidate(candidate, payload)
        reduced_rows = (
            pd.DataFrame(
                [
                    _event_result_row(
                        candidate=candidate,
                        spec=dict(spec["perinatal_live_birth_specs"][analysis_unit_id]),
                        result=payload,
                        fallback_level_used=_optional_string(payload.get("fallback_level_used")),
                        data_source=_optional_string(payload.get("data_source")),
                        result_origin="event_level_primary",
                    )
                ]
            )
            if not reduced_frame.empty
            else pd.DataFrame()
        )
    elif analysis_unit_id in spec.get("stillbirth_specs", {}):
        reduced_frame, payload = _reduce_grouped_candidate_shards(
            root=root,
            analysis_unit_id=analysis_unit_id,
            candidate_spec=dict(spec["stillbirth_specs"][analysis_unit_id]),
        )
        payload = _reducer_payload_with_candidate(candidate, payload)
        reduced_rows = (
            pd.DataFrame(
                [
                    _event_result_row(
                        candidate=candidate,
                        spec=dict(spec["stillbirth_specs"][analysis_unit_id]),
                        result=payload,
                        fallback_level_used=_optional_string(payload.get("fallback_level_used")),
                        data_source=_optional_string(payload.get("data_source")),
                        result_origin="event_level_primary",
                    )
                ]
            )
            if not reduced_frame.empty
            else pd.DataFrame()
        )
    elif analysis_unit_id in spec.get("neonatal_specs", {}):
        reduced_rows, payload = _reduce_neonatal_candidate_shards(
            root=root,
            analysis_unit_id=analysis_unit_id,
            candidate_spec=dict(spec["neonatal_specs"][analysis_unit_id]),
        )
        payload = _reducer_payload_with_candidate(candidate, payload)
        if not reduced_rows.empty:
            reduced_rows["selected_exposure_family_id"] = _optional_string(candidate.get("selected_exposure_family_id"))
            reduced_rows["selected_exposure_family_name"] = _optional_string(candidate.get("selected_exposure_family_name"))
            reduced_rows["preferred_lag_or_window"] = _optional_string(candidate.get("preferred_lag_or_window"))
    else:
        raise ValueError(f"Unsupported Step 7 reducer analysis unit {analysis_unit_id!r}")
    reduced_json = _reduced_json_path(root, analysis_unit_id)
    reduced_parquet = _reduced_parquet_path(root, analysis_unit_id)
    reduced_json.parent.mkdir(parents=True, exist_ok=True)
    reduced_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if not reduced_rows.empty:
        reduced_rows.to_parquet(reduced_parquet, index=False)
    if _optional_string(payload.get("status")) != "completed":
        return {
            "status": "FAIL",
            "job_status": STEP7_JOB_STATUS_FAILED,
            "job_id": _optional_string(job.get("job_id")),
            "output_relpath": _relative(root, reduced_json),
            "fallback_reason": _optional_string(payload.get("fallback_reason")) or "reducer_not_estimable",
        }
    return {
        "status": "PASS",
        "job_status": STEP7_JOB_STATUS_COMPLETED,
        "job_id": _optional_string(job.get("job_id")),
        "output_relpath": _relative(root, reduced_json),
    }


def _compute_step7_summary_and_gates(
    root: Path,
    *,
    step6_report: dict[str, Any],
    allow_running_job_ids: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, bool], list[str]]:
    allowed_running = set(allow_running_job_ids or set())
    acceptable, missing_groups, _ = _step7_start_condition(root)
    path_map, _ = _step7_output_paths(root)
    candidate_catalog = pd.read_csv(path_map["step7_candidate_catalog"]) if path_map["step7_candidate_catalog"].exists() else pd.DataFrame(columns=_candidate_catalog_output_columns())
    step6_candidates = _load_step6_candidate_catalog(root)
    event_level_results = pd.read_csv(path_map["step7_event_level_results"]) if path_map["step7_event_level_results"].exists() else pd.DataFrame()
    headline_surface = pd.read_csv(path_map["step7_headline_group_surface"]) if path_map["step7_headline_group_surface"].exists() else pd.DataFrame()
    all_surface = pd.read_csv(path_map["step7_all_candidate_surface"]) if path_map["step7_all_candidate_surface"].exists() else pd.DataFrame()
    manifest = _load_step7_runtime_manifest(root)
    priority_present = set(
        event_level_results.loc[
            event_level_results["result_origin"].astype(str).eq("event_level_primary"),
            "analysis_unit_id",
        ].astype(str)
    ) if not event_level_results.empty else set()
    maternal_rows = candidate_catalog.loc[candidate_catalog["group_id"].astype(str).eq("maternal")].copy() if not candidate_catalog.empty else pd.DataFrame()
    gates = {
        "step6_start_condition_acceptable": bool(acceptable),
        "runtime_manifest_complete": bool(
            all(
                _optional_string(job.get("status")) in STEP7_FINAL_JOB_STATUSES
                or (
                    _optional_string(job.get("job_id")) in allowed_running
                    and _optional_string(job.get("status")) == STEP7_JOB_STATUS_RUNNING
                )
                for job in manifest.get("jobs", [])
            )
        ),
        "candidate_routes_complete": bool(
            not candidate_catalog.empty
            and candidate_catalog["step7_route_class"].notna().all()
            and int(candidate_catalog.shape[0]) == int(step6_candidates.shape[0])
        ),
        "headline_group_surface_written": path_map["step7_headline_group_surface"].exists() and int(headline_surface.shape[0]) > 0,
        "all_candidate_surface_written": path_map["step7_all_candidate_surface"].exists() and int(all_surface.shape[0]) == int(candidate_catalog.shape[0]),
        "priority_perinatal_escalation_complete": "preterm_birth_count" in priority_present,
        "priority_stillbirth_escalation_complete": "stillbirth_primary_gte28_count" in priority_present,
        "priority_neonatal_escalation_complete": "neonatal_death_count" in priority_present,
        "maternal_relative_only_routing_complete": bool(
            not maternal_rows.empty
            and maternal_rows["step7_route_class"].astype(str).eq("relative_only_inherited").all()
            and maternal_rows["public_use"].astype(str).eq("relative_only").all()
            and ~maternal_rows["absolute_burden_allowed"].fillna(True).astype(bool).any()
            and ~maternal_rows["headline_group_allowed"].fillna(True).astype(bool).any()
        ),
    }
    summary = {
        "candidate_count": int(candidate_catalog.shape[0]),
        "headline_group_surface_count": int(headline_surface.shape[0]),
        "all_candidate_surface_count": int(all_surface.shape[0]),
        "event_level_escalation_count": int(candidate_catalog["step7_route_class"].astype(str).eq("event_level_escalated").sum()) if not candidate_catalog.empty else 0,
        "aggregated_inherited_count": int(candidate_catalog["step7_route_class"].astype(str).eq("aggregated_inherited").sum()) if not candidate_catalog.empty else 0,
        "relative_only_inherited_count": int(candidate_catalog["step7_route_class"].astype(str).eq("relative_only_inherited").sum()) if not candidate_catalog.empty else 0,
        "event_level_result_row_count": int(event_level_results.shape[0]),
    }
    return summary, gates, missing_groups


def _assemble_event_tables(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    path_map, relative_outputs = _step7_output_paths(root)
    candidate_catalog = _load_runtime_candidate_catalog(root)
    reduced_frames = [
        pd.read_parquet(path)
        for analysis_unit_id in STEP7_EVENT_CANDIDATES_IN_RING
        for path in [_reduced_parquet_path(root, analysis_unit_id)]
        if path.exists()
    ]
    event_level_results = pd.concat(reduced_frames, ignore_index=True) if reduced_frames else pd.DataFrame()
    temporal_controls = _build_temporal_controls_appendix(candidate_catalog, spec)
    fallback_log = _build_candidate_fallback_log(root, candidate_catalog)
    event_level_results.to_csv(path_map["step7_event_level_results"], index=False)
    temporal_controls.to_csv(path_map["step7_temporal_controls_appendix"], index=False)
    fallback_log.to_csv(path_map["step7_fallback_log"], index=False)
    return {
        "status": "PASS",
        "job_status": STEP7_JOB_STATUS_COMPLETED,
        "job_id": "assemble_event_tables",
        "output_relpath": relative_outputs["step7_event_level_results"],
    }


def _assemble_surfaces(root: Path) -> dict[str, Any]:
    path_map, relative_outputs = _step7_output_paths(root)
    candidate_catalog = _load_runtime_candidate_catalog(root)
    event_level_results = pd.read_csv(path_map["step7_event_level_results"]) if path_map["step7_event_level_results"].exists() else pd.DataFrame()
    event_primary_map = {
        str(row["analysis_unit_id"]): row
        for row in event_level_results.loc[
            event_level_results["result_origin"].astype(str).eq("event_level_primary")
        ].to_dict(orient="records")
    } if not event_level_results.empty else {}
    headline_surface = pd.DataFrame(
        [
            _build_surface_row(candidate, event_primary_map)
            for candidate in candidate_catalog.loc[
                candidate_catalog["headline_group_surface_flag"].fillna(False).astype(bool)
            ].to_dict(orient="records")
        ]
    )
    all_surface = pd.DataFrame(
        [_build_surface_row(candidate, event_primary_map) for candidate in candidate_catalog.to_dict(orient="records")]
    )
    if not all_surface.empty:
        all_surface = all_surface.merge(
            candidate_catalog.loc[:, ["analysis_unit_id", "step7_route_reason", "absolute_burden_allowed", "headline_group_allowed"]],
            on="analysis_unit_id",
            how="left",
        ).sort_values(["group_label", "public_label"], ignore_index=True)
    if not headline_surface.empty:
        headline_surface = headline_surface.sort_values(["group_label", "public_label"], ignore_index=True)
    headline_surface.to_csv(path_map["step7_headline_group_surface"], index=False)
    all_surface.to_csv(path_map["step7_all_candidate_surface"], index=False)
    return {
        "status": "PASS",
        "job_status": STEP7_JOB_STATUS_COMPLETED,
        "job_id": "assemble_surfaces",
        "output_relpath": relative_outputs["step7_all_candidate_surface"],
    }


def _assemble_publication_cards(root: Path) -> dict[str, Any]:
    path_map, relative_outputs = _step7_output_paths(root)
    candidate_catalog = pd.read_csv(path_map["step7_candidate_catalog"])
    headline_surface = pd.read_csv(path_map["step7_headline_group_surface"])
    all_surface = pd.read_csv(path_map["step7_all_candidate_surface"])
    summary, _, _ = _compute_step7_summary_and_gates(root, step6_report={"status": "REUSED_EXISTING_STEP6_OUTPUTS"})
    _write_publication_workspace(
        root=root,
        summary=summary,
        candidate_catalog=candidate_catalog,
        headline_surface=headline_surface,
        all_surface=all_surface,
    )
    path_map["step7_publication_cards"].write_text(
        _render_publication_cards(
            summary=summary,
            candidate_catalog=candidate_catalog,
            headline_surface=headline_surface,
            all_surface=all_surface,
        ),
        encoding="utf-8",
    )
    return {
        "status": "PASS",
        "job_status": STEP7_JOB_STATUS_COMPLETED,
        "job_id": "assemble_publication_cards",
        "output_relpath": relative_outputs["step7_publication_cards"],
    }


def _assemble_report_and_workbook(
    root: Path,
    *,
    config_path: Path | None,
    step6_report: dict[str, Any] | None,
    workspace_report: dict[str, object],
) -> dict[str, Any]:
    round_config = load_round_config(root, config_path=config_path)
    spec = _step7_spec(root, config_path=config_path)
    upstream_step6_report = _resolve_upstream_step6_report(root, config_path=config_path, step6_report=step6_report)
    path_map, relative_outputs = _step7_output_paths(root)
    event_level_results = pd.read_csv(path_map["step7_event_level_results"]) if path_map["step7_event_level_results"].exists() else pd.DataFrame()
    headline_surface = pd.read_csv(path_map["step7_headline_group_surface"]) if path_map["step7_headline_group_surface"].exists() else pd.DataFrame()
    all_surface = pd.read_csv(path_map["step7_all_candidate_surface"]) if path_map["step7_all_candidate_surface"].exists() else pd.DataFrame()
    temporal_controls = pd.read_csv(path_map["step7_temporal_controls_appendix"]) if path_map["step7_temporal_controls_appendix"].exists() else pd.DataFrame()
    summary, gates, missing_groups = _compute_step7_summary_and_gates(
        root,
        step6_report=upstream_step6_report,
        allow_running_job_ids={"assemble_report_and_workbook"},
    )
    status = "PASS" if all(gates.values()) else "FAIL"
    path_map["step7_report"].write_text(
        _render_step7_report(
            round_config=round_config,
            step6_report=upstream_step6_report,
            gates=gates,
            summary=summary,
            outputs=relative_outputs,
            workspace_report=workspace_report,
            step6_missing_groups=missing_groups,
        ),
        encoding="utf-8",
    )
    _write_workbook(
        path_map["step7_escalation_bundle_xlsx"],
        list(spec.get("output_workbook_tabs", [])),
        {
            "HeadlineGroups": headline_surface,
            "AllCandidates": all_surface,
            "EventLevel": event_level_results,
            "InheritedAggregated": all_surface.loc[
                all_surface["evidence_origin"].astype(str).isin({"inherited_aggregated", "inherited_aggregated_fallback"})
            ].copy() if not all_surface.empty else pd.DataFrame(),
            "MaternalRelativeOnly": all_surface.loc[
                all_surface["evidence_origin"].astype(str).eq("inherited_aggregated_relative_only")
            ].copy() if not all_surface.empty else pd.DataFrame(),
            "TemporalControls": temporal_controls,
        },
    )
    return {
        "status": status,
        "job_status": STEP7_JOB_STATUS_COMPLETED if status == "PASS" else STEP7_JOB_STATUS_FAILED,
        "job_id": "assemble_report_and_workbook",
        "output_relpath": relative_outputs["step7_report"],
        "fallback_reason": "" if status == "PASS" else "required_priority_escalation_missing",
    }


def _run_step7_job_worker(
    *,
    root: Path,
    config_path: Path | None,
    job_id: str,
    step6_report: dict[str, Any] | None,
) -> dict[str, Any]:
    spec = _step7_spec(root, config_path=config_path)
    workspace_report = _scaffold_step7_publication_workspace(root, spec)
    manifest = _load_step7_runtime_manifest(root)
    job_map = _jobs_by_id(manifest)
    if job_id not in job_map:
        raise KeyError(f"Unknown Step 7 job id {job_id!r}")
    job = job_map[job_id]
    if job_id == "initialize_step7_runtime":
        return _run_initialize_step7_runtime_job(root, config_path)
    if _optional_string(job.get("job_class")) == "base_year":
        return _run_base_year_job(root, job)
    if _optional_string(job.get("job_class")) == "candidate_year":
        return _run_candidate_year_job(root, job, spec)
    if _optional_string(job.get("job_class")) == "reducer":
        return _run_reducer_job(root, job, spec)
    if job_id == "assemble_event_tables":
        return _assemble_event_tables(root, spec)
    if job_id == "assemble_surfaces":
        return _assemble_surfaces(root)
    if job_id == "assemble_publication_cards":
        return _assemble_publication_cards(root)
    if job_id == "assemble_report_and_workbook":
        return _assemble_report_and_workbook(
            root,
            config_path=config_path,
            step6_report=step6_report,
            workspace_report=workspace_report,
        )
    raise ValueError(f"Unsupported Step 7 job id {job_id!r}")


def _scheduler_pythonpath() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    src_path = repo_root / "src"
    existing = os.environ.get("PYTHONPATH", "")
    return f"{src_path}{os.pathsep}{existing}" if existing else str(src_path)


def _run_job_subprocess(
    *,
    root: Path,
    config_path: Path | None,
    job: dict[str, Any],
    max_active_jobs: int,
    execution_mode: str,
) -> tuple[dict[str, Any] | None, int, str, str]:
    command = [
        sys.executable,
        "-m",
        "heat_mnch.cli",
        "step-7-escalation-round",
        "--root-override",
        str(root),
        "--execution-mode",
        execution_mode,
        "--max-active-jobs",
        str(max_active_jobs),
        "--job-id",
        _optional_string(job.get("job_id")),
    ]
    if config_path is not None:
        command.extend(["--config-path", str(config_path)])
    nice_path = shutil.which("nice")
    if nice_path:
        command = [nice_path, "-n", "10", *command]
    env = os.environ.copy()
    env.update(STEP7_THREAD_ENV)
    env["PYTHONPATH"] = _scheduler_pythonpath()
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, env=env)
    payload: dict[str, Any] | None = None
    if result.stdout.strip():
        try:
            payload = json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            payload = None
    return payload, int(result.returncode), result.stdout, result.stderr


def _apply_pending_dependency_skips(manifest: dict[str, Any]) -> bool:
    changed = False
    job_map = _jobs_by_id(manifest)
    for job in manifest.get("jobs", []):
        if _optional_string(job.get("status")) != STEP7_JOB_STATUS_PENDING:
            continue
        if _optional_string(job.get("job_class")) != "candidate_year":
            continue
        if not _dependencies_are_final(job, job_map):
            continue
        dependency_statuses = {_optional_string(job_map[dependency].get("status")) for dependency in job.get("depends_on", [])}
        if dependency_statuses & {STEP7_JOB_STATUS_FAILED, STEP7_JOB_STATUS_SKIPPED}:
            job["status"] = STEP7_JOB_STATUS_SKIPPED
            job["finished_at"] = _now_utc()
            job["fallback_reason"] = "base_year_unavailable"
            changed = True
    return changed


def _next_runnable_job(manifest: dict[str, Any]) -> dict[str, Any] | None:
    job_map = _jobs_by_id(manifest)
    for job in manifest.get("jobs", []):
        if _optional_string(job.get("status")) != STEP7_JOB_STATUS_PENDING:
            continue
        if not _dependencies_are_final(job, job_map):
            continue
        return job
    return None


def _build_step7_response(
    *,
    root: Path,
    config_path: Path | None,
    step6_report: dict[str, Any] | None,
    directories: dict[str, str],
) -> dict[str, object]:
    spec = _step7_spec(root, config_path=config_path)
    workspace_report = _scaffold_step7_publication_workspace(root, spec)
    upstream_step6_report = _resolve_upstream_step6_report(root, config_path=config_path, step6_report=step6_report)
    _, relative_outputs = _step7_output_paths(root)
    summary, gates, _ = _compute_step7_summary_and_gates(root, step6_report=upstream_step6_report)
    status = "PASS" if all(gates.values()) else "FAIL"
    round_config = load_round_config(root, config_path=config_path)
    return {
        "status": status,
        "step": STEP7_NAME,
        "analysis_id": round_config["analysis_id"],
        "years": [int(year) for year in round_config.get("years", [])],
        "directories": directories,
        "summary": {key: value for key, value in summary.items() if key != "event_level_result_row_count"},
        "gates": gates,
        "outputs": relative_outputs,
        "workspace_report": workspace_report,
    }


def build_step7_escalation_round(
    root: Path,
    config_path: Path | None = None,
    step6_report: dict[str, Any] | None = None,
    *,
    resume: bool = True,
    max_active_jobs: int = 1,
    execution_mode: str = STEP7_EXECUTION_MODE,
    job_id: str | None = None,
) -> dict[str, object]:
    directories = ensure_step7_directory_structure(root)
    if execution_mode != STEP7_EXECUTION_MODE:
        raise ValueError(f"Unsupported Step 7 execution mode {execution_mode!r}")
    _ensure_step7_runtime_directories(root)
    if job_id:
        return _run_step7_job_worker(
            root=root,
            config_path=config_path,
            job_id=job_id,
            step6_report=step6_report,
        )

    round_config = load_round_config(root, config_path=config_path)
    spec = _step7_spec(root, config_path=config_path)
    route_catalog = _route_catalog_frame(root, config_path=config_path)
    runtime_paths = _step7_runtime_paths(root)
    manifest_path = runtime_paths["manifest_path"]

    if resume and manifest_path.exists():
        manifest = _reset_jobs_for_resume(_load_step7_runtime_manifest(root))
    else:
        manifest = _build_step7_runtime_manifest(
            root,
            round_config=round_config,
            route_catalog=route_catalog,
            spec=spec,
            max_active_jobs=max_active_jobs,
            execution_mode=execution_mode,
        )
    _write_step7_runtime_state(root, manifest)

    while True:
        manifest = _load_step7_runtime_manifest(root)
        if _apply_pending_dependency_skips(manifest):
            _write_step7_runtime_state(root, manifest)
            continue
        job = _next_runnable_job(manifest)
        if job is None:
            break
        job["status"] = STEP7_JOB_STATUS_RUNNING
        job["started_at"] = _now_utc()
        job["attempt_count"] = int(job.get("attempt_count", 0)) + 1
        _write_step7_runtime_state(root, manifest)
        payload, returncode, stdout_text, stderr_text = _run_job_subprocess(
            root=root,
            config_path=config_path,
            job=job,
            max_active_jobs=max_active_jobs,
            execution_mode=execution_mode,
        )
        _job_log_path(root, _optional_string(job.get("job_id"))).write_text(
            (stdout_text or "") + ("\n" if stdout_text and stderr_text else "") + (stderr_text or ""),
            encoding="utf-8",
        )
        manifest = _load_step7_runtime_manifest(root)
        job = _jobs_by_id(manifest)[_optional_string(job.get("job_id"))]
        job["finished_at"] = _now_utc()
        if payload is not None:
            job["output_relpath"] = _optional_string(payload.get("output_relpath"))
            job["fallback_reason"] = _optional_string(payload.get("fallback_reason"))
        if returncode == 0 and payload is not None and _optional_string(payload.get("job_status")) == STEP7_JOB_STATUS_COMPLETED:
            job["status"] = STEP7_JOB_STATUS_COMPLETED
            job["error_relpath"] = ""
        elif payload is not None and _optional_string(payload.get("job_status")) == STEP7_JOB_STATUS_SKIPPED:
            job["status"] = STEP7_JOB_STATUS_SKIPPED
            job["error_relpath"] = ""
        else:
            job["status"] = STEP7_JOB_STATUS_FAILED
            job["error_relpath"] = _job_error_relpath(root, _optional_string(job.get("job_id")))
            _job_error_path(root, _optional_string(job.get("job_id"))).write_text(
                stderr_text or stdout_text or json.dumps(payload or {}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        _write_step7_runtime_state(root, manifest)
        if _optional_string(job.get("job_id")) == "initialize_step7_runtime" and _optional_string(job.get("status")) == STEP7_JOB_STATUS_FAILED:
            break
        if _optional_string(job.get("phase")) in {"base_year", "candidate_year", "reducer"}:
            time.sleep(_step7_cooldown_seconds())

    return _build_step7_response(
        root=root,
        config_path=config_path,
        step6_report=step6_report,
        directories=directories,
    )
