from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd


SCREENING_CACHE_MANIFEST_FILENAME = "cache_manifest.json"
SCREENING_STAGE_VERSION = "stage4.1.0"


def screening_scope_key(years: list[int]) -> str:
    return "years_" + "_".join(str(year) for year in years)


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
    }


def load_cache_manifest(cache_root: Path) -> dict[str, Any]:
    manifest_path = cache_root / SCREENING_CACHE_MANIFEST_FILENAME
    if not manifest_path.exists():
        return {"entries": {}}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_cache_manifest(cache_root: Path, manifest: dict[str, Any]) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / SCREENING_CACHE_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


@dataclass
class ScreeningCacheManager:
    cache_root: Path
    scope_key: str
    stage_version: str = SCREENING_STAGE_VERSION
    manifest: dict[str, Any] = field(init=False)
    cache_hit_count: int = field(default=0, init=False)
    cache_miss_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.scope_root.mkdir(parents=True, exist_ok=True)
        self.manifest = load_cache_manifest(self.cache_root)

    @property
    def scope_root(self) -> Path:
        return self.cache_root / self.scope_key

    @property
    def manifest_path(self) -> Path:
        return self.cache_root / SCREENING_CACHE_MANIFEST_FILENAME

    def _entry_key(self, entry_name: str) -> str:
        return f"{self.scope_key}:{entry_name}"

    def _entry_path(self, entry_name: str, suffix: str) -> Path:
        return self.scope_root / f"{entry_name}{suffix}"

    def _entry_metadata(self, input_paths: list[Path], artifact_path: Path) -> dict[str, Any]:
        return {
            "scope_key": self.scope_key,
            "stage_version": self.stage_version,
            "artifact_path": str(artifact_path),
            "input_signatures": [file_signature(path) for path in input_paths],
        }

    def _is_valid(self, entry_name: str, artifact_path: Path, expected_metadata: dict[str, Any]) -> bool:
        if not artifact_path.exists():
            return False
        entry = self.manifest.get("entries", {}).get(self._entry_key(entry_name))
        return entry == expected_metadata

    def save_manifest(self) -> Path:
        return save_cache_manifest(self.cache_root, self.manifest)

    def load_or_build_parquet(
        self,
        entry_name: str,
        input_paths: list[Path],
        builder: Callable[[], pd.DataFrame],
    ) -> tuple[pd.DataFrame, bool, Path]:
        artifact_path = self._entry_path(entry_name, ".parquet")
        expected_metadata = self._entry_metadata(input_paths, artifact_path)
        if self._is_valid(entry_name, artifact_path, expected_metadata):
            self.cache_hit_count += 1
            return pd.read_parquet(artifact_path), True, artifact_path

        frame = builder()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(artifact_path, index=False)
        self.manifest.setdefault("entries", {})[self._entry_key(entry_name)] = expected_metadata
        self.save_manifest()
        self.cache_miss_count += 1
        return frame, False, artifact_path

    def write_scope_json(self, filename: str, payload: dict[str, Any]) -> Path:
        output_path = self.scope_root / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return output_path

    def read_scope_json(self, scope_key: str, filename: str) -> dict[str, Any] | None:
        path = self.cache_root / scope_key / filename
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def classify_run_cache_state(self) -> str:
        if self.cache_hit_count == 0 and self.cache_miss_count > 0:
            return "cold"
        if self.cache_hit_count > 0 and self.cache_miss_count == 0:
            return "warm"
        if self.cache_hit_count > 0 and self.cache_miss_count > 0:
            return "partial_rebuild"
        return "no_cache_activity"
