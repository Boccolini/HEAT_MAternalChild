from __future__ import annotations

from pathlib import Path
from typing import Any

from heat_mnch.config import load_yaml


DEFAULT_TRACK_A_CONFIG_PATH = Path("config/analysis/track_a.yaml")
DEFAULT_OUTCOME_MAP_PATH = Path("config/analysis/framework_outcome_map.yaml")
DEFAULT_VARIABLE_ALIASES_PATH = Path("config/analysis/track_a_variable_aliases.yaml")


def _resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _load_json_yaml(root: Path, path: Path, expected_key: str | None = None) -> dict[str, Any]:
    resolved = _resolve_path(root, path)
    payload = load_yaml(resolved)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping payload in {resolved}")
    if expected_key is not None and expected_key not in payload:
        raise ValueError(f"Missing expected key {expected_key!r} in {resolved}")
    return payload


def load_track_a_config(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    return _load_json_yaml(root, config_path or DEFAULT_TRACK_A_CONFIG_PATH)


def load_framework_outcome_map(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    return _load_json_yaml(root, config_path or DEFAULT_OUTCOME_MAP_PATH, expected_key="track_a_outcomes")


def load_track_a_variable_aliases(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    return _load_json_yaml(root, config_path or DEFAULT_VARIABLE_ALIASES_PATH, expected_key="aliases")
