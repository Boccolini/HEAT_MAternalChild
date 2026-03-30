from __future__ import annotations

from pathlib import Path
from typing import Any

from heat_mnch.config import load_yaml


DEFAULT_TRACK_B_CONFIG_PATH = Path("config/analysis/track_b.yaml")
DEFAULT_EDITORIAL_FREEZE_PATH = Path("config/analysis/track_b_editorial_freeze.yaml")
DEFAULT_WINDOW_SPEC_PATH = Path("config/analysis/track_b_window_spec.yaml")
DEFAULT_BURDEN_CONFIG_PATH = Path("config/analysis/track_b_burden.yaml")


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


def load_track_b_config(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    return _load_json_yaml(root, config_path or DEFAULT_TRACK_B_CONFIG_PATH)


def load_track_b_editorial_freeze_config(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    return _load_json_yaml(root, config_path or DEFAULT_EDITORIAL_FREEZE_PATH, expected_key="flagship_protocol")


def load_track_b_window_spec(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    return _load_json_yaml(root, config_path or DEFAULT_WINDOW_SPEC_PATH, expected_key="window_spec")


def load_track_b_burden_config(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    return _load_json_yaml(root, config_path or DEFAULT_BURDEN_CONFIG_PATH, expected_key="burden_spec")
