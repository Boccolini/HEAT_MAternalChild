from __future__ import annotations

from pathlib import Path
from typing import Any

from heat_mnch.config import load_yaml


DEFAULT_LRH_AMERICAS_ROUND2_CONFIG_PATH = Path("config/analysis/lrh_americas_round2_validation.yaml")


def _resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_lrh_americas_round2_validation_config(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    resolved = _resolve_path(root, config_path or DEFAULT_LRH_AMERICAS_ROUND2_CONFIG_PATH)
    payload = load_yaml(resolved)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping payload in {resolved}")
    if "round2_validation_spec" not in payload:
        raise ValueError(f"Missing 'round2_validation_spec' in {resolved}")
    spec = payload["round2_validation_spec"]
    if not isinstance(spec, dict):
        raise ValueError(f"'round2_validation_spec' must be a mapping in {resolved}")
    return spec
