from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_project_config(root: Path | None = None) -> dict[str, Any]:
    base = root or repository_root()
    config_path = base / "config" / "project.yaml"
    payload = load_yaml(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid config payload: {config_path}")
    return payload
