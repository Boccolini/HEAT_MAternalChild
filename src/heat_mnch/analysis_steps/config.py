from __future__ import annotations

from pathlib import Path
from typing import Any

from heat_mnch.config import load_yaml


DEFAULT_ROUND_CONFIG_PATH = Path("config/analysis/round_config.yaml")
DEFAULT_OUTCOME_INVENTORY_PATH = Path("config/analysis/outcome_inventory.yaml")
DEFAULT_ANALYSIS_VARIABLE_ALIASES_PATH = Path("config/analysis/analysis_variable_aliases.yaml")
DEFAULT_SEASONALITY_CONTRACT_PATH = Path("config/analysis/seasonality_contract.yaml")
DEFAULT_ESCALATION_PROTOCOL_PATH = Path("config/analysis/escalation_protocol.yaml")
DEFAULT_STEP6_BURDEN_CONFIG_PATH = Path("config/analysis/step6_burden.yaml")
DEFAULT_STEP6_TRANSLATION_CATALOG_PATH = Path("config/analysis/step6_outcome_translation_catalog.yaml")
DEFAULT_STEP7_ESCALATION_CONFIG_PATH = Path("config/analysis/step7_escalation.yaml")
DEFAULT_STEP7_ROUTE_CATALOG_PATH = Path("config/analysis/step7_route_catalog.yaml")


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


def load_round_config(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    payload = _load_json_yaml(root, config_path or DEFAULT_ROUND_CONFIG_PATH)
    if "outcome_inventory_path" not in payload:
        return _load_json_yaml(root, DEFAULT_ROUND_CONFIG_PATH)
    return payload


def load_outcome_inventory_config(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    return _load_json_yaml(root, config_path or DEFAULT_OUTCOME_INVENTORY_PATH, expected_key="outcomes")


def load_analysis_variable_aliases(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    payload = _load_json_yaml(root, config_path or DEFAULT_ANALYSIS_VARIABLE_ALIASES_PATH)
    if "aliases" in payload:
        return payload
    source_path = payload.get("aliases_source_path")
    if not isinstance(source_path, str):
        raise ValueError("Analysis variable aliases config must define either 'aliases' or 'aliases_source_path'")
    return _load_json_yaml(root, Path(source_path), expected_key="aliases")


def load_seasonality_contract(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    return _load_json_yaml(root, config_path or DEFAULT_SEASONALITY_CONTRACT_PATH, expected_key="profiles")


def load_escalation_protocol(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    return _load_json_yaml(root, config_path or DEFAULT_ESCALATION_PROTOCOL_PATH, expected_key="flagship_protocol")


def load_step6_burden_config(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    return _load_json_yaml(root, config_path or DEFAULT_STEP6_BURDEN_CONFIG_PATH, expected_key="step6_spec")


def load_step6_translation_catalog(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    return _load_json_yaml(root, config_path or DEFAULT_STEP6_TRANSLATION_CATALOG_PATH, expected_key="analysis_units")


def load_step7_escalation_config(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    return _load_json_yaml(root, config_path or DEFAULT_STEP7_ESCALATION_CONFIG_PATH, expected_key="step7_spec")


def load_step7_route_catalog(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    return _load_json_yaml(root, config_path or DEFAULT_STEP7_ROUTE_CATALOG_PATH, expected_key="analysis_units")
