from __future__ import annotations

import json
from pathlib import Path

from .config import load_project_config


def detect_storage_risks(root: Path, markers: list[str]) -> dict[str, object]:
    root_text = str(root)
    matched_markers = [marker for marker in markers if marker in root_text]
    documents_path = "Documents" in root.parts
    return {
        "documents_path": documents_path,
        "managed_storage_markers": matched_markers,
        "runtime_root_recommended": bool(matched_markers) or documents_path,
    }


def probe_path(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"path": str(path), "exists": path.exists(), "readable": False}
    if not path.exists():
        return result
    if path.is_dir():
        result["readable"] = True
        return result
    try:
        with path.open("rb") as handle:
            handle.read(1024)
        result["readable"] = True
    except OSError as exc:
        result["error"] = f"{exc.__class__.__name__}: {exc}"
    return result


def build_inventory(root: Path) -> dict[str, object]:
    config = load_project_config(root)
    layer_inventory: dict[str, object] = {}
    for layer, relative_path in config["data_layers"].items():
        layer_path = root / relative_path
        files = [path for path in layer_path.rglob("*") if path.is_file()] if layer_path.exists() else []
        layer_inventory[layer] = {
            "path": str(layer_path),
            "file_count": len(files),
        }
    manifests_dir = root / config["project"]["manifests_dir"]
    manifest_files = sorted(manifests_dir.glob("*.json")) if manifests_dir.exists() else []
    return {
        "layers": layer_inventory,
        "manifest_count": len(manifest_files),
        "manifests": [str(path.relative_to(root)) for path in manifest_files],
    }


def run_preflight(root: Path) -> dict[str, object]:
    config = load_project_config(root)
    project = config["project"]
    risks = detect_storage_risks(root, config["runtime_guardrails"]["managed_storage_markers"])
    probes = [probe_path(root / relative) for relative in config["critical_paths"]["contracts"]]
    probes.extend(probe_path(root / relative) for relative in config["critical_paths"]["rules"])
    probes.append(probe_path(root / project["manifests_dir"]))
    probes.append(probe_path(root / project["run_logs_dir"]))
    probes.append(probe_path(root / project["status_dir"]))
    payload = {
        "repository_root": str(root),
        "runtime_root": project["runtime_root"],
        "storage_risks": risks,
        "critical_probes": probes,
        "inventory": build_inventory(root),
    }
    output_path = root / project["status_dir"] / "preflight.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload
