from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import (
    load_framework_outcome_map,
    load_track_a_config,
    load_track_a_variable_aliases,
)
from .publication import scaffold_publication_workspace
from .registry import ensure_track_a_directory_structure, track_a_metadata_root


PUBLIC_VARIABLE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
DEFAULT_PROVENANCE_COLUMNS = {
    "canonical_dataset",
    "canonical_variable",
    "canonical_field_reference",
}


def _canonical_key(entry: dict[str, Any]) -> tuple[str, str]:
    return (str(entry["canonical_dataset"]), str(entry["canonical_variable"]))


def _public_signature(entry: dict[str, Any]) -> tuple[str, str, str, str, str, bool]:
    return (
        str(entry["description"]),
        str(entry["units"]),
        str(entry["public_role"]),
        str(entry["outcome_family"]),
        str(entry.get("exposure_family") or ""),
        bool(entry["provenance_only"]),
    )


def validate_track_a_aliases(entries: list[dict[str, Any]]) -> None:
    required_keys = {
        "canonical_dataset",
        "canonical_variable",
        "academic_variable_name",
        "academic_label",
        "description",
        "units",
        "public_role",
        "outcome_family",
        "provenance_only",
    }
    errors: list[str] = []
    seen_canonical: set[tuple[str, str]] = set()
    seen_public: dict[str, tuple[str, str, str, str, str, bool]] = {}

    for index, entry in enumerate(entries, start=1):
        missing = sorted(required_keys - set(entry))
        if missing:
            errors.append(f"Alias entry {index} is missing required keys: {', '.join(missing)}")
            continue

        canonical = _canonical_key(entry)
        if canonical in seen_canonical:
            errors.append(
                f"Duplicate canonical mapping for {canonical[0]}.{canonical[1]} in alias entry {index}"
            )
        seen_canonical.add(canonical)

        academic_variable_name = str(entry["academic_variable_name"])
        if not PUBLIC_VARIABLE_PATTERN.fullmatch(academic_variable_name):
            errors.append(
                f"Academic variable name {academic_variable_name!r} does not match lower_snake_case requirements"
            )

        for text_key in ("academic_label", "description", "units"):
            value = str(entry[text_key])
            if not value:
                errors.append(f"Alias entry {index} has an empty {text_key}")
            elif not value.isascii():
                errors.append(
                    f"Alias entry {index} contains non-ASCII text in {text_key}; Track A outputs must be in academic English"
                )

        signature = _public_signature(entry)
        if academic_variable_name in seen_public and seen_public[academic_variable_name] != signature:
            errors.append(
                f"Academic variable name {academic_variable_name!r} has inconsistent metadata across alias entries"
            )
        seen_public[academic_variable_name] = signature

        if not isinstance(entry["provenance_only"], bool):
            errors.append(f"Alias entry {index} must use a boolean provenance_only flag")

    if errors:
        raise ValueError("Track A alias validation failed:\n- " + "\n- ".join(errors))


def validate_public_output_columns(
    columns: Iterable[str],
    provenance_columns: set[str] | None = None,
) -> list[str]:
    allowed_provenance = provenance_columns or DEFAULT_PROVENANCE_COLUMNS
    invalid_columns: list[str] = []
    for column in columns:
        if column in allowed_provenance:
            continue
        if not PUBLIC_VARIABLE_PATTERN.fullmatch(column):
            invalid_columns.append(column)
    return invalid_columns


def _validate_outcome_map_against_aliases(
    outcome_map: dict[str, Any],
    alias_entries: list[dict[str, Any]],
) -> None:
    public_variables = {str(entry["academic_variable_name"]) for entry in alias_entries}
    errors: list[str] = []
    for outcome_entry in outcome_map["track_a_outcomes"]:
        public_name = str(outcome_entry["academic_variable_name"])
        if public_name not in public_variables:
            errors.append(
                f"Outcome map references academic variable {public_name!r} without a matching alias entry"
            )
    if errors:
        raise ValueError("Track A outcome map validation failed:\n- " + "\n- ".join(errors))


def _build_crosswalk_frame(alias_entries: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for entry in alias_entries:
        rows.append(
            {
                "canonical_dataset": entry["canonical_dataset"],
                "canonical_variable": entry["canonical_variable"],
                "academic_variable_name": entry["academic_variable_name"],
                "academic_label": entry["academic_label"],
                "description": entry["description"],
                "units": entry["units"],
                "public_role": entry["public_role"],
                "outcome_family": entry["outcome_family"],
                "exposure_family": entry.get("exposure_family") or "",
                "provenance_only": entry["provenance_only"],
                "canonical_field_reference": f"{entry['canonical_dataset']}.{entry['canonical_variable']}",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["canonical_dataset", "canonical_variable"], ignore_index=True
    )


def _build_dictionary_frame(crosswalk: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "academic_variable_name",
        "academic_label",
        "description",
        "units",
        "public_role",
        "outcome_family",
        "exposure_family",
        "provenance_only",
    ]
    dictionary = crosswalk.loc[:, columns].drop_duplicates().sort_values(
        ["public_role", "outcome_family", "academic_variable_name"], ignore_index=True
    )
    invalid_columns = validate_public_output_columns(dictionary["academic_variable_name"].tolist(), set())
    if invalid_columns:
        raise ValueError(
            "Track A variable dictionary contains invalid public variable names: "
            + ", ".join(invalid_columns)
        )
    return dictionary


def _render_semantic_layer_report(
    config: dict[str, Any],
    alias_entries: list[dict[str, Any]],
    workspace_report: dict[str, object],
    outputs: dict[str, str],
) -> str:
    publication_created = len(workspace_report["created_files"])
    publication_existing = len(workspace_report["existing_files"])
    years = ", ".join(str(year) for year in config["years"])
    outcome_families = ", ".join(config["outcome_families"])
    exposure_families = ", ".join(family["id"] for family in config["exposure_families"])
    return (
        "# Track A Semantic Layer Report\n\n"
        f"- Analysis title: `{config['analysis_title']}`\n"
        f"- Analysis years: `{years}`\n"
        f"- Outcome families: `{outcome_families}`\n"
        f"- Exposure families: `{exposure_families}`\n"
        f"- Alias entries validated: `{len(alias_entries)}`\n"
        "- Naming policy: `academic English`, `lower_snake_case`, "
        "`canonical non-English names allowed only in provenance metadata`\n"
        f"- Variable dictionary: `{outputs['track_a_variable_dictionary']}`\n"
        f"- Canonical crosswalk: `{outputs['canonical_to_academic_variable_map']}`\n"
        f"- Publication workspace: `{workspace_report['workspace_root']}`\n"
        f"- Publication files created now: `{publication_created}`\n"
        f"- Publication files already present: `{publication_existing}`\n\n"
        "## Decision\n\n"
        "Stage 0 established the Track A scaffold and academic-English semantic layer "
        "without modifying frozen canonical gold tables or canonical manifest packages.\n"
    )


def build_track_a_semantic_layer(root: Path, config_path: Path | None = None) -> dict[str, object]:
    directories = ensure_track_a_directory_structure(root)
    config = load_track_a_config(root, config_path=config_path)
    outcome_map = load_framework_outcome_map(root)
    alias_payload = load_track_a_variable_aliases(root)
    alias_entries = list(alias_payload["aliases"])

    validate_track_a_aliases(alias_entries)
    _validate_outcome_map_against_aliases(outcome_map, alias_entries)

    crosswalk = _build_crosswalk_frame(alias_entries)
    dictionary = _build_dictionary_frame(crosswalk)
    workspace_report = scaffold_publication_workspace(root)

    metadata_root = track_a_metadata_root(root)
    metadata_root.mkdir(parents=True, exist_ok=True)
    crosswalk_path = metadata_root / "canonical_to_academic_variable_map.csv"
    dictionary_path = metadata_root / "track_a_variable_dictionary.csv"
    report_path = metadata_root / "semantic_layer_report.md"

    crosswalk.to_csv(crosswalk_path, index=False)
    dictionary.to_csv(dictionary_path, index=False)

    outputs = {
        "track_a_variable_dictionary": str(dictionary_path.relative_to(root)),
        "canonical_to_academic_variable_map": str(crosswalk_path.relative_to(root)),
        "semantic_layer_report": str(report_path.relative_to(root)),
    }
    report_path.write_text(
        _render_semantic_layer_report(config, alias_entries, workspace_report, outputs),
        encoding="utf-8",
    )

    return {
        "status": "PASS",
        "stage": "stage0_scaffold_and_semantic_layer",
        "analysis_id": config["analysis_id"],
        "analysis_title": config["analysis_title"],
        "years": config["years"],
        "directories": directories,
        "metadata_outputs": outputs,
        "publication_workspace": workspace_report,
        "alias_count": len(alias_entries),
        "public_variable_count": int(dictionary.shape[0]),
        "naming_policy": config["naming_policy"],
        "validation": {
            "public_variable_regex": PUBLIC_VARIABLE_PATTERN.pattern,
            "public_descriptions_ascii_only": True,
            "outcome_map_resolved": True,
        },
    }
