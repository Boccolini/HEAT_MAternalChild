from __future__ import annotations

import re
from typing import Any, Iterable


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


def validate_analysis_variable_aliases(entries: list[dict[str, Any]]) -> None:
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
                    f"Alias entry {index} contains non-ASCII text in {text_key}; official outputs must stay in academic English"
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
        raise ValueError("Analysis alias validation failed:\n- " + "\n- ".join(errors))


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
