from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from heat_mnch.statcalc_api.registry import StatCalcRegistry

from .aliases import validate_analysis_variable_aliases
from .config import (
    load_analysis_variable_aliases,
    load_escalation_protocol,
    load_outcome_inventory_config,
    load_round_config,
    load_seasonality_contract,
)
from .registry import analysis_step0_root, ensure_step0_directory_structure, incremental_publication_root


RELEASE_REPORT_LINE_PATTERN = re.compile(r"^- (?P<label>[^:]+): `(?P<value>.*)`$")
PASS_LIKE_VALUES = {"PASS", "pass", "yes", "True", True}
PROMOTED_ROLES = {"primary", "sensitivity"}
ROUND_PUBLICATION_FILES = {
    "README.md": """# HEAT-MNCH Incremental Analysis Round

This workspace stores the unified round-level documentation for the incremental `steps` pipeline.

Step 0 freezes the outcome roster, denominator rules, seasonality contract, and supersession map before any modeling starts.
""",
    "00_round_charter.md": "# Round Charter\n\nPopulate manuscript-facing governance notes for the current analytical round here.\n",
    "01_methods_freeze.md": "# Methods Freeze\n\nDocument the locked denominator and seasonality contracts here.\n",
}
DENOMINATOR_SPECS = {
    "annual_live_births": {
        "offset_id": "log_annual_live_birth_offset",
        "zero_handling": "never_log_zero",
        "scope": "annual",
    },
    "under_five_population": {
        "offset_id": "log_under_five_population_offset",
        "zero_handling": "never_log_zero",
        "scope": "annual",
    },
    "daily_live_births": {
        "offset_id": "daily_live_birth_count_offset_proxy",
        "zero_handling": "never_log_zero",
        "scope": "daily",
    },
    "annual_total_births_primary": {
        "offset_id": "log_annual_total_birth_count_primary_offset",
        "zero_handling": "never_log_zero",
        "scope": "annual",
    },
    "annual_total_births_sensitivity_gte22": {
        "offset_id": "log_annual_total_birth_count_sensitivity_gte22_offset",
        "zero_handling": "never_log_zero",
        "scope": "annual",
    },
    "total_births_primary": {
        "offset_id": "log_total_birth_count_primary_offset",
        "zero_handling": "never_log_zero",
        "scope": "daily",
    },
    "total_births_sensitivity_gte22": {
        "offset_id": "log_total_birth_count_sensitivity_gte22_offset",
        "zero_handling": "never_log_zero",
        "scope": "daily",
    },
    "women_15_49_population": {
        "offset_id": "log_women_age_15_49_population_offset",
        "zero_handling": "never_log_zero",
        "scope": "annual",
    },
    "women_15_45_population": {
        "offset_id": "log_women_age_15_45_sensitivity_population_offset",
        "zero_handling": "never_log_zero",
        "scope": "annual",
    },
    "not_applicable_count_or_descriptor": {
        "offset_id": "no_offset",
        "zero_handling": "not_applicable",
        "scope": "descriptor_only",
    },
}
LOCKED_OUTCOME_DENOMINATOR_RULES = {
    "ACUTE_UTIL": "under_five_population",
    "HOSP_U5": "under_five_population",
    "DEATHS_U1": "annual_live_births",
    "DEATHS_NEONATAL": "annual_live_births",
    "DEATHS_NEONATAL__event_level": "annual_live_births",
    "DEATHS_28_364D": "annual_live_births",
    "DEATHS_1_4Y": "under_five_population",
    "DEATHS_UNDER5_TOTAL": "under_five_population",
    "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL": "under_five_population",
    "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL": "under_five_population",
    "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL": "under_five_population",
    "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL": "under_five_population",
    "CHILD_DIRECT_HEAT_HOSPITALIZATION_TOTAL": "under_five_population",
    "CHILD_INDIRECT_HEAT_HOSPITALIZATION_TOTAL": "under_five_population",
    "CHILD_EXCESSIVE_NATURAL_HEAT_X30_HOSPITALIZATION_TOTAL": "under_five_population",
    "PRENATAL_VISITS_8PLUS_FLAG": "annual_live_births",
    "PRETERM_FLAG": "annual_live_births",
    "PRETERM_FLAG__event_level": "annual_live_births",
    "VERY_PRETERM_FLAG": "annual_live_births",
    "VERY_PRETERM_FLAG__event_level": "annual_live_births",
    "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL": "under_five_population",
    "UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_TOTAL": "under_five_population",
    "ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_TOTAL": "under_five_population",
    "ANY_MENTION_DIRECT_HEAT_ANY_TOTAL": "under_five_population",
    "ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL": "under_five_population",
    "STILLBIRTH_PRIMARY_GTE28_FLAG": "annual_total_births_primary",
    "STILLBIRTH_PRIMARY_GTE28_FLAG__event_level": "annual_total_births_primary",
    "STILLBIRTH_SENS_GTE22_FLAG": "annual_total_births_sensitivity_gte22",
    "STILLBIRTH_SENS_GTE22_FLAG__event_level": "annual_total_births_sensitivity_gte22",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _resolve_first_existing(root: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")


def _parse_release_report(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = RELEASE_REPORT_LINE_PATTERN.match(line.strip())
        if match:
            parsed[match.group("label")] = match.group("value")
    return parsed


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _statcalc_recipe_ids(root: Path) -> list[str]:
    registry = StatCalcRegistry(root)
    return sorted(recipe.recipe_id for recipe in registry.recipes)


def _scaffold_publication_workspace(root: Path) -> dict[str, object]:
    workspace_root = incremental_publication_root(root)
    created_files: list[str] = []
    existing_files: list[str] = []
    workspace_root.mkdir(parents=True, exist_ok=True)
    for relative_path, contents in ROUND_PUBLICATION_FILES.items():
        target = workspace_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing_files.append(_relative(root, target))
            continue
        target.write_text(contents, encoding="utf-8")
        created_files.append(_relative(root, target))
    return {
        "workspace_root": _relative(root, workspace_root),
        "created_files": created_files,
        "existing_files": existing_files,
    }


def _load_inputs(root: Path, round_config: dict[str, Any]) -> dict[str, Any]:
    frozen_inputs = round_config["frozen_inputs"]
    release_report_path = _resolve_first_existing(
        root,
        [
            str(frozen_inputs["gold_release_report_path"]),
            str(frozen_inputs.get("gold_release_report_fallback_path", "")),
        ],
    )
    endpoint_ledger_path = _resolve_first_existing(
        root,
        [
            str(frozen_inputs["endpoint_readiness_ledger_path"]),
            "artifacts/status/endpoint_readiness_ledger.csv",
        ],
    )
    stage5_matrix_path = _resolve_first_existing(
        root,
        [
            str(frozen_inputs["stage5_dictionary_matrix_path"]),
            "artifacts/status/gold_stage5_dictionary_matrix.csv",
        ],
    )
    return {
        "release_report_path": release_report_path,
        "release_report": _parse_release_report(release_report_path),
        "endpoint_ledger_path": endpoint_ledger_path,
        "endpoint_ledger": pd.read_csv(endpoint_ledger_path),
        "stage5_matrix_path": stage5_matrix_path,
        "stage5_matrix": pd.read_csv(stage5_matrix_path),
    }


def _validated_stage5_gate(stage5_matrix: pd.DataFrame) -> bool:
    required_documented = stage5_matrix["required_documented_yes_no"].astype(str).str.lower().eq("yes").all()
    schema_consistent = True
    if "schema_consistent_with_dictionary_yes_no" in stage5_matrix.columns:
        schema_consistent = stage5_matrix["schema_consistent_with_dictionary_yes_no"].astype(str).str.lower().eq("yes").all()
    statuses_pass = stage5_matrix["status"].astype(str).str.upper().eq("PASS").all()
    return bool(required_documented and schema_consistent and statuses_pass)


def _build_alias_lookup(alias_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    alias_entries = list(alias_payload["aliases"])
    validate_analysis_variable_aliases(alias_entries)
    return {str(entry["academic_variable_name"]): entry for entry in alias_entries}


def _build_seasonality_lookup(seasonality_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for profile in seasonality_payload["profiles"]:
        lookup[str(profile["seasonality_profile_id"])] = dict(profile)
    return lookup


def _build_readiness_lookup(endpoint_ledger: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(row["endpoint_name"]): row.to_dict()
        for _, row in endpoint_ledger.fillna("").iterrows()
    }


def _validate_round_path(role: str, round_path: str) -> bool:
    allowed = {
        "primary": {"immediate_main_round"},
        "sensitivity": {"supplementary_sensitivity_round"},
        "descriptive_only": {"descriptive_only_round"},
        "deferred_event_level": {"event_level_escalation_round"},
    }
    return round_path in allowed.get(role, set())


def _build_inventory_frame(
    outcome_inventory_payload: dict[str, Any],
    alias_lookup: dict[str, dict[str, Any]],
    readiness_lookup: dict[str, dict[str, Any]],
    seasonality_lookup: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, bool]]:
    rows: list[dict[str, Any]] = []
    promoted_readiness_ok = True
    promoted_denominator_ok = True
    seasonality_complete = True
    alias_resolution_ok = True
    round_path_ok = True

    for entry in outcome_inventory_payload["outcomes"]:
        row = dict(entry)
        role = str(row["analysis_role"])
        outcome_id = str(row["outcome_id"])
        alias = alias_lookup.get(str(row["academic_variable_name"]))
        alias_resolution_ok = alias_resolution_ok and alias is not None

        readiness_endpoint = str(row.get("readiness_endpoint_name") or "")
        readiness = readiness_lookup.get(readiness_endpoint) if readiness_endpoint else None
        if role in PROMOTED_ROLES:
            promoted_readiness_ok = promoted_readiness_ok and readiness is not None
            promoted_denominator_ok = promoted_denominator_ok and readiness is not None and str(readiness.get("denominator_valid_yes_no", "")).lower() == "yes"

        seasonality_profile = str(row["seasonality_profile_id"])
        seasonality = seasonality_lookup.get(seasonality_profile)
        seasonality_complete = seasonality_complete and seasonality is not None
        round_path_ok = round_path_ok and _validate_round_path(role, str(row["round_path"]))

        row["alias_public_role"] = str(alias["public_role"]) if alias else ""
        row["alias_provenance_only"] = bool(alias["provenance_only"]) if alias else False
        row["validated_from_release"] = bool(
            readiness is not None
            and str(readiness.get("required_fields_present_yes_no", "")).lower() == "yes"
            and str(readiness.get("denominator_valid_yes_no", "")).lower() == "yes"
        )
        row["primary_or_sensitivity_release"] = str(readiness.get("primary_or_sensitivity", "")) if readiness else ""
        row["release_denominator_rule"] = str(readiness.get("denominator_rule", "")) if readiness else ""
        row["seasonality_main_controls"] = ";".join(str(item) for item in seasonality.get("main_controls", [])) if seasonality else ""
        row["seasonality_robustness_controls"] = ";".join(str(item) for item in seasonality.get("robustness_controls", [])) if seasonality else ""
        row["seasonality_notes"] = str(seasonality.get("notes", "")) if seasonality else ""
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["round_path", "analysis_role", "outcome_id"], ignore_index=True)
    return frame, {
        "promoted_readiness_ok": promoted_readiness_ok,
        "promoted_denominator_ok": promoted_denominator_ok,
        "seasonality_complete": seasonality_complete,
        "alias_resolution_ok": alias_resolution_ok,
        "round_path_ok": round_path_ok,
    }


def _build_denominator_contract(inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in inventory.to_dict(orient="records"):
        denominator_id = str(record["denominator_id"])
        spec = DENOMINATOR_SPECS[denominator_id]
        rows.append(
            {
                "outcome_id": record["outcome_id"],
                "denominator_id": denominator_id,
                "offset_id": spec["offset_id"],
                "zero_handling": spec["zero_handling"],
                "scope": spec["scope"],
                "source_table": record["canonical_dataset"],
                "validated_from_release": bool(record["validated_from_release"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["scope", "denominator_id", "outcome_id"], ignore_index=True)


def _validate_locked_denominator_rules(inventory: pd.DataFrame) -> tuple[bool, list[dict[str, str]]]:
    mismatches: list[dict[str, str]] = []
    inventory_lookup = inventory.set_index("outcome_id")["denominator_id"].astype(str).to_dict()
    for outcome_id, expected_denominator_id in LOCKED_OUTCOME_DENOMINATOR_RULES.items():
        if outcome_id not in inventory_lookup:
            continue
        observed_denominator_id = inventory_lookup.get(outcome_id, "")
        if observed_denominator_id != expected_denominator_id:
            mismatches.append(
                {
                    "outcome_id": outcome_id,
                    "expected_denominator_id": expected_denominator_id,
                    "observed_denominator_id": observed_denominator_id,
                }
            )
    return len(mismatches) == 0, mismatches


def _build_seasonality_contract_frame(inventory: pd.DataFrame, seasonality_lookup: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for profile_id, profile in seasonality_lookup.items():
        covered = inventory.loc[inventory["seasonality_profile_id"].astype(str) == profile_id, "outcome_id"].astype(str).tolist()
        rows.append(
            {
                "seasonality_profile_id": profile_id,
                "outcome_family": profile["outcome_family"],
                "outcome_id_optional": str(profile.get("outcome_id_optional") or ""),
                "time_anchor": profile["time_anchor"],
                "main_controls": ";".join(str(item) for item in profile.get("main_controls", [])),
                "robustness_controls": ";".join(str(item) for item in profile.get("robustness_controls", [])),
                "required_from_step": profile["required_from_step"],
                "notes": profile.get("notes", ""),
                "covered_outcomes": ";".join(covered),
            }
        )
    return pd.DataFrame(rows).sort_values(["outcome_family", "seasonality_profile_id"], ignore_index=True)


def _build_supersession_map(round_config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(round_config["supersession_entries"])
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "legacy_surface",
                "legacy_id",
                "replacement_surface",
                "replacement_id",
                "compatibility_mode",
                "sunset_after_step",
            ]
        )
    return frame.sort_values(["legacy_surface", "legacy_id"], ignore_index=True)


def _inventory_summary(inventory: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, Any]:
    role_counts = inventory["analysis_role"].astype(str).value_counts().to_dict()
    round_counts = inventory["round_path"].astype(str).value_counts().to_dict()
    return {
        "total_outcomes": int(len(inventory)),
        "primary_count": int(role_counts.get("primary", 0)),
        "sensitivity_count": int(role_counts.get("sensitivity", 0)),
        "descriptive_only_count": int(role_counts.get("descriptive_only", 0)),
        "deferred_event_level_count": int(role_counts.get("deferred_event_level", 0)),
        "round_path_counts": round_counts,
        "co_primary_escalation_count": int(len(protocol["flagship_protocol"]["co_primary_outcomes"])),
        "supporting_escalation_count": int(len(protocol["flagship_protocol"]["supporting_secondary_outcomes"])),
    }


def _build_round_charter(
    round_config: dict[str, Any],
    release_report: dict[str, str],
    inventory_summary: dict[str, Any],
    inventory: pd.DataFrame,
    supersession_map: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "analysis_id": round_config["analysis_id"],
        "analysis_title": round_config["analysis_title"],
        "strategy": round_config["strategy"],
        "years": list(round_config["years"]),
        "generated_at": _now_utc(),
        "release_status": release_report.get("Overall status", "UNKNOWN"),
        "ready_now_endpoints": release_report.get("Ready now endpoints", "0"),
        "inventory_summary": inventory_summary,
        "immediate_outcomes": inventory.loc[inventory["analysis_role"].astype(str).eq("primary"), "outcome_id"].astype(str).tolist(),
        "supplementary_outcomes": inventory.loc[inventory["analysis_role"].astype(str).eq("sensitivity"), "outcome_id"].astype(str).tolist(),
        "descriptive_only_outcomes": inventory.loc[inventory["analysis_role"].astype(str).eq("descriptive_only"), "outcome_id"].astype(str).tolist(),
        "deferred_event_level_outcomes": inventory.loc[inventory["analysis_role"].astype(str).eq("deferred_event_level"), "outcome_id"].astype(str).tolist(),
        "supersession_count": int(len(supersession_map)),
        "locked_denominator_rules": LOCKED_OUTCOME_DENOMINATOR_RULES,
        "protected_contracts": round_config["protected_contracts"],
    }


def _render_step0_report(
    round_config: dict[str, Any],
    release_report: dict[str, str],
    input_paths: dict[str, Any],
    gates: dict[str, bool],
    inventory_summary: dict[str, Any],
    outputs: dict[str, str],
    workspace_report: dict[str, object],
    compatibility_payload: dict[str, Any],
) -> str:
    lines = [
        "# Step 0 Freeze Report",
        "",
        f"- Generated at: `{_now_utc()}`",
        f"- Analysis ID: `{round_config['analysis_id']}`",
        f"- Analysis title: `{round_config['analysis_title']}`",
        f"- Years: `{', '.join(str(year) for year in round_config['years'])}`",
        f"- Strategy: `{round_config['strategy']}`",
        f"- Gold release report: `{input_paths['release_report_path']}`",
    ]
    lines.extend(
        [
            f"- Endpoint readiness ledger: `{input_paths['endpoint_ledger_path']}`",
            f"- Stage 5 matrix: `{input_paths['stage5_matrix_path']}`",
            f"- Overall release status: `{release_report.get('Overall status', 'UNKNOWN')}`",
            f"- Primary outcomes: `{inventory_summary['primary_count']}`",
            f"- Sensitivity outcomes: `{inventory_summary['sensitivity_count']}`",
            f"- Descriptive only outcomes: `{inventory_summary['descriptive_only_count']}`",
            f"- Deferred event-level outcomes: `{inventory_summary['deferred_event_level_count']}`",
            "",
            "## Gates",
            "",
        ]
    )
    for gate_name, passed in gates.items():
        lines.append(f"- {gate_name}: `{'PASS' if passed else 'FAIL'}`")
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
            f"- Files created: `{len(workspace_report['created_files'])}`",
            f"- Files already present: `{len(workspace_report['existing_files'])}`",
        ]
    )
    if compatibility_payload:
        lines.extend(["", "## Legacy Compatibility", ""])
        lines.append(f"- Alias used: `{compatibility_payload['legacy_alias_used']}`")
        lines.append(f"- Replacement command: `{compatibility_payload['replacement_command']}`")
    return "\n".join(lines) + "\n"


def _write_step0_publication_notes(
    root: Path,
    *,
    round_config: dict[str, Any],
    gates: dict[str, bool],
    inventory_summary: dict[str, Any],
    outputs: dict[str, str],
) -> None:
    workspace_root = incremental_publication_root(root)
    (workspace_root / "README.md").write_text(
        (
            "# HEAT-MNCH Incremental Analysis Round\n\n"
            "This workspace stores the manuscript-facing notes for the official steps-only analytical pipeline.\n\n"
            f"- Analysis ID: `{round_config['analysis_id']}`\n"
            f"- Coverage years: `{', '.join(str(year) for year in round_config['years'])}`\n"
            f"- Step 0 report: `{outputs['step0_report']}`\n"
        ),
        encoding="utf-8",
    )
    (workspace_root / "00_round_charter.md").write_text(
        (
            "# Round Charter\n\n"
            f"- Analysis title: `{round_config['analysis_title']}`\n"
            f"- Primary outcomes frozen: `{inventory_summary['primary_count']}`\n"
            f"- Sensitivity outcomes frozen: `{inventory_summary['sensitivity_count']}`\n"
            f"- Round charter JSON: `{outputs['round_charter']}`\n"
            f"- Freeze status: `{'PASS' if all(gates.values()) else 'FAIL'}`\n"
        ),
        encoding="utf-8",
    )
    (workspace_root / "01_methods_freeze.md").write_text(
        (
            "# Methods Freeze\n\n"
            f"- Denominator contract: `{outputs['denominator_contract']}`\n"
            f"- Seasonality contract: `{outputs['seasonality_contract']}`\n"
            f"- Supersession map: `{outputs['supersession_map']}`\n"
            f"- Dashboard contract unchanged: `{'PASS' if gates['dashboard_contract_unchanged'] else 'FAIL'}`\n"
            f"- StatCalc recipe ids unchanged: `{'PASS' if gates['statcalc_recipe_ids_unchanged'] else 'FAIL'}`\n"
        ),
        encoding="utf-8",
    )


def _materialize_legacy_alias(root: Path, legacy_alias: str | None) -> dict[str, Any]:
    if legacy_alias is None:
        return {}
    raise ValueError(
        f"Legacy pipeline alias {legacy_alias!r} is disabled. Use the official step-* or steps-run commands only."
    )


def build_step0_freeze(
    root: Path,
    config_path: Path | None = None,
    legacy_alias: str | None = None,
) -> dict[str, object]:
    directories = ensure_step0_directory_structure(root)
    round_config = load_round_config(root, config_path=config_path)
    outcome_inventory_payload = load_outcome_inventory_config(root, Path(round_config["outcome_inventory_path"]))
    alias_payload = load_analysis_variable_aliases(root, Path(round_config["analysis_variable_aliases_path"]))
    seasonality_payload = load_seasonality_contract(root, Path(round_config["seasonality_contract_path"]))
    escalation_protocol = load_escalation_protocol(root, Path(round_config["escalation_protocol_path"]))
    input_paths = _load_inputs(root, round_config)
    workspace_report = _scaffold_publication_workspace(root)

    dashboard_manifest = root / str(round_config["protected_contracts"]["dashboard_manifest_path"])
    dashboard_hash_before = _file_hash(dashboard_manifest)
    statcalc_recipe_ids_before = _statcalc_recipe_ids(root)

    alias_lookup = _build_alias_lookup(alias_payload)
    seasonality_lookup = _build_seasonality_lookup(seasonality_payload)
    readiness_lookup = _build_readiness_lookup(input_paths["endpoint_ledger"])
    inventory, inventory_flags = _build_inventory_frame(
        outcome_inventory_payload,
        alias_lookup,
        readiness_lookup,
        seasonality_lookup,
    )
    locked_denominator_rules_respected, locked_denominator_rule_mismatches = _validate_locked_denominator_rules(inventory)
    denominator_contract = _build_denominator_contract(inventory)
    seasonality_contract = _build_seasonality_contract_frame(inventory, seasonality_lookup)
    supersession_map = _build_supersession_map(round_config)
    inventory_summary = _inventory_summary(inventory, escalation_protocol)
    round_charter = _build_round_charter(
        round_config,
        input_paths["release_report"],
        inventory_summary,
        inventory,
        supersession_map,
    )

    compatibility_payload = _materialize_legacy_alias(root, legacy_alias)

    dashboard_hash_after = _file_hash(dashboard_manifest)
    statcalc_recipe_ids_after = _statcalc_recipe_ids(root)

    gates = {
        "gold_release_pass": input_paths["release_report"].get("Overall status") in PASS_LIKE_VALUES,
        "stage5_dictionary_consistent": _validated_stage5_gate(input_paths["stage5_matrix"]),
        "promoted_outcomes_have_release_readiness": inventory_flags["promoted_readiness_ok"],
        "promoted_outcomes_have_valid_denominators": inventory_flags["promoted_denominator_ok"],
        "locked_denominator_rules_respected": locked_denominator_rules_respected,
        "seasonality_profiles_complete": inventory_flags["seasonality_complete"],
        "outcomes_resolve_to_aliases": inventory_flags["alias_resolution_ok"],
        "round_paths_consistent_with_roles": inventory_flags["round_path_ok"],
        "dashboard_contract_unchanged": dashboard_hash_before == dashboard_hash_after,
        "statcalc_recipe_ids_unchanged": statcalc_recipe_ids_before == statcalc_recipe_ids_after,
    }
    status = "PASS" if all(gates.values()) else "FAIL"

    output_root = analysis_step0_root(root)
    frozen_inventory_path = output_root / "frozen_outcome_inventory.csv"
    denominator_contract_path = output_root / "denominator_contract.csv"
    seasonality_contract_path = output_root / "seasonality_contract.csv"
    supersession_map_path = output_root / "supersession_map.csv"
    round_charter_path = output_root / "round_charter.json"
    report_path = output_root / "step0_report.md"

    inventory.to_csv(frozen_inventory_path, index=False)
    denominator_contract.to_csv(denominator_contract_path, index=False)
    seasonality_contract.to_csv(seasonality_contract_path, index=False)
    supersession_map.to_csv(supersession_map_path, index=False)
    round_charter_path.write_text(json.dumps(round_charter, indent=2), encoding="utf-8")

    outputs = {
        "frozen_outcome_inventory": _relative(root, frozen_inventory_path),
        "denominator_contract": _relative(root, denominator_contract_path),
        "seasonality_contract": _relative(root, seasonality_contract_path),
        "supersession_map": _relative(root, supersession_map_path),
        "round_charter": _relative(root, round_charter_path),
        "step0_report": _relative(root, report_path),
    }
    report_path.write_text(
        _render_step0_report(
            round_config,
            input_paths["release_report"],
            {
                "release_report_path": _relative(root, input_paths["release_report_path"]),
                "endpoint_ledger_path": _relative(root, input_paths["endpoint_ledger_path"]),
                "stage5_matrix_path": _relative(root, input_paths["stage5_matrix_path"]),
            },
            gates,
            inventory_summary,
            outputs,
            workspace_report,
            compatibility_payload,
        ),
        encoding="utf-8",
    )
    _write_step0_publication_notes(
        root,
        round_config=round_config,
        gates=gates,
        inventory_summary=inventory_summary,
        outputs=outputs,
    )

    payload: dict[str, object] = {
        "status": status,
        "step": "step0_freeze_governance_reset",
        "analysis_id": round_config["analysis_id"],
        "years": round_config["years"],
        "directories": directories,
        "gates": gates,
        "inventory_summary": inventory_summary,
        "outputs": outputs,
        "workspace_report": workspace_report,
        "compatibility": {
            "mode": round_config["strategy"],
            "legacy_alias_used": False,
            "replacement_command": None,
            "legacy_alias": None,
        },
        "locked_denominator_rule_mismatches": locked_denominator_rule_mismatches,
    }
    payload.update({key: value for key, value in compatibility_payload.items() if key not in {"legacy_reports", "legacy_alias"}})
    return payload
