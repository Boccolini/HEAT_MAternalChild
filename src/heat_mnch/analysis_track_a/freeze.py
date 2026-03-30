from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .config import (
    load_framework_outcome_map,
    load_track_a_config,
    load_track_a_variable_aliases,
)
from .registry import track_a_root


REQUIRED_DENOMINATOR_FLAGS = (
    "DENOMINATOR_ZERO_FLAG",
    "DENOMINATOR_MISSING_FLAG",
    "RATE_NOT_COMPUTABLE_FLAG",
    "OFFSET_VALID_FLAG",
)
CLIMATE_LINEAGE_FIELDS = (
    "FONTE_CLIMA_FINAL",
    "FONTE_TMAX_C_FINAL",
    "FONTE_RH_MEAN_PCT_FINAL",
)
SCHEMA_FILES = {
    "gold_mnch_perinatal": "gold_mnch_perinatal_diario_municipio.schema.csv",
    "gold_mnch_mortality": "gold_mnch_mortality_diario_municipio.schema.csv",
    "gold_mnch_panel": "gold_mnch_panel_diario_municipio.schema.csv",
}
RELEASE_REPORT_LINE_PATTERN = re.compile(r"^- (?P<label>[^:]+): `(?P<value>.*)`$")
PASS_LIKE_VALUES = {"PASS", "pass", "yes", "True", True}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _load_schema_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [str(row["column_name"]) for row in reader if row.get("column_name")]


def _parquet_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def _parquet_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def _status_string(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def _load_track_a_alias_index(root: Path) -> dict[str, dict[str, Any]]:
    alias_payload = load_track_a_variable_aliases(root)
    index: dict[str, dict[str, Any]] = {}
    for row in alias_payload["aliases"]:
        index.setdefault(str(row["academic_variable_name"]), row)
    return index


def _build_manifest_package_freshness(
    root: Path,
    manifest_index: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    overall = True
    for entry in manifest_index["entries"]:
        package_path = root / str(entry["package_path"])
        source_path = root / str(entry["source_path"])
        package_exists = package_path.exists()
        source_exists = source_path.exists()
        freshness_pass = package_exists and source_exists and package_path.stat().st_mtime >= source_path.stat().st_mtime
        overall = overall and freshness_pass
        rows.append(
            {
                "filename": str(entry["filename"]),
                "package_exists": _yes_no(package_exists),
                "source_exists": _yes_no(source_exists),
                "package_freshness_pass": _yes_no(freshness_pass),
            }
        )
    return overall, rows


def _load_stage5_matrix(root: Path) -> pd.DataFrame:
    path = _resolve_first_existing(
        root,
        [
            "gold_stage5_dictionary_matrix.csv",
            "artifacts/status/gold_stage5_dictionary_matrix.csv",
        ],
    )
    return pd.read_csv(path)


def _build_table_summary(
    root: Path,
    config: dict[str, Any],
    stage5_matrix: pd.DataFrame,
    release_manifest_index: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    years_expected = set(config["years"])
    gold_manifest_entries = [
        entry
        for entry in release_manifest_index["entries"]
        if str(entry["filename"]).startswith("gold_gold_")
    ]
    table_summary: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for manifest_entry in gold_manifest_entries:
        package_manifest_path = root / str(manifest_entry["package_path"])
        manifest_payload = _load_json(package_manifest_path)
        table_name = str(manifest_payload["dataset"])
        entry_rows = list(manifest_payload["entries"])
        manifest_years = {int(item["year"]) for item in entry_rows}
        year_coverage_complete = manifest_years == years_expected
        package_freshness_pass = True
        row_count_match = True
        denominator_flags_present = True
        climate_lineage_fields_present = True
        schema_or_dictionary_consistent = True
        all_columns: set[str] = set()
        supporting_paths: list[str] = []

        schema_columns = _load_schema_columns(root / "schemas" / SCHEMA_FILES[table_name]) if table_name in SCHEMA_FILES else []

        for item in entry_rows:
            data_path = root / str(item["path"])
            supporting_paths.append(str(item["path"]))
            columns = _parquet_columns(data_path)
            all_columns.update(columns)
            row_count_match = row_count_match and (_parquet_row_count(data_path) == int(item["row_count"]))
            package_freshness_pass = package_freshness_pass and (package_manifest_path.stat().st_mtime >= data_path.stat().st_mtime)
            denominator_flags_present = denominator_flags_present and all(flag in columns for flag in REQUIRED_DENOMINATOR_FLAGS)
            climate_lineage_fields_present = climate_lineage_fields_present and all(flag in columns for flag in CLIMATE_LINEAGE_FIELDS)
            if schema_columns:
                schema_or_dictionary_consistent = schema_or_dictionary_consistent and all(column in columns for column in schema_columns)

        stage5_rows = stage5_matrix.loc[stage5_matrix["table"].astype(str) == table_name].copy()
        if table_name not in SCHEMA_FILES:
            schema_or_dictionary_consistent = (
                schema_or_dictionary_consistent
                and not stage5_rows.empty
                and all(stage5_rows["schema_consistent_with_dictionary_yes_no"].astype(str).str.lower().eq("yes"))
                and all(stage5_rows["required_documented_yes_no"].astype(str).str.lower().eq("yes"))
            )
        else:
            schema_or_dictionary_consistent = (
                schema_or_dictionary_consistent
                and not stage5_rows.empty
                and all(stage5_rows["required_documented_yes_no"].astype(str).str.lower().eq("yes"))
            )

        climate_documented_complete = (
            not stage5_rows.empty
            and all(stage5_rows["climate_enrichment_complete_yes_no"].astype(str).str.lower().eq("yes"))
        )
        internal_validation_pass = (
            not stage5_rows.empty
            and all(stage5_rows["internal_validation_pass_fail"].astype(str).str.lower().eq("pass"))
        )
        external_validation_pass = (
            not stage5_rows.empty
            and all(stage5_rows["external_validation_pass_fail"].astype(str).str.lower().eq("pass"))
        )

        table_summary[table_name] = {
            "table_name": table_name,
            "years_covered": sorted(manifest_years),
            "year_coverage_complete": year_coverage_complete,
            "row_count_match": row_count_match,
            "denominator_flags_present": denominator_flags_present,
            "climate_lineage_fields_present": climate_lineage_fields_present,
            "schema_or_dictionary_consistent": schema_or_dictionary_consistent,
            "artifact_freshness_pass": package_freshness_pass,
            "climate_documented_complete": climate_documented_complete,
            "internal_validation_pass": internal_validation_pass,
            "external_validation_pass": external_validation_pass,
            "all_columns": all_columns,
            "package_manifest_path": _relative(root, package_manifest_path),
            "data_paths": supporting_paths,
        }
        rows.append(
            {
                "table_name": table_name,
                "years_covered": ";".join(str(year) for year in sorted(manifest_years)),
                "year_coverage_complete": _yes_no(year_coverage_complete),
                "row_count_match": _yes_no(row_count_match),
                "denominator_flags_present": _yes_no(denominator_flags_present),
                "climate_lineage_fields_present": _yes_no(climate_lineage_fields_present),
                "schema_or_dictionary_consistent": _yes_no(schema_or_dictionary_consistent),
                "climate_documented_complete": _yes_no(climate_documented_complete),
                "internal_validation_pass": _yes_no(internal_validation_pass),
                "external_validation_pass": _yes_no(external_validation_pass),
                "artifact_freshness_pass": _yes_no(package_freshness_pass),
                "package_manifest_path": _relative(root, package_manifest_path),
            }
        )
    return table_summary, rows


def _load_readiness_ledger(root: Path, config: dict[str, Any]) -> pd.DataFrame:
    ledger_path = _resolve_first_existing(
        root,
        [
            str(config["frozen_inputs"]["endpoint_readiness_ledger_path"]),
            "artifacts/status/endpoint_readiness_ledger.csv",
        ],
    )
    return pd.read_csv(ledger_path)


def _build_endpoint_row(
    outcome: dict[str, Any],
    alias_index: dict[str, dict[str, Any]],
    readiness_lookup: dict[str, dict[str, Any]],
    table_summary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    academic_variable_name = str(outcome["academic_variable_name"])
    alias_row = alias_index[academic_variable_name]
    readiness_endpoint_name = outcome.get("readiness_endpoint_name") or outcome["canonical_variable"]
    readiness_row = readiness_lookup.get(str(readiness_endpoint_name))

    supporting_table_names = {str(outcome["canonical_dataset"])}
    if readiness_row is not None and str(readiness_row["gold_table"]):
        supporting_table_names.add(str(readiness_row["gold_table"]))

    supporting_summaries = [table_summary[name] for name in sorted(supporting_table_names) if name in table_summary]
    year_coverage_complete = bool(supporting_summaries) and all(item["year_coverage_complete"] for item in supporting_summaries)
    denominator_flags_present = bool(supporting_summaries) and all(item["denominator_flags_present"] for item in supporting_summaries)
    climate_lineage_fields_present = bool(supporting_summaries) and all(item["climate_lineage_fields_present"] for item in supporting_summaries)
    schema_or_dictionary_consistent = bool(supporting_summaries) and all(item["schema_or_dictionary_consistent"] for item in supporting_summaries)
    artifact_freshness_pass = bool(supporting_summaries) and all(item["artifact_freshness_pass"] for item in supporting_summaries)

    canonical_fields = table_summary[str(outcome["canonical_dataset"])]["all_columns"]
    canonical_variable_present = str(outcome["canonical_variable"]) in canonical_fields
    required_fields_present = (
        str(readiness_row["required_fields_present_yes_no"])
        if readiness_row is not None
        else _yes_no(canonical_variable_present)
    )
    denominator_rule = (
        str(readiness_row["denominator_rule"])
        if readiness_row is not None
        else "table-level support only; endpoint-specific denominator certification not available in the frozen readiness ledger"
    )
    denominator_valid = (
        str(readiness_row["denominator_valid_yes_no"])
        if readiness_row is not None
        else _yes_no(denominator_flags_present)
    )
    climate_enrichment_complete = (
        str(readiness_row["climate_enrichment_complete_yes_no"])
        if readiness_row is not None
        else _yes_no(climate_lineage_fields_present)
    )
    internal_validation_status = (
        str(readiness_row["internal_validation_pass_fail"])
        if readiness_row is not None
        else _status_string(all(item["internal_validation_pass"] for item in supporting_summaries))
    )
    external_validation_status = (
        str(readiness_row["external_validation_pass_fail"])
        if readiness_row is not None
        else _status_string(all(item["external_validation_pass"] for item in supporting_summaries))
    )
    readiness_status = str(readiness_row["status"]) if readiness_row is not None else "Table-level support only"

    if str(outcome["analysis_role"]) == "descriptive_only":
        stage1_classification = "descriptive_only"
        freeze_gate_decision = "include_in_track_a_descriptive_only"
    elif not year_coverage_complete or not schema_or_dictionary_consistent or not artifact_freshness_pass:
        stage1_classification = "deferred"
        freeze_gate_decision = "defer_from_track_a_until_release_integrity_is_restored"
    elif readiness_row is None:
        stage1_classification = "screening_only"
        freeze_gate_decision = "include_in_track_a_screening_only"
    elif readiness_status == "Blocked" or required_fields_present != "yes":
        stage1_classification = "deferred"
        freeze_gate_decision = "defer_from_track_a_until_endpoint_readiness_is_restored"
    elif readiness_status == "Partially ready":
        stage1_classification = "screening_only"
        freeze_gate_decision = "include_in_track_a_screening_only"
    elif denominator_valid != "yes" or climate_enrichment_complete != "yes":
        stage1_classification = "screening_only"
        freeze_gate_decision = "include_in_track_a_screening_only"
    else:
        stage1_classification = str(outcome["analysis_role"])
        freeze_gate_decision = (
            "include_in_track_a_benchmark"
            if stage1_classification == "benchmark_ready"
            else "include_in_track_a_screening_only"
        )

    return {
        "prompt_label": str(outcome["prompt_label"]),
        "academic_variable_name": academic_variable_name,
        "academic_label": str(alias_row["academic_label"]),
        "canonical_dataset": str(outcome["canonical_dataset"]),
        "canonical_variable": str(outcome["canonical_variable"]),
        "readiness_endpoint_name": str(readiness_endpoint_name) if readiness_endpoint_name else "",
        "readiness_table": str(readiness_row["gold_table"]) if readiness_row is not None else "",
        "readiness_status": readiness_status,
        "default_analysis_role": str(outcome["analysis_role"]),
        "stage1_classification": stage1_classification,
        "semantic_alignment": str(outcome.get("semantic_alignment") or "unspecified"),
        "semantic_note": str(outcome.get("semantic_note") or ""),
        "denominator_rule": denominator_rule,
        "denominator_valid": denominator_valid,
        "climate_enrichment_complete": climate_enrichment_complete,
        "required_fields_present": required_fields_present,
        "internal_validation_status": internal_validation_status,
        "external_validation_status": external_validation_status,
        "year_coverage_complete": _yes_no(year_coverage_complete),
        "schema_or_dictionary_consistent": _yes_no(schema_or_dictionary_consistent),
        "denominator_flags_present": _yes_no(denominator_flags_present),
        "climate_lineage_fields_present": _yes_no(climate_lineage_fields_present),
        "artifact_freshness_pass": _yes_no(artifact_freshness_pass),
        "freeze_gate_decision": freeze_gate_decision,
        "supporting_tables": ";".join(sorted(supporting_table_names)),
    }


def _augment_endpoint_guardrails(endpoint_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in endpoint_rows:
        grouped[str(row["academic_variable_name"])].append(row)

    for academic_variable_name, rows in grouped.items():
        duplicate_prompt_concept = len(rows) > 1
        for row in rows:
            row["public_concept_name"] = academic_variable_name
            row["public_concept_label"] = rows[0]["academic_label"]
            row["duplicate_prompt_concept_flag"] = _yes_no(duplicate_prompt_concept)
            row["headline_eligible_flag"] = _yes_no(row["stage1_classification"] == "benchmark_ready")
            row["benchmark_model_eligible_flag"] = _yes_no(row["stage1_classification"] == "benchmark_ready")
            row["descriptive_atlas_eligible_flag"] = _yes_no(
                row["stage1_classification"] in {"benchmark_ready", "screening_only", "descriptive_only"}
            )
    return endpoint_rows


def _build_public_concept_registry(endpoint_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in endpoint_rows:
        grouped[str(row["public_concept_name"])].append(row)

    registry_rows: list[dict[str, Any]] = []
    for public_concept_name, rows in sorted(grouped.items()):
        prompt_labels = sorted(str(row["prompt_label"]) for row in rows)
        classifications = sorted({str(row["stage1_classification"]) for row in rows})
        semantic_alignments = sorted({str(row["semantic_alignment"]) for row in rows})
        duplicate_prompt_concept = len(rows) > 1
        proxy_duplicate = duplicate_prompt_concept and all(
            str(row["semantic_alignment"]) == "proxy_endpoint_duplicate_mapping" for row in rows
        )
        if proxy_duplicate:
            concept_resolution_status = "collapsed_proxy_duplicate"
            concept_resolution_note = (
                "Multiple prompt-level concepts resolve to the same certified acute-care proxy and must appear once in public-facing Track A outputs."
            )
        elif duplicate_prompt_concept:
            concept_resolution_status = "shared_public_concept"
            concept_resolution_note = (
                "Multiple prompt-level concepts map to a single public-facing Track A concept and should be reported once per public concept."
            )
        else:
            concept_resolution_status = "one_prompt_one_public_concept"
            concept_resolution_note = "The prompt-level concept maps uniquely to a single public-facing Track A concept."

        registry_rows.append(
            {
                "public_concept_name": public_concept_name,
                "public_concept_label": rows[0]["public_concept_label"],
                "prompt_labels": ";".join(prompt_labels),
                "prompt_label_count": len(prompt_labels),
                "stage1_classifications": ";".join(classifications),
                "semantic_alignments": ";".join(semantic_alignments),
                "headline_eligible_flag": _yes_no(any(row["headline_eligible_flag"] == "yes" for row in rows)),
                "concept_resolution_status": concept_resolution_status,
                "concept_resolution_note": concept_resolution_note,
            }
        )
    return registry_rows


def _render_concept_resolution_report(
    concept_registry_rows: list[dict[str, Any]],
    outputs: dict[str, str],
) -> str:
    lines = [
        "# Track A Concept Resolution Report",
        "",
        f"- Public concept registry: `{outputs['public_concept_registry']}`",
        "",
        "## Concept Resolution",
        "",
    ]
    for row in concept_registry_rows:
        lines.append(
            "- "
            f"{row['public_concept_name']}: prompt_labels={row['prompt_labels']}, "
            f"status={row['concept_resolution_status']}, "
            f"headline_eligible_flag={row['headline_eligible_flag']}"
        )
        lines.append(f"  {row['concept_resolution_note']}")
    return "\n".join(lines) + "\n"


def _render_freeze_report(
    config: dict[str, Any],
    release_metrics: dict[str, str],
    package_freshness_pass: bool,
    package_rows: list[dict[str, Any]],
    table_rows: list[dict[str, Any]],
    endpoint_rows: list[dict[str, Any]],
    concept_registry_rows: list[dict[str, Any]],
    outputs: dict[str, str],
) -> str:
    benchmark_ready = sum(row["stage1_classification"] == "benchmark_ready" for row in endpoint_rows)
    screening_only = sum(row["stage1_classification"] == "screening_only" for row in endpoint_rows)
    descriptive_only = sum(row["stage1_classification"] == "descriptive_only" for row in endpoint_rows)
    deferred = sum(row["stage1_classification"] == "deferred" for row in endpoint_rows)
    headline_eligible = sum(row["headline_eligible_flag"] == "yes" for row in endpoint_rows)
    caveat_rows = [
        row for row in endpoint_rows if row["stage1_classification"] != row["default_analysis_role"]
        or row["semantic_alignment"] != "direct_endpoint_certification"
    ]
    report_lines = [
        "# Track A Gold Freeze Report",
        "",
        f"- Generated at: `{_now_utc()}`",
        f"- Analysis title: `{config['analysis_title']}`",
        f"- Frozen release status: `{release_metrics.get('Overall status', 'UNKNOWN')}`",
        f"- Silver release gate: `{release_metrics.get('Silver release gate', 'UNKNOWN')}`",
        f"- Canonical gold tables PASS: `{release_metrics.get('Canonical gold tables PASS', 'UNKNOWN')}`",
        f"- Derived analytical views PASS: `{release_metrics.get('Derived analytical views PASS', 'UNKNOWN')}`",
        f"- Ready now endpoints in the frozen ledger: `{release_metrics.get('Ready now endpoints', 'UNKNOWN')}`",
        f"- Package freshness checks PASS: `{_yes_no(package_freshness_pass)}`",
        f"- Track A benchmark-ready outcomes: `{benchmark_ready}`",
        f"- Track A screening-only outcomes: `{screening_only}`",
        f"- Track A descriptive-only outcomes: `{descriptive_only}`",
        f"- Track A deferred outcomes: `{deferred}`",
        f"- Headline-eligible outcomes: `{headline_eligible}`",
        f"- Analysis readiness matrix: `{outputs['analysis_ready_matrix']}`",
        f"- Public concept registry: `{outputs['public_concept_registry']}`",
        f"- Concept resolution report: `{outputs['concept_resolution_report']}`",
        "",
        "## Table-Level Validation",
        "",
    ]
    for row in sorted(table_rows, key=lambda item: str(item["table_name"])):
        report_lines.append(
            "- "
            f"{row['table_name']}: year_coverage_complete={row['year_coverage_complete']}, "
            f"schema_or_dictionary_consistent={row['schema_or_dictionary_consistent']}, "
            f"denominator_flags_present={row['denominator_flags_present']}, "
            f"climate_lineage_fields_present={row['climate_lineage_fields_present']}, "
            f"artifact_freshness_pass={row['artifact_freshness_pass']}"
        )
    report_lines.extend(["", "## Package Freshness", ""])
    for row in sorted(package_rows, key=lambda item: str(item["filename"])):
        report_lines.append(
            "- "
            f"{row['filename']}: package_exists={row['package_exists']}, "
            f"source_exists={row['source_exists']}, "
            f"package_freshness_pass={row['package_freshness_pass']}"
        )
    report_lines.extend(["", "## Outcome Classification", ""])
    for row in sorted(endpoint_rows, key=lambda item: str(item["prompt_label"])):
        report_lines.append(
            "- "
            f"{row['prompt_label']} -> {row['academic_variable_name']}: "
            f"classification={row['stage1_classification']}, "
            f"readiness_status={row['readiness_status']}, "
            f"semantic_alignment={row['semantic_alignment']}, "
            f"headline_eligible_flag={row['headline_eligible_flag']}"
        )
    report_lines.extend(["", "## Public Concepts", ""])
    for row in concept_registry_rows:
        report_lines.append(
            "- "
            f"{row['public_concept_name']}: prompt_labels={row['prompt_labels']}, "
            f"concept_resolution_status={row['concept_resolution_status']}, "
            f"headline_eligible_flag={row['headline_eligible_flag']}"
        )
    report_lines.extend(["", "## Caveats", ""])
    if caveat_rows:
        for row in caveat_rows:
            report_lines.append(
                "- "
                f"{row['prompt_label']}: {row['semantic_note'] or 'Classification differs from the default analysis role.'}"
            )
    else:
        report_lines.append("- None")
    return "\n".join(report_lines) + "\n"


def build_track_a_freeze(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    config = load_track_a_config(root, config_path=config_path)
    outcome_map = load_framework_outcome_map(root)
    alias_index = _load_track_a_alias_index(root)

    output_dir = track_a_root(root) / "freeze"
    output_dir.mkdir(parents=True, exist_ok=True)

    release_report_path = _resolve_first_existing(
        root,
        [
            str(config["frozen_inputs"]["gold_release_report_path"]),
            "artifacts/status/gold_release_candidate_report.md",
        ],
    )
    release_manifest_dir = root / str(config["frozen_inputs"]["gold_manifest_package_dir"])
    release_manifest_path = release_manifest_dir / "gold_release_manifest.json"
    release_manifest_index_path = release_manifest_dir / "manifest_index.json"
    release_manifest = _load_json(release_manifest_path)
    release_manifest_index = _load_json(release_manifest_index_path)
    release_metrics = _parse_release_report(release_report_path)

    stage5_matrix = _load_stage5_matrix(root)
    table_summary, table_rows = _build_table_summary(
        root=root,
        config=config,
        stage5_matrix=stage5_matrix,
        release_manifest_index=release_manifest_index,
    )
    readiness_ledger = _load_readiness_ledger(root, config)
    readiness_lookup = {
        str(row["endpoint_name"]): row
        for row in readiness_ledger.to_dict(orient="records")
    }
    endpoint_rows = [
        _build_endpoint_row(
            outcome=outcome,
            alias_index=alias_index,
            readiness_lookup=readiness_lookup,
            table_summary=table_summary,
        )
        for outcome in outcome_map["track_a_outcomes"]
    ]
    endpoint_rows = _augment_endpoint_guardrails(endpoint_rows)
    concept_registry_rows = _build_public_concept_registry(endpoint_rows)
    endpoint_frame = pd.DataFrame(endpoint_rows).sort_values(
        ["stage1_classification", "prompt_label"], ignore_index=True
    )
    public_concept_registry_frame = pd.DataFrame(concept_registry_rows).sort_values(
        ["headline_eligible_flag", "public_concept_name"], ascending=[False, True], ignore_index=True
    )

    package_freshness_pass, package_rows = _build_manifest_package_freshness(root, release_manifest_index)

    analysis_ready_matrix_path = output_dir / "analysis_ready_matrix.csv"
    gold_freeze_report_path = output_dir / "gold_freeze_report.md"
    public_concept_registry_path = output_dir / "public_concept_registry.csv"
    concept_resolution_report_path = output_dir / "concept_resolution_report.md"
    endpoint_frame.to_csv(analysis_ready_matrix_path, index=False)
    public_concept_registry_frame.to_csv(public_concept_registry_path, index=False)

    outputs = {
        "gold_freeze_report": _relative(root, gold_freeze_report_path),
        "analysis_ready_matrix": _relative(root, analysis_ready_matrix_path),
        "public_concept_registry": _relative(root, public_concept_registry_path),
        "concept_resolution_report": _relative(root, concept_resolution_report_path),
    }
    gold_freeze_report_path.write_text(
        _render_freeze_report(
            config=config,
            release_metrics=release_metrics,
            package_freshness_pass=package_freshness_pass,
            package_rows=package_rows,
            table_rows=table_rows,
            endpoint_rows=endpoint_rows,
            concept_registry_rows=concept_registry_rows,
            outputs=outputs,
        ),
        encoding="utf-8",
    )
    concept_resolution_report_path.write_text(
        _render_concept_resolution_report(concept_registry_rows, outputs),
        encoding="utf-8",
    )

    deferred_count = int(endpoint_frame["stage1_classification"].eq("deferred").sum())
    hard_failures = (
        release_manifest.get("status") not in PASS_LIKE_VALUES
        or release_metrics.get("Overall status") not in {"PASS", "pass"}
        or not package_freshness_pass
        or not all(row["schema_or_dictionary_consistent"] == "yes" for row in table_rows)
        or not all(row["artifact_freshness_pass"] == "yes" for row in table_rows)
    )
    status = "PASS" if not hard_failures and deferred_count == 0 else "FAIL"

    return {
        "status": status,
        "stage": "stage1_analytical_freeze_and_readiness_gate",
        "analysis_id": config["analysis_id"],
        "analysis_title": config["analysis_title"],
        "generated_at": _now_utc(),
        "release_status": str(release_manifest.get("status")),
        "release_report_path": _relative(root, release_report_path),
        "release_manifest_path": _relative(root, release_manifest_path),
        "release_manifest_index_path": _relative(root, release_manifest_index_path),
        "package_freshness_pass": package_freshness_pass,
        "table_count": len(table_rows),
        "benchmark_ready_count": int(endpoint_frame["stage1_classification"].eq("benchmark_ready").sum()),
        "screening_only_count": int(endpoint_frame["stage1_classification"].eq("screening_only").sum()),
        "descriptive_only_count": int(endpoint_frame["stage1_classification"].eq("descriptive_only").sum()),
        "deferred_count": deferred_count,
        "headline_eligible_count": int(endpoint_frame["headline_eligible_flag"].eq("yes").sum()),
        "outputs": outputs,
        "table_validation_rows": table_rows,
        "package_freshness_rows": package_rows,
    }
