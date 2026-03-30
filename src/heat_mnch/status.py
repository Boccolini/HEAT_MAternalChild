from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import load_yaml


VALID_STATUSES = {"Ready now", "Partially ready", "Blocked"}

SILVER_RELEASE_MANIFEST = "artifacts/manifests/silver_release_manifest.json"
DICTIONARY_MANIFEST = "artifacts/manifests/mnch_outcome_dictionary_icd10.manifest.json"

OUTCOME_PREREQUISITES = {
    "preterm_birth": {
        "silver_manifests": ["silver_sinasc", "silver_climate"],
        "requires_dictionary": False,
    },
    "stillbirth": {
        "silver_manifests": ["silver_stillbirth", "silver_sinasc", "silver_climate"],
        "requires_dictionary": False,
    },
    "neonatal_infant_under5_mortality": {
        "silver_manifests": ["silver_sim", "silver_population", "silver_climate"],
        "requires_dictionary": False,
    },
    "obstetric_admissions": {
        "silver_manifests": ["silver_sih", "silver_climate"],
        "requires_dictionary": True,
    },
    "hypertensive_disorders_of_pregnancy": {
        "silver_manifests": ["silver_sih", "silver_climate"],
        "requires_dictionary": True,
    },
    "preterm_labor_admissions": {
        "silver_manifests": ["silver_sih", "silver_climate"],
        "requires_dictionary": True,
    },
    "gestational_diabetes_admissions": {
        "silver_manifests": ["silver_sih", "silver_climate"],
        "requires_dictionary": True,
    },
    "acute_care_proxy": {
        "silver_manifests": ["silver_analytical", "silver_climate", "silver_population"],
        "requires_dictionary": False,
    },
}


def load_outcomes(root: Path) -> list[dict[str, object]]:
    payload = load_yaml(root / "config" / "rules" / "outcomes.yaml")
    outcomes = payload.get("outcomes", [])
    if not isinstance(outcomes, list):
        raise ValueError("config/rules/outcomes.yaml must define a list under 'outcomes'")
    return outcomes


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _join_items(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def _silver_release_state(root: Path) -> dict[str, object]:
    manifest_path = root / SILVER_RELEASE_MANIFEST
    payload = _load_json(manifest_path)
    if payload is None:
        return {
            "status": "MISSING",
            "manifest_path": manifest_path,
            "report_path": root / "artifacts" / "status" / "silver_release_candidate_report.md",
            "included_manifests": {},
        }
    included = payload.get("included_manifests", {})
    return {
        "status": str(payload.get("status", "UNKNOWN")).upper(),
        "manifest_path": manifest_path,
        "report_path": root / str(payload.get("report_path", "artifacts/status/silver_release_candidate_report.md")),
        "included_manifests": included if isinstance(included, dict) else {},
    }


def _manifest_exists(root: Path, relative_path: str) -> bool:
    return (root / relative_path).exists()


def _evaluate_outcome(root: Path, outcome: dict[str, object], silver_release: dict[str, object]) -> dict[str, object]:
    outcome_name = str(outcome["outcome"])
    rules = OUTCOME_PREREQUISITES.get(outcome_name, {})
    silver_manifests = list(rules.get("silver_manifests", []))
    requires_dictionary = bool(rules.get("requires_dictionary", False))

    evidence_paths: list[str] = []
    silver_manifest_path = _relative_path(root, Path(silver_release["manifest_path"]))
    evidence_paths.append(silver_manifest_path)

    report_path = Path(silver_release["report_path"])
    if report_path.exists():
        evidence_paths.append(_relative_path(root, report_path))

    silver_status = str(silver_release["status"])
    included_manifests = silver_release["included_manifests"]
    if not isinstance(included_manifests, dict):
        included_manifests = {}

    missing: list[str] = []
    for manifest_name in silver_manifests:
        manifest_relpath = included_manifests.get(manifest_name)
        if not isinstance(manifest_relpath, str):
            missing.append(manifest_name)
            continue
        if not _manifest_exists(root, manifest_relpath):
            missing.append(manifest_name)
            continue
        evidence_paths.append(manifest_relpath)

    if requires_dictionary:
        dictionary_path = root / DICTIONARY_MANIFEST
        if dictionary_path.exists():
            evidence_paths.append(_relative_path(root, dictionary_path))
        else:
            missing.append("mnch_outcome_dictionary_icd10")

    if silver_status != "PASS":
        status = "Blocked"
        reason = "Approved silver release artifacts are not available yet for this outcome."
        next_action = "Resolve the silver release gate before authorizing any canonical gold stage."
    elif missing:
        status = "Partially ready"
        reason = f"Silver release is PASS, but outcome-specific prerequisite manifests are missing: {_join_items(missing)}."
        next_action = "Refresh the missing silver or dictionary manifests before authorizing the canonical gold gate."
    else:
        status = "Ready now"
        reason = "Approved silver release artifacts cover the upstream inputs required for this outcome."
        next_action = f"Proceed to canonical gold promotion for `{outcome['gold_table']}` when that stage is explicitly authorized."

    return {
        "status": status,
        "evidence_paths": evidence_paths,
        "reason": reason,
        "next_action": next_action,
    }


def init_status_note(root: Path) -> Path:
    outcomes = load_outcomes(root)
    output_path = root / "artifacts" / "status" / "gold_readiness_status.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silver_release = _silver_release_state(root)
    evaluations = [_evaluate_outcome(root, item, silver_release) for item in outcomes]
    ready_count = sum(result["status"] == "Ready now" for result in evaluations)
    partial_count = sum(result["status"] == "Partially ready" for result in evaluations)
    blocked_count = sum(result["status"] == "Blocked" for result in evaluations)
    lines = [
        "# Gold Readiness Status",
        "",
        f"- Generated at: `{_utc_now()}`",
        "- Evaluation basis: approved silver release artifacts only; existing gold artifacts are intentionally ignored in this checkpoint.",
        f"- Silver release gate: `{silver_release['status']}` ({_relative_path(root, Path(silver_release['manifest_path']))})",
        f"- Outcomes ready now: `{ready_count}`",
        f"- Outcomes partially ready: `{partial_count}`",
        f"- Outcomes blocked: `{blocked_count}`",
        "",
    ]
    for item, evaluation in zip(outcomes, evaluations):
        required_fields = item.get("required_gold_fields", [])
        required_fields_text = ", ".join(required_fields) if isinstance(required_fields, list) and required_fields else "none declared"
        lines.extend(
            [
                f"## {item['outcome']}",
                "",
                f"- Status: {evaluation['status']}",
                f"- Gold table: {item['gold_table']}",
                f"- Required gold fields: {required_fields_text}",
                f"- Gold-ready criterion: {item['gold_ready_criterion']}",
                f"- Upstream evidence: {'; '.join(evaluation['evidence_paths'])}",
                f"- Reason: {evaluation['reason']}",
                f"- Next corrective action: {evaluation['next_action']}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
