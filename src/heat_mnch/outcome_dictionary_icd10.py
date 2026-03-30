from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .manifest import sha256_file


SIM_ICD10_COLUMNS = ("CAUSABAS", "LINHAA", "LINHAB", "LINHAC", "LINHAD", "LINHAII")
SIH_ANY_DIAG_COLUMNS = (
    "DIAG_PRINC",
    "DIAG_SECUN",
    "DIAGSEC1",
    "DIAGSEC2",
    "DIAGSEC3",
    "DIAGSEC4",
    "DIAGSEC5",
    "DIAGSEC6",
    "DIAGSEC7",
    "DIAGSEC8",
    "DIAGSEC9",
    "CID_ASSO",
    "CID_MORTE",
)
SIH_EXTERNAL_CAUSE_COLUMNS = (
    "DIAG_SECUN",
    "DIAGSEC1",
    "DIAGSEC2",
    "DIAGSEC3",
    "DIAGSEC4",
    "DIAGSEC5",
    "DIAGSEC6",
    "DIAGSEC7",
    "DIAGSEC8",
    "DIAGSEC9",
    "CID_NOTIF",
    "CID_ASSO",
    "CID_MORTE",
)

SIM_SCOPE_UNDERLYING = "UNDERLYING_BASIC_CAUSE"
SIM_SCOPE_ANY_MENTION = "ANY_MENTION"
SIH_SCOPE_PRINCIPAL = "PRINCIPAL_DIAGNOSIS"
SIH_SCOPE_ANY_DIAG = "ANY_DIAGNOSIS"
SIH_SCOPE_EXTERNAL_CAUSE = "EXTERNAL_CAUSE_FIELDS"
OUTCOME_DICTIONARY_BUILD_LOGIC_VERSION = 4
SIM_CAUSE_SPLIT_PATTERN = re.compile(r"[^A-Z0-9.]+")


@dataclass(frozen=True)
class OutcomeGroupSpec:
    outcome_group: str
    heat_relation: str
    group_role: str
    rule_literals: tuple[str, ...]
    canonical_scope_sim: str
    canonical_scope_sih: str


OUTCOME_GROUP_SPECS: tuple[OutcomeGroupSpec, ...] = (
    OutcomeGroupSpec(
        outcome_group="HDP_ANY_PRIMARY",
        heat_relation="MATERNAL_PHENOTYPE",
        group_role="PRIMARY",
        rule_literals=("O10", "O11", "O13", "O14.*", "O15.*", "O16"),
        canonical_scope_sim=SIM_SCOPE_UNDERLYING,
        canonical_scope_sih=SIH_SCOPE_PRINCIPAL,
    ),
    OutcomeGroupSpec(
        outcome_group="HDP_SEVERE_PRIMARY",
        heat_relation="MATERNAL_PHENOTYPE",
        group_role="PRIMARY",
        rule_literals=("O14.1", "O14.2", "O15.*"),
        canonical_scope_sim=SIM_SCOPE_UNDERLYING,
        canonical_scope_sih=SIH_SCOPE_PRINCIPAL,
    ),
    OutcomeGroupSpec(
        outcome_group="HDP_GESTATIONAL_HTN_PRIMARY",
        heat_relation="MATERNAL_PHENOTYPE",
        group_role="PRIMARY",
        rule_literals=("O13",),
        canonical_scope_sim=SIM_SCOPE_UNDERLYING,
        canonical_scope_sih=SIH_SCOPE_PRINCIPAL,
    ),
    OutcomeGroupSpec(
        outcome_group="HDP_PREECLAMPSIA_PRIMARY",
        heat_relation="MATERNAL_PHENOTYPE",
        group_role="PRIMARY",
        rule_literals=("O14.*",),
        canonical_scope_sim=SIM_SCOPE_UNDERLYING,
        canonical_scope_sih=SIH_SCOPE_PRINCIPAL,
    ),
    OutcomeGroupSpec(
        outcome_group="HDP_ECLAMPSIA_PRIMARY",
        heat_relation="MATERNAL_PHENOTYPE",
        group_role="PRIMARY",
        rule_literals=("O15.*",),
        canonical_scope_sim=SIM_SCOPE_UNDERLYING,
        canonical_scope_sih=SIH_SCOPE_PRINCIPAL,
    ),
    OutcomeGroupSpec(
        outcome_group="PRETERM_LABOR_PRIMARY",
        heat_relation="MATERNAL_PHENOTYPE",
        group_role="PRIMARY",
        rule_literals=("O60.*",),
        canonical_scope_sim=SIM_SCOPE_UNDERLYING,
        canonical_scope_sih=SIH_SCOPE_PRINCIPAL,
    ),
    OutcomeGroupSpec(
        outcome_group="THREATENED_PRETERM_LABOR_SENSITIVITY",
        heat_relation="MATERNAL_PHENOTYPE",
        group_role="SENSITIVITY",
        rule_literals=("O47.0", "O60.*"),
        canonical_scope_sim=SIM_SCOPE_ANY_MENTION,
        canonical_scope_sih=SIH_SCOPE_ANY_DIAG,
    ),
    OutcomeGroupSpec(
        outcome_group="GDM_GESTATIONAL_PRIMARY",
        heat_relation="MATERNAL_PHENOTYPE",
        group_role="PRIMARY",
        rule_literals=("O24.4",),
        canonical_scope_sim=SIM_SCOPE_UNDERLYING,
        canonical_scope_sih=SIH_SCOPE_PRINCIPAL,
    ),
    OutcomeGroupSpec(
        outcome_group="APH_ANTEPARTUM_PRIMARY",
        heat_relation="MATERNAL_PHENOTYPE",
        group_role="PRIMARY",
        rule_literals=("O44.*", "O45.*", "O46.*"),
        canonical_scope_sim=SIM_SCOPE_UNDERLYING,
        canonical_scope_sih=SIH_SCOPE_PRINCIPAL,
    ),
    OutcomeGroupSpec(
        outcome_group="DIABETES_PREGNANCY_BROAD_SENSITIVITY",
        heat_relation="MATERNAL_PHENOTYPE",
        group_role="SENSITIVITY",
        rule_literals=("O24.*",),
        canonical_scope_sim=SIM_SCOPE_ANY_MENTION,
        canonical_scope_sih=SIH_SCOPE_ANY_DIAG,
    ),
    OutcomeGroupSpec(
        outcome_group="CHILD_HEAT_THERMAL_PRIMARY",
        heat_relation="DIRECT_HEAT_OUTCOME",
        group_role="PRIMARY",
        rule_literals=("T67.*", "P81.0"),
        canonical_scope_sim=SIM_SCOPE_UNDERLYING,
        canonical_scope_sih=SIH_SCOPE_PRINCIPAL,
    ),
    OutcomeGroupSpec(
        outcome_group="CHILD_DEHYDRATION_PRIMARY",
        heat_relation="DIRECT_HEAT_OUTCOME",
        group_role="PRIMARY",
        rule_literals=("E86", "P74.1"),
        canonical_scope_sim=SIM_SCOPE_UNDERLYING,
        canonical_scope_sih=SIH_SCOPE_PRINCIPAL,
    ),
    OutcomeGroupSpec(
        outcome_group="CHILD_FLUID_ELECTROLYTE_SENSITIVITY",
        heat_relation="INDIRECT_HEAT_SENSITIVE_OUTCOME",
        group_role="SENSITIVITY",
        rule_literals=("E87.0", "E87.1", "P74.2", "P74.3"),
        canonical_scope_sim=SIM_SCOPE_ANY_MENTION,
        canonical_scope_sih=SIH_SCOPE_ANY_DIAG,
    ),
    OutcomeGroupSpec(
        outcome_group="CHILD_DIARRHEA_GI_PRIMARY",
        heat_relation="INDIRECT_HEAT_SENSITIVE_OUTCOME",
        group_role="PRIMARY",
        rule_literals=("A00-A09",),
        canonical_scope_sim=SIM_SCOPE_UNDERLYING,
        canonical_scope_sih=SIH_SCOPE_PRINCIPAL,
    ),
    OutcomeGroupSpec(
        outcome_group="CHILD_RESPIRATORY_ACUTE_PRIMARY",
        heat_relation="INDIRECT_HEAT_SENSITIVE_OUTCOME",
        group_role="PRIMARY",
        rule_literals=("J00-J06", "J09-J18", "J20-J22"),
        canonical_scope_sim=SIM_SCOPE_UNDERLYING,
        canonical_scope_sih=SIH_SCOPE_PRINCIPAL,
    ),
    OutcomeGroupSpec(
        outcome_group="CHILD_ASTHMA_WHEEZE_SENSITIVITY",
        heat_relation="INDIRECT_HEAT_SENSITIVE_OUTCOME",
        group_role="SENSITIVITY",
        rule_literals=("J45", "J46"),
        canonical_scope_sim=SIM_SCOPE_ANY_MENTION,
        canonical_scope_sih=SIH_SCOPE_ANY_DIAG,
    ),
    OutcomeGroupSpec(
        outcome_group="CHILD_RENAL_ACUTE_SENSITIVITY",
        heat_relation="INDIRECT_HEAT_SENSITIVE_OUTCOME",
        group_role="SENSITIVITY",
        rule_literals=("N17.*",),
        canonical_scope_sim=SIM_SCOPE_ANY_MENTION,
        canonical_scope_sih=SIH_SCOPE_ANY_DIAG,
    ),
    OutcomeGroupSpec(
        outcome_group="CHILD_MALNUTRITION_SLOW_MOVING_SENSITIVITY",
        heat_relation="SLOW_MOVING_INDIRECT_OUTCOME",
        group_role="SENSITIVITY",
        rule_literals=("E40-E46",),
        canonical_scope_sim=SIM_SCOPE_ANY_MENTION,
        canonical_scope_sih=SIH_SCOPE_ANY_DIAG,
    ),
    OutcomeGroupSpec(
        outcome_group="EXCESSIVE_NATURAL_HEAT_X30",
        heat_relation="LINKAGE_FLAG",
        group_role="LINKAGE",
        rule_literals=("X30",),
        canonical_scope_sim=SIM_SCOPE_ANY_MENTION,
        canonical_scope_sih=SIH_SCOPE_EXTERNAL_CAUSE,
    ),
)


DIRECT_HEAT_GROUPS = (
    "CHILD_HEAT_THERMAL_PRIMARY",
    "CHILD_DEHYDRATION_PRIMARY",
)
INDIRECT_HEAT_GROUPS = (
    "CHILD_FLUID_ELECTROLYTE_SENSITIVITY",
    "CHILD_DIARRHEA_GI_PRIMARY",
    "CHILD_RESPIRATORY_ACUTE_PRIMARY",
    "CHILD_ASTHMA_WHEEZE_SENSITIVITY",
    "CHILD_RENAL_ACUTE_SENSITIVITY",
)
SLOW_MOVING_HEAT_GROUPS = ("CHILD_MALNUTRITION_SLOW_MOVING_SENSITIVITY",)
MATERNAL_GROUPS = (
    "HDP_ANY_PRIMARY",
    "HDP_SEVERE_PRIMARY",
    "PRETERM_LABOR_PRIMARY",
    "THREATENED_PRETERM_LABOR_SENSITIVITY",
    "GDM_GESTATIONAL_PRIMARY",
    "APH_ANTEPARTUM_PRIMARY",
    "DIABETES_PREGNANCY_BROAD_SENSITIVITY",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_icd10_code(value: object) -> str:
    text = str(value).strip().upper().replace(".", "")
    text = "".join(character for character in text if character.isalnum())
    if not text or set(text) == {"0"}:
        return ""
    return text


def normalize_icd10_series(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).map(normalize_icd10_code)


def split_sim_cause_tokens(value: object) -> tuple[str, ...]:
    text = str(value).strip().upper()
    if not text:
        return ()
    tokens = tuple(
        code
        for code in (
            normalize_icd10_code(fragment)
            for fragment in SIM_CAUSE_SPLIT_PATTERN.split(text)
        )
        if code
    )
    if tokens:
        return tokens
    normalized = normalize_icd10_code(text)
    return (normalized,) if normalized else ()


def _icd10_category3(code: str) -> str:
    normalized = normalize_icd10_code(code)
    return normalized[:3] if len(normalized) >= 3 else normalized


def _expand_category_rule(codes: set[str], category3: str) -> set[str]:
    return {code for code in codes if code.startswith(category3)}


def _expand_prefix_rule(codes: set[str], prefix: str) -> set[str]:
    return {code for code in codes if code.startswith(prefix)}


def _expand_range_rule(codes: set[str], start_category: str, end_category: str) -> set[str]:
    categories = {
        code[:3]
        for code in codes
        if len(code) >= 3 and start_category <= code[:3] <= end_category
    }
    expanded: set[str] = set()
    for category in categories:
        expanded.update(_expand_category_rule(codes, category))
    return expanded


def _expand_rule_literal(codes: set[str], literal: str) -> set[str]:
    normalized_literal = literal.strip().upper()
    if normalized_literal.endswith(".*"):
        return _expand_prefix_rule(codes, normalize_icd10_code(normalized_literal[:-2]))
    if "-" in normalized_literal:
        start_literal, end_literal = normalized_literal.split("-", 1)
        return _expand_range_rule(codes, _icd10_category3(start_literal), _icd10_category3(end_literal))
    normalized = normalize_icd10_code(normalized_literal)
    if len(normalized) <= 3:
        return _expand_category_rule(codes, normalized)
    return _expand_prefix_rule(codes, normalized)


def _collect_observed_codes(root: Path, start_year: int, end_year: int) -> set[str]:
    codes: set[str] = set()
    dataset_specs = {
        "sim": SIM_ICD10_COLUMNS,
        "sih": tuple(dict.fromkeys(SIH_ANY_DIAG_COLUMNS + ("CID_NOTIF",))),
    }
    sim_token_cache: dict[str, tuple[str, ...]] = {}
    for dataset, columns in dataset_specs.items():
        for year in range(start_year, end_year + 1):
            path = root / "data" / "bronze" / dataset / str(year) / f"{dataset}_{year}_source_preserving.csv.gz"
            for chunk in pd.read_csv(
                path,
                compression="gzip",
                sep=";",
                dtype=str,
                usecols=list(columns),
                chunksize=200_000 if dataset == "sim" else 300_000,
                keep_default_na=False,
                low_memory=False,
            ):
                if dataset == "sim":
                    codes.update(
                        code
                        for code in normalize_icd10_series(chunk["CAUSABAS"]).unique().tolist()
                        if code
                    )
                    for column in columns:
                        if column == "CAUSABAS":
                            continue
                        for raw_value in chunk[column].fillna("").astype(str).unique().tolist():
                            tokens = sim_token_cache.get(raw_value)
                            if tokens is None:
                                tokens = split_sim_cause_tokens(raw_value)
                                sim_token_cache[raw_value] = tokens
                            codes.update(tokens)
                    continue
                for column in columns:
                    codes.update(
                        code
                        for code in normalize_icd10_series(chunk[column]).unique().tolist()
                        if code
                    )
    return codes


def build_unified_outcome_dictionary(root: Path, start_year: int, end_year: int) -> dict[str, Any]:
    canonical_path = root / "artifacts" / "dictionaries" / "mnch_outcome_dictionary_icd10.csv"
    manifest_path = root / "artifacts" / "manifests" / "mnch_outcome_dictionary_icd10.manifest.json"
    summary_path = root / "artifacts" / "status" / "mnch_outcome_dictionary_icd10_summary.md"
    expected_years = list(range(start_year, end_year + 1))
    if canonical_path.exists() and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing_manifest.get("source_years") == expected_years
            and existing_manifest.get("build_logic_version") == OUTCOME_DICTIONARY_BUILD_LOGIC_VERSION
        ):
            dictionary_frame = pd.read_csv(canonical_path, dtype={"ICD10_CODE": str})
            if int(len(dictionary_frame)) == int(existing_manifest.get("row_count", -1)):
                versioned_relpath = next(
                    (
                        entry["path"]
                        for entry in existing_manifest.get("entries", [])
                        if str(entry.get("path", "")).endswith(".csv")
                        and "mnch_outcome_dictionary_icd10_v" in str(entry.get("path", ""))
                    ),
                    str(canonical_path.relative_to(root)),
                )
                return {
                    "version": existing_manifest.get("version"),
                    "canonical_path": canonical_path,
                    "versioned_path": root / versioned_relpath,
                    "manifest_path": manifest_path,
                    "summary_path": summary_path,
                    "dictionary": dictionary_frame,
                    "expansion_audit": existing_manifest.get("expansion_audit", []),
                }

    observed_codes = _collect_observed_codes(root, start_year, end_year)
    version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
    dictionary_rows: list[dict[str, object]] = []
    expansion_audit: list[dict[str, object]] = []

    for spec in OUTCOME_GROUP_SPECS:
        expanded_codes: set[str] = set()
        for literal in spec.rule_literals:
            literal_codes = _expand_rule_literal(observed_codes, literal)
            expanded_codes.update(literal_codes)
            expansion_audit.append(
                {
                    "outcome_group": spec.outcome_group,
                    "rule_literal": literal,
                    "expanded_code_count": len(literal_codes),
                }
            )
        for code in sorted(expanded_codes):
            dictionary_rows.append(
                {
                    "DICTIONARY_NAME": "mnch_outcome_dictionary_icd10",
                    "DICTIONARY_VERSION": version,
                    "OUTCOME_GROUP": spec.outcome_group,
                    "HEAT_RELATION": spec.heat_relation,
                    "GROUP_ROLE": spec.group_role,
                    "RULE_LITERAL": "|".join(spec.rule_literals),
                    "ICD10_CODE": code,
                    "ICD10_CATEGORY3": code[:3],
                    "ICD10_CODE_LENGTH": len(code),
                    "CANONICAL_SCOPE_SIM": spec.canonical_scope_sim,
                    "CANONICAL_SCOPE_SIH": spec.canonical_scope_sih,
                    "APPLIES_TO_SIM_UNDERLYING_BASIC_CAUSE": int(spec.outcome_group != "EXCESSIVE_NATURAL_HEAT_X30"),
                    "APPLIES_TO_SIM_ANY_MENTION": int(True),
                    "APPLIES_TO_SIH_PRINCIPAL_DIAGNOSIS": int(spec.outcome_group != "EXCESSIVE_NATURAL_HEAT_X30"),
                    "APPLIES_TO_SIH_ANY_DIAGNOSIS": int(spec.outcome_group != "EXCESSIVE_NATURAL_HEAT_X30"),
                    "APPLIES_TO_SIH_EXTERNAL_CAUSE_FIELDS": int(spec.outcome_group == "EXCESSIVE_NATURAL_HEAT_X30"),
                    "OBSERVED_IN_2020_2024_UNIVERSE": 1,
                }
            )

    dictionary_frame = pd.DataFrame(dictionary_rows).sort_values(
        ["OUTCOME_GROUP", "ICD10_CODE"],
        kind="mergesort",
    ).reset_index(drop=True)
    if dictionary_frame.empty:
        raise ValueError("Outcome dictionary expansion produced no rows")
    if dictionary_frame.duplicated(["OUTCOME_GROUP", "ICD10_CODE"]).any():
        raise ValueError("Outcome dictionary has duplicate OUTCOME_GROUP/ICD10_CODE rows")
    if dictionary_frame["ICD10_CODE"].str.contains(r"[\*\-]", regex=True).any():
        raise ValueError("Outcome dictionary contains unresolved wildcard or range syntax")

    output_dir = root / "artifacts" / "dictionaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    versioned_path = output_dir / f"mnch_outcome_dictionary_icd10_{version}.csv"
    dictionary_frame.to_csv(versioned_path, index=False, quoting=csv.QUOTE_MINIMAL)
    shutil.copy2(versioned_path, canonical_path)

    manifest = {
        "dictionary": "mnch_outcome_dictionary_icd10",
        "version": version,
        "generated_at": utc_now(),
        "source_years": list(range(start_year, end_year + 1)),
        "build_logic_version": OUTCOME_DICTIONARY_BUILD_LOGIC_VERSION,
        "row_count": int(len(dictionary_frame)),
        "group_count": int(dictionary_frame["OUTCOME_GROUP"].nunique()),
        "observed_icd10_code_count": int(len(observed_codes)),
        "expansion_audit": expansion_audit,
        "entries": [
            {
                "path": str(versioned_path.relative_to(root)),
                "size_bytes": versioned_path.stat().st_size,
                "sha256": sha256_file(versioned_path),
                "row_count": int(len(dictionary_frame)),
                "version": version,
            },
            {
                "path": str(canonical_path.relative_to(root)),
                "size_bytes": canonical_path.stat().st_size,
                "sha256": sha256_file(canonical_path),
                "row_count": int(len(dictionary_frame)),
                "version": version,
            },
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "# Unified ICD-10 Outcome Dictionary Summary",
        "",
        f"- Generated at: {manifest['generated_at']}",
        f"- Version: `{version}`",
        f"- Rows: {manifest['row_count']}",
        f"- Outcome groups: {manifest['group_count']}",
        f"- Observed ICD-10 codes scanned in 2020-2024 bronzes: {manifest['observed_icd10_code_count']}",
        f"- Canonical file: `{canonical_path.relative_to(root)}`",
        f"- Versioned file: `{versioned_path.relative_to(root)}`",
        "",
    ]
    for group_name, group_frame in dictionary_frame.groupby("OUTCOME_GROUP", sort=True):
        summary_lines.append(f"- {group_name}: {len(group_frame)} explicit codes")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "version": version,
        "canonical_path": canonical_path,
        "versioned_path": versioned_path,
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "dictionary": dictionary_frame,
        "expansion_audit": expansion_audit,
    }


def load_unified_outcome_dictionary(root: Path) -> pd.DataFrame:
    path = root / "artifacts" / "dictionaries" / "mnch_outcome_dictionary_icd10.csv"
    if not path.exists():
        raise FileNotFoundError(f"Outcome dictionary missing: {path}")
    return pd.read_csv(path, dtype={"ICD10_CODE": str})


def build_group_code_sets(dictionary: pd.DataFrame) -> dict[str, set[str]]:
    group_code_sets = {
        outcome_group: set(group_frame["ICD10_CODE"].astype(str).tolist())
        for outcome_group, group_frame in dictionary.groupby("OUTCOME_GROUP", sort=True)
    }
    for group_name in expected_group_names():
        group_code_sets.setdefault(group_name, set())
    return group_code_sets


def _sim_age_unit_and_value(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    cleaned = series.fillna("").astype(str).str.strip().str.zfill(3)
    unit_code = cleaned.str[:1]
    value = pd.to_numeric(cleaned.str[1:], errors="coerce")
    return unit_code, value


def derive_sim_age_flags(age_series: pd.Series) -> pd.DataFrame:
    unit_code, age_value = _sim_age_unit_and_value(age_series)
    under_28_days = unit_code.isin(["0", "1"]) | (unit_code.eq("2") & age_value.le(27))
    age_28_364_days = (unit_code.eq("2") & age_value.ge(28)) | unit_code.eq("3")
    age_1_4_years = unit_code.eq("4") & age_value.between(1, 4, inclusive="both")
    under_5 = under_28_days | age_28_364_days | (unit_code.eq("4") & age_value.between(0, 4, inclusive="both"))
    wra_15_49 = unit_code.eq("4") & age_value.between(15, 49, inclusive="both")
    wra_15_45 = unit_code.eq("4") & age_value.between(15, 45, inclusive="both")
    return pd.DataFrame(
        {
            "AGE_UNIT_CODE": unit_code,
            "AGE_VALUE": age_value.astype("Float64"),
            "AGE_GROUP_0_27D": under_28_days.astype("Int8"),
            "AGE_GROUP_28_364D": age_28_364_days.astype("Int8"),
            "AGE_GROUP_1_4Y": age_1_4_years.astype("Int8"),
            "AGE_GROUP_U5": under_5.astype("Int8"),
            "AGE_GROUP_WRA_15_49": wra_15_49.astype("Int8"),
            "AGE_GROUP_WRA_15_45_SENSITIVITY": wra_15_45.astype("Int8"),
        }
    )


def derive_sih_age_flags(code_series: pd.Series, age_series: pd.Series) -> pd.DataFrame:
    unit_code = code_series.fillna("").astype(str).str.strip()
    age_value = pd.to_numeric(age_series.fillna("").astype(str).str.strip(), errors="coerce")
    under_28_days = unit_code.isin(["0", "1"]) | (unit_code.eq("2") & age_value.le(27))
    age_28_364_days = (unit_code.eq("2") & age_value.ge(28)) | unit_code.eq("3")
    age_1_4_years = unit_code.eq("4") & age_value.between(1, 4, inclusive="both")
    under_5 = under_28_days | age_28_364_days | (unit_code.eq("4") & age_value.between(0, 4, inclusive="both"))
    wra_15_49 = unit_code.eq("4") & age_value.between(15, 49, inclusive="both")
    wra_15_45 = unit_code.eq("4") & age_value.between(15, 45, inclusive="both")
    return pd.DataFrame(
        {
            "AGE_UNIT_CODE": unit_code,
            "AGE_VALUE": age_value.astype("Float64"),
            "AGE_GROUP_0_27D": under_28_days.astype("Int8"),
            "AGE_GROUP_28_364D": age_28_364_days.astype("Int8"),
            "AGE_GROUP_1_4Y": age_1_4_years.astype("Int8"),
            "AGE_GROUP_U5": under_5.astype("Int8"),
            "AGE_GROUP_WRA_15_49": wra_15_49.astype("Int8"),
            "AGE_GROUP_WRA_15_45_SENSITIVITY": wra_15_45.astype("Int8"),
        }
    )


def dataset_entry(path: Path, root: Path, row_count: int, version: str | None = None) -> dict[str, object]:
    entry = {
        "path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": int(row_count),
    }
    if version is not None:
        entry["version"] = version
    return entry


def expected_group_names() -> tuple[str, ...]:
    return tuple(spec.outcome_group for spec in OUTCOME_GROUP_SPECS)
