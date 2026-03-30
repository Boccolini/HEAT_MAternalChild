from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from .config import load_round_config, load_step6_burden_config, load_step6_translation_catalog
from .registry import (
    analysis_shared_marts_root,
    analysis_step1_root,
    analysis_step3_root,
    analysis_step4_root,
    analysis_step5_root,
    analysis_step6_root,
    ensure_step6_directory_structure,
    incremental_publication_root,
)
from .step2 import _baseline_valid_rows
from .step3 import SCREENING_PROFILE_SPECS, _pool_binary_summary, _q1_q5_summary
from .step5 import build_step5_heterogeneity_synthesis


STEP6_PUBLICATION_FILES = {
    "12_step6_absolute_burden.md": "# Step 6 Absolute Burden Translation\n\nDocument the pooled burden translation and exclusions here.\n",
    "13_step6_policy_translation.md": "# Step 6 Policy Translation Cards\n\nDocument headline and supplementary policy translation notes here.\n",
}
STEP6_NAME = "step6_absolute_burden_translation"
COUNT_ESTIMATE_SCALE = "rate_ratio_per_interquartile_range_increase"
SEVERITY_ESTIMATE_SCALE = "mean_difference_per_interquartile_range_increase"
SECONDARY_ONLY_TRACK = "secondary_parallel_only"
HEADLINE_ORDER_TRACK = "headline_order_candidate"
SUPPLEMENTARY_ONLY_TRACK = "supplementary_absolute_only"
GROUP_LABELS = {
    "maternal": "Maternal",
    "perinatal_nascimento": "Perinatal and Birth",
    "mortalidade_infantil": "Child Mortality",
    "hospitalizacao_infantil": "Child Hospitalization",
}
GROUP_SORT_ORDER = {
    "maternal": 0,
    "perinatal_nascimento": 1,
    "mortalidade_infantil": 2,
    "hospitalizacao_infantil": 3,
}
WORKBOOK_HEADER_FILL = PatternFill(fill_type="solid", fgColor="D9EAF7")
WORKBOOK_HEADLINE_FILL = PatternFill(fill_type="solid", fgColor="D9EAD3")
WORKBOOK_SECONDARY_FILL = PatternFill(fill_type="solid", fgColor="FCE5CD")
WORKBOOK_EXCLUSION_FILL = PatternFill(fill_type="solid", fgColor="F4CCCC")
UF_TO_MACROREGION = {
    "AC": "North",
    "AL": "Northeast",
    "AP": "North",
    "AM": "North",
    "BA": "Northeast",
    "CE": "Northeast",
    "DF": "Central-West",
    "ES": "Southeast",
    "GO": "Central-West",
    "MA": "Northeast",
    "MT": "Central-West",
    "MS": "Central-West",
    "MG": "Southeast",
    "PA": "North",
    "PB": "Northeast",
    "PR": "South",
    "PE": "Northeast",
    "PI": "Northeast",
    "RJ": "Southeast",
    "RN": "Northeast",
    "RS": "South",
    "RO": "North",
    "RR": "North",
    "SC": "South",
    "SP": "Southeast",
    "SE": "Northeast",
    "TO": "North",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _optional_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def _normalize_municipality_code(series: pd.Series) -> pd.Series:
    values: list[str] = []
    for value in series.tolist():
        text = "" if value is None or pd.isna(value) else str(value)
        digits = "".join(character for character in text if character.isdigit())
        values.append(digits.zfill(6)[:6])
    return pd.Series(values, index=series.index, dtype="string")


def _bool_like(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _scaffold_step6_publication_workspace(root: Path) -> dict[str, object]:
    workspace_root = incremental_publication_root(root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    created_files: list[str] = []
    existing_files: list[str] = []
    for relative_path, contents in STEP6_PUBLICATION_FILES.items():
        target = workspace_root / relative_path
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


def _effect_is_significant(estimate_scale: str, estimate: Any, p_value: Any) -> bool:
    try:
        estimate_numeric = float(estimate)
        p_value_numeric = float(p_value)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(estimate_numeric) and math.isfinite(p_value_numeric)):
        return False
    if estimate_scale == SEVERITY_ESTIMATE_SCALE:
        return bool(estimate_numeric > 0 and p_value_numeric < 0.05)
    return bool(estimate_numeric > 1 and p_value_numeric < 0.05)


def _direction_is_harmful(estimate_scale: str, estimate: Any) -> bool:
    try:
        estimate_numeric = float(estimate)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(estimate_numeric):
        return False
    if estimate_scale == SEVERITY_ESTIMATE_SCALE:
        return bool(estimate_numeric > 0)
    return bool(estimate_numeric > 1)


def _same_sign(value_a: Any, value_b: Any, tolerance: float = 1e-12) -> bool:
    try:
        numeric_a = float(value_a)
        numeric_b = float(value_b)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(numeric_a) and math.isfinite(numeric_b)):
        return False
    if abs(numeric_a) <= tolerance and abs(numeric_b) <= tolerance:
        return True
    return (numeric_a > tolerance and numeric_b > tolerance) or (numeric_a < -tolerance and numeric_b < -tolerance)


def _compute_signed_excess(
    *,
    estimate_scale: str,
    estimate: Any,
    lower_ci: Any,
    upper_ci: Any,
    exposed_outcome_sum: Any,
    exposed_denominator_sum: Any,
    reference_outcome_sum: Any,
    reference_denominator_sum: Any,
) -> dict[str, float]:
    try:
        estimate_numeric = float(estimate)
        lower_numeric = float(lower_ci)
        upper_numeric = float(upper_ci)
        exposed_outcome_numeric = float(exposed_outcome_sum)
        exposed_denominator_numeric = float(exposed_denominator_sum)
        reference_outcome_numeric = float(reference_outcome_sum)
        reference_denominator_numeric = float(reference_denominator_sum)
    except (TypeError, ValueError):
        return {
            "signed_excess": float("nan"),
            "signed_excess_lower": float("nan"),
            "signed_excess_upper": float("nan"),
            "raw_excesso": float("nan"),
            "same_sign_flag": False,
        }

    if estimate_scale == SEVERITY_ESTIMATE_SCALE:
        signed_excess = estimate_numeric * exposed_denominator_numeric
        signed_excess_lower = lower_numeric * exposed_denominator_numeric
        signed_excess_upper = upper_numeric * exposed_denominator_numeric
    else:
        if estimate_numeric == 0 or lower_numeric == 0 or upper_numeric == 0:
            signed_excess = float("nan")
            signed_excess_lower = float("nan")
            signed_excess_upper = float("nan")
        else:
            signed_excess = exposed_outcome_numeric * ((estimate_numeric - 1.0) / estimate_numeric)
            signed_excess_lower = exposed_outcome_numeric * ((lower_numeric - 1.0) / lower_numeric)
            signed_excess_upper = exposed_outcome_numeric * ((upper_numeric - 1.0) / upper_numeric)

    if reference_denominator_numeric > 0:
        raw_excesso = exposed_outcome_numeric - (
            (reference_outcome_numeric / reference_denominator_numeric) * exposed_denominator_numeric
        )
    else:
        raw_excesso = float("nan")
    return {
        "signed_excess": float(signed_excess),
        "signed_excess_lower": float(signed_excess_lower),
        "signed_excess_upper": float(signed_excess_upper),
        "raw_excesso": float(raw_excesso),
        "same_sign_flag": _same_sign(signed_excess, raw_excesso),
    }


def _load_step5_scorecard(root: Path) -> pd.DataFrame:
    return pd.read_csv(analysis_step5_root(root) / "step5_scorecard.csv")


def _load_step3_screening_results(root: Path) -> pd.DataFrame:
    return pd.read_csv(analysis_step3_root(root) / "step3_screening_results.csv")


def _load_step4_decisions(root: Path) -> pd.DataFrame:
    return pd.read_csv(analysis_step4_root(root) / "step4_promotion_decisions.csv")


def _load_step1_roster(root: Path) -> pd.DataFrame:
    return pd.read_csv(analysis_step1_root(root) / "analysis_unit_roster.csv")


def _load_step0_inventory(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "artifacts" / "analysis" / "step0" / "frozen_outcome_inventory.csv")


def _load_translation_catalog_frame(root: Path) -> pd.DataFrame:
    payload = load_step6_translation_catalog(root)
    return pd.DataFrame(payload.get("analysis_units", []))


def _burden_spec(root: Path) -> dict[str, Any]:
    payload = load_step6_burden_config(root)
    return dict(payload.get("step6_spec", {}))


def _load_municipality_lookup(root: Path) -> pd.DataFrame:
    path = root / "artifacts" / "reference" / "municipality_context" / "atlas_brasil_municipality_lookup.csv"
    lookup = pd.read_csv(path).copy()
    lookup["municipality_code"] = _normalize_municipality_code(lookup["MUNICIPALITY_CODE_IBGE7"])
    lookup["municipality_name"] = lookup["NM_MUNICIPIO"].astype(str)
    lookup["state_code"] = lookup["UF"].astype(str)
    lookup["macroregion"] = lookup["state_code"].map(UF_TO_MACROREGION).fillna("Unknown")
    return lookup.loc[:, ["municipality_code", "municipality_name", "state_code", "macroregion"]].drop_duplicates(
        subset=["municipality_code"], keep="first"
    )


def _inventory_public_label(linked_outcome_ids: Any, inventory_lookup: dict[str, str]) -> str:
    if linked_outcome_ids is None or pd.isna(linked_outcome_ids):
        return ""
    for token in str(linked_outcome_ids).split("|"):
        label = inventory_lookup.get(token.strip())
        if label:
            return label
    return ""


def _build_candidate_catalog(
    *,
    scorecard: pd.DataFrame,
    step1_roster: pd.DataFrame,
    step0_inventory: pd.DataFrame,
    translation_catalog: pd.DataFrame,
    step4_decisions: pd.DataFrame,
) -> pd.DataFrame:
    roster_subset = step1_roster.loc[
        :,
        [
            "analysis_unit_id",
            "linked_public_labels",
            "denominator_id",
            "denominator_column",
            "seasonality_profile_id",
            "mart_name",
        ],
    ].drop_duplicates(subset=["analysis_unit_id"], keep="first")
    inventory_lookup = dict(
        zip(step0_inventory["outcome_id"].astype(str), step0_inventory["public_label"].astype(str), strict=False)
    )
    catalog = scorecard.merge(roster_subset, on="analysis_unit_id", how="left", suffixes=("", "_step1"))
    catalog = catalog.merge(
        translation_catalog,
        on="analysis_unit_id",
        how="left",
        validate="one_to_one",
    )
    step4_subset = step4_decisions.loc[
        :,
        ["analysis_unit_id", "candidate_track", "selected_exposure_family_id", "preferred_lag_or_window"],
    ].rename(
        columns={
            "selected_exposure_family_id": "step4_selected_exposure_family_id",
            "preferred_lag_or_window": "step4_preferred_lag_or_window",
        }
    )
    catalog = catalog.merge(step4_subset, on=["analysis_unit_id", "candidate_track"], how="left")
    catalog["step4_alignment_flag"] = (
        catalog["selected_exposure_family_id"].astype(str).eq(catalog["step4_selected_exposure_family_id"].astype(str))
        & catalog["preferred_lag_or_window"].astype(str).eq(catalog["step4_preferred_lag_or_window"].astype(str))
    )
    catalog["public_label"] = catalog["linked_public_labels"].fillna("").astype(str)
    missing_public_label = catalog["public_label"].str.len().eq(0)
    catalog.loc[missing_public_label, "public_label"] = catalog.loc[
        missing_public_label, "linked_outcome_ids"
    ].map(lambda value: _inventory_public_label(value, inventory_lookup))
    catalog["group_label"] = catalog["group_id"].map(GROUP_LABELS).fillna(catalog.get("group_label", ""))
    catalog["headline_allowed_flag"] = catalog["headline_allowed_flag"].map(_bool_like)
    catalog["supplementary_absolute_allowed_flag"] = catalog["supplementary_absolute_allowed_flag"].map(_bool_like)
    catalog["headline_priority"] = pd.to_numeric(catalog["headline_priority"], errors="coerce")
    return catalog.sort_values(
        ["group_id", "headline_priority", "analysis_unit_id"],
        ignore_index=True,
    )


def _contrast_operator_for_outcome_family(outcome_family: str) -> str:
    if str(outcome_family) == "acute_daily":
        return "acute_p95_vs_reference"
    return "q5_vs_q1"


def _select_step3_anchor(
    step3_results: pd.DataFrame,
    *,
    analysis_unit_id: str,
    candidate_track: str,
    selected_exposure_family_id: str,
    preferred_lag_or_window: str,
    outcome_family: str,
) -> pd.DataFrame:
    subset = step3_results.loc[
        step3_results["analysis_unit_id"].astype(str).eq(analysis_unit_id)
        & step3_results["candidate_track"].astype(str).eq(candidate_track)
        & step3_results["exposure_family_id"].astype(str).eq(selected_exposure_family_id)
        & step3_results["lag_or_window"].astype(str).eq(preferred_lag_or_window)
    ].copy()
    if str(outcome_family) == "acute_daily":
        return subset.loc[subset["screening_domain"].astype(str).eq("acute_p95_vs_reference")]
    return subset.loc[
        subset["screening_method"].astype(str).eq("within_municipality_year_quintile_extremes")
    ]


def _attach_step3_anchor_metrics(candidate_catalog: pd.DataFrame, step3_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in candidate_catalog.to_dict(orient="records"):
        subset = _select_step3_anchor(
            step3_results,
            analysis_unit_id=str(record["analysis_unit_id"]),
            candidate_track=str(record["candidate_track"]),
            selected_exposure_family_id=str(record["selected_exposure_family_id"]),
            preferred_lag_or_window=str(record["preferred_lag_or_window"]),
            outcome_family=str(record["outcome_family"]),
        )
        row = dict(record)
        row["contrast_operator"] = _contrast_operator_for_outcome_family(str(record["outcome_family"]))
        row["step3_anchor_found_flag"] = len(subset) == 1
        row["step3_anchor_match_count"] = int(len(subset))
        if len(subset) == 1:
            anchor = subset.iloc[0]
            row["estimate"] = float(anchor["estimate"])
            row["lower_ci"] = float(anchor["lower_ci"])
            row["upper_ci"] = float(anchor["upper_ci"])
            row["p_value"] = float(anchor["p_value"])
            row["valid_strata_count"] = int(anchor["valid_strata_count"])
            row["exposed_outcome_sum"] = float(anchor["exposed_outcome_sum"])
            row["reference_outcome_sum"] = float(anchor["reference_outcome_sum"])
            row["exposed_denominator_sum"] = float(anchor["exposed_denominator_sum"])
            row["reference_denominator_sum"] = float(anchor["reference_denominator_sum"])
            row.update(
                _compute_signed_excess(
                    estimate_scale=str(record["estimate_scale"]),
                    estimate=row["estimate"],
                    lower_ci=row["lower_ci"],
                    upper_ci=row["upper_ci"],
                    exposed_outcome_sum=row["exposed_outcome_sum"],
                    exposed_denominator_sum=row["exposed_denominator_sum"],
                    reference_outcome_sum=row["reference_outcome_sum"],
                    reference_denominator_sum=row["reference_denominator_sum"],
                )
            )
            row["harmful_direction_flag"] = _direction_is_harmful(str(record["estimate_scale"]), row["estimate"])
            row["significant_flag"] = _effect_is_significant(
                str(record["estimate_scale"]),
                row["estimate"],
                row["p_value"],
            )
        else:
            row["estimate"] = float("nan")
            row["lower_ci"] = float("nan")
            row["upper_ci"] = float("nan")
            row["p_value"] = float("nan")
            row["valid_strata_count"] = 0
            row["exposed_outcome_sum"] = 0.0
            row["reference_outcome_sum"] = 0.0
            row["exposed_denominator_sum"] = 0.0
            row["reference_denominator_sum"] = 0.0
            row["signed_excess"] = float("nan")
            row["signed_excess_lower"] = float("nan")
            row["signed_excess_upper"] = float("nan")
            row["raw_excesso"] = float("nan")
            row["same_sign_flag"] = False
            row["harmful_direction_flag"] = False
            row["significant_flag"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_public_burden_gates(candidate_catalog: pd.DataFrame, burden_spec: dict[str, Any]) -> pd.DataFrame:
    thresholds = burden_spec.get("publication_thresholds", {})
    national_min_exposed = float(thresholds.get("national_min_exposed_outcome_sum", 50))
    national_min_strata = int(thresholds.get("national_min_valid_strata", 20))
    catalog = candidate_catalog.copy()
    catalog["public_burden_gate_pass"] = (
        catalog["step3_anchor_found_flag"].map(_bool_like)
        & catalog["harmful_direction_flag"].map(_bool_like)
        & catalog["significant_flag"].map(_bool_like)
        & catalog["same_sign_flag"].map(_bool_like)
        & pd.to_numeric(catalog["exposed_outcome_sum"], errors="coerce").fillna(0).ge(national_min_exposed)
        & pd.to_numeric(catalog["valid_strata_count"], errors="coerce").fillna(0).ge(national_min_strata)
        & ~catalog["translation_track"].astype(str).eq(SECONDARY_ONLY_TRACK)
    )
    catalog["gate_failure_reason"] = np.select(
        [
            ~catalog["step3_anchor_found_flag"].map(_bool_like),
            ~catalog["harmful_direction_flag"].map(_bool_like),
            ~catalog["significant_flag"].map(_bool_like),
            ~catalog["same_sign_flag"].map(_bool_like),
            pd.to_numeric(catalog["exposed_outcome_sum"], errors="coerce").fillna(0).lt(national_min_exposed),
            pd.to_numeric(catalog["valid_strata_count"], errors="coerce").fillna(0).lt(national_min_strata),
            catalog["translation_track"].astype(str).eq(SECONDARY_ONLY_TRACK),
        ],
        [
            "missing_step3_anchor",
            "not_harmful",
            "not_significant",
            "step3_qc_sign_mismatch",
            "national_exposed_outcomes_below_threshold",
            "national_valid_strata_below_threshold",
            "secondary_parallel_only_catalog_rule",
        ],
        default="passed",
    )
    return catalog


def _finalize_translation_tracks(candidate_catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog = candidate_catalog.copy()
    catalog["final_translation_stage"] = "secondary_parallel"
    catalog["headline_group_flag"] = False
    catalog["secondary_reason_final"] = catalog["secondary_reason"].fillna("").astype(str)
    group_rows: list[dict[str, Any]] = []
    for group_id, group in catalog.groupby("group_id", sort=False):
        ordered = group.sort_values(["headline_priority", "analysis_unit_id"], ignore_index=False)
        headline_candidates = ordered.loc[
            ordered["headline_allowed_flag"].map(_bool_like)
            & ordered["public_burden_gate_pass"].map(_bool_like)
        ]
        if not headline_candidates.empty:
            headline_index = headline_candidates.index[0]
            catalog.at[headline_index, "headline_group_flag"] = True
            catalog.at[headline_index, "final_translation_stage"] = "headline_absolute"
            headline_record = catalog.loc[headline_index].to_dict()
            group_rows.append(
                {
                    "group_id": group_id,
                    "group_label": headline_record.get("group_label", GROUP_LABELS.get(str(group_id), "")),
                    "headline_analysis_unit_id": headline_record["analysis_unit_id"],
                    "headline_public_label": headline_record.get("public_label", ""),
                    "selected_exposure_family_id": headline_record.get("selected_exposure_family_id", ""),
                    "selected_exposure_family_name": headline_record.get("selected_exposure_family_name", ""),
                    "preferred_lag_or_window": headline_record.get("preferred_lag_or_window", ""),
                    "estimate": headline_record.get("estimate", float("nan")),
                    "p_value": headline_record.get("p_value", float("nan")),
                    "signed_excess": headline_record.get("signed_excess", float("nan")),
                    "headline_selected_flag": True,
                    "headline_missing_flag": False,
                }
            )
        else:
            group_rows.append(
                {
                    "group_id": group_id,
                    "group_label": GROUP_LABELS.get(str(group_id), ""),
                    "headline_analysis_unit_id": "",
                    "headline_public_label": "",
                    "selected_exposure_family_id": "",
                    "selected_exposure_family_name": "",
                    "preferred_lag_or_window": "",
                    "estimate": float("nan"),
                    "p_value": float("nan"),
                    "signed_excess": float("nan"),
                    "headline_selected_flag": False,
                    "headline_missing_flag": True,
                }
            )

        supplementary_indices = ordered.index[
            ordered["public_burden_gate_pass"].map(_bool_like)
            & ordered["supplementary_absolute_allowed_flag"].map(_bool_like)
            & ~ordered.index.isin(headline_candidates.index[:1])
        ]
        for index in supplementary_indices:
            catalog.at[index, "final_translation_stage"] = "supplementary_absolute"

    secondary_mask = catalog["final_translation_stage"].astype(str).eq("secondary_parallel")
    fill_mask = secondary_mask & catalog["secondary_reason_final"].str.len().eq(0)
    catalog.loc[fill_mask, "secondary_reason_final"] = (
        catalog.loc[fill_mask, "gate_failure_reason"].replace(
            {
                "not_harmful": "failed_public_burden_gate",
                "not_significant": "failed_public_burden_gate",
                "step3_qc_sign_mismatch": "failed_public_burden_gate",
                "national_exposed_outcomes_below_threshold": "failed_public_burden_gate",
                "national_valid_strata_below_threshold": "failed_public_burden_gate",
                "missing_step3_anchor": "failed_public_burden_gate",
                "passed": "failed_public_burden_gate",
            }
        )
    )
    return catalog, pd.DataFrame(group_rows).sort_values("group_id", key=lambda series: series.map(GROUP_SORT_ORDER))


def _required_columns_by_mart(candidate_catalog: pd.DataFrame) -> dict[str, set[str]]:
    required: dict[str, set[str]] = {}
    base_columns = {
        "municipality_code",
        "observation_date",
        "calendar_year",
        "quality_control_status",
        "offset_valid_flag",
        "denominator_zero_flag",
        "denominator_missing_flag",
        "rate_not_computable_flag",
    }
    for mart_name, group in candidate_catalog.groupby("mart_name", sort=True):
        columns = set(base_columns)
        for record in group.to_dict(orient="records"):
            columns.add(str(record["analysis_unit_id"]))
            columns.add(str(record["selected_exposure_variable"]))
            columns.add(str(record["denominator_column"]))
        required[str(mart_name)] = columns
    return required


def _load_model_frames(
    root: Path,
    years: list[int],
    required_columns_by_mart: dict[str, set[str]],
    municipality_lookup: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for mart_name, columns in required_columns_by_mart.items():
        partitions: list[pd.DataFrame] = []
        selected_columns = sorted(columns)
        for year in years:
            path = analysis_shared_marts_root(root) / mart_name / f"YEAR={year}" / f"{mart_name}_{year}.parquet"
            partition = pd.read_parquet(path, columns=selected_columns)
            partition["municipality_code"] = _normalize_municipality_code(partition["municipality_code"])
            partition["observation_date"] = pd.to_datetime(partition["observation_date"], errors="coerce")
            partition["epidemiologic_week"] = partition["observation_date"].dt.isocalendar().week.astype("Int64")
            partitions.append(partition)
        frame = pd.concat(partitions, ignore_index=True)
        frames[mart_name] = frame.merge(municipality_lookup, on="municipality_code", how="left")
    return frames


def _tag_candidate_rows(frame: pd.DataFrame, record: dict[str, Any]) -> pd.DataFrame:
    exposure_column = str(record["selected_exposure_variable"])
    outcome_column = str(record["analysis_unit_id"])
    denominator_column = str(record["denominator_column"])
    valid = _baseline_valid_rows(frame, denominator_column, exposure_column).copy()
    if valid.empty:
        return pd.DataFrame()
    valid = valid.reset_index(drop=True)
    valid["outcome_value"] = pd.to_numeric(valid[outcome_column], errors="coerce").fillna(0.0)
    valid["denominator_value"] = pd.to_numeric(valid[denominator_column], errors="coerce").fillna(0.0)
    valid["stratum_id"] = pd.factorize(
        pd.MultiIndex.from_frame(valid.loc[:, ["municipality_code", "calendar_year"]]),
        sort=False,
    )[0]
    if str(record["outcome_family"]) == "acute_daily":
        quantiles = (
            valid.groupby("stratum_id", sort=False)[exposure_column]
            .quantile([0.40, 0.60, 0.95])
            .unstack()
            .rename(columns={0.40: "p40", 0.60: "p60", 0.95: "p95"})
        )
        valid = valid.join(quantiles, on="stratum_id")
        valid["screen_group"] = np.where(
            pd.to_numeric(valid[exposure_column], errors="coerce").ge(pd.to_numeric(valid["p95"], errors="coerce")),
            "exposed",
            np.where(
                pd.to_numeric(valid[exposure_column], errors="coerce").between(
                    pd.to_numeric(valid["p40"], errors="coerce"),
                    pd.to_numeric(valid["p60"], errors="coerce"),
                    inclusive="both",
                ),
                "reference",
                "other",
            ),
        )
        return valid.loc[
            valid["screen_group"].isin(["exposed", "reference"]),
            [
                "municipality_code",
                "municipality_name",
                "state_code",
                "macroregion",
                "calendar_year",
                "stratum_id",
                "screen_group",
                "outcome_value",
                "denominator_value",
            ],
        ].copy()

    quantiles = (
        valid.groupby("stratum_id", sort=False)[exposure_column]
        .quantile([0.20, 0.40, 0.60, 0.80])
        .unstack()
        .rename(columns={0.20: "q20", 0.40: "q40", 0.60: "q60", 0.80: "q80"})
    )
    valid = valid.join(quantiles, on="stratum_id")
    group_size = valid.groupby("stratum_id", sort=False)[exposure_column].transform("size")
    exposure_values = pd.to_numeric(valid[exposure_column], errors="coerce")
    valid["exposure_quintile"] = np.where(
        group_size.to_numpy() >= 5,
        1
        + (exposure_values.to_numpy() >= valid["q20"].to_numpy()).astype(int)
        + (exposure_values.to_numpy() >= valid["q40"].to_numpy()).astype(int)
        + (exposure_values.to_numpy() >= valid["q60"].to_numpy()).astype(int)
        + (exposure_values.to_numpy() >= valid["q80"].to_numpy()).astype(int),
        np.nan,
    )
    valid = valid.loc[valid["exposure_quintile"].notna()].copy()
    valid["exposure_quintile"] = valid["exposure_quintile"].astype(int)
    return valid.loc[
        :,
        [
            "municipality_code",
            "municipality_name",
            "state_code",
            "macroregion",
            "calendar_year",
            "stratum_id",
            "exposure_quintile",
            "outcome_value",
            "denominator_value",
        ],
    ].copy()


def _summary_from_tagged(tagged: pd.DataFrame, outcome_family: str) -> pd.DataFrame:
    if tagged.empty:
        return pd.DataFrame()
    if str(outcome_family) == "acute_daily":
        grouped = (
            tagged.groupby(["stratum_id", "screen_group"], as_index=False)
            .agg(
                outcome_sum=("outcome_value", "sum"),
                denominator_sum=("denominator_value", "sum"),
                row_count=("screen_group", "size"),
            )
        )
        pivot = grouped.pivot_table(
            index=["stratum_id"],
            columns="screen_group",
            values=["outcome_sum", "denominator_sum", "row_count"],
            aggfunc="first",
            fill_value=0,
        )
        pivot.columns = [f"{metric}_{group}" for metric, group in pivot.columns.to_flat_index()]
        summary = pivot.reset_index()
        for column in [
            "outcome_sum_exposed",
            "outcome_sum_reference",
            "denominator_sum_exposed",
            "denominator_sum_reference",
            "row_count_exposed",
            "row_count_reference",
        ]:
            if column not in summary.columns:
                summary[column] = 0.0
        return summary

    grouped = (
        tagged.groupby(["stratum_id", "exposure_quintile"], as_index=False)
        .agg(
            outcome_sum=("outcome_value", "sum"),
            denominator_sum=("denominator_value", "sum"),
            row_count=("exposure_quintile", "size"),
        )
    )
    return _q1_q5_summary(grouped)


def _build_burden_row(
    *,
    summary: pd.DataFrame,
    record: dict[str, Any],
    geography_level: str,
    geography_id: str,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    model_kind = "severity" if str(record["estimate_scale"]) == SEVERITY_ESTIMATE_SCALE else "count"
    effect = _pool_binary_summary(summary, model_kind=model_kind)
    metrics = _compute_signed_excess(
        estimate_scale=str(record["estimate_scale"]),
        estimate=effect["estimate"],
        lower_ci=effect["lower_ci"],
        upper_ci=effect["upper_ci"],
        exposed_outcome_sum=effect["exposed_outcome_sum"],
        exposed_denominator_sum=effect["exposed_denominator_sum"],
        reference_outcome_sum=effect["reference_outcome_sum"],
        reference_denominator_sum=effect["reference_denominator_sum"],
    )
    harmful_flag = _direction_is_harmful(str(record["estimate_scale"]), effect["estimate"])
    significant_flag = _effect_is_significant(str(record["estimate_scale"]), effect["estimate"], effect["p_value"])
    minimum_exposed = {
        "national": float(thresholds.get("national_min_exposed_outcome_sum", 50)),
        "macroregion": float(thresholds.get("macroregion_min_exposed_outcome_sum", 50)),
        "state": float(thresholds.get("state_min_exposed_outcome_sum", 100)),
    }.get(geography_level, float("inf"))
    threshold_met = float(effect["exposed_outcome_sum"]) >= minimum_exposed
    publishable_flag = bool(
        harmful_flag
        and significant_flag
        and metrics["same_sign_flag"]
        and threshold_met
    )
    return {
        "analysis_unit_id": record["analysis_unit_id"],
        "public_label": record.get("public_label", ""),
        "group_id": record.get("group_id", ""),
        "group_label": record.get("group_label", ""),
        "headline_group_flag": bool(record.get("headline_group_flag", False)),
        "selected_exposure_family_id": record.get("selected_exposure_family_id", ""),
        "selected_exposure_family_name": record.get("selected_exposure_family_name", ""),
        "preferred_lag_or_window": record.get("preferred_lag_or_window", ""),
        "contrast_operator": record.get("contrast_operator", ""),
        "estimate": float(effect["estimate"]),
        "lower_ci": float(effect["lower_ci"]),
        "upper_ci": float(effect["upper_ci"]),
        "p_value": float(effect["p_value"]),
        "valid_strata_count": int(effect["valid_strata_count"]),
        "exposed_outcome_sum": float(effect["exposed_outcome_sum"]),
        "reference_outcome_sum": float(effect["reference_outcome_sum"]),
        "exposed_denominator_sum": float(effect["exposed_denominator_sum"]),
        "reference_denominator_sum": float(effect["reference_denominator_sum"]),
        "signed_excess": float(metrics["signed_excess"]),
        "signed_excess_lower": float(metrics["signed_excess_lower"]),
        "signed_excess_upper": float(metrics["signed_excess_upper"]),
        "raw_excesso": float(metrics["raw_excesso"]),
        "same_sign_flag": bool(metrics["same_sign_flag"]),
        "harmful_direction_flag": bool(harmful_flag),
        "significant_flag": bool(significant_flag),
        "geography_level": geography_level,
        "geography_id": geography_id,
        "threshold_met_flag": bool(threshold_met),
        "publishable_flag": publishable_flag,
    }


def _build_geography_outputs(
    *,
    root: Path,
    round_years: list[int],
    public_candidates: pd.DataFrame,
    burden_spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if public_candidates.empty:
        return pd.DataFrame(), pd.DataFrame()

    lookup = _load_municipality_lookup(root)
    frames_by_mart = _load_model_frames(
        root=root,
        years=round_years,
        required_columns_by_mart=_required_columns_by_mart(public_candidates),
        municipality_lookup=lookup,
    )
    thresholds = burden_spec.get("publication_thresholds", {})
    geography_rows: list[dict[str, Any]] = []
    municipality_rows: list[dict[str, Any]] = []

    for record in public_candidates.to_dict(orient="records"):
        frame = frames_by_mart[str(record["mart_name"])]
        tagged = _tag_candidate_rows(frame, record)
        if tagged.empty:
            continue
        summary_national = _summary_from_tagged(tagged, str(record["outcome_family"]))
        geography_rows.append(
            _build_burden_row(
                summary=summary_national,
                record=record,
                geography_level="national",
                geography_id="BR",
                thresholds=thresholds,
            )
        )

        for macroregion, subset in tagged.groupby("macroregion", sort=True, dropna=True):
            summary = _summary_from_tagged(subset.reset_index(drop=True), str(record["outcome_family"]))
            geography_rows.append(
                _build_burden_row(
                    summary=summary,
                    record=record,
                    geography_level="macroregion",
                    geography_id=_optional_string(macroregion),
                    thresholds=thresholds,
                )
            )
        for state_code, subset in tagged.groupby("state_code", sort=True, dropna=True):
            summary = _summary_from_tagged(subset.reset_index(drop=True), str(record["outcome_family"]))
            geography_rows.append(
                _build_burden_row(
                    summary=summary,
                    record=record,
                    geography_level="state",
                    geography_id=_optional_string(state_code),
                    thresholds=thresholds,
                )
            )

        if bool(record.get("headline_group_flag", False)):
            if str(record["outcome_family"]) == "acute_daily":
                exposed_subset = tagged.loc[tagged["screen_group"].astype(str).eq("exposed")].copy()
            else:
                exposed_subset = tagged.loc[pd.to_numeric(tagged["exposure_quintile"], errors="coerce").eq(5)].copy()
            municipality_summary = (
                exposed_subset.groupby(["municipality_code", "municipality_name", "state_code", "macroregion"], as_index=False)
                .agg(observed_exposed_events=("outcome_value", "sum"))
            )
            national_exposed = float(municipality_summary["observed_exposed_events"].sum())
            municipality_summary["share_of_national_exposed_events"] = np.where(
                national_exposed > 0,
                municipality_summary["observed_exposed_events"] / national_exposed,
                0.0,
            )
            municipality_summary["share_of_group_headline_exposed_events"] = municipality_summary[
                "share_of_national_exposed_events"
            ]
            municipality_summary["group_id"] = record["group_id"]
            municipality_summary["group_label"] = record["group_label"]
            municipality_summary["headline_analysis_unit_id"] = record["analysis_unit_id"]
            municipality_summary["headline_public_label"] = record.get("public_label", "")
            municipality_rows.extend(municipality_summary.to_dict(orient="records"))

    geography_frame = pd.DataFrame(geography_rows)
    municipality_frame = pd.DataFrame(municipality_rows)
    return geography_frame, municipality_frame


def _build_secondary_parallel_translation(candidate_catalog: pd.DataFrame) -> pd.DataFrame:
    secondary = candidate_catalog.loc[
        candidate_catalog["final_translation_stage"].astype(str).eq("secondary_parallel")
    ].copy()
    if secondary.empty:
        return secondary
    recommended_public_use = np.select(
        [
            secondary["secondary_reason_final"].astype(str).eq("continuous_severity_non_event"),
            secondary["secondary_reason_final"].astype(str).eq("service_or_non_harmful_proxy"),
        ],
        [
            "continuous_internal_translation_only",
            "relative_only",
        ],
        default="relative_plus_observed_hot_volume",
    )
    secondary["recommended_public_use"] = recommended_public_use
    return secondary.loc[
        :,
        [
            "analysis_unit_id",
            "public_label",
            "group_id",
            "group_label",
            "selected_exposure_family_id",
            "selected_exposure_family_name",
            "preferred_lag_or_window",
            "contrast_operator",
            "estimate",
            "lower_ci",
            "upper_ci",
            "p_value",
            "exposed_outcome_sum",
            "reference_outcome_sum",
            "signed_excess",
            "secondary_reason_final",
            "recommended_public_use",
        ],
    ].rename(columns={"secondary_reason_final": "secondary_reason"})


def _build_exclusions_frame(candidate_catalog: pd.DataFrame) -> pd.DataFrame:
    excluded = candidate_catalog.loc[
        candidate_catalog["final_translation_stage"].astype(str).eq("secondary_parallel")
    ].copy()
    if excluded.empty:
        return excluded
    excluded["exclusion_reason"] = np.where(
        excluded["secondary_reason_final"].astype(str).str.len().gt(0),
        excluded["secondary_reason_final"],
        excluded["gate_failure_reason"],
    )
    return excluded.loc[
        :,
        [
            "analysis_unit_id",
            "public_label",
            "group_id",
            "group_label",
            "translation_track",
            "secondary_reason_final",
            "gate_failure_reason",
            "exclusion_reason",
        ],
    ]


def _build_decision_checklist(
    *,
    candidate_catalog: pd.DataFrame,
    group_headlines: pd.DataFrame,
    headline_frame: pd.DataFrame,
    secondary_frame: pd.DataFrame,
    gates: dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "checklist_item": "All Step 5 candidates were ingested into the Step 6 catalog.",
                "pass_flag": bool(gates["candidate_catalog_complete"]),
                "evidence": f"{len(candidate_catalog)} catalog rows were written.",
            },
            {
                "checklist_item": "Every candidate resolved to exactly one Step 3 contrast anchor.",
                "pass_flag": bool(gates["step3_anchor_resolved_for_all_candidates"]),
                "evidence": f"{int(candidate_catalog['step3_anchor_found_flag'].fillna(False).astype(bool).sum())}/{len(candidate_catalog)} candidates resolved to one anchor.",
            },
            {
                "checklist_item": "Step 4 exposure-family and lag selections aligned with the Step 5 scorecard.",
                "pass_flag": bool(gates["step4_alignment_verified"]),
                "evidence": f"{int(candidate_catalog['step4_alignment_flag'].fillna(False).astype(bool).sum())}/{len(candidate_catalog)} candidates aligned.",
            },
            {
                "checklist_item": "One headline burden was selected for each group.",
                "pass_flag": bool(gates["headline_selected_for_all_groups"]),
                "evidence": f"{int(group_headlines['headline_selected_flag'].fillna(False).astype(bool).sum())}/{len(group_headlines)} groups received a headline.",
            },
            {
                "checklist_item": "Absolute burden outputs were written for all approved headline and supplementary candidates.",
                "pass_flag": bool(gates["absolute_outputs_written"]),
                "evidence": f"{len(headline_frame)} headline rows were written; supplementary rows were written in a parallel table.",
            },
            {
                "checklist_item": "Secondary parallel translation outputs were written for excluded or non-defensible outcomes.",
                "pass_flag": bool(gates["secondary_outputs_written"]),
                "evidence": f"{len(secondary_frame)} secondary rows were written.",
            },
        ]
    )


def _write_sheet(workbook: Workbook, sheet_name: str, frame: pd.DataFrame, *, fill: PatternFill | None = None) -> None:
    sheet = workbook.create_sheet(title=sheet_name)
    if frame.empty:
        sheet.append(["No rows available."])
        return
    header_font = Font(bold=True)
    for row_index, row in enumerate([list(frame.columns)] + frame.astype(object).where(pd.notna(frame), "").values.tolist(), start=1):
        sheet.append(row)
        if row_index == 1:
            for cell in sheet[row_index]:
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center")
                cell.fill = WORKBOOK_HEADER_FILL
        elif fill is not None:
            for cell in sheet[row_index]:
                cell.fill = fill
    for column_cells in sheet.columns:
        length = max(len(str(cell.value or "")) for cell in column_cells)
        sheet.column_dimensions[column_cells[0].column_letter].width = min(max(length + 2, 12), 42)


def _write_workbook(
    *,
    workbook_path: Path,
    tabs: list[str],
    headlines: pd.DataFrame,
    supplementary: pd.DataFrame,
    secondary: pd.DataFrame,
    exclusions: pd.DataFrame,
    states: pd.DataFrame,
    macroregions: pd.DataFrame,
    municipality_support: pd.DataFrame,
) -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)
    frame_map = {
        "Headlines": (headlines, WORKBOOK_HEADLINE_FILL),
        "Supplementary": (supplementary, None),
        "Secondary": (secondary, WORKBOOK_SECONDARY_FILL),
        "Exclusions": (exclusions, WORKBOOK_EXCLUSION_FILL),
        "States": (states, None),
        "Macroregions": (macroregions, None),
        "MunicipalitySupport": (municipality_support, None),
    }
    for sheet_name in tabs:
        frame, fill = frame_map.get(sheet_name, (pd.DataFrame(), None))
        _write_sheet(workbook, sheet_name, frame, fill=fill)
    workbook.save(workbook_path)


def _render_policy_cards(group_headlines: pd.DataFrame, candidate_catalog: pd.DataFrame) -> str:
    lines = ["# Step 6 Policy Translation Cards", ""]
    for group_id in sorted(GROUP_LABELS, key=GROUP_SORT_ORDER.get):
        group_label = GROUP_LABELS[group_id]
        headline = group_headlines.loc[group_headlines["group_id"].astype(str).eq(group_id)]
        if headline.empty or not bool(headline["headline_selected_flag"].iloc[0]):
            lines.extend([f"## {group_label}", "", "- No eligible headline burden was identified in this round.", ""])
            continue
        headline_id = str(headline["headline_analysis_unit_id"].iloc[0])
        record = candidate_catalog.loc[candidate_catalog["analysis_unit_id"].astype(str).eq(headline_id)].iloc[0]
        caution = "secondary heterogeneity caution present" if bool(record.get("has_any_significant_reversal_flag", False)) else "no Step 5 significant reversal flag"
        lines.extend(
            [
                f"## {group_label}",
                "",
                f"- Headline: `{record['analysis_unit_id']}`",
                f"- Public label: `{record.get('public_label', '')}`",
                f"- Exposure family: `{record.get('selected_exposure_family_name', '')}`",
                f"- Lag or window: `{record.get('preferred_lag_or_window', '')}`",
                f"- Contrast operator: `{record.get('contrast_operator', '')}`",
                f"- National signed excess: `{float(record.get('signed_excess', float('nan'))):.3f}`",
                f"- Caution: `{caution}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _write_step6_publication_notes(
    root: Path,
    *,
    status: str,
    summary: dict[str, Any],
    outputs: dict[str, str],
) -> None:
    workspace_root = incremental_publication_root(root)
    (workspace_root / "12_step6_absolute_burden.md").write_text(
        (
            "# Step 6 Absolute Burden\n\n"
            f"- Status: `{status}`\n"
            f"- Candidates translated: `{summary['candidate_count']}`\n"
            f"- Headline groups filled: `{summary['headline_group_filled_count']}` of `{summary['headline_group_count']}`\n"
            f"- Headline burden table: `{outputs['headline_burden_estimates']}`\n"
            f"- Supplementary burden table: `{outputs['supplementary_burden_estimates']}`\n"
            f"- Technical report: `{outputs['step6_report']}`\n"
        ),
        encoding="utf-8",
    )
    (workspace_root / "13_step6_policy_translation.md").write_text(
        (
            "# Step 6 Policy Translation\n\n"
            f"- Status: `{status}`\n"
            f"- Secondary parallel rows: `{summary['secondary_parallel_count']}`\n"
            f"- Group headlines table: `{outputs['step6_group_headlines']}`\n"
            f"- Policy cards: `{outputs['step6_policy_translation_cards']}`\n"
        ),
        encoding="utf-8",
    )


def _render_step6_report(
    *,
    round_config: dict[str, Any],
    step5_report: dict[str, Any],
    gates: dict[str, Any],
    summary: dict[str, Any],
    outputs: dict[str, str],
    workspace_report: dict[str, Any],
    group_headlines: pd.DataFrame,
) -> str:
    lines = [
        f"# Step 6 Report: {round_config['analysis_title']}",
        "",
        f"- Generated at: `{_now_utc()}`",
        f"- Upstream Step 5 status: `{step5_report['status']}`",
        f"- Analysis ID: `{round_config['analysis_id']}`",
        "",
        "## Summary",
        "",
        f"- Candidate count: `{summary['candidate_count']}`",
        f"- Headline group count: `{summary['headline_group_count']}`",
        f"- Headline group filled count: `{summary['headline_group_filled_count']}`",
        f"- Headline absolute count: `{summary['headline_absolute_count']}`",
        f"- Supplementary absolute count: `{summary['supplementary_absolute_count']}`",
        f"- Secondary parallel count: `{summary['secondary_parallel_count']}`",
        "",
        "## Gates",
        "",
    ]
    for gate_name, gate_value in gates.items():
        if gate_name.endswith("_count"):
            lines.append(f"- {gate_name}: `{gate_value}`")
        else:
            lines.append(f"- {gate_name}: `{bool(gate_value)}`")
    lines.extend(["", "## Group Headlines", ""])
    if group_headlines.empty:
        lines.append("- No group headline rows were written.")
    else:
        for record in group_headlines.to_dict(orient="records"):
            lines.append(
                f"- group=`{record['group_id']}`, headline=`{record['headline_analysis_unit_id'] or 'MISSING'}`, selected=`{bool(record['headline_selected_flag'])}`"
            )
    lines.extend(["", "## Outputs", ""])
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
    return "\n".join(lines) + "\n"


def _resolve_upstream_step5_report(
    root: Path,
    *,
    config_path: Path | None,
    step5_report: dict[str, Any] | None,
) -> dict[str, Any]:
    if step5_report is not None:
        return step5_report
    if (analysis_step5_root(root) / "step5_scorecard.csv").exists():
        return {
            "status": "PASS",
            "step": "step5_heterogeneity_synthesis",
            "source": "existing_artifacts",
        }
    return build_step5_heterogeneity_synthesis(root, config_path=config_path)


def build_step6_absolute_burden(
    root: Path,
    config_path: Path | None = None,
    step5_report: dict[str, Any] | None = None,
) -> dict[str, object]:
    directories = ensure_step6_directory_structure(root)
    round_config = load_round_config(root, config_path=config_path)
    burden_spec = _burden_spec(root)
    workspace_report = _scaffold_step6_publication_workspace(root)
    upstream_step5_report = _resolve_upstream_step5_report(
        root,
        config_path=config_path,
        step5_report=step5_report,
    )

    candidate_catalog_path = analysis_step6_root(root) / "step6_candidate_catalog.csv"
    group_headlines_path = analysis_step6_root(root) / "step6_group_headlines.csv"
    headline_path = analysis_step6_root(root) / "headline_burden_estimates.csv"
    supplementary_path = analysis_step6_root(root) / "supplementary_burden_estimates.csv"
    secondary_path = analysis_step6_root(root) / "secondary_parallel_translation.csv"
    exclusions_path = analysis_step6_root(root) / "step6_exclusions.csv"
    state_gate_path = analysis_step6_root(root) / "state_burden_publication_gate.csv"
    municipality_detail_path = analysis_step6_root(root) / "municipality_contribution_detail.csv"
    checklist_path = analysis_step6_root(root) / "step6_decision_checklist.csv"
    policy_cards_path = analysis_step6_root(root) / "step6_policy_translation_cards.md"
    report_path = analysis_step6_root(root) / "step6_report.md"
    workbook_path = analysis_step6_root(root) / "step6_burden_translation.xlsx"

    outputs = {
        "step6_candidate_catalog": _relative(root, candidate_catalog_path),
        "step6_group_headlines": _relative(root, group_headlines_path),
        "headline_burden_estimates": _relative(root, headline_path),
        "supplementary_burden_estimates": _relative(root, supplementary_path),
        "secondary_parallel_translation": _relative(root, secondary_path),
        "step6_exclusions": _relative(root, exclusions_path),
        "state_burden_publication_gate": _relative(root, state_gate_path),
        "municipality_contribution_detail": _relative(root, municipality_detail_path),
        "step6_decision_checklist": _relative(root, checklist_path),
        "step6_policy_translation_cards": _relative(root, policy_cards_path),
        "step6_report": _relative(root, report_path),
        "step6_burden_translation_xlsx": _relative(root, workbook_path),
    }

    empty = pd.DataFrame()
    if upstream_step5_report["status"] != "PASS":
        gates = {
            "upstream_step5_pass": False,
            "candidate_catalog_complete": False,
            "step3_anchor_resolved_for_all_candidates": False,
            "step4_alignment_verified": False,
            "all_candidates_assigned_track": False,
            "headline_selected_for_all_groups": False,
            "absolute_outputs_written": False,
            "secondary_outputs_written": False,
            "headline_group_missing_count": 0,
        }
        for path in (
            candidate_catalog_path,
            group_headlines_path,
            headline_path,
            supplementary_path,
            secondary_path,
            exclusions_path,
            state_gate_path,
            municipality_detail_path,
            checklist_path,
        ):
            empty.to_csv(path, index=False)
        policy_cards_path.write_text("# Step 6 Policy Translation Cards\n\nNo rows available.\n", encoding="utf-8")
        _write_workbook(
            workbook_path=workbook_path,
            tabs=burden_spec.get("output_workbook_tabs", []),
            headlines=empty,
            supplementary=empty,
            secondary=empty,
            exclusions=empty,
            states=empty,
            macroregions=empty,
            municipality_support=empty,
        )
        report_path.write_text(
            _render_step6_report(
                round_config=round_config,
                step5_report=upstream_step5_report,
                gates=gates,
                summary={
                    "candidate_count": 0,
                    "headline_group_count": 0,
                    "headline_group_filled_count": 0,
                    "headline_absolute_count": 0,
                    "supplementary_absolute_count": 0,
                    "secondary_parallel_count": 0,
                },
                outputs=outputs,
                workspace_report=workspace_report,
                group_headlines=pd.DataFrame(),
            ),
            encoding="utf-8",
        )
        _write_step6_publication_notes(
            root,
            status="FAIL",
            summary={
                "candidate_count": 0,
                "headline_group_count": 0,
                "headline_group_filled_count": 0,
                "secondary_parallel_count": 0,
            },
            outputs=outputs,
        )
        return {
            "status": "FAIL",
            "step": STEP6_NAME,
            "analysis_id": round_config["analysis_id"],
            "years": list(round_config["years"]),
            "directories": directories,
            "upstream_step5_status": upstream_step5_report["status"],
            "gates": gates,
            "summary": {
                "candidate_count": 0,
                "headline_group_count": 0,
                "headline_group_filled_count": 0,
                "headline_absolute_count": 0,
                "supplementary_absolute_count": 0,
                "secondary_parallel_count": 0,
            },
            "outputs": outputs,
            "workspace_report": workspace_report,
        }

    step5_scorecard = _load_step5_scorecard(root)
    step3_results = _load_step3_screening_results(root)
    step4_decisions = _load_step4_decisions(root)
    step1_roster = _load_step1_roster(root)
    step0_inventory = _load_step0_inventory(root)
    translation_catalog = _load_translation_catalog_frame(root)

    candidate_catalog = _build_candidate_catalog(
        scorecard=step5_scorecard,
        step1_roster=step1_roster,
        step0_inventory=step0_inventory,
        translation_catalog=translation_catalog,
        step4_decisions=step4_decisions,
    )
    candidate_catalog = _attach_step3_anchor_metrics(candidate_catalog, step3_results)
    candidate_catalog = _apply_public_burden_gates(candidate_catalog, burden_spec)
    candidate_catalog, group_headlines = _finalize_translation_tracks(candidate_catalog)

    public_candidates = candidate_catalog.loc[
        candidate_catalog["final_translation_stage"].astype(str).isin({"headline_absolute", "supplementary_absolute"})
    ].copy()
    geography_frame, municipality_frame = _build_geography_outputs(
        root=root,
        round_years=[int(year) for year in round_config["years"]],
        public_candidates=public_candidates,
        burden_spec=burden_spec,
    )
    headline_frame = geography_frame.loc[
        geography_frame["headline_group_flag"].fillna(False).astype(bool)
    ].copy()
    supplementary_frame = geography_frame.loc[
        ~geography_frame["headline_group_flag"].fillna(False).astype(bool)
    ].copy()
    secondary_frame = _build_secondary_parallel_translation(candidate_catalog)
    exclusions_frame = _build_exclusions_frame(candidate_catalog)
    state_gate_frame = geography_frame.loc[
        geography_frame["geography_level"].astype(str).eq("state")
    ].copy()

    candidate_catalog.to_csv(candidate_catalog_path, index=False)
    group_headlines.to_csv(group_headlines_path, index=False)
    headline_frame.to_csv(headline_path, index=False)
    supplementary_frame.to_csv(supplementary_path, index=False)
    secondary_frame.to_csv(secondary_path, index=False)
    exclusions_frame.to_csv(exclusions_path, index=False)
    state_gate_frame.to_csv(state_gate_path, index=False)
    municipality_frame.to_csv(municipality_detail_path, index=False)

    checklist = _build_decision_checklist(
        candidate_catalog=candidate_catalog,
        group_headlines=group_headlines,
        headline_frame=headline_frame,
        secondary_frame=secondary_frame,
        gates={
            "candidate_catalog_complete": bool(len(candidate_catalog) == len(step5_scorecard) and not candidate_catalog.empty),
            "step3_anchor_resolved_for_all_candidates": bool(candidate_catalog["step3_anchor_found_flag"].fillna(False).astype(bool).all()),
            "step4_alignment_verified": bool(candidate_catalog["step4_alignment_flag"].fillna(False).astype(bool).all()),
            "headline_selected_for_all_groups": bool(group_headlines["headline_selected_flag"].fillna(False).astype(bool).all()),
            "absolute_outputs_written": bool(headline_path.exists()),
            "secondary_outputs_written": bool(secondary_path.exists()),
        },
    )
    checklist.to_csv(checklist_path, index=False)

    policy_cards_path.write_text(_render_policy_cards(group_headlines, candidate_catalog), encoding="utf-8")
    _write_workbook(
        workbook_path=workbook_path,
        tabs=list(burden_spec.get("output_workbook_tabs", [])),
        headlines=headline_frame,
        supplementary=supplementary_frame,
        secondary=secondary_frame,
        exclusions=exclusions_frame,
        states=state_gate_frame,
        macroregions=geography_frame.loc[geography_frame["geography_level"].astype(str).eq("macroregion")].copy(),
        municipality_support=municipality_frame,
    )

    gates = {
        "upstream_step5_pass": True,
        "candidate_catalog_complete": bool(len(candidate_catalog) == len(step5_scorecard) and not candidate_catalog.empty),
        "step3_anchor_resolved_for_all_candidates": bool(candidate_catalog["step3_anchor_found_flag"].fillna(False).astype(bool).all()),
        "step4_alignment_verified": bool(candidate_catalog["step4_alignment_flag"].fillna(False).astype(bool).all()),
        "all_candidates_assigned_track": bool(candidate_catalog["final_translation_stage"].astype(str).str.len().gt(0).all()),
        "headline_selected_for_all_groups": bool(group_headlines["headline_selected_flag"].fillna(False).astype(bool).all()),
        "absolute_outputs_written": bool(headline_path.exists() and supplementary_path.exists() and workbook_path.exists()),
        "secondary_outputs_written": bool(secondary_path.exists() and exclusions_path.exists()),
        "headline_group_missing_count": int(group_headlines["headline_missing_flag"].fillna(False).astype(bool).sum()),
    }
    summary = {
        "candidate_count": int(len(candidate_catalog)),
        "headline_group_count": int(len(group_headlines)),
        "headline_group_filled_count": int(group_headlines["headline_selected_flag"].fillna(False).astype(bool).sum()),
        "headline_absolute_count": int(candidate_catalog["final_translation_stage"].astype(str).eq("headline_absolute").sum()),
        "supplementary_absolute_count": int(candidate_catalog["final_translation_stage"].astype(str).eq("supplementary_absolute").sum()),
        "secondary_parallel_count": int(candidate_catalog["final_translation_stage"].astype(str).eq("secondary_parallel").sum()),
    }
    report_path.write_text(
        _render_step6_report(
            round_config=round_config,
            step5_report=upstream_step5_report,
            gates=gates,
            summary=summary,
            outputs=outputs,
            workspace_report=workspace_report,
            group_headlines=group_headlines,
        ),
        encoding="utf-8",
    )
    status = "PASS" if all(bool(value) for key, value in gates.items() if not key.endswith("_count")) else "FAIL"
    _write_step6_publication_notes(root, status=status, summary=summary, outputs=outputs)
    return {
        "status": status,
        "step": STEP6_NAME,
        "analysis_id": round_config["analysis_id"],
        "years": list(round_config["years"]),
        "directories": directories,
        "upstream_step5_status": upstream_step5_report["status"],
        "gates": gates,
        "summary": summary,
        "outputs": outputs,
        "workspace_report": workspace_report,
    }
