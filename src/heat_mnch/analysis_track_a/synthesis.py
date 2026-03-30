from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill

from .config import load_track_a_config
from .registry import track_a_root


COUNT_NULL_VALUE = 1.0
SEVERITY_NULL_VALUE = 0.0


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _synthesis_root(root: Path) -> Path:
    return track_a_root(root) / "synthesis"


def _load_required_csv(root: Path, relative_path: str, stage_name: str) -> pd.DataFrame:
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Stage 8 requires {stage_name} at {path}.")
    return pd.read_csv(path)


def _load_stage_inputs(root: Path) -> dict[str, pd.DataFrame]:
    return {
        "freeze": _load_required_csv(root, "artifacts/track_a/freeze/analysis_ready_matrix.csv", "Stage 1 readiness matrix"),
        "triage": _load_required_csv(root, "artifacts/track_a/screening/triage_signal_matrix.csv", "Stage 4 triage matrix"),
        "benchmark": _load_required_csv(root, "artifacts/track_a/models/benchmark_results.csv", "Stage 6 benchmark results"),
        "diagnostics": _load_required_csv(root, "artifacts/track_a/models/model_diagnostics.csv", "Stage 6 diagnostics"),
        "sensitivity": _load_required_csv(root, "artifacts/track_a/sensitivity/sensitivity_matrix.csv", "Stage 7 sensitivity matrix"),
        "heterogeneity": _load_required_csv(root, "artifacts/track_a/sensitivity/heterogeneity_results.csv", "Stage 7 heterogeneity results"),
    }


def _safe_int(value: Any) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return 0
    return int(numeric)


def _safe_float(value: Any, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return float(numeric)


def _effect_direction(estimate: float, estimate_scale: str) -> str:
    if not pd.notna(estimate):
        return "not_estimated"
    if str(estimate_scale).startswith("mean_difference"):
        if abs(float(estimate)) < 1e-12:
            return "null"
        return "positive" if float(estimate) > 0 else "negative"
    if abs(float(estimate) - COUNT_NULL_VALUE) < 1e-12:
        return "null"
    return "positive" if float(estimate) > COUNT_NULL_VALUE else "negative"


def _unique_freeze_frame(freeze: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "denominator_valid": "yes",
        "climate_enrichment_complete": "yes",
        "required_fields_present": "yes",
        "internal_validation_status": "pass",
        "external_validation_status": "pass",
        "freeze_gate_decision": "pass",
        "default_analysis_role": "",
    }
    frame = freeze.copy()
    for column, default in defaults.items():
        if column not in frame.columns:
            if column == "default_analysis_role" and "stage1_classification" in frame.columns:
                frame[column] = frame["stage1_classification"].astype(str)
            else:
                frame[column] = default
    columns = [
        "public_concept_name",
        "public_concept_label",
        "stage1_classification",
        "headline_eligible_flag",
        "benchmark_model_eligible_flag",
        "denominator_valid",
        "climate_enrichment_complete",
        "required_fields_present",
        "internal_validation_status",
        "external_validation_status",
        "freeze_gate_decision",
        "semantic_alignment",
        "default_analysis_role",
    ]
    return frame.loc[:, columns].sort_values(
        ["public_concept_name", "stage1_classification"],
        ascending=[True, True],
    ).drop_duplicates("public_concept_name")


def _benchmark_summary(benchmark: pd.DataFrame) -> pd.DataFrame:
    frame = benchmark.copy()
    frame["effect_direction"] = [
        _effect_direction(estimate, estimate_scale)
        for estimate, estimate_scale in zip(frame["estimate"], frame["estimate_scale"], strict=False)
    ]
    summary = frame.groupby("public_concept_name", as_index=False).agg(
        benchmark_row_count=("public_concept_name", "size"),
        benchmark_model_count=("model_label", "nunique"),
        finite_estimate_count=("estimate", lambda series: int(pd.to_numeric(series, errors="coerce").notna().sum())),
        non_null_estimate_count=("effect_direction", lambda series: int(series.isin(["positive", "negative"]).sum())),
        benchmark_null_row_count=("effect_direction", lambda series: int(series.eq("null").sum())),
        benchmark_direction_count=("effect_direction", lambda series: int(series.loc[series.isin(["positive", "negative"])].nunique())),
        benchmark_min_p_value=("p_value", "min"),
    )
    return summary


def _source_summary(sensitivity: pd.DataFrame) -> pd.DataFrame:
    frame = sensitivity.loc[
        sensitivity["sensitivity_domain"].isin(["source_restriction_no_fallback", "climate_source_stratum"])
    ].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "public_concept_name",
                "source_row_count",
                "source_sensitive_flag",
                "source_opposite_direction_flag",
                "source_max_abs_relative_change_pct",
            ]
        )
    frame["abs_relative_change_pct"] = pd.to_numeric(frame["relative_change_pct"], errors="coerce").abs()
    summary = frame.groupby("public_concept_name", as_index=False).agg(
        source_row_count=("public_concept_name", "size"),
        source_sensitive_flag=("source_dependence_flag", lambda series: "yes" if series.astype(str).eq("yes").any() else "no"),
        source_opposite_direction_flag=("direction_vs_reference", lambda series: "yes" if series.astype(str).eq("opposite_direction").any() else "no"),
        source_max_abs_relative_change_pct=("abs_relative_change_pct", "max"),
    )
    return summary


def _heterogeneity_summary(heterogeneity: pd.DataFrame) -> pd.DataFrame:
    if heterogeneity.empty:
        return pd.DataFrame(
            columns=[
                "public_concept_name",
                "heterogeneity_row_count",
                "heterogeneity_modifier_count",
                "heterogeneity_inconsistent_flag",
                "heterogeneity_consistent_row_count",
                "heterogeneity_inconsistent_row_count",
            ]
        )
    summary = heterogeneity.groupby("public_concept_name", as_index=False).agg(
        heterogeneity_row_count=("public_concept_name", "size"),
        heterogeneity_modifier_count=("modifier_name", "nunique"),
        heterogeneity_consistent_row_count=(
            "heterogeneity_signal_classification",
            lambda series: int(series.astype(str).eq("directionally_consistent").sum()),
        ),
        heterogeneity_inconsistent_row_count=(
            "heterogeneity_signal_classification",
            lambda series: int(series.astype(str).eq("directionally_inconsistent").sum()),
        ),
    )
    summary["heterogeneity_inconsistent_flag"] = summary["heterogeneity_inconsistent_row_count"].gt(0).map(
        {True: "yes", False: "no"}
    )
    return summary


def _operational_validity(row: pd.Series) -> tuple[int, str]:
    if (
        str(row.get("stage1_classification", "")) == "benchmark_ready"
        and str(row.get("denominator_valid", "")) == "yes"
        and str(row.get("climate_enrichment_complete", "")) == "yes"
        and str(row.get("required_fields_present", "")) == "yes"
        and str(row.get("internal_validation_status", "")) == "pass"
        and str(row.get("external_validation_status", "")) == "pass"
    ):
        return 3, "The concept passed the analytical freeze gate with valid denominators, complete climate enrichment, and internal and external validation."
    if str(row.get("stage1_classification", "")) == "screening_only":
        return 2, "The concept is present in the frozen gold stack with valid analytical support, but endpoint certification remains limited to screening use."
    if str(row.get("stage1_classification", "")) == "descriptive_only":
        return 1, "The concept is available operationally, but the frozen gold stack limits it to descriptive or proxy use."
    return 0, "The concept did not clear the analytical readiness gate for operational interpretation."


def _construct_validity(row: pd.Series) -> tuple[int, str]:
    triage_class = str(row.get("triage_priority_classification", ""))
    domain_count = _safe_int(row.get("signal_domain_count_p_le_0_10", 0))
    direction_consistency = _safe_float(row.get("signal_direction_consistency", 0.0))
    minimum_q_value = pd.to_numeric(row.get("minimum_q_value"), errors="coerce")
    if triage_class == "advance" and domain_count >= 2 and direction_consistency >= 0.75 and pd.notna(minimum_q_value):
        return 3, "Screening signals were coherent across domains and remained directionally consistent for the concept."
    if triage_class in {"advance", "advance_with_caveats"} and domain_count >= 1:
        return 2, "Screening identified interpretable signals, but coherence across domains or direction was incomplete."
    if _safe_int(row.get("screening_result_count", 0)) > 0:
        return 1, "The concept entered screening, but construct evidence was weak or inconsistent."
    return 0, "No meaningful screening evidence was available for construct validation."


def _robustness(row: pd.Series) -> tuple[int, str]:
    benchmark_rows = _safe_int(row.get("benchmark_row_count", 0))
    non_null_rows = _safe_int(row.get("non_null_estimate_count", 0))
    direction_count = _safe_int(row.get("benchmark_direction_count", 0))
    source_sensitive = str(row.get("source_sensitive_flag", "no")) == "yes"
    heterogeneity_inconsistent = str(row.get("heterogeneity_inconsistent_flag", "no")) == "yes"
    if benchmark_rows <= 0:
        return 0, "The concept did not advance into the benchmark ladder, so cross-model robustness could not be established."
    if non_null_rows <= 0:
        return 1, "The concept reached benchmark modeling, but benchmark effects were essentially null across model rows."
    if direction_count <= 1 and not source_sensitive and not heterogeneity_inconsistent:
        return 3, "Benchmark rows remained directionally aligned and Stage 7 did not identify material source dependence or heterogeneity reversals."
    if direction_count <= 1:
        return 2, "Benchmark rows were broadly aligned, but Stage 7 introduced material caveats from source restriction or heterogeneity."
    return 1, "Benchmark rows diverged across model families or later sensitivity analyses undermined robustness."


def _source_dependence(row: pd.Series) -> tuple[int, int, str]:
    source_row_count = _safe_int(row.get("source_row_count", 0))
    if source_row_count <= 0:
        return 1, 0, "Source dependence was not estimable because the concept did not enter the benchmark-supported sensitivity package."
    max_change = pd.to_numeric(row.get("source_max_abs_relative_change_pct"), errors="coerce")
    opposite = str(row.get("source_opposite_direction_flag", "no")) == "yes"
    sensitive = str(row.get("source_sensitive_flag", "no")) == "yes"
    if opposite or (pd.notna(max_change) and float(max_change) >= 50.0):
        return 0, -2, "Source-stratified estimates changed direction or changed by at least 50%, indicating severe source dependence."
    if sensitive or (pd.notna(max_change) and float(max_change) >= 10.0):
        return 1, -1, "Source-stratified estimates showed material instability, indicating meaningful dependence on climate source handling."
    if pd.notna(max_change) and float(max_change) >= 5.0:
        return 2, 0, "Source-restricted estimates moved modestly, but not enough to trigger a score downgrade."
    return 3, 0, "Source-restricted and source-stratified estimates remained stable."


def _implementation_value(row: pd.Series) -> tuple[int, str]:
    concept = str(row.get("public_concept_name", ""))
    stage1_class = str(row.get("stage1_classification", ""))
    benchmark_recommended = str(row.get("benchmark_stage_recommendation", "no")) == "yes"
    secondary_only_concepts = {
        "gestational_days_lost_below_37_weeks",
        "preterm_labor_admission_count",
        "gestational_diabetes_admission_count",
        "severe_hypertensive_disorder_of_pregnancy_count",
        "acute_care_proxy_count",
    }
    if benchmark_recommended and stage1_class == "benchmark_ready" and concept not in secondary_only_concepts:
        return 3, "The concept is operationally interpretable and aligned with the main Track A framework-validation narrative."
    if benchmark_recommended and stage1_class == "benchmark_ready":
        return 2, "The concept remains useful for implementation, but it is better positioned as a secondary or caveated indicator."
    if stage1_class in {"screening_only", "descriptive_only"}:
        return 1, "The concept has contextual or supplementary implementation value but should not anchor frontline interpretation."
    return 0, "The concept has limited implementation value in the current frozen gold and Track A design."


def _final_classification(
    *,
    final_score: int,
    stage1_classification: str,
    benchmark_stage_recommendation: str,
    operational_validity_score: int,
    construct_validity_score: int,
    robustness_score: int,
    source_dependence_rating: int,
) -> str:
    if (
        final_score >= 10
        and stage1_classification == "benchmark_ready"
        and benchmark_stage_recommendation == "yes"
        and operational_validity_score == 3
        and construct_validity_score >= 2
        and robustness_score >= 2
        and source_dependence_rating >= 2
    ):
        return "strong candidate"
    if final_score >= 7 and stage1_classification == "benchmark_ready":
        return "usable with caveats"
    if final_score >= 4:
        return "secondary only"
    return "not ready for frontline interpretation"


def _classification_rationale(row: pd.Series) -> str:
    classification = str(row["final_classification"])
    if classification == "strong candidate":
        return "The concept combined full operational readiness with coherent screening evidence, benchmark support, and no material source penalty."
    if classification == "usable with caveats":
        return "The concept remained analytically usable, but attenuation in the benchmark or Stage 7 caveats limited frontline confidence."
    if classification == "secondary only":
        return "The concept added contextual or sensitivity value, but the evidence package did not justify headline framework interpretation."
    return "The concept remained below the threshold for frontline framework interpretation in the current Track A evidence stack."


def _rubric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dimension": "Operational validity",
                "score_range": "0-3",
                "rule_summary": "Freeze readiness, denominator validity, climate enrichment, and validation status.",
            },
            {
                "dimension": "Construct validity",
                "score_range": "0-3",
                "rule_summary": "Screening-domain coherence, directional consistency, and signal coverage.",
            },
            {
                "dimension": "Robustness across model families",
                "score_range": "0-3",
                "rule_summary": "Stage 6 benchmark stability plus Stage 7 sensitivity and heterogeneity caveats.",
            },
            {
                "dimension": "Source dependence rating",
                "score_range": "0-3",
                "rule_summary": "Stage 7 source restriction and source-stratified instability; lower ratings trigger penalties.",
            },
            {
                "dimension": "Implementation value",
                "score_range": "0-3",
                "rule_summary": "Practical value for WMO/WHO-oriented framework interpretation.",
            },
            {
                "dimension": "Source dependence penalty",
                "score_range": "0 to -2",
                "rule_summary": "Applied to the four-dimension core score when source dependence is material.",
            },
        ]
    )


def _style_workbook(path: Path) -> None:
    workbook = load_workbook(path)
    header_fill = PatternFill(fill_type="solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
        for column_cells in worksheet.columns:
            column_letter = column_cells[0].column_letter
            max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
            worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 48)
    workbook.save(path)


def _render_synthesis_report(scorecard: pd.DataFrame) -> str:
    class_counts = scorecard["final_classification"].value_counts().to_dict()
    strong_candidates = scorecard.loc[
        scorecard["final_classification"].eq("strong candidate"),
        "public_concept_name",
    ].astype(str).tolist()
    usable_with_caveats = scorecard.loc[
        scorecard["final_classification"].eq("usable with caveats"),
        "public_concept_name",
    ].astype(str).tolist()
    fragile_concepts = scorecard.loc[
        scorecard["source_dependence_penalty"].lt(0) | scorecard["heterogeneity_inconsistent_flag"].eq("yes"),
        "public_concept_name",
    ].astype(str).tolist()
    lines = [
        "# Track A Framework Synthesis Report",
        "",
        "## Classification Summary",
        "",
        f"- Strong candidates: `{class_counts.get('strong candidate', 0)}`",
        f"- Usable with caveats: `{class_counts.get('usable with caveats', 0)}`",
        f"- Secondary only: `{class_counts.get('secondary only', 0)}`",
        f"- Not ready for frontline interpretation: `{class_counts.get('not ready for frontline interpretation', 0)}`",
        "",
        "## Headline Interpretation",
        "",
        "- Track A remains a framework-validation paper rather than a flagship single-outcome manuscript.",
        "- Operational readiness stayed high across the frozen gold stack, but Stage 6 attenuation and Stage 7 caveats limited the number of indicators that qualify as strong candidates.",
        "",
        "## Strong Candidates",
        "",
    ]
    lines.extend([f"- {concept}" for concept in strong_candidates] or ["- None"])
    lines.extend(
        [
            "",
            "## Usable With Caveats",
            "",
        ]
    )
    lines.extend([f"- {concept}" for concept in usable_with_caveats] or ["- None"])
    lines.extend(
        [
            "",
            "## Fragility Flags",
            "",
        ]
    )
    lines.extend([f"- {concept}" for concept in fragile_concepts] or ["- None"])
    lines.append("")
    return "\n".join(lines)


def build_track_a_synthesis(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    config = load_track_a_config(root, config_path=config_path)
    stage_inputs = _load_stage_inputs(root)

    freeze = _unique_freeze_frame(stage_inputs["freeze"])
    triage = stage_inputs["triage"].copy()
    benchmark_summary = _benchmark_summary(stage_inputs["benchmark"])
    source_summary = _source_summary(stage_inputs["sensitivity"])
    heterogeneity_summary = _heterogeneity_summary(stage_inputs["heterogeneity"])

    scorecard = freeze.merge(
        triage,
        on=["public_concept_name", "public_concept_label", "stage1_classification", "headline_eligible_flag", "benchmark_model_eligible_flag"],
        how="left",
    )
    scorecard = scorecard.merge(benchmark_summary, on="public_concept_name", how="left")
    scorecard = scorecard.merge(source_summary, on="public_concept_name", how="left")
    scorecard = scorecard.merge(heterogeneity_summary, on="public_concept_name", how="left")

    score_rows: list[dict[str, Any]] = []
    for row in scorecard.to_dict(orient="records"):
        series = pd.Series(row)
        operational_score, operational_rationale = _operational_validity(series)
        construct_score, construct_rationale = _construct_validity(series)
        robustness_score, robustness_rationale = _robustness(series)
        source_rating, source_penalty, source_rationale = _source_dependence(series)
        implementation_score, implementation_rationale = _implementation_value(series)
        core_score = operational_score + construct_score + robustness_score + implementation_score
        final_score = max(core_score + source_penalty, 0)
        final_classification = _final_classification(
            final_score=final_score,
            stage1_classification=str(series.get("stage1_classification", "")),
            benchmark_stage_recommendation=str(series.get("benchmark_stage_recommendation", "no")),
            operational_validity_score=operational_score,
            construct_validity_score=construct_score,
            robustness_score=robustness_score,
            source_dependence_rating=source_rating,
        )
        score_rows.append(
            {
                "public_concept_name": series["public_concept_name"],
                "public_concept_label": series["public_concept_label"],
                "stage1_classification": series.get("stage1_classification", ""),
                "headline_eligible_flag": series.get("headline_eligible_flag", ""),
                "benchmark_stage_recommendation": series.get("benchmark_stage_recommendation", "no"),
                "best_exposure_variable": series.get("best_exposure_variable", ""),
                "best_lag_or_window": series.get("best_lag_or_window", ""),
                "triage_priority_classification": series.get("triage_priority_classification", ""),
                "operational_validity_score": operational_score,
                "operational_validity_rationale": operational_rationale,
                "construct_validity_score": construct_score,
                "construct_validity_rationale": construct_rationale,
                "robustness_score": robustness_score,
                "robustness_rationale": robustness_rationale,
                "source_dependence_rating": source_rating,
                "source_dependence_penalty": source_penalty,
                "source_dependence_rationale": source_rationale,
                "implementation_value_score": implementation_score,
                "implementation_value_rationale": implementation_rationale,
                "core_score": core_score,
                "final_score": final_score,
                "final_classification": final_classification,
                "classification_rationale": _classification_rationale(
                    pd.Series(
                        {
                            "final_classification": final_classification,
                            "source_dependence_penalty": source_penalty,
                            "heterogeneity_inconsistent_flag": series.get("heterogeneity_inconsistent_flag", "no"),
                        }
                    )
                ),
                "source_sensitive_flag": series.get("source_sensitive_flag", "no"),
                "heterogeneity_inconsistent_flag": series.get("heterogeneity_inconsistent_flag", "no"),
                "benchmark_row_count": _safe_int(series.get("benchmark_row_count", 0)),
                "heterogeneity_row_count": _safe_int(series.get("heterogeneity_row_count", 0)),
                "source_row_count": _safe_int(series.get("source_row_count", 0)),
            }
        )

    scorecard_frame = pd.DataFrame(score_rows).sort_values(
        ["final_score", "public_concept_label"],
        ascending=[False, True],
    )
    evidence_trace = scorecard.merge(
        scorecard_frame.loc[
            :,
            [
                "public_concept_name",
                "core_score",
                "final_score",
                "final_classification",
                "source_dependence_penalty",
            ],
        ],
        on="public_concept_name",
        how="left",
    ).sort_values(["final_score", "public_concept_name"], ascending=[False, True])
    class_summary = (
        scorecard_frame.groupby("final_classification", as_index=False)
        .agg(concept_count=("public_concept_name", "size"))
        .sort_values("concept_count", ascending=False)
    )

    synthesis_root = _synthesis_root(root)
    synthesis_root.mkdir(parents=True, exist_ok=True)
    workbook_path = synthesis_root / "framework_validation_scorecard.xlsx"
    csv_path = synthesis_root / "framework_validation_scorecard.csv"
    evidence_path = synthesis_root / "framework_validation_evidence_trace.csv"
    report_path = synthesis_root / "stageA_synthesis_report.md"

    scorecard_frame.to_csv(csv_path, index=False)
    evidence_trace.to_csv(evidence_path, index=False)
    report_path.write_text(_render_synthesis_report(scorecard_frame), encoding="utf-8")

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        scorecard_frame.to_excel(writer, sheet_name="indicator_scorecard", index=False)
        class_summary.to_excel(writer, sheet_name="class_summary", index=False)
        evidence_trace.to_excel(writer, sheet_name="evidence_trace", index=False)
        _rubric_frame().to_excel(writer, sheet_name="rubric", index=False)
    _style_workbook(workbook_path)

    return {
        "status": "PASS",
        "stage": "stage8_framework_synthesis_and_indicator_scorecard",
        "analysis_id": str(config["analysis_id"]),
        "analysis_title": str(config["analysis_title"]),
        "indicator_count": int(len(scorecard_frame)),
        "classification_counts": class_summary.set_index("final_classification")["concept_count"].to_dict(),
        "outputs": {
            "framework_validation_scorecard_xlsx": _relative(root, workbook_path),
            "framework_validation_scorecard_csv": _relative(root, csv_path),
            "framework_validation_evidence_trace_csv": _relative(root, evidence_path),
            "stageA_synthesis_report": _relative(root, report_path),
        },
    }
