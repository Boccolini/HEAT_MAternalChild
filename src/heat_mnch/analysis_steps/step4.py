from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_round_config
from .registry import (
    analysis_step3_root,
    analysis_step4_root,
    ensure_step4_directory_structure,
    incremental_publication_root,
)
from .step3 import build_step3_threshold_discovery


STEP4_PUBLICATION_FILES = {
    "08_step4_lag_refinement.md": "# Step 4 Lag Refinement\n\nDocument preferred lags and windows here.\n",
    "09_step4_timing_narrative.md": "# Step 4 Timing Narrative\n\nDocument biological and design-constrained timing interpretation here.\n",
}

STEP4_NAME = "step4_lag_refinement"
NEXT_STEP_ROUTE = "step5_heterogeneity_synthesis"
FAMILY_ID_BY_RECOMMENDATION = {
    "tmax": "family_1_tmax_primary",
    "heat_index": "family_4_heat_index_parallel",
}
FAMILY_NAME_BY_RECOMMENDATION = {
    "tmax": "Maximum daily temperature",
    "heat_index": "Approximate heat index",
}
LAG_WINDOW_ORDER = {
    "lag0": 0,
    "lag0_1": 1,
    "lag0_3": 2,
    "lag0_7": 3,
    "0_6": 0,
    "0_13": 1,
    "0_27": 2,
}
SUPPORT_GRADE_RANK = {"D": 0, "C": 1, "B": 2, "A": 3}
ALLOWED_PROMOTION_REASONS = {
    "simplest_supported_window_retained",
    "fallback_to_simpler_window_due_to_no_clear_gain",
    "more_complex_window_shows_stronger_support_than_simpler_alternative",
    "more_complex_window_has_better_stability_than_simpler_alternative",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _scaffold_step4_publication_workspace(root: Path) -> dict[str, object]:
    workspace_root = incremental_publication_root(root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    created_files: list[str] = []
    existing_files: list[str] = []
    for relative_path, contents in STEP4_PUBLICATION_FILES.items():
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


def _load_step3_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    step3_root = analysis_step3_root(root)
    screening_results = pd.read_csv(step3_root / "step3_screening_results.csv")
    family_threshold_comparison = pd.read_csv(step3_root / "step3_family_threshold_comparison.csv")
    return screening_results, family_threshold_comparison


def _lag_order_value(value: Any) -> int:
    return int(LAG_WINDOW_ORDER.get(str(value), 99))


def _lag_support_grade(
    finite_result_count: int,
    screening_domain_count: int,
    sign_consistency_flag: bool,
    best_p_value: float,
) -> str:
    if finite_result_count >= 3 and screening_domain_count >= 2 and sign_consistency_flag and math.isfinite(best_p_value) and best_p_value < 0.05:
        return "A"
    if finite_result_count >= 2 and screening_domain_count >= 2 and sign_consistency_flag:
        return "B"
    if finite_result_count >= 1:
        return "C"
    return "D"


def _build_lag_comparison(
    screening_results: pd.DataFrame,
    family_threshold_comparison: pd.DataFrame,
) -> pd.DataFrame:
    if screening_results.empty or family_threshold_comparison.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for record in family_threshold_comparison.to_dict(orient="records"):
        recommendation = str(record["headline_family_recommendation"])
        selected_family_id = FAMILY_ID_BY_RECOMMENDATION.get(recommendation, "")
        selected_family_name = FAMILY_NAME_BY_RECOMMENDATION.get(recommendation, "")
        candidate_subset = screening_results.loc[
            screening_results["analysis_unit_id"].astype(str).eq(str(record["analysis_unit_id"]))
            & screening_results["candidate_track"].astype(str).eq(str(record["candidate_track"]))
        ].copy()
        outcome_family = (
            str(candidate_subset["outcome_family"].iloc[0])
            if not candidate_subset.empty and "outcome_family" in candidate_subset.columns
            else ""
        )
        subset = screening_results.loc[
            screening_results["analysis_unit_id"].astype(str).eq(str(record["analysis_unit_id"]))
            & screening_results["candidate_track"].astype(str).eq(str(record["candidate_track"]))
            & screening_results["exposure_family_id"].astype(str).eq(selected_family_id)
        ].copy() if selected_family_id else pd.DataFrame()

        if subset.empty:
            rows.append(
                {
                    "analysis_unit_id": str(record["analysis_unit_id"]),
                    "candidate_track": str(record["candidate_track"]),
                    "analysis_role": str(record.get("analysis_role", "")),
                    "outcome_family": outcome_family,
                    "headline_family_recommendation": recommendation,
                    "selected_exposure_family_id": selected_family_id,
                    "selected_exposure_family_name": selected_family_name,
                    "lag_or_window": "",
                    "lag_window_order": 99,
                    "finite_result_count": 0,
                    "screening_domain_count": 0,
                    "sign_consistency_flag": False,
                    "support_grade": "D",
                    "support_grade_rank": 0,
                    "best_screening_domain": "",
                    "best_screening_method": "",
                    "best_estimate": float("nan"),
                    "best_lower_ci": float("nan"),
                    "best_upper_ci": float("nan"),
                    "best_p_value": float("nan"),
                    "best_valid_strata_count": 0,
                    "seasonality_controls_inherited": "",
                    "seasonality_controls_fixed_flag": False,
                }
            )
            continue

        for lag_or_window, lag_group in subset.groupby("lag_or_window", sort=False):
            finite = lag_group.loc[
                lag_group["estimate"].apply(lambda value: math.isfinite(float(value)))
                & lag_group["valid_strata_count"].fillna(0).gt(0)
            ].copy()
            if not finite.empty:
                finite["effect_direction"] = finite.apply(
                    lambda row: "positive"
                    if (str(row["estimate_scale"]) == "mean_difference_per_interquartile_range_increase" and float(row["estimate"]) > 0)
                    or (str(row["estimate_scale"]) != "mean_difference_per_interquartile_range_increase" and float(row["estimate"]) > 1)
                    else "negative"
                    if (str(row["estimate_scale"]) == "mean_difference_per_interquartile_range_increase" and float(row["estimate"]) < 0)
                    or (str(row["estimate_scale"]) != "mean_difference_per_interquartile_range_increase" and float(row["estimate"]) < 1)
                    else "null",
                    axis=1,
                )
                directions = {value for value in finite["effect_direction"].astype(str).tolist() if value}
                sign_consistency_flag = len(directions) <= 1
                screening_domain_count = int(finite["screening_domain"].astype(str).nunique())
                finite_result_count = int(len(finite))
                finite = finite.sort_values(["p_value", "valid_strata_count"], ascending=[True, False], ignore_index=True)
                best = finite.iloc[0]
                best_p_value = float(best["p_value"])
            else:
                sign_consistency_flag = False
                screening_domain_count = 0
                finite_result_count = 0
                best = None
                best_p_value = float("nan")

            support_grade = _lag_support_grade(
                finite_result_count=finite_result_count,
                screening_domain_count=screening_domain_count,
                sign_consistency_flag=sign_consistency_flag,
                best_p_value=best_p_value,
            )
            seasonality_values = {
                value for value in lag_group["seasonality_controls_inherited"].astype(str).tolist() if value
            }
            rows.append(
                {
                    "analysis_unit_id": str(record["analysis_unit_id"]),
                    "candidate_track": str(record["candidate_track"]),
                    "analysis_role": str(record.get("analysis_role", "")),
                    "outcome_family": outcome_family,
                    "headline_family_recommendation": recommendation,
                    "selected_exposure_family_id": selected_family_id,
                    "selected_exposure_family_name": selected_family_name,
                    "lag_or_window": str(lag_or_window),
                    "lag_window_order": _lag_order_value(lag_or_window),
                    "finite_result_count": finite_result_count,
                    "screening_domain_count": screening_domain_count,
                    "sign_consistency_flag": bool(sign_consistency_flag),
                    "support_grade": support_grade,
                    "support_grade_rank": SUPPORT_GRADE_RANK[support_grade],
                    "best_screening_domain": str(best["screening_domain"]) if best is not None else "",
                    "best_screening_method": str(best["screening_method"]) if best is not None else "",
                    "best_estimate": float(best["estimate"]) if best is not None else float("nan"),
                    "best_lower_ci": float(best["lower_ci"]) if best is not None else float("nan"),
                    "best_upper_ci": float(best["upper_ci"]) if best is not None else float("nan"),
                    "best_p_value": best_p_value,
                    "best_valid_strata_count": int(best["valid_strata_count"]) if best is not None else 0,
                    "seasonality_controls_inherited": next(iter(seasonality_values), ""),
                    "seasonality_controls_fixed_flag": len(seasonality_values) <= 1 and len(seasonality_values) > 0,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["candidate_track", "analysis_role", "analysis_unit_id", "lag_window_order", "lag_or_window"],
        ignore_index=True,
    )


def _timing_narrative(outcome_family: str, lag_or_window: str) -> str:
    lag_or_window = str(lag_or_window)
    if not lag_or_window:
        return "No preferred lag or window could be selected from the Step 3 screening output."
    if outcome_family == "acute_daily":
        if lag_or_window == "lag0":
            return "Selected same-day acute timing. Interpretation should stay narrow and avoid claiming a longer persistence pattern."
        if lag_or_window == "lag0_1":
            return "Selected an acute 0 to 1 day window. Interpretation can support a short-lag acute response but not a long carry-over pattern."
        if lag_or_window == "lag0_3":
            return "Selected an acute 0 to 3 day window. Interpretation can support a short cumulative acute burden but should not be framed as a distributed lag surface."
        if lag_or_window == "lag0_7":
            return "Selected an acute 0 to 7 day window. Interpretation should emphasize short cumulative exposure rather than a mechanistic long-lag effect."
    if lag_or_window == "0_6":
        return "Selected the shortest event-aligned 0 to 6 day window. This is an observed delivery or event-aligned window, not a trimester or whole-pregnancy effect."
    if lag_or_window == "0_13":
        return "Selected the 0 to 13 day event-aligned window. Interpretation must remain event-aligned and should not be translated into trimester claims."
    if lag_or_window == "0_27":
        return "Selected the 0 to 27 day event-aligned window. Interpretation remains delivery or event aligned and should not be described as a full gestational causal window."
    return "Selected lag or window should be interpreted only within the observed exposure timing actually available in the aggregated data."


def _build_step4_decisions(
    lag_comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if lag_comparison.empty:
        return pd.DataFrame(), pd.DataFrame()
    decision_rows: list[dict[str, Any]] = []
    narrative_rows: list[dict[str, Any]] = []
    for keys, group in lag_comparison.groupby(
        [
            "analysis_unit_id",
            "candidate_track",
            "analysis_role",
            "outcome_family",
            "headline_family_recommendation",
            "selected_exposure_family_id",
            "selected_exposure_family_name",
        ],
        sort=True,
    ):
        sorted_group = group.sort_values(
            ["support_grade_rank", "best_p_value", "lag_window_order"],
            ascending=[False, True, True],
            ignore_index=True,
        )
        best = sorted_group.iloc[0]
        simplest = group.sort_values(["lag_window_order", "best_p_value"], ascending=[True, True], ignore_index=True).iloc[0]

        selected = best
        selected_reason = "strongest_supported_lag_or_window"
        decision_status = "promote_selected_window"

        if str(keys[4]) == "no_preference_yet" or not str(keys[5]):
            selected = simplest
            selected_reason = "headline_family_not_resolved_in_step3"
            decision_status = "defer_until_family_resolved"
        elif int(best["support_grade_rank"]) == 0:
            selected = simplest
            selected_reason = "no_stable_lag_or_window"
            decision_status = "fallback_to_simpler_window"
        elif int(best["lag_window_order"]) > int(simplest["lag_window_order"]):
            if int(best["support_grade_rank"]) > int(simplest["support_grade_rank"]):
                selected_reason = "more_complex_window_shows_stronger_support_than_simpler_alternative"
            elif (
                int(best["support_grade_rank"]) == int(simplest["support_grade_rank"])
                and bool(best["sign_consistency_flag"])
                and float(best["best_p_value"] if math.isfinite(float(best["best_p_value"])) else 1.0)
                < float(simplest["best_p_value"] if math.isfinite(float(simplest["best_p_value"])) else 1.0)
            ):
                selected_reason = "more_complex_window_has_better_stability_than_simpler_alternative"
            else:
                selected = simplest
                selected_reason = "fallback_to_simpler_window_due_to_no_clear_gain"
        else:
            selected_reason = "simplest_supported_window_retained"

        candidate_ready_for_step5_flag = (
            decision_status == "promote_selected_window"
            and bool(selected["seasonality_controls_fixed_flag"])
            and str(selected["lag_or_window"]) != ""
        )
        narrative_rows.append(
            {
                "analysis_unit_id": str(keys[0]),
                "candidate_track": str(keys[1]),
                "analysis_role": str(keys[2]),
                "outcome_family": str(keys[3]),
                "headline_family_recommendation": str(keys[4]),
                "selected_exposure_family_id": str(keys[5]),
                "selected_exposure_family_name": str(keys[6]),
                "preferred_lag_or_window": str(selected["lag_or_window"]),
                "timing_narrative": _timing_narrative(str(keys[3]), str(selected["lag_or_window"])),
                "timing_scope_note": _timing_narrative(str(keys[3]), str(selected["lag_or_window"])),
            }
        )
        decision_rows.append(
            {
                "analysis_unit_id": str(keys[0]),
                "candidate_track": str(keys[1]),
                "analysis_role": str(keys[2]),
                "outcome_family": str(keys[3]),
                "headline_family_recommendation": str(keys[4]),
                "selected_exposure_family_id": str(keys[5]),
                "selected_exposure_family_name": str(keys[6]),
                "preferred_lag_or_window": str(selected["lag_or_window"]),
                "preferred_support_grade": str(selected["support_grade"]),
                "preferred_support_grade_rank": int(selected["support_grade_rank"]),
                "preferred_best_screening_domain": str(selected["best_screening_domain"]),
                "preferred_best_p_value": float(selected["best_p_value"]),
                "seasonality_controls_fixed_flag": bool(selected["seasonality_controls_fixed_flag"]),
                "decision_status": decision_status,
                "selected_reason": selected_reason,
                "candidate_ready_for_step5_flag": bool(candidate_ready_for_step5_flag),
                "next_step_route": NEXT_STEP_ROUTE if candidate_ready_for_step5_flag else "stay_with_selected_or_simpler_window",
            }
        )
    return (
        pd.DataFrame(decision_rows).sort_values(
            ["candidate_track", "analysis_role", "analysis_unit_id"],
            ignore_index=True,
        ),
        pd.DataFrame(narrative_rows).sort_values(
            ["candidate_track", "analysis_role", "analysis_unit_id"],
            ignore_index=True,
        ),
    )


def _build_decision_checklist(
    family_threshold_comparison: pd.DataFrame,
    lag_comparison: pd.DataFrame,
    decisions: pd.DataFrame,
    gates: dict[str, bool],
) -> pd.DataFrame:
    candidate_count = int(len(family_threshold_comparison))
    selected_count = int(len(decisions))
    return pd.DataFrame(
        [
            {
                "checklist_item": "Every Step 3 candidate received a Step 4 lag or window decision.",
                "pass_flag": bool(gates["selected_lag_written_for_candidates"]),
                "evidence": f"{selected_count}/{candidate_count} Step 3 candidates received a Step 4 lag decision.",
            },
            {
                "checklist_item": "Headline exposure family was resolved before Step 4 promotion.",
                "pass_flag": bool(gates["headline_family_resolved_for_candidates"]),
                "evidence": f"{int(family_threshold_comparison['headline_family_recommendation'].astype(str).ne('no_preference_yet').sum())}/{candidate_count} Step 3 candidates had a resolved family recommendation.",
            },
            {
                "checklist_item": "Lag comparison kept seasonal controls fixed within each candidate.",
                "pass_flag": bool(gates["seasonal_controls_fixed_across_comparisons"]),
                "evidence": f"{int(lag_comparison['seasonality_controls_fixed_flag'].sum()) if not lag_comparison.empty else 0}/{int(len(lag_comparison))} lag rows preserved one inherited control set.",
            },
            {
                "checklist_item": "No unstable complex lag or window was promoted over a simpler alternative without evidence.",
                "pass_flag": bool(gates["unstable_complex_lags_not_promoted"]),
                "evidence": f"{int(decisions['candidate_ready_for_step5_flag'].sum()) if not decisions.empty else 0} candidates were promoted to Step 5 after lag refinement.",
            },
        ]
    )


def _render_step4_report(
    round_config: dict[str, Any],
    step3_report: dict[str, Any],
    gates: dict[str, bool],
    summary: dict[str, Any],
    outputs: dict[str, str],
    workspace_report: dict[str, object],
    decisions: pd.DataFrame,
) -> str:
    lines = [
        "# Step 4 Lag Refinement Report",
        "",
        f"- Generated at: `{_now_utc()}`",
        f"- Analysis ID: `{round_config['analysis_id']}`",
        f"- Analysis title: `{round_config['analysis_title']}`",
        f"- Years: `{', '.join(str(year) for year in round_config['years'])}`",
        f"- Upstream Step 3 status: `{step3_report['status']}`",
        "",
        "## Gates",
        "",
    ]
    for gate_name, passed in gates.items():
        lines.append(f"- {gate_name}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Step 3 candidates reviewed: `{summary['candidate_count']}`",
            f"- Lag-comparison rows: `{summary['lag_comparison_count']}`",
            f"- Candidates with selected lag/window: `{summary['selected_count']}`",
            f"- Candidates promoted to Step 5: `{summary['step5_ready_count']}`",
            f"- Candidates deferred after lag review: `{summary['deferred_count']}`",
        ]
    )
    if not decisions.empty:
        lines.extend(["", "## Step 4 Decisions", ""])
        for record in decisions.to_dict(orient="records"):
            lines.append(
                "- "
                f"{record['analysis_unit_id']}: family=`{record['headline_family_recommendation']}`, "
                f"lag_window=`{record['preferred_lag_or_window']}`, status=`{record['decision_status']}`, "
                f"reason=`{record['selected_reason']}`"
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


def _write_step4_publication_notes(
    root: Path,
    *,
    status: str,
    summary: dict[str, Any],
    outputs: dict[str, str],
) -> None:
    workspace_root = incremental_publication_root(root)
    (workspace_root / "08_step4_lag_refinement.md").write_text(
        (
            "# Step 4 Lag Refinement\n\n"
            f"- Status: `{status}`\n"
            f"- Candidates reviewed: `{summary['candidate_count']}`\n"
            f"- Candidates promoted to Step 5: `{summary['step5_ready_count']}`\n"
            f"- Lag comparison table: `{outputs['lag_comparison']}`\n"
            f"- Technical report: `{outputs['step4_report']}`\n"
        ),
        encoding="utf-8",
    )
    (workspace_root / "09_step4_timing_narrative.md").write_text(
        (
            "# Step 4 Timing Narrative\n\n"
            f"- Status: `{status}`\n"
            f"- Deferred candidates: `{summary['deferred_count']}`\n"
            f"- Timing narratives: `{outputs['timing_narratives']}`\n"
            f"- Promotion decisions: `{outputs['promotion_decisions']}`\n"
        ),
        encoding="utf-8",
    )


def build_step4_lag_refinement(
    root: Path,
    config_path: Path | None = None,
    step3_report: dict[str, Any] | None = None,
) -> dict[str, object]:
    directories = ensure_step4_directory_structure(root)
    round_config = load_round_config(root, config_path=config_path)
    workspace_report = _scaffold_step4_publication_workspace(root)
    upstream_step3_report = step3_report or build_step3_threshold_discovery(root, config_path=config_path)

    lag_comparison_path = analysis_step4_root(root) / "step4_lag_comparison.csv"
    timing_narratives_path = analysis_step4_root(root) / "step4_timing_narratives.csv"
    decisions_path = analysis_step4_root(root) / "step4_promotion_decisions.csv"
    checklist_path = analysis_step4_root(root) / "step4_decision_checklist.csv"
    report_path = analysis_step4_root(root) / "step4_report.md"

    outputs = {
        "lag_comparison": _relative(root, lag_comparison_path),
        "timing_narratives": _relative(root, timing_narratives_path),
        "promotion_decisions": _relative(root, decisions_path),
        "decision_checklist": _relative(root, checklist_path),
        "step4_report": _relative(root, report_path),
    }

    empty = pd.DataFrame()
    if upstream_step3_report["status"] != "PASS":
        gates = {
            "upstream_step3_pass": False,
            "headline_family_resolved_for_candidates": False,
            "lag_comparison_written": False,
            "selected_lag_written_for_candidates": False,
            "seasonal_controls_fixed_across_comparisons": False,
            "unstable_complex_lags_not_promoted": False,
        }
        for path in (lag_comparison_path, timing_narratives_path, decisions_path, checklist_path):
            empty.to_csv(path, index=False)
        report_path.write_text(
            _render_step4_report(
                round_config=round_config,
                step3_report=upstream_step3_report,
                gates=gates,
                summary={
                    "candidate_count": 0,
                    "lag_comparison_count": 0,
                    "selected_count": 0,
                    "step5_ready_count": 0,
                    "deferred_count": 0,
                },
                outputs=outputs,
                workspace_report=workspace_report,
                decisions=pd.DataFrame(),
            ),
            encoding="utf-8",
        )
        _write_step4_publication_notes(
            root,
            status="FAIL",
            summary={
                "candidate_count": 0,
                "step5_ready_count": 0,
                "deferred_count": 0,
            },
            outputs=outputs,
        )
        return {
            "status": "FAIL",
            "step": STEP4_NAME,
            "analysis_id": round_config["analysis_id"],
            "years": list(round_config["years"]),
            "directories": directories,
            "upstream_step3_status": upstream_step3_report["status"],
            "gates": gates,
            "summary": {
                "candidate_count": 0,
                "lag_comparison_count": 0,
                "selected_count": 0,
                "step5_ready_count": 0,
                "deferred_count": 0,
            },
            "outputs": outputs,
            "workspace_report": workspace_report,
        }

    screening_results, family_threshold_comparison = _load_step3_inputs(root)
    lag_comparison = _build_lag_comparison(screening_results, family_threshold_comparison)
    decisions, timing_narratives = _build_step4_decisions(lag_comparison)

    candidate_count = int(len(family_threshold_comparison))
    selected_count = int(len(decisions))
    step5_ready_count = int(decisions["candidate_ready_for_step5_flag"].eq(True).sum()) if not decisions.empty else 0
    deferred_count = int(selected_count - step5_ready_count)
    gates = {
        "upstream_step3_pass": True,
        "headline_family_resolved_for_candidates": bool(
            not family_threshold_comparison.empty
            and family_threshold_comparison["headline_family_recommendation"].astype(str).ne("no_preference_yet").all()
        ),
        "lag_comparison_written": not lag_comparison.empty,
        "selected_lag_written_for_candidates": selected_count == candidate_count if candidate_count > 0 else False,
        "seasonal_controls_fixed_across_comparisons": bool(
            not lag_comparison.empty
            and lag_comparison["seasonality_controls_fixed_flag"].astype(bool).all()
        ),
        "unstable_complex_lags_not_promoted": bool(
            decisions.empty
            or decisions.loc[
                decisions["decision_status"].astype(str).eq("promote_selected_window"),
                "selected_reason",
            ].astype(str).isin(ALLOWED_PROMOTION_REASONS).all()
        ),
    }
    status = "PASS" if all(gates.values()) else "FAIL"

    lag_comparison.to_csv(lag_comparison_path, index=False)
    timing_narratives.to_csv(timing_narratives_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    checklist = _build_decision_checklist(
        family_threshold_comparison=family_threshold_comparison,
        lag_comparison=lag_comparison,
        decisions=decisions,
        gates=gates,
    )
    checklist.to_csv(checklist_path, index=False)

    summary = {
        "candidate_count": candidate_count,
        "lag_comparison_count": int(len(lag_comparison)),
        "selected_count": selected_count,
        "step5_ready_count": step5_ready_count,
        "deferred_count": deferred_count,
    }
    report_path.write_text(
        _render_step4_report(
            round_config=round_config,
            step3_report=upstream_step3_report,
            gates=gates,
            summary=summary,
            outputs=outputs,
            workspace_report=workspace_report,
            decisions=decisions,
        ),
        encoding="utf-8",
    )
    _write_step4_publication_notes(root, status=status, summary=summary, outputs=outputs)

    return {
        "status": status,
        "step": STEP4_NAME,
        "analysis_id": round_config["analysis_id"],
        "years": list(round_config["years"]),
        "directories": directories,
        "upstream_step3_status": upstream_step3_report["status"],
        "gates": gates,
        "summary": summary,
        "outputs": outputs,
        "workspace_report": workspace_report,
    }
