from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from matplotlib import patches

from .config import load_framework_outcome_map, load_track_a_config
from .publication import scaffold_publication_workspace
from .registry import track_a_publication_root, track_a_root


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


STATUS_COLORS = {
    "exposure": "#d55e00",
    "outcome": "#0072b2",
    "estimable_adjustment": "#2c7f4f",
    "conceptual_only_not_estimable": "#9c6ade",
    "post_exposure_descendant_do_not_adjust": "#7a7a7a",
    "stratification_preferred_not_default_adjustment": "#f0a202",
    "policy_structure_only": "#5b7f95",
    "measurement_or_data_quality": "#5f9ea0",
}
STATUS_LABELS = {
    "exposure": "Exposure",
    "outcome": "Outcome",
    "estimable_adjustment": "Estimable adjustment node",
    "conceptual_only_not_estimable": "Conceptual-only, not estimable in Track A",
    "post_exposure_descendant_do_not_adjust": "Post-exposure descendant; do not adjust",
    "stratification_preferred_not_default_adjustment": "Stratification-preferred context node",
    "policy_structure_only": "Policy placeholder structure node",
    "measurement_or_data_quality": "Measurement or reporting node",
}
LEGEND_ORDER = [
    "exposure",
    "outcome",
    "estimable_adjustment",
    "conceptual_only_not_estimable",
    "stratification_preferred_not_default_adjustment",
    "post_exposure_descendant_do_not_adjust",
    "measurement_or_data_quality",
    "policy_structure_only",
]


def _dags_root(root: Path) -> Path:
    return track_a_root(root) / "dags"


def _load_readiness(root: Path) -> pd.DataFrame:
    readiness_path = track_a_root(root) / "freeze" / "analysis_ready_matrix.csv"
    if not readiness_path.exists():
        raise FileNotFoundError(
            f"Stage 5 requires the Track A readiness matrix at {readiness_path}. Run track-a-freeze first."
        )
    return pd.read_csv(readiness_path)


def _readiness_lookup(readiness: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(row["academic_variable_name"]): row
        for row in readiness.to_dict(orient="records")
    }


def _dag_specs() -> list[dict[str, Any]]:
    return [
        {
            "family_id": "acute_daily",
            "title": "Acute Daily Heat-Health DAG",
            "filename_stem": "dag_family_acute_daily",
            "scope_statement": (
                "This DAG represents short-term daily heat exposure analyses for acute neonatal and child outcomes "
                "within municipality-day panel structures."
            ),
            "purpose": (
                "To define the estimable adjustment architecture for acute fixed-effects screening and benchmark models "
                "in which exposure is measured at daily resolution."
            ),
            "outcome_family_bindings": ["acute_daily"],
            "primary_estimable_adjustment_set": [
                "municipality fixed effects",
                "calendar year indicators",
                "day-of-week indicators",
                "smooth or stratified seasonal calendar time",
                "relative humidity when the primary exposure family is maximum daily temperature",
            ],
            "conditional_adjustments": [
                "Exposure-family sensitivity models should remain separate; do not include maximum daily temperature, humidity, and heat index in the same primary specification.",
            ],
            "prohibited_adjustments": [
                "acute-care utilization outcomes or admission proxies that may lie on the exposure-to-outcome pathway",
                "post-exposure clinical deterioration markers",
                "derived exposure metrics from a different exposure family in the same primary model",
            ],
            "model_note": (
                "Municipal context is reserved for heterogeneity and stratification rather than default adjustment "
                "in the acute fixed-effects architecture."
            ),
            "narrative_paragraphs": [
                "The acute DAG is anchored in municipality fixed effects and short-term temporal controls because the principal confounding structure is time-varying within municipality-year rather than cross-sectional between municipalities.",
                "Relative humidity is included as an estimable adjustment node only when the main exposure family is maximum daily temperature. When humidity or heat index is the exposure family itself, those alternative exposure families should be estimated in separate primary models rather than entered jointly.",
                "Health-service pressure and reporting practice are retained as background or measurement nodes. They motivate robustness checks and quality-control restrictions, but they do not justify conditioning on downstream acute-care utilization proxies when mortality is the focal outcome.",
            ],
            "nodes": [
                {
                    "id": "calendar_time",
                    "label": "Calendar time\nand seasonality",
                    "role": "time_control",
                    "track_a_status": "estimable_adjustment",
                    "note": "Implemented with year indicators, day of week, and seasonal time structure.",
                    "position": [0.12, 0.82],
                },
                {
                    "id": "municipality_fe",
                    "label": "Municipality\nfixed effects",
                    "role": "baseline_context",
                    "track_a_status": "estimable_adjustment",
                    "note": "Captures time-invariant structural differences across municipalities.",
                    "position": [0.12, 0.42],
                },
                {
                    "id": "humidity",
                    "label": "Relative humidity\n(conditional)",
                    "role": "meteorology",
                    "track_a_status": "estimable_adjustment",
                    "note": "Included only when maximum daily temperature is the primary exposure family.",
                    "position": [0.38, 0.82],
                },
                {
                    "id": "heat_exposure",
                    "label": "Short-term heat\nexposure",
                    "role": "exposure",
                    "track_a_status": "exposure",
                    "note": "Represented by temperature, heatwave duration, or heat-index families in separate models.",
                    "position": [0.42, 0.56],
                },
                {
                    "id": "service_pressure",
                    "label": "Health-service\npressure",
                    "role": "system_context",
                    "track_a_status": "conceptual_only_not_estimable",
                    "note": "Conceptually relevant but not directly estimable in frozen gold as a daily confounder.",
                    "position": [0.68, 0.82],
                },
                {
                    "id": "reporting_practice",
                    "label": "Reporting and\nquality control",
                    "role": "measurement",
                    "track_a_status": "measurement_or_data_quality",
                    "note": "Handled through PASS filters, denominator flags, and source-aware sensitivity restrictions.",
                    "position": [0.70, 0.34],
                },
                {
                    "id": "acute_outcome",
                    "label": "Acute neonatal and\nchild outcomes",
                    "role": "outcome",
                    "track_a_status": "outcome",
                    "note": "Neonatal, infant, and under-five mortality are the headline acute outcomes.",
                    "position": [0.88, 0.56],
                },
                {
                    "id": "acute_proxy",
                    "label": "Acute-care proxy\n(descendant)",
                    "role": "post_exposure_use",
                    "track_a_status": "post_exposure_descendant_do_not_adjust",
                    "note": "May be screened descriptively but must not be used as a default adjustment variable.",
                    "position": [0.88, 0.18],
                },
            ],
            "edges": [
                {"source": "calendar_time", "target": "heat_exposure", "narrative": "Seasonality and calendar structure shape acute heat exposure."},
                {"source": "calendar_time", "target": "acute_outcome", "narrative": "Seasonality also shapes acute mortality risk."},
                {"source": "municipality_fe", "target": "heat_exposure", "narrative": "Persistent municipal structure influences ambient exposure levels."},
                {"source": "municipality_fe", "target": "acute_outcome", "narrative": "Persistent municipal structure influences baseline outcome risk."},
                {"source": "humidity", "target": "heat_exposure", "narrative": "Humidity modifies apparent thermal burden in temperature-based models."},
                {"source": "service_pressure", "target": "acute_outcome", "narrative": "System pressure can affect vulnerability and care access."},
                {"source": "reporting_practice", "target": "acute_outcome", "narrative": "Measurement quality influences observed event counts."},
                {"source": "heat_exposure", "target": "acute_outcome", "narrative": "The primary causal pathway of interest links acute heat to acute outcomes."},
                {"source": "heat_exposure", "target": "acute_proxy", "narrative": "Care-seeking proxies may be downstream of acute heat and should not be conditioned on."},
                {"source": "acute_outcome", "target": "acute_proxy", "narrative": "Downstream care use may also respond to evolving illness severity."},
            ],
        },
        {
            "family_id": "gestational_delivery_window_aggregate",
            "title": "Gestational Delivery-Window Aggregate DAG",
            "filename_stem": "dag_family_gestational_delivery_window_aggregate",
            "scope_statement": (
                "This DAG represents delivery-aligned and event-aligned observable exposure windows for perinatal "
                "and maternal operational outcomes in frozen Track A gold."
            ),
            "purpose": (
                "To separate estimable aggregated delivery-window models from the richer gestational event-level "
                "causal structures deferred to Track B."
            ),
            "outcome_family_bindings": ["perinatal_delivery_window_aggregate", "maternal_operational"],
            "primary_estimable_adjustment_set": [
                "municipality fixed effects",
                "calendar year indicators",
                "delivery-date or event-date seasonality controls",
                "day-of-week indicators when operationally relevant",
            ],
            "conditional_adjustments": [
                "Municipal context variables should be used for stratification or interaction analyses rather than default inclusion in the main fixed-effects adjustment set.",
                "Do not reinterpret delivery-aligned windows as third-trimester or whole-pregnancy causal exposures in Track A.",
            ],
            "prohibited_adjustments": [
                "obstetric interventions that occur after heat exposure and before recorded outcome ascertainment",
                "preterm-labor or admission events when they are mechanistic descendants or outcomes",
                "delivery-route or neonatal treatment variables that are downstream of heat-triggered clinical progression",
            ],
            "model_note": (
                "Maternal age, education, parity, multiple gestation, and conception season remain conceptually "
                "important but are not estimable as individual-level covariates in frozen Track A gold."
            ),
            "narrative_paragraphs": [
                "The gestational delivery-window DAG is intentionally conservative. It formalizes what Track A can estimate from municipality-day aggregates without pretending that delivery-aligned windows identify trimester-specific or gestational-week-specific effects.",
                "Municipality fixed effects and delivery-date timing controls remain estimable. In contrast, individual-level pregnancy characteristics are retained as conceptual nodes so that the supplement documents the causal structure honestly while acknowledging the current estimation boundary.",
                "Baseline municipal context belongs in heterogeneity and vulnerability analyses. It should not be treated as an automatic confounder in the main fixed-effects models because most of its variation is time-invariant at the municipal level.",
            ],
            "nodes": [
                {
                    "id": "delivery_time",
                    "label": "Delivery or event\ncalendar time",
                    "role": "time_control",
                    "track_a_status": "estimable_adjustment",
                    "note": "Implemented with event-date seasonality and calendar-year controls.",
                    "position": [0.10, 0.84],
                },
                {
                    "id": "municipality_fe",
                    "label": "Municipality\nfixed effects",
                    "role": "baseline_context",
                    "track_a_status": "estimable_adjustment",
                    "note": "Captures stable municipal structure and baseline risk.",
                    "position": [0.10, 0.48],
                },
                {
                    "id": "pregnancy_characteristics",
                    "label": "Maternal age,\nparity, education,\nmultiple gestation",
                    "role": "pregnancy_profile",
                    "track_a_status": "conceptual_only_not_estimable",
                    "note": "Conceptually important but not estimable from frozen municipality-day aggregates.",
                    "position": [0.34, 0.84],
                },
                {
                    "id": "conception_timing",
                    "label": "Conception season\nand baseline timing",
                    "role": "pregnancy_timing",
                    "track_a_status": "conceptual_only_not_estimable",
                    "note": "Deferred to Track B for finer gestational timing analysis.",
                    "position": [0.36, 0.48],
                },
                {
                    "id": "delivery_window_exposure",
                    "label": "Delivery-aligned\nheat exposure window",
                    "role": "exposure",
                    "track_a_status": "exposure",
                    "note": "Observed 7-, 14-, and 28-day windows before delivery or event.",
                    "position": [0.54, 0.66],
                },
                {
                    "id": "baseline_context",
                    "label": "Baseline municipal\ncontext",
                    "role": "context",
                    "track_a_status": "stratification_preferred_not_default_adjustment",
                    "note": "Urbanicity, inequality, human development, altitude, and coastal status are stratification-oriented.",
                    "position": [0.58, 0.22],
                },
                {
                    "id": "post_exposure_interventions",
                    "label": "Post-exposure\nobstetric intervention",
                    "role": "clinical_pathway",
                    "track_a_status": "post_exposure_descendant_do_not_adjust",
                    "note": "A descendant of the exposure-response process and therefore not a default adjustment variable.",
                    "position": [0.78, 0.82],
                },
                {
                    "id": "perinatal_outcomes",
                    "label": "Perinatal and\nmaternal operational\noutcomes",
                    "role": "outcome",
                    "track_a_status": "outcome",
                    "note": "Includes preterm birth, stillbirth, gestational days lost, and maternal operational admissions.",
                    "position": [0.88, 0.56],
                },
                {
                    "id": "reporting_practice",
                    "label": "Reporting and\ncase ascertainment",
                    "role": "measurement",
                    "track_a_status": "measurement_or_data_quality",
                    "note": "Managed with denominator flags, offset validation, and certified lineage checks.",
                    "position": [0.88, 0.20],
                },
            ],
            "edges": [
                {"source": "delivery_time", "target": "delivery_window_exposure", "narrative": "Delivery timing determines the calendar window over which Track A exposure summaries are computed."},
                {"source": "delivery_time", "target": "perinatal_outcomes", "narrative": "Delivery timing also captures seasonal and secular trends in outcomes."},
                {"source": "municipality_fe", "target": "delivery_window_exposure", "narrative": "Stable municipal characteristics shape the local heat profile."},
                {"source": "municipality_fe", "target": "perinatal_outcomes", "narrative": "Stable municipal characteristics influence baseline outcome risk."},
                {"source": "pregnancy_characteristics", "target": "perinatal_outcomes", "narrative": "Individual pregnancy characteristics influence perinatal and maternal risk."},
                {"source": "conception_timing", "target": "delivery_window_exposure", "narrative": "Conception timing influences where delivery falls in the annual heat cycle."},
                {"source": "conception_timing", "target": "perinatal_outcomes", "narrative": "Conception timing also influences the developmental context of pregnancy."},
                {"source": "baseline_context", "target": "perinatal_outcomes", "narrative": "Municipal context informs vulnerability and effect modification."},
                {"source": "delivery_window_exposure", "target": "post_exposure_interventions", "narrative": "Heat exposure may trigger downstream clinical response and intervention."},
                {"source": "delivery_window_exposure", "target": "perinatal_outcomes", "narrative": "This is the principal Track A association of interest for aggregated delivery-aligned windows."},
                {"source": "post_exposure_interventions", "target": "perinatal_outcomes", "narrative": "Post-exposure clinical action lies on the pathway to the observed outcome and must not be adjusted away."},
                {"source": "reporting_practice", "target": "perinatal_outcomes", "narrative": "Observed outcomes depend on ascertainment and recording quality."},
            ],
        },
        {
            "family_id": "policy_placeholder",
            "title": "Policy Placeholder DAG",
            "filename_stem": "dag_family_policy_placeholder",
            "scope_statement": (
                "This structural DAG records how an early-warning and adaptation policy layer could interact with "
                "heat exposure, implementation context, and maternal-child outcomes."
            ),
            "purpose": (
                "To document the intended policy causal architecture for future work without claiming that Track A "
                "currently estimates this policy effect."
            ),
            "outcome_family_bindings": [],
            "primary_estimable_adjustment_set": [],
            "conditional_adjustments": [
                "No Track A primary adjustment set is declared for this family because the policy DAG is structural only in the current cycle.",
            ],
            "prohibited_adjustments": [
                "No causal conditioning set should be inferred from this placeholder without a dedicated policy dataset and design.",
            ],
            "model_note": (
                "The policy DAG is a placeholder for future implementation research and should not be read as an "
                "estimable model specification in Track A."
            ),
            "narrative_paragraphs": [
                "The policy placeholder DAG exists so that the supplement can document the intended implementation-science trajectory of the framework without overstating what the present frozen gold release can estimate.",
                "Its function is architectural rather than inferential. It links early warning, adaptation, health-system preparedness, and implementation context to eventual maternal and child outcomes, but no policy estimand is fit in Track A.",
            ],
            "nodes": [
                {
                    "id": "implementation_context",
                    "label": "Implementation\ncontext",
                    "role": "context",
                    "track_a_status": "policy_structure_only",
                    "note": "Preparedness, governance, and municipal delivery constraints.",
                    "position": [0.10, 0.60],
                },
                {
                    "id": "policy_intervention",
                    "label": "Heat early-warning\nand adaptation policy",
                    "role": "policy",
                    "track_a_status": "policy_structure_only",
                    "note": "Future intervention node for implementation-focused analyses.",
                    "position": [0.34, 0.60],
                },
                {
                    "id": "health_system_preparedness",
                    "label": "Health-system\npreparedness",
                    "role": "system_response",
                    "track_a_status": "policy_structure_only",
                    "note": "Potential mediator of policy uptake and service readiness.",
                    "position": [0.56, 0.82],
                },
                {
                    "id": "heat_exposure",
                    "label": "Ambient heat\nexposure",
                    "role": "exposure",
                    "track_a_status": "exposure",
                    "note": "Operational heat metrics remain observable even though policy effects are not estimated here.",
                    "position": [0.56, 0.38],
                },
                {
                    "id": "targeting_and_behavior",
                    "label": "Targeting,\ncommunication,\nand behavior change",
                    "role": "implementation",
                    "track_a_status": "policy_structure_only",
                    "note": "Potential pathway from warning systems to exposure mitigation.",
                    "position": [0.74, 0.60],
                },
                {
                    "id": "maternal_child_outcomes",
                    "label": "Maternal and child\nhealth outcomes",
                    "role": "outcome",
                    "track_a_status": "outcome",
                    "note": "Ultimate outcomes of framework interest.",
                    "position": [0.90, 0.60],
                },
            ],
            "edges": [
                {"source": "implementation_context", "target": "policy_intervention", "narrative": "Context shapes whether adaptation policy can be deployed."},
                {"source": "implementation_context", "target": "health_system_preparedness", "narrative": "Context also shapes system readiness and capacity."},
                {"source": "policy_intervention", "target": "health_system_preparedness", "narrative": "Warning systems may improve preparedness and response planning."},
                {"source": "policy_intervention", "target": "targeting_and_behavior", "narrative": "Policy can influence communication, targeting, and household protective behavior."},
                {"source": "heat_exposure", "target": "maternal_child_outcomes", "narrative": "Ambient heat remains the primary environmental driver of interest."},
                {"source": "health_system_preparedness", "target": "maternal_child_outcomes", "narrative": "Preparedness may mitigate downstream health consequences."},
                {"source": "targeting_and_behavior", "target": "maternal_child_outcomes", "narrative": "Behavioral response may reduce effective exposure and vulnerability."},
            ],
        },
    ]


def _family_outcomes(
    family_spec: dict[str, Any],
    outcome_map: list[dict[str, Any]],
    readiness_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bindings = set(family_spec["outcome_family_bindings"])
    for item in outcome_map:
        if str(item["outcome_family"]) not in bindings:
            continue
        readiness = readiness_lookup.get(str(item["academic_variable_name"]), {})
        rows.append(
            {
                "prompt_label": str(item["prompt_label"]),
                "public_concept_name": str(item["academic_variable_name"]),
                "public_concept_label": str(readiness.get("public_concept_label", item["academic_variable_name"])),
                "outcome_family": str(item["outcome_family"]),
                "description": str(item["description"]),
                "stage1_classification": str(readiness.get("stage1_classification", "unknown")),
                "headline_eligible_flag": str(readiness.get("headline_eligible_flag", "no")),
                "benchmark_model_eligible_flag": str(readiness.get("benchmark_model_eligible_flag", "no")),
                "semantic_alignment": str(item.get("semantic_alignment", "")),
            }
        )
    return sorted(rows, key=lambda row: (row["outcome_family"], row["public_concept_label"]))


def _wrap_label(label: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(label, width=width)) if "\n" not in label else label


def _draw_dag_figure(family_spec: dict[str, Any], output_path: Path) -> None:
    figure = plt.figure(figsize=(14, 8))
    axis = figure.add_subplot(1, 1, 1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    node_bounds: dict[str, tuple[float, float, float, float]] = {}
    box_width = 0.18
    box_height = 0.11

    for node in family_spec["nodes"]:
        center_x, center_y = node["position"]
        left = center_x - (box_width / 2.0)
        bottom = center_y - (box_height / 2.0)
        patch = patches.FancyBboxPatch(
            (left, bottom),
            box_width,
            box_height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#243746",
            facecolor=STATUS_COLORS[node["track_a_status"]],
            alpha=0.88,
        )
        axis.add_patch(patch)
        axis.text(
            center_x,
            center_y,
            _wrap_label(str(node["label"]), width=16),
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )
        node_bounds[str(node["id"])] = (left, bottom, box_width, box_height)

    for edge in family_spec["edges"]:
        source_left, source_bottom, source_width, source_height = node_bounds[str(edge["source"])]
        target_left, target_bottom, target_width, target_height = node_bounds[str(edge["target"])]
        source_center = (source_left + source_width / 2.0, source_bottom + source_height / 2.0)
        target_center = (target_left + target_width / 2.0, target_bottom + target_height / 2.0)
        arrow = patches.FancyArrowPatch(
            source_center,
            target_center,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.4,
            color="#243746",
            connectionstyle="arc3,rad=0.05",
            shrinkA=30,
            shrinkB=30,
        )
        axis.add_patch(arrow)

    legend_handles = [
        patches.Patch(color=STATUS_COLORS[key], label=STATUS_LABELS[key])
        for key in LEGEND_ORDER
        if key in {node["track_a_status"] for node in family_spec["nodes"]}
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.02),
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    axis.set_title(family_spec["title"], fontsize=16, fontweight="bold")
    axis.text(0.5, 0.96, family_spec["scope_statement"], ha="center", va="top", fontsize=10)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _render_family_markdown(
    family_spec: dict[str, Any],
    covered_outcomes: list[dict[str, Any]],
    deferred_windows: list[str],
) -> str:
    node_rows = [
        "| Node ID | Label | Role | Track A status | Note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for node in family_spec["nodes"]:
        node_rows.append(
            f"| `{node['id']}` | {str(node['label']).replace(chr(10), '<br>')} | {node['role']} | "
            f"{node['track_a_status']} | {node['note']} |"
        )

    edge_rows = [
        "| Source | Target | Narrative |",
        "| --- | --- | --- |",
    ]
    for edge in family_spec["edges"]:
        edge_rows.append(f"| `{edge['source']}` | `{edge['target']}` | {edge['narrative']} |")

    outcome_rows = [
        "| Public concept | Outcome family | Stage 1 classification | Benchmark eligible | Description |",
        "| --- | --- | --- | --- | --- |",
    ]
    if covered_outcomes:
        for outcome in covered_outcomes:
            outcome_rows.append(
                f"| `{outcome['public_concept_name']}` | {outcome['outcome_family']} | "
                f"{outcome['stage1_classification']} | {outcome['benchmark_model_eligible_flag']} | "
                f"{outcome['description']} |"
            )
    else:
        outcome_rows.append("| None assigned in Stage 5 | n/a | n/a | n/a | This DAG is structural only in Track A. |")

    deferred_note = ""
    if family_spec["family_id"] == "gestational_delivery_window_aggregate":
        deferred_items = ", ".join(f"`{item}`" for item in deferred_windows)
        deferred_note = (
            "\n## Deferred Structures for Track B\n\n"
            f"The following gestational structures remain explicitly deferred beyond Track A: {deferred_items}.\n"
        )

    paragraphs = "\n\n".join(family_spec["narrative_paragraphs"])
    primary_set = "\n".join(f"- {item}" for item in family_spec["primary_estimable_adjustment_set"]) or "- None declared in Track A."
    conditional_set = "\n".join(f"- {item}" for item in family_spec["conditional_adjustments"])
    prohibited_set = "\n".join(f"- {item}" for item in family_spec["prohibited_adjustments"])

    return (
        f"# {family_spec['title']}\n\n"
        f"## Purpose\n\n{family_spec['purpose']}\n\n"
        f"## Scope Statement\n\n{family_spec['scope_statement']}\n\n"
        f"## Narrative Rationale\n\n{paragraphs}\n\n"
        "## Primary Estimable Adjustment Set\n\n"
        f"{primary_set}\n\n"
        "## Conditional Guidance\n\n"
        f"{conditional_set}\n\n"
        "## Prohibited Adjustments\n\n"
        f"{prohibited_set}\n\n"
        f"## Model Note\n\n{family_spec['model_note']}\n\n"
        "## Node Register\n\n"
        + "\n".join(node_rows)
        + "\n\n## Edge Register\n\n"
        + "\n".join(edge_rows)
        + "\n\n## Covered Track A Outcomes\n\n"
        + "\n".join(outcome_rows)
        + deferred_note
    )


def _render_adjustment_sets(
    family_specs: list[dict[str, Any]],
    family_outcome_map: dict[str, list[dict[str, Any]]],
    deferred_windows: list[str],
) -> str:
    sections: list[str] = ["# Track A Adjustment Sets\n"]
    for family_spec in family_specs:
        covered_outcomes = family_outcome_map[family_spec["family_id"]]
        covered_list = ", ".join(f"`{item['public_concept_name']}`" for item in covered_outcomes) or "No outcome set is assigned in Track A."
        primary_set = "\n".join(f"- {item}" for item in family_spec["primary_estimable_adjustment_set"]) or "- None declared."
        conditional_set = "\n".join(f"- {item}" for item in family_spec["conditional_adjustments"])
        prohibited_set = "\n".join(f"- {item}" for item in family_spec["prohibited_adjustments"])
        conceptual_only = [
            node["label"]
            for node in family_spec["nodes"]
            if node["track_a_status"] == "conceptual_only_not_estimable"
        ]
        contextual_only = "\n".join(f"- {item}" for item in conceptual_only) or "- None retained."
        sections.append(
            f"## {family_spec['title']}\n\n"
            f"Covered outcomes: {covered_list}\n\n"
            f"Primary estimable set:\n{primary_set}\n\n"
            f"Conceptual-only nodes retained for causal transparency:\n{contextual_only}\n\n"
            f"Conditional guidance:\n{conditional_set}\n\n"
            f"Prohibited descendants or forbidden adjustments:\n{prohibited_set}\n\n"
            f"Interpretive note: {family_spec['model_note']}\n"
        )
    sections.append(
        "## Track B Deferrals\n\n"
        "The following structures remain outside the estimable Track A DAG package and must be reserved for the Track B event-level gestational design:\n"
        + "\n".join(f"- `{item}`" for item in deferred_windows)
    )
    return "\n\n".join(sections) + "\n"


def _render_package_report(
    config: dict[str, Any],
    inventory: pd.DataFrame,
    family_outcome_map: dict[str, list[dict[str, Any]]],
) -> str:
    lines = [
        "# Track A DAG Package Report",
        "",
        f"- Analysis ID: `{config['analysis_id']}`",
        f"- DAG family count: {inventory.shape[0]}",
        f"- Figure count: {inventory['png_path'].notna().sum()}",
        "",
        "## Family Coverage",
        "",
    ]
    for record in inventory.to_dict(orient="records"):
        outcomes = family_outcome_map[record["family_id"]]
        class_counts = pd.Series([row["stage1_classification"] for row in outcomes]).value_counts().to_dict() if outcomes else {}
        lines.append(
            f"- `{record['family_id']}`: {len(outcomes)} covered outcomes; "
            f"classification summary = {class_counts if class_counts else {'structural_only': 1}}"
        )
    lines.extend(
        [
            "",
            "## Package Notes",
            "",
            "- Acute municipal context remains outside the default acute fixed-effects adjustment set.",
            "- Delivery-aligned windows are documented as observable aggregated windows rather than trimester-specific causal exposures.",
            "- The policy DAG is structural only and does not authorize policy estimation in Track A.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_publication_manifest(inventory: pd.DataFrame, root: Path) -> str:
    lines = [
        "# DAG Package Manifest",
        "",
        "This manifest records the Stage 5 DAG assets generated for the Track A supplement.",
        "",
    ]
    for record in inventory.to_dict(orient="records"):
        lines.append(f"## {record['title']}")
        lines.append("")
        lines.append(f"- JSON specification: `{record['json_path']}`")
        lines.append(f"- Narrative note: `{record['markdown_path']}`")
        lines.append(f"- Visual asset: `{record['png_path']}`")
        lines.append("")
    lines.append(
        f"- Shared adjustment sets note: `{(track_a_root(root) / 'dags' / 'adjustment_sets.md').relative_to(root)}`"
    )
    lines.append("")
    return "\n".join(lines)


def build_track_a_dag_package(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    config = load_track_a_config(root, config_path=config_path)
    outcome_map = load_framework_outcome_map(root)["track_a_outcomes"]
    readiness = _load_readiness(root)
    readiness_lookup = _readiness_lookup(readiness)
    deferred_windows = list(config.get("lag_definitions", {}).get("deferred_windows_for_track_b", []))

    scaffold_publication_workspace(root)

    dag_root = _dags_root(root)
    dag_root.mkdir(parents=True, exist_ok=True)
    publication_manifest_path = track_a_publication_root(root) / "manifests" / "dag_package_manifest.md"
    publication_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    inventory_rows: list[dict[str, Any]] = []
    family_outcome_map: dict[str, list[dict[str, Any]]] = {}

    for family_spec in _dag_specs():
        covered_outcomes = _family_outcomes(family_spec, outcome_map, readiness_lookup)
        family_outcome_map[family_spec["family_id"]] = covered_outcomes

        json_path = dag_root / f"{family_spec['filename_stem']}.json"
        markdown_path = dag_root / f"{family_spec['filename_stem']}.md"
        png_path = dag_root / f"{family_spec['filename_stem']}.png"

        json_payload = {
            "family_id": family_spec["family_id"],
            "title": family_spec["title"],
            "scope_statement": family_spec["scope_statement"],
            "purpose": family_spec["purpose"],
            "primary_estimable_adjustment_set": family_spec["primary_estimable_adjustment_set"],
            "conditional_adjustments": family_spec["conditional_adjustments"],
            "prohibited_adjustments": family_spec["prohibited_adjustments"],
            "model_note": family_spec["model_note"],
            "nodes": family_spec["nodes"],
            "edges": family_spec["edges"],
            "covered_track_a_outcomes": covered_outcomes,
            "deferred_windows_for_track_b": deferred_windows if family_spec["family_id"] == "gestational_delivery_window_aggregate" else [],
        }
        json_path.write_text(json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8")
        markdown_path.write_text(
            _render_family_markdown(family_spec, covered_outcomes, deferred_windows),
            encoding="utf-8",
        )
        _draw_dag_figure(family_spec, png_path)

        inventory_rows.append(
            {
                "family_id": family_spec["family_id"],
                "title": family_spec["title"],
                "json_path": str(json_path.relative_to(root)),
                "markdown_path": str(markdown_path.relative_to(root)),
                "png_path": str(png_path.relative_to(root)),
                "node_count": len(family_spec["nodes"]),
                "edge_count": len(family_spec["edges"]),
                "covered_outcome_count": len(covered_outcomes),
            }
        )

    inventory = pd.DataFrame(inventory_rows).sort_values("family_id")
    inventory_path = dag_root / "dag_inventory.csv"
    adjustment_sets_path = dag_root / "adjustment_sets.md"
    report_path = dag_root / "dag_package_report.md"

    inventory.to_csv(inventory_path, index=False)
    adjustment_sets_path.write_text(
        _render_adjustment_sets(_dag_specs(), family_outcome_map, deferred_windows),
        encoding="utf-8",
    )
    report_path.write_text(_render_package_report(config, inventory, family_outcome_map), encoding="utf-8")
    publication_manifest_path.write_text(_render_publication_manifest(inventory, root), encoding="utf-8")

    return {
        "status": "PASS",
        "stage": "stage5_dag_package_and_adjustment_architecture",
        "analysis_id": config["analysis_id"],
        "dag_family_count": int(inventory.shape[0]),
        "figure_count": int(inventory["png_path"].notna().sum()),
        "covered_outcome_count": int(sum(len(rows) for rows in family_outcome_map.values())),
        "outputs": {
            "dags_directory": str(dag_root.relative_to(root)),
            "dag_inventory": str(inventory_path.relative_to(root)),
            "adjustment_sets": str(adjustment_sets_path.relative_to(root)),
            "dag_package_report": str(report_path.relative_to(root)),
            "publication_manifest": str(publication_manifest_path.relative_to(root)),
        },
    }
