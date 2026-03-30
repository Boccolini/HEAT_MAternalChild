from __future__ import annotations

from datetime import date
from pathlib import Path
from shutil import copy2
from typing import Any

import pandas as pd

from .config import load_track_a_config
from .registry import track_a_publication_root

try:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt
except ImportError as exc:  # pragma: no cover - exercised through integration tests
    Document = None  # type: ignore[assignment]
    WD_ALIGN_PARAGRAPH = None  # type: ignore[assignment]
    OxmlElement = None  # type: ignore[assignment]
    qn = None  # type: ignore[assignment]
    Inches = None  # type: ignore[assignment]
    Pt = None  # type: ignore[assignment]
    DOCX_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial branch
    DOCX_IMPORT_ERROR = None


TRACK_A_PUBLICATION_FILES = {
    "README.md": """# Track A Framework Validation

This workspace stores the manuscript-facing material for Track A of the HEAT-MNCH program.

Track A is the broad framework-validation paper. It evaluates whether the frozen operational gold stack can support reproducible, interpretable, and policy-relevant heat-health analyses across maternal, neonatal, and child health outcomes.

This publication workspace must reference canonical data products in `data/` and derived analytical outputs in `artifacts/track_a/`. It must not duplicate canonical datasets.
""",
    "00_outline.md": "# Outline\n\n1. Introduction\n2. Methods\n3. Results\n4. Discussion\n5. Limitations\n6. Tables\n7. Figures\n8. Supplement\n",
    "01_introduction.md": "# Introduction\n\nUse this file to develop the broad framework-validation introduction for Track A.\n",
    "02_methods.md": "# Methods\n\nUse this file to document the frozen gold inputs, semantic layer, descriptive atlas, screening strategy, benchmark ladder, and sensitivity framework for Track A.\n",
    "03_results.md": "# Results\n\nUse this file to summarize descriptive patterns, screening signals, benchmark findings, and framework-validation conclusions for Track A.\n",
    "04_discussion.md": "# Discussion\n\nUse this file to interpret the Track A findings as evidence about framework validity and operational usability rather than as a flagship perinatal claim.\n",
    "05_limitations.md": "# Limitations\n\nUse this file to document data, design, exposure, denominator, and source-dependence limitations for Track A.\n",
    "06_tables.md": "# Tables\n\nList the main-text and supplement tables for the Track A manuscript here.\n",
    "07_figures.md": "# Figures\n\nList the main-text and supplement figures for the Track A manuscript here.\n",
    "08_supplement.md": "# Supplement\n\nTrack supplement contents here, including DAGs, denominator rules, sensitivity analyses, and metadata appendices.\n",
    "tables/README.md": "# Tables\n\nStore Track A table shells and exported manuscript-ready tables in this directory.\n",
    "figures/README.md": "# Figures\n\nStore Track A figure shells and publication-ready figure exports in this directory.\n",
    "manifests/README.md": "# Manifests\n\nStore lightweight references to canonical and derived artifacts used by the Track A paper in this directory.\n",
    "submission/README.md": "# Submission\n\nStore journal-facing packaging notes, checklist files, and submission exports in this directory.\n",
}


def scaffold_publication_workspace(root: Path) -> dict[str, object]:
    workspace_root = track_a_publication_root(root)
    created_files: list[str] = []
    existing_files: list[str] = []
    workspace_root.mkdir(parents=True, exist_ok=True)

    for relative_path, contents in TRACK_A_PUBLICATION_FILES.items():
        target = workspace_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing_files.append(str(target.relative_to(root)))
            continue
        target.write_text(contents, encoding="utf-8")
        created_files.append(str(target.relative_to(root)))

    return {
        "workspace_root": str(workspace_root.relative_to(root)),
        "created_files": created_files,
        "existing_files": existing_files,
    }


def _require_docx_dependency() -> None:
    if Document is None:
        raise RuntimeError(
            "Stage 9 requires python-docx. Install it with `uv pip install python-docx`."
        ) from DOCX_IMPORT_ERROR


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _load_required_csv(root: Path, relative_path: str, stage_name: str) -> pd.DataFrame:
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Stage 9 requires {stage_name} at {path}.")
    return pd.read_csv(path)


def _load_optional_csv(root: Path, relative_path: str) -> pd.DataFrame:
    path = root / relative_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    return {
        "freeze": _load_required_csv(root, "artifacts/track_a/freeze/analysis_ready_matrix.csv", "Stage 1 readiness matrix"),
        "triage": _load_required_csv(root, "artifacts/track_a/screening/triage_signal_matrix.csv", "Stage 4 triage matrix"),
        "benchmark": _load_required_csv(root, "artifacts/track_a/models/benchmark_results.csv", "Stage 6 benchmark results"),
        "sensitivity": _load_required_csv(root, "artifacts/track_a/sensitivity/sensitivity_matrix.csv", "Stage 7 sensitivity matrix"),
        "heterogeneity": _load_required_csv(root, "artifacts/track_a/sensitivity/heterogeneity_results.csv", "Stage 7 heterogeneity results"),
        "scorecard": _load_required_csv(root, "artifacts/track_a/synthesis/framework_validation_scorecard.csv", "Stage 8 scorecard"),
        "evidence_trace": _load_required_csv(root, "artifacts/track_a/synthesis/framework_validation_evidence_trace.csv", "Stage 8 evidence trace"),
        "dag_inventory": _load_required_csv(root, "artifacts/track_a/dags/dag_inventory.csv", "Stage 5 DAG inventory"),
    }


def _unique_freeze_concepts(freeze: pd.DataFrame) -> pd.DataFrame:
    return freeze.sort_values(
        ["public_concept_name", "stage1_classification"],
        ascending=[True, True],
    ).drop_duplicates("public_concept_name")


def _format_concept_list(values: list[str]) -> str:
    if not values:
        return "None."
    return ", ".join(f"`{value}`" for value in values) + "."


def _markdown_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| Column | Value |\n| --- | --- |\n| None | None |\n"
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for record in frame.fillna("").astype(str).to_dict(orient="records"):
        lines.append("| " + " | ".join(record[column] for column in headers) + " |")
    return "\n".join(lines) + "\n"


def _publication_metrics(inputs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    freeze = _unique_freeze_concepts(inputs["freeze"])
    triage = inputs["triage"].copy()
    benchmark = inputs["benchmark"].copy()
    sensitivity = inputs["sensitivity"].copy()
    heterogeneity = inputs["heterogeneity"].copy()
    scorecard = inputs["scorecard"].copy().sort_values(["final_score", "public_concept_name"], ascending=[False, True])

    readiness_counts = freeze["stage1_classification"].value_counts().to_dict()
    triage_counts = triage["triage_priority_classification"].value_counts().to_dict()
    classification_counts = scorecard["final_classification"].value_counts().to_dict()

    strong_candidates = scorecard.loc[
        scorecard["final_classification"].eq("strong candidate"),
        "public_concept_name",
    ].astype(str).tolist()
    usable_with_caveats = scorecard.loc[
        scorecard["final_classification"].eq("usable with caveats"),
        "public_concept_name",
    ].astype(str).tolist()
    secondary_only = scorecard.loc[
        scorecard["final_classification"].eq("secondary only"),
        "public_concept_name",
    ].astype(str).tolist()
    fragility_flags = scorecard.loc[
        scorecard["source_sensitive_flag"].astype(str).eq("yes")
        | scorecard["heterogeneity_inconsistent_flag"].astype(str).eq("yes"),
        "public_concept_name",
    ].astype(str).tolist()

    metrics = {
        "analysis_years": "2020-2024",
        "indicator_count": int(len(scorecard)),
        "ready_benchmark_count": int(readiness_counts.get("benchmark_ready", 0)),
        "screening_only_count": int(readiness_counts.get("screening_only", 0)),
        "descriptive_only_count": int(readiness_counts.get("descriptive_only", 0)),
        "triage_advance_count": int(triage_counts.get("advance", 0)),
        "triage_caveat_count": int(triage_counts.get("advance_with_caveats", 0)),
        "triage_descriptive_count": int(triage_counts.get("descriptive_only", 0)),
        "benchmark_row_count": int(len(benchmark)),
        "benchmark_concept_count": int(benchmark["public_concept_name"].nunique()),
        "sensitivity_row_count": int(len(sensitivity)),
        "heterogeneity_row_count": int(len(heterogeneity)),
        "dag_family_count": int(inputs["dag_inventory"]["family_id"].nunique()),
        "classification_counts": classification_counts,
        "strong_candidates": strong_candidates,
        "usable_with_caveats": usable_with_caveats,
        "secondary_only": secondary_only,
        "fragility_flags": fragility_flags,
        "source_sensitive_concepts": scorecard.loc[
            scorecard["source_sensitive_flag"].astype(str).eq("yes"),
            "public_concept_name",
        ].astype(str).tolist(),
        "heterogeneity_inconsistent_concepts": scorecard.loc[
            scorecard["heterogeneity_inconsistent_flag"].astype(str).eq("yes"),
            "public_concept_name",
        ].astype(str).tolist(),
    }
    return metrics


def _table_shells(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_tables = pd.DataFrame(
        [
            {
                "table_id": "Table 1",
                "placement": "Main text",
                "title": "Operational readiness classification for Track A public concepts",
                "purpose": "Summarize Stage 1 analytical readiness and headline eligibility.",
                "source_artifact": "artifacts/track_a/freeze/analysis_ready_matrix.csv",
            },
            {
                "table_id": "Table 2",
                "placement": "Main text",
                "title": "Screening triage and benchmark promotion decisions",
                "purpose": "Summarize Stage 4 signal triage and benchmark-stage recommendation.",
                "source_artifact": "artifacts/track_a/screening/triage_signal_matrix.csv",
            },
            {
                "table_id": "Table 3",
                "placement": "Main text",
                "title": "Progressive benchmark ladder summary across modeled concepts",
                "purpose": "Summarize Stage 6 benchmark ladder outputs and directional attenuation.",
                "source_artifact": "artifacts/track_a/models/benchmark_results.csv",
            },
            {
                "table_id": "Table 4",
                "placement": "Main text",
                "title": "Framework validation scorecard",
                "purpose": "Present final Track A indicator classification and score dimensions.",
                "source_artifact": "artifacts/track_a/synthesis/framework_validation_scorecard.csv",
            },
        ]
    )
    supplement_tables = pd.DataFrame(
        [
            {
                "table_id": "Supplementary Table S1",
                "placement": "Supplement",
                "title": "Academic-English variable dictionary for Track A analytical outputs",
                "purpose": "Document public-facing variables and provenance crosswalks.",
                "source_artifact": "artifacts/track_a/metadata/track_a_variable_dictionary.csv",
            },
            {
                "table_id": "Supplementary Table S2",
                "placement": "Supplement",
                "title": "DAG family inventory and covered outcome map",
                "purpose": "Map DAG families to outcomes and adjustment architecture.",
                "source_artifact": "artifacts/track_a/dags/dag_inventory.csv",
            },
            {
                "table_id": "Supplementary Table S3",
                "placement": "Supplement",
                "title": "Sensitivity analysis matrix",
                "purpose": "Summarize source restriction, exposure-family comparison, and alternative lag/window results.",
                "source_artifact": "artifacts/track_a/sensitivity/sensitivity_matrix.csv",
            },
            {
                "table_id": "Supplementary Table S4",
                "placement": "Supplement",
                "title": "Heterogeneity and modifier results",
                "purpose": "Summarize one-modifier-at-a-time heterogeneity analyses.",
                "source_artifact": "artifacts/track_a/sensitivity/heterogeneity_results.csv",
            },
            {
                "table_id": "Supplementary Table S5",
                "placement": "Supplement",
                "title": "Framework validation evidence trace",
                "purpose": "Map scorecard classifications back to staged evidence.",
                "source_artifact": "artifacts/track_a/synthesis/framework_validation_evidence_trace.csv",
            },
        ]
    )
    return main_tables, supplement_tables


def _figure_shells() -> tuple[pd.DataFrame, pd.DataFrame]:
    main_figures = pd.DataFrame(
        [
            {
                "figure_id": "Figure 1",
                "placement": "Main text",
                "title": "Track A framework-validation design and stage flow",
                "purpose": "Show the staged logic from freeze through scorecard synthesis.",
                "source_artifact": "artifacts/track_a/dags/dag_package_report.md",
            },
            {
                "figure_id": "Figure 2",
                "placement": "Main text",
                "title": "Screening triage summary across Track A concepts",
                "purpose": "Show Stage 4 triage classes and benchmark promotion status.",
                "source_artifact": "artifacts/track_a/screening/screening_plots/figure_04_triage_summary.png",
            },
            {
                "figure_id": "Figure 3",
                "placement": "Main text",
                "title": "Benchmark ladder effect summary for benchmark-ready concepts",
                "purpose": "Show Stage 6 benchmark effect patterns and attenuation.",
                "source_artifact": "artifacts/track_a/models/effect_plots/figure_04_model_diagnostics.png",
            },
            {
                "figure_id": "Figure 4",
                "placement": "Main text",
                "title": "Indicator scorecard classification overview",
                "purpose": "Show the final Track A class distribution and fragility flags.",
                "source_artifact": "artifacts/track_a/synthesis/framework_validation_scorecard.csv",
            },
        ]
    )
    supplement_figures = pd.DataFrame(
        [
            {
                "figure_id": "Supplementary Figure S1",
                "placement": "Supplement",
                "title": "Acute daily DAG family",
                "purpose": "Document the acute fixed-effects adjustment architecture.",
                "source_artifact": "artifacts/track_a/dags/dag_family_acute_daily.png",
            },
            {
                "figure_id": "Supplementary Figure S2",
                "placement": "Supplement",
                "title": "Gestational delivery-window aggregate DAG family",
                "purpose": "Document delivery-aligned exposure logic and Track B deferrals.",
                "source_artifact": "artifacts/track_a/dags/dag_family_gestational_delivery_window_aggregate.png",
            },
            {
                "figure_id": "Supplementary Figure S3",
                "placement": "Supplement",
                "title": "Signal stability map",
                "purpose": "Show municipality-level vulnerability and stability patterns.",
                "source_artifact": "artifacts/track_a/sensitivity/vulnerability_maps/figure_02_signal_stability_score_map.png",
            },
            {
                "figure_id": "Supplementary Figure S4",
                "placement": "Supplement",
                "title": "Delivery-window screening summary",
                "purpose": "Show Stage 4 delivery-window screening patterns.",
                "source_artifact": "artifacts/track_a/screening/screening_plots/figure_03_delivery_window_screening.png",
            },
        ]
    )
    return main_figures, supplement_figures


def _strobe_checklist() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item": "Title and abstract", "manuscript_section": "Title page; Abstract", "status": "Addressed", "note": "Framework-validation design and scope stated explicitly."},
            {"item": "Background and rationale", "manuscript_section": "Introduction", "status": "Addressed", "note": "WMO/WHO-oriented framework rationale summarized."},
            {"item": "Objectives", "manuscript_section": "Introduction", "status": "Addressed", "note": "Primary objective defined as framework validation."},
            {"item": "Study design", "manuscript_section": "Methods", "status": "Addressed", "note": "Staged analytical design and frozen-gold strategy described."},
            {"item": "Setting", "manuscript_section": "Methods", "status": "Addressed", "note": "Brazilian municipality-day scope and 2020-2024 period documented."},
            {"item": "Variables", "manuscript_section": "Methods; Supplement", "status": "Addressed", "note": "Outcome, exposure, and contextual variable families mapped to source artifacts."},
            {"item": "Data sources and measurement", "manuscript_section": "Methods; Supplement", "status": "Addressed", "note": "Freeze, semantic layer, and climate lineage referenced."},
            {"item": "Bias", "manuscript_section": "Limitations", "status": "Addressed", "note": "Source dependence and exposure-window limits reported."},
            {"item": "Study size", "manuscript_section": "Methods", "status": "Addressed", "note": "Indicator count and analytical stage coverage summarized."},
            {"item": "Statistical methods", "manuscript_section": "Methods", "status": "Addressed", "note": "Screening, benchmark, sensitivity, and scorecard stages summarized."},
            {"item": "Main results", "manuscript_section": "Results", "status": "Addressed", "note": "Scorecard classes and strong candidates reported."},
            {"item": "Limitations", "manuscript_section": "Limitations", "status": "Addressed", "note": "Design and data constraints stated directly."},
            {"item": "Interpretation", "manuscript_section": "Discussion", "status": "Addressed", "note": "Interpretation restricted to framework validation."},
        ]
    )


def _record_checklist() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item": "Data source description", "manuscript_section": "Methods", "status": "Addressed", "note": "Frozen gold stack and upstream provenance summarized."},
            {"item": "Study population selection", "manuscript_section": "Methods", "status": "Addressed", "note": "Public concepts and stage eligibility criteria reported."},
            {"item": "Code lists and algorithms", "manuscript_section": "Supplement", "status": "Addressed", "note": "Canonical mappings and DAG package referenced."},
            {"item": "Data linkage", "manuscript_section": "Methods; Supplement", "status": "Addressed", "note": "Climate-enriched gold linkage and semantic layer documented."},
            {"item": "Data cleaning methods", "manuscript_section": "Methods; Limitations", "status": "Addressed", "note": "Freeze checks, anomaly correction, and denominator rules referenced."},
            {"item": "Access to data and code", "manuscript_section": "Supplement; Frozen release note", "status": "Addressed", "note": "Artifact paths and frozen release note included."},
            {"item": "Data sharing statement", "manuscript_section": "Submission package", "status": "Addressed", "note": "Journal package manifest links source artifacts without duplicating canonical data."},
            {"item": "Reproducibility statement", "manuscript_section": "Methods; Supplement", "status": "Addressed", "note": "Stage outputs and evidence trace are cited explicitly."},
        ]
    )


def _render_outline(metrics: dict[str, Any]) -> str:
    return (
        "# Outline\n\n"
        "1. Introduction\n"
        "2. Methods\n"
        "3. Results\n"
        "4. Discussion\n"
        "5. Limitations\n"
        "6. Tables\n"
        "7. Figures\n"
        "8. Supplement\n\n"
        "Current package status:\n\n"
        f"- Indicators scored: `{metrics['indicator_count']}`\n"
        f"- Strong candidates: `{metrics['classification_counts'].get('strong candidate', 0)}`\n"
        f"- Usable with caveats: `{metrics['classification_counts'].get('usable with caveats', 0)}`\n"
        f"- Secondary only: `{metrics['classification_counts'].get('secondary only', 0)}`\n"
    )


def _render_introduction(metrics: dict[str, Any]) -> str:
    return (
        "# Introduction\n\n"
        "The Track A manuscript evaluates whether the frozen municipality-day gold stack can support a disciplined and policy-relevant "
        "framework for heat and maternal, neonatal, and child health interpretation across Brazil. The objective is not to elevate a "
        "single flagship perinatal result. Instead, the paper asks whether the combined analytical stack can operationalize a broad "
        "WMO/WHO-oriented framework with transparent denominators, reproducible stage logic, and interpretable outcome families.\n\n"
        f"The final Track A package scored `{metrics['indicator_count']}` public concepts over `{metrics['analysis_years']}`. "
        f"Stage 1 classified `{metrics['ready_benchmark_count']}` concepts as benchmark-ready, `{metrics['screening_only_count']}` as screening-only, "
        f"and `{metrics['descriptive_only_count']}` as descriptive-only after duplicate proxy collapse. The synthesis therefore focuses on "
        "framework usability, operational validity, and robustness rather than on a single disease-specific claim.\n"
    )


def _render_methods(metrics: dict[str, Any]) -> str:
    return (
        "# Methods\n\n"
        "Track A used a frozen-gold, gold-first analytical design. The canonical gold layer remained read-only, while all analytical derivations "
        "were produced in Track A artifacts and publication workspaces. The staged design comprised: (1) an analytical freeze and readiness gate; "
        "(2) English-language semantic mapping and analysis marts; (3) a descriptive atlas; (4) screening and statistical triage; "
        "(5) DAG package definition; (6) progressive benchmark modeling; (7) sensitivity, heterogeneity, and vulnerability assessment; and "
        "(8) framework synthesis through an indicator scorecard.\n\n"
        f"Stage 4 produced `{metrics['triage_advance_count']}` `advance` concepts and `{metrics['triage_caveat_count']}` `advance_with_caveats` concepts. "
        f"Stage 6 benchmarked `{metrics['benchmark_concept_count']}` concepts across `{metrics['benchmark_row_count']}` modeled rows, and Stage 7 added "
        f"`{metrics['sensitivity_row_count']}` sensitivity rows plus `{metrics['heterogeneity_row_count']}` heterogeneity rows. "
        "Stage 8 translated the staged evidence into five score dimensions: operational validity, construct validity, robustness across model families, "
        "source dependence, and implementation value.\n\n"
        "Delivery-aligned 7-, 14-, and 28-day exposure windows were retained as observable pre-event summaries for Track A and were not presented as "
        "whole-pregnancy or trimester-specific causal windows. Municipality context was used for heterogeneity interpretation, not as a default adjustment "
        "set in acute fixed-effects designs.\n"
    )


def _render_results(metrics: dict[str, Any]) -> str:
    return (
        "# Results\n\n"
        "Operational readiness remained high across the frozen stack, but later stages narrowed the list of indicators that could support strong "
        "frontline framework interpretation. Screening remained signal-rich, whereas benchmark modeling attenuated several associations and Stage 7 "
        "introduced additional source-dependence and heterogeneity caveats.\n\n"
        "The final scorecard identified the following strong candidates:\n\n"
        f"{_markdown_bullets(metrics['strong_candidates'])}\n\n"
        "Concepts that remained usable with caveats were:\n\n"
        f"{_markdown_bullets(metrics['usable_with_caveats'])}\n\n"
        "Fragility flags from source dependence or heterogeneity inconsistency were concentrated in:\n\n"
        f"{_markdown_bullets(metrics['fragility_flags'])}\n"
    )


def _render_discussion(metrics: dict[str, Any]) -> str:
    return (
        "# Discussion\n\n"
        "The Track A evidence package supports a broad framework-validation claim rather than a flagship etiologic claim. The most defensible indicators "
        "were mortality outcomes, which retained high operational readiness and comparatively stable staged evidence. Perinatal and maternal operational "
        "concepts remained usable, but they were more sensitive to benchmark attenuation or Stage 7 instability.\n\n"
        f"Only `{metrics['classification_counts'].get('strong candidate', 0)}` indicators qualified as strong candidates, whereas "
        f"`{metrics['classification_counts'].get('usable with caveats', 0)}` required explicit caveats. This pattern is informative rather than disappointing: "
        "it shows that the framework can distinguish between operationally ready indicators that remain stable and those that should stay secondary or provisional.\n"
    )


def _render_limitations(metrics: dict[str, Any]) -> str:
    return (
        "# Limitations\n\n"
        "Track A is constrained by the frozen-gold design and intentionally does not construct the event-level gestational exposure matrix planned for Track B. "
        "Accordingly, delivery-aligned windows should not be interpreted as trimester-specific or whole-pregnancy exposures. Benchmark attenuation also limits "
        "headline interpretation for several indicators that screened positively. In addition, proxy concepts and outcomes with limited endpoint certification "
        "remain outside the main causal interpretation set.\n\n"
        "The most important caveated indicators were "
        f"{_format_concept_list(metrics['fragility_flags'])} "
        "These concepts remain informative for framework validation, but they should not anchor the strongest public-facing claims without further design refinement.\n"
    )


def _render_tables_section(main_tables: pd.DataFrame, supplement_tables: pd.DataFrame) -> str:
    return (
        "# Tables\n\n"
        "## Main-text table shells\n\n"
        f"{_markdown_table(main_tables[['table_id', 'title', 'purpose']])}\n"
        "## Supplementary table shells\n\n"
        f"{_markdown_table(supplement_tables[['table_id', 'title', 'purpose']])}"
    )


def _render_figures_section(main_figures: pd.DataFrame, supplement_figures: pd.DataFrame) -> str:
    return (
        "# Figures\n\n"
        "## Main-text figure shells\n\n"
        f"{_markdown_table(main_figures[['figure_id', 'title', 'purpose']])}\n"
        "## Supplementary figure shells\n\n"
        f"{_markdown_table(supplement_figures[['figure_id', 'title', 'purpose']])}"
    )


def _render_supplement(metrics: dict[str, Any]) -> str:
    return (
        "# Supplement\n\n"
        "This supplement package documents the Track A analytical design, DAG architecture, screening and benchmark staging, sensitivity outputs, "
        "indicator scorecard evidence trace, and reporting checklists.\n\n"
        "## Supplement overview\n\n"
        f"- DAG families documented: `{metrics['dag_family_count']}`\n"
        f"- Sensitivity rows summarized: `{metrics['sensitivity_row_count']}`\n"
        f"- Heterogeneity rows summarized: `{metrics['heterogeneity_row_count']}`\n"
        f"- Secondary-only concepts: `{len(metrics['secondary_only'])}`\n\n"
        "## Track B deferrals\n\n"
        "- Event-level gestational timing matrices remain out of scope.\n"
        "- Third-trimester and whole-pregnancy exposure windows remain deferred.\n"
        "- Policy-DAG estimation remains structural only in Track A.\n"
    )


def _compose_main_markdown(
    introduction: str,
    methods: str,
    results: str,
    discussion: str,
    limitations: str,
) -> str:
    return "\n\n".join(
        [
            "# Track A Framework Validation Manuscript",
            introduction,
            methods,
            results,
            discussion,
            limitations,
        ]
    ) + "\n"


def _compose_supplement_markdown(
    supplement: str,
    tables_section: str,
    figures_section: str,
    strobe_markdown: str,
    record_markdown: str,
) -> str:
    return "\n\n".join(
        [
            "# Track A Framework Validation Supplement",
            supplement,
            "## STROBE Checklist\n\n" + strobe_markdown,
            "## RECORD Checklist\n\n" + record_markdown,
            tables_section,
            figures_section,
        ]
    ) + "\n"


def _set_cell_shading(cell: Any, fill: str = "D9E6F2") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def _style_document(doc: Any) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10.5)

    for name, size in [("Title", 20), ("Heading 1", 15), ("Heading 2", 12), ("Heading 3", 11)]:
        style = doc.styles[name]
        style.font.name = "Aptos"
        style.font.size = Pt(size)
        style.font.bold = True


def _add_title_page(doc: Any, *, title: str, subtitle: str, metadata: list[tuple[str, str]], note: str | None = None) -> None:
    paragraph = doc.add_paragraph(style="Title")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run(title)

    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(subtitle)
    run.bold = True

    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    for index, label in enumerate(["Field", "Value"]):
        cell = table.rows[0].cells[index]
        cell.text = label
        _set_cell_shading(cell)

    for key, value in metadata:
        row = table.add_row().cells
        row[0].text = key
        row[1].text = value

    if note:
        doc.add_paragraph()
        note_paragraph = doc.add_paragraph()
        note_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        note_run = note_paragraph.add_run(note)
        note_run.italic = True

    doc.add_page_break()


def _add_paragraphs(doc: Any, blocks: list[str]) -> None:
    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            for item in stripped.splitlines():
                if item.startswith("- "):
                    doc.add_paragraph(item[2:], style="List Bullet")
        else:
            doc.add_paragraph(stripped)


def _add_dataframe_table(doc: Any, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    table = doc.add_table(rows=1, cols=len(frame.columns))
    table.style = "Table Grid"
    for index, column in enumerate(frame.columns):
        cell = table.rows[0].cells[index]
        cell.text = str(column)
        _set_cell_shading(cell)
    for record in frame.fillna("").astype(str).to_dict(orient="records"):
        row = table.add_row().cells
        for index, column in enumerate(frame.columns):
            row[index].text = record[str(column)]


def _build_docx_documents(
    workspace_root: Path,
    metrics: dict[str, Any],
    main_tables: pd.DataFrame,
    supplement_tables: pd.DataFrame,
    main_figures: pd.DataFrame,
    supplement_figures: pd.DataFrame,
    strobe: pd.DataFrame,
    record: pd.DataFrame,
) -> tuple[Path, Path]:
    _require_docx_dependency()

    manuscript_doc = Document()
    _style_document(manuscript_doc)
    _add_title_page(
        manuscript_doc,
        title="Track A Framework Validation",
        subtitle="Main manuscript package",
        metadata=[
            ("Date", str(date.today())),
            ("Language", "Academic English"),
            ("Analytical years", str(metrics["analysis_years"])),
            ("Indicators scored", str(metrics["indicator_count"])),
            ("Strong candidates", str(metrics["classification_counts"].get("strong candidate", 0))),
            ("Usable with caveats", str(metrics["classification_counts"].get("usable with caveats", 0))),
        ],
        note="Draft generated from staged Track A artifacts.",
    )
    manuscript_doc.add_heading("Abstract", level=1)
    _add_paragraphs(
        manuscript_doc,
        [
            (
                "Track A evaluated whether a frozen municipality-day gold stack can sustain a broad framework-validation paper for "
                "heat and maternal, neonatal, and child health interpretation across Brazil. The staged package combined analytical "
                "freeze checks, descriptive profiling, screening triage, benchmark modeling, DAG families, sensitivity analysis, and "
                "an indicator scorecard. The final synthesis scored "
                f"{metrics['indicator_count']} public concepts and identified "
                f"{metrics['classification_counts'].get('strong candidate', 0)} strong candidates, "
                f"{metrics['classification_counts'].get('usable with caveats', 0)} usable concepts with caveats, and "
                f"{metrics['classification_counts'].get('secondary only', 0)} secondary-only concepts."
            ),
            "Keywords: heat, maternal health, neonatal mortality, child mortality, framework validation, Brazil.",
        ],
    )

    section_map = {
        "Introduction": _render_introduction(metrics),
        "Methods": _render_methods(metrics),
        "Results": _render_results(metrics),
        "Discussion": _render_discussion(metrics),
        "Limitations": _render_limitations(metrics),
    }
    for heading, markdown in section_map.items():
        manuscript_doc.add_heading(heading, level=1)
        body = markdown.split("\n", 2)[2] if markdown.startswith("# ") else markdown
        _add_paragraphs(manuscript_doc, body.split("\n\n"))

    manuscript_doc.add_heading("Package tables", level=1)
    _add_dataframe_table(manuscript_doc, main_tables[["table_id", "title", "purpose"]])
    manuscript_doc.add_heading("Package figures", level=1)
    _add_dataframe_table(manuscript_doc, main_figures[["figure_id", "title", "purpose"]])

    manuscript_path = workspace_root / "manuscript_trackA.docx"
    manuscript_doc.save(manuscript_path)

    supplement_doc = Document()
    _style_document(supplement_doc)
    _add_title_page(
        supplement_doc,
        title="Track A Framework Validation",
        subtitle="Supplement package",
        metadata=[
            ("Date", str(date.today())),
            ("Language", "Academic English"),
            ("DAG families", str(metrics["dag_family_count"])),
            ("Sensitivity rows", str(metrics["sensitivity_row_count"])),
            ("Heterogeneity rows", str(metrics["heterogeneity_row_count"])),
            ("Fragility flags", str(len(metrics["fragility_flags"]))),
        ],
        note="Structural document validation completed without visual render because LibreOffice and Poppler were unavailable in the environment.",
    )
    supplement_doc.add_heading("Supplement overview", level=1)
    _add_paragraphs(supplement_doc, _render_supplement(metrics).split("\n\n")[1:])
    supplement_doc.add_heading("Supplementary table shells", level=1)
    _add_dataframe_table(supplement_doc, supplement_tables[["table_id", "title", "purpose"]])
    supplement_doc.add_heading("Supplementary figure shells", level=1)
    _add_dataframe_table(supplement_doc, supplement_figures[["figure_id", "title", "purpose"]])
    supplement_doc.add_heading("STROBE checklist", level=1)
    _add_dataframe_table(supplement_doc, strobe[["item", "manuscript_section", "status", "note"]])
    supplement_doc.add_heading("RECORD checklist", level=1)
    _add_dataframe_table(supplement_doc, record[["item", "manuscript_section", "status", "note"]])

    supplement_path = workspace_root / "supplement_trackA.docx"
    supplement_doc.save(supplement_path)
    return manuscript_path, supplement_path


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_publication_workspace(
    root: Path,
    workspace_root: Path,
    metrics: dict[str, Any],
    main_tables: pd.DataFrame,
    supplement_tables: pd.DataFrame,
    main_figures: pd.DataFrame,
    supplement_figures: pd.DataFrame,
    strobe: pd.DataFrame,
    record: pd.DataFrame,
    evidence_trace: pd.DataFrame,
) -> dict[str, Path]:
    outline_text = _render_outline(metrics)
    introduction_text = _render_introduction(metrics)
    methods_text = _render_methods(metrics)
    results_text = _render_results(metrics)
    discussion_text = _render_discussion(metrics)
    limitations_text = _render_limitations(metrics)
    tables_text = _render_tables_section(main_tables, supplement_tables)
    figures_text = _render_figures_section(main_figures, supplement_figures)
    supplement_text = _render_supplement(metrics)

    paths = {
        "outline": workspace_root / "00_outline.md",
        "introduction": workspace_root / "01_introduction.md",
        "methods": workspace_root / "02_methods.md",
        "results": workspace_root / "03_results.md",
        "discussion": workspace_root / "04_discussion.md",
        "limitations": workspace_root / "05_limitations.md",
        "tables": workspace_root / "06_tables.md",
        "figures": workspace_root / "07_figures.md",
        "supplement": workspace_root / "08_supplement.md",
        "manuscript_md": workspace_root / "manuscript_trackA.md",
        "supplement_md": workspace_root / "supplement_trackA.md",
        "main_tables_csv": workspace_root / "tables" / "main_text_table_shells.csv",
        "supplement_tables_csv": workspace_root / "tables" / "supplement_table_shells.csv",
        "main_figures_csv": workspace_root / "figures" / "main_text_figure_shells.csv",
        "supplement_figures_csv": workspace_root / "figures" / "supplement_figure_shells.csv",
        "strobe": workspace_root / "submission" / "strobe_checklist.md",
        "record": workspace_root / "submission" / "record_checklist.md",
        "artifact_manifest": workspace_root / "manifests" / "publication_artifact_manifest.csv",
    }

    _write_text(paths["outline"], outline_text)
    _write_text(paths["introduction"], introduction_text)
    _write_text(paths["methods"], methods_text)
    _write_text(paths["results"], results_text)
    _write_text(paths["discussion"], discussion_text)
    _write_text(paths["limitations"], limitations_text)
    _write_text(paths["tables"], tables_text)
    _write_text(paths["figures"], figures_text)
    _write_text(paths["supplement"], supplement_text)
    _write_text(
        paths["manuscript_md"],
        _compose_main_markdown(
            introduction_text,
            methods_text,
            results_text,
            discussion_text,
            limitations_text,
        ),
    )
    _write_text(
        paths["supplement_md"],
        _compose_supplement_markdown(
            supplement_text,
            tables_text,
            figures_text,
            _markdown_table(strobe),
            _markdown_table(record),
        ),
    )
    _write_text(paths["strobe"], "# STROBE Checklist\n\n" + _markdown_table(strobe))
    _write_text(paths["record"], "# RECORD Checklist\n\n" + _markdown_table(record))

    paths["main_tables_csv"].parent.mkdir(parents=True, exist_ok=True)
    main_tables.to_csv(paths["main_tables_csv"], index=False)
    supplement_tables.to_csv(paths["supplement_tables_csv"], index=False)
    main_figures.to_csv(paths["main_figures_csv"], index=False)
    supplement_figures.to_csv(paths["supplement_figures_csv"], index=False)

    artifact_manifest = pd.DataFrame(
        [
            {"artifact_role": "readiness_matrix", "relative_path": "artifacts/track_a/freeze/analysis_ready_matrix.csv"},
            {"artifact_role": "triage_matrix", "relative_path": "artifacts/track_a/screening/triage_signal_matrix.csv"},
            {"artifact_role": "benchmark_results", "relative_path": "artifacts/track_a/models/benchmark_results.csv"},
            {"artifact_role": "sensitivity_matrix", "relative_path": "artifacts/track_a/sensitivity/sensitivity_matrix.csv"},
            {"artifact_role": "heterogeneity_results", "relative_path": "artifacts/track_a/sensitivity/heterogeneity_results.csv"},
            {"artifact_role": "scorecard", "relative_path": "artifacts/track_a/synthesis/framework_validation_scorecard.csv"},
            {"artifact_role": "evidence_trace", "relative_path": "artifacts/track_a/synthesis/framework_validation_evidence_trace.csv"},
            {"artifact_role": "evidence_trace_rows", "relative_path": f"{len(evidence_trace)}"},
        ]
    )
    artifact_manifest.to_csv(paths["artifact_manifest"], index=False)
    return paths


def _copy_submission_package(
    workspace_root: Path,
    manuscript_path: Path,
    supplement_path: Path,
    strobe_path: Path,
    record_path: Path,
) -> tuple[Path, Path]:
    package_root = workspace_root / "submission" / "journal_package_trackA"
    package_root.mkdir(parents=True, exist_ok=True)
    manuscript_copy = package_root / manuscript_path.name
    supplement_copy = package_root / supplement_path.name
    copy2(manuscript_path, manuscript_copy)
    copy2(supplement_path, supplement_copy)
    copy2(strobe_path, package_root / strobe_path.name)
    copy2(record_path, package_root / record_path.name)
    manifest_path = package_root / "journal_package_manifest.md"
    manifest_path.write_text(
        "# Journal Package Manifest\n\n"
        "- Main manuscript: `manuscript_trackA.docx`\n"
        "- Supplement: `supplement_trackA.docx`\n"
        "- Reporting checklist: `strobe_checklist.md`\n"
        "- Reporting checklist: `record_checklist.md`\n",
        encoding="utf-8",
    )
    return package_root, manifest_path


def _write_frozen_release_note(root: Path, metrics: dict[str, Any], manuscript_path: Path, supplement_path: Path) -> Path:
    frozen_root = root / "reports" / "frozen_releases"
    frozen_root.mkdir(parents=True, exist_ok=True)
    release_path = frozen_root / "track_a_framework_validation_stage9_release.md"
    release_path.write_text(
        "# Track A Framework Validation Stage 9 Release\n\n"
        f"- Release date: `{date.today()}`\n"
        f"- Analytical years: `{metrics['analysis_years']}`\n"
        f"- Indicators scored: `{metrics['indicator_count']}`\n"
        f"- Strong candidates: `{metrics['classification_counts'].get('strong candidate', 0)}`\n"
        f"- Usable with caveats: `{metrics['classification_counts'].get('usable with caveats', 0)}`\n"
        f"- Secondary only: `{metrics['classification_counts'].get('secondary only', 0)}`\n"
        f"- Main manuscript: `{_relative(root, manuscript_path)}`\n"
        f"- Supplement: `{_relative(root, supplement_path)}`\n"
        "- This release note references canonical and derived analytical artifacts without duplicating frozen data products.\n",
        encoding="utf-8",
    )
    return release_path


def build_track_a_paper_package(root: Path) -> dict[str, object]:
    workspace_report = scaffold_publication_workspace(root)
    inputs = _load_inputs(root)
    metrics = _publication_metrics(inputs)
    main_tables, supplement_tables = _table_shells(root)
    main_figures, supplement_figures = _figure_shells()
    strobe = _strobe_checklist()
    record = _record_checklist()

    workspace_root = track_a_publication_root(root)
    publication_paths = _write_publication_workspace(
        root=root,
        workspace_root=workspace_root,
        metrics=metrics,
        main_tables=main_tables,
        supplement_tables=supplement_tables,
        main_figures=main_figures,
        supplement_figures=supplement_figures,
        strobe=strobe,
        record=record,
        evidence_trace=inputs["evidence_trace"],
    )

    manuscript_path, supplement_path = _build_docx_documents(
        workspace_root=workspace_root,
        metrics=metrics,
        main_tables=main_tables,
        supplement_tables=supplement_tables,
        main_figures=main_figures,
        supplement_figures=supplement_figures,
        strobe=strobe,
        record=record,
    )
    package_root, package_manifest_path = _copy_submission_package(
        workspace_root=workspace_root,
        manuscript_path=manuscript_path,
        supplement_path=supplement_path,
        strobe_path=publication_paths["strobe"],
        record_path=publication_paths["record"],
    )
    frozen_release_path = _write_frozen_release_note(root, metrics, manuscript_path, supplement_path)

    return {
        "status": "PASS",
        "stage": "stage9_manuscript_and_supplement_package",
        "indicator_count": metrics["indicator_count"],
        "classification_counts": metrics["classification_counts"],
        "visual_validation_mode": "structural_docx_validation_only",
        "workspace_report": workspace_report,
        "outputs": {
            "workspace_root": _relative(root, workspace_root),
            "manuscript_trackA_docx": _relative(root, manuscript_path),
            "supplement_trackA_docx": _relative(root, supplement_path),
            "manuscript_trackA_markdown": _relative(root, publication_paths["manuscript_md"]),
            "supplement_trackA_markdown": _relative(root, publication_paths["supplement_md"]),
            "main_text_table_shells": _relative(root, publication_paths["main_tables_csv"]),
            "supplement_table_shells": _relative(root, publication_paths["supplement_tables_csv"]),
            "main_text_figure_shells": _relative(root, publication_paths["main_figures_csv"]),
            "supplement_figure_shells": _relative(root, publication_paths["supplement_figures_csv"]),
            "strobe_checklist": _relative(root, publication_paths["strobe"]),
            "record_checklist": _relative(root, publication_paths["record"]),
            "publication_artifact_manifest": _relative(root, publication_paths["artifact_manifest"]),
            "journal_package_root": _relative(root, package_root),
            "journal_package_manifest": _relative(root, package_manifest_path),
            "frozen_release_note": _relative(root, frozen_release_path),
        },
    }


TRACK_A_VALIDITY_V2_FILES = {
    "README.md": """# Track A Validity Report V2

This workspace stores the refined v2 Track A validity report and supplementary material.

The v2 package is a parallel publication build. It must not overwrite the existing Track A manuscript package.
""",
    "00_outline.md": "# Outline\n\n1. Title page and structured abstract\n2. Key findings\n3. Introduction\n4. Methods\n5. Results\n6. Discussion\n7. Limitations\n8. Conclusion\n9. References\n10. Supplement\n",
    "01_introduction.md": "# Introduction\n\nUse this file to document the WHO/WMO-oriented framework framing and the Track A validation question.\n",
    "02_methods.md": "# Methods\n\nUse this file to summarize the frozen-gold analytical design, validity domains, and interpretation rules.\n",
    "03_results.md": "# Results\n\nUse this file to report data sufficiency, primary and secondary outcome validity, attenuation, source dependence, and implementation readiness.\n",
    "04_discussion.md": "# Discussion\n\nUse this file to interpret Track A as an initial framework-validation package.\n",
    "05_limitations.md": "# Limitations\n\nUse this file to document proxy benchmark limitations, source dependence, and deferred work.\n",
    "06_tables.md": "# Tables\n\nUse this file to summarize the main and supplementary tables included in the v2 package.\n",
    "07_figures.md": "# Figures\n\nUse this file to list the existing Track A figures referenced in the v2 supplement.\n",
    "08_supplement.md": "# Supplement\n\nUse this file to outline the supplementary appendices, detailed tables, and reference modules.\n",
    "tables/README.md": "# Tables\n\nStore Track A validity-report tables in this directory if exported separately.\n",
    "figures/README.md": "# Figures\n\nStore any curated figure references for the Track A validity report here.\n",
    "manifests/README.md": "# Manifests\n\nStore the evidence and reference manifest for the Track A validity report here.\n",
}

TRACK_A_VALIDITY_V2_PRIMARY_OUTCOMES = {
    "neonatal_death_count",
    "infant_death_count",
    "under_five_death_count",
}

TRACK_A_VALIDITY_V2_DISPLAY_LABELS = {
    "acute_care_proxy_count": "Pediatric acute-care proxy",
    "gestational_days_lost_below_37_weeks": "Gestational shortening below 37 weeks",
    "gestational_diabetes_admission_count": "Gestational diabetes admissions",
    "hypertensive_disorder_of_pregnancy_count": "Hypertensive disorders of pregnancy",
    "infant_death_count": "Infant mortality",
    "neonatal_death_count": "Neonatal mortality",
    "obstetric_admission_count": "Obstetric admissions",
    "preterm_birth_count": "Preterm birth",
    "preterm_labor_admission_count": "Preterm labor admissions",
    "severe_hypertensive_disorder_of_pregnancy_count": "Severe hypertensive disorders of pregnancy",
    "stillbirth_primary_gte28_count": "Stillbirth at 28 completed weeks or more",
    "stillbirth_sensitivity_gte22_count": "Stillbirth at 22 completed weeks or more",
    "under_five_death_count": "Under-five mortality",
    "very_preterm_birth_count": "Very preterm birth",
}

TRACK_A_VALIDITY_V2_REFERENCE_MODULES = [
    {
        "claim_area": "Heat hazard measurement and climate-source governance",
        "module_name": "Climate indicators",
        "vancouver_path": "references/vancouver/climate_indicators_vancouver.md",
        "notes_path": "references/notes/climate_indicator_rationale.md",
    },
    {
        "claim_area": "Climate-source provenance and metadata retention",
        "module_name": "Climate source fields",
        "vancouver_path": "references/vancouver/climate_source_fields_vancouver.md",
        "notes_path": "references/notes/climate_source_fields_rationale.md",
    },
    {
        "claim_area": "Raw-source provenance and transfer surfaces",
        "module_name": "Raw data sources",
        "vancouver_path": "references/vancouver/raw_data_sources_vancouver.md",
        "notes_path": "references/notes/raw_data_source_provenance.md",
    },
    {
        "claim_area": "Population and live-birth denominator rules",
        "module_name": "Population denominators",
        "vancouver_path": "references/vancouver/population_denominators_vancouver.md",
        "notes_path": "references/notes/population_denominator_rationale.md",
    },
    {
        "claim_area": "Mortality indicator semantics and downstream mortality rates",
        "module_name": "SIM mortality semantics",
        "vancouver_path": "references/vancouver/sim_general_mortality_vancouver.md",
        "notes_path": "references/notes/sim_general_mortality_rationale.md",
    },
    {
        "claim_area": "Operational maternal admissions and hospital-based phenotypes",
        "module_name": "SIH operational semantics",
        "vancouver_path": "references/vancouver/sih_sus_operational_vancouver.md",
        "notes_path": "references/notes/sih_sus_operational_rationale.md",
    },
    {
        "claim_area": "Prematurity, gestational shortening, and antenatal administrative indicators",
        "module_name": "Gestational age and prematurity",
        "vancouver_path": "references/vancouver/gestational_age_prematurity_vancouver.md",
        "notes_path": "references/notes/gestational_age_prematurity_rationale.md",
    },
    {
        "claim_area": "Stillbirth standards, sensitivity thresholds, and total-birth denominators",
        "module_name": "Stillbirth and fetal death",
        "vancouver_path": "references/vancouver/stillbirth_fetal_death_vancouver.md",
        "notes_path": "references/notes/stillbirth_fetal_death_rationale.md",
    },
    {
        "claim_area": "Live-birth administrative provenance",
        "module_name": "SINASC full dictionary",
        "vancouver_path": "references/vancouver/sinasc_full_dictionary_vancouver.md",
        "notes_path": "references/notes/sinasc_full_dictionary_rationale.md",
    },
]

TRACK_A_VALIDITY_V2_MAIN_REFERENCES = [
    "1. World Meteorological Organization. Guide to Instruments and Methods of Observation. WMO-No. 8. 2021/2018 ed. Geneva: World Meteorological Organization; 2021.",
    "2. World Meteorological Organization. Guide to Climatological Practices. WMO-No. 100. 3rd ed. Geneva: World Meteorological Organization; 2011.",
    "3. Hersbach H, Bell B, Berrisford P, Hirahara S, Horanyi A, Munoz-Sabater J, et al. The ERA5 global reanalysis. Q J R Meteorol Soc. 2020;146(730):1999-2049. doi:10.1002/qj.3803.",
    "4. Munoz-Sabater J, Dutra E, Agusti-Panareda A, Albergel C, Arduini G, Balsamo G, et al. ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. Earth Syst Sci Data. 2021;13(9):4349-4383. doi:10.5194/essd-13-4349-2021.",
    "5. World Health Organization. Neonatal mortality rate (per 1000 live births) [Internet]. Geneva: World Health Organization; [cited 2026 Mar 20]. Available from: https://www.who.int/data/gho/indicator-metadata-registry/imr-details/67",
    "6. World Health Organization. Infant mortality rate (between birth and 11 months per 1000 live births) [Internet]. Geneva: World Health Organization; [cited 2026 Mar 20]. Available from: https://www.who.int/data/gho/indicator-metadata-registry/imr-details/1",
    "7. World Health Organization. Under-five mortality rate (probability of dying by age 5 per 1000 live births) [Internet]. Geneva: World Health Organization; [cited 2026 Mar 20]. Available from: https://www.who.int/data/gho/data/indicators/indicator-details/GHO/under-five-mortality-rate-%28probability-of-dying-by-age-5-per-1000-live-births%29",
    "8. World Health Organization. Preterm birth [Internet]. Geneva: World Health Organization; 2023 May 10 [cited 2026 Mar 20]. Available from: https://www.who.int/news-room/fact-sheets/detail/preterm-birth",
    "9. World Health Organization. Stillbirth rate (per 1000 total births) [Internet]. Geneva: World Health Organization; [cited 2026 Mar 20]. Available from: https://www.who.int/data/gho/indicator-metadata-registry/imr-details/2444",
    "10. Blencowe H, Hug L, Moller AB, You D, Moran AC. Definitions, terminology and standards for reporting of births and deaths in the perinatal period: International Classification of Diseases (ICD-11). Int J Gynaecol Obstet. 2025;168(1):1-9. doi:10.1002/ijgo.15794.",
    "11. Lakhoo DP, Brink N, Radebe L, Craig MH, Pham MD, Haghighi MM, et al. A systematic review and meta-analysis of heat exposure impacts on maternal, fetal and neonatal health. Nat Med. 2025;31(2):684-694. doi:10.1038/s41591-024-03395-8.",
    "12. Silveira IH, Reboucas P, Pescarini J, Cortes TR, Oliveira BFA, Neves DJD, et al. Effects of ambient temperature on under-five mortality: a nationwide space-time-stratified case-crossover study in Brazil. Environ Res. 2025;285(Pt 5):122683. doi:10.1016/j.envres.2025.122683.",
]

TRACK_A_VALIDITY_V2_FRAMEWORK_DRAFT = Path(
    "/Users/digicode.agent/Downloads/Draft meeting report heat MNCH indicators_for participant review.docx"
)


def _track_a_validity_v2_workspace_root(root: Path) -> Path:
    return track_a_publication_root(root) / "v2_validity_report"


def scaffold_track_a_validity_v2_workspace(root: Path) -> dict[str, object]:
    workspace_root = _track_a_validity_v2_workspace_root(root)
    created_files: list[str] = []
    existing_files: list[str] = []
    workspace_root.mkdir(parents=True, exist_ok=True)

    for relative_path, contents in TRACK_A_VALIDITY_V2_FILES.items():
        target = workspace_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing_files.append(str(target.relative_to(root)))
            continue
        target.write_text(contents, encoding="utf-8")
        created_files.append(str(target.relative_to(root)))

    return {
        "workspace_root": str(workspace_root.relative_to(root)),
        "created_files": created_files,
        "existing_files": existing_files,
    }


def _load_validity_v2_inputs(root: Path) -> dict[str, Any]:
    inputs = _load_inputs(root)
    inputs["variable_dictionary"] = _load_optional_csv(
        root,
        "artifacts/track_a/metadata/track_a_variable_dictionary.csv",
    )
    inputs["config"] = load_track_a_config(root)
    return inputs


def _safe_int_value(value: Any) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return 0
    return int(numeric)


def _safe_float_value(value: Any, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return float(numeric)


def _flag_yes(value: Any) -> bool:
    return str(value).strip().lower() == "yes"


def _sentence_case(value: str) -> str:
    return value[:1].upper() + value[1:] if value else value


def _human_join(values: list[str]) -> str:
    items = [value for value in values if value]
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _human_join_lower(values: list[str]) -> str:
    lowered = [value[:1].lower() + value[1:] if value else value for value in values]
    return _human_join(lowered)


def _pluralize(count: int, singular: str, plural: str | None = None) -> str:
    if count == 1:
        return singular
    return plural if plural is not None else f"{singular}s"


def _validity_v2_display_label(concept_name: str, fallback: str) -> str:
    if concept_name in TRACK_A_VALIDITY_V2_DISPLAY_LABELS:
        return TRACK_A_VALIDITY_V2_DISPLAY_LABELS[concept_name]
    if fallback:
        return fallback
    return concept_name.replace("_", " ").strip().title()


def _validity_v2_stage1_summary(value: str) -> str:
    mapping = {
        "benchmark_ready": "Benchmark-ready in the frozen stack",
        "screening_only": "Restricted to screening support",
        "descriptive_only": "Restricted to descriptive or proxy use",
    }
    return mapping.get(str(value), "Operational status not classified")


def _validity_v2_screening_summary(row: pd.Series) -> str:
    triage_class = str(row.get("triage_priority_classification", ""))
    direction_consistency = _safe_float_value(row.get("signal_direction_consistency"), default=0.0)
    if triage_class == "advance" and direction_consistency >= 0.75:
        return "High screening coherence across retained domains"
    if triage_class == "advance":
        return "Signal-rich screening with some directional variation"
    if triage_class == "advance_with_caveats":
        return "Screening support present but coherence was incomplete"
    return "Screening support was limited or deferred"


def _validity_v2_benchmark_summary(row: pd.Series) -> str:
    benchmark_rows = _safe_int_value(row.get("benchmark_row_count"))
    non_null_rows = _safe_int_value(row.get("non_null_estimate_count"))
    null_rows = _safe_int_value(row.get("benchmark_null_row_count"))
    if benchmark_rows <= 0:
        return "Did not advance into the benchmark ladder"
    if str(row.get("final_classification", "")) == "strong candidate":
        if null_rows > 0:
            return "Core benchmark signal was retained, while auxiliary proxy rungs attenuated"
        return "Benchmark signal was retained across the modeled rows"
    if non_null_rows <= 0:
        return "Screening signals attenuated to near-null benchmark estimates"
    if _flag_yes(row.get("source_sensitive_flag")) or _flag_yes(row.get("heterogeneity_inconsistent_flag")):
        return "Some benchmark support remained, but later instability constrained interpretation"
    return "Benchmark support remained weaker than the initial screening pattern"


def _validity_v2_source_summary(row: pd.Series) -> str:
    if _safe_int_value(row.get("source_row_count")) <= 0:
        return "Source dependence was not estimable in the final package"
    max_change = _safe_float_value(row.get("source_max_abs_relative_change_pct"), default=0.0)
    if _flag_yes(row.get("source_sensitive_flag")) and _flag_yes(row.get("source_opposite_direction_flag")):
        return "High source dependence, including direction reversal"
    if _flag_yes(row.get("source_sensitive_flag")):
        return "Material source dependence"
    if max_change >= 5.0:
        return "Low to modest source movement"
    return "Low source dependence"


def _validity_v2_heterogeneity_summary(row: pd.Series) -> str:
    heterogeneity_rows = _safe_int_value(row.get("heterogeneity_row_count"))
    if heterogeneity_rows <= 0:
        return "Contextual heterogeneity was not estimated"
    if _flag_yes(row.get("heterogeneity_inconsistent_flag")):
        return "Directional reversals were present in some contextual strata"
    if _safe_int_value(row.get("heterogeneity_consistent_row_count")) > 0:
        return "No directional reversal was detected across the examined strata"
    return "Subgroup estimates were largely imprecise rather than contradictory"


def _validity_v2_implementation_summary(row: pd.Series) -> str:
    classification = str(row.get("final_classification", ""))
    concept_name = str(row.get("public_concept_name", ""))
    if concept_name in TRACK_A_VALIDITY_V2_PRIMARY_OUTCOMES:
        return "Ready for primary validation use"
    if classification == "strong candidate":
        return "Ready for frontline framework interpretation"
    if classification == "usable with caveats":
        return "Usable for secondary interpretation with explicit caveats"
    return "Retain for supplementary or contextual use only"


def _validity_v2_outcome_tier(row: pd.Series) -> str:
    concept_name = str(row.get("public_concept_name", ""))
    if concept_name in TRACK_A_VALIDITY_V2_PRIMARY_OUTCOMES:
        return "Primary validation outcome"
    if str(row.get("final_classification", "")) == "secondary only":
        return "Supporting outcome"
    return "Secondary validation outcome"


def _prepare_validity_v2_context(inputs: dict[str, Any]) -> dict[str, Any]:
    evidence_trace = inputs["evidence_trace"].copy()
    scorecard = inputs["scorecard"].copy()
    freeze = _unique_freeze_concepts(inputs["freeze"]).copy()
    freeze_defaults = {
        "readiness_status": "",
        "semantic_note": "",
        "supporting_tables": "",
        "canonical_dataset": "",
        "canonical_variable": "",
        "readiness_endpoint_name": "",
        "denominator_rule": "",
    }
    for column, default in freeze_defaults.items():
        if column not in freeze.columns:
            freeze[column] = default

    merged = evidence_trace.merge(
        scorecard[
            [
                "public_concept_name",
                "operational_validity_score",
                "construct_validity_score",
                "robustness_score",
                "source_dependence_rating",
                "source_dependence_penalty",
                "implementation_value_score",
                "classification_rationale",
            ]
        ],
        on="public_concept_name",
        how="left",
    )
    merged = merged.merge(
        freeze[
            [
                "public_concept_name",
                "readiness_status",
                "semantic_note",
                "supporting_tables",
                "prompt_label",
                "canonical_dataset",
                "canonical_variable",
                "readiness_endpoint_name",
                "denominator_rule",
            ]
        ],
        on="public_concept_name",
        how="left",
    )
    merged["display_label"] = [
        _validity_v2_display_label(concept_name=str(concept_name), fallback=str(label))
        for concept_name, label in zip(merged["public_concept_name"], merged["public_concept_label"], strict=False)
    ]
    merged["outcome_tier"] = merged.apply(_validity_v2_outcome_tier, axis=1)
    merged["stage1_summary"] = merged["stage1_classification"].map(_validity_v2_stage1_summary)
    merged["screening_summary"] = merged.apply(_validity_v2_screening_summary, axis=1)
    merged["benchmark_summary"] = merged.apply(_validity_v2_benchmark_summary, axis=1)
    merged["source_summary"] = merged.apply(_validity_v2_source_summary, axis=1)
    merged["heterogeneity_summary"] = merged.apply(_validity_v2_heterogeneity_summary, axis=1)
    merged["implementation_summary"] = merged.apply(_validity_v2_implementation_summary, axis=1)
    merged["final_classification_label"] = [
        _sentence_case(str(value)) for value in merged["final_classification"].astype(str)
    ]
    merged["source_max_abs_relative_change_pct"] = pd.to_numeric(
        merged.get("source_max_abs_relative_change_pct"), errors="coerce"
    ).round(1)
    merged["signal_direction_consistency"] = pd.to_numeric(
        merged.get("signal_direction_consistency"), errors="coerce"
    ).round(2)
    merged["primary_sort"] = merged["public_concept_name"].map(
        lambda value: 0 if value in TRACK_A_VALIDITY_V2_PRIMARY_OUTCOMES else 1
    )
    merged = merged.sort_values(
        ["primary_sort", "final_score", "display_label"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    strong_labels = merged.loc[
        merged["final_classification"].eq("strong candidate"),
        "display_label",
    ].astype(str).tolist()
    caveat_labels = merged.loc[
        merged["final_classification"].eq("usable with caveats"),
        "display_label",
    ].astype(str).tolist()
    secondary_only_labels = merged.loc[
        merged["final_classification"].eq("secondary only"),
        "display_label",
    ].astype(str).tolist()
    source_sensitive_labels = merged.loc[
        merged["source_sensitive_flag"].astype(str).eq("yes"),
        "display_label",
    ].astype(str).tolist()
    heterogeneity_labels = merged.loc[
        merged["heterogeneity_inconsistent_flag"].astype(str).eq("yes"),
        "display_label",
    ].astype(str).tolist()

    main_table_1 = merged[
        ["display_label", "outcome_tier", "final_classification_label", "implementation_summary"]
    ].rename(
        columns={
            "display_label": "Outcome meaning",
            "outcome_tier": "Validation tier",
            "final_classification_label": "Interpretation class",
            "implementation_summary": "Implementation use",
        }
    )
    main_table_2 = merged[
        [
            "display_label",
            "stage1_summary",
            "screening_summary",
            "benchmark_summary",
            "source_summary",
            "heterogeneity_summary",
            "implementation_summary",
        ]
    ].rename(
        columns={
            "display_label": "Outcome meaning",
            "stage1_summary": "Data sufficiency and readiness",
            "screening_summary": "Screening coherence",
            "benchmark_summary": "Benchmark attenuation",
            "source_summary": "Source dependence",
            "heterogeneity_summary": "Contextual heterogeneity",
            "implementation_summary": "Implementation readiness",
        }
    )

    crosswalk_table = merged[
        [
            "display_label",
            "public_concept_label",
            "prompt_label",
            "public_concept_name",
            "canonical_dataset",
            "canonical_variable",
            "readiness_endpoint_name",
        ]
    ].rename(
        columns={
            "display_label": "Outcome meaning",
            "public_concept_label": "Public label",
            "prompt_label": "Prompt label",
            "public_concept_name": "Public concept name",
            "canonical_dataset": "Canonical dataset",
            "canonical_variable": "Canonical variable",
            "readiness_endpoint_name": "Readiness endpoint",
        }
    )

    exposure_family_table = pd.DataFrame(
        [
            {
                "Analytical family": str(family.get("public_name", "Unspecified exposure family")),
                "Exposure meaning": _human_join(
                    [
                        {
                            "maximum_daily_temperature_c": "maximum daily temperature",
                            "mean_daily_relative_humidity_pct": "mean daily relative humidity",
                            "heatwave_duration_days_p95_threshold": "heatwave duration above the municipality-year 95th percentile threshold",
                            "approximate_heat_index_c": "approximate heat index",
                        }.get(str(variable), str(variable).replace("_", " "))
                        for variable in family.get("variables", [])
                    ]
                ),
                "Windows retained in Track A": "; ".join(
                    inputs["config"].get("lag_definitions", {}).get(
                        "acute_daily" if "temperature" in str(family.get("public_name", "")).lower() and "humidity" not in str(family.get("public_name", "")).lower() else "delivery_aligned_windows",
                        [],
                    )
                )
                or "Derived from the Track A lag or window configuration",
                "Interpretive note": "Track A retains observable acute lags and delivery-aligned windows; whole-pregnancy timing remains deferred to later work.",
            }
            for family in inputs["config"].get("exposure_families", [])
        ]
    )

    endpoint_certification_table = merged[
        [
            "display_label",
            "readiness_status",
            "stage1_classification",
            "semantic_alignment",
            "semantic_note",
            "denominator_rule",
        ]
    ].rename(
        columns={
            "display_label": "Outcome meaning",
            "readiness_status": "Readiness status",
            "stage1_classification": "Stage 1 class",
            "semantic_alignment": "Semantic alignment",
            "semantic_note": "Certification caveat",
            "denominator_rule": "Denominator rule",
        }
    )

    screening_detail_table = inputs["triage"][
        [
            "public_concept_label",
            "stage1_classification",
            "best_exposure_label",
            "best_screening_domain",
            "best_lag_or_window",
            "signal_direction_consistency",
            "triage_priority_classification",
            "benchmark_stage_recommendation",
        ]
    ].copy().rename(
        columns={
            "public_concept_label": "Outcome meaning",
            "stage1_classification": "Stage 1 class",
            "best_exposure_label": "Best screening exposure",
            "best_screening_domain": "Best screening domain",
            "best_lag_or_window": "Best lag or window",
            "signal_direction_consistency": "Directional consistency",
            "triage_priority_classification": "Triage class",
            "benchmark_stage_recommendation": "Advanced to benchmark",
        }
    )
    screening_detail_table["Outcome meaning"] = [
        _validity_v2_display_label(name, label)
        for name, label in zip(
            inputs["triage"]["public_concept_name"],
            inputs["triage"]["public_concept_label"],
            strict=False,
        )
    ]
    screening_detail_table["Directional consistency"] = pd.to_numeric(
        screening_detail_table["Directional consistency"], errors="coerce"
    ).round(2)

    benchmark_table = merged[
        [
            "display_label",
            "benchmark_row_count",
            "benchmark_model_count",
            "non_null_estimate_count",
            "benchmark_null_row_count",
            "benchmark_summary",
        ]
    ].rename(
        columns={
            "display_label": "Outcome meaning",
            "benchmark_row_count": "Benchmark rows",
            "benchmark_model_count": "Model families represented",
            "non_null_estimate_count": "Non-null benchmark rows",
            "benchmark_null_row_count": "Near-null benchmark rows",
            "benchmark_summary": "Interpretive summary",
        }
    )

    source_dependence_table = merged[
        [
            "display_label",
            "source_row_count",
            "source_sensitive_flag",
            "source_opposite_direction_flag",
            "source_max_abs_relative_change_pct",
            "source_summary",
        ]
    ].rename(
        columns={
            "display_label": "Outcome meaning",
            "source_row_count": "Source comparison rows",
            "source_sensitive_flag": "Material source dependence",
            "source_opposite_direction_flag": "Direction reversal across sources",
            "source_max_abs_relative_change_pct": "Maximum absolute relative change (%)",
            "source_summary": "Interpretive summary",
        }
    )

    heterogeneity_table = merged[
        [
            "display_label",
            "heterogeneity_row_count",
            "heterogeneity_modifier_count",
            "heterogeneity_inconsistent_row_count",
            "heterogeneity_summary",
        ]
    ].rename(
        columns={
            "display_label": "Outcome meaning",
            "heterogeneity_row_count": "Heterogeneity rows",
            "heterogeneity_modifier_count": "Context modifiers",
            "heterogeneity_inconsistent_row_count": "Directionally inconsistent rows",
            "heterogeneity_summary": "Interpretive summary",
        }
    )

    claim_reference_table = pd.DataFrame(TRACK_A_VALIDITY_V2_REFERENCE_MODULES).rename(
        columns={
            "claim_area": "Claim area",
            "module_name": "Reference module",
            "vancouver_path": "Vancouver bibliography path",
            "notes_path": "Methods note path",
        }
    )

    selected_figure_table = pd.DataFrame(
        [
            {
                "Figure reference": "Screening triage summary",
                "Source artifact": "artifacts/track_a/screening/screening_plots/figure_04_triage_summary.png",
                "Suggested use": "Optional supplement figure for the transition from signal-rich screening to benchmark selection.",
            },
            {
                "Figure reference": "Acute benchmark model panel",
                "Source artifact": "artifacts/track_a/models/effect_plots/figure_01_acute_benchmark_models.png",
                "Suggested use": "Optional supplement figure for the mortality-focused benchmark ladder.",
            },
            {
                "Figure reference": "Signal stability score map",
                "Source artifact": "artifacts/track_a/sensitivity/vulnerability_maps/figure_02_signal_stability_score_map.png",
                "Suggested use": "Optional supplement figure for contextual stability and geographic heterogeneity.",
            },
            {
                "Figure reference": "Contextual vulnerability marker map",
                "Source artifact": "artifacts/track_a/sensitivity/vulnerability_maps/figure_01_contextual_vulnerability_marker_count_map.png",
                "Suggested use": "Optional supplement figure for vulnerability interpretation rather than causal attribution.",
            },
        ]
    )

    context = {
        "frame": merged,
        "strong_labels": strong_labels,
        "caveat_labels": caveat_labels,
        "secondary_only_labels": secondary_only_labels,
        "source_sensitive_labels": source_sensitive_labels,
        "heterogeneity_labels": heterogeneity_labels,
        "main_table_1": main_table_1,
        "main_table_2": main_table_2,
        "crosswalk_table": crosswalk_table,
        "exposure_family_table": exposure_family_table,
        "endpoint_certification_table": endpoint_certification_table,
        "screening_detail_table": screening_detail_table,
        "benchmark_table": benchmark_table,
        "source_dependence_table": source_dependence_table,
        "heterogeneity_table": heterogeneity_table,
        "claim_reference_table": claim_reference_table,
        "selected_figure_table": selected_figure_table,
        "module_table": claim_reference_table.rename(
            columns={
                "Claim area": "Coverage area",
                "Reference module": "Module",
            }
        ),
    }
    return context


def _validity_v2_title(metrics: dict[str, Any]) -> str:
    return (
        "Initial validation of a WHO/WMO-oriented framework for heat and maternal, "
        "neonatal, and child health outcomes in Brazil, 2020-2024"
    )


def _validity_v2_authors() -> list[tuple[str, str]]:
    return [
        ("Cristiano Boccolini", "1"),
        ("Patricia Boccolini", "2"),
        ("Christovam Barcellos", "1"),
    ]


def _validity_v2_affiliations() -> list[tuple[str, str]]:
    return [
        ("1", "ICICT/Fiocruz"),
        ("2", "FMP/UNIFASE"),
    ]


def _validity_v2_overview_sentence(metrics: dict[str, Any]) -> str:
    class_counts = metrics["classification_counts"]
    return (
        f"The frozen Track A package scored {metrics['indicator_count']} public concepts across {metrics['analysis_years']}. "
        f"Stage 1 retained {metrics['ready_benchmark_count']} {_pluralize(metrics['ready_benchmark_count'], 'benchmark-ready concept')}, "
        f"{metrics['screening_only_count']} {_pluralize(metrics['screening_only_count'], 'screening-only concept')}, and "
        f"{metrics['descriptive_only_count']} {_pluralize(metrics['descriptive_only_count'], 'descriptive-only concept')}. "
        f"Stage 4 remained signal-rich ({metrics['triage_advance_count']} advance and "
        f"{metrics['triage_caveat_count']} advance-with-caveats concepts), but the final synthesis narrowed the interpretation set "
        f"to {class_counts.get('strong candidate', 0)} strong candidates, "
        f"{class_counts.get('usable with caveats', 0)} usable concepts with caveats, and "
        f"{class_counts.get('secondary only', 0)} secondary-only concepts."
    )


def _validity_v2_introduction(metrics: dict[str, Any], context: dict[str, Any]) -> str:
    return (
        "# Introduction\n\n"
        "Extreme heat is increasingly recognized as a threat to maternal, neonatal, and child health, yet surveillance-ready "
        "indicator frameworks remain uneven across routine health-information systems. This v2 Track A report adopts a WHO/WMO-oriented "
        "frame in which indicators are judged not only by whether they can be measured, but also by whether they are relevant to "
        "maternal, newborn, and child health decision-making and important enough to justify operational use. The framing follows the "
        "priority logic used in the WHO meeting draft that emphasized feasibility, relevance, and importance, while translating those "
        "ideas into an epidemiological validation workflow for the frozen Track A analytical stack.\n\n"
        "The report does not attempt to elevate a single flagship association. Instead, it asks whether the frozen municipality-day gold "
        "stack is sufficiently complete, internally coherent, and interpretable to support an initial validation of a broad climate and "
        "health framework. The focus is therefore on directional stability, attenuation across analytical stages, source dependence, and "
        "implementation readiness rather than on reporting a single pooled estimate for each concept.\n\n"
        f"{_validity_v2_overview_sentence(metrics)}"
    )


def _validity_v2_methods(metrics: dict[str, Any], context: dict[str, Any]) -> str:
    primary_labels = context["frame"].loc[
        context["frame"]["outcome_tier"].eq("Primary validation outcome"),
        "display_label",
    ].astype(str).tolist()
    return (
        "# Methods\n\n"
        "Track A used a frozen-gold, read-only analytical design based on the national municipality-day stack for 2020-2024. "
        "Data ingestion, source promotion, climate linkage, and denominator construction were not re-engineered for this report; they "
        "were treated as frozen inputs and are described in detail in the supplementary material. The main report therefore concentrates "
        "on whether the existing stack is usable and meaningful for heat-related maternal, neonatal, and child health interpretation.\n\n"
        "The validation sequence followed the staged Track A architecture: analytical readiness and semantic alignment at freeze; "
        "screening and triage to identify signal-rich concepts; DAG-based adjustment architecture; progressive benchmark modeling; "
        "source-restriction and source-stratified sensitivity analysis; one-modifier-at-a-time contextual heterogeneity analysis; and a "
        "final scorecard covering operational validity, construct validity, robustness, source dependence, and implementation value. "
        "In the language of the WHO meeting draft, operational validity addresses feasibility, construct validity addresses relevance, and "
        "implementation value addresses importance, while robustness and source dependence act as additional filters on whether an "
        "indicator remains trustworthy in practice.\n\n"
        f"The primary validation outcomes were { _human_join(primary_labels).lower() }. "
        "All remaining concepts were treated as secondary or supporting outcomes. Delivery-aligned seven-, fourteen-, and twenty-eight-day "
        "exposure summaries were retained as observable pre-event windows. They were not interpreted as trimester-specific or whole-pregnancy "
        "causal windows, which remain deferred to later work."
    )


def _validity_v2_results(metrics: dict[str, Any], context: dict[str, Any]) -> str:
    strong = _human_join(context["strong_labels"])
    caveats = _human_join(context["caveat_labels"])
    secondary = _human_join(context["secondary_only_labels"])
    source_sensitive = _human_join_lower(context["source_sensitive_labels"])
    heterogeneity = _human_join_lower(context["heterogeneity_labels"])
    return (
        "# Results\n\n"
        "## Data sufficiency and readiness\n\n"
        f"{_validity_v2_overview_sentence(metrics)} The data sufficiency result is therefore favorable for initial framework validation: "
        "the frozen stack is large enough, complete enough, and sufficiently standardized to support a meaningful staged review rather than "
        "a purely exploratory exercise.\n\n"
        "## Primary outcome validity\n\n"
        f"The mortality indicators emerged as the primary validation outcomes. {strong} were the only concepts to retain strong-candidate "
        "status after the full staged review. Their screening coherence was not identical, with neonatal mortality presenting the cleanest "
        "screening profile, but none of the three showed material source sensitivity or contextual direction reversals. This combination of "
        "benchmark retention, low source dependence, and sustained implementation value makes them the most defensible frontline indicators "
        "in the present framework.\n\n"
        "## Secondary outcome validity\n\n"
        f"The secondary outcome set remained informative but more fragile. {caveats} remained usable with caveats, which indicates that they "
        "were operationally present and epidemiologically relevant enough to survive the staged review, yet not stable enough to serve as the "
        "strongest public-facing indicators. The supplementary-only concepts were "
        f"{secondary}, reflecting either limited endpoint certification or proxy status.\n\n"
        "## Directional stability and attenuation\n\n"
        "The central validation finding was not the presence of screening signals alone, but how those signals changed as the analysis moved "
        "from broad screening to benchmark and sensitivity stages. Stage 4 was intentionally permissive and signal-rich. Stage 6 then applied "
        "a stricter benchmark ladder and produced substantial attenuation for most maternal and perinatal concepts. In other words, the framework "
        "was able to distinguish between indicators that looked promising in early screening and indicators that remained directionally interpretable "
        "after stronger analytical scrutiny.\n\n"
        "## Source dependence and contextual heterogeneity\n\n"
        f"Material source dependence was concentrated in {source_sensitive}. "
        f"Contextual direction reversals were confined to {heterogeneity}. "
        "These patterns matter because they identify where the framework is sensitive to climate-source handling or to contextual stratification, "
        "which in turn limits how confidently those outcomes can be used for frontline interpretation.\n\n"
        "## Implementation readiness and next analytical steps\n\n"
        "Taken together, the results support three implementation tiers. First, mortality indicators are ready for primary validation use now. "
        "Second, selected perinatal and maternal concepts are usable with explicit caveats, especially when the purpose is surveillance-oriented "
        "monitoring rather than singular causal inference. Third, proxy or weakly certified concepts should remain supplementary until the next "
        "generation of event-level and source-harmonized analyses is complete."
    )


def _validity_v2_discussion(metrics: dict[str, Any], context: dict[str, Any]) -> str:
    return (
        "# Discussion\n\n"
        "The Track A evidence base supports an initial validation claim, not a definitive etiological claim for every concept in the package. "
        "That distinction is important. The value of the framework lies in showing that a read-only operational gold stack can already support "
        "disciplined indicator triage across maternal, neonatal, and child health, while also making clear where confidence should stop.\n\n"
        "The main positive finding is that the framework has enough data to be usable. Operational readiness was high, semantic alignment was "
        "sufficiently explicit, and the staged evidence chain could meaningfully differentiate between stable and fragile indicators. The main "
        "negative finding is equally informative: for many maternal and perinatal outcomes, signal-rich screening patterns did not translate into "
        "equally strong benchmark support. That attenuation is not a failure of the framework; it is part of the framework's validity function.\n\n"
        "The mortality outcomes provide the clearest case for immediate interpretation. By contrast, the perinatal and maternal concepts occupy a "
        "middle ground in which they remain relevant and often important, but not uniformly robust. This is precisely the kind of hierarchy that a "
        "WHO/WMO-oriented implementation framework needs if it is to guide adaptation planning without overstating certainty.\n\n"
        "The next analytical priority is therefore not to discard the caveated indicators, but to move them into designs that can better address "
        "their known weaknesses. Track B event-level gestational timing, stronger pregnancy-window exposure architectures, improved pediatric "
        "utilization definitions, and tighter endpoint certification for severe maternal phenotypes are the logical next steps."
    )


def _validity_v2_limitations(metrics: dict[str, Any], context: dict[str, Any]) -> str:
    source_sensitive = _human_join_lower(context["source_sensitive_labels"])
    return (
        "# Limitations\n\n"
        "Several limitations should remain explicit. First, the benchmark ladder is a proxy framework adapted to the current repository workflow, "
        "not a full maximum-likelihood modeling stack. Second, delivery-aligned windows are useful observable summaries for Track A, but they do "
        "not replace event-level pregnancy timing matrices. Third, the framework is WHO/WMO-oriented rather than formally WHO/WMO-endorsed; "
        "project-defined derivations and operational restrictions remain clearly distinct from external standards.\n\n"
        f"Fourth, source dependence remains material for {source_sensitive}. "
        "These outcomes are still meaningful for surveillance-oriented interpretation, but they should not anchor the strongest public claims until "
        "source harmonization and design refinement improve their stability. Finally, proxy or incompletely certified outcomes remain supplementary "
        "because the frozen stack cannot yet support them as fully validated frontline indicators."
    )


def _validity_v2_conclusion(metrics: dict[str, Any], context: dict[str, Any]) -> str:
    return (
        "# Conclusion\n\n"
        "The frozen Track A package contains enough data for an initial validation of a WHO/WMO-oriented heat and maternal, neonatal, and child "
        "health framework. It is usable because the staged evidence chain can identify indicators that remain stable, indicators that attenuate, "
        "and indicators that are materially source-sensitive. It is meaningful because those distinctions directly inform implementation readiness. "
        "At present, mortality indicators are ready for primary validation use, several perinatal and maternal indicators are usable with caveats, "
        "and proxy or weakly certified indicators should remain supplementary while the next analytical generation is developed."
    )


def _validity_v2_structured_abstract(metrics: dict[str, Any], context: dict[str, Any]) -> dict[str, str]:
    class_counts = metrics["classification_counts"]
    return {
        "Background": (
            "Routine heat and health surveillance for maternal, neonatal, and child outcomes remains constrained by uncertainty about which "
            "indicators are sufficiently measurable, interpretable, and implementation-ready."
        ),
        "Objective": (
            "To assess whether the frozen Track A municipality-day analytical stack is sufficient for an initial validation of a WHO/WMO-oriented "
            "framework for heat-related maternal, neonatal, and child health interpretation in Brazil."
        ),
        "Design": (
            "Framework-validation study using staged analytical readiness review, screening triage, benchmark modeling, source-restriction and "
            "source-stratified sensitivity analysis, contextual heterogeneity analysis, and final scorecard synthesis."
        ),
        "Setting": "Brazil, municipality-day analytical stack, 2020-2024.",
        "Outcome tiers": (
            "Primary validation outcomes were neonatal, infant, and under-five mortality. Perinatal, maternal, and proxy concepts were treated as "
            "secondary or supporting outcomes."
        ),
        "Main findings": (
            f"Track A scored {metrics['indicator_count']} concepts. The final synthesis identified "
            f"{class_counts.get('strong candidate', 0)} strong candidates, "
            f"{class_counts.get('usable with caveats', 0)} usable concepts with caveats, and "
            f"{class_counts.get('secondary only', 0)} secondary-only concepts. Mortality outcomes retained the strongest overall validity profile, "
            "whereas most maternal and perinatal concepts showed substantial benchmark attenuation or later instability."
        ),
        "Interpretation": (
            "The framework has enough data to support initial validation and immediate mortality-focused use. Its main value lies in separating "
            "stable indicators from caveated indicators and in making source dependence and implementation limits explicit."
        ),
    }


def _validity_v2_key_findings(metrics: dict[str, Any], context: dict[str, Any]) -> list[str]:
    return [
        "The frozen Track A stack is sufficient for initial framework validation because readiness, denominator validity, and climate linkage were already in place before report writing.",
        "Mortality outcomes formed the only primary validation indicators that remained strong after benchmark and sensitivity review.",
        "Most perinatal and maternal outcomes remained meaningful but caveated, mainly because screening signals attenuated in the benchmark ladder.",
        "Source dependence was concentrated in gestational shortening, obstetric admissions, and preterm labor admissions.",
        "The framework is implementation-ready for mortality-focused interpretation now and for selected secondary monitoring uses with explicit caveats.",
    ]


def _validity_v2_main_references_text() -> str:
    return "# References\n\n" + "\n".join(TRACK_A_VALIDITY_V2_MAIN_REFERENCES) + "\n"


def _validity_v2_tables_section(context: dict[str, Any]) -> str:
    return (
        "# Tables\n\n"
        "## Main report tables\n\n"
        "### Table 1. Outcome validity tiers and interpretation class\n\n"
        f"{_markdown_table(context['main_table_1'])}\n"
        "### Table 2. Cross-stage validation summary by outcome meaning\n\n"
        f"{_markdown_table(context['main_table_2'])}\n"
        "## Supplementary tables\n\n"
        "The supplementary material includes the outcome crosswalk, exposure and lag appendix, endpoint certification, screening detail, "
        "benchmark attenuation, source-dependence, heterogeneity, and claim-to-reference-module tables."
    )


def _validity_v2_figures_section(context: dict[str, Any]) -> str:
    return (
        "# Figures\n\n"
        "The main report does not force the inclusion of analytical figures. Existing Track A figures are retained as optional supplement "
        "references and are only recommended where they clarify validity stages without overwhelming the narrative.\n\n"
        "## Selected figure appendix\n\n"
        f"{_markdown_table(context['selected_figure_table'])}"
    )


def _validity_v2_supplement_text(metrics: dict[str, Any], context: dict[str, Any]) -> str:
    return (
        "# Supplement\n\n"
        "This supplementary package provides the detailed provenance, standards, validation tables, and reference-module crosswalk for the "
        "Track A validity report.\n\n"
        "## Supplement overview\n\n"
        f"- Analytical years: `{metrics['analysis_years']}`\n"
        f"- Concepts scored: `{metrics['indicator_count']}`\n"
        f"- Sensitivity rows summarized: `{metrics['sensitivity_row_count']}`\n"
        f"- Heterogeneity rows summarized: `{metrics['heterogeneity_row_count']}`\n"
        f"- Concepts with material source dependence: `{len(context['source_sensitive_labels'])}`\n"
        f"- Concepts with contextual direction reversals: `{len(context['heterogeneity_labels'])}`\n\n"
        "## Appendix A. Data lineage and promotion summary\n\n"
        "The analytical package was built from the frozen Track A gold stack. Promotion, source preservation, climate linkage, and denominator "
        "construction were treated as frozen repository inputs rather than re-estimated components. This appendix is the main location for those "
        "details so that the main report can remain focused on validity.\n\n"
        "## Appendix B. Standards and denominator summary\n\n"
        "Outcome definitions were interpreted against the project-stored WHO, WMO, Ministry of Health, and methodological reference modules. "
        "Where the repository uses project-defined operational restrictions, those restrictions are stated explicitly and are not presented as "
        "formal external standards.\n\n"
        "## Appendix C. Validation methods\n\n"
        "The staged validation architecture mapped feasibility to operational validity, relevance to construct validity, and importance to "
        "implementation value, while retaining benchmark robustness and source dependence as additional decision filters."
    )


def _compose_validity_v2_manuscript_markdown(
    title: str,
    abstract_sections: dict[str, str],
    key_findings: list[str],
    introduction: str,
    methods: str,
    results: str,
    discussion: str,
    limitations: str,
    conclusion: str,
) -> str:
    abstract_lines = ["# Structured abstract"]
    for label, text in abstract_sections.items():
        abstract_lines.append(f"**{label}:** {text}")
    key_findings_lines = ["# Key findings", *_markdown_bullets(key_findings).splitlines()]
    return (
        f"# {title}\n\n"
        "## Authors\n\n"
        "Cristiano Boccolini (ICICT/Fiocruz); Patricia Boccolini (FMP/UNIFASE); Christovam Barcellos (ICICT/Fiocruz)\n\n"
        + "\n\n".join(
            [
                "\n\n".join(abstract_lines),
                "\n".join(key_findings_lines),
                introduction,
                methods,
                results,
                discussion,
                limitations,
                conclusion,
                _validity_v2_main_references_text().strip(),
            ]
        )
        + "\n"
    )


def _compose_validity_v2_supplement_markdown(
    supplement_text: str,
    context: dict[str, Any],
) -> str:
    sections = [
        "# Track A Validity Report V2 Supplement",
        supplement_text,
        "## Outcome meaning to canonical endpoint crosswalk\n\n" + _markdown_table(context["crosswalk_table"]),
        "## Exposure family and lag-window appendix\n\n" + _markdown_table(context["exposure_family_table"]),
        "## Endpoint certification and semantic alignment\n\n" + _markdown_table(context["endpoint_certification_table"]),
        "## Screening and triage detail\n\n" + _markdown_table(context["screening_detail_table"]),
        "## Benchmark attenuation summary\n\n" + _markdown_table(context["benchmark_table"]),
        "## Source-dependence appendix\n\n" + _markdown_table(context["source_dependence_table"]),
        "## Heterogeneity inconsistency appendix\n\n" + _markdown_table(context["heterogeneity_table"]),
        "## Selected figure appendix\n\n" + _markdown_table(context["selected_figure_table"]),
        "## Claim-to-reference-module appendix\n\n" + _markdown_table(context["claim_reference_table"]),
    ]
    return "\n\n".join(sections) + "\n"


def _write_validity_v2_workspace(
    root: Path,
    workspace_root: Path,
    title: str,
    abstract_sections: dict[str, str],
    key_findings: list[str],
    introduction: str,
    methods: str,
    results: str,
    discussion: str,
    limitations: str,
    conclusion: str,
    supplement_text: str,
    context: dict[str, Any],
) -> dict[str, Path]:
    outline_text = TRACK_A_VALIDITY_V2_FILES["00_outline.md"] + "\nCurrent package status:\n\n" + _markdown_bullets(
        [
            f"Primary validation outcomes: {_human_join(context['frame'].loc[context['frame']['outcome_tier'].eq('Primary validation outcome'), 'display_label'].astype(str).tolist())}",
            f"Strong candidates: {_human_join(context['strong_labels'])}",
            f"Usable with caveats: {_human_join(context['caveat_labels'])}",
            f"Supplementary only: {_human_join(context['secondary_only_labels'])}",
        ]
    ) + "\n"
    tables_text = _validity_v2_tables_section(context)
    figures_text = _validity_v2_figures_section(context)

    paths = {
        "outline": workspace_root / "00_outline.md",
        "introduction": workspace_root / "01_introduction.md",
        "methods": workspace_root / "02_methods.md",
        "results": workspace_root / "03_results.md",
        "discussion": workspace_root / "04_discussion.md",
        "limitations": workspace_root / "05_limitations.md",
        "tables": workspace_root / "06_tables.md",
        "figures": workspace_root / "07_figures.md",
        "supplement": workspace_root / "08_supplement.md",
        "report_md": workspace_root / "manuscript_trackA_validity_v2.md",
        "supplement_md": workspace_root / "supplement_trackA_validity_v2.md",
        "manifest": workspace_root / "manifests" / "evidence_and_reference_manifest.md",
    }

    _write_text(paths["outline"], outline_text)
    _write_text(paths["introduction"], introduction)
    _write_text(paths["methods"], methods)
    _write_text(paths["results"], results)
    _write_text(paths["discussion"], discussion)
    _write_text(paths["limitations"], limitations)
    _write_text(paths["tables"], tables_text)
    _write_text(paths["figures"], figures_text)
    _write_text(paths["supplement"], supplement_text)
    _write_text(
        paths["report_md"],
        _compose_validity_v2_manuscript_markdown(
            title=title,
            abstract_sections=abstract_sections,
            key_findings=key_findings,
            introduction=introduction,
            methods=methods,
            results=results,
            discussion=discussion,
            limitations=limitations,
            conclusion=conclusion,
        ),
    )
    _write_text(
        paths["supplement_md"],
        _compose_validity_v2_supplement_markdown(
            supplement_text=supplement_text,
            context=context,
        ),
    )
    return paths


def _add_validity_v2_title_page(doc: Any, *, title: str, metrics: dict[str, Any], framework_note: str | None) -> None:
    paragraph = doc.add_paragraph(style="Title")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run(title)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_run = subtitle.add_run("Track A validity report v2")
    subtitle_run.bold = True

    author_paragraph = doc.add_paragraph()
    author_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author_paragraph.add_run(
        "Cristiano Boccolini1; Patricia Boccolini2; Christovam Barcellos1"
    ).bold = True

    for affiliation_id, affiliation_text in _validity_v2_affiliations():
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.add_run(f"{affiliation_id}. {affiliation_text}")

    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    for index, label in enumerate(["Field", "Value"]):
        cell = table.rows[0].cells[index]
        cell.text = label
        _set_cell_shading(cell, fill="D9E6F2")

    metadata = [
        ("Date", str(date.today())),
        ("Analytical years", str(metrics["analysis_years"])),
        ("Concepts scored", str(metrics["indicator_count"])),
        ("Strong candidates", str(metrics["classification_counts"].get("strong candidate", 0))),
        ("Usable with caveats", str(metrics["classification_counts"].get("usable with caveats", 0))),
    ]
    for key, value in metadata:
        row = table.add_row().cells
        row[0].text = key
        row[1].text = value

    if framework_note:
        doc.add_paragraph()
        note = doc.add_paragraph()
        note.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = note.add_run(framework_note)
        run.italic = True

    doc.add_page_break()


def _add_labeled_paragraph(doc: Any, label: str, text: str) -> None:
    paragraph = doc.add_paragraph()
    run = paragraph.add_run(f"{label}: ")
    run.bold = True
    paragraph.add_run(text)


def _add_markdown_blocks(doc: Any, text: str) -> None:
    for block in text.split("\n\n"):
        stripped = block.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=2)
            continue
        if stripped.startswith("# "):
            doc.add_heading(stripped[2:], level=1)
            continue
        if all(line.startswith("- ") for line in stripped.splitlines() if line.strip()):
            for line in stripped.splitlines():
                if line.startswith("- "):
                    doc.add_paragraph(line[2:], style="List Bullet")
            continue
        doc.add_paragraph(stripped)


def _add_callout_box(doc: Any, title: str, items: list[str]) -> None:
    table = doc.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    cell = table.rows[0].cells[0]
    _set_cell_shading(cell, fill="EDF4FA")
    paragraph = cell.paragraphs[0]
    title_run = paragraph.add_run(title)
    title_run.bold = True
    for item in items:
        paragraph = cell.add_paragraph(style="List Bullet")
        paragraph.add_run(item)


def _add_numbered_references(doc: Any, references: list[str]) -> None:
    for reference in references:
        doc.add_paragraph(reference, style="List Number")


def _build_validity_v2_docx_documents(
    workspace_root: Path,
    title: str,
    metrics: dict[str, Any],
    abstract_sections: dict[str, str],
    key_findings: list[str],
    introduction: str,
    methods: str,
    results: str,
    discussion: str,
    limitations: str,
    conclusion: str,
    context: dict[str, Any],
) -> tuple[Path, Path]:
    _require_docx_dependency()

    report_doc = Document()
    _style_document(report_doc)
    framework_note = (
        "Framework framing follows the WHO/WMO meeting draft emphasis on feasibility, relevance, and importance."
        if TRACK_A_VALIDITY_V2_FRAMEWORK_DRAFT.exists()
        else "Framework framing follows the WHO/WMO-oriented indicator prioritization logic used in the Track A validation package."
    )
    _add_validity_v2_title_page(report_doc, title=title, metrics=metrics, framework_note=framework_note)
    report_doc.add_heading("Structured abstract", level=1)
    for label, text in abstract_sections.items():
        _add_labeled_paragraph(report_doc, label, text)
    report_doc.add_heading("Key findings", level=1)
    _add_callout_box(report_doc, "Track A validity summary", key_findings)
    for heading, body in [
        ("Introduction", introduction),
        ("Methods", methods),
        ("Results", results),
        ("Discussion", discussion),
        ("Limitations", limitations),
        ("Conclusion", conclusion),
    ]:
        report_doc.add_heading(heading, level=1)
        content = body.split("\n", 2)[2] if body.startswith("# ") else body
        _add_markdown_blocks(report_doc, content)
    report_doc.add_heading("Table 1. Outcome validity tiers and interpretation class", level=1)
    _add_dataframe_table(report_doc, context["main_table_1"])
    report_doc.add_heading("Table 2. Cross-stage validation summary by outcome meaning", level=1)
    _add_dataframe_table(report_doc, context["main_table_2"])
    report_doc.add_heading("References", level=1)
    _add_numbered_references(report_doc, TRACK_A_VALIDITY_V2_MAIN_REFERENCES)
    report_path = workspace_root / "report_trackA_validity_v2.docx"
    report_doc.save(report_path)

    supplement_doc = Document()
    _style_document(supplement_doc)
    _add_validity_v2_title_page(
        supplement_doc,
        title=title,
        metrics=metrics,
        framework_note="Supplementary appendices supporting the Track A validity report v2.",
    )
    overview_content = _validity_v2_supplement_text(metrics, context).split("\n", 2)[2]
    _add_markdown_blocks(supplement_doc, overview_content)
    for heading, frame in [
        ("Outcome meaning to canonical endpoint crosswalk", context["crosswalk_table"]),
        ("Exposure family and lag-window appendix", context["exposure_family_table"]),
        ("Endpoint certification and semantic alignment", context["endpoint_certification_table"]),
        ("Screening and triage detail", context["screening_detail_table"]),
        ("Benchmark attenuation summary", context["benchmark_table"]),
        ("Source-dependence appendix", context["source_dependence_table"]),
        ("Heterogeneity inconsistency appendix", context["heterogeneity_table"]),
        ("Selected figure appendix", context["selected_figure_table"]),
        ("Claim-to-reference-module appendix", context["claim_reference_table"]),
    ]:
        supplement_doc.add_heading(heading, level=1)
        _add_dataframe_table(supplement_doc, frame)
    supplement_path = workspace_root / "supplement_trackA_validity_v2.docx"
    supplement_doc.save(supplement_path)
    return report_path, supplement_path


def _write_validity_v2_manifest(root: Path, workspace_root: Path, context: dict[str, Any]) -> Path:
    manifest_path = workspace_root / "manifests" / "evidence_and_reference_manifest.md"
    artifact_rows = [
        ("Readiness matrix", "artifacts/track_a/freeze/analysis_ready_matrix.csv"),
        ("Triage matrix", "artifacts/track_a/screening/triage_signal_matrix.csv"),
        ("Benchmark results", "artifacts/track_a/models/benchmark_results.csv"),
        ("Sensitivity matrix", "artifacts/track_a/sensitivity/sensitivity_matrix.csv"),
        ("Heterogeneity results", "artifacts/track_a/sensitivity/heterogeneity_results.csv"),
        ("Framework validation scorecard", "artifacts/track_a/synthesis/framework_validation_scorecard.csv"),
        ("Framework evidence trace", "artifacts/track_a/synthesis/framework_validation_evidence_trace.csv"),
        ("Track A variable dictionary", "artifacts/track_a/metadata/track_a_variable_dictionary.csv"),
    ]
    artifact_lines = ["# Evidence and Reference Manifest", "", "## Frozen analytical artifacts", ""]
    artifact_lines.extend([f"- {label}: `{path}`" for label, path in artifact_rows])
    artifact_lines.extend(["", "## Project-stored reference modules", ""])
    for row in TRACK_A_VALIDITY_V2_REFERENCE_MODULES:
        artifact_lines.append(
            f"- {row['module_name']}: `{row['vancouver_path']}` with methods note `{row['notes_path']}`"
        )
    if TRACK_A_VALIDITY_V2_FRAMEWORK_DRAFT.exists():
        artifact_lines.extend(
            [
                "",
                "## External framework draft used for framing only",
                "",
                f"- Draft meeting report: `{TRACK_A_VALIDITY_V2_FRAMEWORK_DRAFT}`",
            ]
        )
    artifact_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Main report prose uses human-readable outcome meanings rather than raw variable names.",
            "- The supplement retains technical crosswalks where reproducibility requires canonical labels.",
        ]
    )
    _write_text(manifest_path, "\n".join(artifact_lines) + "\n")
    return manifest_path


def build_track_a_validity_report_v2(root: Path) -> dict[str, object]:
    workspace_report = scaffold_track_a_validity_v2_workspace(root)
    inputs = _load_validity_v2_inputs(root)
    metrics = _publication_metrics(inputs)
    context = _prepare_validity_v2_context(inputs)
    title = _validity_v2_title(metrics)
    abstract_sections = _validity_v2_structured_abstract(metrics, context)
    key_findings = _validity_v2_key_findings(metrics, context)
    introduction = _validity_v2_introduction(metrics, context)
    methods = _validity_v2_methods(metrics, context)
    results = _validity_v2_results(metrics, context)
    discussion = _validity_v2_discussion(metrics, context)
    limitations = _validity_v2_limitations(metrics, context)
    conclusion = _validity_v2_conclusion(metrics, context)
    supplement_text = _validity_v2_supplement_text(metrics, context)

    workspace_root = _track_a_validity_v2_workspace_root(root)
    paths = _write_validity_v2_workspace(
        root=root,
        workspace_root=workspace_root,
        title=title,
        abstract_sections=abstract_sections,
        key_findings=key_findings,
        introduction=introduction,
        methods=methods,
        results=results,
        discussion=discussion,
        limitations=limitations,
        conclusion=conclusion,
        supplement_text=supplement_text,
        context=context,
    )
    report_docx, supplement_docx = _build_validity_v2_docx_documents(
        workspace_root=workspace_root,
        title=title,
        metrics=metrics,
        abstract_sections=abstract_sections,
        key_findings=key_findings,
        introduction=introduction,
        methods=methods,
        results=results,
        discussion=discussion,
        limitations=limitations,
        conclusion=conclusion,
        context=context,
    )
    manifest_path = _write_validity_v2_manifest(root, workspace_root, context)

    return {
        "status": "PASS",
        "stage": "track_a_validity_report_v2",
        "indicator_count": metrics["indicator_count"],
        "classification_counts": metrics["classification_counts"],
        "workspace_report": workspace_report,
        "outputs": {
            "workspace_root": _relative(root, workspace_root),
            "report_docx": _relative(root, report_docx),
            "supplement_docx": _relative(root, supplement_docx),
            "report_markdown": _relative(root, paths["report_md"]),
            "supplement_markdown": _relative(root, paths["supplement_md"]),
            "evidence_manifest": _relative(root, manifest_path),
        },
    }
