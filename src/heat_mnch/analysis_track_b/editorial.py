from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_track_b_config, load_track_b_editorial_freeze_config
from .registry import ensure_track_b_directory_structure, track_b_publication_root, track_b_root


TRACK_B_PUBLICATION_FILES = {
    "README.md": """# Track B Flagship Perinatal Paper

This workspace stores the Track B flagship manuscript package for The Lancet Planetary Health.

Track B reuses the validated shared core from Track A and extends it into an event-level perinatal flagship focused on preterm birth, stillbirth, neonatal mortality, burden, and inequity.
""",
    "00_outline.md": """# Outline

1. Problem statement and planetary-health relevance
2. Why maternal-newborn heat risks matter in Brazil now
3. Co-primary outcomes and flagship question
4. National evidence on preterm birth, stillbirth, and neonatal mortality
5. Absolute burden
6. Inequity and vulnerability
7. Policy implications
8. Supplement
""",
    "01_introduction.md": """# Introduction

Use this file to develop the flagship planetary-health framing for Track B.
""",
    "02_methods.md": """# Methods

Use this file to document the event-level exposure matrix, flagship model hierarchy, burden design, and inequity package for Track B.
""",
    "03_results.md": """# Results

Use this file to summarize the three co-primary outcome packages, burden estimates, and inequity findings.
""",
    "04_discussion.md": """# Discussion

Use this file to keep the Track B discussion focused on perinatal risk, burden, and adaptation relevance.
""",
    "05_limitations.md": """# Limitations

Use this file to document observability limits, source dependence, and event-level adjustment constraints for Track B.
""",
    "06_tables.md": """# Tables

List the locked main-text and supplement tables for the Track B manuscript here.
""",
    "07_figures.md": """# Figures

List the locked main-text and supplement figures for the Track B manuscript here.
""",
    "08_supplement.md": """# Supplement

Track supplement contents here, including the event-level matrix, burden appendices, and inequity results.
""",
    "tables/README.md": """# Tables

Store Track B table shells and manuscript-ready exports in this directory.
""",
    "figures/README.md": """# Figures

Store Track B figure shells and manuscript-ready exports in this directory.
""",
    "submission/README.md": """# Submission

Store journal-facing package files for the Track B flagship in this directory.
""",
}


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _render_protocol(track_b_config: dict[str, Any], editorial_freeze: dict[str, Any]) -> str:
    protocol = editorial_freeze["flagship_protocol"]
    co_primary = protocol["co_primary_outcomes"]
    supporting = protocol["supporting_secondary_outcomes"]
    primary_estimates = protocol["primary_main_text_estimates"]
    lines = [
        "# Track B Flagship Protocol",
        "",
        f"- Analysis identifier: `{track_b_config['analysis_id']}`",
        f"- Journal target: `{protocol['journal_target']}`",
        f"- Analytical years: `{', '.join(str(year) for year in track_b_config['years'])}`",
        "",
        "## Flagship Question",
        "",
        protocol["flagship_question"],
        "",
        "## Outcome Hierarchy",
        "",
        "- Co-primary outcomes:",
    ]
    lines.extend([f"  - `{outcome}`" for outcome in co_primary])
    lines.extend(
        [
            "- Supporting secondary outcomes:",
        ]
    )
    lines.extend([f"  - `{outcome}`" for outcome in supporting])
    lines.extend(
        [
            "",
            "## Primary Main-Text Estimates",
            "",
        ]
    )
    for outcome_name, estimate_name in primary_estimates.items():
        lines.append(f"- `{outcome_name}` -> `{estimate_name}`")
    lines.extend(
        [
            "",
            "## Exposure Hierarchy",
            "",
            f"- Primary family: `{protocol['primary_exposure_family']}`",
            "- Secondary families:",
        ]
    )
    lines.extend([f"  - `{family}`" for family in protocol["secondary_exposure_families"]])
    lines.extend(
        [
            "",
            "## Locked Main-Text Package",
            "",
            f"- Main tables: `{protocol['main_text_table_count']}`",
            f"- Main figures: `{protocol['main_text_figure_count']}`",
            "- Includes a Research in Context panel.",
        ]
    )
    return "\n".join(lines) + "\n"


def _research_in_context_panel(protocol: dict[str, Any]) -> str:
    return (
        "# Research in Context\n\n"
        "## Evidence before this study\n\n"
        "Prior heat and perinatal studies have often focused on single outcomes, limited geographies, or exposure windows that were not jointly designed to support burden and inequity interpretation.\n\n"
        "## Added value of this study\n\n"
        "This Track B flagship combines a national event-level gestational exposure matrix with co-primary perinatal outcomes, an acute neonatal arm, attributable burden estimates, and explicit vulnerability stratification.\n\n"
        "## Implications of all available evidence\n\n"
        "If the final Track B estimates remain directionally coherent, the paper will support adaptation-focused interpretation of maternal-newborn heat risks in Brazil rather than a pipeline-validation claim.\n"
    )


def _table_shells(editorial_freeze: dict[str, Any]) -> pd.DataFrame:
    protocol = editorial_freeze["flagship_protocol"]
    return pd.DataFrame(
        [
            {
                "table_id": "Table 1",
                "placement": "Main text",
                "title": "Editorial freeze of co-primary outcomes, exposure hierarchy, and flagship estimands",
                "source_artifact": "reports/publications/track_b_lanplh_flagship_perinatal/flagship_protocol.md",
            },
            {
                "table_id": "Table 2",
                "placement": "Main text",
                "title": "Primary flagship estimates for preterm birth, stillbirth, and neonatal mortality",
                "source_artifact": "artifacts/track_b/models/",
            },
            {
                "table_id": "Table 3",
                "placement": "Main text",
                "title": "Attributable burden by year and macro-region",
                "source_artifact": "artifacts/track_b/burden/flagship_burden_estimates.xlsx",
            },
            {
                "table_id": "Table 4",
                "placement": "Main text",
                "title": "Inequity and vulnerability summary across co-primary outcomes",
                "source_artifact": "artifacts/track_b/inequity/flagship_inequity_results.csv",
            },
            {
                "table_id": "Supplementary Table S1",
                "placement": "Supplement",
                "title": "Event-level gestational exposure matrix specification and QC summary",
                "source_artifact": "artifacts/track_b/event_level/window_qc_report.md",
            },
            {
                "table_id": "Supplementary Table S2",
                "placement": "Supplement",
                "title": "Robustness matrix across outcome families",
                "source_artifact": "artifacts/track_b/robustness/flagship_robustness_matrix.csv",
            },
            {
                "table_id": "Supplementary Table S3",
                "placement": "Supplement",
                "title": "Reporting package crosswalk",
                "source_artifact": "reports/publications/track_b_lanplh_flagship_perinatal/submission/",
            },
        ]
    )


def _figure_shells(editorial_freeze: dict[str, Any]) -> str:
    protocol = editorial_freeze["flagship_protocol"]
    return (
        "# Track B Figure Shells\n\n"
        f"- Main-text figures locked: `{protocol['main_text_figure_count']}`\n"
        "- Figure 1. Event-level exposure matrix and study architecture\n"
        "- Figure 2. Primary flagship estimates across the three co-primary outcomes\n"
        "- Figure 3. Attributable burden by year and macro-region\n"
        "- Figure 4. Inequity and vulnerability summary map or panel\n"
    )


def _scaffold_publication_workspace(root: Path) -> dict[str, object]:
    workspace_root = track_b_publication_root(root)
    created_files: list[str] = []
    existing_files: list[str] = []
    workspace_root.mkdir(parents=True, exist_ok=True)
    for relative_path, contents in TRACK_B_PUBLICATION_FILES.items():
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


def build_track_b_editorial_freeze(root: Path, config_path: Path | None = None) -> dict[str, object]:
    ensure_track_b_directory_structure(root)
    workspace_report = _scaffold_publication_workspace(root)
    track_b_config = load_track_b_config(root, config_path=config_path)
    editorial_freeze = load_track_b_editorial_freeze_config(root)

    artifact_root = track_b_root(root) / "editorial_freeze"
    artifact_root.mkdir(parents=True, exist_ok=True)
    publication_root = track_b_publication_root(root)

    protocol_path = publication_root / "flagship_protocol.md"
    table_shells_path = publication_root / "tables" / "flagship_table_shells.xlsx"
    figure_shells_path = publication_root / "figures" / "flagship_figure_shells.md"
    research_panel_path = publication_root / "research_in_context_panel.md"
    registry_note_path = artifact_root / "editorial_freeze_report.md"

    protocol_path.write_text(_render_protocol(track_b_config, editorial_freeze), encoding="utf-8")
    research_panel_path.write_text(
        _research_in_context_panel(editorial_freeze["flagship_protocol"]),
        encoding="utf-8",
    )
    _table_shells(editorial_freeze).to_excel(table_shells_path, index=False)
    figure_shells_path.write_text(_figure_shells(editorial_freeze), encoding="utf-8")
    registry_note_path.write_text(
        "# Track B Editorial Freeze Report\n\n"
        f"- Co-primary outcomes locked: `{len(editorial_freeze['flagship_protocol']['co_primary_outcomes'])}`\n"
        f"- Supporting secondary outcomes locked: `{len(editorial_freeze['flagship_protocol']['supporting_secondary_outcomes'])}`\n"
        f"- Primary exposure family: `{editorial_freeze['flagship_protocol']['primary_exposure_family']}`\n"
        f"- Main-text tables locked: `{editorial_freeze['flagship_protocol']['main_text_table_count']}`\n"
        f"- Main-text figures locked: `{editorial_freeze['flagship_protocol']['main_text_figure_count']}`\n",
        encoding="utf-8",
    )

    return {
        "status": "PASS",
        "stage": "track_b_stage_b0_editorial_freeze",
        "analysis_id": track_b_config["analysis_id"],
        "co_primary_outcome_count": len(editorial_freeze["flagship_protocol"]["co_primary_outcomes"]),
        "supporting_secondary_outcome_count": len(editorial_freeze["flagship_protocol"]["supporting_secondary_outcomes"]),
        "outputs": {
            "flagship_protocol": _relative(root, protocol_path),
            "flagship_table_shells": _relative(root, table_shells_path),
            "flagship_figure_shells": _relative(root, figure_shells_path),
            "research_in_context_panel": _relative(root, research_panel_path),
            "editorial_freeze_report": _relative(root, registry_note_path),
        },
        "workspace_report": workspace_report,
    }
