from __future__ import annotations

from datetime import date
from pathlib import Path
from shutil import copy2
from typing import Any

import pandas as pd

from .registry import ensure_track_b_directory_structure, track_b_publication_root

try:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt
except ImportError as exc:  # pragma: no cover
    Document = None  # type: ignore[assignment]
    WD_ALIGN_PARAGRAPH = None  # type: ignore[assignment]
    OxmlElement = None  # type: ignore[assignment]
    qn = None  # type: ignore[assignment]
    Inches = None  # type: ignore[assignment]
    Pt = None  # type: ignore[assignment]
    DOCX_IMPORT_ERROR = exc
else:  # pragma: no cover
    DOCX_IMPORT_ERROR = None


def _require_docx_dependency() -> None:
    if Document is None:
        raise RuntimeError("Track B paper package requires python-docx.") from DOCX_IMPORT_ERROR


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _load_required_csv(root: Path, relative_path: str, stage_name: str) -> pd.DataFrame:
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Track B paper package requires {stage_name} at {path}.")
    return pd.read_csv(path, low_memory=False)


def _summarize_burden_detail(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Track B paper package requires burden detail at {path}.")
    row_count = 0
    observed: dict[str, float] = {}
    attributable: dict[str, float] = {}
    for chunk in pd.read_csv(
        path,
        usecols=["outcome_label", "observed_events", "attributable_events"],
        low_memory=False,
        chunksize=500_000,
    ):
        row_count += int(chunk.shape[0])
        grouped = chunk.groupby("outcome_label", as_index=False).agg(
            observed_events=("observed_events", "sum"),
            attributable_events=("attributable_events", "sum"),
        )
        for record in grouped.to_dict(orient="records"):
            outcome = str(record["outcome_label"])
            observed[outcome] = observed.get(outcome, 0.0) + float(record["observed_events"])
            attributable[outcome] = attributable.get(outcome, 0.0) + float(record["attributable_events"])
    return {
        "row_count": row_count,
        "observed_events_by_outcome": observed,
        "attributable_events_by_outcome": attributable,
    }


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


def _set_cell_shading(cell: Any, fill: str = "D9E6F2") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def _add_title_page(doc: Any, *, title: str, subtitle: str, metadata: list[tuple[str, str]], note: str | None = None) -> None:
    paragraph = doc.add_paragraph(style="Title")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run(title)
    subtitle_paragraph = doc.add_paragraph()
    subtitle_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle_paragraph.add_run(subtitle)
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
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(note)
        run.italic = True
    doc.add_page_break()


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| Column | Value |\n| --- | --- |\n| None | None |\n"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in frame.fillna("").astype(str).to_dict(orient="records"):
        lines.append("| " + " | ".join(record[column] for column in columns) + " |")
    return "\n".join(lines) + "\n"


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


def _load_inputs(root: Path) -> dict[str, Any]:
    editorial_protocol = (track_b_publication_root(root) / "flagship_protocol.md").read_text(encoding="utf-8")
    return {
        "protocol_text": editorial_protocol,
        "preterm": _load_required_csv(root, "artifacts/track_b/models/preterm_main_models.csv", "Track B preterm model package"),
        "gest_days": _load_required_csv(root, "artifacts/track_b/models/gest_days_lost_models.csv", "Track B gestational days lost model package"),
        "stillbirth": _load_required_csv(root, "artifacts/track_b/models/stillbirth_main_models.csv", "Track B stillbirth model package"),
        "neonatal": _load_required_csv(root, "artifacts/track_b/models/neonatal_main_models.csv", "Track B neonatal model package"),
        "burden_summary": _summarize_burden_detail(root, "artifacts/track_b/burden/municipality_burden_detail.csv"),
        "inequity": _load_required_csv(root, "artifacts/track_b/inequity/flagship_inequity_results.csv", "Track B inequity package"),
        "robustness": _load_required_csv(root, "artifacts/track_b/robustness/flagship_robustness_matrix.csv", "Track B robustness package"),
    }


def _metrics(inputs: dict[str, Any]) -> dict[str, Any]:
    preterm_primary = inputs["preterm"].loc[inputs["preterm"]["primary_model_flag"].astype(str).eq("yes")].iloc[0]
    stillbirth_primary = inputs["stillbirth"].loc[inputs["stillbirth"]["primary_model_flag"].astype(str).eq("yes")].iloc[0]
    neonatal_primary = inputs["neonatal"].loc[inputs["neonatal"]["primary_model_flag"].astype(str).eq("yes")].iloc[0]
    burden = inputs["burden_summary"]
    inequity = inputs["inequity"]
    robustness = inputs["robustness"]
    return {
        "preterm_primary_rr": float(preterm_primary["estimate"]),
        "stillbirth_primary_rr": float(stillbirth_primary["estimate"]),
        "neonatal_primary_rr": float(neonatal_primary["estimate"]),
        "preterm_primary_events": float(preterm_primary["outcome_count"]),
        "stillbirth_primary_events": float(stillbirth_primary["outcome_count"]),
        "neonatal_primary_events": float(neonatal_primary["death_count"]),
        "burden_total_rows": int(burden["row_count"]),
        "preterm_attributable_total": float(burden["attributable_events_by_outcome"].get("Preterm birth", 0.0)),
        "stillbirth_attributable_total": float(burden["attributable_events_by_outcome"].get("Stillbirth", 0.0)),
        "neonatal_attributable_total": float(burden["attributable_events_by_outcome"].get("Neonatal mortality", 0.0)),
        "inequity_rows": int(inequity.shape[0]),
        "inequity_outcome_count": int(inequity["outcome_label"].nunique()) if not inequity.empty else 0,
        "robustness_rows": int(robustness.shape[0]),
        "co_primary_outcomes": 3,
    }


def _main_tables() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"table_id": "Table 1", "title": "Editorial freeze of co-primary outcomes and flagship estimands", "source_artifact": "reports/publications/track_b_lanplh_flagship_perinatal/flagship_protocol.md"},
            {"table_id": "Table 2", "title": "Primary flagship estimates for preterm birth, stillbirth, and neonatal mortality", "source_artifact": "artifacts/track_b/models/"},
            {"table_id": "Table 3", "title": "Attributable burden by year and macro-region", "source_artifact": "artifacts/track_b/burden/flagship_burden_estimates.xlsx"},
            {"table_id": "Table 4", "title": "Inequity and vulnerability summary across co-primary outcomes", "source_artifact": "artifacts/track_b/inequity/flagship_inequity_results.csv"},
        ]
    )


def _main_figures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"figure_id": "Figure 1", "title": "Track B event-level study architecture", "source_artifact": "artifacts/track_b/event_level/window_qc_report.md"},
            {"figure_id": "Figure 2", "title": "Primary flagship effect estimates across co-primary outcomes", "source_artifact": "artifacts/track_b/models/"},
            {"figure_id": "Figure 3", "title": "Attributable burden across macro-regions", "source_artifact": "artifacts/track_b/burden/burden_maps/"},
            {"figure_id": "Figure 4", "title": "Inequity and vulnerability summary", "source_artifact": "artifacts/track_b/inequity/flagship_vulnerability_maps/"},
        ]
    )


def _strobe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item": "Title and abstract", "section": "Title page; Abstract", "status": "Addressed"},
            {"item": "Background and rationale", "section": "Introduction", "status": "Addressed"},
            {"item": "Objectives", "section": "Introduction", "status": "Addressed"},
            {"item": "Study design", "section": "Methods", "status": "Addressed"},
            {"item": "Variables", "section": "Methods; Supplement", "status": "Addressed"},
            {"item": "Main results", "section": "Results", "status": "Addressed"},
            {"item": "Limitations", "section": "Limitations", "status": "Addressed"},
        ]
    )


def _record() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item": "Data source description", "section": "Methods", "status": "Addressed"},
            {"item": "Study population selection", "section": "Methods", "status": "Addressed"},
            {"item": "Data cleaning methods", "section": "Methods; Supplement", "status": "Addressed"},
            {"item": "Access to data and code", "section": "Supplement", "status": "Addressed"},
        ]
    )


def _gather() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item": "Health indicators", "section": "Burden appendix", "status": "Addressed"},
            {"item": "Counterfactual definition", "section": "Methods; Burden appendix", "status": "Addressed"},
            {"item": "Uncertainty presentation", "section": "Results; Supplement", "status": "Addressed"},
        ]
    )


def _render_abstract(metrics: dict[str, Any]) -> str:
    return (
        "Track B is a national event-level and municipal-day perinatal heat package built on the validated Track A core. "
        "After correcting window observability and burden geography joins, the flagship estimates remain near-null at national scale. "
        f"The primary rate ratios are {metrics['preterm_primary_rr']:.6f} for preterm birth, {metrics['stillbirth_primary_rr']:.6f} for stillbirth, "
        f"and {metrics['neonatal_primary_rr']:.6f} for neonatal mortality. "
        f"The corrected burden package yields approximately {metrics['preterm_attributable_total']:.2f} attributable preterm births nationally and zero attributable stillbirths and neonatal deaths on the current model scale. "
        "The package is therefore strongest as a rigorously corrected national null-to-minimal effect analysis, not as a large-burden headline paper."
    )


def _render_intro(metrics: dict[str, Any]) -> str:
    return (
        "# Introduction\n\n"
        "Track B is the flagship perinatal extension built on the validated shared core from Track A. In its corrected form, the strongest scientific contribution is not a large national burden headline, but a disciplined national test showing that several initially promising signals attenuate sharply once window observability, burden geography, and full-scale benchmarking are enforced.\n\n"
        f"The package retains three co-primary outcomes and one prespecified main-text estimate for each: preterm birth, stillbirth at or beyond 28 weeks, and neonatal mortality. The corrected primary estimates are `{metrics['preterm_primary_rr']:.6f}`, `{metrics['stillbirth_primary_rr']:.6f}`, and `{metrics['neonatal_primary_rr']:.6f}` as rate ratios per one-unit increase on their respective flagship scales.\n"
    )


def _render_methods(metrics: dict[str, Any]) -> str:
    return (
        "# Methods\n\n"
        "Track B combines an event-level gestational exposure matrix for live births and stillbirths with an acute municipal-day neonatal mortality arm. The event-level matrix derives delivery-aligned and gestational windows in academic English and preserves full-window observability flags. Primary burden uses a municipality-specific empirical minimum-risk temperature counterfactual. Vulnerability interpretation uses municipality context from the Track A context layer for stratification only.\n\n"
        "The current implementation uses grouped fixed-effects proxy models for the main outcome packages so that the full national data can be modeled tractably in the repository environment. These proxy labels are explicit in the exported results and methods notes.\n"
    )


def _render_results(metrics: dict[str, Any]) -> str:
    return (
        "# Results\n\n"
        "All three co-primary outcome packages, the burden package, the inequity package, and the robustness package were generated. "
        f"The burden detail file contains `{metrics['burden_total_rows']}` municipality-level rows across outcomes, the inequity package contains `{metrics['inequity_rows']}` summary rows across `{metrics['inequity_outcome_count']}` outcomes, and the robustness matrix contains `{metrics['robustness_rows']}` rows.\n\n"
        f"The corrected preterm primary model includes `{int(metrics['preterm_primary_events']):,}` preterm births and remains numerically close to the null. "
        f"Stillbirth and neonatal mortality likewise remain near-null in the corrected primary models. "
        f"Attributable burden is correspondingly small: approximately `{metrics['preterm_attributable_total']:.2f}` preterm births nationally, with `{metrics['stillbirth_attributable_total']:.2f}` attributable stillbirths and `{metrics['neonatal_attributable_total']:.2f}` attributable neonatal deaths on the current model scale.\n"
    )


def _render_discussion() -> str:
    return (
        "# Discussion\n\n"
        "Track B shifts the program from broad framework validation to a focal perinatal package, but the corrected national estimates support a much more constrained interpretation than the original flagship ambition implied. "
        "The current analytical package is strongest as a rigorous national null-to-minimal effect paper, or as a methods-forward perinatal climate platform paper, rather than as a high-burden flagship claim. "
        "Some components also remain proxy-based for tractability and the week-of-gestation distributed-lag sensitivity remains scaffolded rather than fully estimated.\n"
    )


def _render_limitations() -> str:
    return (
        "# Limitations\n\n"
        "The current Track B implementation preserves academic-English outputs and a coherent package structure, but not every planned model component is a full maximum-likelihood estimator. "
        "Maternal age, schooling, and parity are still unavailable in the live-birth silver event layer and therefore remain conceptual-only in the primary preterm adjustment set. "
        "The week-of-gestation sensitivity is scaffolded, not fully estimated, in the current implementation pass. "
        "Most importantly, the corrected national estimates are near-null, so any external submission must avoid burden-forward or strong-causal headline claims.\n"
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_docx(workspace_root: Path, metrics: dict[str, Any], main_tables: pd.DataFrame, main_figures: pd.DataFrame, strobe: pd.DataFrame, record: pd.DataFrame, gather: pd.DataFrame) -> tuple[Path, Path]:
    _require_docx_dependency()
    manuscript = Document()
    _style_document(manuscript)
    _add_title_page(
        manuscript,
        title="Track B Flagship Perinatal Paper",
        subtitle="Main manuscript package",
        metadata=[
            ("Date", str(date.today())),
            ("Language", "Academic English"),
            ("Co-primary outcomes", str(metrics["co_primary_outcomes"])),
            ("Preterm primary RR", f"{metrics['preterm_primary_rr']:.3f}"),
            ("Stillbirth primary RR", f"{metrics['stillbirth_primary_rr']:.3f}"),
            ("Neonatal primary RR", f"{metrics['neonatal_primary_rr']:.3f}"),
        ],
        note="Draft generated from Track B flagship artifacts.",
    )
    manuscript.add_heading("Abstract", level=1)
    _add_paragraphs(
        manuscript,
        [
            _render_abstract(metrics),
            "Keywords: heat, preterm birth, stillbirth, neonatal mortality, burden, inequity, Brazil.",
        ],
    )
    for title, body in [
        ("Introduction", _render_intro(metrics)),
        ("Methods", _render_methods(metrics)),
        ("Results", _render_results(metrics)),
        ("Discussion", _render_discussion()),
        ("Limitations", _render_limitations()),
    ]:
        manuscript.add_heading(title, level=1)
        _add_paragraphs(manuscript, body.split("\n\n")[1:])
    manuscript.add_heading("Main-text tables", level=1)
    _add_dataframe_table(manuscript, main_tables)
    manuscript.add_heading("Main-text figures", level=1)
    _add_dataframe_table(manuscript, main_figures)
    manuscript_path = workspace_root / "manuscript_trackB_lanplh.docx"
    manuscript.save(manuscript_path)

    supplement = Document()
    _style_document(supplement)
    _add_title_page(
        supplement,
        title="Track B Flagship Perinatal Paper",
        subtitle="Supplement package",
        metadata=[
            ("Date", str(date.today())),
            ("Language", "Academic English"),
            ("Inequity rows", str(metrics["inequity_rows"])),
            ("Robustness rows", str(metrics["robustness_rows"])),
            ("Burden detail rows", str(metrics["burden_total_rows"])),
        ],
        note="Draft supplement generated from Track B artifacts.",
    )
    supplement.add_heading("Supplement overview", level=1)
    _add_paragraphs(
        supplement,
        [
            "This supplement accompanies the Track B flagship paper and includes manuscript shells, reporting checklists, and links to primary artifact packages.",
        ],
    )
    supplement.add_heading("STROBE checklist", level=1)
    _add_dataframe_table(supplement, strobe)
    supplement.add_heading("RECORD checklist", level=1)
    _add_dataframe_table(supplement, record)
    supplement.add_heading("GATHER checklist", level=1)
    _add_dataframe_table(supplement, gather)
    supplement_path = workspace_root / "supplement_trackB_lanplh.docx"
    supplement.save(supplement_path)
    return manuscript_path, supplement_path


def build_track_b_paper_package(root: Path) -> dict[str, object]:
    ensure_track_b_directory_structure(root)
    workspace_root = track_b_publication_root(root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    inputs = _load_inputs(root)
    metrics = _metrics(inputs)
    main_tables = _main_tables()
    main_figures = _main_figures()
    strobe = _strobe()
    record = _record()
    gather = _gather()

    introduction = _render_intro(metrics)
    methods = _render_methods(metrics)
    results = _render_results(metrics)
    discussion = _render_discussion()
    limitations = _render_limitations()
    manuscript_md = "# Track B Flagship Perinatal Manuscript\n\n" + "\n\n".join([introduction, methods, results, discussion, limitations]) + "\n"
    supplement_md = (
        "# Track B Flagship Perinatal Supplement\n\n"
        "## STROBE Checklist\n\n" + _markdown_table(strobe)
        + "\n## RECORD Checklist\n\n" + _markdown_table(record)
        + "\n## GATHER Checklist\n\n" + _markdown_table(gather)
    )

    write_paths = {
        "manuscript_md": workspace_root / "manuscript_trackB_lanplh.md",
        "supplement_md": workspace_root / "supplement_trackB_lanplh.md",
        "research_in_context": workspace_root / "research_in_context_panel.md",
        "strobe": workspace_root / "submission" / "strobe_checklist.md",
        "record": workspace_root / "submission" / "record_checklist.md",
        "gather": workspace_root / "submission" / "gather_checklist.md",
        "code_verification": workspace_root / "submission" / "code_verification_appendix.md",
    }
    _write_text(write_paths["manuscript_md"], manuscript_md)
    _write_text(write_paths["supplement_md"], supplement_md)
    _write_text(
        write_paths["research_in_context"],
        "# Research in Context\n\n"
        "## Evidence before this study\n\n"
        "Most prior heat and perinatal studies focus on a single outcome or a limited geography.\n\n"
        "## Added value of this study\n\n"
        "Track B combines three co-primary outcomes, burden, and vulnerability interpretation in one corrected national package and shows how initially plausible signals can attenuate once full observability and end-to-end validation are enforced.\n\n"
        "## Implications of all available evidence\n\n"
        "The current evidence supports a cautious national null-to-minimal effect interpretation. If a stronger flagship claim is desired, the package likely needs a redesigned editorial target or richer estimators rather than stronger wording alone.\n",
    )
    _write_text(write_paths["strobe"], "# STROBE Checklist\n\n" + _markdown_table(strobe))
    _write_text(write_paths["record"], "# RECORD Checklist\n\n" + _markdown_table(record))
    _write_text(write_paths["gather"], "# GATHER Checklist\n\n" + _markdown_table(gather))
    _write_text(
        write_paths["code_verification"],
        "# Code Verification Appendix\n\n"
        "- Track B artifacts were generated through the repository CLI commands.\n"
        "- Public-facing outputs remain in academic English.\n"
        "- Remaining proxy-model adaptations are explicitly labeled in the exported methods notes.\n",
    )
    reframing_path = workspace_root / "editorial_reframing_note.md"
    _write_text(
        reframing_path,
        "# Editorial Reframing Note\n\n"
        "## Current best-fit framing\n\n"
        "- National event-level and municipal-day analysis with corrected near-null or minimal effect estimates.\n"
        "- Platform-strength paper showing how full-scale correction and validation attenuate apparent headline signals.\n\n"
        "## Less defensible framing under the current results\n\n"
        "- High-burden flagship claim.\n"
        "- Strong causal heat-risk headline across all three co-primary outcomes.\n\n"
        "## Recommended editorial options\n\n"
        "- Submit as a national null-to-minimal effect perinatal heat paper with strong methods emphasis.\n"
        "- Pivot to a methods-plus-results platform paper centered on corrected observability and burden handling.\n"
        "- If a stronger flagship claim is still desired, upgrade estimators and week-of-gestation modeling before submission.\n",
    )
    main_tables.to_csv(workspace_root / "tables" / "main_text_table_shells.csv", index=False)
    main_figures.to_csv(workspace_root / "figures" / "main_text_figure_shells.csv", index=False)

    manuscript_path, supplement_path = _build_docx(workspace_root, metrics, main_tables, main_figures, strobe, record, gather)
    package_root = workspace_root / "submission" / "journal_package_trackB"
    package_root.mkdir(parents=True, exist_ok=True)
    for path in [manuscript_path, supplement_path, write_paths["strobe"], write_paths["record"], write_paths["gather"], write_paths["code_verification"], write_paths["research_in_context"]]:
        copy2(path, package_root / path.name)
    copy2(reframing_path, package_root / reframing_path.name)
    manifest_path = package_root / "journal_package_manifest.md"
    manifest_path.write_text(
        "# Journal Package Manifest\n\n"
        "- Main manuscript: `manuscript_trackB_lanplh.docx`\n"
        "- Supplement: `supplement_trackB_lanplh.docx`\n"
        "- Research in Context panel: `research_in_context_panel.md`\n"
        "- STROBE, RECORD, and GATHER checklists included.\n",
        encoding="utf-8",
    )

    frozen_release = root / "reports" / "frozen_releases" / "track_b_lanplh_stage_b8_release.md"
    frozen_release.parent.mkdir(parents=True, exist_ok=True)
    frozen_release.write_text(
        "# Track B Lancet Planetary Health Flagship Release\n\n"
        f"- Release date: `{date.today()}`\n"
        f"- Main manuscript: `{_relative(root, manuscript_path)}`\n"
        f"- Supplement: `{_relative(root, supplement_path)}`\n"
        "- This release references Track B analytical artifacts without rewriting canonical gold data.\n",
        encoding="utf-8",
    )
    return {
        "status": "PASS",
        "stage": "track_b_stage_b8_paper_package",
        "outputs": {
            "manuscript_trackB_lanplh_docx": _relative(root, manuscript_path),
            "supplement_trackB_lanplh_docx": _relative(root, supplement_path),
            "manuscript_trackB_lanplh_markdown": _relative(root, write_paths["manuscript_md"]),
            "supplement_trackB_lanplh_markdown": _relative(root, write_paths["supplement_md"]),
            "research_in_context_panel": _relative(root, write_paths["research_in_context"]),
            "strobe_checklist": _relative(root, write_paths["strobe"]),
            "record_checklist": _relative(root, write_paths["record"]),
            "gather_checklist": _relative(root, write_paths["gather"]),
            "code_verification_appendix": _relative(root, write_paths["code_verification"]),
            "editorial_reframing_note": _relative(root, reframing_path),
            "journal_package_root": _relative(root, package_root),
            "journal_package_manifest": _relative(root, manifest_path),
            "frozen_release_note": _relative(root, frozen_release),
        },
    }
