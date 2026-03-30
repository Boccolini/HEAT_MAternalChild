from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

from docx_support import add_table, save_doc, style_document


ROOT = Path(__file__).resolve().parents[1]
ROUND_DIR = ROOT / "reports" / "publications" / "incremental_analysis_round"
OUTPUT_DIR = ROOT / "output" / "doc"

DOC_SPECS = [
    (
        ROUND_DIR / "31_lrh_americas_editorial_revision_plan.md",
        OUTPUT_DIR / "lrh_americas_editorial_revision_plan_rev4.docx",
    ),
    (
        ROUND_DIR / "32_lrh_americas_original_article_manuscript_rev4.md",
        OUTPUT_DIR / "lrh_americas_original_article_heat_mnch_brazil_2020_2024_rev4_editorial.docx",
    ),
    (
        ROUND_DIR / "33_lrh_americas_research_in_context_rev4.md",
        OUTPUT_DIR / "lrh_americas_research_in_context_heat_mnch_brazil_2020_2024_rev4_editorial.docx",
    ),
    (
        ROUND_DIR / "34_lrh_americas_supplement_rev4.md",
        OUTPUT_DIR / "lrh_americas_supplement_heat_mnch_brazil_2020_2024_rev4_editorial.docx",
    ),
    (
        ROUND_DIR / "35_lrh_americas_submission_checklists_rev4.md",
        OUTPUT_DIR / "lrh_americas_submission_checklists_heat_mnch_brazil_2020_2024_rev4_editorial.docx",
    ),
    (
        ROUND_DIR / "30_lrh_americas_multiactor_critical_consensus_dossier.md",
        OUTPUT_DIR / "lrh_americas_multiactor_critical_consensus_dossier.docx",
    ),
]


def _strip_inline(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("`", "")
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("*", "")
    text = text.replace("_", "_")
    return text.strip()


def _parse_table(lines: list[str], start: int) -> tuple[list[str], list[list[str]], int]:
    block: list[str] = []
    idx = start
    while idx < len(lines) and lines[idx].lstrip().startswith("|"):
        block.append(lines[idx].rstrip())
        idx += 1
    headers = [_strip_inline(cell) for cell in block[0].strip().strip("|").split("|")]
    rows: list[list[str]] = []
    for raw in block[2:]:
        row = [_strip_inline(cell) for cell in raw.strip().strip("|").split("|")]
        rows.append(row)
    return headers, rows, idx


def _flush_paragraph(doc: Document, buffer: list[str]) -> None:
    if not buffer:
        return
    paragraph = doc.add_paragraph(" ".join(_strip_inline(part) for part in buffer).strip())
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    buffer.clear()


def render_markdown_to_docx(input_path: Path, output_path: Path) -> None:
    doc = Document()
    style_document(doc)
    lines = input_path.read_text(encoding="utf-8").splitlines()
    paragraph_buffer: list[str] = []
    first_heading = True
    idx = 0

    while idx < len(lines):
        line = lines[idx].rstrip()
        stripped = line.strip()

        if not stripped:
            _flush_paragraph(doc, paragraph_buffer)
            idx += 1
            continue

        if stripped.startswith("|"):
            _flush_paragraph(doc, paragraph_buffer)
            headers, rows, idx = _parse_table(lines, idx)
            add_table(doc, headers, rows)
            doc.add_paragraph()
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            _flush_paragraph(doc, paragraph_buffer)
            hashes, content = heading_match.groups()
            content = _strip_inline(content)
            if first_heading:
                para = doc.add_paragraph(style="Title")
                para.add_run(content)
                para.alignment = WD_ALIGN_PARAGRAPH.LEFT
                first_heading = False
            else:
                level = min(max(len(hashes) - 1, 1), 3)
                doc.add_heading(content, level=level)
            idx += 1
            continue

        bullet_match = re.match(r"^-\s+(.*)$", stripped)
        if bullet_match:
            _flush_paragraph(doc, paragraph_buffer)
            doc.add_paragraph(_strip_inline(bullet_match.group(1)), style="List Bullet")
            idx += 1
            continue

        number_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if number_match:
            _flush_paragraph(doc, paragraph_buffer)
            doc.add_paragraph(_strip_inline(number_match.group(1)), style="List Number")
            idx += 1
            continue

        quote_match = re.match(r"^>\s+(.*)$", stripped)
        if quote_match:
            _flush_paragraph(doc, paragraph_buffer)
            para = doc.add_paragraph()
            run = para.add_run(_strip_inline(quote_match.group(1)))
            run.italic = True
            idx += 1
            continue

        paragraph_buffer.append(stripped)
        idx += 1

    _flush_paragraph(doc, paragraph_buffer)
    save_doc(doc, output_path)


def main() -> None:
    for input_path, output_path in DOC_SPECS:
        render_markdown_to_docx(input_path, output_path)
        print(f"wrote {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
