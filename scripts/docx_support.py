from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


HEADER_FILL = "D9E6F2"


def set_cell_shading(cell, fill: str = HEADER_FILL) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def style_document(doc: Document) -> None:
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


def add_title_page(
    doc: Document,
    *,
    title: str,
    subtitle: str,
    prepared_by: str,
    metadata: list[tuple[str, str]],
    note: str | None = None,
) -> None:
    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(title)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(subtitle)
    run.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(prepared_by)

    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    for idx, label in enumerate(["Field", "Value"]):
        cell = table.rows[0].cells[idx]
        cell.text = label
        set_cell_shading(cell)

    for key, value in metadata:
        row = table.add_row().cells
        row[0].text = key
        row[1].text = value

    if note:
        doc.add_paragraph()
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(note)
        run.italic = True

    doc.add_page_break()


def add_heading_paragraph(doc: Document, heading: str, text: str) -> None:
    doc.add_heading(heading, level=1)
    doc.add_paragraph(text)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for idx, header in enumerate(headers):
        cell = table.rows[0].cells[idx]
        cell.text = header
        set_cell_shading(cell)
    for values in rows:
        row = table.add_row().cells
        for idx, value in enumerate(values):
            row[idx].text = value


def add_numbered_references(doc: Document, refs: list[str]) -> None:
    for ref in refs:
        p = doc.add_paragraph(style="List Number")
        p.add_run(ref)


def save_doc(doc: Document, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
