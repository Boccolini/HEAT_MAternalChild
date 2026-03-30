from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document

from docx_support import (
    add_bullets,
    add_heading_paragraph,
    add_table,
    add_title_page,
    save_doc,
    style_document,
)


TITLE = "Child Health Observatory (Observa Infancia) : Brazil's heat exposure on maternal, newborn, and child health - Fiocruz, Brazil."
OUTPUT_PATH = Path("output/doc/child_health_observatory_observa_infancia_wireframes_handoff_v1.docx")


def add_page_wireframe(
    doc: Document,
    page_title: str,
    page_purpose: str,
    rows: list[list[str]],
) -> None:
    doc.add_heading(page_title, level=2)
    doc.add_paragraph(page_purpose)
    add_table(
        doc,
        ["Zone", "Component", "Annotation", "Gold base", "Interaction note"],
        rows,
    )


def main() -> None:
    doc = Document()
    style_document(doc)
    add_title_page(
        doc,
        title=TITLE,
        subtitle="Annotated Page Wireframes and Implementation Handoff Pack",
        prepared_by="Prepared by ICICT/Fiocruz, Brazil",
        metadata=[
            ("Version", "v1.0"),
            ("Date", str(date.today())),
            ("Status", "Implementation handoff draft"),
            ("Language", "English"),
            ("Institution", "ICICT/Fiocruz, Brazil"),
        ],
    )

    add_heading_paragraph(
        doc,
        "1. Package Purpose",
        "This package translates the approved product requirements into annotated page wireframes and an implementation handoff "
        "for frontend, product, data, and quality-assurance teams. It defines the screen architecture, layout logic, component "
        "inventory, design tokens, page behaviors, and technical dependencies required for implementation."
    )

    add_heading_paragraph(
        doc,
        "2. Global Shell and Navigation",
        "The product shell shall remain consistent across all analytical pages."
    )
    add_table(
        doc,
        ["Shell element", "Specification"],
        [
            ["Top navigation", "Product title, page tabs, institution label, reset action, download action"],
            ["Global filter bar", "Time period, geography, indicator family, context stratum, count-versus-rate toggle"],
            ["Metadata rail", "Last update, source tables, denominator, interpretation note"],
            ["Main canvas", "Page-specific analytical components arranged in a stable visual hierarchy"],
            ["Right-side or bottom drawer", "Definitions, glossary terms, and municipality profile drill-down"],
        ],
    )

    add_heading_paragraph(
        doc,
        "3. Annotated Page Wireframes",
        "The following wireframes specify layout logic and component purpose page by page."
    )

    add_page_wireframe(
        doc,
        "3.1 Overview",
        "The Overview page is the high-level entry point and should answer what is changing, where burden is concentrated, and how indicators compare.",
        [
            ["Row 1", "Key indicator ribbon", "Six headline cards with count and rate toggle", "gold_mnch_panel", "Card click sets active indicator"],
            ["Row 2 left", "Trend panel", "Main multi-year line chart with monthly default", "gold_mnch_panel", "Legend filter and hover tooltips"],
            ["Row 2 right", "Choropleth map", "Spatial burden for active indicator", "gold_mnch_panel", "Map click filters state or municipality"],
            ["Row 3 left", "Seasonality heatmap", "Month-by-year intensity matrix", "gold_mnch_panel", "Cell click filters month and year"],
            ["Row 3 right", "Ranking table", "Top and bottom states or municipalities", "gold_mnch_panel", "Sortable and downloadable"],
        ],
    )

    add_page_wireframe(
        doc,
        "3.2 Perinatal Health",
        "This page should foreground prematurity, stillbirth, prenatal care, and gestational profile.",
        [
            ["Row 1", "Perinatal summary cards", "Prematurity, very preterm, stillbirth primary, stillbirth sensitivity", "gold_mnch_perinatal", "Card click changes chart focus"],
            ["Row 2 left", "Prematurity trend", "Time series for prematurity metrics", "gold_mnch_perinatal", "Toggle count versus rate"],
            ["Row 2 right", "Stillbirth comparator", "Primary versus sensitivity comparison", "gold_mnch_perinatal", "Definition toggle"],
            ["Row 3 left", "Gestational profile panel", "Gestational-age and days-lost distribution summary", "gold_mnch_perinatal", "Context filter applies"],
            ["Row 3 right", "Prenatal care panel", "Prenatal visits and six-plus threshold", "gold_mnch_perinatal", "Tooltip with source note"],
        ],
    )

    add_page_wireframe(
        doc,
        "3.3 Child Mortality",
        "This page should support mortality surveillance across neonatal, infant, and under-five outcomes, including age composition.",
        [
            ["Row 1", "Mortality summary cards", "Neonatal, infant, and under-five mortality", "gold_mnch_mortality", "Active card drives map and trend"],
            ["Row 2 left", "Mortality trend panel", "Longitudinal rates and counts", "gold_mnch_mortality", "Monthly default with annual option"],
            ["Row 2 right", "Mortality map", "Spatial distribution for selected mortality metric", "gold_mnch_mortality", "Map click filters geography"],
            ["Row 3 left", "Age composition chart", "Neonatal, post-neonatal, and age 1-4 years", "gold_analysis_child_u5", "Stacked or grouped view toggle"],
            ["Row 3 right", "Cause-group panel", "Selected heat-sensitive causes", "gold_analysis_child_u5", "Cause selection filters lower table"],
        ],
    )

    add_page_wireframe(
        doc,
        "3.4 Maternal Morbidity and Women's Health",
        "This page should organize obstetric burden and WRA sensitivity within a single coherent workflow.",
        [
            ["Row 1", "Maternal summary cards", "Obstetric admissions, HDP, severe HDP, GDM, preterm labour", "gold_analysis_wra_15_49", "Card click sets active metric"],
            ["Row 2 left", "Maternal trend panel", "Temporal trend by selected metric", "gold_analysis_wra_15_49", "Count versus rate toggle"],
            ["Row 2 right", "Maternal map", "Spatial concentration of burden", "gold_analysis_wra_15_49", "Map click filters geography"],
            ["Row 3 left", "WRA comparator", "15-49 versus 15-45 denominator sensitivity", "gold_analysis_wra_15_49; gold_analysis_wra_15_45_sens", "Slope or paired bar display"],
            ["Row 3 right", "Context inequality panel", "Outcome by context quintile or category", "gold_analysis_wra_15_49 + gold_municipality_context", "Stratum click propagates filter"],
        ],
    )

    add_page_wireframe(
        doc,
        "3.5 Climate Exposure and Seasonality",
        "This page should prioritize the exposure story and make seasonality legible before the user moves to outcome interpretation.",
        [
            ["Row 1", "Climate summary cards", "Maximum temperature, humidity, heat index, heatwave duration", "gold_mnch_panel", "Card click changes active metric"],
            ["Row 2 left", "Climate trend panel", "Time series for selected exposure", "gold_mnch_panel", "Monthly default"],
            ["Row 2 right", "Climate map", "Spatial pattern of selected climate metric", "gold_mnch_panel", "Map click filters geography"],
            ["Row 3 left", "Seasonality heatmap", "Month-by-year exposure structure", "gold_mnch_panel", "Cell click filters month-year"],
            ["Row 3 right", "Mountain plot", "Seasonal distribution by year or context", "gold_mnch_panel", "Phase 2 visualization"],
        ],
    )

    add_page_wireframe(
        doc,
        "3.6 Context and Inequalities",
        "This page should separate structural municipal context from immediate health outcomes while keeping their relationship visible.",
        [
            ["Row 1", "Context selector", "Choose IDH, Gini, urban proportion, altitude, or coastal status", "gold_municipality_context", "Variable selection updates all plots"],
            ["Row 2 left", "Context map", "Spatial distribution of selected context variable", "gold_municipality_context", "Map click filters geography"],
            ["Row 2 right", "Outcome by stratum", "Outcome distribution by selected context stratum", "gold_mnch_panel + gold_municipality_context", "Bar click filters stratum"],
            ["Row 3 left", "Inequality matrix", "Indicator versus context comparison matrix", "gold_mnch_panel + gold_municipality_context", "Hover reveals exact value and denominator"],
            ["Row 3 right", "Scatter panel", "Outcome versus context correlation display", "gold_mnch_panel + gold_municipality_context", "Tooltip includes municipality context profile"],
        ],
    )

    add_page_wireframe(
        doc,
        "3.7 Exploratory Cross-variable Analysis",
        "This page should remain controlled and descriptive, suitable for advanced analytical users.",
        [
            ["Row 1", "Variable builder", "Select x-variable, y-variable, colour, and bubble size", "multiple", "Stateful filter builder"],
            ["Row 2 left", "Scatter or bubble plot", "Primary cross-variable view", "multiple", "Context and time filters apply"],
            ["Row 2 right", "2D heatmap", "Alternative density or binned relationship view", "multiple", "Switch between count and rate"],
            ["Row 3", "Dynamic table", "Underlying filtered analytical table", "multiple", "Download current query"],
        ],
    )

    add_page_wireframe(
        doc,
        "3.8 Methods and Data Quality",
        "This page should make interpretation safeguards explicit and easy to access.",
        [
            ["Row 1", "Definitions cards", "Indicator definitions and denominator rules", "dictionary + all gold", "Page-level search"],
            ["Row 2 left", "Quality summary", "Coverage, missingness, and flag counts", "all", "Filterable by page and indicator"],
            ["Row 2 right", "Data provenance panel", "Source tables and update logic", "all", "Expandable details"],
            ["Row 3", "Downloads and citation", "Structured export and recommended citation", "published aggregates", "Version included in export"],
        ],
    )

    add_heading_paragraph(
        doc,
        "4. Design-System Handoff",
        "The frontend shall implement the following visual system and interaction rules."
    )
    add_table(
        doc,
        ["Token family", "Specification"],
        [
            ["Colour palette", "Primary navy #12324A; secondary blue #365C72; canvas #F7F7F4; rule #D8DEE3; heat accent #C76A1B; humidity teal #3E7C78; caution red #8E2C2C"],
            ["Typography", "Aptos or Source Sans 3 for interface; Source Serif 4 for section headers in editorial outputs; tabular numerals for KPI values"],
            ["Spacing", "8-point spacing system with large whitespace between analytical zones"],
            ["Grid", "12-column desktop grid; 8-column tablet adaptation; stacked mobile fallback"],
            ["Cards", "Low-elevation flat cards with explicit titles, units, and metadata links"],
            ["Charts", "No rainbow palettes; no 3D effects; direct labels when feasible; legends always visible"],
            ["Maps", "Single-hue sequential palette for rates; stable class logic across comparable pages; restrained basemap"],
            ["Motion", "Minimal motion; state changes should be immediate and purposeful rather than decorative"],
        ],
    )

    add_heading_paragraph(
        doc,
        "5. Frontend Handoff",
        "The frontend should be implemented as a route-based analytical application with shared filter state and reusable chart components."
    )
    add_table(
        doc,
        ["Area", "Recommendation"],
        [
            ["Route map", "/overview, /perinatal, /child-mortality, /maternal, /climate, /context, /explorer, /methods"],
            ["Shared state", "Global filters stored in a single application state with page-specific local state only when necessary"],
            ["Core components", "FilterBar, MetricCard, TrendPanel, ChoroplethMap, HeatmapMatrix, RankingTable, DefinitionDrawer, MunicipalityProfileDrawer"],
            ["Page composition", "Each page assembles reusable components rather than unique ad hoc layouts"],
            ["Metadata pattern", "Every page receives a metadata payload with update date, denominator, provenance, and caveats"],
            ["Responsive behavior", "Desktop first, but all tables and maps must degrade gracefully for tablet and mobile"],
        ],
    )

    add_heading_paragraph(
        doc,
        "6. Backend and Data Handoff",
        "The backend and data layers should expose stable analytical services aligned with the gold tables."
    )
    add_table(
        doc,
        ["Service", "Requirement"],
        [
            ["Aggregation service", "Return week, month, season, and year aggregations from municipality-day gold tables"],
            ["Context join service", "Apply municipality-year joins from gold contextual tables"],
            ["Indicator registry", "Map public metric identifiers to gold variables, denominators, units, and labels"],
            ["Metadata service", "Return definitions, denominator rules, source tables, and data-quality notes"],
            ["Export service", "Support filtered chart and table export with version stamp"],
            ["Geography service", "Return state and municipality geometries and lookup metadata"],
            ["Caching strategy", "Cache common aggregations by page, metric, and time granularity"],
        ],
    )

    add_heading_paragraph(
        doc,
        "7. Suggested Response Contract",
        "All analytical endpoints should return a stable and self-describing structure."
    )
    add_table(
        doc,
        ["Field", "Meaning"],
        [
            ["metric_id", "Stable frontend metric identifier"],
            ["metric_label", "Human-readable metric label"],
            ["unit", "Count, rate, percentage, or contextual unit"],
            ["filters_applied", "Time, geography, context, and definition filters"],
            ["series", "Analytical payload for charts or tables"],
            ["denominator_note", "Readable denominator rule"],
            ["source_tables", "Gold tables used in the response"],
            ["quality_flags", "Relevant quality or imputation notes"],
            ["last_updated", "Timestamp of last materialized source"],
        ],
    )

    add_heading_paragraph(
        doc,
        "8. Accessibility and QA Handoff",
        "Quality assurance must be integrated into implementation rather than deferred."
    )
    add_bullets(
        doc,
        [
            "Meet WCAG 2.2 AA contrast and keyboard-navigation requirements.",
            "Ensure that colour is never the only carrier of meaning.",
            "Provide alt text and accessible labels for maps, charts, and controls.",
            "Test filter persistence, drill-down behavior, downloads, and mobile fallback.",
            "Validate that every public metric resolves to a documented gold variable and denominator rule.",
        ],
    )

    add_heading_paragraph(
        doc,
        "9. Delivery Sequence",
        "Implementation should proceed in controlled phases."
    )
    add_table(
        doc,
        ["Phase", "Priority deliverables"],
        [
            ["Phase 1", "Global shell, overview, perinatal, child mortality, maternal, methods drawer"],
            ["Phase 2", "Climate and context pages, municipality profile, download and citation"],
            ["Phase 3", "Explorer, mountain plot, expanded quality diagnostics"],
        ],
    )

    save_doc(doc, OUTPUT_PATH)


if __name__ == "__main__":
    main()
