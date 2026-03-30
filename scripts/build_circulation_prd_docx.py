from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document

from docx_support import (
    add_bullets,
    add_heading_paragraph,
    add_numbered_references,
    add_table,
    add_title_page,
    save_doc,
    style_document,
)


TITLE = "Child Health Observatory (Observa Infancia) : Brazil's heat exposure on maternal, newborn, and child health - Fiocruz, Brazil."
OUTPUT_PATH = Path("output/doc/child_health_observatory_observa_infancia_prd_v1.docx")


def main() -> None:
    doc = Document()
    style_document(doc)
    add_title_page(
        doc,
        title=TITLE,
        subtitle="Product Requirements and Functional Design Document",
        prepared_by="Prepared by ICICT/Fiocruz, Brazil",
        metadata=[
            ("Version", "v1.1"),
            ("Date", str(date.today())),
            ("Status", "Circulation draft"),
            ("Language", "English"),
            ("Institution", "ICICT/Fiocruz, Brazil"),
            ("Analytical scope", "Brazilian municipalities, 2020-2024"),
        ],
    )

    add_heading_paragraph(
        doc,
        "1. Executive Summary",
        "Child Health Observatory (Observa Infancia) is a public-facing analytical platform for descriptive surveillance "
        "of maternal, newborn, and child health in relation to heat and climate conditions across Brazilian municipalities "
        "from 2020 to 2024. The product is designed to support technical analysis, public communication, and policy-facing "
        "interpretation through a high-trust, metadata-rich environment. Its analytical backbone is the municipality-day "
        "gold layer, while public display prioritizes weekly, monthly, seasonal, and annual interpretation to improve clarity."
    )
    doc.add_paragraph(
        "The platform supports four principal functions: monitoring temporal patterns, identifying spatial concentration of "
        "burden, describing inequalities across municipal contexts, and examining descriptive relationships between health "
        "outcomes and climate exposures. It is not intended to estimate causal effects. All analytical outputs must therefore "
        "display indicator definitions, denominator rules, data provenance, and interpretation notes directly in the interface."
    )

    add_heading_paragraph(
        doc,
        "2. Product Purpose and Intended Use",
        "The Observatory shall function as a descriptive surveillance and decision-support platform. It shall provide a "
        "formal and operationally useful environment for examining perinatal outcomes, maternal morbidity, child mortality, "
        "heat-related exposures, and contextual heterogeneity at national, state, and municipal levels."
    )
    add_bullets(
        doc,
        [
            "Primary use: descriptive epidemiology, monitoring, and analytical interpretation.",
            "Secondary use: technical communication, stakeholder review, and hypothesis generation for future inferential work.",
            "Explicit non-use: causal claims, unsupported league tables, or decontextualized comparison of municipalities.",
        ],
    )

    add_heading_paragraph(
        doc,
        "3. Product Objectives",
        "The product shall meet the following objectives."
    )
    add_bullets(
        doc,
        [
            "Provide a coherent overview of maternal, newborn, and child health in relation to heat and climate conditions.",
            "Support temporal analysis using weekly, monthly, seasonal, and annual aggregation.",
            "Support spatial analysis at Brazil, state, and municipality levels.",
            "Enable contextual stratification using development, inequality, urbanization, altitude, and coastal status.",
            "Expose methods, metadata, source provenance, quality notes, and downloads on every analytical page.",
            "Maintain an elegant and professionally restrained visual standard consistent with ICICT/Fiocruz ownership.",
        ],
    )

    add_heading_paragraph(
        doc,
        "4. Primary Users, Institutional Ownership, and Acronyms",
        "The product is intended to support specialist and policy-facing audiences under ICICT/Fiocruz ownership."
    )
    add_table(
        doc,
        ["User group", "Primary use case"],
        [
            ["Ministry of Health analysts", "Surveillance, planning, and technical monitoring"],
            ["Climate and health researchers", "Descriptive analysis and hypothesis generation"],
            ["Maternal and child health programmes", "Monitoring burden, disparities, and seasonal patterns"],
            ["Civil-society and operational partners", "Targeting and contextual vulnerability assessment"],
            ["Data and IT teams", "Downloadable, versioned, metadata-rich analytical outputs"],
        ],
    )
    doc.add_paragraph()
    add_table(
        doc,
        ["Acronym", "Label"],
        [
            ["KPI", "Key performance indicator"],
            ["WRA", "Women of reproductive age"],
            ["MNCH", "Maternal, newborn, and child health"],
            ["HDP", "Hypertensive disorders of pregnancy"],
            ["GDM", "Gestational diabetes mellitus"],
            ["SINASC", "Brazilian Information System on Live Births"],
            ["SIM", "Brazilian Mortality Information System"],
            ["SIH/SUS", "Brazilian Hospital Information System of the Unified Health System"],
            ["IDH", "Municipal Human Development Index"],
        ],
    )

    add_heading_paragraph(
        doc,
        "5. Analytical Scope and Gold-Layer Architecture",
        "The analytical architecture is structured around the harmonized municipality-day gold panel and thematic gold tables. "
        "The data pipeline follows a medallion architecture: the raw layer preserves source acquisitions, the bronze layer preserves "
        "source-oriented promoted records, the silver layer standardizes validated analytical entities, and the gold layer delivers "
        "harmonized outputs for dashboard consumption. This structure supports traceability, validation, and reproducible transformation "
        "from source-preserving ingestion to publication-ready analytical products."
    )
    add_table(
        doc,
        ["Domain", "Gold base", "Primary role"],
        [
            ["Operational analytical backbone", "gold_mnch_panel", "Cross-domain harmonized panel for public display and shared filters"],
            ["Perinatal outcomes", "gold_mnch_perinatal", "Prematurity, stillbirth, prenatal care, gestational measures, and obstetric outcomes"],
            ["Child mortality", "gold_mnch_mortality", "Neonatal, infant, and under-five mortality"],
            ["Derived child analytical view", "gold_analysis_child_u5", "Age composition and selected heat-sensitive cause groups"],
            ["Derived maternal analytical views", "gold_analysis_wra_15_49; gold_analysis_wra_15_45_sens", "Maternal morbidity and denominator sensitivity"],
            ["Contextual stratification", "gold_municipality_context", "Development, inequality, urbanization, altitude, and coastal context"],
        ],
    )

    add_heading_paragraph(
        doc,
        "6. Information Architecture",
        "The dashboard shall be organized around a limited number of analytically coherent pages."
    )
    add_table(
        doc,
        ["Page", "Purpose", "Primary outputs"],
        [
            ["Overview", "High-level monitoring across outcome families", "Key indicators, trend chart, choropleth, ranking table"],
            ["Perinatal Health", "Prematurity, stillbirth, prenatal and gestational profile", "Trend lines, sensitivity comparison, seasonal heatmap"],
            ["Child Mortality", "Mortality levels, age composition, and selected causes", "Trend panel, composition chart, map, cause panel"],
            ["Maternal Morbidity and Women's Health", "Obstetric burden and denominator sensitivity", "Trend panel, map, WRA comparator"],
            ["Climate Exposure and Seasonality", "Heat and humidity patterns", "Climate trend panel, seasonal heatmap, mountain plot"],
            ["Context and Inequalities", "Heterogeneity by structural municipal context", "Context maps, stratified bars, inequality matrix"],
            ["Exploratory Cross-variable Analysis", "Controlled descriptive crossing of outcomes, climate, and context", "Scatterplots, 2D heatmaps, tables"],
            ["Methods and Data Quality", "Interpretation support and technical transparency", "Metadata cards, quality summaries, downloads"],
        ],
    )

    add_heading_paragraph(
        doc,
        "7. Functional Requirements",
        "The MVP shall implement the following minimum requirements."
    )
    add_bullets(
        doc,
        [
            "A shared filter model shall control time, geography, indicator family, context stratum, and count-versus-rate display.",
            "Every analytical page shall expose last update date, source tables, denominator logic, and interpretation note.",
            "Default public display shall prioritize monthly views, with weekly as an option and daily only for advanced drill-down.",
            "Municipality drill-down shall benchmark a municipality against its state and the national level.",
            "Stillbirth views shall support both primary and sensitivity definitions.",
            "Maternal views shall support denominator sensitivity comparison between women aged 15-49 and women aged 15-45.",
            "All tables and charts shall support export in a user-facing download workflow.",
            "Context stratification shall preserve an explicit missing-data category where context is absent.",
        ],
    )

    add_heading_paragraph(
        doc,
        "8. Component Backlog",
        "The following component inventory translates the product vision into implementation units."
    )
    add_table(
        doc,
        ["Component", "Analytical question", "Gold base", "Priority", "Dependencies"],
        [
            ["Global Filter Bar", "Which population, place, and period am I viewing?", "all", "MVP", "shared state model; time aggregation layer"],
            ["Key Indicator Ribbon", "What is the current level of the main indicators?", "gold_mnch_panel", "MVP", "indicator registry; denominator logic"],
            ["Overview Trend Chart", "How are headline indicators changing over time?", "gold_mnch_panel", "MVP", "monthly and weekly aggregations"],
            ["Overview Choropleth", "Where is burden highest?", "gold_mnch_panel", "MVP", "municipality geometry; map service"],
            ["Perinatal Trend Panel", "How do prematurity and stillbirth evolve over time?", "gold_mnch_perinatal", "MVP", "perinatal metric registry"],
            ["Stillbirth Comparator", "How sensitive are stillbirth findings to case definition?", "gold_mnch_perinatal", "MVP", "definition toggle"],
            ["Child Mortality Trend Panel", "How do neonatal, infant, and under-five mortality change over time?", "gold_mnch_mortality", "MVP", "mortality rate engine"],
            ["Child Age Composition Chart", "Which age bands account for under-five mortality?", "gold_analysis_child_u5", "MVP", "derived age-band logic"],
            ["Maternal Morbidity Panel", "How do maternal morbidity indicators vary over time and space?", "gold_analysis_wra_15_49", "MVP", "WRA denominator service"],
            ["WRA Sensitivity Comparator", "Do patterns differ under alternate WRA denominators?", "gold_analysis_wra_15_49; gold_analysis_wra_15_45_sens", "MVP", "dual-denominator comparison"],
            ["Climate Metrics Panel", "How do heat and humidity conditions vary over time and geography?", "gold_mnch_panel", "MVP", "climate harmonization layer"],
            ["Seasonality Heatmap", "Are there recurring seasonal peaks?", "gold_mnch_panel", "MVP", "month-season aggregation"],
            ["Mountain Plot", "How do seasonal distributions shift across years or strata?", "gold_mnch_panel", "Phase 2", "advanced chart library"],
            ["Context Stratification Matrix", "How do outcomes differ by municipal context?", "gold_mnch_panel + gold_municipality_context", "MVP", "municipality-year join; quantile classifier"],
            ["Context Profile Maps", "What is the distribution of structural municipal context?", "gold_municipality_context", "MVP", "classification and legend logic"],
            ["Municipality Profile Drawer", "What is the local profile relative to state and national benchmarks?", "multiple gold tables", "MVP", "drill-down API; benchmark logic"],
            ["Cross-variable Builder", "What descriptive relationships are visible between outcomes and exposures?", "multiple gold tables", "Phase 2", "query builder; scatter and heatmap engine"],
            ["Methods Drawer", "What does this indicator mean and how should it be interpreted?", "dictionary + all gold", "MVP", "metadata service"],
            ["Download and Citation Centre", "How can outputs be exported and cited correctly?", "published aggregates", "MVP", "export service; versioning"],
            ["Data Quality Panel", "How complete and reliable are the data?", "all", "Phase 2", "quality-control registry; flag summaries"],
        ],
    )

    add_heading_paragraph(
        doc,
        "9. Methods and Interpretation Annex",
        "The product shall expose a concise but rigorous methods layer."
    )
    add_table(
        doc,
        ["Topic", "Specification"],
        [
            ["Analytical unit", "Municipality-day in backend; public outputs aggregated to week, month, season, or year"],
            ["Temporal scope", "2020-2024"],
            ["Geographic scope", "All 5,570 municipalities represented in the analytical gold layer"],
            ["Stillbirth interpretation", "Primary and sensitivity definitions displayed separately"],
            ["Mortality denominators", "Neonatal and infant rates per 1,000 live births; under-five rates per 100,000 under-five population"],
            ["Maternal denominators", "Counts and WRA-based rates; sensitivity comparison between women 15-49 and women 15-45"],
            ["Climate metrics", "Maximum temperature, relative humidity, approximate heat index, and heatwave duration"],
            ["Context variables", "Urban-rural structure, IDH, Gini, altitude, and coastal status with explicit source-year notes"],
            ["Missing context", "Retained as an explicit missing stratum rather than excluded"],
            ["Interpretation limit", "Descriptive surveillance only; not causal inference"],
        ],
    )

    add_heading_paragraph(
        doc,
        "10. Glossary Annex",
        "The following terms shall appear in the methods and metadata layer."
    )
    add_table(
        doc,
        ["Term", "Definition"],
        [
            ["Municipality-day", "One municipality observed on one calendar day"],
            ["Preterm birth", "Live birth before 37 completed weeks of gestation"],
            ["Very preterm birth", "Live birth before 32 completed weeks of gestation"],
            ["Stillbirth, primary", "Stillbirth meeting the primary analytical threshold used in gold outputs"],
            ["Stillbirth, sensitivity", "Stillbirth meeting the alternative sensitivity threshold used for comparison"],
            ["Neonatal mortality", "Deaths from 0 to 27 completed days of life"],
            ["Infant mortality", "Deaths before one year of age"],
            ["Under-five mortality", "Deaths before five completed years of age"],
            ["HDP", "Hypertensive disorders of pregnancy"],
            ["GDM", "Gestational diabetes mellitus"],
            ["WRA", "Women of reproductive age"],
            ["Heat index", "Composite approximation of apparent heat based on temperature and humidity"],
            ["Heatwave duration", "Duration-based indicator of sustained heat exposure"],
            ["IDH", "Municipal Human Development Index contextual variable"],
            ["Gini", "Municipal inequality contextual variable"],
            ["Urban proportion", "Share of the municipal population classified as urban"],
            ["Coastal status", "Binary indicator of official coastal municipal classification"],
        ],
    )

    doc.add_page_break()
    add_heading_paragraph(
        doc,
        "Appendix A. Indexed Project Bibliography Modules Incorporated",
        "The circulation draft incorporates the indexed bibliography already curated in the project repository. "
        "The following topical modules were used to align definitions, provenance, and interpretation."
    )
    add_table(
        doc,
        ["Bibliography module", "Topic"],
        [
            ["references/vancouver/climate_indicators_vancouver.md", "Heat index, heatwave framing, and climate indicators"],
            ["references/vancouver/climate_source_fields_vancouver.md", "Climate source fields and harmonization"],
            ["references/vancouver/gestational_age_prematurity_vancouver.md", "Gestational age, prematurity, and prenatal-care semantics"],
            ["references/vancouver/population_denominators_vancouver.md", "Population denominator semantics"],
            ["references/vancouver/sih_sus_operational_vancouver.md", "SIH/SUS operational and methodological context"],
            ["references/vancouver/sim_general_mortality_vancouver.md", "SIM mortality semantics and quality context"],
            ["references/vancouver/sinasc_full_dictionary_vancouver.md", "SINASC dictionary and data-quality context"],
            ["references/vancouver/stillbirth_fetal_death_vancouver.md", "Stillbirth and fetal-death definitions and reporting context"],
        ],
    )

    add_heading_paragraph(
        doc,
        "Appendix B. Selected Indexed References",
        "The following selected references were drawn from the indexed project bibliography to support the circulation draft."
    )
    selected_refs = [
        "Instituto Nacional de Meteorologia. BDMEP - Dados Historicos [Internet]. Brasilia: INMET; [cited 2026 Mar 19]. Available from: https://portal.inmet.gov.br/servicos/bdmep-dados-hist%C3%B3ricos",
        "Hersbach H, Bell B, Berrisford P, Hirahara S, Horanyi A, Munoz-Sabater J, et al. The ERA5 global reanalysis. Q J R Meteorol Soc. 2020;146(730):1999-2049. doi:10.1002/qj.3803.",
        "Munoz-Sabater J, Dutra E, Agusti-Panareda A, Albergel C, Arduini G, Balsamo G, et al. ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. Earth Syst Sci Data. 2021;13(9):4349-4383. doi:10.5194/essd-13-4349-2021.",
        "Steadman RG. The assessment of sultriness. Part I: A temperature-humidity index based on human physiology and clothing science. J Appl Meteorol. 1979;18(7):861-873.",
        "Henriques LB, Alves EB, Vieira FMDSB, Cardoso BB, D'Angeles ACR, Cruz OG, et al. Accuracy of gestational age assessment in the Brazilian Information System on Live Births (SINASC): a population study. Cad Saude Publica. 2019;35(3):e00098918. doi:10.1590/0102-311X00098918.",
        "Bonilha EA, Lira MMTA, Freitas M, Aly CMC, Dos Santos PC, Niy DY, et al. Gestational age: comparing estimation methods and live births' profile. Rev Bras Epidemiol. 2023;26:e230016. doi:10.1590/1980-549720230016.",
        "Ohuma EO, Moller A-B, Bradley E, Chakwera S, Hussain-Alkhateeb L, Lewin A, et al. National, regional, and global estimates of preterm birth in 2020, with trends from 2010: a systematic analysis. Lancet. 2023;402(10409):1261-1271. doi:10.1016/S0140-6736(23)00878-4.",
        "Hug L, You D, Blencowe H, Mishra A, Wang Z, Fix MJ, et al. Global, regional, and national estimates and trends in stillbirths from 2000 to 2019: a systematic assessment. Lancet. 2021;398(10302):772-785. doi:10.1016/S0140-6736(21)01112-0.",
        "Barros PS, Aquino EC, Souza MR. Fetal mortality and the challenges for women's health care in Brazil. Rev Saude Publica. 2019;53:12. doi:10.11606/S1518-8787.2019053000714.",
        "Domingues RMSM, Meijinhos LS, Guillen LCT, Dias MAB, Saraceni V, Pinheiro RS, et al. Estudo de validacao das internacoes obstetricas no Sistema de Informacoes Hospitalares do Sistema Unico de Saude para a vigilancia da morbidade materna: Brasil, 2021-2022. Epidemiol Serv Saude. 2024;33:e20231252. doi:10.1590/S2237-96222024v33e20231252.pt.",
        "Marinho MF, Franca EB, Teixeira RA, Ishitani LH, Cunha CC, Santos MR, et al. Dados para a saude: impacto na melhoria da qualidade da informacao sobre causas de obito no Brasil. Rev Bras Epidemiol. 2019;22 Suppl 3:e190005.supl.3. doi:10.1590/1980-549720190005.supl.3.",
        "Silveira IH, Reboucas P, Pescarini J, Cortes TR, Oliveira BFA, Neves DJD, et al. Effects of ambient temperature on under-five mortality: a nationwide space-time-stratified case-crossover study in Brazil. Environ Res. 2025;285(Pt 5):122683. doi:10.1016/j.envres.2025.122683.",
    ]
    add_numbered_references(doc, selected_refs)

    save_doc(doc, OUTPUT_PATH)


if __name__ == "__main__":
    main()
