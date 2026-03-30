from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .gold_documentation import build_gold_variable_dictionary
from .outcome_dictionary_icd10 import DIRECT_HEAT_GROUPS, INDIRECT_HEAT_GROUPS, OUTCOME_GROUP_SPECS, SLOW_MOVING_HEAT_GROUPS

START_YEAR = 2020
END_YEAR = 2024
PLOT_SAMPLE_ROWS = 30000
TABLE_PAGE_ROWS = 22
TABLE_PRIORITY = {
    "gold_mnch_panel": 0,
    "gold_mnch_perinatal": 1,
    "gold_mnch_mortality": 2,
    "gold_municipality_context_static": 3,
    "gold_municipality_context": 4,
    "gold_analysis_child_u5": 5,
    "gold_analysis_wra_15_49": 6,
    "gold_analysis_wra_15_45_sens": 7,
}
TABLE_LABELS = {
    "gold_mnch_panel": "Painel gold harmonizado",
    "gold_mnch_perinatal": "Gold perinatal",
    "gold_mnch_mortality": "Gold mortalidade infantil",
    "gold_municipality_context": "Gold contexto municipal",
    "gold_municipality_context_static": "Gold contexto municipal estatico",
    "gold_analysis_child_u5": "View analitica gold crianca <5",
    "gold_analysis_wra_15_49": "View analitica gold mulheres 15-49",
    "gold_analysis_wra_15_45_sens": "View analitica gold mulheres 15-45 sensibilidade",
}
CONDITIONAL_MISSINGNESS_SCOPES = {
    "PRENATAL_VISITS_COUNT": {
        "required_columns": ("LIVE_BIRTHS_TOTAL",),
        "scope_label": "rows with LIVE_BIRTHS_TOTAL > 0",
    }
}
PRIMARY_METRIC_FIELDS = {
    "PRETERM_BIRTHS_CANONICAL_TOTAL",
    "PRETERM_FLAG",
    "STILLBIRTH_TOTAL_PRIMARY_GTE28",
    "DEATHS_NEONATAL",
    "DEATHS_U1",
    "DEATHS_U5",
    "OBSTETRIC_ADMISSIONS_TOTAL",
    "HDP_ANY_TOTAL",
    "HDP_SEVERE_TOTAL",
    "GDM_GESTATIONAL_TOTAL",
    "PRETERM_LABOR_TOTAL",
}
CLIMATE_FIELDS = {"TMAX_C", "RH_MEAN_PCT", "HEAT_INDEX_APPROX_C", "HEATWAVE_LEN_P95"}
IDENTIFIER_FIELDS = {
    "MUNICIPALITY_CODE",
    "municipality_code",
    "MUNICIPALITY_CODE_IBGE7",
    "DATE",
    "YEAR",
    "year",
    "MONTH",
}
CONTINUOUS_UNIT_HINTS = {"degrees Celsius", "percent", "rate per 1,000", "log(denominator)", "value", "proportion"}
COUNT_UNIT_HINTS = {"count", "population count", "days"}
STRING_TYPE_HINTS = ("VARCHAR", "TEXT", "STRING")
TEMPORAL_TYPE_HINTS = ("DATE", "TIMESTAMP", "TIME")
FLOAT_TYPE_HINTS = ("FLOAT", "DOUBLE", "DECIMAL", "REAL")
INTEGER_TYPE_HINTS = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UBIGINT", "UINTEGER", "USMALLINT", "UTINYINT")
LAYER_PATTERN = re.compile(r"\b(?:silver|bronze|raw)_[A-Za-z0-9_]+\b")

CONTEXT_OVERRIDES = {
    "URBAN_POP": {
        "label": "Populacao urbana",
        "definition": "Contagem da populacao urbana municipal carregada na camada gold contextual.",
        "unit": "population count",
        "construction": "Campo direto da gold contextual. Serve como componente do calculo de PROP_URBAN e PROP_RURAL.",
    },
    "RURAL_POP": {
        "label": "Populacao rural",
        "definition": "Contagem da populacao rural municipal carregada na camada gold contextual.",
        "unit": "population count",
        "construction": "Campo direto da gold contextual. Serve como componente do calculo de PROP_URBAN e PROP_RURAL.",
    },
    "PROP_URBAN": {
        "label": "Proporcao urbana",
        "definition": "Participacao da populacao urbana no total urbano+rural do municipio.",
        "unit": "proportion",
        "construction": "Derivada na gold contextual como URBAN_POP / (URBAN_POP + RURAL_POP).",
        "numerator": "URBAN_POP",
        "denominator": "URBAN_POP + RURAL_POP",
    },
    "PROP_RURAL": {
        "label": "Proporcao rural",
        "definition": "Participacao da populacao rural no total urbano+rural do municipio.",
        "unit": "proportion",
        "construction": "Derivada na gold contextual como RURAL_POP / (URBAN_POP + RURAL_POP).",
        "numerator": "RURAL_POP",
        "denominator": "URBAN_POP + RURAL_POP",
    },
    "GINI_RAW": {
        "label": "Gini observado",
        "definition": "Valor de referencia do coeficiente de Gini municipal armazenado na gold contextual.",
        "unit": "value",
        "construction": "Valor de referencia mantido na gold contextual antes da expansao temporal.",
    },
    "GINI_EXPANDED": {
        "label": "Gini expandido",
        "definition": "Coeficiente de Gini expandido para o painel analitico municipal-ano.",
        "unit": "value",
        "construction": "Valor do Gini carregado na gold contextual apos expansao temporal documentada.",
    },
    "IDH_RAW": {
        "label": "IDH observado",
        "definition": "Valor de referencia do IDH municipal armazenado na gold contextual.",
        "unit": "value",
        "construction": "Valor de referencia mantido na gold contextual antes da expansao temporal.",
    },
    "IDH_EXPANDED": {
        "label": "IDH expandido",
        "definition": "IDH municipal expandido para o painel analitico municipal-ano.",
        "unit": "value",
        "construction": "Valor do IDH carregado na gold contextual apos expansao temporal documentada.",
    },
    "ALTITUDE_MEAN_M": {
        "label": "Altitude media municipal",
        "definition": "Altitude media municipal em metros carregada na gold contextual.",
        "unit": "value",
        "construction": "Resumo altimetrico municipal estatico ligado a todas as observacoes do painel contextual.",
    },
    "COASTAL_FLAG": {
        "label": "Municipio costeiro",
        "definition": "Indicador binario de pertencimento ao recorte costeiro usado na camada contextual gold.",
        "unit": "0/1 indicator",
        "construction": "Indicador binario contextual estatico mantido na gold contextual.",
    },
    "STATIC_CONTEXT_FLAG": {
        "label": "Marcador de contexto estatico",
        "definition": "Indicador binario de que a linha foi preenchida a partir da camada contextual estatica.",
        "unit": "0/1 indicator",
        "construction": "Flag operacional na gold contextual para distinguir componentes estaticos.",
    },
    "JOIN_STATUS_FLAG": {
        "label": "Status de linkage contextual",
        "definition": "Indicador operacional do status de vinculacao do contexto municipal na gold.",
        "unit": "0/1 indicator",
        "construction": "Flag operacional do join da gold contextual.",
    },
    "MUNICIPAL_BOUNDARY_REFERENCE_YEAR": {
        "label": "Ano de referencia territorial",
        "definition": "Ano de referencia da malha municipal usada na gold contextual.",
        "unit": "year",
        "construction": "Metadado territorial carregado na camada contextual gold.",
    },
}


@dataclass(frozen=True)
class GoldTableSpec:
    table_name: str
    parquet_glob: str
    layer: str
    plot_priority: int


@dataclass(frozen=True)
class ArtifactPaths:
    output_dir: Path
    tables_dir: Path
    pdf_path: Path
    manifest_path: Path
    schema_full_path: Path
    schema_compact_path: Path
    missingness_path: Path
    continuous_summary_path: Path
    count_summary_path: Path
    categorical_summary_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sanitize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    text = LAYER_PATTERN.sub("upstream pathway", text)
    return re.sub(r"\s+", " ", text)


def _wrap_text(value: object, width: int) -> str:
    text = _sanitize_text(value)
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def _chunks(items: Iterable[str], size: int) -> Iterable[list[str]]:
    chunk: list[str] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _table_specs(root: Path) -> list[GoldTableSpec]:
    return [
        GoldTableSpec("gold_mnch_panel", str(root / "data/gold/gold_mnch_panel/YEAR=*/*.parquet"), "gold", TABLE_PRIORITY["gold_mnch_panel"]),
        GoldTableSpec("gold_mnch_perinatal", str(root / "data/gold/gold_mnch_perinatal/YEAR=*/*.parquet"), "gold", TABLE_PRIORITY["gold_mnch_perinatal"]),
        GoldTableSpec("gold_mnch_mortality", str(root / "data/gold/gold_mnch_mortality/YEAR=*/*.parquet"), "gold", TABLE_PRIORITY["gold_mnch_mortality"]),
        GoldTableSpec("gold_municipality_context_static", str(root / "data/gold/gold_municipality_context_static/gold_municipality_context_static.parquet"), "gold", TABLE_PRIORITY["gold_municipality_context_static"]),
        GoldTableSpec("gold_municipality_context", str(root / "data/gold/gold_municipality_context/YEAR=*/*.parquet"), "gold", TABLE_PRIORITY["gold_municipality_context"]),
        GoldTableSpec("gold_analysis_child_u5", str(root / "data/gold/gold_analysis_child_u5/YEAR=*/*.parquet"), "derived_gold", TABLE_PRIORITY["gold_analysis_child_u5"]),
        GoldTableSpec("gold_analysis_wra_15_49", str(root / "data/gold/gold_analysis_wra_15_49/YEAR=*/*.parquet"), "derived_gold", TABLE_PRIORITY["gold_analysis_wra_15_49"]),
        GoldTableSpec("gold_analysis_wra_15_45_sens", str(root / "data/gold/gold_analysis_wra_15_45_sens/YEAR=*/*.parquet"), "derived_gold", TABLE_PRIORITY["gold_analysis_wra_15_45_sens"]),
    ]


def _artifact_paths(root: Path, output_dir: Path | None = None) -> ArtifactPaths:
    report_dir = output_dir or root / "reports/round3"
    tables_dir = report_dir / "tables"
    return ArtifactPaths(
        output_dir=report_dir,
        tables_dir=tables_dir,
        pdf_path=report_dir / "round3_gold_analysis.pdf",
        manifest_path=report_dir / "round3_gold_analysis_manifest.json",
        schema_full_path=tables_dir / "round3_gold_schema_full.csv",
        schema_compact_path=tables_dir / "round3_gold_schema_compact.csv",
        missingness_path=tables_dir / "round3_gold_missingness.csv",
        continuous_summary_path=tables_dir / "round3_gold_continuous_summary.csv",
        count_summary_path=tables_dir / "round3_gold_count_summary.csv",
        categorical_summary_path=tables_dir / "round3_gold_categorical_summary.csv",
    )


def _load_registry_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_metric_maps(metrics: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], dict[str, str]], dict[tuple[str, str], str]]:
    numerator_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    denominator_map: dict[tuple[str, str], list[str]] = {}
    for metric in metrics:
        source_table = str(metric.get("source_table", ""))
        numerator_field = str(metric.get("numerator_field", ""))
        denominator_field = str(metric.get("denominator_field", ""))
        if source_table and numerator_field:
            numerator_map.setdefault((source_table, numerator_field), []).append(metric)
        if source_table and denominator_field:
            denominator_map.setdefault((source_table, denominator_field), []).append(str(metric.get("metric_id", "")))
    numerator_summary: dict[tuple[str, str], dict[str, str]] = {}
    for key, rows in numerator_map.items():
        numerator_summary[key] = {
            "metric_ids": "; ".join(sorted({str(row["metric_id"]) for row in rows})),
            "metric_labels": "; ".join(sorted({str(row["label"]) for row in rows})),
            "metric_families": "; ".join(sorted({str(row.get("metric_family", "")) for row in rows if row.get("metric_family")})),
            "metric_definition_types": "; ".join(sorted({str(row.get("definition_type", "")) for row in rows if row.get("definition_type")})),
            "dashboard_denominator_field": "; ".join(sorted({str(row.get("denominator_field", "")) for row in rows if row.get("denominator_field")})),
            "aggregation_methods": "; ".join(sorted({str(row.get("aggregation_method", "")) for row in rows if row.get("aggregation_method")})),
            "rate_display_units": "; ".join(sorted({str(row.get("rate_display_unit", "")) for row in rows if row.get("rate_display_unit")})),
        }
    denominator_summary = {key: "; ".join(sorted({metric_id for metric_id in metric_ids if metric_id})) for key, metric_ids in denominator_map.items()}
    return numerator_summary, denominator_summary


def _build_icd_maps() -> dict[str, dict[str, str]]:
    group_specs = {spec.outcome_group: spec for spec in OUTCOME_GROUP_SPECS}
    mapping = {
        "HDP_ANY_TOTAL": "HDP_ANY_PRIMARY",
        "HDP_SEVERE_TOTAL": "HDP_SEVERE_PRIMARY",
        "HDP_GESTATIONAL_HTN_TOTAL": "HDP_GESTATIONAL_HTN_PRIMARY",
        "HDP_PREECLAMPSIA_TOTAL": "HDP_PREECLAMPSIA_PRIMARY",
        "HDP_ECLAMPSIA_TOTAL": "HDP_ECLAMPSIA_PRIMARY",
        "PRETERM_LABOR_TOTAL": "PRETERM_LABOR_PRIMARY",
        "PRETERM_LABOR_URGENT_TOTAL": "PRETERM_LABOR_PRIMARY",
        "PRETERM_LABOR_ICU_TOTAL": "PRETERM_LABOR_PRIMARY",
        "PRETERM_LABOR_DEATH_TOTAL": "PRETERM_LABOR_PRIMARY",
        "GDM_GESTATIONAL_TOTAL": "GDM_GESTATIONAL_PRIMARY",
        "GDM_GESTATIONAL_URGENT_TOTAL": "GDM_GESTATIONAL_PRIMARY",
        "GDM_GESTATIONAL_ICU_TOTAL": "GDM_GESTATIONAL_PRIMARY",
        "GDM_GESTATIONAL_DEATH_TOTAL": "GDM_GESTATIONAL_PRIMARY",
        "DIABETES_PREGNANCY_BROAD_TOTAL": "DIABETES_PREGNANCY_BROAD_SENSITIVITY",
        "UNDERLYING_BASIC_CAUSE_CHILD_HEAT_THERMAL_PRIMARY_TOTAL": "CHILD_HEAT_THERMAL_PRIMARY",
        "UNDERLYING_BASIC_CAUSE_CHILD_DEHYDRATION_PRIMARY_TOTAL": "CHILD_DEHYDRATION_PRIMARY",
        "ANY_MENTION_CHILD_FLUID_ELECTROLYTE_SENSITIVITY_TOTAL": "CHILD_FLUID_ELECTROLYTE_SENSITIVITY",
        "UNDERLYING_BASIC_CAUSE_CHILD_DIARRHEA_GI_PRIMARY_TOTAL": "CHILD_DIARRHEA_GI_PRIMARY",
        "UNDERLYING_BASIC_CAUSE_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL": "CHILD_RESPIRATORY_ACUTE_PRIMARY",
        "ANY_MENTION_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL": "CHILD_ASTHMA_WHEEZE_SENSITIVITY",
        "ANY_MENTION_CHILD_RENAL_ACUTE_SENSITIVITY_TOTAL": "CHILD_RENAL_ACUTE_SENSITIVITY",
        "ANY_MENTION_CHILD_MALNUTRITION_SLOW_MOVING_SENSITIVITY_TOTAL": "CHILD_MALNUTRITION_SLOW_MOVING_SENSITIVITY",
        "ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_TOTAL": "EXCESSIVE_NATURAL_HEAT_X30",
    }
    icd_map: dict[str, dict[str, str]] = {}
    for variable_name, group_name in mapping.items():
        spec = group_specs[group_name]
        icd_map[variable_name] = {
            "icd10_codes": ", ".join(spec.rule_literals),
            "icd10_scope": spec.canonical_scope_sih if variable_name.startswith(("HDP_", "PRETERM_LABOR", "GDM_", "DIABETES_")) else spec.canonical_scope_sim,
        }
    aggregate_groups = {
        "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL": (DIRECT_HEAT_GROUPS, "UNDERLYING_BASIC_CAUSE"),
        "ANY_MENTION_DIRECT_HEAT_ANY_TOTAL": (DIRECT_HEAT_GROUPS, "ANY_MENTION"),
        "UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_TOTAL": (INDIRECT_HEAT_GROUPS, "UNDERLYING_BASIC_CAUSE"),
        "ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL": (INDIRECT_HEAT_GROUPS, "ANY_MENTION"),
        "UNDERLYING_BASIC_CAUSE_SLOW_MOVING_HEAT_ANY_TOTAL": (SLOW_MOVING_HEAT_GROUPS, "UNDERLYING_BASIC_CAUSE"),
        "ANY_MENTION_SLOW_MOVING_HEAT_ANY_TOTAL": (SLOW_MOVING_HEAT_GROUPS, "ANY_MENTION"),
    }
    for variable_name, (groups, scope) in aggregate_groups.items():
        codes = sorted({code for group_name in groups for code in group_specs[group_name].rule_literals})
        icd_map[variable_name] = {"icd10_codes": ", ".join(codes), "icd10_scope": scope}
    return icd_map


def _context_metadata(field: str) -> dict[str, str]:
    if field in CONTEXT_OVERRIDES:
        return CONTEXT_OVERRIDES[field]
    if field.endswith("_SOURCE_YEAR"):
        return {
            "label": field.replace("_", " ").title(),
            "definition": "Ano de referencia do insumo contextual carregado na gold.",
            "unit": "year",
            "construction": "Metadado de referencia temporal armazenado na gold contextual.",
        }
    if field.endswith("_EXPANSION_METHOD"):
        return {
            "label": field.replace("_", " ").title(),
            "definition": "Metodo de expansao temporal registrado na gold contextual.",
            "unit": "text",
            "construction": "Campo descritivo que documenta a regra de expansao usada no contexto gold.",
        }
    if field.endswith("_SOURCE"):
        return {
            "label": field.replace("_", " ").title(),
            "definition": "Fonte declarada para o insumo contextual carregado na gold.",
            "unit": "text",
            "construction": "Campo de linhagem da gold contextual.",
        }
    if field.endswith("_FLAG"):
        return {
            "label": field.replace("_", " ").title(),
            "definition": "Flag operacional armazenada na camada gold contextual.",
            "unit": "0/1 indicator",
            "construction": "Indicador operacional da gold contextual.",
        }
    return {}


def _is_float_type(column_type: str) -> bool:
    return any(token in column_type.upper() for token in FLOAT_TYPE_HINTS)


def _is_integer_type(column_type: str) -> bool:
    return any(token in column_type.upper() for token in INTEGER_TYPE_HINTS)


def _is_string_type(column_type: str) -> bool:
    return any(token in column_type.upper() for token in STRING_TYPE_HINTS)


def _is_temporal_type(column_type: str) -> bool:
    return any(token in column_type.upper() for token in TEMPORAL_TYPE_HINTS)


def _infer_theme(row: pd.Series) -> str:
    table_name = str(row["table_name"])
    variable_name = str(row["variable_name"])
    variable_class = str(row.get("variable_class", ""))
    metric_definition_types = str(row.get("metric_definition_types", ""))
    metric_families = str(row.get("metric_families", ""))
    if variable_name in IDENTIFIER_FIELDS or _is_temporal_type(str(row["column_type"])):
        return "Metadata"
    if table_name.startswith("gold_municipality_context"):
        return "Contextuais"
    if variable_name in CLIMATE_FIELDS or "climate" in metric_families or variable_class in {"humidity_and_co_exposures", "primary_exposures"}:
        return "Clima"
    if variable_name in PRIMARY_METRIC_FIELDS or "canonical" in metric_definition_types or "primary" in metric_definition_types or variable_class == "primary_outcomes":
        return "KPI"
    if variable_class in {"secondary_or_sensitivity_outcomes", "phenotype_flags"} or "sensitivity" in metric_definition_types:
        return "Secundarias"
    if variable_class in {"denominators_and_offsets", "quality_control_variables", "lineage_and_provenance"}:
        return "Operacionais"
    return "Outras"


def _infer_distribution_bucket(row: pd.Series) -> str:
    variable_name = str(row["variable_name"])
    column_type = str(row["column_type"])
    unit = str(row.get("unit", ""))
    approx_distinct = int(row.get("approx_distinct_non_missing", 0) or 0)
    non_missing_rows = int(row.get("non_missing_rows", 0) or 0)
    if variable_name in IDENTIFIER_FIELDS or unit in {"ISO date", "calendar year", "month number", "year"} or _is_temporal_type(column_type):
        return "metadata"
    if unit == "0/1 indicator" or variable_name.endswith("_FLAG") or (non_missing_rows > 0 and approx_distinct <= 2 and (_is_integer_type(column_type) or _is_float_type(column_type))):
        return "binary"
    if _is_string_type(column_type):
        return "categorical"
    if unit in CONTINUOUS_UNIT_HINTS:
        return "continuous"
    if unit in COUNT_UNIT_HINTS:
        return "count"
    if variable_name.startswith("OFFSET_LOG_"):
        return "continuous"
    if variable_name.endswith("_RATE") or "_RATE_" in variable_name or variable_name.startswith(("CRUDE_RATE_", "NEONATAL_MORTALITY_RATE_", "INFANT_MORTALITY_RATE_", "UNDER5_MORTALITY_RATE_")):
        return "continuous"
    if variable_name.startswith(("PROP_", "GINI_", "IDH_", "ALTITUDE_")):
        return "continuous"
    if _is_float_type(column_type):
        return "continuous"
    if _is_integer_type(column_type):
        if variable_name.endswith("_YEAR"):
            return "metadata"
        return "count"
    return "other"


def _relation_sql(spec: GoldTableSpec) -> str:
    return f"read_parquet('{spec.parquet_glob}', union_by_name=true)"


def _conditional_missingness_overrides(
    con: duckdb.DuckDBPyConnection,
    spec: GoldTableSpec,
    columns: set[str],
) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    prenatal_rule = CONDITIONAL_MISSINGNESS_SCOPES.get("PRENATAL_VISITS_COUNT")
    if prenatal_rule and "PRENATAL_VISITS_COUNT" in columns and set(prenatal_rule["required_columns"]).issubset(columns):
        prenatal_column = _quote_ident("PRENATAL_VISITS_COUNT")
        live_births_column = _quote_ident("LIVE_BIRTHS_TOTAL")
        applicable_expr = f"COALESCE(CAST({live_births_column} AS DOUBLE), 0) > 0"
        scan = con.execute(
            f"""
            SELECT
                SUM(CASE WHEN {applicable_expr} THEN 1 ELSE 0 END) AS applicable_rows,
                SUM(CASE WHEN {applicable_expr} AND {prenatal_column} IS NULL THEN 1 ELSE 0 END) AS missing_rows,
                COUNT(CASE WHEN {applicable_expr} THEN {prenatal_column} END) AS non_missing_rows,
                APPROX_COUNT_DISTINCT(CASE WHEN {applicable_expr} THEN {prenatal_column} END) AS approx_distinct_non_missing
            FROM {_relation_sql(spec)}
            """
        ).df().iloc[0]
        overrides["PRENATAL_VISITS_COUNT"] = {
            "missingness_denominator_rows": int(scan["applicable_rows"]),
            "missing_rows": int(scan["missing_rows"]),
            "non_missing_rows": int(scan["non_missing_rows"]),
            "approx_distinct_non_missing": int(scan["approx_distinct_non_missing"]),
            "missingness_scope": str(prenatal_rule["scope_label"]),
        }
    return overrides


def _scan_table_columns(con: duckdb.DuckDBPyConnection, spec: GoldTableSpec) -> tuple[pd.DataFrame, int]:
    describe = con.execute(f"DESCRIBE SELECT * FROM {_relation_sql(spec)}").df()
    columns = describe["column_name"].tolist()
    exprs = ["COUNT(*) AS total_rows"]
    for column_name in columns:
        qcol = _quote_ident(column_name)
        exprs.append(f"SUM(CASE WHEN {qcol} IS NULL THEN 1 ELSE 0 END) AS {_quote_ident(column_name + '__missing')}")
        exprs.append(f"COUNT({qcol}) AS {_quote_ident(column_name + '__non_missing')}")
        exprs.append(f"APPROX_COUNT_DISTINCT({qcol}) AS {_quote_ident(column_name + '__approx_distinct')}")
    scan = con.execute(f"SELECT {', '.join(exprs)} FROM {_relation_sql(spec)}").df().iloc[0]
    total_rows = int(scan["total_rows"])
    describe["table_name"] = spec.table_name
    describe["layer"] = spec.layer
    describe["plot_priority"] = spec.plot_priority
    describe["total_rows"] = total_rows
    describe["missing_rows"] = describe["column_name"].map(lambda name: int(scan[f"{name}__missing"]))
    describe["non_missing_rows"] = describe["column_name"].map(lambda name: int(scan[f"{name}__non_missing"]))
    describe["approx_distinct_non_missing"] = describe["column_name"].map(lambda name: int(scan[f"{name}__approx_distinct"]))
    describe["missingness_denominator_rows"] = total_rows
    describe["missingness_scope"] = "all rows"
    for column_name, override in _conditional_missingness_overrides(con, spec, set(columns)).items():
        mask = describe["column_name"].eq(column_name)
        if not mask.any():
            continue
        describe.loc[mask, "missing_rows"] = int(override["missing_rows"])
        describe.loc[mask, "non_missing_rows"] = int(override["non_missing_rows"])
        describe.loc[mask, "approx_distinct_non_missing"] = int(override["approx_distinct_non_missing"])
        describe.loc[mask, "missingness_denominator_rows"] = int(override["missingness_denominator_rows"])
        describe.loc[mask, "missingness_scope"] = str(override["missingness_scope"])
    describe["missing_pct"] = np.where(
        describe["missingness_denominator_rows"] == 0,
        0.0,
        describe["missing_rows"] / describe["missingness_denominator_rows"],
    )
    return describe, total_rows


def _build_catalog(
    root: Path,
    con: duckdb.DuckDBPyConnection,
    specs: list[GoldTableSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dictionary = build_gold_variable_dictionary(root, START_YEAR, END_YEAR)["dictionary"].copy()
    doc_lookup = dictionary.set_index(["table_name", "variable_name"]).to_dict(orient="index")
    indicator_registry = _load_registry_json(root / "config/dashboard/indicator_registry.yaml")
    numerator_map, denominator_map = _build_metric_maps(indicator_registry["metrics"])
    icd_map = _build_icd_maps()
    catalog_frames: list[pd.DataFrame] = []
    table_inventory_rows: list[dict[str, Any]] = []
    for spec in specs:
        describe, total_rows = _scan_table_columns(con, spec)
        rows: list[dict[str, Any]] = []
        for item in describe.to_dict(orient="records"):
            variable_name = str(item["column_name"])
            doc = doc_lookup.get((spec.table_name, variable_name), {})
            context_meta = _context_metadata(variable_name) if spec.table_name.startswith("gold_municipality_context") else {}
            metric_meta = numerator_map.get((spec.table_name, variable_name), {})
            row = {
                "table_name": spec.table_name,
                "table_label": TABLE_LABELS.get(spec.table_name, spec.table_name),
                "layer": spec.layer,
                "plot_priority": spec.plot_priority,
                "variable_name": variable_name,
                "column_type": str(item["column_type"]),
                "total_rows": total_rows,
                "missingness_denominator_rows": int(item.get("missingness_denominator_rows", total_rows)),
                "missingness_scope": str(item.get("missingness_scope", "all rows")),
                "non_missing_rows": int(item["non_missing_rows"]),
                "missing_rows": int(item["missing_rows"]),
                "missing_pct": float(item["missing_pct"]),
                "approx_distinct_non_missing": int(item["approx_distinct_non_missing"]),
                "variable_class": _sanitize_text(doc.get("variable_class", "")),
                "label": _sanitize_text(context_meta.get("label") or doc.get("plain_english_label", "")),
                "definition": _sanitize_text(context_meta.get("definition") or doc.get("definition", "")),
                "unit": _sanitize_text(context_meta.get("unit") or doc.get("unit", "")),
                "primary_or_sensitivity": _sanitize_text(doc.get("primary_or_sensitivity", "")),
                "denominator_rule": _sanitize_text(doc.get("denominator_rule", "")),
                "notes_on_use": _sanitize_text(doc.get("notes_on_use", "")),
                "metric_ids": metric_meta.get("metric_ids", ""),
                "metric_labels": metric_meta.get("metric_labels", ""),
                "metric_families": metric_meta.get("metric_families", ""),
                "metric_definition_types": metric_meta.get("metric_definition_types", ""),
                "dashboard_denominator_field": metric_meta.get("dashboard_denominator_field", context_meta.get("denominator", "")),
                "aggregation_methods": metric_meta.get("aggregation_methods", ""),
                "rate_display_units": metric_meta.get("rate_display_units", ""),
                "used_as_denominator_metric_ids": denominator_map.get((spec.table_name, variable_name), ""),
                "icd10_codes": icd_map.get(variable_name, {}).get("icd10_codes", ""),
                "icd10_scope": icd_map.get(variable_name, {}).get("icd10_scope", ""),
                "manual_construction": context_meta.get("construction", ""),
                "manual_numerator": context_meta.get("numerator", metric_meta.get("metric_ids", "") and variable_name or ""),
                "manual_denominator": context_meta.get("denominator", ""),
            }
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame["theme_group"] = frame.apply(_infer_theme, axis=1)
        frame["distribution_bucket"] = frame.apply(_infer_distribution_bucket, axis=1)
        catalog_frames.append(frame)
        table_inventory_rows.append(
            {
                "table_name": spec.table_name,
                "table_label": TABLE_LABELS.get(spec.table_name, spec.table_name),
                "layer": spec.layer,
                "plot_priority": spec.plot_priority,
                "rows": total_rows,
                "variables": int(frame["variable_name"].nunique()),
                "years": "static" if spec.table_name == "gold_municipality_context_static" else f"{START_YEAR}-{END_YEAR}",
            }
        )
    catalog = pd.concat(catalog_frames, ignore_index=True)
    catalog["non_missing_pct"] = np.where(
        catalog["missingness_denominator_rows"] == 0,
        0.0,
        catalog["non_missing_rows"] / catalog["missingness_denominator_rows"],
    )
    catalog["construction_summary"] = catalog.apply(_build_construction_summary, axis=1)
    catalog["label"] = catalog["label"].replace("", pd.NA).fillna(catalog["variable_name"].str.replace("_", " ", regex=False).str.title())
    table_inventory = pd.DataFrame(table_inventory_rows).sort_values(["plot_priority", "table_name"], kind="mergesort")
    return catalog, table_inventory.drop(columns=["plot_priority"], errors="ignore")


def _build_construction_summary(row: pd.Series) -> str:
    parts: list[str] = [f"Presente em {row['table_name']}."]
    manual_construction = _sanitize_text(row.get("manual_construction", ""))
    definition = _sanitize_text(row.get("definition", ""))
    if manual_construction:
        parts.append(manual_construction)
    elif definition:
        parts.append(definition)
    if row.get("metric_ids"):
        parts.append(f"Usada como numerador em: {row['metric_ids']}.")
    if row.get("manual_numerator") and not row.get("metric_ids"):
        parts.append(f"Numerador: {row['manual_numerator']}.")
    denominator_field = _sanitize_text(row.get("manual_denominator") or row.get("dashboard_denominator_field", ""))
    if denominator_field:
        parts.append(f"Denominador: {denominator_field}.")
    if row.get("used_as_denominator_metric_ids"):
        parts.append(f"Usada como denominador em: {row['used_as_denominator_metric_ids']}.")
    if row.get("aggregation_methods"):
        parts.append(f"Agregacao dashboard: {row['aggregation_methods']}.")
    if row.get("icd10_codes"):
        scope = _sanitize_text(row.get("icd10_scope", ""))
        if scope:
            parts.append(f"CID-10: {row['icd10_codes']} (escopo {scope}).")
        else:
            parts.append(f"CID-10: {row['icd10_codes']}.")
    if row.get("notes_on_use"):
        parts.append(_sanitize_text(row["notes_on_use"]))
    return " ".join(part.strip() for part in parts if part).strip()


def _build_plot_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    plot_ready = catalog[
        catalog["distribution_bucket"].isin({"continuous", "count", "categorical", "binary"})
        & ~catalog["distribution_bucket"].eq("metadata")
        & ~catalog["theme_group"].eq("Metadata")
    ].copy()
    plot_ready = plot_ready.sort_values(["variable_name", "plot_priority", "table_name"], kind="mergesort")
    plot_catalog = plot_ready.drop_duplicates(subset=["variable_name"], keep="first").reset_index(drop=True)
    return plot_catalog


def _load_plot_samples(con: duckdb.DuckDBPyConnection, plot_catalog: pd.DataFrame, specs: dict[str, GoldTableSpec]) -> dict[str, pd.DataFrame]:
    samples: dict[str, pd.DataFrame] = {}
    for table_name in sorted(plot_catalog["table_name"].unique(), key=lambda value: TABLE_PRIORITY[value]):
        spec = specs[table_name]
        relation = _relation_sql(spec)
        total_rows = int(plot_catalog.loc[plot_catalog["table_name"] == table_name, "total_rows"].iloc[0])
        if total_rows <= PLOT_SAMPLE_ROWS or table_name.startswith("gold_municipality_context"):
            query = f"SELECT * FROM {relation}"
        else:
            query = f"SELECT * FROM {relation} USING SAMPLE reservoir({PLOT_SAMPLE_ROWS} ROWS) REPEATABLE (42)"
        samples[table_name] = con.execute(query).df()
    return samples


def _build_numeric_summary(
    con: duckdb.DuckDBPyConnection,
    plot_catalog: pd.DataFrame,
    specs: dict[str, GoldTableSpec],
    bucket: str,
) -> pd.DataFrame:
    bucket_rows = plot_catalog[plot_catalog["distribution_bucket"] == bucket].copy()
    if bucket_rows.empty:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for table_name, group in bucket_rows.groupby("table_name", sort=False):
        spec = specs[table_name]
        relation = _relation_sql(spec)
        for column_chunk in _chunks(group["variable_name"].tolist(), 12):
            exprs: list[str] = []
            for variable_name in column_chunk:
                qcol = _quote_ident(variable_name)
                exprs.extend(
                    [
                        f"COUNT({qcol}) AS {_quote_ident(variable_name + '__n')}",
                        f"AVG(CAST({qcol} AS DOUBLE)) AS {_quote_ident(variable_name + '__mean')}",
                        f"STDDEV_SAMP(CAST({qcol} AS DOUBLE)) AS {_quote_ident(variable_name + '__sd')}",
                        f"MIN(CAST({qcol} AS DOUBLE)) AS {_quote_ident(variable_name + '__min')}",
                        f"QUANTILE_CONT(CAST({qcol} AS DOUBLE), 0.25) AS {_quote_ident(variable_name + '__p25')}",
                        f"QUANTILE_CONT(CAST({qcol} AS DOUBLE), 0.5) AS {_quote_ident(variable_name + '__median')}",
                        f"QUANTILE_CONT(CAST({qcol} AS DOUBLE), 0.75) AS {_quote_ident(variable_name + '__p75')}",
                        f"MAX(CAST({qcol} AS DOUBLE)) AS {_quote_ident(variable_name + '__max')}",
                        f"SUM(CASE WHEN {qcol} = 0 THEN 1 ELSE 0 END) AS {_quote_ident(variable_name + '__zeros')}",
                    ]
                )
            stats = con.execute(f"SELECT {', '.join(exprs)} FROM {relation}").df().iloc[0]
            rows: list[dict[str, Any]] = []
            for variable_name in column_chunk:
                meta = group.loc[group["variable_name"] == variable_name].iloc[0]
                rows.append(
                    {
                        "table_name": table_name,
                        "table_label": meta["table_label"],
                        "variable_name": variable_name,
                        "label": meta["label"],
                        "theme_group": meta["theme_group"],
                        "distribution_bucket": bucket,
                        "n_non_missing": int(stats[f"{variable_name}__n"]),
                        "mean": stats[f"{variable_name}__mean"],
                        "sd": stats[f"{variable_name}__sd"],
                        "min": stats[f"{variable_name}__min"],
                        "p25": stats[f"{variable_name}__p25"],
                        "median": stats[f"{variable_name}__median"],
                        "p75": stats[f"{variable_name}__p75"],
                        "max": stats[f"{variable_name}__max"],
                        "zero_count": int(stats[f"{variable_name}__zeros"]),
                        "missing_rows": int(meta["missing_rows"]),
                        "missing_pct": float(meta["missing_pct"]),
                        "unit": meta["unit"],
                    }
                )
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    summary = pd.concat(frames, ignore_index=True)
    return summary.sort_values(["theme_group", "table_name", "variable_name"], kind="mergesort").reset_index(drop=True)


def _build_categorical_summary(
    con: duckdb.DuckDBPyConnection,
    plot_catalog: pd.DataFrame,
    specs: dict[str, GoldTableSpec],
) -> pd.DataFrame:
    bucket_rows = plot_catalog[plot_catalog["distribution_bucket"].isin({"binary", "categorical"})].copy()
    if bucket_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for meta in bucket_rows.to_dict(orient="records"):
        spec = specs[str(meta["table_name"])]
        relation = _relation_sql(spec)
        variable_name = str(meta["variable_name"])
        qcol = _quote_ident(variable_name)
        values = con.execute(
            f"""
            SELECT CAST({qcol} AS VARCHAR) AS value, COUNT(*) AS n
            FROM {relation}
            WHERE {qcol} IS NOT NULL
            GROUP BY 1
            ORDER BY n DESC, value
            """
        ).df()
        total_non_missing = max(int(meta["non_missing_rows"]), 1)
        for order, item in enumerate(values.to_dict(orient="records"), start=1):
            rows.append(
                {
                    "table_name": meta["table_name"],
                    "table_label": meta["table_label"],
                    "variable_name": variable_name,
                    "label": meta["label"],
                    "theme_group": meta["theme_group"],
                    "distribution_bucket": meta["distribution_bucket"],
                    "value": item["value"],
                    "n": int(item["n"]),
                    "pct_non_missing": float(item["n"]) / total_non_missing,
                    "order_in_variable": order,
                    "missing_rows": int(meta["missing_rows"]),
                    "missing_pct": float(meta["missing_pct"]),
                }
            )
        rows.append(
            {
                "table_name": meta["table_name"],
                "table_label": meta["table_label"],
                "variable_name": variable_name,
                "label": meta["label"],
                "theme_group": meta["theme_group"],
                "distribution_bucket": meta["distribution_bucket"],
                "value": "(missing)",
                "n": int(meta["missing_rows"]),
                "pct_non_missing": np.nan,
                "order_in_variable": len(values) + 1,
                "missing_rows": int(meta["missing_rows"]),
                "missing_pct": float(meta["missing_pct"]),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["theme_group", "variable_name", "order_in_variable"], kind="mergesort").reset_index(drop=True)


def _series_from_sample(samples: dict[str, pd.DataFrame], table_name: str, variable_name: str) -> pd.Series:
    frame = samples[table_name]
    if variable_name not in frame.columns:
        return pd.Series(dtype="float64")
    series = pd.to_numeric(frame[variable_name], errors="coerce").dropna()
    if series.empty:
        return pd.Series(dtype="float64")
    return series.astype(float)


def _render_cover(
    pdf: PdfPages,
    table_inventory: pd.DataFrame,
    catalog: pd.DataFrame,
    plot_catalog: pd.DataFrame,
    artifacts: ArtifactPaths,
) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.text(0.03, 0.94, "Round 3: analise descritiva gold-only", fontsize=20, fontweight="bold", transform=ax.transAxes)
    ax.text(0.03, 0.88, "Escopo", fontsize=13, fontweight="bold", transform=ax.transAxes)
    scope_lines = [
        "Somente tabelas em data/gold foram usadas nesta rodada.",
        "Camadas silver, bronze e raw foram ignoradas na construcao do PDF e das tabelas auxiliares.",
        f"Periodo coberto: {START_YEAR}-{END_YEAR}.",
        "As distribuicoes visuais deduplicam variaveis repetidas e priorizam tabelas gold canonicas.",
        f"Artefatos auxiliares: {artifacts.tables_dir.relative_to(artifacts.output_dir.parent.parent)}",
    ]
    ax.text(0.03, 0.78, "\n".join(scope_lines), fontsize=11, va="top", transform=ax.transAxes)
    total_rows = int(table_inventory["rows"].sum())
    total_variables = int(catalog.shape[0])
    unique_variables = int(catalog["variable_name"].nunique())
    visual_variables = int(plot_catalog["variable_name"].nunique())
    overview_lines = [
        f"Tabelas gold/derived gold analisadas: {table_inventory.shape[0]}",
        f"Linhas avaliadas somadas: {total_rows:,}",
        f"Entradas tabela-variavel: {total_variables:,}",
        f"Variaveis unicas observadas: {unique_variables:,}",
        f"Variaveis com painel visual: {visual_variables:,}",
    ]
    ax.text(0.03, 0.52, "Cobertura", fontsize=13, fontweight="bold", transform=ax.transAxes)
    ax.text(0.03, 0.44, "\n".join(overview_lines), fontsize=11, va="top", transform=ax.transAxes)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    ax.text(0.03, 0.08, f"Gerado em {generated}. Saida principal: {artifacts.pdf_path.name}", fontsize=10, color="#444444", transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_inventory_page(pdf: PdfPages, table_inventory: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title("Inventario das tabelas gold", loc="left", fontsize=16, fontweight="bold", pad=16)
    inventory = table_inventory[["table_label", "layer", "rows", "variables", "years"]].copy()
    inventory["rows"] = inventory["rows"].map(lambda value: f"{int(value):,}")
    table = ax.table(cellText=inventory.values, colLabels=["Tabela", "Layer", "Linhas", "Variaveis", "Anos"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_missingness_summary(pdf: PdfPages, missingness: pd.DataFrame) -> None:
    top = missingness.sort_values("missing_pct", ascending=False).head(20).copy()
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.barh(top["table_name"] + " | " + top["variable_name"], top["missing_pct"] * 100, color="#b33c3c")
    ax.invert_yaxis()
    ax.set_title("Top 20 variaveis com maior proporcao de missings", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("% missing")
    ax.grid(axis="x", alpha=0.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_continuous_boxplots(pdf: PdfPages, continuous: pd.DataFrame, samples: dict[str, pd.DataFrame]) -> None:
    if continuous.empty:
        return
    for theme, group in continuous.groupby("theme_group", sort=False):
        records = group.to_dict(orient="records")
        for page_start in range(0, len(records), 6):
            page_records = records[page_start : page_start + 6]
            fig, axes = plt.subplots(len(page_records), 1, figsize=(11.69, 8.27))
            if len(page_records) == 1:
                axes = [axes]
            fig.suptitle(f"Boxplots: variaveis continuas ({theme})", x=0.03, y=0.98, ha="left", fontsize=16, fontweight="bold")
            for axis, record in zip(axes, page_records, strict=False):
                values = _series_from_sample(samples, str(record["table_name"]), str(record["variable_name"]))
                if values.empty:
                    axis.text(0.5, 0.5, "Sem valores nao nulos na amostra", ha="center", va="center")
                    axis.axis("off")
                    continue
                axis.boxplot(values.to_numpy(), vert=False, patch_artist=True, whis=(5, 95), boxprops={"facecolor": "#8ebad9"})
                axis.set_title(f"{record['label']} [{record['table_name']}]", loc="left", fontsize=11)
                axis.grid(axis="x", alpha=0.2)
                axis.set_xlabel(record.get("unit", "") or "valor")
                axis.text(
                    0.99,
                    0.8,
                    f"n={int(record['n_non_missing']):,} | missing={record['missing_pct']*100:.2f}%",
                    ha="right",
                    va="center",
                    transform=axis.transAxes,
                    fontsize=9,
                    color="#444444",
                )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _render_count_histograms(pdf: PdfPages, counts: pd.DataFrame, samples: dict[str, pd.DataFrame]) -> None:
    if counts.empty:
        return
    records = counts.to_dict(orient="records")
    for page_start in range(0, len(records), 4):
        page_records = records[page_start : page_start + 4]
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        axes = axes.flatten()
        fig.suptitle("Histogramas: variaveis de contagem", x=0.03, y=0.98, ha="left", fontsize=16, fontweight="bold")
        for axis, record in zip(axes, page_records, strict=False):
            values = _series_from_sample(samples, str(record["table_name"]), str(record["variable_name"]))
            if values.empty:
                axis.text(0.5, 0.5, "Sem valores nao nulos na amostra", ha="center", va="center")
                axis.axis("off")
                continue
            clip_upper = float(values.quantile(0.99)) if values.nunique() > 20 else float(values.max())
            clipped = values.clip(upper=clip_upper)
            bins = min(30, max(10, int(clipped.nunique())))
            axis.hist(clipped.to_numpy(), bins=bins, color="#d98e73", edgecolor="white")
            axis.set_title(f"{record['label']} [{record['table_name']}]", loc="left", fontsize=11)
            axis.grid(axis="y", alpha=0.2)
            note = f"n={int(record['n_non_missing']):,} | zeros={int(record['zero_count']):,} | missing={record['missing_pct']*100:.2f}%"
            if clip_upper < float(values.max()):
                note = f"{note} | clip p99={clip_upper:.2f}"
            axis.text(0.99, 0.9, note, ha="right", va="top", transform=axis.transAxes, fontsize=8, color="#444444")
        for axis in axes[len(page_records) :]:
            axis.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _table_pages(
    pdf: PdfPages,
    frame: pd.DataFrame,
    title: str,
    subtitle: str,
    columns: list[str],
    rename: dict[str, str],
    rows_per_page: int = TABLE_PAGE_ROWS,
) -> None:
    if frame.empty:
        return
    render = frame[columns].copy()
    render = render.rename(columns=rename)
    for column in render.columns:
        if render[column].dtype.kind in {"f"}:
            if "pct" in column.lower():
                render[column] = render[column].map(lambda value: "" if pd.isna(value) else f"{value*100:.2f}%")
            else:
                render[column] = render[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
        else:
            render[column] = render[column].map(lambda value: _sanitize_text(value))
    wrap_widths = {column: 24 for column in render.columns}
    for column in render.columns:
        if "Resumo" in column or "Construcao" in column or "Definicao" in column:
            wrap_widths[column] = 42
        elif "Tabela" in column or "Variavel" in column:
            wrap_widths[column] = 28
    for start in range(0, len(render), rows_per_page):
        page = render.iloc[start : start + rows_per_page].copy()
        for column in page.columns:
            page[column] = page[column].map(lambda value: _wrap_text(value, wrap_widths[column]))
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=18)
        ax.text(0.0, 1.01, subtitle, fontsize=10, color="#444444", transform=ax.transAxes)
        table = ax.table(cellText=page.values, colLabels=page.columns.tolist(), loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(7.2)
        table.scale(1, 1.55)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def build_round3_gold_report(root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    artifacts = _artifact_paths(root, output_dir=output_dir)
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts.tables_dir.mkdir(parents=True, exist_ok=True)

    specs = _table_specs(root)
    spec_map = {spec.table_name: spec for spec in specs}
    con = duckdb.connect()
    catalog, table_inventory = _build_catalog(root, con, specs)
    plot_catalog = _build_plot_catalog(catalog)
    continuous_summary = _build_numeric_summary(con, plot_catalog, spec_map, "continuous")
    count_summary = _build_numeric_summary(con, plot_catalog, spec_map, "count")
    categorical_summary = _build_categorical_summary(con, plot_catalog, spec_map)
    missingness = catalog[
        [
            "table_name",
            "table_label",
            "variable_name",
            "label",
            "theme_group",
            "distribution_bucket",
            "total_rows",
            "missingness_denominator_rows",
            "missingness_scope",
            "non_missing_rows",
            "missing_rows",
            "missing_pct",
            "approx_distinct_non_missing",
        ]
    ].sort_values(["missing_pct", "table_name", "variable_name"], ascending=[False, True, True], kind="mergesort")
    schema_full = catalog[
        [
            "table_name",
            "table_label",
            "layer",
            "variable_name",
            "column_type",
            "theme_group",
            "distribution_bucket",
            "label",
            "definition",
            "unit",
            "variable_class",
            "primary_or_sensitivity",
            "metric_ids",
            "metric_labels",
            "dashboard_denominator_field",
            "used_as_denominator_metric_ids",
            "aggregation_methods",
            "rate_display_units",
            "icd10_scope",
            "icd10_codes",
            "denominator_rule",
            "construction_summary",
            "notes_on_use",
            "total_rows",
            "missingness_denominator_rows",
            "missingness_scope",
            "non_missing_rows",
            "missing_rows",
            "missing_pct",
            "approx_distinct_non_missing",
        ]
    ].sort_values(["table_name", "variable_name"], kind="mergesort")
    schema_compact = schema_full[
        [
            "table_name",
            "variable_name",
            "theme_group",
            "distribution_bucket",
            "label",
            "metric_ids",
            "dashboard_denominator_field",
            "icd10_codes",
            "construction_summary",
            "missing_pct",
        ]
    ].copy()

    schema_full.to_csv(artifacts.schema_full_path, index=False)
    schema_compact.to_csv(artifacts.schema_compact_path, index=False)
    missingness.to_csv(artifacts.missingness_path, index=False)
    continuous_summary.to_csv(artifacts.continuous_summary_path, index=False)
    count_summary.to_csv(artifacts.count_summary_path, index=False)
    categorical_summary.to_csv(artifacts.categorical_summary_path, index=False)

    samples = _load_plot_samples(con, plot_catalog, spec_map)
    with PdfPages(artifacts.pdf_path) as pdf:
        _render_cover(pdf, table_inventory, catalog, plot_catalog, artifacts)
        _render_inventory_page(pdf, table_inventory)
        _render_missingness_summary(pdf, missingness)
        _render_continuous_boxplots(pdf, continuous_summary, samples)
        _render_count_histograms(pdf, count_summary, samples)
        _table_pages(
            pdf,
            categorical_summary,
            title="Tabela de valores: variaveis dicotomicas e categoricas",
            subtitle="Valores exatos calculados no gold; a linha (missing) resume os ausentes.",
            columns=["table_name", "variable_name", "value", "n", "pct_non_missing", "missing_pct"],
            rename={
                "table_name": "Tabela",
                "variable_name": "Variavel",
                "value": "Valor",
                "n": "N",
                "pct_non_missing": "% entre nao missing",
                "missing_pct": "% missing",
            },
            rows_per_page=24,
        )
        _table_pages(
            pdf,
            missingness,
            title="Missingness de todas as variaveis",
            subtitle="Appendix completa de missings por tabela e variavel.",
            columns=[
                "table_name",
                "variable_name",
                "theme_group",
                "distribution_bucket",
                "missingness_denominator_rows",
                "missingness_scope",
                "missing_rows",
                "missing_pct",
                "approx_distinct_non_missing",
            ],
            rename={
                "table_name": "Tabela",
                "variable_name": "Variavel",
                "theme_group": "Grupo",
                "distribution_bucket": "Tipo",
                "missingness_denominator_rows": "Linhas avaliadas",
                "missingness_scope": "Escopo",
                "missing_rows": "Missings",
                "missing_pct": "% missing",
                "approx_distinct_non_missing": "Distintos aprox",
            },
            rows_per_page=26,
        )
        _table_pages(
            pdf,
            schema_compact,
            title="Esquema descritivo de construcao das variaveis",
            subtitle="Nome na base, papel analitico, CID-10 quando aplicavel, denominadores e nota resumida de construcao.",
            columns=["table_name", "variable_name", "theme_group", "distribution_bucket", "metric_ids", "dashboard_denominator_field", "icd10_codes", "construction_summary"],
            rename={
                "table_name": "Tabela",
                "variable_name": "Variavel",
                "theme_group": "Grupo",
                "distribution_bucket": "Tipo",
                "metric_ids": "Metricas",
                "dashboard_denominator_field": "Denominador",
                "icd10_codes": "CID-10",
                "construction_summary": "Construcao",
            },
            rows_per_page=18,
        )

    manifest = {
        "generated_at_utc": _utc_now(),
        "scope": "gold_only",
        "ignored_layers": ["silver", "bronze", "raw"],
        "years": list(range(START_YEAR, END_YEAR + 1)),
        "source_tables": table_inventory.to_dict(orient="records"),
        "visual_deduplication_rule": "Repeated variable names were plotted once, prioritizing canonical gold tables and the static contextual table before derived views.",
        "sample_rows_per_table_for_plots": PLOT_SAMPLE_ROWS,
        "outputs": {
            "pdf": str(artifacts.pdf_path.relative_to(root)),
            "schema_full_csv": str(artifacts.schema_full_path.relative_to(root)),
            "schema_compact_csv": str(artifacts.schema_compact_path.relative_to(root)),
            "missingness_csv": str(artifacts.missingness_path.relative_to(root)),
            "continuous_summary_csv": str(artifacts.continuous_summary_path.relative_to(root)),
            "count_summary_csv": str(artifacts.count_summary_path.relative_to(root)),
            "categorical_summary_csv": str(artifacts.categorical_summary_path.relative_to(root)),
        },
    }
    artifacts.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    con.close()
    return {
        "report_dir": str(artifacts.output_dir),
        "pdf_path": str(artifacts.pdf_path),
        "manifest_path": str(artifacts.manifest_path),
        "schema_full_path": str(artifacts.schema_full_path),
        "schema_compact_path": str(artifacts.schema_compact_path),
        "missingness_path": str(artifacts.missingness_path),
        "continuous_summary_path": str(artifacts.continuous_summary_path),
        "count_summary_path": str(artifacts.count_summary_path),
        "categorical_summary_path": str(artifacts.categorical_summary_path),
    }
