from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterable

import duckdb

from ..dashboard_registry import (
    GoldDatasetLineageDefinition,
    MetricDefinition,
    PageDefinition,
    SourceSystemDefinition,
    load_context_registry,
    load_gold_dataset_lineage,
    load_medallion_layers,
    load_metric_registry,
    load_page_registry,
    load_source_system_registry,
)
from .schemas import DashboardFilters, RelationshipFilters


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_list(values: Iterable[str | int]) -> str:
    items = []
    for value in values:
        if isinstance(value, int):
            items.append(str(value))
        else:
            items.append(_sql_quote(str(value)))
    return ", ".join(items)


def _uf_to_region_case(alias: str = "m", default: str = "'Unknown region'") -> str:
    clauses = " ".join(
        f"WHEN {alias}.uf = {_sql_quote(uf)} THEN {_sql_quote(region)}"
        for uf, region in UF_TO_REGION_LABEL.items()
    )
    return f"CASE {clauses} ELSE {default} END"


FIELD_LABELS: dict[str, str] = {
    "PRETERM_BIRTHS_CANONICAL_TOTAL": "Canonical preterm live births",
    "PRETERM_FLAG": "Live births before 37 completed weeks",
    "VERY_PRETERM_FLAG": "Live births before 32 completed weeks",
    "STILLBIRTH_TOTAL_PRIMARY_GTE28": "Stillbirth events at 28 or more completed weeks",
    "STILLBIRTH_TOTAL_GTE22_SENSITIVITY": "Stillbirth events at 22 or more completed weeks",
    "PRENATAL_VISITS_6PLUS_FLAG": "Births with six or more prenatal visits",
    "PRENATAL_VISITS_8PLUS_FLAG": "Births with eight or more prenatal visits",
    "LOW_BIRTH_WEIGHT_LT2500_TOTAL": "Live births weighing less than 2500 grams",
    "PRETERM_LOW_BIRTH_WEIGHT_TOTAL": "Preterm live births weighing less than 2500 grams",
    "GEST_AGE_CANONICAL": "Canonical gestational age in completed weeks",
    "GEST_DAYS_LOST_BELOW_37W": "Gestational days lost below 37 completed weeks",
    "DEATHS_NEONATAL": "Deaths from 0 to 27 completed days of life",
    "DEATHS_U1": "Deaths before one year of age",
    "DEATHS_U5": "Deaths before five completed years of age",
    "DEATHS_UNDER5_TOTAL": "Deaths before five completed years of age",
    "DEATHS_0_27D": "Deaths from 0 to 27 completed days of life",
    "DEATHS_28_364D": "Deaths from 28 to 364 completed days of life",
    "DEATHS_1_4Y": "Deaths from one to four completed years of age",
    "OBSTETRIC_ADMISSIONS_TOTAL": "Obstetric hospital admissions",
    "HDP_ANY_TOTAL": "Admissions classified as hypertensive disorders of pregnancy",
    "HDP_SEVERE_TOTAL": "Admissions classified as severe hypertensive disorders of pregnancy",
    "GDM_GESTATIONAL_TOTAL": "Admissions classified as gestational diabetes mellitus",
    "PRETERM_LABOR_TOTAL": "Admissions classified as preterm labour",
    "APH_ANTEPARTUM_TOTAL": "Admissions classified as antepartum haemorrhage",
    "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL": "Child admissions with dehydration in principal diagnosis",
    "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL": "Child admissions with diarrhoeal or gastrointestinal disease in principal diagnosis",
    "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL": "Child admissions with acute respiratory disease in principal diagnosis",
    "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL": "Child admissions with asthma or wheeze in any diagnosis field",
    "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL": "Child direct heat deaths by underlying or basic cause",
    "UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_TOTAL": "Child indirect heat-sensitive deaths by underlying or basic cause",
    "ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_TOTAL": "Child deaths with any mention of excessive natural heat X30",
    "ANY_MENTION_DIRECT_HEAT_ANY_TOTAL": "Child deaths with any mention of direct heat",
    "ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL": "Child deaths with any mention of indirect heat-sensitive causes",
    "TMAX_C": "Daily maximum temperature",
    "RH_MEAN_PCT": "Daily mean relative humidity",
    "HEAT_INDEX_APPROX_C": "Approximate heat index derived from temperature and humidity",
    "HEATWAVE_LEN_P95": "Heatwave duration under the P95 sensitivity definition",
    "REF_LIVE_BIRTHS_DAILY": "Live births total in the municipality-day panel",
    "LIVE_BIRTHS_TOTAL": "Live births total",
    "LIVE_BIRTHS_TOTAL_ANNUAL": "Live births total in the annual mortality denominator",
    "TOTAL_BIRTHS_PRIMARY": "Total births under the primary stillbirth definition",
    "TOTAL_BIRTHS_SENSITIVITY_GTE22": "Total births under the sensitivity stillbirth definition",
    "POP_U5": "Under-five population",
    "POP_WRA_15_49": "Women aged 15 to 49 years population",
    "POP_WRA_15_45_SENSITIVITY": "Women aged 15 to 45 years population",
    "NV_REGISTRO_UTIL_IG_TOTAL": "Birth records with usable gestational age",
}


SOURCE_SYSTEM_BRONZE_MAP: dict[str, list[str]] = {
    "sinasc": ["bronze_sinasc"],
    "sim": ["bronze_sim"],
    "sih_sus": ["bronze_sih"],
    "stillbirth_registry": ["bronze_stillbirth"],
    "ibge_population": ["bronze_population"],
    "era5_land": ["bronze_climate"],
}


SOURCE_SYSTEM_SILVER_MAP: dict[str, list[str]] = {
    "sinasc": ["silver_sinasc_event"],
    "sim": ["silver_sim_deaths"],
    "sih_sus": ["silver_sih_admissions"],
    "stillbirth_registry": ["silver_stillbirth_individual"],
    "ibge_population": ["silver_population"],
    "ibge_geography": ["silver_geography_reference"],
    "era5_land": ["silver_climate_daily"],
}


UF_TO_REGION_LABEL: dict[str, str] = {
    "AC": "North",
    "AP": "North",
    "AM": "North",
    "PA": "North",
    "RO": "North",
    "RR": "North",
    "TO": "North",
    "AL": "Northeast",
    "BA": "Northeast",
    "CE": "Northeast",
    "MA": "Northeast",
    "PB": "Northeast",
    "PE": "Northeast",
    "PI": "Northeast",
    "RN": "Northeast",
    "SE": "Northeast",
    "ES": "Southeast",
    "MG": "Southeast",
    "RJ": "Southeast",
    "SP": "Southeast",
    "PR": "South",
    "RS": "South",
    "SC": "South",
    "DF": "Central-West",
    "GO": "Central-West",
    "MS": "Central-West",
    "MT": "Central-West",
}


class DashboardRepository:
    def __init__(self, root: Path, dashboard_dir: Path):
        self.root = root
        self.dashboard_dir = dashboard_dir
        manifest_path = dashboard_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.pages = load_page_registry(root)
        self.page_map = {page.page_id: page for page in self.pages}
        self.metrics = load_metric_registry(root)
        self.metric_map = {metric.metric_id: metric for metric in self.metrics}
        self.context_variables = load_context_registry(root)
        self.context_variable_map = {variable.context_variable: variable for variable in self.context_variables}
        self.source_systems = load_source_system_registry(root)
        self.source_system_map = {system.source_system_id: system for system in self.source_systems}
        self.medallion_layers = load_medallion_layers(root)
        self.gold_dataset_lineage = load_gold_dataset_lineage(root)
        self.gold_dataset_lineage_map = {lineage.gold_dataset_id: lineage for lineage in self.gold_dataset_lineage}
        self._lock = threading.Lock()
        self._connection = duckdb.connect(str(dashboard_dir / "observa_infancia.duckdb"), read_only=False)
        self._ensure_tables()

    def _fetchdf(self, query: str):
        with self._lock:
            return self._connection.execute(query).fetchdf()

    def _fetchone(self, query: str):
        with self._lock:
            return self._connection.execute(query).fetchone()

    def _execute(self, query: str) -> None:
        with self._lock:
            self._connection.execute(query)

    def _ensure_tables(self) -> None:
        parquet_sources = {
            "dashboard_metrics": self.dashboard_dir / "dashboard_metrics.parquet",
            "dashboard_context": self.dashboard_dir / "dashboard_context.parquet",
            "municipality_profiles": self.dashboard_dir / "municipality_profiles.parquet",
            "dashboard_geographies": self.dashboard_dir / "dashboard_geographies.parquet",
        }
        for table_name, parquet_path in parquet_sources.items():
            self._execute(
                f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet({_sql_quote(str(parquet_path))})"
            )

    def close(self) -> None:
        self._connection.close()

    def _granularity_predicate(self, granularity: str, alias: str = "m") -> str:
        if granularity == "month":
            return f"{alias}.month IS NOT NULL"
        if granularity == "season":
            return f"{alias}.month IS NULL AND {alias}.season IS NOT NULL"
        if granularity == "year":
            return f"{alias}.month IS NULL AND {alias}.season IS NULL"
        raise ValueError("Weekly granularity is not materialized in the internal beta build.")

    def _base_where(self, filters: DashboardFilters, metric: MetricDefinition, granularity: str) -> str:
        parts = [f"m.metric_id = {_sql_quote(metric.metric_id)}", self._granularity_predicate(granularity)]
        years = getattr(filters, "years", []) or self.manifest["years_covered"]
        parts.append(f"m.year IN ({_sql_list(years)})")
        definition_type = getattr(filters, "definition_type", None)
        if definition_type:
            parts.append(f"m.definition_type = {_sql_quote(definition_type)}")
        geo_codes = getattr(filters, "geo_codes", [])
        if geo_codes:
            uf_codes = [code for code in geo_codes if len(code) == 2]
            municipality_codes = [code for code in geo_codes if len(code) != 2]
            geo_parts = []
            if uf_codes:
                geo_parts.append(f"m.uf IN ({_sql_list(uf_codes)})")
            if municipality_codes:
                geo_parts.append(f"m.municipality_code IN ({_sql_list(municipality_codes)})")
            if geo_parts:
                parts.append("(" + " OR ".join(geo_parts) + ")")
        context_variable = getattr(filters, "context_variable", None)
        context_bins = getattr(filters, "context_bins", [])
        if context_variable and context_bins:
            parts.append(f"c.{context_variable} IN ({_sql_list(context_bins)})")
        return " AND ".join(parts)

    def _value_expression(self, metric: MetricDefinition, value_mode: str, numerator_alias: str = "numerator_sum", denominator_alias: str = "denominator_sum") -> str:
        if value_mode == "rate":
            if not metric.denominator_field or metric.rate_multiplier is None:
                return "NULL"
            return (
                f"CASE WHEN {denominator_alias} >= {metric.minimum_denominator_threshold} "
                f"THEN ({metric.rate_multiplier} * {numerator_alias} / NULLIF({denominator_alias}, 0)) ELSE NULL END"
            )
        if metric.aggregation_method == "weighted_mean":
            return f"CASE WHEN {denominator_alias} >= {metric.minimum_denominator_threshold} THEN ({numerator_alias} / NULLIF({denominator_alias}, 0)) ELSE NULL END"
        return numerator_alias

    def _resolve_value_mode(self, metric: MetricDefinition, requested_mode: str | None) -> str:
        if requested_mode and requested_mode in metric.available_value_modes:
            return requested_mode
        return metric.default_value_mode

    def _metric_aggregate(self, filters: DashboardFilters, metric: MetricDefinition, granularity: str, group_clause: str, label_expr: str, code_expr: str, order_clause: str) -> list[dict[str, Any]]:
        where_clause = self._base_where(filters, metric, granularity)
        value_mode = filters.value_mode if filters.value_mode in metric.available_value_modes else metric.default_value_mode
        value_expr = self._value_expression(metric, value_mode)
        query = f"""
            WITH base AS (
                SELECT
                    {code_expr} AS geo_code,
                    {label_expr} AS label,
                    SUM(m.value_count) AS numerator_sum,
                    SUM(COALESCE(m.denominator_value, 0)) AS denominator_sum,
                    MAX(m.quality_flag_any) AS quality_flag_any
                FROM dashboard_metrics m
                LEFT JOIN dashboard_context c
                  ON m.municipality_code = c.municipality_code AND m.context_join_year = c.year
                LEFT JOIN dashboard_geographies g
                  ON m.municipality_code = g.municipality_code
                WHERE {where_clause}
                GROUP BY {group_clause}
            )
            SELECT
                geo_code,
                label,
                {value_expr} AS value,
                denominator_sum AS denominator,
                CAST(quality_flag_any AS INTEGER) AS quality_flag_any
            FROM base
            {order_clause}
        """
        result = self._fetchdf(query)
        return result.to_dict(orient="records")

    def _card_bucket_parts(self, granularity: str) -> tuple[str, str, str, str]:
        if granularity == "month":
            return (
                "strftime('%Y-%m', CAST(m.date_bucket AS DATE))",
                "m.year AS sort_year, m.month AS sort_part",
                "1, 2, 3",
                "sort_year DESC, sort_part DESC",
            )
        if granularity == "season":
            season_order = "CASE m.season WHEN 'DJF' THEN 1 WHEN 'MAM' THEN 2 WHEN 'JJA' THEN 3 ELSE 4 END"
            return (
                "CAST(m.year AS VARCHAR) || '-' || m.season",
                f"m.year AS sort_year, {season_order} AS sort_part",
                "1, 2, 3",
                "sort_year DESC, sort_part DESC",
            )
        return (
            "CAST(m.year AS VARCHAR)",
            "m.year AS sort_year, 0 AS sort_part",
            "1, 2, 3",
            "sort_year DESC, sort_part DESC",
        )

    def _combined_period_label(self, years: list[int]) -> str:
        if not years:
            return "Combined selected years"
        ordered_years = sorted(years)
        if len(ordered_years) == 1:
            return str(ordered_years[0])
        first_year = ordered_years[0]
        last_year = ordered_years[-1]
        contiguous = ordered_years == list(range(first_year, last_year + 1))
        if contiguous:
            return f"Combined {first_year}-{str(last_year)[-2:]}"
        return "Combined selected years"

    def get_manifest(self) -> dict[str, Any]:
        return self.manifest

    def get_metrics(self) -> dict[str, Any]:
        return {
            "pages": [page.to_dict() for page in self.pages],
            "metrics": [metric.to_dict() for metric in self.metrics],
            "context_variables": [variable.to_dict() for variable in self.context_variables],
        }

    def get_filters(self) -> dict[str, Any]:
        states = self._fetchdf(
            """
            SELECT DISTINCT uf AS geo_code, state_name AS label
            FROM dashboard_geographies
            ORDER BY label
            """
        )
        municipalities = self._fetchdf(
            """
            SELECT municipality_code AS geo_code, municipality_name AS label, uf
            FROM dashboard_geographies
            ORDER BY uf, label
            """
        )
        return {
            "years": self.manifest["years_covered"],
            "states": states.to_dict(orient="records"),
            "municipalities": municipalities.to_dict(orient="records"),
            "definitions": sorted({metric.definition_type for metric in self.metrics}),
            "granularities": self.manifest["granularities"],
            "geo_levels": self.manifest["geo_levels"],
            "context_variables": [variable.to_dict() for variable in self.context_variables],
        }

    def _pages_using_metric(self, metric_id: str) -> list[str]:
        return [page.page_id for page in self.pages if metric_id in page.metric_ids]

    def _field_label(self, field_name: str | None) -> str:
        if not field_name:
            return "Not applicable"
        return FIELD_LABELS.get(field_name, field_name.replace("_", " ").title())

    def _source_system_ids_for_metric(self, metric: MetricDefinition) -> list[str]:
        technical_sheet = metric.technical_sheet or {}
        if technical_sheet.get("source_system_ids"):
            return [str(item) for item in technical_sheet["source_system_ids"]]

        if metric.metric_family == "climate":
            return ["era5_land"]
        if "stillbirth" in metric.metric_id:
            return ["stillbirth_registry"]
        if metric.metric_family == "perinatal":
            return ["sinasc"]
        if metric.metric_family == "maternal":
            return ["sih_sus", "ibge_population"]
        if metric.metric_family == "mortality":
            lineage = self.gold_dataset_lineage_map.get(metric.source_table)
            if lineage:
                return list(lineage.source_system_ids)
            return ["sim", "ibge_population"]
        lineage = self.gold_dataset_lineage_map.get(metric.source_table)
        if lineage:
            return list(lineage.source_system_ids)
        return []

    def _source_system_ids_for_context(self, context_variable: str) -> list[str]:
        variable = self.context_variable_map[context_variable]
        technical_sheet = variable.technical_sheet or {}
        return [str(item) for item in technical_sheet.get("source_system_ids", [])]

    def _gold_dataset_id_for_metric(self, metric: MetricDefinition) -> str:
        technical_sheet = metric.technical_sheet or {}
        if technical_sheet.get("gold_dataset_id"):
            return str(technical_sheet["gold_dataset_id"])
        return metric.source_table

    def _gold_lineage_for_metric(self, metric: MetricDefinition) -> GoldDatasetLineageDefinition | None:
        return self.gold_dataset_lineage_map.get(self._gold_dataset_id_for_metric(metric))

    def _bronze_families_for_metric(self, metric: MetricDefinition, source_system_ids: list[str]) -> list[str]:
        technical_sheet = metric.technical_sheet or {}
        if technical_sheet.get("bronze_family_ids"):
            return [str(item) for item in technical_sheet["bronze_family_ids"]]

        bronze_families: list[str] = []
        for source_system_id in source_system_ids:
            for bronze_family in SOURCE_SYSTEM_BRONZE_MAP.get(source_system_id, []):
                if bronze_family not in bronze_families:
                    bronze_families.append(bronze_family)
        return bronze_families

    def _silver_entities_for_metric(self, metric: MetricDefinition, source_system_ids: list[str]) -> list[str]:
        technical_sheet = metric.technical_sheet or {}
        if technical_sheet.get("silver_entity_ids"):
            return [str(item) for item in technical_sheet["silver_entity_ids"]]

        silver_entities: list[str] = []
        for source_system_id in source_system_ids:
            for silver_entity in SOURCE_SYSTEM_SILVER_MAP.get(source_system_id, []):
                if silver_entity not in silver_entities:
                    silver_entities.append(silver_entity)

        if metric.denominator_field == "POP_U5" and "silver_pop_u5" not in silver_entities:
            silver_entities.append("silver_pop_u5")
        if metric.denominator_field == "POP_WRA_15_49" and "silver_pop_wra_15_49" not in silver_entities:
            silver_entities.append("silver_pop_wra_15_49")
        if metric.denominator_field == "POP_WRA_15_45_SENSITIVITY" and "silver_pop_wra_15_45_sensitivity" not in silver_entities:
            silver_entities.append("silver_pop_wra_15_45_sensitivity")
        if metric.metric_id == "gest_age_canonical" and "silver_sinasc_event" not in silver_entities:
            silver_entities.append("silver_sinasc_event")
        return silver_entities

    def _raw_sources_for_metric(self, metric: MetricDefinition, source_system_ids: list[str]) -> list[str]:
        technical_sheet = metric.technical_sheet or {}
        if technical_sheet.get("raw_source_ids"):
            return [str(item) for item in technical_sheet["raw_source_ids"]]
        return source_system_ids

    def _aggregation_logic_for_metric(self, metric: MetricDefinition) -> str:
        if metric.aggregation_method == "weighted_mean":
            if metric.aggregation_weight_field:
                return (
                    f"The dashboard computes a weighted mean of {self._field_label(metric.numerator_field).lower()} "
                    f"using {self._field_label(metric.aggregation_weight_field).lower()} as the aggregation weight "
                    "within the selected geography and time bucket."
                )
            return (
                f"The dashboard computes the arithmetic mean of {self._field_label(metric.numerator_field).lower()} "
                "across the municipality-day observations contributing to the selected time bucket and geography."
            )
        return (
            f"The dashboard sums {self._field_label(metric.numerator_field).lower()} within the selected geography "
            "and time bucket before any optional rate calculation."
        )

    def _numerator_sheet_for_metric(self, metric: MetricDefinition) -> dict[str, Any]:
        technical_sheet = metric.technical_sheet or {}
        numerator = technical_sheet.get("numerator", {})
        if numerator:
            return {
                "label": str(numerator.get("label", self._field_label(metric.numerator_field))),
                "fields": [str(item) for item in numerator.get("fields", [metric.numerator_field])],
                "logic": str(numerator.get("logic", self._aggregation_logic_for_metric(metric))),
            }
        return {
            "label": self._field_label(metric.numerator_field),
            "fields": [metric.numerator_field],
            "logic": self._aggregation_logic_for_metric(metric),
        }

    def _denominator_sheet_for_metric(self, metric: MetricDefinition) -> dict[str, Any]:
        technical_sheet = metric.technical_sheet or {}
        denominator = technical_sheet.get("denominator", {})
        denominator_fields: list[str] = []
        if denominator.get("fields"):
            denominator_fields = [str(item) for item in denominator.get("fields", [])]
        elif metric.denominator_field:
            denominator_fields = [metric.denominator_field]
        elif metric.aggregation_weight_field:
            denominator_fields = [metric.aggregation_weight_field]

        if denominator:
            return {
                "label": str(denominator.get("label", self._field_label(denominator_fields[0] if denominator_fields else None))),
                "fields": denominator_fields,
                "logic": str(denominator.get("logic", "Dashboard denominator logic is documented in the indicator registry.")),
                "minimum_threshold": int(denominator.get("minimum_threshold", metric.minimum_denominator_threshold)),
                "rate_formula": denominator.get("rate_formula"),
            }

        if metric.denominator_field:
            rate_formula = None
            if metric.rate_multiplier is not None:
                rate_formula = f"({self._field_label(metric.numerator_field)} / {self._field_label(metric.denominator_field)}) * {metric.rate_multiplier:g}"
            return {
                "label": self._field_label(metric.denominator_field),
                "fields": [metric.denominator_field],
                "logic": (
                    f"The denominator uses {self._field_label(metric.denominator_field).lower()} aggregated over the same "
                    "selected geography and time bucket. Rate values are suppressed when the denominator falls below the "
                    f"minimum threshold of {metric.minimum_denominator_threshold}."
                ),
                "minimum_threshold": metric.minimum_denominator_threshold,
                "rate_formula": rate_formula,
            }

        if metric.aggregation_method == "weighted_mean" and metric.aggregation_weight_field:
            return {
                "label": self._field_label(metric.aggregation_weight_field),
                "fields": [metric.aggregation_weight_field],
                "logic": (
                    f"The weighted mean uses {self._field_label(metric.aggregation_weight_field).lower()} as the analytic "
                    "weight across the selected geography and time bucket."
                ),
                "minimum_threshold": metric.minimum_denominator_threshold,
                "rate_formula": None,
            }

        if metric.aggregation_method == "weighted_mean":
            return {
                "label": "Available municipality-day observations",
                "fields": [],
                "logic": (
                    "The mean is computed across the available municipality-day observations contributing to the selected "
                    "aggregation. No explicit population denominator is applied."
                ),
                "minimum_threshold": metric.minimum_denominator_threshold,
                "rate_formula": None,
            }

        return {
            "label": "Not applicable",
            "fields": [],
            "logic": "This indicator is displayed as a direct count or sum and does not use a separate denominator in the dashboard.",
            "minimum_threshold": metric.minimum_denominator_threshold,
            "rate_formula": None,
        }

    def _component_variables_for_metric(self, metric: MetricDefinition) -> list[dict[str, Any]]:
        technical_sheet = metric.technical_sheet or {}
        if technical_sheet.get("component_variables"):
            return [
                {
                    "field": str(item["field"]),
                    "label": str(item["label"]),
                    "role": str(item["role"]),
                    "required_for": [str(value) for value in item.get("required_for", [])],
                }
                for item in technical_sheet.get("component_variables", [])
            ]

        component_variables: list[dict[str, Any]] = [
            {
                "field": metric.numerator_field,
                "label": self._field_label(metric.numerator_field),
                "role": "numerator",
                "required_for": ["count", "display"] if metric.default_value_mode == "count" else ["rate", "display"],
            }
        ]
        if metric.denominator_field:
            component_variables.append(
                {
                    "field": metric.denominator_field,
                    "label": self._field_label(metric.denominator_field),
                    "role": "denominator",
                    "required_for": ["rate"],
                }
            )
        if metric.aggregation_weight_field:
            component_variables.append(
                {
                    "field": metric.aggregation_weight_field,
                    "label": self._field_label(metric.aggregation_weight_field),
                    "role": "rate_rule",
                    "required_for": ["display"],
                }
            )
        for flag in metric.quality_flags:
            component_variables.append(
                {
                    "field": flag,
                    "label": flag.replace("_", " ").title(),
                    "role": "quality_rule",
                    "required_for": ["quality"],
                }
            )
        return component_variables

    def _display_rules_for_metric(self, metric: MetricDefinition) -> list[str]:
        rules = ["Results are descriptive and should not be interpreted as causal estimates."]
        if "rate" in metric.available_value_modes:
            rules.insert(
                0,
                f"Rate displays are suppressed when the aggregated denominator is below {metric.minimum_denominator_threshold}.",
            )
            rules.insert(
                1,
                "Visible rate values are clipped at the upper 99th percentile of the current view; exports retain the underlying values.",
            )
        else:
            rules.insert(0, "This metric is displayed as a direct count, sum, or mean rather than as a rate.")
        if metric.quality_flags:
            rules.append(f"Quality flags retained in the gold layer: {', '.join(metric.quality_flags)}.")
        return rules

    def _caveats_for_metric(self, metric: MetricDefinition) -> list[str]:
        caveats: list[str] = []
        technical_sheet = metric.technical_sheet or {}
        if technical_sheet.get("caveats"):
            caveats.extend(str(item) for item in technical_sheet.get("caveats", []))
        if metric.definition_type == "sensitivity":
            caveats.append("This indicator uses a sensitivity definition and should be interpreted alongside the corresponding primary or canonical specification.")
        if metric.metric_family == "climate":
            caveats.append("Climate indicators describe exposure conditions and do not represent direct health outcomes.")
        if metric.aggregation_method == "weighted_mean":
            caveats.append("The displayed estimate is an aggregated mean and may mask within-month or within-season variability.")
        caveats.append("Dashboard estimates are intended for descriptive surveillance and pattern detection.")
        return list(dict.fromkeys(caveats))

    def _indicator_public_definition(self, metric: MetricDefinition) -> str:
        technical_sheet = metric.technical_sheet or {}
        if technical_sheet.get("public_definition"):
            return str(technical_sheet["public_definition"])
        note = metric.tooltip_note.strip()
        if note.endswith("."):
            return note
        if note:
            return f"{note}."
        return f"{metric.label} derived from {metric.source_table}."

    def _build_indicator_sheet(self, metric: MetricDefinition) -> dict[str, Any]:
        source_system_ids = self._source_system_ids_for_metric(metric)
        page_usage = self._pages_using_metric(metric.metric_id)
        gold_dataset_id = self._gold_dataset_id_for_metric(metric)
        lineage = self._gold_lineage_for_metric(metric)
        bronze_family_ids = self._bronze_families_for_metric(metric, source_system_ids)
        silver_entity_ids = self._silver_entities_for_metric(metric, source_system_ids)
        raw_source_ids = self._raw_sources_for_metric(metric, source_system_ids)
        technical_sheet = metric.technical_sheet or {}
        used_as_kpi = bool(technical_sheet.get("used_as_kpi", metric.page_id != "methods"))

        return {
            "sheet_id": metric.metric_id,
            "sheet_type": "indicator",
            "label": metric.label,
            "short_label": metric.short_label,
            "metric_id": metric.metric_id,
            "metric_family": metric.metric_family,
            "page_id": metric.page_id,
            "page_usage": page_usage,
            "relationship_compatible": metric.metric_id in {item.metric_id for item in self.metrics if item.page_id != "methods"},
            "used_as_kpi": used_as_kpi,
            "definition_type": metric.definition_type,
            "display_unit": metric.display_unit,
            "rate_unit": metric.rate_display_unit,
            "public_definition": self._indicator_public_definition(metric),
            "construction_logic": self._aggregation_logic_for_metric(metric),
            "aggregation_method": metric.aggregation_method,
            "allowed_geo_levels": list(metric.allowed_geo_levels),
            "allowed_granularities": list(metric.allowed_granularities),
            "available_value_modes": list(metric.available_value_modes),
            "numerator": self._numerator_sheet_for_metric(metric),
            "denominator": self._denominator_sheet_for_metric(metric),
            "component_variables": self._component_variables_for_metric(metric),
            "source_system_ids": source_system_ids,
            "gold_dataset_id": gold_dataset_id,
            "silver_entity_ids": list(dict.fromkeys(technical_sheet.get("silver_entity_ids", silver_entity_ids))),
            "bronze_family_ids": list(dict.fromkeys(technical_sheet.get("bronze_family_ids", bronze_family_ids))),
            "raw_source_ids": list(dict.fromkeys(technical_sheet.get("raw_source_ids", raw_source_ids))),
            "dashboard_artifact_id": str(
                technical_sheet.get("dashboard_artifact_id", lineage.dashboard_artifact_id if lineage else "dashboard_metrics")
            ),
            "source_table": metric.source_table,
            "numerator_field": metric.numerator_field,
            "denominator_field": metric.denominator_field,
            "quality_flags": list(metric.quality_flags),
            "display_rules": self._display_rules_for_metric(metric),
            "caveats": self._caveats_for_metric(metric),
        }

    def _build_stratifier_sheet(self, context_variable: str) -> dict[str, Any]:
        variable = self.context_variable_map[context_variable]
        technical_sheet = variable.technical_sheet or {}
        lineage = self.gold_dataset_lineage_map.get("gold_municipality_context")
        source_system_ids = self._source_system_ids_for_context(context_variable)
        bronze_family_ids = [item for source_system_id in source_system_ids for item in SOURCE_SYSTEM_BRONZE_MAP.get(source_system_id, [])]
        silver_entity_ids = [item for source_system_id in source_system_ids for item in SOURCE_SYSTEM_SILVER_MAP.get(source_system_id, [])]
        if "silver_geography_reference" not in silver_entity_ids and "ibge_geography" in source_system_ids:
            silver_entity_ids.append("silver_geography_reference")
        return {
            "sheet_id": variable.context_variable,
            "sheet_type": "stratifier",
            "label": variable.label,
            "context_variable": variable.context_variable,
            "source_field": variable.source_field,
            "stratifier_type": "contextual stratifier",
            "applicable_pages": [str(item) for item in technical_sheet.get("used_on_pages", [])],
            "public_definition": str(technical_sheet.get("public_definition", variable.label)),
            "source_system_ids": source_system_ids,
            "observed_source_years": [str(item) for item in technical_sheet.get("observed_source_years", [])],
            "expanded_years": [int(item) for item in technical_sheet.get("expanded_years", self.manifest["years_covered"])],
            "expansion_method": str(technical_sheet.get("expansion_method", "Documented contextual expansion across dashboard years.")),
            "join_key": str(technical_sheet.get("join_key", "municipality_code + year")),
            "binning": variable.binning,
            "order": list(variable.order),
            "thresholds": list(variable.thresholds),
            "threshold_labels": [str(item) for item in technical_sheet.get("threshold_labels", list(variable.order))],
            "component_variables": [
                {
                    "field": str(item["field"]),
                    "label": str(item["label"]),
                    "role": str(item["role"]),
                    "required_for": [],
                }
                for item in technical_sheet.get("component_variables", [])
            ],
            "binning_logic": str(technical_sheet.get("binning_logic", "Context bins are defined in the dashboard registry.")),
            "gold_dataset_id": "gold_municipality_context",
            "silver_entity_ids": list(dict.fromkeys(technical_sheet.get("silver_entity_ids", silver_entity_ids if silver_entity_ids else list(lineage.silver_entity_ids if lineage else [])))),
            "bronze_family_ids": list(dict.fromkeys(technical_sheet.get("bronze_family_ids", bronze_family_ids if bronze_family_ids else list(lineage.bronze_family_ids if lineage else [])))),
            "raw_source_ids": [str(item) for item in technical_sheet.get("raw_source_ids", source_system_ids)],
            "dashboard_artifact_id": str(technical_sheet.get("dashboard_artifact_id", lineage.dashboard_artifact_id if lineage else "dashboard_context")),
            "caveats": [str(item) for item in technical_sheet.get("caveats", [])],
        }

    def get_methods_payload(self) -> dict[str, Any]:
        indicator_sheets = [self._build_indicator_sheet(metric) for metric in sorted(self.metrics, key=lambda item: item.sort_order)]
        stratifier_sheets = [self._build_stratifier_sheet(variable.context_variable) for variable in self.context_variables]
        display_rules = [
            {
                "rule_id": "denominator_suppression",
                "title": "Denominator suppression",
                "description": "Rate values are not displayed when the aggregated denominator falls below the minimum threshold defined for the selected metric.",
            },
            {
                "rule_id": "visual_clipping",
                "title": "Rate clipping",
                "description": "Visible rate values in charts, maps, rankings, stratification cards, and relationship plots are clipped at the upper 99th percentile of the current view.",
            },
            {
                "rule_id": "quality_flags",
                "title": "Quality flags",
                "description": "Gold-layer quality flags are carried into the serving layer and surfaced in technical sheets and query outputs.",
            },
            {
                "rule_id": "descriptive_interpretation",
                "title": "Descriptive interpretation only",
                "description": "Dashboard outputs are descriptive surveillance products and must not be interpreted as causal estimates.",
            },
        ]
        return {
            "generated_at": self.manifest["build_generated_at"],
            "technical_overview": {
                "metric_count": len(indicator_sheets),
                "stratifier_count": len(stratifier_sheets),
                "source_system_count": len(self.source_systems),
                "years_covered": list(self.manifest["years_covered"]),
            },
            "medallion_strategy": {"layers": [layer.to_dict() for layer in self.medallion_layers]},
            "source_systems": [system.to_dict() for system in self.source_systems],
            "display_rules": display_rules,
            "indicator_sheets": indicator_sheets,
            "stratifier_sheets": stratifier_sheets,
        }

    def query_cards(self, filters: DashboardFilters) -> list[dict[str, Any]]:
        if not filters.page_id and not filters.metric_id:
            raise ValueError("page_id or metric_id is required for cards.")
        if filters.metric_id:
            metric_ids = [filters.metric_id]
        else:
            page = self.page_map[filters.page_id or ""]
            metric_ids = list(page.metric_ids)
        cards: list[dict[str, Any]] = []
        for metric_id in metric_ids:
            metric = self.metric_map[metric_id]
            current_filters = filters.model_copy(update={"value_mode": filters.value_mode, "granularity": "year"})
            value_mode = current_filters.value_mode if current_filters.value_mode in metric.available_value_modes else metric.default_value_mode
            value_expr = self._value_expression(metric, value_mode)
            selected_years = current_filters.years or self.manifest["years_covered"]
            where_clause = self._base_where(current_filters, metric, "year")
            query = f"""
                WITH base AS (
                    SELECT
                        SUM(m.value_count) AS numerator_sum,
                        SUM(COALESCE(m.denominator_value, 0)) AS denominator_sum,
                        MAX(m.quality_flag_any) AS quality_flag_any
                    FROM dashboard_metrics m
                    LEFT JOIN dashboard_context c
                      ON m.municipality_code = c.municipality_code AND m.context_join_year = c.year
                    WHERE {where_clause}
                )
                SELECT
                    {value_expr} AS value,
                    CAST(quality_flag_any AS INTEGER) AS quality_flag_any
                FROM base
            """
            current = self._fetchone(query)
            current_value = current[0] if current else None
            quality_flag_any = int(current[1]) if current else 0
            period_label = self._combined_period_label(selected_years)

            width = len(selected_years)
            first_year = min(selected_years)
            prior_years = [year for year in range(first_year - width, first_year)]
            prior_where_clause = self._base_where(
                current_filters.model_copy(update={"years": prior_years}),
                metric,
                "year",
            )
            prior_query = f"""
                WITH base AS (
                    SELECT
                        SUM(m.value_count) AS numerator_sum,
                        SUM(COALESCE(m.denominator_value, 0)) AS denominator_sum
                    FROM dashboard_metrics m
                    LEFT JOIN dashboard_context c
                      ON m.municipality_code = c.municipality_code AND m.context_join_year = c.year
                    WHERE {prior_where_clause}
                )
                SELECT {value_expr} AS value FROM base
            """
            prior_row = self._fetchone(prior_query)
            prior_value = prior_row[0] if prior_row else None
            delta = None
            if current_value is not None and prior_value not in {None, 0}:
                delta = float(current_value - prior_value)
            cards.append(
                {
                    "metric_id": metric.metric_id,
                    "label": metric.label,
                    "value": current_value,
                    "unit": metric.rate_display_unit if value_mode == "rate" and metric.rate_display_unit else metric.display_unit,
                    "period_label": period_label,
                    "delta_vs_prior_period": delta,
                    "quality_note": "Contains one or more quality flags." if quality_flag_any else None,
                }
            )
        cards.sort(key=lambda item: self.metric_map[item["metric_id"]].sort_order)
        return cards

    def query_series(self, filters: DashboardFilters) -> list[dict[str, Any]]:
        if not filters.metric_id:
            raise ValueError("metric_id is required for series.")
        metric = self.metric_map[filters.metric_id]
        if filters.granularity not in metric.allowed_granularities:
            raise ValueError(f"Granularity {filters.granularity} is not available for {metric.metric_id}.")
        where_clause = self._base_where(filters, metric, filters.granularity)
        value_mode = filters.value_mode if filters.value_mode in metric.available_value_modes else metric.default_value_mode
        value_expr = self._value_expression(metric, value_mode)
        if filters.granularity == "month":
            x_expr = "m.date_bucket"
            order_expr = "m.year, m.month"
        elif filters.granularity == "season":
            x_expr = "CAST(m.year AS VARCHAR) || '-' || m.season"
            order_expr = "m.year, CASE m.season WHEN 'DJF' THEN 1 WHEN 'MAM' THEN 2 WHEN 'JJA' THEN 3 ELSE 4 END"
        else:
            x_expr = "CAST(m.year AS VARCHAR)"
            order_expr = "m.year"
        query = f"""
            WITH base AS (
                SELECT
                    {x_expr} AS x,
                    SUM(m.value_count) AS numerator_sum,
                    SUM(COALESCE(m.denominator_value, 0)) AS denominator_sum,
                    MAX(m.quality_flag_any) AS quality_flag_any
                FROM dashboard_metrics m
                LEFT JOIN dashboard_context c
                  ON m.municipality_code = c.municipality_code AND m.context_join_year = c.year
                WHERE {where_clause}
                GROUP BY x, {order_expr}
                ORDER BY {order_expr}
            )
            SELECT
                x,
                {value_expr} AS y,
                denominator_sum AS denominator,
                CAST(quality_flag_any AS INTEGER) AS quality_flag_any
            FROM base
            ORDER BY x
        """
        result = self._fetchdf(query)
        rows = []
        for record in result.to_dict(orient="records"):
            rows.append(
                {
                    "x": record["x"],
                    "y": record["y"],
                    "denominator": record["denominator"],
                    "quality_flag_any": bool(record["quality_flag_any"]),
                    "tooltip_meta": {
                        "metric_id": metric.metric_id,
                        "value_mode": value_mode,
                        "unit": metric.rate_display_unit if value_mode == "rate" and metric.rate_display_unit else metric.display_unit,
                    },
                }
            )
        return rows

    def query_map(self, filters: DashboardFilters) -> list[dict[str, Any]]:
        if not filters.metric_id:
            raise ValueError("metric_id is required for map.")
        metric = self.metric_map[filters.metric_id]
        output_level = filters.geo_level
        if output_level == "state":
            group_clause = "m.uf, g.state_name"
            code_expr = "m.uf"
            label_expr = "g.state_name"
        elif output_level == "municipality":
            group_clause = "m.municipality_code, g.municipality_name"
            code_expr = "m.municipality_code"
            label_expr = "g.municipality_name"
        else:
            group_clause = "'BR', 'Brazil'"
            code_expr = "'BR'"
            label_expr = "'Brazil'"
        granularity = filters.granularity if filters.granularity in {"season", "year"} else "year"
        return self._metric_aggregate(filters, metric, granularity, group_clause, label_expr, code_expr, "ORDER BY value DESC NULLS LAST")

    def query_ranking(self, filters: DashboardFilters) -> list[dict[str, Any]]:
        rows = self.query_map(filters)
        sorted_rows = sorted(
            rows,
            key=lambda item: (item["value"] is None, -(item["value"] if item["value"] is not None else float("-inf"))),
        )
        ranking = []
        for index, row in enumerate(sorted_rows, start=1):
            ranking.append({**row, "rank": index})
        return ranking

    def query_stratification(self, filters: DashboardFilters) -> list[dict[str, Any]]:
        if not filters.metric_id:
            raise ValueError("metric_id is required for stratification.")
        if not filters.context_variable:
            raise ValueError("context_variable is required for stratification.")
        metric = self.metric_map[filters.metric_id]
        context_column = filters.context_variable
        where_clause = self._base_where(filters, metric, "year")
        value_mode = filters.value_mode if filters.value_mode in metric.available_value_modes else metric.default_value_mode
        value_expr = self._value_expression(metric, value_mode)
        query = f"""
            WITH base AS (
                SELECT
                    c.{context_column} AS stratum,
                    SUM(m.value_count) AS numerator_sum,
                    SUM(COALESCE(m.denominator_value, 0)) AS denominator_sum,
                    COUNT(DISTINCT m.municipality_code) AS n_geographies
                FROM dashboard_metrics m
                LEFT JOIN dashboard_context c
                  ON m.municipality_code = c.municipality_code AND m.context_join_year = c.year
                WHERE {where_clause}
                GROUP BY c.{context_column}
            )
            SELECT
                COALESCE(stratum, 'missing') AS stratum,
                {value_expr} AS value,
                denominator_sum AS denominator,
                n_geographies
            FROM base
            ORDER BY stratum
        """
        result = self._fetchdf(query)
        return result.to_dict(orient="records")

    def query_relationship(self, filters: RelationshipFilters) -> list[dict[str, Any]]:
        x_metric = self.metric_map[filters.x_metric_id]
        y_metric = self.metric_map[filters.y_metric_id]

        if filters.granularity not in x_metric.allowed_granularities:
            raise ValueError(f"Granularity {filters.granularity} is not available for {x_metric.metric_id}.")
        if filters.granularity not in y_metric.allowed_granularities:
            raise ValueError(f"Granularity {filters.granularity} is not available for {y_metric.metric_id}.")

        x_value_mode = self._resolve_value_mode(x_metric, filters.x_value_mode)
        y_value_mode = self._resolve_value_mode(y_metric, filters.y_value_mode)
        x_value_expr = self._value_expression(x_metric, x_value_mode)
        y_value_expr = self._value_expression(y_metric, y_value_mode)

        if filters.geo_level == "region":
            region_expr = _uf_to_region_case("m")
            geo_code_expr = region_expr
            label_expr = region_expr
        elif filters.geo_level == "state":
            geo_code_expr = "m.uf"
            label_expr = "g.state_name"
        elif filters.geo_level == "municipality":
            geo_code_expr = "m.municipality_code"
            label_expr = "g.municipality_name"
        else:
            geo_code_expr = "'BR'"
            label_expr = "'Brazil'"

        if filters.granularity == "month":
            time_bucket_expr = "m.date_bucket"
        elif filters.granularity == "season":
            time_bucket_expr = "CAST(m.year AS VARCHAR) || '-' || m.season"
        else:
            time_bucket_expr = "CAST(m.year AS VARCHAR)"

        if filters.context_variable:
            context_expr = f"COALESCE(c.{filters.context_variable}, 'missing')"
            context_group_expr = f", {context_expr}"
            context_join_expr = "AND x.context_stratum = y.context_stratum"
            context_projection = f"{context_expr} AS context_stratum,"
        else:
            context_group_expr = ""
            context_join_expr = ""
            context_projection = "NULL AS context_stratum,"

        group_clause = ", ".join([geo_code_expr, label_expr, time_bucket_expr, "m.year"]) + context_group_expr
        x_where_clause = self._base_where(filters, x_metric, filters.granularity)
        y_where_clause = self._base_where(filters, y_metric, filters.granularity)

        query = f"""
            WITH x_base AS (
                SELECT
                    {geo_code_expr} AS geo_code,
                    {label_expr} AS label,
                    {time_bucket_expr} AS time_bucket,
                    m.year AS year,
                    {context_projection}
                    SUM(m.value_count) AS numerator_sum,
                    SUM(COALESCE(m.denominator_value, 0)) AS denominator_sum,
                    MAX(m.quality_flag_any) AS quality_flag_any
                FROM dashboard_metrics m
                LEFT JOIN dashboard_context c
                  ON m.municipality_code = c.municipality_code AND m.context_join_year = c.year
                LEFT JOIN dashboard_geographies g
                  ON m.municipality_code = g.municipality_code
                WHERE {x_where_clause}
                GROUP BY {group_clause}
            ),
            x_values AS (
                SELECT
                    geo_code,
                    label,
                    time_bucket,
                    year,
                    context_stratum,
                    {x_value_expr} AS x_value,
                    CAST(quality_flag_any AS INTEGER) AS quality_flag_any
                FROM x_base
            ),
            y_base AS (
                SELECT
                    {geo_code_expr} AS geo_code,
                    {label_expr} AS label,
                    {time_bucket_expr} AS time_bucket,
                    m.year AS year,
                    {context_projection}
                    SUM(m.value_count) AS numerator_sum,
                    SUM(COALESCE(m.denominator_value, 0)) AS denominator_sum,
                    MAX(m.quality_flag_any) AS quality_flag_any
                FROM dashboard_metrics m
                LEFT JOIN dashboard_context c
                  ON m.municipality_code = c.municipality_code AND m.context_join_year = c.year
                LEFT JOIN dashboard_geographies g
                  ON m.municipality_code = g.municipality_code
                WHERE {y_where_clause}
                GROUP BY {group_clause}
            ),
            y_values AS (
                SELECT
                    geo_code,
                    label,
                    time_bucket,
                    year,
                    context_stratum,
                    {y_value_expr} AS y_value,
                    CAST(quality_flag_any AS INTEGER) AS quality_flag_any
                FROM y_base
            )
            SELECT
                x.geo_code,
                x.label,
                x.context_stratum,
                x.time_bucket,
                x.year,
                x.x_value,
                y.y_value,
                CAST(GREATEST(x.quality_flag_any, y.quality_flag_any) AS INTEGER) AS quality_flag_any
            FROM x_values x
            INNER JOIN y_values y
              ON x.geo_code = y.geo_code
             AND x.time_bucket = y.time_bucket
             AND x.year = y.year
             {context_join_expr}
            WHERE x.x_value IS NOT NULL AND y.y_value IS NOT NULL
            ORDER BY x.x_value, y.y_value
        """

        result = self._fetchdf(query)
        rows = []
        for record in result.to_dict(orient="records"):
            rows.append(
                {
                    "x_value": record["x_value"],
                    "y_value": record["y_value"],
                    "geo_code": record["geo_code"],
                    "label": record["label"],
                    "context_stratum": record["context_stratum"],
                    "time_bucket": record["time_bucket"],
                    "year": int(record["year"]),
                    "quality_flag_any": bool(record["quality_flag_any"]),
                    "tooltip_meta": {
                        "x_metric_id": x_metric.metric_id,
                        "x_metric_label": x_metric.label,
                        "x_unit": x_metric.rate_display_unit if x_value_mode == "rate" and x_metric.rate_display_unit else x_metric.display_unit,
                        "y_metric_id": y_metric.metric_id,
                        "y_metric_label": y_metric.label,
                        "y_unit": y_metric.rate_display_unit if y_value_mode == "rate" and y_metric.rate_display_unit else y_metric.display_unit,
                    },
                }
            )
        return rows

    def municipality_profile(self, municipality_code: str) -> dict[str, Any]:
        geo = self._fetchone(
            f"""
            SELECT municipality_code, municipality_name, uf
            FROM dashboard_geographies
            WHERE municipality_code = {_sql_quote(municipality_code)}
            """
        )
        if geo is None:
            raise KeyError(municipality_code)
        context = self._fetchdf(
            f"""
            SELECT *
            FROM dashboard_context
            WHERE municipality_code = {_sql_quote(municipality_code)}
            ORDER BY year
            """
        )
        profiles = self._fetchdf(
            f"""
            SELECT *
            FROM municipality_profiles
            WHERE municipality_code = {_sql_quote(municipality_code)}
            ORDER BY page_id, metric_id, year
            """
        )
        bundles: dict[str, list[dict[str, Any]]] = {
            "overview": [],
            "perinatal": [],
            "mortality": [],
            "maternal": [],
            "climate": [],
        }
        page_alias = {
            "overview": "overview",
            "perinatal": "perinatal",
            "child_mortality": "mortality",
            "maternal": "maternal",
            "climate": "climate",
            "context": "overview",
        }
        for row in profiles.to_dict(orient="records"):
            bucket = page_alias.get(row["page_id"], "overview")
            bundles[bucket].append(row)
        return {
            "municipality_code": geo[0],
            "label": geo[1],
            "uf": geo[2],
            "context": {
                "years": context.to_dict(orient="records"),
            },
            **bundles,
        }
