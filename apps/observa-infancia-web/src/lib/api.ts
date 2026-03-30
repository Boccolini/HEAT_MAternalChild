import { z } from "zod";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";

export const pageSchema = z.object({
  page_id: z.string(),
  route: z.string(),
  title: z.string(),
  description: z.string(),
  default_metric_id: z.string().nullable(),
  metric_ids: z.array(z.string())
});

export const metricSchema = z.object({
  metric_id: z.string(),
  label: z.string(),
  short_label: z.string(),
  metric_family: z.string(),
  page_id: z.string(),
  source_table: z.string(),
  numerator_field: z.string(),
  denominator_field: z.string().nullable(),
  definition_type: z.string(),
  default_value_mode: z.string(),
  available_value_modes: z.array(z.string()),
  allowed_granularities: z.array(z.string()),
  allowed_geo_levels: z.array(z.string()),
  display_unit: z.string(),
  rate_display_unit: z.string().nullable(),
  minimum_denominator_threshold: z.number(),
  quality_flags: z.array(z.string()),
  tooltip_note: z.string(),
  aggregation_method: z.string(),
  aggregation_weight_field: z.string().nullable(),
  denominator_aggregation: z.string().nullable(),
  rate_multiplier: z.number().nullable(),
  sort_order: z.number()
});

export const contextVariableSchema = z.object({
  context_variable: z.string(),
  label: z.string(),
  source_field: z.string(),
  binning: z.string(),
  missing_label: z.string(),
  order: z.array(z.string()),
  thresholds: z.array(z.number()).optional().default([])
});

export const manifestSchema = z.object({
  dashboard_version: z.string(),
  dashboard_name: z.string(),
  build_generated_at: z.string(),
  years_covered: z.array(z.number()),
  granularities: z.array(z.string()),
  geo_levels: z.array(z.string()),
  pages: z.array(pageSchema),
  metrics: z.array(metricSchema),
  context_variables: z.array(contextVariableSchema),
  context_bins: z.record(z.any()),
  source_tables: z.array(z.string()),
  artifacts: z.record(z.string()),
  coverage: z.object({
    municipalities: z.number(),
    states: z.number(),
    metric_rows: z.number()
  })
});

const filtersSchema = z.object({
  years: z.array(z.number()),
  states: z.array(z.object({ geo_code: z.string(), label: z.string() })),
  municipalities: z.array(z.object({ geo_code: z.string(), label: z.string(), uf: z.string() })),
  definitions: z.array(z.string()),
  granularities: z.array(z.string()),
  geo_levels: z.array(z.string()),
  context_variables: z.array(contextVariableSchema)
});

const metricsPayloadSchema = z.object({
  pages: z.array(pageSchema),
  metrics: z.array(metricSchema),
  context_variables: z.array(contextVariableSchema)
});

const cardSchema = z.object({
  metric_id: z.string(),
  label: z.string(),
  value: z.number().nullable(),
  unit: z.string(),
  period_label: z.string().nullable().optional(),
  delta_vs_prior_period: z.number().nullable(),
  quality_note: z.string().nullable()
});

const seriesSchema = z.object({
  x: z.string(),
  y: z.number().nullable(),
  denominator: z.number().nullable(),
  quality_flag_any: z.boolean(),
  tooltip_meta: z.record(z.any())
});

const mapDatumSchema = z.object({
  geo_code: z.string(),
  label: z.string(),
  value: z.number().nullable(),
  denominator: z.number().nullable(),
  quality_flag_any: z.boolean()
});

const rankingRowSchema = mapDatumSchema.extend({
  rank: z.number()
});

const stratificationRowSchema = z.object({
  stratum: z.string(),
  value: z.number().nullable(),
  denominator: z.number().nullable(),
  n_geographies: z.number()
});

const municipalityProfileSchema = z.object({
  municipality_code: z.string(),
  label: z.string(),
  uf: z.string(),
  context: z.record(z.any()),
  overview: z.array(z.record(z.any())),
  perinatal: z.array(z.record(z.any())),
  mortality: z.array(z.record(z.any())),
  maternal: z.array(z.record(z.any())),
  climate: z.array(z.record(z.any()))
});

const relationshipPointSchema = z.object({
  x_value: z.number(),
  y_value: z.number(),
  geo_code: z.string(),
  label: z.string(),
  context_stratum: z.string().nullable().optional().default(null),
  time_bucket: z.string(),
  year: z.number(),
  quality_flag_any: z.boolean(),
  tooltip_meta: z.record(z.any())
});

const technicalOverviewSchema = z.object({
  metric_count: z.number(),
  stratifier_count: z.number(),
  source_system_count: z.number(),
  years_covered: z.array(z.number())
});

const medallionLayerSchema = z.object({
  layer_id: z.string(),
  label: z.string(),
  description: z.string(),
  artifact_type: z.string()
});

const sourceSystemSchema = z.object({
  source_system_id: z.string(),
  label: z.string(),
  short_description: z.string(),
  domain: z.string(),
  raw_manifest_ids: z.array(z.string()),
  bronze_manifest_ids: z.array(z.string()),
  silver_manifest_ids: z.array(z.string()),
  notes: z.array(z.string())
});

const displayRuleSchema = z.object({
  rule_id: z.string(),
  title: z.string(),
  description: z.string()
});

const technicalComponentVariableSchema = z.object({
  field: z.string(),
  label: z.string(),
  role: z.string(),
  required_for: z.array(z.string()).default([])
});

const technicalNumeratorSchema = z.object({
  label: z.string(),
  fields: z.array(z.string()),
  logic: z.string()
});

const technicalDenominatorSchema = z.object({
  label: z.string(),
  fields: z.array(z.string()),
  logic: z.string(),
  minimum_threshold: z.number().nullable().optional(),
  rate_formula: z.string().nullable().optional()
});

const indicatorTechnicalSheetSchema = z.object({
  sheet_id: z.string(),
  sheet_type: z.literal("indicator"),
  label: z.string(),
  short_label: z.string(),
  metric_id: z.string(),
  metric_family: z.string(),
  page_id: z.string(),
  page_usage: z.array(z.string()),
  relationship_compatible: z.boolean(),
  used_as_kpi: z.boolean(),
  definition_type: z.string(),
  display_unit: z.string(),
  rate_unit: z.string().nullable().optional(),
  public_definition: z.string(),
  construction_logic: z.string(),
  aggregation_method: z.string(),
  allowed_geo_levels: z.array(z.string()),
  allowed_granularities: z.array(z.string()),
  available_value_modes: z.array(z.string()),
  numerator: technicalNumeratorSchema,
  denominator: technicalDenominatorSchema.nullable().optional(),
  component_variables: z.array(technicalComponentVariableSchema),
  source_system_ids: z.array(z.string()),
  gold_dataset_id: z.string(),
  silver_entity_ids: z.array(z.string()),
  bronze_family_ids: z.array(z.string()),
  raw_source_ids: z.array(z.string()),
  dashboard_artifact_id: z.string(),
  source_table: z.string(),
  numerator_field: z.string(),
  denominator_field: z.string().nullable().optional(),
  quality_flags: z.array(z.string()),
  display_rules: z.array(z.string()),
  caveats: z.array(z.string())
});

const stratifierTechnicalSheetSchema = z.object({
  sheet_id: z.string(),
  sheet_type: z.literal("stratifier"),
  label: z.string(),
  context_variable: z.string(),
  source_field: z.string(),
  stratifier_type: z.string(),
  applicable_pages: z.array(z.string()),
  public_definition: z.string(),
  source_system_ids: z.array(z.string()),
  observed_source_years: z.array(z.string()),
  expanded_years: z.array(z.number()),
  expansion_method: z.string(),
  join_key: z.string(),
  binning: z.string(),
  order: z.array(z.string()),
  thresholds: z.array(z.number()),
  threshold_labels: z.array(z.string()),
  component_variables: z.array(technicalComponentVariableSchema),
  binning_logic: z.string(),
  gold_dataset_id: z.string(),
  silver_entity_ids: z.array(z.string()),
  bronze_family_ids: z.array(z.string()),
  raw_source_ids: z.array(z.string()),
  dashboard_artifact_id: z.string(),
  caveats: z.array(z.string())
});

const methodsPayloadSchema = z.object({
  generated_at: z.string(),
  technical_overview: technicalOverviewSchema,
  medallion_strategy: z.object({
    layers: z.array(medallionLayerSchema)
  }),
  source_systems: z.array(sourceSystemSchema),
  display_rules: z.array(displayRuleSchema),
  indicator_sheets: z.array(indicatorTechnicalSheetSchema),
  stratifier_sheets: z.array(stratifierTechnicalSheetSchema)
});

export type PageDefinition = z.infer<typeof pageSchema>;
export type MetricDefinition = z.infer<typeof metricSchema>;
export type ContextVariable = z.infer<typeof contextVariableSchema>;
export type Manifest = z.infer<typeof manifestSchema>;
export type FiltersPayload = z.infer<typeof filtersSchema>;
export type MetricsPayload = z.infer<typeof metricsPayloadSchema>;
export type MetricCard = z.infer<typeof cardSchema>;
export type SeriesPoint = z.infer<typeof seriesSchema>;
export type MapDatum = z.infer<typeof mapDatumSchema>;
export type RankingRow = z.infer<typeof rankingRowSchema>;
export type StratificationRow = z.infer<typeof stratificationRowSchema>;
export type MunicipalityProfile = z.infer<typeof municipalityProfileSchema>;
export type RelationshipPoint = z.infer<typeof relationshipPointSchema>;
export type TechnicalOverview = z.infer<typeof technicalOverviewSchema>;
export type MedallionLayer = z.infer<typeof medallionLayerSchema>;
export type SourceSystem = z.infer<typeof sourceSystemSchema>;
export type DisplayRule = z.infer<typeof displayRuleSchema>;
export type TechnicalComponentVariable = z.infer<typeof technicalComponentVariableSchema>;
export type IndicatorTechnicalSheet = z.infer<typeof indicatorTechnicalSheetSchema>;
export type StratifierTechnicalSheet = z.infer<typeof stratifierTechnicalSheetSchema>;
export type MethodsPayload = z.infer<typeof methodsPayloadSchema>;

export type DashboardFilters = {
  page_id?: string;
  metric_id?: string;
  years: number[];
  granularity: "month" | "season" | "year";
  geo_level: "national" | "state" | "municipality" | "region";
  geo_codes: string[];
  value_mode: "count" | "rate";
  definition_type?: "primary" | "sensitivity" | "canonical";
  context_variable?: string;
  context_bins: string[];
  compare_mode: "none" | "state_vs_national" | "municipality_vs_state_vs_national";
};

export type RelationshipFilters = {
  x_metric_id: string;
  y_metric_id: string;
  years: number[];
  granularity: "month" | "season" | "year";
  geo_level: "national" | "state" | "municipality" | "region";
  geo_codes: string[];
  x_value_mode?: "count" | "rate";
  y_value_mode?: "count" | "rate";
  context_variable?: string;
  context_bins: string[];
};

async function fetchJson<T>(input: RequestInfo, init: RequestInit, schema: z.ZodType<T>): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `Request failed with status ${response.status}`);
  }
  const payload = await response.json();
  return schema.parse(payload);
}

export function getManifest(): Promise<Manifest> {
  return fetchJson(`${API_BASE}/meta/manifest`, {}, manifestSchema);
}

export function getFiltersMeta(): Promise<FiltersPayload> {
  return fetchJson(`${API_BASE}/meta/filters`, {}, filtersSchema);
}

export function getMetricsMeta(): Promise<MetricsPayload> {
  return fetchJson(`${API_BASE}/meta/metrics`, {}, metricsPayloadSchema);
}

export function getMethodsMeta(): Promise<MethodsPayload> {
  return fetchJson(`${API_BASE}/meta/methods`, {}, methodsPayloadSchema);
}

export function postCards(filters: DashboardFilters): Promise<MetricCard[]> {
  return fetchJson(`${API_BASE}/query/cards`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(filters)
  }, z.array(cardSchema));
}

export function postSeries(filters: DashboardFilters): Promise<SeriesPoint[]> {
  return fetchJson(`${API_BASE}/query/series`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(filters)
  }, z.array(seriesSchema));
}

export function postMap(filters: DashboardFilters): Promise<MapDatum[]> {
  return fetchJson(`${API_BASE}/query/map`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(filters)
  }, z.array(mapDatumSchema));
}

export function postRanking(filters: DashboardFilters): Promise<RankingRow[]> {
  return fetchJson(`${API_BASE}/query/ranking`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(filters)
  }, z.array(rankingRowSchema));
}

export function postStratification(filters: DashboardFilters): Promise<StratificationRow[]> {
  return fetchJson(`${API_BASE}/query/stratification`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(filters)
  }, z.array(stratificationRowSchema));
}

export function getMunicipalityProfile(municipalityCode: string): Promise<MunicipalityProfile> {
  return fetchJson(`${API_BASE}/municipalities/${municipalityCode}/profile`, {}, municipalityProfileSchema);
}

export function postRelationship(filters: RelationshipFilters): Promise<RelationshipPoint[]> {
  return fetchJson(`${API_BASE}/query/relationship`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(filters)
  }, z.array(relationshipPointSchema));
}
