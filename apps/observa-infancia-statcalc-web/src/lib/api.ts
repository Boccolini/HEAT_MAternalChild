/// <reference types="vite/client" />

import { z } from "zod";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api/v1/statcalc";

const selectorOptionSchema = z.object({
  id: z.string(),
  label: z.string(),
  description: z.string().nullable().optional(),
  metadata: z.record(z.any()).default({})
});

export type SelectorOption = {
  id: string;
  label: string;
  description?: string | null;
  metadata: Record<string, any>;
};

export type Recipe = {
  recipe_id: string;
  track_id: string;
  label: string;
  description: string;
  outcome_family: string;
  execution_behavior: string;
  default_template_id: string;
  default_offset_id: string;
  templates: SelectorOption[];
  offsets: SelectorOption[];
  outcomes: SelectorOption[];
  exposures: SelectorOption[];
  lag_windows: SelectorOption[];
  stratifiers: string[];
  source_filters: SelectorOption[];
  grouping_templates: SelectorOption[];
  denominator_variants: SelectorOption[];
  gestational_source_modes: SelectorOption[];
};

export type Stratifier = {
  stratifier_id: string;
  label: string;
  description: string;
  stratum_column: string;
  strata_kind: string;
  stratum_labels: string[];
};

export type RunSpec = {
  recipe_id: string;
  outcome_variable: string;
  exposure_variable: string;
  lag_or_window?: string | null;
  template_id: string;
  stratifier_id?: string | null;
  source_filter: string;
  denominator_variant?: string | null;
  grouping_template?: string | null;
  gestational_source_mode?: string | null;
  offset_id?: string | null;
  adjustment_context_variables: string[];
};

export type Run = {
  run_id: string;
  track_id: string;
  recipe_id: string;
  status: string;
  execution_mode: string;
  is_reference: boolean;
  spec_hash: string;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  warnings: string[];
  error_message?: string | null;
  run_dir: string;
  result_available: boolean;
  spec: RunSpec;
};

export type RunResult = {
  run_id: string;
  track_id: string;
  recipe_id: string;
  execution_mode: string;
  summary: Record<string, any>;
  rows: Array<{
    row_id: string;
    label: string;
    group_label?: string | null;
    estimate?: number | null;
    lower_ci?: number | null;
    upper_ci?: number | null;
    p_value?: number | null;
    estimate_scale: string;
    model_label: string;
    valid_strata_count: number;
    observation_count?: number | null;
    outcome_sum?: number | null;
    denominator_sum?: number | null;
    primary_row_flag: boolean;
    metadata: Record<string, any>;
  }>;
  warnings: string[];
  forest_plot_url?: string | null;
  table_csv_url?: string | null;
  spec: RunSpec;
};

export type Preset = {
  preset_id: string;
  name: string;
  description?: string | null;
  created_at: string;
  updated_at: string;
  run_spec: RunSpec;
};

export const recipeSchema = z.object({
  recipe_id: z.string(),
  track_id: z.string(),
  label: z.string(),
  description: z.string(),
  outcome_family: z.string(),
  execution_behavior: z.string(),
  default_template_id: z.string(),
  default_offset_id: z.string(),
  templates: z.array(selectorOptionSchema),
  offsets: z.array(selectorOptionSchema),
  outcomes: z.array(selectorOptionSchema),
  exposures: z.array(selectorOptionSchema),
  lag_windows: z.array(selectorOptionSchema).default([]),
  stratifiers: z.array(z.string()).default([]),
  source_filters: z.array(selectorOptionSchema).default([]),
  grouping_templates: z.array(selectorOptionSchema).default([]),
  denominator_variants: z.array(selectorOptionSchema).default([]),
  gestational_source_modes: z.array(selectorOptionSchema).default([])
});

export const stratifierSchema = z.object({
  stratifier_id: z.string(),
  label: z.string(),
  description: z.string(),
  stratum_column: z.string(),
  strata_kind: z.string(),
  stratum_labels: z.array(z.string())
});

export const runSpecSchema = z.object({
  recipe_id: z.string(),
  outcome_variable: z.string(),
  exposure_variable: z.string(),
  lag_or_window: z.string().nullable().optional(),
  template_id: z.string(),
  stratifier_id: z.string().nullable().optional(),
  source_filter: z.string(),
  denominator_variant: z.string().nullable().optional(),
  grouping_template: z.string().nullable().optional(),
  gestational_source_mode: z.string().nullable().optional(),
  offset_id: z.string().nullable().optional(),
  adjustment_context_variables: z.array(z.string()).default([])
});

const runSchema = z.object({
  run_id: z.string(),
  track_id: z.string(),
  recipe_id: z.string(),
  status: z.string(),
  execution_mode: z.string(),
  is_reference: z.boolean(),
  spec_hash: z.string(),
  created_at: z.string(),
  updated_at: z.string(),
  started_at: z.string().nullable().optional(),
  completed_at: z.string().nullable().optional(),
  warnings: z.array(z.string()).default([]),
  error_message: z.string().nullable().optional(),
  run_dir: z.string(),
  result_available: z.boolean(),
  spec: runSpecSchema
});

const resultRowSchema = z.object({
  row_id: z.string(),
  label: z.string(),
  group_label: z.string().nullable().optional(),
  estimate: z.number().nullable().optional(),
  lower_ci: z.number().nullable().optional(),
  upper_ci: z.number().nullable().optional(),
  p_value: z.number().nullable().optional(),
  estimate_scale: z.string(),
  model_label: z.string(),
  valid_strata_count: z.number(),
  observation_count: z.number().nullable().optional(),
  outcome_sum: z.number().nullable().optional(),
  denominator_sum: z.number().nullable().optional(),
  primary_row_flag: z.boolean(),
  metadata: z.record(z.any()).default({})
});

const runResultSchema = z.object({
  run_id: z.string(),
  track_id: z.string(),
  recipe_id: z.string(),
  execution_mode: z.string(),
  summary: z.record(z.any()),
  rows: z.array(resultRowSchema),
  warnings: z.array(z.string()).default([]),
  forest_plot_url: z.string().nullable().optional(),
  table_csv_url: z.string().nullable().optional(),
  spec: runSpecSchema
});

const presetSchema = z.object({
  preset_id: z.string(),
  name: z.string(),
  description: z.string().nullable().optional(),
  created_at: z.string(),
  updated_at: z.string(),
  run_spec: runSpecSchema
});

async function request<T>(path: string, init?: RequestInit, schema?: z.ZodTypeAny): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(String(payload.detail ?? response.statusText));
  }
  const payload = await response.json();
  return (schema ? schema.parse(payload) : payload) as T;
}

export function getRecipes() {
  return request<Recipe[]>("/meta/recipes", undefined, z.array(recipeSchema));
}

export function getStratifiers() {
  return request<Stratifier[]>("/meta/stratifiers", undefined, z.array(stratifierSchema));
}

export function getPresets() {
  return request<Preset[]>("/meta/presets", undefined, z.array(presetSchema));
}

export function getRuns() {
  return request<Run[]>("/runs", undefined, z.array(runSchema));
}

export function postRun(spec: RunSpec) {
  return request<Run>("/runs", { method: "POST", body: JSON.stringify(spec) }, runSchema);
}

export function getRun(runId: string) {
  return request<Run>(`/runs/${runId}`, undefined, runSchema);
}

export function getRunResult(runId: string) {
  return request<RunResult>(`/runs/${runId}/result`, undefined, runResultSchema);
}

export function createPreset(payload: { name: string; description?: string; run_spec: RunSpec }) {
  return request<Preset>("/presets", { method: "POST", body: JSON.stringify(payload) }, presetSchema);
}

export function updatePreset(presetId: string, payload: { name: string; description?: string; run_spec: RunSpec }) {
  return request<Preset>(`/presets/${presetId}`, { method: "PUT", body: JSON.stringify(payload) }, presetSchema);
}

export async function deletePreset(presetId: string) {
  await request(`/presets/${presetId}`, { method: "DELETE" });
}
