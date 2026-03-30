import { jsx as _jsx } from "react/jsx-runtime";
import { describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { MethodsPage } from "../pages/MethodsPage";
const methodsPayload = {
    generated_at: new Date().toISOString(),
    technical_overview: {
        metric_count: 2,
        stratifier_count: 1,
        source_system_count: 2,
        years_covered: [2020, 2021, 2022, 2023, 2024],
    },
    medallion_strategy: {
        layers: [
            { layer_id: "raw", label: "Raw", description: "Raw layer", artifact_type: "snapshots" },
            { layer_id: "bronze", label: "Bronze", description: "Bronze layer", artifact_type: "promoted files" },
            { layer_id: "silver", label: "Silver", description: "Silver layer", artifact_type: "entities" },
            { layer_id: "gold", label: "Gold", description: "Gold layer", artifact_type: "analytical tables" },
            { layer_id: "dashboard_mart", label: "Dashboard mart", description: "Serving layer", artifact_type: "mart" },
        ],
    },
    source_systems: [
        {
            source_system_id: "sinasc",
            label: "Brazilian Information System on Live Births (SINASC)",
            short_description: "Live birth source",
            domain: "health",
            raw_manifest_ids: ["raw_sinasc.manifest.json"],
            bronze_manifest_ids: ["bronze_sinasc.manifest.json"],
            silver_manifest_ids: ["silver_sinasc.manifest.json"],
            notes: ["Primary perinatal source"],
        },
        {
            source_system_id: "atlas_brasil",
            label: "Atlas Brasil",
            short_description: "Context source",
            domain: "context",
            raw_manifest_ids: [],
            bronze_manifest_ids: [],
            silver_manifest_ids: [],
            notes: ["Context reference"],
        },
    ],
    display_rules: [
        { rule_id: "denominator_suppression", title: "Denominator suppression", description: "Suppresses unstable rates." },
    ],
    indicator_sheets: [
        {
            sheet_id: "preterm_flag",
            sheet_type: "indicator",
            label: "Preterm births",
            short_label: "Preterm",
            metric_id: "preterm_flag",
            metric_family: "perinatal",
            page_id: "perinatal",
            page_usage: ["perinatal", "context"],
            relationship_compatible: true,
            used_as_kpi: true,
            definition_type: "primary",
            display_unit: "births",
            rate_unit: "per 1,000 live births",
            public_definition: "Live births before 37 completed weeks of gestation.",
            construction_logic: "Summed across the selected geography and time bucket.",
            aggregation_method: "sum",
            allowed_geo_levels: ["national", "state", "municipality"],
            allowed_granularities: ["month", "season", "year"],
            available_value_modes: ["count", "rate"],
            numerator: {
                label: "Live births before 37 completed weeks",
                fields: ["PRETERM_FLAG"],
                logic: "Summed numerator.",
            },
            denominator: {
                label: "Live births total",
                fields: ["LIVE_BIRTHS_TOTAL"],
                logic: "Summed denominator.",
                minimum_threshold: 20,
                rate_formula: "(Preterm births / live births total) * 1000",
            },
            component_variables: [
                { field: "PRETERM_FLAG", label: "Live births before 37 completed weeks", role: "numerator", required_for: ["count", "rate"] },
                { field: "LIVE_BIRTHS_TOTAL", label: "Live births total", role: "denominator", required_for: ["rate"] },
            ],
            source_system_ids: ["sinasc"],
            gold_dataset_id: "gold_mnch_perinatal",
            silver_entity_ids: ["silver_sinasc_event"],
            bronze_family_ids: ["bronze_sinasc"],
            raw_source_ids: ["sinasc"],
            dashboard_artifact_id: "dashboard_metrics",
            source_table: "gold_mnch_perinatal",
            numerator_field: "PRETERM_FLAG",
            denominator_field: "LIVE_BIRTHS_TOTAL",
            quality_flags: ["RATE_NOT_COMPUTABLE_FLAG"],
            display_rules: ["Rates are suppressed when denominators are too small."],
            caveats: ["Descriptive interpretation only."],
        },
    ],
    stratifier_sheets: [
        {
            sheet_id: "idh_quintile",
            sheet_type: "stratifier",
            label: "Municipal Human Development Index quintile",
            context_variable: "idh_quintile",
            source_field: "IDH_EXPANDED",
            stratifier_type: "contextual stratifier",
            applicable_pages: ["overview", "relationships"],
            public_definition: "Expanded IDH grouped into quintiles.",
            source_system_ids: ["atlas_brasil"],
            observed_source_years: ["2010"],
            expanded_years: [2020, 2021, 2022, 2023, 2024],
            expansion_method: "Piecewise constant expansion.",
            join_key: "municipality_code + year",
            binning: "quintile",
            order: ["Q1", "Q2", "Q3", "Q4", "Q5", "missing"],
            thresholds: [],
            threshold_labels: ["Lower", "Intermediate-lower", "Average", "Intermediate-upper", "Higher", "Missing context"],
            component_variables: [
                { field: "IDH_EXPANDED", label: "Expanded IDH", role: "expansion", required_for: [] },
            ],
            binning_logic: "Stable national quintiles.",
            gold_dataset_id: "gold_municipality_context",
            silver_entity_ids: ["silver_population"],
            bronze_family_ids: ["bronze_population"],
            raw_source_ids: ["atlas_brasil"],
            dashboard_artifact_id: "dashboard_context",
            caveats: ["Expanded from 2010 reference values."],
        },
    ],
};
vi.stubGlobal("fetch", vi.fn(async (input) => {
    const url = String(input);
    if (url.endsWith("/api/v1/meta/methods")) {
        return new Response(JSON.stringify(methodsPayload));
    }
    if (url.endsWith("/api/v1/meta/manifest")) {
        return new Response(JSON.stringify({
            dashboard_version: "v1",
            dashboard_name: "Observa Infancia",
            build_generated_at: new Date().toISOString(),
            years_covered: [2020, 2021, 2022, 2023, 2024],
            granularities: ["month", "year", "season"],
            geo_levels: ["state", "municipality", "national"],
            pages: [
                { page_id: "overview", route: "/overview", title: "Overview", description: "Overview", default_metric_id: "preterm_flag", metric_ids: ["preterm_flag"] },
                { page_id: "perinatal", route: "/perinatal", title: "Perinatal Health", description: "Perinatal", default_metric_id: "preterm_flag", metric_ids: ["preterm_flag"] },
                { page_id: "context", route: "/context", title: "Context and Inequalities", description: "Context", default_metric_id: "preterm_flag", metric_ids: ["preterm_flag"] },
                { page_id: "relationships", route: "/relationships", title: "Climate-Health Relationships", description: "Relationships", default_metric_id: null, metric_ids: [] },
                { page_id: "methods", route: "/methods", title: "Methods", description: "Methods", default_metric_id: null, metric_ids: [] },
            ],
            metrics: [],
            context_variables: [],
            context_bins: {},
            source_tables: [],
            artifacts: {},
            coverage: { municipalities: 5570, states: 27, metric_rows: 10 },
        }));
    }
    return new Response(JSON.stringify({}), { status: 200 });
}));
function renderMethods(initialEntry = "/methods") {
    const client = new QueryClient();
    render(_jsx(QueryClientProvider, { client: client, children: _jsx(MemoryRouter, { initialEntries: [initialEntry], children: _jsx(Routes, { children: _jsx(Route, { path: "/methods", element: _jsx(MethodsPage, {}) }) }) }) }));
}
describe("MethodsPage", () => {
    it("renders the technical catalog and opens a drawer from chip click", async () => {
        renderMethods();
        expect(await screen.findByText("Methods and Technical Specifications")).toBeInTheDocument();
        const chip = await screen.findByRole("button", { name: /Preterm births/i });
        fireEvent.click(chip);
        expect(await screen.findByText("Indicator technical sheet")).toBeInTheDocument();
        expect(screen.getByText("Numerator")).toBeInTheDocument();
        expect(screen.getByText("Denominator")).toBeInTheDocument();
    });
    it("opens a technical sheet directly from the query parameter", async () => {
        renderMethods("/methods?sheet=idh_quintile");
        expect(await screen.findByText("Stratifier technical sheet")).toBeInTheDocument();
        expect(screen.getByText("Source and observation year")).toBeInTheDocument();
        expect(screen.getByText("Binning and thresholds")).toBeInTheDocument();
    });
});
