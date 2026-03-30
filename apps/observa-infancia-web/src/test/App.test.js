import { jsx as _jsx } from "react/jsx-runtime";
import { describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { render, screen } from "@testing-library/react";
import App from "../App";
vi.mock("../components/ChoroplethPanel", () => ({
    ChoroplethPanel: () => _jsx("div", { children: "Mock map" })
}));
vi.mock("../components/TrendPanel", () => ({
    TrendPanel: () => _jsx("div", { children: "Mock trend" })
}));
vi.mock("../components/HeatmapPanel", () => ({
    HeatmapPanel: () => _jsx("div", { children: "Mock heatmap" })
}));
vi.stubGlobal("fetch", vi.fn(async (input) => {
    const url = String(input);
    if (url.endsWith("/api/v1/meta/metrics")) {
        return new Response(JSON.stringify({
            pages: [
                { page_id: "overview", route: "/overview", title: "Overview", description: "Overview page", default_metric_id: "deaths_u5", metric_ids: ["deaths_u5"] },
                { page_id: "methods", route: "/methods", title: "Methods", description: "Methods page", default_metric_id: null, metric_ids: [] }
            ],
            metrics: [
                {
                    metric_id: "deaths_u5",
                    label: "Under-five deaths",
                    short_label: "Under 5",
                    metric_family: "mortality",
                    page_id: "child_mortality",
                    source_table: "gold_mnch_mortality",
                    numerator_field: "DEATHS_U5",
                    denominator_field: "POP_U5",
                    definition_type: "primary",
                    default_value_mode: "count",
                    available_value_modes: ["count", "rate"],
                    allowed_granularities: ["month", "year", "season"],
                    allowed_geo_levels: ["national", "state", "municipality"],
                    display_unit: "deaths",
                    rate_display_unit: "per 100,000 under-five population",
                    minimum_denominator_threshold: 100,
                    quality_flags: [],
                    tooltip_note: "Test metric",
                    aggregation_method: "sum",
                    aggregation_weight_field: null,
                    denominator_aggregation: "max",
                    rate_multiplier: 100000,
                    sort_order: 1
                }
            ],
            context_variables: []
        }));
    }
    if (url.endsWith("/api/v1/meta/manifest")) {
        return new Response(JSON.stringify({
            dashboard_version: "v1",
            dashboard_name: "Observa Infancia",
            build_generated_at: new Date().toISOString(),
            years_covered: [2024],
            granularities: ["month", "year", "season"],
            geo_levels: ["state", "municipality", "national"],
            pages: [],
            metrics: [],
            context_variables: [],
            context_bins: {},
            source_tables: [],
            artifacts: {},
            coverage: { municipalities: 5570, states: 27, metric_rows: 10 }
        }));
    }
    if (url.endsWith("/api/v1/meta/filters")) {
        return new Response(JSON.stringify({
            years: [2024],
            states: [],
            municipalities: [],
            definitions: ["primary"],
            granularities: ["month", "year", "season"],
            geo_levels: ["state", "municipality", "national"],
            context_variables: []
        }));
    }
    if (url.includes("/query/")) {
        return new Response(JSON.stringify([]));
    }
    return new Response(JSON.stringify({}), { status: 200 });
}));
describe("App", () => {
    it("renders the overview route", async () => {
        const client = new QueryClient();
        render(_jsx(QueryClientProvider, { client: client, children: _jsx(MemoryRouter, { initialEntries: ["/overview"], children: _jsx(App, {}) }) }));
        expect(await screen.findByText("Overview page")).toBeInTheDocument();
    });
});
