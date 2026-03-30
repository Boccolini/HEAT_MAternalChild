import { fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import App from "../App";

vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo) => {
  const url = String(input);
  if (url.endsWith("/api/v1/statcalc/meta/recipes")) {
    return new Response(JSON.stringify([
      {
        recipe_id: "track_a_acute_fe_proxy",
        track_id: "track_a",
        label: "Aggregated acute analysis",
        description: "Acute recipe",
        outcome_family: "acute_daily",
        execution_behavior: "sync",
        default_template_id: "track_a_benchmark_proxy",
        default_offset_id: "log_annual_live_birth_offset",
        templates: [{ id: "track_a_benchmark_proxy", label: "Benchmark fixed-effects proxy", description: null, metadata: {} }],
        offsets: [
          { id: "log_annual_live_birth_offset", label: "Live births", description: null, metadata: {} },
          { id: "log_under_five_population_offset", label: "Under-five population", description: null, metadata: {} }
        ],
        outcomes: [
          {
            id: "neonatal_death_count",
            label: "Neonatal mortality",
            description: null,
            metadata: { offset_id: "log_annual_live_birth_offset" }
          },
          {
            id: "under_five_death_count",
            label: "Under-five mortality",
            description: null,
            metadata: { offset_id: "log_under_five_population_offset" }
          }
        ],
        exposures: [{ id: "maximum_daily_temperature_c", label: "Maximum daily temperature", description: null, metadata: {} }],
        lag_windows: [{ id: "lag0", label: "lag0", description: null, metadata: {} }],
        stratifiers: [],
        source_filters: [{ id: "all_sources", label: "All sources", description: null, metadata: {} }],
        grouping_templates: [{ id: "track_a_municipality_year", label: "Municipality-year FE", description: null, metadata: {} }],
        denominator_variants: [],
        gestational_source_modes: []
      }
    ]));
  }
  if (url.endsWith("/api/v1/statcalc/meta/stratifiers")) {
    return new Response(JSON.stringify([]));
  }
  if (url.endsWith("/api/v1/statcalc/meta/presets")) {
    return new Response(JSON.stringify([]));
  }
  if (url.endsWith("/api/v1/statcalc/runs")) {
    return new Response(JSON.stringify([]));
  }
  return new Response(JSON.stringify({}), { status: 200 });
}) as unknown as typeof fetch);

function renderApp() {
  const client = new QueryClient();
  render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  );
}

describe("StatCalc App", () => {
  it("renders the builder, help panel, and preview shell in English", async () => {
    renderApp();
    expect(await screen.findByText("Statistical calculator for outcome analyses")).toBeInTheDocument();
    expect(await screen.findByText("Specification builder")).toBeInTheDocument();
    expect(await screen.findByText("Live specification preview")).toBeInTheDocument();
    expect(await screen.findByText("Concept guide")).toBeInTheDocument();
    expect(await screen.findByText("Run history")).toBeInTheDocument();
    expect(await screen.findByText("Saved presets")).toBeInTheDocument();
    expect(await screen.findByRole("option", { name: "Primary fixed-effects model" })).toBeInTheDocument();
    expect(await screen.findByText("Exposure timing: Same-day exposure.")).toBeInTheDocument();
  });

  it("updates the displayed offset and preview when the outcome changes", async () => {
    renderApp();

    expect(await screen.findByDisplayValue("Live-birth denominator")).toBeInTheDocument();
    expect(await screen.findByText("Neonatal mortality as a function of Maximum daily temperature")).toBeInTheDocument();
    expect(await screen.findByText(/within the aggregated current-cycle stream using the Primary fixed-effects model/i)).toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox", { name: /Outcome/i }), { target: { value: "under_five_death_count" } });

    expect(await screen.findByDisplayValue("Under-five population denominator")).toBeInTheDocument();
    expect(await screen.findByText("Under-five mortality as a function of Maximum daily temperature")).toBeInTheDocument();
  });

  it("switches the concept panel from quick help to methodology", async () => {
    renderApp();

    expect(await screen.findByText(/A recipe is a validated analytical envelope/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Methodology" }));
    expect(await screen.findByText(/Recipes are not arbitrary model blueprints/i)).toBeInTheDocument();
  });
});
