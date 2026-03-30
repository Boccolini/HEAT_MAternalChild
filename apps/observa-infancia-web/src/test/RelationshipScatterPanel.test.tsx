import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { forwardRef } from "react";
import type { RelationshipPoint } from "../lib/api";
import { RelationshipScatterPanel } from "../components/RelationshipScatterPanel";

vi.mock("echarts-for-react", () => ({
  default: forwardRef(function MockEcharts(props: { option: { series?: Array<{ name?: string }> } }, _ref) {
    const seriesNames = (props.option.series ?? []).map((series) => series.name ?? "").join("|");
    return <div data-testid="relationship-chart" data-series={seriesNames} />;
  }),
}));

const basePoints: RelationshipPoint[] = [
  {
    x_value: 31.2,
    y_value: 12.4,
    geo_code: "N",
    label: "North",
    context_stratum: null,
    time_bucket: "2024-01",
    year: 2024,
    quality_flag_any: false,
    tooltip_meta: {},
  },
  {
    x_value: 34.1,
    y_value: 18.1,
    geo_code: "S",
    label: "South",
    context_stratum: null,
    time_bucket: "2024-02",
    year: 2024,
    quality_flag_any: false,
    tooltip_meta: {},
  },
];

afterEach(() => {
  cleanup();
});

describe("RelationshipScatterPanel", () => {
  it("uses geographic colouring by region when no context variable is active", () => {
    const { getByTestId } = render(
      <RelationshipScatterPanel
        title="Exposure-response scatter"
        subtitle="Test"
        xLabel="Temperature"
        yLabel="Preterm births"
        points={basePoints}
        linearRegression={null}
        fitMode="overall"
        geoLevel="region"
      />,
    );

    const chart = getByTestId("relationship-chart");
    expect(chart.getAttribute("data-series")).toContain("North");
    expect(chart.getAttribute("data-series")).toContain("South");
    expect(screen.getByText(/Point colours follow Brazilian macroregions/i)).toBeInTheDocument();
  });

  it("normalizes state-coded region points to macroregion legend labels", () => {
    const { getByTestId } = render(
      <RelationshipScatterPanel
        title="Exposure-response scatter"
        subtitle="Test"
        xLabel="Temperature"
        yLabel="Preterm births"
        points={[
          { ...basePoints[0], geo_code: "SP", label: "Sao Paulo" },
          { ...basePoints[1], geo_code: "RS", label: "Rio Grande do Sul" },
        ]}
        linearRegression={null}
        fitMode="overall"
        geoLevel="region"
      />,
    );

    const chart = getByTestId("relationship-chart");
    expect(chart.getAttribute("data-series")).toContain("Southeast");
    expect(chart.getAttribute("data-series")).toContain("South");
    expect(chart.getAttribute("data-series")).not.toContain("Sao Paulo");
    expect(chart.getAttribute("data-series")).not.toContain("Rio Grande do Sul");
  });

  it("shows a warning when contextual colouring overrides geographic colours", () => {
    const { getByTestId } = render(
      <RelationshipScatterPanel
        title="Exposure-response scatter"
        subtitle="Test"
        xLabel="Temperature"
        yLabel="Preterm births"
        points={[
          { ...basePoints[0], context_stratum: "Q1" },
          { ...basePoints[1], context_stratum: "Q5" },
        ]}
        linearRegression={null}
        fitMode="overall"
        geoLevel="state"
        contextVariable="idh_quintile"
        contextVariableLabel="Municipal Human Development Index quintile"
        contextOrder={["Q1", "Q2", "Q3", "Q4", "Q5", "missing"]}
      />,
    );

    expect(
      screen.getByText(/Context variable is active\. Point colours now represent Municipal Human Development Index quintile/i),
    ).toBeInTheDocument();
    const chart = getByTestId("relationship-chart");
    expect(chart.getAttribute("data-series")).toContain("Lower development");
    expect(chart.getAttribute("data-series")).toContain("Higher development");
  });
});
