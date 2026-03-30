import { describe, expect, it } from "vitest";
import { clipSeriesRates, clipStratificationRates } from "../lib/dashboardDisplay";
const rateFilters = {
    years: [2024],
    granularity: "month",
    geo_level: "municipality",
    geo_codes: [],
    value_mode: "rate",
    context_bins: [],
    compare_mode: "none",
};
describe("dashboardDisplay rate stabilisation", () => {
    it("clips extreme series rates at the upper tail in rate mode", () => {
        const points = [
            { x: "2024-01-01", y: 10, denominator: 100, quality_flag_any: false, tooltip_meta: {} },
            { x: "2024-02-01", y: 11, denominator: 100, quality_flag_any: false, tooltip_meta: {} },
            { x: "2024-03-01", y: 12, denominator: 100, quality_flag_any: false, tooltip_meta: {} },
            { x: "2024-04-01", y: 13, denominator: 100, quality_flag_any: false, tooltip_meta: {} },
            { x: "2024-05-01", y: 1000, denominator: 100, quality_flag_any: false, tooltip_meta: {} },
        ];
        const clipped = clipSeriesRates(points, rateFilters);
        expect(clipped.at(-1)?.y).not.toBe(1000);
        expect(clipped.at(-1)?.y).toBeLessThan(1000);
    });
    it("clips stratification cards in rate mode while preserving order inputs", () => {
        const rows = [
            { stratum: "Q1", value: 2, denominator: 120, n_geographies: 100 },
            { stratum: "Q2", value: 3, denominator: 120, n_geographies: 100 },
            { stratum: "Q3", value: 4, denominator: 120, n_geographies: 100 },
            { stratum: "Q4", value: 5, denominator: 120, n_geographies: 100 },
            { stratum: "Q5", value: 500, denominator: 120, n_geographies: 100 },
        ];
        const clipped = clipStratificationRates(rows, rateFilters);
        expect(clipped[4]?.value).toBeLessThan(500);
        expect(clipped[0]?.stratum).toBe("Q1");
    });
    it("does not clip when the dashboard is in count mode", () => {
        const filters = { ...rateFilters, value_mode: "count" };
        const points = [
            { x: "2024-01-01", y: 1, denominator: 100, quality_flag_any: false, tooltip_meta: {} },
            { x: "2024-02-01", y: 1000, denominator: 100, quality_flag_any: false, tooltip_meta: {} },
            { x: "2024-03-01", y: 2, denominator: 100, quality_flag_any: false, tooltip_meta: {} },
            { x: "2024-04-01", y: 3, denominator: 100, quality_flag_any: false, tooltip_meta: {} },
            { x: "2024-05-01", y: 4, denominator: 100, quality_flag_any: false, tooltip_meta: {} },
        ];
        expect(clipSeriesRates(points, filters)[1]?.y).toBe(1000);
    });
});
