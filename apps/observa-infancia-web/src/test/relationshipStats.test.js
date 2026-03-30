import { describe, expect, it } from "vitest";
import { computeLinearRegression, computeLinearRegressionByStratum, formatPValue } from "../lib/relationshipStats";
describe("relationshipStats", () => {
    it("estimates a simple linear fit for a perfect positive relationship", () => {
        const points = [
            { x_value: 1, y_value: 3, geo_code: "A", label: "A", context_stratum: "Q1", time_bucket: "2024-01-01", year: 2024, quality_flag_any: false, tooltip_meta: {} },
            { x_value: 2, y_value: 5, geo_code: "B", label: "B", context_stratum: "Q1", time_bucket: "2024-02-01", year: 2024, quality_flag_any: false, tooltip_meta: {} },
            { x_value: 3, y_value: 7, geo_code: "C", label: "C", context_stratum: "Q2", time_bucket: "2024-03-01", year: 2024, quality_flag_any: false, tooltip_meta: {} },
            { x_value: 4, y_value: 9, geo_code: "D", label: "D", context_stratum: "Q2", time_bucket: "2024-04-01", year: 2024, quality_flag_any: false, tooltip_meta: {} },
            { x_value: 5, y_value: 11, geo_code: "E", label: "E", context_stratum: "Q3", time_bucket: "2024-05-01", year: 2024, quality_flag_any: false, tooltip_meta: {} },
        ];
        const fit = computeLinearRegression(points);
        expect(fit).not.toBeNull();
        expect(fit?.slope).toBeCloseTo(2, 8);
        expect(fit?.intercept).toBeCloseTo(1, 8);
        expect(fit?.rSquared).toBeCloseTo(1, 8);
        expect(fit?.pValue).toBe(0);
        expect(formatPValue(fit?.pValue ?? null)).toBe("<0.001");
    });
    it("returns null when there is no usable x variation", () => {
        const points = [
            { x_value: 2, y_value: 3, geo_code: "A", label: "A", context_stratum: null, time_bucket: "2024-01-01", year: 2024, quality_flag_any: false, tooltip_meta: {} },
            { x_value: 2, y_value: 4, geo_code: "B", label: "B", context_stratum: null, time_bucket: "2024-02-01", year: 2024, quality_flag_any: false, tooltip_meta: {} },
            { x_value: 2, y_value: 5, geo_code: "C", label: "C", context_stratum: null, time_bucket: "2024-03-01", year: 2024, quality_flag_any: false, tooltip_meta: {} },
        ];
        expect(computeLinearRegression(points)).toBeNull();
    });
    it("estimates separate fits by context stratum", () => {
        const points = [
            { x_value: 1, y_value: 2, geo_code: "A", label: "A", context_stratum: "Q1", time_bucket: "2024-01-01", year: 2024, quality_flag_any: false, tooltip_meta: {} },
            { x_value: 2, y_value: 4, geo_code: "B", label: "B", context_stratum: "Q1", time_bucket: "2024-02-01", year: 2024, quality_flag_any: false, tooltip_meta: {} },
            { x_value: 3, y_value: 6, geo_code: "C", label: "C", context_stratum: "Q1", time_bucket: "2024-03-01", year: 2024, quality_flag_any: false, tooltip_meta: {} },
            { x_value: 1, y_value: 5, geo_code: "D", label: "D", context_stratum: "Q2", time_bucket: "2024-01-01", year: 2024, quality_flag_any: false, tooltip_meta: {} },
            { x_value: 2, y_value: 6, geo_code: "E", label: "E", context_stratum: "Q2", time_bucket: "2024-02-01", year: 2024, quality_flag_any: false, tooltip_meta: {} },
            { x_value: 3, y_value: 7, geo_code: "F", label: "F", context_stratum: "Q2", time_bucket: "2024-03-01", year: 2024, quality_flag_any: false, tooltip_meta: {} },
        ];
        const fits = computeLinearRegressionByStratum(points);
        expect(fits).toHaveLength(2);
        expect(fits.find((item) => item.stratum === "Q1")?.regression.slope).toBeCloseTo(2, 8);
        expect(fits.find((item) => item.stratum === "Q2")?.regression.slope).toBeCloseTo(1, 8);
    });
});
