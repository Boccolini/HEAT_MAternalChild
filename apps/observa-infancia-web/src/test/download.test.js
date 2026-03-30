import { describe, expect, it } from "vitest";
import { buildDownloadFilename, rowsToCsvContent } from "../lib/download";
describe("download helpers", () => {
    it("serializes rows to CSV with escaping", () => {
        const csv = rowsToCsvContent([
            { geography: "Rio, Brazil", value: 12.3, note: 'He said "ok"' },
            { geography: "Sao Paulo", value: null, note: "plain" },
        ]);
        expect(csv).toContain('geography,value,note');
        expect(csv).toContain('"Rio, Brazil",12.3,"He said ""ok"""');
        expect(csv).toContain("Sao Paulo,,plain");
    });
    it("prepends title and notes metadata to CSV exports", () => {
        const csv = rowsToCsvContent([{ geography: "Rio", value: 1 }], { title: "Spatial distribution", notes: ["Years: 2020-2024", "Display geography: state"] });
        expect(csv).toContain("# Title: Spatial distribution");
        expect(csv).toContain("# Notes:");
        expect(csv).toContain("# - Years: 2020-2024");
        expect(csv).toContain("geography,value");
    });
    it("builds a predictable filename prefix and extension", () => {
        const filename = buildDownloadFilename("Spatial distribution", "csv");
        expect(filename.startsWith("observa-infancia-spatial-distribution-")).toBe(true);
        expect(filename.endsWith(".csv")).toBe(true);
    });
});
