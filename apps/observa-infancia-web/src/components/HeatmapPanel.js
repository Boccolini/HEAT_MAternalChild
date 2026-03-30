import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useRef } from "react";
import ReactECharts from "echarts-for-react";
import { PanelActionButton } from "./PanelActionButton";
import { PanelDownloadActions } from "./PanelDownloadActions";
import { buildDownloadFilename, downloadChartImage, downloadCsv } from "../lib/download";
function toHeatmap(points) {
    const rows = points
        .map((point) => {
        const [year, month] = point.x.split("-").map(Number);
        if (!year || !month) {
            return null;
        }
        return { year, month, value: point.y ?? null };
    })
        .filter((item) => item !== null);
    const years = Array.from(new Set(rows.map((row) => row.year))).sort();
    const months = Array.from({ length: 12 }, (_, index) => index + 1);
    const matrix = rows.map((row) => [row.month - 1, years.indexOf(row.year), row.value]);
    return { years, months, matrix };
}
function formatNumericValue(value) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return "NA";
    }
    return value.toLocaleString(undefined, { maximumFractionDigits: value < 1 ? 3 : 2 });
}
export function HeatmapPanel({ title, points, valueUnit, note, downloadMetadata }) {
    const chartRef = useRef(null);
    const { years, months, matrix } = toHeatmap(points);
    const option = {
        tooltip: {
            position: "top",
            formatter: (params) => {
                const [monthIndex, yearIndex, value] = params.data;
                const year = years[yearIndex];
                const month = months[monthIndex]?.toString().padStart(2, "0");
                return [
                    `<strong>${year}-${month}</strong>`,
                    `${formatNumericValue(value)}${valueUnit ? ` ${valueUnit}` : ""}`,
                ].join("<br/>");
            },
        },
        grid: { left: 40, right: 20, top: 20, bottom: 40 },
        xAxis: {
            type: "category",
            data: months.map((month) => month.toString().padStart(2, "0")),
            axisLabel: { color: "#365C72" }
        },
        yAxis: {
            type: "category",
            data: years.map(String),
            axisLabel: { color: "#365C72" }
        },
        visualMap: {
            min: 0,
            max: Math.max(...matrix.map((row) => row[2] ?? 0), 1),
            calculable: true,
            orient: "horizontal",
            left: "center",
            bottom: 0,
            text: valueUnit ? [valueUnit, ""] : undefined,
            inRange: { color: ["#F7F7F4", "#365C72"] }
        },
        series: [
            {
                type: "heatmap",
                data: matrix,
                label: { show: false }
            }
        ]
    };
    return (_jsxs("section", { className: "panel", children: [_jsxs("div", { className: "panel-header", children: [_jsxs("div", { children: [_jsx("h3", { className: "text-base font-semibold", children: title }), valueUnit ? _jsxs("p", { className: "mt-1 text-xs text-slate-500", children: ["Unit: ", valueUnit] }) : null] }), _jsxs(PanelDownloadActions, { children: [_jsx(PanelActionButton, { label: "PNG", onClick: () => {
                                    const chart = chartRef.current?.getEchartsInstance();
                                    if (!chart) {
                                        return;
                                    }
                                    void downloadChartImage(chart.getDataURL({ pixelRatio: 2, backgroundColor: "#ffffff" }), buildDownloadFilename(title, "png"), downloadMetadata);
                                } }), _jsx(PanelActionButton, { label: "CSV", onClick: () => downloadCsv(matrix.map(([monthIndex, yearIndex, value]) => ({
                                    year: years[yearIndex] ?? null,
                                    month: months[monthIndex] ?? null,
                                    value,
                                })), buildDownloadFilename(title, "csv"), downloadMetadata) })] })] }), _jsxs("div", { className: "panel-body", children: [_jsx(ReactECharts, { ref: chartRef, option: option, style: { height: 320 } }), note ? _jsx("p", { className: "mt-3 text-xs text-slate-500", children: note }) : null] })] }));
}
