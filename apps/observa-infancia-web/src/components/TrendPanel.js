import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useRef } from "react";
import ReactECharts from "echarts-for-react";
import { PanelActionButton } from "./PanelActionButton";
import { PanelDownloadActions } from "./PanelDownloadActions";
import { buildDownloadFilename, downloadChartImage, downloadCsv } from "../lib/download";
function formatNumericValue(value) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return "NA";
    }
    return value.toLocaleString(undefined, { maximumFractionDigits: value < 1 ? 3 : 2 });
}
export function TrendPanel({ title, points = [], seriesCollections, valueUnit, note, downloadMetadata }) {
    const chartRef = useRef(null);
    const collections = seriesCollections && seriesCollections.length > 0
        ? seriesCollections
        : [{ label: title, color: "#12324A", points }];
    const xAxisValues = collections[0]?.points.map((point) => point.x) ?? [];
    const option = {
        tooltip: {
            trigger: "axis",
            formatter: (items) => {
                if (!items || items.length === 0) {
                    return "";
                }
                const header = `<strong>${items[0].axisValueLabel}</strong>`;
                const rows = items.map((item) => `${item.seriesName}: ${formatNumericValue(item.value)}${valueUnit ? ` ${valueUnit}` : ""}`);
                return [
                    header,
                    ...rows,
                ].join("<br/>");
            },
        },
        grid: { left: 40, right: 20, top: 30, bottom: 40 },
        xAxis: {
            type: "category",
            data: xAxisValues,
            axisLabel: { color: "#365C72" }
        },
        yAxis: {
            type: "value",
            name: valueUnit,
            nameLocation: "middle",
            nameGap: 56,
            axisLabel: { color: "#365C72" },
            splitLine: { lineStyle: { color: "#D8DEE3" } }
        },
        legend: collections.length > 1
            ? {
                top: 0,
                textStyle: { color: "#365C72" },
            }
            : undefined,
        series: [
            ...collections.map((collection, index) => ({
                name: collection.label,
                type: "line",
                data: collection.points.map((point) => point.y),
                smooth: true,
                lineStyle: { color: collection.color, width: 3 },
                itemStyle: { color: collection.color },
                areaStyle: collections.length === 1 && index === 0 ? { color: "rgba(18, 50, 74, 0.08)" } : undefined,
            })),
        ]
    };
    return (_jsxs("section", { className: "panel", children: [_jsxs("div", { className: "panel-header", children: [_jsxs("div", { children: [_jsx("h3", { className: "text-base font-semibold", children: title }), valueUnit ? _jsxs("p", { className: "mt-1 text-xs text-slate-500", children: ["Unit: ", valueUnit] }) : null] }), _jsxs(PanelDownloadActions, { children: [_jsx(PanelActionButton, { label: "PNG", onClick: () => {
                                    const chart = chartRef.current?.getEchartsInstance();
                                    if (!chart) {
                                        return;
                                    }
                                    void downloadChartImage(chart.getDataURL({ pixelRatio: 2, backgroundColor: "#ffffff" }), buildDownloadFilename(title, "png"), downloadMetadata);
                                } }), _jsx(PanelActionButton, { label: "CSV", onClick: () => downloadCsv(collections.flatMap((collection) => collection.points.map((point) => ({
                                    series: collection.label,
                                    time_bucket: point.x,
                                    value: point.y,
                                    denominator: point.denominator,
                                    quality_flag_any: point.quality_flag_any,
                                }))), buildDownloadFilename(title, "csv"), downloadMetadata) })] })] }), _jsxs("div", { className: "panel-body", children: [_jsx(ReactECharts, { ref: chartRef, option: option, style: { height: 320 } }), note ? _jsx("p", { className: "mt-3 text-xs text-slate-500", children: note }) : null] })] }));
}
