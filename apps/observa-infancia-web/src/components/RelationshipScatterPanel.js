import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useRef } from "react";
import ReactECharts from "echarts-for-react";
import { formatContextValue } from "../lib/dashboardDisplay";
import { PanelActionButton } from "./PanelActionButton";
import { PanelDownloadActions } from "./PanelDownloadActions";
import { buildDownloadFilename, downloadChartImage, downloadCsv } from "../lib/download";
const BASE_PALETTE = ["#365C72", "#4C8C8A", "#86A86F", "#C76A1B", "#8E2C2C", "#8A7F70"];
const REGION_ORDER = ["North", "Northeast", "Southeast", "South", "Central-West"];
const UF_TO_REGION = {
    AC: "North",
    AP: "North",
    AM: "North",
    PA: "North",
    RO: "North",
    RR: "North",
    TO: "North",
    AL: "Northeast",
    BA: "Northeast",
    CE: "Northeast",
    MA: "Northeast",
    PB: "Northeast",
    PE: "Northeast",
    PI: "Northeast",
    RN: "Northeast",
    SE: "Northeast",
    ES: "Southeast",
    MG: "Southeast",
    RJ: "Southeast",
    SP: "Southeast",
    PR: "South",
    RS: "South",
    SC: "South",
    DF: "Central-West",
    GO: "Central-West",
    MS: "Central-West",
    MT: "Central-West",
};
function colorForIndex(index) {
    if (index < BASE_PALETTE.length) {
        return BASE_PALETTE[index];
    }
    const hue = (index * 137.508) % 360;
    return `hsl(${hue.toFixed(1)} 48% 42%)`;
}
function buildSmoothedCurve(points) {
    const valid = points
        .filter((point) => Number.isFinite(point.x_value) && Number.isFinite(point.y_value))
        .sort((left, right) => left.x_value - right.x_value);
    if (valid.length < 8) {
        return valid.map((point) => ({ x: point.x_value, y: point.y_value }));
    }
    const bucketCount = Math.min(16, Math.max(6, Math.round(Math.sqrt(valid.length))));
    const bucketSize = Math.ceil(valid.length / bucketCount);
    const curve = [];
    for (let index = 0; index < valid.length; index += bucketSize) {
        const bucket = valid.slice(index, index + bucketSize);
        if (bucket.length === 0) {
            continue;
        }
        const meanX = bucket.reduce((sum, point) => sum + point.x_value, 0) / bucket.length;
        const meanY = bucket.reduce((sum, point) => sum + point.y_value, 0) / bucket.length;
        curve.push({ x: meanX, y: meanY });
    }
    return curve;
}
function normalizeRegionLabel(point) {
    if (REGION_ORDER.includes(point.label)) {
        return point.label;
    }
    if (REGION_ORDER.includes(point.geo_code)) {
        return point.geo_code;
    }
    return UF_TO_REGION[point.geo_code] ?? point.label;
}
export function RelationshipScatterPanel({ title, subtitle, xLabel, yLabel, points, linearRegression, stratifiedRegressions = [], fitMode, geoLevel, contextVariable, contextVariableLabel, contextOrder, fitGroupingLabel, downloadMetadata, }) {
    const chartRef = useRef(null);
    const groupedScatter = new Map();
    const groupLabelByKey = new Map();
    const colorMode = contextVariable ? "context" : geoLevel === "region" ? "region" : geoLevel === "state" ? "state" : "single";
    const groupingLabel = colorMode === "context"
        ? contextVariableLabel ?? "Context variable"
        : colorMode === "region"
            ? "Region"
            : colorMode === "state"
                ? "State"
                : undefined;
    for (const point of points) {
        let groupKey = "all";
        let groupLabel = "Observed points";
        if (colorMode === "context") {
            groupKey = point.context_stratum ?? "missing";
            groupLabel = formatContextValue(contextVariable, groupKey);
        }
        else if (colorMode === "region") {
            groupKey = normalizeRegionLabel(point);
            groupLabel = normalizeRegionLabel(point);
        }
        else if (colorMode === "state") {
            groupKey = point.geo_code;
            groupLabel = point.label;
        }
        const bucket = groupedScatter.get(groupKey) ?? [];
        bucket.push({
            value: [point.x_value, point.y_value],
            label: colorMode === "region" ? groupLabel : point.label,
            timeBucket: point.time_bucket,
            qualityFlag: point.quality_flag_any,
            groupLabel,
        });
        groupedScatter.set(groupKey, bucket);
        groupLabelByKey.set(groupKey, groupLabel);
    }
    const orderedStrata = [...groupedScatter.keys()].sort((left, right) => {
        if (colorMode === "context" && contextOrder) {
            const leftIndex = contextOrder.indexOf(left);
            const rightIndex = contextOrder.indexOf(right);
            return (leftIndex === -1 ? Number.MAX_SAFE_INTEGER : leftIndex) - (rightIndex === -1 ? Number.MAX_SAFE_INTEGER : rightIndex);
        }
        if (colorMode === "region") {
            const leftIndex = REGION_ORDER.indexOf(groupLabelByKey.get(left) ?? left);
            const rightIndex = REGION_ORDER.indexOf(groupLabelByKey.get(right) ?? right);
            return (leftIndex === -1 ? Number.MAX_SAFE_INTEGER : leftIndex) - (rightIndex === -1 ? Number.MAX_SAFE_INTEGER : rightIndex);
        }
        return (groupLabelByKey.get(left) ?? left).localeCompare(groupLabelByKey.get(right) ?? right);
    });
    const smoothCurve = buildSmoothedCurve(points);
    const colorByStratum = new Map(orderedStrata.map((stratum, index) => [stratum, colorForIndex(index)]));
    const visibleStratifiedFits = fitMode === "stratum"
        ? stratifiedRegressions
            .slice()
            .sort((left, right) => {
            if (colorMode === "region") {
                const leftIndex = REGION_ORDER.indexOf(left.stratum);
                const rightIndex = REGION_ORDER.indexOf(right.stratum);
                return (leftIndex === -1 ? Number.MAX_SAFE_INTEGER : leftIndex) - (rightIndex === -1 ? Number.MAX_SAFE_INTEGER : rightIndex);
            }
            if (!contextOrder) {
                return left.stratum.localeCompare(right.stratum);
            }
            const leftIndex = contextOrder.indexOf(left.stratum);
            const rightIndex = contextOrder.indexOf(right.stratum);
            return (leftIndex === -1 ? Number.MAX_SAFE_INTEGER : leftIndex) - (rightIndex === -1 ? Number.MAX_SAFE_INTEGER : rightIndex);
        })
        : [];
    const option = {
        tooltip: {
            trigger: "item",
            formatter(params) {
                const data = params.data;
                if (Array.isArray(data) && params.seriesType === "line") {
                    return [
                        `<strong>${params.seriesName ?? "Model fit"}</strong>`,
                        `${xLabel}: ${data[0].toLocaleString(undefined, { maximumFractionDigits: 2 })}`,
                        `${yLabel}: ${data[1].toLocaleString(undefined, { maximumFractionDigits: 2 })}`,
                    ].join("<br/>");
                }
                if (Array.isArray(data) || !data?.value) {
                    return params.seriesName ?? "";
                }
                const value = data.value;
                const qualityText = data.qualityFlag ? "<br/>Contains one or more quality flags" : "";
                return [
                    `<strong>${data.label ?? "Observation"}</strong>`,
                    groupingLabel && data.groupLabel ? `${groupingLabel}: ${data.groupLabel}` : "",
                    `${data.timeBucket ?? ""}`,
                    `${xLabel}: ${value[0].toLocaleString(undefined, { maximumFractionDigits: 2 })}`,
                    `${yLabel}: ${value[1].toLocaleString(undefined, { maximumFractionDigits: 2 })}${qualityText}`,
                ].filter(Boolean).join("<br/>");
            },
        },
        grid: { left: 64, right: 24, top: 56, bottom: 72 },
        xAxis: {
            type: "value",
            name: xLabel,
            nameLocation: "middle",
            nameGap: 44,
            axisLabel: { color: "#365C72" },
            splitLine: { lineStyle: { color: "#D8DEE3" } },
        },
        yAxis: {
            type: "value",
            name: yLabel,
            nameLocation: "middle",
            nameGap: 56,
            axisLabel: { color: "#365C72" },
            splitLine: { lineStyle: { color: "#D8DEE3" } },
        },
        dataZoom: [
            { type: "inside", xAxisIndex: 0 },
            { type: "inside", yAxisIndex: 0 },
        ],
        legend: {
            type: orderedStrata.length > 8 ? "scroll" : "plain",
            top: 12,
            textStyle: { color: "#365C72" },
        },
        series: [
            ...orderedStrata.map((stratum, index) => ({
                name: groupLabelByKey.get(stratum) ?? "Observed points",
                type: "scatter",
                data: groupedScatter.get(stratum) ?? [],
                itemStyle: {
                    color: colorForIndex(index),
                    opacity: 0.65,
                },
                emphasis: {
                    itemStyle: {
                        color: colorForIndex(index),
                        opacity: 1,
                    },
                },
                symbolSize: 10,
            })),
            {
                name: "Smooth trend",
                type: "line",
                data: smoothCurve.map((point) => [point.x, point.y]),
                smooth: true,
                showSymbol: false,
                lineStyle: {
                    color: "#C76A1B",
                    width: 3,
                },
            },
            ...(fitMode === "overall" && linearRegression
                ? [
                    {
                        name: "Linear fit",
                        type: "line",
                        data: linearRegression.line,
                        showSymbol: false,
                        lineStyle: {
                            color: "#12324A",
                            width: 2,
                            type: "dashed",
                        },
                    },
                ]
                : []),
            ...visibleStratifiedFits.map(({ stratum, regression }) => ({
                name: `${contextVariable ? formatContextValue(contextVariable, stratum) : stratum} fit`,
                type: "line",
                data: regression.line,
                showSymbol: false,
                lineStyle: {
                    color: colorByStratum.get(stratum) ?? "#12324A",
                    width: 2,
                    type: "dashed",
                },
            })),
        ],
    };
    return (_jsxs("section", { className: "panel", children: [_jsxs("div", { className: "panel-header", children: [_jsxs("div", { children: [_jsx("h3", { className: "text-base font-semibold", children: title }), _jsx("p", { className: "mt-1 text-sm text-slate-500", children: subtitle })] }), _jsxs(PanelDownloadActions, { children: [_jsx(PanelActionButton, { label: "PNG", onClick: () => {
                                    const chart = chartRef.current?.getEchartsInstance();
                                    if (!chart) {
                                        return;
                                    }
                                    void downloadChartImage(chart.getDataURL({ pixelRatio: 2, backgroundColor: "#ffffff" }), buildDownloadFilename(title, "png"), downloadMetadata);
                                } }), _jsx(PanelActionButton, { label: "CSV", onClick: () => downloadCsv(points.map((point) => ({
                                    geo_code: point.geo_code,
                                    geography: point.label,
                                    time_bucket: point.time_bucket,
                                    year: point.year,
                                    context_stratum: point.context_stratum,
                                    x_value: point.x_value,
                                    y_value: point.y_value,
                                    quality_flag_any: point.quality_flag_any,
                                })), buildDownloadFilename(title, "csv"), downloadMetadata) })] })] }), _jsxs("div", { className: "panel-body", children: [contextVariable ? (_jsxs("div", { className: "mb-4 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900", children: ["Context variable is active. Point colours now represent ", contextVariableLabel ?? "contextual strata", ", so geographic colouring by region or state is temporarily disabled."] })) : null, _jsx(ReactECharts, { ref: chartRef, option: option, style: { height: 560 } }), _jsxs("p", { className: "mt-4 text-xs text-slate-500", children: ["The orange curve shows a quantile-binned smooth trend across the observed exposure range.", fitMode === "overall"
                                ? " The dashed navy line shows a simple linear fit."
                                : ` Dashed lines show simple linear fits estimated within each visible ${fitGroupingLabel ?? "stratum"}.`, " ", colorMode === "context"
                                ? "Point colours follow the selected contextual stratification."
                                : colorMode === "region"
                                    ? "Point colours follow Brazilian macroregions."
                                    : colorMode === "state"
                                        ? "Point colours follow states."
                                        : "Points use a common colour because no contextual or regional grouping is active."] })] })] }));
}
