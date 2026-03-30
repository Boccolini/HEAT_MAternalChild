import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useMemo, useRef, useState } from "react";
import { ComposableMap, Geographies, Geography } from "react-simple-maps";
import { PanelActionButton } from "./PanelActionButton";
import { PanelDownloadActions } from "./PanelDownloadActions";
import { buildDownloadFilename, downloadCsv, downloadSvg } from "../lib/download";
function interpolateColor(value, min, max) {
    if (max <= min) {
        return "#365C72";
    }
    const ratio = (value - min) / (max - min);
    const start = [247, 247, 244];
    const end = [18, 50, 74];
    const channel = start.map((base, index) => Math.round(base + (end[index] - base) * ratio));
    return `rgb(${channel[0]}, ${channel[1]}, ${channel[2]})`;
}
export function ChoroplethPanel({ title, data, geoLevel, onSelectMunicipality, downloadMetadata }) {
    const [hovered, setHovered] = useState(null);
    const containerRef = useRef(null);
    const geometryPath = geoLevel === "municipality"
        ? "/dashboard-assets/municipalities_2022_simplified.topojson"
        : "/dashboard-assets/states_2022_simplified.topojson";
    const projectionScale = geoLevel === "municipality" ? 1325 : 1025;
    const dataMap = useMemo(() => new Map(data.map((item) => [item.geo_code, item])), [data]);
    const values = data.map((item) => item.value).filter((value) => value !== null);
    const min = values.length ? Math.min(...values) : 0;
    const max = values.length ? Math.max(...values) : 1;
    return (_jsxs("section", { className: "panel", children: [_jsxs("div", { className: "panel-header", children: [_jsxs("div", { children: [_jsx("h3", { className: "text-base font-semibold", children: title }), hovered && _jsxs("p", { className: "text-sm text-slate-600", children: [hovered.label, ": ", hovered.value?.toFixed(2) ?? "NA"] })] }), _jsxs(PanelDownloadActions, { children: [_jsx(PanelActionButton, { label: "SVG", onClick: () => {
                                    const svg = containerRef.current?.querySelector("svg");
                                    if (svg instanceof SVGSVGElement) {
                                        downloadSvg(svg, buildDownloadFilename(title, "svg"), downloadMetadata);
                                    }
                                } }), _jsx(PanelActionButton, { label: "CSV", onClick: () => downloadCsv(data.map((row) => ({
                                    geo_code: row.geo_code,
                                    geography: row.label,
                                    value: row.value,
                                    denominator: row.denominator,
                                    quality_flag_any: row.quality_flag_any,
                                })), buildDownloadFilename(title, "csv"), downloadMetadata) })] })] }), _jsx("div", { ref: containerRef, className: "panel-body overflow-hidden", children: _jsx(ComposableMap, { projection: "geoMercator", projectionConfig: { center: [-54, -15], scale: projectionScale }, width: 980, height: 640, style: { width: "100%", height: "clamp(500px, 58vw, 700px)" }, children: _jsx(Geographies, { geography: geometryPath, children: ({ geographies }) => geographies.map((geo) => {
                            const geoCode = String(geo.properties.geo_code ?? geo.properties.municipality_code ?? geo.id);
                            const datum = dataMap.get(geoCode);
                            const fill = datum?.value !== null && datum?.value !== undefined ? interpolateColor(datum.value, min, max) : "#F1F3F4";
                            return (_jsx(Geography, { geography: geo, fill: fill, stroke: geoLevel === "municipality" ? "#ffffff" : "#d8dee3", strokeWidth: geoLevel === "municipality" ? 0.1 : 0.3, style: {
                                    default: { outline: "none" },
                                    hover: { fill: "#C76A1B", outline: "none" },
                                    pressed: { outline: "none" }
                                }, onMouseEnter: () => setHovered(datum ?? null), onMouseLeave: () => setHovered(null), onClick: () => {
                                    if (geoLevel === "municipality" && datum) {
                                        onSelectMunicipality(datum.geo_code);
                                    }
                                } }, geo.rsmKey));
                        }) }) }) })] }));
}
