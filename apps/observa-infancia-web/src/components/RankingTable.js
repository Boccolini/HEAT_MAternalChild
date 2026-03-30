import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { PanelActionButton } from "./PanelActionButton";
import { PanelDownloadActions } from "./PanelDownloadActions";
import { buildDownloadFilename, downloadCsv } from "../lib/download";
export function RankingTable({ title, rows, onSelectMunicipality, downloadMetadata }) {
    const visibleRows = rows.slice(0, 15);
    return (_jsxs("section", { className: "panel", children: [_jsxs("div", { className: "panel-header", children: [_jsx("h3", { className: "text-base font-semibold", children: title }), _jsx(PanelDownloadActions, { children: _jsx(PanelActionButton, { label: "CSV", onClick: () => downloadCsv(visibleRows.map((row) => ({
                                rank: row.rank,
                                geo_code: row.geo_code,
                                geography: row.label,
                                value: row.value,
                                denominator: row.denominator,
                                quality_flag_any: row.quality_flag_any,
                            })), buildDownloadFilename(title, "csv"), downloadMetadata) }) })] }), _jsx("div", { className: "panel-body overflow-auto", children: _jsxs("table", { className: "w-full text-sm", children: [_jsx("thead", { className: "text-left text-slate-500", children: _jsxs("tr", { children: [_jsx("th", { className: "pb-3", children: "Rank" }), _jsx("th", { className: "pb-3", children: "Geography" }), _jsx("th", { className: "pb-3 text-right", children: "Value" })] }) }), _jsx("tbody", { children: visibleRows.map((row) => (_jsxs("tr", { className: "border-t border-rule", children: [_jsx("td", { className: "py-3", children: row.rank }), _jsx("td", { className: "py-3", children: _jsx("button", { type: "button", className: "text-left text-blue hover:underline", onClick: () => row.geo_code.length === 7 && onSelectMunicipality(row.geo_code), children: row.label }) }), _jsx("td", { className: "py-3 text-right", children: row.value === null ? "NA" : row.value.toFixed(2) })] }, row.geo_code))) })] }) })] }));
}
