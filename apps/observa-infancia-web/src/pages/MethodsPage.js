import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { DisplayRulesPanel } from "../components/DisplayRulesPanel";
import { MedallionStrategyPanel } from "../components/MedallionStrategyPanel";
import { SourceSystemsPanel } from "../components/SourceSystemsPanel";
import { TechnicalSheetCatalog } from "../components/TechnicalSheetCatalog";
import { TechnicalSheetDrawer } from "../components/TechnicalSheetDrawer";
import { getManifest, getMethodsMeta, } from "../lib/api";
import { DASHBOARD_FROZEN_BASELINE_DATE, DASHBOARD_FROZEN_BASELINE_NOTE, DASHBOARD_FROZEN_BASELINE_VERSION } from "../lib/dashboardRelease";
export function MethodsPage() {
    const [searchParams, setSearchParams] = useSearchParams();
    const [searchTerm, setSearchTerm] = useState("");
    const [typeFilter, setTypeFilter] = useState("all");
    const [familyFilter, setFamilyFilter] = useState("all");
    const { data: methodsPayload, error } = useQuery({ queryKey: ["methods-meta"], queryFn: getMethodsMeta });
    const { data: manifest } = useQuery({ queryKey: ["manifest"], queryFn: getManifest });
    const pageTitleMap = useMemo(() => Object.fromEntries((manifest?.pages ?? []).map((page) => [page.page_id, page.title])), [manifest?.pages]);
    const sourceSystemMap = useMemo(() => Object.fromEntries((methodsPayload?.source_systems ?? []).map((sourceSystem) => [sourceSystem.source_system_id, sourceSystem])), [methodsPayload?.source_systems]);
    const activeSheetId = searchParams.get("sheet");
    const activeSheet = useMemo(() => {
        if (!activeSheetId || !methodsPayload) {
            return null;
        }
        return (methodsPayload.indicator_sheets.find((sheet) => sheet.sheet_id === activeSheetId) ??
            methodsPayload.stratifier_sheets.find((sheet) => sheet.sheet_id === activeSheetId) ??
            null);
    }, [activeSheetId, methodsPayload]);
    function setSheet(sheetId) {
        const next = new URLSearchParams(searchParams);
        if (sheetId) {
            next.set("sheet", sheetId);
        }
        else {
            next.delete("sheet");
        }
        setSearchParams(next, { replace: true });
    }
    if (error) {
        return _jsx("div", { className: "text-sm text-caution", children: "Unable to load methods metadata from the dashboard API." });
    }
    if (!methodsPayload || !manifest) {
        return _jsx("div", { className: "text-sm text-slate-500", children: "Loading technical methods metadata..." });
    }
    return (_jsxs("div", { className: "space-y-6", children: [_jsxs("section", { className: "space-y-2", children: [_jsx("h2", { className: "text-2xl font-semibold", children: "Methods and Technical Specifications" }), _jsx("p", { className: "max-w-4xl text-sm text-slate-600", children: "This page documents KPI construction, contextual stratifier construction, source provenance, medallion lineage, and dashboard display rules for the internal beta build." })] }), _jsxs("section", { className: "panel", children: [_jsx("div", { className: "panel-header", children: _jsx("h3", { className: "text-base font-semibold", children: "Method boundary for this release" }) }), _jsxs("div", { className: "panel-body space-y-3 text-sm text-slate-700", children: [_jsx("p", { children: "The dashboard is a descriptive and surveillance-facing interface built on the frozen workflow used in the current study package." }), _jsx("p", { children: "Its role is to display retained indicators, contextual patterns, medallion lineage, and page-level exploration rules. It is not a standalone causal engine." }), _jsx("p", { children: "Principal, supportive, and provisional signals should be read according to the frozen validation hierarchy in this release baseline." })] })] }), _jsxs("section", { className: "panel", children: [_jsx("div", { className: "panel-header", children: _jsx("h3", { className: "text-base font-semibold", children: "Citation and DOI status" }) }), _jsxs("div", { className: "panel-body space-y-3 text-sm text-slate-700", children: [_jsx("p", { children: "All generated downloads should carry the dashboard citation metadata for sharing and reuse." }), _jsx("p", { children: "Suggested citation: Boccolini, CS. Brazil's heat exposure on maternal, newborn, and child health - Dashboard." }), _jsx("p", { children: "DOI status: pending release archival. The exact DOI should be attached after the frozen dashboard release is archived." })] })] }), _jsxs("section", { className: "panel", children: [_jsx("div", { className: "panel-header", children: _jsx("h3", { className: "text-base font-semibold", children: "Technical overview" }) }), _jsxs("div", { className: "panel-body grid gap-4 md:grid-cols-2 xl:grid-cols-6", children: [_jsxs("div", { className: "rounded-2xl border border-rule p-4", children: [_jsx("div", { className: "text-xs uppercase tracking-wide text-slate-500", children: "Build" }), _jsx("div", { className: "mt-2 text-lg font-semibold", children: manifest.dashboard_version }), _jsx("div", { className: "mt-2 text-sm text-slate-600", children: new Date(methodsPayload.generated_at).toLocaleString() })] }), _jsxs("div", { className: "rounded-2xl border border-rule p-4", children: [_jsx("div", { className: "text-xs uppercase tracking-wide text-slate-500", children: "Coverage" }), _jsxs("div", { className: "mt-2 text-lg font-semibold", children: [manifest.coverage.municipalities, " municipalities"] }), _jsxs("div", { className: "mt-2 text-sm text-slate-600", children: [manifest.coverage.states, " states"] })] }), _jsxs("div", { className: "rounded-2xl border border-rule p-4", children: [_jsx("div", { className: "text-xs uppercase tracking-wide text-slate-500", children: "Years" }), _jsx("div", { className: "mt-2 text-lg font-semibold", children: methodsPayload.technical_overview.years_covered.join(", ") })] }), _jsxs("div", { className: "rounded-2xl border border-rule p-4", children: [_jsx("div", { className: "text-xs uppercase tracking-wide text-slate-500", children: "Public metrics" }), _jsx("div", { className: "mt-2 text-lg font-semibold", children: methodsPayload.technical_overview.metric_count })] }), _jsxs("div", { className: "rounded-2xl border border-rule p-4", children: [_jsx("div", { className: "text-xs uppercase tracking-wide text-slate-500", children: "Context stratifiers" }), _jsx("div", { className: "mt-2 text-lg font-semibold", children: methodsPayload.technical_overview.stratifier_count })] }), _jsxs("div", { className: "rounded-2xl border border-rule p-4", children: [_jsx("div", { className: "text-xs uppercase tracking-wide text-slate-500", children: "Source systems" }), _jsx("div", { className: "mt-2 text-lg font-semibold", children: methodsPayload.technical_overview.source_system_count })] })] }), _jsxs("div", { className: "border-t border-rule px-6 py-4 text-sm text-slate-600", children: [_jsx("span", { className: "font-medium text-navy", children: "Frozen dashboard baseline:" }), " ", DASHBOARD_FROZEN_BASELINE_VERSION, " |", " ", DASHBOARD_FROZEN_BASELINE_DATE, ". ", DASHBOARD_FROZEN_BASELINE_NOTE] })] }), _jsx(MedallionStrategyPanel, { layers: methodsPayload.medallion_strategy.layers }), _jsx(SourceSystemsPanel, { sourceSystems: methodsPayload.source_systems }), _jsx(TechnicalSheetCatalog, { indicatorSheets: methodsPayload.indicator_sheets, stratifierSheets: methodsPayload.stratifier_sheets, activeSheetId: activeSheetId, searchTerm: searchTerm, onSearchTermChange: setSearchTerm, typeFilter: typeFilter, onTypeFilterChange: setTypeFilter, familyFilter: familyFilter, onFamilyFilterChange: setFamilyFilter, onSelectSheet: setSheet }), _jsx(DisplayRulesPanel, { rules: methodsPayload.display_rules }), _jsx(TechnicalSheetDrawer, { open: Boolean(activeSheet), sheet: activeSheet, sourceSystemMap: sourceSystemMap, pageTitleMap: pageTitleMap, onClose: () => setSheet(undefined) })] }));
}
