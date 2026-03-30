import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { buildDownloadFilename, downloadCsv } from "../lib/download";
async function getOutcomeMatrix() {
    const response = await fetch("/downloads/36b_full_outcome_progression_matrix.json");
    if (!response.ok) {
        throw new Error("Unable to load outcome progression matrix");
    }
    return response.json();
}
export function OutcomeMatrixPage() {
    const [searchTerm, setSearchTerm] = useState("");
    const [tierFilter, setTierFilter] = useState("all");
    const { data, error, isLoading } = useQuery({
        queryKey: ["outcome-matrix"],
        queryFn: getOutcomeMatrix
    });
    const visibleRows = useMemo(() => {
        const rows = data ?? [];
        const normalized = searchTerm.trim().toLowerCase();
        return rows.filter((row) => {
            const tierMatch = tierFilter === "all" || row["Final tier"] === tierFilter;
            const searchMatch = normalized.length === 0 ||
                [row.Outcome, row.analysis_unit_id, row.Domain, row["Final tier"], row["Decision note"]]
                    .join(" ")
                    .toLowerCase()
                    .includes(normalized);
            return tierMatch && searchMatch;
        });
    }, [data, searchTerm, tierFilter]);
    if (isLoading) {
        return _jsx("div", { className: "text-sm text-slate-500", children: "Loading outcome progression matrix..." });
    }
    if (error || !data) {
        return _jsx("div", { className: "text-sm text-caution", children: "Unable to load the full outcome progression matrix." });
    }
    function handleDownloadCsv() {
        downloadCsv(visibleRows, buildDownloadFilename("full-outcome-progression-matrix", "csv"), {
            title: "Full outcome progression matrix",
            notes: [
                "32 screened outcomes",
                "Frozen dashboard baseline release",
            ]
        });
    }
    return (_jsxs("div", { className: "space-y-6", children: [_jsxs("section", { className: "space-y-2", children: [_jsx("h2", { className: "text-2xl font-semibold", children: "Outcome progression matrix" }), _jsx("p", { className: "max-w-5xl text-sm text-slate-600", children: "This table shows how all 32 screened outcomes moved through the workflow, from feasibility screening to the final hierarchy. The final tier reflects the post-validation interpretation used in the current paper package." })] }), _jsxs("section", { className: "panel", children: [_jsx("div", { className: "panel-header", children: _jsx("h3", { className: "text-base font-semibold", children: "Filters" }) }), _jsxs("div", { className: "panel-body grid gap-4 md:grid-cols-[minmax(0,1fr)_220px]", children: [_jsxs("label", { className: "flex flex-col gap-2 text-sm", children: [_jsx("span", { className: "font-medium", children: "Search outcome or note" }), _jsx("input", { value: searchTerm, onChange: (event) => setSearchTerm(event.target.value), className: "rounded-xl border border-rule bg-white px-3 py-2", placeholder: "e.g. neonatal, maternal, provisional" })] }), _jsxs("label", { className: "flex flex-col gap-2 text-sm", children: [_jsx("span", { className: "font-medium", children: "Final tier" }), _jsxs("select", { value: tierFilter, onChange: (event) => setTierFilter(event.target.value), className: "rounded-xl border border-rule bg-white px-3 py-2", children: [_jsx("option", { value: "all", children: "All outcomes" }), _jsx("option", { value: "Principal", children: "Principal" }), _jsx("option", { value: "Supportive", children: "Supportive" }), _jsx("option", { value: "Provisional", children: "Provisional" }), _jsx("option", { value: "Not in final retained set", children: "Not in final retained set" })] })] })] })] }), _jsxs("section", { className: "panel", children: [_jsxs("div", { className: "panel-header", children: [_jsxs("div", { children: [_jsx("h3", { className: "text-base font-semibold", children: "Full screened roster" }), _jsxs("p", { className: "text-sm text-slate-600", children: [visibleRows.length, " outcomes shown"] })] }), _jsx("button", { type: "button", onClick: handleDownloadCsv, className: "rounded-full border border-rule bg-white px-3 py-1.5 text-sm font-medium text-navy transition-colors hover:border-blue hover:text-blue", children: "Download CSV" })] }), _jsx("div", { className: "panel-body overflow-auto", children: _jsxs("table", { className: "min-w-[1600px] text-sm", children: [_jsx("thead", { className: "sticky top-0 z-10 text-left text-slate-500", children: _jsx("tr", { children: [
                                            "Outcome",
                                            "Domain",
                                            "Feasibility screen",
                                            "Primary analysis",
                                            "Selected exposure/window",
                                            "Context screen",
                                            "Publication stage",
                                            "Step7 public use",
                                            "Temporal validation",
                                            "DLNM",
                                            "Case-crossover",
                                            "Final tier",
                                            "Decision note"
                                        ].map((header) => (_jsx("th", { className: "border-b border-rule bg-slate-50 px-3 py-3 align-top font-medium shadow-sm", children: header }, header))) }) }), _jsx("tbody", { children: visibleRows.map((row) => (_jsxs("tr", { className: "border-t border-rule align-top", children: [_jsx("td", { className: "px-3 py-3 font-medium text-navy", children: row.Outcome }), _jsx("td", { className: "px-3 py-3", children: row.Domain }), _jsx("td", { className: "px-3 py-3", children: row["Feasibility screen"] }), _jsx("td", { className: "px-3 py-3", children: row["Primary analysis"] }), _jsx("td", { className: "px-3 py-3", children: row["Selected exposure/window"] }), _jsx("td", { className: "px-3 py-3", children: row["Context screen"] }), _jsx("td", { className: "px-3 py-3", children: row["Publication stage"] }), _jsx("td", { className: "px-3 py-3", children: row["Step7 public use"] }), _jsx("td", { className: "px-3 py-3", children: row["Temporal validation"] }), _jsx("td", { className: "px-3 py-3", children: row.DLNM }), _jsx("td", { className: "px-3 py-3", children: row["Case-crossover"] }), _jsx("td", { className: "px-3 py-3", children: row["Final tier"] }), _jsx("td", { className: "px-3 py-3 text-slate-600", children: row["Decision note"] })] }, row.analysis_unit_id))) })] }) })] })] }));
}
