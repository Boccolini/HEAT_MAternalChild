import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
function matchesSearch(sheet, searchTerm) {
    if (!searchTerm.trim()) {
        return true;
    }
    const needle = searchTerm.trim().toLowerCase();
    const haystack = [
        sheet.label,
        sheet.sheet_id,
        sheet.public_definition,
        sheet.sheet_type === "indicator" ? sheet.metric_family : "context",
    ]
        .join(" ")
        .toLowerCase();
    return haystack.includes(needle);
}
function ChipButton({ sheet, active, onClick, }) {
    const usage = sheet.sheet_type === "indicator"
        ? sheet.page_usage.slice(0, 3).map((item) => item.replace("_", " "))
        : sheet.applicable_pages.slice(0, 3).map((item) => item.replace("_", " "));
    return (_jsxs("button", { type: "button", className: [
            "rounded-2xl border px-4 py-3 text-left transition-colors",
            active ? "border-navy bg-navy text-white" : "border-rule bg-white hover:border-blue hover:text-blue",
        ].join(" "), onClick: onClick, children: [_jsx("div", { className: "font-medium", children: sheet.label }), _jsx("div", { className: active ? "mt-1 text-[11px] opacity-80" : "mt-1 text-[11px] text-slate-500", children: sheet.sheet_id }), _jsx("div", { className: "mt-2 flex flex-wrap gap-1", children: usage.map((item) => (_jsx("span", { className: active ? "rounded-full border border-white/30 px-2 py-1 text-[11px]" : "rounded-full border border-rule px-2 py-1 text-[11px] text-slate-600", children: item }, `${sheet.sheet_id}-${item}`))) })] }));
}
export function TechnicalSheetCatalog({ indicatorSheets, stratifierSheets, activeSheetId, searchTerm, onSearchTermChange, typeFilter, onTypeFilterChange, familyFilter, onFamilyFilterChange, onSelectSheet, }) {
    const familyOptions = ["all", "overview", "perinatal", "mortality", "child_morbidity", "maternal", "climate", "context"];
    const filteredIndicators = indicatorSheets.filter((sheet) => {
        if (typeFilter === "stratifiers") {
            return false;
        }
        if (familyFilter !== "all" && familyFilter !== "overview" && sheet.metric_family !== familyFilter) {
            return false;
        }
        if (familyFilter === "overview" && !(sheet.page_id === "overview" || sheet.page_usage.includes("overview"))) {
            return false;
        }
        return matchesSearch(sheet, searchTerm);
    });
    const filteredStratifiers = stratifierSheets.filter((sheet) => {
        if (typeFilter === "indicators") {
            return false;
        }
        if (familyFilter !== "all" && familyFilter !== "context") {
            return false;
        }
        return matchesSearch(sheet, searchTerm);
    });
    const sections = [
        {
            title: "Overview and headline KPIs",
            items: filteredIndicators.filter((sheet) => sheet.page_id === "overview" || sheet.page_usage.includes("overview")),
        },
        {
            title: "Perinatal indicators",
            items: filteredIndicators.filter((sheet) => sheet.metric_family === "perinatal" && !sheet.page_usage.includes("overview")),
        },
        {
            title: "Child mortality indicators",
            items: filteredIndicators.filter((sheet) => sheet.metric_family === "mortality"),
        },
        {
            title: "Child morbidity and admissions",
            items: filteredIndicators.filter((sheet) => sheet.metric_family === "child_morbidity"),
        },
        {
            title: "Maternal morbidity indicators",
            items: filteredIndicators.filter((sheet) => sheet.metric_family === "maternal"),
        },
        {
            title: "Climate indicators",
            items: filteredIndicators.filter((sheet) => sheet.metric_family === "climate"),
        },
        {
            title: "Context stratifiers",
            items: filteredStratifiers,
        },
    ].filter((section) => section.items.length > 0);
    return (_jsxs("section", { className: "panel", children: [_jsx("div", { className: "panel-header", children: _jsx("h3", { className: "text-base font-semibold", children: "Technical sheet catalog" }) }), _jsxs("div", { className: "panel-body space-y-6", children: [_jsxs("div", { className: "grid gap-4 md:grid-cols-3", children: [_jsxs("label", { className: "space-y-2 text-sm", children: [_jsx("span", { className: "text-xs uppercase tracking-wide text-slate-500", children: "Search" }), _jsx("input", { value: searchTerm, onChange: (event) => onSearchTermChange(event.target.value), placeholder: "Search indicator, field, or concept", className: "w-full rounded-xl border border-rule bg-white px-3 py-2" })] }), _jsxs("label", { className: "space-y-2 text-sm", children: [_jsx("span", { className: "text-xs uppercase tracking-wide text-slate-500", children: "Type" }), _jsxs("select", { value: typeFilter, onChange: (event) => onTypeFilterChange(event.target.value), className: "w-full rounded-xl border border-rule bg-white px-3 py-2", children: [_jsx("option", { value: "all", children: "All" }), _jsx("option", { value: "indicators", children: "Indicators" }), _jsx("option", { value: "stratifiers", children: "Stratifiers" })] })] }), _jsxs("label", { className: "space-y-2 text-sm", children: [_jsx("span", { className: "text-xs uppercase tracking-wide text-slate-500", children: "Family" }), _jsx("select", { value: familyFilter, onChange: (event) => onFamilyFilterChange(event.target.value), className: "w-full rounded-xl border border-rule bg-white px-3 py-2", children: familyOptions.map((option) => (_jsx("option", { value: option, children: option === "all" ? "All families" : option.replace("_", " ") }, option))) })] })] }), _jsxs("div", { className: "space-y-6", children: [sections.map((section) => (_jsxs("div", { className: "space-y-3", children: [_jsx("h4", { className: "text-sm font-semibold uppercase tracking-wide text-slate-500", children: section.title }), _jsx("div", { className: "grid gap-3 md:grid-cols-2 xl:grid-cols-3", children: section.items.map((sheet) => (_jsx(ChipButton, { sheet: sheet, active: sheet.sheet_id === activeSheetId, onClick: () => onSelectSheet(sheet.sheet_id) }, sheet.sheet_id))) })] }, section.title))), sections.length === 0 ? (_jsx("div", { className: "rounded-2xl border border-rule p-4 text-sm text-slate-600", children: "No technical sheets match the current search and filter combination." })) : null] })] })] }));
}
