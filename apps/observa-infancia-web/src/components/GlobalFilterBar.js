import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { formatContextOption } from "../lib/dashboardDisplay";
function handleMultiSelect(event) {
    return Array.from(event.target.selectedOptions).map((option) => option.value);
}
const COMBINED_YEARS_VALUE = "__combined_years__";
function combinedYearsLabel(years) {
    if (!years || years.length === 0) {
        return "Combined";
    }
    const sorted = [...years].sort((left, right) => left - right);
    if (sorted.length === 1) {
        return `Combined ${sorted[0]}`;
    }
    return `Combined ${sorted[0]}-${String(sorted.at(-1)).slice(-2)}`;
}
export function GlobalFilterBar({ filters, filtersMeta, metricAllowsRate, onChange, viewMode, onViewModeChange, includeRegionOption = false, }) {
    const contextDefinition = filtersMeta?.context_variables.find((item) => item.context_variable === filters.context_variable);
    const allYearsSelected = Boolean(filtersMeta?.years.length) &&
        filtersMeta?.years.every((year) => filters.years.includes(year));
    const yearSelectValue = allYearsSelected
        ? [COMBINED_YEARS_VALUE, ...filters.years.map(String)]
        : filters.years.map(String);
    return (_jsx("div", { className: "panel", children: _jsxs("div", { className: "panel-body grid gap-4 md:grid-cols-2 xl:grid-cols-6", children: [_jsxs("label", { className: "flex flex-col gap-2 text-sm", children: [_jsx("span", { className: "font-medium", children: "Years" }), _jsxs("select", { multiple: true, value: yearSelectValue, onChange: (event) => {
                                const selected = handleMultiSelect(event);
                                if (selected.includes(COMBINED_YEARS_VALUE) && filtersMeta?.years.length) {
                                    onChange("years", filtersMeta.years);
                                    return;
                                }
                                onChange("years", selected
                                    .filter((value) => value !== COMBINED_YEARS_VALUE)
                                    .map(Number));
                            }, className: "min-h-28 rounded-xl border border-rule bg-white px-3 py-2", children: [filtersMeta?.years.length ? (_jsx("option", { value: COMBINED_YEARS_VALUE, children: combinedYearsLabel(filtersMeta.years) })) : null, filtersMeta?.years.map((year) => (_jsx("option", { value: year, children: year }, year)))] })] }), _jsxs("label", { className: "flex flex-col gap-2 text-sm", children: [_jsx("span", { className: "font-medium", children: "Granularity" }), _jsx("select", { value: filters.granularity, onChange: (event) => onChange("granularity", event.target.value), className: "rounded-xl border border-rule bg-white px-3 py-2", children: filtersMeta?.granularities
                                .filter((item) => item !== "week")
                                .map((granularity) => (_jsx("option", { value: granularity, children: granularity }, granularity))) })] }), _jsxs("label", { className: "flex flex-col gap-2 text-sm", children: [_jsx("span", { className: "font-medium", children: viewMode ? "View mode" : "Display geography" }), viewMode && onViewModeChange ? (_jsxs("select", { value: viewMode, onChange: (event) => onViewModeChange(event.target.value), className: "rounded-xl border border-rule bg-white px-3 py-2", children: [_jsx("option", { value: "national", children: "National" }), _jsx("option", { value: "regional", children: "Regional comparison" })] })) : (_jsxs("select", { value: filters.geo_level, onChange: (event) => onChange("geoLevel", event.target.value), className: "rounded-xl border border-rule bg-white px-3 py-2", children: [includeRegionOption ? _jsx("option", { value: "region", children: "Region" }) : null, _jsx("option", { value: "state", children: "State" }), _jsx("option", { value: "municipality", children: "Municipality" }), _jsx("option", { value: "national", children: "National" })] }))] }), _jsxs("label", { className: "flex flex-col gap-2 text-sm", children: [_jsx("span", { className: "font-medium", children: "State filter" }), _jsx("select", { multiple: true, value: filters.geo_codes.filter((code) => code.length === 2), onChange: (event) => onChange("geoCodes", handleMultiSelect(event)), className: "min-h-28 rounded-xl border border-rule bg-white px-3 py-2", children: filtersMeta?.states.map((state) => (_jsx("option", { value: state.geo_code, children: state.label }, state.geo_code))) })] }), _jsxs("label", { className: "flex flex-col gap-2 text-sm", children: [_jsx("span", { className: "font-medium", children: "Value mode" }), _jsxs("select", { value: filters.value_mode, onChange: (event) => onChange("valueMode", event.target.value), className: "rounded-xl border border-rule bg-white px-3 py-2", disabled: !metricAllowsRate, children: [_jsx("option", { value: "count", children: "Value" }), _jsx("option", { value: "rate", children: "Rate" })] })] }), _jsxs("label", { className: "flex flex-col gap-2 text-sm", children: [_jsx("span", { className: "font-medium", children: "Context variable" }), _jsxs("select", { value: filters.context_variable ?? "", onChange: (event) => onChange("contextVariable", event.target.value || undefined), className: "rounded-xl border border-rule bg-white px-3 py-2", children: [_jsx("option", { value: "", children: "None" }), filtersMeta?.context_variables.map((variable) => (_jsx("option", { value: variable.context_variable, children: variable.label }, variable.context_variable)))] })] }), contextDefinition && (_jsxs("label", { className: "flex flex-col gap-2 text-sm md:col-span-2 xl:col-span-6", children: [_jsx("span", { className: "font-medium", children: "Context bins" }), _jsx("select", { multiple: true, value: filters.context_bins, onChange: (event) => onChange("contextBins", handleMultiSelect(event)), className: "min-h-28 rounded-xl border border-rule bg-white px-3 py-2", children: contextDefinition.order.map((option) => (_jsx("option", { value: option, children: formatContextOption(contextDefinition, option) }, option))) })] }))] }) }));
}
