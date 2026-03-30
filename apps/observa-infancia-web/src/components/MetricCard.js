import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export function MetricCard({ card, active, onClick }) {
    return (_jsxs("button", { type: "button", onClick: onClick, className: [
            "rounded-2xl border p-4 text-left transition-colors",
            active ? "border-navy bg-navy text-white" : "border-rule bg-white hover:border-blue"
        ].join(" "), children: [_jsx("div", { className: "text-xs uppercase tracking-wide opacity-75", children: card.label }), _jsx("div", { className: "mt-3 text-3xl font-semibold", children: card.value === null ? "NA" : card.value.toLocaleString(undefined, { maximumFractionDigits: 2 }) }), _jsx("div", { className: "mt-1 text-sm opacity-80", children: card.unit }), card.period_label ? (_jsxs("div", { className: "mt-1 text-xs opacity-70", children: ["Reference period: ", card.period_label] })) : null, _jsx("div", { className: "mt-3 text-xs opacity-80", children: card.delta_vs_prior_period === null
                    ? "No prior comparison"
                    : `${card.delta_vs_prior_period >= 0 ? "+" : ""}${card.delta_vs_prior_period.toFixed(2)} vs prior period` })] }));
}
