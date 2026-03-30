import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Outlet } from "react-router-dom";
import { DASHBOARD_FROZEN_BASELINE_DATE, DASHBOARD_FROZEN_BASELINE_VERSION } from "../lib/dashboardRelease";
import { TopNav } from "./TopNav";
export function AppShell({ pages }) {
    return (_jsxs("div", { className: "min-h-screen bg-canvas", children: [_jsx("header", { className: "border-b border-rule bg-white", children: _jsxs("div", { className: "mx-auto flex max-w-[1600px] flex-col gap-4 px-6 py-5", children: [_jsxs("div", { children: [_jsx("p", { className: "text-xs uppercase tracking-[0.24em] text-blue", children: "ICICT / Fiocruz" }), _jsx("h1", { className: "mt-2 text-2xl font-semibold text-navy", children: "Child Health Observatory (Observa Infancia)" }), _jsx("p", { className: "mt-1 max-w-3xl text-sm text-slate-600", children: "Brazil's heat exposure on maternal, newborn, and child health." }), _jsxs("p", { className: "mt-2 text-xs text-slate-500", children: ["Frozen baseline ", DASHBOARD_FROZEN_BASELINE_VERSION, " | ", DASHBOARD_FROZEN_BASELINE_DATE] })] }), _jsx(TopNav, { pages: pages })] }) }), _jsx("main", { className: "mx-auto max-w-[1600px] px-6 py-6", children: _jsx(Outlet, {}) })] }));
}
