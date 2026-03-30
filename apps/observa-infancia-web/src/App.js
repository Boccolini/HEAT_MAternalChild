import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Navigate, Route, Routes } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getMetricsMeta } from "./lib/api";
import { AppShell } from "./components/AppShell";
import { AnalyticalPage } from "./pages/AnalyticalPage";
import { ClimateHealthRelationshipsPage } from "./pages/ClimateHealthRelationshipsPage";
import { HowToUsePage } from "./pages/HowToUsePage";
import { MethodsPage } from "./pages/MethodsPage";
import { OutcomeMatrixPage } from "./pages/OutcomeMatrixPage";
import { VersionLedgerPage } from "./pages/VersionLedgerPage";
function App() {
    const { data, error, isLoading } = useQuery({ queryKey: ["metrics-meta"], queryFn: getMetricsMeta });
    if (isLoading) {
        return _jsx("div", { className: "p-6 text-sm text-slate-500", children: "Loading application metadata..." });
    }
    if (error || !data) {
        return (_jsx("div", { className: "p-6 text-sm text-caution", children: "Unable to load application metadata. Check that the dashboard API is running on port 8000." }));
    }
    return (_jsx(Routes, { children: _jsxs(Route, { path: "/", element: _jsx(AppShell, { pages: data.pages }), children: [_jsx(Route, { index: true, element: _jsx(Navigate, { to: "/overview", replace: true }) }), _jsx(Route, { path: "overview", element: _jsx(AnalyticalPage, { pageId: "overview" }) }), _jsx(Route, { path: "perinatal", element: _jsx(AnalyticalPage, { pageId: "perinatal" }) }), _jsx(Route, { path: "child-mortality", element: _jsx(AnalyticalPage, { pageId: "child_mortality" }) }), _jsx(Route, { path: "maternal", element: _jsx(AnalyticalPage, { pageId: "maternal" }) }), _jsx(Route, { path: "climate", element: _jsx(AnalyticalPage, { pageId: "climate" }) }), _jsx(Route, { path: "context", element: _jsx(AnalyticalPage, { pageId: "context" }) }), _jsx(Route, { path: "relationships", element: _jsx(ClimateHealthRelationshipsPage, {}) }), _jsx(Route, { path: "how-to-use", element: _jsx(HowToUsePage, {}) }), _jsx(Route, { path: "outcome-matrix", element: _jsx(OutcomeMatrixPage, {}) }), _jsx(Route, { path: "version-ledger", element: _jsx(VersionLedgerPage, {}) }), _jsx(Route, { path: "methods", element: _jsx(MethodsPage, {}) })] }) }));
}
export default App;
