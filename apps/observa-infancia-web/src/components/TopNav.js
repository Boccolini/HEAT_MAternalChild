import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Link, NavLink } from "react-router-dom";
export function TopNav({ pages }) {
    return (_jsxs("nav", { className: "flex flex-wrap gap-2", children: [pages
                .filter((page) => page.route !== "/methods")
                .map((page) => (_jsx(NavLink, { to: page.route, className: ({ isActive }) => [
                    "rounded-full border px-4 py-2 text-sm font-medium transition-colors",
                    isActive
                        ? "border-navy bg-navy text-white"
                        : "border-rule bg-white text-navy hover:border-blue hover:text-blue"
                ].join(" "), children: page.title }, page.page_id))), _jsx(NavLink, { to: "/methods", className: ({ isActive }) => [
                    "rounded-full border px-4 py-2 text-sm font-medium transition-colors",
                    isActive
                        ? "border-blue bg-blue text-white"
                        : "border-rule bg-white text-navy hover:border-blue hover:text-blue"
                ].join(" "), children: "Methods" }), _jsx(Link, { to: "/how-to-use", className: "rounded-full border border-rule bg-white px-4 py-2 text-sm font-medium text-navy transition-colors hover:border-blue hover:text-blue", children: "How to use" }), _jsx(Link, { to: "/outcome-matrix", className: "rounded-full border border-blue bg-blue/5 px-4 py-2 text-sm font-medium text-blue transition-colors hover:bg-blue hover:text-white", children: "32 outcomes matrix" }), _jsx(Link, { to: "/version-ledger", className: "rounded-full border border-rule bg-white px-4 py-2 text-sm font-medium text-navy transition-colors hover:border-blue hover:text-blue", children: "Version ledger" })] }));
}
