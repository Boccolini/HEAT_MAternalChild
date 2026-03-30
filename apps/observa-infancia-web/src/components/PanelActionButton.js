import { jsx as _jsx } from "react/jsx-runtime";
export function PanelActionButton({ label, onClick }) {
    return (_jsx("button", { type: "button", onClick: onClick, className: "rounded-full border border-rule px-3 py-1 text-xs font-medium tracking-wide text-blue transition hover:border-blue hover:bg-slate-50", children: label }));
}
