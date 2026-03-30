import type { DisplayRule } from "../lib/api";

export function DisplayRulesPanel({ rules }: { rules: DisplayRule[] }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3 className="text-base font-semibold">Display and quality rules</h3>
      </div>
      <div className="panel-body grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {rules.map((rule) => (
          <div key={rule.rule_id} className="rounded-2xl border border-rule p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">{rule.title}</div>
            <p className="mt-3 text-sm text-slate-700">{rule.description}</p>
            <p className="mt-3 text-[11px] text-slate-500">
              Technical reference: <span className="font-mono">{rule.rule_id}</span>
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}
