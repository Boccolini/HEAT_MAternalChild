import type { MedallionLayer } from "../lib/api";

export function MedallionStrategyPanel({ layers }: { layers: MedallionLayer[] }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3 className="text-base font-semibold">Data architecture</h3>
      </div>
      <div className="panel-body grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        {layers.map((layer) => (
          <div key={layer.layer_id} className="rounded-2xl border border-rule p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">{layer.label}</div>
            <p className="mt-3 text-sm text-slate-700">{layer.description}</p>
            <p className="mt-3 text-[11px] text-slate-500">
              Technical reference: <span className="font-mono">{layer.layer_id}</span>
            </p>
            <p className="mt-2 text-[11px] text-slate-500">{layer.artifact_type}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
