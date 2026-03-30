import type { SourceSystem } from "../lib/api";

export function SourceSystemsPanel({ sourceSystems }: { sourceSystems: SourceSystem[] }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3 className="text-base font-semibold">Source systems</h3>
      </div>
      <div className="panel-body grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {sourceSystems.map((system) => (
          <div key={system.source_system_id} className="rounded-2xl border border-rule p-4">
            <div className="flex items-center justify-between gap-3">
              <h4 className="text-sm font-semibold">{system.label}</h4>
              <span className="rounded-full border border-rule px-2 py-1 text-[11px] uppercase tracking-wide text-slate-500">
                {system.domain}
              </span>
            </div>
            <p className="mt-3 text-sm text-slate-700">{system.short_description}</p>
            <div className="mt-4 space-y-2 text-[11px] text-slate-500">
              <p>
                Technical reference: <span className="font-mono">{system.source_system_id}</span>
              </p>
              {system.raw_manifest_ids.length > 0 ? (
                <p>
                  Raw manifests: <span className="font-mono">{system.raw_manifest_ids.join(", ")}</span>
                </p>
              ) : null}
              {system.bronze_manifest_ids.length > 0 ? (
                <p>
                  Bronze manifests: <span className="font-mono">{system.bronze_manifest_ids.join(", ")}</span>
                </p>
              ) : null}
              {system.silver_manifest_ids.length > 0 ? (
                <p>
                  Silver manifests: <span className="font-mono">{system.silver_manifest_ids.join(", ")}</span>
                </p>
              ) : null}
            </div>
            {system.notes.length > 0 ? (
              <ul className="mt-4 space-y-2 text-sm text-slate-600">
                {system.notes.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            ) : null}
          </div>
        ))}
      </div>
    </section>
  );
}
