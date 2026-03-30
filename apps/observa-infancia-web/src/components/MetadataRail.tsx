import type { Manifest, MetricDefinition } from "../lib/api";

type MetadataRailProps = {
  manifest?: Manifest;
  metric?: MetricDefinition;
  pageTitle: string;
  geographyFilterLabel?: string;
  geographyFilterName?: string;
  valueMode?: string;
  contextVariableLabel?: string;
  granularity?: string;
  viewModeLabel?: string;
};

export function MetadataRail({
  manifest,
  metric,
  pageTitle,
  geographyFilterLabel,
  geographyFilterName,
  valueMode,
  contextVariableLabel,
  granularity,
  viewModeLabel,
}: MetadataRailProps) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="text-base font-semibold">Notes and overview</h2>
      </div>
      <div className="panel-body grid gap-4 md:grid-cols-2 xl:grid-cols-3 text-sm">
        <div className="rounded-2xl border border-rule p-4">
          <p className="text-xs uppercase tracking-wide text-slate-500">Coverage</p>
          <p className="mt-3">{manifest?.coverage.municipalities ?? "..."} municipalities</p>
          <p>{manifest?.years_covered.join(", ") ?? "..."}</p>
          <p className="mt-2 text-slate-600">{pageTitle}</p>
        </div>
        <div className="rounded-2xl border border-rule p-4">
          <p className="text-xs uppercase tracking-wide text-slate-500">Last updated</p>
          <p className="mt-3">{manifest?.build_generated_at ? new Date(manifest.build_generated_at).toLocaleString() : "..."}</p>
          <p className="mt-2 text-xs uppercase tracking-wide text-slate-500">Granularity</p>
          <p className="mt-2">{granularity ?? "..."}</p>
        </div>
        <div className="rounded-2xl border border-rule p-4">
          <p className="text-xs uppercase tracking-wide text-slate-500">Indicator</p>
          <p className="mt-3">{metric?.label ?? "Select an indicator"}</p>
          <p className="mt-2 text-xs uppercase tracking-wide text-slate-500">Value mode</p>
          <p className="mt-2">{valueMode ?? "..."}</p>
        </div>
        <div className="rounded-2xl border border-rule p-4">
          <p className="text-xs uppercase tracking-wide text-slate-500">Definition note</p>
          <p className="mt-3 text-slate-600">{metric?.tooltip_note ?? "Indicator metadata will appear here."}</p>
        </div>
        <div className="rounded-2xl border border-rule p-4">
          <p className="text-xs uppercase tracking-wide text-slate-500">Current filters</p>
          {viewModeLabel ? <p className="mt-3 text-slate-700">View mode: {viewModeLabel}</p> : null}
          <p className={viewModeLabel ? "mt-2 text-slate-700" : "mt-3 text-slate-700"}>
            {geographyFilterName ?? "State filter"}: {geographyFilterLabel ?? "All states"}
          </p>
          <p className="mt-2 text-slate-700">Context variable: {contextVariableLabel ?? "None"}</p>
        </div>
        <div className="rounded-2xl border border-rule p-4">
          <p className="text-xs uppercase tracking-wide text-slate-500">Interpretation</p>
          <p className="mt-3 text-slate-600">Results are descriptive and should not be interpreted as causal estimates.</p>
        </div>
      </div>
    </section>
  );
}
