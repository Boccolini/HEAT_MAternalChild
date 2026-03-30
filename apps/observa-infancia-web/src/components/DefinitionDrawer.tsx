import type { MetricDefinition } from "../lib/api";

type DefinitionDrawerProps = {
  metric?: MetricDefinition;
  open: boolean;
  onClose: () => void;
};

export function DefinitionDrawer({ metric, open, onClose }: DefinitionDrawerProps) {
  if (!open || !metric) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/20">
      <div className="h-full w-full max-w-xl bg-white p-6 shadow-2xl">
        <div className="mb-6 flex items-start justify-between">
          <div>
            <h2 className="text-xl font-semibold">{metric.label}</h2>
            <p className="text-sm text-slate-500">{metric.metric_id}</p>
          </div>
          <button type="button" className="rounded-full border border-rule px-3 py-1 text-sm" onClick={onClose}>
            Close
          </button>
        </div>
        <dl className="space-y-4 text-sm">
          <div>
            <dt className="font-medium">Definition type</dt>
            <dd>{metric.definition_type}</dd>
          </div>
          <div>
            <dt className="font-medium">Source table</dt>
            <dd>{metric.source_table}</dd>
          </div>
          <div>
            <dt className="font-medium">Primary unit</dt>
            <dd>{metric.display_unit}</dd>
          </div>
          <div>
            <dt className="font-medium">Rate unit</dt>
            <dd>{metric.rate_display_unit ?? "Not applicable"}</dd>
          </div>
          <div>
            <dt className="font-medium">Minimum denominator threshold</dt>
            <dd>
              {metric.minimum_denominator_threshold}
              {metric.denominator_field ? ` (${metric.denominator_field})` : ""}
            </dd>
          </div>
          <div>
            <dt className="font-medium">Quality flags</dt>
            <dd>{metric.quality_flags.join(", ") || "None registered"}</dd>
          </div>
          <div>
            <dt className="font-medium">Display rule</dt>
            <dd className="text-slate-600">
              Rate views are suppressed below the minimum denominator threshold and high visible rates are clipped at the
              upper 99th percentile of the current view.
            </dd>
          </div>
          <div>
            <dt className="font-medium">Analytical note</dt>
            <dd className="text-slate-600">{metric.tooltip_note}</dd>
          </div>
        </dl>
      </div>
    </div>
  );
}
