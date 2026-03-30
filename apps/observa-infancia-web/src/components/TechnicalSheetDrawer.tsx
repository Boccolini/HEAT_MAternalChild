import { useEffect, type ReactNode } from "react";
import type {
  IndicatorTechnicalSheet,
  SourceSystem,
  StratifierTechnicalSheet,
  TechnicalComponentVariable,
} from "../lib/api";

type TechnicalSheet = IndicatorTechnicalSheet | StratifierTechnicalSheet;

type TechnicalSheetDrawerProps = {
  sheet?: TechnicalSheet | null;
  sourceSystemMap: Record<string, SourceSystem>;
  pageTitleMap: Record<string, string>;
  open: boolean;
  onClose: () => void;
};

function DrawerSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-3 border-t border-rule pt-5 first:border-t-0 first:pt-0">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">{title}</h3>
      {children}
    </section>
  );
}

function ReferenceLine({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 text-sm font-mono text-slate-700">{value}</div>
    </div>
  );
}

function ComponentVariablesTable({ variables }: { variables: TechnicalComponentVariable[] }) {
  return (
    <div className="overflow-auto rounded-2xl border border-rule">
      <table className="w-full text-sm">
        <thead className="text-left text-slate-500">
          <tr>
            <th className="px-4 py-3">Variable</th>
            <th className="px-4 py-3">Role</th>
            <th className="px-4 py-3">Required for</th>
          </tr>
        </thead>
        <tbody>
          {variables.map((variable) => (
            <tr key={`${variable.field}-${variable.role}`} className="border-t border-rule align-top">
              <td className="px-4 py-3">
                <div className="font-medium">{variable.label}</div>
                <div className="mt-1 font-mono text-[11px] text-slate-500">{variable.field}</div>
              </td>
              <td className="px-4 py-3 text-slate-700">{variable.role}</td>
              <td className="px-4 py-3 text-slate-700">{variable.required_for.join(", ") || "Display"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LineageStack({
  sourceSystems,
  bronzeFamilyIds,
  silverEntityIds,
  goldDatasetId,
  dashboardArtifactId,
}: {
  sourceSystems: SourceSystem[];
  bronzeFamilyIds: string[];
  silverEntityIds: string[];
  goldDatasetId: string;
  dashboardArtifactId: string;
}) {
  const layers = [
    {
      title: "Raw",
      values: sourceSystems.flatMap((system) => system.raw_manifest_ids.length > 0 ? system.raw_manifest_ids : [system.source_system_id]),
    },
    {
      title: "Bronze",
      values: bronzeFamilyIds,
    },
    {
      title: "Silver",
      values: silverEntityIds,
    },
    {
      title: "Gold",
      values: [goldDatasetId],
    },
    {
      title: "Dashboard mart",
      values: [dashboardArtifactId],
    },
  ];

  return (
    <div className="grid gap-3 md:grid-cols-2">
      {layers.map((layer) => (
        <div key={layer.title} className="rounded-2xl border border-rule p-4">
          <div className="text-xs uppercase tracking-wide text-slate-500">{layer.title}</div>
          <div className="mt-3 space-y-2">
            {layer.values.map((value) => (
              <div key={`${layer.title}-${value}`} className="font-mono text-sm text-slate-700">
                {value}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export function TechnicalSheetDrawer({
  sheet,
  sourceSystemMap,
  pageTitleMap,
  open,
  onClose,
}: TechnicalSheetDrawerProps) {
  useEffect(() => {
    if (!open) {
      return undefined;
    }
    function handleKeydown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }
    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  }, [onClose, open]);

  if (!open || !sheet) {
    return null;
  }

  const sourceSystems = sheet.source_system_ids
    .map((sourceSystemId) => sourceSystemMap[sourceSystemId])
    .filter((sourceSystem): sourceSystem is SourceSystem => Boolean(sourceSystem));

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/20" onClick={onClose}>
      <div className="h-full w-full max-w-2xl overflow-auto bg-white p-6 shadow-2xl" onClick={(event) => event.stopPropagation()}>
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.14em] text-blue">
              {sheet.sheet_type === "indicator" ? "Indicator technical sheet" : "Stratifier technical sheet"}
            </p>
            <h2 className="mt-2 text-2xl font-semibold">{sheet.label}</h2>
            <p className="mt-1 font-mono text-sm text-slate-500">{sheet.sheet_id}</p>
          </div>
          <button type="button" className="rounded-full border border-rule px-3 py-1 text-sm" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="space-y-6">
          {sheet.sheet_type === "indicator" ? (
            <>
              <DrawerSection title="Indicator summary">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-2xl border border-rule p-4 text-sm">
                    <p className="text-xs uppercase tracking-wide text-slate-500">Public label</p>
                    <p className="mt-2 font-medium">{sheet.label}</p>
                    <p className="mt-3 text-xs uppercase tracking-wide text-slate-500">Family</p>
                    <p className="mt-2">{sheet.metric_family}</p>
                    <p className="mt-3 text-xs uppercase tracking-wide text-slate-500">Definition type</p>
                    <p className="mt-2">{sheet.definition_type}</p>
                  </div>
                  <div className="rounded-2xl border border-rule p-4 text-sm">
                    <p className="text-xs uppercase tracking-wide text-slate-500">Page usage</p>
                    <p className="mt-2">{sheet.page_usage.map((pageId) => pageTitleMap[pageId] ?? pageId).join(", ")}</p>
                    <p className="mt-3 text-xs uppercase tracking-wide text-slate-500">KPI usage</p>
                    <p className="mt-2">{sheet.used_as_kpi ? "Yes" : "No"}</p>
                    <p className="mt-3 text-xs uppercase tracking-wide text-slate-500">Display units</p>
                    <p className="mt-2">
                      {sheet.display_unit}
                      {sheet.rate_unit ? `; rate unit ${sheet.rate_unit}` : ""}
                    </p>
                  </div>
                </div>
              </DrawerSection>

              <DrawerSection title="Construction logic">
                <p className="text-sm text-slate-700">{sheet.public_definition}</p>
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-2xl border border-rule p-4 text-sm">
                    <p className="text-xs uppercase tracking-wide text-slate-500">Aggregation method</p>
                    <p className="mt-2">{sheet.aggregation_method}</p>
                    <p className="mt-3 text-slate-600">{sheet.construction_logic}</p>
                  </div>
                  <div className="rounded-2xl border border-rule p-4 text-sm">
                    <p className="text-xs uppercase tracking-wide text-slate-500">Allowed geographies</p>
                    <p className="mt-2">{sheet.allowed_geo_levels.join(", ")}</p>
                    <p className="mt-3 text-xs uppercase tracking-wide text-slate-500">Allowed granularities</p>
                    <p className="mt-2">{sheet.allowed_granularities.join(", ")}</p>
                    <p className="mt-3 text-xs uppercase tracking-wide text-slate-500">Display modes</p>
                    <p className="mt-2">{sheet.available_value_modes.join(", ")}</p>
                  </div>
                </div>
              </DrawerSection>

              <DrawerSection title="Numerator">
                <div className="rounded-2xl border border-rule p-4 text-sm">
                  <p className="font-medium">{sheet.numerator.label}</p>
                  <p className="mt-2 text-slate-700">{sheet.numerator.logic}</p>
                  <p className="mt-3 font-mono text-[11px] text-slate-500">{sheet.numerator.fields.join(", ")}</p>
                </div>
              </DrawerSection>

              <DrawerSection title="Denominator">
                <div className="rounded-2xl border border-rule p-4 text-sm">
                  <p className="font-medium">{sheet.denominator?.label ?? "Not applicable"}</p>
                  <p className="mt-2 text-slate-700">{sheet.denominator?.logic ?? "No separate denominator is applied to this dashboard metric."}</p>
                  <p className="mt-3 font-mono text-[11px] text-slate-500">{sheet.denominator?.fields.join(", ") || "No denominator field"}</p>
                  {sheet.denominator?.minimum_threshold != null ? (
                    <p className="mt-3 text-slate-600">Minimum threshold: {sheet.denominator.minimum_threshold}</p>
                  ) : null}
                  {sheet.denominator?.rate_formula ? <p className="mt-2 text-slate-600">Formula: {sheet.denominator.rate_formula}</p> : null}
                </div>
              </DrawerSection>

              <DrawerSection title="Component variables">
                <ComponentVariablesTable variables={sheet.component_variables} />
              </DrawerSection>

              <DrawerSection title="Source provenance">
                <div className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="rounded-2xl border border-rule p-4">
                      <div className="text-xs uppercase tracking-wide text-slate-500">Source systems</div>
                      <div className="mt-3 space-y-3 text-sm">
                        {sourceSystems.map((sourceSystem) => (
                          <div key={sourceSystem.source_system_id}>
                            <p className="font-medium">{sourceSystem.label}</p>
                            <p className="mt-1 text-slate-600">{sourceSystem.short_description}</p>
                            <p className="mt-1 font-mono text-[11px] text-slate-500">{sourceSystem.source_system_id}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="rounded-2xl border border-rule p-4 space-y-4">
                      <ReferenceLine label="Gold dataset" value={sheet.gold_dataset_id} />
                      <ReferenceLine label="Source table" value={sheet.source_table} />
                      <ReferenceLine label="Numerator field" value={sheet.numerator_field} />
                      <ReferenceLine label="Denominator field" value={sheet.denominator_field ?? "None"} />
                      <ReferenceLine label="Quality flags" value={sheet.quality_flags.join(", ") || "None registered"} />
                    </div>
                  </div>
                </div>
              </DrawerSection>

              <DrawerSection title="Medallion lineage">
                <LineageStack
                  sourceSystems={sourceSystems}
                  bronzeFamilyIds={sheet.bronze_family_ids}
                  silverEntityIds={sheet.silver_entity_ids}
                  goldDatasetId={sheet.gold_dataset_id}
                  dashboardArtifactId={sheet.dashboard_artifact_id}
                />
              </DrawerSection>

              <DrawerSection title="Display and quality rules">
                <ul className="space-y-2 text-sm text-slate-700">
                  {sheet.display_rules.map((rule) => (
                    <li key={rule}>{rule}</li>
                  ))}
                </ul>
              </DrawerSection>

              <DrawerSection title="Caveats">
                <ul className="space-y-2 text-sm text-slate-700">
                  {sheet.caveats.map((caveat) => (
                    <li key={caveat}>{caveat}</li>
                  ))}
                </ul>
              </DrawerSection>
            </>
          ) : (
            <>
              <DrawerSection title="Stratifier summary">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-2xl border border-rule p-4 text-sm">
                    <p className="text-xs uppercase tracking-wide text-slate-500">Type</p>
                    <p className="mt-2">{sheet.stratifier_type}</p>
                    <p className="mt-3 text-xs uppercase tracking-wide text-slate-500">Source field</p>
                    <p className="mt-2 font-mono">{sheet.source_field}</p>
                  </div>
                  <div className="rounded-2xl border border-rule p-4 text-sm">
                    <p className="text-xs uppercase tracking-wide text-slate-500">Applicable pages</p>
                    <p className="mt-2">{sheet.applicable_pages.map((pageId) => pageTitleMap[pageId] ?? pageId).join(", ")}</p>
                    <p className="mt-3 text-xs uppercase tracking-wide text-slate-500">Binning</p>
                    <p className="mt-2">{sheet.binning}</p>
                  </div>
                </div>
              </DrawerSection>

              <DrawerSection title="Construction logic">
                <p className="text-sm text-slate-700">{sheet.public_definition}</p>
                <div className="rounded-2xl border border-rule p-4 text-sm">
                  <p>{sheet.binning_logic}</p>
                  <p className="mt-3 text-slate-600">{sheet.expansion_method}</p>
                </div>
              </DrawerSection>

              <DrawerSection title="Source and observation year">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-2xl border border-rule p-4 text-sm">
                    <p className="text-xs uppercase tracking-wide text-slate-500">Observed source years</p>
                    <p className="mt-2">{sheet.observed_source_years.join(", ")}</p>
                    <p className="mt-3 text-xs uppercase tracking-wide text-slate-500">Expanded coverage years</p>
                    <p className="mt-2">{sheet.expanded_years.join(", ")}</p>
                    <p className="mt-3 text-xs uppercase tracking-wide text-slate-500">Join key</p>
                    <p className="mt-2 font-mono">{sheet.join_key}</p>
                  </div>
                  <div className="rounded-2xl border border-rule p-4 text-sm">
                    <p className="text-xs uppercase tracking-wide text-slate-500">Source systems</p>
                    <div className="mt-2 space-y-3">
                      {sourceSystems.map((sourceSystem) => (
                        <div key={sourceSystem.source_system_id}>
                          <p className="font-medium">{sourceSystem.label}</p>
                          <p className="mt-1 text-slate-600">{sourceSystem.short_description}</p>
                          <p className="mt-1 font-mono text-[11px] text-slate-500">{sourceSystem.source_system_id}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </DrawerSection>

              <DrawerSection title="Binning and thresholds">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-2xl border border-rule p-4 text-sm">
                    <p className="text-xs uppercase tracking-wide text-slate-500">Ordered categories</p>
                    <p className="mt-2">{sheet.order.join(", ")}</p>
                    <p className="mt-3 text-xs uppercase tracking-wide text-slate-500">Human-readable labels</p>
                    <p className="mt-2">{sheet.threshold_labels.join(", ")}</p>
                  </div>
                  <div className="rounded-2xl border border-rule p-4 text-sm">
                    <p className="text-xs uppercase tracking-wide text-slate-500">Threshold values</p>
                    <p className="mt-2">{sheet.thresholds.length > 0 ? sheet.thresholds.join(", ") : "Not threshold-driven"}</p>
                  </div>
                </div>
              </DrawerSection>

              <DrawerSection title="Join logic">
                <div className="rounded-2xl border border-rule p-4 text-sm">
                  <p className="font-medium">Dashboard join grain</p>
                  <p className="mt-2 text-slate-700">{sheet.join_key}</p>
                  <p className="mt-3 text-slate-600">
                    The contextual value is linked at municipality-year level and then carried into the dashboard mart for
                    stratification and relationship analysis.
                  </p>
                </div>
              </DrawerSection>

              <DrawerSection title="Component variables">
                <ComponentVariablesTable variables={sheet.component_variables} />
              </DrawerSection>

              <DrawerSection title="Medallion lineage">
                <LineageStack
                  sourceSystems={sourceSystems}
                  bronzeFamilyIds={sheet.bronze_family_ids}
                  silverEntityIds={sheet.silver_entity_ids}
                  goldDatasetId={sheet.gold_dataset_id}
                  dashboardArtifactId={sheet.dashboard_artifact_id}
                />
              </DrawerSection>

              <DrawerSection title="Caveats">
                <ul className="space-y-2 text-sm text-slate-700">
                  {sheet.caveats.map((caveat) => (
                    <li key={caveat}>{caveat}</li>
                  ))}
                </ul>
              </DrawerSection>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
