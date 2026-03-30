import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { DisplayRulesPanel } from "../components/DisplayRulesPanel";
import { MedallionStrategyPanel } from "../components/MedallionStrategyPanel";
import { SourceSystemsPanel } from "../components/SourceSystemsPanel";
import { TechnicalSheetCatalog } from "../components/TechnicalSheetCatalog";
import { TechnicalSheetDrawer } from "../components/TechnicalSheetDrawer";
import {
  getManifest,
  getMethodsMeta,
  type IndicatorTechnicalSheet,
  type StratifierTechnicalSheet,
} from "../lib/api";
import {
  DASHBOARD_FROZEN_BASELINE_DATE,
  DASHBOARD_FROZEN_BASELINE_NOTE,
  DASHBOARD_FROZEN_BASELINE_VERSION
} from "../lib/dashboardRelease";

export function MethodsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [searchTerm, setSearchTerm] = useState("");
  const [typeFilter, setTypeFilter] = useState<"all" | "indicators" | "stratifiers">("all");
  const [familyFilter, setFamilyFilter] = useState("all");
  const { data: methodsPayload, error } = useQuery({ queryKey: ["methods-meta"], queryFn: getMethodsMeta });
  const { data: manifest } = useQuery({ queryKey: ["manifest"], queryFn: getManifest });

  const pageTitleMap = useMemo(
    () =>
      Object.fromEntries((manifest?.pages ?? []).map((page) => [page.page_id, page.title])),
    [manifest?.pages],
  );
  const sourceSystemMap = useMemo(
    () =>
      Object.fromEntries((methodsPayload?.source_systems ?? []).map((sourceSystem) => [sourceSystem.source_system_id, sourceSystem])),
    [methodsPayload?.source_systems],
  );
  const activeSheetId = searchParams.get("sheet");
  const activeSheet = useMemo<IndicatorTechnicalSheet | StratifierTechnicalSheet | null>(() => {
    if (!activeSheetId || !methodsPayload) {
      return null;
    }
    return (
      methodsPayload.indicator_sheets.find((sheet) => sheet.sheet_id === activeSheetId) ??
      methodsPayload.stratifier_sheets.find((sheet) => sheet.sheet_id === activeSheetId) ??
      null
    );
  }, [activeSheetId, methodsPayload]);

  function setSheet(sheetId?: string) {
    const next = new URLSearchParams(searchParams);
    if (sheetId) {
      next.set("sheet", sheetId);
    } else {
      next.delete("sheet");
    }
    setSearchParams(next, { replace: true });
  }

  if (error) {
    return <div className="text-sm text-caution">Unable to load methods metadata from the dashboard API.</div>;
  }

  if (!methodsPayload || !manifest) {
    return <div className="text-sm text-slate-500">Loading technical methods metadata...</div>;
  }

  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <h2 className="text-2xl font-semibold">Methods and Technical Specifications</h2>
        <p className="max-w-4xl text-sm text-slate-600">
          This page documents KPI construction, contextual stratifier construction, source provenance, medallion lineage,
          and dashboard display rules for the internal beta build.
        </p>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3 className="text-base font-semibold">Method boundary for this release</h3>
        </div>
        <div className="panel-body space-y-3 text-sm text-slate-700">
          <p>The dashboard is a descriptive and surveillance-facing interface built on the frozen workflow used in the current study package.</p>
          <p>Its role is to display retained indicators, contextual patterns, medallion lineage, and page-level exploration rules. It is not a standalone causal engine.</p>
          <p>Principal, supportive, and provisional signals should be read according to the frozen validation hierarchy in this release baseline.</p>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3 className="text-base font-semibold">Citation and DOI status</h3>
        </div>
        <div className="panel-body space-y-3 text-sm text-slate-700">
          <p>All generated downloads should carry the dashboard citation metadata for sharing and reuse.</p>
          <p>Suggested citation: Boccolini, CS. Brazil&apos;s heat exposure on maternal, newborn, and child health - Dashboard.</p>
          <p>DOI status: pending release archival. The exact DOI should be attached after the frozen dashboard release is archived.</p>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3 className="text-base font-semibold">Technical overview</h3>
        </div>
        <div className="panel-body grid gap-4 md:grid-cols-2 xl:grid-cols-6">
          <div className="rounded-2xl border border-rule p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">Build</div>
            <div className="mt-2 text-lg font-semibold">{manifest.dashboard_version}</div>
            <div className="mt-2 text-sm text-slate-600">{new Date(methodsPayload.generated_at).toLocaleString()}</div>
          </div>
          <div className="rounded-2xl border border-rule p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">Coverage</div>
            <div className="mt-2 text-lg font-semibold">{manifest.coverage.municipalities} municipalities</div>
            <div className="mt-2 text-sm text-slate-600">{manifest.coverage.states} states</div>
          </div>
          <div className="rounded-2xl border border-rule p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">Years</div>
            <div className="mt-2 text-lg font-semibold">{methodsPayload.technical_overview.years_covered.join(", ")}</div>
          </div>
          <div className="rounded-2xl border border-rule p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">Public metrics</div>
            <div className="mt-2 text-lg font-semibold">{methodsPayload.technical_overview.metric_count}</div>
          </div>
          <div className="rounded-2xl border border-rule p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">Context stratifiers</div>
            <div className="mt-2 text-lg font-semibold">{methodsPayload.technical_overview.stratifier_count}</div>
          </div>
          <div className="rounded-2xl border border-rule p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">Source systems</div>
            <div className="mt-2 text-lg font-semibold">{methodsPayload.technical_overview.source_system_count}</div>
          </div>
        </div>
        <div className="border-t border-rule px-6 py-4 text-sm text-slate-600">
          <span className="font-medium text-navy">Frozen dashboard baseline:</span> {DASHBOARD_FROZEN_BASELINE_VERSION} |{" "}
          {DASHBOARD_FROZEN_BASELINE_DATE}. {DASHBOARD_FROZEN_BASELINE_NOTE}
        </div>
      </section>

      <MedallionStrategyPanel layers={methodsPayload.medallion_strategy.layers} />
      <SourceSystemsPanel sourceSystems={methodsPayload.source_systems} />

      <TechnicalSheetCatalog
        indicatorSheets={methodsPayload.indicator_sheets}
        stratifierSheets={methodsPayload.stratifier_sheets}
        activeSheetId={activeSheetId}
        searchTerm={searchTerm}
        onSearchTermChange={setSearchTerm}
        typeFilter={typeFilter}
        onTypeFilterChange={setTypeFilter}
        familyFilter={familyFilter}
        onFamilyFilterChange={setFamilyFilter}
        onSelectSheet={setSheet}
      />

      <DisplayRulesPanel rules={methodsPayload.display_rules} />

      <TechnicalSheetDrawer
        open={Boolean(activeSheet)}
        sheet={activeSheet}
        sourceSystemMap={sourceSystemMap}
        pageTitleMap={pageTitleMap}
        onClose={() => setSheet(undefined)}
      />
    </div>
  );
}
