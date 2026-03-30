import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import {
  getFiltersMeta,
  getManifest,
  getMetricsMeta,
  postRelationship,
  type DashboardFilters,
  type MetricDefinition,
  type PageDefinition,
  type RelationshipFilters,
  type RelationshipPoint,
} from "../lib/api";
import { clipRelationshipRates, formatContextValue } from "../lib/dashboardDisplay";
import {
  computeLinearRegression,
  computeLinearRegressionByGroup,
  computeLinearRegressionByStratum,
  formatPValue,
  type StratifiedLinearRegressionSummary,
} from "../lib/relationshipStats";
import { useDashboardFilters } from "../hooks/useDashboardFilters";
import { GlobalFilterBar } from "../components/GlobalFilterBar";
import { PanelActionButton } from "../components/PanelActionButton";
import { PanelDownloadActions } from "../components/PanelDownloadActions";
import { RelationshipScatterPanel } from "../components/RelationshipScatterPanel";
import { buildDownloadFilename, downloadCsv, type DownloadMetadata } from "../lib/download";
import { geoFilterLabel, summarizeGeoFilter } from "../lib/geoLabels";

function uniqueMetrics(metrics: MetricDefinition[]): MetricDefinition[] {
  const seen = new Set<string>();
  const result: MetricDefinition[] = [];
  for (const metric of metrics) {
    if (seen.has(metric.metric_id)) {
      continue;
    }
    seen.add(metric.metric_id);
    result.push(metric);
  }
  return result;
}

const HEALTH_METRIC_GROUPS = [
  {
    label: "Mortality",
    metricIds: [
      "deaths_neonatal",
      "deaths_u1",
      "deaths_1_4y",
      "deaths_28_364d",
      "deaths_u5",
      "underlying_basic_cause_direct_heat_any_total",
      "underlying_basic_cause_indirect_heat_any_total",
      "any_mention_excessive_natural_heat_x30_total",
      "any_mention_direct_heat_any_total",
      "any_mention_indirect_heat_any_total",
    ],
  },
  {
    label: "Births and perinatal outcomes",
    metricIds: [
      "preterm_flag",
      "very_preterm_flag",
      "low_birth_weight_lt2500_total",
      "preterm_low_birth_weight_total",
      "stillbirth_total_primary_gte28",
      "stillbirth_total_gte22_sensitivity",
      "prenatal_visits_8plus_flag",
      "gest_age_canonical",
      "gest_days_lost_below_37w",
    ],
  },
  {
    label: "Hospital admissions and morbidity",
    metricIds: [
      "obstetric_admissions_total",
      "hdp_any_total",
      "preterm_labor_total",
      "gdm_gestational_total",
      "aph_antepartum_total",
      "principal_diagnosis_child_dehydration_primary_total",
      "principal_diagnosis_child_diarrhea_gi_primary_total",
      "principal_diagnosis_child_respiratory_acute_primary_total",
      "any_diagnosis_child_asthma_wheeze_sensitivity_total",
    ],
  },
] as const;

export function ClimateHealthRelationshipsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const { data: manifest } = useQuery({ queryKey: ["manifest"], queryFn: getManifest });
  const { data: filtersMeta } = useQuery({ queryKey: ["filters-meta"], queryFn: getFiltersMeta });
  const { data: metricsMeta, error: metricsMetaError } = useQuery({ queryKey: ["metrics-meta"], queryFn: getMetricsMeta });
  const { filters, setFilter } = useDashboardFilters(manifest);

  const page = useMemo<PageDefinition | undefined>(
    () => metricsMeta?.pages.find((item) => item.page_id === "relationships"),
    [metricsMeta?.pages],
  );

  const climateMetrics = useMemo(
    () =>
      uniqueMetrics(
        (metricsMeta?.metrics ?? [])
          .filter((metric) => metric.metric_family === "climate" || metric.page_id === "climate")
          .sort((left, right) => left.sort_order - right.sort_order),
      ),
    [metricsMeta?.metrics],
  );

  const healthMetricGroups = useMemo(() => {
    const metricsById = new Map(
      uniqueMetrics((metricsMeta?.metrics ?? []).filter((metric) => metric.metric_family !== "climate")).map((metric) => [
        metric.metric_id,
        metric,
      ]),
    );

    return HEALTH_METRIC_GROUPS.map((group) => ({
      label: group.label,
      metrics: group.metricIds
        .map((metricId) => metricsById.get(metricId))
        .filter((metric): metric is MetricDefinition => Boolean(metric)),
    })).filter((group) => group.metrics.length > 0);
  }, [metricsMeta?.metrics]);

  const healthMetrics = useMemo(
    () => healthMetricGroups.flatMap((group) => group.metrics),
    [healthMetricGroups],
  );

  const selectedClimateMetricId = searchParams.get("xMetric") ?? climateMetrics[0]?.metric_id;
  const requestedHealthMetricId = searchParams.get("yMetric");
  const selectedHealthMetricId =
    (requestedHealthMetricId && healthMetrics.some((metric) => metric.metric_id === requestedHealthMetricId)
      ? requestedHealthMetricId
      : undefined) ??
    healthMetrics.find((metric) => metric.metric_id === "preterm_flag")?.metric_id ??
    healthMetrics[0]?.metric_id;

  const selectedClimateMetric = climateMetrics.find((metric) => metric.metric_id === selectedClimateMetricId);
  const selectedHealthMetric = healthMetrics.find((metric) => metric.metric_id === selectedHealthMetricId);

  const requestedValueMode = searchParams.get("valueMode") as DashboardFilters["value_mode"] | null;
  const requestedFitMode = searchParams.get("fitMode");
  const effectiveValueMode: DashboardFilters["value_mode"] =
    requestedValueMode ??
    ((selectedHealthMetric?.available_value_modes.includes("rate")
      ? "rate"
      : (selectedHealthMetric?.default_value_mode as DashboardFilters["value_mode"] | undefined)) ?? "count");

  const effectiveFilters = useMemo<DashboardFilters>(
    () => ({
      ...filters,
      page_id: "relationships",
      geo_level: filters.geo_level ?? "state",
      value_mode: effectiveValueMode,
    }),
    [effectiveValueMode, filters],
  );

  const relationshipFilters = useMemo<RelationshipFilters | null>(() => {
    if (!selectedClimateMetric || !selectedHealthMetric) {
      return null;
    }
    return {
      x_metric_id: selectedClimateMetric.metric_id,
      y_metric_id: selectedHealthMetric.metric_id,
      years: effectiveFilters.years,
      granularity: effectiveFilters.granularity,
      geo_level: effectiveFilters.geo_level,
      geo_codes: effectiveFilters.geo_codes,
      x_value_mode: selectedClimateMetric.default_value_mode as "count" | "rate",
      y_value_mode: effectiveValueMode,
      context_variable: effectiveFilters.context_variable,
      context_bins: effectiveFilters.context_bins,
    };
  }, [effectiveFilters.context_bins, effectiveFilters.context_variable, effectiveFilters.geo_codes, effectiveFilters.geo_level, effectiveFilters.granularity, effectiveFilters.years, effectiveValueMode, selectedClimateMetric, selectedHealthMetric]);

  const { data: rawPoints = [] } = useQuery<RelationshipPoint[]>({
    queryKey: ["relationship", relationshipFilters],
    queryFn: () => postRelationship(relationshipFilters!),
    enabled: Boolean(relationshipFilters),
  });

  const points = useMemo(
    () => clipRelationshipRates(rawPoints, effectiveValueMode),
    [effectiveValueMode, rawPoints],
  );
  const activeContextDefinition = filtersMeta?.context_variables.find(
    (item) => item.context_variable === effectiveFilters.context_variable,
  );
  const fitModeAllowsStrata = Boolean(effectiveFilters.context_variable || effectiveFilters.geo_level === "region");
  const effectiveFitMode: "overall" | "stratum" =
    fitModeAllowsStrata && requestedFitMode === "stratum" ? "stratum" : "overall";
  const linearRegression = useMemo(() => computeLinearRegression(rawPoints), [rawPoints]);
  const stratifiedRegressions = useMemo<StratifiedLinearRegressionSummary[]>(
    () => {
      if (effectiveFilters.context_variable) {
        return computeLinearRegressionByStratum(rawPoints);
      }
      if (effectiveFilters.geo_level === "region") {
        return computeLinearRegressionByGroup(rawPoints, (point) => point.label || point.geo_code);
      }
      return [];
    },
    [effectiveFilters.context_variable, effectiveFilters.geo_level, rawPoints],
  );
  const orderedStratifiedRegressions = useMemo(() => {
    if (effectiveFilters.geo_level === "region" && !effectiveFilters.context_variable) {
      const order = new Map(["North", "Northeast", "Southeast", "South", "Central-West"].map((value, index) => [value, index]));
      return stratifiedRegressions.slice().sort((left, right) => {
        const leftIndex = order.get(left.stratum) ?? Number.MAX_SAFE_INTEGER;
        const rightIndex = order.get(right.stratum) ?? Number.MAX_SAFE_INTEGER;
        return leftIndex - rightIndex;
      });
    }
    if (!activeContextDefinition) {
      return stratifiedRegressions;
    }
    const order = new Map(activeContextDefinition.order.map((value, index) => [value, index]));
    return stratifiedRegressions.slice().sort((left, right) => {
      const leftIndex = order.get(left.stratum) ?? Number.MAX_SAFE_INTEGER;
      const rightIndex = order.get(right.stratum) ?? Number.MAX_SAFE_INTEGER;
      return leftIndex - rightIndex;
    });
  }, [activeContextDefinition, effectiveFilters.context_variable, effectiveFilters.geo_level, stratifiedRegressions]);
  const fitGroupingLabel = effectiveFilters.context_variable
    ? "contextual stratum"
    : effectiveFilters.geo_level === "region"
      ? "macroregion"
      : "stratum";
  const fitGroupColumnLabel = effectiveFilters.context_variable ? "Context stratum" : effectiveFilters.geo_level === "region" ? "Region" : "Stratum";
  const formatFitGroupLabel = (stratum: string) =>
    effectiveFilters.context_variable ? formatContextValue(effectiveFilters.context_variable, stratum) : stratum;

  const summary = useMemo(() => {
    const xValues = points.map((point) => point.x_value).filter(Number.isFinite);
    const yValues = points.map((point) => point.y_value).filter(Number.isFinite);
    const xRange =
      xValues.length > 0
        ? `${Math.min(...xValues).toLocaleString(undefined, { maximumFractionDigits: 2 })} to ${Math.max(...xValues).toLocaleString(undefined, { maximumFractionDigits: 2 })}`
        : "NA";
    const yRange =
      yValues.length > 0
        ? `${Math.min(...yValues).toLocaleString(undefined, { maximumFractionDigits: 2 })} to ${Math.max(...yValues).toLocaleString(undefined, { maximumFractionDigits: 2 })}`
        : "NA";
    return {
      points: points.length,
      xRange,
      yRange,
      slope:
        effectiveFitMode === "overall"
          ? linearRegression?.slope.toLocaleString(undefined, { maximumFractionDigits: 4 }) ?? "NA"
          : "Multiple fits",
      pValue:
        effectiveFitMode === "overall"
          ? formatPValue(linearRegression?.pValue ?? null)
          : `${orderedStratifiedRegressions.filter((item) => item.regression.significantAtFivePercent).length} / ${orderedStratifiedRegressions.length} significant`,
      rSquared:
        effectiveFitMode === "overall"
          ? linearRegression?.rSquared.toLocaleString(undefined, { maximumFractionDigits: 3 }) ?? "NA"
          : orderedStratifiedRegressions.length > 0
            ? (
                orderedStratifiedRegressions.reduce((sum, item) => sum + item.regression.rSquared, 0) /
                orderedStratifiedRegressions.length
              ).toLocaleString(undefined, { maximumFractionDigits: 3 })
            : "NA",
      significance:
        effectiveFitMode === "overall"
          ? linearRegression?.significanceLabel ?? "Insufficient observations"
          : `${orderedStratifiedRegressions.length} fitted strata`,
    };
  }, [effectiveFitMode, linearRegression, orderedStratifiedRegressions, points]);
  const geographySummaryMode = effectiveFilters.geo_level === "region" ? "region" : "state";
  const geographyFilterSummary = useMemo(
    () => summarizeGeoFilter(effectiveFilters.geo_codes, filtersMeta?.states, geographySummaryMode),
    [effectiveFilters.geo_codes, filtersMeta?.states, geographySummaryMode],
  );
  const geographyFilterName = useMemo(
    () => geoFilterLabel(geographySummaryMode),
    [geographySummaryMode],
  );
  const relationshipNotes = useMemo(
    () => [
      `Coverage: ${manifest?.coverage.municipalities ?? "..."} municipalities; years ${manifest?.years_covered.join(", ") ?? "..."}.`,
      `Climate indicator (X): ${selectedClimateMetric.label}.`,
      `Health indicator (Y): ${selectedHealthMetric.label}.`,
      "Display note: the orange curve shows a quantile-binned smooth trend; linear fits are descriptive and not causal estimates.",
      effectiveValueMode === "rate"
        ? "Rate stabilisation: visible scatter values are clipped at the upper 99th percentile to reduce unstable small-denominator extremes."
        : "Rate stabilisation: the current display uses counts, so rate clipping is not applied.",
      "Linear fit note: regression statistics are computed from the underlying paired observations that passed denominator rules, before visual clipping.",
      `Filters: Years ${effectiveFilters.years.join(", ")}; granularity ${effectiveFilters.granularity}; display geography ${effectiveFilters.geo_level}; ${geographyFilterName.toLowerCase()} ${geographyFilterSummary}; value mode ${effectiveFilters.value_mode}; context variable ${activeContextDefinition?.label ?? "None"}.`,
    ],
    [
      activeContextDefinition?.label,
      effectiveFilters.geo_level,
      effectiveFilters.granularity,
      effectiveFilters.value_mode,
      effectiveFilters.years,
      effectiveValueMode,
      geographyFilterName,
      geographyFilterSummary,
      manifest?.coverage.municipalities,
      manifest?.years_covered,
      selectedClimateMetric.label,
      selectedHealthMetric.label,
    ],
  );
  const relationshipDownloadMetadata = useMemo<DownloadMetadata>(
    () => ({
      title: `${page?.title ?? "Climate-Health Relationships"} - ${selectedClimateMetric.label} versus ${selectedHealthMetric.label}`,
      notes: relationshipNotes,
    }),
    [page?.title, relationshipNotes, selectedClimateMetric.label, selectedHealthMetric.label],
  );

  function setRelationshipParam(key: string, value: string | undefined) {
    const next = new URLSearchParams(searchParams);
    if (!value) {
      next.delete(key);
    } else {
      next.set(key, value);
    }
    setSearchParams(next, { replace: true });
  }

  if (metricsMetaError) {
    return <div className="text-sm text-caution">Unable to load relationship metadata from the dashboard API.</div>;
  }

  if (!page || !selectedClimateMetric || !selectedHealthMetric) {
    return <div className="text-sm text-slate-500">Loading dashboard metadata...</div>;
  }

  return (
    <div className="space-y-6">
      <div className="space-y-6">
        <section className="space-y-2">
          <h2 className="text-2xl font-semibold">{page.title}</h2>
          <p className="max-w-4xl text-sm text-slate-600">{page.description}</p>
        </section>

        <GlobalFilterBar
          filters={effectiveFilters}
          filtersMeta={filtersMeta}
          metricAllowsRate={Boolean(selectedHealthMetric.available_value_modes.includes("rate"))}
          onChange={setFilter}
          includeRegionOption
        />

        <div className="grid gap-6 xl:grid-cols-[340px_minmax(0,1fr)]">
          <section className="space-y-4">
            <section className="panel">
              <div className="panel-header">
                <h3 className="text-base font-semibold">Analysis controls</h3>
              </div>
              <div className="panel-body space-y-4">
                <label className="flex flex-col gap-2 text-sm">
                  <span className="font-medium">Climate indicator (X)</span>
                  <select
                    value={selectedClimateMetric.metric_id}
                    onChange={(event) => setRelationshipParam("xMetric", event.target.value)}
                    className="rounded-xl border border-rule bg-white px-3 py-2"
                  >
                    {climateMetrics.map((metric) => (
                      <option key={metric.metric_id} value={metric.metric_id}>
                        {metric.label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="flex flex-col gap-2 text-sm">
                  <span className="font-medium">Health indicator (Y)</span>
                  <select
                    value={selectedHealthMetric.metric_id}
                    onChange={(event) => setRelationshipParam("yMetric", event.target.value)}
                    className="rounded-xl border border-rule bg-white px-3 py-2"
                  >
                    {healthMetricGroups.map((group) => (
                      <optgroup key={group.label} label={group.label}>
                        {group.metrics.map((metric) => (
                          <option key={metric.metric_id} value={metric.metric_id}>
                            {metric.label}
                          </option>
                        ))}
                      </optgroup>
                    ))}
                  </select>
                </label>

                <label className="flex flex-col gap-2 text-sm">
                  <span className="font-medium">Linear fit mode</span>
                  <select
                    value={effectiveFitMode}
                    onChange={(event) => setRelationshipParam("fitMode", event.target.value)}
                    className="rounded-xl border border-rule bg-white px-3 py-2"
                    disabled={!fitModeAllowsStrata}
                  >
                    <option value="overall">Overall fit</option>
                    <option value="stratum">Stratum-specific fits</option>
                  </select>
                </label>
              </div>
            </section>

            <div className="grid gap-4">
              <div className="rounded-2xl border border-rule bg-white px-4 py-4 shadow-panel">
                <div className="text-xs uppercase tracking-wide text-slate-500">Paired observations</div>
                <div className="mt-3 text-3xl font-semibold">{summary.points.toLocaleString()}</div>
              </div>

              <div className="rounded-2xl border border-rule bg-white px-4 py-4 shadow-panel">
                <div className="text-xs uppercase tracking-wide text-slate-500">Linear slope</div>
                <div className="mt-3 text-sm font-medium">{summary.slope}</div>
                <div className="mt-4 text-xs uppercase tracking-wide text-slate-500">P-value</div>
                <div className="mt-2 text-sm font-medium">{summary.pValue}</div>
              </div>

              <div className="rounded-2xl border border-rule bg-white px-4 py-4 shadow-panel">
                <div className="text-xs uppercase tracking-wide text-slate-500">Linear fit</div>
                <div className="mt-3 text-sm font-medium">{summary.significance}</div>
                <div className="mt-4 text-xs uppercase tracking-wide text-slate-500">R-squared</div>
                <div className="mt-2 text-sm font-medium">{summary.rSquared}</div>
              </div>

              <div className="rounded-2xl border border-rule bg-white px-4 py-4 shadow-panel">
                <div className="text-xs uppercase tracking-wide text-slate-500">Outcome range</div>
                <div className="mt-3 text-sm font-medium">{summary.yRange}</div>
                <div className="mt-4 text-xs uppercase tracking-wide text-slate-500">Exposure range</div>
                <div className="mt-2 text-sm font-medium">{summary.xRange}</div>
              </div>
            </div>
          </section>

          <RelationshipScatterPanel
            title="Exposure-response scatter"
            subtitle="Each point represents one geography-period pair under the current filters."
            xLabel={selectedClimateMetric.label}
            yLabel={selectedHealthMetric.rate_display_unit && effectiveValueMode === "rate"
              ? `${selectedHealthMetric.label} (${selectedHealthMetric.rate_display_unit})`
              : `${selectedHealthMetric.label} (${selectedHealthMetric.display_unit})`}
            points={points}
            linearRegression={linearRegression}
            stratifiedRegressions={orderedStratifiedRegressions}
            fitMode={effectiveFitMode}
            geoLevel={effectiveFilters.geo_level}
            contextVariable={effectiveFilters.context_variable}
            contextVariableLabel={activeContextDefinition?.label}
            contextOrder={activeContextDefinition?.order}
            fitGroupingLabel={fitGroupingLabel}
            downloadMetadata={relationshipDownloadMetadata}
          />
        </div>

        {effectiveFitMode === "stratum" && orderedStratifiedRegressions.length > 0 && (
          <section className="panel">
            <div className="panel-header">
              <h3 className="text-base font-semibold">Stratum-specific linear fits</h3>
              <PanelDownloadActions>
                <PanelActionButton
                  label="CSV"
                  onClick={() =>
                    downloadCsv(
                      orderedStratifiedRegressions.map(({ stratum, regression }) => ({
                        group_label: formatFitGroupLabel(stratum),
                        raw_stratum: stratum,
                        observations: regression.n,
                        slope: regression.slope,
                        p_value: regression.pValue,
                        r_squared: regression.rSquared,
                        interpretation: regression.significanceLabel,
                      })),
                      buildDownloadFilename("relationship-stratum-specific-linear-fits", "csv"),
                      {
                        ...relationshipDownloadMetadata,
                        title: `${relationshipDownloadMetadata.title} - Stratum-specific linear fits`,
                      },
                    )
                  }
                />
              </PanelDownloadActions>
            </div>
            <div className="panel-body overflow-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-slate-500">
                  <tr>
                    <th className="pb-3">{fitGroupColumnLabel}</th>
                    <th className="pb-3">Observations</th>
                    <th className="pb-3">Slope</th>
                    <th className="pb-3">P-value</th>
                    <th className="pb-3">R-squared</th>
                    <th className="pb-3">Interpretation</th>
                  </tr>
                </thead>
                <tbody>
                  {orderedStratifiedRegressions.map(({ stratum, regression }) => (
                    <tr key={stratum} className="border-t border-rule align-top">
                      <td className="py-3 font-medium">
                        {formatFitGroupLabel(stratum)}
                      </td>
                      <td className="py-3">{regression.n.toLocaleString()}</td>
                      <td className="py-3">{regression.slope.toLocaleString(undefined, { maximumFractionDigits: 4 })}</td>
                      <td className="py-3">{formatPValue(regression.pValue)}</td>
                      <td className="py-3">{regression.rSquared.toLocaleString(undefined, { maximumFractionDigits: 3 })}</td>
                      <td className="py-3">{regression.significanceLabel}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        <section className="panel">
          <div className="panel-header">
            <h3 className="text-base font-semibold">Notes</h3>
          </div>
          <div className="panel-body grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            <div className="rounded-2xl border border-rule p-4">
              <div className="text-xs uppercase tracking-wide text-slate-500">Coverage</div>
              <p className="mt-3 text-sm text-slate-700">{relationshipNotes[0]}</p>
            </div>
            <div className="rounded-2xl border border-rule p-4">
              <div className="text-xs uppercase tracking-wide text-slate-500">Indicators</div>
              <p className="mt-3 text-sm text-slate-700">{relationshipNotes[1]}</p>
              <p className="mt-2 text-sm text-slate-700">{relationshipNotes[2]}</p>
            </div>
            <div className="rounded-2xl border border-rule p-4">
              <div className="text-xs uppercase tracking-wide text-slate-500">Display note</div>
              <p className="mt-3 text-sm text-slate-700">{relationshipNotes[3]}</p>
            </div>
            <div className="rounded-2xl border border-rule p-4">
              <div className="text-xs uppercase tracking-wide text-slate-500">Rate stabilisation</div>
              <p className="mt-3 text-sm text-slate-700">{relationshipNotes[4]}</p>
            </div>
            <div className="rounded-2xl border border-rule p-4">
              <div className="text-xs uppercase tracking-wide text-slate-500">Linear fit note</div>
              <p className="mt-3 text-sm text-slate-700">{relationshipNotes[5]}</p>
            </div>
            <div className="rounded-2xl border border-rule p-4">
              <div className="text-xs uppercase tracking-wide text-slate-500">Filters</div>
              <p className="mt-3 text-sm text-slate-700">{relationshipNotes[6]}</p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
