import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import {
  getFiltersMeta,
  getManifest,
  getMetricsMeta,
  postCards,
  postMap,
  postRanking,
  postSeries,
  postStratification,
  type DashboardFilters,
  type MetricDefinition,
  type PageDefinition
} from "../lib/api";
import {
  clipMapRates,
  clipRankingRates,
  clipSeriesRates,
  clipStratificationRates,
  formatContextValue,
  sortStratificationRows,
} from "../lib/dashboardDisplay";
import { useDashboardFilters } from "../hooks/useDashboardFilters";
import { GlobalFilterBar } from "../components/GlobalFilterBar";
import { MetadataRail } from "../components/MetadataRail";
import { MetricCard } from "../components/MetricCard";
import { TrendPanel } from "../components/TrendPanel";
import { ChoroplethPanel } from "../components/ChoroplethPanel";
import { HeatmapPanel } from "../components/HeatmapPanel";
import { RankingTable } from "../components/RankingTable";
import { DefinitionDrawer } from "../components/DefinitionDrawer";
import { MunicipalityProfileDrawer } from "../components/MunicipalityProfileDrawer";
import { PanelActionButton } from "../components/PanelActionButton";
import { PanelDownloadActions } from "../components/PanelDownloadActions";
import { buildDownloadFilename, downloadCsv, type DownloadMetadata } from "../lib/download";
import { geoFilterLabel, summarizeGeoFilter } from "../lib/geoLabels";

type AnalyticalPageProps = {
  pageId: string;
};

const REGION_GROUPS: Array<{ id: string; label: string; color: string; states: string[] }> = [
  { id: "north", label: "North", color: "#12324A", states: ["AC", "AP", "AM", "PA", "RO", "RR", "TO"] },
  { id: "northeast", label: "Northeast", color: "#C76A1B", states: ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"] },
  { id: "southeast", label: "Southeast", color: "#3E7C78", states: ["ES", "MG", "RJ", "SP"] },
  { id: "south", label: "South", color: "#6A4C93", states: ["PR", "RS", "SC"] },
  { id: "central_west", label: "Central-West", color: "#8E2C2C", states: ["DF", "GO", "MS", "MT"] },
];

export function AnalyticalPage({ pageId }: AnalyticalPageProps) {
  const [definitionOpen, setDefinitionOpen] = useState(false);
  const [selectedMunicipality, setSelectedMunicipality] = useState<string | null>(null);
  const [searchParams, setSearchParams] = useSearchParams();
  const { data: manifest } = useQuery({ queryKey: ["manifest"], queryFn: getManifest });
  const { data: filtersMeta } = useQuery({ queryKey: ["filters-meta"], queryFn: getFiltersMeta });
  const { data: metricsMeta, error: metricsMetaError } = useQuery({ queryKey: ["metrics-meta"], queryFn: getMetricsMeta });
  const { filters, setFilter } = useDashboardFilters(manifest);

  const viewMode = searchParams.get("viewMode") === "regional" ? "regional" : "national";
  const page = useMemo<PageDefinition | undefined>(
    () => metricsMeta?.pages.find((item) => item.page_id === pageId),
    [metricsMeta?.pages, pageId]
  );

  const metric = useMemo<MetricDefinition | undefined>(() => {
    const metricId = filters.metric_id ?? page?.default_metric_id ?? page?.metric_ids[0];
    return metricsMeta?.metrics.find((item) => item.metric_id === metricId);
  }, [filters.metric_id, metricsMeta?.metrics, page]);

  const effectiveFilters = useMemo<DashboardFilters>(() => {
    if (!page || !metric) {
      return filters;
    }
    return {
      ...filters,
      page_id: page.page_id,
      metric_id: metric.metric_id,
      geo_level: viewMode === "regional" ? "state" : "national"
    };
  }, [filters, metric, page, viewMode]);

  const cardsFilters = useMemo<DashboardFilters>(() => {
    if (!page) {
      return effectiveFilters;
    }
    return {
      ...effectiveFilters,
      page_id: page.page_id,
      metric_id: undefined,
    };
  }, [effectiveFilters, page]);

  const { data: cards = [] } = useQuery({
    queryKey: ["cards", pageId, cardsFilters],
    queryFn: () => postCards(cardsFilters),
    enabled: Boolean(page)
  });

  const { data: series = [] } = useQuery({
    queryKey: ["series", pageId, effectiveFilters],
    queryFn: () => postSeries(effectiveFilters),
    enabled: Boolean(metric) && viewMode === "national"
  });

  const { data: monthlySeries = [] } = useQuery({
    queryKey: ["monthly-series", pageId, effectiveFilters.metric_id, effectiveFilters.years, effectiveFilters.geo_codes, effectiveFilters.value_mode],
    queryFn: () => postSeries({ ...effectiveFilters, granularity: "month" }),
    enabled: Boolean(metric)
  });

  const { data: regionalSeriesCollections = [] } = useQuery({
    queryKey: ["regional-series", pageId, effectiveFilters.metric_id, effectiveFilters.years, effectiveFilters.granularity, effectiveFilters.value_mode, effectiveFilters.geo_codes],
    queryFn: async () => {
      const selectedStates = new Set(effectiveFilters.geo_codes.filter((code) => code.length === 2));
      const collections = await Promise.all(
        REGION_GROUPS.map(async (region) => {
          const regionStates = selectedStates.size > 0
            ? region.states.filter((state) => selectedStates.has(state))
            : region.states;
          if (regionStates.length === 0) {
            return null;
          }
          const points = await postSeries({
            ...effectiveFilters,
            geo_level: "state",
            geo_codes: regionStates,
          });
          return {
            label: region.label,
            color: region.color,
            points: clipSeriesRates(points, effectiveFilters),
          };
        }),
      );
      return collections.filter(
        (
          collection,
        ): collection is {
          label: string;
          color: string;
          points: typeof series;
        } => Boolean(collection) && collection.points.length > 0,
      );
    },
    enabled: Boolean(metric) && viewMode === "regional",
  });

  const mapFilters = useMemo(
    () => ({ ...effectiveFilters, geo_level: "state" as const }),
    [effectiveFilters],
  );
  const { data: mapRows = [] } = useQuery({
    queryKey: ["map", pageId, mapFilters],
    queryFn: () => postMap(mapFilters),
    enabled: Boolean(metric)
  });

  const { data: rankingRows = [] } = useQuery({
    queryKey: ["ranking", pageId, mapFilters],
    queryFn: () => postRanking(mapFilters),
    enabled: Boolean(metric)
  });

  const { data: stratificationRows = [] } = useQuery({
    queryKey: ["stratification", pageId, effectiveFilters],
    queryFn: () => postStratification({ ...effectiveFilters, context_variable: effectiveFilters.context_variable ?? "idh_quintile" }),
    enabled: Boolean(metric)
  });

  const activeContextVariable = effectiveFilters.context_variable ?? "idh_quintile";
  const activeContextDefinition = filtersMeta?.context_variables.find(
    (item) => item.context_variable === activeContextVariable,
  );

  const displaySeries = useMemo(() => clipSeriesRates(series, effectiveFilters), [effectiveFilters, series]);
  const displayMonthlySeries = useMemo(() => clipSeriesRates(monthlySeries, effectiveFilters), [effectiveFilters, monthlySeries]);
  const displayMapRows = useMemo(() => clipMapRates(mapRows, effectiveFilters), [effectiveFilters, mapRows]);
  const displayRankingRows = useMemo(() => clipRankingRates(rankingRows, effectiveFilters), [effectiveFilters, rankingRows]);
  const clippedStratificationRows = useMemo(
    () => clipStratificationRates(stratificationRows, effectiveFilters),
    [effectiveFilters, stratificationRows],
  );
  const displayStratificationRows = useMemo(
    () => sortStratificationRows(clippedStratificationRows, activeContextDefinition),
    [activeContextDefinition, clippedStratificationRows],
  );
  const geographySummaryMode = viewMode === "regional" ? "region" : "state";
  const geographyFilterSummary = useMemo(
    () => summarizeGeoFilter(effectiveFilters.geo_codes, filtersMeta?.states, geographySummaryMode),
    [effectiveFilters.geo_codes, filtersMeta?.states, geographySummaryMode],
  );
  const geographyFilterName = useMemo(
    () => geoFilterLabel(geographySummaryMode),
    [geographySummaryMode],
  );
  const commonDownloadNotes = useMemo<DownloadMetadata>(() => {
    const notes = [
      `Page: ${page.title}`,
      `Indicator: ${metric?.label ?? "Not selected"}`,
      `Years: ${effectiveFilters.years.join(", ")}`,
      `Granularity: ${effectiveFilters.granularity}`,
      `View mode: ${viewMode === "regional" ? "Regional comparison" : "National"}`,
      `${geographyFilterName}: ${geographyFilterSummary}`,
      `Value mode: ${effectiveFilters.value_mode}`,
      `Context variable: ${activeContextDefinition?.label ?? "None"}`,
    ];
    if (effectiveFilters.value_mode === "rate") {
      notes.push("Visible rate values are clipped at the upper 99th percentile within the current view.");
    }
    return {
      title: `${page.title} - ${metric?.label ?? "Indicator"}`,
      notes,
    };
  }, [activeContextDefinition?.label, effectiveFilters.granularity, effectiveFilters.value_mode, effectiveFilters.years, geographyFilterName, geographyFilterSummary, metric?.label, page.title, viewMode]);
  const activeDisplayUnit = useMemo(
    () =>
      effectiveFilters.value_mode === "rate" && metric?.rate_display_unit
        ? metric.rate_display_unit
        : metric?.display_unit,
    [effectiveFilters.value_mode, metric?.display_unit, metric?.rate_display_unit],
  );
  const rateInterpretationNote = useMemo(() => {
    if (effectiveFilters.value_mode !== "rate" || !metric?.rate_display_unit) {
      return undefined;
    }
    if (viewMode === "regional") {
      return `Values remain displayed in ${metric.rate_display_unit}. Regional comparison overlays one curve per Brazilian macroregion.`;
    }
    if (effectiveFilters.granularity === "year") {
      return `Values are displayed in ${metric.rate_display_unit}.`;
    }
    return `Values remain displayed in ${metric.rate_display_unit}. Lower monthly or seasonal points reflect the numerator within each time bucket, not a change in the rate multiplier.`;
  }, [effectiveFilters.granularity, effectiveFilters.value_mode, metric?.rate_display_unit, viewMode]);

  function setViewMode(nextViewMode: "national" | "regional") {
    const next = new URLSearchParams(searchParams);
    next.set("viewMode", nextViewMode);
    setSearchParams(next, { replace: true });
  }

  if (metricsMetaError) {
    return <div className="text-sm text-caution">Unable to load page metadata from the dashboard API.</div>;
  }

  if (!page) {
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
          metricAllowsRate={Boolean(metric?.available_value_modes.includes("rate"))}
          onChange={setFilter}
          viewMode={viewMode}
          onViewModeChange={setViewMode}
        />

        {effectiveFilters.value_mode === "rate" && (
          <p className="text-xs text-slate-500">
            Visible rate values are clipped at the upper 99th percentile within the current view to reduce unstable small-denominator extremes.
          </p>
        )}

        <div className="grid gap-6 xl:grid-cols-[340px_minmax(0,1fr)]">
          <section className="space-y-4">
            <section className="panel">
              <div className="panel-header">
                <h3 className="text-base font-semibold">Key indicators</h3>
              </div>
              <div className="panel-body grid gap-4">
                {cards.map((card) => (
                  <MetricCard
                    key={card.metric_id}
                    card={card}
                    active={card.metric_id === metric?.metric_id}
                    onClick={() => setFilter("metric", card.metric_id)}
                  />
                ))}
              </div>
            </section>
          </section>

          <div className="space-y-6">
            <div className="grid gap-6 xl:grid-cols-2">
              <TrendPanel
                title="Temporal trend"
                points={viewMode === "national" ? displaySeries : undefined}
                seriesCollections={viewMode === "regional" ? regionalSeriesCollections : undefined}
                valueUnit={activeDisplayUnit}
                note={rateInterpretationNote}
                downloadMetadata={{ ...commonDownloadNotes, title: `${commonDownloadNotes.title} - Temporal trend` }}
              />
              <HeatmapPanel
                title="Month by year heatmap"
                points={displayMonthlySeries}
                valueUnit={activeDisplayUnit}
                note={
                  viewMode === "regional"
                    ? `This heatmap remains a national reference view. Regional comparison is shown in the temporal trend panel above.`
                    : rateInterpretationNote
                }
                downloadMetadata={{ ...commonDownloadNotes, title: `${commonDownloadNotes.title} - Month by year heatmap` }}
              />
            </div>

            <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
              <ChoroplethPanel
                title="Spatial distribution"
                data={displayMapRows}
                geoLevel="state"
                onSelectMunicipality={setSelectedMunicipality}
                downloadMetadata={{ ...commonDownloadNotes, title: `${commonDownloadNotes.title} - Spatial distribution` }}
              />
              <RankingTable
                title="Ranking"
                rows={displayRankingRows}
                onSelectMunicipality={setSelectedMunicipality}
                downloadMetadata={{ ...commonDownloadNotes, title: `${commonDownloadNotes.title} - Ranking` }}
              />
            </div>
          </div>
        </div>

        <section className="panel">
          <div className="panel-header">
            <h3 className="text-base font-semibold">Context stratification</h3>
            <div className="flex items-center gap-3">
              <PanelDownloadActions>
                <PanelActionButton
                  label="CSV"
                  onClick={() =>
                    downloadCsv(
                      displayStratificationRows.map((row) => ({
                        context_stratum: formatContextValue(activeContextVariable, row.stratum),
                        raw_stratum: row.stratum,
                        value: row.value,
                        denominator: row.denominator,
                        municipalities: row.n_geographies,
                      })),
                      buildDownloadFilename(`${page.title}-context-stratification`, "csv"),
                      {
                        ...commonDownloadNotes,
                        title: `${commonDownloadNotes.title} - Context stratification`,
                      },
                    )
                  }
                />
              </PanelDownloadActions>
              <button
                type="button"
                onClick={() => setDefinitionOpen(true)}
                className="rounded-full border border-rule px-3 py-1 text-sm"
              >
                Definition
              </button>
            </div>
          </div>
          <div className="panel-body">
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {displayStratificationRows.map((row) => (
                <div key={row.stratum} className="rounded-2xl border border-rule p-4">
                  <div className="text-xs uppercase tracking-wide text-slate-500">
                    {formatContextValue(activeContextVariable, row.stratum)}
                  </div>
                  <div className="mt-3 text-2xl font-semibold">
                    {row.value === null ? "NA" : row.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  </div>
                  <div className="mt-1 text-sm text-slate-500">{row.n_geographies} municipalities</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <MetadataRail
          manifest={manifest}
          metric={metric}
          pageTitle={page.title}
          geographyFilterLabel={geographyFilterSummary}
          geographyFilterName={geographyFilterName}
          valueMode={effectiveFilters.value_mode}
          contextVariableLabel={activeContextDefinition?.label}
          granularity={effectiveFilters.granularity}
          viewModeLabel={viewMode === "regional" ? "Regional comparison" : "National"}
        />
      </div>

      <DefinitionDrawer metric={metric} open={definitionOpen} onClose={() => setDefinitionOpen(false)} />
      <MunicipalityProfileDrawer municipalityCode={selectedMunicipality} onClose={() => setSelectedMunicipality(null)} />
    </div>
  );
}
