import type { ChangeEvent } from "react";
import type { ContextVariable, DashboardFilters, FiltersPayload } from "../lib/api";
import { formatContextOption } from "../lib/dashboardDisplay";

type GlobalFilterBarProps = {
  filters: DashboardFilters;
  filtersMeta?: FiltersPayload;
  metricAllowsRate: boolean;
  onChange: (key: string, value: string | string[] | number[] | undefined) => void;
  viewMode?: "national" | "regional";
  onViewModeChange?: (value: "national" | "regional") => void;
  includeRegionOption?: boolean;
};

function handleMultiSelect(event: ChangeEvent<HTMLSelectElement>) {
  return Array.from(event.target.selectedOptions).map((option) => option.value);
}

const COMBINED_YEARS_VALUE = "__combined_years__";

function combinedYearsLabel(years: number[] | undefined): string {
  if (!years || years.length === 0) {
    return "Combined";
  }
  const sorted = [...years].sort((left, right) => left - right);
  if (sorted.length === 1) {
    return `Combined ${sorted[0]}`;
  }
  return `Combined ${sorted[0]}-${String(sorted.at(-1)).slice(-2)}`;
}

export function GlobalFilterBar({
  filters,
  filtersMeta,
  metricAllowsRate,
  onChange,
  viewMode,
  onViewModeChange,
  includeRegionOption = false,
}: GlobalFilterBarProps) {
  const contextDefinition = filtersMeta?.context_variables.find(
    (item) => item.context_variable === filters.context_variable
  );
  const allYearsSelected =
    Boolean(filtersMeta?.years.length) &&
    filtersMeta?.years.every((year) => filters.years.includes(year));
  const yearSelectValue = allYearsSelected
    ? [COMBINED_YEARS_VALUE, ...filters.years.map(String)]
    : filters.years.map(String);

  return (
    <div className="panel">
      <div className="panel-body grid gap-4 md:grid-cols-2 xl:grid-cols-6">
        <label className="flex flex-col gap-2 text-sm">
          <span className="font-medium">Years</span>
          <select
            multiple
            value={yearSelectValue}
            onChange={(event) => {
              const selected = handleMultiSelect(event);
              if (selected.includes(COMBINED_YEARS_VALUE) && filtersMeta?.years.length) {
                onChange("years", filtersMeta.years);
                return;
              }
              onChange(
                "years",
                selected
                  .filter((value) => value !== COMBINED_YEARS_VALUE)
                  .map(Number),
              );
            }}
            className="min-h-28 rounded-xl border border-rule bg-white px-3 py-2"
          >
            {filtersMeta?.years.length ? (
              <option value={COMBINED_YEARS_VALUE}>
                {combinedYearsLabel(filtersMeta.years)}
              </option>
            ) : null}
            {filtersMeta?.years.map((year) => (
              <option key={year} value={year}>
                {year}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-2 text-sm">
          <span className="font-medium">Granularity</span>
          <select
            value={filters.granularity}
            onChange={(event) => onChange("granularity", event.target.value)}
            className="rounded-xl border border-rule bg-white px-3 py-2"
          >
            {filtersMeta?.granularities
              .filter((item) => item !== "week")
              .map((granularity) => (
                <option key={granularity} value={granularity}>
                  {granularity}
                </option>
              ))}
          </select>
        </label>

        <label className="flex flex-col gap-2 text-sm">
          <span className="font-medium">{viewMode ? "View mode" : "Display geography"}</span>
          {viewMode && onViewModeChange ? (
            <select
              value={viewMode}
              onChange={(event) => onViewModeChange(event.target.value as "national" | "regional")}
              className="rounded-xl border border-rule bg-white px-3 py-2"
            >
              <option value="national">National</option>
              <option value="regional">Regional comparison</option>
            </select>
          ) : (
            <select
              value={filters.geo_level}
              onChange={(event) => onChange("geoLevel", event.target.value)}
              className="rounded-xl border border-rule bg-white px-3 py-2"
            >
              {includeRegionOption ? <option value="region">Region</option> : null}
              <option value="state">State</option>
              <option value="municipality">Municipality</option>
              <option value="national">National</option>
            </select>
          )}
        </label>

        <label className="flex flex-col gap-2 text-sm">
          <span className="font-medium">State filter</span>
          <select
            multiple
            value={filters.geo_codes.filter((code) => code.length === 2)}
            onChange={(event) => onChange("geoCodes", handleMultiSelect(event))}
            className="min-h-28 rounded-xl border border-rule bg-white px-3 py-2"
          >
            {filtersMeta?.states.map((state) => (
              <option key={state.geo_code} value={state.geo_code}>
                {state.label}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-2 text-sm">
          <span className="font-medium">Value mode</span>
          <select
            value={filters.value_mode}
            onChange={(event) => onChange("valueMode", event.target.value)}
            className="rounded-xl border border-rule bg-white px-3 py-2"
            disabled={!metricAllowsRate}
          >
            <option value="count">Value</option>
            <option value="rate">Rate</option>
          </select>
        </label>

        <label className="flex flex-col gap-2 text-sm">
          <span className="font-medium">Context variable</span>
          <select
            value={filters.context_variable ?? ""}
            onChange={(event) => onChange("contextVariable", event.target.value || undefined)}
            className="rounded-xl border border-rule bg-white px-3 py-2"
          >
            <option value="">None</option>
            {filtersMeta?.context_variables.map((variable) => (
              <option key={variable.context_variable} value={variable.context_variable}>
                {variable.label}
              </option>
            ))}
          </select>
        </label>

        {contextDefinition && (
          <label className="flex flex-col gap-2 text-sm md:col-span-2 xl:col-span-6">
            <span className="font-medium">Context bins</span>
            <select
              multiple
              value={filters.context_bins}
              onChange={(event) => onChange("contextBins", handleMultiSelect(event))}
              className="min-h-28 rounded-xl border border-rule bg-white px-3 py-2"
            >
              {contextDefinition.order.map((option) => (
                <option key={option} value={option}>
                  {formatContextOption(contextDefinition, option)}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
    </div>
  );
}
