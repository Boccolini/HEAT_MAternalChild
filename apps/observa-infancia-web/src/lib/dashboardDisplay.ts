import type { ContextVariable, DashboardFilters, MapDatum, RankingRow, RelationshipPoint, SeriesPoint, StratificationRow } from "./api";

const QUINTILE_LABELS: Record<string, string> = {
  Q1: "Lower",
  Q2: "Intermediate-lower",
  Q3: "Average",
  Q4: "Upper",
  Q5: "Higher",
};

const QUARTILE_LABELS: Record<string, string> = {
  Q1: "Lower",
  Q2: "Intermediate-lower",
  Q3: "Upper",
  Q4: "Higher",
};

const VARIABLE_SUFFIXES: Record<string, string> = {
  idh_quintile: "development",
  gini_quintile: "inequality",
  altitude_quartile: "altitude",
};

const FIXED_LABELS: Record<string, Record<string, string>> = {
  urbanicity_band: {
    low: "Lower urbanicity",
    medium: "Intermediate urbanicity",
    high: "Higher urbanicity",
    missing: "Missing context",
  },
  coastal_flag: {
    non_coastal: "Non-coastal",
    coastal: "Coastal",
    missing: "Missing context",
  },
};

type NumericValueRow = {
  value?: number | null;
};

function percentile(values: number[], q: number): number | null {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const position = (sorted.length - 1) * q;
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  if (lowerIndex === upperIndex) {
    return sorted[lowerIndex];
  }
  const lowerWeight = upperIndex - position;
  const upperWeight = position - lowerIndex;
  return sorted[lowerIndex] * lowerWeight + sorted[upperIndex] * upperWeight;
}

export function formatContextValue(contextVariable: string | undefined, rawValue: string): string {
  if (!contextVariable) {
    return rawValue;
  }
  if (rawValue === "missing") {
    return "Missing context";
  }
  if (contextVariable in FIXED_LABELS) {
    return FIXED_LABELS[contextVariable][rawValue] ?? rawValue;
  }
  const suffix = VARIABLE_SUFFIXES[contextVariable];
  if (!suffix) {
    return rawValue;
  }
  if (contextVariable.endsWith("_quartile")) {
    return QUARTILE_LABELS[rawValue] ? `${QUARTILE_LABELS[rawValue]} ${suffix}` : rawValue;
  }
  return QUINTILE_LABELS[rawValue] ? `${QUINTILE_LABELS[rawValue]} ${suffix}` : rawValue;
}

export function formatContextOption(definition: ContextVariable | undefined, rawValue: string): string {
  return formatContextValue(definition?.context_variable, rawValue);
}

export function sortStratificationRows(
  rows: StratificationRow[],
  definition: ContextVariable | undefined,
): StratificationRow[] {
  if (!definition) {
    return rows;
  }
  const order = new Map(definition.order.map((item, index) => [item, index]));
  return [...rows].sort((left, right) => {
    const leftIndex = order.get(left.stratum) ?? Number.MAX_SAFE_INTEGER;
    const rightIndex = order.get(right.stratum) ?? Number.MAX_SAFE_INTEGER;
    return leftIndex - rightIndex;
  });
}

export function clipRateValues<T extends NumericValueRow>(rows: T[], filters: DashboardFilters): T[] {
  if (filters.value_mode !== "rate") {
    return rows;
  }
  const numericValues = rows
    .map((row) => row.value)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (numericValues.length < 5) {
    return rows;
  }
  const upperCap = percentile(numericValues, 0.99);
  if (upperCap === null || !Number.isFinite(upperCap) || upperCap <= 0) {
    return rows;
  }
  return rows.map((row) => {
    if (row.value === null || row.value <= upperCap) {
      return row;
    }
    return { ...row, value: upperCap };
  });
}

export function clipSeriesRates(points: SeriesPoint[], filters: DashboardFilters): SeriesPoint[] {
  if (filters.value_mode !== "rate") {
    return points;
  }
  const numericValues = points
    .map((point) => point.y)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (numericValues.length < 5) {
    return points;
  }
  const upperCap = percentile(numericValues, 0.99);
  if (upperCap === null || !Number.isFinite(upperCap) || upperCap <= 0) {
    return points;
  }
  return points.map((point) => {
    if (point.y === null || point.y <= upperCap) {
      return point;
    }
    return { ...point, y: upperCap };
  });
}

export function clipMapRates(rows: MapDatum[], filters: DashboardFilters): MapDatum[] {
  return clipRateValues(rows, filters);
}

export function clipRankingRates(rows: RankingRow[], filters: DashboardFilters): RankingRow[] {
  return clipRateValues(rows, filters);
}

export function clipStratificationRates(rows: StratificationRow[], filters: DashboardFilters): StratificationRow[] {
  return clipRateValues(rows, filters);
}

export function clipRelationshipRates(points: RelationshipPoint[], yValueMode: "count" | "rate"): RelationshipPoint[] {
  if (yValueMode !== "rate") {
    return points;
  }
  const numericValues = points
    .map((point) => point.y_value)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (numericValues.length < 5) {
    return points;
  }
  const upperCap = percentile(numericValues, 0.99);
  if (upperCap === null || !Number.isFinite(upperCap) || upperCap <= 0) {
    return points;
  }
  return points.map((point) => {
    if (point.y_value <= upperCap) {
      return point;
    }
    return { ...point, y_value: upperCap };
  });
}
