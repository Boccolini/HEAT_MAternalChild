type StateOption = {
  geo_code?: string;
  label?: string;
};

const REGION_ORDER = ["North", "Northeast", "Southeast", "South", "Central-West"] as const;

const UF_TO_REGION: Record<string, (typeof REGION_ORDER)[number]> = {
  AC: "North",
  AP: "North",
  AM: "North",
  PA: "North",
  RO: "North",
  RR: "North",
  TO: "North",
  AL: "Northeast",
  BA: "Northeast",
  CE: "Northeast",
  MA: "Northeast",
  PB: "Northeast",
  PE: "Northeast",
  PI: "Northeast",
  RN: "Northeast",
  SE: "Northeast",
  ES: "Southeast",
  MG: "Southeast",
  RJ: "Southeast",
  SP: "Southeast",
  PR: "South",
  RS: "South",
  SC: "South",
  DF: "Central-West",
  GO: "Central-West",
  MS: "Central-West",
  MT: "Central-West",
};

export function summarizeGeoFilter(
  geoCodes: string[],
  states: StateOption[] | undefined,
  mode: "state" | "region",
): string {
  const stateCodes = [...new Set(geoCodes.filter((code) => code.length === 2))];
  if (stateCodes.length === 0) {
    return mode === "region" ? "All regions" : "All states";
  }

  if (mode === "region") {
    const regions = [...new Set(stateCodes.map((code) => UF_TO_REGION[code]).filter(Boolean))];
    return regions
      .sort((left, right) => REGION_ORDER.indexOf(left) - REGION_ORDER.indexOf(right))
      .join(", ");
  }

  return (states ?? [])
    .filter((state) => state.geo_code && stateCodes.includes(state.geo_code))
    .map((state) => state.label)
    .filter(Boolean)
    .join(", ");
}

export function geoFilterLabel(mode: "state" | "region"): string {
  return mode === "region" ? "Regional filter" : "State filter";
}
