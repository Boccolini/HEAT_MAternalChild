export const DASHBOARD_FROZEN_BASELINE_VERSION = "freeze-2026-03-29";
export const DASHBOARD_FROZEN_BASELINE_DATE = "2026-03-29";
export const DASHBOARD_FROZEN_BASELINE_NOTE =
  "Frozen dashboard baseline. Any subsequent change should be treated as a versioned update.";
export const DASHBOARD_CITATION_AUTHOR = "Boccolini, CS";
export const DASHBOARD_CITATION_TITLE =
  "Brazil's heat exposure on maternal, newborn, and child health - Dashboard";
export const DASHBOARD_DOI = "10.5281/zenodo.19325763";
export const DASHBOARD_SUGGESTED_CITATION = `${DASHBOARD_CITATION_AUTHOR}. ${DASHBOARD_CITATION_TITLE}. ${DASHBOARD_FROZEN_BASELINE_VERSION}. ${DASHBOARD_DOI}.`;
export const DASHBOARD_VERSION_LEDGER = [
  {
    version: "v1_2026_03_29",
    date: "2026-03-29",
    changeType: "major",
    summary: "Frozen baseline established for the public dashboard release.",
    details: [
      "Release boundary fixed for the dashboard baseline.",
      "Internal 32-outcome progression matrix published inside the app.",
      "Citation metadata appended to generated downloads.",
      "Methods page updated with release boundary and DOI status."
    ]
  }
] as const;
