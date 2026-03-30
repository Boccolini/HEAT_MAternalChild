import type { IndicatorTechnicalSheet, StratifierTechnicalSheet } from "../lib/api";

type TechnicalSheetCatalogProps = {
  indicatorSheets: IndicatorTechnicalSheet[];
  stratifierSheets: StratifierTechnicalSheet[];
  activeSheetId?: string | null;
  searchTerm: string;
  onSearchTermChange: (value: string) => void;
  typeFilter: "all" | "indicators" | "stratifiers";
  onTypeFilterChange: (value: "all" | "indicators" | "stratifiers") => void;
  familyFilter: string;
  onFamilyFilterChange: (value: string) => void;
  onSelectSheet: (sheetId: string) => void;
};

type CatalogSection = {
  title: string;
  items: Array<IndicatorTechnicalSheet | StratifierTechnicalSheet>;
};

function matchesSearch(
  sheet: IndicatorTechnicalSheet | StratifierTechnicalSheet,
  searchTerm: string,
): boolean {
  if (!searchTerm.trim()) {
    return true;
  }
  const needle = searchTerm.trim().toLowerCase();
  const haystack = [
    sheet.label,
    sheet.sheet_id,
    sheet.public_definition,
    sheet.sheet_type === "indicator" ? sheet.metric_family : "context",
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(needle);
}

function ChipButton({
  sheet,
  active,
  onClick,
}: {
  sheet: IndicatorTechnicalSheet | StratifierTechnicalSheet;
  active: boolean;
  onClick: () => void;
}) {
  const usage =
    sheet.sheet_type === "indicator"
      ? sheet.page_usage.slice(0, 3).map((item) => item.replace("_", " "))
      : sheet.applicable_pages.slice(0, 3).map((item) => item.replace("_", " "));

  return (
    <button
      type="button"
      className={[
        "rounded-2xl border px-4 py-3 text-left transition-colors",
        active ? "border-navy bg-navy text-white" : "border-rule bg-white hover:border-blue hover:text-blue",
      ].join(" ")}
      onClick={onClick}
    >
      <div className="font-medium">{sheet.label}</div>
      <div className={active ? "mt-1 text-[11px] opacity-80" : "mt-1 text-[11px] text-slate-500"}>{sheet.sheet_id}</div>
      <div className="mt-2 flex flex-wrap gap-1">
        {usage.map((item) => (
          <span
            key={`${sheet.sheet_id}-${item}`}
            className={active ? "rounded-full border border-white/30 px-2 py-1 text-[11px]" : "rounded-full border border-rule px-2 py-1 text-[11px] text-slate-600"}
          >
            {item}
          </span>
        ))}
      </div>
    </button>
  );
}

export function TechnicalSheetCatalog({
  indicatorSheets,
  stratifierSheets,
  activeSheetId,
  searchTerm,
  onSearchTermChange,
  typeFilter,
  onTypeFilterChange,
  familyFilter,
  onFamilyFilterChange,
  onSelectSheet,
}: TechnicalSheetCatalogProps) {
  const familyOptions = ["all", "overview", "perinatal", "mortality", "child_morbidity", "maternal", "climate", "context"];

  const filteredIndicators = indicatorSheets.filter((sheet) => {
    if (typeFilter === "stratifiers") {
      return false;
    }
    if (familyFilter !== "all" && familyFilter !== "overview" && sheet.metric_family !== familyFilter) {
      return false;
    }
    if (familyFilter === "overview" && !(sheet.page_id === "overview" || sheet.page_usage.includes("overview"))) {
      return false;
    }
    return matchesSearch(sheet, searchTerm);
  });

  const filteredStratifiers = stratifierSheets.filter((sheet) => {
    if (typeFilter === "indicators") {
      return false;
    }
    if (familyFilter !== "all" && familyFilter !== "context") {
      return false;
    }
    return matchesSearch(sheet, searchTerm);
  });

  const sections: CatalogSection[] = [
    {
      title: "Overview and headline KPIs",
      items: filteredIndicators.filter((sheet) => sheet.page_id === "overview" || sheet.page_usage.includes("overview")),
    },
    {
      title: "Perinatal indicators",
      items: filteredIndicators.filter((sheet) => sheet.metric_family === "perinatal" && !sheet.page_usage.includes("overview")),
    },
    {
      title: "Child mortality indicators",
      items: filteredIndicators.filter((sheet) => sheet.metric_family === "mortality"),
    },
    {
      title: "Child morbidity and admissions",
      items: filteredIndicators.filter((sheet) => sheet.metric_family === "child_morbidity"),
    },
    {
      title: "Maternal morbidity indicators",
      items: filteredIndicators.filter((sheet) => sheet.metric_family === "maternal"),
    },
    {
      title: "Climate indicators",
      items: filteredIndicators.filter((sheet) => sheet.metric_family === "climate"),
    },
    {
      title: "Context stratifiers",
      items: filteredStratifiers,
    },
  ].filter((section) => section.items.length > 0);

  return (
    <section className="panel">
      <div className="panel-header">
        <h3 className="text-base font-semibold">Technical sheet catalog</h3>
      </div>
      <div className="panel-body space-y-6">
        <div className="grid gap-4 md:grid-cols-3">
          <label className="space-y-2 text-sm">
            <span className="text-xs uppercase tracking-wide text-slate-500">Search</span>
            <input
              value={searchTerm}
              onChange={(event) => onSearchTermChange(event.target.value)}
              placeholder="Search indicator, field, or concept"
              className="w-full rounded-xl border border-rule bg-white px-3 py-2"
            />
          </label>
          <label className="space-y-2 text-sm">
            <span className="text-xs uppercase tracking-wide text-slate-500">Type</span>
            <select
              value={typeFilter}
              onChange={(event) => onTypeFilterChange(event.target.value as "all" | "indicators" | "stratifiers")}
              className="w-full rounded-xl border border-rule bg-white px-3 py-2"
            >
              <option value="all">All</option>
              <option value="indicators">Indicators</option>
              <option value="stratifiers">Stratifiers</option>
            </select>
          </label>
          <label className="space-y-2 text-sm">
            <span className="text-xs uppercase tracking-wide text-slate-500">Family</span>
            <select
              value={familyFilter}
              onChange={(event) => onFamilyFilterChange(event.target.value)}
              className="w-full rounded-xl border border-rule bg-white px-3 py-2"
            >
              {familyOptions.map((option) => (
                <option key={option} value={option}>
                  {option === "all" ? "All families" : option.replace("_", " ")}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="space-y-6">
          {sections.map((section) => (
            <div key={section.title} className="space-y-3">
              <h4 className="text-sm font-semibold uppercase tracking-wide text-slate-500">{section.title}</h4>
              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                {section.items.map((sheet) => (
                  <ChipButton
                    key={sheet.sheet_id}
                    sheet={sheet}
                    active={sheet.sheet_id === activeSheetId}
                    onClick={() => onSelectSheet(sheet.sheet_id)}
                  />
                ))}
              </div>
            </div>
          ))}
          {sections.length === 0 ? (
            <div className="rounded-2xl border border-rule p-4 text-sm text-slate-600">
              No technical sheets match the current search and filter combination.
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}
