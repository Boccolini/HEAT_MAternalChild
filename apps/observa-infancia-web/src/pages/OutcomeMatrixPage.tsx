import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { buildDownloadFilename, downloadCsv } from "../lib/download";

type OutcomeMatrixRow = {
  Outcome: string;
  analysis_unit_id: string;
  Domain: string;
  "Feasibility screen": string;
  "Primary analysis": string;
  "Selected exposure/window": string;
  "Context screen": string;
  "Publication stage": string;
  "Step7 public use": string;
  "Temporal validation": string;
  DLNM: string;
  "Case-crossover": string;
  "Final tier": string;
  "Decision note": string;
};

async function getOutcomeMatrix(): Promise<OutcomeMatrixRow[]> {
  const response = await fetch("/downloads/36b_full_outcome_progression_matrix.json");
  if (!response.ok) {
    throw new Error("Unable to load outcome progression matrix");
  }
  return response.json();
}

export function OutcomeMatrixPage() {
  const [searchTerm, setSearchTerm] = useState("");
  const [tierFilter, setTierFilter] = useState("all");
  const { data, error, isLoading } = useQuery({
    queryKey: ["outcome-matrix"],
    queryFn: getOutcomeMatrix
  });

  const visibleRows = useMemo(() => {
    const rows = data ?? [];
    const normalized = searchTerm.trim().toLowerCase();
    return rows.filter((row) => {
      const tierMatch = tierFilter === "all" || row["Final tier"] === tierFilter;
      const searchMatch =
        normalized.length === 0 ||
        [row.Outcome, row.analysis_unit_id, row.Domain, row["Final tier"], row["Decision note"]]
          .join(" ")
          .toLowerCase()
          .includes(normalized);
      return tierMatch && searchMatch;
    });
  }, [data, searchTerm, tierFilter]);

  if (isLoading) {
    return <div className="text-sm text-slate-500">Loading outcome progression matrix...</div>;
  }

  if (error || !data) {
    return <div className="text-sm text-caution">Unable to load the full outcome progression matrix.</div>;
  }

  function handleDownloadCsv() {
    downloadCsv(
      visibleRows,
      buildDownloadFilename("full-outcome-progression-matrix", "csv"),
      {
        title: "Full outcome progression matrix",
        notes: [
          "32 screened outcomes",
          "Frozen dashboard baseline release",
        ]
      }
    );
  }

  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <h2 className="text-2xl font-semibold">Outcome progression matrix</h2>
        <p className="max-w-5xl text-sm text-slate-600">
          This table shows how all 32 screened outcomes moved through the workflow, from feasibility screening to the
          final hierarchy. The final tier reflects the post-validation interpretation used in the current paper package.
        </p>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3 className="text-base font-semibold">Filters</h3>
        </div>
        <div className="panel-body grid gap-4 md:grid-cols-[minmax(0,1fr)_220px]">
          <label className="flex flex-col gap-2 text-sm">
            <span className="font-medium">Search outcome or note</span>
            <input
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
              className="rounded-xl border border-rule bg-white px-3 py-2"
              placeholder="e.g. neonatal, maternal, provisional"
            />
          </label>
          <label className="flex flex-col gap-2 text-sm">
            <span className="font-medium">Final tier</span>
            <select
              value={tierFilter}
              onChange={(event) => setTierFilter(event.target.value)}
              className="rounded-xl border border-rule bg-white px-3 py-2"
            >
              <option value="all">All outcomes</option>
              <option value="Principal">Principal</option>
              <option value="Supportive">Supportive</option>
              <option value="Provisional">Provisional</option>
              <option value="Not in final retained set">Not in final retained set</option>
            </select>
          </label>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h3 className="text-base font-semibold">Full screened roster</h3>
            <p className="text-sm text-slate-600">{visibleRows.length} outcomes shown</p>
          </div>
          <button
            type="button"
            onClick={handleDownloadCsv}
            className="rounded-full border border-rule bg-white px-3 py-1.5 text-sm font-medium text-navy transition-colors hover:border-blue hover:text-blue"
          >
            Download CSV
          </button>
        </div>
        <div className="panel-body overflow-auto">
          <table className="min-w-[1600px] text-sm">
            <thead className="sticky top-0 z-10 text-left text-slate-500">
              <tr>
                {[
                  "Outcome",
                  "Domain",
                  "Feasibility screen",
                  "Primary analysis",
                  "Selected exposure/window",
                  "Context screen",
                  "Publication stage",
                  "Step7 public use",
                  "Temporal validation",
                  "DLNM",
                  "Case-crossover",
                  "Final tier",
                  "Decision note"
                ].map((header) => (
                  <th key={header} className="border-b border-rule bg-slate-50 px-3 py-3 align-top font-medium shadow-sm">
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {visibleRows.map((row) => (
                <tr key={row.analysis_unit_id} className="border-t border-rule align-top">
                  <td className="px-3 py-3 font-medium text-navy">{row.Outcome}</td>
                  <td className="px-3 py-3">{row.Domain}</td>
                  <td className="px-3 py-3">{row["Feasibility screen"]}</td>
                  <td className="px-3 py-3">{row["Primary analysis"]}</td>
                  <td className="px-3 py-3">{row["Selected exposure/window"]}</td>
                  <td className="px-3 py-3">{row["Context screen"]}</td>
                  <td className="px-3 py-3">{row["Publication stage"]}</td>
                  <td className="px-3 py-3">{row["Step7 public use"]}</td>
                  <td className="px-3 py-3">{row["Temporal validation"]}</td>
                  <td className="px-3 py-3">{row.DLNM}</td>
                  <td className="px-3 py-3">{row["Case-crossover"]}</td>
                  <td className="px-3 py-3">{row["Final tier"]}</td>
                  <td className="px-3 py-3 text-slate-600">{row["Decision note"]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
