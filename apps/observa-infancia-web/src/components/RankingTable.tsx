import type { RankingRow } from "../lib/api";
import { PanelActionButton } from "./PanelActionButton";
import { PanelDownloadActions } from "./PanelDownloadActions";
import { buildDownloadFilename, downloadCsv, type DownloadMetadata } from "../lib/download";

type RankingTableProps = {
  title: string;
  rows: RankingRow[];
  onSelectMunicipality: (municipalityCode: string) => void;
  downloadMetadata?: DownloadMetadata;
};

export function RankingTable({ title, rows, onSelectMunicipality, downloadMetadata }: RankingTableProps) {
  const visibleRows = rows.slice(0, 15);
  return (
    <section className="panel">
      <div className="panel-header">
        <h3 className="text-base font-semibold">{title}</h3>
        <PanelDownloadActions>
          <PanelActionButton
            label="CSV"
            onClick={() =>
              downloadCsv(
                visibleRows.map((row) => ({
                  rank: row.rank,
                  geo_code: row.geo_code,
                  geography: row.label,
                  value: row.value,
                  denominator: row.denominator,
                  quality_flag_any: row.quality_flag_any,
                })),
                buildDownloadFilename(title, "csv"),
                downloadMetadata,
              )
            }
          />
        </PanelDownloadActions>
      </div>
      <div className="panel-body overflow-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-slate-500">
            <tr>
              <th className="pb-3">Rank</th>
              <th className="pb-3">Geography</th>
              <th className="pb-3 text-right">Value</th>
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row) => (
              <tr key={row.geo_code} className="border-t border-rule">
                <td className="py-3">{row.rank}</td>
                <td className="py-3">
                  <button
                    type="button"
                    className="text-left text-blue hover:underline"
                    onClick={() => row.geo_code.length === 7 && onSelectMunicipality(row.geo_code)}
                  >
                    {row.label}
                  </button>
                </td>
                <td className="py-3 text-right">{row.value === null ? "NA" : row.value.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
