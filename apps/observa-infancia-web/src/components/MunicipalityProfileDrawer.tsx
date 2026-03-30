import { useQuery } from "@tanstack/react-query";
import { getMunicipalityProfile, type MunicipalityProfile } from "../lib/api";

type MunicipalityProfileDrawerProps = {
  municipalityCode: string | null;
  onClose: () => void;
};

function ProfileSection({ title, rows }: { title: string; rows: MunicipalityProfile["overview"] }) {
  return (
    <section className="space-y-2">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">{title}</h3>
      <div className="space-y-2">
        {rows.slice(0, 8).map((row) => (
          <div key={`${row.metric_id}-${row.year}`} className="rounded-xl border border-rule p-3 text-sm">
            <div className="font-medium">{row.metric_id}</div>
            <div className="text-slate-500">Year {row.year}</div>
            <div className="mt-1">
              Value {row.value_rate ?? row.value_count ?? "NA"}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

export function MunicipalityProfileDrawer({ municipalityCode, onClose }: MunicipalityProfileDrawerProps) {
  const { data, isLoading } = useQuery({
    queryKey: ["municipality-profile", municipalityCode],
    queryFn: () => getMunicipalityProfile(municipalityCode ?? ""),
    enabled: Boolean(municipalityCode)
  });

  if (!municipalityCode) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/20">
      <div className="h-full w-full max-w-2xl overflow-auto bg-white p-6 shadow-2xl">
        <div className="mb-6 flex items-start justify-between">
          <div>
            <h2 className="text-xl font-semibold">{data?.label ?? "Municipality profile"}</h2>
            <p className="text-sm text-slate-500">{data?.uf ?? ""} {municipalityCode}</p>
          </div>
          <button type="button" className="rounded-full border border-rule px-3 py-1 text-sm" onClick={onClose}>
            Close
          </button>
        </div>
        {isLoading && <p className="text-sm text-slate-500">Loading municipality profile...</p>}
        {data && (
          <div className="space-y-6">
            <section className="rounded-2xl border border-rule p-4">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Context</h3>
              <div className="mt-3 overflow-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-slate-500">
                      <th>Year</th>
                      <th>IDH</th>
                      <th>Gini</th>
                      <th>Urban</th>
                      <th>Altitude</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.context.years.map((row: Record<string, unknown>) => (
                      <tr key={String(row.year)} className="border-t border-rule">
                        <td className="py-2">{String(row.year)}</td>
                        <td className="py-2">{Number(row.IDH_EXPANDED ?? 0).toFixed(3)}</td>
                        <td className="py-2">{Number(row.GINI_EXPANDED ?? 0).toFixed(3)}</td>
                        <td className="py-2">{Number(row.PROP_URBAN ?? 0).toFixed(2)}</td>
                        <td className="py-2">{Number(row.ALTITUDE_MEAN_M ?? 0).toFixed(0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
            <ProfileSection title="Overview" rows={data.overview} />
            <ProfileSection title="Perinatal" rows={data.perinatal} />
            <ProfileSection title="Child mortality" rows={data.mortality} />
            <ProfileSection title="Maternal" rows={data.maternal} />
            <ProfileSection title="Climate" rows={data.climate} />
          </div>
        )}
      </div>
    </div>
  );
}
