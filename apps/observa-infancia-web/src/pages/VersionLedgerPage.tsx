import { DASHBOARD_VERSION_LEDGER } from "../lib/dashboardRelease";

export function VersionLedgerPage() {
  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <h2 className="text-2xl font-semibold">Version ledger</h2>
        <p className="max-w-5xl text-sm text-slate-600">
          This page records frozen dashboard releases and subsequent versioned updates. Small changes should increment a
          decimal point within the current major line, while larger changes should open a new major version.
        </p>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3 className="text-base font-semibold">Versioning rule</h3>
        </div>
        <div className="panel-body space-y-3 text-sm text-slate-700">
          <p>
            Use the frozen baseline as the anchor release. Small UI or wording updates should move from `1.0` to `1.1`,
            `1.2`, and so on. Larger analytical, structural, or release-scope changes should open a new major line.
          </p>
          <p>
            The current baseline is recorded as `v1_2026_03_29`, corresponding to the frozen dashboard state dated March
            29, 2026.
          </p>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3 className="text-base font-semibold">Recorded releases</h3>
        </div>
        <div className="panel-body space-y-4">
          {DASHBOARD_VERSION_LEDGER.map((entry) => (
            <article key={entry.version} className="rounded-2xl border border-rule bg-slate-50 p-4">
              <div className="flex flex-wrap items-center gap-3">
                <h4 className="text-base font-semibold text-navy">{entry.version}</h4>
                <span className="rounded-full bg-blue/10 px-3 py-1 text-xs font-medium uppercase tracking-wide text-blue">
                  {entry.changeType}
                </span>
                <span className="text-sm text-slate-500">{entry.date}</span>
              </div>
              <p className="mt-3 text-sm text-slate-700">{entry.summary}</p>
              <ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-slate-700">
                {entry.details.map((detail) => (
                  <li key={detail}>{detail}</li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
