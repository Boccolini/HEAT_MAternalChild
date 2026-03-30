import { Outlet } from "react-router-dom";
import type { PageDefinition } from "../lib/api";
import { DASHBOARD_FROZEN_BASELINE_DATE, DASHBOARD_FROZEN_BASELINE_VERSION } from "../lib/dashboardRelease";
import { TopNav } from "./TopNav";

type AppShellProps = {
  pages: PageDefinition[];
};

export function AppShell({ pages }: AppShellProps) {
  return (
    <div className="min-h-screen bg-canvas">
      <header className="border-b border-rule bg-white">
        <div className="mx-auto flex max-w-[1600px] flex-col gap-4 px-6 py-5">
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-blue">ICICT / Fiocruz</p>
            <h1 className="mt-2 text-2xl font-semibold text-navy">
              Child Health Observatory (Observa Infancia)
            </h1>
            <p className="mt-1 max-w-3xl text-sm text-slate-600">
              Brazil&apos;s heat exposure on maternal, newborn, and child health.
            </p>
            <p className="mt-2 text-xs text-slate-500">
              Frozen baseline {DASHBOARD_FROZEN_BASELINE_VERSION} | {DASHBOARD_FROZEN_BASELINE_DATE}
            </p>
          </div>
          <TopNav pages={pages} />
        </div>
      </header>
      <main className="mx-auto max-w-[1600px] px-6 py-6">
        <Outlet />
      </main>
    </div>
  );
}
