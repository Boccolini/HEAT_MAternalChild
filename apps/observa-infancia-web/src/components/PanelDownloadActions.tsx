import type { ReactNode } from "react";

type PanelDownloadActionsProps = {
  children: ReactNode;
};

export function PanelDownloadActions({ children }: PanelDownloadActionsProps) {
  return (
    <div className="flex flex-col items-end gap-1">
      <span className="text-[11px] font-medium uppercase tracking-[0.14em] text-slate-500">Download</span>
      <div className="flex items-center gap-2">{children}</div>
    </div>
  );
}
