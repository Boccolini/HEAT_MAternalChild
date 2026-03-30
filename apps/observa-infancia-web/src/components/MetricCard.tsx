import type { MetricCard as MetricCardType } from "../lib/api";

type MetricCardProps = {
  card: MetricCardType;
  active: boolean;
  onClick: () => void;
};

export function MetricCard({ card, active, onClick }: MetricCardProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "rounded-2xl border p-4 text-left transition-colors",
        active ? "border-navy bg-navy text-white" : "border-rule bg-white hover:border-blue"
      ].join(" ")}
    >
      <div className="text-xs uppercase tracking-wide opacity-75">{card.label}</div>
      <div className="mt-3 text-3xl font-semibold">
        {card.value === null ? "NA" : card.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
      </div>
      <div className="mt-1 text-sm opacity-80">{card.unit}</div>
      {card.period_label ? (
        <div className="mt-1 text-xs opacity-70">Reference period: {card.period_label}</div>
      ) : null}
      <div className="mt-3 text-xs opacity-80">
        {card.delta_vs_prior_period === null
          ? "No prior comparison"
          : `${card.delta_vs_prior_period >= 0 ? "+" : ""}${card.delta_vs_prior_period.toFixed(2)} vs prior period`}
      </div>
    </button>
  );
}
