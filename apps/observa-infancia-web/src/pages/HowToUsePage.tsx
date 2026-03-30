export function HowToUsePage() {
  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <h2 className="text-2xl font-semibold">How to use</h2>
        <p className="max-w-5xl text-sm text-slate-600">
          This dashboard is designed for descriptive exploration, surveillance reading, and communication support. It
          should be used to compare patterns, identify priorities, and review how retained heat-health signals behave
          across geography, time, and domain.
        </p>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3 className="text-base font-semibold">Recommended reading order</h3>
        </div>
        <div className="panel-body space-y-3 text-sm text-slate-700">
          <p>1. Start with the overview page to identify the main patterns and current indicators.</p>
          <p>2. Move to perinatal, child mortality, maternal, climate, and context pages to inspect domain-specific distributions.</p>
          <p>3. Use the relationships page to examine paired exposure-outcome patterns under the selected filters.</p>
          <p>4. Use the outcome matrix page to understand which signals remained principal, supportive, provisional, or outside the final retained set.</p>
          <p>5. Use the methods page whenever you need to verify indicator construction, lineage, release boundary, citation metadata, or display rules.</p>
          <p>6. Use the version ledger page to check the current frozen baseline and any future versioned updates.</p>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h3 className="text-base font-semibold">Interpretation guardrails</h3>
        </div>
        <div className="panel-body space-y-3 text-sm text-slate-700">
          <p>Use the dashboard as a surveillance and prioritisation surface, not as a source of final causal burden estimates.</p>
          <p>Read perinatal signals as delivery-aligned operational windows, not as whole-pregnancy etiologic effects.</p>
          <p>Read supportive and provisional signals with more caution than principal ones, especially when validation or benchmark analyses showed partial agreement or discordance.</p>
          <p>Use downloads with the embedded citation metadata whenever a table or figure is shared outside the dashboard.</p>
        </div>
      </section>
    </div>
  );
}
