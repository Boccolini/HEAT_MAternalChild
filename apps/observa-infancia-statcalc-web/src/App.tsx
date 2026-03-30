import { FormEvent, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createPreset,
  deletePreset,
  getPresets,
  getRecipes,
  getRun,
  getRunResult,
  getRuns,
  getStratifiers,
  postRun,
  type Preset,
  type Recipe,
  type Run,
  type RunResult,
  type RunSpec,
  type SelectorOption,
  type Stratifier,
} from "./lib/api";

type HelpTab = "quick" | "methodology";
type HelpTopicId =
  | "track_a"
  | "track_b"
  | "benchmark"
  | "recipe"
  | "template"
  | "outcome"
  | "exposure"
  | "lag_window"
  | "stratifier"
  | "source_filter"
  | "grouping"
  | "gestational_source"
  | "offset"
  | "reference_live"
  | "interpretation";

type HelpTopic = {
  title: string;
  quick: string;
  methodology: string;
};

type EffectDescriptor = {
  label: string;
  shorthand: string;
  nullValue: number;
  interpretationGuide: string;
  previewTerm: string;
};

const EMPTY_SPEC: RunSpec = {
  recipe_id: "",
  outcome_variable: "",
  exposure_variable: "",
  lag_or_window: "",
  template_id: "",
  stratifier_id: "",
  source_filter: "all_sources",
  denominator_variant: "",
  grouping_template: "",
  gestational_source_mode: "",
  offset_id: "",
  adjustment_context_variables: [],
};

const HELP_TOPICS: Record<HelpTopicId, HelpTopic> = {
  track_a: {
    title: "Aggregated current-cycle stream",
    quick: "This stream covers municipality-period analyses fit from validated aggregated marts that are already ready in the current gold cycle. It is designed for rapid local execution and typically returns results immediately when no exact reference artifact is available.",
    methodology:
      "Within this application, the aggregated current-cycle stream represents the municipality-period fixed-effects analyses that can be fit directly from validated marts. The unit of analysis is a municipality-period panel constructed from harmonised analytical marts rather than individual event records. Because these analytical objects are already collapsed and quality-gated, the stream is computationally lighter and therefore executes synchronously when a specification does not match an existing exported reference estimate.",
  },
  track_b: {
    title: "Event-level escalation stream",
    quick: "This stream covers heavier event-level or grouped event-level perinatal analyses that should be used after the aggregated stream identifies a priority signal. Live models are queued locally because the underlying data structures and estimation routines are more demanding.",
    methodology:
      "The event-level escalation stream operationalises the deeper perinatal modelling layer. The application first attempts to resolve the requested specification against frozen exported model tables. If no exact match exists, the run is sent to a local worker that executes the grouped event-level model in a dedicated subprocess, preserving the separation between interactive browsing and computationally expensive escalation analyses.",
  },
  benchmark: {
    title: "Benchmark",
    quick: "A benchmark specification is the principal validated model configuration used as the main analytical reference within a stream.",
    methodology:
      "In this interface, a benchmark configuration denotes the prespecified main model architecture judged to be the primary inferential baseline for a given analytical family. Sensitivity analyses, subgroup analyses, and source restrictions are interpreted relative to this benchmark rather than as free-standing replacements for the main design.",
  },
  recipe: {
    title: "Recipe",
    quick: "A recipe is a validated analytical envelope: it defines which outcomes, exposures, offsets, windows, and execution rules are allowed together.",
    methodology:
      "Recipes are not arbitrary model blueprints. Each recipe encodes a constrained set of combinations that are already coherent with the validated analytical streams. This ensures that users can customise within an approved design space while still preserving reproducibility, denominator compatibility, and the intended distinction between benchmark, sensitivity, heterogeneity, and escalation analyses.",
  },
  template: {
    title: "Template",
    quick: "A template is the prespecified model architecture applied within a recipe, such as the benchmark fixed-effects proxy model.",
    methodology:
      "Templates specify the analytical architecture rather than the substantive variables alone. They determine the modelling family, the intended role of offsets, and the structural logic that links the selected exposure and outcome to a valid inferential design. In this v1 interface, templates remain intentionally constrained in order to protect comparability across the approved incremental analysis tiers.",
  },
  outcome: {
    title: "Outcome",
    quick: "The outcome is the endpoint being modelled, such as neonatal mortality, preterm birth, or gestational days lost.",
    methodology:
      "Outcome selection determines more than the dependent variable. It also constrains denominator compatibility, offset selection, admissible recipes, and the likely effect scale. For that reason, this application treats outcome choice as a structural decision rather than a cosmetic label change.",
  },
  exposure: {
    title: "Exposure",
    quick: "The exposure is the climatic or heat-related measure whose association with the outcome will be estimated.",
    methodology:
      "Exposure definitions in this application are taken from validated analytical families rather than from free-form formulas. The selected exposure may represent acute temperature, trimester-specific averages, heatwave summaries, or interaction proxies, and the exact interpretation of the coefficient depends on both the exposure definition and the analytical template in which it is embedded.",
  },
  lag_window: {
    title: "Lag / Window",
    quick: "Lag or window selection determines the temporal alignment between the exposure metric and the outcome process.",
    methodology:
      "Temporal alignment is analytically substantive. Acute mortality models may use short distributed lags, while perinatal models may use delivery-aligned windows or trimester-specific summaries. The lag/window selector therefore changes the estimand itself by redefining the exposure period to which the outcome is being related.",
  },
  stratifier: {
    title: "Stratifier",
    quick: "A stratifier requests separate estimates across validated contextual subgroups; it does not move contextual variables into the adjustment set.",
    methodology:
      "This application restricts contextual variables to heterogeneity analysis rather than primary adjustment. Selecting a stratifier yields stratum-specific estimates that facilitate the assessment of effect heterogeneity across contextual municipal characteristics while preserving the analytical rule that contextual variables should not be inserted as ordinary adjustment covariates in the main model.",
  },
  source_filter: {
    title: "Source Filter",
    quick: "The source filter restricts the analytical sample according to approved data-source or quality rules, such as excluding fallback-imputed climate rows.",
    methodology:
      "Source filters operationalise prespecified sensitivity choices. They should be interpreted as analytic restrictions on the admissible sample, not as arbitrary cleaning decisions. Inference under a source filter is therefore a sensitivity result that is best read relative to the benchmark configuration.",
  },
  grouping: {
    title: "Grouping Template",
    quick: "The grouping template defines the approved aggregation or fixed-effects grouping structure used by the recipe.",
    methodology:
      "Grouping is part of the design logic of the model. It determines how records are collapsed or indexed for estimation and therefore affects identifiability, variance structure, and interpretability. The application exposes only grouping templates that remain coherent with the validated recipe-specific design.",
  },
  gestational_source: {
    title: "Gestational Source",
    quick: "This option controls which gestational-age source definitions are admitted in event-level escalation models.",
    methodology:
      "Gestational-age source selection is a substantive data-definition choice in the event-level escalation stream. Restricting to exact measurements may improve measurement specificity while reducing sample size, whereas broader source inclusion may preserve coverage at the cost of greater heterogeneity in ascertainment. The application keeps this option explicit because it changes the analytic population.",
  },
  offset: {
    title: "Offset",
    quick: "The offset is applied automatically to preserve denominator compatibility with the chosen outcome. It is displayed for transparency but not manually edited.",
    methodology:
      "Offsets are not exposed as free user controls because they are structurally tied to the outcome definition. In count-type models, the offset anchors the regression to the relevant population or birth denominator, thereby converting the model from a raw-count analysis into a rate-based design. Preventing manual offset reassignment protects interpretability and prevents denominator mismatch.",
  },
  reference_live: {
    title: "Reference vs Live",
    quick: "If the specification exactly matches an exported analytical artifact, the app returns the reference result immediately. Otherwise it fits the corresponding live model locally.",
    methodology:
      "The application implements a dual execution path. Reference mode is a strict reproducibility shortcut: it resolves only exact matches to frozen exported results. Live mode is used when the requested specification lies within the validated recipe but is not already present as a frozen artifact. This design preserves both fast retrieval of established results and local flexibility for approved customisations.",
  },
  interpretation: {
    title: "Interpretation",
    quick: "The interpretation panel converts the numerical result into academic English, explaining the effect direction, scale, uncertainty, and any stratified structure.",
    methodology:
      "Interpretation is generated from the returned effect estimate, confidence interval, p-value, and the selected analytical context. The text is descriptive rather than causal: it frames the result as an estimated association under the chosen specification, notes whether uncertainty intervals exclude the null reference, and highlights when subgroup estimates should be read as heterogeneity evidence rather than a single pooled contrast.",
  },
};

const KEY_CONCEPTS: HelpTopicId[] = ["track_a", "track_b", "benchmark", "recipe", "reference_live", "offset"];

function resolveOffsetId(recipe: Recipe | undefined, outcomeVariable: string): string {
  if (!recipe) {
    return "";
  }
  const selectedOutcome = recipe.outcomes.find((item) => item.id === outcomeVariable);
  return String(selectedOutcome?.metadata?.offset_id ?? recipe.default_offset_id ?? "");
}

function normalizeSpecForRecipe(recipe: Recipe | undefined, draftSpec: RunSpec): RunSpec {
  if (!recipe) {
    return draftSpec;
  }
  return {
    ...draftSpec,
    offset_id: resolveOffsetId(recipe, draftSpec.outcome_variable),
  };
}

function buildDefaultSpec(recipe: Recipe | undefined): RunSpec {
  if (!recipe) {
    return EMPTY_SPEC;
  }
  const outcomeVariable = recipe.outcomes[0]?.id ?? "";
  return {
    recipe_id: recipe.recipe_id,
    outcome_variable: outcomeVariable,
    exposure_variable: recipe.exposures[0]?.id ?? "",
    lag_or_window: recipe.lag_windows[0]?.id ?? String(recipe.exposures[0]?.metadata?.lag_or_window ?? ""),
    template_id: recipe.default_template_id,
    stratifier_id: "",
    source_filter: recipe.source_filters[0]?.id ?? "all_sources",
    denominator_variant: recipe.denominator_variants[0]?.id ?? "",
    grouping_template: recipe.grouping_templates[0]?.id ?? "",
    gestational_source_mode: recipe.gestational_source_modes[0]?.id ?? "",
    offset_id: resolveOffsetId(recipe, outcomeVariable),
    adjustment_context_variables: [],
  };
}

function formatNumber(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return "NA";
  }
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) {
    return value.toExponential(3);
  }
  return value.toFixed(4);
}

function formatPValue(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return "NA";
  }
  if (value < 0.001) {
    return "<0.001";
  }
  if (value < 0.01) {
    return value.toFixed(3);
  }
  return value.toFixed(2);
}

function runIsPending(run: Run | undefined): boolean {
  return run?.status === "queued" || run?.status === "running";
}

function optionLabel(options: SelectorOption[] | undefined, id: string | null | undefined, fallback = ""): string {
  if (!id) {
    return fallback;
  }
  return options?.find((item) => item.id === id)?.label ?? id;
}

function humanizeFallback(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\bfe\b/gi, "fixed-effects")
    .replace(/\btmax\b/gi, "maximum temperature")
    .replace(/\brh\b/gi, "relative humidity")
    .replace(/\bga\b/gi, "gestational age")
    .replace(/\s+/g, " ")
    .trim();
}

function naturalRecipeLabel(value: string, recipeId = ""): string {
  const normalized = `${recipeId} ${value}`.toLowerCase();
  if (normalized.includes("track_a_acute")) {
    return "Aggregated acute outcome analysis";
  }
  if (normalized.includes("track_a_perinatal")) {
    return "Aggregated perinatal outcome analysis";
  }
  if (normalized.includes("track_a_maternal")) {
    return "Aggregated maternal outcome analysis";
  }
  if (normalized.includes("track_b_preterm")) {
    return "Event-level preterm birth analysis";
  }
  if (normalized.includes("track_b_stillbirth")) {
    return "Event-level stillbirth analysis";
  }
  if (normalized.includes("track_b_neonatal")) {
    return "Event-level neonatal analysis";
  }
  return value;
}

function naturalTemplateLabel(value: string): string {
  const normalized = value.toLowerCase();
  if (normalized.includes("benchmark") && (normalized.includes("fixed-effects") || normalized.includes("fe"))) {
    return "Primary fixed-effects model";
  }
  if (normalized.includes("flagship") && normalized.includes("grouped")) {
    return "Primary grouped fixed-effects model";
  }
  return value
    .replace(/\bbenchmark\b/gi, "Primary")
    .replace(/\bflagship\b/gi, "Primary")
    .replace(/fixed-effects proxy/gi, "fixed-effects model")
    .replace(/\bfe proxy\b/gi, "fixed-effects model")
    .replace(/\bproxy\b/gi, "model")
    .replace(/\bFE\b/g, "fixed-effects");
}

function naturalLagLabel(value: string): string {
  const normalized = value.toLowerCase().replace(/\s+/g, "_");
  const simpleLag = normalized.match(/^lag(\d+)$/);
  if (simpleLag) {
    const lag = Number(simpleLag[1]);
    return lag === 0 ? "Same-day exposure" : `Exposure ${lag} day${lag === 1 ? "" : "s"} earlier`;
  }
  const meanLag = normalized.match(/^lag0_(\d+)_mean(?:_[a-z0-9]+(?:_[a-z0-9]+)*)?$/);
  if (meanLag) {
    const lag = Number(meanLag[1]);
    return lag === 1
      ? "Average exposure across the current day and previous day"
      : `Average exposure across the current day and previous ${lag} days`;
  }
  const eventWindow = normalized.match(/^days_0_(\d+)_before_event(?:_[a-z0-9]+(?:_[a-z0-9]+)*)?$/);
  if (eventWindow) {
    const days = Number(eventWindow[1]) + 1;
    return `Average exposure during the ${days}-day period before the event`;
  }
  if (normalized === "third_trimester") {
    return "Average exposure during the third trimester";
  }
  if (normalized === "whole_pregnancy") {
    return "Average exposure across the whole pregnancy";
  }
  return humanizeFallback(value);
}

function naturalOutcomeLabel(value: string, outcomeId = ""): string {
  if (value && !value.includes("_")) {
    return value;
  }
  return humanizeFallback(value || outcomeId);
}

function naturalExposureLabel(value: string, exposureId = ""): string {
  if (value && !value.includes("_")) {
    return value;
  }
  return humanizeFallback(value || exposureId);
}

function naturalOffsetLabel(value: string): string {
  const normalized = value.toLowerCase();
  if (normalized.includes("automatic")) {
    return "Automatic denominator selected by recipe";
  }
  if (normalized.includes("daily live-birth")) {
    return "Daily live-birth denominator";
  }
  if (normalized.includes("annual live-birth") || normalized.includes("live births")) {
    return "Live-birth denominator";
  }
  if (normalized.includes("under-five population")) {
    return "Under-five population denominator";
  }
  if (normalized.includes("total-birth")) {
    return "Total-birth denominator";
  }
  if (normalized.includes("women aged 15-49")) {
    return "Women aged 15-49 denominator";
  }
  if (normalized.includes("women aged 15-45")) {
    return "Women aged 15-45 denominator";
  }
  return value
    .replace(/\blog\b/gi, "")
    .replace(/\boffset proxy\b/gi, "denominator")
    .replace(/\boffset\b/gi, "denominator")
    .replace(/\s+/g, " ")
    .trim();
}

function naturalSourceFilterLabel(value: string): string {
  const normalized = value.toLowerCase();
  if (normalized.includes("all sources") || normalized.includes("all climate rows") || normalized.includes("all observed rows")) {
    return "Use all eligible rows";
  }
  if (normalized.includes("no fallback")) {
    return "Exclude rows that rely on fallback-imputed climate data";
  }
  return humanizeFallback(value);
}

function naturalGroupingLabel(value: string): string {
  const normalized = value.toLowerCase();
  if (normalized.includes("municipality-year")) {
    return "Municipality-by-year comparison structure";
  }
  if (normalized.includes("flagship")) {
    return "Predefined grouped comparison structure";
  }
  return humanizeFallback(value);
}

function naturalGestationalSourceLabel(value: string): string {
  const normalized = value.toLowerCase();
  if (normalized.includes("all gestational-age sources")) {
    return "Use all available gestational-age sources";
  }
  if (normalized.includes("exact gestational-age measurements only")) {
    return "Restrict to exact gestational-age measurements";
  }
  return humanizeFallback(value);
}

function naturalModelLabel(value: string): string {
  const normalized = value.toLowerCase();
  if (normalized.includes("modified_poisson")) {
    return "Modified Poisson fixed-effects model";
  }
  if (normalized.includes("poisson") && normalized.includes("total_birth")) {
    return "Poisson fixed-effects model with a total-birth denominator";
  }
  if (normalized.includes("poisson")) {
    return "Poisson fixed-effects model";
  }
  if (normalized.includes("gaussian")) {
    return "Linear fixed-effects model";
  }
  return humanizeFallback(value)
    .replace(/\bproxy\b/gi, "model")
    .replace(/\bfe\b/gi, "fixed-effects");
}

function naturalExecutionModeLabel(value: string): string {
  if (value === "reference") {
    return "Reference result";
  }
  if (value === "live") {
    return "Live model";
  }
  return humanizeFallback(value);
}

function currentExposureLabel(recipe: Recipe | undefined, spec: RunSpec): string {
  return naturalExposureLabel(optionLabel(recipe?.exposures, spec.exposure_variable, spec.exposure_variable), spec.exposure_variable);
}

function currentOutcomeLabel(recipe: Recipe | undefined, spec: RunSpec): string {
  return naturalOutcomeLabel(optionLabel(recipe?.outcomes, spec.outcome_variable, spec.outcome_variable), spec.outcome_variable);
}

function currentTemplateLabel(recipe: Recipe | undefined, spec: RunSpec): string {
  return naturalTemplateLabel(optionLabel(recipe?.templates, spec.template_id, spec.template_id));
}

function currentLagLabel(recipe: Recipe | undefined, spec: RunSpec): string {
  if (!spec.lag_or_window) {
    return "Recipe-defined exposure timing";
  }
  return naturalLagLabel(optionLabel(recipe?.lag_windows, spec.lag_or_window, spec.lag_or_window));
}

function currentOffsetLabel(recipe: Recipe | undefined, spec: RunSpec): string {
  const resolvedOffsetId = resolveOffsetId(recipe, spec.outcome_variable);
  return naturalOffsetLabel(optionLabel(recipe?.offsets, resolvedOffsetId, resolvedOffsetId));
}

function currentStratifierLabel(stratifiers: Stratifier[], spec: RunSpec): string {
  if (!spec.stratifier_id) {
    return "No stratification";
  }
  return stratifiers.find((item) => item.stratifier_id === spec.stratifier_id)?.label ?? spec.stratifier_id;
}

function currentSourceFilterLabel(recipe: Recipe | undefined, spec: RunSpec): string {
  return naturalSourceFilterLabel(optionLabel(recipe?.source_filters, spec.source_filter, spec.source_filter));
}

function currentGroupingLabel(recipe: Recipe | undefined, spec: RunSpec): string {
  if (!spec.grouping_template) {
    return "Recipe default grouping";
  }
  return naturalGroupingLabel(optionLabel(recipe?.grouping_templates, spec.grouping_template, spec.grouping_template));
}

function currentGestationalSourceLabel(recipe: Recipe | undefined, spec: RunSpec): string {
  if (!spec.gestational_source_mode) {
    return "Recipe default gestational source";
  }
  return naturalGestationalSourceLabel(optionLabel(recipe?.gestational_source_modes, spec.gestational_source_mode, spec.gestational_source_mode));
}

function effectDescriptorFromScale(estimateScale: string | undefined, outcomeVariable: string): EffectDescriptor {
  if (estimateScale?.includes("mean_difference") || outcomeVariable.includes("gestational_days_lost")) {
    return {
      label: "Mean difference",
      shorthand: "MD",
      nullValue: 0,
      interpretationGuide: "Values above 0 indicate a higher mean outcome; values below 0 indicate a lower mean outcome.",
      previewTerm: "difference in the mean outcome",
    };
  }
  return {
    label: "Rate ratio",
    shorthand: "RR",
    nullValue: 1,
    interpretationGuide: "Values above 1 indicate a higher rate; values below 1 indicate a lower rate.",
    previewTerm: "relative rate of the outcome",
  };
}

function previewTrackLabel(recipe: Recipe | undefined): string {
  if (!recipe) {
    return "the selected analytical stream";
  }
  return recipe.track_id === "track_a" ? "the aggregated current-cycle stream" : "the event-level escalation stream";
}

function primaryResultRow(result: RunResult): RunResult["rows"][number] | undefined {
  return result.rows.find((row) => row.primary_row_flag) ?? result.rows[0];
}

function buildPreviewNarrative(recipe: Recipe | undefined, spec: RunSpec, stratifiers: Stratifier[]) {
  const outcomeLabel = currentOutcomeLabel(recipe, spec);
  const exposureLabel = currentExposureLabel(recipe, spec);
  const templateLabel = currentTemplateLabel(recipe, spec);
  const lagLabel = currentLagLabel(recipe, spec);
  const offsetLabel = currentOffsetLabel(recipe, spec);
  const stratifierLabel = currentStratifierLabel(stratifiers, spec);
  const sourceFilterLabel = currentSourceFilterLabel(recipe, spec);
  const groupingLabel = currentGroupingLabel(recipe, spec);
  const gestationalLabel = currentGestationalSourceLabel(recipe, spec);
  const effect = effectDescriptorFromScale(undefined, spec.outcome_variable);
  const trackLabel = previewTrackLabel(recipe);
  const executionText = recipe?.track_id === "track_b"
    ? "If no exact reference artifact is available, the live model will be queued and executed by the local event-level worker."
    : "If no exact reference artifact is available, the live model will be fit immediately from the local aggregated marts.";

  const body = `This specification will estimate the association between ${exposureLabel} and ${outcomeLabel} within ${trackLabel} using the ${templateLabel}. The principal result will be reported as a ${effect.label.toLowerCase()} for the selected exposure definition, with ${offsetLabel} applied automatically to preserve denominator compatibility.`;
  const interpretation = effect.label === "Rate ratio"
    ? "In substantive terms, the result will indicate whether higher exposure values are associated with higher or lower outcome rates under the chosen design."
    : "In substantive terms, the result will indicate whether higher exposure values are associated with an increase or decrease in the mean outcome under the chosen design.";

  const outputs = [
    `${effect.label} estimate with 95% confidence interval and p-value.`,
    `Coefficient table, enlarged forest plot, and reproducible JSON specification.`,
    `Execution path resolved as reference lookup first, then approved live estimation if needed.`,
    spec.stratifier_id
      ? `Separate stratum-specific estimates for ${stratifierLabel}, to support heterogeneity assessment.`
      : "A primary estimate without contextual stratification.",
  ];

  const designNotes = [
    `Exposure timing: ${lagLabel}.`,
    `Source restriction: ${sourceFilterLabel}.`,
    `Grouping structure: ${groupingLabel}.`,
    recipe?.gestational_source_modes.length ? `Gestational-age source rule: ${gestationalLabel}.` : "No gestational-source override is active for this specification.",
    executionText,
  ];

  return {
    title: `${outcomeLabel} as a function of ${exposureLabel}`,
    body,
    interpretation,
    effect,
    outputs,
    designNotes,
  };
}

function buildResultInterpretation(result: RunResult, stratifiers: Stratifier[]): string[] {
  const row = primaryResultRow(result);
  if (!row || row.estimate == null) {
    return [
      "This run did not yield an estimable primary effect. The table and forest plot should therefore be read as a record of model execution rather than as interpretable inferential output.",
    ];
  }

  const outcomeLabel = naturalOutcomeLabel(String(result.summary.outcome_label ?? result.spec.outcome_variable), result.spec.outcome_variable);
  const exposureLabel = naturalExposureLabel(String(result.summary.exposure_label ?? result.spec.exposure_variable), result.spec.exposure_variable);
  const effect = effectDescriptorFromScale(row.estimate_scale, result.spec.outcome_variable);
  const ciIncludesNull = row.lower_ci != null && row.upper_ci != null
    ? row.lower_ci <= effect.nullValue && row.upper_ci >= effect.nullValue
    : true;
  const ciText = row.lower_ci != null && row.upper_ci != null
    ? `${formatNumber(row.lower_ci)} to ${formatNumber(row.upper_ci)}`
    : "not available";
  const pText = formatPValue(row.p_value);

  let mainSentence = "";
  if (effect.label === "Rate ratio") {
    const percentChange = (row.estimate - 1) * 100;
    const direction = percentChange > 0 ? "higher" : percentChange < 0 ? "lower" : "similar";
    const magnitude = Math.abs(percentChange).toFixed(1);
    mainSentence = direction === "similar"
      ? `In the primary estimate, higher ${exposureLabel} was associated with a very small change in the rate of ${outcomeLabel} (${effect.shorthand} ${formatNumber(row.estimate)}, 95% CI ${ciText}; p = ${pText}).`
      : `In the primary estimate, higher ${exposureLabel} was associated with an estimated ${magnitude}% ${direction} rate of ${outcomeLabel} (${effect.shorthand} ${formatNumber(row.estimate)}, 95% CI ${ciText}; p = ${pText}).`;
  } else {
    const direction = row.estimate > 0 ? "higher" : row.estimate < 0 ? "lower" : "similar";
    const magnitude = Math.abs(row.estimate).toFixed(2);
    mainSentence = direction === "similar"
      ? `In the primary estimate, higher ${exposureLabel} was associated with little difference in ${outcomeLabel} (${effect.shorthand} ${formatNumber(row.estimate)}, 95% CI ${ciText}; p = ${pText}).`
      : `In the primary estimate, higher ${exposureLabel} was associated with ${magnitude} ${direction === "higher" ? "additional" : "fewer"} units of ${outcomeLabel} on average (${effect.shorthand} ${formatNumber(row.estimate)}, 95% CI ${ciText}; p = ${pText}).`;
  }

  const uncertaintySentence = ciIncludesNull
    ? `The interval includes the null reference (${effect.nullValue}), so the estimate should be read as compatible with both modest effects and no clear departure from the null.`
    : `The interval excludes the null reference (${effect.nullValue}), indicating that the estimated association is directionally consistent with a non-null effect under the selected specification.`;

  const stratifierSentence = result.spec.stratifier_id && result.rows.length > 1
    ? `Because the model was stratified by ${stratifiers.find((item) => item.stratifier_id === result.spec.stratifier_id)?.label ?? result.spec.stratifier_id}, the forest plot and coefficient table should be used to assess heterogeneity across strata rather than to infer a single pooled contrast.`
    : `The forest plot and coefficient table provide the primary inferential summary for this specification.`;

  const warningSentence = result.warnings.length > 0
    ? `Interpretation should be tempered by the recorded warnings: ${result.warnings.join(" ")}`
    : "";

  return [mainSentence, uncertaintySentence, stratifierSentence, warningSentence].filter(Boolean);
}

function InfoButton({
  topicId,
  onOpen,
}: {
  topicId: HelpTopicId;
  onOpen: (topicId: HelpTopicId) => void;
}) {
  return (
    <button
      aria-label={`Explain ${HELP_TOPICS[topicId].title}`}
      className="info-button"
      onClick={() => onOpen(topicId)}
      type="button"
    >
      ?
    </button>
  );
}

function HelpPanel({
  topicId,
  tab,
  onSelectTopic,
  onSelectTab,
}: {
  topicId: HelpTopicId;
  tab: HelpTab;
  onSelectTopic: (topicId: HelpTopicId) => void;
  onSelectTab: (tab: HelpTab) => void;
}) {
  const topic = HELP_TOPICS[topicId];
  return (
    <div className="panel">
      <div className="panel-header">
        <div>
          <h2>Concept guide</h2>
          <p className="panel-copy">Use the question-mark buttons in the builder to load short or extended explanations.</p>
        </div>
      </div>
      <div className="chip-row">
        {KEY_CONCEPTS.map((item) => (
          <button
            className={`concept-chip${item === topicId ? " concept-chip-active" : ""}`}
            key={item}
            onClick={() => onSelectTopic(item)}
            type="button"
          >
            {HELP_TOPICS[item].title}
          </button>
        ))}
      </div>
      <div className="tab-row">
        <button
          className={`tab-button${tab === "quick" ? " tab-button-active" : ""}`}
          onClick={() => onSelectTab("quick")}
          type="button"
        >
          Quick Help
        </button>
        <button
          className={`tab-button${tab === "methodology" ? " tab-button-active" : ""}`}
          onClick={() => onSelectTab("methodology")}
          type="button"
        >
          Methodology
        </button>
      </div>
      <div className="help-body">
        <span className="banner-label">Selected topic</span>
        <h3>{topic.title}</h3>
        <p>{tab === "quick" ? topic.quick : topic.methodology}</p>
      </div>
    </div>
  );
}

function SpecificationPreview({
  recipe,
  spec,
  stratifiers,
  onOpenHelp,
}: {
  recipe: Recipe | undefined;
  spec: RunSpec;
  stratifiers: Stratifier[];
  onOpenHelp: (topicId: HelpTopicId) => void;
}) {
  const preview = buildPreviewNarrative(recipe, spec, stratifiers);

  return (
    <div className="panel">
      <div className="panel-header">
        <div>
          <h2>Live specification preview</h2>
          <p className="panel-copy">This panel updates as you change the analytical controls.</p>
        </div>
        <button className="link-button" onClick={() => onOpenHelp("reference_live")} type="button">
          How execution is resolved
        </button>
      </div>
      <div className="preview-grid">
        <div className="preview-highlight">
          <span className="banner-label">Expected question</span>
          <strong>{preview.title}</strong>
          <p>{preview.body}</p>
        </div>
        <div className="summary-card">
          <span>Effect scale</span>
          <strong>{preview.effect.label}</strong>
          <p>{preview.effect.interpretationGuide}</p>
        </div>
        <div className="summary-card">
          <span>Analytical reading</span>
          <strong>{preview.effect.previewTerm}</strong>
          <p>{preview.interpretation}</p>
        </div>
      </div>
      <div className="preview-columns">
        <div className="preview-list">
          <span className="banner-label">Expected outputs</span>
          <ul>
            {preview.outputs.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div className="preview-list">
          <span className="banner-label">Design notes</span>
          <ul>
            {preview.designNotes.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

function InterpretationPanel({ result, stratifiers }: { result: RunResult; stratifiers: Stratifier[] }) {
  const interpretation = buildResultInterpretation(result, stratifiers);
  return (
    <div className="interpretation-block">
      <div className="panel-header">
        <h3>Academic interpretation</h3>
        <span className="pill">Generated from the returned estimate</span>
      </div>
      <div className="interpretation-copy">
        {interpretation.map((paragraph) => (
          <p key={paragraph}>{paragraph}</p>
        ))}
      </div>
    </div>
  );
}

function ResultTable({
  result,
  stratifiers,
}: {
  result: RunResult;
  stratifiers: Stratifier[];
}) {
  const effect = effectDescriptorFromScale(primaryResultRow(result)?.estimate_scale, result.spec.outcome_variable);

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Current results</h2>
        <a className="link-button" href={result.table_csv_url ?? "#"}>Download CSV</a>
      </div>
      <div className="summary-grid">
        <div className="summary-card">
          <span>Outcome</span>
          <strong>{naturalOutcomeLabel(String(result.summary.outcome_label ?? "NA"), result.spec.outcome_variable)}</strong>
        </div>
        <div className="summary-card">
          <span>Exposure</span>
          <strong>{naturalExposureLabel(String(result.summary.exposure_label ?? "NA"), result.spec.exposure_variable)}</strong>
        </div>
        <div className="summary-card">
          <span>Execution mode</span>
          <strong>{naturalExecutionModeLabel(result.execution_mode)}</strong>
        </div>
        <div className="summary-card">
          <span>Offset</span>
          <strong>{naturalOffsetLabel(String(result.summary.offset_label ?? "NA"))}</strong>
        </div>
        <div className="summary-card">
          <span>Effect scale</span>
          <strong>{effect.label}</strong>
        </div>
      </div>
      {result.warnings.length > 0 ? (
        <div className="warning-box">
          {result.warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      ) : null}
      <InterpretationPanel result={result} stratifiers={stratifiers} />
      <div className="result-grid">
        <div className="figure-shell figure-shell-large">
          {result.forest_plot_url ? (
            <img src={result.forest_plot_url} alt="Forest plot" />
          ) : (
            <div className="empty-figure">No forest plot is available for this run.</div>
          )}
        </div>
        <div className="table-shell">
          <table>
            <thead>
              <tr>
                <th>Group</th>
                <th>Estimate</th>
                <th>95% CI</th>
                <th>p-value</th>
                <th>Model</th>
              </tr>
            </thead>
            <tbody>
              {result.rows.map((row) => (
                <tr key={row.row_id}>
                  <td>{row.group_label ?? naturalOutcomeLabel(row.label)}</td>
                  <td>{formatNumber(row.estimate)}</td>
                  <td>{`${formatNumber(row.lower_ci)} to ${formatNumber(row.upper_ci)}`}</td>
                  <td>{formatPValue(row.p_value)}</td>
                  <td>{naturalModelLabel(row.model_label)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="spec-block">
        <div className="panel-header">
          <h3>Reproducible specification</h3>
          <span className="pill">{String(result.summary.reference_artifact ?? "live")}</span>
        </div>
        <pre>{JSON.stringify(result.spec, null, 2)}</pre>
      </div>
    </div>
  );
}

function HistoryList({
  runs,
  recipes,
  activeRunId,
  onSelect,
  onDuplicate,
}: {
  runs: Run[];
  recipes: Recipe[];
  activeRunId: string | null;
  onSelect: (run: Run) => void;
  onDuplicate: (run: Run) => void;
}) {
  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Run history</h2>
      </div>
      <div className="list-stack">
        {runs.map((run) => (
          <button
            key={run.run_id}
            className={`list-card${run.run_id === activeRunId ? " list-card-active" : ""}`}
            onClick={() => onSelect(run)}
            type="button"
          >
            <div>
              <strong>{naturalRecipeLabel(recipes.find((item) => item.recipe_id === run.recipe_id)?.label ?? run.recipe_id, run.recipe_id)}</strong>
              <p>{naturalOutcomeLabel(optionLabel(recipes.find((item) => item.recipe_id === run.recipe_id)?.outcomes, run.spec.outcome_variable, run.spec.outcome_variable), run.spec.outcome_variable)} · {naturalExposureLabel(optionLabel(recipes.find((item) => item.recipe_id === run.recipe_id)?.exposures, run.spec.exposure_variable, run.spec.exposure_variable), run.spec.exposure_variable)}</p>
            </div>
            <div className="list-actions">
              <span className={`status status-${run.status}`}>{run.status}</span>
              <span className="secondary-action" onClick={(event) => { event.stopPropagation(); onDuplicate(run); }}>Duplicate</span>
            </div>
          </button>
        ))}
        {runs.length === 0 ? <p className="empty-copy">No runs yet.</p> : null}
      </div>
    </div>
  );
}

function PresetList({
  presets,
  recipes,
  onLoad,
  onDelete,
}: {
  presets: Preset[];
  recipes: Recipe[];
  onLoad: (preset: Preset) => void;
  onDelete: (preset: Preset) => void;
}) {
  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Saved presets</h2>
      </div>
      <div className="list-stack">
        {presets.map((preset) => (
          <div className="list-card" key={preset.preset_id}>
            <div>
              <strong>{preset.name}</strong>
              <p>{preset.description || naturalRecipeLabel(recipes.find((item) => item.recipe_id === preset.run_spec.recipe_id)?.label ?? preset.run_spec.recipe_id, preset.run_spec.recipe_id)}</p>
            </div>
            <div className="list-actions">
              <span className="secondary-action" onClick={() => onLoad(preset)}>Load</span>
              <span className="secondary-action destructive" onClick={() => onDelete(preset)}>Delete</span>
            </div>
          </div>
        ))}
        {presets.length === 0 ? <p className="empty-copy">No saved presets.</p> : null}
      </div>
    </div>
  );
}

function App() {
  const queryClient = useQueryClient();
  const [spec, setSpec] = useState<RunSpec>(EMPTY_SPEC);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [helpTopicId, setHelpTopicId] = useState<HelpTopicId>("recipe");
  const [helpTab, setHelpTab] = useState<HelpTab>("quick");

  const { data: recipes = [], isLoading, error } = useQuery({ queryKey: ["recipes"], queryFn: getRecipes });
  const { data: stratifiers = [] } = useQuery({ queryKey: ["stratifiers"], queryFn: getStratifiers });
  const { data: presets = [] } = useQuery({ queryKey: ["presets"], queryFn: getPresets });
  const { data: runs = [] } = useQuery({
    queryKey: ["runs"],
    queryFn: getRuns,
    refetchInterval: activeRunId ? 2000 : false,
  });

  const recipe = useMemo(() => recipes.find((item) => item.recipe_id === spec.recipe_id), [recipes, spec.recipe_id]);
  const activeRun = useQuery({
    queryKey: ["run", activeRunId],
    queryFn: () => getRun(activeRunId!),
    enabled: Boolean(activeRunId),
    refetchInterval: (query) => runIsPending(query.state.data as Run | undefined) ? 2000 : false,
  });
  const runResult = useQuery({
    queryKey: ["result", activeRunId],
    queryFn: () => getRunResult(activeRunId!),
    enabled: Boolean(activeRunId && activeRun.data && !runIsPending(activeRun.data)),
  });

  useEffect(() => {
    if (recipes.length > 0 && !spec.recipe_id) {
      setSpec(buildDefaultSpec(recipes[0]));
    }
  }, [recipes, spec.recipe_id]);

  useEffect(() => {
    if (recipe) {
      setSpec((current) => {
        if (current.recipe_id !== recipe.recipe_id) {
          return current;
        }
        const outcomeVariable = recipe.outcomes.some((item) => item.id === current.outcome_variable)
          ? current.outcome_variable
          : (recipe.outcomes[0]?.id ?? "");
        return {
          ...current,
          template_id: recipe.templates.some((item) => item.id === current.template_id) ? current.template_id : recipe.default_template_id,
          outcome_variable: outcomeVariable,
          exposure_variable: recipe.exposures.some((item) => item.id === current.exposure_variable) ? current.exposure_variable : (recipe.exposures[0]?.id ?? ""),
          lag_or_window: recipe.lag_windows.some((item) => item.id === current.lag_or_window)
            ? current.lag_or_window
            : (recipe.lag_windows[0]?.id ?? String(recipe.exposures[0]?.metadata?.lag_or_window ?? "")),
          source_filter: recipe.source_filters.some((item) => item.id === current.source_filter) ? current.source_filter : (recipe.source_filters[0]?.id ?? "all_sources"),
          grouping_template: recipe.grouping_templates.some((item) => item.id === current.grouping_template) ? current.grouping_template : (recipe.grouping_templates[0]?.id ?? ""),
          gestational_source_mode: recipe.gestational_source_modes.some((item) => item.id === current.gestational_source_mode) ? current.gestational_source_mode : (recipe.gestational_source_modes[0]?.id ?? ""),
          offset_id: resolveOffsetId(recipe, outcomeVariable),
        };
      });
    }
  }, [recipe]);

  const createRunMutation = useMutation({
    mutationFn: postRun,
    onSuccess: async (run) => {
      setActiveRunId(run.run_id);
      await queryClient.invalidateQueries({ queryKey: ["runs"] });
      if (!runIsPending(run)) {
        await queryClient.invalidateQueries({ queryKey: ["result", run.run_id] });
      }
    },
  });

  const createPresetMutation = useMutation({
    mutationFn: createPreset,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["presets"] });
    },
  });

  const deletePresetMutation = useMutation({
    mutationFn: deletePreset,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["presets"] });
    },
  });

  const openHelpTopic = (topicId: HelpTopicId) => {
    setHelpTopicId(topicId);
  };

  const onRecipeChange = (recipeId: string) => {
    const nextRecipe = recipes.find((item) => item.recipe_id === recipeId);
    setSpec(buildDefaultSpec(nextRecipe));
    openHelpTopic(nextRecipe?.track_id === "track_b" ? "track_b" : "track_a");
  };

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const normalizedSpec = normalizeSpecForRecipe(recipe, spec);
    await createRunMutation.mutateAsync({
      ...normalizedSpec,
      stratifier_id: normalizedSpec.stratifier_id || null,
      denominator_variant: normalizedSpec.denominator_variant || null,
      grouping_template: normalizedSpec.grouping_template || null,
      gestational_source_mode: normalizedSpec.gestational_source_mode || null,
      lag_or_window: normalizedSpec.lag_or_window || null,
      offset_id: normalizedSpec.offset_id || null,
    });
  };

  const savePreset = async () => {
    const name = window.prompt("Preset name:");
    if (!name) {
      return;
    }
    const description = window.prompt("Short preset description:") ?? "";
    await createPresetMutation.mutateAsync({ name, description, run_spec: normalizeSpecForRecipe(recipe, spec) });
  };

  if (isLoading) {
    return <div className="loading-shell">Loading statistical calculator metadata...</div>;
  }

  if (error instanceof Error) {
    return <div className="loading-shell error-shell">{error.message}</div>;
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Observa Infancia · Local analytical workstation</p>
          <h1>Statistical calculator for outcome analyses</h1>
          <p className="hero-copy">
            A local-first interface for aggregated current-cycle analyses and event-level escalation analyses, with constrained analytical recipes,
            reference-first execution, and reproducible result bundles.
          </p>
        </div>
        <div className="hero-meta">
          <span className="pill">Mac mini · port 8010</span>
          <span className="pill">Reference-first execution</span>
          <span className="pill">Automatic offset governance</span>
          <span className="pill">Context restricted to stratification</span>
        </div>
      </header>

      <main className="layout">
        <section className="main-column">
          <form className="panel form-panel" onSubmit={onSubmit}>
            <div className="panel-header">
              <div>
                <h2>Specification builder</h2>
                <p className="panel-copy">Select a validated recipe, refine the approved options, and review the expected estimand before running the model.</p>
              </div>
              <button className="primary-button" disabled={createRunMutation.isPending} type="submit">
                {createRunMutation.isPending ? "Running..." : "Run analysis"}
              </button>
            </div>
            <div className="form-grid">
              <div className="field-shell">
                <span className="label-row">
                  <label className="field-label" htmlFor="recipe-field">Recipe</label>
                  <InfoButton onOpen={openHelpTopic} topicId="recipe" />
                </span>
                <select id="recipe-field" value={spec.recipe_id} onChange={(event) => onRecipeChange(event.target.value)}>
                  {recipes.map((item) => (
                    <option key={item.recipe_id} value={item.recipe_id}>{naturalRecipeLabel(item.label, item.recipe_id)}</option>
                  ))}
                </select>
              </div>
              <div className="field-shell">
                <span className="label-row">
                  <label className="field-label" htmlFor="template-field">Template</label>
                  <InfoButton onOpen={openHelpTopic} topicId="template" />
                </span>
                <select id="template-field" value={spec.template_id} onChange={(event) => setSpec((current) => ({ ...current, template_id: event.target.value }))}>
                  {recipe?.templates.map((item) => (
                    <option key={item.id} value={item.id}>{naturalTemplateLabel(item.label)}</option>
                  ))}
                </select>
              </div>
              <div className="field-shell">
                <span className="label-row">
                  <label className="field-label" htmlFor="outcome-field">Outcome</label>
                  <InfoButton onOpen={openHelpTopic} topicId="outcome" />
                </span>
                <select
                  id="outcome-field"
                  value={spec.outcome_variable}
                  onChange={(event) => {
                    const outcomeVariable = event.target.value;
                    setSpec((current) => ({
                      ...current,
                      outcome_variable: outcomeVariable,
                      offset_id: resolveOffsetId(recipe, outcomeVariable),
                    }));
                  }}
                >
                  {recipe?.outcomes.map((item) => (
                    <option key={item.id} value={item.id}>{naturalOutcomeLabel(item.label, item.id)}</option>
                  ))}
                </select>
              </div>
              <div className="field-shell">
                <span className="label-row">
                  <label className="field-label" htmlFor="exposure-field">Exposure</label>
                  <InfoButton onOpen={openHelpTopic} topicId="exposure" />
                </span>
                <select
                  id="exposure-field"
                  value={spec.exposure_variable}
                  onChange={(event) => {
                    const exposure = recipe?.exposures.find((item) => item.id === event.target.value);
                    setSpec((current) => ({
                      ...current,
                      exposure_variable: event.target.value,
                      lag_or_window: recipe?.lag_windows.length ? current.lag_or_window : String(exposure?.metadata?.lag_or_window ?? ""),
                    }));
                  }}
                >
                  {recipe?.exposures.map((item) => (
                    <option key={item.id} value={item.id}>{naturalExposureLabel(item.label, item.id)}</option>
                  ))}
                </select>
              </div>
              {recipe?.lag_windows.length ? (
                <div className="field-shell">
                  <span className="label-row">
                    <label className="field-label" htmlFor="lag-window-field">Lag / window</label>
                    <InfoButton onOpen={openHelpTopic} topicId="lag_window" />
                  </span>
                  <select id="lag-window-field" value={spec.lag_or_window ?? ""} onChange={(event) => setSpec((current) => ({ ...current, lag_or_window: event.target.value }))}>
                    {recipe.lag_windows.map((item) => (
                      <option key={item.id} value={item.id}>{naturalLagLabel(item.label || item.id)}</option>
                    ))}
                  </select>
                </div>
              ) : null}
              <div className="field-shell">
                <span className="label-row">
                  <label className="field-label" htmlFor="stratifier-field">Stratifier</label>
                  <InfoButton onOpen={openHelpTopic} topicId="stratifier" />
                </span>
                <select id="stratifier-field" value={spec.stratifier_id ?? ""} onChange={(event) => setSpec((current) => ({ ...current, stratifier_id: event.target.value }))}>
                  <option value="">No stratification</option>
                  {recipe?.stratifiers.map((item) => {
                    const stratifier = stratifiers.find((entry) => entry.stratifier_id === item);
                    return (
                      <option key={item} value={item}>{stratifier?.label ?? item}</option>
                    );
                  })}
                </select>
              </div>
              <div className="field-shell">
                <span className="label-row">
                  <label className="field-label" htmlFor="source-filter-field">Source filter</label>
                  <InfoButton onOpen={openHelpTopic} topicId="source_filter" />
                </span>
                <select id="source-filter-field" value={spec.source_filter} onChange={(event) => setSpec((current) => ({ ...current, source_filter: event.target.value }))}>
                  {recipe?.source_filters.map((item) => (
                    <option key={item.id} value={item.id}>{naturalSourceFilterLabel(item.label)}</option>
                  ))}
                </select>
              </div>
              {recipe?.grouping_templates.length ? (
                <div className="field-shell">
                  <span className="label-row">
                    <label className="field-label" htmlFor="grouping-field">Grouping template</label>
                    <InfoButton onOpen={openHelpTopic} topicId="grouping" />
                  </span>
                  <select id="grouping-field" value={spec.grouping_template ?? ""} onChange={(event) => setSpec((current) => ({ ...current, grouping_template: event.target.value }))}>
                    {recipe.grouping_templates.map((item) => (
                      <option key={item.id} value={item.id}>{naturalGroupingLabel(item.label)}</option>
                    ))}
                  </select>
                </div>
              ) : null}
              {recipe?.gestational_source_modes.length ? (
                <div className="field-shell">
                  <span className="label-row">
                    <label className="field-label" htmlFor="gestational-source-field">Gestational source</label>
                    <InfoButton onOpen={openHelpTopic} topicId="gestational_source" />
                  </span>
                  <select id="gestational-source-field" value={spec.gestational_source_mode ?? ""} onChange={(event) => setSpec((current) => ({ ...current, gestational_source_mode: event.target.value }))}>
                    {recipe.gestational_source_modes.map((item) => (
                      <option key={item.id} value={item.id}>{naturalGestationalSourceLabel(item.label)}</option>
                    ))}
                  </select>
                </div>
              ) : null}
              <div className="field-shell">
                <span className="label-row">
                  <label className="field-label" htmlFor="offset-field">Offset</label>
                  <InfoButton onOpen={openHelpTopic} topicId="offset" />
                </span>
                <input id="offset-field" readOnly value={currentOffsetLabel(recipe, spec)} />
              </div>
            </div>
            <div className="spec-banner">
              <div>
                <span className="banner-label">Current specification</span>
                <strong>{currentOutcomeLabel(recipe, spec)} · {currentExposureLabel(recipe, spec)}</strong>
              </div>
              <button className="secondary-button" onClick={savePreset} type="button">Save preset</button>
            </div>
            {createRunMutation.error instanceof Error ? <p className="inline-error">{createRunMutation.error.message}</p> : null}
          </form>

          <SpecificationPreview
            onOpenHelp={openHelpTopic}
            recipe={recipe}
            spec={spec}
            stratifiers={stratifiers}
          />

          {activeRun.data ? (
            <div className="panel status-panel">
              <div className="panel-header">
                <h2>Execution details</h2>
                <span className={`status status-${activeRun.data.status}`}>{activeRun.data.status}</span>
              </div>
              <div className="detail-grid">
                <p><strong>Run ID:</strong> {activeRun.data.run_id}</p>
                <p><strong>Execution mode:</strong> {naturalExecutionModeLabel(activeRun.data.execution_mode)}</p>
                <p><strong>Created at:</strong> {activeRun.data.created_at}</p>
                <p><strong>Updated at:</strong> {activeRun.data.updated_at}</p>
              </div>
              {activeRun.data.error_message ? <p className="inline-error">{activeRun.data.error_message}</p> : null}
              {activeRun.data.warnings.length > 0 ? (
                <div className="warning-box">
                  {activeRun.data.warnings.map((warning) => (
                    <p key={warning}>{warning}</p>
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}

          {runResult.data ? <ResultTable result={runResult.data} stratifiers={stratifiers} /> : null}
        </section>

        <aside className="side-column">
          <HelpPanel
            onSelectTab={setHelpTab}
            onSelectTopic={setHelpTopicId}
            tab={helpTab}
            topicId={helpTopicId}
          />
          <HistoryList
            runs={runs}
            recipes={recipes}
            activeRunId={activeRunId}
            onSelect={(run) => {
              setActiveRunId(run.run_id);
              setSpec(normalizeSpecForRecipe(recipes.find((item) => item.recipe_id === run.spec.recipe_id), run.spec));
            }}
            onDuplicate={(run) => setSpec(normalizeSpecForRecipe(recipes.find((item) => item.recipe_id === run.spec.recipe_id), run.spec))}
          />
          <PresetList
            presets={presets}
            recipes={recipes}
            onLoad={(preset) => setSpec(normalizeSpecForRecipe(recipes.find((item) => item.recipe_id === preset.run_spec.recipe_id), preset.run_spec))}
            onDelete={(preset) => deletePresetMutation.mutate(preset.preset_id)}
          />
        </aside>
      </main>
    </div>
  );
}

export default App;
